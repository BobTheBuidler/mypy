"""Code generation for native function bodies."""

from __future__ import annotations

from typing import Final

from mypyc.analysis.blockfreq import frequently_executed_blocks
from mypyc.codegen.emit import DEBUG_ERRORS, Emitter, TracebackAndGotoHandler, c_array_initializer
from mypyc.common import (
    GENERATOR_ATTRIBUTE_PREFIX,
    HAVE_IMMORTAL,
    MODULE_PREFIX,
    NATIVE_PREFIX,
    REG_PREFIX,
    STATIC_PREFIX,
    TYPE_PREFIX,
    TYPE_VAR_PREFIX,
)
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FUNC_CLASSMETHOD, FUNC_STATICMETHOD, FuncDecl, FuncIR, all_values
from mypyc.ir.ops import (
    ERR_FALSE,
    NAMESPACE_MODULE,
    NAMESPACE_STATIC,
    NAMESPACE_TYPE,
    NAMESPACE_TYPE_VAR,
    Assign,
    AssignMulti,
    BasicBlock,
    Box,
    Branch,
    Call,
    CallC,
    Cast,
    ComparisonOp,
    ControlOp,
    CString,
    DecRef,
    Extend,
    Float,
    FloatComparisonOp,
    FloatNeg,
    FloatOp,
    GetAttr,
    GetElementPtr,
    Goto,
    IncRef,
    InitStatic,
    Integer,
    IntOp,
    KeepAlive,
    LoadAddress,
    LoadErrorValue,
    LoadGlobal,
    LoadLiteral,
    LoadMem,
    LoadStatic,
    MethodCall,
    Op,
    OpVisitor,
    PrimitiveOp,
    RaiseStandardError,
    Register,
    Return,
    SetAttr,
    SetElement,
    SetMem,
    Truncate,
    TupleGet,
    TupleSet,
    Unborrow,
    Unbox,
    Undef,
    Unreachable,
    Value,
)
from mypyc.ir.pprint import generate_names_for_ir
from mypyc.ir.rtypes import (
    RArray,
    RInstance,
    RStruct,
    RTuple,
    RType,
    is_bool_or_bit_rprimitive,
    is_int32_rprimitive,
    is_int64_rprimitive,
    is_int_rprimitive,
    is_none_rprimitive,
    is_pointer_rprimitive,
    is_tagged,
)


def native_function_type(fn: FuncIR, emitter: Emitter) -> str:
    args = ", ".join(emitter.ctype(arg.type) for arg in fn.args) or "void"
    ret = emitter.ctype(fn.ret_type)
    return f"{ret} (*)({args})"


def native_function_header(fn: FuncDecl, emitter: Emitter) -> str:
    args = []
    for arg in fn.sig.args:
        args.append(f"{emitter.ctype_spaced(arg.type)}{REG_PREFIX}{arg.name}")

    return "{ret_type}{name}({args})".format(
        ret_type=emitter.ctype_spaced(fn.sig.ret_type),
        name=emitter.native_function_name(fn),
        args=", ".join(args) or "void",
    )


def generate_native_function(
    fn: FuncIR, emitter: Emitter, source_path: str, module_name: str
) -> None:
    declarations = Emitter(emitter.context)
    names = generate_names_for_ir(fn.arg_regs, fn.blocks)
    body = Emitter(emitter.context, names)
    visitor = FunctionEmitterVisitor(body, declarations, source_path, module_name)

    declarations.emit_line(f"{native_function_header(fn.decl, emitter)} {{")
    body.indent()

    for r in all_values(fn.arg_regs, fn.blocks):
        if isinstance(r.type, RTuple):
            emitter.declare_tuple_struct(r.type)
        if isinstance(r.type, RArray):
            continue  # Special: declared on first assignment

        if r in fn.arg_regs:
            continue  # Skip the arguments

        ctype = emitter.ctype_spaced(r.type)
        init = ""
        declarations.emit_line(
            "{ctype}{prefix}{name}{init};".format(
                ctype=ctype, prefix=REG_PREFIX, name=names[r], init=init
            )
        )

    # Before we emit the blocks, give them all labels
    blocks = fn.blocks
    for i, block in enumerate(blocks):
        block.label = i

    # Find blocks that are never jumped to or are only jumped to from the
    # block directly above it. This allows for more labels and gotos to be
    # eliminated during code generation.
    for block in fn.blocks:
        terminator = block.terminator
        assert isinstance(terminator, ControlOp), terminator

        for target in terminator.targets():
            is_next_block = target.label == block.label + 1

            # Always emit labels for GetAttr error checks since the emit code that
            # generates them will add instructions between the branch and the
            # next label, causing the label to be wrongly removed. A better
            # solution would be to change the IR so that it adds a basic block
            # in between the calls.
            is_problematic_op = isinstance(terminator, Branch) and any(
                isinstance(s, GetAttr) for s in terminator.sources()
            )

            if not is_next_block or is_problematic_op:
                fn.blocks[target.label].referenced = True

    common = frequently_executed_blocks(fn.blocks[0])

    for i in range(len(blocks)):
        block = blocks[i]
        visitor.rare = block not in common
        next_block = None
        if i + 1 < len(blocks):
            next_block = blocks[i + 1]
        body.emit_label(block)
        visitor.next_block = next_block

        ops = block.ops
        visitor.ops = ops
        visitor.op_index = 0
        while visitor.op_index < len(ops):
            ops[visitor.op_index].accept(visitor)
            visitor.op_index += 1

    body.emit_line("}")

    emitter.emit_from_emitter(declarations)
    emitter.emit_from_emitter(body)


class FunctionEmitterVisitor(OpVisitor[None]):
    def __init__(
        self, emitter: Emitter, declarations: Emitter, source_path: str, module_name: str
    ) -> None:
        self.emitter = emitter
        self.names = emitter.names
        self.declarations = declarations
        self.source_path = source_path
        self.module_name = module_name
        self.literals = emitter.context.literals
        self.rare = False
        # Next basic block to be processed after the current one (if any), set by caller
        self.next_block: BasicBlock | None = None
        # Ops in the basic block currently being processed, set by caller
        self.ops: list[Op] = []
        # Current index within ops; visit methods can increment this to skip/merge ops
        self.op_index = 0

    def temp_name(self) -> str:
        return self.emitter.temp_name()

    def visit_goto(self, op: Goto) -> None:
        if op.label is not self.next_block:
            self.emit_line("goto %s;" % self.label(op.label))

    def error_value_check(self, value: Value, compare: str) -> str:
        typ = value.type
        if isinstance(typ, RTuple):
            # TODO: What about empty tuple?
            return self.emitter.tuple_undefined_check_cond(
                typ, self.reg(value), self.c_error_value, compare
            )
        else:
            return f"{self.reg(value)} {compare} {self.c_error_value(typ)}"

    def visit_branch(self, op: Branch) -> None:
        true, false = op.true, op.false
        negated = op.negated
        negated_rare = False
        if true is self.next_block and op.traceback_entry is None:
            # Switch true/false since it avoids an else block.
            true, false = false, true
            negated = not negated
            negated_rare = True

        neg = "!" if negated else ""
        cond = ""
        if op.op == Branch.BOOL:
            expr_result = self.reg(op.value)
            cond = f"{neg}{expr_result}"
        elif op.op == Branch.IS_ERROR:
            compare = "!=" if negated else "=="
            cond = self.error_value_check(op.value, compare)
        else:
            assert False, "Invalid branch"

        # For error checks, tell the compiler the branch is unlikely
        if op.traceback_entry is not None or op.rare:
            if not negated_rare:
                cond = f"unlikely({cond})"
            else:
                cond = f"likely({cond})"

        if false is self.next_block:
            if op.traceback_entry is None:
                if true is not self.next_block:
                    self.emit_line(f"if ({cond}) goto {self.label(true)};")
            else:
                self.emit_line(f"if ({cond}) {{")
                self.emit_traceback(op)
                self.emit_lines("goto %s;" % self.label(true), "}")
        else:
            self.emit_line(f"if ({cond}) {{")
            self.emit_traceback(op)

            if true is not self.next_block:
                self.emit_line("goto %s;" % self.label(true))

            self.emit_lines("} else", "    goto %s;" % self.label(false))

    def visit_return(self, op: Return) -> None:
        value_str = self.reg(op.value)
        self.emit_line("return %s;" % value_str)

    def visit_tuple_set(self, op: TupleSet) -> None:
        dest = self.reg(op)
        tuple_type = op.tuple_type
        self.emitter.declare_tuple_struct(tuple_type)
        if len(op.items) == 0:  # empty tuple
            self.emit_line(f"{dest}.empty_struct_error_flag = 0;")
        else:
            for i, item in enumerate(op.items):
                self.emit_line(f"{dest}.f{i} = {self.reg(item)};")

    def visit_assign(self, op: Assign) -> None:
        dest = self.reg(op.dest)
        src = self.reg(op.src)
        # clang whines about self assignment (which we might generate
        # for some casts), so don't emit it.
        if dest != src:
            # We sometimes assign from an integer prepresentation of a pointer
            # to a real pointer, and C compilers insist on a cast.
            if op.src.type.is_unboxed and not op.dest.type.is_unboxed:
                src = f"(void *){src}"
            self.emit_line(f"{dest} = {src};")

    def visit_assign_multi(self, op: AssignMulti) -> None:
        typ = op.dest.type
        assert isinstance(typ, RArray), typ
        dest = self.reg(op.dest)
        # RArray values can only be assigned to once, so we can always
        # declare them on initialization.
        self.emit_line(
            "%s%s[%d] = %s;"
            % (
                self.emitter.ctype_spaced(typ.item_type),
                dest,
                len(op.src),
                c_array_initializer([self.reg(s) for s in op.src], indented=True),
            )
        )

    def visit_load_error_value(self, op: LoadErrorValue) -> None:
        if isinstance(op.type, RTuple):
            values = [self.c_undefined_value(item) for item in op.type.types]
            tmp = self.temp_name()
            self.emit_line("{} {} = {{ {} }};".format(self.ctype(op.type), tmp, ", ".join(values)))
            self.emit_line(f"{self.reg(op)} = {tmp};")
        else:
            self.emit_line(f"{self.reg(op)} = {self.c_error_value(op.type)};")

    def visit_load_literal(self, op: LoadLiteral) -> None:
        index = self.literals.literal_index(op.value)
        if not is_int_rprimitive(op.type):
            self.emit_line("%s = CPyStatics[%d];" % (self.reg(op), index), ann=op.value)
        else:
            self.emit_line(
                "%s = (CPyTagged)CPyStatics[%d] | 1;" % (self.reg(op), index), ann=op.value
            )

    def get_attr_expr(self, obj: str, op: GetAttr | SetAttr, decl_cl: ClassIR) -> str:
        """Generate attribute accessor for normal (non-property) access.

        This either has a form like obj->attr_name for attributes defined in non-trait
        classes, and *(obj + attr_offset) for attributes defined by traits. We also
        insert all necessary C casts here.
        """
        cast = f"({op.class_type.struct_name(self.emitter.names)} *)"
        if decl_cl.is_trait and op.class_type.class_ir.is_trait:
            # For pure trait access find the offset first, offsets
            # are ordered by attribute position in the cl.attributes dict.
            # TODO: pre-calculate the mapping to make this faster.
            trait_attr_index = list(decl_cl.attributes).index(op.attr)
            # TODO: reuse these names somehow?
            offset = self.emitter.temp_name()
            self.declarations.emit_line(f"size_t {offset};")
            self.emitter.emit_line(
                "{} = {};".format(
                    offset,
                    "CPy_FindAttrOffset({}, {}, {})".format(
                        self.emitter.type_struct_name(decl_cl),
                        f"({cast}{obj})->vtable",
                        trait_attr_index,
                    ),
                )
            )
            attr_cast = f"({self.ctype(op.class_type.attr_type(op.attr))} *)"
            return f"*{attr_cast}((char *){obj} + {offset})"
        else:
            # Cast to something non-trait. Note: for this to work, all struct
            # members for non-trait classes must obey monotonic linear growth.
            if op.class_type.class_ir.is_trait:
                assert not decl_cl.is_trait
                cast = f"({decl_cl.struct_name(self.emitter.names)} *)"
            return f"({cast}{obj})->{self.emitter.attr(op.attr)}"

    def visit_get_attr(self, op: GetAttr) -> None:
        if op.allow_error_value:
            self.get_attr_with_allow_error_value(op)
            return
        dest = self.reg(op)
        obj = self.reg(op.obj)
        rtype = op.class_type
        cl = rtype.class_ir
        attr_rtype, decl_cl = cl.attr_details(op.attr)
        prefer_method = cl.is_trait and attr_rtype.error_overlap
        if cl.get_method(op.attr, prefer_method=prefer_method):
            # Properties are essentially methods, so use vtable access for them.
            if cl.is_method_final(op.attr):
                self.emit_method_call(f"{dest} = ", op.obj, op.attr, [])
            else:
                version = "_TRAIT" if cl.is_trait else ""
                self.emit_line(
                    "%s = CPY_GET_ATTR%s(%s, %s, %d, %s, %s); /* %s */"
                    % (
                        dest,
                        version,
                        obj,
                        self.emitter.type_struct_name(rtype.class_ir),
                        rtype.getter_index(op.attr),
                        rtype.struct_name(self.names),
                        self.ctype(rtype.attr_type(op.attr)),
                        op.attr,
                    )
                )
        else:
            # Otherwise, use direct or offset struct access.
            attr_expr = self.get_attr_expr(obj, op, decl_cl)
            self.emitter.emit_line(f"{dest} = {attr_expr};")
            always_defined = cl.is_always_defined(op.attr)
            merged_branch = None
            if not always_defined:
                self.emitter.emit_undefined_attr_check(
                    attr_rtype, dest, "==", obj, op.attr, cl, unlikely=True
                )
                branch = self.next_branch()
                if branch is not None:
                    if (
                        branch.value is op
                        and branch.op == Branch.IS_ERROR
                        and branch.traceback_entry is not None
                        and not branch.negated
                    ):
                        # Generate code for the following branch here to avoid
                        # redundant branches in the generated code.
                        self.emit_attribute_error(branch, cl, op.attr)
                        self.emit_line("goto %s;" % self.label(branch.true))
                        merged_branch = branch
                        self.emitter.emit_line("}")
                if not merged_branch:
                    var_name = op.attr.removeprefix(GENERATOR_ATTRIBUTE_PREFIX)
                    if cl.is_environment:
                        # Environment classes represent locals, so missing attrs are unbound vars.
                        exc_class = "PyExc_UnboundLocalError"
                        exc_msg = f"local variable {var_name!r} referenced before assignment"
                    else:
                        exc_class = "PyExc_AttributeError"
                        exc_msg = f"attribute {var_name!r} of {cl.name!r} undefined"
                    self.emitter.emit_line(f'PyErr_SetString({exc_class}, "{exc_msg}");')

            if attr_rtype.is_refcounted and not op.is_borrowed:
                if not merged_branch and not always_defined:
                    self.emitter.emit_line("} else {")
                self.emitter.emit_inc_ref(dest, attr_rtype)
            if merged_branch:
                if merged_branch.false is not self.next_block:
                    self.emit_line("goto %s;" % self.label(merged_branch.false))
                self.op_index += 1
            elif not always_defined:
                self.emitter.emit_line("}")

    def get_attr_with_allow_error_value(self, op: GetAttr) -> None:
        """Handle GetAttr with allow_error_value=True.

        This allows NULL or other error value without raising AttributeError.
        """
        dest = self.reg(op)
        obj = self.reg(op.obj)
        rtype = op.class_type
        cl = rtype.class_ir
        attr_rtype, decl_cl = cl.attr_details(op.attr)

        # Direct struct access without NULL check
        attr_expr = self.get_attr_expr(obj, op, decl_cl)
        self.emitter.emit_line(f"{dest} = {attr_expr};")

        # Only emit inc_ref if not NULL
        if attr_rtype.is_refcounted and not op.is_borrowed:
            check = self.error_value_check(op, "!=")
            self.emitter.emit_line(f"if ({check}) {{")
            self.emitter.emit_inc_ref(dest, attr_rtype)
            self.emitter.emit_line("}")

    def next_branch(self) -> Branch | None:
        if self.op_index + 1 < len(self.ops):
            next_op = self.ops[self.op_index + 1]
            if isinstance(next_op, Branch):
                return next_op
        return None

    def visit_set_attr(self, op: SetAttr) -> None:
        if op.error_kind == ERR_FALSE:
            dest = self.reg(op)
        obj = self.reg(op.obj)
        src = self.reg(op.src)
        rtype = op.class_type
        cl = rtype.class_ir
        attr_rtype, decl_cl = cl.attr_details(op.attr)
        if cl.get_method(op.attr):
            # Again, use vtable access for properties...
            assert not op.is_init and op.error_kind == ERR_FALSE, "%s %d %d %s" % (
                op.attr,
                op.is_init,
                op.error_kind,
                rtype,
            )
            version = "_TRAIT" if cl.is_trait else ""
            self.emit_line(
                "%s = CPY_SET_ATTR%s(%s, %s, %d, %s, %s, %s); /* %s */"
                % (
                    dest,
                    version,
                    obj,
                    self.emitter.type_struct_name(rtype.class_ir),
                    rtype.setter_index(op.attr),
                    src,
                    rtype.struct_name(self.names),
                    self.ctype(rtype.attr_type(op.attr)),
                    op.attr,
                )
            )
        else:
            # ...and struct access for normal attributes.
            attr_expr = self.get_attr_expr(obj, op, decl_cl)
            if not op.is_init and attr_rtype.is_refcounted:
                # This is not an initialization (where we know that the attribute was
                # previously undefined), so decref the old value.
                always_defined = cl.is_always_defined(op.attr)
                if not always_defined:
                    self.emitter.emit_undefined_attr_check(
                        attr_rtype, attr_expr, "!=", obj, op.attr, cl
                    )
                self.emitter.emit_dec_ref(attr_expr, attr_rtype)
                if not always_defined:
                    self.emitter.emit_line("}")
            self.emitter.emit_line(f"{attr_expr} = {src};")
            if not op.is_init and attr_rtype.is_refcounted:
                if not always_defined:
                    self.emitter.emit_line("if (!({})) {{".format(
                        self.emitter.error_value_check(attr_rtype, attr_expr, "==")
                    ))
                self.emitter.emit_inc_ref(attr_expr, attr_rtype)
                if not always_defined:
                    self.emitter.emit_line("}")

    def visit_set_element(self, op: SetElement) -> None:
        # TODO: Always decref then incref? Consider parts that might not execute
        # if above exception occurs.
        base = self.reg(op.base)
        index = self.reg(op.index)
        src = self.reg(op.src)
        # NOTE: We don't need to emit an exception check here because any exception
        # will propagate through the next error check, so it doesn't need a traceback.
        # It's only necessary to emit error checks for the latter if we modify this
        # in future and allow the traceback entry for the current op to be None.
        op_expr = f"CPyList_SetItemUnsafe({base}, {index}, {src});"
        self.emit_line(op_expr)

    def visit_set_mem(self, op: SetMem) -> None:
        dest = self.reg(op.dest)
        src = self.reg(op.src)
        self.emitter.emit_line(f"*{dest} = {src};")

    def visit_init_static(self, op: InitStatic) -> None:
        # When initializing a static variable, just assign it directly.
        self.emit_line(f"{self.reg(op.dest)} = {self.reg(op.src)};")

    def visit_tuple_get(self, op: TupleGet) -> None:
        if op.src.type.is_refcounted:
            self.emitter.emit_inc_ref(self.reg(op.src), op.src.type)
        item_rtype = op.src_type
        if isinstance(item_rtype, RStruct):
            # Access an unboxed tuple struct directly
            self.emit_line(f"{self.reg(op)} = {self.reg(op.src)}.f{op.index};")
        else:
            # Access a boxed tuple using its C API
            # TODO: We should be using a more efficient method for this
            self.emit_line(f"{self.reg(op)} = PyTuple_GetItem({self.reg(op.src)}, {op.index});")
            # We don't need to check PyTuple_GetItem for error because we already
            # know that the tuple is the right size
            self.emitter.emit_inc_ref(self.reg(op), item_rtype)

    def visit_keep_alive(self, op: KeepAlive) -> None:
        # Generate a branch that is always false to keep something alive without
        # adding any run-time overhead.
        self.emit_line("if (0) {}".format(self.reg(op.src)))

    def visit_primitive_op(self, op: PrimitiveOp) -> None:
        # These are special and can handle return values of different types.
        if op.prim_op in (int_op, int_op_unsafe):
            assert len(op.args) == 1
            self.emit_line("%s = %s;" % (self.reg(op), self.reg(op.args[0])))
        else:
            args = ", ".join(self.reg(arg) for arg in op.args)
            if op.prim_op in (ckres_op, ckres_int_op, ckres_sized_op, ckres_nop_exc_op):
                # Special case: these ops return an int directly (0 or 1, typically)
                # where 0 indicates error. This is not a special "error" value.
                result = f"{self.reg(op)} = {op.prim_op.c_function_name}({args});"
                self.emit_line(result)

                # If the op has a traceback position, check it immediately. We can't
                # use emit_line_loan_with_traceback since we need the result for
                # the check. Note that we assume the op is not borrowed (it isn't
                # currently possible for a primitive op to return a borrowed value).
                if op.traceback_entry is not None:
                    self.emitter.emit_traceback(self.source_path, self.module_name, op.traceback_entry)
                    self.emitter.emit_line(
                        f"if (unlikely({self.reg(op)} == 2)) goto {self.label(op.error_label)};"
                    )
            else:
                self.emit_line(
                    f"{self.reg(op)} = {op.prim_op.c_function_name}({args});",
                    ann=op.value,
                )

                if op.traceback_entry is not None:
                    self.emitter.emit_traceback(self.source_path, self.module_name, op.traceback_entry)
                    self.emitter.emit_line(
                        f"if (unlikely({self.reg(op)} == {self.c_error_value(op.type)})) goto {self.label(op.error_label)};"
                    )

    def visit_raise_standard_error(self, op: RaiseStandardError) -> None:
        # This includes temporary variable declarations for some of the
        # error types that need a variable to be passed to the generic
        # emit_type_error_traceback function.
        rtype = op.class_ir
        if rtype is None:
            rtype = op.value.type
        if isinstance(rtype, RInstance):
            rtype = rtype.class_ir
        error_kind = op.error_kind
        if error_kind == ERR_FALSE:
            rtype = None  # type: ignore[assignment]
        if error_kind == ERR_FALSE and op.prim_op is not None:
            # Special case: these ops return an int directly (0 or 1, typically)
            # where 0 indicates error. This is not a special "error" value.
            # So we need to ignore the type in those cases.
            rtype = None  # type: ignore[assignment]
        if rtype is not None and rtype.cdef is not None:
            name = rtype.cdef.name
        elif error_kind == ERR_FALSE:
            name = None
        elif error_kind == ERR_NEG_INT:
            name = "OverflowError"
        elif error_kind == ERR_MAGIC:
            name = "StopIteration"
        elif error_kind == ERR_FALSE:
            name = "AssertionError"
        else:
            name = ""

        if error_kind == ERR_MAGIC:
            # TODO: We could move some of this logic into the traceback handling code
            # to avoid doing it twice.
            if op.traceback_entry is not None:
                self.emitter.emit_traceback(self.source_path, self.module_name, op.traceback_entry)
            self.emitter.emit_line(
                "if (unlikely(PyErr_Occurred() != NULL)) goto %s;" % self.label(op.error_label)
            )
        else:
            if op.traceback_entry is not None:
                if rtype is None:
                    self.emitter.emit_traceback(self.source_path, self.module_name, op.traceback_entry)
                    self.emitter.emit_line(
                        "if (unlikely(PyErr_Occurred() != NULL)) goto %s;"
                        % self.label(op.error_label)
                    )
                else:
                    self.emit_type_error_traceback(rtype, self.reg(op.value), op.traceback_entry)

            # Emit the error handler call or just set the exception
            if op.traceback_entry is None:
                # If there is no traceback, we can just set the error and return.
                if error_kind == ERR_FALSE:
                    # With ERR_FALSE, the caller already set the error.
                    self.emitter.emit_line(f"{self.reg(op)} = 0;")
                elif error_kind == ERR_NEG_INT:
                    self.emitter.emit_line(f"{self.reg(op)} = -1;")
                elif error_kind == ERR_MAGIC:
                    self.emitter.emit_line(f"{self.reg(op)} = -1;")
                else:
                    self.emitter.emit_line(f"{self.reg(op)} = 0;")
                return

            # If there is a traceback, call the error handler.
            if error_kind == ERR_MAGIC:
                if op.traceback_entry is not None:
                    self.emit_line("goto %s;" % self.label(op.error_label))
            elif error_kind == ERR_FALSE:
                # With ERR_FALSE, the caller already set the error.
                if op.traceback_entry is not None:
                    self.emit_line("goto %s;" % self.label(op.error_label))
            else:
                if rtype is None:
                    if op.prim_op is not None:
                        self.emit_line("goto %s;" % self.label(op.error_label))
                    else:
                        self.emit_line("goto %s;" % self.label(op.error_label))
                else:
                    self.emit_type_error_traceback(rtype, self.reg(op.value), op.traceback_entry)
                    self.emit_line("goto %s;" % self.label(op.error_label))
