"""Utilities for emitting C code."""

from __future__ import annotations

import pprint
import sys
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from mypyc.codegen.cstring import c_string_initializer
from mypyc.codegen.literals import Literals
from mypyc.common import (
    ATTR_PREFIX,
    BITMAP_BITS,
    FAST_ISINSTANCE_MAX_SUBCLASSES,
    HAVE_IMMORTAL,
    MODULE_PREFIX,
    NATIVE_PREFIX,
    PREFIX,
    REG_PREFIX,
    STATIC_PREFIX,
    TYPE_PREFIX,
)
from mypyc.ir.class_ir import ClassIR, all_concrete_classes
from mypyc.ir.func_ir import FUNC_STATICMETHOD, FuncDecl, FuncIR, get_text_signature
from mypyc.ir.ops import BasicBlock, Value
from mypyc.ir.rtypes import (
    RInstance,
    RPrimitive,
    RTuple,
    RType,
    RUnion,
    int_rprimitive,
    is_bool_or_bit_rprimitive,
    is_bytearray_rprimitive,
    is_bytes_rprimitive,
    is_dict_rprimitive,
    is_fixed_width_rtype,
    is_float_rprimitive,
    is_frozenset_rprimitive,
    is_int16_rprimitive,
    is_int32_rprimitive,
    is_int64_rprimitive,
    is_int_rprimitive,
    is_list_rprimitive,
    is_native_rprimitive,
    is_none_rprimitive,
    is_object_rprimitive,
    is_optional_type,
    is_range_rprimitive,
    is_set_rprimitive,
    is_short_int_rprimitive,
    is_str_rprimitive,
    is_tuple_rprimitive,
    is_uint8_rprimitive,
    object_rprimitive,
    optional_value_type,
)
from mypyc.namegen import NameGenerator, exported_name
from mypyc.sametype import is_same_type

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

# Whether to insert debug asserts for all error handling, to quickly
# catch errors propagating without exceptions set.
DEBUG_ERRORS: Final = False


class HeaderDeclaration:
    """A representation of a declaration in C.

    This is used to generate declarations in header files and
    (optionally) definitions in source files.

    Attributes:
      decl: C source code for the declaration.
      defn: Optionally, C source code for a definition.
      dependencies: The names of any objects that must be declared prior.
      is_type: Whether the declaration is of a C type. (C types will be declared in
               external header files and not marked 'extern'.)
      needs_export: Whether the declared object needs to be exported to
                    other modules in the linking table.
    """

    def __init__(
        self,
        decl: str | list[str],
        defn: list[str] | None = None,
        *,
        dependencies: set[str] | None = None,
        is_type: bool = False,
        needs_export: bool = False,
    ) -> None:
        self.decl = [decl] if isinstance(decl, str) else decl
        self.defn = defn
        self.dependencies = dependencies or set()
        self.is_type = is_type
        self.needs_export = needs_export


class EmitterContext:
    """Shared emitter state for a compilation group."""

    def __init__(
        self,
        names: NameGenerator,
        group_name: str | None = None,
        group_map: dict[str, str | None] | None = None,
    ) -> None:
        """Setup shared emitter state.

        Args:
            names: The name generator to use
            group_map: Map from module names to group name
            group_name: Current group name
        """
        self.temp_counter = 0
        self.names = names
        self.group_name = group_name
        self.group_map = group_map or {}
        # Groups that this group depends on
        self.group_deps: set[str] = set()

        # The map below is used for generating declarations and
        # definitions at the top of the C file. The main idea is that they can
        # be generated at any time during the emit phase.

        # A map of a C identifier to whatever the C identifier declares. Currently this is
        # used for declaring structs and the key corresponds to the name of the struct.
        # The declaration contains the body of the struct.
        self.declarations: dict[str, HeaderDeclaration] = {}

        self.literals = Literals()


class ErrorHandler:
    """Describes handling errors in unbox/cast operations."""


class AssignHandler(ErrorHandler):
    """Assign an error value on error."""


class GotoHandler(ErrorHandler):
    """Goto label on error."""

    def __init__(self, label: str) -> None:
        self.label = label


class TracebackAndGotoHandler(ErrorHandler):
    """Add traceback item and goto label on error."""

    def __init__(
        self, label: str, source_path: str, module_name: str, traceback_entry: tuple[str, int]
    ) -> None:
        self.label = label
        self.source_path = source_path
        self.module_name = module_name
        self.traceback_entry = traceback_entry


class ReturnHandler(ErrorHandler):
    """Return a constant value on error."""

    def __init__(self, value: str) -> None:
        self.value = value


class Emitter:
    """Helper for C code generation."""

    def __init__(
        self,
        context: EmitterContext,
        value_names: dict[Value, str] | None = None,
        capi_version: tuple[int, int] | None = None,
        filepath: str | None = None,
    ) -> None:
        self.context = context
        self.capi_version = capi_version or sys.version_info[:2]
        self.names = context.names
        self.value_names = value_names or {}
        self.fragments: list[str] = []
        self._indent = 0
        self.filepath = filepath

    # Low-level operations

    def indent(self) -> None:
        self._indent += 4

    def dedent(self) -> None:
        self._indent -= 4
        assert self._indent >= 0

    def label(self, label: BasicBlock) -> str:
        return "CPyL%s" % label.label

    def reg(self, reg: Value) -> str:
        return REG_PREFIX + self.value_names[reg]

    def attr(self, name: str) -> str:
        return ATTR_PREFIX + name

    def object_annotation(self, obj: object, line: str) -> str:
        """Build a C comment with an object's string representation.

        If the comment exceeds the line length limit, it's wrapped into a
        multiline string (with the extra lines indented to be aligned with
        the first line's comment).

        If it contains illegal characters, an empty string is returned."""
        line_width = self._indent + len(line)
        formatted = pformat_deterministic(obj, max(90 - line_width, 20))

        if any(x in formatted for x in ("/*", "*/", "\0")):
            return ""

        if "\n" in formatted:
            first_line, rest = formatted.split("\n", maxsplit=1)
            comment_continued = textwrap.indent(rest, (line_width + 3) * " ")
            return f" /* {first_line}\n{comment_continued} */"
        else:
            return f" /* {formatted} */"

    def emit_line(self, line: str = "", *, ann: object = None) -> None:
        if line.startswith("}"):
            self.dedent()
        comment = self.object_annotation(ann, line) if ann is not None else ""
        self.fragments.append(self._indent * " " + line + comment + "\n")
        if line.endswith("{"):
            self.indent()

    def emit_lines(self, *lines: str) -> None:
        for line in lines:
            self.emit_line(line)

    def emit_label(self, label: BasicBlock | str) -> None:
        if isinstance(label, str):
            text = label
        else:
            if label.label == 0 or not label.referenced:
                return

            text = self.label(label)
        # Extra semicolon prevents an error when the next line declares a tempvar
        self.fragments.append(f"{text}: ;\n")

    def emit_from_emitter(self, emitter: Emitter) -> None:
        self.fragments.extend(emitter.fragments)

    def emit_printf(self, fmt: str, *args: str) -> None:
        fmt = fmt.replace("\n", "\\n")
        self.emit_line("printf(%s);" % ", ".join(['"%s"' % fmt] + list(args)))
        self.emit_line("fflush(stdout);")

    def temp_name(self) -> str:
        self.context.temp_counter += 1
        return "__tmp%d" % self.context.temp_counter

    def new_label(self) -> str:
        self.context.temp_counter += 1
        return "__LL%d" % self.context.temp_counter

    def get_module_group_prefix(self, module_name: str) -> str:
        """Get the group prefix for a module (relative to the current group).

        The prefix should be prepended to the object name whenever
        accessing an object from this module.

        If the module lives is in the current compilation group, there is
        no prefix.  But if it lives in a different group (and hence a separate
        extension module), we need to access objects from it indirectly via an
        export table.

        For example, for code in group `a` to call a function `bar` in group `b`,
        it would need to do `exports_b.CPyDef_bar(...)`, while code that is
        also in group `b` can simply do `CPyDef_bar(...)`.

        Thus the prefix for a module in group `b` is 'exports_b.' if the current
        group is *not* b and just '' if it is.
        """
        groups = self.context.group_map
        target_group_name = groups.get(module_name)
        if target_group_name and target_group_name != self.context.group_name:
            self.context.group_deps.add(target_group_name)
            return f"exports_{exported_name(target_group_name)}."
        else:
            return ""

    def get_group_prefix(self, obj: ClassIR | FuncDecl) -> str:
        """Get the group prefix for an object."""
        # See docs above
        return self.get_module_group_prefix(obj.module_name)

    def static_name(self, id: str, module: str | None, prefix: str = STATIC_PREFIX) -> str:
        """Create name of a C static variable.

        These are used for literals and imported modules, among other
        things.

        The caller should ensure that the (id, module) pair cannot
        overlap with other calls to this method within a compilation
        group.
        """
        lib_prefix = "" if not module else self.get_module_group_prefix(module)
        # If we are accessing static via the export table, we need to dereference
        # the pointer also.
        star_maybe = "*" if lib_prefix else ""
        suffix = self.names.private_name(module or "", id)
        return f"{star_maybe}{lib_prefix}{prefix}{suffix}"

    def type_struct_name(self, cl: ClassIR) -> str:
        return self.static_name(cl.name, cl.module_name, prefix=TYPE_PREFIX)

    def ctype(self, rtype: RType) -> str:
        return rtype._ctype

    def ctype_spaced(self, rtype: RType) -> str:
        """Adds a space after ctype for non-pointers."""
        ctype = self.ctype(rtype)
        if ctype[-1] == "*":
            return ctype
        else:
            return ctype + " "

    def c_undefined_value(self, rtype: RType) -> str:
        if not rtype.is_unboxed:
            return "NULL"
        elif isinstance(rtype, RPrimitive):
            return rtype.c_undefined
        elif isinstance(rtype, RTuple):
            return self.tuple_undefined_value(rtype)
        assert False, rtype

    def c_error_value(self, rtype: RType) -> str:
        return self.c_undefined_value(rtype)

    def native_function_name(self, fn: FuncDecl) -> str:
        return f"{NATIVE_PREFIX}{fn.cname(self.names)}"

    def tuple_c_declaration(self, rtuple: RTuple) -> list[str]:
        result = [
            f"#ifndef MYPYC_DECLARED_{rtuple.struct_name}",
            f"#define MYPYC_DECLARED_{rtuple.struct_name}",
            f"typedef struct {rtuple.struct_name} {{",
        ]
        if len(rtuple.types) == 0:  # empty tuple
            # Empty tuples contain a flag so that they can still indicate
            # error values.
            result.append("int empty_struct_error_flag;")
        else:
            i = 0
            for typ in rtuple.types:
                result.append(f"{self.ctype_spaced(typ)}f{i};")
                i += 1
        result.append(f"}} {rtuple.struct_name};")
        result.append("#endif")
        result.append("")

        return result

    def bitmap_field(self, index: int) -> str:
        """Return C field name used for attribute bitmap."""
        n = index // BITMAP_BITS
        if n == 0:
            return "bitmap"
        return f"bitmap{n + 1}"

    def attr_bitmap_expr(self, obj: str, cl: ClassIR, index: int) -> str:
        """Return reference to the attribute definedness bitmap."""
        cast = f"({cl.struct_name(self.names)} *)"
        attr = self.bitmap_field(index)
        return f"({cast}{obj})->{attr}"

    def emit_attr_bitmap_set(
        self, value: str, obj: str, rtype: RType, cl: ClassIR, attr: str
    ) -> None:
        """Mark an attribute as defined in the attribute bitmap.

        Assumes that the attribute is tracked in the bitmap (only some attributes
        use the bitmap). If 'value' is not equal to the error value, do nothing.
        """
        self._emit_attr_bitmap_update(value, obj, rtype, cl, attr, clear=False)

    def emit_attr_bitmap_clear(self, obj: str, rtype: RType, cl: ClassIR, attr: str) -> None:
        """Mark an attribute as undefined in the attribute bitmap.

        Unlike emit_attr_bitmap_set, clear unconditionally.
        """
        self._emit_attr_bitmap_update("", obj, rtype, cl, attr, clear=True)

    def _emit_attr_bitmap_update(
        self, value: str, obj: str, rtype: RType, cl: ClassIR, attr: str, clear: bool
    ) -> None:
        if value:
            check = self.error_value_check(rtype, value, "==")
            self.emit_line(f"if (unlikely({check})) {{")
        index = cl.bitmap_attrs.index(attr)
        mask = 1 << (index & (BITMAP_BITS - 1))
        bitmap = self.attr_bitmap_expr(obj, cl, index)
        if clear:
            self.emit_line(f"{bitmap} &= ~{mask};")
        else:
            self.emit_line(f"{bitmap} |= {mask};")
        if value:
            self.emit_line("}")

    def emit_undefined_attr_check(
        self,
        rtype: RType,
        attr_expr: str,
        compare: str,
        obj: str,
        attr: str,
        cl: ClassIR,
        *,
        unlikely: bool = False,
    ) -> None:
        check = self.error_value_check(rtype, attr_expr, compare)
        if unlikely:
            check = f"unlikely({check})"
        if rtype.error_overlap:
            index = cl.bitmap_attrs.index(attr)
            bit = 1 << (index & (BITMAP_BITS - 1))
            attr = self.bitmap_field(index)
            obj_expr = f"({cl.struct_name(self.names)} *){obj}"
            check = f"{check} && !(({obj_expr})->{attr} & {bit})"
        self.emit_line(f"if ({check}) {{")

    def error_value_check(self, rtype: RType, value: str, compare: str) -> str:
        if isinstance(rtype, RTuple):
            return self.tuple_undefined_check_cond(
                rtype, value, self.c_error_value, compare, check_exception=False
            )
        else:
            return f"{value} {compare} {self.c_error_value(rtype)}"

    def tuple_undefined_check_cond(
        self,
        rtuple: RTuple,
        tuple_expr_in_c: str,
        c_type_compare_val: Callable[[RType], str],
        compare: str,
        *,
        check_exception: bool = True,
    ) -> str:
        if len(rtuple.types) == 0:
            # empty tuple
            return "{}.empty_struct_error_flag {} {}".format(
                tuple_expr_in_c, compare, c_type_compare_val(int_rprimitive)
            )
        if rtuple.error_overlap:
            i = 0
            item_type = rtuple.types[0]
        else:
            for i, typ in enumerate(rtuple.types):
                if not typ.error_overlap:
                    item_type = rtuple.types[i]
                    break
            else:
                assert False, "not expecting tuple with error overlap"
        if isinstance(item_type, RTuple):
            return self.tuple_undefined_check_cond(
                item_type, tuple_expr_in_c + f".f{i}", c_type_compare_val, compare
            )
        else:
            check = f"{tuple_expr_in_c}.f{i} {compare} {c_type_compare_val(item_type)}"
            if rtuple.error_overlap and check_exception:
                check += " && PyErr_Occurred()"
            return check

    def tuple_undefined_value(self, rtuple: RTuple) -> str:
        """Undefined tuple value suitable in an expression."""
        return f"({rtuple.struct_name}) {self.c_initializer_undefined_value(rtuple)}"

    def c_initializer_undefined_value(self, rtype: RType) -> str:
        """Undefined value represented in a form suitable for variable initialization."""
        if isinstance(rtype, RTuple):
            if not rtype.types:
                # Empty tuples contain a flag so that they can still indicate
                # error values.
                return f"{{ {int_rprimitive.c_undefined} }}"
            items = ", ".join([self.c_initializer_undefined_value(t) for t in rtype.types])
            return f"{{ {items} }}"
        else:
            return self.c_undefined_value(rtype)

    # Higher-level operations

    def declare_tuple_struct(self, tuple_type: RTuple) -> None:
        if tuple_type.struct_name not in self.context.declarations:
            dependencies = set()
            for typ in tuple_type.types:
                # XXX other types might eventually need similar behavior
                if isinstance(typ, RTuple):
                    dependencies.add(typ.struct_name)

            self.context.declarations[tuple_type.struct_name] = HeaderDeclaration(
                self.tuple_c_declaration(tuple_type), dependencies=dependencies, is_type=True
            )

    def emit_inc_ref(self, dest: str, rtype: RType, *, rare: bool = False) -> None:
        """Increment reference count of C expression `dest`.

        For composite unboxed structures (e.g. tuples) recursively
        increment reference counts for each component.

        If rare is True, optimize for code size and compilation speed.
        """
        if is_int_rprimitive(rtype):
            if rare:
                self.emit_line("CPyTagged_IncRef(%s);" % dest)
            else:
                self.emit_line("CPyTagged_INCREF(%s);" % dest)
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_inc_ref(f"{dest}.f{i}", item_type)
        elif not rtype.is_unboxed:
            # Always inline, since this is a simple but very hot op
            if rtype.may_be_immortal or not HAVE_IMMORTAL:
                self.emit_line("CPy_INCREF(%s);" % dest)
            else:
                self.emit_line("CPy_INCREF_NO_IMM(%s);" % dest)
        # Otherwise assume it's an unboxed, pointerless value and do nothing.

    def emit_dec_ref(
        self, dest: str, rtype: RType, *, is_xdec: bool = False, rare: bool = False
    ) -> None:
        """Decrement reference count of C expression `dest`.

        For composite unboxed structures (e.g. tuples) recursively
        decrement reference counts for each component.

        If rare is True, optimize for code size and compilation speed.
        """
        x = "X" if is_xdec else ""
        if is_int_rprimitive(rtype):
            if rare:
                self.emit_line("CPyTagged_DecRef(%s);" % dest)
            else:
                self.emit_line("CPyTagged_DECREF(%s);" % dest)
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_dec_ref(f"{dest}.f{i}", item_type, is_xdec=is_xdec)
        elif not rtype.is_unboxed:
            if rare:
                self.emit_line("CPy_DecRef(%s);" % dest)
            else:
                if rtype.may_be_immortal or not HAVE_IMMORTAL:
                    self.emit_line("CPy_DECREF(%s);" % dest)
                else:
                    self.emit_line("CPy_DECREF_NO_IMM(%s);" % dest)
            if is_xdec:
                self.emit_line("CPy_XDECREF(%s);" % dest)
        # Otherwise assume it's an unboxed, pointerless value and do nothing.

    def emit_dec_ref_no_non_zero(self, dest: str, rtype: RType) -> None:
        """Decrement reference count of C expression `dest`.

        For composite unboxed structures (e.g. tuples) recursively
        decrement reference counts for each component.

        Unlike emit_dec_ref(), when reference count becomes zero, don't call the
        tp_clear method (i.e. avoid decrefing foreign references).

        If rtype is a boxed object, use CPy_DECREF.
        """
        if is_int_rprimitive(rtype):
            self.emit_line("CPyTagged_DECREF(%s);" % dest)
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_dec_ref_no_non_zero(f"{dest}.f{i}", item_type)
        elif not rtype.is_unboxed:
            self.emit_line("CPy_DECREF(%s);" % dest)
        # Otherwise assume it's an unboxed, pointerless value and do nothing.

    def emit_inc_ref_rare(self, dest: str, rtype: RType) -> None:
        """Increment reference count of C expression `dest`.

        For composite unboxed structures (e.g. tuples) recursively
        increment reference counts for each component.

        But this always uses the rare variant of inc/decr ops.
        """
        if is_int_rprimitive(rtype):
            self.emit_line("CPyTagged_IncRef(%s);" % dest)
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_inc_ref_rare(f"{dest}.f{i}", item_type)
        elif not rtype.is_unboxed:
            self.emit_line("CPy_IncRef(%s);" % dest)
        # Otherwise assume it's an unboxed, pointerless value and do nothing.

    def emit_dec_ref_rare(self, dest: str, rtype: RType) -> None:
        """Decrement reference count of C expression `dest`.

        For composite unboxed structures (e.g. tuples) recursively
        decrement reference counts for each component.

        But this always uses the rare variant of inc/decr ops.
        """
        if is_int_rprimitive(rtype):
            self.emit_line("CPyTagged_DecRef(%s);" % dest)
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_dec_ref_rare(f"{dest}.f{i}", item_type)
        elif not rtype.is_unboxed:
            self.emit_line("CPy_DecRef(%s);" % dest)
        # Otherwise assume it's an unboxed, pointerless value and do nothing.

    def emit_attr_bitmap_check(self, rtype: RType, obj: str, attr: str, cl: ClassIR) -> str:
        """Check if attribute is present in the attribute bitmap.

        This returns a C bool expression.
        """
        # If attribute is not in bitmap, then it is always present.
        if attr not in cl.bitmap_attrs:
            return "1"
        index = cl.bitmap_attrs.index(attr)
        return f"({self.attr_bitmap_expr(obj, cl, index)} & {1 << (index & (BITMAP_BITS - 1))})"

    def emit_attr_bitmap_set_helper(
        self,
        obj: str,
        rtype: RType,
        cl: ClassIR,
        attr: str,
    ) -> str:
        """Emit a helper function that sets a bit in the attribute bitmap."""
        helper = f"CPyDef_{attr}_set"
        helper_name = f"{NATIVE_PREFIX}{helper}"
        proto = f"void {helper_name}({cl.struct_name(self.names)} *self, {self.ctype(rtype)} value)"
        self.context.declarations[helper_name] = HeaderDeclaration(
            proto + ";",
            dependencies={self.type_struct_name(cl)},
        )
        emitter = Emitter(self.context, self.value_names)
        emitter.emit_line(proto)
        emitter.emit_line("{")
        emitter.emit_attr_bitmap_set("value", "self", rtype, cl, attr)
        emitter.emit_line("}")
        self.emit_from_emitter(emitter)
        return helper

    def emit_attr_bitmap_clear_helper(
        self, obj: str, rtype: RType, cl: ClassIR, attr: str
    ) -> str:
        """Emit a helper function that clears a bit in the attribute bitmap."""
        helper = f"CPyDef_{attr}_clear"
        helper_name = f"{NATIVE_PREFIX}{helper}"
        proto = f"void {helper_name}({cl.struct_name(self.names)} *self)"
        self.context.declarations[helper_name] = HeaderDeclaration(
            proto + ";", dependencies={self.type_struct_name(cl)}
        )
        emitter = Emitter(self.context, self.value_names)
        emitter.emit_line(proto)
        emitter.emit_line("{")
        emitter.emit_attr_bitmap_clear("self", rtype, cl, attr)
        emitter.emit_line("}")
        self.emit_from_emitter(emitter)
        return helper

    def emit_method_address(self, op: str, obj: str, rtype: RType, name: str, cl: ClassIR) -> str:
        assert not is_bool_or_bit_rprimitive(rtype)
        cl = rtype.class_ir
        assert cl.get_method(name), f"{name} not found in {cl.name}"
        return f"{self.attr(name)}({obj})"

    def emit_cast_error(self, src: str, dest: RType, failure: str) -> None:
        """Emit runtime type checking for cast operations.

        Must only be used if RType is a pointer type.
        """
        if isinstance(dest, RInstance) and dest.class_ir.is_trait:
            self.emit_line(f"if (!PyObject_TypeCheck({src}, {self.type_struct_name(dest.class_ir)})) {{")
            self.emit_line(failure)
            self.emit_line("}")
            return

        # Some types have fast path for case where the object has a default
        # type, since the class may be subclassed. Fast path is case where
        # we don't need to check for subclasses. There is also a fast path
        # for checking the MRO.
        if isinstance(dest, RInstance) and not dest.class_ir.allow_interpreted_subclasses:
            # Check object's type directly. Note that cannot use type(obj) as
            # it would cause ambiguity with typing.Type, etc.
            self.emit_line(f"if (Py_TYPE({src}) == {self.type_struct_name(dest.class_ir)}) {{")
            self.emit_line("}")
            self.emit_line("else {")
            self.emit_line(failure)
            self.emit_line("}")
        elif isinstance(dest, RInstance) and len(dest.class_ir.mro) > FAST_ISINSTANCE_MAX_SUBCLASSES:
            # Check the object's mro for (direct or indirect) subclassing.
            # This is technically different from what PyObject_TypeCheck does
            # but should be OK given the acceptable types of generated classes.
            # (It should be a subset of those that are safe to check).
            type_expr = self.type_struct_name(dest.class_ir)
            self.emit_line(f"if (unlikely(!PyObject_TypeCheck({src}, {type_expr}))) {{")
            self.emit_line(failure)
            self.emit_line("}")
        else:
            # Check if object is an instance of a subclass
            instance_check = f"PyObject_TypeCheck({src}, {self.type_struct_name(dest.class_ir)})"
            self.emit_line(f"if (unlikely(!{instance_check})) {{")
            self.emit_line(failure)
            self.emit_line("}")

    def emit_cast_error_with_overlapping_error_value(
        self, src: str, dest: RType, failure: str
    ) -> None:
        # In this case, we can't test if the value is a valid as a normal value, so we need to also check
        # for a raised exception.
            self.emit_line(f"if ({value} == {self.c_error_value(rtype)} && PyErr_Occurred()) {{")
        else:
            self.emit_line(f"if ({value} == {self.c_error_value(rtype)}) {{")
        self.emit_lines(failure, "}")

    def emit_gc_visit(self, target: str, rtype: RType) -> None:
        """Emit code for GC visiting a C variable reference.

        Assume that 'target' represents a C expression that refers to a
        struct member, such as 'self->x'.
        """
        if not rtype.is_refcounted:
            # Not refcounted -> no pointers -> no GC interaction.
            return
        elif isinstance(rtype, RPrimitive) and rtype.name == "builtins.int":
            self.emit_line(f"if (CPyTagged_CheckLong({target})) {{")
            self.emit_line(f"Py_VISIT(CPyTagged_LongAsObject({target}));")
            self.emit_line("}")
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_gc_visit(f"{target}.f{i}", item_type)
        elif self.ctype(rtype) == "PyObject *":
            # The simplest case.
            self.emit_line(f"Py_VISIT({target});")
        else:
            assert False, "emit_gc_visit() not implemented for %s" % repr(rtype)

    def emit_gc_clear(self, target: str, rtype: RType) -> None:
        """Emit code for clearing a C attribute reference for GC.

        Assume that 'target' represents a C expression that refers to a
        struct member, such as 'self->x'.

        If a last_ref disappears we have to deal with resurrected objects. The source
        code of `tp_clear` cannot resurrect, so we can safely use `tp_clear`.
        """
        if not rtype.is_refcounted:
            # Not refcounted -> no pointers -> no GC interaction.
            return
        elif isinstance(rtype, RPrimitive) and rtype.name == "builtins.int":
            self.emit_line(f"if (CPyTagged_CheckLong({target})) {{")
            self.emit_line(f"CPyTagged __tmp = {target};")
            self.emit_line(f"{target} = {self.c_undefined_value(rtype)};")
            self.emit_line("Py_XDECREF(CPyTagged_LongAsObject(__tmp));")
            self.emit_line("}")
        elif isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_gc_clear(f"{target}.f{i}", item_type)
        elif self.ctype(rtype) == "PyObject *":
            # The simplest case.
            self.emit_line(f"Py_CLEAR({target});")
        else:
            assert False, "emit_gc_clear() not implemented for %s" % repr(rtype)

    def emit_reuse_clear(self, target: str, rtype: RType) -> None:
        """Emit attribute clear before object is added into freelist.

        Assume that 'target' represents a C expression that refers to a
        struct member, such as 'self->x'.

        Unlike emit_gc_clear(), initialize attribute value to match a freshly
        allocated object.
        """
        if isinstance(rtype, RTuple):
            for i, item_type in enumerate(rtype.types):
                self.emit_reuse_clear(f"{target}.f{i}", item_type)
        elif not rtype.is_refcounted:
            self.emit_line(f"{target} = {rtype.c_undefined};")
        elif isinstance(rtype, RPrimitive) and rtype.name == "builtins.int":
            self.emit_line(f"if (CPyTagged_CheckLong({target})) {{")
            self.emit_line(f"CPyTagged __tmp = {target};")
            self.emit_line(f"{target} = {self.c_undefined_value(rtype)};")
            self.emit_line("Py_XDECREF(CPyTagged_LongAsObject(__tmp));")
            self.emit_line("} else {")
            self.emit_line(f"{target} = {self.c_undefined_value(rtype)};")
            self.emit_line("}")
        else:
            self.emit_gc_clear(target, rtype)

    def emit_traceback(
        self, source_path: str, module_name: str, traceback_entry: tuple[str, int]
    ) -> None:
        return self._emit_traceback("CPy_AddTraceback", source_path, module_name, traceback_entry)

    def emit_type_error_traceback(
        self,
        source_path: str,
        module_name: str,
        traceback_entry: tuple[str, int],
        *,
        typ: RType,
        src: str,
    ) -> None:
        func = "CPy_TypeErrorTraceback"
        type_str = f'"{self.pretty_name(typ)}"'
        return self._emit_traceback(
            func, source_path, module_name, traceback_entry, type_str=type_str, src=src
        )

    def _emit_traceback(
        self,
        func: str,
        source_path: str,
        module_name: str,
        traceback_entry: tuple[str, int],
        type_str: str = "",
        src: str = "",
    ) -> None:
        globals_static = self.static_name("globals", module_name)
        line = '%s("%s", "%s", %d, %s' % (
            func,
            source_path.replace("\\", "\\\\"),
            traceback_entry[0],
            traceback_entry[1],
            globals_static,
        )
        if type_str:
            assert src
            line += f", {type_str}, {src}"
        line += ");"
        self.emit_line(line)
        if DEBUG_ERRORS:
            self.emit_line('assert(PyErr_Occurred() != NULL && "failure w/o err!");')

    def emit_unbox_failure_with_overlapping_error_value(
        self, dest: str, typ: RType, failure: str
    ) -> None:
        self.emit_line(f"if ({dest} == {self.c_error_value(typ)} && PyErr_Occurred()) {{")
        self.emit_line(failure)
        self.emit_line("}")

    def emit_cpyfunction_instance(
        self, fn: FuncIR, name: str, filepath: str, error_stmt: str
    ) -> str:
        module = self.static_name(fn.decl.module_name, None, prefix=MODULE_PREFIX)
        cname = f"{PREFIX}{fn.cname(self.names)}"
        wrapper_name = f"{cname}_wrapper"
        cfunc = f"(PyCFunction){cname}"
        func_flags = "METH_FASTCALL | METH_KEYWORDS"
        doc = f"PyDoc_STR({native_function_doc_initializer(fn)})"
        has_self_arg = "true" if fn.class_name and fn.decl.kind != FUNC_STATICMETHOD else "false"

        code_flags = "CO_COROUTINE"
        self.emit_line(
            f'PyObject* {wrapper_name} = CPyFunction_New({module}, "{filepath}", "{name}", {cfunc}, {func_flags}, {doc}, {fn.line}, {code_flags}, {has_self_arg});'
        )
        self.emit_line(f"if (unlikely(!{wrapper_name}))")
        self.emit_line(error_stmt)
        return wrapper_name


def c_array_initializer(components: list[str], *, indented: bool = False) -> str:
    """Construct an initializer for a C array variable.

    Components are C expressions valid in an initializer.

    For example, if components are ["1", "2"], the result
    would be "{1, 2}", which can be used like this:

        int a[] = {1, 2};

    If the result is long, split it into multiple lines.
    """
    indent = " " * 4 if indented else ""
    res = []
    current: list[str] = []
    cur_len = 0
    for c in components:
        if not current or cur_len + 2 + len(indent) + len(c) < 70:
            current.append(c)
            cur_len += len(c) + 2
        else:
            res.append(indent + ", ".join(current))
            current = [c]
            cur_len = len(c)
    if not res:
        # Result fits on a single line
        return "{%s}" % ", ".join(current)
    # Multi-line result
    res.append(indent + ", ".join(current))
    return "{\n    " + ",\n    ".join(res) + "\n" + indent + "}"


def native_function_doc_initializer(func: FuncIR) -> str:
    text_sig = get_text_signature(func)
    if text_sig is None:
        return "NULL"
    docstring = f"{text_sig}\n--\n\n"
    return c_string_initializer(docstring.encode("ascii", errors="backslashreplace"))


def pformat_deterministic(obj: object, width: int) -> str:
    """Pretty-print `obj` with deterministic sorting for mypyc literal types."""
    printer = _DeterministicPrettyPrinter(width=width, compact=True, sort_dicts=True)
    return printer.pformat(obj)


def _literal_sort_key(obj: object) -> tuple[str, object]:
    """Return a deterministic sort key for mypyc literal types."""
    if isinstance(obj, tuple):
        return ("tuple", tuple(_literal_sort_key(item) for item in obj))
    if isinstance(obj, frozenset):
        items = sorted((_literal_sort_key(item) for item in obj))
        return ("frozenset", tuple(items))
    return (type(obj).__name__, repr(obj))


def _mypyc_safe_key(obj: object) -> tuple[str, object]:
    """A custom sort key implementation for pprint that makes output deterministic."""
    return _literal_sort_key(obj)


class _DeterministicPrettyPrinter(pprint.PrettyPrinter):
    """PrettyPrinter that sorts set/frozenset elements deterministically."""

    _dispatch = pprint.PrettyPrinter._dispatch.copy()

    def format(
        self, object: object, context: dict[int, int], maxlevels: int | None, level: int
    ) -> tuple[str, bool, bool]:
        if isinstance(object, (set, frozenset)) and type(object).__repr__ in (
            set.__repr__,
            frozenset.__repr__,
        ):
            return self._safe_set_repr(object, context, maxlevels, level)
        return super().format(object, context, maxlevels, level)

    def _safe_set_repr(
        self,
        object: set[object] | frozenset[object],
        context: dict[int, int],
        maxlevels: int | None,
        level: int,
    ) -> tuple[str, bool, bool]:
        if not object:
            return repr(object), True, False
        objid = id(object)
        if maxlevels and level >= maxlevels:
            if isinstance(object, frozenset):
                return "frozenset({...})", False, objid in context
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        components = []
        level += 1
        for item in sorted(object, key=_mypyc_safe_key):
            item_repr, item_readable, item_recursive = self.format(
                item, context, maxlevels, level
            )
            components.append(item_repr)
            readable = readable and item_readable
            if item_recursive:
                recursive = True
        del context[objid]
        if isinstance(object, frozenset):
            return f"frozenset({{{', '.join(components)}}})", readable, recursive
        return "{" + ", ".join(components) + "}", readable, recursive

    def _pprint_set(
        self,
        object: set[object] | frozenset[object],
        stream: SupportsWrite[str],
        indent: int,
        allowance: int,
        context: dict[int, int],
        level: int,
    ) -> None:
        if not object:
            stream.write(repr(object))
            return
        typ = type(object)
        if typ is set:
            stream.write("{")
            endchar = "}"
        else:
            stream.write("frozenset({")
            endchar = "})"
            indent += len("frozenset(")
        items = sorted(object, key=_mypyc_safe_key)
        self._format_items(items, stream, indent, allowance + len(endchar), context, level)
        stream.write(endchar)

    _dispatch[set.__repr__] = _pprint_set
    _dispatch[frozenset.__repr__] = _pprint_set
