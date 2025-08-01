"""Transform mypy statement ASTs to mypyc IR (Intermediate Representation).

The top-level AST transformation logic is implemented in mypyc.irbuild.visitor
and mypyc.irbuild.builder.

A few statements are transformed in mypyc.irbuild.function (yield, for example).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Callable

import mypy.nodes
from mypy.nodes import (
    ARG_NAMED,
    ARG_POS,
    AssertStmt,
    AssignmentStmt,
    AwaitExpr,
    Block,
    BreakStmt,
    ContinueStmt,
    DelStmt,
    Expression,
    ExpressionStmt,
    ForStmt,
    IfStmt,
    Import,
    ImportAll,
    ImportFrom,
    ListExpr,
    Lvalue,
    MatchStmt,
    OperatorAssignmentStmt,
    RaiseStmt,
    ReturnStmt,
    StarExpr,
    StrExpr,
    TempNode,
    TryStmt,
    TupleExpr,
    TypeAliasStmt,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)
from mypyc.common import TEMP_ATTR_NAME
from mypyc.ir.ops import (
    ERR_NEVER,
    NAMESPACE_MODULE,
    NO_TRACEBACK_LINE_NO,
    Assign,
    BasicBlock,
    Branch,
    Call,
    InitStatic,
    Integer,
    LoadAddress,
    LoadErrorValue,
    LoadLiteral,
    LoadStatic,
    MethodCall,
    PrimitiveDescription,
    RaiseStandardError,
    Register,
    Return,
    TupleGet,
    Unborrow,
    Unreachable,
    Value,
)
from mypyc.ir.rtypes import (
    RInstance,
    RTuple,
    c_pyssize_t_rprimitive,
    exc_rtuple,
    is_tagged,
    none_rprimitive,
    object_pointer_rprimitive,
    object_rprimitive,
)
from mypyc.irbuild.ast_helpers import is_borrow_friendly_expr, process_conditional
from mypyc.irbuild.builder import IRBuilder, create_type_params, int_borrow_friendly_op
from mypyc.irbuild.for_helpers import for_loop_helper
from mypyc.irbuild.generator import add_raise_exception_blocks_to_generator_class
from mypyc.irbuild.nonlocalcontrol import (
    ExceptNonlocalControl,
    FinallyNonlocalControl,
    TryFinallyNonlocalControl,
)
from mypyc.irbuild.prepare import GENERATOR_HELPER_NAME
from mypyc.irbuild.targets import (
    AssignmentTarget,
    AssignmentTargetAttr,
    AssignmentTargetIndex,
    AssignmentTargetRegister,
    AssignmentTargetTuple,
)
from mypyc.primitives.exc_ops import (
    error_catch_op,
    exc_matches_op,
    get_exc_info_op,
    get_exc_value_op,
    keep_propagating_op,
    no_err_occurred_op,
    propagate_if_error_op,
    raise_exception_op,
    reraise_exception_op,
    restore_exc_info_op,
)
from mypyc.primitives.generic_ops import iter_op, next_raw_op, py_delattr_op
from mypyc.primitives.misc_ops import (
    check_stop_op,
    coro_op,
    import_from_many_op,
    import_many_op,
    import_op,
    send_op,
    set_type_alias_compute_function_op,
    type_op,
    yield_from_except_op,
)

from .match import MatchVisitor

GenFunc = Callable[[], None]
ValueGenFunc = Callable[[], Value]


def transform_block(builder: IRBuilder, block: Block) -> None:
    if not block.is_unreachable:
        builder.block_reachable_stack.append(True)
        for stmt in block.body:
            builder.accept(stmt)
            if not builder.block_reachable_stack[-1]:
                # The rest of the block is unreachable, so skip it
                break
        builder.block_reachable_stack.pop()
    # Raise a RuntimeError if we hit a non-empty unreachable block.
    # Don't complain about empty unreachable blocks, since mypy inserts
    # those after `if MYPY`.
    elif block.body:
        builder.add(
            RaiseStandardError(
                RaiseStandardError.RUNTIME_ERROR, "Reached allegedly unreachable code!", block.line
            )
        )
        builder.add(Unreachable())


def transform_expression_stmt(builder: IRBuilder, stmt: ExpressionStmt) -> None:
    if isinstance(stmt.expr, StrExpr):
        # Docstring. Ignore
        return
    # ExpressionStmts do not need to be coerced like other Expressions, so we shouldn't
    # call builder.accept here.
    stmt.expr.accept(builder.visitor)
    builder.flush_keep_alives()


def transform_return_stmt(builder: IRBuilder, stmt: ReturnStmt) -> None:
    if stmt.expr:
        retval = builder.accept(stmt.expr)
    else:
        retval = builder.builder.none()
    retval = builder.coerce(retval, builder.ret_types[-1], stmt.line)
    builder.nonlocal_control[-1].gen_return(builder, retval, stmt.line)


def transform_assignment_stmt(builder: IRBuilder, stmt: AssignmentStmt) -> None:
    lvalues = stmt.lvalues
    assert lvalues
    builder.disallow_class_assignments(lvalues, stmt.line)
    first_lvalue = lvalues[0]
    if stmt.type and isinstance(stmt.rvalue, TempNode):
        # This is actually a variable annotation without initializer. Don't generate
        # an assignment but we need to call get_assignment_target since it adds a
        # name binding as a side effect.
        builder.get_assignment_target(first_lvalue, stmt.line)
        return

    # Special case multiple assignments like 'x, y = e1, e2'.
    if (
        isinstance(first_lvalue, (TupleExpr, ListExpr))
        and isinstance(stmt.rvalue, (TupleExpr, ListExpr))
        and len(first_lvalue.items) == len(stmt.rvalue.items)
        and all(is_simple_lvalue(item) for item in first_lvalue.items)
        and len(lvalues) == 1
    ):
        temps = []
        for right in stmt.rvalue.items:
            rvalue_reg = builder.accept(right)
            temp = Register(rvalue_reg.type)
            builder.assign(temp, rvalue_reg, stmt.line)
            temps.append(temp)
        for left, temp in zip(first_lvalue.items, temps):
            assignment_target = builder.get_assignment_target(left)
            builder.assign(assignment_target, temp, stmt.line)
        builder.flush_keep_alives()
        return

    line = stmt.rvalue.line
    rvalue_reg = builder.accept(stmt.rvalue)

    if builder.non_function_scope() and stmt.is_final_def:
        builder.init_final_static(first_lvalue, rvalue_reg)

    # Special-case multiple assignments like 'x, y = expr' to reduce refcount ops.
    if (
        isinstance(first_lvalue, (TupleExpr, ListExpr))
        and isinstance(rvalue_reg.type, RTuple)
        and len(rvalue_reg.type.types) == len(first_lvalue.items)
        and len(lvalues) == 1
        and all(is_simple_lvalue(item) for item in first_lvalue.items)
        and any(t.is_refcounted for t in rvalue_reg.type.types)
    ):
        n = len(first_lvalue.items)
        borrows = [builder.add(TupleGet(rvalue_reg, i, borrow=True)) for i in range(n)]
        builder.builder.keep_alive([rvalue_reg], steal=True)
        for lvalue_item, rvalue_item in zip(first_lvalue.items, borrows):
            rvalue_item = builder.add(Unborrow(rvalue_item))
            builder.assign(builder.get_assignment_target(lvalue_item), rvalue_item, line)
        builder.flush_keep_alives()
        return

    for lvalue in lvalues:
        target = builder.get_assignment_target(lvalue)
        builder.assign(target, rvalue_reg, line)
        builder.flush_keep_alives()


def is_simple_lvalue(expr: Expression) -> bool:
    return not isinstance(expr, (StarExpr, ListExpr, TupleExpr))


def transform_operator_assignment_stmt(builder: IRBuilder, stmt: OperatorAssignmentStmt) -> None:
    """Operator assignment statement such as x += 1"""
    builder.disallow_class_assignments([stmt.lvalue], stmt.line)
    if (
        is_tagged(builder.node_type(stmt.lvalue))
        and is_tagged(builder.node_type(stmt.rvalue))
        and stmt.op in int_borrow_friendly_op
    ):
        can_borrow = is_borrow_friendly_expr(builder, stmt.rvalue) and is_borrow_friendly_expr(
            builder, stmt.lvalue
        )
    else:
        can_borrow = False
    target = builder.get_assignment_target(stmt.lvalue)
    target_value = builder.read(target, stmt.line, can_borrow=can_borrow)
    rreg = builder.accept(stmt.rvalue, can_borrow=can_borrow)
    # the Python parser strips the '=' from operator assignment statements, so re-add it
    op = stmt.op + "="
    res = builder.binary_op(target_value, rreg, op, stmt.line)
    # usually operator assignments are done in-place
    # but when target doesn't support that we need to manually assign
    builder.assign(target, res, res.line)
    builder.flush_keep_alives()


def import_globals_id_and_name(module_id: str, as_name: str | None) -> tuple[str, str]:
    """Compute names for updating the globals dict with the appropriate module.

    * For 'import foo.bar as baz' we add 'foo.bar' with the name 'baz'
    * For 'import foo.bar' we add 'foo' with the name 'foo'

    Typically we then ignore these entries and access things directly
    via the module static, but we will use the globals version for
    modules that mypy couldn't find, since it doesn't analyze module
    references from those properly."""
    if as_name:
        globals_id = module_id
        globals_name = as_name
    else:
        globals_id = globals_name = module_id.split(".")[0]

    return globals_id, globals_name


def transform_import(builder: IRBuilder, node: Import) -> None:
    if node.is_mypy_only:
        return

    # Imports (not from imports!) are processed in an odd way so they can be
    # table-driven and compact. Here's how it works:
    #
    # Import nodes are divided in groups (in the prebuild visitor). Each group
    # consists of consecutive Import nodes:
    #
    #   import mod         <| group #1
    #   import mod2         |
    #
    #   def foo() -> None:
    #       import mod3    <- group #2 (*)
    #
    #   import mod4        <| group #3
    #   import mod5         |
    #
    # Every time we encounter the first import of a group, build IR to call a
    # helper function that will perform all of the group's imports in one go.
    if not node.is_top_level:
        # (*) Unless the import is within a function. In that case, prioritize
        # speed over codesize when generating IR.
        globals = builder.load_globals_dict()
        for mod_id, as_name in node.ids:
            builder.gen_import(mod_id, node.line)
            globals_id, globals_name = import_globals_id_and_name(mod_id, as_name)
            builder.gen_method_call(
                globals,
                "__setitem__",
                [builder.load_str(globals_name), builder.get_module(globals_id, node.line)],
                result_type=None,
                line=node.line,
            )
        return

    if node not in builder.module_import_groups:
        return

    modules = []
    static_ptrs = []
    # To show the right line number on failure, we have to add the traceback
    # entry within the helper function (which is admittedly ugly). To drive
    # this, we need the line number corresponding to each module.
    mod_lines = []
    for import_node in builder.module_import_groups[node]:
        for mod_id, as_name in import_node.ids:
            builder.imports[mod_id] = None
            modules.append((mod_id, *import_globals_id_and_name(mod_id, as_name)))
            mod_static = LoadStatic(object_rprimitive, mod_id, namespace=NAMESPACE_MODULE)
            static_ptrs.append(builder.add(LoadAddress(object_pointer_rprimitive, mod_static)))
            mod_lines.append(Integer(import_node.line, c_pyssize_t_rprimitive))

    static_array_ptr = builder.builder.setup_rarray(object_pointer_rprimitive, static_ptrs)
    import_line_ptr = builder.builder.setup_rarray(c_pyssize_t_rprimitive, mod_lines)
    builder.call_c(
        import_many_op,
        [
            builder.add(LoadLiteral(tuple(modules), object_rprimitive)),
            static_array_ptr,
            builder.load_globals_dict(),
            builder.load_str(builder.module_path),
            builder.load_str(builder.fn_info.name),
            import_line_ptr,
        ],
        NO_TRACEBACK_LINE_NO,
    )


def transform_import_from(builder: IRBuilder, node: ImportFrom) -> None:
    if node.is_mypy_only:
        return

    module_state = builder.graph[builder.module_name]
    if builder.module_path.endswith("__init__.py"):
        module_package = builder.module_name
    elif module_state.ancestors is not None and module_state.ancestors:
        module_package = module_state.ancestors[0]
    else:
        module_package = ""

    id = importlib.util.resolve_name("." * node.relative + node.id, module_package)
    builder.imports[id] = None

    names = [name for name, _ in node.names]
    as_names = [as_name or name for name, as_name in node.names]
    names_literal = builder.add(LoadLiteral(tuple(names), object_rprimitive))
    if as_names == names:
        # Reuse names tuple to reduce verbosity.
        as_names_literal = names_literal
    else:
        as_names_literal = builder.add(LoadLiteral(tuple(as_names), object_rprimitive))
    # Note that we miscompile import from inside of functions here,
    # since that case *shouldn't* load everything into the globals dict.
    # This probably doesn't matter much and the code runs basically right.
    module = builder.call_c(
        import_from_many_op,
        [builder.load_str(id), names_literal, as_names_literal, builder.load_globals_dict()],
        node.line,
    )
    builder.add(InitStatic(module, id, namespace=NAMESPACE_MODULE))


def transform_import_all(builder: IRBuilder, node: ImportAll) -> None:
    if node.is_mypy_only:
        return
    builder.gen_import(node.id, node.line)


def transform_if_stmt(builder: IRBuilder, stmt: IfStmt) -> None:
    if_body, next = BasicBlock(), BasicBlock()
    else_body = BasicBlock() if stmt.else_body else next

    # If statements are normalized
    assert len(stmt.expr) == 1

    process_conditional(builder, stmt.expr[0], if_body, else_body)
    builder.activate_block(if_body)
    builder.accept(stmt.body[0])
    builder.goto(next)
    if stmt.else_body:
        builder.activate_block(else_body)
        builder.accept(stmt.else_body)
        builder.goto(next)
    builder.activate_block(next)


def transform_while_stmt(builder: IRBuilder, s: WhileStmt) -> None:
    body, next, top, else_block = BasicBlock(), BasicBlock(), BasicBlock(), BasicBlock()
    normal_loop_exit = else_block if s.else_body is not None else next

    builder.push_loop_stack(top, next)

    # Split block so that we get a handle to the top of the loop.
    builder.goto_and_activate(top)
    process_conditional(builder, s.expr, body, normal_loop_exit)

    builder.activate_block(body)
    builder.accept(s.body)
    # Add branch to the top at the end of the body.
    builder.goto(top)

    builder.pop_loop_stack()

    if s.else_body is not None:
        builder.activate_block(else_block)
        builder.accept(s.else_body)
        builder.goto(next)

    builder.activate_block(next)


def transform_for_stmt(builder: IRBuilder, s: ForStmt) -> None:
    def body() -> None:
        builder.accept(s.body)

    def else_block() -> None:
        assert s.else_body is not None
        builder.accept(s.else_body)

    for_loop_helper(
        builder, s.index, s.expr, body, else_block if s.else_body else None, s.is_async, s.line
    )


def transform_break_stmt(builder: IRBuilder, node: BreakStmt) -> None:
    builder.nonlocal_control[-1].gen_break(builder, node.line)


def transform_continue_stmt(builder: IRBuilder, node: ContinueStmt) -> None:
    builder.nonlocal_control[-1].gen_continue(builder, node.line)


def transform_raise_stmt(builder: IRBuilder, s: RaiseStmt) -> None:
    if s.expr is None:
        builder.call_c(reraise_exception_op, [], NO_TRACEBACK_LINE_NO)
        builder.add(Unreachable())
        return

    exc = builder.accept(s.expr)
    builder.call_c(raise_exception_op, [exc], s.line)
    builder.add(Unreachable())


def transform_try_except(
    builder: IRBuilder,
    body: GenFunc,
    handlers: Sequence[tuple[tuple[ValueGenFunc, int] | None, Expression | None, GenFunc]],
    else_body: GenFunc | None,
    line: int,
) -> None:
    """Generalized try/except/else handling that takes functions to gen the bodies.

    The point of this is to also be able to support with."""
    assert handlers, "try needs except"

    except_entry, exit_block, cleanup_block = BasicBlock(), BasicBlock(), BasicBlock()
    double_except_block = BasicBlock()
    # If there is an else block, jump there after the try, otherwise just leave
    else_block = BasicBlock() if else_body else exit_block

    # Compile the try block with an error handler
    builder.builder.push_error_handler(except_entry)
    builder.goto_and_activate(BasicBlock())
    body()
    builder.goto(else_block)
    builder.builder.pop_error_handler()

    # The error handler catches the error and then checks it
    # against the except clauses. We compile the error handler
    # itself with an error handler so that it can properly restore
    # the *old* exc_info if an exception occurs.
    # The exception chaining will be done automatically when the
    # exception is raised, based on the exception in exc_info.
    builder.builder.push_error_handler(double_except_block)
    builder.activate_block(except_entry)
    old_exc = builder.maybe_spill(builder.call_c(error_catch_op, [], line))
    # Compile the except blocks with the nonlocal control flow overridden to clear exc_info
    builder.nonlocal_control.append(ExceptNonlocalControl(builder.nonlocal_control[-1], old_exc))

    # Process the bodies
    for type, var, handler_body in handlers:
        next_block = None
        if type:
            type_f, type_line = type
            next_block, body_block = BasicBlock(), BasicBlock()
            matches = builder.call_c(exc_matches_op, [type_f()], type_line)
            builder.add(Branch(matches, body_block, next_block, Branch.BOOL))
            builder.activate_block(body_block)
        if var:
            target = builder.get_assignment_target(var)
            builder.assign(target, builder.call_c(get_exc_value_op, [], var.line), var.line)
        handler_body()
        builder.goto(cleanup_block)
        if next_block:
            builder.activate_block(next_block)

    # Reraise the exception if needed
    if next_block:
        builder.call_c(reraise_exception_op, [], NO_TRACEBACK_LINE_NO)
        builder.add(Unreachable())

    builder.nonlocal_control.pop()
    builder.builder.pop_error_handler()

    # Cleanup for if we leave except through normal control flow:
    # restore the saved exc_info information and continue propagating
    # the exception if it exists.
    builder.activate_block(cleanup_block)
    builder.call_c(restore_exc_info_op, [builder.read(old_exc)], line)
    builder.goto(exit_block)

    # Cleanup for if we leave except through a raised exception:
    # restore the saved exc_info information and continue propagating
    # the exception.
    builder.activate_block(double_except_block)
    builder.call_c(restore_exc_info_op, [builder.read(old_exc)], line)
    builder.call_c(keep_propagating_op, [], NO_TRACEBACK_LINE_NO)
    builder.add(Unreachable())

    # If present, compile the else body in the obvious way
    if else_body:
        builder.activate_block(else_block)
        else_body()
        builder.goto(exit_block)

    builder.activate_block(exit_block)


def transform_try_except_stmt(builder: IRBuilder, t: TryStmt) -> None:
    def body() -> None:
        builder.accept(t.body)

    # Work around scoping woes
    def make_handler(body: Block) -> GenFunc:
        return lambda: builder.accept(body)

    def make_entry(type: Expression) -> tuple[ValueGenFunc, int]:
        return (lambda: builder.accept(type), type.line)

    handlers = [
        (make_entry(type) if type else None, var, make_handler(body))
        for type, var, body in zip(t.types, t.vars, t.handlers)
    ]
    else_body = (lambda: builder.accept(t.else_body)) if t.else_body else None
    transform_try_except(builder, body, handlers, else_body, t.line)


def try_finally_try(
    builder: IRBuilder,
    err_handler: BasicBlock,
    return_entry: BasicBlock,
    main_entry: BasicBlock,
    try_body: GenFunc,
) -> Register | AssignmentTarget | None:
    # Compile the try block with an error handler
    control = TryFinallyNonlocalControl(return_entry)
    builder.builder.push_error_handler(err_handler)

    builder.nonlocal_control.append(control)
    builder.goto_and_activate(BasicBlock())
    try_body()
    builder.goto(main_entry)
    builder.nonlocal_control.pop()
    builder.builder.pop_error_handler()

    return control.ret_reg


def try_finally_entry_blocks(
    builder: IRBuilder,
    err_handler: BasicBlock,
    return_entry: BasicBlock,
    main_entry: BasicBlock,
    finally_block: BasicBlock,
    ret_reg: Register | AssignmentTarget | None,
) -> Value:
    old_exc = Register(exc_rtuple)

    # Entry block for non-exceptional flow
    builder.activate_block(main_entry)
    if ret_reg:
        builder.assign(ret_reg, builder.add(LoadErrorValue(builder.ret_types[-1])), -1)
    builder.goto(return_entry)

    builder.activate_block(return_entry)
    builder.add(Assign(old_exc, builder.add(LoadErrorValue(exc_rtuple))))
    builder.goto(finally_block)

    # Entry block for errors
    builder.activate_block(err_handler)
    if ret_reg:
        builder.assign(ret_reg, builder.add(LoadErrorValue(builder.ret_types[-1])), -1)
    builder.add(Assign(old_exc, builder.call_c(error_catch_op, [], -1)))
    builder.goto(finally_block)

    return old_exc


def try_finally_body(
    builder: IRBuilder, finally_block: BasicBlock, finally_body: GenFunc, old_exc: Value
) -> tuple[BasicBlock, FinallyNonlocalControl]:
    cleanup_block = BasicBlock()
    # Compile the finally block with the nonlocal control flow overridden to restore exc_info
    builder.builder.push_error_handler(cleanup_block)
    finally_control = FinallyNonlocalControl(builder.nonlocal_control[-1], old_exc)
    builder.nonlocal_control.append(finally_control)
    builder.activate_block(finally_block)
    finally_body()
    builder.nonlocal_control.pop()

    return cleanup_block, finally_control


def try_finally_resolve_control(
    builder: IRBuilder,
    cleanup_block: BasicBlock,
    finally_control: FinallyNonlocalControl,
    old_exc: Value,
    ret_reg: Register | AssignmentTarget | None,
) -> BasicBlock:
    """Resolve the control flow out of a finally block.

    This means returning if there was a return, propagating
    exceptions, break/continue (soon), or just continuing on.
    """
    reraise, rest = BasicBlock(), BasicBlock()
    builder.add(Branch(old_exc, rest, reraise, Branch.IS_ERROR))

    # Reraise the exception if there was one
    builder.activate_block(reraise)
    builder.call_c(reraise_exception_op, [], NO_TRACEBACK_LINE_NO)
    builder.add(Unreachable())
    builder.builder.pop_error_handler()

    # If there was a return, keep returning
    if ret_reg:
        builder.activate_block(rest)
        return_block, rest = BasicBlock(), BasicBlock()
        # For spill targets in try/finally, use nullable read to avoid AttributeError
        if isinstance(ret_reg, AssignmentTargetAttr) and ret_reg.attr.startswith(TEMP_ATTR_NAME):
            ret_val = builder.read_nullable_attr(ret_reg.obj, ret_reg.attr, -1)
        else:
            ret_val = builder.read(ret_reg)
        builder.add(Branch(ret_val, rest, return_block, Branch.IS_ERROR))

        builder.activate_block(return_block)
        builder.nonlocal_control[-1].gen_return(builder, ret_val, -1)

    # TODO: handle break/continue
    builder.activate_block(rest)
    out_block = BasicBlock()
    builder.goto(out_block)

    # If there was an exception, restore again
    builder.activate_block(cleanup_block)
    finally_control.gen_cleanup(builder, -1)
    builder.call_c(keep_propagating_op, [], NO_TRACEBACK_LINE_NO)
    builder.add(Unreachable())

    return out_block


def transform_try_finally_stmt(
    builder: IRBuilder, try_body: GenFunc, finally_body: GenFunc, line: int = -1
) -> None:
    """Generalized try/finally handling that takes functions to gen the bodies.

    The point of this is to also be able to support with."""
    # Finally is a big pain, because there are so many ways that
    # exits can occur. We emit 10+ basic blocks for every finally!

    err_handler, main_entry, return_entry, finally_block = (
        BasicBlock(),
        BasicBlock(),
        BasicBlock(),
        BasicBlock(),
    )

    # Compile the body of the try
    ret_reg = try_finally_try(builder, err_handler, return_entry, main_entry, try_body)

    # Set up the entry blocks for the finally statement
    old_exc = try_finally_entry_blocks(
        builder, err_handler, return_entry, main_entry, finally_block, ret_reg
    )

    # Compile the body of the finally
    cleanup_block, finally_control = try_finally_body(
        builder, finally_block, finally_body, old_exc
    )

    # Resolve the control flow out of the finally block
    out_block = try_finally_resolve_control(
        builder, cleanup_block, finally_control, old_exc, ret_reg
    )

    builder.activate_block(out_block)


def transform_try_finally_stmt_async(
    builder: IRBuilder, try_body: GenFunc, finally_body: GenFunc, line: int = -1
) -> None:
    """Async-aware try/finally handling for when finally contains await.

    This version uses a modified approach that preserves exceptions across await."""

    # We need to handle returns properly, so we'll use TryFinallyNonlocalControl
    # to track return values, similar to the regular try/finally implementation

    err_handler, main_entry, return_entry, finally_entry = (
        BasicBlock(),
        BasicBlock(),
        BasicBlock(),
        BasicBlock(),
    )

    # Track if we're returning from the try block
    control = TryFinallyNonlocalControl(return_entry)
    builder.builder.push_error_handler(err_handler)
    builder.nonlocal_control.append(control)
    builder.goto_and_activate(BasicBlock())
    try_body()
    builder.goto(main_entry)
    builder.nonlocal_control.pop()
    builder.builder.pop_error_handler()
    ret_reg = control.ret_reg

    # Normal case - no exception or return
    builder.activate_block(main_entry)
    builder.goto(finally_entry)

    # Return case
    builder.activate_block(return_entry)
    builder.goto(finally_entry)

    # Exception case - need to catch to clear the error indicator
    builder.activate_block(err_handler)
    # Catch the error to clear Python's error indicator
    builder.call_c(error_catch_op, [], line)
    # We're not going to use old_exc since it won't survive await
    # The exception is now in sys.exc_info()
    builder.goto(finally_entry)

    # Finally block
    builder.activate_block(finally_entry)

    # Execute finally body
    finally_body()

    # After finally, we need to handle exceptions carefully:
    # 1. If finally raised a new exception, it's in the error indicator - let it propagate
    # 2. If finally didn't raise, check if we need to reraise the original from sys.exc_info()
    # 3. If there was a return, return that value
    # 4. Otherwise, normal exit

    # First, check if there's a current exception in the error indicator
    # (this would be from the finally block)
    no_current_exc = builder.call_c(no_err_occurred_op, [], line)
    finally_raised = BasicBlock()
    check_original = BasicBlock()
    builder.add(Branch(no_current_exc, check_original, finally_raised, Branch.BOOL))

    # Finally raised an exception - let it propagate naturally
    builder.activate_block(finally_raised)
    builder.call_c(keep_propagating_op, [], NO_TRACEBACK_LINE_NO)
    builder.add(Unreachable())

    # No exception from finally, check if we need to handle return or original exception
    builder.activate_block(check_original)

    # Check if we have a return value
    if ret_reg:
        return_block, check_old_exc = BasicBlock(), BasicBlock()
        builder.add(Branch(builder.read(ret_reg), check_old_exc, return_block, Branch.IS_ERROR))

        builder.activate_block(return_block)
        builder.nonlocal_control[-1].gen_return(builder, builder.read(ret_reg), -1)

        builder.activate_block(check_old_exc)

    # Check if we need to reraise the original exception from sys.exc_info
    exc_info = builder.call_c(get_exc_info_op, [], line)
    exc_type = builder.add(TupleGet(exc_info, 0, line))

    # Check if exc_type is None
    none_obj = builder.none_object()
    has_exc = builder.binary_op(exc_type, none_obj, "is not", line)

    reraise_block, exit_block = BasicBlock(), BasicBlock()
    builder.add(Branch(has_exc, reraise_block, exit_block, Branch.BOOL))

    # Reraise the original exception
    builder.activate_block(reraise_block)
    builder.call_c(reraise_exception_op, [], NO_TRACEBACK_LINE_NO)
    builder.add(Unreachable())

    # Normal exit
    builder.activate_block(exit_block)


# A simple visitor to detect await expressions
class AwaitDetector(mypy.traverser.TraverserVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.has_await = False

    def visit_await_expr(self, o: mypy.nodes.AwaitExpr) -> None:
        self.has_await = True
        super().visit_await_expr(o)


def transform_try_stmt(builder: IRBuilder, t: TryStmt) -> None:
    # Our compilation strategy for try/except/else/finally is to
    # treat try/except/else and try/finally as separate language
    # constructs that we compile separately. When we have a
    # try/except/else/finally, we treat the try/except/else as the
    # body of a try/finally block.
    if t.is_star:
        builder.error("Exception groups and except* cannot be compiled yet", t.line)

    # Check if we're in an async function with a finally block that contains await
    use_async_version = False
    if t.finally_body and builder.fn_info.is_coroutine:
        detector = AwaitDetector()
        t.finally_body.accept(detector)

        if detector.has_await:
            # Use the async version that handles exceptions correctly
            use_async_version = True

    if t.finally_body:

        def transform_try_body() -> None:
            if t.handlers:
                transform_try_except_stmt(builder, t)
            else:
                builder.accept(t.body)

        body = t.finally_body

        if use_async_version:
            transform_try_finally_stmt_async(
                builder, transform_try_body, lambda: builder.accept(body), t.line
            )
        else:
            transform_try_finally_stmt(
                builder, transform_try_body, lambda: builder.accept(body), t.line
            )
    else:
        transform_try_except_stmt(builder, t)


def get_sys_exc_info(builder: IRBuilder) -> list[Value]:
    exc_info = builder.call_c(get_exc_info_op, [], -1)
    return [builder.add(TupleGet(exc_info, i, -1)) for i in range(3)]


def transform_with(
    builder: IRBuilder,
    expr: Expression,
    target: Lvalue | None,
    body: GenFunc,
    is_async: bool,
    line: int,
) -> None:
    # This is basically a straight transcription of the Python code in PEP 343.
    # I don't actually understand why a bunch of it is the way it is.
    # We could probably optimize the case where the manager is compiled by us,
    # but that is not our common case at all, so.

    al = "a" if is_async else ""

    mgr_v = builder.accept(expr)
    is_native = isinstance(mgr_v.type, RInstance)
    if is_native:
        value = builder.add(MethodCall(mgr_v, f"__{al}enter__", args=[], line=line))
        exit_ = None
    else:
        typ = builder.primitive_op(type_op, [mgr_v], line)
        exit_ = builder.maybe_spill(builder.py_get_attr(typ, f"__{al}exit__", line))
        value = builder.py_call(builder.py_get_attr(typ, f"__{al}enter__", line), [mgr_v], line)

    mgr = builder.maybe_spill(mgr_v)
    exc = builder.maybe_spill_assignable(builder.true())
    if is_async:
        value = emit_await(builder, value, line)

    def maybe_natively_call_exit(exc_info: bool) -> Value:
        if exc_info:
            args = get_sys_exc_info(builder)
        else:
            none = builder.none_object()
            args = [none, none, none]

        if is_native:
            assert isinstance(mgr_v.type, RInstance), mgr_v.type
            exit_val = builder.gen_method_call(
                builder.read(mgr),
                f"__{al}exit__",
                arg_values=args,
                line=line,
                result_type=none_rprimitive,
            )
        else:
            assert exit_ is not None
            exit_val = builder.py_call(builder.read(exit_), [builder.read(mgr)] + args, line)

        if is_async:
            return emit_await(builder, exit_val, line)
        else:
            return exit_val

    def try_body() -> None:
        if target:
            builder.assign(builder.get_assignment_target(target), value, line)
        body()

    def except_body() -> None:
        builder.assign(exc, builder.false(), line)
        out_block, reraise_block = BasicBlock(), BasicBlock()
        builder.add_bool_branch(maybe_natively_call_exit(exc_info=True), out_block, reraise_block)
        builder.activate_block(reraise_block)
        builder.call_c(reraise_exception_op, [], NO_TRACEBACK_LINE_NO)
        builder.add(Unreachable())
        builder.activate_block(out_block)

    def finally_body() -> None:
        out_block, exit_block = BasicBlock(), BasicBlock()
        builder.add(Branch(builder.read(exc), exit_block, out_block, Branch.BOOL))
        builder.activate_block(exit_block)

        maybe_natively_call_exit(exc_info=False)
        builder.goto_and_activate(out_block)

    transform_try_finally_stmt(
        builder,
        lambda: transform_try_except(builder, try_body, [(None, None, except_body)], None, line),
        finally_body,
        line,
    )


def transform_with_stmt(builder: IRBuilder, o: WithStmt) -> None:
    # Generate separate logic for each expr in it, left to right
    def generate(i: int) -> None:
        if i >= len(o.expr):
            builder.accept(o.body)
        else:
            transform_with(
                builder, o.expr[i], o.target[i], lambda: generate(i + 1), o.is_async, o.line
            )

    generate(0)


def transform_assert_stmt(builder: IRBuilder, a: AssertStmt) -> None:
    if builder.options.strip_asserts:
        return
    cond = builder.accept(a.expr)
    ok_block, error_block = BasicBlock(), BasicBlock()
    builder.add_bool_branch(cond, ok_block, error_block)
    builder.activate_block(error_block)
    if a.msg is None:
        # Special case (for simpler generated code)
        builder.add(RaiseStandardError(RaiseStandardError.ASSERTION_ERROR, None, a.line))
    elif isinstance(a.msg, StrExpr):
        # Another special case
        builder.add(RaiseStandardError(RaiseStandardError.ASSERTION_ERROR, a.msg.value, a.line))
    else:
        # The general case -- explicitly construct an exception instance
        message = builder.accept(a.msg)
        exc_type = builder.load_module_attr_by_fullname("builtins.AssertionError", a.line)
        exc = builder.py_call(exc_type, [message], a.line)
        builder.call_c(raise_exception_op, [exc], a.line)
    builder.add(Unreachable())
    builder.activate_block(ok_block)


def transform_del_stmt(builder: IRBuilder, o: DelStmt) -> None:
    transform_del_item(builder, builder.get_assignment_target(o.expr), o.line)


def transform_del_item(builder: IRBuilder, target: AssignmentTarget, line: int) -> None:
    if isinstance(target, AssignmentTargetIndex):
        builder.gen_method_call(
            target.base, "__delitem__", [target.index], result_type=None, line=line
        )
    elif isinstance(target, AssignmentTargetAttr):
        if isinstance(target.obj_type, RInstance):
            cl = target.obj_type.class_ir
            if not cl.is_deletable(target.attr):
                builder.error(f'"{target.attr}" cannot be deleted', line)
                builder.note(
                    'Using "__deletable__ = '
                    + '[\'<attr>\']" in the class body enables "del obj.<attr>"',
                    line,
                )
        key = builder.load_str(target.attr)
        builder.primitive_op(py_delattr_op, [target.obj, key], line)
    elif isinstance(target, AssignmentTargetRegister):
        # Delete a local by assigning an error value to it, which will
        # prompt the insertion of uninit checks.
        builder.add(
            Assign(target.register, builder.add(LoadErrorValue(target.type, undefines=True)))
        )
    elif isinstance(target, AssignmentTargetTuple):
        for subtarget in target.items:
            transform_del_item(builder, subtarget, line)


# yield/yield from/await

# These are really expressions, not statements... but they depend on try/except/finally


def emit_yield(builder: IRBuilder, val: Value, line: int) -> Value:
    retval = builder.coerce(val, builder.ret_types[-1], line)

    cls = builder.fn_info.generator_class
    # Create a new block for the instructions immediately following the yield expression, and
    # set the next label so that the next time '__next__' is called on the generator object,
    # the function continues at the new block.
    next_block = BasicBlock()
    next_label = len(cls.continuation_blocks)
    cls.continuation_blocks.append(next_block)
    builder.assign(cls.next_label_target, Integer(next_label), line)
    builder.add(Return(retval, yield_target=next_block))
    builder.activate_block(next_block)

    add_raise_exception_blocks_to_generator_class(builder, line)

    assert cls.send_arg_reg is not None
    return cls.send_arg_reg


def emit_yield_from_or_await(
    builder: IRBuilder, val: Value, line: int, *, is_await: bool
) -> Value:
    # This is basically an implementation of the code in PEP 380.

    # TODO: do we want to use the right types here?
    result = Register(object_rprimitive)
    to_yield_reg = Register(object_rprimitive)
    received_reg = Register(object_rprimitive)

    helper_method = GENERATOR_HELPER_NAME
    if (
        isinstance(val, (Call, MethodCall))
        and isinstance(val.type, RInstance)
        and val.type.class_ir.has_method(helper_method)
    ):
        # This is a generated native generator class, and we can use a fast path.
        # This allows two optimizations:
        # 1) No need to call CPy_GetCoro() or iter() since for native generators
        #    it just returns the generator object (implemented here).
        # 2) Instead of calling next(), call generator helper method directly,
        #    since next() just calls __next__ which calls the helper method.
        iter_val: Value = val
    else:
        get_op = coro_op if is_await else iter_op
        if isinstance(get_op, PrimitiveDescription):
            iter_val = builder.primitive_op(get_op, [val], line)
        else:
            iter_val = builder.call_c(get_op, [val], line)

    iter_reg = builder.maybe_spill_assignable(iter_val)

    stop_block, main_block, done_block = BasicBlock(), BasicBlock(), BasicBlock()

    if isinstance(iter_reg.type, RInstance) and iter_reg.type.class_ir.has_method(helper_method):
        # Second fast path optimization: call helper directly (see also comment above).
        #
        # Calling a generated generator, so avoid raising StopIteration by passing
        # an extra PyObject ** argument to helper where the stop iteration value is stored.
        fast_path = True
        obj = builder.read(iter_reg)
        nn = builder.none_object()
        stop_iter_val = Register(object_rprimitive)
        err = builder.add(LoadErrorValue(object_rprimitive, undefines=True))
        builder.assign(stop_iter_val, err, line)
        ptr = builder.add(LoadAddress(object_pointer_rprimitive, stop_iter_val))
        m = MethodCall(obj, helper_method, [nn, nn, nn, nn, ptr], line)
        # Generators have custom error handling, so disable normal error handling.
        m.error_kind = ERR_NEVER
        _y_init = builder.add(m)
    else:
        fast_path = False
        _y_init = builder.call_c(next_raw_op, [builder.read(iter_reg)], line)

    builder.add(Branch(_y_init, stop_block, main_block, Branch.IS_ERROR))

    builder.activate_block(stop_block)
    if fast_path:
        builder.primitive_op(propagate_if_error_op, [stop_iter_val], line)
        builder.assign(result, stop_iter_val, line)
    else:
        # Try extracting a return value from a StopIteration and return it.
        # If it wasn't, this reraises the exception.
        builder.assign(result, builder.call_c(check_stop_op, [], line), line)
    # Clear the spilled iterator/coroutine so that it will be freed.
    # Otherwise, the freeing of the spilled register would likely be delayed.
    err = builder.add(LoadErrorValue(iter_reg.type))
    builder.assign(iter_reg, err, line)
    builder.goto(done_block)

    builder.activate_block(main_block)
    builder.assign(to_yield_reg, _y_init, line)

    # OK Now the main loop!
    loop_block = BasicBlock()
    builder.goto_and_activate(loop_block)

    def try_body() -> None:
        builder.assign(received_reg, emit_yield(builder, builder.read(to_yield_reg), line), line)

    def except_body() -> None:
        # The body of the except is all implemented in a C function to
        # reduce how much code we need to generate. It returns a value
        # indicating whether to break or yield (or raise an exception).
        val = Register(object_rprimitive)
        val_address = builder.add(LoadAddress(object_pointer_rprimitive, val))
        to_stop = builder.call_c(yield_from_except_op, [builder.read(iter_reg), val_address], line)

        ok, stop = BasicBlock(), BasicBlock()
        builder.add(Branch(to_stop, stop, ok, Branch.BOOL))

        # The exception got swallowed. Continue, yielding the returned value
        builder.activate_block(ok)
        builder.assign(to_yield_reg, val, line)
        builder.nonlocal_control[-1].gen_continue(builder, line)

        # The exception was a StopIteration. Stop iterating.
        builder.activate_block(stop)
        builder.assign(result, val, line)
        builder.nonlocal_control[-1].gen_break(builder, line)

    def else_body() -> None:
        # Do a next() or a .send(). It will return NULL on exception
        # but it won't automatically propagate.
        _y = builder.call_c(send_op, [builder.read(iter_reg), builder.read(received_reg)], line)
        ok, stop = BasicBlock(), BasicBlock()
        builder.add(Branch(_y, stop, ok, Branch.IS_ERROR))

        # Everything's fine. Yield it.
        builder.activate_block(ok)
        builder.assign(to_yield_reg, _y, line)
        builder.nonlocal_control[-1].gen_continue(builder, line)

        # Try extracting a return value from a StopIteration and return it.
        # If it wasn't, this rereaises the exception.
        builder.activate_block(stop)
        builder.assign(result, builder.call_c(check_stop_op, [], line), line)
        builder.nonlocal_control[-1].gen_break(builder, line)

    builder.push_loop_stack(loop_block, done_block)
    transform_try_except(builder, try_body, [(None, None, except_body)], else_body, line)
    builder.pop_loop_stack()

    builder.goto_and_activate(done_block)
    return builder.read(result)


def emit_await(builder: IRBuilder, val: Value, line: int) -> Value:
    return emit_yield_from_or_await(builder, val, line, is_await=True)


def transform_yield_expr(builder: IRBuilder, expr: YieldExpr) -> Value:
    if builder.fn_info.is_coroutine:
        builder.error("async generators are unimplemented", expr.line)

    if expr.expr:
        retval = builder.accept(expr.expr)
    else:
        retval = builder.builder.none()
    return emit_yield(builder, retval, expr.line)


def transform_yield_from_expr(builder: IRBuilder, o: YieldFromExpr) -> Value:
    return emit_yield_from_or_await(builder, builder.accept(o.expr), o.line, is_await=False)


def transform_await_expr(builder: IRBuilder, o: AwaitExpr) -> Value:
    return emit_yield_from_or_await(builder, builder.accept(o.expr), o.line, is_await=True)


def transform_match_stmt(builder: IRBuilder, m: MatchStmt) -> None:
    m.accept(MatchVisitor(builder, m))


def transform_type_alias_stmt(builder: IRBuilder, s: TypeAliasStmt) -> None:
    line = s.line
    # Use "_typing" to avoid importing "typing", as the latter can be expensive.
    # "_typing" includes everything we need here.
    mod = builder.call_c(import_op, [builder.load_str("_typing")], line)
    type_params = create_type_params(builder, mod, s.type_args, s.line)

    type_alias_type = builder.py_get_attr(mod, "TypeAliasType", line)
    args = [builder.load_str(s.name.name), builder.none()]
    arg_names: list[str | None] = [None, None]
    arg_kinds = [ARG_POS, ARG_POS]
    if s.type_args:
        args.append(builder.new_tuple(type_params, line))
        arg_names.append("type_params")
        arg_kinds.append(ARG_NAMED)
    alias = builder.py_call(type_alias_type, args, line, arg_names=arg_names, arg_kinds=arg_kinds)

    # Use primitive to set function used to lazily compute type alias type value.
    # The value needs to be lazily computed to match Python runtime behavior, but
    # Python public APIs don't support this, so we use a C primitive.
    compute_fn = s.value.accept(builder.visitor)
    builder.builder.primitive_op(set_type_alias_compute_function_op, [alias, compute_fn], line)

    target = builder.get_assignment_target(s.name)
    builder.assign(target, alias, line)
