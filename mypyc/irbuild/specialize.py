"""Special case IR generation of calls to specific builtin functions.

Most special cases should be handled using the data driven "primitive
ops" system, but certain operations require special handling that has
access to the AST/IR directly and can make decisions/optimizations
based on it. These special cases can be implemented here.

For example, we use specializers to statically emit the length of a
fixed length tuple and to emit optimized code for any()/all() calls with
generator comprehensions as the argument.

See comment below for more documentation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

from mypy.nodes import (
    ARG_NAMED,
    ARG_POS,
    CallExpr,
    DictExpr,
    Expression,
    GeneratorExpr,
    IndexExpr,
    IntExpr,
    ListExpr,
    MemberExpr,
    NameExpr,
    RefExpr,
    StrExpr,
    SuperExpr,
    TupleExpr,
    Var,
)
from mypy.types import AnyType, TypeOfAny
from mypyc.ir.ops import (
    BasicBlock,
    Call,
    Extend,
    Integer,
    PrimitiveDescription,
    RaiseStandardError,
    Register,
    SetAttr,
    Truncate,
    Unreachable,
    Value,
)
from mypyc.ir.rtypes import (
    RInstance,
    RPrimitive,
    RTuple,
    RType,
    bool_rprimitive,
    bytes_rprimitive,
    bytes_writer_rprimitive,
    c_int_rprimitive,
    dict_rprimitive,
    int16_rprimitive,
    int32_rprimitive,
    int64_rprimitive,
    int_rprimitive,
    is_bool_rprimitive,
    is_dict_rprimitive,
    is_fixed_width_rtype,
    is_float_rprimitive,
    is_int16_rprimitive,
    is_int32_rprimitive,
    is_int64_rprimitive,
    is_int_rprimitive,
    is_list_rprimitive,
    is_sequence_rprimitive,
    is_str_rprimitive,
    is_tagged,
    is_uint8_rprimitive,
    list_rprimitive,
    object_rprimitive,
    set_rprimitive,
    str_rprimitive,
    uint8_rprimitive,
)
from mypyc.irbuild.builder import IRBuilder
from mypyc.irbuild.constant_fold import constant_fold_expr
from mypyc.irbuild.for_helpers import (
    comprehension_helper,
    get_expr_length_value,
    sequence_from_generator_preallocate_helper,
    translate_list_comprehension,
    translate_set_comprehension,
)
from mypyc.irbuild.format_str_tokenizer import (
    FormatOp,
    convert_format_expr_to_str,
    join_formatted_strings,
    tokenizer_format_call,
)
from mypyc.primitives.bytearray_ops import isinstance_bytearray
from mypyc.primitives.bytes_ops import (
    bytes_adjust_index_op,
    bytes_get_item_unsafe_op,
    bytes_range_check_op,
    isinstance_bytes,
)
from mypyc.primitives.dict_ops import (
    dict_items_op,
    dict_keys_op,
    dict_setdefault_spec_init_op,
    dict_values_op,
    isinstance_dict,
)
from mypyc.primitives.float_ops import isinstance_float
from mypyc.primitives.generic_ops import generic_setattr, setup_object
from mypyc.primitives.int_ops import (
    int_to_big_endian_op,
    int_to_bytes_op,
    int_to_little_endian_op,
    isinstance_int,
)
from mypyc.primitives.librt_strings_ops import (
    bytes_writer_adjust_index_op,
    bytes_writer_get_item_unsafe_op,
    bytes_writer_range_check_op,
    bytes_writer_set_item_unsafe_op,
)
from mypyc.primitives.list_ops import isinstance_list, new_list_set_item_op
from mypyc.primitives.misc_ops import isinstance_bool
from mypyc.primitives.set_ops import isinstance_frozenset, isinstance_set
from mypyc.primitives.str_ops import (
    bytes_decode_ascii_strict,
    bytes_decode_latin1_strict,
    bytes_decode_utf8_strict,
    isinstance_str,
    str_adjust_index_op,
    str_encode_ascii_strict,
    str_encode_latin1_strict,
    str_encode_utf8_strict,
    str_get_item_unsafe_as_int_op,
    str_range_check_op,
)
from mypyc.primitives.tuple_ops import isinstance_tuple, new_tuple_set_item_op

# Specializers are attempted before compiling the arguments to the
# function.  Specializers can return None to indicate that they failed
# and the call should be compiled normally. Otherwise they should emit
# code for the call and return a Value containing the result.
#
# Specializers take three arguments: the IRBuilder, the CallExpr being
# compiled, and the RefExpr that is the left hand side of the call.
Specializer = Callable[["IRBuilder", CallExpr, RefExpr], Value | None]

# Dunder specializers are for special method calls like __getitem__, __setitem__, etc.
# that don't naturally map to CallExpr nodes (e.g., from IndexExpr).
#
# They take four arguments: the IRBuilder, the base expression (target object),
# the list of argument expressions (positional arguments to the dunder), and the
# context expression (e.g., IndexExpr) for error reporting.
DunderSpecializer = Callable[["IRBuilder", Expression, list[Expression], Expression], Value | None]

# Dictionary containing all configured specializers.
#
# Specializers can operate on methods as well, and are keyed on the
# name and RType in that case.
specializers: dict[tuple[str, RType | None], list[Specializer]] = {}

# Dictionary containing all configured dunder specializers.
#
# Dunder specializers are keyed on the dunder name and RType (always a method call).
dunder_specializers: dict[tuple[str, RType], list[DunderSpecializer]] = {}


def _apply_specialization(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr, name: str | None, typ: RType | None = None
) -> Value | None:
    # TODO: Allow special cases to have default args or named args. Currently they don't since
    #       they check that everything in arg_kinds is ARG_POS.

    # If there is a specializer for this function, try calling it.
    # Return the first successful one.
    if name and (name, typ) in specializers:
        for specializer in specializers[name, typ]:
            val = specializer(builder, expr, callee)
            if val is not None:
                return val
    return None


def apply_function_specialization(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Invoke the Specializer callback for a function if one has been registered"""
    return _apply_specialization(builder, expr, callee, callee.fullname)


def apply_method_specialization(
    builder: IRBuilder, expr: CallExpr, callee: MemberExpr, typ: RType | None = None
) -> Value | None:
    """Invoke the Specializer callback for a method if one has been registered"""
    name = callee.fullname if typ is None else callee.name
    return _apply_specialization(builder, expr, callee, name, typ)


def specialize_function(
    name: str, typ: RType | None = None
) -> Callable[[Specializer], Specializer]:
    """Decorator to register a function as being a specializer.

    There may exist multiple specializers for one function. When
    translating method calls, the earlier appended specializer has
    higher priority.
    """

    def wrapper(f: Specializer) -> Specializer:
        specializers.setdefault((name, typ), []).append(f)
        return f

    return wrapper


def specialize_dunder(name: str, typ: RType) -> Callable[[DunderSpecializer], DunderSpecializer]:
    """Decorator to register a function as being a dunder specializer.

    Dunder specializers handle special method calls like __getitem__ that
    don't naturally map to CallExpr nodes.

    There may exist multiple specializers for one dunder. When translating
    dunder calls, the earlier appended specializer has higher priority.
    """

    def wrapper(f: DunderSpecializer) -> DunderSpecializer:
        dunder_specializers.setdefault((name, typ), []).append(f)
        return f

    return wrapper


def apply_dunder_specialization(
    builder: IRBuilder,
    base_expr: Expression,
    args: list[Expression],
    name: str,
    ctx_expr: Expression,
) -> Value | None:
    """Invoke the DunderSpecializer callback if one has been registered.

    Args:
        builder: The IR builder
        base_expr: The base expression (target object)
        args: List of argument expressions (positional arguments to the dunder)
        name: The dunder method name (e.g., "__getitem__")
        ctx_expr: The context expression for error reporting (e.g., IndexExpr)

    Returns:
        The specialized value, or None if no specialization was found.
    """
    base_type = builder.node_type(base_expr)

    # Check if there's a specializer for this dunder method and type
    if (name, base_type) in dunder_specializers:
        for specializer in dunder_specializers[name, base_type]:
            val = specializer(builder, base_expr, args, ctx_expr)
            if val is not None:
                return val
    return None


@specialize_function("builtins.globals")
def translate_globals(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if len(expr.args) == 0:
        return builder.load_globals_dict()
    return None


@specialize_function("builtins.abs")
@specialize_function("builtins.int")
@specialize_function("builtins.float")
@specialize_function("builtins.complex")
@specialize_function("mypy_extensions.i64")
@specialize_function("mypy_extensions.i32")
@specialize_function("mypy_extensions.i16")
@specialize_function("mypy_extensions.u8")
def translate_builtins_with_unary_dunder(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Specialize calls on native classes that implement the associated dunder.

    E.g. i64(x) gets specialized to x.__int__() if x is a native instance.
    """
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS] and isinstance(callee, NameExpr):
        arg = expr.args[0]
        arg_typ = builder.node_type(arg)
        shortname = callee.fullname.split(".")[1]
        if shortname in ("i64", "i32", "i16", "u8"):
            method = "__int__"
        else:
            method = f"__{shortname}__"
        if isinstance(arg_typ, RInstance) and arg_typ.class_ir.has_method(method):
            obj = builder.accept(arg)
            return builder.gen_method_call(obj, method, [], None, expr.line)

    return None


@specialize_function("builtins.len")
def translate_len(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if len(expr.args) == 1 and expr.arg_kinds == [ARG_POS]:
        arg = expr.args[0]
        expr_rtype = builder.node_type(arg)
        # NOTE (?) I'm not sure if my handling of can_borrow is correct here
        obj = builder.accept(arg, can_borrow=is_list_rprimitive(expr_rtype))
        if is_sequence_rprimitive(expr_rtype) or isinstance(expr_rtype, RTuple):
            return get_expr_length_value(builder, arg, obj, expr.line, use_pyssize_t=False)
        else:
            return builder.builtin_len(obj, expr.line)
    return None


@specialize_function("builtins.list")
def dict_methods_fast_path(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    """Specialize a common case when list() is called on a dictionary
    view method call.

    For example:
        foo = list(bar.keys())
    """
    if not (len(expr.args) == 1 and expr.arg_kinds == [ARG_POS]):
        return None
    arg = expr.args[0]
    if not (isinstance(arg, CallExpr) and not arg.args and isinstance(arg.callee, MemberExpr)):
        return None
    base = arg.callee.expr
    attr = arg.callee.name
    rtype = builder.node_type(base)
    if not (is_dict_rprimitive(rtype) and attr in ("keys", "values", "items")):
        return None

    obj = builder.accept(base)
    # Note that it is not safe to use fast methods on dict subclasses,
    # so the corresponding helpers in CPy.h fallback to (inlined)
    # generic logic.
    if attr == "keys":
        return builder.call_c(dict_keys_op, [obj], expr.line)
    elif attr == "values":
        return builder.call_c(dict_values_op, [obj], expr.line)
    else:
        return builder.call_c(dict_items_op, [obj], expr.line)


@specialize_function("builtins.list")
def translate_list_from_generator_call(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Special case for simplest list comprehension.

    For example:
        list(f(x) for x in some_list/some_tuple/some_str)
    'translate_list_comprehension()' would take care of other cases
    if this fails.
    """
    if (
        len(expr.args) == 1
        and expr.arg_kinds[0] == ARG_POS
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return sequence_from_generator_preallocate_helper(
            builder,
            expr.args[0],
            empty_op_llbuilder=builder.builder.new_list_op_with_length,
            set_item_op=new_list_set_item_op,
        )
    return None


@specialize_function("builtins.tuple")
def translate_tuple_from_generator_call(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Special case for simplest tuple creation from a generator.

    For example:
        tuple(f(x) for x in some_list/some_tuple/some_str/some_bytes)
    'translate_safe_generator_call()' would take care of other cases
    if this fails.
    """
    if (
        len(expr.args) == 1
        and expr.arg_kinds[0] == ARG_POS
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return sequence_from_generator_preallocate_helper(
            builder,
            expr.args[0],
            empty_op_llbuilder=builder.builder.new_tuple_with_length,
            set_item_op=new_tuple_set_item_op,
        )
    return None


@specialize_function("builtins.set")
def translate_set_from_generator_call(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Special case for set creation from a generator.

    For example:
        set(f(...) for ... in iterator/nested_generators...)
    """
    if (
        len(expr.args) == 1
        and expr.arg_kinds[0] == ARG_POS
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return translate_set_comprehension(builder, expr.args[0])
    return None


@specialize_function("builtins.min")
@specialize_function("builtins.max")
def faster_min_max(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if expr.arg_kinds == [ARG_POS, ARG_POS]:
        x, y = builder.accept(expr.args[0]), builder.accept(expr.args[1])
        result = Register(builder.node_type(expr))
        # CPython evaluates arguments reversely when calling min(...) or max(...)
        if callee.fullname == "builtins.min":
            comparison = builder.binary_op(y, x, "<", expr.line)
        else:
            comparison = builder.binary_op(y, x, ">", expr.line)

        true_block, false_block, next_block = BasicBlock(), BasicBlock(), BasicBlock()
        builder.add_bool_branch(comparison, true_block, false_block)

        builder.activate_block(true_block)
        builder.assign(result, builder.coerce(y, result.type, expr.line), expr.line)
        builder.goto(next_block)

        builder.activate_block(false_block)
        builder.assign(result, builder.coerce(x, result.type, expr.line), expr.line)
        builder.goto(next_block)

        builder.activate_block(next_block)
        return result
    return None


@specialize_function("builtins.tuple")
@specialize_function("builtins.frozenset")
@specialize_function("builtins.dict")
@specialize_function("builtins.min")
@specialize_function("builtins.max")
@specialize_function("builtins.sorted")
@specialize_function("collections.OrderedDict")
@specialize_function("join", str_rprimitive)
@specialize_function("extend", list_rprimitive)
@specialize_function("update", dict_rprimitive)
@specialize_function("update", set_rprimitive)
def translate_safe_generator_call(
    builder: IRBuilder, expr: CallExpr, callee: RefExpr
) -> Value | None:
    """Special cases for things that consume iterators where we know we
    can safely compile a generator into a list.
    """
    if (
        len(expr.args) > 0
        and expr.arg_kinds[0] == ARG_POS
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        if isinstance(callee, MemberExpr):
            return builder.gen_method_call(
                builder.accept(callee.expr),
                callee.name,
                (
                    [translate_list_comprehension(builder, expr.args[0])]
                    + [builder.accept(arg) for arg in expr.args[1:]]
                ),
                builder.node_type(expr),
                expr.line,
                expr.arg_kinds,
                expr.arg_names,
            )
        else:
            return builder.call_refexpr_with_args(
                expr,
                callee,
                (
                    [translate_list_comprehension(builder, expr.args[0])]
                    + [builder.accept(arg) for arg in expr.args[1:]]
                ),
            )
    return None


@specialize_function("builtins.any")
def translate_any_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if (
        len(expr.args) == 1
        and expr.arg_kinds == [ARG_POS]
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return any_all_helper(builder, expr.args[0], builder.false, lambda x: x, builder.true)
    return None


@specialize_function("builtins.all")
def translate_all_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if (
        len(expr.args) == 1
        and expr.arg_kinds == [ARG_POS]
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return any_all_helper(
            builder,
            expr.args[0],
            builder.true,
            lambda x: builder.unary_op(x, "not", expr.line),
            builder.false,
        )
    return None


def any_all_helper(
    builder: IRBuilder,
    gen: GeneratorExpr,
    initial_value: Callable[[], Value],
    modify: Callable[[Value], Value],
    new_value: Callable[[], Value],
) -> Value:
    retval = Register(bool_rprimitive)
    builder.assign(retval, initial_value(), -1)
    loop_params = list(zip(gen.indices, gen.sequences, gen.condlists, gen.is_async))
    true_block, false_block, exit_block = BasicBlock(), BasicBlock(), BasicBlock()

    def gen_inner_stmts() -> None:
        comparison = modify(builder.accept(gen.left_expr))
        builder.add_bool_branch(comparison, true_block, false_block)
        builder.activate_block(true_block)
        builder.assign(retval, new_value(), -1)
        builder.goto(exit_block)
        builder.activate_block(false_block)

    comprehension_helper(builder, loop_params, gen_inner_stmts, gen.line)
    builder.goto_and_activate(exit_block)

    return retval


@specialize_function("builtins.sum")
def translate_sum_call(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    # specialized implementation is used if:
    # - only one or two arguments given (if not, sum() has been given invalid arguments)
    # - first argument is a Generator (there is no benefit to optimizing the performance of eg.
    #   sum([1, 2, 3]), so non-Generator Iterables are not handled)
    if not (
        len(expr.args) in (1, 2)
        and expr.arg_kinds[0] == ARG_POS
        and isinstance(expr.args[0], GeneratorExpr)
    ):
        return None

    # handle 'start' argument, if given
    if len(expr.args) == 2:
        # ensure call to sum() was properly constructed
        if expr.arg_kinds[1] not in (ARG_POS, ARG_NAMED):
            return None
        start_expr = expr.args[1]
    else:
        start_expr = IntExpr(0)

    gen_expr = expr.args[0]
    target_type = builder.node_type(expr)
    retval = Register(target_type)
    builder.assign(retval, builder.coerce(builder.accept(start_expr), target_type, -1), -1)

    def gen_inner_stmts() -> None:
        call_expr = builder.accept(gen_expr.left_expr)
        builder.assign(retval, builder.binary_op(retval, call_expr, "+", -1), -1)

    loop_params = list(
        zip(gen_expr.indices, gen_expr.sequences, gen_expr.condlists, gen_expr.is_async)
    )
    comprehension_helper(builder, loop_params, gen_inner_stmts, gen_expr.line)

    return retval


@specialize_function("dataclasses.field")
@specialize_function("attr.ib")
@specialize_function("attr.attrib")
@specialize_function("attr.Factory")
def translate_attr_field(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    """Special case for attr.ib/attr.attrib/attr.Factory in class bodies.

    We use these as a marker for initializing attributes in
    the native class constructor.
    """
    if not builder.fn_info or not builder.fn_info.is_decorated_builder:
        return None
    if not builder.fn_info.func_decl or not builder.fn_info.func_decl.class_name:
        return None
    if not builder.fn_info.func_decl.self_type:
        return None

    # Only allow attr.ib/attr.attrib in extension classes
    typ = builder.fn_info.func_decl.self_type
    if not isinstance(typ, RInstance):
        return None
    if not typ.class_ir.is_ext_class:
        builder.error(
            f"can't use {callee.fullname} with non-extension class", expr.line
        )
        return None

    if callee.fullname == "attr.Factory":
        if len(expr.args) != 1:
            return None
        # Factory just unwraps the callable, which will be called in the
        # generated __init__.
        if expr.arg_kinds[0] != ARG_POS:
            return None
        return builder.accept(expr.args[0])
    else:
        # Mark that we need to use the class helper to initialize defaults.
        typ.class_ir.has_dict = True
        # Return an uninitialized instance of the annotation type, or 1
        # if there is no annotation (special-cased in the builder).
        if expr.arg_kinds and expr.arg_kinds[0] != ARG_POS:
            return None
        if expr.args:
            return builder.accept(expr.args[0])
        else:
            return Integer(1)


@specialize_function("builtins.isinstance")
def translate_isinstance(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    if len(expr.args) != 2:
        return None

    obj_expr = expr.args[0]
    type_expr = expr.args[1]

    obj_rtype = builder.node_type(obj_expr)
    type_rtype = builder.node_type(type_expr)
    if not (isinstance(type_rtype, RPrimitive) and type_rtype.name == "builtins.type"):
        return None

    # Some types we can do specialized checks for
    if is_bool_rprimitive(obj_rtype):
        op = isinstance_bool
    elif is_list_rprimitive(obj_rtype):
        op = isinstance_list
    elif is_tuple_rprimitive(obj_rtype):
        op = isinstance_tuple
    elif is_dict_rprimitive(obj_rtype):
        op = isinstance_dict
    elif is_set_rprimitive(obj_rtype):
        op = isinstance_set
    elif isinstance(obj_rtype, RTuple):
        # We could potentially also allow more specialization in specific cases
        # (e.g., a TupleType with a compatible fallback)
        return None
    else:
        return None

    if isinstance(type_expr, CallExpr):
        # The typed_ast parser doesn't preserve enough information to determine
        # whether the argument of the type instance was a keyword or positional,
        # so we assume it might have been a keyword and just skip the specialization.
        return None
    if isinstance(type_expr, NameExpr) and type_expr.fullname == "builtins.bool":
        return builder.call_c(isinstance_bool, [builder.accept(obj_expr)], expr.line)

    return builder.call_c(op, [builder.accept(obj_expr)], expr.line)


@specialize_function("builtins.min")
@specialize_function("builtins.max")
def translate_min_max(builder: IRBuilder, expr: CallExpr, callee: RefExpr) -> Value | None:
    # [rest of file unchanged]
