"""Build mypy IR from the parse tree (and type-checking results).

Constructors IRs are also built here, along with the mypyc glue classes,
Python wrappers, etc.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Callable

from mypy import messages
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    GDEF,
    IMPLICITLY_ABSTRACT,
    LDEF,
    MDEF,
    VARIADIC_STAR,
    Decorator,
    FuncDef,
    LambdaExpr,
    NameExpr,
    TypeInfo,
    Var,
)
from mypy.types import AnyType, CallableType, Type, TypeOfAny, UnboundType, get_proper_type
from mypyc.common import (
    ATTR_PREFIX,
    DEF_SELF_NAME,
    FAST_ISINSTANCE_MAX_SUBCLASSES,
    LAMBDA_NAME,
    MODULE_PREFIX,
    NAME_PREFIX,
    TYPE_PREFIX,
    TYPE_VAR_PREFIX,
)
from mypyc.errors import Errors
from mypyc.ir.class_ir import ClassIR, FunctionInfo
from mypyc.ir.func_ir import (
    FUNC_CLASSMETHOD,
    FUNC_NORMAL,
    FUNC_STATICMETHOD,
    FuncDecl,
    FuncIR,
    FuncSignature,
    FuncInfo,
    RuntimeArg,
)
from mypyc.ir.ops import (
    Assign,
    BasicBlock,
    Branch,
    Call,
    CallC,
    ControlOp,
    DecRef,
    Extend,
    FloatComparisonOp,
    GetAttr,
    GetElementPtr,
    Goto,
    IncRef,
    InitStatic,
    Integer,
    IntOp,
    LoadAddress,
    LoadErrorValue,
    LoadGlobal,
    LoadLiteral,
    LoadMem,
    LoadStatic,
    MethodCall,
    Return,
    SetAttr,
    Truncate,
    TupleGet,
    Unbox,
)
from mypyc.ir.pprint import generate_names_for_ir
from mypyc.ir.rtypes import (
    RInstance,
    RType,
    bool_rprimitive,
    dict_rprimitive,
    int_rprimitive,
    object_rprimitive,
    optional_value_type,
    str_rprimitive,
)
from mypyc.irbuild.constant_fold import constant_fold_expr
from mypyc.irbuild.context import FuncInfo, FuncInfoBase, IRBuilder, ImplicitClass, Liveness
from mypyc.irbuild.env_class import add_self_to_env, setup_env_class
from mypyc.irbuild.format_str_tokenizer import convert_format_str_to_str
from mypyc.irbuild.nonlocalcontrol import cleanup_nonlocal_control
from mypyc.irbuild.prebuildvisitor import PreBuildVisitor
from mypyc.irbuild.specialize import translate_fast_isinstance
from mypyc.primitives.dict_ops import dict_get_item_op, dict_set_item_op
from mypyc.primitives.misc_ops import check_unpack_count_op, iter_op, py_calc_hash_op
from mypyc.primitives.registry import load_address_op, py_setattr_op
from mypyc.sametype import is_same_type


def transform_func_def(builder: IRBuilder, fdef: FuncDef, can_borrow: bool = False) -> FuncIR:
    """Transform a mypy AST FuncDef to IR.

    The body of the function is transformed into IR. This also deals with
    nested functions within the function body, and with generators.

    The mypy AST is desugared during semantic analysis and type checking and
    thus corresponds to Python bytecode already. The IR resembles Python
    bytecode to a degree and it is easier to generate correct code by closely
    following the structure of the function.

    If the function is nested inside another function, then it will use an
    implicit first argument that is the environment of the enclosing function.
    """
    builder.enter(FuncInfo(fdef))

    # Load all arguments to registers.
    for arg in builder.mapper.fdef_to_sig(fdef).args:
        builder.add_argument(arg.name, arg.type)

    # If this is a method, then define the implicit 'self' variable. This
    # has to happen before visiting the body, in case the body contains
    # any references to the 'self' variable.
    if isinstance(fdef, FuncDef):
        builder.add_self_to_env(builder.fn_info.class_ir)

    # Do this after setting up self if we are in a class,
    # as __init__ needs self in env.
    setup_env_class(builder)

    # Do this after setting up the environment
    visit = PreBuildVisitor(builder)
    fdef.body.accept(visit)

    # Fill in default values for arguments and then do the function body.
    setup_func_def(builder, fdef)

    builder.leave()
    return builder.fn_info.fitem.fn_ir


@specialize_function
@singledispatch_function
@in_external_trait
@in_trait
@in_external_class
def transform_method(builder: IRBuilder, fdef: FuncDef) -> None:
    if builder.fn_info.is_generator:
        builder.fn_info.add_self_to_env()  # Must happen before setup_env_class

    setup_env_class(builder)

    # Do this after setting up the environment
    visit = PreBuildVisitor(builder)
    fdef.body.accept(visit)

    # Fill in default values for arguments and then do the function body.
    setup_func_def(builder, fdef)

    builder.leave()


def transform_overloaded_method(
    builder: IRBuilder,
    fdefs: list[FuncDef],
    implementation: FuncDef,
    impl_sig: FuncSignature,
) -> None:
    overloads: dict[str, list[FuncDef]] = defaultdict(list)
    for fdef in fdefs:
        overloads[fdef.name].append(fdef)

    # In generated code, the overrides dispatch based on argument count, so
    # we need to check the argument kinds at runtime.
    transform_method(builder, implementation)
    builder.fn_info.overload_map[implementation.name] = overloads


def transform_decorator(builder: IRBuilder, dec: Decorator) -> None:
    sig = builder.mapper.fdef_to_sig(dec.func)
    func_ir, func_reg = transform_func_def(builder, dec.func)

    base = dec.var.type
    assert isinstance(base, CallableType)
    wrapper = dec.func
    if dec.func.is_generator:
        add_yield_from_to_wrapper(dec.func, builder)
    func_ir = gen_glue(builder, func_ir, base, sig)
    func_reg = builder.instantiate_callable_class(func_ir, func_reg)

    # Run the decorator on the function
    decorator = dec.decorator
    # Some decorators (like @property) return non-callables
    func_obj = builder.builder.py_call(decorator, [func_reg], dec.line)
    builder.assign(builder.lookup(dec.var), func_obj, dec.line)


def transform_class_def(builder: IRBuilder, cdef: ClassDef) -> None:
    name = cdef.name
    fullname = cdef.fullname

    # Build IR for class body.
    builder.enter_class(cdef)
    setup_env_class(builder)
    build_ir_for_class_body(builder, cdef)
    builder.leave_class()

    class_ir = builder.mapper.type_to_ir[cdef.info]
    # If this is a nested class, need to add the implicit class environment
    # argument
    if class_ir.is_nested:
        add_nested_env_class(builder, class_ir)

    # Create non-extension class class_dict
    non_ext = not class_ir.is_ext_class
    if non_ext and class_ir.is_generated:
        cls_dict = setup_non_ext_dict(builder, cdef)
    else:
        cls_dict = None

    # Generate a constructor
    if class_ir.is_ext_class:
        gen_native_init(builder, cdef)
        if class_ir.allow_interpreted_subclasses:
            gen_init_for_interpreted_subclasses(builder, cdef)

    # Create the class for native extension classes
    if class_ir.is_ext_class:
        gen_native_type_object(builder, cdef)

    # Generate glue methods for calling native methods in interpreted subclasses
    if class_ir.allow_interpreted_subclasses:
        gen_glue_methods(builder)

    # Generate properties for read-only attributes
    if non_ext and class_ir.is_generated:
        for attr, typ in class_ir.attributes.items():
            if class_ir.is_attr_readonly(attr) and class_ir.is_always_defined(attr):
                gen_glue_property(builder, None, attr, typ, class_ir, class_ir, is_setter=False)

    # Create a tuple representing the MRO for non-extension classes.
    if non_ext and class_ir.is_generated:
        # For generated classes, we first construct their namespaces so that
        # we can instantiate the class object.
        # For normal classes, the class namespace gets stored directly
        # in a PyObject * or a dict object. This is done in
gen_module_def().
        cls_ns = builder.builder.make_dict([], -1)
        cls_qualname = fullname.rsplit(".", maxsplit=1)[-1]
        builder.builder.add(SetAttr(cls_ns, "__qualname__", cls_qualname, -1))

        # Add __module__
        cls_mod = builder.load_global_str(fullname.rsplit(".")[0], -1)
        builder.builder.add(SetAttr(cls_ns, "__module__", cls_mod, -1))

        # Create the class
        cdef = ClassDef(class_ir.name, Block([]))
        cdef.fullname = fullname
        cdef.info = TypeInfo(SymbolTable(), cdef, fullname)
        cdef.info.bases = [TypeInfo(SymbolTable(), cdef, "builtins.object")]
        cdef.info.mro = [cdef.info] + cdef.info.bases
        bases = [builder.builder.get_native_type(base) for base in cdef.info.bases]
        class_tuple = builder.builder.new_tuple(bases, -1)
        meta = builder.load_module_attr_by_fullname("builtins.type", -1)
        class_obj = builder.builder.call(meta, [fullname, class_tuple, cls_ns], -1)
        cls_obj_reg = Register(object_rprimitive)
        builder.assign(cls_obj_reg, class_obj, -1)

        builder.instantiate_callable_class(cls_obj_reg, class_ir.class_name)

    # Setup final MRO
    if class_ir.is_ext_class:
        class_ir.mro = [class_ir] + class_ir.mro


def transform_func_body(builder: IRBuilder, fdef: FuncDef) -> None:
    # Build IR for a function in two passes:
    # * First transform the AST (body) into IR ops
    # * Then insert error handling and do optimizations
    if fdef.is_decorated:
        old_env = builder.fn_info
        transform_decorator(builder, fdef)
        builder = old_env
    else:
        transform_func_def(builder, fdef)


def setup_env_class(builder: IRBuilder) -> None:
    """Prepare environment class, if needed, for a function.

    Environment classes are needed for nested functions or generators. This also
    takes care of adding the environment class to the main class namespace if
    needed.
    """
    if builder.fn_info.generator_class is not None:
        # Generators currently always use environments.
        setup_env_class(builder)
        if builder.fn_info.can_merge_generator_and_env_classes():
            builder.fn_info.generator_class.ir.is_environment = True

    if builder.fn_info.env_class is not None:
        ir = builder.fn_info.env_class
        builder.fn_info.env_class = ir

        if builder.fn_info.is_nested:
            add_nested_env_class(builder, ir)

    if builder.fn_info.generator_class is not None and builder.fn_info.env_class is not None:
        add_nested_env_class(builder, builder.fn_info.generator_class.ir)


def gen_module_builder(module: MypyFile, mapper: Mapper, options: Options, is_group: bool = False) -> ModuleBuilder:
    """Create a ModuleBuilder for a module.

    Args:
        module: A MypyFile
        mapper: The mapper to use
        options: The options to use
        is_group: True if this is a group member; used to produce different
            output after group generation
    """
    errors = Errors(options)
    builder = IRBuilder(mapper, options, errors)

    # The groups are laid out in the same order as top-level file list and
    # top-level dependency list. We do this so that modules have stable
    # indexes, which ensures stable C names across different runs.
    order = {m.fullname: i for i, m in enumerate(options.modules)}
    builder.group_map = build_group_map(options, order)

    if module.id not in mapper.modules:
        module_builder = ModuleBuilder(builder)
        mapper.modules[module.id] = module_builder
        module_builder.mapper = mapper
        module_builder.builder = builder
        module_builder.options = options

    module_builder.visit_mypy_file(module)
    return module_builder
