"""Data structures for representing mypyc classes.

There is a fairly direct mapping from mypy IR classes to this form.

These IR classes are also used for describing extension class
representations.
"""

from __future__ import annotations

from mypy.nodes import TypeInfo

from mypyc.common import FAST_ISINSTANCE_MAX_SUBCLASSES
from mypyc.ir.func_ir import FuncDecl, FuncIR, FuncSignature, all_values
from mypyc.ir.rtypes import RInstance, RType, deserialize_type, serialize_type
from mypyc.ir.vtable import VTable
from mypyc.serialize import JsonDict, DeserMaps

# Map from name of a special method to data about the method for generating it
from mypyc.specialize import SpecializeMethod  # noqa: E402
from mypyc.symtables import SymbolTable

# Anything added to this constant is assumed to be a hard-coded builtin
BUILTIN_CLASS_NAMES = {
    "builtins.object",
    "builtins.type",
    "builtins.int",
    "builtins.float",
    "builtins.str",
    "builtins.bool",
    "builtins.tuple",
    "builtins.list",
    "builtins.dict",
    "builtins.set",
    "builtins.frozenset",
}


class ClassIR:
    def __init__(
        self,
        name: str,
        module_name: str,
        is_trait: bool = False,
        is_generated: bool = False,
        is_abstract: bool = False,
        is_ext_class: bool = True,
        is_final_class: bool = False,
        is_augmented: bool = False,
    ) -> None:
        self.name = name
        self.module_name = module_name
        self.is_trait = is_trait
        self.is_generated = is_generated
        # Environment classes represent locals and should emit UnboundLocalError for missing vars.
        self.is_environment = False
        self.is_abstract = is_abstract
        self.is_ext_class = is_ext_class
        self.is_final_class = is_final_class
        self.is_augmented = is_augmented

        self.deletable = False
        # Whether to generate getseters for member attributes.
        self.needs_getseters = False
        # Whether to generate getseters for implicit property access on some members
        self.needs_getseters_for_properties = False
        # Members that are accessed from C that can bypass accessors
        self._properties = set()

        # We use these to track if we can create a special merged
        # getter/setter method for a property. If a property appears
        # in _property_readable, then it is readable in the target
        # class and all subclasses. If it appears in _property_writable,
        # then it is writable in the target class and all subclasses.
        # If it appears in neither, then it's only readable/writable on
        # the target class and we generate a property accessor method.
        self._property_readable: set[str] = set()
        self._property_writable: set[str] = set()

        # If true, we should generate a property accessor for each property
        # in property_readable or property_writable
        self.property_getter_or_setter: bool | None = None

        # Attributes that are initialized in __init__ or elsewhere.
        # If an attribute is initialized in __init__ or elsewhere, then
        # there is no reason to check if it is defined when accessing it.
        self._always_initialized_attrs: set[str] | None = None

        # Attributes of the class represented as a mapping from the attribute name
        # to the type (RType) of the attribute
        self.attributes: dict[str, RType] = {}
        # Some attributes are always initialized to None (only applicable to reference types).
        self.always_initialized_attrs: dict[str, bool] = {}
        # Attributes that may be initialized
        self._init_self_attributes: list[tuple[str, RType]] | None = None
        # Attributes that are in a bitmap (for tracking definedness)
        self.bitmap_attrs: list[str] = []

        # Maps attribute names to their corresponding getters and setters.
        # If a getter/setter is defined, it is preferred over direct slot access.
        self.properties: dict[str, tuple[FuncIR | None, FuncIR | None]] = {}

        # List of methods, keyed by name
        self.methods: dict[str, FuncIR] = {}

        # List of other methods, keyed by name
        # (patches and property accessors)
        self.glue_methods: dict[str, FuncIR] = {}

        # Nro if this is a trait
        self.mro: list[ClassIR] = []

        # Base class, if this is a trait
        self.trait_base: ClassIR | None = None

        # Base classes, if this is a non-trait class
        self.base_mro: list[ClassIR] = []

        # Direct subclasses (for trait vtables)
        self.children: set[ClassIR] = set()

        # Description of class attributes
        # (<name>, <type>, <is_final>)
        self.class_attributes: dict[str, tuple[RType, bool]] = {}

        # Meta attributes for class attributes to be used by generated code.
        # (<name>, <value>, <is_final>)
        self.meta_attributes: dict[str, tuple[object, bool]] = {}

        self.vtable_entries: list[VTableEntry] | None = None
        self.vtable_signature: dict[str, FuncSignature] = {}
        # Not including entries for methods defined in trait ancestors.
        self.vtable_entries_for_trait: list[VTableEntry] | None = None

        # Unboxed types for which this class is used to wrap as a boxed object
        self.ctor_for: list[RType] = []

        # method declarations for mypy shadow classes
        self.method_decls: list[FuncDecl] = []

        # information about how to serialize this class
        self.serialize_info: list[tuple[str, RType]] = []

        # slots for each attribute (not necessarily in source code order)
        self.slots: list[str] = []

        # class for trait methods
        self.trait_vtable: VTable | None = None

        # dict of attribute name to index in vtable
        self.vtable_attrs: dict[str, int] = {}

        # map of attribute name to its property object (None if not a property)
        # (in the interpreted version)
        self.properties_info: dict[str, TypeInfo | None] = {}

    def __repr__(self) -> str:
        return (
            "ClassIR("
            "name={self.name}, module_name={self.module_name}, "
            "is_trait={self.is_trait}, is_generated={self.is_generated}, "
            "is_environment={self.is_environment}, "
            "is_abstract={self.is_abstract}, is_ext_class={self.is_ext_class}, "
            "is_final_class={self.is_final_class}"
            ")".format(self=self)
        )

    @property
    def fullname(self) -> str:
        return self.module_name + "." + self.name

    def serialize(self) -> JsonDict:
        return {
            "name": self.name,
            "module_name": self.module_name,
            "is_trait": self.is_trait,
            "is_ext_class": self.is_ext_class,
            "is_abstract": self.is_abstract,
            "is_generated": self.is_generated,
            "is_environment": self.is_environment,
            "is_augmented": self.is_augmented,
            "is_final_class": self.is_final_class,
            "inherits_python": self.inherits_python,
            "has_dict": self.has_dict,
            "allow_interpreted_subclasses": self.allow_interpreted_subclasses,
            "needs_getseters": self.needs_getseters,
            "needs_getseters_for_properties": self.needs_getseters_for_properties,
            "attributes": [(k, serialize_type(v)) for k, v in self.attributes.items()],
            "always_initialized_attrs": self.always_initialized_attrs,
            "init_self_attributes": [
                (name, serialize_type(typ))
                for name, typ in self._init_self_attributes or []
            ],
            "bitmap_attrs": self.bitmap_attrs,
            "properties": [
                (k, v[0].name if v[0] else None, v[1].name if v[1] else None)
                for k, v in self.properties.items()
            ],
            "methods": [v.name for v in self.methods.values()],
            "glue_methods": [v.name for v in self.glue_methods.values()],
            "mro": [c.fullname for c in self.mro],
            "trait_base": self.trait_base.fullname if self.trait_base else None,
            "base_mro": [c.fullname for c in self.base_mro],
            "children": [c.fullname for c in self.children],
            "constructor": self.ctor.name if self.ctor else None,
            "attrs_with_defaults": self.attrs_with_defaults,
            "vtable_entries": [v.serialize() for v in self.vtable_entries or []],
            "vtable_entries_for_trait": [
                v.serialize() for v in self.vtable_entries_for_trait or []
            ],
            "vtable_attrs": self.vtable_attrs,
            "has_attr_richcompare": self.has_attr_richcompare,
            "is_pyc_only": self.is_pyc_only,
        }

    @classmethod
    def deserialize(cls, data: JsonDict, ctx: DeserMaps) -> ClassIR:
        name = data["name"]
        module_name = data["module_name"]
        ir = ClassIR(name, module_name)

        ir.is_trait = data["is_trait"]
        ir.is_generated = data["is_generated"]
        ir.is_environment = data.get("is_environment", False)
        ir.is_abstract = data["is_abstract"]
        ir.is_ext_class = data["is_ext_class"]
        ir.is_augmented = data["is_augmented"]
        ir.is_final_class = data["is_final_class"]
        ir.inherits_python = data["inherits_python"]
        ir.has_dict = data["has_dict"]
        ir.allow_interpreted_subclasses = data["allow_interpreted_subclasses"]
        ir.needs_getseters = data["needs_getseters"]
        ir.needs_getseters_for_properties = data["needs_getseters_for_properties"]
        ir.attributes = dict(
            (k, deserialize_type(t, ctx)) for k, t in data["attributes"]
        )
        ir.always_initialized_attrs = data["always_initialized_attrs"]
        ir._init_self_attributes = [
            (name, deserialize_type(typ, ctx))
            for name, typ in data["init_self_attributes"]
        ]
        ir.bitmap_attrs = data["bitmap_attrs"]
        ir.vtable_attrs = data["vtable_attrs"]

        # These will be populated later.
        ir.properties = {}
        ir.methods = {}
        ir.glue_methods = {}
        ir.vtable_entries = None
        ir.vtable_entries_for_trait = None
        ir.trait_vtable = None
        ir._properties = set()

        ir._always_initialized_attrs = None
        ir._property_readable = set()
        ir._property_writable = set()

        # XXX: should we handle these?
        ir.mro = []
        ir.trait_base = None
        ir.base_mro = []

        ir.children = set()

        # other things to fix up
        ir.ctor = None
        ir.attributes_readonly = set()
        ir.vtable_signature = {}
        ir.class_attributes = {}
        ir.meta_attributes = {}
        ir.method_decls = []
        ir.serialize_info = []

        return ir
