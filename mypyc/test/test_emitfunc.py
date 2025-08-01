from __future__ import annotations

import unittest

from mypy.test.helpers import assert_string_arrays_equal
from mypyc.codegen.emit import Emitter, EmitterContext
from mypyc.codegen.emitfunc import FunctionEmitterVisitor, generate_native_function
from mypyc.common import HAVE_IMMORTAL, PLATFORM_SIZE
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FuncDecl, FuncIR, FuncSignature, RuntimeArg
from mypyc.ir.ops import (
    ERR_NEVER,
    Assign,
    AssignMulti,
    BasicBlock,
    Box,
    Branch,
    Call,
    CallC,
    Cast,
    ComparisonOp,
    CString,
    DecRef,
    Extend,
    GetAttr,
    GetElementPtr,
    Goto,
    IncRef,
    Integer,
    IntOp,
    LoadAddress,
    LoadLiteral,
    LoadMem,
    Op,
    Register,
    Return,
    SetAttr,
    SetElement,
    SetMem,
    TupleGet,
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
    bool_rprimitive,
    c_int_rprimitive,
    cstring_rprimitive,
    dict_rprimitive,
    int32_rprimitive,
    int64_rprimitive,
    int_rprimitive,
    list_rprimitive,
    none_rprimitive,
    object_rprimitive,
    pointer_rprimitive,
    short_int_rprimitive,
)
from mypyc.irbuild.vtable import compute_vtable
from mypyc.namegen import NameGenerator
from mypyc.primitives.dict_ops import (
    dict_get_item_op,
    dict_new_op,
    dict_set_item_op,
    dict_update_op,
)
from mypyc.primitives.int_ops import int_neg_op
from mypyc.primitives.list_ops import list_append_op, list_get_item_op, list_set_item_op
from mypyc.primitives.misc_ops import none_object_op
from mypyc.primitives.registry import binary_ops
from mypyc.subtype import is_subtype


class TestFunctionEmitterVisitor(unittest.TestCase):
    """Test generation of fragments of C from individual IR ops."""

    def setUp(self) -> None:
        self.registers: list[Register] = []

        def add_local(name: str, rtype: RType) -> Register:
            reg = Register(rtype, name)
            self.registers.append(reg)
            return reg

        self.n = add_local("n", int_rprimitive)
        self.m = add_local("m", int_rprimitive)
        self.k = add_local("k", int_rprimitive)
        self.l = add_local("l", list_rprimitive)
        self.ll = add_local("ll", list_rprimitive)
        self.o = add_local("o", object_rprimitive)
        self.o2 = add_local("o2", object_rprimitive)
        self.d = add_local("d", dict_rprimitive)
        self.b = add_local("b", bool_rprimitive)
        self.s1 = add_local("s1", short_int_rprimitive)
        self.s2 = add_local("s2", short_int_rprimitive)
        self.i32 = add_local("i32", int32_rprimitive)
        self.i32_1 = add_local("i32_1", int32_rprimitive)
        self.i64 = add_local("i64", int64_rprimitive)
        self.i64_1 = add_local("i64_1", int64_rprimitive)
        self.ptr = add_local("ptr", pointer_rprimitive)
        self.t = add_local("t", RTuple([int_rprimitive, bool_rprimitive]))
        self.tt = add_local(
            "tt", RTuple([RTuple([int_rprimitive, bool_rprimitive]), bool_rprimitive])
        )
        ir = ClassIR("A", "mod")
        ir.attributes = {
            "x": bool_rprimitive,
            "y": int_rprimitive,
            "i1": int64_rprimitive,
            "i2": int32_rprimitive,
            "t": RTuple([object_rprimitive, object_rprimitive]),
        }
        ir.bitmap_attrs = ["i1", "i2"]
        compute_vtable(ir)
        ir.mro = [ir]
        self.r = add_local("r", RInstance(ir))
        self.none = add_local("none", none_rprimitive)

        self.struct_type = RStruct(
            "Foo", ["b", "x", "y"], [bool_rprimitive, int32_rprimitive, int64_rprimitive]
        )
        self.st = add_local("st", self.struct_type)

        self.context = EmitterContext(NameGenerator([["mod"]]))

    def test_goto(self) -> None:
        self.assert_emit(Goto(BasicBlock(2)), "goto CPyL2;")

    def test_goto_next_block(self) -> None:
        next_block = BasicBlock(2)
        self.assert_emit(Goto(next_block), "", next_block=next_block)

    def test_return(self) -> None:
        self.assert_emit(Return(self.m), "return cpy_r_m;")

    def test_integer(self) -> None:
        self.assert_emit(Assign(self.n, Integer(5)), "cpy_r_n = 10;")
        self.assert_emit(Assign(self.i32, Integer(5, c_int_rprimitive)), "cpy_r_i32 = 5;")

    def test_tuple_get(self) -> None:
        self.assert_emit(TupleGet(self.t, 1, 0), "cpy_r_r0 = cpy_r_t.f1;")

    def test_load_None(self) -> None:  # noqa: N802
        self.assert_emit(
            LoadAddress(none_object_op.type, none_object_op.src, 0),
            "cpy_r_r0 = (PyObject *)&_Py_NoneStruct;",
        )

    def test_assign_int(self) -> None:
        self.assert_emit(Assign(self.m, self.n), "cpy_r_m = cpy_r_n;")

    def test_int_add(self) -> None:
        self.assert_emit_binary_op(
            "+", self.n, self.m, self.k, "cpy_r_r0 = CPyTagged_Add(cpy_r_m, cpy_r_k);"
        )

    def test_int_sub(self) -> None:
        self.assert_emit_binary_op(
            "-", self.n, self.m, self.k, "cpy_r_r0 = CPyTagged_Subtract(cpy_r_m, cpy_r_k);"
        )

    def test_int_neg(self) -> None:
        assert int_neg_op.c_function_name is not None
        self.assert_emit(
            CallC(
                int_neg_op.c_function_name,
                [self.m],
                int_neg_op.return_type,
                int_neg_op.steals,
                int_neg_op.is_borrowed,
                int_neg_op.is_borrowed,
                int_neg_op.error_kind,
                55,
            ),
            "cpy_r_r0 = CPyTagged_Negate(cpy_r_m);",
        )

    def test_branch(self) -> None:
        self.assert_emit(
            Branch(self.b, BasicBlock(8), BasicBlock(9), Branch.BOOL),
            """if (cpy_r_b) {
                                goto CPyL8;
                            } else
                                goto CPyL9;
                         """,
        )
        b = Branch(self.b, BasicBlock(8), BasicBlock(9), Branch.BOOL)
        b.negated = True
        self.assert_emit(
            b,
            """if (!cpy_r_b) {
                                goto CPyL8;
                            } else
                                goto CPyL9;
                         """,
        )

    def test_branch_no_else(self) -> None:
        next_block = BasicBlock(9)
        b = Branch(self.b, BasicBlock(8), next_block, Branch.BOOL)
        self.assert_emit(b, """if (cpy_r_b) goto CPyL8;""", next_block=next_block)
        next_block = BasicBlock(9)
        b = Branch(self.b, BasicBlock(8), next_block, Branch.BOOL)
        b.negated = True
        self.assert_emit(b, """if (!cpy_r_b) goto CPyL8;""", next_block=next_block)

    def test_branch_no_else_negated(self) -> None:
        next_block = BasicBlock(1)
        b = Branch(self.b, next_block, BasicBlock(2), Branch.BOOL)
        self.assert_emit(b, """if (!cpy_r_b) goto CPyL2;""", next_block=next_block)
        next_block = BasicBlock(1)
        b = Branch(self.b, next_block, BasicBlock(2), Branch.BOOL)
        b.negated = True
        self.assert_emit(b, """if (cpy_r_b) goto CPyL2;""", next_block=next_block)

    def test_branch_is_error(self) -> None:
        b = Branch(self.b, BasicBlock(8), BasicBlock(9), Branch.IS_ERROR)
        self.assert_emit(
            b,
            """if (cpy_r_b == 2) {
                                goto CPyL8;
                            } else
                                goto CPyL9;
                         """,
        )
        b = Branch(self.b, BasicBlock(8), BasicBlock(9), Branch.IS_ERROR)
        b.negated = True
        self.assert_emit(
            b,
            """if (cpy_r_b != 2) {
                                goto CPyL8;
                            } else
                                goto CPyL9;
                         """,
        )

    def test_branch_is_error_next_block(self) -> None:
        next_block = BasicBlock(8)
        b = Branch(self.b, next_block, BasicBlock(9), Branch.IS_ERROR)
        self.assert_emit(b, """if (cpy_r_b != 2) goto CPyL9;""", next_block=next_block)
        b = Branch(self.b, next_block, BasicBlock(9), Branch.IS_ERROR)
        b.negated = True
        self.assert_emit(b, """if (cpy_r_b == 2) goto CPyL9;""", next_block=next_block)

    def test_branch_rare(self) -> None:
        self.assert_emit(
            Branch(self.b, BasicBlock(8), BasicBlock(9), Branch.BOOL, rare=True),
            """if (unlikely(cpy_r_b)) {
                                goto CPyL8;
                            } else
                                goto CPyL9;
                         """,
        )
        next_block = BasicBlock(9)
        self.assert_emit(
            Branch(self.b, BasicBlock(8), next_block, Branch.BOOL, rare=True),
            """if (unlikely(cpy_r_b)) goto CPyL8;""",
            next_block=next_block,
        )
        next_block = BasicBlock(8)
        b = Branch(self.b, next_block, BasicBlock(9), Branch.BOOL, rare=True)
        self.assert_emit(b, """if (likely(!cpy_r_b)) goto CPyL9;""", next_block=next_block)
        next_block = BasicBlock(8)
        b = Branch(self.b, next_block, BasicBlock(9), Branch.BOOL, rare=True)
        b.negated = True
        self.assert_emit(b, """if (likely(cpy_r_b)) goto CPyL9;""", next_block=next_block)

    def test_call(self) -> None:
        decl = FuncDecl(
            "myfn", None, "mod", FuncSignature([RuntimeArg("m", int_rprimitive)], int_rprimitive)
        )
        self.assert_emit(Call(decl, [self.m], 55), "cpy_r_r0 = CPyDef_myfn(cpy_r_m);")

    def test_call_two_args(self) -> None:
        decl = FuncDecl(
            "myfn",
            None,
            "mod",
            FuncSignature(
                [RuntimeArg("m", int_rprimitive), RuntimeArg("n", int_rprimitive)], int_rprimitive
            ),
        )
        self.assert_emit(
            Call(decl, [self.m, self.k], 55), "cpy_r_r0 = CPyDef_myfn(cpy_r_m, cpy_r_k);"
        )

    def test_inc_ref(self) -> None:
        self.assert_emit(IncRef(self.o), "CPy_INCREF(cpy_r_o);")
        self.assert_emit(IncRef(self.o), "CPy_INCREF(cpy_r_o);", rare=True)

    def test_dec_ref(self) -> None:
        self.assert_emit(DecRef(self.o), "CPy_DECREF(cpy_r_o);")
        self.assert_emit(DecRef(self.o), "CPy_DecRef(cpy_r_o);", rare=True)

    def test_inc_ref_int(self) -> None:
        self.assert_emit(IncRef(self.m), "CPyTagged_INCREF(cpy_r_m);")
        self.assert_emit(IncRef(self.m), "CPyTagged_IncRef(cpy_r_m);", rare=True)

    def test_dec_ref_int(self) -> None:
        self.assert_emit(DecRef(self.m), "CPyTagged_DECREF(cpy_r_m);")
        self.assert_emit(DecRef(self.m), "CPyTagged_DecRef(cpy_r_m);", rare=True)

    def test_dec_ref_tuple(self) -> None:
        self.assert_emit(DecRef(self.t), "CPyTagged_DECREF(cpy_r_t.f0);")

    def test_dec_ref_tuple_nested(self) -> None:
        self.assert_emit(DecRef(self.tt), "CPyTagged_DECREF(cpy_r_tt.f0.f0);")

    def test_list_get_item(self) -> None:
        self.assert_emit(
            CallC(
                str(list_get_item_op.c_function_name),
                [self.m, self.k],
                list_get_item_op.return_type,
                list_get_item_op.steals,
                list_get_item_op.is_borrowed,
                list_get_item_op.error_kind,
                55,
            ),
            """cpy_r_r0 = CPyList_GetItem(cpy_r_m, cpy_r_k);""",
        )

    def test_list_set_item(self) -> None:
        self.assert_emit(
            CallC(
                str(list_set_item_op.c_function_name),
                [self.l, self.n, self.o],
                list_set_item_op.return_type,
                list_set_item_op.steals,
                list_set_item_op.is_borrowed,
                list_set_item_op.error_kind,
                55,
            ),
            """cpy_r_r0 = CPyList_SetItem(cpy_r_l, cpy_r_n, cpy_r_o);""",
        )

    def test_box_int(self) -> None:
        self.assert_emit(Box(self.n), """cpy_r_r0 = CPyTagged_StealAsObject(cpy_r_n);""")

    def test_unbox_int(self) -> None:
        self.assert_emit(
            Unbox(self.m, int_rprimitive, 55),
            """if (likely(PyLong_Check(cpy_r_m)))
                                cpy_r_r0 = CPyTagged_FromObject(cpy_r_m);
                            else {
                                CPy_TypeError("int", cpy_r_m); cpy_r_r0 = CPY_INT_TAG;
                            }
                         """,
        )

    def test_box_i64(self) -> None:
        self.assert_emit(Box(self.i64), """cpy_r_r0 = PyLong_FromLongLong(cpy_r_i64);""")

    def test_unbox_i64(self) -> None:
        self.assert_emit(
            Unbox(self.o, int64_rprimitive, 55), """cpy_r_r0 = CPyLong_AsInt64(cpy_r_o);"""
        )

    def test_list_append(self) -> None:
        self.assert_emit(
            CallC(
                str(list_append_op.c_function_name),
                [self.l, self.o],
                list_append_op.return_type,
                list_append_op.steals,
                list_append_op.is_borrowed,
                list_append_op.error_kind,
                1,
            ),
            """cpy_r_r0 = PyList_Append(cpy_r_l, cpy_r_o);""",
        )

    def test_get_attr(self) -> None:
        self.assert_emit(
            GetAttr(self.r, "y", 1),
            """cpy_r_r0 = ((mod___AObject *)cpy_r_r)->_y;
               if (unlikely(cpy_r_r0 == CPY_INT_TAG)) {
                   PyErr_SetString(PyExc_AttributeError, "attribute 'y' of 'A' undefined");
               } else {
                   CPyTagged_INCREF(cpy_r_r0);
               }
            """,
        )

    def test_get_attr_non_refcounted(self) -> None:
        self.assert_emit(
            GetAttr(self.r, "x", 1),
            """cpy_r_r0 = ((mod___AObject *)cpy_r_r)->_x;
               if (unlikely(cpy_r_r0 == 2)) {
                   PyErr_SetString(PyExc_AttributeError, "attribute 'x' of 'A' undefined");
               }
            """,
        )

    def test_get_attr_merged(self) -> None:
        op = GetAttr(self.r, "y", 1)
        branch = Branch(op, BasicBlock(8), BasicBlock(9), Branch.IS_ERROR)
        branch.traceback_entry = ("foobar", 123)
        self.assert_emit(
            op,
            """\
            cpy_r_r0 = ((mod___AObject *)cpy_r_r)->_y;
            if (unlikely(cpy_r_r0 == CPY_INT_TAG)) {
                CPy_AttributeError("prog.py", "foobar", "A", "y", 123, CPyStatic_prog___globals);
                goto CPyL8;
            }
            CPyTagged_INCREF(cpy_r_r0);
            goto CPyL9;
            """,
            next_branch=branch,
            skip_next=True,
        )

    def test_get_attr_with_bitmap(self) -> None:
        self.assert_emit(
            GetAttr(self.r, "i1", 1),
            """cpy_r_r0 = ((mod___AObject *)cpy_r_r)->_i1;
               if (unlikely(cpy_r_r0 == -113) && !(((mod___AObject *)cpy_r_r)->bitmap & 1)) {
                   PyErr_SetString(PyExc_AttributeError, "attribute 'i1' of 'A' undefined");
               }
            """,
        )

    def test_get_attr_nullable_with_tuple(self) -> None:
        self.assert_emit(
            GetAttr(self.r, "t", 1, allow_error_value=True),
            """cpy_r_r0 = ((mod___AObject *)cpy_r_r)->_t;
               if (cpy_r_r0.f0 != NULL) {
                   CPy_INCREF(cpy_r_r0.f0);
                   CPy_INCREF(cpy_r_r0.f1);
               }
            """,
        )

    def test_set_attr(self) -> None:
        self.assert_emit(
            SetAttr(self.r, "y", self.m, 1),
            """if (((mod___AObject *)cpy_r_r)->_y != CPY_INT_TAG) {
                   CPyTagged_DECREF(((mod___AObject *)cpy_r_r)->_y);
               }
               ((mod___AObject *)cpy_r_r)->_y = cpy_r_m;
               cpy_r_r0 = 1;
            """,
        )

    def test_set_attr_non_refcounted(self) -> None:
        self.assert_emit(
            SetAttr(self.r, "x", self.b, 1),
            """((mod___AObject *)cpy_r_r)->_x = cpy_r_b;
               cpy_r_r0 = 1;
            """,
        )

    def test_set_attr_no_error(self) -> None:
        op = SetAttr(self.r, "y", self.m, 1)
        op.error_kind = ERR_NEVER
        self.assert_emit(
            op,
            """if (((mod___AObject *)cpy_r_r)->_y != CPY_INT_TAG) {
                   CPyTagged_DECREF(((mod___AObject *)cpy_r_r)->_y);
               }
               ((mod___AObject *)cpy_r_r)->_y = cpy_r_m;
            """,
        )

    def test_set_attr_non_refcounted_no_error(self) -> None:
        op = SetAttr(self.r, "x", self.b, 1)
        op.error_kind = ERR_NEVER
        self.assert_emit(
            op,
            """((mod___AObject *)cpy_r_r)->_x = cpy_r_b;
            """,
        )

    def test_set_attr_with_bitmap(self) -> None:
        # For some rtypes the error value overlaps a valid value, so we need
        # to use a separate bitmap to track defined attributes.
        self.assert_emit(
            SetAttr(self.r, "i1", self.i64, 1),
            """if (unlikely(cpy_r_i64 == -113)) {
                   ((mod___AObject *)cpy_r_r)->bitmap |= 1;
               }
               ((mod___AObject *)cpy_r_r)->_i1 = cpy_r_i64;
               cpy_r_r0 = 1;
            """,
        )
        self.assert_emit(
            SetAttr(self.r, "i2", self.i32, 1),
            """if (unlikely(cpy_r_i32 == -113)) {
                   ((mod___AObject *)cpy_r_r)->bitmap |= 2;
               }
               ((mod___AObject *)cpy_r_r)->_i2 = cpy_r_i32;
               cpy_r_r0 = 1;
            """,
        )

    def test_set_attr_init_with_bitmap(self) -> None:
        op = SetAttr(self.r, "i1", self.i64, 1)
        op.is_init = True
        self.assert_emit(
            op,
            """if (unlikely(cpy_r_i64 == -113)) {
                   ((mod___AObject *)cpy_r_r)->bitmap |= 1;
               }
               ((mod___AObject *)cpy_r_r)->_i1 = cpy_r_i64;
               cpy_r_r0 = 1;
            """,
        )

    def test_dict_get_item(self) -> None:
        self.assert_emit(
            CallC(
                str(dict_get_item_op.c_function_name),
                [self.d, self.o2],
                dict_get_item_op.return_type,
                dict_get_item_op.steals,
                dict_get_item_op.is_borrowed,
                dict_get_item_op.error_kind,
                1,
            ),
            """cpy_r_r0 = CPyDict_GetItem(cpy_r_d, cpy_r_o2);""",
        )

    def test_dict_set_item(self) -> None:
        self.assert_emit(
            CallC(
                str(dict_set_item_op.c_function_name),
                [self.d, self.o, self.o2],
                dict_set_item_op.return_type,
                dict_set_item_op.steals,
                dict_set_item_op.is_borrowed,
                dict_set_item_op.error_kind,
                1,
            ),
            """cpy_r_r0 = CPyDict_SetItem(cpy_r_d, cpy_r_o, cpy_r_o2);""",
        )

    def test_dict_update(self) -> None:
        self.assert_emit(
            CallC(
                str(dict_update_op.c_function_name),
                [self.d, self.o],
                dict_update_op.return_type,
                dict_update_op.steals,
                dict_update_op.is_borrowed,
                dict_update_op.error_kind,
                1,
            ),
            """cpy_r_r0 = CPyDict_Update(cpy_r_d, cpy_r_o);""",
        )

    def test_new_dict(self) -> None:
        self.assert_emit(
            CallC(
                dict_new_op.c_function_name,
                [],
                dict_new_op.return_type,
                dict_new_op.steals,
                dict_new_op.is_borrowed,
                dict_new_op.error_kind,
                1,
            ),
            """cpy_r_r0 = PyDict_New();""",
        )

    def test_dict_contains(self) -> None:
        self.assert_emit_binary_op(
            "in", self.b, self.o, self.d, """cpy_r_r0 = PyDict_Contains(cpy_r_d, cpy_r_o);"""
        )

    def test_int_op(self) -> None:
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.ADD, 1),
            """cpy_r_r0 = cpy_r_s1 + cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.SUB, 1),
            """cpy_r_r0 = cpy_r_s1 - cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.MUL, 1),
            """cpy_r_r0 = cpy_r_s1 * cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.DIV, 1),
            """cpy_r_r0 = cpy_r_s1 / cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.MOD, 1),
            """cpy_r_r0 = cpy_r_s1 % cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.AND, 1),
            """cpy_r_r0 = cpy_r_s1 & cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.OR, 1),
            """cpy_r_r0 = cpy_r_s1 | cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.XOR, 1),
            """cpy_r_r0 = cpy_r_s1 ^ cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.LEFT_SHIFT, 1),
            """cpy_r_r0 = cpy_r_s1 << cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.s1, self.s2, IntOp.RIGHT_SHIFT, 1),
            """cpy_r_r0 = (Py_ssize_t)cpy_r_s1 >> (Py_ssize_t)cpy_r_s2;""",
        )
        self.assert_emit(
            IntOp(short_int_rprimitive, self.i64, self.i64_1, IntOp.RIGHT_SHIFT, 1),
            """cpy_r_r0 = cpy_r_i64 >> cpy_r_i64_1;""",
        )

    def test_comparison_op(self) -> None:
        # signed
        self.assert_emit(
            ComparisonOp(self.s1, self.s2, ComparisonOp.SLT, 1),
            """cpy_r_r0 = (Py_ssize_t)cpy_r_s1 < (Py_ssize_t)cpy_r_s2;""",
        )
        self.assert_emit(
            ComparisonOp(self.i32, self.i32_1, ComparisonOp.SLT, 1),
            """cpy_r_r0 = cpy_r_i32 < cpy_r_i32_1;""",
        )
        self.assert_emit(
            ComparisonOp(self.i64, self.i64_1, ComparisonOp.SLT, 1),
            """cpy_r_r0 = cpy_r_i64 < cpy_r_i64_1;""",
        )
        # unsigned
        self.assert_emit(
            ComparisonOp(self.s1, self.s2, ComparisonOp.ULT, 1),
            """cpy_r_r0 = cpy_r_s1 < cpy_r_s2;""",
        )
        self.assert_emit(
            ComparisonOp(self.i32, self.i32_1, ComparisonOp.ULT, 1),
            """cpy_r_r0 = (uint32_t)cpy_r_i32 < (uint32_t)cpy_r_i32_1;""",
        )
        self.assert_emit(
            ComparisonOp(self.i64, self.i64_1, ComparisonOp.ULT, 1),
            """cpy_r_r0 = (uint64_t)cpy_r_i64 < (uint64_t)cpy_r_i64_1;""",
        )

        # object type
        self.assert_emit(
            ComparisonOp(self.o, self.o2, ComparisonOp.EQ, 1),
            """cpy_r_r0 = cpy_r_o == cpy_r_o2;""",
        )
        self.assert_emit(
            ComparisonOp(self.o, self.o2, ComparisonOp.NEQ, 1),
            """cpy_r_r0 = cpy_r_o != cpy_r_o2;""",
        )

    def test_load_mem(self) -> None:
        self.assert_emit(LoadMem(bool_rprimitive, self.ptr), """cpy_r_r0 = *(char *)cpy_r_ptr;""")

    def test_set_mem(self) -> None:
        self.assert_emit(
            SetMem(bool_rprimitive, self.ptr, self.b), """*(char *)cpy_r_ptr = cpy_r_b;"""
        )

    def test_get_element_ptr(self) -> None:
        r = RStruct(
            "Foo", ["b", "i32", "i64"], [bool_rprimitive, int32_rprimitive, int64_rprimitive]
        )
        self.assert_emit(
            GetElementPtr(self.o, r, "b"), """cpy_r_r0 = (CPyPtr)&((Foo *)cpy_r_o)->b;"""
        )
        self.assert_emit(
            GetElementPtr(self.o, r, "i32"), """cpy_r_r0 = (CPyPtr)&((Foo *)cpy_r_o)->i32;"""
        )
        self.assert_emit(
            GetElementPtr(self.o, r, "i64"), """cpy_r_r0 = (CPyPtr)&((Foo *)cpy_r_o)->i64;"""
        )

    def test_set_element(self) -> None:
        # Use compact syntax when setting the initial element of an undefined value
        self.assert_emit(
            SetElement(Undef(self.struct_type), "b", self.b), """cpy_r_r0.b = cpy_r_b;"""
        )
        # We propagate the unchanged values in subsequent assignments
        self.assert_emit(
            SetElement(self.st, "x", self.i32),
            """cpy_r_r0 = (Foo) { cpy_r_st.b, cpy_r_i32, cpy_r_st.y };""",
        )

    def test_load_address(self) -> None:
        self.assert_emit(
            LoadAddress(object_rprimitive, "PyDict_Type"),
            """cpy_r_r0 = (PyObject *)&PyDict_Type;""",
        )

    def test_assign_multi(self) -> None:
        t = RArray(object_rprimitive, 2)
        a = Register(t, "a")
        self.registers.append(a)
        self.assert_emit(
            AssignMulti(a, [self.o, self.o2]), """PyObject *cpy_r_a[2] = {cpy_r_o, cpy_r_o2};"""
        )

    def test_long_unsigned(self) -> None:
        a = Register(int64_rprimitive, "a")
        self.assert_emit(
            Assign(a, Integer(1 << 31, int64_rprimitive)), """cpy_r_a = 2147483648LL;"""
        )
        self.assert_emit(
            Assign(a, Integer((1 << 31) - 1, int64_rprimitive)), """cpy_r_a = 2147483647;"""
        )

    def test_long_signed(self) -> None:
        a = Register(int64_rprimitive, "a")
        self.assert_emit(
            Assign(a, Integer(-(1 << 31) + 1, int64_rprimitive)), """cpy_r_a = -2147483647;"""
        )
        self.assert_emit(
            Assign(a, Integer(-(1 << 31), int64_rprimitive)), """cpy_r_a = -2147483648LL;"""
        )

    def test_cast_and_branch_merge(self) -> None:
        op = Cast(self.r, dict_rprimitive, 1)
        next_block = BasicBlock(9)
        branch = Branch(op, BasicBlock(8), next_block, Branch.IS_ERROR)
        branch.traceback_entry = ("foobar", 123)
        self.assert_emit(
            op,
            """\
if (likely(PyDict_Check(cpy_r_r)))
    cpy_r_r0 = cpy_r_r;
else {
    CPy_TypeErrorTraceback("prog.py", "foobar", 123, CPyStatic_prog___globals, "dict", cpy_r_r);
    goto CPyL8;
}
""",
            next_block=next_block,
            next_branch=branch,
            skip_next=True,
        )

    def test_cast_and_branch_no_merge_1(self) -> None:
        op = Cast(self.r, dict_rprimitive, 1)
        branch = Branch(op, BasicBlock(8), BasicBlock(9), Branch.IS_ERROR)
        branch.traceback_entry = ("foobar", 123)
        self.assert_emit(
            op,
            """\
            if (likely(PyDict_Check(cpy_r_r)))
                cpy_r_r0 = cpy_r_r;
            else {
                CPy_TypeError("dict", cpy_r_r);
                cpy_r_r0 = NULL;
            }
            """,
            next_block=BasicBlock(10),
            next_branch=branch,
            skip_next=False,
        )

    def test_cast_and_branch_no_merge_2(self) -> None:
        op = Cast(self.r, dict_rprimitive, 1)
        next_block = BasicBlock(9)
        branch = Branch(op, BasicBlock(8), next_block, Branch.IS_ERROR)
        branch.negated = True
        branch.traceback_entry = ("foobar", 123)
        self.assert_emit(
            op,
            """\
            if (likely(PyDict_Check(cpy_r_r)))
                cpy_r_r0 = cpy_r_r;
            else {
                CPy_TypeError("dict", cpy_r_r);
                cpy_r_r0 = NULL;
            }
            """,
            next_block=next_block,
            next_branch=branch,
        )

    def test_cast_and_branch_no_merge_3(self) -> None:
        op = Cast(self.r, dict_rprimitive, 1)
        next_block = BasicBlock(9)
        branch = Branch(op, BasicBlock(8), next_block, Branch.BOOL)
        branch.traceback_entry = ("foobar", 123)
        self.assert_emit(
            op,
            """\
            if (likely(PyDict_Check(cpy_r_r)))
                cpy_r_r0 = cpy_r_r;
            else {
                CPy_TypeError("dict", cpy_r_r);
                cpy_r_r0 = NULL;
            }
            """,
            next_block=next_block,
            next_branch=branch,
        )

    def test_cast_and_branch_no_merge_4(self) -> None:
        op = Cast(self.r, dict_rprimitive, 1)
        next_block = BasicBlock(9)
        branch = Branch(op, BasicBlock(8), next_block, Branch.IS_ERROR)
        self.assert_emit(
            op,
            """\
            if (likely(PyDict_Check(cpy_r_r)))
                cpy_r_r0 = cpy_r_r;
            else {
                CPy_TypeError("dict", cpy_r_r);
                cpy_r_r0 = NULL;
            }
            """,
            next_block=next_block,
            next_branch=branch,
        )

    def test_extend(self) -> None:
        a = Register(int32_rprimitive, "a")
        self.assert_emit(Extend(a, int64_rprimitive, signed=True), """cpy_r_r0 = cpy_r_a;""")
        self.assert_emit(
            Extend(a, int64_rprimitive, signed=False), """cpy_r_r0 = (uint32_t)cpy_r_a;"""
        )
        if PLATFORM_SIZE == 4:
            self.assert_emit(
                Extend(self.n, int64_rprimitive, signed=True),
                """cpy_r_r0 = (Py_ssize_t)cpy_r_n;""",
            )
            self.assert_emit(
                Extend(self.n, int64_rprimitive, signed=False), """cpy_r_r0 = cpy_r_n;"""
            )
        if PLATFORM_SIZE == 8:
            self.assert_emit(Extend(a, int_rprimitive, signed=True), """cpy_r_r0 = cpy_r_a;""")
            self.assert_emit(
                Extend(a, int_rprimitive, signed=False), """cpy_r_r0 = (uint32_t)cpy_r_a;"""
            )

    def test_inc_ref_none(self) -> None:
        b = Box(self.none)
        self.assert_emit([b, IncRef(b)], "" if HAVE_IMMORTAL else "CPy_INCREF(cpy_r_r0);")

    def test_inc_ref_bool(self) -> None:
        b = Box(self.b)
        self.assert_emit([b, IncRef(b)], "" if HAVE_IMMORTAL else "CPy_INCREF(cpy_r_r0);")

    def test_inc_ref_int_literal(self) -> None:
        for x in -5, 0, 1, 5, 255, 256:
            b = LoadLiteral(x, object_rprimitive)
            self.assert_emit([b, IncRef(b)], "" if HAVE_IMMORTAL else "CPy_INCREF(cpy_r_r0);")
        for x in -1123355, -6, 257, 123235345:
            b = LoadLiteral(x, object_rprimitive)
            self.assert_emit([b, IncRef(b)], "CPy_INCREF(cpy_r_r0);")

    def test_c_string(self) -> None:
        s = Register(cstring_rprimitive, "s")
        self.assert_emit(Assign(s, CString(b"foo")), """cpy_r_s = "foo";""")
        self.assert_emit(Assign(s, CString(b'foo "o')), r"""cpy_r_s = "foo \"o";""")
        self.assert_emit(Assign(s, CString(b"\x00")), r"""cpy_r_s = "\x00";""")
        self.assert_emit(Assign(s, CString(b"\\")), r"""cpy_r_s = "\\";""")
        for i in range(256):
            b = bytes([i])
            if b == b"\n":
                target = "\\n"
            elif b == b"\r":
                target = "\\r"
            elif b == b"\t":
                target = "\\t"
            elif b == b'"':
                target = '\\"'
            elif b == b"\\":
                target = "\\\\"
            elif i < 32 or i >= 127:
                target = "\\x%.2x" % i
            else:
                target = b.decode("ascii")
            self.assert_emit(Assign(s, CString(b)), f'cpy_r_s = "{target}";')

    def assert_emit(
        self,
        op: Op | list[Op],
        expected: str,
        next_block: BasicBlock | None = None,
        *,
        rare: bool = False,
        next_branch: Branch | None = None,
        skip_next: bool = False,
    ) -> None:
        block = BasicBlock(0)
        if isinstance(op, Op):
            block.ops.append(op)
        else:
            block.ops.extend(op)
            op = op[-1]
        value_names = generate_names_for_ir(self.registers, [block])
        emitter = Emitter(self.context, value_names)
        declarations = Emitter(self.context, value_names)
        emitter.fragments = []
        declarations.fragments = []

        visitor = FunctionEmitterVisitor(emitter, declarations, "prog.py", "prog")
        visitor.next_block = next_block
        visitor.rare = rare
        if next_branch:
            visitor.ops = [op, next_branch]
        else:
            visitor.ops = [op]
        visitor.op_index = 0

        op.accept(visitor)
        frags = declarations.fragments + emitter.fragments
        actual_lines = [line.strip(" ") for line in frags]
        assert all(line.endswith("\n") for line in actual_lines)
        actual_lines = [line.rstrip("\n") for line in actual_lines]
        if not expected.strip():
            expected_lines = []
        else:
            expected_lines = expected.rstrip().split("\n")
        expected_lines = [line.strip(" ") for line in expected_lines]
        assert_string_arrays_equal(
            expected_lines, actual_lines, msg="Generated code unexpected", traceback=True
        )
        if skip_next:
            assert visitor.op_index == 1
        else:
            assert visitor.op_index == 0

    def assert_emit_binary_op(
        self, op: str, dest: Value, left: Value, right: Value, expected: str
    ) -> None:
        if op in binary_ops:
            ops = binary_ops[op]
            for desc in ops:
                if is_subtype(left.type, desc.arg_types[0]) and is_subtype(
                    right.type, desc.arg_types[1]
                ):
                    args = [left, right]
                    if desc.ordering is not None:
                        args = [args[i] for i in desc.ordering]
                    # This only supports primitives that map to C calls
                    assert desc.c_function_name is not None
                    self.assert_emit(
                        CallC(
                            desc.c_function_name,
                            args,
                            desc.return_type,
                            desc.steals,
                            desc.is_borrowed,
                            desc.error_kind,
                            55,
                        ),
                        expected,
                    )
                    return
        else:
            assert False, "Could not find matching op"


class TestGenerateFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.arg = RuntimeArg("arg", int_rprimitive)
        self.reg = Register(int_rprimitive, "arg")
        self.block = BasicBlock(0)

    def test_simple(self) -> None:
        self.block.ops.append(Return(self.reg))
        fn = FuncIR(
            FuncDecl("myfunc", None, "mod", FuncSignature([self.arg], int_rprimitive)),
            [self.reg],
            [self.block],
        )
        value_names = generate_names_for_ir(fn.arg_regs, fn.blocks)
        emitter = Emitter(EmitterContext(NameGenerator([["mod"]])), value_names)
        generate_native_function(fn, emitter, "prog.py", "prog")
        result = emitter.fragments
        assert_string_arrays_equal(
            ["CPyTagged CPyDef_myfunc(CPyTagged cpy_r_arg) {\n", "    return cpy_r_arg;\n", "}\n"],
            result,
            msg="Generated code invalid",
        )

    def test_register(self) -> None:
        reg = Register(int_rprimitive)
        op = Assign(reg, Integer(5))
        self.block.ops.append(op)
        self.block.ops.append(Unreachable())
        fn = FuncIR(
            FuncDecl("myfunc", None, "mod", FuncSignature([self.arg], list_rprimitive)),
            [self.reg],
            [self.block],
        )
        value_names = generate_names_for_ir(fn.arg_regs, fn.blocks)
        emitter = Emitter(EmitterContext(NameGenerator([["mod"]])), value_names)
        generate_native_function(fn, emitter, "prog.py", "prog")
        result = emitter.fragments
        assert_string_arrays_equal(
            [
                "PyObject *CPyDef_myfunc(CPyTagged cpy_r_arg) {\n",
                "    CPyTagged cpy_r_r0;\n",
                "    cpy_r_r0 = 10;\n",
                "    CPy_Unreachable();\n",
                "}\n",
            ],
            result,
            msg="Generated code invalid",
        )
