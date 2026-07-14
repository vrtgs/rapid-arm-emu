#![allow(missing_docs)]

use crate::helper::u64_const;
use emu_abi::exec_state::X_REGISTER_COUNT;
use jit_compiler_core::ir::{ExecIrBuilder, IConst, IntCmp, IntWidth, Terminator};

mod helper;

#[test]
#[should_panic(expected = "arithmetic size mismatch")]
fn builder_rejects_arithmetic_width_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(1));
    let narrow = builder.iconst(IConst::u32(1));

    let _ = builder.iadd(wide, narrow);
}

#[test]
#[should_panic(expected = "can only store 64 bit integers to processor registers")]
fn builder_rejects_storing_narrow_value_to_processor_register() {
    let mut builder = ExecIrBuilder::default();

    let narrow = builder.iconst(IConst::u32(1));
    builder.store_x_reg::<0>(narrow);
}

#[test]
#[should_panic]
fn load_x_reg_dyn_rejects_out_of_range_register() {
    let mut builder = ExecIrBuilder::default();

    let _ = builder.load_x_reg_dyn(X_REGISTER_COUNT, IntWidth::W64);
}

#[test]
#[should_panic]
fn store_x_reg_dyn_rejects_out_of_range_register() {
    let mut builder = ExecIrBuilder::default();

    let value = u64_const(&mut builder, 1);
    builder.store_x_reg_dyn(X_REGISTER_COUNT, value);
}

#[test]
#[should_panic(expected = "condition must have bool type")]
fn builder_rejects_select_with_integer_condition() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.iconst(IConst::u64(1));
    let if_true = builder.iconst(IConst::u64(2));
    let if_false = builder.iconst(IConst::u64(3));

    let _ = builder.select(cond, if_true, if_false);
}

#[test]
#[should_panic(expected = "select type mismatch")]
fn builder_rejects_select_value_type_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let one = builder.iconst(IConst::u64(1));
    let cond = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));

    let if_true = builder.iconst(IConst::u64(2));
    let if_false = builder.iconst(IConst::u32(3));

    let _ = builder.select(cond, if_true, if_false);
}

#[test]
#[should_panic(expected = "arithmetic size mismatch")]
fn builder_rejects_comparison_width_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(1));
    let _ = builder.icmp_imm(IntCmp::Equal, wide, IConst::u32(1));
}

#[test]
#[should_panic(expected = "mismatched integer widths used for bitwise op")]
fn builder_rejects_bitwise_width_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(1));
    let narrow = builder.iconst(IConst::u32(1));

    let _ = builder.bitor(wide, narrow);
}

#[test]
#[should_panic(expected = "mismatched integer widths used for bitwise op")]
fn builder_rejects_bitwise_imm_width_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(1));

    let _ = builder.bitand_imm(wide, IConst::u32(1));
}

#[test]
fn terminator_accept_bool_branch_condition() {
    let mut builder = ExecIrBuilder::default();

    let one = builder.iconst(IConst::u64(1));
    let cond = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));
    let target = builder.create_block();

    builder.terminate(Terminator::BrZ(cond, target, target));
}

#[test]
#[should_panic]
fn terminator_rejects_bool_return_fail_reason() {
    let mut builder = ExecIrBuilder::default();

    let one = builder.iconst(IConst::u64(1));
    let halt_reason = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));

    builder.terminate(Terminator::ReturnCode(halt_reason));
}

#[test]
#[should_panic(expected = "can only store 32 bit integers to pstate")]
fn builder_rejects_storing_non_w32_to_pstate() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(0));
    builder.store_pstate(wide);
}
