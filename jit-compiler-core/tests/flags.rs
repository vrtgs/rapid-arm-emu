#![allow(missing_docs)]

use crate::helper::{clear_pstate, run_success, store_pstate_equals_as_x_reg};
use emu_abi::exec_state::{ExecState, PState};
use jit_compiler_core::ir::{ExecIrBuilder, IConst, IntCmp, IntWidth};

mod helper;

#[test]
fn set_nzcv_flags_preserves_non_flag_bits_and_replaces_old_flags() {
    let mut builder = ExecIrBuilder::default();

    let preserved = 0x00ff_00ff & !PState::NZCV_MASK.0;
    let initial = preserved | PState::NZCV_MASK.0;

    let initial = builder.iconst(IConst::u32(initial));
    builder.store_pstate(initial);

    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));

    let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
    let false_bool = builder.icmp(IntCmp::Equal, one, zero);

    builder.set_nzcv_flags(true_bool, false_bool, true_bool, false_bool);

    let expected = preserved | PState::N.0 | PState::C.0;
    store_pstate_equals_as_x_reg::<0>(&mut builder, expected);

    let mut state = ExecState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn adds_and_subs_set_arm_nzcv_flags_for_wrap_carry_and_overflow() {
    let mut builder = ExecIrBuilder::default();

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::u64(u64::MAX));
    let rhs = builder.iconst(IConst::u64(1));
    let value = builder.iadds(lhs, rhs);
    builder.store_x_reg::<0>(value);
    store_pstate_equals_as_x_reg::<1>(&mut builder, PState::Z.0 | PState::C.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::i64(i64::MAX));
    let rhs = builder.iconst(IConst::i64(1));
    let value = builder.iadds(lhs, rhs);
    builder.store_x_reg::<2>(value);
    store_pstate_equals_as_x_reg::<3>(&mut builder, PState::N.0 | PState::V.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::u64(0));
    let rhs = builder.iconst(IConst::u64(1));
    let value = builder.isubs(lhs, rhs);
    builder.store_x_reg::<4>(value);
    store_pstate_equals_as_x_reg::<5>(&mut builder, PState::N.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::i64(i64::MIN));
    let rhs = builder.iconst(IConst::i64(1));
    let value = builder.isubs(lhs, rhs);
    builder.store_x_reg::<6>(value);
    store_pstate_equals_as_x_reg::<7>(&mut builder, PState::C.0 | PState::V.0);

    let mut state = ExecState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 1);

    assert_eq!(state.x_registers[2], i64::MIN.cast_unsigned());
    assert_eq!(state.x_registers[3], 1);

    assert_eq!(state.x_registers[4], u64::MAX);
    assert_eq!(state.x_registers[5], 1);

    assert_eq!(state.x_registers[6], i64::MAX.cast_unsigned());
    assert_eq!(state.x_registers[7], 1);
}

#[test]
fn flag_setting_binops_produce_storable_values() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.load_x_reg::<0>(IntWidth::W64);
    let rhs = builder.load_x_reg::<1>(IntWidth::W64);
    let sum = builder.iadds(lhs, rhs);
    builder.store_x_reg::<2>(sum);

    let lhs = builder.load_x_reg::<0>(IntWidth::W64);
    let rhs = builder.load_x_reg::<1>(IntWidth::W64);
    let diff = builder.isubs(lhs, rhs);
    builder.store_x_reg::<3>(diff);

    let mut state = ExecState::initial();
    state.x_registers[0] = 40;
    state.x_registers[1] = 58;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[2], 98);
    assert_eq!(state.x_registers[3], 40_u64.wrapping_sub(58));
}
