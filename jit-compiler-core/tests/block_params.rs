#![allow(missing_docs)]

use crate::helper::{call_compiled, compile, run_success, u64_const};
use emu_abi::exec_state::ExecState;
use jit_compiler_core::ir::{ExecIrBuilder, IntWidth, Terminator, Type};

mod helper;

#[test]
fn block_parameter_passed_by_unconditional_branch_is_visible_in_target() {
    let mut builder = ExecIrBuilder::default();

    let target = builder.create_block();
    let param = builder.add_block_parameter_at(target, Type::I64);

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let delta = u64_const(&mut builder, 0x20);
    let arg = builder.iadd(x0, delta);

    builder.terminate(Terminator::Br((target, vec![arg])));

    builder.switch_to(target);
    let one = u64_const(&mut builder, 1);
    let result = builder.iadd(param, one);
    builder.store_x_reg::<1>(result);

    let mut state = ExecState::initial();
    state.x_registers[0] = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0x1021);
}

#[test]
fn block_parameter_arguments_act_like_phi_for_conditional_branch() {
    let mut builder = ExecIrBuilder::default();

    let join = builder.create_block();
    let value = builder.add_block_parameter_at(join, Type::I64);

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    let zero_value = u64_const(&mut builder, 2);
    let non_zero_value = u64_const(&mut builder, 40);

    builder.terminate(Terminator::BrZ(
        cond,
        (join, vec![zero_value]),
        (join, vec![non_zero_value]),
    ));

    builder.switch_to(join);
    let one = u64_const(&mut builder, 1);
    let result = builder.iadd(value, one);
    builder.store_x_reg::<1>(result);

    let compiled = compile(builder);

    let mut zero_state = ExecState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 3);

    let mut non_zero_state = ExecState::initial();
    non_zero_state.x_registers[0] = 1;
    assert_eq!(call_compiled(&compiled, &mut non_zero_state), 0);
    assert_eq!(non_zero_state.x_registers[1], 41);
}

#[test]
fn brz_same_target_same_arguments_keeps_block_arguments() {
    let mut builder = ExecIrBuilder::default();

    let join = builder.create_block();
    let value = builder.add_block_parameter_at(join, Type::I64);

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    let arg = u64_const(&mut builder, 0xfeed_face);

    builder.terminate(Terminator::BrZ(cond, (join, vec![arg]), (join, vec![arg])));

    builder.switch_to(join);
    builder.store_x_reg::<1>(value);

    let compiled = compile(builder);

    let mut zero_state = ExecState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 0xfeed_face);

    let mut non_zero_state = ExecState::initial();
    non_zero_state.x_registers[0] = 123;
    assert_eq!(call_compiled(&compiled, &mut non_zero_state), 0);
    assert_eq!(non_zero_state.x_registers[1], 0xfeed_face);
}

#[test]
fn loop_carried_block_parameters_update_on_backedge() {
    let mut builder = ExecIrBuilder::default();

    let loop_block = builder.create_block();
    let exit_block = builder.create_block();

    let remaining = builder.add_block_parameter_at(loop_block, Type::I64);
    let acc = builder.add_block_parameter_at(loop_block, Type::I64);
    let result = builder.add_block_parameter_at(exit_block, Type::I64);

    let initial_remaining = builder.load_x_reg::<0>(IntWidth::W64);
    let initial_acc = u64_const(&mut builder, 0);

    builder.terminate(Terminator::Br((
        loop_block,
        vec![initial_remaining, initial_acc],
    )));

    builder.switch_to(loop_block);
    builder.add_safepoint();

    let one = u64_const(&mut builder, 1);
    let next_remaining = builder.isub(remaining, one);
    let next_acc = builder.iadd(acc, remaining);

    builder.terminate(Terminator::BrZ(
        next_remaining,
        (exit_block, vec![next_acc]),
        (loop_block, vec![next_remaining, next_acc]),
    ));

    builder.switch_to(exit_block);
    builder.store_x_reg::<1>(result);

    let mut state = ExecState::initial();
    state.x_registers[0] = 4;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 10);
}
