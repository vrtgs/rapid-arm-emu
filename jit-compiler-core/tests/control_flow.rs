#![allow(missing_docs)]

use crate::helper::{branch_to_store_x1, run_success, store_x_const, u64_const};
use emu_abi::exec_state::ExecState;
use jit_compiler_core::ir::{Block, ExecIrBuilder, IntWidth, Terminator};

mod helper;

#[test]
fn brnz_takes_zero_path_for_zero_condition() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

    let mut state = ExecState::initial();
    state.x_registers[0] = 0;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xbbbb);
}

#[test]
fn brnz_takes_non_zero_path_for_non_zero_condition() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

    let mut state = ExecState::initial();
    state.x_registers[0] = 42;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xaaaa);
}

#[test]
fn brnz_accepts_narrow_integer_conditions() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W8);
    branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

    let mut state = ExecState::initial();
    state.x_registers[0] = 0x0100;

    run_success(builder, &mut state);

    assert_eq!(
        state.x_registers[1], 0x2222,
        "low byte is zero, so W8 condition must be false",
    );

    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W8);
    branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

    let mut state = ExecState::initial();
    state.x_registers[0] = 0x0101;

    run_success(builder, &mut state);

    assert_eq!(
        state.x_registers[1], 0x1111,
        "low byte is non-zero, so W8 condition must be true",
    );
}

#[test]
#[allow(clippy::unusual_byte_groupings)]
fn unconditional_branch_executes_target_block() {
    let mut builder = ExecIrBuilder::default();

    let target = builder.create_block();
    builder.terminate(Terminator::Br(target));

    builder.switch_to(target);
    store_x_const::<0>(&mut builder, 0xdecaf_bad);

    let mut state = ExecState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0xdecaf_bad);
}

#[test]
fn diamond_control_flow_rejoins_through_processor_state() {
    fn build_program() -> ExecIrBuilder {
        let mut builder = ExecIrBuilder::default();

        let non_zero = builder.create_block();
        let zero = builder.create_block();
        let join = builder.create_block();

        let cond = builder.load_x_reg::<0>(IntWidth::W64);
        builder.terminate(Terminator::BrZ(cond, zero, non_zero));

        builder.switch_to(non_zero);
        store_x_const::<1>(&mut builder, 40);
        builder.terminate(Terminator::Br(join));

        builder.switch_to(zero);
        store_x_const::<1>(&mut builder, 2);
        builder.terminate(Terminator::Br(join));

        builder.switch_to(join);
        let x1 = builder.load_x_reg::<1>(IntWidth::W64);
        let one = u64_const(&mut builder, 1);
        let result = builder.iadd(x1, one);
        builder.store_x_reg::<2>(result);

        builder
    }

    let mut state = ExecState::initial();
    state.x_registers[0] = 0;
    run_success(build_program(), &mut state);
    assert_eq!(state.x_registers[1], 2);
    assert_eq!(state.x_registers[2], 3);

    let mut state = ExecState::initial();
    state.x_registers[0] = 1;
    run_success(build_program(), &mut state);
    assert_eq!(state.x_registers[1], 40);
    assert_eq!(state.x_registers[2], 41);
}

#[test]
fn manually_cold_block_still_executes_when_reached() {
    let mut builder = ExecIrBuilder::default();

    let hot = builder.create_block();
    let cold = builder.create_block();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    builder.terminate(Terminator::BrZ(cond, hot, cold));

    builder.switch_to(hot);
    store_x_const::<1>(&mut builder, 0x1234);

    builder.switch_to(cold);
    builder.mark_block_bold(cold);
    store_x_const::<1>(&mut builder, 0xc01d);

    let mut state = ExecState::initial();
    state.x_registers[0] = 1;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xc01d);
}

#[test]
fn simple_counted_loop_executes_until_condition_is_zero() {
    let mut builder = ExecIrBuilder::default();

    let loop_block = builder.create_block();
    let exit_block = builder.create_block();

    builder.terminate(Terminator::Br(loop_block));
    builder.switch_to(loop_block);

    builder.add_safepoint();

    let current = builder.load_x_reg::<0>(IntWidth::W64);
    let one = u64_const(&mut builder, 1);
    let next = builder.isub(current, one);
    builder.store_x_reg::<0>(next);
    builder.terminate(Terminator::BrZ(next, exit_block, loop_block));

    builder.switch_to(exit_block);
    store_x_const::<1>(&mut builder, 0x0600_d100_u64);

    let mut state = ExecState::initial();
    state.x_registers[0] = 7;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 0x0600_d100_u64);
}

#[test]
fn unreachable_blocks_do_not_execute() {
    let mut builder = ExecIrBuilder::default();

    let unreachable = builder.create_block();

    builder.switch_to(unreachable);
    store_x_const::<0>(&mut builder, 0xbad);

    builder.switch_to(Block::ENTRYPOINT);
    store_x_const::<0>(&mut builder, 0x600d);

    let mut state = ExecState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x600d);
}
