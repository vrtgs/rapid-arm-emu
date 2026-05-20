use crate::helper::{call_compiled, compile, empty_io_mmu, run_full, run_success, store_x_const};
use emu_abi::halt_reason::HaltReason;
use emu_abi::processor_state::ProcessorState;
use exec_ir::{ExecIrBuilder, IConst, IntWidth, IrBuilderConfig, Terminator};
use std::num::NonZero;

mod helper;

#[test]
fn instruction_done_increments_pc_by_four_each_time() {
    let mut builder = ExecIrBuilder::default();

    builder.next_insn();
    builder.next_insn();
    builder.next_insn();

    let mut state = ProcessorState::initial();
    state.pc = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.pc, 0x1000 + 4 * 3);
}

#[test]
fn explicit_halt_check_every_instruction_still_retires_all_instructions() {
    let mut builder = ExecIrBuilder::with_config(IrBuilderConfig {
        halt_check_every: NonZero::new(1).unwrap(),
    });

    builder.next_insn();
    builder.next_insn();
    builder.next_insn();
    builder.next_insn();

    let mut state = ProcessorState::initial();
    state.pc = 0x40;

    run_success(builder, &mut state);

    assert_eq!(state.pc, 0x50);
}

#[test]
fn automatic_halt_check_split_at_default_interval_preserves_pc_progress() {
    let instructions = 1024_u64;

    let halt_check_every = NonZero::new(instructions / 4).unwrap().try_into().unwrap();

    let mut builder = ExecIrBuilder::with_config(IrBuilderConfig { halt_check_every });

    for _ in 0..instructions {
        builder.next_insn();
    }

    let compiled = compile(builder);

    let mut state = ProcessorState::initial();
    for inital_pc in [0x1000, 0x1100, 0x8080] {
        state.pc = inital_pc;
        call_compiled(&compiled, &mut state);
        assert_eq!(state.pc, inital_pc.strict_add(instructions.strict_mul(4)));
    }
}

#[test]
fn backedge_halt_guard_after_instruction_done_preserves_loop_retirement() {
    let mut builder = ExecIrBuilder::default();

    let loop_block = builder.create_block();
    let exit_block = builder.create_block();

    builder.terminate(Terminator::Br(loop_block));

    builder.switch_to(loop_block);
    let current = builder.load_x_reg::<0>(IntWidth::W64);
    let one = builder.iconst(IConst::u64(1));
    let next = builder.sub(current, one);
    builder.store_x_reg::<0>(next);
    builder.next_insn();
    builder.terminate(Terminator::BrZ(next, exit_block, loop_block));

    builder.switch_to(exit_block);
    store_x_const::<1>(&mut builder, 0x5151);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 5;
    state.pc = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 0x5151);
    assert_eq!(state.pc, 0x1000 + 5 * 4);
}

#[test]
fn halts_inifnite_loop() {
    let mut builder = ExecIrBuilder::default();

    let new_block = builder.create_block();
    // can't loop back to entry point; invalid IR
    builder.terminate(Terminator::Br(new_block));
    builder.switch_to(new_block);
    builder.add_safepoint();
    builder.terminate(Terminator::Br(new_block));

    let expected_code = HaltReason {
        opcode: NonZero::new(121).unwrap(),
        metadata: 0xbeef,
    };

    let code = run_full(
        builder,
        &mut ProcessorState::initial(),
        &empty_io_mmu(),
        |_processor_state, _io_mmu, halt_reason| {
            halt_reason.halt(expected_code);
        },
        |_, _, halt| assert!(halt.take().is_none()),
    );

    assert_eq!(expected_code.as_nz_u32().get(), code)
}

#[test]
#[should_panic]
fn inifnite_loop_with_no_safepoint() {
    let mut builder = ExecIrBuilder::default();

    let new_block = builder.create_block();
    // can't loop back to entry point; invalid IR
    builder.terminate(Terminator::Br(new_block));
    builder.switch_to(new_block);
    builder.terminate(Terminator::Br(new_block));

    let _ = builder.build();
}
