use crate::helper::{call_compiled, compile, run_success, u64_const};
use emu_abi::processor_state::ProcessorState;
use exec_ir::{Block, ExecIrBuilder, IntWidth};

mod helper;

#[test]
fn empty_ir_returns_success_and_preserves_basic_state() {
    let mut state = ProcessorState::initial();
    state.sp = 0x1111;
    state.pc = 0x2222;
    state.x_registers[0] = 0x3333;
    state.x_registers[1] = 0x4444;

    let builder = ExecIrBuilder::default();

    run_success(builder, &mut state);

    assert_eq!(state.sp, 0x1111);
    assert_eq!(state.pc, 0x2222);
    assert_eq!(state.x_registers[0], 0x3333);
    assert_eq!(state.x_registers[1], 0x4444);
}

#[test]
fn compiled_block_can_be_called_more_than_once() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let one = u64_const(&mut builder, 1);
    let incremented = builder.add(x0, one);
    builder.store_x_reg::<0>(incremented);

    let compiled = compile(builder);

    let mut first = ProcessorState::initial();
    first.x_registers[0] = 10;
    assert_eq!(call_compiled(&compiled, &mut first), 0);
    assert_eq!(first.x_registers[0], 11);

    let mut second = ProcessorState::initial();
    second.x_registers[0] = u64::MAX;
    assert_eq!(call_compiled(&compiled, &mut second), 0);
    assert_eq!(second.x_registers[0], 0);
}

#[test]
fn builder_current_block_tracks_switches() {
    let mut builder = ExecIrBuilder::default();

    assert_eq!(builder.current_block(), Block::ENTRYPOINT);

    let other = builder.create_block();
    builder.switch_to(other);

    assert_eq!(builder.current_block(), other);
}
