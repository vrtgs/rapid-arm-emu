use crate::helper::{call_compiled, compile, run, store_x_const};
use emu_abi::processor_state::ProcessorState;
use exec_ir::{ExecIrBuilder, IConst, IntWidth, Terminator};

mod helper;

#[test]
fn return_fail_returns_halt_reason_and_preserves_prior_stores() {
    let mut builder = ExecIrBuilder::default();

    store_x_const::<0>(&mut builder, 0x1234);

    let halt_reason = builder.iconst(IConst::u32(0x4d2));
    builder.terminate(Terminator::ReturnCode(halt_reason));

    let mut state = ProcessorState::initial();
    assert_eq!(run(builder, &mut state), 0x4d2);
    assert_eq!(state.x_registers[0], 0x1234);
}

#[test]
fn branch_to_return_fail_only_fails_on_taken_path() {
    let mut builder = ExecIrBuilder::default();

    let fail = builder.create_block();
    let ok = builder.create_block();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    builder.terminate(Terminator::BrZ(cond, ok, fail));

    builder.switch_to(fail);
    let halt_reason = builder.iconst(IConst::u32(0xbeef));
    builder.terminate(Terminator::ReturnCode(halt_reason));

    builder.switch_to(ok);
    store_x_const::<1>(&mut builder, 0x600d);

    let compiled = compile(builder);

    let mut fail_state = ProcessorState::initial();
    fail_state.x_registers[0] = 1;
    assert_eq!(call_compiled(&compiled, &mut fail_state), 0xbeef);
    assert_eq!(fail_state.x_registers[1], 0);

    let mut ok_state = ProcessorState::initial();
    ok_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut ok_state), 0);
    assert_eq!(ok_state.x_registers[1], 0x600d);
}
