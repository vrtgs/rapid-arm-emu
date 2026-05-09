use crate::helper::{run_success, store_int_equals_as_x_reg, store_x_const, u64_const};
use emu_abi::processor_state::ProcessorState;
use exec_ir::{ExecIrBuilder, IConst, IntWidth, Terminator};

mod helper;

#[test]
fn iconst_can_store_to_x_registers_sp_and_pc() {
    let mut builder = ExecIrBuilder::default();

    store_x_const::<0>(&mut builder, 0x0123_4567_89ab_cdef);
    store_x_const::<1>(&mut builder, 0xfedc_ba98_7654_3210);

    let sp = u64_const(&mut builder, 0x1000_2000_3000_4000);
    builder.store_sp(sp);

    let pc = u64_const(&mut builder, 0x5555_6666_7777_8888);
    builder.store_pc(pc);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x0123_4567_89ab_cdef);
    assert_eq!(state.x_registers[1], 0xfedc_ba98_7654_3210);
    assert_eq!(state.sp, 0x1000_2000_3000_4000);
    assert_eq!(state.pc, 0x5555_6666_7777_8888);
}

#[test]
fn load_x_reg_const_and_dyn_then_store_roundtrip() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    builder.store_x_reg::<2>(x0);

    let x1 = builder.load_x_reg_dyn(1, IntWidth::W64);
    builder.store_x_reg_dyn(3, x1);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xaaaa_bbbb_cccc_dddd;
    state.x_registers[1] = 0x1111_2222_3333_4444;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[2], 0xaaaa_bbbb_cccc_dddd);
    assert_eq!(state.x_registers[3], 0x1111_2222_3333_4444);
}

#[test]
fn load_sp_and_pc_can_feed_arithmetic_and_stores() {
    let mut builder = ExecIrBuilder::default();

    let sp = builder.load_sp();
    let sp_delta = u64_const(&mut builder, 0x20);
    let adjusted_sp = builder.add(sp, sp_delta);
    builder.store_x_reg::<0>(adjusted_sp);

    let pc = builder.load_pc();
    let pc_delta = u64_const(&mut builder, 0x44);
    let adjusted_pc = builder.add(pc, pc_delta);
    builder.store_x_reg::<1>(adjusted_pc);

    let mut state = ProcessorState::initial();
    state.sp = 0x1000;
    state.pc = 0x8000;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x1020);
    assert_eq!(state.x_registers[1], 0x8044);
}

#[test]
fn lower_width_x_register_loads_read_low_order_bits() {
    let mut builder = ExecIrBuilder::default();

    let bad = builder.create_block();

    let got8 = builder.load_x_reg::<0>(IntWidth::W8);
    let expected8 = builder.iconst(IConst::u8(0x88));
    let diff8 = builder.sub(got8, expected8);
    let check16 = builder.create_block();

    builder.terminate(Terminator::BrZ(diff8, check16, bad));

    builder.switch_to(check16);
    let got16 = builder.load_x_reg::<0>(IntWidth::W16);
    let expected16 = builder.iconst(IConst::u16(0x7788));
    let diff16 = builder.sub(got16, expected16);
    let check32 = builder.create_block();

    builder.terminate(Terminator::BrZ(diff16, check32, bad));

    builder.switch_to(check32);
    let got32 = builder.load_x_reg::<0>(IntWidth::W32);
    let expected32 = builder.iconst(IConst::u32(0x5566_7788));
    let diff32 = builder.sub(got32, expected32);
    let good = builder.create_block();

    builder.terminate(Terminator::BrZ(diff32, good, bad));

    builder.switch_to(good);
    store_x_const::<1>(&mut builder, 0x600d);

    builder.switch_to(bad);
    store_x_const::<1>(&mut builder, 0xbad);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x1122_3344_5566_7788;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0x600d);
}

#[test]
fn dynamic_narrow_register_loads_read_low_order_bits() {
    let mut builder = ExecIrBuilder::default();

    let got = builder.load_x_reg_dyn(0, IntWidth::W8);
    store_int_equals_as_x_reg::<1>(&mut builder, got, IConst::u8(0xef));

    let got = builder.load_x_reg_dyn(0, IntWidth::W16);
    store_int_equals_as_x_reg::<2>(&mut builder, got, IConst::u16(0xcdef));

    let got = builder.load_x_reg_dyn(0, IntWidth::W32);
    store_int_equals_as_x_reg::<3>(&mut builder, got, IConst::u32(0x89ab_cdef));

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x0123_4567_89ab_cdef;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 1);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(state.x_registers[3], 1);
}

#[test]
fn explicit_pc_store_then_instruction_done_uses_new_pc() {
    let mut builder = ExecIrBuilder::default();

    let pc = u64_const(&mut builder, 0x2000);
    builder.store_pc(pc);
    builder.next_insn();

    let mut state = ProcessorState::initial();
    state.pc = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.pc, 0x2004);
}
