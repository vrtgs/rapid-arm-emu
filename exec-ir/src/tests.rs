use super::*;
use crate::compiler::{CompiledExecChunk, ExecIrCompiler};
use emu_abi::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
use emu_abi::memory::PAGE_SIZE;
use io_mmu::IoMMU;
use io_mmu::cpu_fabric::CpuFabric;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::LazyLock;

fn empty_io_mmu() -> IoMMU {
    IoMMU::new(CpuFabric::new())
}

static COMPILER: LazyLock<ExecIrCompiler> = LazyLock::new(ExecIrCompiler::new);

fn compile(builder: ExecIrBuilder) -> CompiledExecChunk {
    COMPILER.compile(builder.build())
}

fn call_compiled_full(
    compiled: &CompiledExecChunk,
    processor_state: &mut ProcessorState,
    io_mmu: &IoMMU,
    setup: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
    post_process: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
) -> u32 {
    let halt_reason = AtomicHaltReason::new();
    setup(processor_state, io_mmu, &halt_reason);
    let trap = compiled.call(processor_state, &halt_reason, io_mmu);
    post_process(processor_state, io_mmu, &halt_reason);
    trap
}

fn call_compiled(compiled: &CompiledExecChunk, processor_state: &mut ProcessorState) -> u32 {
    call_compiled_full(
        compiled,
        processor_state,
        &empty_io_mmu(),
        |_, _, _| {},
        |_, _, _| {},
    )
}

fn run_full(
    builder: ExecIrBuilder,
    processor_state: &mut ProcessorState,
    io_mmu: &IoMMU,
    setup: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
    post_process: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
) -> u32 {
    let compiled = compile(builder);
    call_compiled_full(&compiled, processor_state, io_mmu, setup, post_process)
}

fn run_with_mmu(
    builder: ExecIrBuilder,
    processor_state: &mut ProcessorState,
    io_mmu: &IoMMU,
) -> u32 {
    run_full(builder, processor_state, io_mmu, |_, _, _| {}, |_, _, _| {})
}

fn run(builder: ExecIrBuilder, processor_state: &mut ProcessorState) -> u32 {
    let compiled = compile(builder);
    call_compiled(&compiled, processor_state)
}

fn run_success(builder: ExecIrBuilder, processor_state: &mut ProcessorState) {
    assert_eq!(run(builder, processor_state), 0);
}

fn u64_const(builder: &mut ExecIrBuilder, value: u64) -> SSAValue {
    builder.iconst(IConst::u64(value))
}

fn store_x_const<const REG_IDX: u8>(builder: &mut ExecIrBuilder, value: u64) {
    let value = u64_const(builder, value);
    builder.store_x_reg::<REG_IDX>(value);
}

fn branch_to_store_x1(
    cond: SSAValue,
    builder: &mut ExecIrBuilder,
    non_zero_value: u64,
    zero_value: u64,
) {
    let non_zero = builder.create_block();
    let zero = builder.create_block();

    builder.terminate(Terminator::BrZ(cond, zero, non_zero));

    builder.switch_to(non_zero);
    store_x_const::<1>(builder, non_zero_value);

    builder.switch_to(zero);
    store_x_const::<1>(builder, zero_value);
}

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
fn wrapping_add_sub_mul_and_neg_work() {
    let mut builder = ExecIrBuilder::default();

    let max = u64_const(&mut builder, u64::MAX);
    let one = u64_const(&mut builder, 1);
    let add_wrapped = builder.add(max, one);
    builder.store_x_reg::<0>(add_wrapped);

    let zero = u64_const(&mut builder, 0);
    let sub_wrapped = builder.sub(zero, one);
    builder.store_x_reg::<1>(sub_wrapped);

    let high_bit = u64_const(&mut builder, 0x8000_0000_0000_0000);
    let two = u64_const(&mut builder, 2);
    let mul_wrapped = builder.mul(high_bit, two);
    builder.store_x_reg::<2>(mul_wrapped);

    let five = u64_const(&mut builder, 5);
    let neg_five = builder.neg(five);
    builder.store_x_reg::<3>(neg_five);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], u64::MAX);
    assert_eq!(state.x_registers[2], 0);
    assert_eq!(state.x_registers[3], (!5_u64).wrapping_add(1));
}

#[test]
fn arithmetic_can_use_loaded_registers() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let x1 = builder.load_x_reg::<1>(IntWidth::W64);
    let sum = builder.add(x0, x1);
    builder.store_x_reg::<2>(sum);

    let x2 = builder.load_x_reg::<2>(IntWidth::W64);
    let three = u64_const(&mut builder, 3);
    let product = builder.mul(x2, three);
    builder.store_x_reg::<3>(product);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 9;
    state.x_registers[1] = 11;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[2], 20);
    assert_eq!(state.x_registers[3], 60);
}

#[test]
fn unsigned_division_handles_normal_and_zero_divisors() {
    let mut builder = ExecIrBuilder::default();

    let hundred = u64_const(&mut builder, 100);
    let seven = u64_const(&mut builder, 7);
    let quotient = builder.udiv(hundred, seven);
    builder.store_x_reg::<0>(quotient);

    let numerator = u64_const(&mut builder, 1234);
    let zero = u64_const(&mut builder, 0);
    let div_by_zero = builder.udiv(numerator, zero);
    builder.store_x_reg::<1>(div_by_zero);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 14);
    assert_eq!(state.x_registers[1], 0);
}

#[test]
fn signed_division_handles_normal_zero_and_overflow_cases() {
    let mut builder = ExecIrBuilder::default();

    let minus_seventeen = u64_const(&mut builder, (-17_i64).cast_unsigned());
    let five = u64_const(&mut builder, 5);
    let quotient = builder.sdiv(minus_seventeen, five);
    builder.store_x_reg::<0>(quotient);

    let numerator = u64_const(&mut builder, (-123_i64).cast_unsigned());
    let zero = u64_const(&mut builder, 0);
    let div_by_zero = builder.sdiv(numerator, zero);
    builder.store_x_reg::<1>(div_by_zero);

    let int_min = u64_const(&mut builder, i64::MIN.cast_unsigned());
    let minus_one = u64_const(&mut builder, (-1_i64).cast_unsigned());
    let overflow = builder.sdiv(int_min, minus_one);
    builder.store_x_reg::<2>(overflow);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
    assert_eq!(state.x_registers[1], 0);
    assert_eq!(state.x_registers[2], i64::MIN.cast_unsigned());
}

#[test]
fn flag_setting_binops_produce_storable_values() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.load_x_reg::<0>(IntWidth::W64);
    let rhs = builder.load_x_reg::<1>(IntWidth::W64);
    let sum = builder.adds(lhs, rhs);
    builder.store_x_reg::<2>(sum);

    let lhs = builder.load_x_reg::<0>(IntWidth::W64);
    let rhs = builder.load_x_reg::<1>(IntWidth::W64);
    let diff = builder.subs(lhs, rhs);
    builder.store_x_reg::<3>(diff);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 40;
    state.x_registers[1] = 58;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[2], 98);
    assert_eq!(state.x_registers[3], 40_u64.wrapping_sub(58));
}

#[test]
fn brnz_takes_zero_path_for_zero_condition() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xbbbb);
}

#[test]
fn brnz_takes_non_zero_path_for_non_zero_condition() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W64);
    branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 42;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xaaaa);
}

#[test]
fn brnz_accepts_narrow_integer_conditions() {
    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W8);
    branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x0100;

    run_success(builder, &mut state);

    assert_eq!(
        state.x_registers[1], 0x2222,
        "low byte is zero, so W8 condition must be false",
    );

    let mut builder = ExecIrBuilder::default();

    let cond = builder.load_x_reg::<0>(IntWidth::W8);
    branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

    let mut state = ProcessorState::initial();
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

    let mut state = ProcessorState::initial();
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
        let result = builder.add(x1, one);
        builder.store_x_reg::<2>(result);

        builder
    }

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0;
    run_success(build_program(), &mut state);
    assert_eq!(state.x_registers[1], 2);
    assert_eq!(state.x_registers[2], 3);

    let mut state = ProcessorState::initial();
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

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 1;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 0xc01d);
}

#[test]
fn instruction_done_increments_pc_by_four_each_time() {
    let mut builder = ExecIrBuilder::default();

    builder.next_insn();
    builder.next_insn();
    builder.next_insn();

    let mut state = ProcessorState::initial();
    state.pc = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.pc, 0x100c);
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
    let next = builder.sub(current, one);
    builder.store_x_reg::<0>(next);
    builder.terminate(Terminator::BrZ(next, exit_block, loop_block));

    builder.switch_to(exit_block);
    store_x_const::<1>(&mut builder, 0x0600_d100_u64);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 7;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 0x0600_d100_u64);
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

#[test]
#[should_panic(expected = "arithmetic size mismatch")]
fn builder_rejects_arithmetic_width_mismatch() {
    let mut builder = ExecIrBuilder::default();

    let wide = builder.iconst(IConst::u64(1));
    let narrow = builder.iconst(IConst::u32(1));

    let _ = builder.add(wide, narrow);
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

fn store_bool_as_x_reg<const REG_IDX: u8>(builder: &mut ExecIrBuilder, cond: SSAValue) {
    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let value = builder.select(cond, one, zero);
    builder.store_x_reg::<REG_IDX>(value);
}

fn store_int_equals_as_x_reg<const REG_IDX: u8>(
    builder: &mut ExecIrBuilder,
    value: SSAValue,
    expected: IConst,
) {
    let ok = builder.icmp_imm(IntCmp::Equal, value, expected);
    store_bool_as_x_reg::<REG_IDX>(builder, ok);
}

fn clear_pstate(builder: &mut ExecIrBuilder) {
    let zero = builder.iconst(IConst::u32(0));
    builder.store_pstate(zero);
}

fn store_pstate_equals_as_x_reg<const REG_IDX: u8>(builder: &mut ExecIrBuilder, expected: u32) {
    let pstate = builder.load_pstate();
    let ok = builder.icmp_imm(IntCmp::Equal, pstate, IConst::u32(expected));
    store_bool_as_x_reg::<REG_IDX>(builder, ok);
}

#[test]
fn int_width_metadata_is_exact() {
    assert_eq!(IntWidth::from_bits(8), Some(IntWidth::W8));
    assert_eq!(IntWidth::from_bits(16), Some(IntWidth::W16));
    assert_eq!(IntWidth::from_bits(32), Some(IntWidth::W32));
    assert_eq!(IntWidth::from_bits(64), Some(IntWidth::W64));

    assert_eq!(IntWidth::from_bits(0), None);
    assert_eq!(IntWidth::from_bits(1), None);
    assert_eq!(IntWidth::from_bits(7), None);
    assert_eq!(IntWidth::from_bits(128), None);

    assert_eq!(IntWidth::W8.bits(), 8);
    assert_eq!(IntWidth::W16.bits(), 16);
    assert_eq!(IntWidth::W32.bits(), 32);
    assert_eq!(IntWidth::W64.bits(), 64);

    assert_eq!(IntWidth::W8.bytes(), 1);
    assert_eq!(IntWidth::W16.bytes(), 2);
    assert_eq!(IntWidth::W32.bytes(), 4);
    assert_eq!(IntWidth::W64.bytes(), 8);
}

#[test]
fn integer_comparisons_cover_signed_unsigned_and_immediates() {
    let mut builder = ExecIrBuilder::default();

    let minus_one = builder.iconst(IConst::i64(-1));
    let plus_one = builder.iconst(IConst::u64(1));

    let cond = builder.icmp(IntCmp::Equal, minus_one, plus_one);
    store_bool_as_x_reg::<0>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::NotEqual, minus_one, plus_one);
    store_bool_as_x_reg::<1>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedLessThan, minus_one, plus_one);
    store_bool_as_x_reg::<2>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedGreaterThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<3>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedGreaterThan, minus_one, plus_one);
    store_bool_as_x_reg::<4>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::SignedLessThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<5>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedLessThan, minus_one, plus_one);
    store_bool_as_x_reg::<6>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedGreaterThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<7>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedGreaterThan, minus_one, plus_one);
    store_bool_as_x_reg::<8>(&mut builder, cond);

    let cond = builder.icmp(IntCmp::UnsignedLessThanOrEqual, minus_one, plus_one);
    store_bool_as_x_reg::<9>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::SignedLessThan, minus_one, IConst::u64(1));
    store_bool_as_x_reg::<10>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::UnsignedGreaterThan, minus_one, IConst::u64(1));
    store_bool_as_x_reg::<11>(&mut builder, cond);

    let cond = builder.icmp_imm(IntCmp::Equal, minus_one, IConst::i64(-1));
    store_bool_as_x_reg::<12>(&mut builder, cond);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0);
    assert_eq!(state.x_registers[1], 1);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(state.x_registers[3], 0);
    assert_eq!(state.x_registers[4], 0);
    assert_eq!(state.x_registers[5], 1);
    assert_eq!(state.x_registers[6], 0);
    assert_eq!(state.x_registers[7], 1);
    assert_eq!(state.x_registers[8], 1);
    assert_eq!(state.x_registers[9], 0);
    assert_eq!(state.x_registers[10], 1);
    assert_eq!(state.x_registers[11], 1);
    assert_eq!(state.x_registers[12], 1);
}

#[test]
fn select_handles_both_paths_and_bool_values() {
    let mut builder = ExecIrBuilder::default();

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let cond = builder.icmp_imm(IntCmp::NotEqual, x0, IConst::u64(0));

    let true_value = builder.iconst(IConst::u64(0xaaaa));
    let false_value = builder.iconst(IConst::u64(0xbbbb));
    let selected = builder.select(cond, true_value, false_value);
    builder.store_x_reg::<1>(selected);

    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
    let false_bool = builder.icmp(IntCmp::Equal, one, zero);
    let selected_bool = builder.select(cond, true_bool, false_bool);
    store_bool_as_x_reg::<2>(&mut builder, selected_bool);

    let compiled = compile(builder);

    let mut zero_state = ProcessorState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 0xbbbb);
    assert_eq!(zero_state.x_registers[2], 0);

    let mut non_zero_state = ProcessorState::initial();
    non_zero_state.x_registers[0] = 1;
    assert_eq!(call_compiled(&compiled, &mut non_zero_state), 0);
    assert_eq!(non_zero_state.x_registers[1], 0xaaaa);
    assert_eq!(non_zero_state.x_registers[2], 1);
}

#[test]
fn bitwise_integer_and_bool_ops_cover_reg_and_immediate_forms() {
    let mut builder = ExecIrBuilder::default();

    let a = builder.iconst(IConst::u64(0xca));
    let b = builder.iconst(IConst::u64(0xac));

    let value = builder.bitand(a, b);
    builder.store_x_reg::<0>(value);

    let value = builder.bitor(a, b);
    builder.store_x_reg::<1>(value);

    let value = builder.bitxor(a, b);
    builder.store_x_reg::<2>(value);

    let value = builder.bitand_imm(a, IConst::u64(0xf0));
    builder.store_x_reg::<3>(value);

    let value = builder.bitor_imm(b, IConst::u64(0x03));
    builder.store_x_reg::<4>(value);

    let value = builder.bitxor_imm(a, IConst::u64(0xff));
    builder.store_x_reg::<5>(value);

    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
    let false_bool = builder.icmp(IntCmp::Equal, one, zero);

    let value = builder.bitand(true_bool, false_bool);
    store_bool_as_x_reg::<6>(&mut builder, value);

    let value = builder.bitor(true_bool, false_bool);
    store_bool_as_x_reg::<7>(&mut builder, value);

    let value = builder.bitxor(true_bool, false_bool);
    store_bool_as_x_reg::<8>(&mut builder, value);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x88);
    assert_eq!(state.x_registers[1], 0xee);
    assert_eq!(state.x_registers[2], 0x66);
    assert_eq!(state.x_registers[3], 0xc0);
    assert_eq!(state.x_registers[4], 0xaf);
    assert_eq!(state.x_registers[5], 0x35);
    assert_eq!(state.x_registers[6], 0);
    assert_eq!(state.x_registers[7], 1);
    assert_eq!(state.x_registers[8], 1);
}

#[test]
fn narrow_integer_ops_wrap_compare_and_divide_correctly() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(10));
    let value = builder.add(lhs, rhs);
    store_int_equals_as_x_reg::<0>(&mut builder, value, IConst::u8(4));

    let lhs = builder.iconst(IConst::u16(0));
    let rhs = builder.iconst(IConst::u16(1));
    let value = builder.sub(lhs, rhs);
    store_int_equals_as_x_reg::<1>(&mut builder, value, IConst::u16(u16::MAX));

    let lhs = builder.iconst(IConst::u32(0x8000_0000));
    let rhs = builder.iconst(IConst::u32(2));
    let value = builder.mul(lhs, rhs);
    store_int_equals_as_x_reg::<2>(&mut builder, value, IConst::u32(0));

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(10));
    let value = builder.udiv(lhs, rhs);
    store_int_equals_as_x_reg::<3>(&mut builder, value, IConst::u8(25));

    let lhs = builder.iconst(IConst::u8(250));
    let rhs = builder.iconst(IConst::u8(0));
    let value = builder.udiv(lhs, rhs);
    store_int_equals_as_x_reg::<4>(&mut builder, value, IConst::u8(0));

    let lhs = builder.iconst(IConst::i16(-9));
    let rhs = builder.iconst(IConst::i16(2));
    let value = builder.sdiv(lhs, rhs);
    store_int_equals_as_x_reg::<5>(&mut builder, value, IConst::i16(-4));

    let lhs = builder.iconst(IConst::i16(i16::MIN));
    let rhs = builder.iconst(IConst::i16(-1));
    let value = builder.sdiv(lhs, rhs);
    store_int_equals_as_x_reg::<6>(&mut builder, value, IConst::i16(i16::MIN));

    let value = builder.iconst(IConst::u8(1));
    let value = builder.neg(value);
    store_int_equals_as_x_reg::<7>(&mut builder, value, IConst::u8(255));

    let value = builder.iconst(IConst::u16(0x00ff));
    let value = builder.bitxor_imm(value, IConst::u16(0x0ff0));
    store_int_equals_as_x_reg::<8>(&mut builder, value, IConst::u16(0x0f0f));

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    for idx in 0..=8 {
        assert_eq!(state.x_registers[idx], 1, "x{idx}");
    }
}

#[test]
fn signed_and_unsigned_division_cover_more_rounding_edges() {
    let mut builder = ExecIrBuilder::default();

    let lhs = builder.iconst(IConst::i64(7));
    let rhs = builder.iconst(IConst::i64(-2));
    let value = builder.sdiv(lhs, rhs);
    builder.store_x_reg::<0>(value);

    let lhs = builder.iconst(IConst::i64(-7));
    let rhs = builder.iconst(IConst::i64(-2));
    let value = builder.sdiv(lhs, rhs);
    builder.store_x_reg::<1>(value);

    let lhs = builder.iconst(IConst::u64(u64::MAX));
    let rhs = builder.iconst(IConst::u64(u64::MAX));
    let value = builder.udiv(lhs, rhs);
    builder.store_x_reg::<2>(value);

    let lhs = builder.iconst(IConst::u64(7));
    let rhs = builder.iconst(IConst::u64(8));
    let value = builder.udiv(lhs, rhs);
    builder.store_x_reg::<3>(value);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
    assert_eq!(state.x_registers[1], 3);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(state.x_registers[3], 0);
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

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn adds_and_subs_set_arm_nzcv_flags_for_wrap_carry_and_overflow() {
    let mut builder = ExecIrBuilder::default();

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::u64(u64::MAX));
    let rhs = builder.iconst(IConst::u64(1));
    let value = builder.adds(lhs, rhs);
    builder.store_x_reg::<0>(value);
    store_pstate_equals_as_x_reg::<1>(&mut builder, PState::Z.0 | PState::C.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::i64(i64::MAX));
    let rhs = builder.iconst(IConst::i64(1));
    let value = builder.adds(lhs, rhs);
    builder.store_x_reg::<2>(value);
    store_pstate_equals_as_x_reg::<3>(&mut builder, PState::N.0 | PState::V.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::u64(0));
    let rhs = builder.iconst(IConst::u64(1));
    let value = builder.subs(lhs, rhs);
    builder.store_x_reg::<4>(value);
    store_pstate_equals_as_x_reg::<5>(&mut builder, PState::N.0);

    clear_pstate(&mut builder);
    let lhs = builder.iconst(IConst::i64(i64::MIN));
    let rhs = builder.iconst(IConst::i64(1));
    let value = builder.subs(lhs, rhs);
    builder.store_x_reg::<6>(value);
    store_pstate_equals_as_x_reg::<7>(&mut builder, PState::C.0 | PState::V.0);

    let mut state = ProcessorState::initial();
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

#[test]
fn unreachable_blocks_do_not_execute() {
    let mut builder = ExecIrBuilder::default();

    let unreachable = builder.create_block();

    builder.switch_to(unreachable);
    store_x_const::<0>(&mut builder, 0xbad);

    builder.switch_to(Block::ENTRYPOINT);
    store_x_const::<0>(&mut builder, 0x600d);

    let mut state = ProcessorState::initial();
    run_success(builder, &mut state);

    assert_eq!(state.x_registers[0], 0x600d);
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
    let mut builder = ExecIrBuilder::default();

    for _ in 0..520 {
        builder.next_insn();
    }

    let mut state = ProcessorState::initial();
    state.pc = 0x1000;

    run_success(builder, &mut state);

    assert_eq!(state.pc, 0x1000 + 520_u64 * 4);
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
#[should_panic(expected = "can't do pointer bitwise operations currently")]
fn builder_rejects_pointer_bitwise_operations() {
    let mut builder = ExecIrBuilder::default();

    let _ = builder.bitor(SSAValue::ARG_PROCESSOR_STATE, SSAValue::ARG_PAGES);
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
        payload: 0xbeef,
    };

    let code = run_full(
        builder,
        &mut ProcessorState::initial(),
        &empty_io_mmu(),
        |_processor_state, _io_mmu, halt_reason| {
            halt_reason.halt(expected_code);
        },
        |_, _, halt| assert_eq!(halt.take().bits(), 0),
    );

    assert_eq!(
        Some(expected_code),
        HaltReason::from_inner(HaltReasonInner::from_bits_retain(code))
    )
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

#[test]
fn block_parameter_passed_by_unconditional_branch_is_visible_in_target() {
    let mut builder = ExecIrBuilder::default();

    let target = builder.create_block();
    let param = builder.add_block_parameter_at(target, Type::I64);

    let x0 = builder.load_x_reg::<0>(IntWidth::W64);
    let delta = u64_const(&mut builder, 0x20);
    let arg = builder.add(x0, delta);

    builder.terminate(Terminator::Br((target, vec![arg])));

    builder.switch_to(target);
    let one = u64_const(&mut builder, 1);
    let result = builder.add(param, one);
    builder.store_x_reg::<1>(result);

    let mut state = ProcessorState::initial();
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
    let result = builder.add(value, one);
    builder.store_x_reg::<1>(result);

    let compiled = compile(builder);

    let mut zero_state = ProcessorState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 3);

    let mut non_zero_state = ProcessorState::initial();
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

    let mut zero_state = ProcessorState::initial();
    zero_state.x_registers[0] = 0;
    assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
    assert_eq!(zero_state.x_registers[1], 0xfeed_face);

    let mut non_zero_state = ProcessorState::initial();
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
    let next_remaining = builder.sub(remaining, one);
    let next_acc = builder.add(acc, remaining);

    builder.terminate(Terminator::BrZ(
        next_remaining,
        (exit_block, vec![next_acc]),
        (loop_block, vec![next_remaining, next_acc]),
    ));

    builder.switch_to(exit_block);
    builder.store_x_reg::<1>(result);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 4;

    run_success(builder, &mut state);

    assert_eq!(state.x_registers[1], 10);
}

const VM_BASE: u64 = 0;

fn vm_page_addr(page: usize) -> u64 {
    u64::try_from(page.strict_mul(PAGE_SIZE)).unwrap()
}

#[allow(clippy::cast_possible_truncation)]
fn vm_pattern_byte(i: usize) -> u8 {
    (i as u8).wrapping_mul(37).wrapping_add(0x51)
}

fn vm_pattern_array<const N: usize>(start: usize) -> [u8; N] {
    std::array::from_fn(|i| vm_pattern_byte(start.wrapping_add(i)))
}

struct VmPageBacking {
    ptr: NonNull<u8>,
    len: usize,
}

impl Drop for VmPageBacking {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(
                self.ptr.as_ptr(),
                Layout::from_size_align_unchecked(self.len, PAGE_SIZE),
            );
        }
    }
}

impl VmPageBacking {
    fn new(pages: usize) -> Self {
        assert!(pages > 0);

        let len = pages.strict_mul(PAGE_SIZE);
        let layout = Layout::from_size_align(len, PAGE_SIZE).unwrap();

        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

        Self { ptr, len }
    }

    fn get_page(&mut self, page: usize) -> &mut [u8; PAGE_SIZE] {
        let index = page.checked_mul(PAGE_SIZE).unwrap();
        assert!(index < self.len);

        unsafe { &mut *self.ptr.as_ptr().add(index).cast() }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

struct VmFixture {
    mmu: IoMMU,
    _backing: VmPageBacking,
}

impl VmFixture {
    fn new(pages: usize, protections: MemProt) -> Self {
        Self::with_bytes(pages, protections, |_| 0)
    }

    fn with_bytes(pages: usize, protections: MemProt, byte: impl FnMut(usize) -> u8) -> Self {
        let protections = vec![protections; pages];
        Self::with_page_protections_and_bytes(&protections, byte)
    }

    fn with_page_protections_and_bytes(
        protections: &[MemProt],
        mut byte: impl FnMut(usize) -> u8,
    ) -> Self {
        assert!(!protections.is_empty());

        let mut backing = VmPageBacking::new(protections.len());

        for (i, dst) in backing.as_mut_slice().iter_mut().enumerate() {
            *dst = byte(i);
        }

        let mut mmu = IoMMU::new(CpuFabric::new());
        let page_size = u64::try_from(PAGE_SIZE).unwrap();

        for (page, protections) in protections.iter().copied().enumerate() {
            unsafe {
                mmu.map_memory(
                    vm_page_addr(page),
                    backing.get_page(page).as_mut_ptr(),
                    page_size,
                    protections,
                )
                .unwrap();
            }
        }

        Self {
            mmu,
            _backing: backing,
        }
    }
}

fn run_success_with_mmu(
    builder: ExecIrBuilder,
    processor_state: &mut ProcessorState,
    io_mmu: &IoMMU,
) {
    assert_eq!(run_with_mmu(builder, processor_state, io_mmu), 0);
}

fn assert_memory_trap(code: u32) {
    assert_eq!(
        Some(HaltReason::MEMORY_TRAP),
        HaltReason::from_inner(HaltReasonInner::from_bits_retain(code))
    );
}

#[test]
fn vm_load64_fast_path_reads_mapped_memory() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_pattern_array::<8>(8)),
    );
}

#[test]
fn vm_store64_fast_path_writes_mapped_memory() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 16);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(fixture.mmu.load64_le(16).unwrap(), 0x0123_4567_89ab_cdef,);
}

#[test]
fn vm_unaligned_load32_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 3);
    let loaded = builder.vm_load(addr, IntWidth::W32);
    store_int_equals_as_x_reg::<0>(
        &mut builder,
        loaded,
        IConst::u32(u32::from_le_bytes(vm_pattern_array::<4>(3))),
    );

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn vm_unaligned_store32_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 3);
    let value = builder.iconst(IConst::u32(0x89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(fixture.mmu.load32_le(3).unwrap(), 0x89ab_cdef);
}

#[test]
fn vm_cross_page_load64_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let start = u64::try_from(PAGE_SIZE - 4).unwrap();
    let addr = u64_const(&mut builder, start);
    let loaded = builder.vm_load(addr, IntWidth::W64);

    store_int_equals_as_x_reg::<0>(
        &mut builder,
        loaded,
        IConst::u64(u64::from_le_bytes(vm_pattern_array::<8>(PAGE_SIZE - 4))),
    );

    let fixture = VmFixture::with_bytes(2, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn vm_cross_page_store64_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let start = u64::try_from(PAGE_SIZE - 4).unwrap();
    let addr = u64_const(&mut builder, start);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(fixture.mmu.load64_le(start).unwrap(), 0x0123_4567_89ab_cdef,);
}

#[test]
fn vm_load_traps_when_page_is_out_of_bounds() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, VM_BASE);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xfeed_face;

    let code = run(builder, &mut state);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xfeed_face);
}

#[test]
fn vm_load_traps_on_missing_read_permission() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, VM_BASE);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::new(1, MemProt::WRITE);
    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xfeed_face;

    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xfeed_face);
}

#[test]
fn vm_store_traps_on_missing_write_permission_and_does_not_modify_memory() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::with_bytes(1, MemProt::READ, vm_pattern_byte);
    let expected = u64::from_le_bytes(vm_pattern_array::<8>(8));

    let mut state = ProcessorState::initial();
    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(fixture.mmu.load64_le(8).unwrap(), expected);
}

#[test]
#[should_panic]
fn vm_load_rejects_non_i64_virtual_address() {
    let mut builder = ExecIrBuilder::default();

    let addr = builder.iconst(IConst::u32(0));
    let _ = builder.vm_load(addr, IntWidth::W64);
}

#[test]
#[should_panic(expected = "can only do integer vm stores on integers")]
fn vm_store_rejects_non_integer_value() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, VM_BASE);
    let one = u64_const(&mut builder, 1);
    let two = u64_const(&mut builder, 2);
    let bool_value = builder.icmp(IntCmp::Equal, one, two);

    builder.vm_store(addr, bool_value);
}

#[allow(clippy::cast_possible_truncation)]
fn vm_page_tagged_byte(i: usize) -> u8 {
    let page = i / PAGE_SIZE;
    let offset = i % PAGE_SIZE;

    (page as u8)
        .wrapping_mul(0x31)
        .wrapping_add(offset as u8)
        .wrapping_add(0x10)
}

fn vm_page_tagged_array<const N: usize>(start: usize) -> [u8; N] {
    std::array::from_fn(|i| vm_page_tagged_byte(start.wrapping_add(i)))
}

fn vm_expected_iconst(width: IntWidth, start: usize) -> IConst {
    match width {
        IntWidth::W8 => IConst::u8(vm_page_tagged_byte(start)),
        IntWidth::W16 => IConst::u16(u16::from_le_bytes(vm_page_tagged_array::<2>(start))),
        IntWidth::W32 => IConst::u32(u32::from_le_bytes(vm_page_tagged_array::<4>(start))),
        IntWidth::W64 => IConst::u64(u64::from_le_bytes(vm_page_tagged_array::<8>(start))),
    }
}

#[test]
fn vm_aligned_fast_path_loads_all_widths_from_page0_nonzero_offsets() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 1);
    let value = builder.vm_load(addr, IntWidth::W8);
    store_int_equals_as_x_reg::<0>(&mut builder, value, vm_expected_iconst(IntWidth::W8, 1));

    let addr = u64_const(&mut builder, 2);
    let value = builder.vm_load(addr, IntWidth::W16);
    store_int_equals_as_x_reg::<1>(&mut builder, value, vm_expected_iconst(IntWidth::W16, 2));

    let addr = u64_const(&mut builder, 4);
    let value = builder.vm_load(addr, IntWidth::W32);
    store_int_equals_as_x_reg::<2>(&mut builder, value, vm_expected_iconst(IntWidth::W32, 4));

    let addr = u64_const(&mut builder, 8);
    let value = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<3>(value);

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 1);
    assert_eq!(state.x_registers[1], 1);
    assert_eq!(state.x_registers[2], 1);
    assert_eq!(
        state.x_registers[3],
        u64::from_le_bytes(vm_page_tagged_array::<8>(8)),
    );
}

#[test]
fn vm_aligned_fast_path_stores_all_widths_to_page0_nonzero_offsets() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 1);
    let value = builder.iconst(IConst::u8(0xa7));
    builder.vm_store(addr, value);

    let addr = u64_const(&mut builder, 2);
    let value = builder.iconst(IConst::u16(0xb8c9));
    builder.vm_store(addr, value);

    let addr = u64_const(&mut builder, 4);
    let value = builder.iconst(IConst::u32(0xdade_beef));
    builder.vm_store(addr, value);

    let addr = u64_const(&mut builder, 8);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(fixture.mmu.load_byte(1).unwrap(), 0xa7);
    assert_eq!(fixture.mmu.load16_le(2).unwrap(), 0xb8c9);
    assert_eq!(fixture.mmu.load32_le(4).unwrap(), 0xdade_beef);
    assert_eq!(fixture.mmu.load64_le(8).unwrap(), 0x0123_4567_89ab_cdef);
}

#[test]
fn vm_fast_path_load_uses_page_number_not_page_offset_for_page0_offset8() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mut protections = vec![MemProt::READ | MemProt::WRITE; 9];

    // If the fast path incorrectly indexes page-table entries by page_offset,
    // vaddr 8 on page 0 will read Page[8].mem_prot instead of Page[0].mem_prot.
    protections[8] = MemProt::WRITE;

    let fixture = VmFixture::with_page_protections_and_bytes(&protections, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(8)),
    );
}

#[test]
fn vm_fast_path_load_uses_page_number_not_zero_for_page1_offset0() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::with_bytes(2, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(PAGE_SIZE)),
    );
}

#[test]
fn vm_fast_path_store_uses_page_number_not_zero_for_page1_offset0() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let value = builder.iconst(IConst::u64(0xfeed_face_cafe_beef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(fixture.mmu.load64_le(vm_page_addr(0)).unwrap(), 0);
    assert_eq!(
        fixture.mmu.load64_le(vm_page_addr(1)).unwrap(),
        0xfeed_face_cafe_beef,
    );
}

#[test]
fn vm_fast_path_load_uses_page_number_not_page_offset_for_nonzero_page_nonzero_offset() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE.strict_mul(2).strict_add(24);
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    // This makes the old bug deterministic:
    // correct page-table index is 2, but the broken index would be 24.
    let fixture = VmFixture::with_bytes(32, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(start)),
    );

    assert_ne!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(24 * PAGE_SIZE + 24)),
        "this would mean the fast path loaded from page table entry 24 instead of entry 2",
    );
}

#[test]
fn vm_fast_path_load_permission_check_uses_target_page_not_offset_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::WRITE],
        vm_page_tagged_byte,
    );

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x1111_2222_3333_4444;

    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0x1111_2222_3333_4444);
}

#[test]
fn vm_fast_path_store_permission_check_uses_target_page_not_offset_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let value = builder.iconst(IConst::u64(0xfeed_face_cafe_beef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::READ],
        |_| 0,
    );

    let mut state = ProcessorState::initial();
    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);

    assert_eq!(fixture.mmu.load64_le(vm_page_addr(0)).unwrap(), 0);
    assert_eq!(fixture.mmu.load64_le(vm_page_addr(1)).unwrap(), 0);
}

#[test]
fn vm_fast_path_load_at_exact_end_of_page_succeeds_for_w16_w32_and_w64() {
    let mut builder = ExecIrBuilder::default();

    let start16 = PAGE_SIZE - 2;
    let addr = u64_const(&mut builder, u64::try_from(start16).unwrap());
    let value = builder.vm_load(addr, IntWidth::W16);
    store_int_equals_as_x_reg::<0>(
        &mut builder,
        value,
        vm_expected_iconst(IntWidth::W16, start16),
    );

    let start32 = PAGE_SIZE - 4;
    let addr = u64_const(&mut builder, u64::try_from(start32).unwrap());
    let value = builder.vm_load(addr, IntWidth::W32);
    store_int_equals_as_x_reg::<1>(
        &mut builder,
        value,
        vm_expected_iconst(IntWidth::W32, start32),
    );

    let start64 = PAGE_SIZE - 8;
    let addr = u64_const(&mut builder, u64::try_from(start64).unwrap());
    let value = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<2>(value);

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 1);
    assert_eq!(state.x_registers[1], 1);
    assert_eq!(
        state.x_registers[2],
        u64::from_le_bytes(vm_page_tagged_array::<8>(start64)),
    );
}

#[test]
fn vm_fast_path_store_at_exact_end_of_page_succeeds_for_w16_w32_and_w64() {
    {
        let mut builder = ExecIrBuilder::default();

        let start = PAGE_SIZE - 2;
        let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
        let value = builder.iconst(IConst::u16(0x1234));
        builder.vm_store(addr, value);

        let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &fixture.mmu);

        assert_eq!(
            fixture
                .mmu
                .load16_le(u64::try_from(start).unwrap())
                .unwrap(),
            0x1234,
        );
    }

    {
        let mut builder = ExecIrBuilder::default();

        let start = PAGE_SIZE - 4;
        let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
        let value = builder.iconst(IConst::u32(0x4567_89ab));
        builder.vm_store(addr, value);

        let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &fixture.mmu);

        assert_eq!(
            fixture
                .mmu
                .load32_le(u64::try_from(start).unwrap())
                .unwrap(),
            0x4567_89ab,
        );
    }

    {
        let mut builder = ExecIrBuilder::default();

        let start = PAGE_SIZE - 8;
        let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
        let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
        builder.vm_store(addr, value);

        let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &fixture.mmu);

        assert_eq!(
            fixture
                .mmu
                .load64_le(u64::try_from(start).unwrap())
                .unwrap(),
            0x0123_4567_89ab_cdef,
        );
    }
}

#[test]
fn vm_byte_access_at_last_byte_of_page_uses_fast_path_and_succeeds() {
    let mut builder = ExecIrBuilder::default();

    let last = PAGE_SIZE - 1;

    let addr = u64_const(&mut builder, u64::try_from(last).unwrap());
    let loaded = builder.vm_load(addr, IntWidth::W8);
    store_int_equals_as_x_reg::<0>(&mut builder, loaded, vm_expected_iconst(IntWidth::W8, last));

    let value = builder.iconst(IConst::u8(0xee));
    builder.vm_store(addr, value);

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 1);
    assert_eq!(
        fixture.mmu.load_byte(u64::try_from(last).unwrap()).unwrap(),
        0xee,
    );
}

#[test]
fn vm_dynamic_load_can_take_fast_path_and_fallback_path_in_same_compiled_chunk() {
    let mut builder = ExecIrBuilder::default();

    let addr = builder.load_x_reg::<0>(IntWidth::W64);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<1>(loaded);

    let compiled = compile(builder);
    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);

    let mut aligned_state = ProcessorState::initial();
    aligned_state.x_registers[0] = 8;
    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut aligned_state,
            &fixture.mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        aligned_state.x_registers[1],
        u64::from_le_bytes(vm_page_tagged_array::<8>(8)),
    );

    let mut unaligned_state = ProcessorState::initial();
    unaligned_state.x_registers[0] = 3;
    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut unaligned_state,
            &fixture.mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        unaligned_state.x_registers[1],
        u64::from_le_bytes(vm_page_tagged_array::<8>(3)),
    );
}

#[test]
fn vm_dynamic_store_can_take_fast_path_and_fallback_path_in_same_compiled_chunk() {
    let mut builder = ExecIrBuilder::default();

    let addr = builder.load_x_reg::<0>(IntWidth::W64);
    let value = builder.load_x_reg::<1>(IntWidth::W64);
    builder.vm_store(addr, value);

    let compiled = compile(builder);

    let aligned_fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut aligned_state = ProcessorState::initial();
    aligned_state.x_registers[0] = 8;
    aligned_state.x_registers[1] = 0x1111_2222_3333_4444;

    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut aligned_state,
            &aligned_fixture.mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        aligned_fixture.mmu.load64_le(8).unwrap(),
        0x1111_2222_3333_4444,
    );

    let fallback_fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut fallback_state = ProcessorState::initial();
    fallback_state.x_registers[0] = 3;
    fallback_state.x_registers[1] = 0xaaaa_bbbb_cccc_dddd;

    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut fallback_state,
            &fallback_fixture.mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        fallback_fixture.mmu.load64_le(3).unwrap(),
        0xaaaa_bbbb_cccc_dddd,
    );
}

#[test]
fn vm_cross_page_load16_traps_when_second_page_lacks_read_permission() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 1;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let loaded = builder.vm_load(addr, IntWidth::W16);
    store_int_equals_as_x_reg::<0>(&mut builder, loaded, IConst::u16(0));

    let fixture = VmFixture::with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::WRITE],
        vm_page_tagged_byte,
    );

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xdead_beef;

    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xdead_beef);
}

#[test]
fn vm_cross_page_store32_traps_when_second_page_lacks_write_and_does_not_partially_store() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 2;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let value = builder.iconst(IConst::u32(0x0123_4567));
    builder.vm_store(addr, value);

    let fixture = VmFixture::with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::READ],
        vm_page_tagged_byte,
    );

    let before0 = fixture
        .mmu
        .load_byte(u64::try_from(PAGE_SIZE - 2).unwrap())
        .unwrap();
    let before1 = fixture
        .mmu
        .load_byte(u64::try_from(PAGE_SIZE - 1).unwrap())
        .unwrap();
    let before2 = fixture.mmu.load_byte(vm_page_addr(1)).unwrap();
    let before3 = fixture.mmu.load_byte(vm_page_addr(1) + 1).unwrap();

    let mut state = ProcessorState::initial();
    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);

    assert_eq!(
        fixture
            .mmu
            .load_byte(u64::try_from(PAGE_SIZE - 2).unwrap())
            .unwrap(),
        before0,
    );
    assert_eq!(
        fixture
            .mmu
            .load_byte(u64::try_from(PAGE_SIZE - 1).unwrap())
            .unwrap(),
        before1,
    );
    assert_eq!(fixture.mmu.load_byte(vm_page_addr(1)).unwrap(), before2);
    assert_eq!(fixture.mmu.load_byte(vm_page_addr(1) + 1).unwrap(), before3);
}

#[test]
fn vm_cross_page_load64_roundtrips_exact_little_endian_bytes() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 3;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::with_bytes(2, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(start)),
    );
}

#[test]
fn vm_cross_page_store64_roundtrips_exact_little_endian_bytes() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 3;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let value = builder.iconst(IConst::u64(0x1020_3040_5060_7080));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(
        fixture
            .mmu
            .load64_le(u64::try_from(start).unwrap())
            .unwrap(),
        0x1020_3040_5060_7080,
    );
}

#[test]
fn vm_store_then_load_same_address_in_same_ir_sees_new_value_fast_path() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 32);
    let stored = builder.iconst(IConst::u64(0x9988_7766_5544_3322));
    builder.vm_store(addr, stored);

    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 0x9988_7766_5544_3322);
    assert_eq!(fixture.mmu.load64_le(32).unwrap(), 0x9988_7766_5544_3322);
}

#[test]
fn vm_store_then_load_same_unaligned_address_in_same_ir_sees_new_value_fallback_path() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 33);
    let stored = builder.iconst(IConst::u64(0x8877_6655_4433_2211));
    builder.vm_store(addr, stored);

    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &fixture.mmu);

    assert_eq!(state.x_registers[0], 0x8877_6655_4433_2211);
    assert_eq!(fixture.mmu.load64_le(33).unwrap(), 0x8877_6655_4433_2211);
}

#[test]
fn vm_load_fast_path_traps_when_page_number_equals_page_count() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let fixture = VmFixture::with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xaaaa_bbbb_cccc_dddd;

    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xaaaa_bbbb_cccc_dddd);
}

#[test]
fn vm_store_fast_path_traps_when_page_number_equals_page_count() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let fixture = VmFixture::new(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    let code = run_with_mmu(builder, &mut state, &fixture.mmu);

    assert_memory_trap(code);
    assert_eq!(fixture.mmu.load64_le(0).unwrap(), 0);
}
