#![allow(missing_docs)]
#![allow(
    dead_code,
    reason = "not all of this module is always used by the test modules"
)]

use emu_abi::exec_state::ExecState;
use emu_abi::halt_reason::AtomicHaltReason;
use emu_abi::internal_traits::InitInPlace;
use emu_abi::memory::{PagePointer, Tlb};
use io_mmu::cpu_fabric::CpuFabric;
use io_mmu::icache::ICache;
use jit_compiler_core::chunk::CompiledExecChunk;
use jit_compiler_core::compiler::{CompileTier, ExecIrCompiler};
use jit_compiler_core::exec_context::ExecContext;
use jit_compiler_core::ir::{ExecIrBuilder, IConst, IntCmp, SSAValue, Terminator};
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::sync::LazyLock;

pub(crate) struct ICacheSink(());

impl ICache for ICacheSink {
    fn invalidate(&self, _: PagePointer) {}
}

unsafe impl InitInPlace for ICacheSink {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        this.write(ICacheSink(()))
    }
}

thread_local! {
    pub static TLB: RefCell<Tlb> = const { RefCell::new(Tlb::new()) };
}

pub(crate) type IoMMU<T = ICacheSink> = io_mmu::IoMMU<T>;

pub(crate) fn empty_io_mmu() -> IoMMU {
    IoMMU::new(CpuFabric::new())
}

static COMPILER: LazyLock<ExecIrCompiler> =
    LazyLock::new(|| ExecIrCompiler::default().with_show_disassembly());

pub(crate) fn compile(builder: ExecIrBuilder) -> CompiledExecChunk {
    COMPILER.compile(&builder.build(), CompileTier::Tier1)
}

pub(crate) fn call_compiled_full<T: ICache>(
    compiled: &CompiledExecChunk,
    exec_state: &mut ExecState,
    io_mmu: &IoMMU<T>,
    setup: impl FnOnce(&mut ExecState, &IoMMU<T>, &AtomicHaltReason),
    post_process: impl FnOnce(&mut ExecState, &IoMMU<T>, &AtomicHaltReason),
) -> u32 {
    let halt_reason = AtomicHaltReason::new();
    setup(exec_state, io_mmu, &halt_reason);

    TLB.with_borrow_mut(|tlb| {
        let mut extra = ExecContext::initial();
        let trap = compiled.call::<T>(exec_state, &mut extra, tlb, &halt_reason, io_mmu);
        post_process(exec_state, io_mmu, &halt_reason);
        trap
    })
}

pub(crate) fn call_compiled(compiled: &CompiledExecChunk, processor_state: &mut ExecState) -> u32 {
    call_compiled_full(
        compiled,
        processor_state,
        &empty_io_mmu(),
        |_, _, _| {},
        |_, _, _| {},
    )
}

pub(crate) fn run_full<T: ICache>(
    builder: ExecIrBuilder,
    processor_state: &mut ExecState,
    io_mmu: &IoMMU<T>,
    setup: impl FnOnce(&mut ExecState, &IoMMU<T>, &AtomicHaltReason),
    post_process: impl FnOnce(&mut ExecState, &IoMMU<T>, &AtomicHaltReason),
) -> u32 {
    let compiled = compile(builder);
    call_compiled_full(&compiled, processor_state, io_mmu, setup, post_process)
}

pub(crate) fn run_with_mmu<T: ICache>(
    builder: ExecIrBuilder,
    processor_state: &mut ExecState,
    io_mmu: &IoMMU<T>,
) -> u32 {
    run_full(builder, processor_state, io_mmu, |_, _, _| {}, |_, _, _| {})
}

pub(crate) fn run(builder: ExecIrBuilder, processor_state: &mut ExecState) -> u32 {
    let compiled = compile(builder);
    call_compiled(&compiled, processor_state)
}

pub(crate) fn run_success(builder: ExecIrBuilder, processor_state: &mut ExecState) {
    assert_eq!(run(builder, processor_state), 0);
}

pub(crate) fn u64_const(builder: &mut ExecIrBuilder, value: u64) -> SSAValue {
    builder.iconst(IConst::u64(value))
}

pub(crate) fn store_x_const<const REG_IDX: u8>(builder: &mut ExecIrBuilder, value: u64) {
    let value = u64_const(builder, value);
    builder.store_x_reg::<REG_IDX>(value);
}

pub(crate) fn branch_to_store_x1(
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

pub(crate) fn store_bool_as_x_reg<const REG_IDX: u8>(builder: &mut ExecIrBuilder, cond: SSAValue) {
    let one = builder.iconst(IConst::u64(1));
    let zero = builder.iconst(IConst::u64(0));
    let value = builder.select(cond, one, zero);
    builder.store_x_reg::<REG_IDX>(value);
}

pub(crate) fn store_int_equals_as_x_reg<const REG_IDX: u8>(
    builder: &mut ExecIrBuilder,
    value: SSAValue,
    expected: IConst,
) {
    let ok = builder.icmp_imm(IntCmp::Equal, value, expected);
    store_bool_as_x_reg::<REG_IDX>(builder, ok);
}

pub(crate) fn clear_pstate(builder: &mut ExecIrBuilder) {
    let zero = builder.iconst(IConst::u32(0));
    builder.store_pstate(zero);
}

pub(crate) fn store_pstate_equals_as_x_reg<const REG_IDX: u8>(
    builder: &mut ExecIrBuilder,
    expected: u32,
) {
    let pstate = builder.load_pstate();
    let ok = builder.icmp_imm(IntCmp::Equal, pstate, IConst::u32(expected));
    store_bool_as_x_reg::<REG_IDX>(builder, ok);
}
