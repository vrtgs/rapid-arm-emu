use crate::helper::{
    TLB, call_compiled_full, compile, run_with_mmu, store_int_equals_as_x_reg, u64_const,
};
use emu_abi::halt_reason::HaltReason;
use emu_abi::internal_traits::{CpuFabricPrivate, ICache, InitInPlace};
use emu_abi::memory::{MemProt, PAGE_SIZE, PAGE_SIZE_U64, PagePointer, TLB_SIZE};
use emu_abi::processor_state::ProcessorState;
use exec_ir::{ExecIrBuilder, IConst, IntCmp, IntWidth};
use io_mmu::NoCache;
use io_mmu::cpu_fabric::CpuFabric;
use std::cell::RefCell;
use std::collections::HashSet;
use std::mem::MaybeUninit;
use std::num::NonZero;

mod helper;

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

struct ICacheSink(RefCell<HashSet<PagePointer>>);

impl ICache for ICacheSink {
    fn invalidate(&self, page: PagePointer) {
        self.0.borrow_mut().insert(page);
    }
}

unsafe impl InitInPlace for ICacheSink {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        this.write(ICacheSink(RefCell::new(HashSet::new())))
    }
}

type IoMMU = io_mmu::IoMMU<ICacheSink>;

fn iommu_with_sparse_page_protections_and_bytes(
    protections: &[(usize, MemProt)],
    mut byte: impl FnMut(usize, usize) -> u8,
) -> IoMMU {
    assert!(!protections.is_empty());

    let mut mmu = IoMMU::new(CpuFabric::new());
    let page_size = u64::try_from(PAGE_SIZE).unwrap();

    let mut needs_flush = false;

    for &(page, protections) in protections {
        let base = vm_page_addr(page);
        mmu.map(base, page_size, protections).unwrap();

        let mut needs_store = false;
        let page = std::array::from_fn::<u8, PAGE_SIZE, _>(|i| {
            let byte = byte(page, i);
            needs_store |= byte != 0;
            byte
        });

        if needs_store {
            TLB.with_borrow_mut(|tlb| {
                mmu.store_force(tlb, base, &page).unwrap();
            });
            if protections.contains(MemProt::EXECUTE) {
                needs_flush = true;
            }
        }
    }

    if needs_flush {
        mmu.flush_dirty_pages();
        mmu.get_fabric().icache().0.borrow_mut().clear();
    }

    mmu
}

fn iommu_with_page_protections_and_bytes(
    protections: &[MemProt],
    mut byte: impl FnMut(usize) -> u8,
) -> IoMMU {
    iommu_with_sparse_page_protections_and_bytes(
        &protections.iter().copied().enumerate().collect::<Vec<_>>(),
        |page, i| byte(page.strict_mul(PAGE_SIZE).strict_add(i)),
    )
}

fn iommu_with_bytes(pages: usize, protections: MemProt, byte: impl FnMut(usize) -> u8) -> IoMMU {
    let protections = (0..pages).map(|_| protections).collect::<Vec<_>>();
    iommu_with_page_protections_and_bytes(&protections, byte)
}

fn new_iommu(pages: usize, protections: MemProt) -> IoMMU {
    iommu_with_bytes(pages, protections, |_| 0)
}

fn run_success_with_mmu(
    builder: ExecIrBuilder,
    processor_state: &mut ProcessorState,
    io_mmu: &IoMMU,
) {
    assert_eq!(run_with_mmu(builder, processor_state, io_mmu), 0);
}

#[track_caller]
#[inline(always)]
fn assert_memory_trap(code: u32) {
    let code = NonZero::new(code)
        .map(HaltReason::from_u32)
        .map(|reason| reason.opcode);

    assert_eq!(Some(HaltReason::OPCODE_MEMORY_TRAP), code);
}

#[test]
fn vm_load64_reads_mapped_memory_after_tlb_miss() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_pattern_array::<8>(8)),
    );
}

#[test]
fn vm_store64_writes_mapped_memory_after_tlb_miss() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 16);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(mmu.load64_le(NoCache, 16).unwrap(), 0x0123_4567_89ab_cdef,);
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

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn vm_unaligned_store32_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 3);
    let value = builder.iconst(IConst::u32(0x89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(mmu.load32_le(NoCache, 3).unwrap(), 0x89ab_cdef);
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

    let mmu = iommu_with_bytes(2, MemProt::READ | MemProt::WRITE, vm_pattern_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(state.x_registers[0], 1);
}

#[test]
fn vm_cross_page_store64_uses_iommu_fallback() {
    let mut builder = ExecIrBuilder::default();

    let start = u64::try_from(PAGE_SIZE - 4).unwrap();
    let addr = u64_const(&mut builder, start);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        mmu.load64_le(NoCache, start).unwrap(),
        0x0123_4567_89ab_cdef,
    );
}

#[test]
fn vm_load_traps_when_page_is_out_of_bounds() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, VM_BASE);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xfeed_face;

    let mmu = IoMMU::new(CpuFabric::new());
    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xfeed_face);
}

#[test]
fn vm_load_traps_on_missing_read_permission() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, VM_BASE);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = new_iommu(1, MemProt::WRITE);
    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xfeed_face;

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xfeed_face);
}

#[test]
fn vm_store_traps_on_missing_write_permission_and_does_not_modify_memory() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = iommu_with_bytes(1, MemProt::READ, vm_pattern_byte);
    let expected = u64::from_le_bytes(vm_pattern_array::<8>(8));

    let mut state = ProcessorState::initial();
    let code = run_with_mmu(builder, &mut state, &mmu);
    assert_memory_trap(code);

    assert_eq!(mmu.load64_le(NoCache, 8).unwrap(), expected);
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

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

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

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    TLB.with_borrow_mut(|tlb| {
        assert_eq!(mmu.load_byte(&mut *tlb, 1).unwrap(), 0xa7);
        assert_eq!(mmu.load16_le(&mut *tlb, 2).unwrap(), 0xb8c9);
        assert_eq!(mmu.load32_le(&mut *tlb, 4).unwrap(), 0xdade_beef);
        assert_eq!(mmu.load64_le(&mut *tlb, 8).unwrap(), 0x0123_4567_89ab_cdef);
    });
}

#[test]
fn vm_tlb_hit_load_uses_cached_entry_for_same_page_and_offset() {
    let mut builder = ExecIrBuilder::default();

    // The first access populates the TLB through the IO-MMU fallback. The second
    // access should use the cached TLB entry rather than direct Page metadata.
    let addr = u64_const(&mut builder, 8);
    let _ = builder.vm_load(addr, IntWidth::W64);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(8)),
    );
}

#[test]
fn vm_tlb_collision_load_does_not_reuse_different_virtual_page_entry() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    // These pages map to the same TLB slot when the index is page_index & TLB_MASK.
    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let page0_loaded = builder.vm_load(page0_addr, IntWidth::W64);
    builder.store_x_reg::<0>(page0_loaded);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let alias_loaded = builder.vm_load(alias_addr, IntWidth::W64);
    builder.store_x_reg::<1>(alias_loaded);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[
            (0, MemProt::READ | MemProt::WRITE),
            (alias_page, MemProt::READ | MemProt::WRITE),
        ],
        |page, offset| tlb_collision_byte(page, offset, alias_page),
    );
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(tlb_collision_array::<8>(0, 0, alias_page)),
    );
    assert_eq!(
        state.x_registers[1],
        u64::from_le_bytes(tlb_collision_array::<8>(alias_page, 0, alias_page)),
    );
    assert_ne!(
        state.x_registers[0], state.x_registers[1],
        "aliasing TLB pages must not resolve to the same host page",
    );
}

#[test]
fn vm_tlb_collision_store_does_not_write_through_different_virtual_page_entry() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let page0_value = builder.iconst(IConst::u64(0x1111_2222_3333_4444));
    builder.vm_store(page0_addr, page0_value);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let alias_value = builder.iconst(IConst::u64(0xaaaa_bbbb_cccc_dddd));
    builder.vm_store(alias_addr, alias_value);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[
            (0, MemProt::READ | MemProt::WRITE),
            (alias_page, MemProt::READ | MemProt::WRITE),
        ],
        |_, _| 0,
    );
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    TLB.with_borrow_mut(|tlb| {
        assert_eq!(
            mmu.load64_le(&mut *tlb, vm_page_addr(0)).unwrap(),
            0x1111_2222_3333_4444,
        );
        assert_eq!(
            mmu.load64_le(&mut *tlb, vm_page_addr(alias_page)).unwrap(),
            0xaaaa_bbbb_cccc_dddd,
        );
    });
}

#[test]
fn vm_tlb_collision_load_to_unmapped_page_traps_instead_of_reusing_old_entry() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let _ = builder.vm_load(page0_addr, IntWidth::W64);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let loaded = builder.vm_load(alias_addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[(0, MemProt::READ | MemProt::WRITE)],
        |page, offset| tlb_collision_byte(page, offset, alias_page),
    );
    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x1234_5678_9abc_def0;

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0x1234_5678_9abc_def0);
}

#[test]
fn vm_tlb_collision_store_to_unmapped_page_traps_instead_of_reusing_old_entry() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let page0_loaded = builder.vm_load(page0_addr, IntWidth::W64);
    builder.store_x_reg::<0>(page0_loaded);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let alias_value = builder.iconst(IConst::u64(0xaaaa_bbbb_cccc_dddd));
    builder.vm_store(alias_addr, alias_value);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[(0, MemProt::READ | MemProt::WRITE)],
        |page, offset| tlb_collision_byte(page, offset, alias_page),
    );
    let page0_before = TLB.with_borrow_mut(|tlb| mmu.load64_le(tlb, vm_page_addr(0)).unwrap());
    let mut state = ProcessorState::initial();

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(
        TLB.with_borrow_mut(|tlb| mmu.load64_le(tlb, vm_page_addr(0)).unwrap()),
        page0_before,
    );
}

#[test]
fn vm_tlb_collision_load_permission_check_uses_colliding_target_page() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let _ = builder.vm_load(page0_addr, IntWidth::W64);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let loaded = builder.vm_load(alias_addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[
            (0, MemProt::READ | MemProt::WRITE),
            (alias_page, MemProt::WRITE),
        ],
        |page, offset| tlb_collision_byte(page, offset, alias_page),
    );

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0x1111_2222_3333_4444;

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0x1111_2222_3333_4444);
}

#[test]
fn vm_tlb_collision_store_permission_check_uses_colliding_target_page() {
    let mut builder = ExecIrBuilder::default();
    let alias_page = TLB_SIZE;

    let page0_addr = u64_const(&mut builder, vm_page_addr(0));
    let page0_loaded = builder.vm_load(page0_addr, IntWidth::W64);
    builder.store_x_reg::<0>(page0_loaded);

    let alias_addr = u64_const(&mut builder, vm_page_addr(alias_page));
    let alias_value = builder.iconst(IConst::u64(0xaaaa_bbbb_cccc_dddd));
    builder.vm_store(alias_addr, alias_value);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[
            (0, MemProt::READ | MemProt::WRITE),
            (alias_page, MemProt::READ),
        ],
        |page, offset| tlb_collision_byte(page, offset, alias_page),
    );

    let mut state = ProcessorState::initial();

    let (page0_before, alias_before) = TLB.with_borrow_mut(|tlb| {
        let page0_before = mmu.load64_le(&mut *tlb, vm_page_addr(0)).unwrap();
        let alias_before = mmu.load64_le(&mut *tlb, vm_page_addr(alias_page)).unwrap();

        (page0_before, alias_before)
    });

    let code = run_with_mmu(builder, &mut state, &mmu);
    assert_memory_trap(code);

    TLB.with_borrow_mut(|tlb| {
        assert_eq!(
            mmu.load64_le(&mut *tlb, vm_page_addr(0)).unwrap(),
            page0_before
        );
        assert_eq!(
            mmu.load64_le(&mut *tlb, vm_page_addr(alias_page)).unwrap(),
            alias_before,
        );
    });
}

#[test]
fn vm_tlb_hit_load_uses_page_number_not_zero_for_page1_offset0() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let _ = builder.vm_load(addr, IntWidth::W64);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_bytes(2, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_page_tagged_array::<8>(PAGE_SIZE)),
    );
}

#[test]
fn vm_tlb_hit_store_uses_page_number_not_zero_for_page1_offset0() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let warm_value = builder.iconst(IConst::u64(0x1111_2222_3333_4444));
    builder.vm_store(addr, warm_value);
    let value = builder.iconst(IConst::u64(0xfeed_face_cafe_beef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    TLB.with_borrow_mut(|tlb| {
        assert_eq!(mmu.load64_le(&mut *tlb, vm_page_addr(0)).unwrap(), 0);
        assert_eq!(
            mmu.load64_le(&mut *tlb, vm_page_addr(1)).unwrap(),
            0xfeed_face_cafe_beef,
        );
    });
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

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

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

        let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &mmu);

        assert_eq!(
            mmu.load16_le(NoCache, u64::try_from(start).unwrap())
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

        let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &mmu);

        assert_eq!(
            mmu.load32_le(NoCache, u64::try_from(start).unwrap())
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

        let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
        let mut state = ProcessorState::initial();

        run_success_with_mmu(builder, &mut state, &mmu);

        assert_eq!(
            mmu.load64_le(NoCache, u64::try_from(start).unwrap())
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

    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(state.x_registers[0], 1);
    assert_eq!(
        mmu.load_byte(NoCache, u64::try_from(last).unwrap())
            .unwrap(),
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
    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);

    let mut aligned_state = ProcessorState::initial();
    aligned_state.x_registers[0] = 8;
    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut aligned_state,
            &mmu,
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
            &mmu,
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

    let aligned_mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut aligned_state = ProcessorState::initial();
    aligned_state.x_registers[0] = 8;
    aligned_state.x_registers[1] = 0x1111_2222_3333_4444;

    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut aligned_state,
            &aligned_mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        aligned_mmu.load64_le(NoCache, 8).unwrap(),
        0x1111_2222_3333_4444,
    );

    let fallback_mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut fallback_state = ProcessorState::initial();
    fallback_state.x_registers[0] = 3;
    fallback_state.x_registers[1] = 0xaaaa_bbbb_cccc_dddd;

    assert_eq!(
        call_compiled_full(
            &compiled,
            &mut fallback_state,
            &fallback_mmu,
            |_, _, _| {},
            |_, _, _| {},
        ),
        0
    );
    assert_eq!(
        fallback_mmu.load64_le(NoCache, 3).unwrap(),
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

    let mmu = iommu_with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::WRITE],
        vm_page_tagged_byte,
    );

    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xdead_beef;

    let code = run_with_mmu(builder, &mut state, &mmu);

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

    let mmu = iommu_with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::READ],
        vm_page_tagged_byte,
    );

    let addrs = [
        PAGE_SIZE_U64.strict_sub(2),
        PAGE_SIZE_U64.strict_sub(1),
        vm_page_addr(1),
        vm_page_addr(1) + 1,
    ];

    let read_bytes =
        || TLB.with_borrow_mut(|tlb| addrs.map(|vaddr| mmu.load_byte(&mut *tlb, vaddr).unwrap()));

    let before = read_bytes();

    let mut state = ProcessorState::initial();
    let code = run_with_mmu(builder, &mut state, &mmu);
    assert_memory_trap(code);

    let after = read_bytes();
    assert_eq!(before, after);
}

#[test]
fn vm_cross_page_load64_roundtrips_exact_little_endian_bytes() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 3;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_bytes(2, MemProt::READ | MemProt::WRITE, vm_page_tagged_byte);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

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

    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        mmu.load64_le(NoCache, u64::try_from(start).unwrap())
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

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(state.x_registers[0], 0x9988_7766_5544_3322);
    assert_eq!(mmu.load64_le(NoCache, 32).unwrap(), 0x9988_7766_5544_3322);
}

#[test]
fn vm_store_then_load_same_unaligned_address_in_same_ir_sees_new_value_fallback_path() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 33);
    let stored = builder.iconst(IConst::u64(0x8877_6655_4433_2211));
    builder.vm_store(addr, stored);

    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(state.x_registers[0], 0x8877_6655_4433_2211);
    assert_eq!(mmu.load64_le(NoCache, 33).unwrap(), 0x8877_6655_4433_2211);
}

#[test]
fn vm_load_traps_when_tlb_miss_fallback_reports_unmapped_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[(0, MemProt::READ | MemProt::WRITE)],
        |page, offset| tlb_collision_byte(page, offset, usize::MAX),
    );
    let mut state = ProcessorState::initial();
    state.x_registers[0] = 0xaaaa_bbbb_cccc_dddd;

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(state.x_registers[0], 0xaaaa_bbbb_cccc_dddd);
}

#[test]
fn vm_store_traps_when_tlb_miss_fallback_reports_unmapped_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1));
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = iommu_with_sparse_page_protections_and_bytes(
        &[(0, MemProt::READ | MemProt::WRITE)],
        |_, _| 0,
    );
    let mut state = ProcessorState::initial();

    let code = run_with_mmu(builder, &mut state, &mmu);

    assert_memory_trap(code);
    assert_eq!(mmu.load64_le(NoCache, 0).unwrap(), 0);
}

#[allow(clippy::cast_possible_truncation)]
fn tlb_collision_byte(page: usize, offset: usize, alias_page: usize) -> u8 {
    let base = if page == alias_page { 0xa0u8 } else { 0x10u8 };
    base.wrapping_add(offset as u8)
}

fn tlb_collision_array<const N: usize>(page: usize, start: usize, alias_page: usize) -> [u8; N] {
    std::array::from_fn(|i| tlb_collision_byte(page, start.wrapping_add(i), alias_page))
}

fn drain_dirty_icache_pages(mmu: &IoMMU) -> usize {
    mmu.flush_dirty_pages();
    let lock = &mut *mmu.get_fabric().icache().0.borrow_mut();
    let count = lock.len();
    lock.clear();
    count
}

#[test]
fn vm_store_to_non_executable_page_does_not_dirty_icache_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 16);
    let value = builder.iconst(IConst::u64(0x0123_4567_89ab_cdef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let mut state = ProcessorState::initial();

    assert_eq!(drain_dirty_icache_pages(&mmu), 0);

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(mmu.load64_le(NoCache, 16).unwrap(), 0x0123_4567_89ab_cdef);
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        0,
        "stores to non-executable pages must not dirty icache state",
    );
}

#[test]
fn vm_store_to_executable_page_dirties_icache_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 16);
    let value = builder.iconst(IConst::u64(0xfeed_face_cafe_beef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE | MemProt::EXECUTE);
    let mut state = ProcessorState::initial();

    assert_eq!(drain_dirty_icache_pages(&mmu), 0);

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(mmu.load64_le(NoCache, 16).unwrap(), 0xfeed_face_cafe_beef);
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        1,
        "stores to executable pages must mark exactly that page dirty",
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        0,
        "draining dirty icache pages must clear the dirty state",
    );
}

#[test]
fn vm_load_from_executable_page_does_not_dirty_icache_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 8);
    let loaded = builder.vm_load(addr, IntWidth::W64);
    builder.store_x_reg::<0>(loaded);

    let mmu = iommu_with_bytes(
        1,
        MemProt::READ | MemProt::WRITE | MemProt::EXECUTE,
        vm_pattern_byte,
    );
    let mut state = ProcessorState::initial();

    assert_eq!(drain_dirty_icache_pages(&mmu), 0);

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        state.x_registers[0],
        u64::from_le_bytes(vm_pattern_array::<8>(8)),
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        0,
        "loads from executable pages must not dirty icache state",
    );
}

#[test]
fn repeated_vm_stores_to_same_executable_page_report_one_dirty_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, 16);
    let value = builder.iconst(IConst::u64(0x1111_2222_3333_4444));
    builder.vm_store(addr, value);

    let addr = u64_const(&mut builder, 24);
    let value = builder.iconst(IConst::u64(0x5555_6666_7777_8888));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE | MemProt::EXECUTE);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    TLB.with_borrow_mut(|tlb| {
        assert_eq!(mmu.load64_le(&mut *tlb, 16).unwrap(), 0x1111_2222_3333_4444);
        assert_eq!(mmu.load64_le(&mut *tlb, 24).unwrap(), 0x5555_6666_7777_8888);
    });
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        1,
        "multiple stores to one executable page should still report one dirty page",
    );
}

#[test]
fn vm_store_to_executable_page1_dirties_only_one_page() {
    let mut builder = ExecIrBuilder::default();

    let addr = u64_const(&mut builder, vm_page_addr(1).strict_add(32));
    let value = builder.iconst(IConst::u64(0xaabb_ccdd_eeff_0011));
    builder.vm_store(addr, value);

    let protections = [
        MemProt::READ | MemProt::WRITE | MemProt::EXECUTE,
        MemProt::READ | MemProt::WRITE | MemProt::EXECUTE,
    ];
    let mmu = iommu_with_page_protections_and_bytes(&protections, |_| 0);
    let mut state = ProcessorState::initial();

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        mmu.load64_le(NoCache, vm_page_addr(1).strict_add(32))
            .unwrap(),
        0xaabb_ccdd_eeff_0011,
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        1,
        "a store to page 1 should not dirty every executable page",
    );
}

#[test]
fn vm_unaligned_store64_fallback_dirties_executable_icache_page() {
    let mut builder = ExecIrBuilder::default();

    // Unaligned W64 forces the IO-MMU fallback path.
    let addr = u64_const(&mut builder, 3);
    let value = builder.iconst(IConst::u64(0xfeed_face_cafe_beef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE | MemProt::EXECUTE);
    let mut state = ProcessorState::initial();

    assert_eq!(drain_dirty_icache_pages(&mmu), 0);

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(mmu.load64_le(NoCache, 3).unwrap(), 0xfeed_face_cafe_beef);
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        1,
        "unaligned fallback stores to executable pages must dirty icache state",
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        0,
        "draining dirty icache pages must clear the dirty state",
    );
}

#[test]
fn vm_cross_page_store16_fallback_dirties_both_executable_icache_pages() {
    let mut builder = ExecIrBuilder::default();

    let start = PAGE_SIZE - 1;
    let addr = u64_const(&mut builder, u64::try_from(start).unwrap());
    let value = builder.iconst(IConst::u16(0xbeef));
    builder.vm_store(addr, value);

    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE | MemProt::EXECUTE);
    let mut state = ProcessorState::initial();

    assert_eq!(drain_dirty_icache_pages(&mmu), 0);

    run_success_with_mmu(builder, &mut state, &mmu);

    assert_eq!(
        mmu.load16_le(NoCache, u64::try_from(start).unwrap())
            .unwrap(),
        0xbeef,
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        2,
        "cross-page fallback stores touching two executable pages must dirty both pages",
    );
    assert_eq!(
        drain_dirty_icache_pages(&mmu),
        0,
        "draining dirty icache pages must clear both dirty states",
    );
}
