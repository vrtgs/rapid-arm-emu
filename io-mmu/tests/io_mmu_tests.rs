use std::collections::HashSet;

use emu_abi::convert::usize_to_u64;
use emu_abi::internal_traits::{CpuFabricPrivate, ICache, InitInPlace, IoMMUPrivate};
use emu_abi::memory::{MemProt, PAGE_SIZE, PAGE_SIZE_U64, PageNumber, PagePointer};
use io_mmu::cpu_fabric::CpuFabric;
use parking_lot::Mutex;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ptr::NonNull;

const BASE: u64 = PAGE_SIZE_U64.strict_mul(51);

const _: () = assert!(BASE.is_multiple_of(PAGE_SIZE_U64));

fn page_addr(page: usize) -> u64 {
    BASE.strict_add(u64::try_from(page.strict_mul(PAGE_SIZE)).unwrap())
}

fn pattern_byte_raw(i: u8) -> u8 {
    i.wrapping_mul(37).wrapping_add(0x51)
}

fn pattern_byte(i: u64) -> u8 {
    pattern_byte_raw((i & 0xFF) as u8)
}

#[allow(clippy::cast_possible_truncation)]
fn pattern_array<const N: usize>(start: u64) -> [u8; N] {
    let byte1 = start as u8;
    std::array::from_fn(move |i| {
        let byte2 = i as u8;
        pattern_byte_raw(byte1.wrapping_add(byte2))
    })
}

struct TestICache {
    set: Mutex<HashSet<PagePointer>>,
}

unsafe impl InitInPlace for TestICache {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        this.write(TestICache {
            set: Mutex::new(HashSet::new()),
        })
    }
}

impl ICache for TestICache {
    fn invalidate(&self, page: PagePointer) {
        self.set.lock().insert(page);
    }
}

type IoMMU = io_mmu::IoMMU<TestICache>;

fn write_pages_to_new_mmu(mmu: &IoMMU, start: u64, size: usize, mut byte: impl FnMut(u64) -> u8) {
    let mut bytes = None::<Box<[u8]>>;
    for i in 0..size {
        let vaddr = start.strict_add(u64::try_from(i).unwrap());
        let byte = byte(vaddr);
        if byte != 0 {
            let bytes = bytes.get_or_insert_with(|| bytemuck::zeroed_slice_box(size));

            bytes[i] = byte;
        }
    }

    if let Some(bytes) = bytes {
        mmu.store_force(start, &bytes).unwrap()
    }
}

fn iommu_with_bytes(pages: usize, protections: MemProt, byte: impl FnMut(u64) -> u8) -> IoMMU {
    let size = pages.strict_mul(PAGE_SIZE);

    let start = BASE;
    let size_u64 = u64::try_from(size).expect("requeted pages can't fit in 64 bti address space");

    let mut mmu = IoMMU::new(CpuFabric::new());
    mmu.map(start, size_u64, protections).unwrap();
    write_pages_to_new_mmu(&mmu, start, size, byte);
    mmu
}

fn iommu_with_page_protections_and_bytes(
    protections: &[MemProt],
    byte: impl FnMut(u64) -> u8,
) -> IoMMU {
    assert!(!protections.is_empty());

    let mut mmu = IoMMU::new(CpuFabric::new());
    for (page, protections) in protections.iter().copied().enumerate() {
        mmu.map(page_addr(page), PAGE_SIZE_U64, protections)
            .unwrap();
    }

    write_pages_to_new_mmu(&mmu, BASE, protections.len().strict_mul(PAGE_SIZE), byte);

    mmu
}

fn new_iommu(pages: usize, protections: MemProt) -> IoMMU {
    iommu_with_bytes(pages, protections, |_| 0)
}

fn iommu_with_page_protections(protections: &[MemProt]) -> IoMMU {
    iommu_with_page_protections_and_bytes(protections, |_| 0)
}

fn read_array<const N: usize>(mmu: &IoMMU, vaddr: u64) -> [u8; N] {
    let mut out = [0; N];
    mmu.load(vaddr, &mut out).unwrap();
    out
}

fn is_dirty(mmu: &IoMMU, page: usize) -> bool {
    mmu.flush_dirty_pages();

    const BASE_PAGE: u64 = {
        assert!(BASE % PAGE_SIZE_U64 == 0);
        BASE / PAGE_SIZE_U64
    };

    let offset_page = usize_to_u64(page).unwrap();
    let page = mmu
        .get_page(PageNumber(BASE_PAGE.strict_add(offset_page)))
        .unwrap();
    let pointer = page.ptr.page_ptr();
    mmu.get_fabric().icache().set.lock().contains(&pointer)
}

#[test]
fn new_mmu_faults_non_empty_accesses() {
    let mmu = IoMMU::new(CpuFabric::new());

    let mut one = [MaybeUninit::<u8>::uninit()];
    assert!(mmu.load_into_uninit(BASE, &mut one).is_err());
    assert!(mmu.store(BASE, &[1]).is_err());

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 1).is_err());

    assert!(mmu.load16_le(BASE).is_err());
    assert!(mmu.load32_le(BASE).is_err());
    assert!(mmu.load64_le(BASE).is_err());

    assert!(mmu.store16_le(BASE, 0x1234).is_err());
    assert!(mmu.store32_le(BASE, 0x1234_5678).is_err());
    assert!(mmu.store64_le(BASE, 0x1234_5678_9abc_def0).is_err());
}

#[test]
fn zero_length_load_and_store_do_not_require_mapping() {
    let mmu = IoMMU::new(CpuFabric::new());

    let mut empty: [MaybeUninit<u8>; 0] = [];

    let loaded = mmu.load_into_uninit(0x1234_5678, &mut empty).unwrap();
    assert!(loaded.is_empty());

    assert!(mmu.store(0x1234_5678, &[]).is_ok());
}

struct AllocManager {
    ptr: NonNull<u8>,
    len: usize,
}

impl AllocManager {
    fn new(pages: usize) -> Self {
        let len = pages.strict_mul(PAGE_SIZE);
        let ptr = match len {
            0 => NonNull::without_provenance(const { NonZero::new(PAGE_SIZE).unwrap() }),
            len => {
                let layout = std::alloc::Layout::from_size_align(len, PAGE_SIZE).unwrap();

                NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap()
            }
        };

        Self { ptr, len }
    }

    fn get_mem(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AllocManager {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.len, PAGE_SIZE);

                std::alloc::dealloc(self.ptr.as_ptr(), layout)
            }
        }
    }
}

#[test]
fn map_memory_rejects_unaligned_base_size_ptr_and_overflow() {
    let mut mmu = IoMMU::new(CpuFabric::new());

    let pages = 2;
    let mut manager = AllocManager::new(pages);
    let ptr = manager.get_mem().as_mut_ptr();

    // TODO
    //   make a test to make sure
    //   let overflow_base = u64::MAX.strict_sub((PAGE_SIZE as u64).strict_sub(1));
    //   (overflow_base, PAGE_SIZE_U64, 0)
    //   is accepted

    let overflow_base = u64::MAX.strict_sub((PAGE_SIZE as u64).strict_sub(1));

    let rejected_combinations = [
        (1, PAGE_SIZE_U64, 0_usize),
        (BASE, PAGE_SIZE_U64, 1),
        (BASE, PAGE_SIZE_U64 - 1, 0),
        (overflow_base, PAGE_SIZE_U64.strict_mul(2), 0),
    ];

    let total_allocated_bytes = u64::try_from(pages.strict_mul(PAGE_SIZE)).unwrap();

    for rejected_combination in rejected_combinations {
        let (base, allocation_size, offset) = rejected_combination;
        let prot = MemProt::READ;
        if offset == 0 {
            assert!(mmu.map(base, allocation_size, prot).is_err());
        }

        let res = unsafe {
            assert!(allocation_size <= total_allocated_bytes);
            mmu.map_shared(base, allocation_size, ptr.byte_add(offset), prot)
        };

        assert!(res.is_err())
    }
}

#[test]
fn nonzero_base_maps_only_requested_page() {
    let pages = 1;

    let mut manager = AllocManager::new(pages);
    let slice = manager.get_mem();

    {
        let first_byte = &mut slice[0];
        assert_eq!(*first_byte, 0);
        *first_byte = 0xaa;
    }

    let mut mmu = IoMMU::new(CpuFabric::new());
    let base = page_addr(2);

    unsafe {
        mmu.map_shared(base, PAGE_SIZE_U64, slice.as_mut_ptr(), MemProt::READ)
            .unwrap();
    }

    assert!(mmu.load_byte(0).is_err());
    assert!(mmu.load_byte(page_addr(1)).is_err());
    assert_eq!(mmu.load_byte(base).unwrap(), 0xaa);
}

#[test]
fn read_only_page_allows_loads_and_rejects_stores() {
    let mmu = new_iommu(1, MemProt::READ);

    assert_eq!(mmu.load_byte(BASE).unwrap(), 0);

    let mut one = [MaybeUninit::<u8>::uninit()];
    assert!(mmu.load_into_uninit(BASE, &mut one).is_ok());

    assert!(mmu.store_byte(BASE, 1).is_err());
    assert!(mmu.store(BASE, &[1]).is_err());
    assert!(mmu.store16_le(BASE, 0xbeef).is_err());
    assert!(mmu.store32_le(BASE, 0xfeed_beef).is_err());
    assert!(mmu.store64_le(BASE, 0xfeed_beef_dead_cafe).is_err());
}

#[test]
fn write_only_page_allows_stores_and_rejects_loads() {
    let mmu = new_iommu(1, MemProt::WRITE);

    assert!(mmu.store_byte(BASE, 1).is_ok());
    assert!(mmu.store(BASE, &[1, 2, 3]).is_ok());
    assert!(mmu.store16_le(BASE, 0xbeef).is_ok());

    assert!(mmu.load(BASE, &mut [0]).is_err());
    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.load16_le(BASE).is_err());
    assert!(mmu.load32_le(BASE).is_err());
    assert!(mmu.load64_le(BASE).is_err());
}

#[test]
fn execute_only_page_rejects_data_loads_and_stores() {
    let mmu = new_iommu(1, MemProt::EXECUTE);

    assert!(mmu.load(BASE, &mut [0]).is_err());
    assert!(mmu.store(BASE, &[1]).is_err());

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 1).is_err());

    assert!(mmu.load16_le(BASE).is_err());
    assert!(mmu.store16_le(BASE, 0xbeef).is_err());
}

#[test]
fn empty_protection_mapping_rejects_all_data_accesses() {
    let mmu = new_iommu(1, MemProt::NONE);

    assert!(mmu.load(BASE, &mut [0]).is_err());
    assert!(mmu.store(BASE, &[1]).is_err());

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 1).is_err());
}

#[test]
fn byte_load_store_roundtrip() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    for i in 0..64 {
        mmu.store_byte(BASE.strict_add(i), pattern_byte(i)).unwrap();
    }

    for i in 0..64 {
        assert_eq!(mmu.load_byte(BASE.strict_add(i)).unwrap(), pattern_byte(i));
    }
}

#[test]
fn slice_store_load_roundtrip_inside_one_page() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    const BYTES: usize = 128;

    let pattern_start = 1009;
    let data = pattern_array::<BYTES>(pattern_start);

    let store_start = BASE.strict_add(17u64);
    mmu.store(store_start, &data).unwrap();

    assert_eq!(read_array::<BYTES>(&mmu, store_start), data);
}

#[test]
fn slice_store_load_roundtrip_across_two_pages() {
    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);

    let start = BASE.strict_add(PAGE_SIZE_U64.strict_sub(5));
    const BYTES: usize = 13;

    let data = pattern_array::<BYTES>(0);

    mmu.store(start, &data).unwrap();
    assert_eq!(read_array::<BYTES>(&mmu, start), data);
}

#[test]
fn slice_store_across_boundary_requires_write_on_both_pages_and_does_not_partially_write() {
    let mmu = iommu_with_page_protections(&[MemProt::READ | MemProt::WRITE, MemProt::READ]);

    let start = BASE.strict_add(PAGE_SIZE_U64.strict_sub(1));
    let byte = 0xFA;
    mmu.store_byte(start, byte).unwrap();
    assert!(mmu.store(start, &[1, 2]).is_err());
    assert_eq!(read_array::<2>(&mmu, start), [byte, 0]);
}

#[test]
fn slice_load_across_boundary_requires_read_on_both_pages() {
    let mmu = iommu_with_page_protections(&[MemProt::READ | MemProt::WRITE, MemProt::WRITE]);

    let start = PAGE_SIZE_U64.strict_sub(1);
    assert!(mmu.load(start, &mut [0, 0]).is_err());
}

#[test]
fn scalar_loads_inside_one_page_are_little_endian() {
    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, pattern_byte);

    {
        let off16 = BASE.strict_add(11_u64);
        assert_eq!(
            mmu.load16_le(off16).unwrap(),
            u16::from_le_bytes(pattern_array(off16))
        );
    }

    {
        let off32 = BASE.strict_add(19_u64);
        assert_eq!(
            mmu.load32_le(off32).unwrap(),
            u32::from_le_bytes(pattern_array(off32))
        );
    }

    {
        let off64 = BASE.strict_add(29_u64);
        assert_eq!(
            mmu.load64_le(off64).unwrap(),
            u64::from_le_bytes(pattern_array(off64))
        );
    }
}

#[test]
fn scalar_stores_inside_one_page_are_little_endian() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    {
        let off16 = BASE.strict_add(11_u64);
        let value = 0xbeef_u16;
        mmu.store16_le(off16, value).unwrap();
        assert_eq!(read_array::<2>(&mmu, off16), value.to_le_bytes());
    }

    {
        let off32 = BASE.strict_add(19_u64);
        let value = 0xaabb_ccdd_u32;
        mmu.store32_le(off32, value).unwrap();
        assert_eq!(read_array::<4>(&mmu, off32), value.to_le_bytes());
    }

    {
        let off64 = BASE.strict_add(29_u64);
        let value = 0x0123_4567_89ab_cdef_u64;
        mmu.store64_le(off64, value).unwrap();
        assert_eq!(read_array::<8>(&mmu, off64), value.to_le_bytes());
    }
}

#[test]
fn scalar_loads_at_last_non_crossing_offsets_succeed() {
    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, pattern_byte);

    let base_end = BASE.strict_add(PAGE_SIZE_U64);

    {
        let off16 = base_end.strict_sub(2);
        assert_eq!(
            mmu.load16_le(off16).unwrap(),
            u16::from_le_bytes(pattern_array(off16))
        );
    }

    {
        let off32 = base_end.strict_sub(4);
        assert_eq!(
            mmu.load32_le(off32).unwrap(),
            u32::from_le_bytes(pattern_array(off32))
        );
    }

    {
        let off64 = base_end.strict_sub(8);
        assert_eq!(
            mmu.load64_le(off64).unwrap(),
            u64::from_le_bytes(pattern_array(off64))
        );
    }
}

#[test]
fn scalar_loads_crossing_page_boundary_read_expected_bytes() {
    let mmu = iommu_with_bytes(2, MemProt::READ | MemProt::WRITE, pattern_byte);

    let base_end = BASE.strict_add(PAGE_SIZE_U64);

    {
        let off16 = base_end.strict_sub(1);
        assert_eq!(
            mmu.load16_le(off16).unwrap(),
            u16::from_le_bytes(pattern_array(off16))
        );
    }

    for bytes_in_first_page in 1..4 {
        let start = base_end.strict_sub(bytes_in_first_page);
        assert_eq!(
            mmu.load32_le(start).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(start)),
            "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
        );
    }

    for bytes_in_first_page in 1..8 {
        let start = base_end.strict_sub(bytes_in_first_page);
        assert_eq!(
            mmu.load64_le(start).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(start)),
            "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
        );
    }
}

#[test]
fn scalar_stores_crossing_page_boundary_write_expected_bytes() {
    let mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);

    let base_end = BASE.strict_add(PAGE_SIZE_U64);

    {
        let off16 = base_end.strict_sub(1);
        let value16 = 0xbeefu16;
        mmu.store16_le(off16, value16).unwrap();
        assert_eq!(read_array(&mmu, off16), value16.to_le_bytes());
    }

    for bytes_in_first_page in 1..4_u8 {
        let start = base_end.strict_sub(bytes_in_first_page as u64);
        let value = 0xaabb_ccdd_u32 ^ (bytes_in_first_page as u32);

        mmu.store32_le(start, value).unwrap();
        assert_eq!(
            read_array::<4>(&mmu, start),
            value.to_le_bytes(),
            "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
        );
    }

    for bytes_in_first_page in 1..8_u8 {
        let start = base_end.strict_sub(bytes_in_first_page as u64);
        let value = 0x0123_4567_89ab_cdefu64 ^ bytes_in_first_page as u64;

        mmu.store64_le(start, value).unwrap();

        assert_eq!(
            read_array::<8>(&mmu, start),
            value.to_le_bytes(),
            "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
        );
    }
}

#[test]
fn crossing_scalar_load_requires_read_on_both_pages() {
    let mmu = iommu_with_page_protections(&[MemProt::READ, MemProt::WRITE]);

    assert!(mmu.load16_le(PAGE_SIZE_U64.strict_sub(1)).is_err());
}

#[test]
fn crossing_scalar_store_requires_write_on_both_pages() {
    let mmu = iommu_with_page_protections(&[MemProt::WRITE, MemProt::READ]);

    assert!(mmu.store16_le(PAGE_SIZE_U64.strict_sub(1), 0xbeef).is_err());
}

#[test]
fn crossing_scalar_access_to_unmapped_second_page_faults() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    let last = BASE.strict_add(PAGE_SIZE_U64.strict_sub(1));

    mmu.store_byte(last, 0xaa).unwrap();

    assert!(mmu.load16_le(last).is_err());
    assert!(mmu.store16_le(last, 0xbeef).is_err());

    assert_eq!(mmu.load_byte(last).unwrap(), 0xaa);
}

#[test]
fn store_byte_marks_executable_page_dirty() {
    let mmu = new_iommu(1, MemProt::WRITE | MemProt::EXECUTE);

    assert!(!is_dirty(&mmu, 0));
    mmu.store_byte(BASE, 0xaa).unwrap();
    assert!(is_dirty(&mmu, 0));
}

#[test]
fn store_slice_marks_executable_page_dirty() {
    let mmu = new_iommu(1, MemProt::WRITE | MemProt::EXECUTE);
    assert!(!is_dirty(&mmu, 0));
    mmu.store(BASE.strict_add(8), &[1, 2, 3, 4]).unwrap();
    assert!(is_dirty(&mmu, 0));
}

#[test]
fn store_slice_crossing_pages_marks_both_executable_pages_dirty() {
    let mmu = new_iommu(2, MemProt::WRITE | MemProt::EXECUTE);

    assert!(!is_dirty(&mmu, 0));
    assert!(!is_dirty(&mmu, 1));

    mmu.store(BASE.strict_add(PAGE_SIZE_U64.strict_sub(2)), &[1, 2, 3, 4])
        .unwrap();

    assert!(is_dirty(&mmu, 0));
    assert!(is_dirty(&mmu, 1));
}

#[test]
fn single_page_scalar_store_marks_executable_page_dirty() {
    let mmu = new_iommu(1, MemProt::WRITE | MemProt::EXECUTE);

    assert!(!is_dirty(&mmu, 0));

    mmu.store64_le(BASE, 0x0123_4567_89ab_cdef).unwrap();

    assert!(is_dirty(&mmu, 0));
}

#[test]
fn crossing_page_scalar_store_marks_executable_page_dirty() {
    let mmu = new_iommu(2, MemProt::WRITE | MemProt::EXECUTE);

    assert!(!is_dirty(&mmu, 0));
    assert!(!is_dirty(&mmu, 1));

    mmu.store64_le(
        BASE.strict_add(PAGE_SIZE_U64.strict_sub(1)),
        0x0123_4567_89ab_cdef,
    )
    .unwrap();

    assert!(is_dirty(&mmu, 0));
    assert!(is_dirty(&mmu, 1));
}

#[test]
fn store_to_non_executable_page_does_not_mark_dirty() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.store_byte(BASE, 0xaa).unwrap();
    mmu.store16_le(BASE.strict_add(8), 0xbeef).unwrap();
    mmu.store(BASE.strict_add(16), &[1, 2, 3]).unwrap();

    assert!(!is_dirty(&mmu, 0));
}

// BUG:
// `for_each_page_chunk` treats `vaddr_end` as an inclusive touched address.
// Ranges are normally `[start, end)`, so a load of exactly one page from the
// start of a one-page mapping should not require page 1 only page 0 to exist.
#[test]
fn bug_load_exactly_one_page_should_not_require_next_page() {
    let mmu = new_iommu(1, MemProt::READ);

    let mut out = Box::new_uninit_slice(PAGE_SIZE);
    assert!(mmu.load_into_uninit(BASE, &mut out).is_ok());
    let out = unsafe { out.assume_init() };
    assert!(out.iter().all(|&byte| byte == 0));
}

// BUG:
// Same exclusive-end bug as above, but through store.
#[test]
fn bug_store_exactly_one_page_should_not_require_next_page() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let data = vec![0x5a; PAGE_SIZE];
    assert!(mmu.store(BASE, &data).is_ok());
}

// BUG:
// Ending exactly at a page boundary should not check permissions on the next page,
// because zero bytes are accessed there.
#[test]
fn bug_load_ending_at_boundary_should_not_require_read_on_next_page() {
    let mmu = iommu_with_page_protections_and_bytes(&[MemProt::READ, MemProt::WRITE], pattern_byte);

    let out = read_array::<PAGE_SIZE>(&mmu, BASE);
    let expected = pattern_array(BASE);
    assert_eq!(out, expected);
}

// BUG:
// Cross-page scalar stores write executable pages but never call `set_insn_dirty`
// on either touched page.
#[test]
fn bug_cross_page_scalar_store_should_mark_touched_executable_pages_dirty() {
    let mmu = new_iommu(2, MemProt::WRITE | MemProt::EXECUTE);

    assert!(!is_dirty(&mmu, 0));
    assert!(!is_dirty(&mmu, 1));

    let addr = BASE.strict_add(PAGE_SIZE_U64.strict_sub(1));

    mmu.store16_le(addr, 0xbeef).unwrap();

    assert!(is_dirty(&mmu, 0));
    assert!(is_dirty(&mmu, 1));
}

fn sparse_fixture_with_hole() -> IoMMU {
    let mut mmu = IoMMU::new(CpuFabric::new());

    mmu.map(page_addr(0), PAGE_SIZE_U64, MemProt::READ | MemProt::WRITE)
        .unwrap();

    // Intentionally skip page 1.

    mmu.map(page_addr(2), PAGE_SIZE_U64, MemProt::READ | MemProt::WRITE)
        .unwrap();

    mmu
}

#[test]
fn unmap_memory_unmaps_one_page() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let page = PAGE_SIZE_U64;

    mmu.store_byte(BASE, 0xaa).unwrap();
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);

    mmu.unmap(BASE, page).unwrap();

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 0xbb).is_err());

    assert!(mmu.load(BASE, &mut [0]).is_err());
    assert!(mmu.store(BASE, &[0xcc]).is_err());

    assert!(mmu.load16_le(BASE).is_err());
    assert!(mmu.store16_le(BASE, 0xbeef).is_err());
}

#[test]
fn unmap_memory_unmaps_only_requested_page_range() {
    let mut mmu = new_iommu(3, MemProt::READ | MemProt::WRITE);
    let page = PAGE_SIZE_U64;

    mmu.store_byte(page_addr(0), 0x10).unwrap();
    mmu.store_byte(page_addr(1), 0x20).unwrap();
    mmu.store_byte(page_addr(2), 0x30).unwrap();

    mmu.unmap(page_addr(1), page).unwrap();

    assert_eq!(mmu.load_byte(page_addr(0)).unwrap(), 0x10);
    assert!(mmu.load_byte(page_addr(1)).is_err());
    assert_eq!(mmu.load_byte(page_addr(2)).unwrap(), 0x30);

    assert!(mmu.store_byte(page_addr(0), 0x11).is_ok());
    assert!(mmu.store_byte(page_addr(1), 0x21).is_err());
    assert!(mmu.store_byte(page_addr(2), 0x31).is_ok());
}

#[test]
fn unmap_memory_can_unmap_multiple_pages() {
    let mut mmu = new_iommu(4, MemProt::READ | MemProt::WRITE);

    mmu.unmap(page_addr(1), PAGE_SIZE_U64.strict_mul(2))
        .unwrap();

    assert!(mmu.load_byte(page_addr(0)).is_ok());
    assert!(mmu.load_byte(page_addr(1)).is_err());
    assert!(mmu.load_byte(page_addr(2)).is_err());
    assert!(mmu.load_byte(page_addr(3)).is_ok());
}

#[test]
fn unmap_memory_is_idempotent_for_existing_page_entries() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let page = PAGE_SIZE_U64;

    mmu.unmap(BASE, page).unwrap();
    mmu.unmap(BASE, page).unwrap();

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 0xaa).is_err());
}

#[test]
fn unmap_memory_rejects_unaligned_start_and_leaves_mapping_unchanged() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let page = PAGE_SIZE_U64;

    mmu.store_byte(BASE, 0xaa).unwrap();

    assert!(mmu.unmap(1, page).is_err());

    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
    assert!(mmu.store_byte(BASE, 0xbb).is_ok());
}

#[test]
fn unmap_memory_rejects_unaligned_size_and_leaves_mapping_unchanged() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    let page = PAGE_SIZE_U64;

    mmu.store_byte(BASE, 0xaa).unwrap();

    assert!(mmu.unmap(BASE, page - 1).is_err());

    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
    assert!(mmu.store_byte(BASE, 0xbb).is_ok());
}

#[test]
fn unmap_memory_already_unmapped_passes_and_is_noop() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(mmu.unmap(page_addr(1), PAGE_SIZE_U64).is_ok());

    assert!(mmu.load_byte(BASE).is_ok());
    assert!(mmu.store_byte(BASE, 0xaa).is_ok());
}

#[test]
fn unmap_memory_rejects_address_overflow() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(mmu.unmap(u64::MAX, 1).is_err());

    assert!(mmu.load_byte(BASE).is_ok());
}

#[test]
fn unmap_memory_allows_zero_sized_noop_at_start() {
    let mut mmu = IoMMU::new(CpuFabric::new());

    assert!(mmu.unmap(BASE, 0).is_ok());
}

#[test]
fn unmap_memory_then_remap_restores_access() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.store_byte(BASE, 0xaa).unwrap();
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);

    mmu.unmap(BASE, PAGE_SIZE_U64).unwrap();
    assert!(mmu.store_byte(BASE, 0xff).is_err());
    assert!(mmu.load_byte(BASE).is_err());

    mmu.map(BASE, PAGE_SIZE_U64, MemProt::READ | MemProt::WRITE)
        .unwrap();

    mmu.store_byte(BASE, 0xbb).unwrap();
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xbb);
}

#[test]
fn mem_protect_can_make_page_read_only() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.store_byte(BASE, 0xaa).unwrap();

    mmu.protect(BASE, PAGE_SIZE_U64, MemProt::READ).unwrap();

    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
    assert!(mmu.store_byte(BASE, 0xbb).is_err());
    assert!(mmu.store(BASE, &[0xcc]).is_err());
    assert!(mmu.store16_le(BASE, 0xbeef).is_err());
}

#[test]
fn mem_protect_can_make_page_write_only() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.store_byte(BASE, 0xaa).unwrap();
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);

    mmu.protect(BASE, PAGE_SIZE_U64, MemProt::WRITE).unwrap();

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.load16_le(BASE).is_err());

    assert!(mmu.store_byte(BASE, 0xbb).is_ok());
    assert!(mmu.store(BASE, &[0xcc]).is_ok());
    assert!(mmu.store16_le(BASE, 0xbeef).is_ok());
}

#[test]
fn mem_protect_can_make_page_execute_only_for_data_accesses() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.protect(BASE, PAGE_SIZE_U64, MemProt::EXECUTE).unwrap();

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 0xaa).is_err());

    let mut out = [MaybeUninit::<u8>::uninit()];
    assert!(mmu.load_into_uninit(BASE, &mut out).is_err());
    assert!(mmu.store(BASE, &[0xaa]).is_err());
}

#[test]
fn mem_protect_can_restore_read_write_after_read_only() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.protect(BASE, PAGE_SIZE_U64, MemProt::READ).unwrap();

    assert!(mmu.store_byte(BASE, 0xaa).is_err());

    mmu.protect(BASE, PAGE_SIZE_U64, MemProt::READ | MemProt::WRITE)
        .unwrap();

    mmu.store_byte(BASE, 0xbb).unwrap();
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xbb);
}

#[test]
fn mem_protect_only_changes_requested_pages() {
    let mut mmu = new_iommu(2, MemProt::READ | MemProt::WRITE);

    mmu.protect(page_addr(1), PAGE_SIZE_U64, MemProt::READ)
        .unwrap();

    assert!(mmu.store_byte(page_addr(0), 0xaa).is_ok());
    assert!(mmu.store_byte(page_addr(1), 0xbb).is_err());

    assert!(mmu.load_byte(page_addr(0)).is_ok());
    assert!(mmu.load_byte(page_addr(1)).is_ok());
}

#[test]
fn mem_protect_can_change_multiple_pages() {
    let mut mmu = new_iommu(3, MemProt::READ | MemProt::WRITE);

    mmu.protect(page_addr(0), PAGE_SIZE_U64.strict_mul(2), MemProt::READ)
        .unwrap();

    assert!(mmu.store_byte(page_addr(0), 0xaa).is_err());
    assert!(mmu.store_byte(page_addr(1), 0xbb).is_err());
    assert!(mmu.store_byte(page_addr(2), 0xcc).is_ok());

    assert!(mmu.load_byte(page_addr(0)).is_ok());
    assert!(mmu.load_byte(page_addr(1)).is_ok());
    assert!(mmu.load_byte(page_addr(2)).is_ok());
}

#[test]
fn mem_protect_rejects_unmapped_page() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    mmu.unmap(BASE, PAGE_SIZE_U64).unwrap();

    assert!(mmu.protect(BASE, PAGE_SIZE_U64, MemProt::READ).is_err());

    assert!(mmu.load_byte(BASE).is_err());
    assert!(mmu.store_byte(BASE, 0xaa).is_err());
}

#[test]
fn mem_protect_rejects_sparse_range_and_does_not_partially_update() {
    let mut mmu = sparse_fixture_with_hole();

    assert!(
        mmu.protect(page_addr(0), PAGE_SIZE_U64.strict_mul(3), MemProt::READ)
            .is_err()
    );

    // Page 0 and page 2 must still be writable. This catches accidental
    // partial updates if the implementation protects pages as it checks them.
    assert!(mmu.store_byte(page_addr(0), 0xaa).is_ok());
    assert!(mmu.store_byte(page_addr(2), 0xbb).is_ok());

    assert_eq!(mmu.load_byte(page_addr(0)).unwrap(), 0xaa);
    assert_eq!(mmu.load_byte(page_addr(2)).unwrap(), 0xbb);
}

#[test]
fn mem_protect_rejects_unaligned_start_and_leaves_mapping_unchanged() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(mmu.protect(1, PAGE_SIZE_U64, MemProt::READ).is_err());

    assert!(mmu.store_byte(BASE, 0xaa).is_ok());
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
}

#[test]
fn mem_protect_rejects_unaligned_size_and_leaves_mapping_unchanged() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(
        mmu.protect(BASE, PAGE_SIZE_U64.strict_sub(1), MemProt::READ)
            .is_err()
    );

    assert!(mmu.store_byte(BASE, 0xaa).is_ok());
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
}

#[test]
fn mem_protect_rejects_range_past_page_table() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(
        mmu.protect(page_addr(1), PAGE_SIZE_U64, MemProt::READ)
            .is_err()
    );

    assert!(mmu.store_byte(BASE, 0xaa).is_ok());
    assert_eq!(mmu.load_byte(BASE).unwrap(), 0xaa);
}

#[test]
fn mem_protect_rejects_address_overflow() {
    let mut mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(mmu.protect(u64::MAX, 1, MemProt::READ).is_err());
    assert!(mmu.store_byte(BASE, 0xaa).is_ok());
}

#[test]
fn mem_protect_allows_zero_sized_noop_at_start() {
    let mut mmu = IoMMU::new(CpuFabric::new());
    assert!(mmu.protect(BASE, 0, MemProt::READ).is_ok());
}

#[test]
fn unaligned_scalar_loads_inside_page_succeed() {
    let mmu = iommu_with_bytes(1, MemProt::READ | MemProt::WRITE, pattern_byte);

    let vaddr = BASE.strict_add(1);

    assert_eq!(
        mmu.load16_le(vaddr).unwrap(),
        u16::from_le_bytes(pattern_array::<2>(1))
    );

    assert_eq!(
        mmu.load32_le(vaddr).unwrap(),
        u32::from_le_bytes(pattern_array::<4>(1))
    );

    assert_eq!(
        mmu.load64_le(vaddr).unwrap(),
        u64::from_le_bytes(pattern_array::<8>(1))
    );
}

#[test]
fn unaligned_scalar_stores_inside_page_succeed() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    let addr16 = BASE.strict_add(1);
    mmu.store16_le(addr16, 0xbeef).unwrap();
    assert_eq!(read_array::<2>(&mmu, addr16), 0xbeefu16.to_le_bytes());

    let addr32 = BASE.strict_add(3);
    mmu.store32_le(addr32, 0xaabb_ccdd).unwrap();
    assert_eq!(read_array::<4>(&mmu, addr32), 0xaabb_ccddu32.to_le_bytes());

    let addr64 = BASE.strict_add(5);
    mmu.store64_le(addr64, 0x0123_4567_89ab_cdef).unwrap();
    assert_eq!(
        read_array::<8>(&mmu, addr64),
        0x0123_4567_89ab_cdefu64.to_le_bytes()
    );
}

#[test]
fn failed_crossing_scalar_store_does_not_write_first_page_when_second_page_lacks_write() {
    let mmu = iommu_with_page_protections_and_bytes(
        &[MemProt::READ | MemProt::WRITE, MemProt::READ],
        pattern_byte,
    );

    let start = BASE.strict_add(PAGE_SIZE_U64);
    let before = mmu.load_byte(start).unwrap();

    assert!(mmu.store16_le(start, 0xbeef).is_err());

    assert_eq!(mmu.load_byte(start).unwrap(), before);
}

#[test]
fn failed_crossing_scalar_store_does_not_write_second_page_when_first_page_lacks_write() {
    let mmu = iommu_with_page_protections_and_bytes(
        &[MemProt::READ, MemProt::READ | MemProt::WRITE],
        pattern_byte,
    );

    let start = PAGE_SIZE_U64.strict_sub(1);
    let second_page_before = mmu.load_byte(page_addr(1)).unwrap();

    assert!(mmu.store16_le(start, 0xbeef).is_err());

    assert_eq!(mmu.load_byte(page_addr(1)).unwrap(), second_page_before);
}

#[test]
fn scalar_accesses_near_u64_max_fault_instead_of_wrapping() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);

    assert!(mmu.load16_le(u64::MAX).is_err());
    assert!(mmu.load32_le(u64::MAX - 1).is_err());
    assert!(mmu.load64_le(u64::MAX - 3).is_err());

    assert!(mmu.store16_le(u64::MAX, 0xbeef).is_err());
    assert!(mmu.store32_le(u64::MAX - 1, 0xaabb_ccdd).is_err());
    assert!(mmu.store64_le(u64::MAX - 3, 0x0123_4567_89ab_cdef).is_err());
}

#[test]
fn fetch_aarch64_faults_on_non_execute_page() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE);
    assert!(mmu.fetch_aarch64(BASE).is_err());
}

#[test]
fn fetch_aarch64_faults_on_unaligned_address() {
    let mmu = new_iommu(1, MemProt::EXECUTE);
    assert!(mmu.fetch_aarch64(BASE.strict_add(1)).is_err());
    assert!(mmu.fetch_aarch64(BASE.strict_add(2)).is_err());
    assert!(mmu.fetch_aarch64(BASE.strict_add(3)).is_err());
}

#[test]
fn fetch_aarch64_returns_correct_little_endian_word() {
    let mmu = new_iommu(1, MemProt::READ | MemProt::WRITE | MemProt::EXECUTE);

    let addr = BASE.strict_add(8u64);
    // NOP
    let insn_bytes = [0x1f, 0x20, 0x03, 0xd5];
    mmu.store(addr, &insn_bytes).unwrap();

    assert_eq!(
        mmu.fetch_aarch64(addr).unwrap(),
        u32::from_le_bytes(insn_bytes)
    );
}

#[test]
fn map_and_access_last_possible_page() {
    let base = u64::MAX.strict_sub(PAGE_SIZE_U64.strict_sub(1));

    let mut mmu = IoMMU::new(CpuFabric::new());
    mmu.map(
        base,
        PAGE_SIZE_U64,
        MemProt::READ | MemProt::WRITE | MemProt::EXECUTE,
    )
    .unwrap();

    mmu.store_byte(base, 0xaa).unwrap();
    assert_eq!(mmu.load_byte(base).unwrap(), 0xaa);

    mmu.store_byte(u64::MAX, 0xbb).unwrap();
    assert_eq!(mmu.load_byte(u64::MAX).unwrap(), 0xbb);

    assert!(mmu.fetch_aarch64(base).is_ok());

    type Word = u32;

    let last_word_addr = const {
        let size = usize_to_u64(size_of::<Word>()).unwrap();
        u64::MAX.strict_sub(size.strict_sub(1))
    };

    let to_store: Word = 0xbaad_beef;
    mmu.store32_le(last_word_addr, to_store).unwrap();
    let found_word: Word = mmu.fetch_aarch64(last_word_addr).unwrap();
    assert_eq!(found_word, to_store)
}
