//! Page-based virtual memory abstraction backed by host memory.
//!
//! This module implements a small MMU-like layer that maps Armv9 guest virtual
//! pages onto host memory and enforces per-page access permissions.
//!
//! Core behavior:
//! - Memory is mapped in page-sized chunks.
//! - Each page may be readable, writable, and/or executable.
//! - Unmapped access or permission violations return [`MemoryFault`].
//! - Byte-range reads and writes may span multiple pages.
//! - Scalar load/store helpers are provided for 8/16/32/64-bit little-endian
//!   accesses.
//! - Scalar accesses may cross page boundaries when both pages are mapped with
//!   the required permissions.
//!
//! Concurrency model:
//! - Backing memory may be accessed concurrently through multiple threads.
//! - Concurrent loads and stores through the MMU are allowed.
//! - Public safe APIs must not produce undefined behavior, even when accesses
//!   race.
//! - Atomicity follows the Armv9 memory model for the active [`CpuFabric`].
//! - Naturally aligned scalar accesses use the single-copy atomicity guarantees
//!   provided by that fabric.
//! - Operations outside the fabric's single-copy atomic width, or operations
//!   split across pages, may be observed as multiple smaller operations.
//!
//! Safety model:
//! - All public safe functions are required to be UB-free.
//! - Unsafe functions may rely on their documented caller obligations.
//! - Mapping requires the caller to provide valid, page-aligned backing memory
//!   for the lifetime of the MMU mapping.
//! - Once memory is mapped into an MMU, the backing pointer must not be accessed
//!   directly while the mapping is alive, except for use as backing memory for an
//!   MMU mapping under the same aliasing/concurrency rules and the same
//!   [`CpuFabric`] memory model.
//!
//! Typical usage:
//! 1. Construct an [`IoMMU`].
//! 2. Map one or more host memory regions with [`IoMMU::map_memory`].
//! 3. Access byte ranges through [`IoMMU::load`] / [`IoMMU::store`].
//! 4. Access scalars through `load_byte/load16/load32/load64` and
//!    `store_byte/store16/store32/store64`.

use std::collections::HashSet;
use std::hint::cold_path;
use std::mem::MaybeUninit;
use std::num::NonZero;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};
use parking_lot::Mutex;
use crate::cpu_fabric::CpuFabric;


mod memops;

// FIXME feature(const_convert)
macro_rules! make_checked_usize_cast {
    ($from: ident => $to: ident) => {
        pastey::paste! {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "this function ensures no truncation happens"
            )]
            #[inline(always)]
            const fn [<$from _to_ $to>](int: $from) -> Option<$to> {
                match $to::BITS >= $from::BITS {
                    true => Some(int as $to),
                    false => {
                        // this would be a widening cast
                        let max: $from = $to::MAX as $from;
                        if int > max {
                            cold_path();
                            return None
                        }
                        Some(int as $to)
                    },
                }
            }
        }

    };
}

make_checked_usize_cast! { u64 => usize }
make_checked_usize_cast! { usize => u64 }

#[inline(always)]
const fn u64_add_usize(x: u64, y: usize) -> Option<u64> {
    let Some(y) = usize_to_u64(y) else {
        cold_path();
        return None
    };
    x.checked_add(y)
}

pub const PAGE_SIZE_U64: u64 = 4096;
pub const PAGE_SIZE: usize = u64_to_usize(PAGE_SIZE_U64).unwrap();

#[allow(
    clippy::cast_possible_truncation,
    reason = "bits must be less than 24 bits; which always fits in a u8"
)]
pub const PAGE_SHIFT: u8 = {
    assert!(PAGE_SIZE.is_power_of_two());
    let bits = PAGE_SIZE.ilog2();
    assert!(bits < 24, "page size too big");
    bits as u8
};


bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub(crate) struct PageFlags: u8 {
        const READ       = 0b0001;
        const WRITE      = 0b0010;
        const EXECUTE    = 0b0100;
        const INSN_DIRTY = 0b1000;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MemoryProtections: u8 {
        const READ = PageFlags::READ.bits();
        const WRITE = PageFlags::WRITE.bits();
        const EXECUTE = PageFlags::EXECUTE.bits();
    }
}


/// Fault returned when a memory access is invalid.
///
/// This is returned when an access:
/// - targets an unmapped page,
/// - violates page permissions,
/// - overflows the virtual address range,
/// - fails an address-alignment check required by a specific operation,
/// - crosses into an unmapped or insufficiently-permitted page,
/// - or otherwise fails MMU validation.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid memory access")]
pub struct MemoryFault(());

impl MemoryFault {
    #[inline(always)]
    #[cold]
    pub const fn fault() -> Self {
        cold_path();
        Self(())
    }
}

macro_rules! ensure {
    ($($expr: expr),+ $(,)?) => {
        if !($({ $expr })&+) {
            return Err(MemoryFault::fault())
        }
    };
}


#[inline]
fn div_rem_page_size(vaddr: u64) -> (u64, usize) {
    (
        vaddr >> PAGE_SHIFT,
        {
            let remainder = vaddr & const { PAGE_SIZE_U64.strict_sub(1) };
            // Safety: PAGE_SIZE fits in usize, and we are taking the remainder by PAGE_SIZE
            //         therefore the remainder **MUST** fit in a usize
            unsafe { u64_to_usize(remainder).unwrap_unchecked() }
        }
    )
}

#[inline]
fn div_page_size_checked(vaddr: u64) -> Result<u64, MemoryFault> {
    let (page, offset) = div_rem_page_size(vaddr);
    ensure!(offset == 0);
    Ok(page)
}



impl MemoryProtections {
    // Self is a superset of PageFlags
    fn into_page_flags(self) -> PageFlags {
        PageFlags::from_bits_retain((self & Self::all()).bits())
    }
}


/// Per-page invariants:
///
/// Mapping state:
///
/// - `ptr == None` means the page is unmapped.
/// - `ptr == Some(_)` means the page is mapped.
/// - If any guest memory-protection bit is set in `page_flags`
///   (`READ`, `WRITE`, or `EXECUTE`), then `ptr` must be `Some`.
///
/// Mapping identity:
///
/// Once a page is mapped, its backing pointer is stable for the lifetime of that
/// mapping. A mapped page's pointer must not be replaced directly.
///
/// To change the backing pointer, the page must first be unmapped through
/// [`Page::unmap`], and only then mapped again with the new pointer.
///
/// Valid transition:
///
/// ```text
/// Some(old_ptr) -> None -> Some(new_ptr)
/// ```
///
/// Invalid transition:
///
/// ```text
/// Some(old_ptr) -> Some(new_ptr)
/// ```
///
/// Exclusive access requirement:
///
/// Changing either the page's memory protections or its mapping state requires
/// exclusive access to the [`Page`].
///
/// In practice, this means:
///
/// - changing guest permission bits (`READ`, `WRITE`, `EXECUTE`) requires
///   `&mut Page`;
/// - mapping a page requires creating/replacing the [`Page`] through exclusive
///   access to the page table entry;
/// - unmapping a page requires `&mut Page`;
/// - replacing a mapped page's backing pointer requires first calling
///   [`Page::unmap`] with `&mut Page`.
///
/// Shared access to a page may perform guest loads/stores and may update
/// non-permission bookkeeping bits such as `INSN_DIRTY`, but it must not change
/// the page's mapping state or guest memory protections.
pub(crate) struct Page {
    /// Host backing pointer for this guest page.
    ///
    /// Mapping invariants:
    /// - `None` means this page is unmapped.
    /// - `Some(ptr)` means this page is mapped.
    /// - If any guest memory-protection bit is set in `page_flags`
    ///   (`READ`, `WRITE`, or `EXECUTE`), this field must be `Some`.
    /// - Once mapped, the backing pointer is stable for the lifetime of that
    ///   mapping.
    /// - A mapped pointer must not be replaced directly. To change the backing
    ///   pointer, the page must first transition through `None`:
    ///
    ///   ```text
    ///   Some(old_ptr) -> None -> Some(new_ptr)
    ///   ```
    ///
    /// - The invalid transition is:
    ///
    ///   ```text
    ///   Some(old_ptr) -> Some(new_ptr)
    ///   ```
    ///
    /// Access invariants:
    /// - Direct access to the pointed-to memory must go through the operations in
    ///   `memops`.
    /// - The pointer is stored as `NonNull<AtomicU8>` because `memops` accepts
    ///   pointers to `AtomicU8` and performs the actual byte, scalar,
    ///   aligned, and unaligned memory operations.
    pub(crate) ptr: Option<NonNull<AtomicU8>>,

    // TODO seperate page_dirty from memory protection flags
    /// Atomic per-page state.
    ///
    /// Stores the raw bits of `PageFlags`:
    /// - `READ`
    /// - `WRITE`
    /// - `EXECUTE`
    /// - `INSN_DIRTY`
    ///
    /// Permission invariants:
    /// - `READ`, `WRITE`, and `EXECUTE` are guest-visible memory-protection bits.
    /// - If any of those permission bits are set, `ptr` must be `Some`.
    /// - Changing guest permissions requires exclusive access to the page.
    /// - Shared access may read these flags for permission checks.
    ///
    /// `INSN_DIRTY` invariants:
    /// - `INSN_DIRTY` is not a guest permission bit.
    /// - `INSN_DIRTY` is bookkeeping for executable pages whose backing bytes may
    ///   have changed since instruction cache / decoded instruction state was
    ///   last synchronized.
    /// - A successful store to a page that was executable at the time of the
    ///   permission check must set `INSN_DIRTY`.
    /// - If `EXECUTE` permission is removed from a page, `INSN_DIRTY` must be
    ///   preserved or set, because previously executable contents may have cached
    ///   instruction state that needs invalidation.
    /// - Updating ordinary guest permissions must preserve an existing
    ///   `INSN_DIRTY` bit.
    /// - Clearing `INSN_DIRTY` is only done by the instruction-cache drain path,
    ///   via `take_insn_dirty`.
    /// - If an executable dirty page is unmapped, its backing pointer must be
    ///   recorded in `pending_dirty_pages` before the page transitions to
    ///   unmapped, so the invalidation range is not lost.
    ///
    /// Concurrency invariants:
    /// - Permission changes and mapping-state changes require exclusive access.
    /// - `INSN_DIRTY` may be updated through shared access after successful
    ///   stores.
    /// - Setting `INSN_DIRTY` uses release ordering so the guest writes that made
    ///   the page dirty happen-before an acquire/acqrel dirty drain observes the
    ///   bit.
    pub(crate) page_flags: AtomicU8,
}

impl Page {
    #[allow(clippy::declare_interior_mutable_const)]
    pub(crate) const UNMAPPED: Self = Self {
        ptr: None,
        page_flags: AtomicU8::new(0),
    };

    pub(crate) fn mapped(
        ptr: NonNull<AtomicU8>,
        memory_protections: MemoryProtections,
    ) -> Self {
        let page_flags = memory_protections.into_page_flags().bits();
        Self {
            ptr: Some(ptr),
            page_flags: AtomicU8::new(page_flags),
        }
    }

    pub(crate) fn unmap(&mut self) {
        *self = Self::UNMAPPED
    }

    pub(crate) fn is_mapped(&self) -> bool {
        self.ptr.is_some()
    }

    /// # Safety
    /// `self` must be mapped
    pub(crate) unsafe fn mem_protect(&mut self, memory_protections: MemoryProtections) {
        debug_assert!(self.is_mapped());

        let flags = self.page_flags.get_mut();
        let mut new_flags = memory_protections.into_page_flags();
        let old_flags = PageFlags::from_bits_retain(*flags);

        if old_flags.contains(PageFlags::INSN_DIRTY) {
            new_flags |= PageFlags::INSN_DIRTY
        }

        // if we had the execute permision; and then suddenly its gone
        // that means we can no longer execute the instructions in this page
        // therefore the page now contains dirty instructions
        if old_flags.contains(PageFlags::EXECUTE) && !new_flags.contains(PageFlags::EXECUTE) {
            new_flags |= PageFlags::INSN_DIRTY
        }

        *flags = new_flags.bits();
    }

    #[inline(always)]
    fn load_flags(&self) -> PageFlags {
        PageFlags::from_bits_retain(self.page_flags.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn load_flags_mut(&mut self) -> PageFlags {
        PageFlags::from_bits_retain(*self.page_flags.get_mut())
    }

    #[inline(always)]
    fn has_access(&self, flags: PageFlags) -> bool {
        if cfg!(debug_assertions) && self.ptr.is_none() {
            assert!(self.load_flags().is_empty());
        }
        self.load_flags().contains(flags)
    }

    #[inline(always)]
    pub(crate) unsafe fn get_data_ptr_unchecked(&self) -> NonNull<AtomicU8> {
        debug_assert!(self.ptr.is_some());

        unsafe { self.ptr.unwrap_unchecked() }
    }

    #[inline(always)]
    fn get_data_ptr(&self, flags: PageFlags) -> Result<NonNull<AtomicU8>, MemoryFault> {
        ensure!(self.has_access(flags));
        // Safety: if self.has_access returns true, its always ok to call get_data_ptr_unchecked
        Ok(unsafe { self.get_data_ptr_unchecked() })
    }


    /// Marks this page as requiring instruction-cache invalidation.
    ///
    /// This is called after every successful store. It only changes the page state
    /// when the page is currently executable: writes to executable memory may change
    /// the instruction stream, so any cached decoded/translated instructions for
    /// this page must be invalidated before execution can safely use them again.
    ///
    /// This function must be called after the write permission check succeeds and
    /// after the store operation succeeds.
    #[inline(always)]
    fn set_insn_dirty(&self, initial_flags: PageFlags) {
        if initial_flags.contains(PageFlags::EXECUTE) {
            cold_path();
            if !initial_flags.contains(PageFlags::INSN_DIRTY) {
                cold_path();
                // Use Release so that the successful guest write happens-before any thread that
                // observes and consumes INSN_DIRTY with Acquire ordering. Once this bit becomes
                // visible, the instruction-cache invalidation path must also be able to observe
                // the memory contents that made the page dirty.
                self.page_flags.fetch_or(PageFlags::INSN_DIRTY.bits(), Ordering::Release);
            }
        }
    }


    pub(crate) fn take_insn_dirty(&self) -> bool {
        // Use AcqRel because this operation both observes and clears INSN_DIRTY.
        //
        // Acquire pairs with the Release in set_insn_dirty, ensuring that if we observe
        // the dirty bit, the writes that dirtied the executable page are visible before
        // we invalidate cached instructions for it.
        //
        // Release covers the clearing side of the RMW: once we clear INSN_DIRTY, later
        // observers must not treat the old dirty state as still pending through this
        // operation.

        let flag_bit = PageFlags::INSN_DIRTY.bits();
        let mask = !flag_bit;
        let old_flags = self.page_flags.fetch_and(mask, Ordering::AcqRel);
        (old_flags & flag_bit) != 0
    }


    /// # Safety
    ///
    /// the whole store operation must fit in the page
    #[inline(always)]
    pub(crate) unsafe fn load(&self, offset: usize, mem: &mut [MaybeUninit<u8>]) -> Result<(), MemoryFault> {
        unsafe {
            let page_ptr = self.get_data_ptr(PageFlags::READ)?;

            let data_ptr = page_ptr.add(offset).as_ptr();

            let len = mem.len();
            let mem_ptr = mem.as_mut_ptr().cast::<u8>();

            for i in 0..len {
                let value = memops::load_byte(data_ptr.add(i));
                std::ptr::write(mem_ptr.add(i), value)
            }

            Ok(())
        }
    }

    /// # Safety
    ///
    /// the whole store operation must fit in the page
    #[inline(always)]
    pub(crate) unsafe fn store(&self, offset: usize, mem: &[u8]) -> Result<(), MemoryFault> {
        let flags = self.load_flags();
        ensure!(flags.contains(PageFlags::WRITE));

        unsafe {
            let page_ptr = self.get_data_ptr_unchecked();

            let write_ptr = page_ptr.add(offset).as_ptr();

            let len = mem.len();
            let mem_ptr = mem.as_ptr();

            for i in 0..len {
                let value = std::ptr::read(mem_ptr.add(i));
                memops::store_byte(write_ptr.add(i), value)
            }

            self.set_insn_dirty(flags);

            Ok(())
        }
    }
}

macro_rules! impl_load_ops {
    {
        $(bits: $bits: tt,
        ty: $ty: ty,
        load_function: $load_op_name: ident,
        store_function: $store_op_name: ident,
        load: $load_name: ident,
        store: $store_name: ident
        ),+
        $(,)?
    } => {
        impl Page {$(
            /// # Safety
            ///
            #[doc = concat!("`offset` must be <= PAGE_SIZE - size_of::<", stringify!($ty), ">()")]
            #[inline(always)]
            pub(crate) unsafe fn $load_name(&self, offset: usize) -> Result<$ty, MemoryFault> {
                let ptr = self.get_data_ptr(PageFlags::READ)?;
                let value = unsafe { memops::$load_op_name(ptr.as_ptr().add(offset)) };
                Ok(value)
            }

            /// # Safety
            ///
            #[doc = concat!("`offset` must be <= A::PAGE_SIZE - size_of<", stringify!($ty), ">()")]
            #[inline(always)]
            pub(crate) unsafe fn $store_name(&self, offset: usize, value: $ty) -> Result<(), MemoryFault> {
                let flags = self.load_flags();
                ensure!(flags.contains(PageFlags::WRITE));
                unsafe {
                    let ptr = self.get_data_ptr_unchecked();
                    memops::$store_op_name(ptr.as_ptr().add(offset), value);
                    self.set_insn_dirty(flags);
                }
                Ok(())
            }
        )+}
    };

    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            impl_load_ops! {$(
                bits: $bits,
                ty: [<u $bits>],
                load_function: [<load $bits _le>],
                store_function: [<store $bits _le>],
                load: [<load $bits>],
                store: [<store $bits>]
            ),+}
        }
    };
}

impl_load_ops! { 64, 32, 16 }

impl_load_ops! {
    bits: 8,
    ty: u8,
    load_function: load_byte,
    store_function: store_byte,
    load: load_byte,
    store: store_byte
}


#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub(crate) struct HostPointer(pub(crate) NonNull<AtomicU8>);

impl HostPointer {
    pub(crate) const fn new(ptr: NonNull<AtomicU8>) -> Self {
        Self(ptr)
    }
}

unsafe impl Send for HostPointer {}
unsafe impl Sync for HostPointer {}


/// Page-mapped virtual memory view over host-backed storage.
///
/// `IoMMU` maps Armv9 virtual addresses onto page-aligned host memory.
/// Access permissions are checked per page, and invalid access returns
/// [`MemoryFault`].
///
/// The implementation permits concurrent access and models the armv9 memory model.
pub struct IoMMU {
    pages: Vec<Page>,
    pending_dirty_pages: Mutex<HashSet<HostPointer>>,
    // FIXME whenever a write is in
    //       range of an active monitor,
    //       invalidate it and lock it,
    //       until the store is complete
    fabric: CpuFabric,
}

// Safety: all interior mutability is guarded explicitly with mutable references;
//         and all access is atomic/tearing and doesn't lead to UB
unsafe impl Send for IoMMU {}
unsafe impl Sync for IoMMU {}

impl IoMMU {
    /// Creates an empty MMU with no mapped pages.
    ///
    /// All accesses fault until memory is mapped with [`IoMMU::map_memory`].
    pub fn new(fabric: CpuFabric) -> Self {
        Self {
            pages: vec![],
            pending_dirty_pages: Mutex::new(HashSet::new()),
            fabric
        }
    }

    pub fn get_fabric(&self) -> &CpuFabric {
        &self.fabric
    }

    fn unmap_page(page: &mut Page, pending_dirty_pages: &mut HashSet<HostPointer>) {
        if page.load_flags_mut().contains(PageFlags::INSN_DIRTY) {
            let page = unsafe { page.get_data_ptr_unchecked() };
            pending_dirty_pages.insert(HostPointer::new(page));
        }
        page.unmap()
    }


    /// Maps a host memory region into the MMU page table.
    ///
    /// `base` is the starting virtual address, `ptr` is the backing host pointer,
    /// and `size` is the mapping size in bytes.
    ///
    /// Requirements:
    /// - `base` must be page-aligned,
    /// - `size` must be page-aligned,
    /// - `ptr` must be aligned to the page size,
    /// - `base + size` must not overflow.
    ///
    /// Permissions are applied to every mapped page in the region.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr .. ptr + size` is valid for the lifetime of this MMU mapping,
    /// - the pointed-to memory is initialized,
    /// - the pointed-to memory is valid for both reads and writes at the host-memory
    ///   level, regardless of the guest permissions applied by `protections`,
    /// - the backing memory is not accessed directly while this mapping is alive,
    ///   except by other MMUs that use the same `CpuFabric`
    /// - the backing memory remains page-aligned and is not deallocated, reallocated,
    ///   or otherwise invalidated while mapped.
    ///
    /// Guest read/write/execute permissions are enforced by the MMU. They do not
    /// relax the host-memory validity requirements above.
    ///
    /// Returns [`MemoryFault`] if alignment or address validation fails.
    pub unsafe fn map_memory(
        &mut self,
        base: u64,
        ptr: *mut u8,
        size: u64,
        protections: MemoryProtections,
    ) -> Result<(), MemoryFault> {
        ensure!(ptr.addr().is_multiple_of(PAGE_SIZE));

        let start_vaddr = base;
        let end_vaddr = base.checked_add(size).ok_or_else(MemoryFault::fault)?;

        let end_page = div_page_size_checked(end_vaddr)?;
        let start_page = div_page_size_checked(start_vaddr)?;

        let Some(end_page) = u64_to_usize(end_page) else {
            panic!("could not map pages into view; out of host memory")
        };

        unsafe {
            // Safety: start page is smaller than end page, and end page fits in ram
            let start_page = u64_to_usize(start_page).unwrap_unchecked();

            if self.pages.len() < end_page {
                self.pages.resize_with(end_page, || Page::UNMAPPED)
            }


            let base_ptr = NonNull::new_unchecked(ptr.cast::<AtomicU8>());
            for page_idx in start_page..end_page {
                let page = self.pages.get_unchecked_mut(page_idx);
                Self::unmap_page(page, self.pending_dirty_pages.get_mut());
                let backing_page_idx = page_idx.unchecked_sub(start_page);
                let page_ptr = base_ptr.add(backing_page_idx.unchecked_mul(PAGE_SIZE));
                *page = Page::mapped(page_ptr, protections);
            }
        }

        Ok(())
    }

    fn get_pages_mut(
        &mut self,
        start: u64,
        size: u64
    ) -> Result<(&mut [Page], &mut HashSet<HostPointer>), MemoryFault> {
        let end = start.checked_add(size).ok_or_else(MemoryFault::fault)?;
        let end_page = div_page_size_checked(end)?;
        let start_page = div_page_size_checked(start)?;

        let (start_page, end_page) = usize::try_from(start_page)
            .and_then(|start_page| Ok((start_page, usize::try_from(end_page)?)))
            .ok()
            .ok_or_else(MemoryFault::fault)?;


        let pages = self
            .pages
            .get_mut(start_page..end_page)
            .ok_or_else(MemoryFault::fault)?;

        Ok((pages, self.pending_dirty_pages.get_mut()))
    }


    pub fn unmap_memory(&mut self, start: u64, size: u64) -> Result<(), MemoryFault> {
        let (pages, pending_dirty_pages) = self.get_pages_mut(start, size)?;
        for page in pages {
            Self::unmap_page(page, pending_dirty_pages);
        }
        Ok(())
    }

    pub fn mem_protect(
        &mut self,
        start: u64,
        size: u64,
        protections: MemoryProtections
    ) -> Result<(), MemoryFault> {
        // changing memory protections doesn't change the mapping of the host pointer
        // therefore there is no need to touch the pending dirty pages
        let (pages, _) = self.get_pages_mut(start, size)?;
        for page in &mut *pages {
            ensure!(page.is_mapped());
        }

        // Safety: all pages are mapped
        for page in pages {
            unsafe { page.mem_protect(protections) }
        }

        Ok(())
    }

    /// # Safety
    ///
    /// `vaddr_start` <= `vaddr_end`
    unsafe fn for_each_page_chunk(
        &self,
        vaddr_start: u64,
        vaddr_end: u64,
        required: PageFlags,
        mut f: impl FnMut(&Page, usize, usize, usize),
    ) -> Result<(), MemoryFault> {
        unsafe { core::hint::assert_unchecked(vaddr_start <= vaddr_end) }

        let (end_page, end_offset) = div_rem_page_size(vaddr_end);
        let end_page = u64_to_usize(end_page).ok_or_else(MemoryFault::fault)?;

        ensure!(end_page < self.pages.len());

        let (start_page, start_offset) = div_rem_page_size(vaddr_start);
        // Safety: end_page fits in usize and end_page >= start_page
        let start_page = unsafe { u64_to_usize(start_page).unwrap_unchecked() };

        // Safety: start_page < end_page < self.pages.len()
        let pages = unsafe { self.pages.get_unchecked(start_page..=end_page) };
        for page in pages {
            ensure!(page.has_access(required))
        }

        let mut buf_offset = 0usize;
        for (i, page) in pages.iter().enumerate() {
            let page_idx = unsafe { start_page.unchecked_add(i) };

            let page_off = if page_idx == start_page { start_offset } else { 0 };
            let page_end = if page_idx == end_page {
                // end_offset is < A::PAGE_SIZE
                // which is some usize, that means there is some usize bigger than us
                // so this can be incremented safely
                unsafe { end_offset.unchecked_add(1) }
            } else {
                PAGE_SIZE
            };

            let chunk_len = unsafe { page_end.unchecked_sub(page_off) };

            f(page, page_off, buf_offset, chunk_len);
            buf_offset = unsafe { buf_offset.unchecked_add(chunk_len) };
        }

        Ok(())
    }

    fn for_each_page_chunk_len(
        &self,
        vaddr: u64,
        len: usize,
        required: PageFlags,
        f: impl FnMut(&Page, usize, usize, usize),
    ) -> Result<(), MemoryFault> {
        if len == 0 {
            return Ok(())
        }

        let extra = unsafe { len.unchecked_sub(1) };
        let end = u64_add_usize(vaddr, extra).ok_or_else(MemoryFault::fault)?;
        // Safety: end is vaddr + len, with no overflow, and so this it must be bigger
        unsafe {
            self.for_each_page_chunk(
                vaddr,
                end,
                required,
                f
            )
        }
    }


    /// Loads a byte slice from virtual memory into `mem`.
    ///
    /// The load may span multiple pages. Every covered page must be mapped and have
    /// read permission.
    ///
    /// Concurrent stores are allowed. The returned bytes may reflect a mixture of
    /// values from racing stores, according to the atomicity guarantees of the
    /// underlying target operations.
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unreadable, or if address
    /// arithmetic overflows.
    ///
    /// This safe function must not invoke undefined behavior.
    pub fn load<'a>(
        &self,
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>]
    ) -> Result<&'a mut [u8], MemoryFault> {
        let result = self.for_each_page_chunk_len(
            vaddr,
            mem.len(),
            PageFlags::READ,
            |page, page_off, buf_off, chunk_len| unsafe {
                let range = buf_off..buf_off.unchecked_add(chunk_len);
                let dst = mem.get_unchecked_mut(range);
                page.load(page_off, dst).unwrap_unchecked();
            },
        );

        // Safety: mem has been filled
        result.map(|()| unsafe { mem.assume_init_mut() })
    }

    /// Stores a byte slice into virtual memory.
    ///
    /// The store may span multiple pages. Every covered page must be mapped and have
    /// write permission.
    ///
    /// Concurrent loads and stores are allowed. Other threads may observe the write
    /// as a sequence of byte or scalar operations according to the atomicity
    /// guarantees of the underlying target operations.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unwritable, or if address
    /// arithmetic overflows.
    ///
    /// This safe function must not invoke undefined behavior.
    #[inline(always)]
    pub fn store(&self, vaddr: u64, mem: &[u8]) -> Result<(), MemoryFault> {
        self.for_each_page_chunk_len(
            vaddr,
            mem.len(),
            PageFlags::WRITE,
            |page, page_off, buf_off, chunk_len| unsafe {
                let range = buf_off..buf_off.unchecked_add(chunk_len);
                let src = mem.get_unchecked(range);
                page.store(page_off, src).unwrap_unchecked();
            },
        )
    }
}

struct SecondPage<'a> {
    page: &'a Page,
    overflow_amount: NonZero<u8>,
}

struct SmallAccess<'a> {
    base_page: &'a Page,
    base_page_offset: usize,
    second_page: Option<SecondPage<'a>>
}

impl IoMMU {
    fn static_small_multibyte_acces<const BYTES: u8>(
        &self,
        vaddr: u64,
    ) -> Result<SmallAccess<'_>, MemoryFault> {
        const {
            assert!(BYTES > 1);
            assert!((BYTES as usize) < PAGE_SIZE);
            assert!(PAGE_SIZE.checked_add(BYTES as usize).is_some());
        }

        let (base_page_idx, base_page_offset) = div_rem_page_size(vaddr);
        let (base_page_idx, base_page) = u64_to_usize(base_page_idx)
            .and_then(|page_idx| Some((page_idx, self.pages.get(page_idx)?)))
            .ok_or_else(MemoryFault::fault)?;

        // TODO safety comments

        let end_offset = unsafe { base_page_offset.unchecked_add(usize::from(BYTES)) };

        let second_page = match end_offset > PAGE_SIZE {
            false => None,
            true => {
                cold_path();
                let overflow_amount = unsafe {
                    u8::try_from(end_offset.unchecked_sub(PAGE_SIZE)).unwrap_unchecked()
                };

                unsafe { core::hint::assert_unchecked(overflow_amount < BYTES) }

                let overflow_amount = unsafe {
                    NonZero::new_unchecked(overflow_amount)
                };

                let second_page = base_page_idx
                    .checked_add(1)
                    .and_then(|second_page_idx| self.pages.get(second_page_idx))
                    .ok_or_else(MemoryFault::fault)?;

                Some(SecondPage {
                    page: second_page,
                    overflow_amount
                })
            }
        };

        Ok(SmallAccess {
            base_page,
            base_page_offset,
            second_page
        })
    }
}


macro_rules! emit_multi_word_load_store {
    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            impl IoMMU {$(
                /// Loads a little-endian scalar from virtual memory.
                ///
                /// The access requires read permission for every page it touches. If the access
                /// crosses a page boundary, both pages must be mapped and readable.
                ///
                /// Atomicity follows the target ARM CPU's single-copy atomicity guarantees for
                /// naturally aligned scalar accesses of this width. Cross-page accesses may be
                /// implemented as multiple operations and should not be treated as a single
                /// atomic access.
                ///
                /// Returns [`MemoryFault`] on unmapped access, permission failure, or overflow.
                ///
                /// This safe function must not invoke undefined behavior.
                #[inline(always)]
                pub fn [<load $bits _le>](&self, vaddr: u64) -> Result<[<u $bits>], MemoryFault> {
                    let access = self.static_small_multibyte_acces::<{ $bits / 8 }>(vaddr)?;
                    match access.second_page {
                        // SAFETY:
                        // `static_small_multibyte_acces` returned `None` for `second_page`, so
                        // this access is fully contained in `base_page`.
                        //
                        // Therefore:
                        //
                        //   base_page_offset + size_of::<u$bits>() <= PAGE_SIZE
                        //
                        // which satisfies the safety requirement of `Page::load$bits`.
                        None => unsafe { access.base_page.[<load $bits>](access.base_page_offset) },

                        Some(second_page) => unsafe {
                            cold_path();

                            // The access crosses the page boundary.
                            //
                            // `overflow_amount` is the number of bytes that must be read from
                            // the start of the second page. The remaining bytes come from the
                            // end of the base page.
                            //
                            // Instead of doing byte-by-byte loads, we load:
                            //
                            //   1. one aligned little-endian word ending at the end of the base page
                            //   2. one aligned little-endian word starting at the beginning of the second page
                            //
                            // Then we shift/or the two words to reconstruct the requested
                            // little-endian value.

                            let hi_page_ptr = second_page.page.get_data_ptr(PageFlags::READ)?;
                            let lo_page_ptr = access.base_page.get_data_ptr(PageFlags::READ)?;


                            // SAFETY:
                            // In the crossing case:
                            //
                            //   end_offset      = base_page_offset + BYTES
                            //   overflow_amount = end_offset - PAGE_SIZE
                            //
                            // Therefore:
                            //
                            //   base_page_offset - overflow_amount
                            // = base_page_offset - (base_page_offset + BYTES - PAGE_SIZE)
                            // = PAGE_SIZE - BYTES
                            let lo_offset = const { PAGE_SIZE.strict_sub($bits / 8) };

                            // SAFETY:
                            // `hi_page_ptr` points to the start of the second page's readable
                            // data. Offset `0` is valid for a full `$bits`-wide load because
                            // `BYTES == $bits / 8` and `BYTES < PAGE_SIZE`.
                            //
                            // The aligned operation is valid because page data is assumed to be
                            // aligned sufficiently for these aligned page-boundary loads.
                            let hi_ptr = hi_page_ptr.as_ptr();

                            // SAFETY:
                            // From the proof above:
                            //
                            //   lo_offset == PAGE_SIZE - BYTES
                            //
                            // so `lo_ptr` points to the first byte of the final `$bits`-wide
                            // word in the base page.
                            //
                            // This means the load is fully contained inside the base page and
                            // ends exactly at the page boundary.
                            //
                            // The aligned operation is valid because this pointer is page-end
                            // aligned for a `$bits`-wide word, assuming `PAGE_SIZE` is a
                            // multiple of `BYTES`.
                            let lo_ptr = lo_page_ptr.byte_add(lo_offset).as_ptr();


                            let hi = memops::[<load $bits _le_aligned>](hi_ptr);
                            let lo = memops::[<load $bits _le_aligned>](lo_ptr);

                            // SAFETY:
                            // `overflow_amount` is a `NonZero<u8>`, so it is at least `1`.
                            // `static_small_multibyte_acces` also guarantees
                            // `overflow_amount < BYTES`.
                            //
                            // Therefore:
                            //
                            //   0 < overflow_amount * 8 < $bits
                            //
                            // So multiplying by 8 cannot overflow `u8` for the supported
                            // widths, and the resulting bit offset is strictly less than the
                            // integer width.
                            let bit_offset = u32::from(
                                second_page.overflow_amount.get().unchecked_mul(8)
                            );

                            // SAFETY:
                            // Since `bit_offset < $bits`, this subtraction cannot underflow.
                            //
                            // Also, because `bit_offset > 0`, `hi_shift` is strictly less than
                            // `$bits`.
                            let hi_shift = ($bits as u32).unchecked_sub(bit_offset);
                            let lo_shift = bit_offset;

                            // SAFETY:
                            // Both shift amounts are in `1..$bits`, so neither unchecked shift
                            // uses an invalid shift amount.
                            //
                            // `lo >> lo_shift` discards the bytes before the requested virtual
                            // address in the base-page word.
                            //
                            // `hi << hi_shift` moves the bytes from the second page into the
                            // high end of the result.
                            //
                            // OR-ing both pieces reconstructs the requested little-endian
                            // `$bits` value spanning the two pages.
                            Ok(hi.unchecked_shl(hi_shift) | lo.unchecked_shr(lo_shift))
                        }
                    }
                }

                /// Stores a little-endian scalar into virtual memory.
                ///
                /// The access requires write permission for every page it touches. If the access
                /// crosses a page boundary, both pages must be mapped and writable.
                ///
                /// Atomicity follows the target ARM CPU's single-copy atomicity guarantees for
                /// naturally aligned scalar accesses of this width. Cross-page accesses may be
                /// implemented as multiple operations and should not be treated as a single
                /// atomic access.
                ///
                /// Returns [`MemoryFault`] on unmapped access, permission failure, overflow, or
                /// required alignment failure.
                ///
                /// This safe function must not invoke undefined behavior.
                #[inline(always)]
                pub fn [<store $bits _le>](&self, vaddr: u64, value: [<u $bits>]) -> Result<(), MemoryFault> {
                    let access = self.static_small_multibyte_acces::<{ $bits / 8 }>(vaddr)?;

                    match access.second_page {
                        // SAFETY:
                        // `static_small_multibyte_acces` returned `None` for `second_page`, so
                        // this access is fully contained in `base_page`.
                        //
                        // Therefore:
                        //
                        //   base_page_offset + size_of::<u$bits>() <= A::PAGE_SIZE
                        //
                        // which satisfies the safety requirement of `Page::store$bits`.
                        None => unsafe {
                            access.base_page.[<store $bits>](access.base_page_offset, value)
                        },

                        Some(second_page) => unsafe {
                            // Note: we can't load 2 words and combine them like the load case
                            //       since that would alter/mess with the atomicity of the bytes
                            //       next to the value
                            let bytes = value.to_le_bytes();

                            let hi_page = &second_page.page;
                            let lo_page = &access.base_page;

                            let hi_page_flags = hi_page.load_flags();
                            let lo_page_flags = lo_page.load_flags();

                            ensure!(
                                hi_page_flags.contains(PageFlags::WRITE),
                                lo_page_flags.contains(PageFlags::WRITE),
                            );

                            let hi_page_ptr = hi_page.get_data_ptr_unchecked();
                            let lo_page_ptr = lo_page.get_data_ptr_unchecked();

                            let overflow = usize::from(second_page.overflow_amount.get());


                            let mut active_ptr = hi_page_ptr.add(overflow).as_ptr();
                            let mut i = bytes.len();
                            for _ in 0..overflow {
                                active_ptr = active_ptr.sub(1);
                                i = i.unchecked_sub(1);
                                let byte = *bytes.get_unchecked(i);
                                memops::store_byte(active_ptr, byte)
                            }

                            active_ptr = lo_page_ptr
                                .add(const { PAGE_SIZE.strict_sub(1) })
                                .as_ptr();

                            loop {
                                i = i.unchecked_sub(1);
                                let byte = *bytes.get_unchecked(i);
                                memops::store_byte(active_ptr, byte);

                                if i == 0 {
                                    break
                                }
                                active_ptr = active_ptr.sub(1)
                            }

                            lo_page.set_insn_dirty(lo_page_flags);
                            hi_page.set_insn_dirty(hi_page_flags);
                            Ok(())
                        }
                    }
                }
            )+}
        }
    };
}

emit_multi_word_load_store! { 64, 32, 16 }

impl IoMMU {
    pub(crate) fn single_page_aligned_access<const ALIGN: u8>(
        &self,
        vaddr: u64
    ) -> Result<(&Page, usize), MemoryFault> {
        const {
            assert!(ALIGN.is_power_of_two());
            assert!(PAGE_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_multiple_of(ALIGN as usize));
        }

        ensure!(vaddr.is_multiple_of(u64::from(ALIGN)));

        let (page, offset) = div_rem_page_size(vaddr);
        let page = u64_to_usize(page)
            .and_then(|page_idx| self.pages.get(page_idx))
            .ok_or_else(MemoryFault::fault)?;

        Ok((page, offset))
    }

    pub fn load_byte(&self, vaddr: u64) -> Result<u8, MemoryFault> {
        const ALIGN: u8 = 1;
        let (page, offset) = self.single_page_aligned_access::<ALIGN>(vaddr)?;
        // Safety: offset is the result of x % PAGE_SIZE and so must be smaller than page size
        unsafe { page.load_byte(offset) }
    }

    pub fn store_byte(&self, vaddr: u64, value: u8) -> Result<(), MemoryFault> {
        const ALIGN: u8 = 1;
        let (page, offset) = self.single_page_aligned_access::<ALIGN>(vaddr)?;
        // Safety: offset is the result of x % PAGE_SIZE and so must be smaller than page size
        unsafe { page.store_byte(offset, value) }
    }
}


impl IoMMU {
    pub(crate) fn fetch_aarch64_full(&self, vaddr: u64) -> Result<(HostPointer, u32), MemoryFault> {
        const ALIGN: u8 = 4;
        let (page, offset) = self.single_page_aligned_access::<ALIGN>(vaddr)?;
        let data_ptr = page.get_data_ptr(PageFlags::EXECUTE)?;
        unsafe {
            let word_ptr = data_ptr.add(offset);
            let word = memops::load32_le_aligned(word_ptr.as_ptr());
            Ok((HostPointer::new(word_ptr), word))
        }
    }

    pub fn fetch_aarch64(&self, vaddr: u64) -> Result<u32, MemoryFault> {
        self.fetch_aarch64_full(vaddr).map(|(_ptr, word)| word)
    }

    pub(crate) fn drain_dirty_icache(&self) -> impl Iterator<Item=Range<HostPointer>> {
        let mut dirty_pages = {
            let mut lock = self.pending_dirty_pages.lock();
            core::mem::take(&mut *lock)
        };

        for page in &self.pages {
            if page.take_insn_dirty() {
                let page_ptr = unsafe { page.get_data_ptr_unchecked() };
                dirty_pages.insert(HostPointer::new(page_ptr));
            }
        }

        dirty_pages.into_iter().map(|ptr| {
            let page_start = ptr;
            let page_end = HostPointer::new(unsafe { ptr.0.add(PAGE_SIZE) });
            page_start..page_end
        })
    }
}

impl IoMMU {
    pub(crate) fn pages(&self) -> &[Page] {
        &self.pages
    }
}


#[cfg(test)]
mod mmu_tests {
    use super::*;

    use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    const BASE: u64 = 0;

    fn page_size() -> usize {
        PAGE_SIZE
    }

    fn page_addr(page: usize) -> u64 {
        BASE.strict_add(u64::try_from(page.strict_mul(page_size())).unwrap())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn pattern_byte(i: usize) -> u8 {
        (i as u8).wrapping_mul(37).wrapping_add(0x51)
    }

    fn pattern_array<const N: usize>(start: usize) -> [u8; N] {
        std::array::from_fn(|i| pattern_byte(start.wrapping_add(i)))
    }

    struct PageBacking {
        ptr: NonNull<u8>,
        len: usize,
    }

    impl PageBacking {
        fn new(pages: usize) -> Self {
            let len = pages.checked_mul(page_size()).unwrap();
            let layout = Layout::from_size_align(len, page_size()).unwrap();

            let raw = match len{
                0 => core::ptr::dangling_mut(),
                _ => unsafe { alloc_zeroed(layout) }
            };
            let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));

            Self { ptr, len }
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            self.ptr.as_ptr()
        }

        fn get_page(&mut self, page: usize) -> &mut [u8; PAGE_SIZE] {
            let index = page.strict_mul(page_size());
            assert!(index < self.len);
            unsafe { &mut *self.ptr.as_ptr().add(page.strict_mul(page_size())).cast() }
        }

        unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }

    impl Drop for PageBacking {
        fn drop(&mut self) {
            if self.len != 0 {
                unsafe {
                    dealloc(
                        self.ptr.as_ptr(),
                        Layout::from_size_align_unchecked(self.len, PAGE_SIZE)
                    );
                }
            }
        }
    }

    struct Fixture {
        mmu: IoMMU,
        _backing: PageBacking,
    }

    impl Fixture {
        fn new(pages: usize, protections: MemoryProtections) -> Self {
            Self::with_bytes(pages, protections, |_| 0)
        }

        fn with_bytes(
            pages: usize,
            protections: MemoryProtections,
            mut byte: impl FnMut(usize) -> u8,
        ) -> Self {
            let mut backing = PageBacking::new(pages);

            unsafe {
                for (i, dst) in backing.as_mut_slice().iter_mut().enumerate() {
                    *dst = byte(i);
                }
            }

            let mut mmu = IoMMU::new(CpuFabric::new());
            unsafe {
                mmu.map_memory(
                    BASE,
                    backing.as_mut_ptr(),
                    pages.strict_mul(page_size()) as u64,
                    protections,
                )
                    .unwrap();
            }

            Self {
                mmu,
                _backing: backing,
            }
        }

        fn with_page_protections(protections: &[MemoryProtections]) -> Self {
            Self::with_page_protections_and_bytes(protections, |_| 0)
        }

        fn with_page_protections_and_bytes(
            protections: &[MemoryProtections],
            mut byte: impl FnMut(usize) -> u8,
        ) -> Self {
            assert!(!protections.is_empty());

            let mut backing = PageBacking::new(protections.len());

            unsafe {
                for (i, dst) in backing.as_mut_slice().iter_mut().enumerate() {
                    *dst = byte(i);
                }
            }

            let mut mmu = IoMMU::new(CpuFabric::new());
            for (page, protections) in protections.iter().copied().enumerate() {
                unsafe {
                    mmu.map_memory(
                        page_addr(page),
                        backing.get_page(page).as_mut_ptr(),
                        page_size() as u64,
                        protections,
                    ).unwrap();
                }
            }

            Self {
                mmu,
                _backing: backing,
            }
        }

        fn read_vec(&self, vaddr: u64, len: usize) -> Vec<u8> {
            let mut out = vec![MaybeUninit::<u8>::uninit(); len];
            self.mmu.load(vaddr, &mut out).unwrap().to_vec()
        }

        fn flags(&self, page: usize) -> PageFlags {
            self.mmu.pages[page].load_flags()
        }

        fn is_dirty(&self, page: usize) -> bool {
            self.flags(page).contains(PageFlags::INSN_DIRTY)
        }
    }

    #[test]
    fn new_mmu_faults_non_empty_accesses() {
        let mmu = IoMMU::new(CpuFabric::new());

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(mmu.load(BASE, &mut one).is_err());
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

        let loaded = mmu.load(0x1234_5678, &mut empty).unwrap();
        assert!(loaded.is_empty());

        assert!(mmu.store(0x1234_5678, &[]).is_ok());
    }

    #[test]
    fn map_memory_rejects_unaligned_base_size_ptr_and_overflow() {
        let mut backing = PageBacking::new(2);
        let mut mmu = IoMMU::new(CpuFabric::new());

        unsafe {
            assert!(mmu
                .map_memory(
                    1,
                    backing.as_mut_ptr(),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    BASE,
                    backing.get_page(0).as_mut_ptr().add(1),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            assert!(mmu
                .map_memory(
                    BASE,
                    backing.as_mut_ptr(),
                    (page_size() - 1) as u64,
                    MemoryProtections::READ,
                )
                .is_err());

            let overflow_base = u64::MAX - ((page_size() as u64) - 1);

            assert!(mmu
                .map_memory(
                    overflow_base,
                    backing.as_mut_ptr(),
                    page_size() as u64,
                    MemoryProtections::READ,
                )
                .is_err());
        }
    }

    #[test]
    fn nonzero_base_maps_only_requested_page() {
        let mut backing = PageBacking::new(1);
        unsafe {
            backing.as_mut_slice()[0] = 0xaa;
        }

        let mut mmu = IoMMU::new(CpuFabric::new());
        let base = page_addr(2);

        unsafe {
            mmu.map_memory(
                base,
                backing.as_mut_ptr(),
                page_size() as u64,
                MemoryProtections::READ,
            )
                .unwrap();
        }

        assert!(mmu.load_byte(0).is_err());
        assert!(mmu.load_byte(page_addr(1)).is_err());
        assert_eq!(mmu.load_byte(base).unwrap(), 0xaa);
    }

    #[test]
    fn read_only_page_allows_loads_and_rejects_stores() {
        let fixture = Fixture::new(1, MemoryProtections::READ);

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0);

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_ok());

        assert!(fixture.mmu.store_byte(BASE, 1).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
        assert!(fixture.mmu.store32_le(BASE, 0xfeed_beef).is_err());
        assert!(fixture.mmu.store64_le(BASE, 0xfeed_beef_dead_cafe).is_err());
    }

    #[test]
    fn write_only_page_allows_stores_and_rejects_loads() {
        let fixture = Fixture::new(1, MemoryProtections::WRITE);

        assert!(fixture.mmu.store_byte(BASE, 1).is_ok());
        assert!(fixture.mmu.store(BASE, &[1, 2, 3]).is_ok());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_ok());

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.load32_le(BASE).is_err());
        assert!(fixture.mmu.load64_le(BASE).is_err());
    }

    #[test]
    fn execute_only_page_rejects_data_loads_and_stores() {
        let fixture = Fixture::new(1, MemoryProtections::EXECUTE);

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 1).is_err());

        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn empty_protection_mapping_rejects_all_data_accesses() {
        let fixture = Fixture::new(1, MemoryProtections::empty());

        let mut one = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut one).is_err());
        assert!(fixture.mmu.store(BASE, &[1]).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 1).is_err());
    }

    #[test]
    fn byte_load_store_roundtrip() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        for i in 0..64_u8 {
            fixture.mmu.store_byte(i.into(), pattern_byte(i.into())).unwrap();
        }

        for i in 0..64_u8 {
            assert_eq!(
                fixture.mmu.load_byte(i.into()).unwrap(),
                pattern_byte(i.into())
            );
        }
    }

    #[test]
    fn slice_store_load_roundtrip_inside_one_page() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let data = pattern_array::<128>(0);
        let start = 17u64;

        fixture.mmu.store(start, &data).unwrap();

        assert_eq!(fixture.read_vec(start, data.len()), data);
    }

    #[test]
    fn slice_store_load_roundtrip_across_two_pages() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE
        );

        let start = u64::try_from(page_size().strict_sub(5)).unwrap();
        let data = pattern_array::<13>(0);

        fixture.mmu.store(start, &data).unwrap();

        assert_eq!(fixture.read_vec(start, data.len()), data);
    }

    #[test]
    fn slice_store_across_boundary_requires_write_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ | MemoryProtections::WRITE,
            MemoryProtections::READ,
        ]);

        let start = (page_size() - 1) as u64;

        assert!(fixture.mmu.store(start, &[1, 2]).is_err());
    }

    #[test]
    fn slice_load_across_boundary_requires_read_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ | MemoryProtections::WRITE,
            MemoryProtections::WRITE,
        ]);

        let start = (page_size() - 1) as u64;
        let mut out = [MaybeUninit::<u8>::uninit(); 2];

        assert!(fixture.mmu.load(start, &mut out).is_err());
    }

    #[test]
    fn failed_slice_store_across_unmapped_page_does_not_partially_write() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let last = (page_size() - 1) as u64;
        fixture.mmu.store_byte(last, 0xaa).unwrap();

        assert!(fixture.mmu.store(last, &[0xbb, 0xcc]).is_err());

        assert_eq!(fixture.mmu.load_byte(last).unwrap(), 0xaa);
    }

    #[test]
    fn scalar_loads_inside_one_page_are_little_endian() {
        let fixture = Fixture::with_bytes(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let off16 = 11usize;
        let off32 = 19usize;
        let off64 = 29usize;

        assert_eq!(
            fixture.mmu.load16_le(off16 as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(off16))
        );

        assert_eq!(
            fixture.mmu.load32_le(off32 as u64).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(off32))
        );

        assert_eq!(
            fixture.mmu.load64_le(off64 as u64).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(off64))
        );
    }

    #[test]
    fn scalar_stores_inside_one_page_are_little_endian() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        fixture.mmu.store16_le(11, 0xbeef).unwrap();
        assert_eq!(fixture.read_vec(11, 2), 0xbeefu16.to_le_bytes().to_vec());

        fixture.mmu.store32_le(19, 0xaabb_ccdd).unwrap();
        assert_eq!(
            fixture.read_vec(19, 4),
            0xaabb_ccddu32.to_le_bytes().to_vec()
        );

        fixture.mmu.store64_le(29, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(
            fixture.read_vec(29, 8),
            0x0123_4567_89ab_cdefu64.to_le_bytes().to_vec()
        );
    }

    #[test]
    fn scalar_loads_at_last_non_crossing_offsets_succeed() {
        let fixture = Fixture::with_bytes(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let p = page_size();

        assert_eq!(
            fixture.mmu.load16_le((p - 2) as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(p - 2))
        );

        assert_eq!(
            fixture.mmu.load32_le((p - 4) as u64).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(p - 4))
        );

        assert_eq!(
            fixture.mmu.load64_le((p - 8) as u64).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(p - 8))
        );
    }

    #[test]
    fn scalar_loads_crossing_page_boundary_read_expected_bytes() {
        let fixture = Fixture::with_bytes(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        let p = page_size();

        assert_eq!(
            fixture.mmu.load16_le((p - 1) as u64).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(p - 1))
        );

        for bytes_in_first_page in 1..4 {
            let start = p - bytes_in_first_page;
            assert_eq!(
                fixture.mmu.load32_le(start as u64).unwrap(),
                u32::from_le_bytes(pattern_array::<4>(start)),
                "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }

        for bytes_in_first_page in 1..8 {
            let start = p - bytes_in_first_page;
            assert_eq!(
                fixture.mmu.load64_le(start as u64).unwrap(),
                u64::from_le_bytes(pattern_array::<8>(start)),
                "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }
    }

    #[test]
    fn scalar_stores_crossing_page_boundary_write_expected_bytes() {
        let fixture = Fixture::new(2, MemoryProtections::READ | MemoryProtections::WRITE);
        let p = page_size();

        let value16 = 0xbeefu16;
        fixture.mmu.store16_le((p - 1) as u64, value16).unwrap();
        assert_eq!(
            fixture.read_vec((p - 1) as u64, 2),
            value16.to_le_bytes().to_vec()
        );

        for bytes_in_first_page in 1..4_u8 {
            let start = p - bytes_in_first_page as usize;
            let value = 0xaabb_ccddu32 ^ bytes_in_first_page as u32;

            fixture.mmu.store32_le(start as u64, value).unwrap();

            assert_eq!(
                fixture.read_vec(start as u64, 4),
                value.to_le_bytes().to_vec(),
                "u32 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }

        for bytes_in_first_page in 1..8 {
            let start = p - bytes_in_first_page;
            let value = 0x0123_4567_89ab_cdefu64 ^ bytes_in_first_page as u64;

            fixture.mmu.store64_le(start as u64, value).unwrap();

            assert_eq!(
                fixture.read_vec(start as u64, 8),
                value.to_le_bytes().to_vec(),
                "u64 crossing with {bytes_in_first_page} byte(s) in the first page"
            );
        }
    }

    #[test]
    fn crossing_scalar_load_requires_read_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::READ,
            MemoryProtections::WRITE,
        ]);

        assert!(fixture.mmu.load16_le((page_size() - 1) as u64).is_err());
    }

    #[test]
    fn crossing_scalar_store_requires_write_on_both_pages() {
        let fixture = Fixture::with_page_protections(&[
            MemoryProtections::WRITE,
            MemoryProtections::READ,
        ]);

        assert!(fixture
            .mmu
            .store16_le((page_size() - 1) as u64, 0xbeef)
            .is_err());
    }

    #[test]
    fn crossing_scalar_access_to_unmapped_second_page_faults() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        let last = (page_size() - 1) as u64;

        fixture.mmu.store_byte(last, 0xaa).unwrap();

        assert!(fixture.mmu.load16_le(last).is_err());
        assert!(fixture.mmu.store16_le(last, 0xbeef).is_err());

        assert_eq!(fixture.mmu.load_byte(last).unwrap(), 0xaa);
    }

    #[test]
    fn store_byte_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_slice_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store(8, &[1, 2, 3, 4]).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_slice_crossing_pages_marks_both_executable_pages_dirty() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));
        assert!(!fixture.is_dirty(1));

        fixture
            .mmu
            .store((page_size() - 2) as u64, &[1, 2, 3, 4])
            .unwrap();

        assert!(fixture.is_dirty(0));
        assert!(fixture.is_dirty(1));
    }

    #[test]
    fn single_page_scalar_store_marks_executable_page_dirty() {
        let fixture = Fixture::new(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));

        fixture.mmu.store64_le(8, 0x0123_4567_89ab_cdef).unwrap();

        assert!(fixture.is_dirty(0));
    }

    #[test]
    fn store_to_non_executable_page_does_not_mark_dirty() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        fixture.mmu.store16_le(8, 0xbeef).unwrap();
        fixture.mmu.store(16, &[1, 2, 3]).unwrap();

        assert!(!fixture.is_dirty(0));
    }

    // BUG:
    // `for_each_page_chunk` treats `vaddr_end` as an inclusive touched address.
    // Ranges are normally `[start, end)`, so a load of exactly one page from the
    // start of a one-page mapping should not require page 1 to exist.
    #[test]
    fn bug_load_exactly_one_page_should_not_require_next_page() {
        let fixture = Fixture::with_bytes(1, MemoryProtections::READ, pattern_byte);

        let mut out = Box::new_uninit_slice(page_size());

        assert!(fixture.mmu.load(BASE, &mut out).is_ok());
    }

    // BUG:
    // Same exclusive-end bug as above, but through store.
    #[test]
    fn bug_store_exactly_one_page_should_not_require_next_page() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let data = vec![0x5a; page_size()];

        assert!(fixture.mmu.store(BASE, &data).is_ok());
    }

    // BUG:
    // Ending exactly at a page boundary should not check permissions on the next page,
    // because zero bytes are accessed there.
    #[test]
    fn bug_load_ending_at_boundary_should_not_require_read_on_next_page() {
        let fixture = Fixture::with_page_protections_and_bytes(
            &[
                MemoryProtections::READ,
                MemoryProtections::WRITE,
            ],
            pattern_byte,
        );

        let mut out = vec![MaybeUninit::<u8>::uninit(); page_size()];

        assert!(fixture.mmu.load(BASE, &mut out).is_ok());
    }

    // BUG:
    // Cross-page scalar stores write executable pages but never call `set_insn_dirty`
    // on either touched page.
    #[test]
    fn bug_cross_page_scalar_store_should_mark_touched_executable_pages_dirty() {
        let fixture = Fixture::new(
            2,
            MemoryProtections::READ
                | MemoryProtections::WRITE
                | MemoryProtections::EXECUTE,
        );

        assert!(!fixture.is_dirty(0));
        assert!(!fixture.is_dirty(1));

        let addr = (page_size() - 1) as u64;

        fixture
            .mmu
            .store16_le(addr, 0xbeef)
            .unwrap();

        assert!(fixture.is_dirty(0));
        assert!(fixture.is_dirty(1));
    }

    fn sparse_fixture_with_hole() -> Fixture {
        let mut backing = PageBacking::new(3);

        unsafe {
            for (i, byte) in backing.as_mut_slice().iter_mut().enumerate() {
                *byte = pattern_byte(i);
            }
        }

        let mut mmu = IoMMU::new(CpuFabric::new());
        let page = page_size() as u64;

        unsafe {
            mmu.map_memory(
                page_addr(0),
                backing.get_page(0).as_mut_ptr(),
                page,
                MemoryProtections::READ | MemoryProtections::WRITE,
            )
                .unwrap();

            // Intentionally skip page 1.

            mmu.map_memory(
                page_addr(2),
                backing.get_page(2).as_mut_ptr(),
                page,
                MemoryProtections::READ | MemoryProtections::WRITE,
            )
                .unwrap();
        }

        Fixture {
            mmu,
            _backing: backing,
        }
    }

    #[test]
    fn unmap_memory_unmaps_one_page() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);

        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_err());

        let mut out = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut out).is_err());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_err());

        assert!(fixture.mmu.load16_le(BASE).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn unmap_memory_unmaps_only_requested_page_range() {
        let mut fixture = Fixture::new(3, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(page_addr(0), 0x10).unwrap();
        fixture.mmu.store_byte(page_addr(1), 0x20).unwrap();
        fixture.mmu.store_byte(page_addr(2), 0x30).unwrap();

        fixture.mmu.unmap_memory(page_addr(1), page).unwrap();

        assert_eq!(fixture.mmu.load_byte(page_addr(0)).unwrap(), 0x10);
        assert!(fixture.mmu.load_byte(page_addr(1)).is_err());
        assert_eq!(fixture.mmu.load_byte(page_addr(2)).unwrap(), 0x30);

        assert!(fixture.mmu.store_byte(page_addr(0), 0x11).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(1), 0x21).is_err());
        assert!(fixture.mmu.store_byte(page_addr(2), 0x31).is_ok());
    }

    #[test]
    fn unmap_memory_can_unmap_multiple_pages() {
        let mut fixture = Fixture::new(4, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(page_addr(1), page * 2).unwrap();

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_err());
        assert!(fixture.mmu.load_byte(page_addr(2)).is_err());
        assert!(fixture.mmu.load_byte(page_addr(3)).is_ok());
    }

    #[test]
    fn unmap_memory_is_idempotent_for_existing_page_entries() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(BASE, page).unwrap();
        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());
    }

    #[test]
    fn unmap_memory_rejects_unaligned_start_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.mmu.unmap_memory(1, page).is_err());

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_unaligned_size_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        assert!(fixture.mmu.unmap_memory(BASE, page - 1).is_err());

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_range_past_page_table() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture.mmu.unmap_memory(page_addr(1), page).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_ok());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
    }

    #[test]
    fn unmap_memory_rejects_address_overflow() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        assert!(fixture.mmu.unmap_memory(u64::MAX, 1).is_err());

        assert!(fixture.mmu.load_byte(BASE).is_ok());
    }

    #[test]
    fn unmap_memory_allows_zero_sized_noop_at_start() {
        let mut mmu = IoMMU::new(CpuFabric::new());

        assert!(mmu.unmap_memory(BASE, 0).is_ok());
    }

    #[test]
    fn unmap_memory_then_remap_restores_access() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());

        unsafe {
            fixture
                .mmu
                .map_memory(
                    BASE,
                    fixture._backing.as_mut_ptr(),
                    page,
                    MemoryProtections::READ | MemoryProtections::WRITE,
                )
                .unwrap();
        }

        fixture.mmu.store_byte(BASE, 0xbb).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_can_make_page_read_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .unwrap();

        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_err());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_err());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_err());
    }

    #[test]
    fn mem_protect_can_make_page_write_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.store_byte(BASE, 0xaa).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::WRITE)
            .unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.load16_le(BASE).is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xbb).is_ok());
        assert!(fixture.mmu.store(BASE, &[0xcc]).is_ok());
        assert!(fixture.mmu.store16_le(BASE, 0xbeef).is_ok());
    }

    #[test]
    fn mem_protect_can_make_page_execute_only_for_data_accesses() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::EXECUTE)
            .unwrap();

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());

        let mut out = [MaybeUninit::<u8>::uninit()];
        assert!(fixture.mmu.load(BASE, &mut out).is_err());
        assert!(fixture.mmu.store(BASE, &[0xaa]).is_err());
    }

    #[test]
    fn mem_protect_can_restore_read_write_after_read_only() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());

        fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ | MemoryProtections::WRITE)
            .unwrap();

        fixture.mmu.store_byte(BASE, 0xbb).unwrap();
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_only_changes_requested_pages() {
        let mut fixture = Fixture::new(2, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(page_addr(1), page, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(1), 0xbb).is_err());

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_ok());
    }

    #[test]
    fn mem_protect_can_change_multiple_pages() {
        let mut fixture = Fixture::new(3, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture
            .mmu
            .mem_protect(page_addr(0), page * 2, MemoryProtections::READ)
            .unwrap();

        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_err());
        assert!(fixture.mmu.store_byte(page_addr(1), 0xbb).is_err());
        assert!(fixture.mmu.store_byte(page_addr(2), 0xcc).is_ok());

        assert!(fixture.mmu.load_byte(page_addr(0)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(1)).is_ok());
        assert!(fixture.mmu.load_byte(page_addr(2)).is_ok());
    }

    #[test]
    fn mem_protect_rejects_unmapped_page() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        fixture.mmu.unmap_memory(BASE, page).unwrap();

        assert!(fixture
            .mmu
            .mem_protect(BASE, page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.load_byte(BASE).is_err());
        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_err());
    }

    #[test]
    fn mem_protect_rejects_sparse_range_and_does_not_partially_update() {
        let mut fixture = sparse_fixture_with_hole();
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(page_addr(0), page * 3, MemoryProtections::READ)
            .is_err());

        // Page 0 and page 2 must still be writable. This catches accidental
        // partial updates if the implementation protects pages as it checks them.
        assert!(fixture.mmu.store_byte(page_addr(0), 0xaa).is_ok());
        assert!(fixture.mmu.store_byte(page_addr(2), 0xbb).is_ok());

        assert_eq!(fixture.mmu.load_byte(page_addr(0)).unwrap(), 0xaa);
        assert_eq!(fixture.mmu.load_byte(page_addr(2)).unwrap(), 0xbb);
    }

    #[test]
    fn mem_protect_rejects_unaligned_start_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(1, page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_unaligned_size_and_leaves_mapping_unchanged() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(BASE, page - 1, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_range_past_page_table() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);
        let page = page_size() as u64;

        assert!(fixture
            .mmu
            .mem_protect(page_addr(1), page, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
        assert_eq!(fixture.mmu.load_byte(BASE).unwrap(), 0xaa);
    }

    #[test]
    fn mem_protect_rejects_address_overflow() {
        let mut fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        assert!(fixture
            .mmu
            .mem_protect(u64::MAX, 1, MemoryProtections::READ)
            .is_err());

        assert!(fixture.mmu.store_byte(BASE, 0xaa).is_ok());
    }

    #[test]
    fn mem_protect_allows_zero_sized_noop_at_start() {
        let mut mmu = IoMMU::new(CpuFabric::new());
        assert!(mmu.mem_protect(BASE, 0, MemoryProtections::READ).is_ok());
    }

    #[test]
    fn unaligned_scalar_loads_inside_page_succeed() {
        let fixture = Fixture::with_bytes(
            1,
            MemoryProtections::READ | MemoryProtections::WRITE,
            pattern_byte,
        );

        assert_eq!(
            fixture.mmu.load16_le(1).unwrap(),
            u16::from_le_bytes(pattern_array::<2>(1))
        );

        assert_eq!(
            fixture.mmu.load32_le(1).unwrap(),
            u32::from_le_bytes(pattern_array::<4>(1))
        );

        assert_eq!(
            fixture.mmu.load64_le(1).unwrap(),
            u64::from_le_bytes(pattern_array::<8>(1))
        );
    }

    #[test]
    fn unaligned_scalar_stores_inside_page_succeed() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        fixture.mmu.store16_le(1, 0xbeef).unwrap();
        assert_eq!(fixture.read_vec(1, 2), 0xbeefu16.to_le_bytes().to_vec());

        fixture.mmu.store32_le(3, 0xaabb_ccdd).unwrap();
        assert_eq!(fixture.read_vec(3, 4), 0xaabb_ccddu32.to_le_bytes().to_vec());

        fixture.mmu.store64_le(5, 0x0123_4567_89ab_cdef).unwrap();
        assert_eq!(
            fixture.read_vec(5, 8),
            0x0123_4567_89ab_cdefu64.to_le_bytes().to_vec()
        );
    }

    #[test]
    fn failed_crossing_scalar_store_does_not_write_first_page_when_second_page_lacks_write() {
        let fixture = Fixture::with_page_protections_and_bytes(
            &[
                MemoryProtections::READ | MemoryProtections::WRITE,
                MemoryProtections::READ,
            ],
            pattern_byte,
        );

        let start = (page_size() - 1) as u64;
        let before = fixture.mmu.load_byte(start).unwrap();

        assert!(fixture.mmu.store16_le(start, 0xbeef).is_err());

        assert_eq!(fixture.mmu.load_byte(start).unwrap(), before);
    }

    #[test]
    fn failed_crossing_scalar_store_does_not_write_second_page_when_first_page_lacks_write() {
        let fixture = Fixture::with_page_protections_and_bytes(
            &[
                MemoryProtections::READ,
                MemoryProtections::READ | MemoryProtections::WRITE,
            ],
            pattern_byte,
        );

        let start = (page_size() - 1) as u64;
        let second_page_before = fixture.mmu.load_byte(page_addr(1)).unwrap();

        assert!(fixture.mmu.store16_le(start, 0xbeef).is_err());

        assert_eq!(fixture.mmu.load_byte(page_addr(1)).unwrap(), second_page_before);
    }

    #[test]
    fn scalar_accesses_near_u64_max_fault_instead_of_wrapping() {
        let fixture = Fixture::new(1, MemoryProtections::READ | MemoryProtections::WRITE);

        assert!(fixture.mmu.load16_le(u64::MAX).is_err());
        assert!(fixture.mmu.load32_le(u64::MAX - 1).is_err());
        assert!(fixture.mmu.load64_le(u64::MAX - 3).is_err());

        assert!(fixture.mmu.store16_le(u64::MAX, 0xbeef).is_err());
        assert!(fixture.mmu.store32_le(u64::MAX - 1, 0xaabb_ccdd).is_err());
        assert!(fixture
            .mmu
            .store64_le(u64::MAX - 3, 0x0123_4567_89ab_cdef)
            .is_err());
    }
}