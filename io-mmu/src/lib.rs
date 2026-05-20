// TODO: replace all of this documentation; as it is outdated

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
//! 2. Map one or more host memory regions with [`IoMMU::map`].
//! 3. Access byte ranges through [`IoMMU::load_into_uninit`] / [`IoMMU::store`].
//! 4. Access scalars through `load_byte/load16_le/load32_le/load64_le` and
//!    `store_byte/store16_le/store32_le/store64_le`.

use crate::cpu_fabric::CpuFabric;
use crate::page_table::PageTable;
use emu_abi::abort::AbortGuard;
use emu_abi::convert::u64_to_usize;
use emu_abi::internal_traits::{AsFFI, CpuFabricPrivate, ICache};
use emu_abi::memory::{
    HostPointer, IoMMUIdentifier, IoMMUIdentifierRef, MemProt, PAGE_OFFSET_MASK_U64, PAGE_SHIFT,
    PAGE_SIZE, PAGE_SIZE_U64, Page, PageNumber, PagePointer, Tlb,
};
use std::hint::cold_path;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::num::NonZero;
use std::process::abort;
use std::ptr::NonNull;
use std::sync::atomic::AtomicU8;

pub mod cpu_fabric;
mod memops;
mod page_table;

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
        if !($({ $expr })&&+) {
            return Err(MemoryFault::fault())
        }
    };
}

pub(crate) use ensure;

#[inline]
fn div_rem_page_size(vaddr: u64) -> (PageNumber, usize) {
    PageNumber::from_vaddr_with_offset(vaddr)
}

#[inline]
fn div_page_size_checked(vaddr: u64) -> Result<PageNumber, MemoryFault> {
    let (page, offset) = div_rem_page_size(vaddr);
    ensure!(offset == 0);
    Ok(page)
}

#[derive(Copy, Clone)]
pub(crate) struct PageTableAccess {
    start_page: PageNumber,
    page_count: NonZero<u64>,
}

impl PageTableAccess {
    fn get_pages(base: u64, size: u64) -> Result<Option<Self>, MemoryFault> {
        let start_page = div_page_size_checked(base)?;
        ensure!(size & PAGE_OFFSET_MASK_U64 == 0);
        let page_count = size >> PAGE_SHIFT;
        let (end_addr, overflow) = base.overflowing_add(size);
        if overflow {
            const { assert!(PAGE_SIZE > 1) }
            ensure!(end_addr == 0)
        }

        let Some(page_count) = NonZero::new(page_count) else {
            return Ok(None);
        };

        Ok(Some(Self {
            start_page,
            page_count,
        }))
    }

    unsafe fn new(start_page: PageNumber, page_count: NonZero<u64>) -> Self {
        debug_assert!(start_page.get().checked_add(page_count.get()).is_some());
        Self {
            start_page,
            page_count,
        }
    }

    pub fn iter(self) -> impl DoubleEndedIterator<Item = PageNumber> {
        let start = self.start_page.get();
        let end = unsafe { start.unchecked_add(self.page_count.get()) };
        (start..end).map(|page| unsafe { PageNumber::from_page_number_unchecked(page) })
    }
}

/// Page-mapped virtual memory view over host-backed storage.
///
/// `IoMMU` maps Armv9 virtual addresses onto page-aligned host memory.
/// Access permissions are checked per page, and invalid access returns
/// [`MemoryFault`].
///
/// The implementation permits concurrent access and models the armv9 memory model.
pub struct IoMMU<T: ?Sized + ICache> {
    identifier: IoMMUIdentifier,
    table: PageTable,
    fabric: CpuFabric<T>,
}

impl<T: ?Sized + ICache> IoMMU<T> {
    /// Creates an empty MMU with no mapped pages.
    ///
    /// All accesses fault until memory is mapped with [`IoMMU::map`].
    pub fn new(fabric: CpuFabric<T>) -> Self {
        Self {
            identifier: IoMMUIdentifier::unique_token(),
            table: PageTable::new(),
            fabric,
        }
    }

    #[inline]
    pub fn get_ident(&self) -> IoMMUIdentifierRef<'_> {
        self.identifier.get_ref()
    }

    fn change_token(&mut self) {
        self.identifier = IoMMUIdentifier::unique_token();
    }

    pub fn get_fabric(&self) -> &CpuFabric<T> {
        &self.fabric
    }

    unsafe fn modify_table(
        &mut self,
        transform: impl FnOnce(&mut PageTable, &T) -> Result<bool, MemoryFault>,
    ) -> Result<(), MemoryFault> {
        struct ChangeIdentifier<'a, T: ?Sized + ICache>(&'a mut IoMMU<T>);

        impl<T: ?Sized + ICache> Drop for ChangeIdentifier<'_, T> {
            fn drop(&mut self) {
                self.0.change_token()
            }
        }

        let identifier_change = ChangeIdentifier(self);
        match transform(
            &mut identifier_change.0.table,
            identifier_change.0.fabric.icache(),
        ) {
            Ok(true) => {
                drop(identifier_change);
                Ok(())
            }
            Ok(false) => {
                std::mem::forget(identifier_change);
                Ok(())
            }
            Err(err) => {
                std::mem::forget(identifier_change);
                Err(err)
            }
        }
    }

    // TODO add the ability to map and unmap in a single operation
    //      by simply using inclusive map ranges, OR by making you
    //      make size == page count directly
    unsafe fn map_region(
        &mut self,
        base: u64,
        size: u64,
        mem_prot: MemProt,
        base_ptr: Option<PagePointer>,
    ) -> Result<(), MemoryFault> {
        let Some(access) = PageTableAccess::get_pages(base, size)? else {
            return Ok(());
        };

        unsafe {
            self.modify_table(|pages, _icache| {
                pages.map(access, mem_prot, base_ptr)?;
                let modified = true;
                Ok(modified)
            })
        }
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
    /// - no mapping exists in the range `base..(base + size)`
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
    pub unsafe fn map_shared(
        &mut self,
        base: u64,
        size: u64,
        ptr: *mut u8,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        assert!(ptr.addr().is_multiple_of(PAGE_SIZE));
        assert!(!ptr.is_null());

        unsafe {
            self.map_region(
                base,
                size,
                protections,
                Some(PagePointer::new(NonNull::new_unchecked(ptr.cast()))),
            )
        }
    }

    pub fn map(&mut self, base: u64, size: u64, protections: MemProt) -> Result<(), MemoryFault> {
        unsafe { self.map_region(base, size, protections, None) }
    }

    pub fn unmap(&mut self, start: u64, size: u64) -> Result<(), MemoryFault> {
        let Some(access) = PageTableAccess::get_pages(start, size)? else {
            return Ok(());
        };

        unsafe {
            self.modify_table(|table, icache| {
                let mut modified = false;
                table.unmap(access, |page| {
                    modified = true;
                    if page.get_insn_dirty() {
                        icache.invalidate(page.ptr.page_ptr());
                        page.unset_insn_dirty();
                    }
                });

                Ok(modified)
            })
        }
    }

    pub fn protect(
        &mut self,
        start: u64,
        size: u64,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        let Some(access) = PageTableAccess::get_pages(start, size)? else {
            return Ok(());
        };

        unsafe {
            self.modify_table(|table, _icache| {
                let mut modified = false;
                table.modify(access, |_, entry| {
                    modified |= entry.prot() != protections;
                    entry.memprot(protections);
                })?;

                Ok(modified)
            })
        }
    }
}

/// A cache layer that sits between the CPU and the page table, translating
/// [`PageNumber`]s into live [`Page`] handles.
///
/// Implementors may satisfy lookups from a fast local structure (e.g. a TLB)
/// or fall through to the page table on every call. Either way, the returned
/// [`Page`] must be a valid view into the [`IoMMU`]'s backing memory for the
/// requested page number — callers rely on this to uphold memory safety across
/// the verify-then-access split in [`LookupCacheExt::access`].
///
/// # Safety
///
/// If `get_page` returns `Ok` for a given `page`, then every subsequent call
/// with the same `page` and the same [`IoMMU`] - without any intervening
/// mutation of that [`IoMMU`] - must also return `Ok`. Returning `Err` after
/// a prior `Ok` is **undefined behaviour**: the cache's consistency guarantee
/// is a precondition that callers are permitted to assume without checking.
///
/// TODO seal trait
pub unsafe trait LookupCache {
    /// Resolves `page` to a [`Page`] handle valid for the lifetime of `io_mmu`.
    ///
    /// See the trait-level safety docs for the consistency requirement that
    /// implementations must uphold.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault>;
}

pub struct NoCache;

unsafe impl LookupCache for NoCache {
    /// Bypasses any caching layer and faults directly to the page table.
    /// Useful when the overhead of a TLB lookup outweighs its benefit -
    /// e.g. single-access patterns where a cached entry would never be reused.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        io_mmu.table.get_page(page)
    }
}

unsafe impl LookupCache for Tlb {
    /// Attempts a TLB hit before falling back to the page table, amortising
    /// translation cost across repeated accesses to the same page.
    ///
    /// The `get_ident` check ensures we don't serve a stale entry after an
    /// address-space switch - the TLB is logically scoped to one MMU identity.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        let page = unsafe {
            self.lookup(page, io_mmu.get_ident(), |page_num| {
                io_mmu.table.get_page(page_num).ok()
            })
        };

        page.ok_or_else(MemoryFault::fault)
    }
}

unsafe impl<C: LookupCache> LookupCache for &mut C {
    #[inline(always)]
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        <C as LookupCache>::get_page(*self, io_mmu, page)
    }
}

pub(crate) trait LookupCacheExt: LookupCache {
    /// Resolves a virtual address into its page number, in-page byte offset,
    /// and a handle to the backing page - all in one step so callers don't
    /// have to re-derive the page number from the address themselves.
    #[inline(always)]
    fn lookup_addr<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        vaddr: u64,
    ) -> Result<(PageNumber, usize, Page<'a>), MemoryFault> {
        let (page_num, offset) = div_rem_page_size(vaddr);
        let page = self.get_page(io_mmu, page_num)?;
        Ok((page_num, offset, page))
    }

    /// Verifies that every page in `pages` satisfies the caller's predicate,
    /// then calls `access` on each one.
    ///
    /// The split into two phases is intentional: all-or-nothing semantics.
    /// If any page fails verification the access closure is never invoked,
    /// so callers don't need to reason about partially-applied side effects.
    ///
    /// No ordering guarantee is made on `access`; the iteration order is an
    /// internal implementation detail and may change.
    fn access<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        pages: PageTableAccess,
        mut verify: impl FnMut(PageNumber, Page<'a>) -> bool,
        mut access: impl FnMut(PageNumber, Page<'a>),
    ) -> Result<(), MemoryFault> {
        // Verify in reverse so the cache is hottest at the head of the range
        // when access begins - the forward access pass then walks into warm
        // entries rather than cold ones.
        for page_num in pages.iter().rev() {
            let page = self.get_page(io_mmu, page_num)?;
            ensure!(verify(page_num, page))
        }

        // Forward walk gives the hardware prefetcher a predictable ascending
        // stride, which matters more on the access loop than verify because
        // this is where real work happens.
        //
        // Unwrap is sound here: verify already confirmed every page is
        // reachable, so a fault now would mean the page table was mutated
        // underneath us - unrecoverable either way.
        for page_num in pages.iter() {
            let page = self.get_page(io_mmu, page_num).unwrap_or_else(|_| abort());
            access(page_num, page)
        }

        Ok(())
    }
}

impl<C: LookupCache> LookupCacheExt for C {}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline(always)]
    pub fn get_page(
        &self,
        mut cache: impl LookupCache,
        page: PageNumber,
    ) -> Result<Page<'_>, MemoryFault> {
        cache.get_page(self, page)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn mem_access(
        &self,
        mut cache: impl LookupCache,
        vaddr: u64,
        len: usize,
        mem_ptr: *mut u8,
        required_prot: impl Into<Option<MemProt>> + Copy,
        mut per_page_commit: impl FnMut(PageNumber, Page<'_>),
        mut step: impl FnMut(*const AtomicU8, *mut u8),
    ) -> Result<(), MemoryFault> {
        let len = u64::try_from(len).map_err(|_| MemoryFault::fault())?;

        let Some(len) = NonZero::new(len) else {
            return Ok(());
        };

        const { assert!(PAGE_SIZE >= 2) }

        let (last_vaddr_maybe, overflowed) = vaddr.overflowing_add(len.get());
        if overflowed {
            ensure!(last_vaddr_maybe == 0)
        }

        let last_addr_inclusive = last_vaddr_maybe.wrapping_sub(1);
        let (start_page, start_offset) = div_rem_page_size(vaddr);
        let (last_page, end_offset) = div_rem_page_size(last_addr_inclusive);

        let page_count = unsafe {
            NonZero::new_unchecked(
                last_page
                    .get()
                    .unchecked_sub(start_page.get())
                    .unchecked_add(1),
            )
        };

        let access = unsafe { PageTableAccess::new(start_page, page_count) };

        let last_page_end = unsafe { end_offset.unchecked_add(1) };

        cache.access(
            self,
            access,
            move |_, page| match required_prot.into() {
                Some(required) => page.ptr.prot().contains(required),
                None => true,
            },
            |page_num, page| unsafe {
                // use select unpredictable, as all loads and stores
                // of all different sizes, reach this function
                // the cpu can't predict the branches here reliably
                // since one loop might, load big, then small, then big....

                let page_start =
                    std::hint::select_unpredictable(page_num == start_page, start_offset, 0);

                let page_end = std::hint::select_unpredictable(
                    page_num == last_page,
                    last_page_end,
                    PAGE_SIZE,
                );

                let full_pages_before = page_num.get().unchecked_sub(start_page.get());
                let page_base_offset =
                    u64_to_usize(full_pages_before.unchecked_mul(PAGE_SIZE_U64)).unwrap_unchecked();

                let dst_start = page_base_offset
                    .unchecked_add(page_start)
                    .unchecked_sub(start_offset);

                let src = { page.ptr.page_ptr().byte_add(page_start).as_ptr() };

                let dst = mem_ptr.add(dst_start).cast::<u8>();
                for i in 0..page_end.unchecked_sub(page_start) {
                    let vm_ptr = src.add(i);
                    let mem_ptr = dst.add(i);
                    step(vm_ptr, mem_ptr)
                }

                // only run per page after running the access loop
                // this is so that when storing; instructions dirty
                // only gets set **after** the bytes have changed
                // so that no situation happens where the insn dirty flag
                // can be consumed **before** the page actually gets dirtied
                per_page_commit(page_num, page);
            },
        )?;

        Ok(())
    }

    fn inner_load_with_prot<'a>(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>],
        required_prot: impl Into<Option<MemProt>> + Copy,
    ) -> Result<&'a mut [u8], MemoryFault> {
        // get the length **before** getting the pointer
        // so that mem isn't reborrowed
        let len = mem.len();
        let ptr = mem.as_mut_ptr().cast::<u8>();
        unsafe {
            self.mem_access(
                cache,
                vaddr,
                len,
                ptr,
                required_prot,
                |_, _| {},
                |vm_ptr, mem_ptr| {
                    let byte = memops::load_byte(vm_ptr);
                    std::ptr::write(mem_ptr, byte);
                },
            )?
        }

        Ok(unsafe { mem.assume_init_mut() })
    }

    /// Loads a byte slice from virtual memory into `mem`.
    ///
    /// The load may span multiple pages. Every covered page must be mapped and have
    /// read permission.
    ///
    /// Concurrent stores are allowed. The returned bytes may reflect a mixture of
    /// values from racing stores, and this only guarentees single-copy atomicity
    /// at the byte level
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, unreadable, or if address
    /// arithmetic overflows.
    pub fn load_into_uninit<'a>(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>],
    ) -> Result<&'a mut [u8], MemoryFault> {
        self.inner_load_with_prot(cache, vaddr, mem, MemProt::READ)
    }

    /// Loads a byte slice from virtual memory into `mem`.
    ///
    /// The load may span multiple pages. Every covered page must be mapped.
    ///
    /// Concurrent stores are allowed. The returned bytes may reflect a mixture of
    /// values from racing stores, and this only guarentees single-copy atomicity
    /// at the byte level
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, or if address arithmetic overflows.
    pub fn load_into_uninit_force<'a>(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>],
    ) -> Result<&'a mut [u8], MemoryFault> {
        self.inner_load_with_prot(cache, vaddr, mem, None)
    }

    fn load_wrapper<F, C>(
        &self,
        cache: C,
        vaddr: u64,
        mem: &mut [u8],
        wrapper: F,
    ) -> Result<(), MemoryFault>
    where
        C: LookupCache,
        F: for<'a> FnOnce(
            &Self,
            C,
            u64,
            &'a mut [MaybeUninit<u8>],
        ) -> Result<&'a mut [u8], MemoryFault>,
    {
        let mem_ptr = mem as *mut [u8];
        let maybe_uninit_ptr = mem_ptr as *mut [MaybeUninit<u8>];
        // Safety: `Self::load_into_uninit` never deinitializes
        let mem_as_uninit = unsafe { &mut *maybe_uninit_ptr };
        let slice = wrapper(self, cache, vaddr, mem_as_uninit)?;
        debug_assert!(std::ptr::eq::<[u8]>(slice as *mut [u8], mem_ptr));
        Ok(())
    }

    /// See [`Self::load_into_uninit`]
    pub fn load(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &mut [u8],
    ) -> Result<(), MemoryFault> {
        self.load_wrapper(cache, vaddr, mem, Self::load_into_uninit)
    }

    /// See [`Self::load_into_uninit_force`]
    pub fn load_force(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &mut [u8],
    ) -> Result<(), MemoryFault> {
        self.load_wrapper(cache, vaddr, mem, Self::load_into_uninit_force)
    }

    fn inner_store_with_prot(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &[u8],
        required_prot: impl Into<Option<MemProt>> + Copy,
    ) -> Result<(), MemoryFault> {
        unsafe {
            self.mem_access(
                cache,
                vaddr,
                mem.len(),
                mem.as_ptr().cast_mut(),
                required_prot,
                |_page_num, page| page.set_insn_dirty(),
                |vm_ptr, mem_ptr| {
                    let byte = std::ptr::read(mem_ptr);
                    memops::store_byte(vm_ptr, byte)
                },
            )
        }
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
    #[inline(always)]
    pub fn store(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &[u8],
    ) -> Result<(), MemoryFault> {
        self.inner_store_with_prot(cache, vaddr, mem, MemProt::WRITE)
    }

    /// Stores a byte slice into virtual memory.
    ///
    /// The store may span multiple pages. Every covered page must be mapped.
    ///
    /// Concurrent loads and stores are allowed. Other threads may observe the write
    /// as a sequence of byte operations
    ///
    /// Returns [`MemoryFault`] if the range is unmapped, or if address
    /// arithmetic overflows.
    #[inline(always)]
    pub fn store_force(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        mem: &[u8],
    ) -> Result<(), MemoryFault> {
        self.inner_store_with_prot(cache, vaddr, mem, None)
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline]
    pub(crate) fn single_page_aligned_access<const ACCESS_SIZE: u8>(
        &self,
        mut cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<(Page<'_>, usize), MemoryFault> {
        const {
            assert!(ACCESS_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_multiple_of(ACCESS_SIZE as usize));
        }

        ensure!(vaddr.is_multiple_of(ACCESS_SIZE as u64));

        let (_page_num, offset, page) = cache.lookup_addr(self, vaddr)?;
        unsafe {
            // SAFETY:
            //
            // div_rem_page_size(vaddr) == (vaddr / PAGE_SIZE, vaddr % PAGE_SIZE),
            // therefore:
            //
            //     offset < PAGE_SIZE
            //
            // We also know:
            //
            //     ACCESS_SIZE | vaddr
            //     ACCESS_SIZE | PAGE_SIZE
            //
            // Therefore:
            //
            //     ACCESS_SIZE | (vaddr % PAGE_SIZE)
            //
            // so `offset` is itself an ACCESS_SIZE-aligned offset into the page.
            //
            // Since:
            //
            //     offset < PAGE_SIZE
            //
            // and `offset` is an ACCESS_SIZE-multiple, the largest possible offset
            // is:
            //
            //     PAGE_SIZE - ACCESS_SIZE
            //
            // Therefore:
            //
            //     offset <= PAGE_SIZE - ACCESS_SIZE
            //
            // which is equivalent to:
            //
            //     offset < PAGE_SIZE - ACCESS_SIZE + 1
            //
            // Thus an ACCESS_SIZE-byte access starting at `offset` cannot cross the
            // page boundary.
            let max_offset_exclusive =
                const { PAGE_SIZE.strict_sub(ACCESS_SIZE as usize).strict_add(1) };

            std::hint::assert_unchecked(offset < max_offset_exclusive)
        };

        Ok((page, offset))
    }

    pub(crate) fn single_page_access(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<(Page<'_>, usize), MemoryFault> {
        const ALIGN: u8 = 1;
        self.single_page_aligned_access::<ALIGN>(cache, vaddr)
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline(always)]
    pub fn load_byte(&self, cache: impl LookupCache, vaddr: u64) -> Result<u8, MemoryFault> {
        let (page, offset) = self.single_page_access(cache, vaddr)?;
        ensure!(page.ptr.prot().contains(MemProt::READ));
        let byte = unsafe { memops::load_byte(page.ptr.page_ptr().byte_add(offset).as_ptr()) };
        Ok(byte)
    }

    #[inline(always)]
    pub fn store_byte(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        value: u8,
    ) -> Result<(), MemoryFault> {
        let (page, offset) = self.single_page_access(cache, vaddr)?;
        ensure!(page.ptr.prot().contains(MemProt::WRITE));
        unsafe { memops::store_byte(page.ptr.page_ptr().byte_add(offset).as_ptr(), value) }
        page.set_insn_dirty();
        Ok(())
    }
}

// TODO re implement scalar acess propperly
//      and add a direct load_with_tlb, store_with_tlb
#[derive(Copy, Clone)]
struct SecondPage<'a> {
    page: Page<'a>,
    overflow_amount: NonZero<usize>,
}

struct SmallAccess<'a> {
    base_page: Page<'a>,
    base_page_offset: usize,
    second_page: Option<SecondPage<'a>>,
}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline(always)]
    fn static_small_multibyte_acces<const BYTES: usize>(
        &self,
        mut cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<SmallAccess<'_>, MemoryFault> {
        const {
            assert!(BYTES > 0);
            assert!(BYTES <= PAGE_SIZE * 2);
        }

        let (base_page_num, base_page_offset) = div_rem_page_size(vaddr);
        let base_page = cache.get_page(self, base_page_num)?;

        let bytes_left_in_base_page = unsafe { PAGE_SIZE.unchecked_sub(base_page_offset) };

        let second_page = BYTES
            .checked_sub(bytes_left_in_base_page)
            .and_then(NonZero::new)
            .map(|overflow_amount| {
                let second_page_num = base_page_num.inc().ok_or_else(MemoryFault::fault)?;

                let page = cache.get_page(self, second_page_num)?;

                Ok(SecondPage {
                    page,
                    overflow_amount,
                })
            })
            .transpose()?;

        Ok(SmallAccess {
            base_page,
            base_page_offset,
            second_page,
        })
    }
}

macro_rules! emit_multi_word_load_store {
    {
        @folded
        $([
            bits: $bits: tt,
            ty: $ty: ty,
            load_name: $load_name: ident,
            aligned_load_name: $memops_aligned_load_name: ident,
            store_name: $store_name: ident $(,)?
        ])+
    } => {
        impl<T: ?Sized + ICache> IoMMU<T> {$(
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
            pub fn $load_name(&self, cache: impl LookupCache, vaddr: u64) -> Result<$ty, MemoryFault> {
                let access = self.static_small_multibyte_acces::<{ size_of::<$ty>() }>(cache, vaddr)?;

                let value = match access.second_page {
                    // SAFETY:
                    // `static_small_multibyte_acces` returned `None` for `second_page`, so
                    // this access is fully contained in `base_page`.
                    //
                    // Therefore:
                    //
                    //   base_page_offset + size_of::<u$bits>() <= PAGE_SIZE
                    //
                    // which satisfies the safety requirement of `Page::load$bits`.
                    None => unsafe {
                        ensure!(access.base_page.ptr.prot().contains(MemProt::READ));
                        let ptr = access.base_page.ptr.page_ptr().byte_add(access.base_page_offset);
                        memops::$load_name(ptr.as_ptr())
                    },

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

                        ensure!(
                            second_page.page.ptr.prot().contains(MemProt::READ),
                            access.base_page.ptr.prot().contains(MemProt::READ),
                        );

                        let hi_page_ptr = second_page.page.ptr.page_ptr();
                        let lo_page_ptr = access.base_page.ptr.page_ptr();


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
                        let hi_ptr = hi_page_ptr.as_non_null_ptr().as_ptr();

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


                        let hi = memops::$memops_aligned_load_name(hi_ptr);
                        let lo = memops::$memops_aligned_load_name(lo_ptr);

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
                        let bit_offset = u32::try_from(
                            second_page.overflow_amount.get().unchecked_mul(8)
                        )
                        .unwrap_unchecked();

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
                        hi.unchecked_shl(hi_shift) | lo.unchecked_shr(lo_shift)
                    }
                };

                Ok(value)
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
            pub fn $store_name(&self, cache: impl LookupCache, vaddr: u64, value: $ty) -> Result<(), MemoryFault> {
                let access = self.static_small_multibyte_acces::<{ size_of::<$ty>() }>(cache, vaddr)?;

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
                        let page = access.base_page;
                        ensure!(page.ptr.prot().contains(MemProt::WRITE));
                        let ptr = page.ptr.page_ptr().byte_add(access.base_page_offset);
                        memops::$store_name(ptr.as_ptr(), value);
                        page.set_insn_dirty();
                    },

                    Some(second_page) => unsafe {
                        // Note: we can't load 2 words and combine them like the load case
                        //       since that would alter/mess with the atomicity of the bytes
                        //       next to the value
                        let bytes = value.to_le_bytes();

                        let hi_page = second_page.page;
                        let lo_page = access.base_page;


                        ensure!(
                            hi_page.ptr.prot().contains(MemProt::WRITE),
                            lo_page.ptr.prot().contains(MemProt::WRITE),
                        );

                        let hi_page_ptr = hi_page.ptr.page_ptr().as_non_null_ptr();
                        let lo_page_ptr = lo_page.ptr.page_ptr().as_non_null_ptr();

                        let overflow = usize::from(second_page.overflow_amount.get());


                        let mut active_ptr = hi_page_ptr.byte_add(overflow).as_ptr();
                        let mut i = bytes.len();
                        for _ in 0..overflow {
                            active_ptr = active_ptr.byte_sub(1);
                            i = i.unchecked_sub(1);
                            let byte = *bytes.get_unchecked(i);
                            memops::store_byte(active_ptr, byte)
                        }

                        active_ptr = lo_page_ptr
                            .byte_add(const { PAGE_SIZE.strict_sub(1) })
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

                        lo_page.set_insn_dirty();
                        hi_page.set_insn_dirty();
                    }
                }

                Ok(())
            }
        )+}
    };

    ($($bits: tt),+ $(,)?) => {
        pastey::paste! {
            emit_multi_word_load_store! {
                @folded
                $([
                    bits: $bits,
                    ty: [<u $bits>],
                    load_name: [<load $bits _le>],
                    aligned_load_name: [<load $bits _le_aligned>],
                    store_name: [<store $bits _le>]
                ])+
            }
        }
    };
}

emit_multi_word_load_store! { 64, 32, 16 }

const AARCH64_WORD_ALIGN: u8 = 4;

impl<T: ?Sized + ICache> IoMMU<T> {
    pub fn fetch_aarch64_full(
        &self,
        mut cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<(PagePointer, u32), MemoryFault> {
        ensure!(vaddr.is_multiple_of(AARCH64_WORD_ALIGN as u64));

        // note since vaddr is aligned there is no need to check overflow or alignment;
        // for more on why this is always true look at `single_page_aligned_access`
        let (_page_number, offset, page) = cache.lookup_addr(self, vaddr)?;

        ensure!(page.ptr.prot().contains(MemProt::EXECUTE));
        unsafe {
            let page_ptr = page.ptr.page_ptr();
            let word_ptr = page_ptr.byte_add(offset);
            let word = memops::load32_le_aligned(word_ptr.as_ptr());
            Ok((page_ptr, word))
        }
    }

    pub fn fetch_aarch64(&self, cache: impl LookupCache, vaddr: u64) -> Result<u32, MemoryFault> {
        self.fetch_aarch64_full(cache, vaddr).map(|(_, word)| word)
    }

    pub fn flush_dirty_pages(&self) {
        for (_page_num, page) in self.table.pages() {
            if page.get_insn_dirty() {
                self.fabric.icache().invalidate(page.ptr.page_ptr());
                page.unset_insn_dirty();
            }
        }
    }
}

impl<T: Sized + ICache> AsFFI for IoMMU<T> {
    type Interface<'a>
        = (IoMMUIdentifierRef<'a>, ManuallyDrop<IoMMU<dyn ICache + 'a>>)
    where
        T: 'a;

    fn as_ffi<'a>(&'a self) -> Self::Interface<'a>
    where
        Self: 'a,
    {
        let ident_ref = self.get_ident();
        let guard = AbortGuard(());
        let identifier = unsafe { ident_ref.copy_identifier() };
        let table = unsafe { std::ptr::read(&self.table) };
        let fabric = unsafe { std::ptr::read(&self.fabric) };
        let fabric = fabric.into_dyn();
        let new = ManuallyDrop::new(IoMMU {
            identifier,
            table,
            fabric,
        });

        guard.disarm();
        (ident_ref, new)
    }
}

impl<'a> AsFFI for IoMMU<dyn ICache + 'a> {
    type Interface<'b>
        = (IoMMUIdentifierRef<'b>, ManuallyDrop<IoMMU<dyn ICache + 'b>>)
    where
        Self: 'b;

    fn as_ffi<'b>(&'b self) -> Self::Interface<'b>
    where
        Self: 'b,
    {
        let ident_ref = self.get_ident();
        let guard = AbortGuard(());
        let this: IoMMU<dyn ICache + 'a> = unsafe { std::ptr::read(self) };
        let new: IoMMU<dyn ICache + 'b> = this;
        let borrowed = ManuallyDrop::new(new);
        guard.disarm();
        (ident_ref, borrowed)
    }
}
