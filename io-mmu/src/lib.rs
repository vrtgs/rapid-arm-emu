//! Software IO-MMU for the emulator: a page-granular virtual address space
//! mapped onto host memory.
//!
//! The central type is [`IoMMU`], which owns a page table translating Armv9
//! guest virtual addresses to host pages. Pages can be backed by anonymous
//! memory, externally provided host memory, or demand-paged
//! [`MemoryObject`](MemoryObject)s. All guest memory accesses
//! go through the audited atomic primitives in [`memops`], and translation
//! can be speeded up by a [`LookupCache`] such as a [`TLB`].

use crate::cpu_fabric::CpuFabric;
use crate::fault::{MemoryFault, ensure};
use crate::icache::ICache;
use crate::lookup_cache::{LookupCache, LookupCacheExt};
use crate::memory_object::MemoryObject;
use crate::page_table::{MapRegion, MemMapFlags, PageTable};
use emu_abi::abort::AbortGuard;
use emu_abi::convert::{u64_to_usize, usize_to_u64};
use emu_abi::memory::{
    IoMMUIdentifier, IoMMUIdentifierRef, MemFlags, MemProt, PAGE_OFFSET_MASK_U64, PAGE_SHIFT,
    PAGE_SIZE, PAGE_SIZE_U64, Page, PageNumber, PagePointer,
};
use std::convert::Infallible;
use std::hint::cold_path;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::num::NonZero;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;

pub mod cpu_fabric;
pub mod fault;
pub mod icache;
pub mod lookup_cache;
pub mod memops;
pub mod memory_object;
mod page_table;

#[inline]
fn div_rem_page_size(vaddr: u64) -> (PageNumber, usize) {
    PageNumber::from_vaddr_with_offset(vaddr)
}

#[inline]
fn div_page_size_checked(vaddr: u64) -> Result<PageNumber, MemoryFault> {
    let (page, offset) = div_rem_page_size(vaddr);
    ensure!(vaddr: vaddr, offset == 0);
    Ok(page)
}

#[derive(Copy, Clone)]
pub(crate) struct PageTableAccess {
    start_page: PageNumber,
    page_count: NonZero<u64>,
}

impl PageTableAccess {
    fn get_pages(base_vaddr: u64, size: u64) -> Result<Option<Self>, MemoryFault> {
        let start_page = div_page_size_checked(base_vaddr)?;
        ensure!(vaddr: base_vaddr, size & PAGE_OFFSET_MASK_U64 == 0);
        let page_count = size >> PAGE_SHIFT;
        let (end_addr, overflow) = base_vaddr.overflowing_add(size);
        if overflow {
            const { assert!(PAGE_SIZE > 1) }
            ensure!(vaddr: u64::MAX, end_addr == 0)
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

    pub(crate) fn iter(self) -> impl DoubleEndedIterator<Item = PageNumber> {
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
    /// Creates a new, empty `IoMMU` attached to `fabric`.
    ///
    /// The address space starts with no mappings; every access faults until
    /// regions are mapped with [`map`](Self::map), [`map_extern`](Self::map_extern),
    /// or [`map_device`](Self::map_device).
    pub fn new(fabric: CpuFabric<T>) -> Self {
        Self {
            identifier: IoMMUIdentifier::unique_token(),
            table: PageTable::new(),
            fabric,
        }
    }

    /// Returns the current identity token of this address space.
    ///
    /// The identifier changes whenever the page table is mutated, so caches
    /// such as a TLB can use it to detect that their entries are stale.
    #[inline]
    pub fn get_ident(&self) -> IoMMUIdentifierRef<'_> {
        self.identifier.get_ref()
    }

    fn change_ident(&mut self) {
        self.identifier = IoMMUIdentifier::unique_token();
    }

    /// Returns the [`CpuFabric`] this MMU is attached to.
    pub fn get_fabric(&self) -> &CpuFabric<T> {
        &self.fabric
    }

    unsafe fn modify_table_full<U, E>(
        &mut self,
        transform: impl FnOnce(&mut PageTable, &CpuFabric<T>) -> Result<(bool, U), E>,
    ) -> Result<U, E> {
        struct ChangeIdentifier<'a, T: ?Sized + ICache>(&'a mut IoMMU<T>);

        impl<T: ?Sized + ICache> Drop for ChangeIdentifier<'_, T> {
            fn drop(&mut self) {
                self.0.change_ident()
            }
        }

        let identifier_change = ChangeIdentifier(self);
        match transform(&mut identifier_change.0.table, &identifier_change.0.fabric) {
            Ok((true, res)) => {
                drop(identifier_change);
                Ok(res)
            }
            Ok((false, res)) => {
                std::mem::forget(identifier_change);
                Ok(res)
            }
            Err(err) => {
                std::mem::forget(identifier_change);
                Err(err)
            }
        }
    }

    unsafe fn modify_table<E>(
        &mut self,
        transform: impl FnOnce(&mut PageTable, &CpuFabric<T>) -> Result<bool, E>,
    ) -> Result<(), E> {
        unsafe {
            self.modify_table_full(move |table, fabric| {
                let bool = transform(table, fabric)?;
                Ok((bool, ()))
            })
        }
    }

    // TODO(low priority) add the ability to map and unmap in a single operation
    unsafe fn map_region(
        &mut self,
        base: u64,
        size: u64,
        mem_prot: MemProt,
        region: MapRegion,
    ) -> Result<(), MemoryFault> {
        let Some(access) = PageTableAccess::get_pages(base, size)? else {
            return Ok(());
        };

        unsafe {
            self.modify_table(|pages, icache| {
                pages.map(icache, access, mem_prot, region)?;
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
    /// - No mapping exists in the range `base..(base + size)`
    ///
    /// Permissions are applied to every mapped page in the region.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr..(ptr + size)` is valid for the lifetime of this MMU mapping,
    ///
    /// - The pointed-to memory is initialized
    ///
    /// - The pointed-to memory is valid for both reads at the host-memory
    ///   level, regardless of the guest permissions applied by `protections`
    ///   and it also must have write permissions set if `readonly = false`
    ///
    /// - If `readonly = false`, the backing memory must not be accessed directly
    ///   while this mapping is alive, except by other MMUs that use the same `CpuFabric`.
    ///
    /// - If `readonly = true`, then the memory pointed to by the pointer may be read from
    ///   while this mapping is alive, but it must not be written to.
    pub unsafe fn map_extern(
        &mut self,
        base: u64,
        size: u64,
        ptr: *mut u8,
        readonly: bool,
        shared: bool,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        assert!(ptr.addr().is_multiple_of(PAGE_SIZE));
        assert!(!ptr.is_null());

        let base_ptr = unsafe { PagePointer::new(NonNull::new_unchecked(ptr.cast())) };

        let region = match (readonly, shared) {
            (false, false) => MapRegion::Extern {
                flags: MemMapFlags::Private,
                base_ptr,
            },
            (true, false) => MapRegion::Extern {
                flags: MemMapFlags::Cow,
                base_ptr,
            },
            (false, true) => MapRegion::Extern {
                flags: MemMapFlags::Shared,
                base_ptr,
            },
            (true, true) => panic!("there is no such thing as shared readonly mappings"),
        };

        unsafe { self.map_region(base, size, protections, region) }
    }

    /// Maps anonymous zero-initialized memory into the MMU page table.
    ///
    /// `base` is the starting virtual address and `size` is the mapping size
    /// in bytes; both must be page-aligned and `base + size` must not
    /// overflow. No mapping may exist in `base..(base + size)`.
    /// `protections` is applied to every mapped page.
    ///
    /// If `shared` is `true`, the pages are shared with forked address
    /// spaces. If `lazy` is `true` (and `shared` is `false`), pages start as
    /// copy-on-write views of the zero page and only get their own backing
    /// storage when first written.
    ///
    /// Returns [`MemoryFault`] on misalignment, overflow, or an already
    /// mapped page in the range.
    pub fn map(
        &mut self,
        base: u64,
        size: u64,
        lazy: bool,
        shared: bool,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        let region = match (lazy, shared) {
            (true | false, true) => MapRegion::Anon(MemMapFlags::Shared),
            (false, false) => MapRegion::Anon(MemMapFlags::Private),
            (true, false) => MapRegion::Anon(MemMapFlags::Cow),
        };

        unsafe { self.map_region(base, size, protections, region) }
    }

    /// Maps a demand-paged [`MemoryObject`] into the MMU page table.
    ///
    /// Same as [`map_device`](Self::map_device), but takes an already
    /// type-erased `Arc<dyn MemoryObject>`.
    ///
    /// `base` and `size` must be page-aligned, `base + size` must not
    /// overflow, and no mapping may already exist in the range. Page contents
    /// are faulted in from `device` on first access and dirty pages are
    /// written back to it. If `shared` is `true`, the object's pages are
    /// shared with forked address spaces.
    ///
    /// Returns [`MemoryFault`] on misalignment, overflow, or an already
    /// mapped page in the range.
    pub fn map_device_dyn(
        &mut self,
        base: u64,
        size: u64,
        device: Arc<dyn MemoryObject>,
        shared: bool,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        let region = MapRegion::Object {
            shared,
            object: device,
        };

        unsafe { self.map_region(base, size, protections, region) }
    }

    /// Maps a demand-paged [`MemoryObject`] into the MMU page table.
    ///
    /// Convenience wrapper around [`map_device_dyn`](Self::map_device_dyn)
    /// that wraps `device` in an [`Arc`]; see that method for the mapping
    /// requirements and semantics.
    pub fn map_device(
        &mut self,
        base: u64,
        size: u64,
        device: impl MemoryObject,
        shared: bool,
        protections: MemProt,
    ) -> Result<(), MemoryFault> {
        self.map_device_dyn(base, size, Arc::new(device), shared, protections)
    }

    /// Unmaps every mapped page in `start..(start + size)`.
    ///
    /// `start` and `size` must be page-aligned and `start + size` must not
    /// overflow. Pages in the range that are not mapped are skipped; dirty
    /// pages are dropped without being flushed back to their backing
    /// [`MemoryObject`] (use [`flush`](Self::flush) first if that matters).
    ///
    /// Returns [`MemoryFault`] on misalignment or overflow.
    pub fn unmap(&mut self, start: u64, size: u64) -> Result<(), MemoryFault> {
        let Some(access) = PageTableAccess::get_pages(start, size)? else {
            return Ok(());
        };

        unsafe {
            self.modify_table(|table, _fabric| {
                let mut modified = false;

                // no need to flush
                table.unmap(access, |_| modified = true);

                Ok(modified)
            })
        }
    }

    /// Sets the guest protection of every page in `start..(start + size)` to
    /// `protections`.
    ///
    /// `start` and `size` must be page-aligned and `start + size` must not
    /// overflow. Every page in the range must currently be mapped; if any
    /// page is unmapped, no page is modified.
    ///
    /// Returns [`MemoryFault`] on misalignment, overflow, or an unmapped page
    /// in the range.
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
                    entry.protect(protections);
                })?;

                Ok(modified)
            })
        }
    }

    /// Forks this address space, returning a new `IoMMU` with a copy of the
    /// current page table.
    ///
    /// Shared mappings keep referring to the same backing pages in both
    /// address spaces. Private mappings become copy-on-write: both sides see
    /// the same contents until one of them writes, at which point the writer
    /// gets its own copy. The new MMU is attached to the same [`CpuFabric`] as `self`.
    pub fn fork(&mut self) -> Self {
        let Ok(table) = unsafe {
            self.modify_table_full(|table, _| {
                let new_table = table.fork();
                Ok::<_, Infallible>((new_table.is_empty(), new_table))
            })
        };

        Self {
            table,
            identifier: self.identifier.clone(),
            fabric: self.fabric.clone(),
        }
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    /// Resolves `page` to a live [`Page`] handle through the given
    /// [`LookupCache`].
    ///
    /// Returns [`MemoryFault`] if the page is not mapped.
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
        no_cow: bool,
        mut per_page_commit: impl FnMut(PageNumber, Page<'_>),
        copy: impl Fn(*const AtomicU8, *mut u8, usize),
    ) -> Result<(), MemoryFault> {
        let len = u64::try_from(len).map_err(|_| {
            // len > u64::MAX
            MemoryFault::general_protection(u64::MAX)
        })?;

        let Some(len) = NonZero::new(len) else {
            return Ok(());
        };

        const { assert!(PAGE_SIZE >= 2) }

        let (last_vaddr_maybe, overflowed) = vaddr.overflowing_add(len.get());
        if overflowed {
            ensure!(vaddr: u64::MAX, last_vaddr_maybe == 0)
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
            move |_, page| {
                if no_cow && page.ptr.flags().contains_any(MemFlags::COW) {
                    return false;
                }

                match required_prot.into() {
                    Some(required) => page.ptr.flags().contains_any(required.into()),
                    None => true,
                }
            },
            |page_num, page| unsafe {
                // use select unpredictable, as all loads and stores
                // of all different sizes, reach this function
                // the cpu can't predict the branches here reliably
                // since one loop might load big, then small, then big...

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

                let vm_ptr = { page.ptr.page_ptr().byte_add(page_start).as_ptr() };
                let host_ptr = mem_ptr.add(dst_start).cast::<u8>();
                let count = page_end.unchecked_sub(page_start);
                copy(vm_ptr, host_ptr, count);

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
        // you can always read from a cow page
        let no_cow = false;
        unsafe {
            self.mem_access(
                cache,
                vaddr,
                len,
                ptr,
                required_prot,
                no_cow,
                |_, _| {},
                |vm_ptr, mem_ptr, count| {
                    memops::copy_nonoverlapping_vm_to_host(vm_ptr, mem_ptr, count);
                },
            )?
        }

        Ok(unsafe { mem.assume_init_mut() })
    }

    /// Loads a byte slice from virtual memory into `mem`.
    ///
    /// The load may span multiple pages. Every covered page must be mapped and have
    /// `READ` permission.
    ///
    /// Concurrent stores are allowed. The returned bytes may reflect a mixture of
    /// values from racing stores, and this only guarantees single-copy atomicity
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
    /// values from racing stores, and this only guarantees single-copy atomicity
    /// at the byte level
    ///
    /// On success, returns `mem` as an initialized `&mut [u8]`.
    ///
    /// Returns [`MemoryFault`] if the range is unmapped or if address arithmetic overflows.
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
        // you can't store to a cow page that can lead to big bad UB
        let no_cow = true;
        unsafe {
            self.mem_access(
                cache,
                vaddr,
                mem.len(),
                mem.as_ptr().cast_mut(),
                required_prot,
                no_cow,
                |_page_num, page| page.set_dirty(),
                |vm_ptr, mem_ptr, count| {
                    memops::copy_nonoverlapping_host_to_vm(mem_ptr, vm_ptr, count);
                },
            )
        }
    }

    /// Stores a byte slice into virtual memory.
    ///
    /// The store may span multiple pages. Every covered page must be mapped and have
    /// `WRITE` permission.
    ///
    /// Concurrent loads and stores are allowed. Other threads may observe the write
    /// as a sequence of byte operations
    ///
    /// Returns [`MemoryFault`] if the range is unmapped or if address
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
    /// Returns [`MemoryFault`] if the range is unmapped or if address
    /// arithmetic overflows.
    ///
    /// Note: writes will still fail if you try to write into a `COW` page
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
    /// Resolves `vaddr` to the page that contains it, together with the
    /// in-page byte offset, for a naturally aligned scalar access of
    /// `ACCESS_SIZE` bytes.
    ///
    /// `ACCESS_SIZE` is the width of the access in bytes (e.g. `4` for a
    /// 32-bit load/store) and must be a power of two smaller than
    /// `PAGE_SIZE`, with `PAGE_SIZE` an exact multiple of it; these are
    /// compile-time invariants of the type system, not runtime conditions.
    ///
    /// `vaddr` must be aligned to `ACCESS_SIZE`; misalignment is reported
    /// as a [`MemoryFault`], not a panic.
    ///
    /// # Guarantee
    ///
    /// Because `vaddr` is `ACCESS_SIZE`-aligned, the returned offset can
    /// never be closer than `ACCESS_SIZE` bytes to the end of the page.
    /// Callers may therefore read or write `ACCESS_SIZE` bytes starting at
    /// the returned offset without any page-boundary check — the access is
    /// statically guaranteed to stay within the single returned page.
    ///
    /// This function does **not** check `MemProt`/`MemFlags` permissions
    /// (`READ`/`WRITE`/`EXECUTE`); callers must check the flags on the
    /// returned [`Page`] themselves before accessing it.
    ///
    /// Returns [`MemoryFault`] if `vaddr` is misaligned or the containing
    /// page is unmapped.
    #[inline(always)]
    pub fn resolve_aligned_scalar_access<const ACCESS_SIZE: usize>(
        &self,
        mut cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<(Page<'_>, usize), MemoryFault> {
        const {
            assert!(
                ACCESS_SIZE.is_power_of_two(),
                "ACCESS_SIZE must be a power of two"
            );
            assert!(
                ACCESS_SIZE < PAGE_SIZE,
                "ACCESS_SIZE must be smaller than PAGE_SIZE this function only \
                 handles accesses that fit within a single page"
            );

            // given that ACCESS_SIZE is a power of 2, and PAGE_SIZE is also a power of 2
            // then this is more or less a sanity check
            assert!(PAGE_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_multiple_of(ACCESS_SIZE));
        }

        ensure!(
            vaddr: vaddr,
            vaddr.is_multiple_of(const { usize_to_u64(ACCESS_SIZE).unwrap() })
        );

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
            //      - `offset <= PAGE_SIZE - ACCESS_SIZE`
            // which is equivalent to:
            //      - `offset < PAGE_SIZE - ACCESS_SIZE + 1`
            // Thus access of size `ACCESS_SIZE` starting at `offset`
            // cannot cross the page boundary.
            let max_offset_exclusive = const { PAGE_SIZE.strict_sub(ACCESS_SIZE).strict_add(1) };

            std::hint::assert_unchecked(offset < max_offset_exclusive)
        };

        Ok((page, offset))
    }

    /// Resolves `vaddr` to the page that contains it, together with the
    /// in-page byte offset, for a single-byte access.
    ///
    /// Equivalent to
    /// [`resolve_aligned_scalar_access::<1>`](Self::resolve_aligned_scalar_access),
    /// specialized for byte accesses. A width-1 access has no alignment
    /// requirement (every offset is trivially "1-byte aligned"), so unlike
    /// the general form this never faults on alignment grounds - only on an
    /// unmapped page.
    ///
    /// As with the general form, this does not check `MemProt`/`MemFlags`
    /// permissions; callers must check the returned [`Page`]'s flags
    /// themselves.
    #[inline(always)]
    pub fn resolve_byte_access(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
    ) -> Result<(Page<'_>, usize), MemoryFault> {
        const ALIGN: usize = 1;
        self.resolve_aligned_scalar_access::<ALIGN>(cache, vaddr)
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    /// Loads a single byte from virtual memory.
    ///
    /// The page containing `vaddr` must be mapped with `READ` permission.
    /// The load is a single-byte atomic access, so it is always
    /// single-copy atomic with respect to concurrent stores.
    ///
    /// Returns [`MemoryFault`] on unmapped access or permission failure.
    #[inline(always)]
    pub fn load_byte(&self, cache: impl LookupCache, vaddr: u64) -> Result<u8, MemoryFault> {
        let (page, offset) = self.resolve_byte_access(cache, vaddr)?;
        ensure!(vaddr: vaddr, page.ptr.flags().contains_any(MemFlags::READ));
        let byte = unsafe { memops::load_byte(page.ptr.page_ptr().byte_add(offset).as_ptr()) };
        Ok(byte)
    }

    /// Stores a single byte into virtual memory.
    ///
    /// The page containing `vaddr` must be mapped with `WRITE` permission.
    /// The store is a single-byte atomic access, so it is always
    /// single-copy atomic with respect to concurrent loads and stores.
    ///
    /// Returns [`MemoryFault`] on unmapped access or permission failure.
    #[inline(always)]
    pub fn store_byte(
        &self,
        cache: impl LookupCache,
        vaddr: u64,
        value: u8,
    ) -> Result<(), MemoryFault> {
        let (page, offset) = self.resolve_byte_access(cache, vaddr)?;
        ensure!(vaddr: vaddr, page.ptr.flags().contains_any(MemFlags::WRITE));
        unsafe { memops::store_byte(page.ptr.page_ptr().byte_add(offset).as_ptr(), value) }
        page.set_dirty();
        Ok(())
    }
}

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
    fn static_small_multibyte_access<const BYTES: usize>(
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
                let second_page_num = base_page_num
                    .inc()
                    .ok_or_else(|| MemoryFault::general_protection(u64::MAX))?;

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
                let access = self.static_small_multibyte_access::<{ size_of::<$ty>() }>(cache, vaddr)?;

                let value = match access.second_page {
                    // SAFETY:
                    // `static_small_multibyte_access` returned `None` for `second_page`, so
                    // this access is fully contained in `base_page`.
                    //
                    // Therefore:
                    //
                    //   base_page_offset + size_of::<u$bits>() <= PAGE_SIZE
                    //
                    // which satisfies the safety requirement of `Page::load$bits`.
                    None => unsafe {
                        ensure!(vaddr: vaddr, access.base_page.ptr.flags().contains_any(MemFlags::READ));
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

                        let lo_page_tagged_ptr = access.base_page.ptr;
                        let hi_page_tagged_ptr = second_page.page.ptr;

                        ensure!(
                            vaddr: vaddr,
                            lo_page_tagged_ptr.flags().contains_any(MemFlags::READ)
                        );

                        ensure!(
                            vaddr: {
                                vaddr
                                    .checked_next_multiple_of(PAGE_SIZE_U64)
                                    .unwrap_unchecked()
                            },
                            hi_page_tagged_ptr.flags().contains_any(MemFlags::READ)
                        );

                        let hi_page_ptr = hi_page_tagged_ptr.page_ptr();
                        let lo_page_ptr = lo_page_tagged_ptr.page_ptr();


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
                        // `static_small_multibyte_access` also guarantees
                        // `overflow_amount < BYTES`.
                        //
                        // Therefore:
                        //
                        //   `0 < overflow_amount * 8 < $bits`
                        //
                        // So multiplying by 8 cannot overflow `u8` for the supported
                        // widths, and the resulting bit offset is strictly less than the
                        // integer width.
                        let bit_offset = u32::try_from(
                            second_page.overflow_amount.get().unchecked_mul(8)
                        ).unwrap_unchecked();

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
                let access = self.static_small_multibyte_access::<{ size_of::<$ty>() }>(cache, vaddr)?;

                match access.second_page {
                    // SAFETY:
                    // `static_small_multibyte_access` returned `None` for `second_page`, so
                    // this access is fully contained in `base_page`.
                    //
                    // Therefore:
                    //
                    //   base_page_offset + size_of::<u$bits>() <= A::PAGE_SIZE
                    //
                    // which satisfies the safety requirement of `Page::store$bits`.
                    None => unsafe {
                        let page = access.base_page;
                        ensure!(vaddr: vaddr, page.ptr.flags().contains_any(MemFlags::WRITE));
                        let ptr = page.ptr.page_ptr().byte_add(access.base_page_offset);
                        memops::$store_name(ptr.as_ptr(), value);
                        page.set_dirty();
                    },

                    Some(second_page) => unsafe {
                        cold_path();

                        // Note: we can't load 2 words and combine them like the load case
                        //       since that would alter/mess with the atomicity of the bytes
                        //       next to the value
                        let bytes = value.to_le_bytes();

                        let hi_page = second_page.page;
                        let lo_page = access.base_page;

                        ensure!(
                            vaddr: vaddr,
                            lo_page.ptr.flags().contains_any(MemFlags::WRITE),
                        );

                        ensure!(
                            vaddr: {
                                vaddr
                                    .checked_next_multiple_of(PAGE_SIZE_U64)
                                    .unwrap_unchecked()
                            },
                            hi_page.ptr.flags().contains_any(MemFlags::WRITE),
                        );


                        let hi_page_ptr = hi_page.ptr.page_ptr().as_non_null_ptr();
                        let lo_page_ptr = lo_page.ptr.page_ptr().as_non_null_ptr();

                        let overflow = second_page.overflow_amount.get();

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

                        lo_page.set_dirty();
                        hi_page.set_dirty();
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

impl<T: ?Sized + ICache> IoMMU<T> {
    /// Fetches the 32-bit little-endian AArch64 instruction word at `vaddr`.
    ///
    /// `vaddr` must be 4-byte aligned, and the containing page must be mapped
    /// with `EXECUTE` permission. The fetch is a naturally aligned 32-bit
    /// access and is therefore single-copy atomic.
    ///
    /// Returns [`MemoryFault`] on unmapped access, permission failure, or
    /// misaligned `vaddr`.
    ///
    /// # Note
    ///
    /// This method isn't a very fast way to fetch instructions,
    /// and is better handled by doing bulk compilation per page.
    pub fn fetch_aarch64(&self, cache: impl LookupCache, vaddr: u64) -> Result<u32, MemoryFault> {
        const AARCH64_INSN_ALIGN: usize = size_of::<u32>();

        let (page, offset) =
            self.resolve_aligned_scalar_access::<AARCH64_INSN_ALIGN>(cache, vaddr)?;

        ensure!(vaddr: vaddr, page.ptr.flags().contains_any(MemFlags::EXECUTE));
        let page_ptr = page.ptr.page_ptr();
        let word = unsafe {
            let word_ptr = page_ptr.byte_add(offset);
            memops::load32_le_aligned(word_ptr.as_ptr())
        };
        Ok(word)
    }

    /// Kicks off the asynchronous write-back of every dirty shared
    /// [`MemoryObject`] page in this address space.
    ///
    /// Unlike [`flush`](Self::flush), this does not wait for the write-back
    /// to complete. This walks the whole page table, so it is slow and not
    /// intended for hot paths.
    pub fn refresh(&self) {
        for (_page_num, page) in self.table.pages() {
            page.refresh(self.get_fabric());
        }
    }

    /// Writes every dirty shared [`MemoryObject`] page in this address space
    /// back to its backing object, waiting for all write-back to complete.
    ///
    /// This walks the whole page table and blocks on I/O, so it is slow and
    /// not intended for hot paths.
    ///
    /// # Errors
    ///
    /// Returns an error if a backing object reports a write failure or if
    /// the flusher thread exited.
    pub fn flush(&self) -> anyhow::Result<()> {
        // TODO better api or at least just keep reusing ONE channel
        let pending_jobs = self
            .table
            .pages()
            .flat_map(|(_page_num, page)| page.flush_sync_inner(self.get_fabric()))
            .collect::<Vec<_>>();

        // there is a collect and try for each deliberately
        pending_jobs.into_iter().try_for_each(|pending_job| {
            pending_job
                .recv()
                .unwrap_or_else(|_| anyhow::bail!("memory_object flusher thread exited"))
        })
    }

    /// Faults in every not-yet-loaded [`MemoryObject`] page in this address
    /// space, waiting until all of them are resident.
    ///
    /// After this returns successfully, no access in the mapped ranges will
    /// need to demand-page from a backing object. This walks the whole page
    /// table and blocks on I/O, so it is slow and not intended for hot paths.
    ///
    /// # Errors
    ///
    /// Returns an error if a backing object reports a read failure or if the
    /// flusher thread exited.
    pub fn fault_in_all_memory_objects(&self) -> anyhow::Result<()> {
        // TODO better api or at least just keep reusing ONE channel
        let pending_jobs = self
            .table
            .pages()
            .flat_map(|(_page_num, page)| page.load_obj_memory_sync_inner(self.get_fabric()))
            .collect::<Vec<_>>();

        // there is a collect and try for each deliberately
        pending_jobs.into_iter().try_for_each(|pending_job| {
            pending_job
                .recv()
                .unwrap_or_else(|_| anyhow::bail!("memory_object flusher thread exited"))
        })
    }

    /// Eagerly resolves every copy-on-write page in this address space,
    /// giving each private page its own backing storage now.
    ///
    /// This walks the whole page table and may copy a lot of memory, so it
    /// is slow and not intended for hot paths.
    pub fn copy_all_cow_pages(&mut self) {
        let Ok(()) = unsafe {
            self.modify_table(|table, _| {
                let mut modified = false;
                table.pages_mut().for_each(|(_, page)| {
                    modified |= page.un_cow();
                });

                Ok::<bool, Infallible>(modified)
            })
        };
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    /// # Safety
    ///
    /// `ManuallyDrop<IoMMU<dyn ICache>>` must only be used whilst self is alive
    pub unsafe fn as_ffi<'a>(
        &'a self,
    ) -> (IoMMUIdentifierRef<'a>, ManuallyDrop<IoMMU<dyn ICache>>) {
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
