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
use emu_abi::convert::u64_to_usize;
use emu_abi::internal_traits::{
    AsFFI, CpuFabricPrivate, ICache, IoMMUByteRawAccess, IoMMUPrivate, IoMMURawIntAccess,
};
use emu_abi::memory::{
    HostPointer, IoMMUIdentifier, IoMMUIdentifierRef, MemProt, PAGE_OFFSET_MASK_U64, PAGE_SHIFT,
    PAGE_SIZE, PAGE_SIZE_U64, Page, PageNumber, PagePointer, Tlb,
};
use std::hint::cold_path;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::num::NonZero;
use std::process::abort;
use std::ptr::NonNull;
use std::sync::OnceLock;
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

use crate::page_table::PageTable;
pub(crate) use ensure;

#[inline]
fn div_rem_page_size(vaddr: u64) -> (PageNumber, usize) {
    let remainder = vaddr & PAGE_OFFSET_MASK_U64;
    // Safety: PAGE_SIZE fits in usize, and we are taking the remainder by PAGE_SIZE
    //         therefore the remainder **MUST** fit in a usize
    let remainder = unsafe { u64_to_usize(remainder).unwrap_unchecked() };

    (PageNumber(vaddr >> PAGE_SHIFT), remainder)
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
        let page_count = div_page_size_checked(size)?;
        let (end_addr, overflow) = base.overflowing_add(size);
        if overflow {
            const { assert!(PAGE_SIZE > 1) }
            ensure!(end_addr == 0)
        }

        let Some(page_count) = NonZero::new(page_count.0) else {
            return Ok(None);
        };

        Ok(Some(Self {
            start_page,
            page_count,
        }))
    }

    unsafe fn new(start_page: PageNumber, page_count: NonZero<u64>) -> Self {
        debug_assert!(start_page.0.checked_add(page_count.get()).is_some());
        Self {
            start_page,
            page_count,
        }
    }

    #[inline(always)]
    fn start(self) -> PageNumber {
        self.start_page
    }

    #[inline(always)]
    fn end(self) -> PageNumber {
        let page = unsafe { self.start_page.0.unchecked_add(self.page_count.get()) };
        PageNumber(page)
    }

    pub fn iter(self) -> impl Iterator<Item = PageNumber> {
        (self.start().0..self.end().0).map(PageNumber)
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
    identifier: OnceLock<IoMMUIdentifier>,
    table: PageTable,
    fabric: CpuFabric<T>,
}

impl<T: ?Sized + ICache> IoMMU<T> {
    /// Creates an empty MMU with no mapped pages.
    ///
    /// All accesses fault until memory is mapped with [`IoMMU::map`].
    pub fn new(fabric: CpuFabric<T>) -> Self {
        Self {
            identifier: OnceLock::new(),
            table: PageTable::new(),
            fabric,
        }
    }

    #[inline]
    fn get_ident(&self) -> IoMMUIdentifierRef<'_> {
        self.identifier
            .get_or_init(IoMMUIdentifier::unique_token)
            .get_ref()
    }

    fn change_token(&mut self) {
        self.identifier.take();
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
        ensure!(ptr.addr().is_multiple_of(PAGE_SIZE), !ptr.is_null());

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

    #[inline(always)]
    unsafe fn mem_access(
        &self,
        vaddr: u64,
        len: usize,
        mem_ptr: *mut u8,
        required_prot: impl Into<Option<MemProt>> + Copy,
        mut per_page: impl FnMut(PageNumber, Page<'_>),
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
            NonZero::new_unchecked(last_page.0.unchecked_sub(start_page.0).unchecked_add(1))
        };

        let access = unsafe { PageTableAccess::new(start_page, page_count) };

        let last_page_end = unsafe { end_offset.unchecked_add(1) };

        self.table.access(
            access,
            move |_, page| match required_prot.into() {
                Some(required) => page.ptr.prot().contains(required),
                None => true,
            },
            |page_num, page| unsafe {
                per_page(page_num, page);

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

                let full_pages_before = page_num.0.unchecked_sub(start_page.0);
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
            },
        )?;

        Ok(())
    }

    fn inner_load_with_prot<'a>(
        &self,
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
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>],
    ) -> Result<&'a mut [u8], MemoryFault> {
        self.inner_load_with_prot(vaddr, mem, MemProt::READ)
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
        vaddr: u64,
        mem: &'a mut [MaybeUninit<u8>],
    ) -> Result<&'a mut [u8], MemoryFault> {
        self.inner_load_with_prot(vaddr, mem, None)
    }

    fn load_wrapper<F>(&self, vaddr: u64, mem: &mut [u8], wrapper: F) -> Result<(), MemoryFault>
    where
        F: for<'a> FnOnce(
            &Self,
            u64,
            &'a mut [MaybeUninit<u8>],
        ) -> Result<&'a mut [u8], MemoryFault>,
    {
        let mem_ptr = mem as *mut [u8];
        let maybe_uninit_ptr = mem_ptr as *mut [MaybeUninit<u8>];
        // Safety: `Self::load_into_uninit` never deinitializes
        let mem_as_uninit = unsafe { &mut *maybe_uninit_ptr };
        let slice = wrapper(self, vaddr, mem_as_uninit)?;
        debug_assert!(std::ptr::eq::<[u8]>(slice as *mut [u8], mem_ptr));
        Ok(())
    }

    /// See [`Self::load_into_uninit`]
    pub fn load(&self, vaddr: u64, mem: &mut [u8]) -> Result<(), MemoryFault> {
        self.load_wrapper(vaddr, mem, Self::load_into_uninit)
    }

    /// See [`Self::load_into_uninit_force`]
    pub fn load_force(&self, vaddr: u64, mem: &mut [u8]) -> Result<(), MemoryFault> {
        self.load_wrapper(vaddr, mem, Self::load_into_uninit_force)
    }

    fn inner_store_with_prot(
        &self,
        vaddr: u64,
        mem: &[u8],
        required_prot: impl Into<Option<MemProt>> + Copy,
    ) -> Result<(), MemoryFault> {
        unsafe {
            self.mem_access(
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
    pub fn store(&self, vaddr: u64, mem: &[u8]) -> Result<(), MemoryFault> {
        self.inner_store_with_prot(vaddr, mem, MemProt::WRITE)
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
    pub fn store_force(&self, vaddr: u64, mem: &[u8]) -> Result<(), MemoryFault> {
        self.inner_store_with_prot(vaddr, mem, None)
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline]
    pub(crate) fn single_page_aligned_access<const ACCESS_SIZE: u8>(
        &self,
        vaddr: u64,
    ) -> Result<(PageNumber, Page<'_>, usize), MemoryFault> {
        const {
            assert!(ACCESS_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_power_of_two());
            assert!(PAGE_SIZE.is_multiple_of(ACCESS_SIZE as usize));
        }

        ensure!(vaddr.is_multiple_of(u64::from(ACCESS_SIZE)));

        let (page_num, offset) = div_rem_page_size(vaddr);
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

        let page = self.table.get_page(page_num)?;
        Ok((page_num, page, offset))
    }

    pub(crate) fn single_page_access(
        &self,
        vaddr: u64,
    ) -> Result<(PageNumber, Page<'_>, usize), MemoryFault> {
        const ALIGN: u8 = 1;
        self.single_page_aligned_access::<ALIGN>(vaddr)
    }
}

impl<T: ?Sized + ICache> IoMMUByteRawAccess for IoMMU<T> {
    type Error = MemoryFault;

    #[inline(always)]
    fn load_byte_raw(&self, vaddr: u64) -> Result<(PageNumber, Page<'_>, u8), MemoryFault> {
        let (page_num, page, offset) = self.single_page_access(vaddr)?;
        ensure!(page.ptr.prot().contains(MemProt::READ));
        let byte = unsafe { memops::load_byte(page.ptr.page_ptr().byte_add(offset).as_ptr()) };
        Ok((page_num, page, byte))
    }

    #[inline(always)]
    fn store_byte_raw(&self, vaddr: u64, value: u8) -> Result<(PageNumber, Page<'_>), MemoryFault> {
        let (page_num, page, offset) = self.single_page_access(vaddr)?;
        ensure!(page.ptr.prot().contains(MemProt::WRITE));
        unsafe { memops::store_byte(page.ptr.page_ptr().byte_add(offset).as_ptr(), value) }
        page.set_insn_dirty();
        Ok((page_num, page))
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    pub fn load_byte(&self, vaddr: u64) -> Result<u8, MemoryFault> {
        let (_page_num, _page, byte) = self.load_byte_raw(vaddr)?;
        Ok(byte)
    }

    pub fn store_byte(&self, vaddr: u64, value: u8) -> Result<(), MemoryFault> {
        let (_page_num, _page) = self.store_byte_raw(vaddr, value)?;
        Ok(())
    }
}

#[derive(Copy, Clone)]
struct SecondPage<'a> {
    page: Page<'a>,
    overflow_amount: NonZero<usize>,
}

struct SmallAccess<'a> {
    base_page_num: PageNumber,
    base_page: Page<'a>,
    base_page_offset: usize,
    second_page: Option<SecondPage<'a>>,
}

impl<T: ?Sized + ICache> IoMMU<T> {
    #[inline(always)]
    fn static_small_multibyte_acces<const BYTES: usize>(
        &self,
        vaddr: u64,
    ) -> Result<SmallAccess<'_>, MemoryFault> {
        const {
            assert!(BYTES > 0);
            assert!(BYTES <= PAGE_SIZE * 2);
        }

        let (base_page_num, base_page_offset) = div_rem_page_size(vaddr);
        let base_page = self.table.get_page(base_page_num)?;

        let bytes_left_in_base_page = unsafe { PAGE_SIZE.unchecked_sub(base_page_offset) };

        let second_page = BYTES
            .checked_sub(bytes_left_in_base_page)
            .and_then(NonZero::new)
            .map(|overflow_amount| {
                let second_page_num = base_page_num
                    .0
                    .checked_add(1)
                    .ok_or_else(MemoryFault::fault)
                    .map(PageNumber)?;

                let page = self.table.get_page(second_page_num)?;

                Ok(SecondPage {
                    page,
                    overflow_amount,
                })
            })
            .transpose()?;

        Ok(SmallAccess {
            base_page_num,
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
            memops_load_name: $memops_load_name: ident,
            store_name: $store_name: ident $(,)?
        ])+
    } => {
        // FIXME actual atomic insn
        $(impl<T: ?Sized + ICache> IoMMURawIntAccess<$ty> for IoMMU<T> {
            #[inline(always)]
            fn load_raw(&self, vaddr: u64) -> Result<(PageNumber, Page<'_>, Option<Page<'_>>, $ty), MemoryFault> {
                let mut data = [0; size_of::<$ty>()];
                self.load(vaddr, &mut data)?;
                let value = <$ty>::from_le_bytes(data);

                let pages = self.static_small_multibyte_acces::<{ size_of::<$ty>() }>(vaddr)?;

                Ok((pages.base_page_num, pages.base_page, pages.second_page.map(|p| p.page), value))
            }

            #[inline(always)]
            fn store_raw(&self, vaddr: u64, value: $ty) -> Result<(PageNumber, Page<'_>, Option<Page<'_>>), MemoryFault> {
                let pages = self.static_small_multibyte_acces::<{ size_of::<$ty>() }>(vaddr)?;
                self.store(vaddr, &value.to_le_bytes())?;
                Ok((pages.base_page_num, pages.base_page, pages.second_page.map(|p| p.page)))
            }
        })+

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
            pub fn $load_name(&self, vaddr: u64) -> Result<$ty, MemoryFault> {
                let (_page_num, _page, _second_page, value) = self.load_raw(vaddr)?;
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
            pub fn $store_name(&self, vaddr: u64, value: $ty) -> Result<(), MemoryFault> {
                let (_page_num, _page, _second_page) = self.store_raw(vaddr, value)?;
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
                    memops_load_name: [<load $bits _le_aligned>],
                    store_name: [<store $bits _le>]
                ])+
            }
        }
    };
}

emit_multi_word_load_store! { 64, 32, 16 }

const AARCH64_WORD_ALIGN: u8 = 4;

impl<T: ?Sized + ICache> IoMMUPrivate for IoMMU<T> {
    type MemoryFault = MemoryFault;

    unsafe fn get_ident_unchecked(&self) -> IoMMUIdentifierRef<'_> {
        let ident = unsafe { self.identifier.get().unwrap_unchecked() };
        ident.get_ref()
    }

    fn get_page(&self, page_number: PageNumber) -> Result<Page<'_>, MemoryFault> {
        self.table.get_page(page_number)
    }

    fn fetch_aarch64_full(&self, vaddr: u64) -> Result<(PageNumber, Page<'_>, u32), MemoryFault> {
        let (page_num, page, offset) =
            self.single_page_aligned_access::<AARCH64_WORD_ALIGN>(vaddr)?;

        ensure!(page.ptr.prot().contains(MemProt::EXECUTE));
        unsafe {
            let word_ptr = page.ptr.page_ptr().byte_add(offset);
            let word = memops::load32_le_aligned(word_ptr.as_ptr());
            Ok((page_num, page, word))
        }
    }

    fn fetch_aarch64_with_tlb(&self, tlb: &mut Tlb, vaddr: u64) -> Result<u32, Self::MemoryFault> {
        let ident = self.get_ident();
        ensure!(vaddr.is_multiple_of(AARCH64_WORD_ALIGN as u64));
        let (page_num, offset) = div_rem_page_size(vaddr);

        // note since vaddr is aligned there is no need to check overflow or alignment;
        // for more on why this is always true look at `single_page_aligned_access`

        let entry = tlb.entry(page_num);
        if !std::ptr::addr_eq(entry.tlb_identifier.as_ptr(), ident.ptr().as_ptr())
            || entry.virtual_page_number != page_num
        {
            cold_path();
            let page = self.get_page(page_num)?;
            entry.update_entry(ident, page_num, page)
        }

        let tagged_ptr = unsafe { entry.tagged_page_ptr.unwrap_unchecked() };
        ensure!(tagged_ptr.prot().contains(MemProt::EXECUTE));
        unsafe {
            let word_ptr = tagged_ptr.page_ptr().byte_add(offset);
            let word = memops::load32_le_aligned(word_ptr.as_ptr());
            Ok(word)
        }
    }
}

impl<T: ?Sized + ICache> IoMMU<T> {
    pub fn fetch_aarch64(&self, vaddr: u64) -> Result<u32, MemoryFault> {
        self.fetch_aarch64_full(vaddr).map(|(_, _, word)| word)
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

impl<T: ?Sized + ICache> Clone for IoMMU<T> {
    fn clone(&self) -> Self {
        let ident = self.get_ident().clone_identifier();
        Self {
            identifier: OnceLock::from(ident),
            table: self.table.clone(),
            fabric: self.fabric.clone(),
        }
    }
}

struct AbortGuard(());

impl AbortGuard {
    fn disarm(self) {
        std::mem::forget(self)
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        abort()
    }
}

impl<T: Sized + ICache> AsFFI for IoMMU<T> {
    type Inetrface<'a>
        = (IoMMUIdentifierRef<'a>, ManuallyDrop<IoMMU<dyn ICache + 'a>>)
    where
        T: 'a;

    fn as_ffi<'a>(&'a self) -> Self::Inetrface<'a>
    where
        Self: 'a,
    {
        let ident_ref = self.get_ident();
        let guard = AbortGuard(());
        let identifier = OnceLock::from(unsafe { ident_ref.copy_identifier() });
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
    type Inetrface<'b>
        = (IoMMUIdentifierRef<'b>, ManuallyDrop<IoMMU<dyn ICache + 'b>>)
    where
        Self: 'b;

    fn as_ffi<'b>(&'b self) -> Self::Inetrface<'b>
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
