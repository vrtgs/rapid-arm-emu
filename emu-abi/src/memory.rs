use crate::convert::u64_to_usize;
use std::hint::cold_path;
use std::num::NonZero;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

pub const PAGE_SIZE_U64: u64 = 4096;
pub const PAGE_SIZE: usize = u64_to_usize(PAGE_SIZE_U64).unwrap();

pub const CACHE_LINE_SIZE_U64: u64 = 64;
pub const CACHE_LINE_SIZE: usize = u64_to_usize(CACHE_LINE_SIZE_U64).unwrap();

const fn compue_shift(power_of_2: u64) -> u8 {
    let power_of_2 = NonZero::new(power_of_2).unwrap();
    assert!(power_of_2.is_power_of_two());
    let bits = power_of_2.trailing_zeros();
    assert!(bits < 24, "page size too big");
    bits as u8
}

pub const PAGE_SHIFT: u8 = compue_shift(PAGE_SIZE_U64);
pub const CACHE_LINE_SHIFT: u8 = compue_shift(CACHE_LINE_SIZE_U64);

pub const PAGE_OFFSET_MASK: u64 = PAGE_SIZE_U64.strict_sub(1);

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct MemProt: u8 {
        const READ       = 0b0001;
        const WRITE      = 0b0010;
        const EXECUTE    = 0b0100;
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
pub struct Page {
    pub ptr: Option<NonNull<AtomicU8>>,
    pub mem_prot: MemProt,
    pub insn_dirty: AtomicBool,
}

impl Page {
    #[allow(clippy::declare_interior_mutable_const)]
    pub const UNMAPPED: Self = Self {
        ptr: None,
        mem_prot: MemProt::empty(),
        insn_dirty: AtomicBool::new(false),
    };

    #[inline(always)]
    pub fn mapped(ptr: NonNull<AtomicU8>, memory_protections: MemProt) -> Self {
        Self {
            ptr: Some(ptr),
            mem_prot: memory_protections,
            insn_dirty: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    pub fn unmap(&mut self) {
        *self = Self::UNMAPPED
    }

    #[inline(always)]
    pub fn is_mapped(&self) -> bool {
        self.ptr.is_some()
    }

    #[inline(always)]
    pub fn has_access(&self, flags: MemProt) -> bool {
        if cfg!(debug_assertions) && self.ptr.is_none() {
            assert!(self.mem_prot.is_empty() && !self.insn_dirty.load(Ordering::Relaxed));
        }
        self.mem_prot.contains(flags)
    }

    /// # Safety
    /// `self` must be mapped
    pub unsafe fn mem_protect(&mut self, memory_protections: MemProt) {
        debug_assert!(self.is_mapped());

        let old_prot = self.mem_prot;
        let new_prot = memory_protections;

        // if we had the execute permision; and then suddenly its gone
        // that means we can no longer execute the instructions in this page
        // therefore the page now contains dirty instructions
        if old_prot.contains(MemProt::EXECUTE) && !old_prot.contains(MemProt::EXECUTE) {
            *self.insn_dirty.get_mut() = true
        }

        self.mem_prot = new_prot;
    }

    #[inline(always)]
    pub fn insn_dirty_mut(&mut self) -> &mut bool {
        self.insn_dirty.get_mut()
    }

    /// # Safety
    ///
    /// `self` must be mapped
    /// this can be ensured by either making sure that
    /// you have access to one or more memory protections
    ///
    /// or by calling `self.is_mapped()`
    #[inline(always)]
    pub unsafe fn get_data_ptr_unchecked(&self) -> NonNull<AtomicU8> {
        debug_assert!(self.is_mapped());
        unsafe { self.ptr.unwrap_unchecked() }
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
    pub fn set_insn_dirty(&self) {
        if self.mem_prot.contains(MemProt::EXECUTE) {
            cold_path();
            if !self.insn_dirty.load(Ordering::Relaxed) {
                cold_path();
                self.insn_dirty.store(true, Ordering::Release)
            }
        }
    }

    pub fn is_insn_dirty(&self) -> bool {
        self.insn_dirty.load(Ordering::Acquire)
    }

    pub fn take_insn_dirty(&self) -> bool {
        // Use AcqRel because this operation both observes and clears insn_dirty.
        //
        // Acquire pairs with the Release in set_insn_dirty, ensuring that if we observe
        // the dirty bit, the writes that dirtied the executable page are visible before
        // we invalidate cached instructions for it.
        //
        // Release covers the clearing side of the RMW: once we clear insn_dirty, later
        // observers must not treat the old dirty state as still pending through this
        // operation.
        self.insn_dirty.swap(false, Ordering::AcqRel)
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct HostPointer(pub NonNull<AtomicU8>);

impl HostPointer {
    pub const fn new(ptr: NonNull<AtomicU8>) -> Self {
        Self(ptr)
    }
}

unsafe impl Send for HostPointer {}
unsafe impl Sync for HostPointer {}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct PagePointer(HostPointer);

impl PagePointer {
    /// # Safety
    ///
    /// `page` must be aligned to PAGE_SIZE
    /// and if `page + PAGE_SIZE` must exist In particular,
    /// this range must not "wrap around" the edge of the address space.
    pub unsafe fn new(page: NonNull<AtomicU8>) -> Self {
        Self(HostPointer(page))
    }

    pub const fn as_range(&self) -> std::ops::Range<HostPointer> {
        let start = self.0;
        let end = HostPointer(unsafe { start.0.add(PAGE_SIZE) });
        start..end
    }
}
