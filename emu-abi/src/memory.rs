use crate::convert::u64_to_usize;
use bytemuck::{Zeroable, ZeroableInOption};
use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::num::NonZero;
use std::ops::{BitAnd, BitOr};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[allow(
    clippy::cast_possible_truncation,
    reason = "shifts never overflow a u8"
)]
const fn compue_shift(power_of_2: u64) -> u8 {
    let power_of_2 = NonZero::new(power_of_2).unwrap();
    assert!(power_of_2.is_power_of_two());
    let bits = power_of_2.trailing_zeros();
    assert!(bits < 24, "page size too big");
    bits as u8
}

const fn compute_mask(size: u64) -> u64 {
    assert!(size.is_power_of_two());
    size.strict_sub(1)
}

pub const PAGE_SIZE_U64: u64 = 4096;
pub const PAGE_SIZE: usize = u64_to_usize(PAGE_SIZE_U64).unwrap();

pub const CACHE_LINE_SIZE_U64: u64 = 64;
pub const CACHE_LINE_SIZE: usize = u64_to_usize(CACHE_LINE_SIZE_U64).unwrap();

pub const PAGE_SHIFT: u8 = compue_shift(PAGE_SIZE_U64);
pub const CACHE_LINE_SHIFT: u8 = compue_shift(CACHE_LINE_SIZE_U64);

pub const PAGE_OFFSET_MASK_U64: u64 = compute_mask(PAGE_SIZE_U64);
pub const PAGE_OFFSET_MASK: usize = u64_to_usize(PAGE_OFFSET_MASK_U64).unwrap();

// important note
// do **not** use bitflags
// MemProt is used for bit tagging pointers
// and must therefore only have the bits explicitly
// allowed here set
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemProt(u8);

impl MemProt {
    pub const NONE: Self = Self(0);

    pub const READ: Self = Self(0b001);
    pub const WRITE: Self = Self(0b010);
    pub const EXECUTE: Self = Self(0b100);

    pub const ALL: Self = Self::READ.union(Self::WRITE).union(Self::EXECUTE);

    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[inline(always)]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline(always)]
    pub const fn retain(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[inline(always)]
    pub const fn contains(self, other: MemProt) -> bool {
        (self.0 & other.0) != 0
    }
}

impl BitOr for MemProt {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitAnd for MemProt {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.retain(rhs)
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
        unsafe { std::hint::assert_unchecked((page.addr().get() & PAGE_OFFSET_MASK) == 0) }
        Self(HostPointer(page))
    }

    #[inline(always)]
    pub const fn as_non_null_ptr(self) -> NonNull<AtomicU8> {
        self.0.0
    }

    /// # Safety
    ///
    /// same as calling `<*mut T>::add(self, count)` where
    /// `T` has a a layout of size `PAGE_SIZE` and align of `PAGE_SIZE`
    #[inline(always)]
    pub unsafe fn add_pages(self, count: usize) -> Self {
        #[repr(C)]
        struct PageMemLayout([AtomicU8; PAGE_SIZE]);

        const _: () = assert!(size_of::<PageMemLayout>() == PAGE_SIZE);

        unsafe {
            Self::new(
                self.as_non_null_ptr()
                    .cast::<PageMemLayout>()
                    .add(count)
                    .cast::<AtomicU8>(),
            )
        }
    }

    /// # Safety
    ///
    /// same as calling `<ptr>::byte_add(count)`
    #[inline(always)]
    pub unsafe fn byte_add(self, count: usize) -> NonNull<AtomicU8> {
        unsafe { self.as_non_null_ptr().byte_add(count) }
    }
}

#[derive(Copy, Clone)]
pub struct Page<'a> {
    pub ptr: TaggedPagePtr,
    pub insn_dirty: &'a AtomicBool,
}

impl Page<'_> {
    #[inline(always)]
    pub fn set_insn_dirty(&self) {
        if self.ptr.prot().contains(MemProt::EXECUTE) {
            cold_path();
            if !self.insn_dirty.load(Ordering::Relaxed) {
                cold_path();
                self.insn_dirty.store(true, Ordering::Release)
            }
        }
    }

    #[inline(always)]
    pub fn get_insn_dirty(&self) -> bool {
        self.insn_dirty.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn unset_insn_dirty(&self) {
        self.insn_dirty.store(false, Ordering::Release)
    }
}

// use MaybeUninit<u8> so that it is explicitly not a zst and will always be a unique alloc
type IoMMUIdentPointee = MaybeUninit<u8>;
type IoMMUIdentifierInner = Arc<IoMMUIdentPointee>;

#[derive(Eq, PartialEq)]
#[repr(transparent)]
pub struct IoMMUIdentifier(NonNull<IoMMUIdentPointee>);

unsafe impl ZeroableInOption for IoMMUIdentifier {}

impl IoMMUIdentifier {
    pub fn unique_token() -> Self {
        // note we use a byte to make absoulely sure that we are getting a new
        // `Arc` and that this isn't some cached ZST Arc
        let mut alloc: IoMMUIdentifierInner = Arc::<u8>::new_uninit();
        assert!(
            Arc::get_mut(&mut alloc).is_some(),
            "allocation is not unique"
        );
        let ptr = Arc::into_raw(alloc);
        Self(unsafe { NonNull::new_unchecked(ptr.cast_mut()) })
    }

    pub fn get_ref(&self) -> IoMMUIdentifierRef<'_> {
        IoMMUIdentifierRef {
            ptr: self.0,
            _marker: PhantomData,
        }
    }
}

impl Clone for IoMMUIdentifier {
    fn clone(&self) -> Self {
        unsafe { IoMMUIdentifierInner::increment_strong_count(self.0.as_ptr()) }

        Self(self.0)
    }

    fn clone_from(&mut self, source: &Self) {
        if (*self) == (*source) {
            return;
        }

        *self = source.clone();
    }
}

impl Drop for IoMMUIdentifier {
    fn drop(&mut self) {
        unsafe { IoMMUIdentifierInner::decrement_strong_count(self.0.as_ptr()) }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct IoMMUIdentifierRef<'a> {
    ptr: NonNull<IoMMUIdentPointee>,
    _marker: PhantomData<&'a IoMMUIdentifier>,
}

impl IoMMUIdentifierRef<'_> {
    #[inline(always)]
    pub fn clone_identifier(self) -> IoMMUIdentifier {
        let ptr = ManuallyDrop::new(IoMMUIdentifier(self.ptr));
        (*ptr).clone()
    }

    /// # Safety
    ///
    /// **must** not ever drop the returned identifier
    #[inline(always)]
    pub unsafe fn copy_identifier(self) -> IoMMUIdentifier {
        IoMMUIdentifier(self.ptr)
    }

    #[inline(always)]
    pub fn ptr(self) -> NonNull<()> {
        self.ptr.cast()
    }
}

const _: () = {
    assert!(
        (MemProt::ALL.bits() as u64 & !PAGE_OFFSET_MASK_U64) == 0,
        "MemProt bits must fit in the low page-alignment bits"
    );
};

#[derive(Debug, Zeroable, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct PageNumber(u64);

impl PageNumber {
    pub const MAX: Self = Self::from_vaddr(u64::MAX);

    #[inline(always)]
    pub const fn from_vaddr_with_offset(vaddr: u64) -> (Self, usize) {
        let remainder = vaddr & PAGE_OFFSET_MASK_U64;
        // Safety: PAGE_SIZE fits in usize, and we are taking the remainder by PAGE_SIZE
        //         therefore the remainder **MUST** fit in a usize
        let remainder = unsafe { u64_to_usize(remainder).unwrap_unchecked() };

        (Self::from_vaddr(vaddr), remainder)
    }

    #[inline(always)]
    pub const fn from_vaddr(vaddr: u64) -> Self {
        Self(vaddr >> PAGE_SHIFT)
    }

    /// # Safety
    /// TODO
    #[inline(always)]
    pub const unsafe fn from_page_number_unchecked(page: u64) -> Self {
        unsafe { std::hint::assert_unchecked(page <= Self::MAX.0) }
        Self(page)
    }

    #[inline]
    pub const fn from_page_number_checked(page: u64) -> Option<Self> {
        if page > Self::MAX.0 {
            return None;
        }

        Some(unsafe { Self::from_page_number_unchecked(page) })
    }

    #[inline]
    pub const fn from_page_number(page: u64) -> Self {
        match Self::from_page_number_checked(page) {
            Some(page) => page,
            None => panic!("page out of bounds"),
        }
    }

    #[inline(always)]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[inline(always)]
    pub const fn inc(self) -> Option<Self> {
        const { assert!(Self::MAX.0 != u64::MAX) }

        Self::from_page_number_checked(unsafe { self.0.unchecked_add(1) })
    }
}

const _: () = assert!(PageNumber::MAX.inc().is_none());

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct TaggedPagePtr(NonNull<AtomicU8>);

impl TaggedPagePtr {
    pub fn new(ptr: PagePointer, prot: MemProt) -> Self {
        let prot: u8 = prot.bits();
        const { assert!((u8::MAX as usize) < PAGE_SIZE) }
        let ptr = ptr.as_non_null_ptr();
        let tag_bits = usize::from(prot);
        Self(ptr.map_addr(|addr| addr | tag_bits))
    }

    pub fn page_ptr(self) -> PagePointer {
        let ptr = self.0.map_addr(|addr| {
            let mask = !usize::from(MemProt::ALL.bits());
            unsafe { NonZero::new_unchecked(addr.get() & mask) }
        });

        unsafe { PagePointer::new(ptr) }
    }

    pub fn prot(self) -> MemProt {
        let mask: u8 = MemProt::ALL.bits();
        let prot_usize = self.0.addr().get() & usize::from(mask);
        let prot_raw: u8 = unsafe { u8::try_from(prot_usize).unwrap_unchecked() };
        MemProt(prot_raw)
    }
}

unsafe impl ZeroableInOption for TaggedPagePtr {}

#[derive(Zeroable)]
#[repr(transparent)]
pub struct MaybeInvalidIdentifier(Option<IoMMUIdentifier>);

impl MaybeInvalidIdentifier {
    pub const fn invalid() -> Self {
        bytemuck::zeroed()
    }

    pub fn new(ident: IoMMUIdentifier) -> Self {
        Self(Some(ident))
    }

    pub fn as_ptr(&self) -> *const () {
        unsafe { std::mem::transmute_copy::<Self, *const ()>(self) }
    }
}

/// # Safety
///
/// if `tagged_page_ptr` is `Some`
/// then `insn_dirty_ptr` must also be `Some`
#[derive(Zeroable)]
pub struct TlbEntry {
    pub tlb_identifier: MaybeInvalidIdentifier,
    pub virtual_page_number: PageNumber,
    pub tagged_page_ptr: Option<TaggedPagePtr>,
    pub insn_dirty_ptr: Option<NonNull<AtomicBool>>,
}

impl TlbEntry {
    pub fn update_entry(
        &mut self,
        identifier: IoMMUIdentifierRef,
        new_page_number: PageNumber,
        page: Page,
    ) {
        let Self {
            tlb_identifier,
            virtual_page_number,
            tagged_page_ptr,
            insn_dirty_ptr,
        } = self;

        let tagged_ptr = page.ptr;

        let new_insn_dirty_ptr = NonNull::<AtomicBool>::from_ref(page.insn_dirty);

        if !std::ptr::addr_eq(tlb_identifier.as_ptr(), identifier.ptr().as_ptr()) {
            *tlb_identifier = MaybeInvalidIdentifier::new(identifier.clone_identifier());
        }

        *tagged_page_ptr = Some(tagged_ptr);
        *insn_dirty_ptr = Some(new_insn_dirty_ptr);
        *virtual_page_number = new_page_number;
    }
}

pub const TLB_SIZE_U64: u64 = match cfg!(test) {
    true => 64,
    false => 1024,
};

pub const TLB_SIZE: usize = u64_to_usize(TLB_SIZE_U64).unwrap();
pub const TLB_MASK: u64 = compute_mask(TLB_SIZE_U64);

#[derive(Zeroable)]
#[repr(transparent)]
pub struct Tlb {
    pub entries: [TlbEntry; TLB_SIZE],
}

impl Default for Tlb {
    fn default() -> Self {
        Self::new()
    }
}

impl Tlb {
    pub fn new_boxed() -> Box<Self> {
        bytemuck::allocation::zeroed_box()
    }

    pub const fn new() -> Self {
        bytemuck::zeroed()
    }

    pub fn entry(&mut self, page_number: PageNumber) -> &mut TlbEntry {
        unsafe {
            let index = u64_to_usize(page_number.0 & TLB_MASK).unwrap_unchecked();
            self.entries.get_unchecked_mut(index)
        }
    }

    /// # Safety
    ///
    /// TODO
    #[inline(always)]
    pub unsafe fn lookup<'a>(
        &mut self,
        page_num: PageNumber,
        ident: IoMMUIdentifierRef,
        fallback: impl FnOnce(PageNumber) -> Option<Page<'a>>,
    ) -> Option<Page<'a>> {
        let entry = self.entry(page_num);

        if !std::ptr::addr_eq(entry.tlb_identifier.as_ptr(), ident.ptr().as_ptr())
            || entry.virtual_page_number != page_num
        {
            cold_path();
            let page = fallback(page_num)?;
            entry.update_entry(ident, page_num, page);
            return Some(page);
        }

        Some(unsafe {
            Page {
                ptr: entry.tagged_page_ptr.unwrap_unchecked(),
                insn_dirty: entry.insn_dirty_ptr.unwrap_unchecked().as_ref(),
            }
        })
    }

    #[inline(always)]
    pub fn update_entry(
        &mut self,
        page_number: PageNumber,
        identifier: IoMMUIdentifierRef,
        page: Page,
    ) {
        self.entry(page_number)
            .update_entry(identifier, page_number, page)
    }
}
