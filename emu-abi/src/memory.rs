use crate::convert::u64_to_usize;
use bytemuck::{Zeroable, ZeroableInOption};
use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::num::NonZero;
use std::ops::{BitAnd, BitOr};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

#[allow(
    clippy::cast_possible_truncation,
    reason = "shifts never overflow a u8"
)]
const fn compute_shift(power_of_2: u64) -> u8 {
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

#[repr(C, align(4096))]
pub struct UninitPageMut([MaybeUninit<u8>; PAGE_SIZE]);

impl UninitPageMut {
    #[inline(always)]
    pub const fn new() -> Self {
        const { Self([MaybeUninit::uninit(); PAGE_SIZE]) }
    }

    pub fn page_pointer_mut(&mut self) -> PagePointer {
        unsafe { PagePointer::new(NonNull::new_unchecked(self.0.as_mut_ptr()).cast()) }
    }

    pub fn page_pointer_ref(&self) -> PagePointer {
        unsafe { PagePointer::new(NonNull::new_unchecked(self.0.as_ptr().cast_mut()).cast()) }
    }

    /// # Safety
    ///
    /// must point to a live - non-dangling - page
    pub unsafe fn from_ptr<'a>(ptr: PagePointer) -> &'a mut Self {
        unsafe { ptr.as_non_null_ptr().cast::<Self>().as_mut() }
    }
}

impl Default for UninitPageMut {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

const _: () = {
    assert!(align_of::<UninitPageMut>() == PAGE_SIZE && size_of::<UninitPageMut>() == PAGE_SIZE)
};

pub const CACHE_LINE_SIZE_U64: u64 = 64;
pub const CACHE_LINE_SIZE: usize = u64_to_usize(CACHE_LINE_SIZE_U64).unwrap();

pub const PAGE_SHIFT: u8 = compute_shift(PAGE_SIZE_U64);
pub const CACHE_LINE_SHIFT: u8 = compute_shift(CACHE_LINE_SIZE_U64);

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
    pub const fn contains_any(self, other: Self) -> bool {
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

impl std::ops::Not for MemProt {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self((!self.bits()) & Self::ALL.bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemFlags(u8);

impl MemFlags {
    const fn new_flag(bit: u8) -> Self {
        assert!(bit.count_ones() == 1);
        assert!((bit & MemProt::ALL.0) == 0);
        Self(bit)
    }

    #[inline(always)]
    pub const fn from_prot(prot: MemProt) -> Self {
        Self(prot.0)
    }

    pub const NONE: Self = Self(0);

    pub const READ: Self = Self::from_prot(MemProt::READ);
    pub const WRITE: Self = Self::from_prot(MemProt::WRITE);
    pub const EXECUTE: Self = Self::from_prot(MemProt::EXECUTE);

    pub const DMA_DEV: Self = Self::new_flag(0b001_000);
    pub const COW: Self = Self::new_flag(0b010_000);

    pub const MUST_DIRTY: Self = Self::EXECUTE.union(Self::DMA_DEV);

    pub const ALL: Self = Self::from_prot(MemProt::ALL)
        .union(Self::COW)
        .union(Self::DMA_DEV);

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
    pub const fn contains_any(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl From<MemProt> for MemFlags {
    #[inline(always)]
    fn from(value: MemProt) -> Self {
        Self::from_prot(value)
    }
}

impl<T: Into<MemFlags>> BitOr<T> for MemFlags {
    type Output = Self;

    fn bitor(self, rhs: T) -> Self::Output {
        self.union(rhs.into())
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
    /// # Note:
    ///
    /// you can never know if a page is truly `Self::DANGLING`
    /// therefore you must ensure that if you have `Self::DANGLING`
    /// you never access it in any way, and checking for it equalling dangling
    /// may exclude real pages
    pub const DANGLING: Self = {
        // make sure you can always offset into any place inside the dangling page
        assert!(PAGE_SIZE.checked_add(PAGE_SIZE).is_some());
        Self(HostPointer(NonNull::without_provenance(
            NonZero::new(PAGE_SIZE).unwrap(),
        )))
    };

    /// # Safety
    ///
    /// `page` must be aligned to PAGE_SIZE
    /// and if `page + PAGE_SIZE` must exist In particular,
    /// this range must not "wrap around" the edge of the address space.
    pub unsafe fn new(page: NonNull<AtomicU8>) -> Self {
        unsafe { std::hint::assert_unchecked((page.addr().get() & PAGE_OFFSET_MASK) == 0) }
        unsafe { std::hint::assert_unchecked(page.addr().get().is_multiple_of(PAGE_SIZE)) }
        Self(HostPointer(page))
    }

    #[inline(always)]
    pub fn as_non_null_ptr(self) -> NonNull<AtomicU8> {
        let ptr = self.0.0;
        unsafe { std::hint::assert_unchecked((ptr.addr().get() & PAGE_OFFSET_MASK) == 0) }
        unsafe { std::hint::assert_unchecked(ptr.addr().get().is_multiple_of(PAGE_SIZE)) }
        ptr
    }

    /// # Safety
    ///
    /// same as calling `<*mut T>::add(self, count)` where
    /// `T` has a a layout of size `PAGE_SIZE` and align of `PAGE_SIZE`
    #[inline(always)]
    pub unsafe fn add_pages(self, count: usize) -> Self {
        unsafe {
            Self::new(
                self.as_non_null_ptr()
                    .cast::<UninitPageMut>()
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
    pub dirty_flags: &'a AtomicU8,
}

impl Page<'_> {
    pub const DIRTY_FLAG_FLUSHING: u8 = 0b10;
    pub const DIRTY_FLAG_IS_DIRTY: u8 = 0b01;

    #[inline(always)]
    pub fn set_dirty(&self) {
        if self.ptr.flags().contains_any(MemFlags::MUST_DIRTY) {
            cold_path();

            std::sync::atomic::fence(Ordering::SeqCst);

            // knowing that `set_insn_dirty` always executes AFTER the pages have been modified,
            // skipping the `fetch_or` when the dirty bit is already set is only safe `iff`
            // OUR modifications to the page are guaranteed to be globally visible before some
            // future flusher decides "nothing more to do" and clears the flag.
            //
            // that guarantee is a STORE-LOAD ordering: our store to the page must be ordered
            // before our load of the flag, as observed by every other thread. Acquire/Release
            // never provide that - they order StoreStore and LoadLoad/LoadStore,
            // but a CPU is still free to have our page write sitting in its store buffer
            // while our flag load is satisfied from cache. SeqCst is the only ordering
            // that forbids this: every SeqCst operation is additionally placed on one
            // single global total order, so our SeqCst load of the flag cannot be
            // reordered ahead of our own prior SeqCst-adjacent store to the page from
            // any other thread's point of view, closing exactly the reordering window
            // that let a flusher race ahead, invalidate, and clean the flag while our
            // write was still invisible.
            //
            // this is necessary as `fetch_or` on every write operation would be too
            // expensive especially under contention, so we still skip it when we can —
            // we're just no longer trying to get that skip for free with a cheaper
            // ordering, since Acquire/Release was paying for a guarantee it never gave us.
            // do pay close attention that this uses
            // **`SeqCst`** specifically, not **`Acquire`** nor **`Relaxed`**, on this load.
            //
            // the `fetch_or` below does NOT need that same upgrade. its job is narrower:
            // publish the new flag *value* to whoever reads it next, so that a future
            // acquire/SeqCst read of `01`/`11` synchronizes-with this store and correctly
            // sees our page write as happens-before. that publish is the textbook
            // Release pattern, and `Release` is sufficient for it - by the time we reach
            // this `fetch_or`, the StoreLoad hazard is already closed, because it only
            // runs after the SeqCst load above has completed, and that load is what forced
            // our page write to be globally visible. `B_A` is also already sequenced-before
            // this `fetch_or` by plain program order. there is no second StoreLoad gap left
            // for this operation to plug, so bumping it to `SeqCst` would only buy us a
            // place in the global total order we don't need, at a cost we'd rather not pay.
            if (self.dirty_flags.load(Ordering::SeqCst) & Self::DIRTY_FLAG_IS_DIRTY) == 0 {
                cold_path();
                self.dirty_flags
                    .fetch_or(Self::DIRTY_FLAG_IS_DIRTY, Ordering::SeqCst);
            }
        }
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
        (MemFlags::ALL.bits() as u64 & !PAGE_OFFSET_MASK_U64) == 0,
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

    pub const fn vaddr_base(self) -> u64 {
        self.0 << PAGE_SHIFT
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
    pub fn new(ptr: PagePointer, prot: MemFlags) -> Self {
        let prot: u8 = prot.bits();
        const { assert!((u8::MAX as usize) < PAGE_SIZE) }
        let ptr = ptr.as_non_null_ptr();
        let tag_bits = usize::from(prot);
        Self(ptr.map_addr(|addr| addr | tag_bits))
    }

    pub fn page_ptr(self) -> PagePointer {
        let ptr = self.0.map_addr(|addr| {
            let mask = !usize::from(MemFlags::ALL.bits());
            unsafe { NonZero::new_unchecked(addr.get() & mask) }
        });

        unsafe { PagePointer::new(ptr) }
    }

    pub fn flags(self) -> MemFlags {
        let mask: u8 = MemFlags::ALL.bits();
        let prot_usize = self.0.addr().get() & usize::from(mask);
        let prot_raw: u8 = unsafe { u8::try_from(prot_usize).unwrap_unchecked() };
        MemFlags(prot_raw)
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
    pub io_mmu_ident: MaybeInvalidIdentifier,
    pub virtual_page_number: PageNumber,
    pub tagged_page_ptr: Option<TaggedPagePtr>,
    pub insn_dirty_ptr: Option<NonNull<AtomicU8>>,
}

impl TlbEntry {
    pub fn update_entry(
        &mut self,
        identifier: IoMMUIdentifierRef,
        new_page_number: PageNumber,
        page: Page,
    ) {
        let Self {
            io_mmu_ident: tlb_identifier,
            virtual_page_number,
            tagged_page_ptr,
            insn_dirty_ptr,
        } = self;

        let tagged_ptr = page.ptr;

        let new_insn_dirty_ptr = NonNull::<AtomicU8>::from_ref(page.dirty_flags);

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
    pub unsafe fn lookup<'a, E>(
        &mut self,
        page_num: PageNumber,
        ident: IoMMUIdentifierRef,
        fallback: impl FnOnce(PageNumber) -> Result<Page<'a>, E>,
    ) -> Result<Page<'a>, E> {
        let entry = self.entry(page_num);

        if !std::ptr::addr_eq(entry.io_mmu_ident.as_ptr(), ident.ptr().as_ptr())
            || entry.virtual_page_number != page_num
        {
            cold_path();
            let page = fallback(page_num)?;
            entry.update_entry(ident, page_num, page);
            return Ok(page);
        }

        Ok(unsafe {
            Page {
                ptr: entry.tagged_page_ptr.unwrap_unchecked(),
                dirty_flags: entry.insn_dirty_ptr.unwrap_unchecked().as_ref(),
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
