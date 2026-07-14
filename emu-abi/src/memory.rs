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

/// Size of a single memory page, in bytes, as a `u64`.
pub const PAGE_SIZE_U64: u64 = 4096;
/// Size of a single memory page, in bytes, as a `usize`.
pub const PAGE_SIZE: usize = u64_to_usize(PAGE_SIZE_U64).unwrap();
/// Number of bits to shift a virtual address right by to get a page number.
pub const PAGE_SHIFT: u8 = compute_shift(PAGE_SIZE_U64);
/// Bitmask that extracts the in-page offset from a virtual address, as a `u64`.
pub const PAGE_OFFSET_MASK_U64: u64 = compute_mask(PAGE_SIZE_U64);
/// Bitmask that extracts the in-page offset from a virtual address, as a `usize`.
pub const PAGE_OFFSET_MASK: usize = u64_to_usize(PAGE_OFFSET_MASK_U64).unwrap();

/// An uninitialized, page-sized and page-aligned block of memory.
#[repr(C, align(4096))]
pub struct UninitPage([MaybeUninit<u8>; PAGE_SIZE]);

const _: () =
    assert!(align_of::<UninitPage>() == PAGE_SIZE && size_of::<UninitPage>() == PAGE_SIZE);

impl UninitPage {
    /// Creates a new, uninitialized page.
    #[inline(always)]
    pub const fn new() -> Self {
        const { Self([MaybeUninit::uninit(); PAGE_SIZE]) }
    }

    /// Returns a [`PagePointer`] to this page, usable for mutable access.
    pub fn page_pointer_mut(&mut self) -> PagePointer {
        unsafe { PagePointer::new(NonNull::new_unchecked(self.0.as_mut_ptr()).cast()) }
    }

    /// Returns a [`PagePointer`] to this page, usable for shared access.
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

impl Default for UninitPage {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Size of a CPU cache line, in bytes, as a `u64`.
pub const CACHE_LINE_SIZE_U64: u64 = 64;
/// Size of a CPU cache line, in bytes, as a `usize`.
pub const CACHE_LINE_SIZE: usize = u64_to_usize(CACHE_LINE_SIZE_U64).unwrap();
/// Number of bits to shift an address right by to get a cache-line number.
pub const CACHE_LINE_SHIFT: u8 = compute_shift(CACHE_LINE_SIZE_U64);

/// Memory protection bits (read/write/execute).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemProt(u8);

impl MemProt {
    /// No permissions.
    pub const NONE: Self = Self(0);

    /// Read permission.
    pub const READ: Self = Self(0b001);
    /// Write permission.
    pub const WRITE: Self = Self(0b010);
    /// Execute permission.
    pub const EXECUTE: Self = Self(0b100);

    /// All permissions (read, write, and execute).
    const ALL: Self = Self::READ.union(Self::WRITE).union(Self::EXECUTE);

    /// Returns the raw bits backing this value.
    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns the union of `self` and `other`.
    #[inline(always)]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the bits of `self` that are also set in `other`.
    #[inline(always)]
    pub const fn retain(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns `true` if `self` and `other` share any set bits.
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

/// Memory flags: [`MemProt`] permission bits plus additional metadata bits
/// (such as [`MemFlags::COW`] and [`MemFlags::OBJ_BACKED`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemFlags(u8);

impl MemFlags {
    const fn new_flag(bit: u8) -> Self {
        assert!(bit.count_ones() == 1);
        assert!((bit & MemProt::ALL.0) == 0);
        Self(bit)
    }

    /// Creates a [`MemFlags`] from a [`MemProt`], carrying over just the
    /// protection bits.
    #[inline(always)]
    pub const fn from_prot(prot: MemProt) -> Self {
        Self(prot.0)
    }

    /// No flags set.
    pub const NONE: Self = Self(0);

    /// Read permission.
    pub const READ: Self = Self::from_prot(MemProt::READ);
    /// Write permission.
    pub const WRITE: Self = Self::from_prot(MemProt::WRITE);
    /// Execute permission.
    pub const EXECUTE: Self = Self::from_prot(MemProt::EXECUTE);

    /// Marks a page as backed by an object.
    pub const OBJ_BACKED: Self = Self::new_flag(0b001_000);
    /// Marks a page as copy-on-write.
    pub const COW: Self = Self::new_flag(0b010_000);

    /// Flags for which writes must always be tracked as dirty.
    pub const MUST_DIRTY: Self = Self::EXECUTE.union(Self::OBJ_BACKED);

    /// All valid flag bits.
    pub const ALL: Self = Self::from_prot(MemProt::ALL)
        .union(Self::COW)
        .union(Self::OBJ_BACKED);

    /// Returns the raw bits backing this value.
    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns the union of `self` and `other`.
    #[inline(always)]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the bits of `self` that are also set in `other`.
    #[inline(always)]
    pub const fn retain(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Returns `true` if `self` and `other` share any set bits.
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

/// A raw, non-null pointer into host memory.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct HostPointer(pub NonNull<AtomicU8>);

impl HostPointer {
    /// Wraps a raw pointer as a [`HostPointer`].
    pub const fn new(ptr: NonNull<AtomicU8>) -> Self {
        Self(ptr)
    }
}

unsafe impl Send for HostPointer {}
unsafe impl Sync for HostPointer {}

/// A pointer to the start of a page-aligned, page-sized region of host memory.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct PagePointer(HostPointer);

impl PagePointer {
    /// A sentinel, dangling page pointer.
    ///
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

    /// Returns the underlying non-null pointer to the start of the page.
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
                    .cast::<UninitPage>()
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

/// A resolved, tagged page pointer paired with the atomic dirty-flags byte
/// that tracks its dirty state.
#[derive(Copy, Clone)]
pub struct Page<'a> {
    /// The tagged pointer to the backing page, including its [`MemFlags`].
    pub ptr: TaggedPagePtr,
    /// The atomic byte tracking this page's dirty/flushing state.
    pub mutable_flags: &'a AtomicU8,
}

impl Page<'_> {
    /// Flag bit indicating a currently outbound flush.
    pub const IS_FLUSHING_DIRTY_PAGE: u8 = 0b10;
    /// Flag bit indicating the page has been written to since it was last clean.
    pub const IS_DIRTY_FLAG: u8 = 0b01;

    /// Marks this page as dirty if it is a page that requires dirty tracking.
    #[inline(always)]
    pub fn set_dirty(&self) {
        if self.ptr.flags().contains_any(MemFlags::MUST_DIRTY) {
            cold_path();
            self.mutable_flags
                .fetch_or(Self::IS_DIRTY_FLAG, Ordering::Release);
        }
    }
}

// use MaybeUninit<u8> so that it is explicitly not a zst and will always be a unique alloc
type IoMMUIdentPointee = MaybeUninit<u8>;
type IoMMUIdentifierInner = Arc<IoMMUIdentPointee>;

/// A unique, reference-counted token identifying an IOMMU address space.
#[derive(Eq, PartialEq)]
#[repr(transparent)]
pub struct IoMMUIdentifier(NonNull<IoMMUIdentPointee>);

unsafe impl ZeroableInOption for IoMMUIdentifier {}

impl IoMMUIdentifier {
    /// Creates a new, globally unique [`IoMMUIdentifier`].
    pub fn unique_token() -> Self {
        // note we use a byte to make sure that we are getting a new
        // `Arc` and that this isn't some cached ZST Arc
        let mut alloc: IoMMUIdentifierInner = Arc::<u8>::new_uninit();
        assert!(
            Arc::get_mut(&mut alloc).is_some(),
            "allocation is not unique"
        );
        let ptr = Arc::into_raw(alloc);
        Self(unsafe { NonNull::new_unchecked(ptr.cast_mut()) })
    }

    /// Borrows this identifier as a lightweight, `Copy`-able reference.
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

/// A borrowed, `Copy`-able reference to an [`IoMMUIdentifier`].
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct IoMMUIdentifierRef<'a> {
    ptr: NonNull<IoMMUIdentPointee>,
    _marker: PhantomData<&'a IoMMUIdentifier>,
}

impl IoMMUIdentifierRef<'_> {
    /// Clones the referenced identifier, producing an owned, ref-counted
    /// [`IoMMUIdentifier`].
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

    /// Returns the raw, type-erased pointer identifying this address space.
    #[inline(always)]
    pub fn ptr(self) -> NonNull<()> {
        self.ptr.cast()
    }
}

const _: () = {
    assert!(
        (MemFlags::ALL.bits() as u64 & !PAGE_OFFSET_MASK_U64) == 0,
        "MemFlags bits must fit in the low page-alignment bits"
    );
};

/// A virtual page number: a virtual address with the in-page offset bits
/// stripped off.
#[derive(Debug, Zeroable, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct PageNumber(u64);

impl PageNumber {
    /// The largest representable page number.
    pub const MAX: Self = Self::from_vaddr(u64::MAX);

    /// Splits a virtual address into its [`PageNumber`] and in-page offset.
    #[inline(always)]
    pub const fn from_vaddr_with_offset(vaddr: u64) -> (Self, usize) {
        let remainder = vaddr & PAGE_OFFSET_MASK_U64;
        // Safety: PAGE_SIZE fits in usize, and we are taking the remainder by PAGE_SIZE,
        //         therefore, the remainder **MUST** fit in a usize
        let remainder = unsafe { u64_to_usize(remainder).unwrap_unchecked() };

        (Self::from_vaddr(vaddr), remainder)
    }

    /// Computes the [`PageNumber`] containing the given virtual address.
    #[inline(always)]
    pub const fn from_vaddr(vaddr: u64) -> Self {
        Self(vaddr >> PAGE_SHIFT)
    }

    /// Creates a [`PageNumber`] from a raw page-number value.
    ///
    /// # Safety
    /// TODO
    #[inline(always)]
    pub const unsafe fn from_page_number_unchecked(page: u64) -> Self {
        unsafe { std::hint::assert_unchecked(page <= Self::MAX.0) }
        Self(page)
    }

    /// Creates a [`PageNumber`] from a raw page-number value, returning
    /// `None` if it exceeds [`Self::MAX`].
    #[inline]
    pub const fn from_page_number_checked(page: u64) -> Option<Self> {
        if page > Self::MAX.0 {
            return None;
        }

        Some(unsafe { Self::from_page_number_unchecked(page) })
    }

    /// Creates a [`PageNumber`] from a raw page-number value.
    ///
    /// # Panics
    ///
    /// Panics if `page` exceeds [`Self::MAX`].
    #[inline]
    pub const fn from_page_number(page: u64) -> Self {
        match Self::from_page_number_checked(page) {
            Some(page) => page,
            None => panic!("page out of bounds"),
        }
    }

    /// Returns the raw page-number value.
    #[inline(always)]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the virtual address of the first byte of this page.
    pub const fn vaddr_base(self) -> u64 {
        self.0 << PAGE_SHIFT
    }

    /// Returns the next page number, or `None` if this is [`Self::MAX`].
    #[inline(always)]
    pub const fn inc(self) -> Option<Self> {
        const { assert!(Self::MAX.0 != u64::MAX) }

        Self::from_page_number_checked(unsafe { self.0.unchecked_add(1) })
    }
}

const _: () = assert!(PageNumber::MAX.inc().is_none());

/// A [`PagePointer`] with its low, unused address bits repurposed to store
/// [`MemFlags`].
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct TaggedPagePtr(NonNull<AtomicU8>);

impl TaggedPagePtr {
    /// Creates a tagged pointer from a [`PagePointer`] and the [`MemFlags`]
    /// to tag it with.
    pub fn new(ptr: PagePointer, prot: MemFlags) -> Self {
        let prot: u8 = prot.bits();
        const { assert!((u8::MAX as usize) < PAGE_SIZE) }
        let ptr = ptr.as_non_null_ptr();
        let tag_bits = usize::from(prot);
        Self(ptr.map_addr(|addr| addr | tag_bits))
    }

    /// Returns the untagged [`PagePointer`], with the flag bits masked off.
    pub fn page_ptr(self) -> PagePointer {
        let ptr = self.0.map_addr(|addr| {
            let mask = !usize::from(MemFlags::ALL.bits());
            unsafe { NonZero::new_unchecked(addr.get() & mask) }
        });

        unsafe { PagePointer::new(ptr) }
    }

    /// Returns the [`MemFlags`] tagged into this pointer's low bits.
    pub fn flags(self) -> MemFlags {
        let mask: u8 = MemFlags::ALL.bits();
        let prot_usize = self.0.addr().get() & usize::from(mask);
        let prot_raw: u8 = unsafe { u8::try_from(prot_usize).unwrap_unchecked() };
        MemFlags(prot_raw)
    }
}

unsafe impl ZeroableInOption for TaggedPagePtr {}

/// An [`IoMMUIdentifier`] slot that may or may not currently hold a valid
/// identifier; a zeroed value represents "invalid".
#[derive(Zeroable)]
#[repr(transparent)]
pub struct MaybeInvalidIdentifier(Option<IoMMUIdentifier>);

impl MaybeInvalidIdentifier {
    /// Returns an invalid (empty) identifier slot.
    pub const fn invalid() -> Self {
        bytemuck::zeroed()
    }

    /// Wraps a valid [`IoMMUIdentifier`] into a slot.
    pub fn new(ident: IoMMUIdentifier) -> Self {
        Self(Some(ident))
    }

    /// Returns a raw, type-erased pointer identifying the held identifier
    /// (or a sentinel value if invalid).
    pub fn as_ptr(&self) -> *const () {
        unsafe { std::mem::transmute_copy::<Self, *const ()>(self) }
    }
}

/// A single TLB entry, mapping a virtual page number (within an IOMMU
/// address space) to a resolved, tagged host page.
///
/// # Safety
///
/// - If `tagged_page_ptr` is `Some`, then `insn_dirty_ptr` must also be `Some`
/// - If `io_mmu_ident` points to a live allocation, then `tagged_page_ptr` is `Some`
#[derive(Zeroable)]
pub struct TlbEntry {
    /// Identifier of the IOMMU address space this entry was resolved in.
    pub io_mmu_ident: MaybeInvalidIdentifier,
    /// The virtual page number this entry caches a translation for.
    pub virtual_page_number: PageNumber,
    /// The resolved, tagged host page pointer, if this entry is valid.
    pub tagged_page_ptr: Option<TaggedPagePtr>,
    /// The dirty-flags pointer for the resolved page, if this entry is valid.
    pub mut_page_flags: Option<NonNull<AtomicU8>>,
}

impl TlbEntry {
    /// Overwrites this entry with a fresh translation for `new_page_number`
    /// within the address space identified by `identifier`.
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
            mut_page_flags,
        } = self;

        let tagged_ptr = page.ptr;

        let new_mut_flags = NonNull::<AtomicU8>::from_ref(page.mutable_flags);

        if !std::ptr::addr_eq(tlb_identifier.as_ptr(), identifier.ptr().as_ptr()) {
            *tlb_identifier = MaybeInvalidIdentifier::new(identifier.clone_identifier());
        }

        *tagged_page_ptr = Some(tagged_ptr);
        *mut_page_flags = Some(new_mut_flags);
        *virtual_page_number = new_page_number;
    }
}

/// Number of entries in a [`Tlb`], as a `u64`. Reduced under `cfg(test)` to
/// make tests exercise TLB eviction more easily.
pub const TLB_SIZE_U64: u64 = match cfg!(test) {
    true => 64,
    false => 1024,
};

/// Number of entries in a [`Tlb`], as a `usize`.
pub const TLB_SIZE: usize = u64_to_usize(TLB_SIZE_U64).unwrap();
/// Bitmask used to index into a [`Tlb`] from a [`PageNumber`].
pub const TLB_MASK: u64 = compute_mask(TLB_SIZE_U64);

/// A direct-mapped translation-lookaside buffer caching recent virtual
/// page-number to host-page translations.
#[derive(Zeroable)]
#[repr(transparent)]
pub struct Tlb {
    /// The backing array of TLB entries.
    pub entries: [TlbEntry; TLB_SIZE],
}

impl Default for Tlb {
    fn default() -> Self {
        Self::new()
    }
}

impl Tlb {
    /// Allocates a new, zeroed [`Tlb`] on the heap directly, without
    /// constructing it on the stack first.
    pub fn new_boxed() -> Box<Self> {
        bytemuck::allocation::zeroed_box()
    }

    /// Creates a new, empty [`Tlb`].
    pub const fn new() -> Self {
        bytemuck::zeroed()
    }

    /// Returns the (direct-mapped) entry slot for the given page number.
    pub fn entry(&mut self, page_number: PageNumber) -> &mut TlbEntry {
        unsafe {
            let index = u64_to_usize(page_number.0 & TLB_MASK).unwrap_unchecked();
            self.entries.get_unchecked_mut(index)
        }
    }

    /// Looks up the translation for `page_num` in address space `ident`,
    /// calling `fallback` to resolve it on a cache miss and populating the
    /// entry with the result.
    ///
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
                mutable_flags: entry.mut_page_flags.unwrap_unchecked().as_ref(),
            }
        })
    }

    /// Populates the entry for `page_number` in address space `identifier`
    /// with `page`, unconditionally overwriting any existing entry.
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
