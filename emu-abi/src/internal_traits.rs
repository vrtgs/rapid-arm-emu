use crate::memory::{IoMMUIdentifierRef, Page, PageNumber, PagePointer, Tlb};
use std::mem::MaybeUninit;

/// This allows exposing things only inbetween the internal crates
pub trait AsFFI {
    type Inetrface<'a>
    where
        Self: 'a;

    fn as_ffi<'a>(&'a self) -> Self::Inetrface<'a>
    where
        Self: 'a;
}

pub trait IoMMUByteRawAccess {
    type Error;

    fn load_byte_raw(&self, vaddr: u64) -> Result<(PageNumber, Page<'_>, u8), Self::Error>;

    fn store_byte_raw(&self, vaddr: u64, value: u8) -> Result<(PageNumber, Page<'_>), Self::Error>;
}

pub trait IoMMURawIntAccess<T: bytemuck::Pod>: IoMMUByteRawAccess {
    fn load_raw(
        &self,
        vaddr: u64,
    ) -> Result<(PageNumber, Page<'_>, Option<Page<'_>>, T), Self::Error>;

    fn store_raw(
        &self,
        vaddr: u64,
        value: T,
    ) -> Result<(PageNumber, Page<'_>, Option<Page<'_>>), Self::Error>;
}

/// # Safety
///
/// A type that can initialize itself directly inside caller-provided storage.
///
/// `InitInPlace::init` writes a valid `Self` into the provided `MaybeUninit<Self>`
/// and returns a mutable reference to the initialized value.
pub unsafe trait InitInPlace: Sized {
    /// Initializes `this` in place and returns a mutable reference to the
    /// initialized value.
    ///
    /// After this function returns normally, the memory referenced by `this`
    /// must contain a fully initialized, valid `Self`.
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self;
}

pub trait ICache {
    fn invalidate(&self, page: PagePointer);
}

pub trait IoMMUPrivate {
    type MemoryFault;

    /// # Safety
    ///
    /// must have already gotten access to an identifier from this same `self`
    /// and must not have modified this with any method that takes a mutable self
    unsafe fn get_ident_unchecked(&self) -> IoMMUIdentifierRef<'_>;

    fn get_page(&self, page_number: PageNumber) -> Result<Page<'_>, Self::MemoryFault>;

    fn fetch_aarch64_full(
        &self,
        vaddr: u64,
    ) -> Result<(PageNumber, Page<'_>, u32), Self::MemoryFault>;

    fn fetch_aarch64_with_tlb(&self, tlb: &mut Tlb, vaddr: u64) -> Result<u32, Self::MemoryFault>;
}

pub trait CpuFabricPrivate {
    type ICache: ?Sized + ICache;

    fn icache(&self) -> &Self::ICache;
}
