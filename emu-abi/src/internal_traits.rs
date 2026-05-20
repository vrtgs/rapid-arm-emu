use crate::memory::PagePointer;
use std::mem::MaybeUninit;

/// This allows exposing things only inbetween the internal crates
pub trait AsFFI {
    type Interface<'a>
    where
        Self: 'a;

    fn as_ffi<'a>(&'a self) -> Self::Interface<'a>
    where
        Self: 'a;
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

pub trait CpuFabricPrivate {
    type ICache: ?Sized + ICache;

    fn icache(&self) -> &Self::ICache;
}
