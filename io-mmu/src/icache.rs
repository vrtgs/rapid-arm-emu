//! Instruction-cache invalidation callbacks.
//!
//! The MMU notifies the [`ICache`] attached to its
//! [`CpuFabric`](crate::cpu_fabric::CpuFabric) whenever the contents of an
//! executable page may have changed or when a page is freed and no longer in use,
//! so cached translations (e.g., JIT compiled blocks) can be discarded.

use emu_abi::memory::PagePointer;

mod sealed {
    use crate::icache::ICache;

    /// # Safety
    ///
    /// must provide a pointer with the metadata of self without panicking
    pub unsafe trait DynUpgrade {
        extern "C" fn get_metadata_ptr<'a>(&self) -> *const (dyn ICache + 'a)
        where
            Self: 'a;
    }

    unsafe impl<T: ICache> DynUpgrade for T {
        // Note: we use `extern "C"` to mark the function no_unwind
        #[allow(improper_ctypes_definitions)]
        extern "C" fn get_metadata_ptr<'a>(&self) -> *const (dyn ICache + 'a)
        where
            T: 'a,
        {
            self
        }
    }
}

/// An instruction cache that must be notified when executable memory changes.
///
/// Implementors are attached to a [`CpuFabric`](crate::cpu_fabric::CpuFabric)
/// and receive [`invalidate`](Self::invalidate) callbacks whenever the MMU
/// determines that a page's contents may no longer match what the cache has
/// translated from it.
pub trait ICache: 'static + Send + Sync + sealed::DynUpgrade {
    /// Invalidates any cached state derived from the contents of `page`.
    fn invalidate(&self, page: PagePointer);
}
