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

    unsafe impl<T: Sized + ICache> DynUpgrade for T {
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

pub trait ICache: 'static + Send + Sync + sealed::DynUpgrade {
    fn invalidate(&self, page: PagePointer);
}
