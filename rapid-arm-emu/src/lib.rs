mod a64;
pub mod armv9;

pub mod halt_reason {
    pub use ::emu_abi::halt_reason::HaltReason;
}

pub mod io_mmu {
    pub use ::io_mmu::*;

    use emu_abi::internal_traits::ICache;
    use emu_abi::memory::PagePointer;

    pub struct InsnCache {}

    impl ICache for InsnCache {
        fn invalidate(&self, _page: PagePointer) {
            todo!()
        }
    }

    pub type IoMMU = io_mmu::IoMMU<InsnCache>;
}
