use std::fmt::{Debug, Formatter};
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub(crate) struct HaltReasonInner: u32 {
        // internal reasons; these don't override, and instead stack
        // example one can step on an invalidate instruction cache instruction
        // in that case we stopped for 2 reasons; 1st of all we have successfully stepped
        // and second is that we invalidated the instruction cache
        const Step                = 0x00_00_00_01;
        const InvalidateInsnCache = 0x00_00_00_02;
        // the top 16 bits are for the trap's code
        // and the last bit represents the trap type
        const TrapBits            = 0xff_ff_0f_00;
    }
}



#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct HaltReason(NonZero<u32>);

impl Debug for HaltReason {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl HaltReason {
    pub(crate) fn into_inner(self) -> HaltReasonInner {
        todo!()
    }

    pub(crate) fn from_inner(_reason: HaltReasonInner) -> Option<HaltReason> {
        todo!()
    }
}

// TODO figure out the best memory ordering for this
//      seqcst is fine for now
pub(crate) struct AtomicHaltReason(AtomicU32);

impl AtomicHaltReason {
    pub fn new(reason: HaltReasonInner) -> Self {
        Self(AtomicU32::new(reason.bits()))
    }

    pub fn load(&self) -> HaltReasonInner {
        HaltReasonInner::from_bits_retain(self.0.load(Ordering::SeqCst))
    }

    pub fn add_reasons(&self, reason: HaltReasonInner) {
        // CAS loop
        // the lsb gets `or`ed and the top 24 bits get replaced
        let _ = reason;
        todo!()
    }
    
    
    pub fn take(&self) -> HaltReasonInner {
        HaltReasonInner::from_bits_retain(self.0.swap(0, Ordering::SeqCst))
    }
}
