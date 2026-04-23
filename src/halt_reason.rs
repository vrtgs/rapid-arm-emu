use std::sync::atomic::{AtomicU32, Ordering};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub(crate) struct HaltReasonInner: u32 {
        // internal reasons
        const Step = 0x00000001;
        const InvalidateInstructionCache = 0x00000004;

        const MemoryFault = 0x00000004;
        const UserDefined1 = 0x01000000;
        const UserDefined2 = 0x02000000;
        const UserDefined3 = 0x04000000;
        const UserDefined4 = 0x08000000;
        const UserDefined5 = 0x10000000;
        const UserDefined6 = 0x20000000;
        const UserDefined7 = 0x40000000;
        const UserDefined8 = 0x80000000;
    }
}


bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct HaltReason: u32 {
        const MemoryFault = 0x00000004;
        const UserDefined1 = 0x01000000;
        const UserDefined2 = 0x02000000;
        const UserDefined3 = 0x04000000;
        const UserDefined4 = 0x08000000;
        const UserDefined5 = 0x10000000;
        const UserDefined6 = 0x20000000;
        const UserDefined7 = 0x40000000;
        const UserDefined8 = 0x80000000;
    }
}

impl HaltReason {
    pub(crate) fn into_inner(self) -> HaltReasonInner {
        // HaltReason is a superset of HaltReason
        HaltReasonInner::from_bits_retain(self.bits())
    }

    pub(crate) fn from_inner(reason: HaltReasonInner) -> HaltReason {
        Self::from_bits_truncate(reason.bits())
    }
}

pub(crate) struct AtomicHaltReason(AtomicU32);

impl AtomicHaltReason {
    pub fn new(reason: HaltReasonInner) -> Self {
        Self(AtomicU32::new(reason.bits()))
    }

    pub fn load(&self) -> HaltReasonInner {
        HaltReasonInner::from_bits_retain(self.0.load(Ordering::SeqCst))
    }

    pub fn add_reasons(&self, reason: HaltReasonInner) {
        self.0.fetch_or(reason.bits(), Ordering::SeqCst);
    }
}
