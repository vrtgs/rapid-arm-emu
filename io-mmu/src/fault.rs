//! Memory fault types returned by failed MMU operations.
//!
//! Every fallible translation or access in this crate reports failure as a
//! [`MemoryFault`], which pairs the faulting virtual address with a
//! [`MemoryFaultReason`].

/// The reason a [`MemoryFault`] was raised.
#[derive(Debug, thiserror::Error)]
pub enum MemoryFaultReason {
    /// The access violated MMU validation: an unmapped page, insufficient
    /// page permissions, address overflow, or a failed alignment check.
    #[error("invalid memory permissions")]
    GeneralProtection,
    /// The backing [`MemoryObject`](crate::memory_object::MemoryObject)
    /// reported an error while faulting a page in or out.
    #[error("invalid memory access {0}")]
    MemoryBus(anyhow::Error),
}

/// This is returned when a memory access:
/// - Targets an unmapped page,
/// - Violates page permissions,
/// - Overflows the virtual address range,
/// - Fails an address-alignment check required by a specific operation,
/// - Crosses into an unmapped or insufficiently-permitted page,
/// - Or otherwise fails MMU validation.
#[derive(Debug, thiserror::Error)]
#[error("memory fault at {vaddr}: {reason}")]
pub struct MemoryFault {
    vaddr: u64,
    reason: MemoryFaultReason,
}

impl MemoryFault {
    /// Creates a [`MemoryFaultReason::GeneralProtection`] fault at `vaddr`.
    #[cold]
    #[inline(always)]
    pub const fn general_protection(vaddr: u64) -> Self {
        Self {
            vaddr,
            reason: MemoryFaultReason::GeneralProtection,
        }
    }

    /// Creates a [`MemoryFaultReason::MemoryBus`] fault at `vaddr` carrying
    /// the underlying backing-object error.
    #[cold]
    #[inline(always)]
    pub const fn memory_bus(vaddr: u64, reason: anyhow::Error) -> Self {
        Self {
            vaddr,
            reason: MemoryFaultReason::MemoryBus(reason),
        }
    }

    /// Returns the virtual address at which the fault occurred.
    #[inline(always)]
    pub const fn vaddr(&self) -> u64 {
        self.vaddr
    }

    /// Consumes the fault, returning the underlying reason for the memory fault.
    #[inline(always)]
    pub fn into_reason(self) -> MemoryFaultReason {
        self.reason
    }
}

macro_rules! ensure {
    (vaddr: $vaddr: expr, $($expr: expr),+ $(,)?) => {
        if !($({ $expr })&&+) {
            ::std::hint::cold_path();
            return Err(MemoryFault::general_protection($vaddr))
        }
    };
}

pub(crate) use ensure;
