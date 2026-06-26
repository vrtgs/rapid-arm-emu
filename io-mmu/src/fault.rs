#[derive(Debug, thiserror::Error)]
pub enum MemoryFaultReason {
    #[error("invalid memory permissions")]
    GeneralProtection,
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
    #[cold]
    #[inline(always)]
    pub const fn general_protection(vaddr: u64) -> Self {
        Self {
            vaddr,
            reason: MemoryFaultReason::GeneralProtection,
        }
    }

    #[cold]
    #[inline(always)]
    pub const fn memory_bus(vaddr: u64, reason: anyhow::Error) -> Self {
        Self {
            vaddr,
            reason: MemoryFaultReason::MemoryBus(reason),
        }
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
