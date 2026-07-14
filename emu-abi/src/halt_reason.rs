use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};

/// The reason why an emulated CPU core stopped execution.
///
/// Encoded as a packed `(opcode, metadata)` pair that fits in a single `u32`.
/// The `opcode` is always non-zero so that a zeroed word represents "no halt pending".
#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub struct HaltReason {
    /// Identifies the kind of halt (see `OPCODE_*` constants).
    pub opcode: NonZero<u16>,
    /// Additional data whose meaning depends on `opcode`.
    pub metadata: u16,
}

impl fmt::Debug for HaltReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.opcode {
            Self::OPCODE_FLUSH_INSN_CACHE => write!(f, "HaltReason::FlushInsnCache"),
            Self::OPCODE_INVALID_INSN => write!(f, "HaltReason::InvalidInsn"),
            Self::OPCODE_UNALIGNED_PC => write!(f, "HaltReason::UnalignedPc"),
            Self::OPCODE_MEMORY_TRAP => {
                write!(f, "HaltReason::MemoryTrap({} bytes)", self.metadata)
            }
            Self::OPCODE_IPI => write!(f, "HaltReason::Ipi({})", self.metadata),
            _ => write!(
                f,
                "HaltReason::Unknown({:#06x}, {})",
                self.opcode, self.metadata
            ),
        }
    }
}

const _: () = {
    assert!(size_of::<u32>() == 4);
    assert!(size_of::<HaltReason>() == 4);
    assert!(align_of::<HaltReason>() == 4);
};

impl HaltReason {
    /// Reinterprets this halt reason as a [`NonZero<u32>`] for atomic storage.
    pub const fn as_nz_u32(self) -> NonZero<u32> {
        unsafe { core::mem::transmute::<Self, NonZero<u32>>(self) }
    }

    /// Reconstructs a [`HaltReason`] from its [`NonZero<u32>`] representation.
    pub const fn from_u32(bits: NonZero<u32>) -> Self {
        unsafe { core::mem::transmute::<NonZero<u32>, Self>(bits) }
    }
}

impl PartialEq for HaltReason {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_nz_u32() == other.as_nz_u32()
    }
}

impl Eq for HaltReason {}

impl PartialOrd for HaltReason {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Self::cmp(self, other))
    }

    #[inline(always)]
    fn lt(&self, other: &Self) -> bool {
        self.as_nz_u32() < other.as_nz_u32()
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        self.as_nz_u32() <= other.as_nz_u32()
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> bool {
        self.as_nz_u32() > other.as_nz_u32()
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> bool {
        self.as_nz_u32() >= other.as_nz_u32()
    }
}

impl Ord for HaltReason {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        <NonZero<u32> as Ord>::cmp(&self.as_nz_u32(), &other.as_nz_u32())
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let this = self.as_nz_u32();
        let other = other.as_nz_u32();
        let max = <NonZero<u32> as Ord>::max(this, other);
        // SAFETY: The maximum of two trap codes is still a trap code.
        Self::from_u32(max)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let this = self.as_nz_u32();
        let other = other.as_nz_u32();
        let max = <NonZero<u32> as Ord>::min(this, other);
        // SAFETY: The minimum of two trap codes is still a trap code.
        Self::from_u32(max)
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        let this = self.as_nz_u32();
        let min = min.as_nz_u32();
        let max = max.as_nz_u32();
        let clamped = <NonZero<u32> as Ord>::clamp(this, min, max);
        // SAFETY: A trap code value clamped between two trap code values is still a trap code.
        Self::from_u32(clamped)
    }
}

impl Hash for HaltReason {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.as_nz_u32().get())
    }

    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        const {
            assert!(size_of::<Self>() == size_of::<u32>());
            assert!(align_of::<Self>() >= align_of::<u32>());
        }

        let data = unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<u32>(), data.len()) };

        <u32 as Hash>::hash_slice(data, state)
    }
}

impl HaltReason {
    /// Constructs a [`HaltReason`] with the given opcode and metadata.
    #[inline(always)]
    pub const fn new(opcode: NonZero<u16>, metadata: u16) -> Self {
        Self { opcode, metadata }
    }

    // Associated constants instead of module-level private consts
    /// Opcode for an invalid or undefined instruction trap.
    pub const OPCODE_INVALID_INSN: NonZero<u16> = NonZero::new(1).unwrap();
    /// Opcode requesting an instruction cache flush.
    pub const OPCODE_FLUSH_INSN_CACHE: NonZero<u16> = NonZero::new(2).unwrap();
    /// Opcode for an unaligned program counter fault.
    pub const OPCODE_UNALIGNED_PC: NonZero<u16> = NonZero::new(3).unwrap();
    /// Opcode for a memory access fault; `metadata` carries the access size in bytes.
    pub const OPCODE_MEMORY_TRAP: NonZero<u16> = NonZero::new(4).unwrap();
    /// Opcode for an inter-processor interrupt; `metadata` carries the IPI tag.
    pub const OPCODE_IPI: NonZero<u16> = NonZero::new(5).unwrap();

    /// Metadata tag identifying a synchronous IPI.
    pub const IPI_SYNC_TAG: u16 = 1;

    /// Halt reason requesting a full instruction cache flush.
    pub const FLUSH_INSN_CACHE: Self = Self::new(Self::OPCODE_FLUSH_INSN_CACHE, 0);

    /// Halt reason for an invalid or undefined instruction.
    pub const INVALID_INSN: Self = Self::new(Self::OPCODE_INVALID_INSN, 0);

    /// Halt reason for an unaligned program counter.
    pub const UNALIGNED_PC: Self = Self::new(Self::OPCODE_UNALIGNED_PC, 0);

    /// Creates a memory trap halt reason for an access of `access_size_in_bytes` bytes.
    pub const fn memory_trap(access_size_in_bytes: u8) -> Self {
        Self::new(Self::OPCODE_MEMORY_TRAP, access_size_in_bytes as u16)
    }

    /// Halt reason for a synchronous inter-processor interrupt.
    pub const IPI_SYNC: Self = Self::new(Self::OPCODE_IPI, Self::IPI_SYNC_TAG);
}

/// An atomically updated halt reason used to signal a CPU core from another thread.
pub struct AtomicHaltReason(AtomicU32);

impl Default for AtomicHaltReason {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicHaltReason {
    /// Creates a new [`AtomicHaltReason`] with no pending halt.
    pub const fn new() -> Self {
        Self(AtomicU32::new(0))
    }

    /// Stores `reason` as the pending halt, signalling the CPU core to stop.
    pub fn halt(&self, reason: HaltReason) {
        self.0.store(reason.as_nz_u32().get(), Ordering::Release)
    }

    /// Atomically takes and clears the pending halt reason.
    ///
    /// Returns `None` if no halt was pending.
    #[inline]
    pub fn take(&self) -> Option<HaltReason> {
        let bits = self.0.swap(0, Ordering::AcqRel);
        NonZero::new(bits).map(HaltReason::from_u32)
    }

    /// Signals the CPU to trap if it isn't already halting
    pub fn try_signal_sync(&self) -> bool {
        let trap = const { HaltReason::IPI_SYNC.as_nz_u32() };
        let res = self
            .0
            .compare_exchange(0, trap.get(), Ordering::AcqRel, Ordering::Acquire);

        res.is_ok()
    }
}

impl AtomicHaltReason {
    /// Returns a reference to the underlying [`AtomicU32`] for FFI use.
    pub const fn as_ffi(&self) -> &AtomicU32 {
        &self.0
    }
}
