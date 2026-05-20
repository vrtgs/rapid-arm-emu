use crate::internal_traits::AsFFI;
use std::num::NonZero;
use std::sync::atomic::{AtomicU32, Ordering};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct HaltReasonInner: u32 {
        // internal reasons; these don't override, and instead stack
        // example one can step on an invalidate instruction cache instruction
        // in that case we stopped for 2 reasons; 1st of all we have successfully stepped
        // and second is that we invalidated the instruction cache
        const Step                = 0x00_00_00_01;
        const InvalidateInsnCache = 0x00_00_00_02;
        // the top 16 bits are for the trap's code
        // and the second byte represents the trap opcode
        const TrapBits            = 0xFF_FF_FF_00;
    }
}

impl HaltReasonInner {
    fn merge_halt_reason(a: Self, b: Self) -> Self {
        let internal_reasons = (a.bits() | b.bits()) & 0xFF;
        let trap_mask = HaltReasonInner::TrapBits.bits();
        let a_trap = a.bits() & trap_mask;
        let b_trap = b.bits() & trap_mask;
        let trap = std::hint::select_unpredictable(a_trap != 0, a_trap, b_trap);

        Self::from_bits_retain(trap | internal_reasons)
    }
}

const INVALID_INSN: NonZero<u8> = NonZero::new(1).unwrap();
const UNALIGNED_PC: NonZero<u8> = NonZero::new(2).unwrap();
const MEMORY_TRAP: NonZero<u8> = NonZero::new(3).unwrap();

#[derive(Debug)] // TODO: better debug repr
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HaltReason {
    pub opcode: NonZero<u8>,
    pub payload: u16,
}

impl HaltReason {
    pub const fn new(opcode: NonZero<u8>, payload: u16) -> Self {
        Self { opcode, payload }
    }

    pub const INVALID_INSN: Self = Self {
        opcode: INVALID_INSN,
        payload: 0,
    };

    pub const UNALIGNED_PC: Self = Self {
        opcode: UNALIGNED_PC,
        payload: 0,
    };

    pub const MEMORY_TRAP: Self = Self {
        opcode: MEMORY_TRAP,
        payload: 0,
    };
}

impl HaltReason {
    pub const fn from_inner(reason: HaltReasonInner) -> Option<Self> {
        let bits = reason.bits();
        let Some(opcode) = NonZero::new(((bits >> 8) & 0xFF) as u8) else {
            return None;
        };
        let payload = (bits >> 16) as u16;
        Some(Self { opcode, payload })
    }

    pub const fn into_inner(self) -> HaltReasonInner {
        let bits = ((self.payload as u32) << 16) | ((self.opcode.get() as u32) << 8);
        HaltReasonInner::from_bits_retain(bits)
    }
}

pub struct AtomicHaltReason(AtomicU32);

impl Default for AtomicHaltReason {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicHaltReason {
    pub fn new() -> Self {
        Self(AtomicU32::new(0))
    }

    #[inline]
    pub fn add_reasons_full(&self, reason: HaltReasonInner) -> HaltReasonInner {
        // CAS loop
        // the lsb gets `or`ed and the top 24 bits get replaced
        let bits = self.0.update(Ordering::Release, Ordering::Relaxed, |bits| {
            let new_reason =
                HaltReasonInner::merge_halt_reason(reason, HaltReasonInner::from_bits_retain(bits));

            new_reason.bits()
        });

        HaltReasonInner::from_bits_retain(bits)
    }

    pub fn halt(&self, reason: HaltReason) {
        self.add_reasons_full(reason.into_inner());
    }

    pub fn take(&self) -> HaltReasonInner {
        HaltReasonInner::from_bits_retain(self.0.swap(0, Ordering::AcqRel))
    }
}

impl AsFFI for AtomicHaltReason {
    type Inetrface<'a> = &'a AtomicU32;

    fn as_ffi<'a>(&'a self) -> Self::Inetrface<'a>
    where
        Self: 'a,
    {
        &self.0
    }
}
