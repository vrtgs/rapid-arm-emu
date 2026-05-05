use parking_lot::Mutex;
use crate::armv9::jit::CodeCache;
use crate::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
use crate::io_mmu::IoMMU;

pub(crate) mod jit;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(16))]
pub struct Vector(pub u128);

const _: () = assert!(align_of::<Vector>() == 16 && size_of::<Vector>() == 16);

pub(crate) const X_REGISTER_COUNT: u8 = 31;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub(crate) struct PState(pub(crate) u32);

impl PState {
    pub(crate) const NEGATIVE: Self = Self(1 << 31);
    pub(crate) const ZERO: Self = Self(1 << 30);
    pub(crate) const CARRY: Self = Self(1 << 29);
    pub(crate) const OVERFLOW: Self = Self(1 << 28);


    pub(crate) const N: Self = Self::NEGATIVE;
    pub(crate) const Z: Self = Self::ZERO;
    pub(crate) const C: Self = Self::CARRY;
    pub(crate) const V: Self = Self::OVERFLOW;


    pub(crate) const NZCV_MASK: Self = Self(Self::N.0 | Self::Z.0 | Self::C.0 | Self::V.0);
}

pub(crate) struct ProcessorState {
    pub(crate) sp: u64,
    pub(crate) pc: u64,
    pub(crate) x_registers: [u64; X_REGISTER_COUNT as usize],
    pub(crate) pstate: PState,
    pub(crate) fpsr: u32,
    pub(crate) fpcr: u32,
    pub(crate) vectors: [Vector; 32],
}

impl ProcessorState {
    pub fn initial() -> Self {
        Self {
            sp: 0,
            pc: 0,
            x_registers: [0; X_REGISTER_COUNT as usize],
            pstate: PState(0),
            fpsr: 0,
            fpcr: 0,
            vectors: [Vector(0); 32],
        }
    }
}

struct ExecutingState {
    processor_state: ProcessorState,
    code_cache: CodeCache,
}

impl ExecutingState {
    fn resume(&mut self, cpu: &Armv9CpuCore) -> HaltReasonInner {
        self.code_cache.run(&mut self.processor_state, cpu)
    }

    fn invalidate_instruction_cache(&mut self, cpu: &Armv9CpuCore) {
        for dirty_range in cpu.mmu.drain_dirty_icache() {
            self.code_cache.invalidate_cache(dirty_range)
        }
    }
}

pub struct Armv9CpuCore {
    mmu: IoMMU,
    halt_reason: AtomicHaltReason,
    executing: Mutex<ExecutingState>,
}

impl Armv9CpuCore {
    pub fn new(mmu: IoMMU) -> Self {
        Self {
            mmu,
            halt_reason: AtomicHaltReason::new(),
            executing: Mutex::new(ExecutingState {
                processor_state: ProcessorState::initial(),
                code_cache: CodeCache::new()
            })
        }
    }

    pub fn mmu(&self) -> &IoMMU {
        &self.mmu
    }

    pub fn mmu_mut(&mut self) -> &mut IoMMU {
        &mut self.mmu
    }

    /// Runs the emulated CPU.
    /// Cannot be recursively called.
    pub fn resume(&self) -> HaltReason {
        let mut lock = self.executing.try_lock()
            .unwrap_or_else(|| panic!("the CPU is already executing"));

        let state: &mut ExecutingState = &mut lock;
        loop {
            let halt_reason = match self.halt_reason.take() {
                reason if reason.is_empty() => state.resume(self),
                reason => reason,
            };

            debug_assert!(!halt_reason.is_empty());

            if halt_reason.contains(HaltReasonInner::InvalidateInsnCache) {
                state.invalidate_instruction_cache(self);
            }

            if let Some(reason) = HaltReason::from_inner(halt_reason) {
                break reason
            }
        }
    }

    pub fn halt(&self, reason: HaltReason) {
        self.halt_reason.add_reasons(reason.into_inner())
    }
}
