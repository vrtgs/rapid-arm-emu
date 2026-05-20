use crate::armv9::jit::CodeCache;
use crate::io_mmu::IoMMU;
use emu_abi::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
use emu_abi::processor_state::ProcessorState;
use parking_lot::Mutex;

pub(crate) mod jit;

struct ExecutingState {
    processor_state: ProcessorState,
    code_cache: CodeCache,
}

impl ExecutingState {
    fn resume(&mut self, cpu: &Armv9CpuCore) -> HaltReasonInner {
        self.code_cache.run(&mut self.processor_state, cpu)
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
                code_cache: CodeCache::new(),
            }),
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
        let mut lock = self
            .executing
            .try_lock()
            .unwrap_or_else(|| panic!("the CPU is already executing"));

        let state: &mut ExecutingState = &mut lock;
        loop {
            let halt_reason = match self.halt_reason.take() {
                reason if reason.is_empty() => state.resume(self),
                reason => reason,
            };

            debug_assert!(!halt_reason.is_empty());

            if halt_reason.contains(HaltReasonInner::InvalidateInsnCache) {
                state.code_cache.invalidate();
                self.mmu.flush_dirty_pages();
            }

            if let Some(reason) = HaltReason::from_inner(halt_reason) {
                break reason;
            }
        }
    }

    pub fn halt(&self, reason: HaltReason) {
        self.halt_reason.halt(reason)
    }
}
