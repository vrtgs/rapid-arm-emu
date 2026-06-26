use crate::address_space::IoMMU;
use crate::armv9::jit::CodeCache;
use emu_abi::exec_state::ExecState;
use emu_abi::halt_reason::{AtomicHaltReason, HaltReason};
use parking_lot::Mutex;

pub(crate) mod jit;

struct ExecutingState {
    processor_state: ExecState,
    code_cache: CodeCache,
}

impl ExecutingState {
    fn resume(&mut self, cpu: &Armv9CpuCore) -> Option<HaltReason> {
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
                processor_state: ExecState::initial(),
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
                None => state.resume(self),
                reason => reason,
            };

            if let Some(reason) = halt_reason {
                if reason == HaltReason::FLUSH_INSN_CACHE {
                    state.code_cache.invalidate();
                    self.mmu.refresh();
                    continue;
                }

                break reason;
            }
        }
    }

    pub fn halt(&self, reason: HaltReason) {
        self.halt_reason.halt(reason)
    }
}
