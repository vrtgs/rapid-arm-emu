use crate::ExecIr;
use crate::compiler::cranelift_backend::CraneliftCompiler;
use crate::compiler::sync_cell::SyncCell;
use emu_abi::halt_reason::AtomicHaltReason;
use emu_abi::internal_traits::{AsFFI, ICache};
use emu_abi::memory::{IoMMUIdentifierRef, Tlb};
use emu_abi::processor_state::ProcessorState;
use io_mmu::IoMMU;
use std::any::Any;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

mod cranelift_backend;
mod gcc_backend;
mod llvm_backend;
mod sync_cell;

type ExecBlockFFI = for<'a> unsafe extern "C" fn(
    processor_state: &mut ProcessorState,
    tlb_entries: &mut Tlb,
    io_mmu_ident: IoMMUIdentifierRef<'a>,
    halt_reason_ptr: &AtomicU32,
    io_mmu: &IoMMU<dyn ICache + 'a>,
) -> u32;

const _: () = assert!(size_of::<&IoMMU<dyn ICache>>() == size_of::<usize>());

#[derive(Clone)]
pub struct CompiledExecChunk {
    ffi: ExecBlockFFI,

    // Keeps the JIT resources alive for at least as long as the fn pointer.
    // If this is dropped while `ffi` may still be called, we get very very bad UB
    _resources: Arc<SyncCell<dyn Any + Send>>,
}

impl CompiledExecChunk {
    fn new_with_recources(ffi: ExecBlockFFI, resources: impl Any + Send) -> Self {
        Self {
            ffi,
            _resources: Arc::new(SyncCell::new(resources)),
        }
    }

    // TODO add a test to make sure an IoMMU<dyn ICache> can infact be used to call
    //      a compiled chunk
    #[inline]
    pub fn call<'a, T: ?Sized + ICache + 'a>(
        &self,
        processor_state: &mut ProcessorState,
        tlb: &mut Tlb,
        halt_reason: &AtomicHaltReason,
        io_mmu: &'a IoMMU<T>,
    ) -> u32
    where
        IoMMU<T>:
            AsFFI<Interface<'a> = (IoMMUIdentifierRef<'a>, ManuallyDrop<IoMMU<dyn ICache + 'a>>)>,
    {
        let halt_reason = halt_reason.as_ffi();
        let (io_mmu_ident, io_mmu) = io_mmu.as_ffi();
        unsafe { (self.ffi)(processor_state, tlb, io_mmu_ident, halt_reason, &io_mmu) }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum CompileTier {
    // the mystical Tier0, when there is an even faster backend
    Tier1,
    Tier2,
}

// currently we only support cranelift but that should change soon with LLVM support
struct CompileBlockOptions {
    function_name: String,
    show_disasm: bool,
}

pub struct ExecIrCompiler {
    next_function_id: AtomicUsize,
    cranelift_compiler: CraneliftCompiler,
    show_disasm: bool,
}

impl Default for ExecIrCompiler {
    fn default() -> Self {
        Self {
            next_function_id: AtomicUsize::new(0),
            cranelift_compiler: CraneliftCompiler::new().unwrap(),
            show_disasm: false,
        }
    }
}

impl ExecIrCompiler {
    pub fn with_show_disassmbly(mut self) -> Self {
        self.show_disasm = true;
        self
    }

    pub fn compile(&self, exec_ir: &ExecIr, tier: CompileTier) -> CompiledExecChunk {
        self.try_compile(exec_ir, tier)
            .unwrap_or_else(|err| panic!("failed to compile ExecIr: {err}"))
    }

    pub fn try_compile(
        &self,
        exec_ir: &ExecIr,
        tier: CompileTier,
    ) -> anyhow::Result<CompiledExecChunk> {
        let function_name = {
            let id = self.next_function_id.fetch_add(1, Ordering::Relaxed);
            format!("exec_chunk_{id}")
        };

        let options = CompileBlockOptions {
            function_name,
            show_disasm: self.show_disasm,
        };

        match tier {
            CompileTier::Tier1 => {
                let optimized = false;
                self.cranelift_compiler
                    .try_compile(options, exec_ir, optimized)
            }
            CompileTier::Tier2 => {
                let optimized = true;
                self.cranelift_compiler
                    .try_compile(options, exec_ir, optimized)
            }
        }
    }
}
