use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use crate::armv9::ProcessorState;
use crate::halt_reason::AtomicHaltReason;
use crate::io_mmu;
use crate::io_mmu::IoMMU;
use crate::ir::compiler::cranelift_backend::CraneliftCompiler;
use crate::ir::ExecIr;
use crate::ir::compiler::sync_cell::SyncCell;

mod sync_cell;
mod cranelift_backend;
mod llvm_backend;
mod gcc_backend;

type ExecBlockFFI = unsafe extern "C" fn(
    processor_state: &mut ProcessorState,
    pages: *const io_mmu::Page,
    page_count: u64,
    halt_reason_ptr: *const AtomicU32,
    io_mmu: *const IoMMU,
) -> u32;

#[derive(Clone)]
pub(crate) struct CompiledExecBlock {
    ffi: ExecBlockFFI,

    // Keeps the JIT module alive for at least as long as the fn pointer.
    //
    // If this is dropped while `ffi` may still be called, we get very very bad UB
    _resources: Option<Arc<SyncCell<dyn Any + Send>>>,
}

impl CompiledExecBlock {
    fn new_with_recources(
        ffi: ExecBlockFFI,
        resources: impl Any + Send
    ) -> Self {
        Self {
            ffi,
            _resources: Some(Arc::new(SyncCell::new(resources)))
        }
    }

    #[inline]
    pub fn call(
        &self,
        processor_state: &mut ProcessorState,
        halt_reason: &AtomicHaltReason,
        io_mmu: &IoMMU,
    ) -> u32 {
        let halt_reason: *const AtomicU32 = halt_reason.as_ffi();
        let (pages, page_count) = io_mmu.pages_ffi();
        unsafe {
            (self.ffi)(
                processor_state,
                pages,
                page_count,
                halt_reason,
                io_mmu
            )
        }
    }
}


// currently we only support cranelift but that should change soon with LLVM support

pub struct ExecIrCompiler {
    next_function_id: AtomicUsize,
    cranelift_compiler: CraneliftCompiler,
}

impl ExecIrCompiler {
    pub fn new() -> Self {
        Self {
            next_function_id: AtomicUsize::new(0),
            cranelift_compiler: CraneliftCompiler::new(
                cranelift_backend::OptLevel::Speed
            ).unwrap(),
        }
    }

    pub fn compile(&self, exec_ir: ExecIr) -> CompiledExecBlock {
        self.try_compile(exec_ir)
            .unwrap_or_else(|err| panic!("failed to compile ExecIr with Cranelift: {err}"))
    }

    pub fn try_compile(&self, exec_ir: ExecIr) -> anyhow::Result<CompiledExecBlock> {
        let function_name = {
            let id = self.next_function_id.fetch_add(1, Ordering::Relaxed);
            format!("exec_block_{id}")
        };

        self.cranelift_compiler.try_compile(function_name, exec_ir)
    }
}