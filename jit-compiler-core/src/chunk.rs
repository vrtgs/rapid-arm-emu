//! The compiled, host-callable code chunk and the FFI signature it is called
//! through.
//!
//! A [`CompiledExecChunk`] pairs a raw function pointer with
//! the JIT resources that keep the emitted machine code alive: cloning a chunk
//! is inexpensive and shares those resources, and they are freed only once the last
//! clone is dropped. Run a chunk with [`CompiledExecChunk::call`], threading in
//! the guest [`ExecState`] and the per-CPU [`ExecContext`].

use crate::exec_context::ExecContext;
use crate::sync_cell::SyncCell;
use emu_abi::exec_state::ExecState;
use emu_abi::halt_reason::AtomicHaltReason;
use emu_abi::memory::{IoMMUIdentifierRef, Tlb};
use io_mmu::IoMMU;
use io_mmu::icache::ICache;
use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

// IMPORTANT NOTE TO IMPLEMENTORS this **MUST** be in sync with the ordering of `carte::ir::Arg`
pub(crate) type ExecChunkFFI = unsafe extern "C" fn(
    // same first 3 arguments of mmu fallback; this gives better regalloc
    io_mmu: &IoMMU<dyn ICache>,
    tlb: &mut Tlb,
    exec_context: &mut ExecContext,
    // ---
    exec_state: &mut ExecState,
    io_mmu_ident: IoMMUIdentifierRef<'_>,
    halt_reason_ptr: &AtomicU32,
) -> u32;

const _: () = assert!(size_of::<&IoMMU<dyn ICache>>() == size_of::<usize>());

/// A compiled, host-callable chunk of guest code.
///
/// Produced by [`ExecIrCompiler`](compiler::ExecIrCompiler); run it with
/// [`call`](Self::call). Cloning is inexpensive and keeps the underlying JIT
/// resources (the emitted machine code and its allocations) alive for as
/// long as any clone exists.
#[derive(Clone)]
pub struct CompiledExecChunk {
    ffi: ExecChunkFFI,

    // Keeps the JIT resources alive for at least as long as the fn pointer.
    // If this is dropped while `ffi` may still be called, we get very, very bad UB
    _resources: Arc<SyncCell<dyn Any + Send>>,
}

impl CompiledExecChunk {
    pub(crate) fn new_with_resources(ffi: ExecChunkFFI, resources: impl Any + Send) -> Self {
        Self {
            ffi,
            _resources: Arc::new(SyncCell::new(resources)),
        }
    }

    /// Runs the compiled chunk against the given guest CPU state and address
    /// space, returning the raw halt-reason code it exited with (`0` for a plain return).
    ///
    /// `context` must not carry a pending memory fault from a previous run;
    /// take it with [`FFISafeMemoryFault::take_memory_fault`] first,
    /// otherwise this panics/aborts.
    #[inline]
    pub fn call<T: ?Sized + ICache>(
        &self,
        exec_state: &mut ExecState,
        context: &mut ExecContext,
        tlb: &mut Tlb,
        halt_reason: &AtomicHaltReason,
        io_mmu: &IoMMU<T>,
    ) -> u32 {
        if cfg!(debug_assertions)
            && let Some(fault) = context.current_mem_fault.take_memory_fault()
        {
            let (fault, op) = fault;
            panic!("called compiled chunk with a pending memory fault {fault} from {op:?}");
        }

        let halt_reason = halt_reason.as_ffi();
        let (io_mmu_ident, io_mmu) = unsafe { io_mmu.as_ffi() };
        unsafe { (self.ffi)(&io_mmu, tlb, context, exec_state, io_mmu_ident, halt_reason) }
    }
}
