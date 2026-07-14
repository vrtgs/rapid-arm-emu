//! Lowering [`ExecIr`] to executable machine code.
//!
//! [`ExecIrCompiler`] is the entry point: it lazily initializes the
//! available JIT backends and compiles IR through the one selected by a
//! [`CompileTier`], yielding a runnable [`CompiledExecChunk`].

use crate::chunk::CompiledExecChunk;
use crate::compiler::cranelift_backend::CraneliftCompiler;
use crate::compiler::gcc_backend::GccJit;
use crate::compiler::llvm_backend::LLVMJit;
use crate::ir::ExecIr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

mod cranelift_backend;
mod gcc_backend;
mod llvm_backend;

/// Which JIT backend (and optimization level) to compile with.
///
/// Lower tiers compile faster and run slower; higher tiers cost more compile
/// time for better code.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum CompileTier {
    // the mystical Tier0, when there is an even faster backend
    /// Cranelift without optimizations: fastest compile time.
    Tier1,
    /// Cranelift with optimizations enabled.
    Tier2,

    /// The `libgccjit` backend.
    GccJit,
    /// The LLVM backend.
    LLVM,
}

struct CompileBlockOptions {
    function_name: String,
    show_disasm: bool,
}

/// Compiles [`ExecIr`] into [`CompiledExecChunk`]s.
///
/// Each backend is created lazily on first use and then reused for
/// later compilations. The compiler is `Send + Sync` and can be shared across threads.
pub struct ExecIrCompiler {
    next_function_id: AtomicUsize,
    cranelift_compiler: OnceLock<CraneliftCompiler>,
    gccjit: OnceLock<GccJit>,
    llvm: OnceLock<LLVMJit>,
    show_disasm: bool,
}

impl Default for ExecIrCompiler {
    fn default() -> Self {
        Self {
            next_function_id: AtomicUsize::new(0),
            cranelift_compiler: OnceLock::new(),
            gccjit: OnceLock::new(),
            llvm: OnceLock::new(),
            show_disasm: false,
        }
    }
}

impl ExecIrCompiler {
    /// Enables printing the disassembly of every compiled function for
    /// debugging the backends.
    pub fn with_show_disassembly(mut self) -> Self {
        self.show_disasm = true;
        self
    }

    /// Compiles `exec_ir` with the backend selected by `tier`.
    ///
    /// # Panics
    ///
    /// Panics if compilation fails; use [`try_compile`](Self::try_compile)
    /// to handle the error instead.
    pub fn compile(&self, exec_ir: &ExecIr, tier: CompileTier) -> CompiledExecChunk {
        self.try_compile(exec_ir, tier)
            .unwrap_or_else(|err| panic!("failed to compile ExecIr: {err}"))
    }

    fn cranelift_compiler(&self) -> &CraneliftCompiler {
        self.cranelift_compiler
            .get_or_init(|| CraneliftCompiler::new().unwrap())
    }

    fn gccjit(&self) -> &GccJit {
        self.gccjit.get_or_init(|| GccJit::new().unwrap())
    }

    fn llvm(&self) -> &LLVMJit {
        self.llvm.get_or_init(|| LLVMJit::new().unwrap())
    }

    /// Compiles `exec_ir` with the backend selected by `tier`, returning an
    /// error if backend initialization or compilation fails.
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
                self.cranelift_compiler()
                    .try_compile(options, exec_ir, optimized)
            }
            CompileTier::Tier2 => {
                let optimized = true;
                self.cranelift_compiler()
                    .try_compile(options, exec_ir, optimized)
            }
            CompileTier::GccJit => self.gccjit().try_compile(options, exec_ir),
            CompileTier::LLVM => self.llvm().try_compile(options, exec_ir),
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ExecIrCompiler>()
};
