// use crate::compiler::{CompileBlockOptions, CompiledExecChunk};
// use crate::{ExecIr, SSAValue, StackSlot};
// use crate::arena::ArenaMap;
// use crate::compiler::llvm_backend::llvm_bindings::{Builder, FunctionValue, LLJit, LLVMContext, Module, OptimizationLevel, TargetTriple};

// TODO
// cpu_fabric llvm_bindings;

// struct FunctionLowering<'ctx, 'ir> {
//     ctx: &'ctx LLVMContext,
//     builder: Builder<'ctx>,
//     module: &'ir Module<'ctx>,
//     exec_ir: &'ir ExecIr,
//     func: FunctionValue<'ctx>,
//
//     values: ArenaMap<SSAValue, BasicValueEnum<'ctx>>,
//     phis: ArenaMap<SSAValue, inkwell::values::PhiValue<'ctx>>,
//     stack_slots: ArenaMap<StackSlot, PointerValue<'ctx>>,
// }

use crate::ExecIr;
use crate::compiler::{CompileBlockOptions, CompiledExecChunk};
use anyhow::bail;

pub(crate) struct LLVMJit {
    // target_triple: TargetTriple,
    // jit: LLJit,
}

impl LLVMJit {
    pub(crate) fn new() -> anyhow::Result<Self> {
        // llvm_bindings::init_llvm().and_then(|()| {
        //     Ok(Self {
        //         jit: LLJit::new()?,
        //         target_triple: TargetTriple::get_default_triple()?,
        //     })
        // })

        bail!("llvm jit still unimplemented")
    }

    pub(crate) fn try_compile(
        &self,
        options: CompileBlockOptions,
        exec_ir: &ExecIr,
    ) -> anyhow::Result<CompiledExecChunk> {
        let _ = (options, exec_ir);
        unreachable!()
    }
}
