use std::mem::ManuallyDrop;
use anyhow::{anyhow, bail, ensure, Context};
use arrayvec::ArrayVec;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module};
use crate::ir::arena::ArenaMap;
use crate::ir::{ArithBinOp, ExecIr, OverflowingBinOp, IConst, IntWidth, LValue, StmtKind, Terminator, IntCmp, BitwiseOp, MAX_STMT_OUTPUTS, StmtData};
use crate::ir::{Block as IrBlock, Type as IrType};
use cranelift::codegen::ir as clif_ir;
use cranelift::prelude::{AbiParam, Configurable, InstBuilder, IntCC, MemFlags};
use cranelift::prelude::isa::OwnedTargetIsa;
use crate::ir::compiler::{CompileOptions, CompiledExecBlock, ExecBlockFFI};

struct FunctionLowering<'a> {
    builder: FunctionBuilder<'a>,
    ptr_ty: clif_ir::Type,

    live_ordered_blocks: &'a [IrBlock],
    values: ArenaMap<LValue, clif_ir::Value>,
    blocks: ArenaMap<IrBlock, clif_ir::Block>,
}

impl<'a> FunctionLowering<'a> {
    fn bind_entry_args(&mut self, entry_block: clif_ir::Block) -> anyhow::Result<()> {
        let params = self.builder.block_params(entry_block);
        let &[
        processor_state,
        pages,
        pages_count,
        halt_reason,
        io_mmu
        ] = params else {
            bail!(
                "internal compiler error: expected 5 entry params, got {}",
                params.len(),
            )
        };

        self.values.insert(LValue::ARG_PROCESSOR_STATE, processor_state);
        self.values.insert(LValue::ARG_PAGES, pages);
        self.values.insert(LValue::ARG_PAGE_COUNT, pages_count);
        self.values.insert(LValue::ARG_HALT_REASON_PTR, halt_reason);
        self.values.insert(LValue::ARG_IO_MMU, io_mmu);

        Ok(())
    }

    fn new(
        mut builder: FunctionBuilder<'a>,
        exec_ir: &'a ExecIr,
        ptr_ty: clif_ir::Type,
    ) -> anyhow::Result<Self> {
        let mut blocks = ArenaMap::with_capacity(exec_ir.blocks.len());

        for &ir_block in &exec_ir.block_compile_order {
            let clif_block = builder.create_block();

            if exec_ir.blocks[ir_block].is_cold {
                builder.set_cold_block(clif_block);
            }

            blocks.insert(ir_block, clif_block);
        }

        let entry_block = *blocks
            .get(IrBlock::ENTRYPOINT)
            .context("internal compiler error: missing entry block")?;

        builder.append_block_params_for_function_params(entry_block);

        let mut this = Self {
            builder,
            ptr_ty,
            live_ordered_blocks: &exec_ir.block_compile_order,
            values: ArenaMap::with_capacity(exec_ir.lvalues.len()),
            blocks,
        };

        this.bind_entry_args(entry_block)?;

        Ok(this)
    }

    fn use_value(&self, lvalue: LValue) -> anyhow::Result<clif_ir::Value> {
        self.values
            .get(lvalue)
            .copied()
            .context("internal compiler error: lvalue used before being lowered")
    }

    fn clif_block(&self, block: IrBlock) -> anyhow::Result<clif_ir::Block> {
        let res = self.blocks
            .get(block)
            .copied()
            .expect("internal compiler error: missing cranelift block");

        Ok(res)
    }


    fn int_ty(width: IntWidth) -> clif_ir::Type {
        match width {
            IntWidth::W8 => clif_ir::types::I8,
            IntWidth::W16 => clif_ir::types::I16,
            IntWidth::W32 => clif_ir::types::I32,
            IntWidth::W64 => clif_ir::types::I64,
        }
    }

    fn iconst(&mut self, iconst: IConst) -> clif_ir::Value {
        let bits = iconst.bits.cast_signed();
        let ty = Self::int_ty(iconst.width);
        self.builder.ins().iconst(ty, bits)
    }


    fn assert_value_ty(
        &self,
        value: clif_ir::Value,
        expected: clif_ir::Type,
        context: &'static str,
    ) -> anyhow::Result<()> {
        let found = self.builder.func.dfg.value_type(value);
        ensure!(
            found == expected,
            "{context} has wrong Cranelift type: expected {expected}, found {found}"
        );

        Ok(())
    }


    fn lower_stmt(&mut self, _exec_ir: &ExecIr, stmt: &StmtData) -> anyhow::Result<()> {
        let values = self.lower_rvalue(&stmt.rvalue)?;
        let outputs = stmt.outputs.as_slice();

        ensure!(
            outputs.len() == values.len(),
            "statement lowering produced mismatched values"
        );

        for (value, &lvalue) in values.into_iter().zip(outputs) {
            self.values.insert(lvalue, value);
        }

        Ok(())
    }

    fn host_memory_flags() -> MemFlags {
        MemFlags::trusted().with_aligned().with_can_move()
    }

    fn lower_host_load(
        &mut self,
        width: IntWidth,
        base_ptr: LValue,
        offset: usize
    ) -> anyhow::Result<clif_ir::Value> {
        let base_ptr = self.use_value(base_ptr)?;
        let offset = i32::try_from(offset)
            .context("internal compiler error host load offset too large")?;

        Ok(self.builder.ins().load(
            Self::int_ty(width),
            Self::host_memory_flags(),
            base_ptr,
            clif_ir::immediates::Offset32::new(offset)
        ))
    }

    fn lower_host_store(
        &mut self,
        base_ptr: LValue,
        offset: usize,
        value: clif_ir::Value
    ) -> anyhow::Result<()> {
        let base_ptr = self.use_value(base_ptr)?;

        let offset = i32::try_from(offset)
            .context("internal compiler error host load offset too large")?;

        self.builder.ins().store(
            Self::host_memory_flags(),
            value,
            base_ptr,
            clif_ir::immediates::Offset32::new(offset)
        );

        Ok(())
    }

    fn convert_to_intcc(cmp: IntCmp) -> IntCC {
        match cmp {
            IntCmp::Equal => IntCC::Equal,
            IntCmp::NotEqual => IntCC::NotEqual,
            IntCmp::SignedLessThan => IntCC::SignedLessThan,
            IntCmp::SignedGreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
            IntCmp::SignedGreaterThan => IntCC::SignedGreaterThan,
            IntCmp::SignedLessThanOrEqual => IntCC::SignedLessThanOrEqual,
            IntCmp::UnsignedLessThan => IntCC::UnsignedLessThan,
            IntCmp::UnsignedGreaterThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
            IntCmp::UnsignedGreaterThan => IntCC::UnsignedGreaterThan,
            IntCmp::UnsignedLessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
        }
    }

    fn lower_rvalue(
        &mut self,
        rvalue: &StmtKind,
    ) -> anyhow::Result<ArrayVec<clif_ir::Value, MAX_STMT_OUTPUTS>> {
        let value = match *rvalue {
            StmtKind::IConst(iconst) => array_helper::from_arr([self.iconst(iconst)]),

            StmtKind::ArithBinOp {
                op,
                lhs,
                rhs,
            } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();
                let value = match op {
                    ArithBinOp::Add => ins.iadd(lhs, rhs),
                    ArithBinOp::Sub => ins.isub(lhs, rhs),
                    ArithBinOp::Mul => ins.imul(lhs, rhs),
                    ArithBinOp::UncheckedUDiv => ins.udiv(lhs, rhs),
                    ArithBinOp::UncheckedSDiv => ins.sdiv(lhs, rhs),
                };

                array_helper::from_arr([value])
            }


            StmtKind::AddImm {
                value,
                imm64
            } => {
                let value = self.use_value(value)?;
                let new_value = self.builder.ins().iadd_imm(value, imm64.cast_signed());
                array_helper::from_arr([new_value])
            },


            StmtKind::IntCmp { cmp, lhs, rhs } => {
                let op = Self::convert_to_intcc(cmp);
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                array_helper::from_arr([self.builder.ins().icmp(op, lhs, rhs)])
            }

            StmtKind::IntCmpImm { cmp, lhs, rhs } => {
                let op = Self::convert_to_intcc(cmp);
                let lhs = self.use_value(lhs)?;
                let rhs = rhs.cast_signed();
                array_helper::from_arr([self.builder.ins().icmp_imm(op, lhs, rhs)])
            }

            StmtKind::Select { cond, if_true, if_false } => {
                let (cond, if_true, if_false) = (
                    self.use_value(cond)?,
                    self.use_value(if_true)?,
                    self.use_value(if_false)?,
                );

                array_helper::from_arr([self.builder.ins().select(cond, if_true, if_false)])
            }

            StmtKind::Bitwise { op, lhs, rhs } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();

                let value = match op {
                    BitwiseOp::And => ins.band(lhs, rhs),
                    BitwiseOp::Or => ins.bor(lhs, rhs),
                    BitwiseOp::Xor => ins.bxor(lhs, rhs),
                };

                array_helper::from_arr([value])
            }

            StmtKind::BitwiseImm { op, lhs, rhs } => {
                let lhs = self.use_value(lhs)?;
                let rhs = rhs.cast_signed();
                let ins = self.builder.ins();
                let value = match op {
                    BitwiseOp::And => ins.band_imm(lhs, rhs),
                    BitwiseOp::Or => ins.bor_imm(lhs, rhs),
                    BitwiseOp::Xor => ins.bxor_imm(lhs, rhs),
                };

                array_helper::from_arr([value])
            }

            StmtKind::OverflowingBinOp {
                op,
                lhs,
                rhs,
            } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();
                let (value, overflow) = match op {
                    OverflowingBinOp::Add => ins.sadd_overflow(lhs, rhs),
                    OverflowingBinOp::Sub => ins.ssub_overflow(lhs, rhs),
                };

                array_helper::from_arr([value, overflow])
            }

            StmtKind::LoadHost { width, base_ptr, offset } => {
                array_helper::from_arr([self.lower_host_load(width, base_ptr, offset)?])
            }

            StmtKind::StoreHost {
                base_ptr,
                offset,
                value,
            } => {
                let value = self.use_value(value)?;
                self.lower_host_store(base_ptr, offset, value)?;
                array_helper::from_arr([])
            }

            StmtKind::LoadHaltReason => {
                let ptr = self.use_value(LValue::ARG_HALT_REASON_PTR)?;
                let value = self.builder.ins().atomic_load(
                    clif_ir::types::I32,
                    Self::host_memory_flags(),
                    ptr
                );
                array_helper::from_arr([value])
            }

            StmtKind::Safepoint => array_helper::from_arr([])
        };

        Ok(value)
    }

    fn lower_terminator(&mut self, terminator: &Terminator) -> anyhow::Result<()> {
        match *terminator {
            Terminator::Return => {
                let zero = self.builder.ins().iconst(clif_ir::types::I32, 0);
                self.builder.ins().return_(&[zero]);
            }

            Terminator::ReturnCode { halt_reason } => {
                let halt_reason = self.use_value(halt_reason)?;
                self.assert_value_ty(
                    halt_reason,
                    clif_ir::types::I32,
                    "ReturnFail halt_reason"
                )?;
                self.builder.ins().return_(&[halt_reason]);
            }

            Terminator::Br(target) => {
                let target = self.clif_block(target)?;
                self.builder.ins().jump(target, &[]);
            }

            Terminator::BrNZ {
                cond,
                non_zero,
                zero,
            } => {
                let cond = self.use_value(cond)?;
                let cond_is_nonzero = self.int_nonzero(cond)?;

                let non_zero = self.clif_block(non_zero)?;
                let zero = self.clif_block(zero)?;

                self.builder
                    .ins()
                    .brif(cond_is_nonzero, non_zero, &[], zero, &[]);
            }
        }

        Ok(())
    }

    fn lower_blocks(
        &mut self,
        exec_ir: &ExecIr,
    ) -> anyhow::Result<()> {
        for &ir_block in self.live_ordered_blocks {
            let clif_block = self.clif_block(ir_block)?;

            self.builder.switch_to_block(clif_block);

            let block_data = &exec_ir.blocks[ir_block];

            for &stmt in &block_data.stmts {
                let stmt = &exec_ir.stmts[stmt];
                self.lower_stmt(exec_ir, stmt)?;
            }

            self.lower_terminator(&block_data.terminator)?;
        }

        Ok(())
    }


    fn int_nonzero(&mut self, value: clif_ir::Value) -> anyhow::Result<clif_ir::Value> {
        let ty = self.builder.func.dfg.value_type(value);
        ensure!(ty.is_int(), "BrNZ condition must be an integer value");
        Ok(self.builder.ins().icmp_imm(IntCC::NotEqual, value, 0))
    }

    #[allow(dead_code)]
    fn ir_ty_to_clif_ty(&self, ty: IrType) -> anyhow::Result<clif_ir::Type> {
        match ty {
            IrType::Bool => Ok(clif_ir::types::I8),
            IrType::Int(width) => Ok(Self::int_ty(width)),
            IrType::HostPtr => Ok(self.ptr_ty),
        }
    }

    fn finish(mut self) {
        self.builder.seal_all_blocks();
        self.builder.finalize();
    }
}


fn exec_block_signature(module: &JITModule) -> clif_ir::Signature {
    let ptr_ty = module.target_config().pointer_type();

    // this makes a signature with the target tripples default calling convention
    // which is basically the C calling convention
    let mut sig = module.make_signature();

    sig.params.push(AbiParam::new(ptr_ty));     // processor_state
    sig.params.push(AbiParam::new(ptr_ty));     // pages
    sig.params.push(AbiParam::new(clif_ir::types::I64)); // page_count
    sig.params.push(AbiParam::new(ptr_ty));     // halt_reason_ptr
    sig.params.push(AbiParam::new(ptr_ty));     // io_mmu

    sig.returns.push(AbiParam::new(clif_ir::types::I32));

    sig
}

pub use cranelift::codegen::settings::OptLevel;
use crate::array_helper;

pub struct CraneliftCompiler {
    isa: OwnedTargetIsa
}

impl CraneliftCompiler {
    pub fn new(opt_level: OptLevel) -> anyhow::Result<Self> {
        let mut flag_builder = cranelift::codegen::settings::builder();

        // JIT code is in-process and not being linked as a PIC object.
        flag_builder
            .set("is_pic", "false")
            .map_err(|err| anyhow!("Cranelift flag is_pic failed: {err}"))?;

        // This mirrors the common Cranelift JIT setup.
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|err| anyhow!("Cranelift flag use_colocated_libcalls failed: {err}"))?;


        // JIT code is in-process and not being linked as a PIC object.
        flag_builder.set("is_pic", "false")?;

        // On at least AArch64, "colocated" calls use shorter-range relocations,
        // which might not reach all definitions; we can't handle that here, so
        // we require long-range relocation types.
        flag_builder.set("use_colocated_libcalls", "false")?;

        flag_builder.set("preserve_frame_pointers", "false")?;

        flag_builder.set("opt_level", match opt_level {
            OptLevel::None => "none",
            OptLevel::Speed => "speed",
            OptLevel::SpeedAndSize => "speed_and_size",
        })?;

        let isa_builder = cranelift::native::builder()
            .map_err(|msg| anyhow!("host machine is not supported by Cranelift: {msg}"))?;

        let isa = isa_builder
            .finish(cranelift::codegen::settings::Flags::new(flag_builder))
            .map_err(|err| anyhow!("Cranelift ISA creation failed: {err}"))?;

        Ok(Self { isa })
    }

    pub fn try_compile(&self, options: CompileOptions, exec_ir: ExecIr) -> anyhow::Result<CompiledExecBlock> {
        let builder = JITBuilder::with_isa(
            self.isa.clone(),
            cranelift::module::default_libcall_names()
        );

        let mut module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let mut builder_ctx = FunctionBuilderContext::new();

        if options.show_disasm {
            ctx.set_disasm(true);
        }

        ctx.func.signature = exec_block_signature(&module);

        {
            let ptr_ty = module.target_config().pointer_type();
            let builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

            let mut lowering =
                FunctionLowering::new(builder, &exec_ir, ptr_ty)?;

            lowering.lower_blocks(&exec_ir)?;
            lowering.finish();
        }

        let func_id = module
            .declare_function(&options.function_name, Linkage::Export, &ctx.func.signature)
            .map_err(|err| anyhow!("declare_function failed: {err}"))?;

        module
            .define_function(func_id, &mut ctx)
            .map_err(|err| anyhow!("define_function failed: {err}"))?;


        if options.show_disasm {
            let code = ctx.compiled_code()
                .expect("Cranelift did not leave compiled code in the context");

            eprintln!("{}:", options.function_name);
            eprintln!("{}", code.vcode.as_deref().unwrap_or("no disassembly was produced"));
        }


        module.clear_context(&mut ctx);

        module
            .finalize_definitions()
            .map_err(|err| anyhow!("finalize_definitions failed: {err}"))?;

        let code_ptr = module.get_finalized_function(func_id);

        let ffi: ExecBlockFFI = unsafe {
            // Safety:
            // - `exec_block_signature` must exactly match `ExecBlockFFI`.
            // - `module` is moved into `resources`, keeping finalized code alive.
            std::mem::transmute::<*const u8, ExecBlockFFI>(code_ptr)
        };

        struct DropModule(ManuallyDrop<JITModule>);

        impl Drop for DropModule {
            fn drop(&mut self) {
                unsafe { ManuallyDrop::take(&mut self.0).free_memory() };
            }
        }

        Ok(CompiledExecBlock::new_with_recources(ffi, DropModule(ManuallyDrop::new(module))))
    }
}


