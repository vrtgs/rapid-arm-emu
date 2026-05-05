use std::collections::HashSet;
use std::mem::{offset_of, ManuallyDrop};
use anyhow::{anyhow, bail, ensure, Context};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module};
use crate::ir::arena::ArenaMap;
use crate::ir::{ArithBinOp, ExecIr, ExecIrBuilder, FlagSettingBinOp, IConst, IntWidth, LValue, RValue, Stmt, Terminator};
use crate::ir::{Block as IrBlock, Type as IrType};
use cranelift::codegen::ir as clif_ir;
use cranelift::prelude::{AbiParam, Configurable, InstBuilder, IntCC, MemFlags};
use cranelift::prelude::isa::{CallConv, OwnedTargetIsa};
use crate::armv9::{PState, ProcessorState};
use crate::ir::compiler::{CompiledExecBlock, ExecBlockFFI};

// does 2 jobs:
// 1. do it in reverse post order, i.e. topo sort
// 2. dead code elimination (very minimal)
fn reverse_post_order_exec_ir(exec_ir: &ExecIr) -> Vec<IrBlock> {
    #[derive(Debug, Copy, Clone)]
    enum DfsFrame {
        Enter(IrBlock),
        Exit(IrBlock),
    }

    let mut seen = HashSet::new();
    let mut postorder = Vec::with_capacity(exec_ir.blocks.len());

    let mut dfs_stack = vec![DfsFrame::Enter(IrBlock::ENTRYPOINT)];

    while let Some(frame) = dfs_stack.pop() {
        match frame {
            DfsFrame::Enter(block) => {
                if !seen.insert(block) {
                    continue;
                }

                dfs_stack.push(DfsFrame::Exit(block));

                let block_terminator = &exec_ir.blocks[block].terminator;

                // Stack is LIFO, so reverse the target order when pushing.
                //
                // If terminator_targets yields [non_zero, zero], this preserves
                // recursive DFS behavior: non_zero is visited before zero.
                for target in ExecIrBuilder::terminator_targets(block_terminator).rev() {
                    dfs_stack.push(DfsFrame::Enter(target));
                }
            }

            DfsFrame::Exit(block) => {
                assert!(postorder.len() < exec_ir.blocks.len());
                postorder.push(block);
            }
        }
    }

    assert!(postorder.len() <= exec_ir.blocks.len());

    postorder.reverse();
    postorder
}

struct FunctionLowering<'a> {
    builder: FunctionBuilder<'a>,
    ptr_ty: clif_ir::Type,

    live_ordered_blocks: &'a [IrBlock],
    values: ArenaMap<LValue, clif_ir::Value>,
    blocks: ArenaMap<IrBlock, clif_ir::Block>,
}

fn int_min_imm(ty: clif_ir::Type) -> i64 {
    let bits = ty.bits();
    assert!(ty.is_int());
    assert!(bits <= 64, "this helper handles scalar ints up to i64");

    match bits {
        64 => i64::MIN,
        _ => 1_i64.strict_shl(bits.strict_sub(1)).strict_neg()
    }
}

fn emit_udiv(
    builder: &mut FunctionBuilder<'_>,
    lhs: clif_ir::Value,
    rhs: clif_ir::Value
) -> clif_ir::Value {
    let ty = builder.func.dfg.value_type(lhs);

    let zero = builder.ins().iconst(ty, 0);
    let one = builder.ins().iconst(ty, 1);

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);

    // `select` is not lazy, so make the divisor safe before dividing.
    let safe_rhs = builder.ins().select(rhs_is_zero, one, rhs);

    let quotient = builder.ins().udiv(lhs, safe_rhs);

    // rhs == 0 produces 0.
    builder.ins().select(rhs_is_zero, zero, quotient)
}

fn emit_sdiv(
    builder: &mut FunctionBuilder<'_>,
    lhs: clif_ir::Value,
    rhs: clif_ir::Value,
) ->  clif_ir::Value {
    let ty = builder.func.dfg.value_type(lhs);

    let zero = builder.ins().iconst(ty, 0);
    let one = builder.ins().iconst(ty, 1);
    let int_min = builder.ins().iconst(ty, int_min_imm(ty));

    let rhs_is_zero = builder.ins().icmp_imm(IntCC::Equal, rhs, 0);

    let lhs_is_min = builder.ins().icmp(IntCC::Equal, lhs, int_min);
    let rhs_is_minus_one = builder.ins().icmp_imm(IntCC::Equal, rhs, -1);
    let is_overflow = builder.ins().band(lhs_is_min, rhs_is_minus_one);

    // Avoid both Cranelift trap cases:
    //   rhs == 0
    //   lhs == INT_MIN && rhs == -1
    let safe_rhs_for_zero = builder.ins().select(rhs_is_zero, one, rhs);
    let safe_rhs = builder.ins().select(is_overflow, one, safe_rhs_for_zero);

    let quotient = builder.ins().sdiv(lhs, safe_rhs);

    // INT_MIN / -1 should produce INT_MIN.
    // Since safe_rhs is 1 in the overflow case, quotient is already lhs,
    // but this makes the intended semantics explicit.
    let quotient = builder.ins().select(is_overflow, lhs, quotient);

    // rhs == 0 should produce 0.
    builder.ins().select(rhs_is_zero, zero, quotient)
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
        exec_ir: &ExecIr,
        live_ordered_blocks: &'a [IrBlock],
        ptr_ty: clif_ir::Type,
    ) -> anyhow::Result<Self> {
        let mut blocks = ArenaMap::with_capacity(exec_ir.blocks.len());

        for &ir_block in live_ordered_blocks {
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
            live_ordered_blocks,
            values: ArenaMap::with_capacity(exec_ir.lvalues.len()),
            blocks,
        };

        this.bind_entry_args(entry_block)?;

        Ok(this)
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
        match iconst {
            IConst::U8(value) => self.builder.ins().iconst(clif_ir::types::I8, value as i64),
            IConst::U16(value) => self.builder.ins().iconst(clif_ir::types::I16, value as i64),
            IConst::U32(value) => self.builder.ins().iconst(clif_ir::types::I32, value as i64),
            IConst::U64(value) => self.builder.ins().iconst(clif_ir::types::I64, value.cast_signed()),
        }
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


    fn lower_stmt(&mut self, exec_ir: &ExecIr, stmt: &Stmt) -> anyhow::Result<()> {
        let dst_ty = exec_ir.lvalues[stmt.lvalue].ty;

        let value = self.lower_rvalue(exec_ir, stmt.lvalue, dst_ty, &stmt.rvalue)?;

        match (dst_ty, value) {
            (IrType::Unit, None) => Ok(()),

            (IrType::Unit, Some(_)) => bail!(
                "statement lowering produced a value for a unit-typed lvalue"
            ),

            (_, Some(value)) => {
                self.values.insert(stmt.lvalue, value);
                Ok(())
            }

            (_, None) => bail!("statement lowering produced no value for a non-unit lvalue"),
        }
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
        let base_ptr = self.values[base_ptr];
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
        let base_ptr = self.values[base_ptr];

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

    fn lower_set_nzcv_flag_set(
        &mut self,
        n: clif_ir::Value,
        z: clif_ir::Value,
        c: clif_ir::Value,
        v: clif_ir::Value,
    ) -> anyhow::Result<()> {
        let base_ptr = LValue::ARG_PROCESSOR_STATE;
        let offset = offset_of!(ProcessorState, pstate);

        let old_flags = self.lower_host_load(IntWidth::W32, base_ptr, offset)?;

        let mut u32_const =
            |x: u32| self.builder.ins().iconst(clif_ir::types::I32, i64::from(x));

        let zeroed = u32_const(0);

        let n_flag_true = u32_const(PState::N.0);
        let z_flag_true = u32_const(PState::Z.0);
        let c_flag_true = u32_const(PState::C.0);
        let v_flag_true = u32_const(PState::V.0);

        let n_flag = self.builder.ins().select(n, n_flag_true, zeroed);
        let z_flag = self.builder.ins().select(z, z_flag_true, zeroed);
        let c_flag = self.builder.ins().select(c, c_flag_true, zeroed);
        let v_flag = self.builder.ins().select(v, v_flag_true, zeroed);

        let nz_flag = self.builder.ins().bor(n_flag, z_flag);
        let cv_flag = self.builder.ins().bor(c_flag, v_flag);
        let nzcv_flags = self.builder.ins().bor(nz_flag, cv_flag);

        let masked_out_flags = self.builder.ins().band_imm(old_flags, i64::from(!PState::NZCV_MASK.0));
        let new_flags = self.builder.ins().bor(masked_out_flags, nzcv_flags);

        self.lower_host_store(base_ptr, offset, new_flags)?;

        Ok(())
    }

    fn lower_rvalue(
        &mut self,
        _exec_ir: &ExecIr,
        _dst: LValue,
        _dst_ty: IrType,
        rvalue: &RValue,
    ) -> anyhow::Result<Option<clif_ir::Value>> {
        let value = match *rvalue {
            RValue::IConst(iconst) => Some(self.iconst(iconst)),

            RValue::ArithBinOp {
                op,
                lhs,
                rhs,
            } => {
                let (lhs, rhs) = (self.values[lhs], self.values[rhs]);
                let ins = self.builder.ins();

                Some(match op {
                    ArithBinOp::Add => ins.iadd(lhs, rhs),
                    ArithBinOp::Sub => ins.isub(lhs, rhs),
                    ArithBinOp::Mul => ins.imul(lhs, rhs),
                    ArithBinOp::UDiv => emit_udiv(&mut self.builder, lhs, rhs),
                    ArithBinOp::SDiv => emit_sdiv(&mut self.builder, lhs, rhs),
                })
            }

            RValue::FlagSettingBinOp {
                op,
                lhs,
                rhs,
            } => {
                let (lhs, rhs) = (self.values[lhs], self.values[rhs]);

                let builder = &mut self.builder;

                let (value, c, v) = match op {
                    FlagSettingBinOp::Add => {
                        // signed overflow flag
                        let (value, overflow) = builder.ins().sadd_overflow(lhs, rhs);
                        // unsigned carry-out
                        let c = builder.ins().icmp(IntCC::UnsignedLessThan, value, lhs);

                        (value, c, overflow)
                    }

                    FlagSettingBinOp::Sub => {
                        let (value, overflow) = builder.ins().ssub_overflow(lhs, rhs);

                        // ARM C for SUB: NOT borrow, so lhs >= rhs unsigned.
                        let c = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs);

                        (value, c, overflow)
                    }
                };

                let n = self.builder.ins().icmp_imm(IntCC::SignedLessThan, value, 0);
                let z = self.builder.ins().icmp_imm(IntCC::Equal, value, 0);

                self.lower_set_nzcv_flag_set(n, z, c, v)?;

                Some(value)
            }

            RValue::LoadHost { width, base_ptr, offset } => {
                Some(self.lower_host_load(width, base_ptr, offset)?)
            }

            RValue::StoreHost {
                base_ptr,
                offset,
                value,
            } => {
                let value = self.values[value];
                self.lower_host_store(base_ptr, offset, value)?;
                None
            }

            RValue::LoadHaltReason => {
                // todo!("lower halt reason atomic load")
                Some(self.builder.ins().iconst(clif_ir::types::I32, 0))
            }

            RValue::InstructionDone => {
                let base_ptr = LValue::ARG_PROCESSOR_STATE;
                let offset = offset_of!(ProcessorState, pc);

                let pc = self.lower_host_load(IntWidth::W64, base_ptr, offset)?;

                let four = self.builder.ins().iconst(clif_ir::types::I64, 4);
                let next_pc = self.builder.ins().iadd(pc, four);
                self.lower_host_store(base_ptr, offset, next_pc)?;

                None
            }
        };

        Ok(value)
    }

    fn lower_terminator(&mut self, terminator: &Terminator) -> anyhow::Result<()> {
        match *terminator {
            Terminator::Return => {
                let zero = self.builder.ins().iconst(clif_ir::types::I32, 0);
                self.builder.ins().return_(&[zero]);
            }

            Terminator::ReturnFail { halt_reason } => {
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

            for stmt in &block_data.stmts {
                self.lower_stmt(exec_ir, stmt)?;
            }

            self.lower_terminator(&block_data.terminator)?;
        }

        Ok(())
    }

    fn use_value(&self, lvalue: LValue) -> anyhow::Result<clif_ir::Value> {
        self.values
            .get(lvalue)
            .copied()
            .context("internal compiler error: lvalue used before being lowered")
    }

    fn clif_block(&self, block: IrBlock) -> anyhow::Result<clif_ir::Block> {
        self.blocks
            .get(block)
            .copied()
            .context("internal compiler error: missing Cranelift block")
    }

    fn int_nonzero(&mut self, value: clif_ir::Value) -> anyhow::Result<clif_ir::Value> {
        let ty = self.builder.func.dfg.value_type(value);
        ensure!(ty.is_int(), "BrNZ condition must be an integer value");
        Ok(self.builder.ins().icmp_imm(IntCC::NotEqual, value, 0))
    }

    #[allow(dead_code)]
    fn ir_ty_to_clif_ty(&self, ty: IrType) -> anyhow::Result<Option<clif_ir::Type>> {
        match ty {
            IrType::Unit => Ok(None),
            IrType::Int(width) => Ok(Some(Self::int_ty(width))),
            IrType::HostPtr => Ok(Some(self.ptr_ty)),
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

    pub fn try_compile(&self, function_name: String, exec_ir: ExecIr) -> anyhow::Result<CompiledExecBlock> {
        let builder = JITBuilder::with_isa(
            self.isa.clone(),
            cranelift::module::default_libcall_names()
        );

        let mut module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let mut builder_ctx = FunctionBuilderContext::new();

        ctx.set_disasm(true);

        ctx.func.signature = exec_block_signature(&module);

        {
            let ptr_ty = module.target_config().pointer_type();
            let builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

            let live_ordered_blocks = reverse_post_order_exec_ir(&exec_ir);

            let mut lowering =
                FunctionLowering::new(builder, &exec_ir, &live_ordered_blocks, ptr_ty)?;

            lowering.lower_blocks(&exec_ir)?;
            lowering.finish();
        }

        let func_id = module
            .declare_function(&function_name, Linkage::Export, &ctx.func.signature)
            .map_err(|err| anyhow!("declare_function failed: {err}"))?;

        module
            .define_function(func_id, &mut ctx)
            .map_err(|err| anyhow!("define_function failed: {err}"))?;


        // After define_function, the Context still contains the compiled result.
        let code = ctx.compiled_code()
            .expect("Cranelift did not leave compiled code in the context");

        if let Some(disasm) = &code.vcode {
            eprintln!("{disasm}");
        } else {
            eprintln!("no disassembly was produced");
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


