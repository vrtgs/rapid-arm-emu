use crate::arena::ArenaMap;
use crate::array_helper;
use crate::compiler::{CompileBlockOptions, CompiledExecChunk, ExecBlockFFI};
use crate::{
    Arg, ArithBinOp, BitwiseOp, CallbackSignature, ExecIr, HOST_CB_SMALL_ARGS, IConst, IntCmp,
    IntWidth, LoadType, MAX_STMT_OUTPUTS, OverflowingBinOp, SSAValue, ShiftOp, StackSlot, StmtData,
    StmtKind, Terminator, TerminatorKind,
};
use crate::{Block as IrBlock, Jump as IrJump, Type as IrType};
use anyhow::{Context, anyhow, ensure};
use arrayvec::ArrayVec;
use cranelift::codegen::ir as clif_ir;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module};
use cranelift::prelude::isa::OwnedTargetIsa;
use cranelift::prelude::{AbiParam, Configurable, InstBuilder, IntCC, MemFlags};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::mem::ManuallyDrop;
use std::num::NonZero;

fn clif_int_ty(width: IntWidth) -> clif_ir::Type {
    match width {
        IntWidth::W8 => clif_ir::types::I8,
        IntWidth::W16 => clif_ir::types::I16,
        IntWidth::W32 => clif_ir::types::I32,
        IntWidth::W64 => clif_ir::types::I64,
    }
}

fn ir_ty_to_clif_ty(ptr_ty: clif_ir::Type, ty: IrType) -> clif_ir::Type {
    match ty {
        IrType::Bool => clif_ir::types::I8,
        IrType::Int(width) => clif_int_ty(width),
        IrType::HostPtr => ptr_ty,
    }
}

struct FunctionLowering<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a JITModule,
    exec_ir: &'a ExecIr,
    ptr_ty: clif_ir::Type,

    live_ordered_blocks: &'a [IrBlock],
    stack_slots: ArenaMap<StackSlot, clif_ir::StackSlot>,
    values: ArenaMap<SSAValue, clif_ir::Value>,
    blocks: ArenaMap<IrBlock, clif_ir::Block>,
    signatures: ArenaMap<CallbackSignature, clif_ir::SigRef>,
}

impl<'a> FunctionLowering<'a> {
    fn assert_entry_args(&mut self, entry_block: clif_ir::Block) -> anyhow::Result<()> {
        let params = self.builder.block_params(entry_block);

        let args = Arg::args();

        ensure!(
            args.len() == params.len(),
            "internal compiler error: expected {} entry params, got {}",
            args.len(),
            params.len(),
        );

        for (arg, &param) in args.zip(params) {
            assert_eq!(self.values[arg.as_ssa_value()], param);
        }

        Ok(())
    }

    fn new(
        mut builder: FunctionBuilder<'a>,
        module: &'a JITModule,
        exec_ir: &'a ExecIr,
        ptr_ty: clif_ir::Type,
    ) -> anyhow::Result<Self> {
        let mut blocks = ArenaMap::with_capacity(exec_ir.blocks.len());
        let mut values = ArenaMap::with_capacity(exec_ir.ssa_values.len());
        let mut stack_slots = ArenaMap::with_capacity(exec_ir.stack_slots.len());
        let signatures = ArenaMap::with_capacity(exec_ir.signatures.len());

        for (ir_stack_slot, data) in exec_ir.stack_slots.iter() {
            debug_assert!(data.align.is_power_of_two());
            let align_shift = u8::try_from(data.align.ilog2())?;
            let clif_stack_slot = builder.create_sized_stack_slot(clif_ir::StackSlotData {
                kind: clif_ir::StackSlotKind::ExplicitSlot,
                size: data.size,
                align_shift,
                key: None,
            });

            stack_slots.insert_unique(ir_stack_slot, clif_stack_slot);
        }

        for &ir_block in &exec_ir.block_compile_order {
            let clif_block = builder.create_block();

            let block_ref = &exec_ir.blocks[ir_block];
            for &param in &block_ref.parameters {
                let ty = ir_ty_to_clif_ty(ptr_ty, exec_ir.ssa_values[param].ty);

                let clif_param = builder.append_block_param(clif_block, ty);
                values.insert(param, clif_param);
            }

            if block_ref.is_cold {
                builder.set_cold_block(clif_block);
            }

            blocks.insert_unique(ir_block, clif_block);
        }

        let entry_block = *blocks
            .get(IrBlock::ENTRYPOINT)
            .context("internal compiler error: missing entry block")?;

        let mut this = Self {
            builder,
            module,
            exec_ir,
            ptr_ty,

            live_ordered_blocks: &exec_ir.block_compile_order,
            values,
            blocks,
            stack_slots,
            signatures,
        };

        this.assert_entry_args(entry_block)?;

        Ok(this)
    }

    fn use_stack_slot(&self, stack_slot: StackSlot) -> anyhow::Result<clif_ir::StackSlot> {
        self.stack_slots
            .get(stack_slot)
            .copied()
            .context("internal compiler error: missing stack slot")
    }

    fn use_value(&self, ssa_value: SSAValue) -> anyhow::Result<clif_ir::Value> {
        self.values
            .get(ssa_value)
            .copied()
            .context("internal compiler error: ssa_value used before being lowered")
    }

    fn clif_block(&self, block: IrBlock) -> anyhow::Result<clif_ir::Block> {
        let res = self
            .blocks
            .get(block)
            .copied()
            .expect("internal compiler error: missing cranelift block");

        Ok(res)
    }

    fn clif_jump(&self, jump: &IrJump) -> anyhow::Result<(clif_ir::Block, Vec<clif_ir::BlockArg>)> {
        let block = self.clif_block(jump.target)?;
        let values = jump
            .parameters
            .iter()
            .map(|&ssa_value| self.use_value(ssa_value).map(clif_ir::BlockArg::Value))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok((block, values))
    }

    fn iconst(&mut self, iconst: IConst) -> clif_ir::Value {
        let bits = iconst.bits.cast_signed();
        let ty = clif_int_ty(iconst.width);
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

        for (value, &ssa_value) in values.into_iter().zip(outputs) {
            self.values.insert_unique(ssa_value, value)
        }

        Ok(())
    }

    fn host_memory_flags(can_move: bool) -> MemFlags {
        let mut flags = MemFlags::trusted();
        if can_move {
            flags.set_can_move();
        }

        flags
    }

    fn guest_memory_flags() -> MemFlags {
        MemFlags::trusted().with_endianness(clif_ir::Endianness::Little)
    }

    fn lower_host_load(
        &mut self,
        ty: LoadType,
        base_ptr: SSAValue,
        offset: usize,
        can_move: bool,
    ) -> anyhow::Result<clif_ir::Value> {
        let base_ptr = self.use_value(base_ptr)?;
        let offset =
            i32::try_from(offset).context("internal compiler error host load offset too large")?;

        let ty = match ty {
            LoadType::Int(width) => clif_int_ty(width),
            LoadType::HostPtr => self.ptr_ty,
        };

        Ok(self.builder.ins().load(
            ty,
            Self::host_memory_flags(can_move),
            base_ptr,
            clif_ir::immediates::Offset32::new(offset),
        ))
    }

    fn lower_host_store(
        &mut self,
        base_ptr: SSAValue,
        offset: usize,
        value: clif_ir::Value,
        can_move: bool,
    ) -> anyhow::Result<()> {
        let base_ptr = self.use_value(base_ptr)?;

        let offset =
            i32::try_from(offset).context("internal compiler error host load offset too large")?;

        self.builder.ins().store(
            Self::host_memory_flags(can_move),
            value,
            base_ptr,
            clif_ir::immediates::Offset32::new(offset),
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

    fn fn_signature(&mut self, signature: CallbackSignature) -> anyhow::Result<clif_ir::SigRef> {
        Ok(*self.signatures.get_or_insert_with(signature, || {
            let mut clif_signature = self.module.make_signature();

            let sig = &self.exec_ir.signatures[signature];
            for &arg in &sig.args[..] {
                clif_signature
                    .params
                    .push(AbiParam::new(ir_ty_to_clif_ty(self.ptr_ty, arg)));
            }

            if let Some(ty) = sig.ret {
                clif_signature
                    .returns
                    .push(AbiParam::new(ir_ty_to_clif_ty(self.ptr_ty, ty)))
            }

            self.builder.import_signature(clif_signature)
        }))
    }

    fn ptr_add(
        &mut self,
        base_ptr: SSAValue,
        offset: SSAValue,
        elem_size: NonZero<usize>,
    ) -> anyhow::Result<clif_ir::Value> {
        let base_ptr = self.use_value(base_ptr)?;
        let offset = self.use_value(offset)?;

        self.assert_value_ty(base_ptr, self.ptr_ty, "PtrAdd base_ptr")?;

        let offset_ty = self.builder.func.dfg.value_type(offset);
        let base_ptr_ty = self.ptr_ty;

        let offset = match offset_ty.bits().cmp(&base_ptr_ty.bits()) {
            Ordering::Less => self.builder.ins().uextend(base_ptr_ty, offset),
            Ordering::Equal => offset,
            Ordering::Greater => self.builder.ins().ireduce(base_ptr_ty, offset),
        };

        let offset = match elem_size.get() {
            1 => offset,
            imm => self.builder.ins().imul_imm(offset, i64::try_from(imm)?),
        };

        let offset_ptr = self.builder.ins().iadd(base_ptr, offset);

        Ok(offset_ptr)
    }

    fn ptr_byte_add(
        &mut self,
        base_ptr: SSAValue,
        offset: SSAValue,
    ) -> anyhow::Result<clif_ir::Value> {
        self.ptr_add(base_ptr, offset, const { NonZero::new(1).unwrap() })
    }

    fn lower_rvalue(
        &mut self,
        rvalue: &StmtKind,
    ) -> anyhow::Result<ArrayVec<clif_ir::Value, MAX_STMT_OUTPUTS>> {
        let value = match *rvalue {
            StmtKind::IConst(iconst) => array_helper::from_arr([self.iconst(iconst)]),

            StmtKind::ArithBinOp { op, lhs, rhs } => {
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

            StmtKind::AddImm { value, imm64 } => {
                let value = self.use_value(value)?;
                let new_value = self.builder.ins().iadd_imm(value, imm64.cast_signed());
                array_helper::from_arr([new_value])
            }

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

            StmtKind::Select {
                cond,
                if_true,
                if_false,
            } => {
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

            StmtKind::ShiftImm {
                op,
                value,
                shift_ammount,
            } => {
                let value = self.use_value(value)?;
                let ins = self.builder.ins();
                let shift_ammount = i64::from(shift_ammount);
                let output = match op {
                    // ShiftOp::SignExtendShr => ins.sshr_imm(value, shift_ammount),
                    ShiftOp::ZeroExtendShr => ins.ushr_imm(value, shift_ammount),
                    // ShiftOp::Shl => ins.ishl_imm(value, shift_ammount),
                };
                array_helper::from_arr([output])
            }

            StmtKind::OverflowingBinOp { op, lhs, rhs } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();
                let (value, overflow) = match op {
                    OverflowingBinOp::Add => ins.sadd_overflow(lhs, rhs),
                    OverflowingBinOp::Sub => ins.ssub_overflow(lhs, rhs),
                };

                array_helper::from_arr([value, overflow])
            }

            StmtKind::LoadHost {
                ty,
                base_ptr,
                offset,
                can_move,
            } => {
                let loaded_val = self.lower_host_load(ty, base_ptr, offset, can_move)?;
                array_helper::from_arr([loaded_val])
            }

            StmtKind::StoreHost {
                base_ptr,
                offset,
                value,
                can_move,
            } => {
                let value = self.use_value(value)?;
                self.lower_host_store(base_ptr, offset, value, can_move)?;
                array_helper::from_arr([])
            }

            StmtKind::LoadStackPtr { slot } => {
                let slot = self.use_stack_slot(slot)?;
                let addr = self.builder.ins().stack_addr(self.ptr_ty, slot, 0);
                array_helper::from_arr([addr])
            }

            StmtKind::PtrAdd {
                base_ptr,
                offset,
                elem_size,
            } => {
                let out_ptr = self.ptr_add(base_ptr, offset, elem_size)?;
                array_helper::from_arr([out_ptr])
            }

            StmtKind::PtrEq(ptr_a, ptr_b) => {
                let ptr_a = self.use_value(ptr_a)?;
                let ptr_b = self.use_value(ptr_b)?;
                self.assert_value_ty(ptr_a, self.ptr_ty, "PtrEq ptr_a")?;
                self.assert_value_ty(ptr_b, self.ptr_ty, "PtrEq ptr_b")?;
                let output = self.builder.ins().icmp(IntCC::Equal, ptr_a, ptr_b);
                array_helper::from_arr([output])
            }

            StmtKind::HasTag { ptr, tag_bits } => {
                let ptr = self.use_value(ptr)?;
                let output = self.builder.ins().band_imm(ptr, i64::from(tag_bits));
                array_helper::from_arr([output])
            }

            StmtKind::Untag { ptr, tag_bits } => {
                let ptr = self.use_value(ptr)?;
                let mask = !i64::from(tag_bits);
                let output = self.builder.ins().band_imm(ptr, mask);
                array_helper::from_arr([output])
            }

            StmtKind::HostCallback {
                func,
                signature,
                ref args,
            } => {
                let signature = self.fn_signature(signature)?;
                let args: &SmallVec<SSAValue, HOST_CB_SMALL_ARGS> = args;
                let args = args
                    .iter()
                    .map(|&val| self.use_value(val))
                    .collect::<anyhow::Result<SmallVec<_, HOST_CB_SMALL_ARGS>>>()?;

                let ptr_value = (func as *const ()).expose_provenance();
                let value = u64::try_from(ptr_value)?.cast_signed();
                let fn_ptr_val = self.builder.ins().iconst(self.ptr_ty, value);

                let ins = self
                    .builder
                    .ins()
                    .call_indirect(signature, fn_ptr_val, &args);
                self.builder.inst_results(ins).iter().copied().collect()
            }

            StmtKind::VMLoadRaw {
                aligned_page_ptr,
                page_offset,
                width,
            } => {
                let int_ty = clif_int_ty(width);
                let ptr = self.ptr_byte_add(aligned_page_ptr, page_offset)?;
                let flags = Self::guest_memory_flags();
                let loaded_val = self.builder.ins().atomic_load(int_ty, flags, ptr);
                array_helper::from_arr([loaded_val])
            }
            StmtKind::VMStoreRaw {
                aligned_page_ptr,
                page_offset,
                value,
            } => {
                let value = self.use_value(value)?;
                let ptr = self.ptr_byte_add(aligned_page_ptr, page_offset)?;
                let flags = Self::guest_memory_flags();
                self.builder.ins().atomic_store(flags, value, ptr);
                array_helper::from_arr([])
            }

            StmtKind::LoadHaltReason => {
                // although accessing the halt reason is **always** safe to do
                // having the load halt reason operation move is quite unexpected
                // and can lead to just well never halting
                // because the halt check moved to the end of the start of the function
                // and was just reasding stale halt reasons
                let can_move = false;
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let value = self.builder.ins().atomic_load(
                    clif_ir::types::I32,
                    Self::host_memory_flags(can_move),
                    ptr,
                );
                array_helper::from_arr([value])
            }

            StmtKind::TakeHaltReason => {
                // same reasons as `StmtKind::LoadHaltReason` stated above
                let can_move = false;
                let zero = self.iconst(IConst::zero(IntWidth::W32));
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let value = self.builder.ins().atomic_rmw(
                    clif_ir::types::I32,
                    Self::host_memory_flags(can_move),
                    clif_ir::AtomicRmwOp::Xchg,
                    ptr,
                    zero,
                );

                array_helper::from_arr([value])
            }

            StmtKind::GetInstructionDirtyFlag(insn_dirty) => {
                // same reasons as `StmtKind::LoadHaltReason` stated above
                let can_move = false;
                let ptr = self.use_value(insn_dirty)?;
                let value = self.builder.ins().atomic_load(
                    clif_ir::types::I8,
                    Self::host_memory_flags(can_move),
                    ptr,
                );
                array_helper::from_arr([value])
            }

            StmtKind::SetInstructionDirtyFlag(insn_dirty) => {
                // same reasons as `StmtKind::LoadHaltReason` stated above
                let can_move = false;
                let ptr = self.use_value(insn_dirty)?;
                let true_val = self.iconst(IConst::u8(1));
                self.builder
                    .ins()
                    .atomic_store(Self::host_memory_flags(can_move), true_val, ptr);
                array_helper::empty()
            }

            StmtKind::Safepoint => array_helper::from_arr([]),
        };

        Ok(value)
    }

    fn lower_terminator(&mut self, terminator: &Terminator) -> anyhow::Result<()> {
        match terminator.kind {
            TerminatorKind::Return => {
                let zero = self.builder.ins().iconst(clif_ir::types::I32, 0);
                self.builder.ins().return_(&[zero]);
            }

            TerminatorKind::ReturnCode { halt_reason } => {
                let halt_reason = self.use_value(halt_reason)?;
                self.assert_value_ty(halt_reason, clif_ir::types::I32, "ReturnFail halt_reason")?;
                self.builder.ins().return_(&[halt_reason]);
            }

            TerminatorKind::Br => {
                let target = &terminator.targets[0];
                let (target, args) = self.clif_jump(target)?;
                self.builder.ins().jump(target, &args);
            }

            TerminatorKind::BrZ { cond } => {
                let [zero, non_zero] = terminator.targets.as_array().unwrap();

                let cond = self.use_value(cond)?;
                let cond_is_nonzero = self.int_nonzero(cond)?;

                let (non_zero, nz_args) = self.clif_jump(non_zero)?;
                let (zero, z_args) = self.clif_jump(zero)?;

                self.builder
                    .ins()
                    .brif(cond_is_nonzero, non_zero, &nz_args, zero, &z_args);
            }
        }

        Ok(())
    }

    fn lower_blocks(&mut self, exec_ir: &ExecIr) -> anyhow::Result<()> {
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

    for arg in Arg::args() {
        sig.params
            .push(AbiParam::new(ir_ty_to_clif_ty(ptr_ty, arg.ty())));
    }

    sig.returns.push(AbiParam::new(clif_ir::types::I32));

    sig
}

pub struct CraneliftCompiler {
    compile_fast_isa: OwnedTargetIsa,
    compile_optimized_isa: OwnedTargetIsa,
}

impl CraneliftCompiler {
    pub fn new() -> anyhow::Result<Self> {
        fn bool_to_str(bool: bool) -> &'static str {
            match bool {
                true => "true",
                false => "false",
            }
        }

        let isa_builder = cranelift::native::builder()
            .map_err(|msg| anyhow!("host machine is not supported by Cranelift: {msg}"))?;

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

        // JIT frames are not intended to be stack-walked by profilers/debuggers.
        // Omitting frame pointers can reduce prologue/epilogue work and keeps an
        // extra register available on targets where the frame pointer would otherwise
        // be reserved.
        flag_builder.set("preserve_frame_pointers", "false")?;

        // Generated JIT functions must not unwind across this boundary. We do not need
        // DWARF/Windows unwind metadata for exceptions, panics, GC stack walking, or
        // debugger frame reconstruction, so skip emitting it to reduce compile-time
        // metadata work.
        flag_builder.set("unwind_info", "false")?;

        // The embedder does not need machine-code CFG metadata after codegen. We call
        // the finalized function pointer directly and do not post-process/analyze basic
        // block offsets or machine-code edges, so avoid generating that metadata.
        flag_builder.set("machine_code_cfg_info", "false")?;

        // On at least AArch64, "colocated" calls use shorter-range relocations,
        // which might not reach all definitions; we can't handle that here, so
        // we require long-range relocation types.
        flag_builder.set("use_colocated_libcalls", "false")?;

        flag_builder.set("preserve_frame_pointers", "false")?;

        // logs are not wanted
        flag_builder.set("regalloc_verbose_logs", "false")?;

        // lower compile time by removing verfification
        // since the IR is already pre checked and verified/trusted
        let check_flag = bool_to_str(cfg!(debug_assertions));
        flag_builder.set("enable_verifier", check_flag)?;
        flag_builder.set("regalloc_checker", check_flag)?;

        // we aren't compiling sandboxed code; the sandbox comes as a natural extension of
        // the IR building process where VM access is lowered
        flag_builder.set("enable_heap_access_spectre_mitigation", "false")?;
        flag_builder.set("enable_table_access_spectre_mitigation", "false")?;

        let unoptimized_builder = {
            let mut flags = flag_builder.clone();
            flags.set("opt_level", "none")?;
            flags.set("regalloc_algorithm", "single_pass")?;
            flags
        };

        flag_builder.set("opt_level", "speed")?;
        flag_builder.set("regalloc_algorithm", "backtracking")?;

        let optimized_builder = flag_builder;

        let (compile_fast_isa, compile_optimized_isa) = isa_builder
            .finish(cranelift::codegen::settings::Flags::new(
                unoptimized_builder,
            ))
            .and_then(|unopt_isa| {
                isa_builder
                    .finish(cranelift::codegen::settings::Flags::new(optimized_builder))
                    .map(|opt_isa| (unopt_isa, opt_isa))
            })
            .map_err(|err| anyhow!("Cranelift ISA creation failed: {err}"))?;

        Ok(Self {
            compile_fast_isa,
            compile_optimized_isa,
        })
    }

    pub fn try_compile(
        &self,
        options: CompileBlockOptions,
        exec_ir: &ExecIr,
        optimized: bool,
    ) -> anyhow::Result<CompiledExecChunk> {
        let isa = match optimized {
            true => &self.compile_optimized_isa,
            false => &self.compile_fast_isa,
        };

        let builder = JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names());

        let mut module = JITModule::new(builder);
        let mut ctx = module.make_context();
        let mut builder_ctx = FunctionBuilderContext::new();

        ctx.set_disasm(options.show_disasm);

        ctx.func.signature = exec_block_signature(&module);

        let mut clif_disasm_output = String::new();

        {
            let ptr_ty = module.target_config().pointer_type();
            let builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

            let mut lowering = FunctionLowering::new(builder, &module, exec_ir, ptr_ty)?;

            lowering.lower_blocks(exec_ir)?;
            lowering.finish();

            if options.show_disasm {
                // do it here because afterwards the clif will get optimized
                clif_disasm_output = ctx.func.display().to_string()
            }
        }

        let func_id = module
            .declare_function(&options.function_name, Linkage::Export, &ctx.func.signature)
            .map_err(|err| anyhow!("declare_function failed: {err}"))?;

        module
            .define_function(func_id, &mut ctx)
            .map_err(|err| anyhow!("define_function failed: {err}"))?;

        if options.show_disasm {
            let code = ctx
                .compiled_code()
                .expect("Cranelift did not leave compiled code in the context");

            eprintln!(
                "{name}:\nclif:\n{clif_disasm_output}\nassembly:\n{disasm}",
                name = options.function_name,
                disasm = code
                    .vcode
                    .as_deref()
                    .unwrap_or("no disassembly was produced")
            );
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

        Ok(CompiledExecChunk::new_with_recources(
            ffi,
            DropModule(ManuallyDrop::new(module)),
        ))
    }
}
