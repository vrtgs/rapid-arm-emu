use crate::arena::ArenaMap;
use crate::chunk::ExecChunkFFI;
use crate::compiler::{CompileBlockOptions, CompiledExecChunk};
use crate::ir::{
    AliasRegion as IrAliasRegion, Block as IrBlock, Jump as IrJump, TypeFull as IrType,
};
use crate::ir::{
    Arg, ArithBinOp, BitwiseOp, CallbackSignature, ExecIr, HOST_CB_SMALL_ARGS, IConst, IntCmp,
    IntWidth, LoadType, MAX_STMT_OUTPUTS, OverflowingBinOp, SSAValue, ShiftOp, StackSlot, StmtData,
    StmtKind, Terminator, TerminatorKind,
};
use anyhow::{Context as _, anyhow, ensure};
use arrayvec::ArrayVec;
use cranelift::codegen::Context;
use cranelift::codegen::ir as clif_ir;
use cranelift::codegen::ir::{AliasRegion, AliasRegionData};
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift::jit::{JITBuilder, JITModule};
use cranelift::module::{Linkage, Module};
use cranelift::prelude::isa::OwnedTargetIsa;
use cranelift::prelude::{AbiParam, Configurable, InstBuilder, IntCC, MemFlagsData};
use emu_abi::memory::Page;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cell::Cell;
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
        IrType::HostPtr(_) => ptr_ty,
    }
}

struct FunctionLowering<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a JITModule,
    exec_ir: &'a ExecIr,
    ptr_ty: clif_ir::Type,

    alias_regions: [Option<AliasRegion>; IrAliasRegion::COUNT],
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

            alias_regions: [None; IrAliasRegion::COUNT],
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
        let bits = iconst.bits().cast_signed();
        let ty = clif_int_ty(iconst.width());
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

    fn lower_stmt(&mut self, stmt: &StmtData) -> anyhow::Result<()> {
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

    fn alias_region(&mut self, alias_region: IrAliasRegion) -> AliasRegion {
        *self.alias_regions[alias_region as usize].get_or_insert_with(|| {
            self.builder.func.dfg.alias_regions.insert(AliasRegionData {
                user_id: u32::try_from(alias_region as usize).unwrap(),
                description: Cow::Owned(format!("{alias_region:?}")),
            })
        })
    }

    // the exact alias regions don't really mean ANYTHING
    // they are just 3 buckets. I decided that mutable VM stuff is `VmCtx`
    // and `ExecState` stuff is `Table` and `VirtualMemory` is `Heap`
    // but these names could be anything
    fn host_memory_flags(&mut self, can_move: bool, alias_region: IrAliasRegion) -> MemFlagsData {
        assert_ne!(alias_region, IrAliasRegion::VirtualMemory);

        let alias = self.alias_region(alias_region);
        let mut flags = MemFlagsData::trusted().with_alias_region(Some(alias));
        if can_move {
            flags.set_can_move();
        }

        if let IrAliasRegion::ReadOnly = alias_region {
            flags.set_readonly();
        }

        flags
    }

    fn guest_memory_flags(&mut self, alias_region: IrAliasRegion) -> MemFlagsData {
        assert_eq!(alias_region, IrAliasRegion::VirtualMemory);

        MemFlagsData::trusted()
            .with_endianness(clif_ir::Endianness::Little)
            .with_alias_region(Some(self.alias_region(IrAliasRegion::VirtualMemory)))
    }

    fn lower_host_load(
        &mut self,
        ty: LoadType,
        base_ptr: SSAValue,
        alias_region: IrAliasRegion,
        offset: usize,
        can_move: bool,
    ) -> anyhow::Result<clif_ir::Value> {
        let base_ptr = self.use_value(base_ptr)?;
        let offset =
            i32::try_from(offset).context("internal compiler error host load offset too large")?;

        let ty = match ty {
            LoadType::Int(width) => clif_int_ty(width),
            LoadType::HostPtr(_) => self.ptr_ty,
        };

        let flags = self.host_memory_flags(can_move, alias_region);
        Ok(self.builder.ins().load(
            ty,
            flags,
            base_ptr,
            clif_ir::immediates::Offset32::new(offset),
        ))
    }

    fn lower_host_store(
        &mut self,
        base_ptr: SSAValue,
        alias_region: IrAliasRegion,
        offset: usize,
        value: clif_ir::Value,
        can_move: bool,
    ) -> anyhow::Result<()> {
        let base_ptr = self.use_value(base_ptr)?;

        let offset =
            i32::try_from(offset).context("internal compiler error host load offset too large")?;

        let flags = self.host_memory_flags(can_move, alias_region);
        self.builder.ins().store(
            flags,
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
        use emu_abi::array_helper::{empty, from_arr};

        let value = match *rvalue {
            StmtKind::IConst(iconst) => from_arr([self.iconst(iconst)]),

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

                from_arr([value])
            }

            StmtKind::IntNeg(val) => {
                let val = self.use_value(val)?;
                let res = self.builder.ins().ineg(val);
                from_arr([res])
            }

            StmtKind::IntCmp { cmp, lhs, rhs } => {
                let op = Self::convert_to_intcc(cmp);
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                from_arr([self.builder.ins().icmp(op, lhs, rhs)])
            }

            StmtKind::IntCmpImm { cmp, lhs, rhs } => {
                let op = Self::convert_to_intcc(cmp);
                let lhs = self.use_value(lhs)?;
                let rhs = rhs.cast_signed();
                from_arr([self.builder.ins().icmp_imm(op, lhs, rhs)])
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

                from_arr([self.builder.ins().select(cond, if_true, if_false)])
            }

            StmtKind::Bitwise { op, lhs, rhs } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();

                let value = match op {
                    BitwiseOp::And => ins.band(lhs, rhs),
                    BitwiseOp::Or => ins.bor(lhs, rhs),
                    BitwiseOp::Xor => ins.bxor(lhs, rhs),
                };

                from_arr([value])
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

                from_arr([value])
            }

            StmtKind::BitNot(val) => {
                let val = self.use_value(val)?;
                let res = self.builder.ins().bnot(val);
                from_arr([res])
            }

            StmtKind::Shift { op, value, shift } => {
                let value = self.use_value(value)?;
                let shift = self.use_value(shift)?;
                let ins = self.builder.ins();
                let output = match op {
                    ShiftOp::SignExtendShr => ins.sshr(value, shift),
                    ShiftOp::ZeroExtendShr => ins.ushr(value, shift),
                    ShiftOp::Shl => ins.ishl(value, shift),
                };

                from_arr([output])
            }

            StmtKind::ShiftUnbounded { op, value, shift } => {
                let value = self.use_value(value)?;
                let shift = self.use_value(shift)?;
                let ty = self.builder.func.dfg.value_type(value);

                let bits = ty.bits();

                // if shift < bits; no overflow
                let in_range =
                    self.builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedLessThan, shift, i64::from(bits));

                let shifted = match op {
                    ShiftOp::SignExtendShr => self.builder.ins().sshr(value, shift),
                    ShiftOp::ZeroExtendShr => self.builder.ins().ushr(value, shift),
                    ShiftOp::Shl => self.builder.ins().ishl(value, shift),
                };

                let overflow_val = match op {
                    ShiftOp::SignExtendShr => self
                        .builder
                        .ins()
                        .sshr_imm(value, i64::from(bits.strict_sub(1))),
                    ShiftOp::ZeroExtendShr | ShiftOp::Shl => self.builder.ins().iconst(ty, 0),
                };

                let output = self.builder.ins().select(in_range, shifted, overflow_val);
                from_arr([output])
            }

            StmtKind::ShiftImm { op, value, shift } => {
                let value = self.use_value(value)?;
                let ins = self.builder.ins();
                let shift_amount = i64::from(shift);
                let output = match op {
                    ShiftOp::SignExtendShr => ins.sshr_imm(value, shift_amount),
                    ShiftOp::ZeroExtendShr => ins.ushr_imm(value, shift_amount),
                    ShiftOp::Shl => ins.ishl_imm(value, shift_amount),
                };

                from_arr([output])
            }

            StmtKind::OverflowingBinOp { op, lhs, rhs } => {
                let (lhs, rhs) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ins = self.builder.ins();
                let (value, overflow) = match op {
                    OverflowingBinOp::Add => ins.sadd_overflow(lhs, rhs),
                    OverflowingBinOp::Sub => ins.ssub_overflow(lhs, rhs),
                };

                from_arr([value, overflow])
            }

            StmtKind::LoadHost {
                ty,
                base_ptr,
                offset,
                can_move,
            } => {
                let IrType::HostPtr(alias) = self.exec_ir.ssa_values[base_ptr].ty else {
                    panic!("can't load non pointer type")
                };
                let loaded_val = self.lower_host_load(ty, base_ptr, alias, offset, can_move)?;
                from_arr([loaded_val])
            }

            StmtKind::StoreHost {
                base_ptr,
                offset,
                value,
                can_move,
            } => {
                let IrType::HostPtr(alias) = self.exec_ir.ssa_values[base_ptr].ty else {
                    panic!("can't store non pointer type")
                };
                let value = self.use_value(value)?;
                self.lower_host_store(base_ptr, alias, offset, value, can_move)?;
                empty()
            }

            StmtKind::LoadStackPtr { slot } => {
                let slot = self.use_stack_slot(slot)?;
                let addr = self.builder.ins().stack_addr(self.ptr_ty, slot, 0);
                from_arr([addr])
            }

            StmtKind::PtrAdd {
                base_ptr,
                offset,
                elem_size,
            } => {
                let out_ptr = self.ptr_add(base_ptr, offset, elem_size)?;
                from_arr([out_ptr])
            }

            StmtKind::PtrEq(ptr_a, ptr_b) => {
                let ptr_a = self.use_value(ptr_a)?;
                let ptr_b = self.use_value(ptr_b)?;
                self.assert_value_ty(ptr_a, self.ptr_ty, "PtrEq ptr_a")?;
                self.assert_value_ty(ptr_b, self.ptr_ty, "PtrEq ptr_b")?;
                let output = self.builder.ins().icmp(IntCC::Equal, ptr_a, ptr_b);
                from_arr([output])
            }

            StmtKind::HasTag { ptr, tag_bits } => {
                let ptr = self.use_value(ptr)?;
                let output = self.builder.ins().band_imm(ptr, i64::from(tag_bits));
                from_arr([output])
            }

            StmtKind::Untag { ptr, tag_bits } => {
                let ptr = self.use_value(ptr)?;
                let mask = !i64::from(tag_bits);
                let output = self.builder.ins().band_imm(ptr, mask);
                from_arr([output])
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
                // loads are always `SeqCst` in cranelift,
                // which is stronger than relaxed or acquire
                seq_cst: _,
            } => {
                let int_ty = clif_int_ty(width);
                let ptr = self.ptr_byte_add(aligned_page_ptr, page_offset)?;
                let flags = self.guest_memory_flags(IrAliasRegion::VirtualMemory);
                let loaded_val = self.builder.ins().atomic_load(int_ty, flags, ptr);
                from_arr([loaded_val])
            }

            StmtKind::VMStoreRaw {
                aligned_page_ptr,
                page_offset,
                value,
                // stores are always `SeqCst` in cranelift,
                // which is stronger than relaxed or release
                seq_cst: _,
            } => {
                let value = self.use_value(value)?;
                let ptr = self.ptr_byte_add(aligned_page_ptr, page_offset)?;
                let flags = self.guest_memory_flags(IrAliasRegion::VirtualMemory);
                self.builder.ins().atomic_store(flags, value, ptr);
                from_arr([])
            }

            // `AqcRel` swap; but cranelift only has SeqCst operations
            StmtKind::LoadHaltReason => {
                // although accessing the halt reason is **always** safe to do,
                // having the load halt reason operation move is quite unexpected
                // and can lead to well, never quite halting
                // because the halt check moved to the end of the start of the function
                // and was just reading stale `HaltReason`
                let can_move = false;
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let flags = self.host_memory_flags(can_move, IrAliasRegion::HaltReason);
                let value = self
                    .builder
                    .ins()
                    .atomic_load(clif_ir::types::I32, flags, ptr);
                from_arr([value])
            }

            // `AqcRel` swap; but cranelift only has SeqCst operations
            StmtKind::TakeHaltReason => {
                // same reasons as `StmtKind::LoadHaltReason` stated above
                let can_move = false;
                let zero = self.iconst(IConst::zero(IntWidth::W32));
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let flags = self.host_memory_flags(can_move, IrAliasRegion::HaltReason);
                let value = self.builder.ins().atomic_rmw(
                    clif_ir::types::I32,
                    flags,
                    clif_ir::AtomicRmwOp::Xchg,
                    ptr,
                    zero,
                );

                from_arr([value])
            }

            // `SetPageDirtyFlag` requires **Release** semantics
            // cranelift uses seq_cst semantics, which is strictly stronger
            StmtKind::SetPageDirtyFlag(insn_dirty) => {
                // this can't move and MUST happen **after** writing the value into the page
                // doing it before that may lead to never actually marking the page dirty
                let can_move = false;
                let ptr = self.use_value(insn_dirty)?;
                let flag_val = self.iconst(IConst::u8(Page::IS_DIRTY_FLAG));

                let flags = self.host_memory_flags(can_move, IrAliasRegion::PageFlags);
                self.builder.ins().atomic_rmw(
                    clif_ir::types::I8,
                    flags,
                    clif_ir::AtomicRmwOp::Or,
                    ptr,
                    flag_val,
                );
                empty()
            }

            StmtKind::Safepoint => from_arr([]),
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

    fn lower_blocks(&mut self) -> anyhow::Result<()> {
        for &ir_block in &self.exec_ir.block_compile_order {
            let clif_block = self.clif_block(ir_block)?;

            self.builder.switch_to_block(clif_block);

            let block_data = &self.exec_ir.blocks[ir_block];

            for &stmt in &block_data.stmts {
                self.lower_stmt(&self.exec_ir.stmts[stmt])?;
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

    // this makes a signature with the target triples default calling convention,
    // which is basically the C calling convention
    let mut sig = module.make_signature();

    for arg in Arg::args() {
        sig.params
            .push(AbiParam::new(ir_ty_to_clif_ty(ptr_ty, arg.ty())));
    }

    sig.returns.push(AbiParam::new(clif_ir::types::I32));

    sig
}

thread_local! {
    static SCRATCH: Cell<Option<(Context, FunctionBuilderContext)>> = const {
        Cell::new(None)
    };
}

pub(crate) struct CraneliftCompiler {
    compile_fast_isa: OwnedTargetIsa,
    compile_optimized_isa: OwnedTargetIsa,
}

impl Drop for CraneliftCompiler {
    fn drop(&mut self) {
        // best effort cleanup
        SCRATCH.take();
    }
}

impl CraneliftCompiler {
    pub(crate) fn new() -> anyhow::Result<Self> {
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
        flag_builder.set("is_pic", "false")?;

        // On at least AArch64, "colocated" calls use shorter-range relocations,
        // which might not reach all definitions; we can't handle that here, so
        // we require long-range relocation types.
        flag_builder.set("use_colocated_libcalls", "false")?;

        // JIT frames are not intended to be stack-walked by profilers/debuggers.
        // Omitting frame pointers can reduce prologue/epilogue work and keep an
        // extra register available on targets where the frame pointer would otherwise
        // be reserved.
        flag_builder.set("preserve_frame_pointers", "false")?;

        // Generated JIT functions must not unwind across this boundary. We do not need any
        // DWARF/Windows unwind metadata for exceptions, panics, GC stack walking, or
        // debugger frame reconstruction, so skip emitting it to reduce compile-time
        // metadata work.
        flag_builder.set("unwind_info", "false")?;

        // The embedder does not need machine-code CFG metadata after codegen. We call
        // the finalized function pointer directly and do not post-process/analyze basic
        // block offsets or machine-code edges, so avoid generating that metadata.
        flag_builder.set("machine_code_cfg_info", "false")?;

        // logs are not wanted
        flag_builder.set("regalloc_verbose_logs", "false")?;

        // lower compile time by removing verification
        // since the IR is already pre-checked and verified/trusted
        let check_flag = bool_to_str(cfg!(debug_assertions));
        flag_builder.set("enable_verifier", check_flag)?;
        flag_builder.set("regalloc_checker", check_flag)?;

        // Take the hit on sandboxing in the name of speed
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

    pub(crate) fn try_compile(
        &self,
        options: CompileBlockOptions,
        exec_ir: &ExecIr,
        optimized: bool,
    ) -> anyhow::Result<CompiledExecChunk> {
        let (mut ctx, mut builder_ctx) = SCRATCH
            .take()
            .unwrap_or_else(|| (Context::new(), FunctionBuilderContext::new()));

        let isa = match optimized {
            true => &self.compile_optimized_isa,
            false => &self.compile_fast_isa,
        };

        ctx.func.signature.call_conv = isa.default_call_conv();

        let builder = JITBuilder::with_isa(isa.clone(), cranelift::module::default_libcall_names());
        let mut module = JITModule::new(builder);

        ctx.set_disasm(options.show_disasm);

        ctx.func.signature = exec_block_signature(&module);

        let mut clif_disasm_output = String::new();

        {
            let ptr_ty = module.target_config().pointer_type();
            let builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

            let mut lowering = FunctionLowering::new(builder, &module, exec_ir, ptr_ty)?;

            lowering.lower_blocks()?;
            lowering.finish();

            if options.show_disasm {
                // do it here because afterward the `clif` will get optimized
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

        SCRATCH.set(Some((ctx, builder_ctx)));

        module
            .finalize_definitions()
            .map_err(|err| anyhow!("finalize_definitions failed: {err}"))?;

        let code_ptr = module.get_finalized_function(func_id);

        let ffi: ExecChunkFFI = unsafe {
            // Safety:
            // - `exec_block_signature` must exactly match `ExecBlockFFI`.
            // - `module` is moved into `resources`, keeping finalized code alive.
            std::mem::transmute::<*const u8, ExecChunkFFI>(code_ptr)
        };

        struct DropModule(ManuallyDrop<JITModule>);

        impl Drop for DropModule {
            fn drop(&mut self) {
                unsafe { ManuallyDrop::take(&mut self.0).free_memory() };
            }
        }

        Ok(CompiledExecChunk::new_with_resources(
            ffi,
            DropModule(ManuallyDrop::new(module)),
        ))
    }
}
