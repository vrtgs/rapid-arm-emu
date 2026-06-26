use crate::arena::{ArenaMap, to_raw};
use crate::compiler::{CompileBlockOptions, CompiledExecChunk, ExecBlockFFI};
use crate::{
    Arg, ArithBinOp, BitwiseOp, CallbackSignature, ExecIr, IConst, IntCmp, IntWidth, LoadType,
    MAX_STMT_OUTPUTS, OverflowingBinOp, SSAValue, ShiftOp, StackSlot, StmtData, StmtKind,
    Terminator, TerminatorKind,
};
use crate::{Block as IrBlock, Jump as IrJump, Type as IrType};
use anyhow::{Context, bail, ensure};
use arrayvec::ArrayVec;
use gccjit::{
    BinaryOp, Block as GccBlock, ComparisonOp, Context as GccContext, Function, FunctionType,
    LValue, OptimizationLevel, RValue, ToLValue, ToRValue, Type as GccType, UnaryOp,
};
use std::num::NonZero;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

fn convert_to_gcc_cmp(cmp: IntCmp) -> ComparisonOp {
    match cmp {
        IntCmp::Equal => ComparisonOp::Equals,
        IntCmp::NotEqual => ComparisonOp::NotEquals,
        IntCmp::SignedLessThan => ComparisonOp::LessThan,
        IntCmp::SignedGreaterThanOrEqual => ComparisonOp::GreaterThanEquals,
        IntCmp::SignedGreaterThan => ComparisonOp::GreaterThan,
        IntCmp::SignedLessThanOrEqual => ComparisonOp::LessThanEquals,
        // Signed comparisons: cast operands to unsigned before comparing
        IntCmp::UnsignedLessThan => ComparisonOp::LessThan,
        IntCmp::UnsignedGreaterThanOrEqual => ComparisonOp::GreaterThanEquals,
        IntCmp::UnsignedGreaterThan => ComparisonOp::GreaterThan,
        IntCmp::UnsignedLessThanOrEqual => ComparisonOp::LessThanEquals,
    }
}

fn is_signed_cmp(cmp: IntCmp) -> bool {
    matches!(
        cmp,
        IntCmp::SignedLessThan
            | IntCmp::SignedGreaterThanOrEqual
            | IntCmp::SignedGreaterThan
            | IntCmp::SignedLessThanOrEqual
    )
}

/// Values of GCC's __ATOMIC_* memory model macros, as defined in <stdatomic.h>.
/// These are stable ABI - GCC has never changed them.
mod atomic_model {
    pub(super) const RELAXED: i32 = 0;
    // pub(super) const CONSUME: i32 = 1;
    pub(super) const ACQUIRE: i32 = 2;
    pub(super) const RELEASE: i32 = 3;
    pub(super) const ACQ_REL: i32 = 4;
    pub(super) const SEQ_CST: i32 = 5;
}

struct TypeCache<'ctx> {
    bool_ty: Option<GccType<'ctx>>,
    usize_ty: Option<GccType<'ctx>>,
    void_ptr_ty: Option<GccType<'ctx>>,
    void_ty: Option<GccType<'ctx>>,
    atomic_read_ptr: Option<GccType<'ctx>>,
    atomic_write_ptr: Option<GccType<'ctx>>,
    c_int_ty: Option<GccType<'ctx>>,

    i128_ty: Option<GccType<'ctx>>,

    u64_ty: Option<GccType<'ctx>>,
    u32_ty: GccType<'ctx>,
    u16_ty: Option<GccType<'ctx>>,
    u8_ty: Option<GccType<'ctx>>,

    i64_ty: Option<GccType<'ctx>>,
    i32_ty: Option<GccType<'ctx>>,
    i16_ty: Option<GccType<'ctx>>,
    i8_ty: Option<GccType<'ctx>>,
}

impl<'ctx> TypeCache<'ctx> {
    fn new(ctx: &'ctx GccContext<'ctx>) -> Self {
        Self {
            bool_ty: None,
            usize_ty: None,
            void_ptr_ty: None,
            void_ty: None,
            atomic_read_ptr: None,
            atomic_write_ptr: None,
            c_int_ty: None,
            i128_ty: None,
            u64_ty: None,
            u32_ty: ctx.new_type::<u32>(),
            u16_ty: None,
            u8_ty: None,
            i64_ty: None,
            i32_ty: None,
            i16_ty: None,
            i8_ty: None,
        }
    }

    fn resolve_int(
        &mut self,
        ctx: &'ctx GccContext<'ctx>,
        signed: bool,
        width: IntWidth,
    ) -> GccType<'ctx> {
        let ty = match (signed, width) {
            (false, IntWidth::W64) => self.u64_ty.get_or_insert_with(|| ctx.new_type::<u64>()),
            (false, IntWidth::W32) => &mut self.u32_ty,
            (false, IntWidth::W16) => self.u16_ty.get_or_insert_with(|| ctx.new_type::<u16>()),
            (false, IntWidth::W8) => self.u8_ty.get_or_insert_with(|| ctx.new_type::<u8>()),

            (true, IntWidth::W64) => self.i64_ty.get_or_insert_with(|| ctx.new_type::<i64>()),
            (true, IntWidth::W32) => self.i32_ty.get_or_insert_with(|| ctx.new_type::<i32>()),
            (true, IntWidth::W16) => self.i16_ty.get_or_insert_with(|| ctx.new_type::<i16>()),
            (true, IntWidth::W8) => self.i8_ty.get_or_insert_with(|| ctx.new_type::<i8>()),
        };

        *ty
    }

    fn resolve_uint(&mut self, ctx: &'ctx GccContext<'ctx>, width: IntWidth) -> GccType<'ctx> {
        self.resolve_int(ctx, false, width)
    }

    fn resolve_sint(&mut self, ctx: &'ctx GccContext<'ctx>, width: IntWidth) -> GccType<'ctx> {
        self.resolve_int(ctx, true, width)
    }

    fn void_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self.void_ty.get_or_insert_with(|| ctx.new_type::<()>())
    }

    fn void_ptr_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self
            .void_ptr_ty
            .get_or_insert_with(|| ctx.new_type::<*mut ()>())
    }

    fn atomic_read_ptr_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self.atomic_read_ptr.get_or_insert_with(|| {
            ctx.new_type::<()>()
                .make_const()
                .make_volatile()
                .make_pointer()
        })
    }

    fn atomic_rmw_ptr_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self
            .atomic_write_ptr
            .get_or_insert_with(|| ctx.new_type::<()>().make_volatile().make_pointer())
    }

    fn usize_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self.usize_ty.get_or_insert_with(|| ctx.new_type::<usize>())
    }

    fn i128_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self
            .i128_ty
            .get_or_insert_with(|| ctx.new_c_type(gccjit::CType::Int128t))
    }

    fn c_int_ty(&mut self, ctx: &'ctx GccContext<'ctx>) -> GccType<'ctx> {
        *self
            .c_int_ty
            .get_or_insert_with(|| ctx.new_c_type(gccjit::CType::Int))
    }

    fn resolve_ty(&mut self, ctx: &'ctx GccContext<'ctx>, ty: IrType) -> GccType<'ctx> {
        match ty {
            IrType::Bool => *self.bool_ty.get_or_insert_with(|| ctx.new_type::<bool>()),
            IrType::Int(width) => self.resolve_uint(ctx, width),
            IrType::HostPtr(_) => self.void_ptr_ty(ctx),
        }
    }
}

struct FunctionLowering<'ctx> {
    ctx: &'ctx GccContext<'ctx>,
    exec_ir: &'ctx ExecIr,
    func: Function<'ctx>,

    current_block: Option<GccBlock<'ctx>>,

    values: ArenaMap<SSAValue, LValue<'ctx>>,
    blocks: ArenaMap<IrBlock, GccBlock<'ctx>>,
    stack_slots: ArenaMap<StackSlot, LValue<'ctx>>,
    sig_types: ArenaMap<CallbackSignature, GccType<'ctx>>,

    type_cache: TypeCache<'ctx>,
}

impl<'ctx> FunctionLowering<'ctx> {
    fn new(
        ctx: &'ctx GccContext,
        exec_ir: &'ctx ExecIr,
        function_name: &str,
    ) -> anyhow::Result<Self> {
        let mut values = ArenaMap::with_capacity(exec_ir.ssa_values.len());
        let mut blocks = ArenaMap::with_capacity(exec_ir.blocks.len());
        let mut stack_slots = ArenaMap::with_capacity(exec_ir.stack_slots.len());
        let sig_types = ArenaMap::with_capacity(exec_ir.signatures.len());

        let mut type_cache = TypeCache::new(ctx);

        let params = Arg::args().enumerate().map(|(i, arg)| {
            assert_eq!(i, arg as usize);

            let ty = type_cache.resolve_ty(ctx, arg.ty());
            let name = format!("arg_{}", arg as usize);
            ctx.new_parameter(None, ty, &name)
        });

        let param_list = params.collect::<Vec<_>>();

        let func = ctx.new_function(
            None,
            FunctionType::Exported,
            type_cache.u32_ty,
            &param_list,
            function_name,
            false,
        );

        // Create a gcc basic block for every IR block in compile order.
        // Block parameters become local variables written at the jump site
        // (gccjit has no block-parameter concept, so we emulate SSA with locals).
        let entry_gcc_block = func.new_block("entry");

        for (idx, &ir_block) in exec_ir.block_compile_order.iter().enumerate() {
            let name = format!("block_{}", to_raw(ir_block).get());
            let gcc_block = if idx == 0 {
                entry_gcc_block
            } else {
                func.new_block(&name)
            };
            blocks.insert_unique(ir_block, gcc_block);
        }

        // Create locals for every SSA value (including block parameters).
        for (ssa_val, ssa_data) in exec_ir.ssa_values.iter() {
            let lvalue = match Arg::from_ssa_value(ssa_val) {
                Some(arg) => param_list[arg as usize].to_lvalue(),
                None => {
                    let ty = type_cache.resolve_ty(ctx, ssa_data.ty);
                    let name = format!("v{}", to_raw(ssa_val).get());
                    func.new_local(None, ty, &name)
                }
            };

            values.insert(ssa_val, lvalue);
        }

        // Entry-block arguments: assign from the gcc function parameters.
        // The entry block's IR parameters are filled from the function params.
        let entry_block_ref = &exec_ir.blocks[IrBlock::ENTRYPOINT];

        ensure!(
            Arg::args().count() == entry_block_ref.parameters.len(),
            "internal compiler error: entry block parameter count mismatch"
        );

        // allocate char arrays, then use their address as the "slot pointer".
        for (slot, slot_data) in exec_ir.stack_slots.iter() {
            let u8 = type_cache.resolve_uint(ctx, IntWidth::W8);

            let arr_ty = ctx.new_array_type(None, u8, slot_data.size.into());

            let slot_ty = arr_ty.get_aligned(slot_data.align.into());

            let name = format!("stack_slot_{}", to_raw(slot).get());
            let local = func.new_local(None, slot_ty, &name);
            stack_slots.insert_unique(slot, local);
        }

        Ok(Self {
            ctx,
            exec_ir,
            func,
            current_block: None,
            values,
            blocks,
            stack_slots,
            sig_types,
            type_cache,
        })
    }

    fn use_value(&self, ssa: SSAValue) -> anyhow::Result<RValue<'ctx>> {
        Ok(self
            .values
            .get(ssa)
            .copied()
            .context("internal compiler error: ssa value used before being lowered")?
            .to_rvalue())
    }

    fn gcc_block(&self, ir_block: IrBlock) -> anyhow::Result<GccBlock<'ctx>> {
        self.blocks
            .get(ir_block)
            .copied()
            .context("internal compiler error: missing gcc block")
    }

    fn iconst_rvalue(&mut self, iconst: IConst) -> RValue<'ctx> {
        let ty = self.type_cache.resolve_uint(self.ctx, iconst.width);
        self.ctx.new_rvalue_from_long(ty, iconst.bits.cast_signed())
    }

    fn u32_const(&self, v: u32) -> RValue<'ctx> {
        self.ctx
            .new_rvalue_from_int(self.type_cache.u32_ty, v.cast_signed())
    }

    fn u8_const(&mut self, v: u8) -> RValue<'ctx> {
        let ty = self.type_cache.resolve_uint(self.ctx, IntWidth::W8);
        self.ctx.new_rvalue_from_int(ty, v as i32)
    }

    fn cast(&self, val: RValue<'ctx>, ty: GccType<'ctx>) -> RValue<'ctx> {
        self.ctx.new_cast(None, val, ty)
    }

    /// Dereference a void pointer cast to the given integer type.
    fn load_ptr(&self, ptr: RValue<'ctx>, ty: GccType<'ctx>) -> RValue<'ctx> {
        let typed_ptr = self.ctx.new_cast(None, ptr, ty.make_pointer());
        typed_ptr.dereference(None).to_rvalue()
    }

    fn store_ptr(&self, ptr: RValue<'ctx>, val: RValue<'ctx>) {
        let ty = val.get_type();
        let typed_ptr = self.ctx.new_cast(None, ptr, ty.make_pointer());
        let lval = typed_ptr.dereference(None);
        self.current_block.unwrap().add_assignment(None, lval, val);
    }

    /// Compute `base_ptr + offset * elem_size` as a void pointer.
    fn ptr_add(
        &mut self,
        base_ptr: RValue<'ctx>,
        offset: RValue<'ctx>,
        elem_size: NonZero<usize>,
    ) -> anyhow::Result<RValue<'ctx>> {
        let u8_ty = self.type_cache.resolve_uint(self.ctx, IntWidth::W8);
        let element_struct_ty = match elem_size.get() {
            1 => u8_ty,
            _ => self
                .ctx
                .new_array_type(None, u8_ty, elem_size.get().try_into()?),
        };

        let element_ptr_ty = element_struct_ty.make_pointer();

        let typed_base = self.cast(base_ptr, element_ptr_ty);

        let element_lvalue = self.ctx.new_array_access(None, typed_base, offset);

        Ok(element_lvalue.get_address(None))
    }

    fn ptr_byte_add(
        &mut self,
        base_ptr: RValue<'ctx>,
        offset: RValue<'ctx>,
    ) -> anyhow::Result<RValue<'ctx>> {
        self.ptr_add(base_ptr, offset, const { NonZero::new(1).unwrap() })
    }

    fn ptr_add_const(
        &mut self,
        base_ptr: RValue<'ctx>,
        offset: usize,
    ) -> anyhow::Result<RValue<'ctx>> {
        let size_t = self.type_cache.usize_ty(self.ctx);
        let off_const = self.ctx.new_rvalue_from_long(
            size_t,
            i64::try_from(offset).context("host offset too large")?,
        );

        self.ptr_byte_add(base_ptr, off_const)
    }

    fn fn_ptr_type(&mut self, signature: CallbackSignature) -> GccType<'ctx> {
        if let Some(&ty) = self.sig_types.get(signature) {
            return ty;
        }

        let sig = &self.exec_ir.signatures[signature];
        let ret_ty = match sig.ret {
            Some(t) => self.type_cache.resolve_ty(self.ctx, t),
            None => self.type_cache.void_ty(self.ctx),
        };

        let param_types = sig
            .args
            .iter()
            .map(|&t| self.type_cache.resolve_ty(self.ctx, t))
            .collect::<Vec<_>>();

        let fn_ptr_ty = self
            .ctx
            .new_function_pointer_type(None, ret_ty, &param_types, false);

        self.sig_types.insert(signature, fn_ptr_ty);
        fn_ptr_ty
    }

    fn lower_cmp(
        &mut self,
        cmp: IntCmp,
        ty_val: SSAValue,
        mut l: RValue<'ctx>,
        mut r: RValue<'ctx>,
    ) -> anyhow::Result<RValue<'ctx>> {
        if is_signed_cmp(cmp) {
            let ty = match self.exec_ir.ssa_values[ty_val].ty {
                IrType::Int(width) => self.type_cache.resolve_sint(self.ctx, width),
                ty => bail!("invalid integer comparison of type: {ty:?}"),
            };
            l = self.cast(l, ty);
            r = self.cast(r, ty);
        }

        let gcc_cmp = convert_to_gcc_cmp(cmp);

        Ok(self.ctx.new_comparison(None, gcc_cmp, l, r))
    }

    fn lower_swap_bytes(&mut self, width: IntWidth, int: RValue<'ctx>) -> RValue<'ctx> {
        let bswap_builtin = match width {
            IntWidth::W8 => return int,
            IntWidth::W16 => "__builtin_bswap16",
            IntWidth::W32 => "__builtin_bswap32",
            IntWidth::W64 => "__builtin_bswap64",
        };

        let bswap = self.ctx.get_builtin_function(bswap_builtin);
        self.ctx.new_call(None, bswap, &[int])
    }

    fn lower_bswap_to_le(&mut self, width: IntWidth, int: RValue<'ctx>) -> RValue<'ctx> {
        const { assert!(cfg!(target_endian = "little") || cfg!(target_endian = "big")) }

        match cfg!(target_endian = "little") {
            true => int,
            false => self.lower_swap_bytes(width, int),
        }
    }

    fn ordering_int(&mut self, ordering: Ordering) -> RValue<'ctx> {
        let ordering = match ordering {
            Ordering::Relaxed => atomic_model::RELAXED,
            Ordering::Release => atomic_model::RELEASE,
            Ordering::Acquire => atomic_model::ACQUIRE,
            Ordering::AcqRel => atomic_model::ACQ_REL,
            Ordering::SeqCst => atomic_model::SEQ_CST,
            _ => unreachable!("unknown ordering"),
        };

        self.ctx
            .new_rvalue_from_int(self.type_cache.c_int_ty(self.ctx), ordering)
    }

    fn lower_atomic_load(
        &mut self,
        width: IntWidth,
        ptr: RValue<'ctx>,
        ordering: Ordering,
    ) -> RValue<'ctx> {
        let int_ty = self.type_cache.resolve_uint(self.ctx, width);
        let builtin_name = match width {
            IntWidth::W8 => "__atomic_load_1",
            IntWidth::W16 => "__atomic_load_2",
            IntWidth::W32 => "__atomic_load_4",
            IntWidth::W64 => "__atomic_load_8",
        };

        let atomic_load_fn = self.ctx.get_builtin_function(builtin_name);
        let ordering = self.ordering_int(ordering);
        let ptr_type = self.type_cache.atomic_read_ptr_ty(self.ctx);
        let typed_ptr = self.cast(ptr, ptr_type);
        let value = self
            .ctx
            .new_call(None, atomic_load_fn, &[typed_ptr, ordering]);
        self.cast(value, int_ty)
    }

    fn lower_atomic_fence(&mut self, ordering: Ordering) {
        let atomic_load_fn = self.ctx.get_builtin_function("__atomic_thread_fence");
        let ordering = self.ordering_int(ordering);
        let value = self.ctx.new_call(None, atomic_load_fn, &[ordering]);
        self.current_block.unwrap().add_eval(None, value);
    }

    fn lower_atomic_store(
        &mut self,
        width: IntWidth,
        ptr: RValue<'ctx>,
        value: RValue<'ctx>,
        ordering: Ordering,
    ) {
        let ptr_type = self.type_cache.atomic_rmw_ptr_ty(self.ctx);

        let typed_ptr = self.cast(ptr, ptr_type);

        let builtin_name = match width {
            IntWidth::W8 => "__atomic_store_1",
            IntWidth::W16 => "__atomic_store_2",
            IntWidth::W32 => "__atomic_store_4",
            IntWidth::W64 => "__atomic_store_8",
        };

        let store_builtin = self.ctx.get_builtin_function(builtin_name);
        let ordering = self.ordering_int(ordering);

        let int_ty = self.type_cache.resolve_sint(self.ctx, width);
        let value = self.cast(value, int_ty);

        let eval = self
            .ctx
            .new_call(None, store_builtin, &[typed_ptr, value, ordering]);

        self.current_block.unwrap().add_eval(None, eval);
    }

    fn lower_atomic_fetch_or(
        &mut self,
        width: IntWidth,
        ptr: RValue<'ctx>,
        value: RValue<'ctx>,
        ordering: Ordering,
    ) -> RValue<'ctx> {
        let ptr_type = self.type_cache.atomic_rmw_ptr_ty(self.ctx);
        let typed_ptr = self.cast(ptr, ptr_type);
        let builtin_name = match width {
            IntWidth::W8 => "__atomic_fetch_or_1",
            IntWidth::W16 => "__atomic_fetch_or_2",
            IntWidth::W32 => "__atomic_fetch_or_4",
            IntWidth::W64 => "__atomic_fetch_or_8",
        };
        let fetch_or_fn = self.ctx.get_builtin_function(builtin_name);
        let ordering = self.ordering_int(ordering);
        let int_ty = self.type_cache.resolve_sint(self.ctx, width);
        let value = self.cast(value, int_ty);
        let result = self
            .ctx
            .new_call(None, fetch_or_fn, &[typed_ptr, value, ordering]);
        let int_ty = self.type_cache.resolve_uint(self.ctx, width);
        self.cast(result, int_ty)
    }

    fn lower_atomic_swap(
        &mut self,
        width: IntWidth,
        ptr: RValue<'ctx>,
        value: RValue<'ctx>,
        ordering: Ordering,
    ) -> RValue<'ctx> {
        let ptr_type = self.type_cache.atomic_rmw_ptr_ty(self.ctx);
        let typed_ptr = self.cast(ptr, ptr_type);

        let builtin_name = match width {
            IntWidth::W8 => "__atomic_exchange_1",
            IntWidth::W16 => "__atomic_exchange_2",
            IntWidth::W32 => "__atomic_exchange_4",
            IntWidth::W64 => "__atomic_exchange_8",
        };

        let store_builtin = self.ctx.get_builtin_function(builtin_name);
        let ordering = self.ordering_int(ordering);

        let sint_ty = self.type_cache.resolve_sint(self.ctx, width);
        let uint_ty = self.type_cache.resolve_uint(self.ctx, width);
        let value = self.cast(value, sint_ty);

        let old_value = self
            .ctx
            .new_call(None, store_builtin, &[typed_ptr, value, ordering]);

        self.cast(old_value, uint_ty)
    }

    fn lower_shift(
        &mut self,
        op: ShiftOp,
        width: IntWidth,
        val: RValue<'ctx>,
        shift: RValue<'ctx>,
    ) -> anyhow::Result<RValue<'ctx>> {
        let ty = val.get_type();
        let result = match op {
            // all variables are stored as unsigned integers, so this is good
            ShiftOp::ZeroExtendShr => {
                self.ctx
                    .new_binary_op(None, BinaryOp::RShift, ty, val, shift)
            }
            ShiftOp::SignExtendShr => {
                let signed_int_ty = self.type_cache.resolve_sint(self.ctx, width);
                let signed_val = self.cast(val, signed_int_ty);
                let result = self.ctx.new_binary_op(
                    None,
                    BinaryOp::RShift,
                    signed_int_ty,
                    signed_val,
                    shift,
                );

                self.cast(result, ty)
            }
            ShiftOp::Shl => self
                .ctx
                .new_binary_op(None, BinaryOp::LShift, ty, val, shift),
        };

        Ok(result)
    }

    fn lower_ternary(
        &mut self,
        out_lval: LValue<'ctx>,
        cond_val: RValue<'ctx>,
        true_val: RValue<'ctx>,
        false_val: RValue<'ctx>,
    ) {
        // gccjit has no ternary; emulate with blocks

        assert!(cond_val.get_type().is_bool());
        let then_block = self.func.new_block("ternary_then");
        let else_block = self.func.new_block("ternary_else");
        let merge_block = self.func.new_block("ternary_merge");

        self.current_block
            .unwrap()
            .end_with_conditional(None, cond_val, then_block, else_block);

        then_block.add_assignment(None, out_lval, true_val);
        then_block.end_with_jump(None, merge_block);

        else_block.add_assignment(None, out_lval, false_val);
        else_block.end_with_jump(None, merge_block);

        self.current_block = Some(merge_block);
    }

    fn lower_rvalue(
        &mut self,
        rvalue: &StmtKind,
        outputs: &[SSAValue],
    ) -> anyhow::Result<Option<ArrayVec<RValue<'ctx>, MAX_STMT_OUTPUTS>>> {
        use crate::array_helper::{empty, from_arr};

        Ok(Some(match *rvalue {
            StmtKind::IConst(iconst) => from_arr([self.iconst_rvalue(iconst)]),

            StmtKind::ArithBinOp { op, lhs, rhs } => {
                let (l, r) = (self.use_value(lhs)?, self.use_value(rhs)?);

                let gcc_op = match op {
                    ArithBinOp::Add => BinaryOp::Plus,
                    ArithBinOp::Sub => BinaryOp::Minus,
                    ArithBinOp::Mul => BinaryOp::Mult,
                    ArithBinOp::UncheckedUDiv | ArithBinOp::UncheckedSDiv => BinaryOp::Divide,
                };

                let out = l.get_type();

                // For signed div, cast to signed type first

                let res = match op {
                    ArithBinOp::UncheckedSDiv => {
                        let signed_ty = match self.exec_ir.ssa_values[lhs].ty {
                            IrType::Int(width) => self.type_cache.resolve_sint(self.ctx, width),
                            ty => bail!("invalid integer division of type: {ty:?}"),
                        };

                        let res = self.ctx.new_binary_op(
                            None,
                            gcc_op,
                            signed_ty,
                            self.cast(l, signed_ty),
                            self.cast(r, signed_ty),
                        );

                        self.cast(res, out)
                    }
                    _ => {
                        // We trust the IR to have the right signedness already via IrType
                        self.ctx.new_binary_op(None, gcc_op, out, l, r)
                    }
                };

                from_arr([res])
            }

            StmtKind::AddImm { value, imm64 } => {
                let val = self.use_value(value)?;
                let ty = val.get_type();
                let imm = self.ctx.new_rvalue_from_long(ty, imm64.cast_signed());
                from_arr([self.ctx.new_binary_op(None, BinaryOp::Plus, ty, val, imm)])
            }

            StmtKind::IntNeg(val) => {
                let val = self.use_value(val)?;
                let res = self
                    .ctx
                    .new_unary_op(None, UnaryOp::Minus, val.get_type(), val);
                from_arr([res])
            }

            StmtKind::IntCmp { cmp, lhs, rhs } => {
                let (l, r) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let res = self.lower_cmp(cmp, lhs, l, r)?;
                from_arr([res])
            }

            StmtKind::IntCmpImm { cmp, lhs, rhs } => {
                let l = self.use_value(lhs)?;
                let rhs = self
                    .ctx
                    .new_rvalue_from_long(l.get_type(), rhs.cast_signed());
                let res = self.lower_cmp(cmp, lhs, l, rhs)?;
                from_arr([res])
            }

            StmtKind::Select {
                cond,
                if_true,
                if_false,
            } => {
                let cond_val = self.use_value(cond)?;
                let true_val = self.use_value(if_true)?;
                let false_val = self.use_value(if_false)?;

                let &[output] = outputs.as_array().unwrap();
                let out_lval = self.values[output];
                self.lower_ternary(out_lval, cond_val, true_val, false_val);
                return Ok(None);
            }

            StmtKind::Bitwise { op, lhs, rhs } => {
                let (l, r) = (self.use_value(lhs)?, self.use_value(rhs)?);
                let ty = l.get_type();
                let gcc_op = match op {
                    BitwiseOp::And => BinaryOp::BitwiseAnd,
                    BitwiseOp::Or => BinaryOp::BitwiseOr,
                    BitwiseOp::Xor => BinaryOp::BitwiseXor,
                };
                let result = self.ctx.new_binary_op(None, gcc_op, ty, l, r);
                from_arr([result])
            }

            StmtKind::BitwiseImm { op, lhs, rhs } => {
                let l = self.use_value(lhs)?;
                let ty = l.get_type();
                let imm = self.ctx.new_rvalue_from_long(ty, rhs.cast_signed());
                let gcc_op = match op {
                    BitwiseOp::And => BinaryOp::BitwiseAnd,
                    BitwiseOp::Or => BinaryOp::BitwiseOr,
                    BitwiseOp::Xor => BinaryOp::BitwiseXor,
                };
                let result = self.ctx.new_binary_op(None, gcc_op, ty, l, imm);
                from_arr([result])
            }

            StmtKind::BitNot(val) => {
                let val = self.use_value(val)?;
                let res = self
                    .ctx
                    .new_unary_op(None, UnaryOp::BitwiseNegate, val.get_type(), val);
                from_arr([res])
            }

            StmtKind::Shift { op, value, shift } => {
                let val = self.use_value(value)?;
                let shift = self.use_value(shift)?;
                let ty = val.get_type();

                let IrType::Int(width) = self.exec_ir.ssa_values[value].ty else {
                    bail!("can't shift non integer")
                };

                // if shift >= bits, clamp result
                // aka in range is !(shift >= bits)
                // aka shift < bits
                let in_range = self.ctx.new_comparison(
                    None,
                    ComparisonOp::LessThan,
                    shift,
                    self.ctx.new_rvalue_from_int(
                        shift.get_type(),
                        i32::try_from(width.bits()).context("int width too large")?,
                    ),
                );

                let &[output] = outputs.as_array().unwrap();
                let out_lval = self.values[output];

                // note RValues are lazily evaluated
                // this means that the `lower_ternary`
                // will only evaluate the RValue selected;
                // therefore, there is no chance for UB
                let shift_casted = self.cast(shift, ty);
                let shifted = self.lower_shift(op, width, val, shift_casted)?;

                let overflow_val = match op {
                    ShiftOp::SignExtendShr => {
                        let shift = self.ctx.new_rvalue_from_int(
                            ty,
                            i32::try_from(width.bits().strict_sub(1)).context("shift too wide")?,
                        );

                        self.lower_shift(ShiftOp::SignExtendShr, width, val, shift)?
                    }
                    ShiftOp::ZeroExtendShr | ShiftOp::Shl => {
                        self.ctx.new_rvalue_zero(val.get_type())
                    }
                };

                self.lower_ternary(out_lval, in_range, shifted, overflow_val);

                return Ok(None);
            }

            StmtKind::ShiftImm { op, value, shift } => {
                let IrType::Int(width) = self.exec_ir.ssa_values[value].ty else {
                    bail!("can't shift non integer")
                };

                let val = self.use_value(value)?;
                let shift = self
                    .ctx
                    .new_rvalue_from_int(val.get_type(), i32::from(shift));
                let result = self.lower_shift(op, width, val, shift)?;

                from_arr([result])
            }

            StmtKind::OverflowingBinOp { op, lhs, rhs } => {
                // gccjit has no native overflow intrinsic;
                // use __builtin_add_overflow / __builtin_sub_overflow via inline approach.
                // We compute a widened result and extract the overflow bit manually.

                let l = self.use_value(lhs)?;
                let r = self.use_value(rhs)?;
                let ty = l.get_type();
                let signed_ty = match self.exec_ir.ssa_values[lhs].ty {
                    IrType::Int(width) => self.type_cache.resolve_sint(self.ctx, width),
                    ty => bail!("invalid integer overflowing op on type: {ty:?}"),
                };

                // Widen to i128 for overflow detection (works for all widths <= 64).
                const { assert!(IntWidth::MAX.bits() <= 128) }
                let i128_ty = self.type_cache.i128_ty(self.ctx);

                let l128 = self.cast(self.cast(l, signed_ty), i128_ty);
                let r128 = self.cast(self.cast(r, signed_ty), i128_ty);

                let gcc_op = match op {
                    OverflowingBinOp::Add => BinaryOp::Plus,
                    OverflowingBinOp::Sub => BinaryOp::Minus,
                };
                let wide_result = self.ctx.new_binary_op(None, gcc_op, i128_ty, l128, r128);

                // Truncate back to original type
                let truncated = self.cast(wide_result, ty);

                // Overflow: sign-extend truncated result back to i128 and compare with wide_result
                let sign_extended = self.cast(self.cast(truncated, signed_ty), i128_ty);
                let overflow = self.ctx.new_comparison(
                    None,
                    ComparisonOp::NotEquals,
                    wide_result,
                    sign_extended,
                );

                from_arr([truncated, overflow])
            }

            StmtKind::LoadHost {
                ty,
                base_ptr,
                offset,
                can_move: _,
            } => {
                let ptr_val = self.use_value(base_ptr)?;
                let offset_ptr = self.ptr_add_const(ptr_val, offset)?;
                let load_ty = match ty {
                    LoadType::Int(w) => self.type_cache.resolve_uint(self.ctx, w),
                    LoadType::HostPtr(_) => self.type_cache.void_ptr_ty(self.ctx),
                };

                let loaded = self.load_ptr(offset_ptr, load_ty);
                from_arr([loaded])
            }

            StmtKind::StoreHost {
                base_ptr,
                offset,
                value,
                can_move: _,
            } => {
                let ptr_val = self.use_value(base_ptr)?;
                let value = self.use_value(value)?;
                let offset_ptr = self.ptr_add_const(ptr_val, offset)?;
                self.store_ptr(offset_ptr, value);
                empty()
            }

            StmtKind::LoadStackPtr { slot } => from_arr([self.stack_slots[slot].get_address(None)]),

            StmtKind::PtrAdd {
                base_ptr,
                offset,
                elem_size,
            } => {
                let base = self.use_value(base_ptr)?;
                let off = self.use_value(offset)?;
                let result = self.ptr_add(base, off, elem_size)?;
                from_arr([result])
            }

            StmtKind::PtrEq(ptr_a, ptr_b) => {
                let a = self.use_value(ptr_a)?;
                let b = self.use_value(ptr_b)?;
                let cmp = self.ctx.new_comparison(None, ComparisonOp::Equals, a, b);
                from_arr([cmp])
            }

            StmtKind::HasTag { ptr, tag_bits } => {
                let ptr_val = self.use_value(ptr)?;
                let usize_ty = self.type_cache.usize_ty(self.ctx);
                let ptr_bits = self.ctx.new_bitcast(None, ptr_val, usize_ty);
                let mask = self.ctx.new_rvalue_from_int(usize_ty, tag_bits.into());
                let masked =
                    self.ctx
                        .new_binary_op(None, BinaryOp::BitwiseAnd, usize_ty, ptr_bits, mask);

                let result = self.ctx.new_comparison(
                    None,
                    ComparisonOp::NotEquals,
                    masked,
                    self.ctx.new_rvalue_zero(usize_ty),
                );

                from_arr([result])
            }

            StmtKind::Untag { ptr, tag_bits } => {
                let ptr_val = self.use_value(ptr)?;
                let ptr_ty = ptr_val.get_type();
                let usize_ty = self.type_cache.usize_ty(self.ctx);

                let ptr_bits = self.ctx.new_bitcast(None, ptr_val, usize_ty);
                let not_mask = self.ctx.new_rvalue_from_int(usize_ty, tag_bits.into());
                let mask = self
                    .ctx
                    .new_unary_op(None, UnaryOp::BitwiseNegate, usize_ty, not_mask);

                let masked =
                    self.ctx
                        .new_binary_op(None, BinaryOp::BitwiseAnd, usize_ty, ptr_bits, mask);

                let new_ptr = self.ctx.new_bitcast(None, masked, ptr_ty);

                from_arr([new_ptr])
            }

            StmtKind::HostCallback {
                func: fn_ptr,
                signature,
                ref args,
            } => {
                let fn_ptr_ty = self.fn_ptr_type(signature);
                let ptr_val = fn_ptr as *mut ();
                let fn_ptr_val = self.ctx.new_rvalue_from_ptr(fn_ptr_ty, ptr_val);

                let sig = &self.exec_ir.signatures[signature];
                let gcc_args: Vec<RValue<'ctx>> = args
                    .iter()
                    .map(|&ssa| self.use_value(ssa))
                    .collect::<anyhow::Result<Vec<_>>>()?;

                match sig.ret {
                    Some(_) => {
                        let ret = self.ctx.new_call_through_ptr(None, fn_ptr_val, &gcc_args);
                        from_arr([ret])
                    }
                    None => {
                        let call = self.ctx.new_call_through_ptr(None, fn_ptr_val, &gcc_args);
                        self.current_block.unwrap().add_eval(None, call);
                        empty()
                    }
                }
            }

            StmtKind::VMLoadRaw {
                aligned_page_ptr,
                page_offset,
                width,
            } => {
                let base = self.use_value(aligned_page_ptr)?;
                let off = self.use_value(page_offset)?;
                let ptr = self.ptr_byte_add(base, off)?;
                let load_res = self.lower_atomic_load(width, ptr, Ordering::Relaxed);
                from_arr([self.lower_bswap_to_le(width, load_res)])
            }

            StmtKind::VMStoreRaw {
                aligned_page_ptr,
                page_offset,
                value,
            } => {
                let width = match self.exec_ir.ssa_values[value].ty {
                    IrType::Int(width) => width,
                    _ => unreachable!(),
                };

                let base = self.use_value(aligned_page_ptr)?;
                let off = self.use_value(page_offset)?;
                let val = self.use_value(value)?;
                let ptr = self.ptr_byte_add(base, off)?;
                let le_value = self.lower_bswap_to_le(width, val);

                self.lower_atomic_store(width, ptr, le_value, Ordering::Relaxed);

                empty()
            }

            StmtKind::LoadHaltReason => {
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let reason = self.lower_atomic_load(IntWidth::W32, ptr, Ordering::Relaxed);
                from_arr([reason])
            }

            StmtKind::TakeHaltReason => {
                let ptr = self.use_value(SSAValue::ARG_HALT_REASON_PTR)?;
                let zero = self.u32_const(0);
                let old_val = self.lower_atomic_swap(IntWidth::W32, ptr, zero, Ordering::AcqRel);

                from_arr([old_val])
            }

            StmtKind::GetInstructionDirtyFlag(insn_dirty) => {
                let ptr = self.use_value(insn_dirty)?;
                self.lower_atomic_fence(Ordering::SeqCst);
                let flag = self.lower_atomic_load(IntWidth::W8, ptr, Ordering::SeqCst);
                from_arr([flag])
            }

            StmtKind::SetInstructionDirtyFlag(insn_dirty) => {
                let ptr = self.use_value(insn_dirty)?;
                let true_val = self.u8_const(1);
                let rvalue =
                    self.lower_atomic_fetch_or(IntWidth::W8, ptr, true_val, Ordering::SeqCst);
                self.current_block.unwrap().add_eval(None, rvalue);
                empty()
            }

            StmtKind::Safepoint => {
                // no-op in the JIT; the ir-builder handles the safepoint mechanism
                empty()
            }
        }))
    }

    fn lower_stmt(&mut self, stmt: &StmtData) -> anyhow::Result<()> {
        let output_locs = stmt.outputs.as_slice();
        if let Some(outputs) = self.lower_rvalue(&stmt.rvalue, output_locs)? {
            ensure!(output_locs.len() == outputs.len());
            let block = self.current_block.unwrap();
            for (&lval, rval) in output_locs.iter().zip(outputs) {
                let lval = self.values[lval];
                block.add_assignment(None, lval, rval);
            }
        }

        Ok(())
    }

    fn emit_jump(&mut self, from_block: GccBlock<'ctx>, jump: &IrJump) -> anyhow::Result<()> {
        let target_ir_block = jump.target;
        let target_ref = &self.exec_ir.blocks[target_ir_block];
        let target_gcc = self.gcc_block(target_ir_block)?;

        // Assign jump arguments into the target block's parameter locals.
        ensure!(
            jump.parameters.len() == target_ref.parameters.len(),
            "internal compiler error: jump parameter count mismatch"
        );
        for (&arg_ssa, &param_ssa) in jump.parameters.iter().zip(&target_ref.parameters) {
            let arg_val = self.use_value(arg_ssa)?;
            let param_lval = self.values[param_ssa];
            from_block.add_assignment(None, param_lval, arg_val);
        }

        from_block.end_with_jump(None, target_gcc);
        Ok(())
    }

    fn lower_terminator(
        &mut self,
        gcc_block: GccBlock<'ctx>,
        terminator: &Terminator,
    ) -> anyhow::Result<()> {
        match terminator.kind {
            TerminatorKind::Return => {
                let zero = self.u32_const(0);
                gcc_block.end_with_return(None, zero);
            }

            TerminatorKind::ReturnCode { halt_reason } => {
                let val = self.use_value(halt_reason)?;
                gcc_block.end_with_return(None, val);
            }

            TerminatorKind::Br => {
                let target = &terminator.targets[0];
                self.emit_jump(gcc_block, target)?;
            }

            TerminatorKind::BrZ { cond } => {
                let [zero_target, non_zero_target] = terminator.targets.as_array().unwrap();

                let cond_val = self.use_value(cond)?;
                let cond_ty = cond_val.get_type();

                let is_nonzero = match cond_ty.is_bool() {
                    true => cond_val,
                    false => {
                        // this is needed as the only conditional libgccjit accepts is bool
                        let zero_const = self.ctx.new_rvalue_zero(cond_ty);
                        self.ctx
                            .new_comparison(None, ComparisonOp::NotEquals, cond_val, zero_const)
                    }
                };

                // We need two intermediate blocks to handle the parameter assignments
                // before the actual target jumps.
                let true_trampoline = self.func.new_block("brz_true");
                let false_trampoline = self.func.new_block("brz_false");

                gcc_block.end_with_conditional(None, is_nonzero, true_trampoline, false_trampoline);

                self.emit_jump(true_trampoline, non_zero_target)?;
                self.emit_jump(false_trampoline, zero_target)?;
            }
        }

        Ok(())
    }

    fn lower_blocks(&mut self, exec_ir: &ExecIr) -> anyhow::Result<()> {
        for &ir_block in exec_ir.block_compile_order.iter() {
            let gcc_block = self.gcc_block(ir_block)?;
            let block_data = &exec_ir.blocks[ir_block];

            self.current_block = Some(gcc_block);

            for &stmt_ref in &block_data.stmts {
                let stmt = &exec_ir.stmts[stmt_ref];
                self.lower_stmt(stmt)?;
            }

            self.lower_terminator(self.current_block.unwrap(), &block_data.terminator)?;
        }

        Ok(())
    }
}

pub(crate) struct GccJit(());

impl GccJit {
    pub(crate) fn new() -> anyhow::Result<Self> {
        // gccjit doesn't require any global initialization; contexts are
        // created per-compilation.  We keep the struct as a ZST for now.
        Ok(Self(()))
    }

    pub(crate) fn try_compile(
        &self,
        options: CompileBlockOptions,
        exec_ir: &ExecIr,
    ) -> anyhow::Result<CompiledExecChunk> {
        // funny enough, this actually doesn't help compile times THAT much
        // since well, parallel compilation just simply doesn't work currently
        // gcc_jit_context_compile() - that's where it acquires the global GCC mutex,
        // spins up toplev::main, parses the CLI options, runs the full GCC pipeline.
        // The context creation itself is trivial
        // https://doc.rust-lang.org/std/thread/struct.LocalKey.html#platform-specific-behavior
        thread_local! {
            static PARENT_CONTEX: GccContext<'static> = {
                let ctx = GccContext::default();

                ctx.set_optimization_level(OptimizationLevel::Aggressive);

                ctx.set_debug_info(false);
                ctx.set_dump_everything(false);

                // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66594
                // ctx.add_command_line_option("-march=native");
                ctx.add_command_line_option("-mtune=native");

                if !cfg!(debug_assertions) {
                    ctx.add_command_line_option("-fno-stack-protector");
                }

                if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
                    ctx.add_command_line_option("-masm=intel");
                }

                ctx
            };
        }

        PARENT_CONTEX.with(|parent_ctx| {
            let ctx = parent_ctx.new_child_context();

            ctx.set_print_errors_to_stderr(
                cfg!(any(debug_assertions, test)) || options.show_disasm,
            );

            if options.show_disasm {
                ctx.set_dump_initial_gimple(true);
                ctx.set_dump_code_on_compile(true);
            }

            let mut lowering = FunctionLowering::new(&ctx, exec_ir, &options.function_name)?;
            lowering.lower_blocks(exec_ir)?;

            let result = ctx.compile();
            drop(ctx);

            let code_ptr =
                NonNull::new(result.get_function(&options.function_name)).with_context(|| {
                    format!(
                        "gccjit: compiled function '{}' not found",
                        options.function_name
                    )
                })?;

            let ffi: ExecBlockFFI = unsafe {
                // Safety:
                // - The signature declared above must exactly match ExecBlockFFI.
                // - `result` is kept alive inside the CompiledExecChunk resources.
                std::mem::transmute::<*const (), ExecBlockFFI>(code_ptr.as_ptr())
            };

            struct DropResult {
                _keep_alive: gccjit::CompileResult,
            }

            // Safety: gccjit documents that it is safe to transfer `gcc_jit_result`
            //         between threads, and also use `gcc_jit_contexts` and share them between threads
            // https://gcc.gnu.org/onlinedocs/jit/topics/contexts.html#thread-safety
            // TODO: open an issue in gccjit
            unsafe impl Send for DropResult {}

            Ok(CompiledExecChunk::new_with_recources(
                ffi,
                DropResult {
                    _keep_alive: result,
                },
            ))
        })
    }
}
