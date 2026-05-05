use std::mem::offset_of;
use crate::armv9::{ProcessorState, X_REGISTER_COUNT};
use crate::ir::arena::{impl_storable, Arena, ArenaMap};

mod arena;
mod compiler;


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntWidth {
    W8 = 1,
    W16 = 2,
    W32 = 4,
    W64 = 8,
}

impl IntWidth {
    pub const fn bits(self) -> u32 {
        (self as u32).strict_mul(8)
    }

    pub const fn bytes(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Type {
    Unit,
    Int(IntWidth),
    HostPtr,
}

pub struct LValueData {
    pub ty: Type,
}

// Conceptual signature of every translated basic block:

impl_storable! {
    LValueData as impl pub LValue;
    init: {
        const ARG_PROCESSOR_STATE = LValueData { ty: Type::HostPtr };
        const ARG_PAGES = LValueData { ty: Type::HostPtr };
        const ARG_PAGE_COUNT = LValueData { ty: Type::Int(IntWidth::W64) };
        const ARG_HALT_REASON_PTR = LValueData { ty: Type::HostPtr };
        const ARG_IO_MMU = LValueData { ty: Type::HostPtr };
    }
}

pub enum Arg {
    ProcessorState,
    Pages,
    PageCount,
    HaltReasonPtr,
    IoMMU
}

impl LValue {
    pub fn as_arg(self) -> Option<Arg> {
        match self {
            Self::ARG_PROCESSOR_STATE => Some(Arg::ProcessorState),
            Self::ARG_PAGES => Some(Arg::Pages),
            Self::ARG_PAGE_COUNT => Some(Arg::PageCount),
            Self::ARG_HALT_REASON_PTR => Some(Arg::HaltReasonPtr),
            Self::ARG_IO_MMU => Some(Arg::IoMMU),
            _ => None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArithBinOp {
    /// Wrapping integer add.
    Add,

    /// Wrapping integer subtract.
    Sub,

    /// Wrapping integer multiply.
    Mul,

    /// Unsigned Integer division.
    ///
    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block.
    /// If `rhs == 0`, the result is `0`.
    UDiv,

    /// Signed integer division.
    ///
    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block
    /// and does not update condition flags.
    /// The result is the signed quotient of `lhs / rhs`, rounded toward zero.
    /// If `rhs == 0`, the result is `0`.
    /// If the signed quotient is not representable, i.e. `INT_MIN / -1`,
    /// the result is `INT_MIN`.
    SDiv,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FlagSettingBinOp {
    Add,
    Sub,
}

#[derive(Debug, Copy, Clone)]
pub enum IConst {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64)
}

impl IConst {
    pub const fn width(self) -> IntWidth {
        match self {
            IConst::U8(_) => IntWidth::W8,
            IConst::U16(_) => IntWidth::W16,
            IConst::U32(_) => IntWidth::W32,
            IConst::U64(_) => IntWidth::W64,
        }
    }

    pub const fn zero(width: IntWidth) -> Self {
        match width {
            IntWidth::W8 => Self::U8(0),
            IntWidth::W16 => Self::U16(0),
            IntWidth::W32 => Self::U32(0),
            IntWidth::W64 => Self::U64(0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RValue {
    /// Integer constant.
    IConst(IConst),

    /// Integer arithmetic.
    ///
    /// `lhs` and `rhs` must have type `Int(width)`.
    /// The result also has type `Int(width)`.
    ArithBinOp {
        op: ArithBinOp,
        lhs: LValue,
        rhs: LValue,
    },

    // resulting flags all go into ProcessorState, pstate and in the right posisitons
    FlagSettingBinOp {
        op: FlagSettingBinOp,
        lhs: LValue,
        rhs: LValue,
    },

    /// Load from a host pointer plus a constant byte offset.
    ///
    /// This is used for things like reading `ProcessorState` fields:
    ///
    /// ```text
    /// LoadHost64(processor_state, offset_of!(ProcessorState, x_registers) + 8 * n)
    /// ```
    LoadHost {
        width: IntWidth,
        base_ptr: LValue,
        offset: usize,
    },

    /// Store to a host pointer plus a constant byte offset.
    StoreHost {
        base_ptr: LValue,
        offset: usize,
        value: LValue,
    },

    /// loads the halt reason found at Arg::HaltReasonPtr
    LoadHaltReason,

    /// increments `pc` by 4
    InstructionDone,
}


pub struct Stmt {
    pub lvalue: LValue,
    pub rvalue: RValue,
}


pub enum Terminator {
    /// Return "0" i.e. return success.
    Return,
    /// Return a `NonZero<u32>` block-exit reason.
    ReturnFail {
        halt_reason: LValue
    },
    BrNZ {
        cond: LValue,
        non_zero: Block,
        zero: Block,
    },
    Br(Block),
}

pub struct BlockData {
    pub stmts: Vec<Stmt>,
    pub terminator: Terminator,
    pub is_cold: bool,
}

impl BlockData {
    pub fn empty() -> Self {
        Self {
            stmts: vec![],
            terminator: Terminator::Return,
            is_cold: false,
        }
    }
}


impl_storable!(
    BlockData as impl pub Block;
    init: {
        const ENTRYPOINT = BlockData::empty();
    }
);


pub(crate) struct ExecIr {
    lvalues: Arena<LValueData>,
    blocks: Arena<BlockData>,
}

pub(crate) struct ExecIrBuilder {
    lvalues: Arena<LValueData>,
    blocks: Arena<BlockData>,
    current_block: Block,
    accurate_step: bool,
}

impl ExecIrBuilder {
    pub fn new() -> Self {
        Self {
            lvalues: Arena::new(),
            blocks: Arena::new(),
            current_block: Block::ENTRYPOINT,
            accurate_step: false
        }
    }

    pub fn current_block(&self) -> Block {
        self.current_block
    }

    pub fn create_block(&mut self) -> Block {
        self.blocks.store(BlockData::empty())
    }

    pub fn switch_to(&mut self, block: Block) {
        self.current_block = block;
    }

    pub fn terminate(&mut self, terminator: Terminator) {
        match terminator {
            Terminator::BrNZ { cond: int, .. }
            | Terminator::ReturnFail { halt_reason: int } => {
                assert!(matches!(self.lvalues[int].ty, Type::Int(_)))
            }
            Terminator::Br(_) | Terminator::Return => {}
        }

        let mark_cold = matches!(terminator, Terminator::ReturnFail { .. });
        self.blocks[self.current_block].terminator = terminator;
        if mark_cold {
            self.mark_block_bold()
        }
    }

    pub fn mark_block_bold(&mut self) {
        // a cold entrypoint is insane and should never be true
        // it will always run when the resulting block is compiled
        // it can't be cold
        // this is made just so that if an exec block always unconditionally fails at the end
        // this doesn't accedentally mark that cold
        let block = self.current_block;
        if block != Block::ENTRYPOINT {
            self.blocks[block].is_cold = true;
        }
    }

    fn type_of(&self, rvalue: &RValue) -> Type {
        match *rvalue {
            RValue::IConst(iconst) => Type::Int(iconst.width()),

            RValue::FlagSettingBinOp {
                op: _,
                lhs,
                rhs: _,
            } | RValue::ArithBinOp {
                op: _,
                lhs,
                rhs: _
            } => self.lvalues[lhs].ty,

            RValue::LoadHost { width, .. } => Type::Int(width),
            RValue::StoreHost { .. } => Type::Unit,
            RValue::LoadHaltReason => Type::Int(IntWidth::W32),
            RValue::InstructionDone => Type::Unit,
        }
    }

    unsafe fn emit_stmt(&mut self, rvalue: RValue) -> LValue {
        let ty = self.type_of(&rvalue);
        let current_block = &mut self.blocks[self.current_block];
        let (lvalue, reservation) = self.lvalues.reserve();
        let stmt = Stmt { lvalue, rvalue };
        current_block.stmts.push(stmt);
        reservation.store(LValueData { ty });
        lvalue
    }

    pub fn iconst(&mut self, iconst: IConst) -> LValue {
        unsafe { self.emit_stmt(RValue::IConst(iconst)) }
    }


    unsafe fn load_processor_register(&mut self, offset: usize, width: IntWidth) -> LValue {
        unsafe {
            // SAFETY: `offset` is assumed to be in-bounds for a full processor register
            // stored as a host `u64`. For sub-register loads, the byte offset must be
            // adjusted on big-endian hosts so that loading fewer than 8 bytes reads the
            // low-order bytes of the register. Since `offset` is already within the
            // register slot and the adjustment is at most `size_of::<u64>() - width.bytes()`,
            // the resulting offset remains within that same register.
            let offset = offset.unchecked_add(cfg_select! {
                target_endian = "little" => 0,
                target_endian = "big" => size_of::<u64>().strict_sub(width.bytes()),
            });

            self.emit_stmt(RValue::LoadHost {
                width,
                base_ptr: LValue::ARG_PROCESSOR_STATE,
                offset,
            })
        }
    }

    unsafe fn x_reg_offset(x_reg: u8) -> usize {
        unsafe {
            core::hint::assert_unchecked(x_reg < X_REGISTER_COUNT);

            offset_of!(ProcessorState, x_registers)
                .unchecked_add((x_reg as usize).unchecked_mul(size_of::<u64>()))
        }
    }

    pub fn load_x_reg_dyn(&mut self, x_reg: u8, width: IntWidth) -> LValue {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.load_processor_register(Self::x_reg_offset(x_reg), width) }
    }

    pub fn load_x_reg<const REG_IDX: u8>(&mut self, width: IntWidth) -> LValue {
        const { assert!(REG_IDX < X_REGISTER_COUNT) }
        unsafe { self.load_processor_register(Self::x_reg_offset(REG_IDX), width) }
    }

    pub fn load_sp(&mut self) -> LValue {
        unsafe { self.load_processor_register(offset_of!(ProcessorState, sp), IntWidth::W64) }
    }

    pub fn load_pc(&mut self) -> LValue {
        unsafe { self.load_processor_register(offset_of!(ProcessorState, pc), IntWidth::W64) }
    }

    unsafe fn store_processor_register(&mut self, offset: usize, value: LValue) {
        let Type::Int(IntWidth::W64) = self.lvalues[value].ty else {
            panic!("can only store 64 bit integers to processor registers")
        };

        unsafe {
            self.emit_stmt(RValue::StoreHost {
                base_ptr: LValue::ARG_PROCESSOR_STATE,
                offset,
                value
            });
        }
    }

    pub fn store_x_reg_dyn(&mut self, x_reg: u8, value: LValue) {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.store_processor_register(Self::x_reg_offset(x_reg), value) }
    }

    pub fn store_x_reg<const REG_IDX: u8>(&mut self, value: LValue) {
        const { assert!(REG_IDX < X_REGISTER_COUNT) }
        unsafe { self.store_processor_register(Self::x_reg_offset(REG_IDX), value) }
    }

    pub fn store_sp(&mut self, value: LValue) {
        unsafe { self.store_processor_register(offset_of!(ProcessorState, sp), value) }
    }

    pub fn store_pc(&mut self, value: LValue) {
        unsafe { self.store_processor_register(offset_of!(ProcessorState, pc), value) }
    }

    fn emit_binop(
        &mut self,
        lhs: LValue,
        rhs: LValue,
        func: impl FnOnce(&mut Self, IntWidth) -> LValue
    ) -> LValue {
        let lhs_ty = self.lvalues[lhs].ty;
        let rhs_ty = self.lvalues[rhs].ty;

        let (Type::Int(width), Type::Int(width2)) = (lhs_ty, rhs_ty) else {
            panic!("can only do arithmetic on integers");
        };

        assert_eq!(
            width,
            width2,
            "arithmetic size mismatch; lhs: {expected}, rhs: {found}",
            expected = width.bits(),
            found = width2.bits()
        );

        func(self, width)
    }

    fn emit_arith_binop(&mut self, op: ArithBinOp, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_binop(lhs, rhs, move |this, _width| unsafe {
            this.emit_stmt(RValue::ArithBinOp {
                op,
                lhs,
                rhs,
            })
        })
    }

    fn emit_flag_setting_binop(&mut self, op: FlagSettingBinOp, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_binop(lhs, rhs, move |this, _width| unsafe {
            this.emit_stmt(RValue::FlagSettingBinOp {
                op,
                lhs,
                rhs,
            })
        })
    }

    pub fn add(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Add, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Mul, lhs, rhs)
    }

    pub fn udiv(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::UDiv, lhs, rhs)
    }

    pub fn sdiv(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::SDiv, lhs, rhs)
    }


    pub fn neg(&mut self, value: LValue) -> LValue {
        let Type::Int(width) = self.lvalues[value].ty else {
            panic!("can only negate an integer")
        };

        let zero = self.iconst(IConst::zero(width));

        // TODO add native support for a negate stmt
        self.sub(zero, value)
    }


    pub fn adds(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_flag_setting_binop(FlagSettingBinOp::Add, lhs, rhs)
    }

    pub fn subs(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_flag_setting_binop(FlagSettingBinOp::Sub, lhs, rhs)
    }

    fn insert_halt_check_at(
        &mut self,
        block: Block,
        insert_at: impl FnOnce(&mut BlockData) -> usize
    ) -> Block {
        let (tail_stmts, old_terminator, old_is_cold) = {
            let block_data = &mut self.blocks[block];
            let insert_at = insert_at(block_data);

            if let Some(instruction_end) = insert_at.checked_sub(1) {
                assert!(matches!(
                    block_data.stmts[instruction_end].rvalue,
                    RValue::InstructionDone
                ));
            }

            let tail_stmts = block_data.stmts.split_off(insert_at);
            let old_terminator =
                std::mem::replace(&mut block_data.terminator, Terminator::Return);
            let old_is_cold = block_data.is_cold;
            (tail_stmts, old_terminator, old_is_cold)
        };

        let (halt_reason, reservation) = self.lvalues.reserve();
        reservation.store(LValueData {
            ty: Type::Int(IntWidth::W32),
        });

        let continuation = self.blocks.store(BlockData {
            stmts: tail_stmts,
            terminator: old_terminator,
            is_cold: old_is_cold,
        });

        let fail = self.blocks.store(BlockData {
            stmts: Vec::new(),
            terminator: Terminator::ReturnFail { halt_reason },
            is_cold: true,
        });

        let block_data = &mut self.blocks[block];

        block_data.stmts.push(Stmt {
            lvalue: halt_reason,
            rvalue: RValue::LoadHaltReason,
        });

        block_data.terminator = Terminator::BrNZ {
            cond: halt_reason,
            non_zero: fail,
            zero: continuation,
        };

        continuation
    }

    pub fn instruction_done(&mut self) {
        unsafe {
            self.emit_stmt(RValue::InstructionDone);
            if self.accurate_step {
                // insert at the end, aka just after the instruction ends
                self.current_block = self.insert_halt_check_at(
                    self.current_block,
                    |data| data.stmts.len()
                )
            }
        }
    }

    fn terminator_targets(terminator: &Terminator) -> impl DoubleEndedIterator<Item=Block> {
        match *terminator {
            Terminator::Br(target) => {
                // Keep every match arm returning the same concrete iterator type.
                // This avoids boxing/dynamic dispatch while still letting callers treat this as
                // "zero, one, or two targets".
                let mut iter = [Block::ENTRYPOINT, target].into_iter();

                // Discard the sentinel so the real target is yielded from slot 1.
                // The sentinel value is never observed by callers.
                iter.next();

                iter
            },

            Terminator::Return | Terminator::ReturnFail { .. } => {
                let mut iter = [Block::ENTRYPOINT, Block::ENTRYPOINT].into_iter();
                iter.next();
                iter.next();

                iter
            }

            Terminator::BrNZ {
                non_zero,
                zero,
                ..
            } => [non_zero, zero].into_iter(),
        }
    }

    fn collect_backedge_sources(&self) -> Vec<Block> {
        use std::collections::HashSet;

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        enum DfsState {
            InStack,
            Done,
        }

        let mut state: ArenaMap<Block, DfsState> = ArenaMap::with_capacity(self.blocks.len());
        let mut already_emitted: HashSet<Block> = HashSet::new();
        let mut out = Vec::new();

        // `(block, iter)`.
        let mut stack = Vec::new();

        state.insert(Block::ENTRYPOINT, DfsState::InStack);
        stack.push((
            Block::ENTRYPOINT,
            Self::terminator_targets(&self.blocks[Block::ENTRYPOINT].terminator)
        ));

        while !stack.is_empty() {
            let (source, target) = {
                let &mut (block, ref mut next_target_idx) = stack
                    .last_mut()
                    .expect("checked non-empty stack above");

                (block, next_target_idx.next())
            };

            let Some(target) = target else {
                let (finished, _) = stack.pop().unwrap();
                state.insert(finished, DfsState::Done);
                continue;
            };

            match state.get(target).copied() {
                Some(DfsState::InStack) => {
                    // `source -> target` points to an ancestor in the DFS stack.
                    // That is a DFS backedge, so guard the source block.
                    if already_emitted.insert(source) {
                        out.push(source);
                    }
                }

                Some(DfsState::Done) => { /* Already fully explored */ }

                None => {
                    state.insert(target, DfsState::InStack);
                    stack.push((
                        target,
                        Self::terminator_targets(&self.blocks[target].terminator)
                    ));
                }
            }
        }

        out
    }

    fn insert_halt_check_guard(&mut self, block: Block) {
        // place the halt poll after instruction completion.
        // That preserves the invariant that a translated instruction either fully retires
        // before observing an external halt request, or has not started the next instruction yet.
        self.insert_halt_check_at(block, |block_data| {
            block_data
                .stmts
                .iter()
                .rposition(|stmt| matches!(stmt.rvalue, RValue::InstructionDone))
                .map_or(0, |idx| idx.strict_add(1))
        });
    }

    pub fn build(mut self) -> ExecIr {
        if !self.accurate_step {
            // TODO: only check every N instructions or so
            //       that also means adding in halt checks
            //       and that means simplifying
            //       the accurate_step to be just when N = 0
            //       that also has a speedup effect since it
            //       means we don't poll for halt every single
            //       step of a hot small loop
            for block in self.collect_backedge_sources() {
                self.insert_halt_check_guard(block);
            }
        }


        ExecIr {
            lvalues: self.lvalues,
            blocks: self.blocks,
        }
    }
}


#[cfg(test)]
mod exec_ir_tests {
    use std::sync::LazyLock;
    use crate::cpu_fabric::CpuFabric;
    use crate::halt_reason::AtomicHaltReason;
    use crate::io_mmu::IoMMU;
    use crate::ir::compiler::{CompiledExecBlock, ExecIrCompiler};
    use super::*;

    fn empty_io_mmu() -> IoMMU {
        IoMMU::new(CpuFabric::new())
    }

    static COMPILER: LazyLock<ExecIrCompiler> = LazyLock::new(ExecIrCompiler::new);

    fn compile(builder: ExecIrBuilder) -> CompiledExecBlock {
        COMPILER.compile(builder.build())
    }

    fn call_compiled(
        compiled: &CompiledExecBlock,
        processor_state: &mut ProcessorState,
    ) -> u32 {
        let halt_reason = AtomicHaltReason::new();
        let io_mmu = empty_io_mmu();
        compiled.call(processor_state, &halt_reason, &io_mmu)
    }

    fn run(builder: ExecIrBuilder, processor_state: &mut ProcessorState) -> u32 {
        let compiled = compile(builder);
        call_compiled(&compiled, processor_state)
    }

    fn run_success(builder: ExecIrBuilder, processor_state: &mut ProcessorState) {
        assert_eq!(run(builder, processor_state), 0);
    }

    fn u64_const(builder: &mut ExecIrBuilder, value: u64) -> LValue {
        builder.iconst(IConst::U64(value))
    }

    fn store_x_const<const REG_IDX: u8>(
        builder: &mut ExecIrBuilder,
        value: u64,
    ) {
        let value = u64_const(builder, value);
        builder.store_x_reg::<REG_IDX>(value);
    }

    fn branch_to_store_x1(
        cond: LValue,
        builder: &mut ExecIrBuilder,
        non_zero_value: u64,
        zero_value: u64,
    ) {
        let non_zero = builder.create_block();
        let zero = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond,
            non_zero,
            zero,
        });

        builder.switch_to(non_zero);
        store_x_const::<1>(builder, non_zero_value);

        builder.switch_to(zero);
        store_x_const::<1>(builder, zero_value);
    }

    #[test]
    fn empty_ir_returns_success_and_preserves_basic_state() {
        let mut state = ProcessorState::initial();
        state.sp = 0x1111;
        state.pc = 0x2222;
        state.x_registers[0] = 0x3333;
        state.x_registers[1] = 0x4444;

        let builder = ExecIrBuilder::new();

        run_success(builder, &mut state);

        assert_eq!(state.sp, 0x1111);
        assert_eq!(state.pc, 0x2222);
        assert_eq!(state.x_registers[0], 0x3333);
        assert_eq!(state.x_registers[1], 0x4444);
    }

    #[test]
    fn iconst_can_store_to_x_registers_sp_and_pc() {
        let mut builder = ExecIrBuilder::new();

        store_x_const::<0>(&mut builder, 0x0123_4567_89ab_cdef);
        store_x_const::<1>(&mut builder, 0xfedc_ba98_7654_3210);

        let sp = u64_const(&mut builder, 0x1000_2000_3000_4000);
        builder.store_sp(sp);

        let pc = u64_const(&mut builder, 0x5555_6666_7777_8888);
        builder.store_pc(pc);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0x0123_4567_89ab_cdef);
        assert_eq!(state.x_registers[1], 0xfedc_ba98_7654_3210);
        assert_eq!(state.sp, 0x1000_2000_3000_4000);
        assert_eq!(state.pc, 0x5555_6666_7777_8888);
    }

    #[test]
    fn load_x_reg_const_and_dyn_then_store_roundtrip() {
        let mut builder = ExecIrBuilder::new();

        let x0 = builder.load_x_reg::<0>(IntWidth::W64);
        builder.store_x_reg::<2>(x0);

        let x1 = builder.load_x_reg_dyn(1, IntWidth::W64);
        builder.store_x_reg_dyn(3, x1);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0xaaaa_bbbb_cccc_dddd;
        state.x_registers[1] = 0x1111_2222_3333_4444;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[2], 0xaaaa_bbbb_cccc_dddd);
        assert_eq!(state.x_registers[3], 0x1111_2222_3333_4444);
    }

    #[test]
    fn load_sp_and_pc_can_feed_arithmetic_and_stores() {
        let mut builder = ExecIrBuilder::new();

        let sp = builder.load_sp();
        let sp_delta = u64_const(&mut builder, 0x20);
        let adjusted_sp = builder.add(sp, sp_delta);
        builder.store_x_reg::<0>(adjusted_sp);

        let pc = builder.load_pc();
        let pc_delta = u64_const(&mut builder, 0x44);
        let adjusted_pc = builder.add(pc, pc_delta);
        builder.store_x_reg::<1>(adjusted_pc);

        let mut state = ProcessorState::initial();
        state.sp = 0x1000;
        state.pc = 0x8000;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0x1020);
        assert_eq!(state.x_registers[1], 0x8044);
    }

    #[test]
    fn lower_width_x_register_loads_read_low_order_bits() {
        let mut builder = ExecIrBuilder::new();

        let bad = builder.create_block();

        let got8 = builder.load_x_reg::<0>(IntWidth::W8);
        let expected8 = builder.iconst(IConst::U8(0x88));
        let diff8 = builder.sub(got8, expected8);
        let check16 = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond: diff8,
            non_zero: bad,
            zero: check16,
        });

        builder.switch_to(check16);
        let got16 = builder.load_x_reg::<0>(IntWidth::W16);
        let expected16 = builder.iconst(IConst::U16(0x7788));
        let diff16 = builder.sub(got16, expected16);
        let check32 = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond: diff16,
            non_zero: bad,
            zero: check32,
        });

        builder.switch_to(check32);
        let got32 = builder.load_x_reg::<0>(IntWidth::W32);
        let expected32 = builder.iconst(IConst::U32(0x5566_7788));
        let diff32 = builder.sub(got32, expected32);
        let good = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond: diff32,
            non_zero: bad,
            zero: good,
        });

        builder.switch_to(good);
        store_x_const::<1>(&mut builder, 0x600d);

        builder.switch_to(bad);
        store_x_const::<1>(&mut builder, 0xbad);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0x1122_3344_5566_7788;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[1], 0x600d);
    }

    #[test]
    fn wrapping_add_sub_mul_and_neg_work() {
        let mut builder = ExecIrBuilder::new();

        let max = u64_const(&mut builder, u64::MAX);
        let one = u64_const(&mut builder, 1);
        let add_wrapped = builder.add(max, one);
        builder.store_x_reg::<0>(add_wrapped);

        let zero = u64_const(&mut builder, 0);
        let sub_wrapped = builder.sub(zero, one);
        builder.store_x_reg::<1>(sub_wrapped);

        let high_bit = u64_const(&mut builder, 0x8000_0000_0000_0000);
        let two = u64_const(&mut builder, 2);
        let mul_wrapped = builder.mul(high_bit, two);
        builder.store_x_reg::<2>(mul_wrapped);

        let five = u64_const(&mut builder, 5);
        let neg_five = builder.neg(five);
        builder.store_x_reg::<3>(neg_five);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0);
        assert_eq!(state.x_registers[1], u64::MAX);
        assert_eq!(state.x_registers[2], 0);
        assert_eq!(state.x_registers[3], (!5_u64).wrapping_add(1));
    }

    #[test]
    fn arithmetic_can_use_loaded_registers() {
        let mut builder = ExecIrBuilder::new();

        let x0 = builder.load_x_reg::<0>(IntWidth::W64);
        let x1 = builder.load_x_reg::<1>(IntWidth::W64);
        let sum = builder.add(x0, x1);
        builder.store_x_reg::<2>(sum);

        let x2 = builder.load_x_reg::<2>(IntWidth::W64);
        let three = u64_const(&mut builder, 3);
        let product = builder.mul(x2, three);
        builder.store_x_reg::<3>(product);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 9;
        state.x_registers[1] = 11;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[2], 20);
        assert_eq!(state.x_registers[3], 60);
    }

    #[test]
    fn unsigned_division_handles_normal_and_zero_divisors() {
        let mut builder = ExecIrBuilder::new();

        let hundred = u64_const(&mut builder, 100);
        let seven = u64_const(&mut builder, 7);
        let quotient = builder.udiv(hundred, seven);
        builder.store_x_reg::<0>(quotient);

        let numerator = u64_const(&mut builder, 1234);
        let zero = u64_const(&mut builder, 0);
        let div_by_zero = builder.udiv(numerator, zero);
        builder.store_x_reg::<1>(div_by_zero);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 14);
        assert_eq!(state.x_registers[1], 0);
    }

    #[test]
    fn signed_division_handles_normal_zero_and_overflow_cases() {
        let mut builder = ExecIrBuilder::new();

        let minus_seventeen = u64_const(&mut builder, (-17_i64).cast_unsigned());
        let five = u64_const(&mut builder, 5);
        let quotient = builder.sdiv(minus_seventeen, five);
        builder.store_x_reg::<0>(quotient);

        let numerator = u64_const(&mut builder, (-123_i64).cast_unsigned());
        let zero = u64_const(&mut builder, 0);
        let div_by_zero = builder.sdiv(numerator, zero);
        builder.store_x_reg::<1>(div_by_zero);

        let int_min = u64_const(&mut builder, i64::MIN.cast_unsigned());
        let minus_one = u64_const(&mut builder, (-1_i64).cast_unsigned());
        let overflow = builder.sdiv(int_min, minus_one);
        builder.store_x_reg::<2>(overflow);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
        assert_eq!(state.x_registers[1], 0);
        assert_eq!(state.x_registers[2], i64::MIN.cast_unsigned());
    }

    #[test]
    fn flag_setting_binops_produce_storable_values() {
        let mut builder = ExecIrBuilder::new();

        let lhs = builder.load_x_reg::<0>(IntWidth::W64);
        let rhs = builder.load_x_reg::<1>(IntWidth::W64);
        let sum = builder.adds(lhs, rhs);
        builder.store_x_reg::<2>(sum);

        let lhs = builder.load_x_reg::<0>(IntWidth::W64);
        let rhs = builder.load_x_reg::<1>(IntWidth::W64);
        let diff = builder.subs(lhs, rhs);
        builder.store_x_reg::<3>(diff);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 40;
        state.x_registers[1] = 58;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[2], 98);
        assert_eq!(state.x_registers[3], 40_u64.wrapping_sub(58));
    }

    #[test]
    fn brnz_takes_zero_path_for_zero_condition() {
        let mut builder = ExecIrBuilder::new();

        let cond = builder.load_x_reg::<0>(IntWidth::W64);
        branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[1], 0xbbbb);
    }

    #[test]
    fn brnz_takes_non_zero_path_for_non_zero_condition() {
        let mut builder = ExecIrBuilder::new();

        let cond = builder.load_x_reg::<0>(IntWidth::W64);
        branch_to_store_x1(cond, &mut builder, 0xaaaa, 0xbbbb);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 42;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[1], 0xaaaa);
    }

    #[test]
    fn brnz_accepts_narrow_integer_conditions() {
        let mut builder = ExecIrBuilder::new();

        let cond = builder.load_x_reg::<0>(IntWidth::W8);
        branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0x0100;

        run_success(builder, &mut state);

        assert_eq!(
            state.x_registers[1],
            0x2222,
            "low byte is zero, so W8 condition must be false",
        );

        let mut builder = ExecIrBuilder::new();

        let cond = builder.load_x_reg::<0>(IntWidth::W8);
        branch_to_store_x1(cond, &mut builder, 0x1111, 0x2222);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0x0101;

        run_success(builder, &mut state);

        assert_eq!(
            state.x_registers[1],
            0x1111,
            "low byte is non-zero, so W8 condition must be true",
        );
    }

    #[test]
    #[allow(clippy::unusual_byte_groupings)]
    fn unconditional_branch_executes_target_block() {
        let mut builder = ExecIrBuilder::new();

        let target = builder.create_block();
        builder.terminate(Terminator::Br(target));

        builder.switch_to(target);
        store_x_const::<0>(&mut builder, 0xdecaf_bad);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0xdecaf_bad);
    }

    #[test]
    fn diamond_control_flow_rejoins_through_processor_state() {
        fn build_program() -> ExecIrBuilder {
            let mut builder = ExecIrBuilder::new();

            let non_zero = builder.create_block();
            let zero = builder.create_block();
            let join = builder.create_block();

            let cond = builder.load_x_reg::<0>(IntWidth::W64);
            builder.terminate(Terminator::BrNZ {
                cond,
                non_zero,
                zero,
            });

            builder.switch_to(non_zero);
            store_x_const::<1>(&mut builder, 40);
            builder.terminate(Terminator::Br(join));

            builder.switch_to(zero);
            store_x_const::<1>(&mut builder, 2);
            builder.terminate(Terminator::Br(join));

            builder.switch_to(join);
            let x1 = builder.load_x_reg::<1>(IntWidth::W64);
            let one = u64_const(&mut builder, 1);
            let result = builder.add(x1, one);
            builder.store_x_reg::<2>(result);

            builder
        }

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0;
        run_success(build_program(), &mut state);
        assert_eq!(state.x_registers[1], 2);
        assert_eq!(state.x_registers[2], 3);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 1;
        run_success(build_program(), &mut state);
        assert_eq!(state.x_registers[1], 40);
        assert_eq!(state.x_registers[2], 41);
    }

    #[test]
    fn manually_cold_block_still_executes_when_reached() {
        let mut builder = ExecIrBuilder::new();

        let hot = builder.create_block();
        let cold = builder.create_block();

        let cond = builder.load_x_reg::<0>(IntWidth::W64);
        builder.terminate(Terminator::BrNZ {
            cond,
            non_zero: cold,
            zero: hot,
        });

        builder.switch_to(hot);
        store_x_const::<1>(&mut builder, 0x1234);

        builder.switch_to(cold);
        builder.mark_block_bold();
        store_x_const::<1>(&mut builder, 0xc01d);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 1;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[1], 0xc01d);
    }

    #[test]
    fn instruction_done_increments_pc_by_four_each_time() {
        let mut builder = ExecIrBuilder::new();

        builder.instruction_done();
        builder.instruction_done();
        builder.instruction_done();

        let mut state = ProcessorState::initial();
        state.pc = 0x1000;

        run_success(builder, &mut state);

        assert_eq!(state.pc, 0x100c);
    }

    #[test]
    fn explicit_pc_store_then_instruction_done_uses_new_pc() {
        let mut builder = ExecIrBuilder::new();

        let pc = u64_const(&mut builder, 0x2000);
        builder.store_pc(pc);
        builder.instruction_done();

        let mut state = ProcessorState::initial();
        state.pc = 0x1000;

        run_success(builder, &mut state);

        assert_eq!(state.pc, 0x2004);
    }

    #[test]
    fn simple_counted_loop_executes_until_condition_is_zero() {
        let mut builder = ExecIrBuilder::new();

        let loop_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.terminate(Terminator::Br(loop_block));

        builder.switch_to(loop_block);
        let current = builder.load_x_reg::<0>(IntWidth::W64);
        let one = u64_const(&mut builder, 1);
        let next = builder.sub(current, one);
        builder.store_x_reg::<0>(next);
        builder.terminate(Terminator::BrNZ {
            cond: next,
            non_zero: loop_block,
            zero: exit_block,
        });

        builder.switch_to(exit_block);
        store_x_const::<1>(&mut builder, 0x0600_d100_u64);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 7;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0);
        assert_eq!(state.x_registers[1], 0x0600_d100_u64);
    }

    #[test]
    fn compiled_block_can_be_called_more_than_once() {
        let mut builder = ExecIrBuilder::new();

        let x0 = builder.load_x_reg::<0>(IntWidth::W64);
        let one = u64_const(&mut builder, 1);
        let incremented = builder.add(x0, one);
        builder.store_x_reg::<0>(incremented);

        let compiled = compile(builder);

        let mut first = ProcessorState::initial();
        first.x_registers[0] = 10;
        assert_eq!(call_compiled(&compiled, &mut first), 0);
        assert_eq!(first.x_registers[0], 11);

        let mut second = ProcessorState::initial();
        second.x_registers[0] = u64::MAX;
        assert_eq!(call_compiled(&compiled, &mut second), 0);
        assert_eq!(second.x_registers[0], 0);
    }

    #[test]
    fn builder_current_block_tracks_switches() {
        let mut builder = ExecIrBuilder::new();

        assert_eq!(builder.current_block(), Block::ENTRYPOINT);

        let other = builder.create_block();
        builder.switch_to(other);

        assert_eq!(builder.current_block(), other);
    }

    #[test]
    #[should_panic(expected = "arithmetic size mismatch")]
    fn builder_rejects_arithmetic_width_mismatch() {
        let mut builder = ExecIrBuilder::new();

        let wide = builder.iconst(IConst::U64(1));
        let narrow = builder.iconst(IConst::U32(1));

        let _ = builder.add(wide, narrow);
    }

    #[test]
    #[should_panic(expected = "can only store 64 bit integers to processor registers")]
    fn builder_rejects_storing_narrow_value_to_processor_register() {
        let mut builder = ExecIrBuilder::new();

        let narrow = builder.iconst(IConst::U32(1));
        builder.store_x_reg::<0>(narrow);
    }

    #[test]
    #[should_panic]
    fn load_x_reg_dyn_rejects_out_of_range_register() {
        let mut builder = ExecIrBuilder::new();

        let _ = builder.load_x_reg_dyn(X_REGISTER_COUNT, IntWidth::W64);
    }

    #[test]
    #[should_panic]
    fn store_x_reg_dyn_rejects_out_of_range_register() {
        let mut builder = ExecIrBuilder::new();

        let value = u64_const(&mut builder, 1);
        builder.store_x_reg_dyn(X_REGISTER_COUNT, value);
    }
}