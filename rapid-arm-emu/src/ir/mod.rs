use std::mem::offset_of;
use std::num::NonZero;
use arrayvec::ArrayVec;
use crate::armv9::{PState, ProcessorState, X_REGISTER_COUNT};
use crate::array_helper;
use crate::ir::arena::{impl_storable, Arena, ArenaSet};

mod arena;
pub(crate) mod compiler;
pub(crate) mod ffi_support;
mod halt_check_pass;


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntWidth {
    W8 = 1,
    W16 = 2,
    W32 = 4,
    W64 = 8,
}

impl IntWidth {
    pub const fn from_bits(bits: u32) -> Option<Self> {
        Some(match bits {
            8 => Self::W8,
            16 => Self::W16,
            32 => Self::W32,
            64 => Self::W64,
            _ => return None
        })
    }

    pub const fn bits(self) -> u32 {
        (self as u32).strict_mul(8)
    }

    pub const fn bytes(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Type {
    Int(IntWidth),
    Bool,
    HostPtr,
}

impl Type {
    pub fn assert_int(self, op_name: &str) -> IntWidth {
        let Type::Int(width) = self else {
            panic!("can only do integer {op_name} on integers");
        };
        width
    }
}

#[derive(Debug)]
struct LValueData {
    pub ty: Type,
}

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

#[derive(Debug, Copy, Clone)]
pub struct IConst {
    width: IntWidth,
    bits: u64,
}

impl IConst {
    pub const fn width(self) -> IntWidth {
        self.width
    }

    pub const fn zero(width: IntWidth) -> Self {
        Self {
            width,
            bits: 0,
        }
    }

    pub const fn one(width: IntWidth) -> Self {
        Self {
            width,
            bits: 1,
        }
    }

    pub const fn min_negative(width: IntWidth) -> Self {
        match width {
            IntWidth::W8 => const { Self::i8(i8::MIN) }
            IntWidth::W16 => const { Self::i16(i16::MIN) }
            IntWidth::W32 => const { Self::i32(i32::MIN) }
            IntWidth::W64 => const { Self::i64(i64::MIN) }
        }
    }


    pub const fn negative_one(width: IntWidth) -> Self {
        let bits = width.bits();
        assert!(bits <= 64);
        Self {
            width,
            // its 2^n - 1 which encodes -1 in the given bit range
            // except when n == 64 then its  0 - 1 which is still -1 ofr 64 bit integers
            bits: 1_u64.unbounded_shl(bits).wrapping_sub(1)
        }
    }
}

macro_rules! zero_extend_u64 {
    (u64, $value: expr) => { $value };
    (i64, $value: expr) => { ($value).cast_unsigned() };


    (u32, $value: expr) => { $value as u64 };
    (u16, $value: expr) => { $value as u64 };
    (u8, $value: expr) => { $value as u64 };

    (i32, $value: expr) => { ($value).cast_unsigned() as u64 };
    (i16, $value: expr) => { ($value).cast_unsigned() as u64 };
    (i8, $value: expr) => { ($value).cast_unsigned() as u64 };
}

macro_rules! impl_primitive_constructors {
    ($($int_ty: ident)+) => {
        impl IConst {
            $(
            #[inline(always)]
            pub const fn $int_ty(value: $int_ty) -> Self {
                let width = const { IntWidth::from_bits($int_ty::BITS).unwrap() };
                Self {
                    width,
                    bits: zero_extend_u64!($int_ty, value)
                }
            }
            )+
        }
    };
}

impl_primitive_constructors! {
    u64 u32 u16 u8
    i64 i32 i16 i8
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
    /// it is UB if `rhs == 0`
    UncheckedUDiv,

    /// Signed integer division.
    ///
    /// This is a normal value-producing bin-op.
    /// it is UB if `rhs == 0` OR `lhs == <ty>::MIN AND rhs == -1`
    UncheckedSDiv,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OverflowingBinOp {
    Add,
    Sub,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum BitwiseOp {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum IntCmp {
    /// `==`.
    Equal,
    /// `!=`.
    NotEqual,
    /// Signed `<`.
    SignedLessThan,
    /// Signed `>=`.
    SignedGreaterThanOrEqual,
    /// Signed `>`.
    SignedGreaterThan,
    /// Signed `<=`.
    SignedLessThanOrEqual,
    /// Unsigned `<`.
    UnsignedLessThan,
    /// Unsigned `>=`.
    UnsignedGreaterThanOrEqual,
    /// Unsigned `>`.
    UnsignedGreaterThan,
    /// Unsigned `<=`.
    UnsignedLessThanOrEqual,
}


const MAX_STMT_OUTPUTS: usize = 2;

#[derive(Debug, Clone)]
enum StmtKind {
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

    AddImm {
        value: LValue,
        imm64: u64
    },

    /// Produces:
    ///   0: arithmetic result
    ///   1: overflow flag
    OverflowingBinOp {
        op: OverflowingBinOp,
        lhs: LValue,
        rhs: LValue,
    },


    IntCmp {
        cmp: IntCmp,
        lhs: LValue,
        rhs: LValue,
    },

    IntCmpImm {
        cmp: IntCmp,
        lhs: LValue,
        rhs: u64,
    },

    Select {
        cond: LValue,
        if_true: LValue,
        if_false: LValue,
    },

    Bitwise {
        op: BitwiseOp,
        lhs: LValue,
        rhs: LValue,
    },

    BitwiseImm {
        op: BitwiseOp,
        lhs: LValue,
        rhs: u64,
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

    Safepoint
}

#[derive(Debug)]
struct Stmt {
    outputs: ArrayVec<LValue, MAX_STMT_OUTPUTS>,
    rvalue: StmtKind,
}


#[derive(Debug)]
pub enum Terminator {
    /// Return "0" i.e. return success.
    Return,
    /// Return a `NonZero<u32>` block-exit reason.
    ReturnCode {
        halt_reason: LValue
    },
    BrNZ {
        cond: LValue,
        non_zero: Block,
        zero: Block,
    },
    Br(Block),
}

impl Terminator {
    pub const MAX_TARGETS: usize = 2;

    pub fn targets(&self) -> arrayvec::IntoIter<Block, { Self::MAX_TARGETS }> {
        match *self {
            Terminator::Br(target) => array_helper::iter_from_arr([target]),


            Terminator::Return | Terminator::ReturnCode { .. } => array_helper::empty(),

            Terminator::BrNZ {
                non_zero,
                zero,
                ..
            } => array_helper::iter_from_arr([non_zero, zero]),
        }
    }
}


#[derive(Debug)]
struct BlockData {
    predecessors: Vec<Block>,
    stmts: Vec<Stmt>,
    terminated: bool,
    terminator: Terminator,
    is_cold: bool,
}

impl BlockData {
    pub fn empty() -> Self {
        Self {
            predecessors: vec![],
            stmts: vec![],
            terminated: false,
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
    block_compile_order: Vec<Block>,
    halt_check_every: NonZero<u32>,
}


pub(crate) struct ExecIrBuilder {
    lvalues: Arena<LValueData>,
    blocks: Arena<BlockData>,
    current_block: Block,
    halt_check_every: NonZero<u32>,
}

pub(crate) struct IrBuilderConfig {
    halt_check_every: NonZero<u32>,
}

impl ExecIrBuilder {
    pub fn with_config(config: IrBuilderConfig) -> Self {
        Self {
            lvalues: Arena::new(),
            blocks: Arena::new(),
            current_block: Block::ENTRYPOINT,
            halt_check_every: config.halt_check_every,
        }
    }

    pub fn new() -> Self {
        let halt_check_every = const { NonZero::new(64).unwrap() };
        Self::with_config(IrBuilderConfig { halt_check_every })
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


    pub fn predecessors(&self, block: Block) -> &[Block] {
        self.blocks[block].predecessors.as_slice()
    }

    pub fn successors(
        &self,
        block: Block
    ) -> arrayvec::IntoIter<Block, { Terminator::MAX_TARGETS }> {
        self.blocks[block].terminator.targets()
    }

    pub fn terminate_block(
        &mut self,
        block: Block,
        mut terminator: Terminator
    ) {

        match terminator {
            Terminator::Return => {},

            Terminator::ReturnCode { halt_reason: int } => {
                assert!(matches!(self.lvalues[int].ty, Type::Int(_)))
            }

            Terminator::BrNZ { cond: int, zero, non_zero } => {
                assert!(matches!(self.lvalues[int].ty, Type::Bool | Type::Int(_)));
                if zero == non_zero {
                    terminator = Terminator::Br(zero)
                }
            }

            Terminator::Br(_) => {},
        }

        for target in terminator.targets() {
            assert_ne!(target, Block::ENTRYPOINT, "can't branch to entrypoint");
        }

        let mark_cold = matches!(terminator, Terminator::ReturnCode { .. });

        for target in terminator.targets() {
            self.blocks[target].predecessors.push(block)
        }

        let block_data = &mut self.blocks[block];
        assert!(!block_data.terminated);
        block_data.terminator = terminator;
        block_data.terminated = true;

        if mark_cold {
            self.mark_block_bold()
        }
    }

    pub fn terminate(&mut self, terminator: Terminator) {
        let current_block = self.current_block;
        self.terminate_block(current_block, terminator)
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

    fn type_of(&self, rvalue: &StmtKind) -> arrayvec::IntoIter<Type, MAX_STMT_OUTPUTS> {
        match *rvalue {
            StmtKind::IConst(iconst) => {
                array_helper::iter_from_arr([Type::Int(iconst.width())])
            },


            StmtKind::ArithBinOp { lhs, .. }
            | StmtKind::AddImm { value: lhs, .. }
            | StmtKind::Bitwise { lhs, .. }
            | StmtKind::BitwiseImm { lhs, .. }
            | StmtKind::Select { cond: _, if_true: lhs, if_false: _ } => {
                array_helper::iter_from_arr([self.lvalues[lhs].ty])
            },

            StmtKind::OverflowingBinOp {
                op: _,
                lhs,
                rhs: _,
            } => array_helper::iter_from_arr([self.lvalues[lhs].ty, Type::Bool]),

            StmtKind::IntCmp { .. } | StmtKind::IntCmpImm { .. } => {
                array_helper::iter_from_arr([Type::Bool])
            },

            StmtKind::LoadHost { width, .. } => {
                array_helper::iter_from_arr([Type::Int(width)])
            },

            StmtKind::StoreHost { .. } => array_helper::empty(),
            StmtKind::LoadHaltReason => array_helper::iter_from_arr([Type::Int(IntWidth::W32)]),
            StmtKind::Safepoint => array_helper::empty(),
        }
    }

    /// # Safety
    ///
    /// the IR must not produce UB when run after compilation
    unsafe fn emit_stmt_full<const N: usize>(&mut self, rvalue: StmtKind) -> [LValue; N] {
        let outputs = self.type_of(&rvalue)
            .map(|ty| self.lvalues.store(LValueData { ty }))
            .collect::<ArrayVec<LValue, MAX_STMT_OUTPUTS>>();

        let emit_out: &[LValue] = outputs.as_slice();
        let emit_out: [LValue; N] = *emit_out.as_array()
            .expect("invalid stmt output amount");

        self.blocks[self.current_block].stmts.push(Stmt { outputs, rvalue });

        emit_out
    }


    #[inline]
    unsafe fn emit_void_stmt(&mut self, rvalue: StmtKind) {
        let [] = unsafe { self.emit_stmt_full(rvalue) };
    }

    #[inline]
    #[must_use]
    unsafe fn emit_1ret_stmt(&mut self, rvalue: StmtKind) -> LValue {
        let [value] = unsafe { self.emit_stmt_full(rvalue) };
        value
    }

    #[inline]
    #[must_use]
    unsafe fn emit_2ret_stmt(&mut self, rvalue: StmtKind) -> (LValue, LValue) {
        let [value1, value2] = unsafe { self.emit_stmt_full(rvalue) };
        (value1, value2)
    }

    pub fn iconst(&mut self, iconst: IConst) -> LValue {
        unsafe { self.emit_1ret_stmt(StmtKind::IConst(iconst)) }
    }

    unsafe fn load_from_processor_state(&mut self, offset: usize, width: IntWidth) -> LValue {
        unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                width,
                base_ptr: LValue::ARG_PROCESSOR_STATE,
                offset,
            })
        }
    }

    unsafe fn load_from_64_bit_processor_register(&mut self, offset: usize, width: IntWidth) -> LValue {
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

            self.load_from_processor_state(offset, width)
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
        unsafe { self.load_from_64_bit_processor_register(Self::x_reg_offset(x_reg), width) }
    }

    pub fn load_x_reg<const REG_IDX: u8>(&mut self, width: IntWidth) -> LValue {
        const { assert!(REG_IDX < X_REGISTER_COUNT) }
        unsafe { self.load_from_64_bit_processor_register(Self::x_reg_offset(REG_IDX), width) }
    }


    pub fn load_sp(&mut self) -> LValue {
        unsafe {
            self.load_from_processor_state(offset_of!(ProcessorState, sp), IntWidth::W64)
        }
    }

    pub fn load_pc(&mut self) -> LValue {
        unsafe {
            self.load_from_processor_state(offset_of!(ProcessorState, pc), IntWidth::W64)
        }
    }

    pub fn load_pstate(&mut self) -> LValue {
        unsafe {
            self.load_from_processor_state(offset_of!(ProcessorState, pstate), IntWidth::W32)
        }
    }

    unsafe fn store_to_processor_state(&mut self, offset: usize, value: LValue) {
        unsafe {
            self.emit_void_stmt(StmtKind::StoreHost {
                base_ptr: LValue::ARG_PROCESSOR_STATE,
                offset,
                value
            })
        }
    }


    unsafe fn store_processor_register(&mut self, offset: usize, value: LValue) {
        let Type::Int(IntWidth::W64) = self.lvalues[value].ty else {
            panic!("can only store 64 bit integers to processor registers")
        };

        unsafe { self.store_to_processor_state(offset, value) }
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

    pub fn store_pstate(&mut self, value: LValue) {
        let Type::Int(IntWidth::W32) = self.lvalues[value].ty else {
            panic!("can only store 32 bit integers to pstate")
        };

        unsafe {
            self.store_to_processor_state(
                offset_of!(ProcessorState, pstate),
                value
            )
        }
    }

    pub fn select(&mut self, cond: LValue, if_true: LValue, if_false: LValue) -> LValue {
        assert_eq!(self.lvalues[cond].ty, Type::Bool, "condition must have bool type");
        assert_eq!(self.lvalues[if_true].ty, self.lvalues[if_false].ty, "select type mismatch");
        unsafe { self.emit_1ret_stmt(StmtKind::Select { cond, if_true, if_false }) }
    }



    fn emit_same_int_ty_imm<T>(
        &mut self,
        op_name: &'static str,
        lhs: LValue,
        rhs: IntWidth,
        func: impl FnOnce(&mut Self) -> T
    ) -> T {
        let lhs_ty = self.lvalues[lhs].ty;
        let width = lhs_ty.assert_int(op_name);

        assert_eq!(
            width,
            rhs,
            "arithmetic size mismatch; lhs: {expected}, rhs: {found}",
            expected = width.bits(),
            found = rhs.bits()
        );

        func(self)
    }

    fn emit_same_int_ty_binop<T>(
        &mut self,
        op_name: &'static str,
        lhs: LValue,
        rhs: LValue,
        func: impl FnOnce(&mut Self, IntWidth) -> T
    ) -> T {
        let rhs = self.lvalues[rhs].ty.assert_int(op_name);
        self.emit_same_int_ty_imm(
            op_name,
            lhs,
            rhs,
            |this| func(this, rhs)
        )
    }


    pub fn icmp(
        &mut self,
        cmp: IntCmp,
        lhs: LValue,
        rhs: LValue,
    ) -> LValue {
        self.emit_same_int_ty_binop(
            "comparisons",
            lhs,
            rhs,
            |this, _width| unsafe {
                this.emit_1ret_stmt(StmtKind::IntCmp {
                    cmp,
                    lhs,
                    rhs,
                })
            }
        )
    }

    pub fn icmp_imm(
        &mut self,
        cmp: IntCmp,
        lhs: LValue,
        rhs: IConst,
    ) -> LValue {
        self.emit_same_int_ty_imm(
            "comparisons",
            lhs,
            rhs.width,
            |this| unsafe {
                this.emit_1ret_stmt(StmtKind::IntCmpImm {
                    cmp,
                    lhs,
                    rhs: rhs.bits,
                })
            }
        )
    }

    fn binop_type_guard<T>(
        &mut self,
        lhs: Type,
        rhs: Type,
        emit: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match (lhs, rhs) {
            (Type::Bool, Type::Bool) => emit(self),
            (Type::Int(width1), Type::Int(width2)) => match width1 == width2 {
                true => emit(self),
                false => panic!("mismatched integer widths used for bitwise op")
            },

            (Type::HostPtr, Type::HostPtr) => {
                panic!("can't do pointer bitwise operations currently")
            },

            _ => panic!("mismatched types used for bitwise operation")
        }
    }


    fn emit_binop(&mut self, op: BitwiseOp, lhs: LValue, rhs: LValue) -> LValue {
        let lhs_ty = self.lvalues[lhs].ty;
        let rhs_ty = self.lvalues[rhs].ty;
        self.binop_type_guard(lhs_ty, rhs_ty, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::Bitwise { op, lhs, rhs })
        })
    }

    fn emit_binop_imm(&mut self, op: BitwiseOp, lhs: LValue, rhs: IConst) -> LValue {
        let lhs_ty = self.lvalues[lhs].ty;
        self.binop_type_guard(lhs_ty, Type::Int(rhs.width), |this| unsafe {
            this.emit_1ret_stmt(StmtKind::BitwiseImm { op, lhs, rhs: rhs.bits })
        })
    }

    pub fn bitor(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_binop(BitwiseOp::Or, lhs, rhs)
    }

    pub fn bitand(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_binop(BitwiseOp::And, lhs, rhs)
    }

    pub fn bitxor(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_binop(BitwiseOp::Xor, lhs, rhs)
    }

    pub fn bitor_imm(&mut self, lhs: LValue, rhs: IConst) -> LValue {
        self.emit_binop_imm(BitwiseOp::Or, lhs, rhs)
    }

    pub fn bitand_imm(&mut self, lhs: LValue, rhs: IConst) -> LValue {
        self.emit_binop_imm(BitwiseOp::And, lhs, rhs)
    }

    pub fn bitxor_imm(&mut self, lhs: LValue, rhs: IConst) -> LValue {
        self.emit_binop_imm(BitwiseOp::Xor, lhs, rhs)
    }

    pub fn set_nzcv_flags(
        &mut self,
        n: LValue,
        z: LValue,
        c: LValue,
        v: LValue,
    ) {
        let old_flags = self.load_pstate();

        let zeroed = self.iconst(IConst::u32(0));

        let n_flag_true = self.iconst(IConst::u32(PState::N.0));
        let z_flag_true = self.iconst(IConst::u32(PState::Z.0));
        let c_flag_true = self.iconst(IConst::u32(PState::C.0));
        let v_flag_true = self.iconst(IConst::u32(PState::V.0));

        let n_flag = self.select(n, n_flag_true, zeroed);
        let z_flag = self.select(z, z_flag_true, zeroed);
        let c_flag = self.select(c, c_flag_true, zeroed);
        let v_flag = self.select(v, v_flag_true, zeroed);

        let nz_flag = self.bitor(n_flag, z_flag);
        let cv_flag = self.bitor(c_flag, v_flag);
        let nzcv_flags = self.bitor(nz_flag, cv_flag);

        let masked_out_flags = self.bitand_imm(old_flags, IConst::u32(!PState::NZCV_MASK.0));
        let new_flags = self.bitor(masked_out_flags, nzcv_flags);

        self.store_pstate(new_flags);
    }


    fn emit_arith_binop(&mut self, op: ArithBinOp, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_same_int_ty_binop(
            "arithmetic",
            lhs,
            rhs,
            move |this, _width| unsafe {
                this.emit_1ret_stmt(StmtKind::ArithBinOp {
                    op,
                    lhs,
                    rhs,
                })
            }
        )
    }

    fn emit_flag_setting_binop(
        &mut self,
        op: OverflowingBinOp,
        lhs: LValue,
        rhs: LValue
    ) -> LValue {
        let (value, overflow, width) = {
            self.emit_same_int_ty_binop(
                "overflowing arithmetic",
                lhs,
                rhs,
                move |this, width| {
                    let (value, overflow) = unsafe {
                        this.emit_2ret_stmt(StmtKind::OverflowingBinOp {
                            op,
                            lhs,
                            rhs,
                        })
                    };

                    (value, overflow, width)
                }
            )
        };

        let zero_imm = IConst::zero(width);

        let negative = self.icmp_imm(IntCmp::SignedLessThan, value, zero_imm);
        let zero = self.icmp_imm(IntCmp::Equal, value, zero_imm);
        let carry = match op {
            OverflowingBinOp::Add => self.icmp(IntCmp::UnsignedLessThan, value, lhs),
            OverflowingBinOp::Sub => self.icmp(IntCmp::UnsignedGreaterThanOrEqual, lhs, rhs),
        };

        self.set_nzcv_flags(negative, zero, carry, overflow);

        value
    }

    pub fn add(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Add, lhs, rhs)
    }

    pub fn add_imm(&mut self, value: LValue, amount: IConst) -> LValue {
        // FIXME clean up code duplication
        self.emit_same_int_ty_imm(
            "arithmetic",
            value,
            amount.width,
            move |this| unsafe {
                this.emit_1ret_stmt(StmtKind::AddImm {
                    value,
                    imm64: amount.bits,
                })
            }
        )
    }

    pub fn sub(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_arith_binop(ArithBinOp::Mul, lhs, rhs)
    }

    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block.
    /// If `rhs == 0`, the result is `0`.
    pub fn udiv(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_same_int_ty_binop(
            "division",
            lhs,
            rhs,
            move |this, width| {
                let zero_imm = IConst::zero(width);

                let zero = this.iconst(zero_imm);
                let one = this.iconst(IConst::one(width));

                let rhs_is_zero = this.icmp_imm(IntCmp::Equal, rhs, zero_imm);

                // `select` is not lazy, so make the divisor safe before dividing.
                let safe_rhs = this.select(rhs_is_zero, one, rhs);

                let quotient = unsafe {
                    this.emit_1ret_stmt(StmtKind::ArithBinOp {
                        op: ArithBinOp::UncheckedUDiv,
                        lhs,
                        rhs: safe_rhs,
                    })
                };

                this.select(rhs_is_zero, zero, quotient)
            }
        )
    }

    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block
    /// and does not update condition flags.
    /// The result is the signed quotient of `lhs / rhs`, rounded toward zero.
    /// If `rhs == 0`, the result is `0`.
    /// If the signed quotient is not representable, i.e. `INT_MIN / -1`,
    /// the result is `INT_MIN`.
    pub fn sdiv(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_same_int_ty_binop(
            "division",
            lhs,
            rhs,
            move |this, width| {
                let zero_imm = IConst::zero(width);
                let int_min_imm = IConst::min_negative(width);
                let negative_one_imm = IConst::negative_one(width);

                let zero = this.iconst(zero_imm);
                let one = this.iconst(IConst::one(width));

                let rhs_is_zero = this.icmp_imm(IntCmp::Equal, rhs, zero_imm);
                let lhs_is_min = this.icmp_imm(IntCmp::Equal, lhs, int_min_imm);
                let rhs_is_minus_one = this.icmp_imm(IntCmp::Equal, rhs, negative_one_imm);

                let is_overflow = this.bitand(lhs_is_min, rhs_is_minus_one);

                // Avoid both UB cases:
                //   rhs == 0
                //   lhs == INT_MIN && rhs == -1
                let use_safe_rhs = this.bitor(rhs_is_zero, is_overflow);

                // INT_MIN / -1 should produce INT_MIN.
                // Since safe_rhs is 1 in the overflow case, quotient is already lhs,
                // but this makes the intended semantics explicit.
                let safe_rhs = this.select(use_safe_rhs, one, rhs);

                let quotient = unsafe {
                    this.emit_1ret_stmt(StmtKind::ArithBinOp {
                        op: ArithBinOp::UncheckedSDiv,
                        lhs,
                        rhs: safe_rhs,
                    })
                };

                // rhs == 0 should produce 0.
                this.select(rhs_is_zero, zero, quotient)
            }
        )
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
        self.emit_flag_setting_binop(OverflowingBinOp::Add, lhs, rhs)
    }

    pub fn subs(&mut self, lhs: LValue, rhs: LValue) -> LValue {
        self.emit_flag_setting_binop(OverflowingBinOp::Sub, lhs, rhs)
    }

    fn insert_halt_check_at(
        &mut self,
        block: Block,
        insert_at: usize
    ) -> Block {
        let (tail_stmts, old_terminated, old_terminator, old_is_cold) = {
            let block_data = &mut self.blocks[block];

            let is_after_safepoint = insert_at
                .checked_sub(1)
                .is_some_and(|instruction_end| {
                    matches!(
                        block_data.stmts[instruction_end].rvalue,
                        StmtKind::Safepoint
                    )
                });

            assert!(
                is_after_safepoint,
                "internal error: halt check must be inserted immediately after a safepoint"
            );

            let tail_stmts = block_data.stmts.split_off(insert_at);
            let old_terminator =
                std::mem::replace(&mut block_data.terminator, Terminator::Return);
            let old_is_cold = block_data.is_cold;
            let old_terminated = block_data.terminated;
            (tail_stmts, old_terminated, old_terminator, old_is_cold)
        };


        let halt_reason = self.lvalues.store(LValueData {
            ty: Type::Int(IntWidth::W32),
        });

        let continuation = self.blocks.store(BlockData {
            predecessors: vec![],
            stmts: tail_stmts,
            terminated: old_terminated,
            terminator: old_terminator,
            is_cold: old_is_cold,
        });

        let fail = self.blocks.store(BlockData {
            predecessors: vec![],
            stmts: Vec::new(),
            terminated: true,
            terminator: Terminator::ReturnCode { halt_reason },
            is_cold: true,
        });


        let block_data = &mut self.blocks[block];
        block_data.stmts.push(Stmt {
            outputs: array_helper::from_arr([halt_reason]),
            rvalue: StmtKind::LoadHaltReason,
        });

        // only time we need to "unterminate" a block
        block_data.terminated = false;
        self.terminate_block(block, Terminator::BrNZ {
            cond: halt_reason,
            non_zero: fail,
            zero: continuation,
        });

        debug_assert_eq!(self.blocks[fail].predecessors, [block]);
        debug_assert_eq!(self.blocks[continuation].predecessors, [block]);


        continuation
    }

    pub fn add_safepoint(&mut self) {
        unsafe { self.emit_void_stmt(StmtKind::Safepoint) }
    }

    /// Shorthand to both increment `pc` and insert a safepoint
    pub fn next_insn(&mut self) {
        let pc = self.load_pc();
        let new_pc = self.add_imm(pc, IConst::u64(4));
        self.store_pc(new_pc);
        self.add_safepoint()
    }
}



impl ExecIrBuilder {
    fn topo_sort(&self) -> Vec<Block> {
        #[derive(Debug, Copy, Clone)]
        enum DfsFrame {
            Enter(Block),
            Exit(Block),
        }

        let mut seen = ArenaSet::with_capacity(self.blocks.len());

        let mut postorder = Vec::with_capacity(self.blocks.len());

        let mut dfs_stack = vec![DfsFrame::Enter(Block::ENTRYPOINT)];

        while let Some(frame) = dfs_stack.pop() {
            match frame {
                DfsFrame::Enter(block) => {
                    if !seen.insert(block) {
                        continue;
                    }

                    dfs_stack.push(DfsFrame::Exit(block));

                    let terminator = &self.blocks[block].terminator;

                    for target in terminator.targets().rev() {
                        if !seen.contains(target) {
                            dfs_stack.push(DfsFrame::Enter(target));
                        }
                    }
                }

                DfsFrame::Exit(block) => {
                    assert!(postorder.len() < self.blocks.len());
                    postorder.push(block);
                }
            }
        }

        assert!(postorder.len() <= self.blocks.len());

        postorder.reverse();
        postorder
    }

    #[must_use]
    pub fn build(mut self) -> ExecIr {
        halt_check_pass::insert_halt_checks(&mut self);
        let reverse_post_order  = self.topo_sort();
        ExecIr {
            lvalues: self.lvalues,
            blocks: self.blocks,
            halt_check_every: self.halt_check_every,
            block_compile_order: reverse_post_order
        }
    }
}


#[cfg(test)]
mod exec_ir_tests {
    use std::sync::LazyLock;
    use crate::cpu_fabric::CpuFabric;
    use crate::halt_reason::{AtomicHaltReason, HaltReason, HaltReasonInner};
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


    fn call_compiled_full(
        compiled: &CompiledExecBlock,
        processor_state: &mut ProcessorState,
        setup: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
    ) -> u32 {
        let halt_reason = AtomicHaltReason::new();
        let io_mmu = empty_io_mmu();
        setup(processor_state, &io_mmu, &halt_reason);
        compiled.call(processor_state, &halt_reason, &io_mmu)
    }

    fn call_compiled(
        compiled: &CompiledExecBlock,
        processor_state: &mut ProcessorState,
    ) -> u32 {
        call_compiled_full(compiled, processor_state, |_, _, _| {})
    }


    fn run_full(
        builder: ExecIrBuilder,
        processor_state: &mut ProcessorState,
        setup: impl FnOnce(&mut ProcessorState, &IoMMU, &AtomicHaltReason),
    ) -> u32 {
        let compiled = compile(builder);
        call_compiled_full(&compiled, processor_state, setup)
    }

    fn run(builder: ExecIrBuilder, processor_state: &mut ProcessorState) -> u32 {
        let compiled = compile(builder);
        call_compiled(&compiled, processor_state)
    }

    fn run_success(builder: ExecIrBuilder, processor_state: &mut ProcessorState) {
        assert_eq!(run(builder, processor_state), 0);
    }

    fn u64_const(builder: &mut ExecIrBuilder, value: u64) -> LValue {
        builder.iconst(IConst::u64(value))
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
        let expected8 = builder.iconst(IConst::u8(0x88));
        let diff8 = builder.sub(got8, expected8);
        let check16 = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond: diff8,
            non_zero: bad,
            zero: check16,
        });

        builder.switch_to(check16);
        let got16 = builder.load_x_reg::<0>(IntWidth::W16);
        let expected16 = builder.iconst(IConst::u16(0x7788));
        let diff16 = builder.sub(got16, expected16);
        let check32 = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond: diff16,
            non_zero: bad,
            zero: check32,
        });

        builder.switch_to(check32);
        let got32 = builder.load_x_reg::<0>(IntWidth::W32);
        let expected32 = builder.iconst(IConst::u32(0x5566_7788));
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

        builder.next_insn();
        builder.next_insn();
        builder.next_insn();

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
        builder.next_insn();

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


        builder.add_safepoint();

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

        let wide = builder.iconst(IConst::u64(1));
        let narrow = builder.iconst(IConst::u32(1));

        let _ = builder.add(wide, narrow);
    }

    #[test]
    #[should_panic(expected = "can only store 64 bit integers to processor registers")]
    fn builder_rejects_storing_narrow_value_to_processor_register() {
        let mut builder = ExecIrBuilder::new();

        let narrow = builder.iconst(IConst::u32(1));
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


    fn store_bool_as_x_reg<const REG_IDX: u8>(
        builder: &mut ExecIrBuilder,
        cond: LValue,
    ) {
        let one = builder.iconst(IConst::u64(1));
        let zero = builder.iconst(IConst::u64(0));
        let value = builder.select(cond, one, zero);
        builder.store_x_reg::<REG_IDX>(value);
    }

    fn store_int_equals_as_x_reg<const REG_IDX: u8>(
        builder: &mut ExecIrBuilder,
        value: LValue,
        expected: IConst,
    ) {
        let ok = builder.icmp_imm(IntCmp::Equal, value, expected);
        store_bool_as_x_reg::<REG_IDX>(builder, ok);
    }

    fn clear_pstate(builder: &mut ExecIrBuilder) {
        let zero = builder.iconst(IConst::u32(0));
        builder.store_pstate(zero);
    }

    fn store_pstate_equals_as_x_reg<const REG_IDX: u8>(
        builder: &mut ExecIrBuilder,
        expected: u32,
    ) {
        let pstate = builder.load_pstate();
        let ok = builder.icmp_imm(IntCmp::Equal, pstate, IConst::u32(expected));
        store_bool_as_x_reg::<REG_IDX>(builder, ok);
    }

    #[test]
    fn int_width_metadata_is_exact() {
        assert_eq!(IntWidth::from_bits(8), Some(IntWidth::W8));
        assert_eq!(IntWidth::from_bits(16), Some(IntWidth::W16));
        assert_eq!(IntWidth::from_bits(32), Some(IntWidth::W32));
        assert_eq!(IntWidth::from_bits(64), Some(IntWidth::W64));

        assert_eq!(IntWidth::from_bits(0), None);
        assert_eq!(IntWidth::from_bits(1), None);
        assert_eq!(IntWidth::from_bits(7), None);
        assert_eq!(IntWidth::from_bits(128), None);

        assert_eq!(IntWidth::W8.bits(), 8);
        assert_eq!(IntWidth::W16.bits(), 16);
        assert_eq!(IntWidth::W32.bits(), 32);
        assert_eq!(IntWidth::W64.bits(), 64);

        assert_eq!(IntWidth::W8.bytes(), 1);
        assert_eq!(IntWidth::W16.bytes(), 2);
        assert_eq!(IntWidth::W32.bytes(), 4);
        assert_eq!(IntWidth::W64.bytes(), 8);
    }

    #[test]
    fn integer_comparisons_cover_signed_unsigned_and_immediates() {
        let mut builder = ExecIrBuilder::new();

        let minus_one = builder.iconst(IConst::i64(-1));
        let plus_one = builder.iconst(IConst::u64(1));

        let cond = builder.icmp(IntCmp::Equal, minus_one, plus_one);
        store_bool_as_x_reg::<0>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::NotEqual, minus_one, plus_one);
        store_bool_as_x_reg::<1>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::SignedLessThan, minus_one, plus_one);
        store_bool_as_x_reg::<2>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::SignedGreaterThanOrEqual, minus_one, plus_one);
        store_bool_as_x_reg::<3>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::SignedGreaterThan, minus_one, plus_one);
        store_bool_as_x_reg::<4>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::SignedLessThanOrEqual, minus_one, plus_one);
        store_bool_as_x_reg::<5>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::UnsignedLessThan, minus_one, plus_one);
        store_bool_as_x_reg::<6>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::UnsignedGreaterThanOrEqual, minus_one, plus_one);
        store_bool_as_x_reg::<7>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::UnsignedGreaterThan, minus_one, plus_one);
        store_bool_as_x_reg::<8>(&mut builder, cond);

        let cond = builder.icmp(IntCmp::UnsignedLessThanOrEqual, minus_one, plus_one);
        store_bool_as_x_reg::<9>(&mut builder, cond);

        let cond = builder.icmp_imm(IntCmp::SignedLessThan, minus_one, IConst::u64(1));
        store_bool_as_x_reg::<10>(&mut builder, cond);

        let cond = builder.icmp_imm(IntCmp::UnsignedGreaterThan, minus_one, IConst::u64(1));
        store_bool_as_x_reg::<11>(&mut builder, cond);

        let cond = builder.icmp_imm(IntCmp::Equal, minus_one, IConst::i64(-1));
        store_bool_as_x_reg::<12>(&mut builder, cond);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0);
        assert_eq!(state.x_registers[1], 1);
        assert_eq!(state.x_registers[2], 1);
        assert_eq!(state.x_registers[3], 0);
        assert_eq!(state.x_registers[4], 0);
        assert_eq!(state.x_registers[5], 1);
        assert_eq!(state.x_registers[6], 0);
        assert_eq!(state.x_registers[7], 1);
        assert_eq!(state.x_registers[8], 1);
        assert_eq!(state.x_registers[9], 0);
        assert_eq!(state.x_registers[10], 1);
        assert_eq!(state.x_registers[11], 1);
        assert_eq!(state.x_registers[12], 1);
    }

    #[test]
    fn select_handles_both_paths_and_bool_values() {
        let mut builder = ExecIrBuilder::new();

        let x0 = builder.load_x_reg::<0>(IntWidth::W64);
        let cond = builder.icmp_imm(IntCmp::NotEqual, x0, IConst::u64(0));

        let true_value = builder.iconst(IConst::u64(0xaaaa));
        let false_value = builder.iconst(IConst::u64(0xbbbb));
        let selected = builder.select(cond, true_value, false_value);
        builder.store_x_reg::<1>(selected);

        let one = builder.iconst(IConst::u64(1));
        let zero = builder.iconst(IConst::u64(0));
        let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
        let false_bool = builder.icmp(IntCmp::Equal, one, zero);
        let selected_bool = builder.select(cond, true_bool, false_bool);
        store_bool_as_x_reg::<2>(&mut builder, selected_bool);

        let compiled = compile(builder);

        let mut zero_state = ProcessorState::initial();
        zero_state.x_registers[0] = 0;
        assert_eq!(call_compiled(&compiled, &mut zero_state), 0);
        assert_eq!(zero_state.x_registers[1], 0xbbbb);
        assert_eq!(zero_state.x_registers[2], 0);

        let mut non_zero_state = ProcessorState::initial();
        non_zero_state.x_registers[0] = 1;
        assert_eq!(call_compiled(&compiled, &mut non_zero_state), 0);
        assert_eq!(non_zero_state.x_registers[1], 0xaaaa);
        assert_eq!(non_zero_state.x_registers[2], 1);
    }

    #[test]
    fn bitwise_integer_and_bool_ops_cover_reg_and_immediate_forms() {
        let mut builder = ExecIrBuilder::new();

        let a = builder.iconst(IConst::u64(0xca));
        let b = builder.iconst(IConst::u64(0xac));

        let value = builder.bitand(a, b);
        builder.store_x_reg::<0>(value);

        let value = builder.bitor(a, b);
        builder.store_x_reg::<1>(value);

        let value = builder.bitxor(a, b);
        builder.store_x_reg::<2>(value);

        let value = builder.bitand_imm(a, IConst::u64(0xf0));
        builder.store_x_reg::<3>(value);

        let value = builder.bitor_imm(b, IConst::u64(0x03));
        builder.store_x_reg::<4>(value);

        let value = builder.bitxor_imm(a, IConst::u64(0xff));
        builder.store_x_reg::<5>(value);

        let one = builder.iconst(IConst::u64(1));
        let zero = builder.iconst(IConst::u64(0));
        let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
        let false_bool = builder.icmp(IntCmp::Equal, one, zero);

        let value = builder.bitand(true_bool, false_bool);
        store_bool_as_x_reg::<6>(&mut builder, value);

        let value = builder.bitor(true_bool, false_bool);
        store_bool_as_x_reg::<7>(&mut builder, value);

        let value = builder.bitxor(true_bool, false_bool);
        store_bool_as_x_reg::<8>(&mut builder, value);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0x88);
        assert_eq!(state.x_registers[1], 0xee);
        assert_eq!(state.x_registers[2], 0x66);
        assert_eq!(state.x_registers[3], 0xc0);
        assert_eq!(state.x_registers[4], 0xaf);
        assert_eq!(state.x_registers[5], 0x35);
        assert_eq!(state.x_registers[6], 0);
        assert_eq!(state.x_registers[7], 1);
        assert_eq!(state.x_registers[8], 1);
    }

    #[test]
    fn narrow_integer_ops_wrap_compare_and_divide_correctly() {
        let mut builder = ExecIrBuilder::new();

        let lhs = builder.iconst(IConst::u8(250));
        let rhs = builder.iconst(IConst::u8(10));
        let value = builder.add(lhs, rhs);
        store_int_equals_as_x_reg::<0>(&mut builder, value, IConst::u8(4));

        let lhs = builder.iconst(IConst::u16(0));
        let rhs = builder.iconst(IConst::u16(1));
        let value = builder.sub(lhs, rhs);
        store_int_equals_as_x_reg::<1>(&mut builder, value, IConst::u16(u16::MAX));

        let lhs = builder.iconst(IConst::u32(0x8000_0000));
        let rhs = builder.iconst(IConst::u32(2));
        let value = builder.mul(lhs, rhs);
        store_int_equals_as_x_reg::<2>(&mut builder, value, IConst::u32(0));

        let lhs = builder.iconst(IConst::u8(250));
        let rhs = builder.iconst(IConst::u8(10));
        let value = builder.udiv(lhs, rhs);
        store_int_equals_as_x_reg::<3>(&mut builder, value, IConst::u8(25));

        let lhs = builder.iconst(IConst::u8(250));
        let rhs = builder.iconst(IConst::u8(0));
        let value = builder.udiv(lhs, rhs);
        store_int_equals_as_x_reg::<4>(&mut builder, value, IConst::u8(0));

        let lhs = builder.iconst(IConst::i16(-9));
        let rhs = builder.iconst(IConst::i16(2));
        let value = builder.sdiv(lhs, rhs);
        store_int_equals_as_x_reg::<5>(&mut builder, value, IConst::i16(-4));

        let lhs = builder.iconst(IConst::i16(i16::MIN));
        let rhs = builder.iconst(IConst::i16(-1));
        let value = builder.sdiv(lhs, rhs);
        store_int_equals_as_x_reg::<6>(&mut builder, value, IConst::i16(i16::MIN));

        let value = builder.iconst(IConst::u8(1));
        let value = builder.neg(value);
        store_int_equals_as_x_reg::<7>(&mut builder, value, IConst::u8(255));

        let value = builder.iconst(IConst::u16(0x00ff));
        let value = builder.bitxor_imm(value, IConst::u16(0x0ff0));
        store_int_equals_as_x_reg::<8>(&mut builder, value, IConst::u16(0x0f0f));

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        for idx in 0..=8 {
            assert_eq!(state.x_registers[idx], 1, "x{idx}");
        }
    }

    #[test]
    fn signed_and_unsigned_division_cover_more_rounding_edges() {
        let mut builder = ExecIrBuilder::new();

        let lhs = builder.iconst(IConst::i64(7));
        let rhs = builder.iconst(IConst::i64(-2));
        let value = builder.sdiv(lhs, rhs);
        builder.store_x_reg::<0>(value);

        let lhs = builder.iconst(IConst::i64(-7));
        let rhs = builder.iconst(IConst::i64(-2));
        let value = builder.sdiv(lhs, rhs);
        builder.store_x_reg::<1>(value);

        let lhs = builder.iconst(IConst::u64(u64::MAX));
        let rhs = builder.iconst(IConst::u64(u64::MAX));
        let value = builder.udiv(lhs, rhs);
        builder.store_x_reg::<2>(value);

        let lhs = builder.iconst(IConst::u64(7));
        let rhs = builder.iconst(IConst::u64(8));
        let value = builder.udiv(lhs, rhs);
        builder.store_x_reg::<3>(value);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], (-3_i64).cast_unsigned());
        assert_eq!(state.x_registers[1], 3);
        assert_eq!(state.x_registers[2], 1);
        assert_eq!(state.x_registers[3], 0);
    }

    #[test]
    fn dynamic_narrow_register_loads_read_low_order_bits() {
        let mut builder = ExecIrBuilder::new();

        let got = builder.load_x_reg_dyn(0, IntWidth::W8);
        store_int_equals_as_x_reg::<1>(&mut builder, got, IConst::u8(0xef));

        let got = builder.load_x_reg_dyn(0, IntWidth::W16);
        store_int_equals_as_x_reg::<2>(&mut builder, got, IConst::u16(0xcdef));

        let got = builder.load_x_reg_dyn(0, IntWidth::W32);
        store_int_equals_as_x_reg::<3>(&mut builder, got, IConst::u32(0x89ab_cdef));

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 0x0123_4567_89ab_cdef;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[1], 1);
        assert_eq!(state.x_registers[2], 1);
        assert_eq!(state.x_registers[3], 1);
    }

    #[test]
    fn set_nzcv_flags_preserves_non_flag_bits_and_replaces_old_flags() {
        let mut builder = ExecIrBuilder::new();

        let preserved = 0x00ff_00ff & !PState::NZCV_MASK.0;
        let initial = preserved | PState::NZCV_MASK.0;

        let initial = builder.iconst(IConst::u32(initial));
        builder.store_pstate(initial);

        let one = builder.iconst(IConst::u64(1));
        let zero = builder.iconst(IConst::u64(0));

        let true_bool = builder.icmp(IntCmp::NotEqual, one, zero);
        let false_bool = builder.icmp(IntCmp::Equal, one, zero);

        builder.set_nzcv_flags(
            true_bool,
            false_bool,
            true_bool,
            false_bool,
        );

        let expected = preserved | PState::N.0 | PState::C.0;
        store_pstate_equals_as_x_reg::<0>(&mut builder, expected);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 1);
    }

    #[test]
    fn adds_and_subs_set_arm_nzcv_flags_for_wrap_carry_and_overflow() {
        let mut builder = ExecIrBuilder::new();

        clear_pstate(&mut builder);
        let lhs = builder.iconst(IConst::u64(u64::MAX));
        let rhs = builder.iconst(IConst::u64(1));
        let value = builder.adds(lhs, rhs);
        builder.store_x_reg::<0>(value);
        store_pstate_equals_as_x_reg::<1>(&mut builder, PState::Z.0 | PState::C.0);

        clear_pstate(&mut builder);
        let lhs = builder.iconst(IConst::i64(i64::MAX));
        let rhs = builder.iconst(IConst::i64(1));
        let value = builder.adds(lhs, rhs);
        builder.store_x_reg::<2>(value);
        store_pstate_equals_as_x_reg::<3>(&mut builder, PState::N.0 | PState::V.0);

        clear_pstate(&mut builder);
        let lhs = builder.iconst(IConst::u64(0));
        let rhs = builder.iconst(IConst::u64(1));
        let value = builder.subs(lhs, rhs);
        builder.store_x_reg::<4>(value);
        store_pstate_equals_as_x_reg::<5>(&mut builder, PState::N.0);

        clear_pstate(&mut builder);
        let lhs = builder.iconst(IConst::i64(i64::MIN));
        let rhs = builder.iconst(IConst::i64(1));
        let value = builder.subs(lhs, rhs);
        builder.store_x_reg::<6>(value);
        store_pstate_equals_as_x_reg::<7>(&mut builder, PState::C.0 | PState::V.0);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0);
        assert_eq!(state.x_registers[1], 1);

        assert_eq!(state.x_registers[2], i64::MIN.cast_unsigned());
        assert_eq!(state.x_registers[3], 1);

        assert_eq!(state.x_registers[4], u64::MAX);
        assert_eq!(state.x_registers[5], 1);

        assert_eq!(state.x_registers[6], i64::MAX.cast_unsigned());
        assert_eq!(state.x_registers[7], 1);
    }

    #[test]
    fn return_fail_returns_halt_reason_and_preserves_prior_stores() {
        let mut builder = ExecIrBuilder::new();

        store_x_const::<0>(&mut builder, 0x1234);

        let halt_reason = builder.iconst(IConst::u32(0x4d2));
        builder.terminate(Terminator::ReturnCode { halt_reason });

        let mut state = ProcessorState::initial();
        assert_eq!(run(builder, &mut state), 0x4d2);
        assert_eq!(state.x_registers[0], 0x1234);
    }

    #[test]
    fn branch_to_return_fail_only_fails_on_taken_path() {
        let mut builder = ExecIrBuilder::new();

        let fail = builder.create_block();
        let ok = builder.create_block();

        let cond = builder.load_x_reg::<0>(IntWidth::W64);
        builder.terminate(Terminator::BrNZ {
            cond,
            non_zero: fail,
            zero: ok,
        });

        builder.switch_to(fail);
        let halt_reason = builder.iconst(IConst::u32(0xbeef));
        builder.terminate(Terminator::ReturnCode { halt_reason });

        builder.switch_to(ok);
        store_x_const::<1>(&mut builder, 0x600d);

        let compiled = compile(builder);

        let mut fail_state = ProcessorState::initial();
        fail_state.x_registers[0] = 1;
        assert_eq!(call_compiled(&compiled, &mut fail_state), 0xbeef);
        assert_eq!(fail_state.x_registers[1], 0);

        let mut ok_state = ProcessorState::initial();
        ok_state.x_registers[0] = 0;
        assert_eq!(call_compiled(&compiled, &mut ok_state), 0);
        assert_eq!(ok_state.x_registers[1], 0x600d);
    }

    #[test]
    fn unreachable_blocks_do_not_execute() {
        let mut builder = ExecIrBuilder::new();

        let unreachable = builder.create_block();

        builder.switch_to(unreachable);
        store_x_const::<0>(&mut builder, 0xbad);

        builder.switch_to(Block::ENTRYPOINT);
        store_x_const::<0>(&mut builder, 0x600d);

        let mut state = ProcessorState::initial();
        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0x600d);
    }

    #[test]
    fn explicit_halt_check_every_instruction_still_retires_all_instructions() {
        let mut builder = ExecIrBuilder::with_config(IrBuilderConfig {
            halt_check_every: NonZero::new(1).unwrap()
        });

        builder.next_insn();
        builder.next_insn();
        builder.next_insn();
        builder.next_insn();

        let mut state = ProcessorState::initial();
        state.pc = 0x40;

        run_success(builder, &mut state);

        assert_eq!(state.pc, 0x50);
    }

    #[test]
    fn automatic_halt_check_split_at_default_interval_preserves_pc_progress() {
        let mut builder = ExecIrBuilder::new();

        for _ in 0..520 {
            builder.next_insn();
        }

        let mut state = ProcessorState::initial();
        state.pc = 0x1000;

        run_success(builder, &mut state);

        assert_eq!(state.pc, 0x1000 + 520_u64 * 4);
    }

    #[test]
    fn backedge_halt_guard_after_instruction_done_preserves_loop_retirement() {
        let mut builder = ExecIrBuilder::new();

        let loop_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.terminate(Terminator::Br(loop_block));

        builder.switch_to(loop_block);
        let current = builder.load_x_reg::<0>(IntWidth::W64);
        let one = builder.iconst(IConst::u64(1));
        let next = builder.sub(current, one);
        builder.store_x_reg::<0>(next);
        builder.next_insn();
        builder.terminate(Terminator::BrNZ {
            cond: next,
            non_zero: loop_block,
            zero: exit_block,
        });

        builder.switch_to(exit_block);
        store_x_const::<1>(&mut builder, 0x5151);

        let mut state = ProcessorState::initial();
        state.x_registers[0] = 5;
        state.pc = 0x1000;

        run_success(builder, &mut state);

        assert_eq!(state.x_registers[0], 0);
        assert_eq!(state.x_registers[1], 0x5151);
        assert_eq!(state.pc, 0x1000 + 5 * 4);
    }

    #[test]
    #[should_panic(expected = "condition must have bool type")]
    fn builder_rejects_select_with_integer_condition() {
        let mut builder = ExecIrBuilder::new();

        let cond = builder.iconst(IConst::u64(1));
        let if_true = builder.iconst(IConst::u64(2));
        let if_false = builder.iconst(IConst::u64(3));

        let _ = builder.select(cond, if_true, if_false);
    }

    #[test]
    #[should_panic(expected = "select type mismatch")]
    fn builder_rejects_select_value_type_mismatch() {
        let mut builder = ExecIrBuilder::new();

        let one = builder.iconst(IConst::u64(1));
        let cond = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));

        let if_true = builder.iconst(IConst::u64(2));
        let if_false = builder.iconst(IConst::u32(3));

        let _ = builder.select(cond, if_true, if_false);
    }

    #[test]
    #[should_panic(expected = "arithmetic size mismatch")]
    fn builder_rejects_comparison_width_mismatch() {
        let mut builder = ExecIrBuilder::new();

        let wide = builder.iconst(IConst::u64(1));
        let _ = builder.icmp_imm(IntCmp::Equal, wide, IConst::u32(1));
    }

    #[test]
    #[should_panic(expected = "mismatched integer widths used for bitwise op")]
    fn builder_rejects_bitwise_width_mismatch() {
        let mut builder = ExecIrBuilder::new();

        let wide = builder.iconst(IConst::u64(1));
        let narrow = builder.iconst(IConst::u32(1));

        let _ = builder.bitor(wide, narrow);
    }

    #[test]
    #[should_panic(expected = "mismatched integer widths used for bitwise op")]
    fn builder_rejects_bitwise_imm_width_mismatch() {
        let mut builder = ExecIrBuilder::new();

        let wide = builder.iconst(IConst::u64(1));

        let _ = builder.bitand_imm(wide, IConst::u32(1));
    }

    #[test]
    #[should_panic(expected = "can't do pointer bitwise operations currently")]
    fn builder_rejects_pointer_bitwise_operations() {
        let mut builder = ExecIrBuilder::new();

        let _ = builder.bitor(LValue::ARG_PROCESSOR_STATE, LValue::ARG_PAGES);
    }

    #[test]
    fn terminator_accept_bool_branch_condition() {
        let mut builder = ExecIrBuilder::new();

        let one = builder.iconst(IConst::u64(1));
        let cond = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));
        let target = builder.create_block();

        builder.terminate(Terminator::BrNZ {
            cond,
            non_zero: target,
            zero: target,
        });
    }

    #[test]
    #[should_panic]
    fn terminator_rejects_bool_return_fail_reason() {
        let mut builder = ExecIrBuilder::new();

        let one = builder.iconst(IConst::u64(1));
        let halt_reason = builder.icmp_imm(IntCmp::Equal, one, IConst::u64(1));

        builder.terminate(Terminator::ReturnCode { halt_reason });
    }

    #[test]
    #[should_panic(expected = "can only store 32 bit integers to pstate")]
    fn builder_rejects_storing_non_w32_to_pstate() {
        let mut builder = ExecIrBuilder::new();

        let wide = builder.iconst(IConst::u64(0));
        builder.store_pstate(wide);
    }


    #[test]
    fn halts_inifnite_loop() {
        let mut builder = ExecIrBuilder::new();

        let new_block = builder.create_block();
        // can't loop back to entry point; invalid IR
        builder.terminate(Terminator::Br(new_block));
        builder.switch_to(new_block);
        builder.add_safepoint();
        builder.terminate(Terminator::Br(new_block));

        let expected_code = HaltReason {
            opcode: NonZero::new(121).unwrap(),
            payload: 0xbeef,
        };

        let code = run_full(
            builder,
            &mut ProcessorState::initial(),
            |_, _io_mmu, halt_reason| {
                halt_reason.halt(expected_code);
            }
        );

        assert_eq!(
            Some(expected_code),
            HaltReason::from_inner(HaltReasonInner::from_bits_retain(code))
        )
    }


    #[test]
    #[should_panic]
    fn inifnite_loop_with_no_safepoint() {
        let mut builder = ExecIrBuilder::new();

        let new_block = builder.create_block();
        // can't loop back to entry point; invalid IR
        builder.terminate(Terminator::Br(new_block));
        builder.switch_to(new_block);
        builder.terminate(Terminator::Br(new_block));
        
        let _ = builder.build();
    }
}