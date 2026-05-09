use crate::arena::{Arena, ArenaSet, impl_storable};
use crate::ffi_support::IoMmuStatus;
use arrayvec::ArrayVec;
use emu_abi::array_helper;
use emu_abi::halt_reason::{HaltReason, HaltReasonInner};
use emu_abi::memory::{MemProt, PAGE_OFFSET_MASK, PAGE_SHIFT, PAGE_SIZE_U64, Page};
use emu_abi::processor_state::{PState, ProcessorState, X_REGISTER_COUNT};
use io_mmu::IoMMU;
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use std::mem::offset_of;
use std::num::NonZero;

mod arena;
pub mod compiler;
mod ffi_support;
mod halt_check_pass;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntWidth {
    W8 = 1,
    W16 = 2,
    W32 = 4,
    W64 = 8,
}

impl IntWidth {
    pub const MAX: Self = Self::W64;

    pub const fn from_bits(bits: u32) -> Option<Self> {
        Some(match bits {
            8 => Self::W8,
            16 => Self::W16,
            32 => Self::W32,
            64 => Self::W64,
            _ => return None,
        })
    }

    pub const fn bits(self) -> u32 {
        (self as u32).strict_mul(8)
    }

    pub const fn bytes_u64(self) -> u64 {
        self as u64
    }

    pub const fn bytes(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Type {
    Int(IntWidth),
    Bool,
    HostPtr,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LoadType {
    Int(IntWidth),
    HostPtr,
}

impl Type {
    pub const I64: Self = Self::Int(IntWidth::W64);
    pub const I32: Self = Self::Int(IntWidth::W32);
    pub const I16: Self = Self::Int(IntWidth::W16);
    pub const I8: Self = Self::Int(IntWidth::W8);

    pub fn assert_int(self, op_name: &str) -> IntWidth {
        let Type::Int(width) = self else {
            panic!("can only do integer {op_name} on integers");
        };
        width
    }
}

#[derive(Debug)]
struct SSAValueData {
    pub ty: Type,
}

impl_storable! {
    SSAValueData as impl pub SSAValue;
    init: {
        const ARG_PROCESSOR_STATE = SSAValueData { ty: Type::HostPtr };
        const ARG_PAGES = SSAValueData { ty: Type::HostPtr };
        const ARG_PAGE_COUNT = SSAValueData { ty: Type::I64 };
        const ARG_HALT_REASON_PTR = SSAValueData { ty: Type::HostPtr };
        const ARG_IO_MMU = SSAValueData { ty: Type::HostPtr };
    }
}

#[derive(Debug)]
struct StackSlotData {
    size: u32,
    align: u8,
}

impl_storable! {
    StackSlotData as impl pub StackSlot;
    init: {}
}

#[derive(Copy, Clone)]
pub enum Arg {
    ProcessorState,
    Pages,
    PageCount,
    HaltReasonPtr,
    IoMMU,
}

impl Arg {
    pub fn args() -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator {
        macro_rules! make_arr {
            ($($name: ident),+ $(,)?) => {{
                fn _assert_handles_all_cases(this: Arg) {
                    match this { $(Arg::$name => ()),+ }
                }

                const _: () = {
                    let mut expected = 0;
                    $(
                    assert!(Arg::$name as u32 == expected);
                    expected += 1;
                    )+

                    let _ = expected;
                };

                [$(Arg::$name),+]
            }};
        }

        let this = make_arr![ProcessorState, Pages, PageCount, HaltReasonPtr, IoMMU];

        this.into_iter()
    }

    pub fn ty(self) -> Type {
        match self {
            Arg::ProcessorState => Type::HostPtr,
            Arg::Pages => Type::HostPtr,
            Arg::PageCount => Type::I64,
            Arg::HaltReasonPtr => Type::HostPtr,
            Arg::IoMMU => Type::HostPtr,
        }
    }

    pub fn as_ssa_value(self) -> SSAValue {
        match self {
            Arg::ProcessorState => SSAValue::ARG_PROCESSOR_STATE,
            Arg::Pages => SSAValue::ARG_PAGES,
            Arg::PageCount => SSAValue::ARG_PAGE_COUNT,
            Arg::HaltReasonPtr => SSAValue::ARG_HALT_REASON_PTR,
            Arg::IoMMU => SSAValue::ARG_IO_MMU,
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
        Self { width, bits: 0 }
    }

    pub const fn one(width: IntWidth) -> Self {
        Self { width, bits: 1 }
    }

    pub const fn min_negative(width: IntWidth) -> Self {
        match width {
            IntWidth::W8 => const { Self::i8(i8::MIN) },
            IntWidth::W16 => const { Self::i16(i16::MIN) },
            IntWidth::W32 => const { Self::i32(i32::MIN) },
            IntWidth::W64 => const { Self::i64(i64::MIN) },
        }
    }

    pub const fn negative_one(width: IntWidth) -> Self {
        let bits = width.bits();
        assert!(bits <= 64);
        Self {
            width,
            // its 2^n - 1 which encodes -1 in the given bit range
            // except when n == 64 then its  0 - 1 which is still -1 ofr 64 bit integers
            bits: 1_u64.unbounded_shl(bits).wrapping_sub(1),
        }
    }
}

macro_rules! zero_extend_u64 {
    (u64, $value: expr) => {
        $value
    };
    (i64, $value: expr) => {
        ($value).cast_unsigned()
    };

    (u32, $value: expr) => {
        $value as u64
    };
    (u16, $value: expr) => {
        $value as u64
    };
    (u8, $value: expr) => {
        $value as u64
    };

    (i32, $value: expr) => {
        ($value).cast_unsigned() as u64
    };
    (i16, $value: expr) => {
        ($value).cast_unsigned() as u64
    };
    (i8, $value: expr) => {
        ($value).cast_unsigned() as u64
    };
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ShiftOp {
    // SignExtendShr,
    ZeroExtendShr,
    // Shl,
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

type HostCallback = unsafe extern "C" fn(...);

#[derive(Debug)]
struct CallbackSignatureData {
    args: Vec<Type>,
    ret: Option<Type>,
}

impl_storable! {
    CallbackSignatureData as impl CallbackSignature;
    init: {
        const LOAD_FALLBACK = CallbackSignatureData {
            // IoMMU, vaddr, out_param
            args: vec![Type::HostPtr, Type::I64, Type::HostPtr],
            // IoMmuStatus
            ret: Some(Type::I8),
        };

        const STORE_I16_FALLBACK = CallbackSignatureData {
            // IoMMU, vaddr, value
            args: vec![Type::HostPtr, Type::I64, Type::I16],
            // IoMmuStatus
            ret: Some(Type::I8),
        };

        const STORE_I32_FALLBACK = CallbackSignatureData {
            // IoMMU, vaddr, value
            args: vec![Type::HostPtr, Type::I64, Type::I32],
            // IoMmuStatus
            ret: Some(Type::I8),
        };

        const STORE_I64_FALLBACK = CallbackSignatureData {
            // IoMMU, vaddr, value
            args: vec![Type::HostPtr, Type::I64, Type::I64],
            // IoMmuStatus
            ret: Some(Type::I8),
        };
    }
}

fn io_mmu_load_callback_for_width(width: IntWidth) -> (HostCallback, CallbackSignature) {
    fn cast<T>(f: unsafe extern "C" fn(*const IoMMU, u64, *mut T) -> IoMmuStatus) -> HostCallback {
        unsafe { std::mem::transmute(f) }
    }

    let host_cb = match width {
        IntWidth::W8 => unreachable!("8-bit load fallback is currently unnecessary"),
        IntWidth::W16 => cast(ffi_support::io_mmu_load16_le),
        IntWidth::W32 => cast(ffi_support::io_mmu_load32_le),
        IntWidth::W64 => cast(ffi_support::io_mmu_load64_le),
    };

    (host_cb, CallbackSignature::LOAD_FALLBACK)
}

fn io_mmu_store_callback_for_width(width: IntWidth) -> (HostCallback, CallbackSignature) {
    fn cast<T>(
        f: unsafe extern "C" fn(*const IoMMU, u64, T) -> IoMmuStatus,
    ) -> unsafe extern "C" fn(...) {
        unsafe { std::mem::transmute(f) }
    }

    match width {
        IntWidth::W8 => unreachable!("8-bit store fallback is currently unnecessary"),
        IntWidth::W16 => (
            cast(ffi_support::io_mmu_store16_le),
            CallbackSignature::STORE_I16_FALLBACK,
        ),
        IntWidth::W32 => (
            cast(ffi_support::io_mmu_store32_le),
            CallbackSignature::STORE_I32_FALLBACK,
        ),
        IntWidth::W64 => (
            cast(ffi_support::io_mmu_store64_le),
            CallbackSignature::STORE_I64_FALLBACK,
        ),
    }
}

const HOST_CB_SMALL_ARGS: usize = 4;

const MAX_STMT_OUTPUTS: usize = 2;

#[derive(Debug)]
enum StmtKind {
    /// Integer constant.
    IConst(IConst),

    /// Integer arithmetic.
    ///
    /// `lhs` and `rhs` must have type `Int(width)`.
    /// The result also has type `Int(width)`.
    ArithBinOp {
        op: ArithBinOp,
        lhs: SSAValue,
        rhs: SSAValue,
    },

    AddImm {
        value: SSAValue,
        imm64: u64,
    },

    /// Produces:
    ///   0: arithmetic result
    ///   1: overflow flag
    OverflowingBinOp {
        op: OverflowingBinOp,
        lhs: SSAValue,
        rhs: SSAValue,
    },

    IntCmp {
        cmp: IntCmp,
        lhs: SSAValue,
        rhs: SSAValue,
    },

    IntCmpImm {
        cmp: IntCmp,
        lhs: SSAValue,
        rhs: u64,
    },

    Select {
        cond: SSAValue,
        if_true: SSAValue,
        if_false: SSAValue,
    },

    Bitwise {
        op: BitwiseOp,
        lhs: SSAValue,
        rhs: SSAValue,
    },

    BitwiseImm {
        op: BitwiseOp,
        lhs: SSAValue,
        rhs: u64,
    },

    ShiftImm {
        op: ShiftOp,
        value: SSAValue,
        shift_ammount: u8,
    },

    /// Load from a host pointer plus a constant byte offset.
    ///
    /// This is used for things like reading `ProcessorState` fields:
    ///
    /// ```text
    /// LoadHost64(processor_state, offset_of!(ProcessorState, x_registers) + 8 * n)
    /// ```
    LoadHost {
        ty: LoadType,
        base_ptr: SSAValue,
        offset: usize,
        /// if true this means that accessing the memory location at
        /// `base_ptr` is always safe regardless of any condition
        can_move: bool,
    },

    /// Store to a host pointer plus a constant byte offset.
    StoreHost {
        base_ptr: SSAValue,
        offset: usize,
        value: SSAValue,
        can_move: bool,
    },

    LoadStackPtr {
        slot: StackSlot,
    },

    PtrAdd {
        base_ptr: SSAValue,
        offset: SSAValue,
        elem_size: NonZero<usize>,
    },

    PtrOffsetImm {
        base_ptr: SSAValue,
        offset: isize,
    },

    HostCallback {
        func: HostCallback,
        signature: CallbackSignature,
        args: SmallVec<SSAValue, HOST_CB_SMALL_ARGS>,
    },

    /// little endian relaxed atomic load
    VMLoadRaw {
        aligned_page_ptr: SSAValue,
        page_offset: SSAValue,
        width: IntWidth,
    },

    /// little endian relaxed atomic store
    VMStoreRaw {
        aligned_page_ptr: SSAValue,
        page_offset: SSAValue,
        value: SSAValue,
    },

    /// loads the halt reason found at Arg::HaltReasonPtr
    /// implemntation:
    /// its a relaxed atomic 32 bit native entain load
    /// this is more like `HasPendingHaltReasonButReturnsBitsBecauseBrZNeedsAValue`
    /// if it returns yes then, and only then do you syncronize, because this makes
    /// the fast path (no halt) very fast
    LoadHaltReason,

    /// takes the halt reason found at Arg::HaltReasonPtr
    /// and replaces it with 0
    /// implementation: AcqRel xchg \[HaltReasonPtr] 0
    TakeHaltReason,

    /// **relaxed** atomic byte load
    GetInstructionDirtyFlag(SSAValue),

    /// **release** atomic byte store to the value `1`
    SetInstructionDirtyFlag(SSAValue),

    Safepoint,
}

#[derive(Debug)]
struct StmtData {
    outputs: ArrayVec<SSAValue, MAX_STMT_OUTPUTS>,
    rvalue: StmtKind,
}

impl_storable! {
    StmtData as impl Stmt;
    init: { }
}

const JUMP_PARAM_SMALL: usize = 8;

#[derive(Debug)]
pub struct Jump {
    parameters: SmallVec<SSAValue, JUMP_PARAM_SMALL>,
    target: Block,
}

impl From<Block> for Jump {
    fn from(value: Block) -> Self {
        Jump {
            parameters: smallvec![],
            target: value,
        }
    }
}

impl<I: Into<SmallVec<SSAValue, JUMP_PARAM_SMALL>>> From<(Block, I)> for Jump {
    fn from((target, parameters): (Block, I)) -> Self {
        Jump {
            target,
            parameters: parameters.into(),
        }
    }
}

#[derive(Debug)]
enum TerminatorKind {
    /// Return "0" i.e. return success.
    Return,
    /// Return a `NonZero<u32>` block-exit reason.
    ReturnCode { halt_reason: SSAValue },
    /// branch targets
    /// at index: `0` branch `zero`
    /// at index: `1` branch `non_zero`
    BrZ { cond: SSAValue },

    /// has only a single branch target
    Br,
}

#[derive(Debug)]
pub struct Terminator {
    targets: ArrayVec<Jump, { Self::MAX_TARGETS }>,
    kind: TerminatorKind,
}

impl Terminator {
    pub const MAX_TARGETS: usize = 2;

    pub fn block_targets(&self) -> arrayvec::IntoIter<Block, { Terminator::MAX_TARGETS }> {
        match self.targets.as_slice() {
            [] => array_helper::empty_iter(),
            [one] => array_helper::iter_from_arr([one.target]),
            [one, two] if one.target != two.target => {
                array_helper::iter_from_arr([one.target, two.target])
            }
            [target, _duplicate_target_different_params] => {
                array_helper::iter_from_arr([target.target])
            }

            _ => {
                const { assert!(Terminator::MAX_TARGETS == 2) }
                unreachable!()
            }
        }
    }
}

#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
impl Terminator {
    pub const Return: Self = Self {
        targets: ArrayVec::new_const(),
        kind: TerminatorKind::Return,
    };

    pub fn ReturnCode(halt_reason: SSAValue) -> Self {
        Self {
            targets: ArrayVec::new_const(),
            kind: TerminatorKind::ReturnCode { halt_reason },
        }
    }

    pub fn Br(jump: impl Into<Jump>) -> Self {
        Self {
            targets: array_helper::from_arr([jump.into()]),
            kind: TerminatorKind::Br,
        }
    }

    pub fn BrZ(cond: SSAValue, zero: impl Into<Jump>, non_zero: impl Into<Jump>) -> Self {
        Self {
            targets: array_helper::from_arr([zero.into(), non_zero.into()]),
            kind: TerminatorKind::BrZ { cond },
        }
    }

    pub fn BrNZ(cond: SSAValue, non_zero: impl Into<Jump>, zero: impl Into<Jump>) -> Self {
        // just swap the arguments
        Self::BrZ(cond, zero, non_zero)
    }
}

#[derive(Debug)]
struct BlockData {
    parameters: SmallVec<SSAValue, JUMP_PARAM_SMALL>,
    stmts: Vec<Stmt>,
    terminated: bool,
    terminator: Terminator,
    is_cold: bool,
}

impl BlockData {
    pub fn empty() -> Self {
        Self {
            parameters: smallvec![],
            stmts: vec![],
            terminated: false,
            terminator: Terminator::Return,
            is_cold: false,
        }
    }
}

impl_storable! {
    BlockData as impl pub Block;
    init: {
        pub const ENTRYPOINT = Self {
            parameters: Arg::args().map(Arg::as_ssa_value).collect(),
            ..BlockData::empty()
        };
    }
}

#[derive(Debug)]
pub struct ExecIr {
    ssa_values: Arena<SSAValueData>,
    blocks: Arena<BlockData>,
    stmts: Arena<StmtData>,
    stack_slots: Arena<StackSlotData>,
    signatures: Arena<CallbackSignatureData>,
    block_compile_order: Vec<Block>,
}

pub struct ExecIrBuilder {
    ssa_values: Arena<SSAValueData>,
    blocks: Arena<BlockData>,
    stmts: Arena<StmtData>,
    stack_slots: Arena<StackSlotData>,
    signatures: Arena<CallbackSignatureData>,
    scratch_space: Option<StackSlot>,
    leave_and_take_halt: Option<Block>,
    halt_blocks: HashMap<HaltReasonInner, Block>,
    current_block: Block,
    halt_check_every: NonZero<u32>,
}

pub struct IrBuilderConfig {
    pub halt_check_every: NonZero<u32>,
}

impl Default for IrBuilderConfig {
    fn default() -> Self {
        Self {
            halt_check_every: const { NonZero::new(128).unwrap() },
        }
    }
}

impl Default for ExecIrBuilder {
    fn default() -> Self {
        Self::with_config(IrBuilderConfig::default())
    }
}

impl ExecIrBuilder {
    pub fn with_config(config: IrBuilderConfig) -> Self {
        Self {
            ssa_values: Arena::new(),
            blocks: Arena::new(),
            stmts: Arena::new(),
            stack_slots: Arena::new(),
            signatures: Arena::new(),
            scratch_space: None,
            leave_and_take_halt: None,
            halt_blocks: HashMap::new(),
            current_block: Block::ENTRYPOINT,
            halt_check_every: config.halt_check_every,
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

    pub fn successors(
        &self,
        block: Block,
    ) -> arrayvec::IntoIter<Block, { Terminator::MAX_TARGETS }> {
        self.blocks[block].terminator.block_targets()
    }

    pub fn mark_block_bold(&mut self, block: Block) {
        // a cold entrypoint is insane and should never be true
        // it will always run when the resulting block is compiled
        // it can't be cold
        // this is made just so that if an exec block always unconditionally fails at the end
        // this doesn't accedentally mark that cold
        if block != Block::ENTRYPOINT {
            self.blocks[block].is_cold = true;
        }
    }

    pub fn mark_current_block_cold(&mut self) {
        self.mark_block_bold(self.current_block)
    }

    pub fn terminate_block(&mut self, block: Block, mut terminator: Terminator) {
        match terminator.kind {
            TerminatorKind::Return => {}

            TerminatorKind::ReturnCode { halt_reason: int } => {
                assert!(matches!(self.ssa_values[int].ty, Type::Int(_)))
            }

            TerminatorKind::BrZ { cond: int } => {
                assert!(matches!(self.ssa_values[int].ty, Type::Bool | Type::Int(_)));
                let [zero, non_zero] = terminator.targets.as_array::<2>().unwrap();
                if zero.target == non_zero.target && zero.parameters == non_zero.parameters {
                    terminator.targets.pop();
                    terminator.kind = TerminatorKind::Br
                }
            }

            TerminatorKind::Br => {}
        }

        for target in terminator.block_targets() {
            assert_ne!(target, Block::ENTRYPOINT, "can't branch to entrypoint");
        }

        let mark_cold = matches!(terminator.kind, TerminatorKind::ReturnCode { .. });

        let block_data = &mut self.blocks[block];
        assert!(!block_data.terminated);
        block_data.terminator = terminator;
        block_data.terminated = true;

        if mark_cold && !block_data.is_cold {
            // this makes sure we don't mark the entrypoint cold
            self.mark_block_bold(block)
        }
    }

    pub fn terminate(&mut self, terminator: Terminator) {
        let current_block = self.current_block;
        self.terminate_block(current_block, terminator)
    }

    pub fn add_block_parameter_at(&mut self, block: Block, ty: Type) -> SSAValue {
        let ssa_value = self.ssa_values.store(SSAValueData { ty });
        self.blocks[block].parameters.push(ssa_value);
        ssa_value
    }

    pub fn add_block_parameter(&mut self, ty: Type) -> SSAValue {
        let block = self.current_block;
        self.add_block_parameter_at(block, ty)
    }

    fn type_of(&self, rvalue: &StmtKind) -> arrayvec::IntoIter<Type, MAX_STMT_OUTPUTS> {
        use array_helper::{empty_iter, iter_from_arr};

        match *rvalue {
            StmtKind::IConst(iconst) => iter_from_arr([Type::Int(iconst.width())]),

            StmtKind::ArithBinOp { lhs, .. }
            | StmtKind::AddImm { value: lhs, .. }
            | StmtKind::Bitwise { lhs, .. }
            | StmtKind::BitwiseImm { lhs, .. }
            | StmtKind::ShiftImm { value: lhs, .. }
            | StmtKind::Select {
                cond: _,
                if_true: lhs,
                if_false: _,
            } => iter_from_arr([self.ssa_values[lhs].ty]),

            StmtKind::OverflowingBinOp { op: _, lhs, rhs: _ } => {
                iter_from_arr([self.ssa_values[lhs].ty, Type::Bool])
            }

            StmtKind::IntCmp { .. } | StmtKind::IntCmpImm { .. } => iter_from_arr([Type::Bool]),

            StmtKind::LoadHost { ty, .. } => match ty {
                LoadType::Int(width) => iter_from_arr([Type::Int(width)]),
                LoadType::HostPtr => iter_from_arr([Type::HostPtr]),
            },

            StmtKind::StoreHost { .. } => empty_iter(),

            StmtKind::PtrOffsetImm { .. }
            | StmtKind::LoadStackPtr { .. }
            | StmtKind::PtrAdd { .. } => iter_from_arr([Type::HostPtr]),

            StmtKind::HostCallback { signature, .. } => match self.signatures[signature].ret {
                None => empty_iter(),
                Some(ret) => iter_from_arr([ret]),
            },

            StmtKind::VMLoadRaw { width, .. } => iter_from_arr([Type::Int(width)]),
            StmtKind::VMStoreRaw { .. } => empty_iter(),

            StmtKind::LoadHaltReason | StmtKind::TakeHaltReason => iter_from_arr([Type::I32]),

            StmtKind::GetInstructionDirtyFlag { .. } => iter_from_arr([Type::I8]),
            StmtKind::SetInstructionDirtyFlag { .. } => empty_iter(),

            StmtKind::Safepoint => empty_iter(),
        }
    }

    pub fn block_scope<T>(&mut self, block: Block, emmitter: impl FnOnce(&mut Self) -> T) -> T {
        struct SetOnDrop<'a> {
            builder: &'a mut ExecIrBuilder,
            original_block: Block,
        }

        impl Drop for SetOnDrop<'_> {
            fn drop(&mut self) {
                self.builder.current_block = self.original_block;
            }
        }

        let original_block = self.current_block;

        let set_on_drop = SetOnDrop {
            builder: self,
            original_block,
        };

        let builder = &mut *set_on_drop.builder;
        builder.current_block = block;

        emmitter(builder)
    }

    /// # Safety
    ///
    /// the IR must not produce UB when run after compilation
    unsafe fn emit_stmt_full<const N: usize>(&mut self, rvalue: StmtKind) -> [SSAValue; N] {
        let outputs = self
            .type_of(&rvalue)
            .map(|ty| self.ssa_values.store(SSAValueData { ty }))
            .collect::<ArrayVec<SSAValue, MAX_STMT_OUTPUTS>>();

        let emit_out: &[SSAValue] = outputs.as_slice();
        let emit_out: [SSAValue; N] = *emit_out.as_array().expect("invalid stmt output amount");

        let stmt = self.stmts.store(StmtData { outputs, rvalue });
        self.blocks[self.current_block].stmts.push(stmt);

        emit_out
    }

    #[inline]
    unsafe fn emit_void_stmt(&mut self, rvalue: StmtKind) {
        let [] = unsafe { self.emit_stmt_full(rvalue) };
    }

    #[inline]
    #[must_use]
    unsafe fn emit_1ret_stmt(&mut self, rvalue: StmtKind) -> SSAValue {
        let [value] = unsafe { self.emit_stmt_full(rvalue) };
        value
    }

    #[inline]
    #[must_use]
    unsafe fn emit_2ret_stmt(&mut self, rvalue: StmtKind) -> (SSAValue, SSAValue) {
        let [value1, value2] = unsafe { self.emit_stmt_full(rvalue) };
        (value1, value2)
    }

    pub fn create_stack_slot(&mut self, size: u32, align: u8) -> StackSlot {
        assert!(align.is_power_of_two());

        self.stack_slots.store(StackSlotData { size, align })
    }

    pub fn use_stack_slot(&mut self, slot: StackSlot) -> SSAValue {
        unsafe { self.emit_1ret_stmt(StmtKind::LoadStackPtr { slot }) }
    }

    pub fn iconst(&mut self, iconst: IConst) -> SSAValue {
        unsafe { self.emit_1ret_stmt(StmtKind::IConst(iconst)) }
    }

    unsafe fn load_from_processor_state(&mut self, offset: usize, width: IntWidth) -> SSAValue {
        unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::Int(width),
                base_ptr: SSAValue::ARG_PROCESSOR_STATE,
                offset,
                // it is always safe to access processor state
                can_move: true,
            })
        }
    }

    unsafe fn load_from_64_bit_processor_register(
        &mut self,
        offset: usize,
        width: IntWidth,
    ) -> SSAValue {
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

    pub fn load_x_reg_dyn(&mut self, x_reg: u8, width: IntWidth) -> SSAValue {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.load_from_64_bit_processor_register(Self::x_reg_offset(x_reg), width) }
    }

    pub fn load_x_reg<const REG_IDX: u8>(&mut self, width: IntWidth) -> SSAValue {
        const { assert!(REG_IDX < X_REGISTER_COUNT) }
        unsafe { self.load_from_64_bit_processor_register(Self::x_reg_offset(REG_IDX), width) }
    }

    pub fn load_sp(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ProcessorState, sp), IntWidth::W64) }
    }

    pub fn load_pc(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ProcessorState, pc), IntWidth::W64) }
    }

    pub fn load_pstate(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ProcessorState, pstate), IntWidth::W32) }
    }

    unsafe fn store_to_processor_state(&mut self, offset: usize, value: SSAValue) {
        unsafe {
            self.emit_void_stmt(StmtKind::StoreHost {
                base_ptr: SSAValue::ARG_PROCESSOR_STATE,
                offset,
                value,
                // it is always safe to access processor state
                can_move: true,
            })
        }
    }

    unsafe fn store_processor_register(&mut self, offset: usize, value: SSAValue) {
        let Type::I64 = self.ssa_values[value].ty else {
            panic!("can only store 64 bit integers to processor registers")
        };

        unsafe { self.store_to_processor_state(offset, value) }
    }

    pub fn store_x_reg_dyn(&mut self, x_reg: u8, value: SSAValue) {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.store_processor_register(Self::x_reg_offset(x_reg), value) }
    }

    pub fn store_x_reg<const REG_IDX: u8>(&mut self, value: SSAValue) {
        const { assert!(REG_IDX < X_REGISTER_COUNT) }
        unsafe { self.store_processor_register(Self::x_reg_offset(REG_IDX), value) }
    }

    pub fn store_sp(&mut self, value: SSAValue) {
        unsafe { self.store_processor_register(offset_of!(ProcessorState, sp), value) }
    }

    pub fn store_pc(&mut self, value: SSAValue) {
        unsafe { self.store_processor_register(offset_of!(ProcessorState, pc), value) }
    }

    pub fn store_pstate(&mut self, value: SSAValue) {
        let Type::I32 = self.ssa_values[value].ty else {
            panic!("can only store 32 bit integers to pstate")
        };

        unsafe { self.store_to_processor_state(offset_of!(ProcessorState, pstate), value) }
    }

    pub fn select(&mut self, cond: SSAValue, if_true: SSAValue, if_false: SSAValue) -> SSAValue {
        assert_eq!(
            self.ssa_values[cond].ty,
            Type::Bool,
            "condition must have bool type"
        );
        assert_eq!(
            self.ssa_values[if_true].ty, self.ssa_values[if_false].ty,
            "select type mismatch"
        );
        unsafe {
            self.emit_1ret_stmt(StmtKind::Select {
                cond,
                if_true,
                if_false,
            })
        }
    }

    unsafe fn call_host(
        &mut self,
        host_cb: HostCallback,
        signature: CallbackSignature,
        args: SmallVec<SSAValue, HOST_CB_SMALL_ARGS>,
    ) -> Option<SSAValue> {
        let sig = &self.signatures[signature];
        let args_ty = sig.args.as_slice();
        assert_eq!(args_ty.len(), args.len(), "mismatched host call lengths");
        for (&arg_ty, &arg) in args_ty.iter().zip(args.iter()) {
            assert_eq!(
                arg_ty, self.ssa_values[arg].ty,
                "mismatched host call arg types"
            );
        }

        let stmt = StmtKind::HostCallback {
            func: host_cb,
            signature,
            args,
        };

        match sig.ret {
            Some(_) => Some(unsafe { self.emit_1ret_stmt(stmt) }),
            None => {
                unsafe { self.emit_void_stmt(stmt) }
                None
            }
        }
    }

    fn emit_same_int_ty_imm<T>(
        &mut self,
        op_name: &'static str,
        lhs: SSAValue,
        rhs: IntWidth,
        func: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let lhs_ty = self.ssa_values[lhs].ty;
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
        lhs: SSAValue,
        rhs: SSAValue,
        func: impl FnOnce(&mut Self, IntWidth) -> T,
    ) -> T {
        let rhs = self.ssa_values[rhs].ty.assert_int(op_name);
        self.emit_same_int_ty_imm(op_name, lhs, rhs, |this| func(this, rhs))
    }

    pub fn icmp(&mut self, cmp: IntCmp, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_same_int_ty_binop("comparisons", lhs, rhs, |this, _width| unsafe {
            this.emit_1ret_stmt(StmtKind::IntCmp { cmp, lhs, rhs })
        })
    }

    pub fn icmp_imm(&mut self, cmp: IntCmp, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_same_int_ty_imm("comparisons", lhs, rhs.width, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::IntCmpImm {
                cmp,
                lhs,
                rhs: rhs.bits,
            })
        })
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
                false => panic!("mismatched integer widths used for bitwise op"),
            },

            (Type::HostPtr, Type::HostPtr) => {
                panic!("can't do pointer bitwise operations currently")
            }

            _ => panic!("mismatched types used for bitwise operation"),
        }
    }

    fn emit_binop(&mut self, op: BitwiseOp, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        let lhs_ty = self.ssa_values[lhs].ty;
        let rhs_ty = self.ssa_values[rhs].ty;
        self.binop_type_guard(lhs_ty, rhs_ty, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::Bitwise { op, lhs, rhs })
        })
    }

    fn emit_binop_imm(&mut self, op: BitwiseOp, lhs: SSAValue, rhs: IConst) -> SSAValue {
        let lhs_ty = self.ssa_values[lhs].ty;
        self.binop_type_guard(lhs_ty, Type::Int(rhs.width), |this| unsafe {
            this.emit_1ret_stmt(StmtKind::BitwiseImm {
                op,
                lhs,
                rhs: rhs.bits,
            })
        })
    }

    pub fn bitor(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_binop(BitwiseOp::Or, lhs, rhs)
    }

    pub fn bitand(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_binop(BitwiseOp::And, lhs, rhs)
    }

    pub fn bitxor(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_binop(BitwiseOp::Xor, lhs, rhs)
    }

    pub fn bitor_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_binop_imm(BitwiseOp::Or, lhs, rhs)
    }

    pub fn bitand_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_binop_imm(BitwiseOp::And, lhs, rhs)
    }

    pub fn bitxor_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_binop_imm(BitwiseOp::Xor, lhs, rhs)
    }

    fn emit_shift_op(&mut self, op: ShiftOp, value: SSAValue, bits: u8) -> SSAValue {
        let Type::Int(width) = self.ssa_values[value].ty else {
            panic!("can only shift integers")
        };

        assert!((bits as u32) < width.bits());

        unsafe {
            self.emit_1ret_stmt(StmtKind::ShiftImm {
                op,
                value,
                shift_ammount: bits,
            })
        }
    }

    pub fn ushr_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_shift_op(ShiftOp::ZeroExtendShr, value, bits)
    }

    pub fn set_nzcv_flags(&mut self, n: SSAValue, z: SSAValue, c: SSAValue, v: SSAValue) {
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

    fn emit_arith_binop(&mut self, op: ArithBinOp, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_same_int_ty_binop("arithmetic", lhs, rhs, move |this, _width| unsafe {
            this.emit_1ret_stmt(StmtKind::ArithBinOp { op, lhs, rhs })
        })
    }

    fn emit_flag_setting_binop(
        &mut self,
        op: OverflowingBinOp,
        lhs: SSAValue,
        rhs: SSAValue,
    ) -> SSAValue {
        let (value, overflow, width) = {
            self.emit_same_int_ty_binop("overflowing arithmetic", lhs, rhs, move |this, width| {
                let (value, overflow) =
                    unsafe { this.emit_2ret_stmt(StmtKind::OverflowingBinOp { op, lhs, rhs }) };

                (value, overflow, width)
            })
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

    pub fn add(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_arith_binop(ArithBinOp::Add, lhs, rhs)
    }

    pub fn add_imm(&mut self, value: SSAValue, amount: IConst) -> SSAValue {
        // FIXME clean up code duplication
        self.emit_same_int_ty_imm("arithmetic", value, amount.width, move |this| unsafe {
            this.emit_1ret_stmt(StmtKind::AddImm {
                value,
                imm64: amount.bits,
            })
        })
    }

    pub fn sub(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_arith_binop(ArithBinOp::Sub, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_arith_binop(ArithBinOp::Mul, lhs, rhs)
    }

    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block.
    /// If `rhs == 0`, the result is `0`.
    pub fn udiv(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_same_int_ty_binop("division", lhs, rhs, move |this, width| {
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
        })
    }

    /// This is a normal value-producing bin-op.
    ///
    /// It does not branch, does not panic, and does not terminate the block
    /// and does not update condition flags.
    /// The result is the signed quotient of `lhs / rhs`, rounded toward zero.
    /// If `rhs == 0`, the result is `0`.
    /// If the signed quotient is not representable, i.e. `INT_MIN / -1`,
    /// the result is `INT_MIN`.
    pub fn sdiv(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_same_int_ty_binop("division", lhs, rhs, move |this, width| {
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
        })
    }

    pub fn neg(&mut self, value: SSAValue) -> SSAValue {
        let Type::Int(width) = self.ssa_values[value].ty else {
            panic!("can only negate an integer")
        };

        let zero = self.iconst(IConst::zero(width));

        // TODO add native support for a negate stmt
        self.sub(zero, value)
    }

    pub fn adds(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_flag_setting_binop(OverflowingBinOp::Add, lhs, rhs)
    }

    pub fn subs(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_flag_setting_binop(OverflowingBinOp::Sub, lhs, rhs)
    }

    fn get_scratch_space_ptr(&mut self, width: IntWidth) -> SSAValue {
        let _ = width;
        const { assert!(IntWidth::MAX.bytes() == IntWidth::W64.bytes()) }

        if self.scratch_space.is_none() {
            self.scratch_space = Some(self.create_stack_slot(
                size_of::<u64>().try_into().unwrap(),
                align_of::<u64>().try_into().unwrap(),
            ))
        }

        self.use_stack_slot(self.scratch_space.unwrap())
    }

    pub fn assert_or_jmp_to(&mut self, cond: SSAValue, expected: bool, fail: Block) -> Block {
        self.mark_block_bold(fail);
        let success = self.create_block();
        let (zero, non_zero) = match expected {
            // value expected not zero
            true => (fail, success),
            // value expected zero
            false => (success, fail),
        };
        self.terminate(Terminator::BrZ(cond, zero, non_zero));
        self.switch_to(success);
        success
    }

    fn get_leave_and_take_halt(&mut self) -> Block {
        if self.leave_and_take_halt.is_none() {
            let fail = self.create_block();

            self.block_scope(fail, |this| {
                this.mark_current_block_cold();
                let final_halt_reason = unsafe { this.emit_1ret_stmt(StmtKind::TakeHaltReason) };

                this.terminate(Terminator::ReturnCode(final_halt_reason));
            });

            assert!(self.leave_and_take_halt.is_none());
            self.leave_and_take_halt = Some(fail)
        }

        self.leave_and_take_halt.unwrap()
    }

    fn get_halt_block(&mut self, reason: HaltReasonInner) -> Block {
        assert_ne!(reason.bits(), 0);

        if let Some(&existing_block) = self.halt_blocks.get(&reason) {
            return existing_block;
        }

        let fail_block = self.create_block();

        self.block_scope(fail_block, |this| {
            let trap_value = this.iconst(IConst::u32(reason.bits()));
            this.terminate(Terminator::ReturnCode(trap_value));
            this.mark_current_block_cold()
        });

        let old_value = self.halt_blocks.insert(reason, fail_block);
        assert!(old_value.is_none());

        fail_block
    }

    /// Inserts a halt check immediately after a safepoint in `block`.
    ///
    /// `insert_at` is the statement insertion index, so the safepoint must be at
    /// `insert_at - 1`.
    ///
    /// This splits `block` at `insert_at`:
    ///
    /// - the original block keeps `stmts[..insert_at]`;
    /// - the original block then loads the halt reason and branches;
    /// - the zero branch goes to a newly-created continuation block containing
    ///   the old `stmts[insert_at..]` and the old terminator;
    /// - the non-zero branch goes to a cold fail block that returns the halt reason.
    ///
    /// Returns the continuation block. Callers that are scanning forward through
    /// the original instruction stream should resume from the returned block.
    fn insert_halt_check_at(&mut self, block: Block, insert_at: usize) -> Block {
        let (tail_stmts, old_terminated, old_terminator, old_is_cold) = {
            let block_data = &mut self.blocks[block];

            let is_after_safepoint = insert_at.checked_sub(1).is_some_and(|instruction_end| {
                matches!(
                    self.stmts[block_data.stmts[instruction_end]].rvalue,
                    StmtKind::Safepoint
                )
            });

            assert!(
                is_after_safepoint,
                "internal error: halt check must be inserted immediately after a safepoint"
            );

            let tail_stmts = block_data.stmts.split_off(insert_at);
            let old_terminator = std::mem::replace(&mut block_data.terminator, Terminator::Return);
            let old_is_cold = block_data.is_cold;
            let old_terminated = block_data.terminated;
            (tail_stmts, old_terminated, old_terminator, old_is_cold)
        };

        let continuation = self.blocks.store(BlockData {
            parameters: smallvec![],
            stmts: tail_stmts,
            terminated: old_terminated,
            terminator: old_terminator,
            is_cold: old_is_cold,
        });

        let leave_and_take_halt = self.get_leave_and_take_halt();

        let maybe_halt_reason = self.block_scope(block, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::LoadHaltReason)
        });

        let block_data = &mut self.blocks[block];

        // only time we need to "unterminate" a block
        // if we ever add a `predecessors` field to BlockData
        // this would be a perfect place to actually remove the
        // this block from all the blocks found in `old_terminator`s
        // predecessors and assign those to `continuation`
        block_data.terminated = false;
        self.terminate_block(
            block,
            Terminator::BrZ(maybe_halt_reason, continuation, leave_and_take_halt),
        );

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

#[derive(Copy, Clone)]
enum VmAccessKind {
    Load { width: IntWidth },
    Store { value: SSAValue },
}

impl VmAccessKind {
    fn width(&self, builder: &ExecIrBuilder) -> IntWidth {
        match *self {
            VmAccessKind::Load { width } => width,
            VmAccessKind::Store { value } => builder.ssa_values[value].ty.assert_int("vm stores"),
        }
    }

    fn required_perms(&self) -> MemProt {
        match *self {
            VmAccessKind::Load { .. } => MemProt::READ,
            VmAccessKind::Store { .. } => MemProt::WRITE,
        }
    }
}

struct FallbackAccess {
    block: Block,
    ok_block: Block,
    value: Option<SSAValue>,
}

impl ExecIrBuilder {
    fn emit_io_mmu_fallback(
        &mut self,
        access: VmAccessKind,
        width: IntWidth,
        vaddr: SSAValue,
        return_trap_block: Block,
    ) -> FallbackAccess {
        let fallback_block = self.create_block();
        self.block_scope(fallback_block, |this| {
            this.mark_current_block_cold();
            let (out_param, (host_cb, sig), args) = match access {
                VmAccessKind::Load { .. } => {
                    let stack_ptr = this.get_scratch_space_ptr(width);
                    let func = io_mmu_load_callback_for_width(width);
                    let args = smallvec![SSAValue::ARG_IO_MMU, vaddr, stack_ptr];
                    (Some(stack_ptr), func, args)
                }
                VmAccessKind::Store { value } => {
                    let func = io_mmu_store_callback_for_width(width);
                    let args = smallvec![SSAValue::ARG_IO_MMU, vaddr, value];
                    (None, func, args)
                }
            };

            let status = unsafe { this.call_host(host_cb, sig, args).unwrap() };
            const { assert!(IoMmuStatus::Ok as u8 == 0) };
            const { assert!(IoMmuStatus::Fault as u8 != 0) };

            this.assert_or_jmp_to(status, false, return_trap_block);

            let value = out_param.map(|out_param| unsafe {
                this.emit_1ret_stmt(StmtKind::LoadHost {
                    ty: LoadType::Int(width),
                    base_ptr: out_param,
                    offset: 0,
                    // this operation is only safe **after** calling the host function and
                    // ensuring that the operation did not trap
                    can_move: false,
                })
            });

            FallbackAccess {
                block: fallback_block,
                ok_block: this.current_block(),
                value,
            }
        })
    }

    fn vm_access(&mut self, vaddr: SSAValue, access: VmAccessKind) -> Option<SSAValue> {
        assert!(matches!(self.ssa_values[vaddr].ty, Type::I64));

        let width = access.width(self);

        let page_index = self.ushr_imm(vaddr, PAGE_SHIFT);
        let page_is_in_bounds = self.icmp(
            IntCmp::UnsignedLessThan,
            page_index,
            SSAValue::ARG_PAGE_COUNT,
        );

        let return_trap_block = self.get_halt_block(HaltReason::MEMORY_TRAP.into_inner());
        self.assert_or_jmp_to(page_is_in_bounds, true, return_trap_block);

        let page_offset = self.bitand_imm(vaddr, IConst::u64(PAGE_OFFSET_MASK));

        let is_aligned_and_fits_page_check = match width {
            // a byte ptr only accesses the byte it is on
            // and is always aligned
            IntWidth::W8 => None,
            _ => {
                let bytes = width.bytes_u64();
                debug_assert!(bytes.is_power_of_two());

                let align_mask = bytes.strict_sub(1);
                let alignment_bits = self.bitand_imm(vaddr, IConst::u64(align_mask));
                let is_aligned = self.icmp_imm(IntCmp::Equal, alignment_bits, IConst::u64(0));

                // access_fits_in_page = page_offset + bytes <= PAGE_SIZE_U64;
                // access_fits_in_page = page_offset <= PAGE_SIZE_U64 - bytes;
                // access_fits_in_page = page_offset < PAGE_SIZE_U64 - bytes + 1;
                let access_fits_in_page = self.icmp_imm(
                    IntCmp::UnsignedLessThan,
                    page_offset,
                    IConst::u64(PAGE_SIZE_U64.strict_sub(bytes).strict_add(1)),
                );

                Some(self.bitand(access_fits_in_page, is_aligned))
            }
        };

        let mut fallback_access = None::<FallbackAccess>;
        if let Some(normal_access) = is_aligned_and_fits_page_check {
            let fallback_access = fallback_access.insert(self.emit_io_mmu_fallback(
                access,
                width,
                vaddr,
                return_trap_block,
            ));

            self.assert_or_jmp_to(normal_access, true, fallback_access.block);
        }

        let page_info_ptr = unsafe {
            self.emit_1ret_stmt(StmtKind::PtrAdd {
                base_ptr: SSAValue::ARG_PAGES,
                offset: page_index,
                elem_size: const { NonZero::new(size_of::<Page>()).unwrap() },
            })
        };

        let page_protections = unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::Int(IntWidth::W8),
                base_ptr: page_info_ptr,
                offset: offset_of!(Page, mem_prot),
                // this page might be out of bounds; not safe to access
                can_move: false,
            })
        };

        // this is loaded before the assert, so that it can be moved to inside after the assert
        // or stay here; whichever the optimizer finds best
        let aligned_page_ptr = unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::HostPtr,
                base_ptr: page_info_ptr,
                offset: offset_of!(Page, ptr),
                // this page might be out of bounds; not safe to access
                can_move: false,
            })
        };

        let required_perms = IConst::u8(access.required_perms().bits());
        let op_allowed = self.bitand_imm(page_protections, required_perms);
        self.assert_or_jmp_to(op_allowed, true, return_trap_block);

        let ret_value = {
            match access {
                VmAccessKind::Load { .. } => Some(unsafe {
                    self.emit_1ret_stmt(StmtKind::VMLoadRaw {
                        aligned_page_ptr,
                        page_offset,
                        width,
                    })
                }),
                VmAccessKind::Store { value } => {
                    unsafe {
                        self.emit_void_stmt(StmtKind::VMStoreRaw {
                            aligned_page_ptr,
                            page_offset,
                            value,
                        })
                    }
                    None
                }
            }
        };

        if let VmAccessKind::Store { .. } = access {
            // if page_is_executable # cold
            //    if page_not_alredy_dirty # cold
            //       set_page_dirty

            let executable = IConst::u8(MemProt::EXECUTE.bits());
            let page_is_executable = self.bitand_imm(page_protections, executable);
            let check_if_not_already_dirty = self.create_block();
            let continuation =
                self.assert_or_jmp_to(page_is_executable, false, check_if_not_already_dirty);

            self.block_scope(check_if_not_already_dirty, |this| {
                let insn_dirty_ptr = unsafe {
                    this.emit_1ret_stmt(StmtKind::PtrOffsetImm {
                        base_ptr: page_info_ptr,
                        offset: isize::try_from(offset_of!(Page, insn_dirty)).unwrap(),
                    })
                };

                let is_dirty = unsafe {
                    this.emit_1ret_stmt(StmtKind::GetInstructionDirtyFlag(insn_dirty_ptr))
                };

                let set_insn_dirty = this.create_block();
                // if is_dirty != 0; goto continue; else goto set_insn_dirty;
                this.terminate(Terminator::BrNZ(is_dirty, continuation, set_insn_dirty));

                this.block_scope(set_insn_dirty, |this| {
                    this.mark_current_block_cold();
                    unsafe {
                        this.emit_void_stmt(StmtKind::SetInstructionDirtyFlag(insn_dirty_ptr))
                    }

                    this.terminate(Terminator::Br(continuation))
                })
            })
        }

        let Some(fallback_access) = fallback_access else {
            return ret_value;
        };

        let returns = match (ret_value, fallback_access.value) {
            (Some(ret_normal), Some(ret_fallback)) => Some((ret_normal, ret_fallback)),
            (None, None) => None,
            _ => unreachable!(),
        };

        match returns {
            Some((ret_normal, ret_fallback)) => {
                let normal_access_block = self.current_block();
                let merge_block = self.create_block();
                self.switch_to(merge_block);

                let param = self.add_block_parameter(Type::Int(width));

                self.terminate_block(
                    normal_access_block,
                    Terminator::Br((merge_block, smallvec![ret_normal])),
                );
                self.terminate_block(
                    fallback_access.ok_block,
                    Terminator::Br((merge_block, smallvec![ret_fallback])),
                );

                Some(param)
            }
            None => {
                let current_block = self.current_block();
                self.terminate_block(fallback_access.ok_block, Terminator::Br(current_block));

                None
            }
        }
    }

    pub fn vm_load(&mut self, vaddr: SSAValue, width: IntWidth) -> SSAValue {
        match self.vm_access(vaddr, VmAccessKind::Load { width }) {
            Some(value) => value,
            None => unreachable!("load access must produce a value"),
        }
    }

    pub fn vm_store(&mut self, vaddr: SSAValue, value: SSAValue) {
        let out = self.vm_access(vaddr, VmAccessKind::Store { value });
        debug_assert!(out.is_none());
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

                    for target in terminator.block_targets().rev() {
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

        let reverse_post_order = self.topo_sort();
        ExecIr {
            ssa_values: self.ssa_values,
            blocks: self.blocks,
            stmts: self.stmts,
            stack_slots: self.stack_slots,
            signatures: self.signatures,
            block_compile_order: reverse_post_order,
        }
    }
}
