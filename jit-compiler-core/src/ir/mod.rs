//! `ExecIr`: the SSA intermediate representation consumed by the JIT
//! backends.
//!
//! An [`ExecIr`] is a control-flow graph of [`Block`]s holding typed,
//! single-assignment values ([`SSAValue`]). Build one with
//! [`ExecIrBuilder`], which offers instruction-level emitters (arithmetic,
//! bitwise ops, guest register and guest memory access, branches) plus the
//! safepoint/halt-check machinery, then finish with
//! [`ExecIrBuilder::build`] and hand the result to a
//! [`compiler::ExecIrCompiler`](crate::compiler::ExecIrCompiler).

use arrayvec::ArrayVec;
use emu_abi::array_helper;
use emu_abi::array_helper::{empty_iter, iter_from_arr};
use emu_abi::exec_state::{ExecState, PState, X_REGISTER_COUNT};
use emu_abi::halt_reason::HaltReason;
use emu_abi::memory::{
    MemFlags, MemProt, PAGE_OFFSET_MASK_U64, PAGE_SHIFT, PAGE_SIZE, TLB_MASK, Tlb, TlbEntry,
};
use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::mem::{MaybeUninit, offset_of};
use std::num::NonZero;

use crate::arena::{Arena, ArenaSet, impl_storable};
use crate::exec_context::{ExecContext, MemOp};
use crate::ir::ffi_support::{IoMmuStatus, StrexStatus};
use io_mmu::IoMMU;
use io_mmu::icache::ICache;

mod ffi_support;
mod halt_check_pass;
mod optimization_pass;

/// The width of an IR integer, in bytes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntWidth {
    /// 8-bit integer.
    W8 = 1,
    /// 16-bit integer.
    W16 = 2,
    /// 32-bit integer.
    W32 = 4,
    /// 64-bit integer.
    W64 = 8,
}

impl IntWidth {
    /// The widest supported integer width ([`W64`](Self::W64)).
    pub const MAX: Self = Self::W64;

    /// Returns the width for a bit count of 8, 16, 32, or 64, and `None`
    /// for any other count.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        Some(match bits {
            8 => Self::W8,
            16 => Self::W16,
            32 => Self::W32,
            64 => Self::W64,
            _ => return None,
        })
    }

    /// Returns the width in bits.
    pub const fn bits(self) -> u32 {
        (self as u32).strict_mul(8)
    }

    /// Returns the width in bytes as a `u64`.
    pub const fn bytes_u64(self) -> u64 {
        self as u64
    }

    /// Returns the width in bytes.
    pub const fn bytes(self) -> u8 {
        self as u8
    }
}

/// An integer constant of a specific [`IntWidth`], stored as its zero-extended bit pattern.
#[derive(Debug, Copy, Clone)]
pub struct IConst {
    width: IntWidth,
    bits: u64,
}

impl IConst {
    /// The constant `0` at the given width.
    pub const fn zero(width: IntWidth) -> Self {
        Self { width, bits: 0 }
    }

    /// The constant `1` at the given width.
    pub const fn one(width: IntWidth) -> Self {
        Self { width, bits: 1 }
    }

    /// The most negative signed value at the given width (e.g. `i32::MIN` for [`IntWidth::W32`]).
    pub const fn min_negative(width: IntWidth) -> Self {
        let bits = 1_u64.strict_shl(width.bits().strict_sub(1));
        Self { width, bits }
    }

    /// The constant `-1` (all bits set) at the given width.
    pub const fn negative_one(width: IntWidth) -> Self {
        let bit_width = width.bits();
        assert!(bit_width <= 64);
        Self {
            width,
            // its 2^n - 1 which encodes -1 in the given bit range
            // except when n == 64 then its 0 - 1 which is still -1 for 64 bit integers
            bits: 1_u64.unbounded_shl(bit_width).wrapping_sub(1),
        }
    }

    /// Returns the constant's integer width.
    pub const fn width(self) -> IntWidth {
        self.width
    }

    /// Returns the constant's bits as its zero-extended bit pattern.
    pub const fn bits(self) -> u64 {
        self.bits
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
            #[doc = concat!(
                "Creates a `", stringify!($int_ty), "`-width constant from `value`, ",
                "storing its bit pattern zero-extended to 64 bits.",
            )]
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

macro_rules! define_alias_regions {
    ($($(#[$meta: meta])* $name: ident),+ $(,)?) => {
        /// A disjoint host-memory region as seen by the IR's alias analysis.
        ///
        /// Two host pointers with different `AliasRegion`s are assumed never
        /// to alias, which lets the backends reorder or eliminate loads and
        /// stores across regions.
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub(crate) enum AliasRegion {
            $($(#[$meta])* $name),+
        }

        impl AliasRegion {
            /// The number of alias regions.
            pub(crate) const COUNT: usize = <[Self]>::len(&[$(Self::$name),+]);
        }
    };
}

define_alias_regions! {
    /// The guest CPU's architectural state (`ExecState`).
    ExecState,
    /// The per-CPU [`ExecContext`].
    ExecContext,
    /// The translation lookaside buffer.
    Tlb,
    /// The halt-reason word polled at safepoints.
    HaltReason,
    /// Opaque memory that is never written through tracked pointers.
    ReadOnly,
    /// JIT-internal scratch stack slots.
    ScratchSpace,
    /// Guest virtual memory (page contents).
    VirtualMemory,
    /// Page metadata such as permission and dirty flags.
    PageFlags,
}

/// The type of [`SSAValue`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum TypeFull {
    /// An integer of the given width.
    Int(IntWidth),
    /// A boolean (the result of comparisons and conditions).
    Bool,
    /// A host pointer into the given [`AliasRegion`].
    HostPtr(AliasRegion),
}

macro_rules! integer_type_constants {
    ($vis: vis) => {
        /// 64-bit integer type.
        $vis const I64: Self = Self::Int(IntWidth::W64);
        /// 32-bit integer type.
        $vis const I32: Self = Self::Int(IntWidth::W32);
        /// 16-bit integer type.
        $vis const I16: Self = Self::Int(IntWidth::W16);
        /// 8-bit integer type.
        $vis const I8: Self = Self::Int(IntWidth::W8);
    };
}

impl TypeFull {
    integer_type_constants!(pub(crate));

    fn assert_int(self, op_name: &str) -> IntWidth {
        let TypeFull::Int(width) = self else {
            panic!("can only do integer {op_name} on integers");
        };

        width
    }
}

/// The type of [`SSAValue`] without any pointers available.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Type {
    /// An integer of the given width.
    Int(IntWidth),
    /// A boolean (the result of comparisons and conditions).
    Bool,
}

impl Type {
    integer_type_constants!(pub);
}

impl From<Type> for TypeFull {
    fn from(value: Type) -> Self {
        match value {
            Type::Int(width) => TypeFull::Int(width),
            Type::Bool => TypeFull::Bool,
        }
    }
}

/// The type of value produced by a host-memory load
/// (a [`TypeFull`] without `Bool`, which cannot be loaded directly).
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum LoadType {
    /// Load an integer of the given width.
    Int(IntWidth),
    /// Load a host pointer belonging to the given [`AliasRegion`].
    HostPtr(AliasRegion),
}

#[derive(Debug)]
pub(crate) struct SSAValueData {
    pub ty: TypeFull,
}

impl_storable! {
    SSAValueData as impl
    /// A handle to a typed, single-assignment IR value.
    ///
    /// Values are produced by [`ExecIrBuilder`] emitters and by block
    ///  parameters and are only meaningful within the builder/IR that
    /// created them.
    pub SSAValue;

    init: {
        pub(crate) const ARG_EXEC_STATE = SSAValueData { ty: Arg::ExecState.ty() };
        pub(crate) const ARG_EXEC_CONTEXT = SSAValueData { ty: Arg::ExecContext.ty() };
        pub(crate) const ARG_TLB_PTR = SSAValueData { ty: Arg::Tlb.ty() };
        pub(crate) const ARG_IO_MMU_IDENT = SSAValueData { ty: Arg::IoMMUIdentifier.ty() };
        pub(crate) const ARG_HALT_REASON_PTR = SSAValueData { ty: Arg::HaltReasonPtr.ty() };
        pub(crate) const ARG_IO_MMU = SSAValueData { ty: Arg::IoMMU.ty() };
    }
}

#[derive(Debug)]
pub(crate) struct StackSlotData {
    pub(crate) size: u32,
    pub(crate) align: u8,
}

impl_storable! {
    StackSlotData as impl
    /// A handle to a JIT-stack allocation created with
    /// [`ExecIrBuilder::create_stack_slot`].
    ///
    /// Turn it into an addressable pointer with
    /// [`ExecIrBuilder::use_stack_slot`].
    pub(crate) StackSlot;

    init: {}
}

/// One of the fixed arguments every compiled exec chunk receives.
///
/// These map one-to-one onto the parameters of the compiled function's
/// `extern "C"` signature and are available in the IR as the entry block's
/// parameters (see [`Arg::as_ssa_value`]).
// IMPORTANT NOTE TO IMPLEMENTORS this **MUST** be in sync with the `ExecBlockFFI` type
#[derive(Copy, Clone)]
pub(crate) enum Arg {
    /// Pointer to the [`IoMMU`] servicing guest memory accesses.
    IoMMU,
    /// Pointer to the CPU's TLB.
    Tlb,
    /// Pointer to the per-CPU [`ExecContext`].
    ExecContext,
    /// Pointer to the guest CPU's architectural state.
    ExecState,
    /// The MMU identity token used to validate TLB entries.
    IoMMUIdentifier,
    /// Pointer to the halt-reason word polled at safepoints.
    HaltReasonPtr,
}

impl Arg {
    /// Returns every argument in its ABI order.
    pub(crate) fn args() -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator {
        macro_rules! make_arr {
            ($($name: ident),+ $(,)?) => {{
                #[deny(unreachable_patterns)]
                fn _assert_handles_all_cases(this: Arg) {
                    match this { $(Arg::$name => ()),+ }
                }

                const _: () = {
                    let mut expected = 0;
                    $(
                    assert!(Arg::$name as u32 == expected);
                    expected = expected.strict_add(1);
                    )+

                    let _ = expected;
                };

                [$(Arg::$name),+]
            }};
        }

        let this = make_arr![
            IoMMU,
            Tlb,
            ExecContext,
            ExecState,
            IoMMUIdentifier,
            HaltReasonPtr,
        ];

        this.into_iter()
    }

    /// Returns the IR type of this argument.
    pub(crate) fn ty(self) -> TypeFull {
        match self {
            Arg::ExecState => TypeFull::HostPtr(AliasRegion::ExecState),
            Arg::ExecContext => TypeFull::HostPtr(AliasRegion::ExecContext),
            Arg::Tlb => TypeFull::HostPtr(AliasRegion::Tlb),
            // for the `IoMMUIdentifier`
            // alias analysis doesn't care about interior mutability, or ref count bumps,
            // it only cares about the guarantee: nothing dereferences it, and it doesn't alias
            // anything else tracked by the analysis. from the IRs perspective, these are opaque.
            Arg::IoMMUIdentifier => TypeFull::HostPtr(AliasRegion::ReadOnly),
            Arg::HaltReasonPtr => TypeFull::HostPtr(AliasRegion::HaltReason),
            Arg::IoMMU => TypeFull::HostPtr(AliasRegion::ReadOnly),
        }
    }

    /// Returns the pre-allocated [`SSAValue`] representing this argument in the entry block.
    pub(crate) fn as_ssa_value(self) -> SSAValue {
        match self {
            Arg::IoMMU => SSAValue::ARG_IO_MMU,
            Arg::Tlb => SSAValue::ARG_TLB_PTR,
            Arg::ExecContext => SSAValue::ARG_EXEC_CONTEXT,
            Arg::ExecState => SSAValue::ARG_EXEC_STATE,
            Arg::IoMMUIdentifier => SSAValue::ARG_IO_MMU_IDENT,
            Arg::HaltReasonPtr => SSAValue::ARG_HALT_REASON_PTR,
        }
    }

    /// Returns the argument that `value` represents, or `None` if `value`
    /// is not one of the pre-allocated argument values.
    pub(crate) fn from_ssa_value(value: SSAValue) -> Option<Self> {
        Some(match value {
            SSAValue::ARG_EXEC_STATE => Arg::ExecState,
            SSAValue::ARG_EXEC_CONTEXT => Arg::ExecContext,
            SSAValue::ARG_TLB_PTR => Arg::Tlb,
            SSAValue::ARG_IO_MMU_IDENT => Arg::IoMMUIdentifier,
            SSAValue::ARG_HALT_REASON_PTR => Arg::HaltReasonPtr,
            SSAValue::ARG_IO_MMU => Arg::IoMMU,

            _ => return None,
        })
    }
}

/// A two-operand integer arithmetic operation.
///
/// Both operands must have the same integer type, which is also the result type.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArithBinOp {
    /// Wrapping integer add.
    Add,

    /// Wrapping integer subtract.
    Sub,

    /// Wrapping integer multiply.
    Mul,

    /// Unsigned Integer division.
    /// It is UB if `rhs == 0`
    UncheckedUDiv,

    /// Signed integer division.
    ///
    /// This is a normal value-producing bin-op.
    /// it is UB if `rhs == 0` OR `lhs == <ty>::MIN AND rhs == -1`
    UncheckedSDiv,
}

/// A two-operand integer operation that also produces a signed-overflow flag as a second output.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OverflowingBinOp {
    /// Wrapping add, plus a signed-overflow flag.
    Add,
    /// Wrapping subtract, plus a signed-overflow flag.
    Sub,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum BitwiseOp {
    And,
    Or,
    Xor,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum ShiftOp {
    SignExtendShr,
    ZeroExtendShr,
    Shl,
}

/// An integer comparison predicate, producing a [`TypeFull::Bool`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum IntCmp {
    /// `==`.
    Equal,
    /// `!=`.
    NotEqual,
    /// Signed `<`.
    SignedLessThan,
    /// Signed `<=`.
    SignedLessThanOrEqual,
    /// Signed `>`.
    SignedGreaterThan,
    /// Signed `>=`.
    SignedGreaterThanOrEqual,
    /// Unsigned `<`.
    UnsignedLessThan,
    /// Unsigned `<=`.
    UnsignedLessThanOrEqual,
    /// Unsigned `>`.
    UnsignedGreaterThan,
    /// Unsigned `>=`.
    UnsignedGreaterThanOrEqual,
}

type HostCallback = unsafe extern "C" fn(...);

#[derive(Debug)]
pub(crate) struct CallbackSignatureData {
    pub(crate) args: Cow<'static, [TypeFull]>,
    pub(crate) ret: Option<TypeFull>,
}

macro_rules! make_store_signature {
    ($ty: ident) => {
        CallbackSignatureData {
            args: Cow::Borrowed(&[
                // io_mmu
                TypeFull::HostPtr(AliasRegion::ReadOnly),
                // tlb
                TypeFull::HostPtr(AliasRegion::Tlb),
                // exec context
                TypeFull::HostPtr(AliasRegion::ExecContext),
                // vaddr
                TypeFull::I64,
                // stored_value
                TypeFull::$ty,
            ]),
            // IoMmuStatus
            ret: Some(TypeFull::I8),
        }
    };
}

impl_storable! {
    CallbackSignatureData as impl pub(crate) CallbackSignature;
    init: {
        const LOAD = CallbackSignatureData {
            args: Cow::Borrowed(&[
                // io_mmu
                TypeFull::HostPtr(AliasRegion::ReadOnly),
                // tlb
                TypeFull::HostPtr(AliasRegion::Tlb),
                // exec context
                TypeFull::HostPtr(AliasRegion::ExecContext),
                // vaddr
                TypeFull::I64,
                // out_param
                TypeFull::HostPtr(AliasRegion::ScratchSpace)
            ]),
            // IoMmuStatus
            ret: Some(TypeFull::I8),
        };

        const STORE_I8 = make_store_signature!(I8);
        const STORE_I16 = make_store_signature!(I16);
        const STORE_I32 = make_store_signature!(I32);
        const STORE_I64 = make_store_signature!(I64);

        // if this becomes hot enough, consider passing in IO_MMU and TLB as well
        // this can lead to much better regalloc
        const CLREX = CallbackSignatureData {
            args: Cow::Borrowed(&[TypeFull::HostPtr(AliasRegion::ExecContext)]),
            ret: None,
        };
    }
}

fn load_callback(width: IntWidth, exclusive: bool) -> (HostCallback, CallbackSignature) {
    fn cast<T>(
        f: unsafe extern "C" fn(
            &IoMMU<dyn ICache + '_>,
            &mut Tlb,
            &mut ExecContext,
            u64,
            &mut MaybeUninit<T>,
        ) -> IoMmuStatus,
    ) -> HostCallback {
        unsafe { std::mem::transmute(f) }
    }

    let host_cb = match (width, exclusive) {
        (IntWidth::W8, false) => cast::<u8>(ffi_support::load_byte),
        (IntWidth::W16, false) => cast::<u16>(ffi_support::load16_le),
        (IntWidth::W32, false) => cast::<u32>(ffi_support::load32_le),
        (IntWidth::W64, false) => cast::<u64>(ffi_support::load64_le),

        (IntWidth::W8, true) => cast::<u8>(ffi_support::ldrexb),
        (IntWidth::W16, true) => cast::<u16>(ffi_support::ldrexh),
        (IntWidth::W32, true) => cast::<u32>(ffi_support::ldrex),
        (IntWidth::W64, true) => cast::<u64>(ffi_support::ldrexd),
    };

    (host_cb, CallbackSignature::LOAD)
}

// note: exclusive returns `StRexStatus` and not `IoMmuStatus`
fn store_callback(width: IntWidth, exclusive: bool) -> (HostCallback, CallbackSignature) {
    fn cast<T>(
        f: unsafe extern "C" fn(
            &IoMMU<dyn ICache + '_>,
            &mut Tlb,
            &mut ExecContext,
            u64,
            T,
        ) -> IoMmuStatus,
    ) -> unsafe extern "C" fn(...) {
        unsafe { std::mem::transmute(f) }
    }

    fn cast_strex<T>(
        f: unsafe extern "C" fn(
            &IoMMU<dyn ICache + '_>,
            &mut Tlb,
            &mut ExecContext,
            u64,
            T,
        ) -> StrexStatus,
    ) -> unsafe extern "C" fn(...) {
        unsafe { std::mem::transmute(f) }
    }

    macro_rules! ret_sig {
        ($bits: tt; if $exclusive: ident { $strex_fn: ident } else { $fallback: ident }) => {{
            type StTy = pastey::paste!([<u $bits>]);

            let fb = ffi_support::$fallback;
            let strex = ffi_support::$strex_fn;
            (
                if $exclusive { cast_strex::<StTy>(strex) } else { cast::<StTy>(fb) },
                pastey::paste!(CallbackSignature::[<STORE_I $bits>]),
            )
        }};
    }

    match width {
        IntWidth::W8 => ret_sig!(8; if exclusive { strexb } else { store_byte }),
        IntWidth::W16 => ret_sig!(16; if exclusive { strexh } else { store16_le }),
        IntWidth::W32 => ret_sig!(32; if exclusive { strex } else { store32_le }),
        IntWidth::W64 => ret_sig!(64; if exclusive { strexd } else { store64_le }),
    }
}

pub(crate) const HOST_CB_SMALL_ARGS: usize = 6;
pub(crate) const MAX_STMT_OUTPUTS: usize = 2;

#[derive(Debug)]
pub(crate) enum StmtKind {
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

    IntNeg(SSAValue),

    /// Produces:
    ///   0: arithmetic result
    ///   1: signed overflow flag
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

    BitNot(SSAValue),

    /// Shift with the amount reduced modulo the value's width.
    ///
    /// The shift amount is masked to `value_width - 1` before shifting
    /// (e.g., shifting a 32-bit value by 33 shifts by 1), matching what
    /// AArch64 register shifts and host ISAs do natively.
    Shift {
        op: ShiftOp,
        value: SSAValue,
        shift: SSAValue,
    },

    /// Shift using unbounded language semantics.
    ///
    /// If the runtime shift amount is greater than or equal to the integer
    /// width:
    ///
    /// - `Shl` and `ZeroExtendShr` produce `0`.
    /// - `SignExtendShr` behaves as though shifted by `width - 1`,
    ///   broadcasting the sign bit across the result.
    ///
    /// Unlike `Shift`, the shift amount is **not** reduced modulo the integer width.
    ShiftUnbounded {
        op: ShiftOp,
        value: SSAValue,
        shift: SSAValue,
    },

    /// Shift by a compile-time-known amount.
    ///
    /// Invariant: `shift` is always less than `(value.ty as Int).width`,
    /// so backends never see an out-of-range immediate.
    ShiftImm {
        op: ShiftOp,
        value: SSAValue,
        shift: u8,
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

    PtrEq(SSAValue, SSAValue),

    HasTag {
        ptr: SSAValue,
        tag_bits: u8,
    },

    Untag {
        ptr: SSAValue,
        tag_bits: u8,
    },

    HostCallback {
        func: HostCallback,
        signature: CallbackSignature,
        args: SmallVec<SSAValue, HOST_CB_SMALL_ARGS>,
    },

    VMLoadRaw {
        aligned_page_ptr: SSAValue,
        page_offset: SSAValue,
        width: IntWidth,
        // the arm `LDAR` does not use acquire semantics it uses seq_cst
        seq_cst: bool,
    },

    VMStoreRaw {
        aligned_page_ptr: SSAValue,
        page_offset: SSAValue,
        value: SSAValue,
        // the arm `STLR` does not use release semantics it uses seq_cst
        seq_cst: bool,
    },

    /// loads the halt reason found at [`Arg::HaltReasonPtr`]
    /// implementation:
    /// it is a relaxed atomic 32-bit native endian load
    /// this is more like `HasPendingHaltReasonButReturnsBitsBecauseBrZNeedsAValue`
    /// if it returns yes then, and only then do you synchronize, because this makes
    /// the fast path (no halt) very fast
    LoadHaltReason,

    /// takes the halt reason found at [`Arg::HaltReasonPtr`]
    /// and replaces it with 0
    /// implementation: `AcqRel` xchg [`Arg::HaltReasonPtr`] 0
    TakeHaltReason,

    /// atomic **Release** `fetch_or` of the [`Page::IS_DIRTY_FLAG`] bit
    /// look at [`Page::set_dirty`] for further details
    SetPageDirtyFlag(SSAValue),

    Safepoint,
}

#[derive(Debug)]
pub(crate) struct StmtData {
    pub(crate) outputs: ArrayVec<SSAValue, MAX_STMT_OUTPUTS>,
    pub(crate) rvalue: StmtKind,
}

impl_storable! {
    StmtData as impl pub(crate) Stmt;
    init: { }
}

const JUMP_PARAM_SMALL: usize = 8;

/// A branch edge: a target [`Block`] plus the values passed as its block
/// parameters.
///
/// Converts from a bare [`Block`] (no parameters) or a
/// `(Block, parameters)` pair.
#[derive(Debug)]
pub struct Jump {
    pub(crate) parameters: SmallVec<SSAValue, JUMP_PARAM_SMALL>,
    pub(crate) target: Block,
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
pub(crate) enum TerminatorKind {
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

/// How a [`Block`] ends: a return or a branch, with its target [`Jump`]s.
///
/// Construct one with [`Terminator::Return`], [`Terminator::ReturnCode`],
/// [`Terminator::Br`], [`Terminator::BrZ`], or [`Terminator::BrNZ`], and
/// attach it with [`ExecIrBuilder::terminate`].
#[derive(Debug)]
pub struct Terminator {
    pub(crate) targets: ArrayVec<Jump, { Self::MAX_TARGETS }>,
    pub(crate) kind: TerminatorKind,
}

impl Terminator {
    pub(crate) const MAX_TARGETS: usize = 2;

    pub(crate) fn block_targets(&self) -> arrayvec::IntoIter<Block, { Terminator::MAX_TARGETS }> {
        match self.targets.as_slice() {
            [] => empty_iter(),
            [one] => iter_from_arr([one.target]),
            [one, two] if one.target != two.target => iter_from_arr([one.target, two.target]),
            [target, _duplicate_target_different_params] => iter_from_arr([target.target]),

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
    /// Return from the exec chunk with the success code `0`.
    pub const Return: Self = Self {
        targets: ArrayVec::new_const(),
        kind: TerminatorKind::Return,
    };

    /// Return from the exec chunk with `halt_reason`, usually a non-zero `u32` block-exit reason.
    ///
    /// # Note
    /// an exit code of zero is legal but is suboptimal
    pub fn ReturnCode(halt_reason: SSAValue) -> Self {
        Self {
            targets: ArrayVec::new_const(),
            kind: TerminatorKind::ReturnCode { halt_reason },
        }
    }

    /// Unconditional branch to `jump`.
    pub fn Br(jump: impl Into<Jump>) -> Self {
        Self {
            targets: array_helper::from_arr([jump.into()]),
            kind: TerminatorKind::Br,
        }
    }

    /// Conditional branch: to `zero` if `cond` is zero/false,
    /// otherwise to `non_zero`. `cond` must be a boolean or integer value.
    pub fn BrZ(cond: SSAValue, zero: impl Into<Jump>, non_zero: impl Into<Jump>) -> Self {
        Self {
            targets: array_helper::from_arr([zero.into(), non_zero.into()]),
            kind: TerminatorKind::BrZ { cond },
        }
    }

    /// Conditional branch: to `zero` if `cond` is non-zero/true,
    /// otherwise to `non_zero`. `cond` must be a boolean or integer value.
    pub fn BrNZ(cond: SSAValue, non_zero: impl Into<Jump>, zero: impl Into<Jump>) -> Self {
        // just swap the arguments
        Self::BrZ(cond, zero, non_zero)
    }
}

#[derive(Debug)]
pub(crate) struct BlockData {
    pub(crate) parameters: SmallVec<SSAValue, JUMP_PARAM_SMALL>,
    pub(crate) stmts: Vec<Stmt>,
    pub(crate) terminated: bool,
    pub(crate) terminator: Terminator,
    pub(crate) is_cold: bool,
}

impl BlockData {
    fn empty() -> Self {
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
    BlockData as impl
    /// A handle to a basic block in the IR's control-flow graph.
    ///
    /// Blocks are created with [`ExecIrBuilder::create_block`], filled by
    /// emitting statements while they are the current block, and ended with
    /// a [`Terminator`].
    pub Block;
    init: {
        /// The function entry block. It implicitly receives every [`Arg`]
        /// as a block parameter and cannot be branched to.
        pub const ENTRYPOINT = Self {
            parameters: Arg::args().map(Arg::as_ssa_value).collect(),
            ..BlockData::empty()
        };
    }
}

/// A finished, immutable IR function ready to hand to an
/// [`ExecIrCompiler`](crate::compiler::ExecIrCompiler).
///
/// Produced by [`ExecIrBuilder::build`], which also runs the halt-check and
/// optimization passes and fixes the block compile order.
#[derive(Debug)]
pub struct ExecIr {
    pub(crate) ssa_values: Arena<SSAValueData>,
    pub(crate) blocks: Arena<BlockData>,
    pub(crate) stmts: Arena<StmtData>,
    pub(crate) stack_slots: Arena<StackSlotData>,
    pub(crate) signatures: Arena<CallbackSignatureData>,
    pub(crate) block_compile_order: Vec<Block>,
}

/// An in-progress [`ExecIr`] function under construction.
///
/// The builder tracks a *current block* that all emitters append to; use
/// [`create_block`](Self::create_block), [`switch_to`](Self::switch_to), and
/// [`terminate`](Self::terminate) to shape the control-flow graph, and call
/// [`build`](Self::build) when the function is complete.
#[derive(Debug)]
pub struct ExecIrBuilder {
    ssa_values: Arena<SSAValueData>,
    blocks: Arena<BlockData>,
    stmts: Arena<StmtData>,
    stack_slots: Arena<StackSlotData>,
    signatures: Arena<CallbackSignatureData>,
    scratch_space: Option<StackSlot>,
    leave_and_take_halt: Option<Block>,
    current_block: Block,
    halt_check_every: NonZero<u32>,
}

/// Configuration for an [`ExecIrBuilder`].
pub struct IrBuilderConfig {
    /// How many safepoints may pass between halt checks: the halt-check
    /// pass guarantees at least one check every `halt_check_every`
    /// safepoints along any execution path.
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
    /// Creates a new builder with the given configuration, positioned at
    /// [`Block::ENTRYPOINT`].
    pub fn with_config(config: IrBuilderConfig) -> Self {
        Self {
            ssa_values: Arena::new(),
            blocks: Arena::new(),
            stmts: Arena::new(),
            stack_slots: Arena::new(),
            signatures: Arena::new(),
            scratch_space: None,
            leave_and_take_halt: None,
            current_block: Block::ENTRYPOINT,
            halt_check_every: config.halt_check_every,
        }
    }

    /// Returns the block that emitters currently append to.
    pub fn current_block(&self) -> Block {
        self.current_block
    }

    /// Creates a new, empty block. The current block is left unchanged.
    pub fn create_block(&mut self) -> Block {
        self.blocks.store(BlockData::empty())
    }

    /// Makes `block` the current block that emitters append to.
    pub fn switch_to(&mut self, block: Block) {
        self.current_block = block;
    }

    /// Returns the distinct blocks that `block`'s terminator can branch to.
    pub fn successors(
        &self,
        block: Block,
    ) -> arrayvec::IntoIter<Block, { Terminator::MAX_TARGETS }> {
        self.blocks[block].terminator.block_targets()
    }

    /// Marks `block` as cold, hinting the backends to place it out of the
    /// hot path.
    ///
    /// If the block is [`Block::ENTRYPOINT`], it is silently exempt,
    /// since it always runs when the chunk is called, so it can never be cold.
    pub fn mark_block_bold(&mut self, block: Block) {
        // a cold entrypoint is insane and should never be true
        // it will always run when the resulting block is compiled
        // it can't be cold
        // this is made just so that if an exec block always unconditionally fails at the end,
        // this doesn't accidentally mark that cold
        if block != Block::ENTRYPOINT {
            self.blocks[block].is_cold = true;
        }
    }

    /// Marks the current block as cold; see
    /// [`mark_block_bold`](Self::mark_block_bold).
    pub fn mark_current_block_cold(&mut self) {
        self.mark_block_bold(self.current_block)
    }

    /// Ends `block` with `terminator`.
    ///
    /// # Panics
    ///
    /// Panics if `block` is already terminated, or if a terminator's operand
    /// has the wrong type, or if any target is [`Block::ENTRYPOINT`].
    pub fn terminate_block(&mut self, block: Block, mut terminator: Terminator) {
        match terminator.kind {
            TerminatorKind::Return => {}

            TerminatorKind::ReturnCode { halt_reason: int } => {
                std::assert_matches!(self.ssa_values[int].ty, TypeFull::Int(_))
            }

            TerminatorKind::BrZ { cond: int } => {
                std::assert_matches!(self.ssa_values[int].ty, TypeFull::Bool | TypeFull::Int(_));
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

        let block_data = &mut self.blocks[block];
        assert!(!block_data.terminated);
        block_data.terminator = terminator;
        block_data.terminated = true;
    }

    /// Ends the current block with `terminator`; see
    /// [`terminate_block`](Self::terminate_block).
    pub fn terminate(&mut self, terminator: Terminator) {
        let current_block = self.current_block;
        self.terminate_block(current_block, terminator)
    }

    /// Appends a parameter of type `ty` to `block`, returning the
    /// [`SSAValue`] the parameter binds. Every [`Jump`] to `block` must pass
    /// a matching value.
    pub fn add_block_parameter_at(&mut self, block: Block, ty: Type) -> SSAValue {
        let ty = TypeFull::from(ty);
        let ssa_value = self.ssa_values.store(SSAValueData { ty });
        self.blocks[block].parameters.push(ssa_value);
        ssa_value
    }

    /// Appends a parameter of type `ty` to the current block; see
    /// [`add_block_parameter_at`](Self::add_block_parameter_at).
    pub fn add_block_parameter(&mut self, ty: Type) -> SSAValue {
        let block = self.current_block;
        self.add_block_parameter_at(block, ty)
    }

    fn type_of(&self, rvalue: &StmtKind) -> arrayvec::IntoIter<TypeFull, MAX_STMT_OUTPUTS> {
        use array_helper::{empty_iter, iter_from_arr};

        match *rvalue {
            StmtKind::IConst(iconst) => iter_from_arr([TypeFull::Int(iconst.width())]),

            StmtKind::ArithBinOp { lhs, .. }
            | StmtKind::Bitwise { lhs, .. }
            | StmtKind::BitwiseImm { lhs, .. }
            | StmtKind::BitNot(lhs)
            | StmtKind::IntNeg(lhs)
            | StmtKind::Shift { value: lhs, .. }
            | StmtKind::ShiftUnbounded { value: lhs, .. }
            | StmtKind::ShiftImm { value: lhs, .. }
            | StmtKind::Select {
                cond: _,
                if_true: lhs,
                if_false: _,
            } => iter_from_arr([self.ssa_values[lhs].ty]),

            StmtKind::OverflowingBinOp { op: _, lhs, rhs: _ } => {
                iter_from_arr([self.ssa_values[lhs].ty, TypeFull::Bool])
            }

            StmtKind::IntCmp { .. } | StmtKind::IntCmpImm { .. } => iter_from_arr([TypeFull::Bool]),

            StmtKind::LoadHost { ty, .. } => match ty {
                LoadType::Int(width) => iter_from_arr([TypeFull::Int(width)]),
                LoadType::HostPtr(alias) => iter_from_arr([TypeFull::HostPtr(alias)]),
            },

            StmtKind::StoreHost { .. } => empty_iter(),

            StmtKind::LoadStackPtr { .. } => {
                iter_from_arr([TypeFull::HostPtr(AliasRegion::ScratchSpace)])
            }

            StmtKind::PtrAdd { base_ptr, .. } => {
                let ty = self.ssa_values[base_ptr].ty;
                std::debug_assert_matches!(ty, TypeFull::HostPtr(_));
                iter_from_arr([ty])
            }

            StmtKind::PtrEq(..) => iter_from_arr([TypeFull::Bool]),
            StmtKind::HasTag { .. } => iter_from_arr([TypeFull::Bool]),

            StmtKind::Untag { ptr, .. } => {
                let ty = self.ssa_values[ptr].ty;
                std::debug_assert_matches!(ty, TypeFull::HostPtr(_));
                iter_from_arr([ty])
            }

            StmtKind::HostCallback { signature, .. } => match self.signatures[signature].ret {
                None => empty_iter(),
                Some(ret) => iter_from_arr([ret]),
            },

            StmtKind::VMLoadRaw { width, .. } => iter_from_arr([TypeFull::Int(width)]),
            StmtKind::VMStoreRaw { .. } => empty_iter(),

            StmtKind::LoadHaltReason | StmtKind::TakeHaltReason => iter_from_arr([TypeFull::I32]),

            StmtKind::SetPageDirtyFlag { .. } => empty_iter(),

            StmtKind::Safepoint => empty_iter(),
        }
    }

    /// Runs `emitter` with `block` as the current block, restoring the
    /// previous current block afterward (even on unwind).
    pub fn block_scope<T>(&mut self, block: Block, emitter: impl FnOnce(&mut Self) -> T) -> T {
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

        emitter(builder)
    }

    /// # Safety
    ///
    /// the IR must not produce UB when run after compilation
    ///
    /// and the IR must have consistent expected types in and out
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

    /// Reserves `size` bytes of JIT stack space with the given power-of-two
    /// alignment.
    pub(crate) fn create_stack_slot(&mut self, size: u32, align: u8) -> StackSlot {
        assert!(align.is_power_of_two());

        self.stack_slots.store(StackSlotData { size, align })
    }

    /// Emits a statement producing a host pointer to `slot`.
    pub(crate) fn use_stack_slot(&mut self, slot: StackSlot) -> SSAValue {
        unsafe { self.emit_1ret_stmt(StmtKind::LoadStackPtr { slot }) }
    }

    /// Emits `iconst` as an integer constant of its width.
    pub fn iconst(&mut self, iconst: IConst) -> SSAValue {
        unsafe { self.emit_1ret_stmt(StmtKind::IConst(iconst)) }
    }

    unsafe fn load_from_processor_state(&mut self, offset: usize, width: IntWidth) -> SSAValue {
        unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::Int(width),
                base_ptr: SSAValue::ARG_EXEC_STATE,
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

    const unsafe fn x_reg_offset(x_reg: u8) -> usize {
        const {
            let _: [u64; X_REGISTER_COUNT as usize] = ExecState::initial().x_registers;
        }

        unsafe {
            core::hint::assert_unchecked(x_reg < X_REGISTER_COUNT);

            offset_of!(ExecState, x_registers)
                .unchecked_add((x_reg as usize).unchecked_mul(size_of::<u64>()))
        }
    }

    /// Loads the low `width` bytes of guest register `x_reg`, where `x_reg` is chosen at runtime.
    ///
    /// # Panics
    ///
    /// Panics if `x_reg` is not a valid X-register index.
    pub fn load_x_reg_dyn(&mut self, x_reg: u8, width: IntWidth) -> SSAValue {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.load_from_64_bit_processor_register(Self::x_reg_offset(x_reg), width) }
    }

    /// Loads the low `width` bytes of guest register `REG_IDX`, validated at compile time.
    pub fn load_x_reg<const REG_IDX: u8>(&mut self, width: IntWidth) -> SSAValue {
        let offset = const {
            assert!(REG_IDX < X_REGISTER_COUNT);
            unsafe { Self::x_reg_offset(REG_IDX) }
        };

        unsafe { self.load_from_64_bit_processor_register(offset, width) }
    }

    /// Loads the guest stack pointer as a 64-bit integer.
    pub fn load_sp(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ExecState, sp), IntWidth::W64) }
    }

    /// Loads the guest program counter as a 64-bit integer.
    pub fn load_pc(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ExecState, pc), IntWidth::W64) }
    }

    /// Loads the guest `PSTATE` flags word as a 32-bit integer.
    pub fn load_pstate(&mut self) -> SSAValue {
        unsafe { self.load_from_processor_state(offset_of!(ExecState, pstate), IntWidth::W32) }
    }

    unsafe fn store_to_processor_state(&mut self, offset: usize, value: SSAValue) {
        unsafe {
            self.emit_void_stmt(StmtKind::StoreHost {
                base_ptr: SSAValue::ARG_EXEC_STATE,
                offset,
                value,
                // it is always safe to access processor state
                can_move: true,
            })
        }
    }

    unsafe fn store_processor_register(&mut self, offset: usize, value: SSAValue) {
        let TypeFull::I64 = self.ssa_values[value].ty else {
            panic!("can only store 64 bit integers to processor registers")
        };

        unsafe { self.store_to_processor_state(offset, value) }
    }

    /// Stores a 64-bit `value` to guest register `x_reg`, chosen at
    /// runtime.
    ///
    /// # Panics
    ///
    /// Panics if `x_reg` is not a valid X-register index or if `value` is
    /// not a 64-bit integer.
    pub fn store_x_reg_dyn(&mut self, x_reg: u8, value: SSAValue) {
        assert!(x_reg < X_REGISTER_COUNT);
        unsafe { self.store_processor_register(Self::x_reg_offset(x_reg), value) }
    }

    /// Stores a 64-bit `value` to guest register `REG_IDX`, validated at
    /// compile time.
    pub fn store_x_reg<const REG_IDX: u8>(&mut self, value: SSAValue) {
        let offset = const {
            assert!(REG_IDX < X_REGISTER_COUNT);
            unsafe { Self::x_reg_offset(REG_IDX) }
        };

        unsafe { self.store_processor_register(offset, value) }
    }

    /// Stores a 64-bit `value` to the guest stack pointer.
    pub fn store_sp(&mut self, value: SSAValue) {
        unsafe { self.store_processor_register(offset_of!(ExecState, sp), value) }
    }

    /// Stores a 64-bit `value` to the guest program counter.
    pub fn store_pc(&mut self, value: SSAValue) {
        unsafe { self.store_processor_register(offset_of!(ExecState, pc), value) }
    }

    /// Stores a 32-bit `value` to the guest `PSTATE` flags word.
    pub fn store_pstate(&mut self, value: SSAValue) {
        let TypeFull::I32 = self.ssa_values[value].ty else {
            panic!("can only store 32 bit integers to pstate")
        };

        unsafe { self.store_to_processor_state(offset_of!(ExecState, pstate), value) }
    }

    /// Emits a branchless select: `if_true` when the boolean `cond` holds, otherwise `if_false`.
    ///
    /// Both arms are always evaluated and must have the same type.
    /// `cond` must be of type [`bool`](Type::Bool)
    pub fn select(&mut self, cond: SSAValue, if_true: SSAValue, if_false: SSAValue) -> SSAValue {
        assert_eq!(
            self.ssa_values[cond].ty,
            TypeFull::Bool,
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
        let args_ty: &[TypeFull] = &sig.args;
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

    /// Compares two integers of the same width with predicate `cmp`, producing a boolean.
    pub fn icmp(&mut self, cmp: IntCmp, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_same_int_ty_binop("comparisons", lhs, rhs, |this, _width| unsafe {
            this.emit_1ret_stmt(StmtKind::IntCmp { cmp, lhs, rhs })
        })
    }

    /// Compares an integer against the constant `rhs`
    /// (which must have the same width) with predicate `cmp`.
    ///
    /// Produces a boolean.
    pub fn icmp_imm(&mut self, cmp: IntCmp, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_same_int_ty_imm("comparisons", lhs, rhs.width, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::IntCmpImm {
                cmp,
                lhs,
                rhs: rhs.bits,
            })
        })
    }

    #[inline(always)]
    fn bitwise_type_guard<T>(
        &mut self,
        lhs: TypeFull,
        rhs: TypeFull,
        emit: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match (lhs, rhs) {
            (TypeFull::Bool, TypeFull::Bool) => emit(self),
            (TypeFull::HostPtr(_), TypeFull::HostPtr(_)) => {
                panic!("bitwise operations on pointers are not allowed")
            }
            (TypeFull::Int(width1), TypeFull::Int(width2)) => match width1 == width2 {
                true => emit(self),
                false => panic!("mismatched integer widths used for bitwise op"),
            },

            _ => panic!("mismatched types used for bitwise operation"),
        }
    }

    fn emit_bitwise(&mut self, op: BitwiseOp, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        let lhs_ty = self.ssa_values[lhs].ty;
        let rhs_ty = self.ssa_values[rhs].ty;
        self.bitwise_type_guard(lhs_ty, rhs_ty, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::Bitwise { op, lhs, rhs })
        })
    }

    fn emit_bitwise_imm(&mut self, op: BitwiseOp, lhs: SSAValue, rhs: IConst) -> SSAValue {
        let lhs_ty = self.ssa_values[lhs].ty;
        self.bitwise_type_guard(lhs_ty, TypeFull::Int(rhs.width), |this| unsafe {
            this.emit_1ret_stmt(StmtKind::BitwiseImm {
                op,
                lhs,
                rhs: rhs.bits,
            })
        })
    }

    /// Emits a bitwise OR of two booleans or two same-width integers.
    pub fn bitor(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_bitwise(BitwiseOp::Or, lhs, rhs)
    }

    /// Emits a bitwise AND of two booleans or two same-width integers.
    pub fn bitand(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_bitwise(BitwiseOp::And, lhs, rhs)
    }

    /// Emits a bitwise XOR of two booleans or two same-width integers.
    pub fn bitxor(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_bitwise(BitwiseOp::Xor, lhs, rhs)
    }

    /// Emits a bitwise OR of an integer and a same-width constant.
    pub fn bitor_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_bitwise_imm(BitwiseOp::Or, lhs, rhs)
    }

    /// Emits a bitwise AND of an integer and a same-width constant.
    pub fn bitand_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_bitwise_imm(BitwiseOp::And, lhs, rhs)
    }

    /// Emits a bitwise XOR of an integer and a same-width constant.
    pub fn bitxor_imm(&mut self, lhs: SSAValue, rhs: IConst) -> SSAValue {
        self.emit_bitwise_imm(BitwiseOp::Xor, lhs, rhs)
    }

    /// Emits a bitwise NOT of a boolean or integer `value`.
    pub fn bitnot(&mut self, value: SSAValue) -> SSAValue {
        let ty = self.ssa_values[value].ty;
        self.bitwise_type_guard(ty, ty, |this| unsafe {
            this.emit_1ret_stmt(StmtKind::BitNot(value))
        })
    }
}

impl ExecIrBuilder {
    fn emit_shift_op(
        &mut self,
        op: ShiftOp,
        value: SSAValue,
        shift: SSAValue,
        unbounded: bool,
    ) -> SSAValue {
        let TypeFull::Int(_) = self.ssa_values[value].ty else {
            panic!("can only shift integers")
        };

        let TypeFull::Int(_) = self.ssa_values[shift].ty else {
            panic!("can only shift integers by other integers")
        };

        let stmt = match unbounded {
            true => StmtKind::ShiftUnbounded { op, value, shift },
            false => StmtKind::Shift { op, value, shift },
        };

        unsafe { self.emit_1ret_stmt(stmt) }
    }

    /// Logical (zero-extending) right shift.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.
    pub fn ushr(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = false;
        self.emit_shift_op(ShiftOp::ZeroExtendShr, value, shift, unbounded)
    }

    /// Arithmetic (sign-extending) right shift.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.`
    pub fn sshr(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = false;
        self.emit_shift_op(ShiftOp::SignExtendShr, value, shift, unbounded)
    }

    /// Left shift.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.
    pub fn shl(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = false;
        self.emit_shift_op(ShiftOp::Shl, value, shift, unbounded)
    }

    /// Logical (zero-extending) right shift using unbounded semantics.
    ///
    /// If the runtime shift amount is greater than or equal to the integer
    /// width, the result is `0`.
    pub fn ushr_unbounded(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = true;
        self.emit_shift_op(ShiftOp::ZeroExtendShr, value, shift, unbounded)
    }

    /// Arithmetic (sign-extending) right shift using unbounded semantics.
    ///
    /// If the runtime shift amount is greater than or equal to the integer
    /// width, the operation behaves as though shifted by `width - 1`,
    /// broadcasting the sign bit across the result.
    pub fn sshr_unbounded(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = true;
        self.emit_shift_op(ShiftOp::SignExtendShr, value, shift, unbounded)
    }

    /// Left shift using unbounded semantics.
    ///
    /// If the runtime shift amount is greater than or equal to the integer
    /// width, the result is `0`.
    pub fn shl_unbounded(&mut self, value: SSAValue, shift: SSAValue) -> SSAValue {
        let unbounded = true;
        self.emit_shift_op(ShiftOp::Shl, value, shift, unbounded)
    }

    fn emit_shift_imm_op(
        &mut self,
        op: ShiftOp,
        value: SSAValue,
        bits: impl FnOnce(IntWidth) -> Result<u8, IConst>,
    ) -> SSAValue {
        let TypeFull::Int(width) = self.ssa_values[value].ty else {
            panic!("can only shift integers")
        };

        let bits = match bits(width) {
            Ok(bits) => bits,
            Err(const_val) => return self.iconst(const_val),
        };

        unsafe {
            self.emit_1ret_stmt(StmtKind::ShiftImm {
                op,
                value,
                shift: bits,
            })
        }
    }

    fn emit_modulo_shift_imm_op(&mut self, op: ShiftOp, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_shift_imm_op(op, value, move |width| {
            Ok(bits.strict_rem(u8::try_from(width.bits()).unwrap()))
        })
    }

    fn emit_panicking_shift_imm_op(&mut self, op: ShiftOp, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_shift_imm_op(op, value, move |width| {
            let width = width.bits();
            let (shift_direction, ty_name) = match op {
                ShiftOp::SignExtendShr => ("right", format_args!("i{width}")),
                ShiftOp::ZeroExtendShr => ("right", format_args!("u{width}")),
                ShiftOp::Shl => ("left", format_args!("u{width}")),
            };

            assert!(
                u32::from(bits) < width,
                "{shift_direction} shift by {bits} is out of bounds for {ty_name}",
            );
            Ok(bits)
        })
    }

    fn emit_unbounded_shift_imm_op(&mut self, op: ShiftOp, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_shift_imm_op(op, value, move |width| {
            let mut bits = bits;
            if (bits as u32) >= width.bits() {
                match op {
                    ShiftOp::ZeroExtendShr | ShiftOp::Shl => {
                        return Err(IConst::zero(width));
                    }
                    ShiftOp::SignExtendShr => {
                        bits = width.bits().strict_sub(1).try_into().unwrap();
                    }
                }
            }

            Ok(bits)
        })
    }

    /// Logical (zero-extending) right shift by a constant `bits`.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.`
    pub fn ushr_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_modulo_shift_imm_op(ShiftOp::ZeroExtendShr, value, bits)
    }

    /// Arithmetic (sign-extending) right shift by a constant `bits`.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.`
    pub fn sshr_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_modulo_shift_imm_op(ShiftOp::SignExtendShr, value, bits)
    }

    /// Left shift by a constant `bits`.
    ///
    /// The runtime shift amount is reduced modulo the integer width before
    /// shifting. For example, shifting a 32-bit value by `33` is equivalent
    /// to shifting by `1`.`
    pub fn shl_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_modulo_shift_imm_op(ShiftOp::Shl, value, bits)
    }

    /// Logical (zero-extending) right shift by a constant `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is greater than or equal to the integer width.
    pub fn ushr_exact_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_panicking_shift_imm_op(ShiftOp::ZeroExtendShr, value, bits)
    }

    /// Arithmetic (sign-extending) right shift by a constant `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is greater than or equal to the integer width.
    pub fn sshr_exact_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_panicking_shift_imm_op(ShiftOp::SignExtendShr, value, bits)
    }

    /// Left shift by a constant `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is greater than or equal to the integer width.
    pub fn shl_exact_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_panicking_shift_imm_op(ShiftOp::Shl, value, bits)
    }

    /// Logical (zero-extending) right shift by a constant `bits` using
    /// unbounded semantics.
    ///
    /// Shifts of the full width or more produce `0`.
    pub fn ushr_unbounded_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_unbounded_shift_imm_op(ShiftOp::ZeroExtendShr, value, bits)
    }

    /// Arithmetic (sign-extending) right shift by a constant `bits` using
    /// unbounded semantics.
    ///
    /// Shifts of the full width or more behave as though shifted by
    /// `width - 1`, broadcasting the sign bit.
    pub fn sshr_unbounded_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_unbounded_shift_imm_op(ShiftOp::SignExtendShr, value, bits)
    }

    /// Left shift by a constant `bits` using unbounded semantics.
    ///
    /// Shifts of the full width or more produce `0`.
    pub fn shl_unbounded_imm(&mut self, value: SSAValue, bits: u8) -> SSAValue {
        self.emit_unbounded_shift_imm_op(ShiftOp::Shl, value, bits)
    }
}

impl ExecIrBuilder {
    /// Stores the four boolean N, Z, C, V values into the guest `PSTATE`
    /// condition flags, leaving the other `PSTATE` bits untouched.
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

    /// Wrapping integer add of two same-width integers.
    pub fn iadd(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_arith_binop(ArithBinOp::Add, lhs, rhs)
    }

    /// Wrapping integer add of an integer and a same-width constant.
    pub fn iadd_imm(&mut self, value: SSAValue, amount: IConst) -> SSAValue {
        let rhs = self.iconst(amount);
        self.iadd(value, rhs)
    }

    /// Wrapping integer subtract of two same-width integers.
    pub fn isub(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_arith_binop(ArithBinOp::Sub, lhs, rhs)
    }

    /// Wrapping integer multiply of two same-width integers.
    pub fn imul(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
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

    /// Wrapping integer negation.
    pub fn ineg(&mut self, value: SSAValue) -> SSAValue {
        std::assert_matches!(self.ssa_values[value].ty, TypeFull::Int(_));
        unsafe { self.emit_1ret_stmt(StmtKind::IntNeg(value)) }
    }

    /// Wrapping add that also updates the guest NZCV condition flags, like
    /// an `ADDS` instruction.
    pub fn iadds(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_flag_setting_binop(OverflowingBinOp::Add, lhs, rhs)
    }

    /// Wrapping subtract that also updates the guest NZCV condition flags,
    /// like a `SUBS`/`CMP` instruction.
    pub fn isubs(&mut self, lhs: SSAValue, rhs: SSAValue) -> SSAValue {
        self.emit_flag_setting_binop(OverflowingBinOp::Sub, lhs, rhs)
    }

    /// Terminates the current block with a branch to `fail` unless `cond`
    /// equals `expected`, otherwise falling through to a freshly created
    /// success block.
    ///
    /// `fail` is marked cold. The builder is switched to the success block,
    /// which is also returned.
    ///
    /// Plainly worded `if cond != expected { goto fail }`
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

    fn make_halt_block(&mut self, reason: HaltReason) -> Block {
        let fail_block = self.create_block();

        self.block_scope(fail_block, |this| {
            let trap_value = this.iconst(IConst::u32(reason.as_nz_u32().get()));
            this.terminate(Terminator::ReturnCode(trap_value));
            this.mark_current_block_cold()
        });

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

    /// Emits a safepoint marking a guest-instruction boundary.
    ///
    /// The halt-check pass run by [`build`](Self::build) inserts halt
    /// checks at a subset of safepoints, so a halt request can only be
    /// observed between guest instructions (see
    /// [`IrBuilderConfig::halt_check_every`]).
    pub fn add_safepoint(&mut self) {
        unsafe { self.emit_void_stmt(StmtKind::Safepoint) }
    }

    /// Shorthand to both increment `pc` and insert a safepoint
    pub fn next_insn(&mut self) {
        let pc = self.load_pc();
        let new_pc = self.iadd_imm(pc, IConst::u64(4));
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
    fn has_tag(&mut self, ptr: SSAValue, tag_bits: u8) -> SSAValue {
        std::assert_matches!(self.ssa_values[ptr].ty, TypeFull::HostPtr(_));
        unsafe { self.emit_1ret_stmt(StmtKind::HasTag { ptr, tag_bits }) }
    }

    fn untag_ptr(&mut self, ptr: SSAValue, tag_bits: u8) -> SSAValue {
        std::assert_matches!(self.ssa_values[ptr].ty, TypeFull::HostPtr(_));
        unsafe { self.emit_1ret_stmt(StmtKind::Untag { ptr, tag_bits }) }
    }

    fn get_scratch_space_ptr(&mut self, width: IntWidth) -> SSAValue {
        const { assert!(IntWidth::MAX.bytes() == IntWidth::W64.bytes()) }
        let _ = width;

        if self.scratch_space.is_none() {
            self.scratch_space = Some(self.create_stack_slot(
                size_of::<u64>().try_into().unwrap(),
                align_of::<u64>().try_into().unwrap(),
            ))
        }

        self.use_stack_slot(self.scratch_space.unwrap())
    }

    fn emit_host_mem_fallback(
        &mut self,
        access: VmAccessKind,
        width: IntWidth,
        vaddr: SSAValue,
    ) -> FallbackAccess {
        let fallback_block = self.create_block();
        self.block_scope(fallback_block, |this| {
            this.mark_current_block_cold();
            let (out_param, (host_cb, sig), arg) = match access {
                VmAccessKind::Load { .. } => {
                    let stack_ptr = this.get_scratch_space_ptr(width);
                    let exclusive = false;
                    let func = load_callback(width, exclusive);
                    (Some(stack_ptr), func, stack_ptr)
                }
                VmAccessKind::Store { value } => {
                    let exclusive = false;
                    let func = store_callback(width, exclusive);
                    (None, func, value)
                }
            };

            let args = smallvec![
                SSAValue::ARG_IO_MMU,
                SSAValue::ARG_TLB_PTR,
                SSAValue::ARG_EXEC_CONTEXT,
                vaddr,
                // either an out param or a value to store
                arg
            ];

            let status = unsafe { this.call_host(host_cb, sig, args).unwrap() };

            const { assert!(IoMmuStatus::Ok as u8 == 0) };
            const { assert!(IoMmuStatus::Fault as u8 != 0) };

            let fail = this.make_halt_block(HaltReason::memory_trap(width.bytes()));
            this.assert_or_jmp_to(status, false, fail);

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

    // TODO/FIXME: figure out if you can actually **resume** the block for real
    //             and have a way to actually add a resume functionality
    //             this can be delayed for a pretty long time, but this comment
    //             should stay as long as we aren't sure this isn't possible
    fn vm_access(
        &mut self,
        vaddr: SSAValue,
        access: VmAccessKind,
        acq_rel: bool,
    ) -> Option<SSAValue> {
        std::assert_matches!(self.ssa_values[vaddr].ty, TypeFull::I64);

        let width = access.width(self);

        let fallback_access = self.emit_host_mem_fallback(access, width, vaddr);

        let page_number = self.sshr_exact_imm(vaddr, PAGE_SHIFT);
        let tlb_index = self.bitand_imm(page_number, IConst::u64(TLB_MASK));

        let tlb_entry_ptr = unsafe {
            self.emit_1ret_stmt(StmtKind::PtrAdd {
                base_ptr: SSAValue::ARG_TLB_PTR,
                offset: tlb_index,
                elem_size: const { NonZero::new(size_of::<TlbEntry>()).unwrap() },
            })
        };

        let io_mmu_ident = unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::HostPtr(AliasRegion::ReadOnly),
                base_ptr: tlb_entry_ptr,
                offset: offset_of!(TlbEntry, io_mmu_ident),
                // tlb access is always valid; since there are always TLB_SIZE
                // entries
                can_move: true,
            })
        };

        let io_mmu_matches = unsafe {
            self.emit_1ret_stmt(StmtKind::PtrEq(io_mmu_ident, SSAValue::ARG_IO_MMU_IDENT))
        };

        self.assert_or_jmp_to(io_mmu_matches, true, fallback_access.block);

        let page_number_found = unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::Int(IntWidth::W64),
                base_ptr: tlb_entry_ptr,
                offset: offset_of!(TlbEntry, virtual_page_number),
                // tlb access is always valid; since there are always TLB_SIZE
                // entries
                can_move: true,
            })
        };

        let page_number_matches = self.icmp(IntCmp::Equal, page_number, page_number_found);

        self.assert_or_jmp_to(page_number_matches, true, fallback_access.block);

        let tagged_page_ptr = unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::HostPtr(AliasRegion::VirtualMemory),
                base_ptr: tlb_entry_ptr,
                offset: offset_of!(TlbEntry, tagged_page_ptr),
                // tlb access is always valid; since there are always TLB_SIZE
                // entries
                can_move: true,
            })
        };

        let page_offset = self.bitand_imm(vaddr, IConst::u64(PAGE_OFFSET_MASK_U64));

        // a byte ptr only accesses the byte it is on and is always aligned
        if !matches!(width, IntWidth::W8) {
            let bytes = width.bytes_u64();
            debug_assert!(bytes.is_power_of_two());

            let align_mask = bytes.strict_sub(1);
            let alignment_bits = self.bitand_imm(vaddr, IConst::u64(align_mask));
            let is_aligned = self.icmp_imm(IntCmp::Equal, alignment_bits, IConst::u64(0));

            // naturally aligned access into a page
            // always fits in said page
            const { assert!(PAGE_SIZE.is_multiple_of(IntWidth::MAX.bytes() as usize)) }

            self.assert_or_jmp_to(is_aligned, true, fallback_access.block);
        }

        let required_perms = access.required_perms().bits();
        let op_allowed = self.has_tag(tagged_page_ptr, required_perms);

        let return_trap_block = {
            let fail_block = self.create_block();

            self.block_scope(fail_block, |this| {
                unsafe {
                    let store_ctx = |this: &mut Self, offset: usize, value: SSAValue| {
                        this.emit_void_stmt(StmtKind::StoreHost {
                            base_ptr: SSAValue::ARG_EXEC_CONTEXT,
                            offset,
                            value,
                            // it is always safe to access processor context
                            can_move: true,
                        })
                    };

                    store_ctx(
                        this,
                        offset_of!(ExecContext, current_mem_fault.vaddr),
                        vaddr,
                    );

                    let mem_op = match access {
                        VmAccessKind::Load { .. } => MemOp::Load,
                        VmAccessKind::Store { .. } => MemOp::Store,
                    };

                    const { assert!(size_of::<MemOp>() == size_of::<u8>()) }
                    let mem_op_val = this.iconst(IConst::u8(mem_op as u8));

                    store_ctx(
                        this,
                        offset_of!(ExecContext, current_mem_fault.mem_op),
                        mem_op_val,
                    );

                    let true_val = this.iconst(IConst::u8(1));
                    store_ctx(
                        this,
                        offset_of!(ExecContext, current_mem_fault.was_real_memory_trap),
                        true_val,
                    );
                };

                let reason = HaltReason::memory_trap(width.bytes());
                let trap_value = this.iconst(IConst::u32(reason.as_nz_u32().get()));
                this.terminate(Terminator::ReturnCode(trap_value));
                this.mark_current_block_cold()
            });

            fail_block
        };

        self.assert_or_jmp_to(op_allowed, true, return_trap_block);

        let aligned_page_ptr = self.untag_ptr(tagged_page_ptr, MemFlags::ALL.bits());
        let ret_value = {
            match access {
                VmAccessKind::Load { .. } => Some(unsafe {
                    self.emit_1ret_stmt(StmtKind::VMLoadRaw {
                        aligned_page_ptr,
                        page_offset,
                        width,
                        seq_cst: acq_rel,
                    })
                }),
                VmAccessKind::Store { value } => {
                    unsafe {
                        self.emit_void_stmt(StmtKind::VMStoreRaw {
                            aligned_page_ptr,
                            page_offset,
                            value,
                            seq_cst: acq_rel,
                        })
                    }
                    None
                }
            }
        };

        match access {
            VmAccessKind::Store { .. } => {
                assert!(ret_value.is_none());
                assert!(fallback_access.value.is_none());

                let page_must_dirty = self.has_tag(tagged_page_ptr, MemFlags::MUST_DIRTY.bits());

                let check_if_not_already_dirty = self.create_block();
                let continuation =
                    self.assert_or_jmp_to(page_must_dirty, false, check_if_not_already_dirty);

                self.block_scope(check_if_not_already_dirty, |this| {
                    let insn_dirty_ptr = unsafe {
                        this.emit_1ret_stmt(StmtKind::LoadHost {
                            ty: LoadType::HostPtr(AliasRegion::PageFlags),
                            base_ptr: tlb_entry_ptr,
                            offset: offset_of!(TlbEntry, mut_page_flags),
                            // tlb access is always valid; since there are always TLB_SIZE
                            // entries
                            can_move: true,
                        })
                    };

                    unsafe { this.emit_void_stmt(StmtKind::SetPageDirtyFlag(insn_dirty_ptr)) }

                    this.terminate(Terminator::Br(continuation))
                });

                // note that the fallback block independently executes this dirty checking
                // regardless of what the inline impl does, so it only just continues
                self.terminate_block(fallback_access.ok_block, Terminator::Br(continuation));

                None
            }

            VmAccessKind::Load { .. } => {
                let ret_normal = ret_value.unwrap();
                let ret_fallback = fallback_access.value.unwrap();

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
        }
    }

    /// Loads `width` bytes of guest memory at `vaddr`.
    ///
    /// # Replayability
    ///
    /// This operation may trap. On memory fault, the JIT exits with a memory fault
    /// reporting `vaddr`; a dispatcher above the JIT decides whether to
    /// resolve it (e.g., CoW break-and-retry) or deliver it to the guest. If
    /// resolved, **the faulting instruction is recompiled and re-executed
    /// from scratch starting at the current pc** - there is no resumption mid-instruction.
    ///
    /// This means every `vm_load`/`vm_store` for a guest instruction must
    /// happen before any guest-visible state derived from them is
    /// committed (register writes, flag updates, PC advance). If a load or
    /// store traps and is later replayed, the whole instruction reruns from
    /// the top — so nothing guest-visible may have been written yet when
    /// the trap fires. Pure recomputation (re-reading a register,
    /// re-deriving an address) is always safe to repeat; only writes to
    /// guest-visible state are not.
    pub fn vm_load(&mut self, vaddr: SSAValue, width: IntWidth) -> SSAValue {
        let acquire = false;
        match self.vm_access(vaddr, VmAccessKind::Load { width }, acquire) {
            Some(value) => value,
            None => unreachable!("load access must produce a value"),
        }
    }

    /// Stores `value` to guest memory at `vaddr`.
    ///
    /// # Replayability
    ///
    /// Same contract as [`Self::vm_load`]: this operation may trap, and a
    /// resolved trap means the instruction is recompiled and re-executed
    /// from scratch, not resumed. All memory accesses for the instruction —
    /// loads and stores alike — must happen before any guest-visible
    /// register or flag commit, since a later trap would replay the whole
    /// instruction including any commit that already ran.
    pub fn vm_store(&mut self, vaddr: SSAValue, value: SSAValue) {
        let release = false;
        let out = self.vm_access(vaddr, VmAccessKind::Store { value }, release);
        assert!(out.is_none());
    }
}

impl ExecIrBuilder {
    /// Loads `width` bytes of guest memory at `vaddr`, establishing an
    /// exclusive reservation on the accessed location for a matching
    /// [`Self::strex`].
    ///
    /// Implements the guest `LDXR`/`LDREX` family. Unlike [`Self::vm_load`],
    /// there is no inlined TLB fast path - every call goes through the host
    /// IoMMU callback, since arming the exclusive monitor is itself a
    /// host-side side effect, note ldrex always implements arm acquire semantics (seq_cst).
    ///
    /// # Replayability
    ///
    /// Same contract as [`Self::vm_load`]: this operation may trap, and a
    /// resolved trap means the instruction is recompiled and re-executed from
    /// scratch, not resumed. All memory accesses for the instruction must
    /// happen before any guest-visible state is committed.
    pub fn ldrex(&mut self, vaddr: SSAValue, width: IntWidth) -> SSAValue {
        std::assert_matches!(self.ssa_values[vaddr].ty, TypeFull::I64);

        let exclusive = true;
        let (host_cb, signature) = load_callback(width, exclusive);
        let stack_ptr = self.get_scratch_space_ptr(width);
        let args = smallvec![
            SSAValue::ARG_IO_MMU,
            SSAValue::ARG_TLB_PTR,
            SSAValue::ARG_EXEC_CONTEXT,
            vaddr,
            stack_ptr,
        ];

        let status = unsafe { self.call_host(host_cb, signature, args).unwrap() };

        const { assert!(IoMmuStatus::Ok as u8 == 0) };
        const { assert!(IoMmuStatus::Fault as u8 != 0) };

        let fail = self.make_halt_block(HaltReason::memory_trap(width.bytes()));
        self.assert_or_jmp_to(status, false, fail);

        unsafe {
            self.emit_1ret_stmt(StmtKind::LoadHost {
                ty: LoadType::Int(width),
                base_ptr: stack_ptr,
                offset: 0,
                // this operation is only safe **after** calling the host function and
                // ensuring that the operation did not trap
                can_move: false,
            })
        }
    }

    /// Conditionally stores `value` to guest memory at `vaddr` if the
    /// exclusive monitor armed by a matching [`Self::ldrex`] is still held,
    /// clearing the monitor either way.
    ///
    /// Implements the guest `STXR`/`STREX` family. Like [`Self::ldrex`], this
    /// always goes through the host IoMMU callback rather than the inlined TLB
    /// fast path, note strex always implements arm release semantics (seq_cst).
    ///
    /// Returns the raw exclusive-store status as an 8-bit value: `0` if the
    /// store succeeded, `1` if it failed because the monitor was not held —
    /// both are ordinary, non-trapping outcomes the caller branches on. A
    /// memory fault instead halts the JIT chunk with a memory trap rather than
    /// returning normally.
    ///
    /// # Replayability
    ///
    /// Same contract as [`Self::vm_store`]: this operation may trap, and a
    /// resolved trap means the instruction is recompiled and re-executed from
    /// scratch.
    pub fn strex(&mut self, vaddr: SSAValue, value: SSAValue) -> SSAValue {
        std::assert_matches!(self.ssa_values[vaddr].ty, TypeFull::I64);
        let width = self.ssa_values[value].ty.assert_int("strex");

        let exclusive = true;
        let (host_cb, signature) = store_callback(width, exclusive);
        let args = smallvec![
            SSAValue::ARG_IO_MMU,
            SSAValue::ARG_TLB_PTR,
            SSAValue::ARG_EXEC_CONTEXT,
            vaddr,
            value,
        ];

        let status = unsafe { self.call_host(host_cb, signature, args).unwrap() };

        const { assert!(StrexStatus::Stored as u8 == 0) };
        const { assert!(StrexStatus::Failed as u8 == 1) };
        const { assert!(StrexStatus::Fault as u8 == u8::MAX) };

        let failed = self.icmp_imm(IntCmp::Equal, status, IConst::u8(u8::MAX));

        let fail = self.make_halt_block(HaltReason::memory_trap(width.bytes()));
        self.assert_or_jmp_to(failed, false, fail);

        status
    }

    /// Clears the exclusive monitor for the current CPU without touching guest
    /// memory.
    ///
    /// Implements the guest `CLREX` instruction: any reservation armed by a
    /// prior [`Self::ldrex`] is dropped, so a subsequent [`Self::strex`]
    /// against that address is guaranteed to fail. This call cannot trap.
    pub fn clrex(&mut self) {
        let args = smallvec![SSAValue::ARG_EXEC_CONTEXT];
        let ret = unsafe {
            type ClrexCB = unsafe extern "C" fn(&mut ExecContext);
            let cb = std::mem::transmute::<ClrexCB, HostCallback>(ffi_support::clrex);
            self.call_host(cb, CallbackSignature::CLREX, args)
        };

        assert!(ret.is_none())
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

        let mut dfs_stack: SmallVec<DfsFrame, 128> = smallvec![DfsFrame::Enter(Block::ENTRYPOINT)];

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

    /// Finishes the function: runs the halt-check and optimization passes,
    /// computes the block compile order, and returns the immutable [`ExecIr`].
    #[must_use]
    pub fn build(mut self) -> ExecIr {
        halt_check_pass::insert_halt_checks(&mut self);
        optimization_pass::optimize(&mut self);
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
