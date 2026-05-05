use std::sync::atomic::AtomicU32;
use crate::armv9::ProcessorState;
use crate::io_mmu;
use crate::io_mmu::IoMMU;
use crate::ir::arena::{impl_storable, Arena};

mod arena;


type ExecBlock = unsafe extern "C" fn(
    processor_state: &mut ProcessorState,
    pages: *const io_mmu::Page,
    page_count: u64,
    halt_reason_ptr: *const AtomicU32,
    io_mmu: *const IoMMU,
) -> u32;


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
pub enum ArithOp {
    /// Wrapping integer add.
    Add,

    /// Wrapping integer subtract.
    Sub,

    /// Wrapping integer multiply.
    Mul,

    /// Integer division.
    ///
    /// This is a normal value-producing instruction.
    ///
    /// It does not branch, does not panic, and does not terminate the block.
    /// If `rhs == 0`, the result is `0`.
    Div,
}


#[derive(Debug, Clone)]
pub enum RValue {
    /// Integer constant.
    Iconst {
        width: IntWidth,
        value: u64,
    },

    /// Integer arithmetic.
    ///
    /// `lhs` and `rhs` must have type `Int(width)`.
    /// The result also has type `Int(width)`.
    Arith {
        op: ArithOp,
        width: IntWidth,
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
        width: IntWidth,
        base_ptr: LValue,
        offset: usize,
        value: LValue,
    },

    LoadHaltReason(LValue),
}


pub struct Stmt {
    pub lvalue: LValue,
    pub rvalue: RValue,
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Return "0" i.e. return success.
    Return,
    /// Return a `NonZero<u32>` block-exit reason.
    ReturnFail {
        value: LValue
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


impl_storable!(
    BlockData as impl pub Block;
    init: {
        const ENTRYPOINT = BlockData {
            stmts: vec![],
            terminator: Terminator::Return,
            is_cold: false,
        };
    }
);


pub struct ExecIrBuilder {
    pub lvalues: Arena<LValueData>,
    pub blocks: Arena<BlockData>,
    pub current_block: Block,
}

impl ExecIrBuilder {
    pub fn new() -> Self {
        Self {
            lvalues: Arena::new(),
            blocks: Arena::new(),
            current_block: Block::ENTRYPOINT
        }
    }

    #[must_use]
    fn make_lvalue(&mut self, ty: Type) -> LValue {
        self.lvalues.store(LValueData { ty })
    }

}
