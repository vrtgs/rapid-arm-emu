use crate::ir::arena::make_handle;

mod arena;


make_handle!(Value);
make_handle!(Stmt);


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum IntWidth {
    W8 = 1,
    W16 = 2,
    W32 = 4,
    W64 = 8,
}

impl IntWidth {
    pub const fn bits(self) -> u8 {
        self as u8 * 8
    }

    pub const fn bytes(self) -> usize {
        self as usize
    }
}


#[derive(Debug, Clone)]
pub struct StmtData {
    pub kind: StmtKind,
    pub result: Option<Value>,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
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
        lhs: Value,
        rhs: Value,
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
        base: Value,
        offset: usize,
    },

    /// Store to a host pointer plus a constant byte offset.
    StoreHost {
        width: IntWidth,
        base: Value,
        offset: usize,
        value: Value,
    },

    /// Load from VM memory through the `IoMMU`.
    ///
    /// `vaddr` must be an `i64` value.
    LoadVm {
        width: IntWidth,
        io_mmu: Value,
        vaddr: Value,
    },

    /// Store to VM memory through the `IoMMU`.
    ///
    /// `vaddr` must be an `i64` value.
    /// `value` must have type `Int(width)`.
    StoreVm {
        width: IntWidth,
        io_mmu: Value,
        vaddr: Value,
        value: Value,
    },
}

pub struct SSABlock {

    terminator: Option<Terminator>,
}




/// Conceptual signature of every translated basic block:
///
/// ```text
/// fn(
///     io_mmu: *const IoMMU,
///     halt_reason: *const AtomicU32,
///     processor_state: *mut ProcessorState,
/// ) -> NonZero<u32>
/// ```
#[derive(Debug, Clone)]
pub struct VMBasicBlockBuilder {
    values: Vec<ValueData>,
    insts: Vec<InstData>,
    params: BasicBlockParams,
}



#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BasicBlockParams {
    pub processor_state: Value,
    pub io_mmu: Value,
    pub halt_reason: Value,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Type {
    Int(IntWidth),
    Ptr(PointerKind),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PointerKind {
    IoMmu,
    AtomicU32,
    ProcessorState,
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
pub struct ValueData {
    pub ty: Type,
    pub def: ValueDef,
}

#[derive(Debug, Clone)]
pub enum ValueDef {
    Param(Param),
    Inst(Inst),
}

#[derive(Debug, Clone)]
pub struct Param {
    pub index: u8,
    pub name: &'static str,
}


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Return a non-zero `u32` block-exit reason.
    ///
    /// The IR expects this value to be non-zero. The builder can check the type,
    /// but cannot prove non-zero-ness for arbitrary SSA values.
    ReturnNonZero {
        value: Value,
    },
}

impl BasicBlockIr {
    pub fn new() -> Self {
        let mut ir = Self {
            values: Vec::new(),
            insts: Vec::new(),
            params: BasicBlockParams {
                io_mmu: Value(u32::MAX),
                halt_reason: Value(u32::MAX),
                processor_state: Value(u32::MAX),
            },
            terminator: None,
        };

        let io_mmu = ir.push_param(Type::Ptr(PointerKind::IoMmu), 0, "io_mmu");
        let halt_reason = ir.push_param(Type::Ptr(PointerKind::AtomicU32), 1, "halt_reason");
        let processor_state =
            ir.push_param(Type::Ptr(PointerKind::ProcessorState), 2, "processor_state");

        ir.params = BasicBlockParams {
            io_mmu,
            halt_reason,
            processor_state,
        };

        ir
    }

    pub fn params(&self) -> BasicBlockParams {
        self.params
    }

    pub fn values(&self) -> &[ValueData] {
        &self.values
    }

    pub fn insts(&self) -> &[InstData] {
        &self.insts
    }

    pub fn terminator(&self) -> Option<Terminator> {
        self.terminator
    }

    pub fn value_type(&self, value: Value) -> Type {
        self.values[value.index()].ty
    }

    fn push_param(
        &mut self,
        ty: Type,
        index: u8,
        name: &'static str,
    ) -> Value {
        self.push_value(ValueData {
            ty,
            def: ValueDef::Param(Param { index, name }),
        })
    }

    fn push_value(&mut self, data: ValueData) -> Value {
        let index = self.values.len();
        let value = Value(u32::try_from(index).unwrap());
        self.values.push(data);
        value
    }

    fn push_inst(&mut self, kind: StmtKind, result_ty: Option<Type>) -> (Inst, Option<Value>) {
        let inst_index = self.insts.len();
        let inst = Inst(u32::try_from(inst_index).unwrap());

        let result = result_ty.map(|ty| {
            self.push_value(ValueData {
                ty,
                def: ValueDef::Inst(inst),
            })
        });

        self.insts.push(InstData { kind, result });

        (inst, result)
    }
}

impl Value {
    pub const fn raw(self) -> u32 {
        self.0
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

impl Inst {
    pub const fn raw(self) -> u32 {
        self.0
    }
}

