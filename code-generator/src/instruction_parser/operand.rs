use crate::instruction_parser::isa::Isa;

enum OperandKind {
    Undefined,
    Register,
    ConstAndImmediate,
    Immediate,
    Const,
    Value,
    Enum,
    Memory,
    OptionalGroup,
    RegisterGroup,
    Select,
}

pub struct Operand<Arch: Isa> {
    kind: OperandKind,
    inner: Arch::Operand,
    has_bang: bool,
}

