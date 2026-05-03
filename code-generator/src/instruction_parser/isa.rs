pub enum IsaEnum {
    A64
}

pub trait Isa: Send + Sync + 'static {
    type Operand: Send + Sync;
    type Register: Send + Sync;

    const NAME: &str;
    const AS_ENUM: IsaEnum;
}


pub enum A64 {}

pub enum A64Operand {
    /// Operand is not defined.
    None,
    /// Register
    Register = 1,
    /// Register group {X0,X1...}
    RegisterGroup = 2,
    /// SystemRegister
    SystemRegister = 3,
    /// A memory operand
    Memory = 4,
    /// An immediate
    Immediate = 5,
    /// Label.
    Label = 6,
    /// Shift.
    Shift = 7,
    /// Shift.
    Extend = 8,
    /// An enum value
    Enum = 9,
}

impl Isa for A64 {
    type Operand = A64Operand;
    type Register = ();

    const NAME: &str = "aarch64";
    const AS_ENUM: IsaEnum = IsaEnum::A64;
}
