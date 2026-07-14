/// A 128-bit SIMD/vector register value, stored as a 16-byte-aligned `u128`.
///
/// This represents one of the AArch64 `V0`–`V31` vector/FP registers.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(16))]
pub struct Vector(pub u128);

const _: () = assert!(align_of::<Vector>() == 16 && size_of::<Vector>() == 16);

/// The number of general-purpose `X` registers (`X0`–`X30`), excluding the
/// stack pointer / zero register, which are tracked separately.
pub const X_REGISTER_COUNT: u8 = 31;

/// The AArch64 processor state (condition flags), as stored in `NZCV`.
///
/// Only the top 4 bits (`N`, `Z`, `C`, `V`) are currently modeled; the
/// remaining bits are reserved/unused.
#[derive(bytemuck::Zeroable, Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct PState(pub u32);

impl PState {
    /// The `Negative` condition flag (bit 31): set when the result of an
    /// operation was negative in two's-complement representation.
    pub const NEGATIVE: Self = Self(1 << 31);

    /// The `Zero` condition flag (bit 30): set when the result of an
    /// operation was zero.
    pub const ZERO: Self = Self(1 << 30);

    /// The `Carry` condition flag (bit 29): set on unsigned overflow (or
    /// borrow, for subtraction).
    pub const CARRY: Self = Self(1 << 29);

    /// The `Overflow` condition flag (bit 28): set on signed overflow.
    pub const OVERFLOW: Self = Self(1 << 28);

    /// Short alias for [`Self::NEGATIVE`].
    pub const N: Self = Self::NEGATIVE;

    /// Short alias for [`Self::ZERO`].
    pub const Z: Self = Self::ZERO;

    /// Short alias for [`Self::CARRY`].
    pub const C: Self = Self::CARRY;

    /// Short alias for [`Self::OVERFLOW`].
    pub const V: Self = Self::OVERFLOW;

    /// A mask covering all four condition flag bits (`N`, `Z`, `C`, `V`).
    pub const NZCV_MASK: Self = Self(Self::N.0 | Self::Z.0 | Self::C.0 | Self::V.0);
}

/// The full architectural execution state of an emulated AArch64 core.
///
/// This includes the program counter, general-purpose and vector register
/// files, stack pointer, condition flags, and floating-point control/status
/// registers.
#[derive(bytemuck::Zeroable, Clone)]
// use `repr(C)` so that we can put hot field s next to each other
// so they land on the same cacheline and so that hot fields have
// smaller constant indices to fit inline in an instruction encoding
// rather than an integer immediate, but do note that repr(C) is NOT
// required for safety, and all offset calculations must use `offset_of!`
// this is only here as an optimization and not for correctness
// that is why, we target `repr(Rust)` on debug and miri builds
// to catch any bugs caused by not using `offset_of!`
// we exclude doc so rustdoc doesn't advertise this as a public layout guarantee
#[cfg_attr(
    any(not(any(doc, debug_assertions, miri)), all(test, not(miri))),
    repr(C)
)]
pub struct ExecState {
    /// The program counter, i.e., the address of the instruction that is about to execute.
    pub pc: u64,
    /// The stack pointer (`SP`).
    pub sp: u64,
    /// The current condition flags (`NZCV`).
    pub pstate: PState,
    /// The general-purpose registers `X0`–`X30`.
    pub x_registers: [u64; X_REGISTER_COUNT as usize],
    /// The floating-point status register (`FPSR`).
    pub fpsr: u32,
    /// The floating-point control register (`FPCR`).
    pub fpcr: u32,
    /// The vector/FP registers `V0`–`V31`.
    pub vectors: [Vector; 32],
}

impl ExecState {
    /// Returns the initial (reset) execution state, with all registers,
    /// flags, and the program counter zeroed.
    #[inline(always)]
    pub const fn initial() -> Self {
        bytemuck::zeroed()
    }
}
