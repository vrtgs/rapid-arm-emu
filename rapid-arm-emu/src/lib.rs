#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(
    // F32 -> F64, Int -> Float, etc.
    clippy::cast_precision_loss,
    // U32 -> U8, etc.
    clippy::cast_possible_truncation,
    // Signed -> Unsigned, etc.
    clippy::cast_possible_wrap,
    // Signed -> Unsigned
    clippy::cast_sign_loss,
    clippy::arithmetic_side_effects,
    reason = "emulators require precise bit-level accuracy; \
              implicit casts can introduce subtle, hard-to-debug architectural discrepancies"
)]

// FIXME; remove generic VAddr, there is no a64 and a32 CPU, there is just a an Armv9-A CPU
//        which also happens to have 2 execution states; AArch32 and AArch64

mod a64;
pub mod armv9;
mod array_helper;
pub mod cpu_fabric;
pub mod halt_reason;
pub mod io_mmu;
mod ir;
pub mod vaddr;
