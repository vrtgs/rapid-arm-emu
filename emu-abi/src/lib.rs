//! ABI types shared between the emulator runtime and generated code.

/// Abort utilities for unwinding-safe process termination.
pub mod abort;
/// Helpers for converting fixed-size arrays into [`arrayvec::ArrayVec`] and its iterator.
pub mod array_helper;
/// Checked integer conversion utilities
pub mod convert;
/// CPU execution state: registers, flags, and SIMD vectors.
pub mod exec_state;
/// Halt reason codes used to signal why an emulated CPU core stopped.
pub mod halt_reason;
/// Internal traits not intended for use outside this workspace.
pub mod internal_traits;
/// Memory subsystem types: page tables, protection flags, TLB, and page pointers.
pub mod memory;
