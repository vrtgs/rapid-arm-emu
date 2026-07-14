//! JIT compilation core for the emulator.
//!
//! Guest code is first translated into [`ExecIr`](ir::ExecIr), an SSA-based
//! intermediate representation built with an
//! [`ExecIrBuilder`](ir::ExecIrBuilder). An
//! [`ExecIrCompiler`](compiler::ExecIrCompiler) then lowers the IR through
//! one of several backends (selected by a
//! [`CompileTier`](compiler::CompileTier)) into a host-callable
//! [`CompiledExecChunk`].

pub(crate) mod arena;
pub mod chunk;
pub mod compiler;
pub mod exec_context;
pub mod ir;
mod sync_cell;
