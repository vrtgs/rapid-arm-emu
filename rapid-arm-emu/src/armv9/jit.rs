use crate::armv9::{Armv9CpuCore, ExecState};
use emu_abi::halt_reason::HaltReason;
use emu_abi::memory::{HostPointer, PagePointer};
use std::collections::HashMap;

// this might seem weird, but when compiling a basic block,
// we might start from one place and go back
// like:
//               top:
//               nop
//               nop
// jumps here -> add x, y;
//               jump top
// do note code blocks must not cross pages

// TODO:
//   - Make compiled code shared across CPU cores.
//   - Keep instruction-byte invalidation separate from virtual
//     mapping/protection invalidation.
//   - Move cross-core code-cache bookkeeping into CpuFabric.

pub(crate) struct CodeBlock {
    _page: PagePointer,
    _machine_code_handle: exec_ir::compiler::CompiledExecChunk,
}

impl CodeBlock {}

pub(crate) struct CodeCache {
    cache: HashMap<HostPointer, CodeBlock>,
}

impl CodeCache {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub(crate) fn run(
        &mut self,
        _state: &mut ExecState,
        _cpu: &Armv9CpuCore,
    ) -> Option<HaltReason> {
        todo!()
    }

    pub(crate) fn invalidate(&mut self) {
        self.cache.clear()
    }
}
