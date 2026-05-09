use crate::armv9::{Armv9CpuCore, ProcessorState};
use emu_abi::halt_reason::HaltReasonInner;
use emu_abi::memory::HostPointer;
use emu_abi::memory::PagePointer;
use std::collections::HashMap;
use std::ops::Range;

// this might seem wierd, but when compiling a basic block,
// we might start from one place, and go back
// like:
//               top:
//               nop
//               nop
// jumps here -> add x, y;
//               jump top

// TODO:
//   - Make compiled code shared across CPU cores.
//   - Keep instruction-byte invalidation separate from virtual
//     mapping/protection invalidation.
//   - Move cross-core code-cache bookkeeping into CpuFabric.

pub(crate) struct CodeBlock {
    /// Half-open real address range touched while decoding.
    ///
    /// Range semantics:
    ///     [start, end)
    ///
    /// This is used for cache invalidation, not for dispatch lookup.
    ///
    /// `start` is not guaranteed to be the entrypoint of the chunk.
    addr: Range<HostPointer>,

    machine_code_handle: exec_ir::compiler::CompiledExecChunk,
}

impl CodeBlock {
    fn execute(&self, _state: &mut ProcessorState, _cpu: &Armv9CpuCore) -> HaltReasonInner {
        todo!()
    }
}

pub(crate) struct CodeCache {
    cache: HashMap<HostPointer, CodeBlock>,
}

impl CodeCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn run(&mut self, _state: &mut ProcessorState, _cpu: &Armv9CpuCore) -> HaltReasonInner {
        todo!()
    }

    pub fn invalidate_cached_page(&mut self, page: PagePointer) {
        let range = page.as_range();
        self.cache.retain(move |_entrypoint, block| {
            let collides = range.start < block.addr.end && block.addr.start < range.end;
            !collides
        })
    }
}
