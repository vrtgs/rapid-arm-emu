//! Per-CPU execution state threaded through every compiled chunk.
//!
//! This is the JIT-side state that is *not* part of the architectural guest
//! state: the [`ExecContext`] holds the current exclusive-monitor reservation
//! and a [`FFISafeMemoryFault`] slot where compiled code records guest memory
//! faults across the `extern "C"` boundary.

use io_mmu::cpu_fabric::exclusive_monitor::Reservation;
use io_mmu::fault::{MemoryFault, MemoryFaultReason};
use std::hint::cold_path;
use std::mem::MaybeUninit;

/// The kind of guest memory operation that raised a memory fault.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum MemOp {
    /// A guest load faulted.
    Load,
    /// A guest store faulted.
    Store,
    /// A guest read-modify-write (e.g., an atomic) faulted.
    Rwm,
}

/// FFI-safe storage for a pending guest [`MemoryFault`].
///
/// Compiled code cannot return a rich Rust error across its `extern "C"`
/// boundary, so the JIT's memory-access fallbacks record faults in here
/// (via [`set_fault`](Self::set_fault)),
/// and the dispatcher retrieves them afterward with [`take_memory_fault`](Self::take_memory_fault).
pub struct FFISafeMemoryFault {
    pub(crate) was_real_memory_trap: bool,
    pub(crate) vaddr: u64,
    pub(crate) mem_op: MemOp,
    has_bus_error: bool,
    bus_error: MaybeUninit<anyhow::Error>,
}

impl Default for FFISafeMemoryFault {
    fn default() -> Self {
        Self::new()
    }
}

impl FFISafeMemoryFault {
    /// Creates an empty slot with no pending fault.
    pub(crate) const fn new() -> Self {
        const {
            Self {
                was_real_memory_trap: false,
                vaddr: 0,
                mem_op: MemOp::Load,
                has_bus_error: false,
                bus_error: MaybeUninit::uninit(),
            }
        }
    }

    fn drop_bus_error(&mut self) {
        if self.has_bus_error {
            self.has_bus_error = false;
            unsafe { self.bus_error.assume_init_drop() }
        }
    }

    fn take_bus_error(&mut self) -> Option<anyhow::Error> {
        match self.has_bus_error {
            true => {
                self.has_bus_error = false;
                Some(unsafe { self.bus_error.assume_init_read() })
            }
            false => None,
        }
    }

    /// Records `error` as the pending fault, remembering `mem_op` as the
    /// kind of access that raised it. Any previously stored bus error is
    /// dropped.
    pub fn set_fault(&mut self, mem_op: MemOp, error: MemoryFault) {
        self.vaddr = error.vaddr();
        self.mem_op = mem_op;
        self.was_real_memory_trap = true;

        match error.into_reason() {
            MemoryFaultReason::GeneralProtection => {}
            MemoryFaultReason::MemoryBus(error) => {
                cold_path();
                self.drop_bus_error();
                self.bus_error.write(error);
                self.has_bus_error = true;
            }
        }
    }

    /// Takes the pending fault, if any, clearing the slot.
    ///
    /// Returns the reconstructed [`MemoryFault`] together with the kind of
    /// memory operation that raised it, or `None` if no fault is pending.
    pub fn take_memory_fault(&mut self) -> Option<(MemoryFault, MemOp)> {
        if !self.was_real_memory_trap {
            return None;
        }

        self.was_real_memory_trap = false;

        let bus_error = self.take_bus_error();

        let fault = match bus_error {
            None => MemoryFault::general_protection(self.vaddr),
            Some(bus_error) => {
                cold_path();
                MemoryFault::memory_bus(self.vaddr, bus_error)
            }
        };

        Some((fault, self.mem_op))
    }
}

impl Drop for FFISafeMemoryFault {
    fn drop(&mut self) {
        self.drop_bus_error();
    }
}

/// Per-CPU execution context threaded through every compiled chunk.
///
/// Holds JIT-side state that is not part of the architectural guest state:
/// the current exclusive-monitor reservation and the pending-memory-fault slot.
///
/// # Note
///
/// This doesn't need to be kept or stored across separate executions (like when context switching),
/// and it can always just be recreated, though it should generally be reused so that
/// exclusive monitor updates have a lower chance of failing
pub struct ExecContext {
    pub(crate) exclusive_monitor_reservation: Option<Reservation>,
    /// Where compiled code records a guest memory fault; check and clear it
    /// with [`FFISafeMemoryFault::take_memory_fault`] after a chunk exits
    /// with a memory-fault halt reason.
    pub current_mem_fault: FFISafeMemoryFault,
}

impl ExecContext {
    /// Creates the initial context: no exclusive reservation and no pending memory fault.
    #[inline(always)]
    pub const fn initial() -> Self {
        const {
            Self {
                exclusive_monitor_reservation: None,
                current_mem_fault: FFISafeMemoryFault::new(),
            }
        }
    }
}
