pub use std::process::abort;

/// A guard that aborts the process when dropped unless explicitly disarmed.
///
/// Wrap a section that must not unwind with `AbortGuard` to convert any panic
/// unwind into an unconditional process abort, preventing partially modified
/// state from being observed by other threads.
///
/// # Constructing
///
/// `AbortGuard` has no `new()` function — construct it directly with the
/// unit field:
///
/// ```
/// use emu_abi::abort::AbortGuard;
///
/// let guard = AbortGuard(());
/// guard.disarm(); // or guard.abort()
/// ```
pub struct AbortGuard(#[allow(missing_docs)] pub ());

impl AbortGuard {
    /// Consumes the guard without aborting, allowing the current scope to exit normally.
    pub fn disarm(self) {
        core::mem::forget(self)
    }

    /// Aborts the process immediately.
    pub fn abort(self) -> ! {
        abort()
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        abort()
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! __panic_abort {
    ($($arg:tt)*) => {{
        let _abort_guard = $crate::abort::AbortGuard(());
        panic!($($arg)*)
    }};
}

/// panic without unwinding
#[doc(inline)]
pub use __panic_abort as panic_abort;
