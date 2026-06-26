pub use std::process::abort;

pub struct AbortGuard(pub ());

impl AbortGuard {
    pub fn disarm(self) {
        core::mem::forget(self)
    }

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
