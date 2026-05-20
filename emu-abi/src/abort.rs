pub use std::process::abort;

pub struct AbortGuard(pub ());

impl AbortGuard {
    pub fn disarm(self) {
        core::mem::forget(self)
    }
}

impl Drop for AbortGuard {
    fn drop(&mut self) {
        abort()
    }
}
