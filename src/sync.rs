
cfg_select! {
    test => {
        use loom::sync::Mutex as LoomMutex;
        use std::sync::TryLockError;

        pub use loom::sync::MutexGuard;

        pub struct Mutex<T>(LoomMutex<T>);

        impl<T> Mutex<T> {
            pub fn new(value: T) -> Self {
                Self(LoomMutex::new(value))
            }

            pub fn lock(&self) -> MutexGuard<'_, T> {
                self.0.lock().unwrap_or_else(|err| err.into_inner())
            }

            pub fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
                self.0.try_lock().map(Some).unwrap_or_else(|err| match err {
                    TryLockError::WouldBlock => None,
                    TryLockError::Poisoned(err) => Some(err.into_inner())
                })
            }
        }
    }
    _ => {
        pub use parking_lot::Mutex;
        pub use parking_lot::MutexGuard;
    }
}
