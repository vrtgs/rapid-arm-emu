pub(crate) struct SyncCell<T: ?Sized>(T);

impl<T> SyncCell<T> {
    pub(crate) fn new(value: T) -> Self {
        SyncCell(value)
    }
}

unsafe impl<T> Sync for SyncCell<T> {}
