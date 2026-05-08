pub(crate) struct SyncCell<T: ?Sized>(T);

impl<T> SyncCell<T> {
    pub fn new(value: T) -> Self {
        SyncCell(value)
    }
}

impl<T: ?Sized> SyncCell<T> {
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

unsafe impl<T> Sync for SyncCell<T> {}
