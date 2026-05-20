use emu_abi::abort::AbortGuard;
use emu_abi::halt_reason::AtomicHaltReason;
use emu_abi::internal_traits::ICache;
use emu_abi::memory::PagePointer;
use parking_lot::{Condvar, Mutex, MutexGuard};
use slab::Slab;
use std::cell::UnsafeCell;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::Arc;

pub struct InsnCache {}

impl ICache for InsnCache {
    fn invalidate(&self, _page: PagePointer) {
        todo!()
    }
}

pub(crate) type IoMMU = io_mmu::IoMMU<InsnCache>;

/// Stack-allocated container for a single `enqueue_modify` invocation.
///
/// Lives on the requesting thread's stack frame; that thread is parked
/// inside `refresh_inner` for the entire window during which `IoMMUOp`
/// holds a pointer to it.
struct ThunkData<F, T> {
    func: Option<F>,
    data: Option<T>,
}

/// Type-erased trampoline. Moves `F` out of the `ThunkData`, calls it,
/// and stores the return value back.
///
/// # Safety
///
/// - `data` must point to a valid, initialized `ThunkData<F, T>` for the
///   exact `F` and `T` this thunk was monomorphized for.
/// - The `ThunkData` must contain `func: Some(_)` and `data: None`
///   (i.e., this thunk has not been called before for this data).
/// - The caller must have exclusive access to the `ThunkData` for the
///   duration of the call.
unsafe fn thunk<F: FnOnce(&mut IoMMU) -> T, T>(mmu: &mut IoMMU, data: *mut ()) {
    unsafe {
        let data = data.cast::<ThunkData<F, T>>().as_mut_unchecked();
        std::hint::assert_unchecked(data.data.is_none());
        std::hint::assert_unchecked(data.func.is_some());
        let func: F = data.func.take().unwrap_unchecked();
        let output: T = func(mmu);
        data.data = Some(output);
    }
}

/// A type-erased IoMMU modification.
///
/// `data` points to a `ThunkData<F, T>` living on the requesting thread's
/// stack. The requesting thread is suspended inside `enqueue_modify` for
/// the entire lifetime of this entry, so the pointer is guaranteed valid
/// until `func` consumes it.
struct IoMMUOp {
    func: unsafe fn(&mut IoMMU, *mut ()),
    data: *mut (),
}

struct AddrSpaceQueue {
    /// Every live `AddrSpaceGuard` has an entry here pointing at its thread's
    /// halt reason. The slab being empty is the drain-ready condition.
    threads: Slab<NonNull<AtomicHaltReason>>,
    /// Queued IoMMU modifications awaiting the next drain.
    pending: Vec<IoMMUOp>,
}

struct AddrSpaceInner {
    /// The IoMMU itself. Accessed as `&IoMMU` by readers (when their key is
    /// in the slab) and as `&mut IoMMU` by whichever thread performs the
    /// drain (when the slab is empty under the mutex).
    iommu: UnsafeCell<IoMMU>,
    queue: Mutex<AddrSpaceQueue>,
    /// Signalled by the draining thread after the pending queue is emptied.
    /// Wakes (a) threads parked in `enter` waiting to join during a pending
    /// shootdown, and (b) threads parked in `refresh_inner` waiting for
    /// someone else to drain.
    resume: Condvar,
}

// SAFETY: pointers in `threads` come from `&AtomicHaltReason` borrows
// outliving their guards; pointers in `pending` reference stack frames
// of threads parked inside `enqueue_modify`. Both are only ever touched
// under the queue mutex, so transferring the queue between threads (as
// the mutex itself does on each lock/unlock) is sound.
unsafe impl Send for AddrSpaceInner {}

// SAFETY: access to `iommu` is gated by the drain protocol — `&mut` is
// only taken under the queue mutex with an empty slab, and `&` is only
// handed out via `AddrSpaceGuard::deref` while its entry is in the slab.
// Both invariants are preserved across threads.
unsafe impl Sync for AddrSpaceInner {}

#[derive(Clone)]
pub struct AddrSpace(Arc<AddrSpaceInner>);

pub(crate) struct AddrSpaceGuard<'a> {
    space: &'a AddrSpaceInner,
    /// Slab key for our entry in `threads`. Valid for the entire lifetime
    /// of this guard, updated in place by `refresh_inner`.
    key: usize,
}

impl<'a> AddrSpaceGuard<'a> {
    /// Join the address space.
    ///
    /// If a shootdown is in progress (pending non-empty), block on the
    /// condvar until it completes. This prevents late joiners from stalling
    /// the drain barrier and from seeing the pre-drain IoMMU identity.
    fn enter(
        space: &'a AddrSpaceInner,
        halt_reason: &'a AtomicHaltReason,
    ) -> (Self, MutexGuard<'a, AddrSpaceQueue>) {
        let mut lock = space.queue.lock();
        while !lock.pending.is_empty() {
            space.resume.wait(&mut lock);
        }
        let key = lock.threads.insert(NonNull::from_ref(halt_reason));

        let this = Self { space, key };

        (this, lock)
    }

    /// Remove self from the slab, and if we're the last reader with pending
    /// ops, drain them inline.
    ///
    /// Returns our halt pointer (so the caller can re-insert it later) and
    /// whether we performed the drain ourselves.
    ///
    /// # Safety
    ///
    /// Caller **must not** use `self` again without first inserting the
    /// returned pointer back into the slab and updating `self.key`
    /// accordingly. `queue` must be from the same `AddrSpaceInner` as `self`.
    #[inline(always)]
    unsafe fn pop(&mut self, queue: &mut AddrSpaceQueue) -> (NonNull<AtomicHaltReason>, bool) {
        // SAFETY: `self.key` is always a valid slab index - it was assigned
        // by `enter` or by the previous `refresh_inner`, and nothing else
        // removes our entry except this method (called from `refresh_inner`
        // or `Drop`, both of which run at most once per key).
        let halt_ptr = unsafe { queue.threads.try_remove(self.key).unwrap_unchecked() };

        let drain = queue.threads.is_empty() && !queue.pending.is_empty();

        if drain {
            // SAFETY: We hold the queue mutex, and `threads.is_empty()` means
            // no `AddrSpaceGuard` currently has shared access to the IoMMU. The
            // mutex prevents any new `AddrSpaceGuard` from being constructed
            // (`enter` blocks on `lock()`), so we have exclusive access for
            // the duration of this borrow.
            let iommu = unsafe { &mut *self.space.iommu.get() };
            for op in queue.pending.drain(..) {
                // SAFETY: `op.data` points at a `ThunkData<F, T>` on the
                // stack of the thread that called `enqueue_modify`. That
                // thread is currently parked inside `refresh_inner` waiting
                // for the queue to drain, so the stack frame is still live.
                // `op.func` is the matching `thunk::<F, T>` for that data -
                // paired together at push time in `enqueue_modify`.
                unsafe { (op.func)(iommu, op.data) }
            }
            self.space.resume.notify_all();
        }

        (halt_ptr, drain)
    }

    /// Conceptually drop-and-reacquire in place: pop, optionally run an
    /// extra operation under the lock, wait for the drain to complete if
    /// someone else is doing it, then re-insert with the same halt pointer.
    ///
    /// # Safety
    ///
    /// `in_between_op` must not corrupt `AddrSpaceQueue`'s invariants -
    /// in particular, it must not modify `threads` or push entries into
    /// `pending` whose `data` pointers don't satisfy `IoMMUOp`'s liveness
    /// requirements.
    unsafe fn refresh_inner(
        &mut self,
        in_between_op: impl FnOnce(&mut AddrSpaceQueue),
        mut guard: MutexGuard<'_, AddrSpaceQueue>,
    ) {
        // The slab transitions through a state where our key has been
        // removed but not yet replaced. A panic in `in_between_op` or in
        // the condvar wait would leave the queue inconsistent and could
        // deadlock other threads waiting at the drain barrier. Aborting
        // is the only safe option.
        let abort_guard = AbortGuard(());

        // SAFETY: we re-insert `ptr` into the slab and update `self.key`
        // before this function returns, satisfying `pop`'s contract.
        // `guard` is locked from `self.space.queue`.
        let (ptr, drained) = unsafe { self.pop(&mut guard) };

        in_between_op(&mut guard);

        if !drained {
            while !guard.pending.is_empty() {
                self.space.resume.wait(&mut guard)
            }
        }

        self.key = guard.threads.insert(ptr);
        abort_guard.disarm();
    }
}

impl AddrSpaceGuard<'_> {
    pub fn sync_up(&mut self) {
        let guard = self.space.queue.lock();
        if guard.pending.is_empty() {
            // this is a simple fast path opt
            // nothing to sync to, halt reason was stale
            return;
        }

        // SAFETY: the no-op `in_between_op` cannot corrupt queue invariants.
        unsafe { self.refresh_inner(|_| {}, guard) }
    }

    /// Queue an IoMMU modification, signal all other readers to drain, then
    /// participate in the drain ourselves. Synchronous - does not return
    /// until the closure has been executed and its result is in hand.
    ///
    /// `F: Send` because the closure may be executed by a different thread
    /// (whichever reader exits last). `T: Send` for the same reason - the
    /// return value is produced on the draining thread and read here.
    ///
    /// # Safety
    /// TODO
    unsafe fn enqueue_modify_locked<F, T>(&mut self, lock: MutexGuard<AddrSpaceQueue>, op: F) -> T
    where
        F: FnOnce(&mut IoMMU) -> T,
        F: Send,
        T: Send,
    {
        let mut guard = lock;

        let mut data: ThunkData<F, T> = ThunkData {
            func: Some(op),
            data: None,
        };

        guard.pending.push(IoMMUOp {
            func: thunk::<F, T>,
            // `data` lives on this stack frame. It stays valid because
            // `refresh_inner` below blocks until the drain has completed,
            // and the `AbortGuard` prevents unwinding from invalidating
            // this pointer while it's still in the queue.
            data: (&raw mut data).cast::<()>(),
        });

        // If we unwind past this point with `data`'s pointer still in the
        // pending queue, a later drain would invoke `thunk` against a
        // freed stack slot. Abort instead.
        let abort_guard = AbortGuard(());

        // SAFETY: `in_between_op` only reads `threads` and calls
        // `try_signal_sync` on the contained pointers; it does not mutate
        // the queue, so invariants are preserved.
        unsafe {
            self.refresh_inner(
                |guard| {
                    for (_, &ptr) in &guard.threads {
                        // SAFETY: slab pointers come from `&AtomicHaltReason` references
                        // whose lifetimes outlive their owning `AddrSpaceGuard`. Entries
                        // are only removed under the queue mutex, which we hold via
                        // `guard`, so every pointer here is currently dereferenceable.
                        let reference = ptr.as_ref();
                        reference.try_signal_sync();
                    }
                },
                guard,
            );
        }

        abort_guard.disarm();

        // SAFETY: `refresh_inner` returned, meaning the drain completed.
        // The drain invokes `thunk`, which sets `data.data = Some(_)`.
        unsafe { data.data.unwrap_unchecked() }
    }

    /// Queue an IoMMU modification, signal all other readers to drain, then
    /// participate in the drain ourselves. Synchronous - does not return
    /// until the closure has been executed and its result is in hand.
    ///
    /// `F: Send` because the closure may be executed by a different thread
    /// (whichever reader exits last). `T: Send` for the same reason - the
    /// return value is produced on the draining thread and read here.
    pub fn enqueue_modify<F, T>(&mut self, op: F) -> T
    where
        F: FnOnce(&mut IoMMU) -> T,
        F: Send,
        T: Send,
    {
        unsafe { self.enqueue_modify_locked(self.space.queue.lock(), op) }
    }
}

impl Drop for AddrSpaceGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: `self` is being dropped, so `pop`'s "must re-insert
        // before reuse" clause is vacuously satisfied.
        unsafe {
            self.pop(&mut self.space.queue.lock());
        }
    }
}

impl Deref for AddrSpaceGuard<'_> {
    type Target = IoMMU;

    fn deref(&self) -> &Self::Target {
        // SAFETY: while this `AddrSpaceGuard` exists, our entry is in the slab,
        // which means no thread can acquire `&mut IoMMU` - a drain requires
        // `threads.is_empty()` under the mutex, and we're in it. So a
        // shared reference is sound for the lifetime of `&self`.
        unsafe { self.space.iommu.get().as_ref_unchecked() }
    }
}

impl AddrSpace {
    /// # Safety
    ///
    /// The returned `AddrSpaceGuard` **must** be dropped - not leaked via
    /// `mem::forget` or similar. Leaking it would let a dangling reference
    /// to `atomic_halt` outlive its borrow, causing undefined behavior on
    /// later operations.
    pub(crate) unsafe fn enter<'a: 'b, 'b>(
        &'a self,
        atomic_halt: &'b AtomicHaltReason,
    ) -> AddrSpaceGuard<'b> {
        AddrSpaceGuard::enter(&self.0, atomic_halt).0
    }

    pub fn modify<F, T>(&self, op: F) -> T
    where
        F: FnOnce(&mut IoMMU) -> T,
        F: Send,
        T: Send,
    {
        // TODO this works, and is kind of cheap..... well not exactly
        //      there is still two lock operations
        //      one for drop; currently this also inserts, then pops, then reinserts
        //      which isn't great
        let reason = AtomicHaltReason::new();
        let (mut guard, lock) = AddrSpaceGuard::enter(&self.0, &reason);

        unsafe { guard.enqueue_modify_locked(lock, op) }
    }
}
