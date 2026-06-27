use crate::cpu_fabric::CpuFabricWeak;
use crate::icache::ICache;
use crate::memory_object::MemoryObject;
use crate::page_table::{MemoryBackedPage, ObjectSlot};
use anyhow::bail;
use emu_abi::abort::AbortGuard;
use emu_abi::abort::panic_abort;
use emu_abi::convert::{u64_to_usize, usize_to_u64};
use emu_abi::memory::PageNumber;
use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Weak};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ObjSlotId(*const ObjectSlot);

impl ObjSlotId {
    fn of(arc: &Arc<ObjectSlot>) -> Self {
        Self(Arc::as_ptr(arc))
    }

    fn from_weak(weak: &Weak<ObjectSlot>) -> Self {
        Self(weak.as_ptr())
    }
}

unsafe impl Send for ObjSlotId {}
unsafe impl Sync for ObjSlotId {}

pub(crate) trait ObjectSyncCallbacks: Send {
    fn on_success(self: Box<Self>);

    fn on_failure(self: Box<Self>, error: &anyhow::Error);
}

pub(crate) struct FnCallback<F>(F);

impl<F: FnOnce(Result<(), &anyhow::Error>) + Send> FnCallback<F> {
    pub(crate) const fn new_flush(callback: F) -> Self {
        Self(callback)
    }
}

// Note: with inlining this usually gets 2 specialized functions for success and failure
//       avoiding the cost of picking the Err and Ok branch, which is why we use this API
//       instead of FnOnce() and also because `ObjectFlushCallbacks` can be implemented on
//       concrete types, which is the major reason we use a trait instead of FnOnce()
impl<F: FnOnce(Result<(), &anyhow::Error>) + Send> ObjectSyncCallbacks for FnCallback<F> {
    fn on_success(self: Box<Self>) {
        let cb = (*self).0;
        cb(Ok(()))
    }

    fn on_failure(self: Box<Self>, error: &anyhow::Error) {
        let cb = (*self).0;
        cb(Err(error))
    }
}

#[repr(transparent)]
struct SucceedOnDrop(ManuallyDrop<Box<dyn ObjectSyncCallbacks>>);

impl SucceedOnDrop {
    fn wrap(callbacks: Box<dyn ObjectSyncCallbacks>) -> Self {
        Self(ManuallyDrop::new(callbacks))
    }

    fn into_callbacks(self) -> Box<dyn ObjectSyncCallbacks> {
        let mut this = ManuallyDrop::new(self);
        unsafe { ManuallyDrop::take(&mut this.0) }
    }
}

impl Drop for SucceedOnDrop {
    fn drop(&mut self) {
        let cb = unsafe { ManuallyDrop::take(&mut self.0) };
        if !std::thread::panicking() {
            let _ = catch_unwind(|| cb.on_success());
        }
    }
}

fn catch_unwind<F: FnOnce() -> R, R>(f: F) -> std::thread::Result<R> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
}

enum OpType {
    Init(CpuFabricWeak<dyn ICache>),
    Flush,
}

struct QueueEntry {
    slot: Weak<ObjectSlot>,
    callbacks: SmallVec<Box<dyn ObjectSyncCallbacks>, 2>,
    op_type: OpType,
}

struct EnqueueEntry<'a> {
    slot: &'a Arc<ObjectSlot>,
    op_type: OpType,
    callback: Option<SucceedOnDrop>,
}

struct Queue {
    queue: VecDeque<QueueEntry>,
    already_in_queue: HashMap<ObjSlotId, u64>,
    // Even at 1 billion entries per second, this u64 takes ~585 years to overflow.
    // not a practical concern on any platform.
    base: u64,
}

impl Queue {
    fn assert_invariant(&self) {
        let abort = AbortGuard(());
        assert_eq!(self.queue.len(), self.already_in_queue.len());
        abort.disarm();
    }

    fn dequeue(&mut self) -> Option<QueueEntry> {
        self.assert_invariant();
        let entry = self.queue.pop_front()?;

        let abort = AbortGuard(());
        let object_id = ObjSlotId::from_weak(&entry.slot);
        let removed = self.already_in_queue.remove(&object_id);

        // must be the last entry (the one just popped)
        assert_eq!(removed, Some(self.base));

        self.base = match self.queue.is_empty() {
            // this isn't necessary but makes debugging slightly easier
            // as it makes the indices nicer to reason about and compute in your head / verify
            true => 0,
            false => self.base.strict_add(1),
        };

        abort.disarm();

        Some(entry)
    }

    fn enqueue<'a>(
        &mut self,
        enqueue: EnqueueEntry<'a>,
        queue_limit: usize,
    ) -> Result<(), Option<EnqueueEntry<'a>>> {
        self.assert_invariant();

        // Safe: OnceCell only goes None->Some, never back, so `Some` here is
        // permanent - there's no future state where this slot needs init again.
        if let OpType::Init(_) = enqueue.op_type
            && enqueue.slot.page.get().is_some()
        {
            // the callback succeeds immediately because of the drop impl of `SucceedOnDrop`
            return Err(None);
        }

        match self.already_in_queue.entry(ObjSlotId::of(enqueue.slot)) {
            Entry::Occupied(index) => {
                let abort = AbortGuard(());
                let index = u64_to_usize((*index.get()).strict_sub(self.base)).unwrap();
                let entry = &mut self.queue[index];
                abort.disarm();

                match (enqueue.op_type, &entry.op_type) {
                    (OpType::Init(new_enqueue_cpu), OpType::Init(already_queued)) => {
                        if !new_enqueue_cpu.is(already_queued) {
                            panic_abort!("unreachable: iommu can't switch CpuFabric")
                        }
                    }

                    (OpType::Flush, OpType::Flush) => {}

                    (OpType::Init(_), OpType::Flush) => {
                        // if `Flush` is already queued,
                        // that means that slot.page is init,
                        // but above we just checked and ensured that page is uninit
                        panic_abort!("unreachable: queue contains flush for uninit page")
                    }

                    (OpType::Flush, OpType::Init(_)) => {
                        if enqueue.slot.page.get().is_none() {
                            panic_abort!("enqueued flush for uninit page")
                        }

                        entry.op_type = OpType::Flush;
                    }
                };

                if let Some(callback) = enqueue.callback {
                    entry.callbacks.push(callback.into_callbacks());
                }

                Ok(())
            }

            Entry::Vacant(entry) => {
                if self.queue.len() >= queue_limit {
                    return Err(Some(enqueue));
                }

                let queue_entry = QueueEntry {
                    slot: Arc::downgrade(enqueue.slot),
                    callbacks: match enqueue.callback {
                        Some(cb) => smallvec::smallvec![cb.into_callbacks()],
                        None => smallvec::smallvec![],
                    },
                    op_type: enqueue.op_type,
                };

                let abort = AbortGuard(());
                let pushed_val_idx = usize_to_u64(self.queue.len()).unwrap();
                let abs_index = self.base.strict_add(pushed_val_idx);
                self.queue.push_back(queue_entry);

                entry.insert(abs_index);
                abort.disarm();

                Ok(())
            }
        }
    }
}

struct QueueInner {
    queue: Queue,
    closed: bool,
}

struct Inner {
    queue_limit: usize,
    queue: Mutex<QueueInner>,
    enqueued: Condvar,
    dequeued: Condvar,
}

#[cold]
#[inline(never)]
fn worker_thread_died() -> ! {
    panic_abort!("async flusher thread died unexpectedly")
}

impl Inner {
    fn close(&self) {
        let mut lock = self.queue.lock();
        lock.closed = true;
        drop(lock);
        // there is no chance of a lost wake-up
        // since a thread that grabs the lock between `drop` and `notify_all` will see
        // `closed = true` immediately - so it doesn't need the notification.
        //
        // and a thread already waiting on the condvar gets woken by `notify_all`
        // and will see `closed = true` when it re-acquires the lock.
        //
        // lastly thread that arrives after `notify_all`
        // will see `closed = true` the moment it locks,
        // so it never needs to wait on the condvar at all.
        self.enqueued.notify_all();
        self.dequeued.notify_all();
    }

    fn enqueue(
        &self,
        slot: &Arc<ObjectSlot>,
        op_type: OpType,
        callback: Option<Box<dyn ObjectSyncCallbacks>>,
    ) {
        // note if enqueue gets canceled
        // its assumed that it is because the operation already completed successfully
        let succeed_on_drop = callback.map(SucceedOnDrop::wrap);

        // No TOCTOU here: a flush request means "write back whatever page is
        // resident right now," not "write back whatever page exists by the
        // time this gets processed." If `page.get()` is `None` at the instant
        // this call is made, there is categorically nothing resident to write
        // back - that's the correct, final answer for *this* flush request,
        // not a stale read of a value that might change underneath us.
        //
        // This also doesn't depend on OnceCell's None->Some monotonicity the
        // way the `Init` check below does. Even if a concurrent `Init` is
        // in flight and completes a moment later, that doesn't retroactively
        // mean this flush needed to do anything: the page wasn't there when
        // the flush was requested, so "trivially succeeded" is correct.
        if let OpType::Flush = op_type
            && slot.page.get().is_none()
        {
            // the callback succeeds immediately because on drop
            return;
        }

        let mut entry = EnqueueEntry {
            slot,
            op_type,
            callback: succeed_on_drop,
        };

        let mut lock = self.queue.lock();
        loop {
            if lock.closed {
                worker_thread_died()
            }

            match lock.queue.enqueue(entry, self.queue_limit) {
                Ok(()) => break,
                // nothing got queued because the operation is no longer necessary
                Err(None) => return,
                Err(Some(requeue)) => entry = requeue,
            }

            self.dequeued.wait(&mut lock);
        }

        drop(lock);

        self.enqueued.notify_one();
    }

    fn dequeue(&self) -> Option<QueueEntry> {
        let mut lock = self.queue.lock();
        loop {
            if lock.closed {
                return None;
            }

            if let Some(entry) = lock.queue.dequeue() {
                self.dequeued.notify_one();
                return Some(entry);
            }

            self.enqueued.wait(&mut lock);
        }
    }
}

fn handle_task(entry: QueueEntry) {
    let QueueEntry {
        slot,
        callbacks,
        op_type,
    } = entry;

    let Some(slot) = slot.upgrade() else {
        return;
    };

    let res: std::thread::Result<anyhow::Result<()>> = match op_type {
        OpType::Init(cpu_fabric) => {
            let res = catch_unwind(|| {
                slot.page.get_or_try_init(|| {
                    let object: &dyn MemoryObject = &*slot.object;
                    let offset: PageNumber = slot.page_offset;
                    let fault_res = MemoryBackedPage::fault_object(object, offset, || cpu_fabric);

                    fault_res.map(Arc::new)
                })
            });

            match res {
                Ok(Ok(_)) => Ok(Ok(())),
                Ok(Err(anyhow)) => Ok(Err(anyhow)),
                Err(panic) => Err(panic),
            }
        }

        OpType::Flush => {
            let slot_ref: &ObjectSlot = &slot;
            let page = slot_ref
                .page
                .get()
                .unwrap_or_else(|| panic_abort!("queued flush without backing page"));

            let page_ptr = page.page_pointer();

            catch_unwind(move || unsafe {
                slot_ref
                    .object
                    .fault_out(slot_ref.page_offset, page_ptr.as_non_null_ptr())
            })
        }
    };

    // drop the slot before ever calling the callbacks
    drop(slot);

    let res = res.unwrap_or_else(|_| bail!("memory object panicked whilst faulting page"));

    // note leaking any callback handles can cause deadlocks in the program
    // if any callback is lost and forgotten during iteration bad things can happen
    // abort if a panic can lead to leaking
    let abort = AbortGuard(());
    let callbacks = callbacks.into_iter();

    match res {
        Ok(()) => callbacks.for_each(|cb| {
            let _panic = catch_unwind(|| cb.on_success());
        }),
        Err(ref error) => callbacks.for_each(|cb| {
            let _panic = catch_unwind(|| cb.on_failure(error));
        }),
    }

    abort.disarm();
}

struct ObjectManagerWorker(Arc<Inner>);

impl Drop for ObjectManagerWorker {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl ObjectManagerWorker {
    fn run_worker(self) {
        while let Some(entry) = self.0.dequeue() {
            handle_task(entry)
        }
    }
}

// TODO add parallel task scheduling / handling with a threadpool
pub(crate) struct ObjectManager {
    // make the handle to `Inner` weak
    // so that we avoid any ref-cycles that may happen from storing `CpuFabric`
    // inside a callback; and also all we really need inner for is to queue things
    // we don't ever need any additional data from it, so there is no need for an `Arc`
    inner: Weak<Inner>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for ObjectManager {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.upgrade() {
            inner.close();
        }

        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
        }
    }
}

impl ObjectManager {
    pub(crate) fn new(queue_limit: usize) -> Self {
        let inner = Arc::new(Inner {
            queue_limit,
            queue: Mutex::new(QueueInner {
                queue: Queue {
                    queue: VecDeque::new(),
                    already_in_queue: HashMap::new(),
                    base: 0,
                },
                closed: false,
            }),
            enqueued: Condvar::new(),
            dequeued: Condvar::new(),
        });

        // construct self first
        // this way if the thread construction panics inner still closes
        let mut this = Self {
            inner: Arc::downgrade(&inner),
            thread: None,
        };

        let worker = ObjectManagerWorker(inner);

        let thread = std::thread::spawn(move || worker.run_worker());

        this.thread = Some(thread);

        this
    }

    pub(crate) fn enqueue_flush(
        &self,
        page: &Arc<ObjectSlot>,
        callback: Option<Box<dyn ObjectSyncCallbacks>>,
    ) {
        self.inner
            .upgrade()
            .unwrap_or_else(|| worker_thread_died())
            .enqueue(page, OpType::Flush, callback)
    }

    pub(crate) fn enqueue_init(
        &self,
        page: &Arc<ObjectSlot>,
        cpu_fabric_weak: CpuFabricWeak<dyn ICache>,
        callback: Option<Box<dyn ObjectSyncCallbacks>>,
    ) {
        self.inner
            .upgrade()
            .unwrap_or_else(|| worker_thread_died())
            .enqueue(page, OpType::Init(cpu_fabric_weak), callback)
    }
}

impl Default for ObjectManager {
    fn default() -> Self {
        let queue_limit = 1024;
        Self::new(queue_limit)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    #[test]
    fn sanity_check() {
        // note this is required for the correctness of the object manager,
        // but the current documentation of
        // [`Weak::as_ptr`](https://doc.rust-lang.org/std/sync/struct.Weak.html#method.as_ptr)
        // explicitly states that this relation doesn't need to hold as of rust version 1.96.0
        let strong = Arc::new("sanity check");
        let weak = Arc::downgrade(&strong);
        let strong_ptr = Arc::as_ptr(&strong);

        drop(strong);

        // Both point to the same object
        assert!(std::ptr::eq(strong_ptr, weak.as_ptr()));
    }
}
