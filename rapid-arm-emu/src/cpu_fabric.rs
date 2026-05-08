//! CPU fabric state shared by emulated CPU contexts.
//!
//! The `cpu_fabric` module owns state that is conceptually shared between CPUs, cores.
//! Today, that shared state is the exclusive monitor which is used to model
//! load-exclusive/store-exclusive style atomic sequences.
//!
//! In the future, `CpuFabric` may grow to include other CPU-adjacent shared
//! resources, such as interrupt routing, cache-coherency metadata, shared
//! timing state, or other cross-core coordination structures. For now, it is a
//! small wrapper around the exclusive monitor so cloned CPU contexts can observe
//! and invalidate the same reservation state.

use crate::io_mmu::HostPointer;
use crossbeam_utils::CachePadded;
use parking_lot::Mutex;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub(crate) const BUCKET_COUNT: u16 = 257;

struct SplitMixConstants {
    shift1: u32,
    c1: usize,
    shift2: u32,
    c2: usize,
    shift3: u32,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "usize's size is checked and we only convert from the native size"
)]
const SPLIT_MIX_CONSTANTS: SplitMixConstants = match usize::BITS {
    // 128 bit targets are exotic,
    // and never seen, but if tehy ever come out, just use the low 64 bits
    64.. => SplitMixConstants {
        shift1: 30,
        c1: 0xbf58_476d_1ce4_e5b9_u64 as usize,
        shift2: 27,
        c2: 0x94d0_49bb_1331_11eb_u64 as usize,
        shift3: 31,
    },

    32 => SplitMixConstants {
        shift1: 16,
        c1: 0x7feb_352d_u32 as usize,
        shift2: 15,
        c2: 0x846c_a68b_u32 as usize,
        shift3: 16,
    },

    16 => SplitMixConstants {
        shift1: 7,
        c1: 0x2c1b_u16 as usize,
        shift2: 9,
        c2: 0x297a_u16 as usize,
        shift3: 7,
    },

    _ => panic!("usize::BITS is a power of 2; that is at least 16 bits"),
};

#[inline(always)]
fn reservation_index(ptr: HostPointer) -> usize {
    let mut x = ptr.0.addr().get();
    x ^= x >> SPLIT_MIX_CONSTANTS.shift1;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c1);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift2;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c2);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift3;
    x.strict_rem(usize::from(BUCKET_COUNT))
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Version(u64);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum ExclusiveMonitorLoad {
    U128(u128),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

pub(crate) struct ReservationToken {
    version: Version,
    value: ExclusiveMonitorLoad,
}

pub(crate) struct ReservationSlot {
    ptr: Option<HostPointer>,
    version: Version,
}

pub(crate) struct ExclusiveMonitor {
    reservations: [CachePadded<Mutex<ReservationSlot>>; BUCKET_COUNT as usize],
}

impl ExclusiveMonitor {
    /// #
    pub fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            let ptr = this.as_mut_ptr();
            for i in 0..BUCKET_COUNT {
                std::ptr::write(
                    &raw mut (*ptr).reservations[usize::from(i)],
                    CachePadded::new(Mutex::new(ReservationSlot {
                        ptr: None,
                        version: Version(0),
                    })),
                )
            }

            this.assume_init_mut()
        }
    }

    pub fn new_arc() -> Arc<Self> {
        let mut uninit = Arc::new_uninit();
        Self::init(Arc::get_mut(&mut uninit).unwrap());
        unsafe { uninit.assume_init() }
    }

    #[must_use]
    pub(crate) fn ldrex(
        &self,
        ptr: HostPointer,
        load_op: impl FnOnce() -> ExclusiveMonitorLoad,
    ) -> ReservationToken {
        let reserve_idx = reservation_index(ptr);
        let mut lock = self.reservations[reserve_idx].lock();

        lock.ptr = Some(ptr);
        let version = lock.version;

        let value = load_op();
        ReservationToken { version, value }
    }

    pub(crate) fn strex<T>(
        &self,
        ptr: HostPointer,
        tok: ReservationToken,
        store_op: impl FnOnce(ExclusiveMonitorLoad) -> Result<T, ()>,
    ) -> Result<T, ()> {
        let reserve_idx = reservation_index(ptr);
        let mut lock = self.reservations[reserve_idx].lock();

        if lock.ptr != Some(ptr) || lock.version != tok.version {
            return Err(());
        }

        // Wrapping is acceptable here: token reuse would require 2^64 successful
        // invalidations of the same reservation slot before an old token could match again.
        // and there aren't any better alternatives
        lock.version.0 = lock.version.0.wrapping_add(1);

        store_op(tok.value)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct CpuFabric(Arc<ExclusiveMonitor>);

impl CpuFabric {
    pub fn new() -> Self {
        Self(ExclusiveMonitor::new_arc())
    }
}

impl Default for CpuFabric {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for CpuFabric {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CpuFabric {}

const _: () = {
    const fn is_sync<T: Sync>() {}
    const fn is_send<T: Send>() {}

    is_sync::<CpuFabric>();
    is_send::<CpuFabric>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZero;
    use std::ptr::NonNull;

    use loom::sync::{Arc, Condvar, Mutex};

    struct BarrierState {
        count: usize,
        generation_id: usize,
    }

    // port of std::sync::Barrier
    struct Barrier {
        num_threads: usize,
        state: Mutex<BarrierState>,
        cond: Condvar,
    }

    impl Barrier {
        fn new(n: usize) -> Self {
            Self {
                num_threads: n,
                state: Mutex::new(BarrierState {
                    count: 0,
                    generation_id: 0,
                }),
                cond: Condvar::new(),
            }
        }

        fn wait(&self) {
            let mut lock = self.state.lock().unwrap();
            let local_gen = lock.generation_id;
            lock.count = lock.count.strict_add(1);
            if lock.count < self.num_threads {
                while local_gen == lock.generation_id {
                    lock = self.cond.wait(lock).unwrap();
                }
            } else {
                lock.count = 0;
                lock.generation_id = lock.generation_id.wrapping_add(1);
                self.cond.notify_all();
            }
        }
    }

    #[test]
    fn test_exclusive_monitor() {
        if cfg!(miri) {
            return;
        }

        loom::model(move || {
            let monitor = Arc::from_std(ExclusiveMonitor::new_arc());
            let memory = Arc::new(Mutex::new(0_u32));
            let barrier = Arc::new(Barrier::new(2));

            let thread_run = || {
                let memory = Arc::clone(&memory);
                let monitor = Arc::clone(&monitor);
                let barrier = Arc::clone(&barrier);
                loom::thread::spawn(move || {
                    let ptr = const {
                        HostPointer::new(NonNull::without_provenance(
                            NonZero::new(0x10000DEAD00BEEF).unwrap(),
                        ))
                    };

                    let token = monitor.ldrex(ptr, || ExclusiveMonitorLoad::U8(0));
                    barrier.wait();
                    let _ = monitor.strex(ptr, token, |val| match val {
                        ExclusiveMonitorLoad::U8(0) => {
                            *memory.try_lock().unwrap() += 1;
                            Ok(())
                        }
                        _ => Err(()),
                    });
                })
            };

            let jh1 = thread_run();
            let jh2 = thread_run();

            jh1.join().unwrap();
            jh2.join().unwrap();

            assert_eq!(*memory.try_lock().unwrap(), 1);
        });
    }
}
