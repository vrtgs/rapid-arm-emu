//! A global exclusive monitor implementing ARM load-/store-exclusive
//! semantics (`LDXR`/`STXR`) in software.
//!
//! [`ExclusiveMonitor::ldrex`] takes a reservation on a host address and
//! [`ExclusiveMonitor::strex`] only commits its store if that reservation is
//! still intact - i.e., no other reservation was taken on a conflicting
//! address in between. Reservations are tracked in a fixed number of hashed,
//! versioned buckets, so unrelated addresses may alias into the same bucket;
//! that can cause spurious [`ReservationLost`] failures but never a wrongly
//! successful store, which matches the architectural allowance for spurious
//! `STXR` failure.

use crossbeam_utils::CachePadded;
use emu_abi::abort::AbortGuard;
use emu_abi::internal_traits::InitInPlace;
use emu_abi::memory::{CACHE_LINE_SHIFT, CACHE_LINE_SIZE, HostPointer};
use parking_lot::{Mutex, MutexGuard};
use std::mem::MaybeUninit;
use std::num::NonZero;

// use a prime number to reduce collisions
const BUCKET_COUNT: u16 = 257;

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
    // 128-bit+ targets are exotic
    // and never seen in std targets, but if they ever come out,
    // just use the low 64 bits; it should be good enough for now
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
    const { assert!(CACHE_LINE_SIZE >= size_of::<u128>()) }

    let mut x = ptr.0.addr().get() >> CACHE_LINE_SHIFT;
    x ^= x >> SPLIT_MIX_CONSTANTS.shift1;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c1);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift2;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c2);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift3;

    x % const { NonZero::new(BUCKET_COUNT as usize).unwrap() }
}

/// A monotonically increasing version number for a reservation bucket.
///
/// Each successful [`strex`](ExclusiveMonitor::strex) bumps the bucket's
/// version, invalidating every outstanding [`Reservation`] that was
/// taken against the old version.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Version(u128);

impl Version {
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn exhausted<T>() -> T {
        panic!("version number exhausted, did we reach the heat death of the universe yet?")
    }
}

/// The value observed by an exclusive load, tagged with its access width.
///
/// This is what the `load_op` passed to [`ExclusiveMonitor::ldrex`] returns
/// and what the `store_op` passed to [`ExclusiveMonitor::strex`] receives
/// back for its compare step.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum ExclusiveMonitorLoad {
    /// A 128-bit exclusive load (e.g. `LDXP` on two 64-bit registers).
    U128(u128),
    /// A 64-bit exclusive load.
    U64(u64),
    /// A 32-bit exclusive load.
    U32(u32),
    /// A 16-bit exclusive load.
    U16(u16),
    /// An 8-bit exclusive load.
    U8(u8),
}

macro_rules! impl_from {
    ($($name:ident($ty: ty)),+ $(,)?) => {
        $(impl From<$ty> for ExclusiveMonitorLoad {
            fn from(value: $ty) -> Self {
                Self::$name(value)
            }
        })+
    };
}

impl_from!(U128(u128), U64(u64), U32(u32), U16(u16), U8(u8),);

/// Proof of a reservation taken by [`ExclusiveMonitor::ldrex`].
///
/// Pass the version to [`ExclusiveMonitor::strex`] to attempt the matching exclusive store;
/// the store only succeeds if the reservation is still intact.
#[non_exhaustive]
pub struct Reservation {
    /// The bucket's [`Version`] at the time this reservation was taken.
    ///
    /// Passed back to [`ExclusiveMonitor::strex`], which only commits the
    /// store if the bucket's current version still matches this value.
    pub version: Version,
    /// the value observed by the exclusive load which created this token.
    pub value: ExclusiveMonitorLoad,
}

/// The state of one reservation bucket: the last reserved address (if any)
/// and the bucket's current [`Version`].
struct ReservationSlot {
    ptr: Option<HostPointer>,
    version: Version,
}

/// Error returned by [`ExclusiveMonitor::strex`] when the reservation was
/// lost between the exclusive load and the exclusive store, so the store was
/// not performed. This corresponds to a failing `STXR`; the usual response
/// is to retry the load/store-exclusive loop.
#[derive(Debug, Copy, Clone, thiserror::Error)]
#[error("exclusive reservation was lost")]
pub struct ReservationLost;

/// A software exclusive monitor shared by every CPU of a machine.
///
/// See the [module docs](self) for the reservation protocol and its aliasing
/// behavior.
pub struct ExclusiveMonitor {
    reservations: [CachePadded<Mutex<ReservationSlot>>; BUCKET_COUNT as usize],
}

#[inline(always)]
fn cast_maybe_uninit<T>(ptr: *mut T) -> *mut MaybeUninit<T> {
    ptr.cast()
}

unsafe impl InitInPlace for ExclusiveMonitor {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
        unsafe {
            let ptr = this.as_mut_ptr();
            let reservations = cast_maybe_uninit(&raw mut (*ptr).reservations).as_mut_unchecked();

            let guard = AbortGuard(());

            let reservations: &mut [MaybeUninit<_>; BUCKET_COUNT as usize] = reservations.as_mut();

            for reservation in reservations {
                reservation.write(
                    const {
                        CachePadded::new(Mutex::new(ReservationSlot {
                            ptr: None,
                            version: Version(0),
                        }))
                    },
                );
            }

            guard.disarm();

            this.assume_init_mut()
        }
    }
}

impl ExclusiveMonitor {
    fn ptr_lock(&self, ptr: HostPointer) -> MutexGuard<'_, ReservationSlot> {
        let reserve_idx = reservation_index(ptr);
        let lock = unsafe { self.reservations.get_unchecked(reserve_idx) };

        lock.lock()
    }

    /// Performs an exclusive load on `ptr` (the `LDXR` half of the
    /// protocol).
    ///
    /// Takes a reservation on `ptr`'s bucket, invokes `load_op` to read the
    /// current value while the bucket is held, and returns a
    /// [`Reservation`] capturing both the observed value and the
    /// bucket's version. Taking a new reservation displaces any previous
    /// reservation aliasing into the same bucket.
    #[must_use]
    pub fn ldrex<T: Copy + Into<ExclusiveMonitorLoad>>(
        &self,
        ptr: HostPointer,
        load_op: impl FnOnce() -> T,
    ) -> (T, Reservation) {
        let mut lock = self.ptr_lock(ptr);

        lock.ptr = Some(ptr);
        let version = lock.version;

        let value = load_op();
        drop(lock);

        (
            value,
            Reservation {
                version,
                value: value.into(),
            },
        )
    }

    /// Attempts an exclusive store on `ptr` (the `STXR` half of the
    /// protocol).
    ///
    /// If the reservation captured by `tok` is still intact — the bucket
    /// still holds a reservation on `ptr` at the same version — the bucket's
    /// version is bumped and `store_op` is invoked with the originally
    /// loaded value to perform the store. Otherwise [`ReservationLost`] is
    /// returned and `store_op` is never called. `store_op` may itself fail
    /// with [`ReservationLost`] (e.g., after re-checking that the value is
    /// unchanged); its result is returned as-is.
    pub fn strex(
        &self,
        ptr: HostPointer,
        version: Version,
        store_op: impl FnOnce() -> Result<(), ReservationLost>,
    ) -> Result<(), ReservationLost> {
        let mut lock = self.ptr_lock(ptr);

        if lock.ptr != Some(ptr) || lock.version != version {
            return Err(ReservationLost);
        }

        // version overflow would require 2^128 successful
        // invalidations of the same reservation slot
        // before an old token could match again.
        // and there aren't any better alternatives
        lock.version.0 = lock
            .version
            .0
            .checked_add(1)
            .unwrap_or_else(Version::exhausted);

        store_op()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZero;
    use std::ptr::NonNull;

    use crate::cpu_fabric::exclusive_monitor::ExclusiveMonitorLoad;
    cfg_select! {
        miri => { use std::{thread, sync::{Arc, Condvar, Mutex}}; }
        _ => { use loom::{thread, sync::{Arc, Condvar, Mutex}}; }
    }

    struct BarrierState {
        count: usize,
        generation_id: usize,
    }

    // port of std::sync::Barrier that works with loom
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

    #[allow(unused_qualifications)]
    fn new_arc_monitor_inner() -> std::sync::Arc<ExclusiveMonitor> {
        let mut uninit = std::sync::Arc::new_uninit();
        ExclusiveMonitor::init(std::sync::Arc::get_mut(&mut uninit).unwrap());
        unsafe { uninit.assume_init() }
    }

    cfg_select! {
        miri => { use new_arc_monitor_inner as new_arc_monitor; }
        _ => {
            fn new_arc_monitor() -> Arc<ExclusiveMonitor> {
                Arc::from_std(new_arc_monitor_inner())
            }
        }
    }

    #[test]
    fn test_exclusive_monitor() {
        let run = move || {
            let monitor = new_arc_monitor();
            let memory = Arc::new(Mutex::new(0_u32));
            let barrier = Arc::new(Barrier::new(2));

            let thread_run = || {
                let memory = Arc::clone(&memory);
                let monitor = Arc::clone(&monitor);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let ptr = const {
                        HostPointer::new(NonNull::without_provenance(
                            NonZero::new(0x10000DEAD00BEEF).unwrap(),
                        ))
                    };

                    let (_, token) = monitor.ldrex(ptr, || ExclusiveMonitorLoad::U8(0));
                    barrier.wait();
                    let _ = monitor.strex(ptr, token.version, || match token.value {
                        ExclusiveMonitorLoad::U8(0) => {
                            *memory.try_lock().unwrap() += 1;
                            Ok(())
                        }
                        _ => Err(ReservationLost),
                    });
                })
            };

            let jh1 = thread_run();
            let jh2 = thread_run();

            jh1.join().unwrap();
            jh2.join().unwrap();

            assert_eq!(*memory.try_lock().unwrap(), 1);
        };

        cfg_select! {
            miri => run(),
            _ => loom::model(run),
        }
    }
}
