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

use crate::HostPointer;
use crossbeam_utils::CachePadded;
use emu_abi::internal_traits::{CpuFabricPrivate, ICache, InitInPlace};
use emu_abi::memory::{CACHE_LINE_SHIFT, CACHE_LINE_SIZE};
use parking_lot::Mutex;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::Arc;

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
    const { assert!(CACHE_LINE_SIZE >= size_of::<u128>()) }
    let mut x = ptr.0.addr().get() >> CACHE_LINE_SHIFT;
    x ^= x >> SPLIT_MIX_CONSTANTS.shift1;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c1);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift2;
    x = x.wrapping_mul(SPLIT_MIX_CONSTANTS.c2);
    x ^= x >> SPLIT_MIX_CONSTANTS.shift3;
    x.strict_rem(usize::from(BUCKET_COUNT))
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Version(u128);

impl Version {
    #[cold]
    #[inline(never)]
    #[track_caller]
    fn exhausted<T>() -> T {
        panic!("version number exaughsted, did we reach the heat death of the universe yet?")
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum ExclusiveMonitorLoad {
    U128(u128),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

pub struct ReservationToken {
    version: Version,
    value: ExclusiveMonitorLoad,
}

impl ReservationToken {
    pub const fn value(&self) -> ExclusiveMonitorLoad {
        self.value
    }
}

pub struct ReservationSlot {
    ptr: Option<HostPointer>,
    version: Version,
}

#[derive(Debug, Copy, Clone, thiserror::Error)]
#[error("exclusive reservation was lost")]
pub struct ReservationLost;

pub struct ExclusiveMonitor {
    reservations: [CachePadded<Mutex<ReservationSlot>>; BUCKET_COUNT as usize],
}

unsafe impl InitInPlace for ExclusiveMonitor {
    fn init(this: &mut MaybeUninit<Self>) -> &mut Self {
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
}

impl ExclusiveMonitor {
    #[must_use]
    pub fn ldrex(
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

    pub fn strex<T>(
        &self,
        ptr: HostPointer,
        tok: ReservationToken,
        store_op: impl FnOnce(ExclusiveMonitorLoad) -> Result<T, ReservationLost>,
    ) -> Result<T, ReservationLost> {
        let reserve_idx = reservation_index(ptr);
        let mut lock = self.reservations[reserve_idx].lock();

        if lock.ptr != Some(ptr) || lock.version != tok.version {
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

        store_op(tok.value)
    }
}

// dirty page manager
struct CpuFabricInner<T: ?Sized + ICache> {
    monitor: ExclusiveMonitor,
    instruction_cache: T,
}

#[repr(transparent)]
pub struct CpuFabric<T: ?Sized + ICache>(Arc<CpuFabricInner<T>>);

impl<T: Sized + ICache> CpuFabric<T> {
    pub(crate) fn into_dyn<'a>(self) -> CpuFabric<dyn ICache + 'a>
    where
        T: 'a,
    {
        CpuFabric(self.0)
    }
}

impl<T: ?Sized + ICache> Clone for CpuFabric<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: ICache + InitInPlace> Default for CpuFabric<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ICache> CpuFabric<T> {
    pub fn new() -> Self
    where
        T: InitInPlace,
    {
        macro_rules! impl_init_in_place {
            (@{ folded } $($field: ident: $ty: ty),*) => {{
                fn _assert_all_fields_mentioned_and_unique<T: ICache>(inner: &CpuFabricInner<T>) {
                    let CpuFabricInner { $($field),* } = inner;
                    $(let _: &$ty = $field;)*
                }

                struct InitGuard<'a, T>(&'a mut ManuallyDrop<T>);

                impl<T> Drop for InitGuard<'_, T> {
                    fn drop(&mut self) {
                        unsafe { ManuallyDrop::<T>::drop(self.0) }
                    }
                }

                let mut arc: Arc<MaybeUninit<CpuFabricInner<T>>> = Arc::new_uninit();
                let init_mut: &mut MaybeUninit<CpuFabricInner<T>> = Arc::get_mut(&mut arc).unwrap();

                let init_ptr: *mut CpuFabricInner<T> = init_mut.as_mut_ptr();

                $(let $field: InitGuard<$ty> = {
                    let ptr: *mut $ty = unsafe { &raw mut ((*init_ptr).$field) };
                    let maybe_uninit_ref: &mut MaybeUninit<$ty> = unsafe {
                        ptr.cast::<MaybeUninit<$ty>>().as_mut_unchecked()
                    };
                    let init_ref: &mut $ty = <$ty as InitInPlace>::init(maybe_uninit_ref);
                    let manualy_drop: &mut ManuallyDrop<$ty> = unsafe {
                        &mut *(init_ref as *mut $ty as *mut ManuallyDrop<$ty>)
                    };

                    InitGuard(manualy_drop)
                };)*

                $(std::mem::forget($field);)*

                unsafe { arc.assume_init() }
            }};
            () => { impl_init_in_place!(@{ folded }) };
            ($($field: ident : $ty: ty),+ $(,)?) => {
                impl_init_in_place!(@{ folded } $($field: $ty),*)
            }
        }

        let inner = impl_init_in_place! {
            monitor: ExclusiveMonitor,
            instruction_cache: T,
        };

        CpuFabric(inner)
    }
}

impl<T: ICache> PartialEq for CpuFabric<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T: ICache> Eq for CpuFabric<T> {}

const _: () = {
    fn _assert_cpu_fabric_send_sync<T: ICache + Send + Sync>() {
        const fn is_sync<T: Sync>() {}
        const fn is_send<T: Send>() {}

        is_send::<CpuFabric<T>>();
        is_sync::<CpuFabric<T>>();
    }
};

impl<T: ?Sized + ICache> CpuFabricPrivate for CpuFabric<T> {
    type ICache = T;

    fn icache(&self) -> &Self::ICache {
        &self.0.instruction_cache
    }
}

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

    pub fn new_arc_monitor() -> std::sync::Arc<ExclusiveMonitor> {
        let mut uninit = std::sync::Arc::new_uninit();
        ExclusiveMonitor::init(std::sync::Arc::get_mut(&mut uninit).unwrap());
        unsafe { uninit.assume_init() }
    }

    #[test]
    fn test_exclusive_monitor() {
        if cfg!(miri) {
            return;
        }

        loom::model(move || {
            let monitor = Arc::from_std(new_arc_monitor());
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
                        _ => Err(ReservationLost),
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
