//! FIXME/SOUNDNESS:
//! This memory backend relies on mixed-size atomic accesses that are currently
//! UB under Rust's memory model, although current rustc/LLVM codegen preserves
//! the intended hardware behavior on supported targets as of rust 1.95.0.
//!
//! This is accepted for the initial 0.0.x emulator backend. The implementation
//! is intentionally isolated in `memops` so it can be replaced by a sound
//! backend if Rust does not legalize mixed-size atomics.

#![allow(
    clippy::cast_possible_truncation,
    reason = "all truncation here is actually handled,\
                  it used to split up one large integer to a bunch of smaller ones"
)]
#![allow(
    clippy::missing_safety_doc,
    reason = "this module is in progress, \
              and if there will be documentation it would be oon the module not on the memops"
)]

use std::hint::cold_path;
use std::sync::atomic::AtomicU8;

cfg_select! {
    any(miri, true) => {
        use crossbeam_utils::CachePadded;
        use parking_lot::RwLock;
        use std::mem::MaybeUninit;

        // use Mersenne prime so that collisions are rarer
        const SHARDED_LOCK_COUNT: usize = 127;

        static ADDRESS_LOCKS: [CachePadded<RwLock<()>>; SHARDED_LOCK_COUNT] = {
            [const { CachePadded::new(RwLock::new(())) }; SHARDED_LOCK_COUNT]
        };

        const CACHE_LINE: usize = 64;
        const _: () = assert!(CACHE_LINE.is_power_of_two());
        const CACHE_LINE_SHIFT: u32 = CACHE_LINE.ilog2();

        #[track_caller]
        const fn assert_can_load_and_store<T: Sized>() {
            let fits_in_cache = 1 <= size_of::<T>() && size_of::<T>() <= CACHE_LINE;
            assert!(fits_in_cache && align_of::<T>() <= size_of::<T>())
        }

        // this actually works since all reads are truly aligned
        // and because all of our atomic operations are smaller than this simulated cache_line.
        #[inline]
        unsafe fn get_cache_line_aligned<T>(ptr: *const AtomicU8) -> &'static RwLock<()> {
            const { assert_can_load_and_store::<T>() }
            unsafe { core::hint::assert_unchecked(ptr.addr().is_multiple_of(size_of::<T>())) }

            let cache_line = ptr.addr() >> CACHE_LINE_SHIFT;
            unsafe { ADDRESS_LOCKS.get_unchecked(cache_line % SHARDED_LOCK_COUNT) }
        }

        unsafe fn read_aligned<T>(ptr: *const AtomicU8) -> T {
            let lock = (unsafe { get_cache_line_aligned::<T>(ptr) }).read();
            let ret = unsafe { std::ptr::read(ptr.cast::<T>()) };
            drop(lock);
            ret
        }

        unsafe fn write_aligned<T>(ptr: *const AtomicU8, value: T) {
            let lock = (unsafe { get_cache_line_aligned::<T>(ptr) }).write();
            unsafe { std::ptr::write(ptr.cast_mut().cast::<T>(), value) }
            drop(lock);
        }

        // We use up to two locks for unaligned access here, and a variable number of
        // locks in `bulk_mem_op`'s bulk-copy walk. These must agree on ONE global lock
        // acquisition order, or this function's deadlock-freedom proof doesn't transfer
        // to the cross-function case -- two call sites can each be individually
        // deadlock-free in isolation and still deadlock against *each other* if they
        // pick different orderings over the same shard set.
        //
        // The global order chosen here: always acquire the numerically SMALLER shard
        // index before the LARGER one. A deadlock needs a cycle in the wait-for graph,
        // which requires some thread to acquire a higher index after a lower one. If
        // every site always acquires in increasing index order, that never happens, so
        // no cycle can form -- the standard global lock-ordering argument.
        //
        // IMPORTANT: `bulk_mem_op`'s walk over `shards_start..=shards_end` visits
        // cache *lines* in ascending order, which is ascending in *shard index* only
        // while the walk doesn't cross the COUNT-1 -> 0 wraparound boundary. A walk
        // that does cross it visits shard COUNT-1 before shard 0, which is descending
        // in shard-index terms at that one boundary -- the same hazard this function
        // has to handle below. `bulk_mem_op` must be fixed to lock by shard index
        // (e.g., by walking the line range twice, or pre-sorting touched shards)
        // rather than assuming line order implies shard-index order. This function's
        // correctness depends on that fix landing too; one site obeying this comment
        // and the other not is exactly how the deadlock reappears.
        //
        // Why "cache-line order" (i.e., first cache line's shard, then second cache
        // line's shard, ignoring index value) is NOT safe on its own: shard_i and
        // shard_j are always exactly (n % COUNT, (n+1) % COUNT) for starting cache
        // line n, i.e., adjacent mod COUNT. Locking "first line's shard, then second
        // line's shard" for every n produces the edge set {(0,1), (1,2), ...,
        // (COUNT-1,0)} -- a Hamiltonian cycle around the ring of shards. This isn't
        // just a theoretical concern between two accesses -- with enough concurrent
        // threads (and we may have thousands of OS threads hammering this table),
        // it's a real, reachable deadlock:
        //
        //   - Suppose COUNT threads each do an unaligned access, thread k spanning
        //     cache lines k and k+1 (mod COUNT), for every k from 0 to COUNT-1, all
        //     at once.
        //   - Each thread's FIRST lock is shard k -- all `COUNT` of these are
        //     distinct shards, so there's no contention yet, and every thread
        //     acquires its first lock immediately, regardless of scheduling order.
        //     No adversarial timing is required.
        //   - Now every thread tries to acquire its SECOND lock: shard (k+1) % COUNT.
        //     But that shard is already held by thread (k+1) % COUNT, which is itself
        //     blocked on shard (k+2) % COUNT, and so on around the ring.
        //   - Thread 0 waits on thread 1, thread 1 waits on thread 2, ..., thread
        //     COUNT-1 waits on thread 0. Every thread is blocked holding a resource
        //     another blocked thread needs. Nothing can ever release, because release
        //     only happens after the second lock is acquired -- permanent deadlock,
        //     not a transient stall.
        //
        // The fix costs one comparison, not a general sort: since the two shards
        // always differ by exactly 1 mod COUNT, ascending order is just
        // (shard_i, shard_j) directly, except at the single wraparound start line
        // n = COUNT - 1, where the pair (COUNT - 1, 0) must be locked as
        // (0, COUNT - 1) instead -- i.e., always lock shard 0 first there, not
        // whichever of the pair came from the lower line number.
        #[inline]
        unsafe fn get_cache_line_unaligned<T>(
            ptr: *const AtomicU8
        ) -> (&'static RwLock<()>, Option<&'static RwLock<()>>) {
            use std::num::NonZero;

            const { assert_can_load_and_store::<T>() }

            let Some(end_offset) = (const { NonZero::new(size_of::<T>().strict_sub(1)) }) else {
                return (unsafe { get_cache_line_aligned::<T>(ptr) }, None)
            };

            let i = ptr.addr() >> CACHE_LINE_SHIFT;
            let j = (unsafe { ptr.byte_add(end_offset.get()) }).addr() >> CACHE_LINE_SHIFT;
            let sorted = match usize::cmp(&i, &j) {
                std::cmp::Ordering::Equal => {
                    let lock = unsafe {
                        ADDRESS_LOCKS.get_unchecked(i % SHARDED_LOCK_COUNT)
                    };
                    return (lock, None)
                },
                // lock the smaller shard index first, in both cases
                std::cmp::Ordering::Less => [i, j],
                std::cmp::Ordering::Greater => [j, i],
            };

            cold_path();

            let [first, second] = sorted.map(|idx| {
                unsafe { ADDRESS_LOCKS.get_unchecked(idx % SHARDED_LOCK_COUNT) }
            });

            (first, Some(second))
    }

        unsafe fn read_unaligned<T>(ptr: *const AtomicU8) -> T {
            let (a, b) = unsafe { get_cache_line_unaligned::<T>(ptr) };
            let lock_a = a.read();
            let lock_b = b.map(|lock| lock.read());
            let ret = unsafe { std::ptr::read_unaligned(ptr.cast::<T>()) };
            drop(lock_a);
            drop(lock_b);
            ret
        }

        unsafe fn write_unaligned<T>(ptr: *const AtomicU8, value: T) {
            let (a, b) = unsafe { get_cache_line_unaligned::<T>(ptr) };
            let lock_a = a.write();
            let lock_b = b.map(|lock| lock.write());
            unsafe { std::ptr::write_unaligned(ptr.cast_mut().cast::<T>(), value) };
            drop(lock_a);
            drop(lock_b);
        }

        macro_rules! make_load_store_inner {
            ($($bits: tt)*) => {
                pastey::paste! {$(
                    #[inline]
                    unsafe fn [<load $bits _ne_aligned_inner>](ptr: *const AtomicU8) -> [<u $bits>] {
                        unsafe { read_aligned::<[<u $bits>]>(ptr) }
                    }

                    #[inline]
                    unsafe fn [<store $bits _ne_aligned_inner>](ptr: *const AtomicU8, value: [<u $bits>]) {
                        unsafe { write_aligned::<[<u $bits>]>(ptr, value) }
                    }

                    #[inline(always)]
                    unsafe fn [<load $bits _le_inner>](ptr: *const AtomicU8) -> [<u $bits>] {
                        (unsafe { read_unaligned::<[<u $bits>]>(ptr) }).to_le()
                    }

                    #[inline(always)]
                    unsafe fn [<store $bits _le_inner>](ptr: *const AtomicU8, value: [<u $bits>]) {
                        unsafe { write_unaligned::<[<u $bits>]>(ptr, value.to_le()) }
                    }
                )*}
            };
        }

        #[inline]
        unsafe fn load8(ptr: *const AtomicU8) -> u8 {
            unsafe { read_aligned::<u8>(ptr) }
        }

        #[inline]
        unsafe fn store8(ptr: *const AtomicU8, value: u8) {
            unsafe { write_aligned::<u8>(ptr, value) }
        }

        /// # Safety
        /// must not cause UB when used as an argument for bulk memory
        unsafe trait BulkMemOp {
            type Lock<'a>;

            fn get_lock(shard: &RwLock<()>) -> Self::Lock<'_>;

            fn src_ptr(vm_ptr: *const AtomicU8, host_ptr: *const u8) -> *const u8;

            fn dst_ptr(vm_ptr: *mut AtomicU8, host_ptr: *mut u8) -> *mut u8;
        }

        enum VmToHost {}

        unsafe impl BulkMemOp for VmToHost {
            type Lock<'a> = parking_lot::RwLockReadGuard<'a, ()>;

            fn get_lock(shard: &RwLock<()>) -> Self::Lock<'_> {
                shard.read()
            }

            #[inline(always)]
            fn src_ptr(vm_ptr: *const AtomicU8, host_ptr: *const u8) -> *const u8 {
                let _dst = host_ptr;
                vm_ptr.cast::<u8>()
            }

            #[inline(always)]
            fn dst_ptr(vm_ptr: *mut AtomicU8, host_ptr: *mut u8) -> *mut u8 {
                let _src = vm_ptr;
                host_ptr
            }
        }

        enum HostToVm {}

        unsafe impl BulkMemOp for HostToVm {
            type Lock<'a> = parking_lot::RwLockWriteGuard<'a, ()>;

            fn get_lock(shard: &RwLock<()>) -> Self::Lock<'_> {
                shard.write()
            }

            #[inline(always)]
            fn src_ptr(vm_ptr: *const AtomicU8, host_ptr: *const u8) -> *const u8 {
                let _dst = vm_ptr;
                host_ptr
            }

            #[inline(always)]
            fn dst_ptr(vm_ptr: *mut AtomicU8, host_ptr: *mut u8) -> *mut u8 {
                let _src = host_ptr;
                vm_ptr.cast::<u8>()
            }
        }

        /// Locks every distinct shard covering cache lines `[cacheline_start, cacheline_end]`
        /// (inclusive), in ascending shard-index order, runs the copy, then unlocks
        /// everything.
        ///
        /// The ascending shard-index order is not optional: it MUST be the same global
        /// lock-acquisition order used by `get_cache_line_unaligned` above. Both sites
        /// touch the same shard table, so if they disagreed on ordering, they could form a
        /// wait-for cycle *across* the two functions even though each is deadlock-free in
        /// isolation. "Smaller shard index first, always" is that shared order.
        ///
        /// Two things make naive "walk the cache lines in order" wrong, and both are
        /// handled below:
        ///
        ///   - Saturation: if the copy spans `>= SHARDED_LOCK_COUNT` cache lines, every
        ///     shard is covered and `line % COUNT` repeats. Walking lines would try to
        ///     lock the same shard twice -- a self-deadlock (parking_lot is not reentrant)
        ///     and a double-init of one storage slot. So we collapse to "lock all shards,
        ///     0..COUNT" in that case.
        ///   - Wraparound: when the run crosses a multiple of COUNT (so `last_shard
        ///     first_shard`), the touched shards are `{0..=last_shard}` followed in *line
        ///     order* by `{first_shard..=COUNT-1}` -- i.e., line order goes high, drops to
        ///     0, climbs again, which is the exact high-then-low pattern that breaks the
        ///     ordering invariant. We instead lock the low group first, then the high
        ///     group, each internally ascending; since `last_shard < first_shard` the two
        ///     groups chained together are globally ascending.
        ///
        /// (One could also model the wrap case as two `ShardsGuardDropGuard`s over disjoint
        /// `split_at_mut` halves; storing guards sequentially in acquisition order instead
        /// keeps it to a single guard and avoids per-shard index arithmetic in the storage.)
        #[inline]
        unsafe fn bulk_mem_op<Op: BulkMemOp>(
            vm_ptr: *const AtomicU8,
            host_ptr: *mut u8,
            count: usize,
        ) {
            let Some(end_offset) = count.checked_sub(1) else {
                return;
            };

            let cacheline_start: usize = vm_ptr.addr() >> CACHE_LINE_SHIFT;
            let cacheline_end: usize = {
                (unsafe { vm_ptr.byte_add(end_offset) }).addr() >> CACHE_LINE_SHIFT
            };

            // cacheline_end >= cacheline_start, so this is >= 1 and never underflows.
            // lines <= count <= isize::MAX <= usize::MAX, and so this never overflows
            let lines = unsafe {
                cacheline_end.unchecked_sub(cacheline_start).unchecked_add(1)
            };


            // Build the ascending lock plan as up to two ascending, value-disjoint ranges
            // [a_start, a_end) then [b_start, b_end), chained. The chain is globally
            // ascending in shard index in every case:
            let ranges = match lines >= SHARDED_LOCK_COUNT {
                // Saturated: all shards touched. Lock 0..COUNT ascending.
                true => [0..SHARDED_LOCK_COUNT, 0..0],
                false => {
                    let first_shard = cacheline_start % SHARDED_LOCK_COUNT;
                    let last_shard = cacheline_end % SHARDED_LOCK_COUNT;
                    // last_shard < SHARDED_LOCK_COUNT <= usize::MAX
                    let last_shard_idx = unsafe { last_shard.unchecked_add(1) };

                    match first_shard <= last_shard {
                        // No wrap: a single ascending run.
                        true => [first_shard..last_shard_idx, 0..0],
                        // Wrapped: low group {0..=last_shard} strictly precedes high group
                        // {first_shard..=COUNT-1} in value, so locking low-then-high stays
                        // globally ascending.
                        false => [0..last_shard_idx, first_shard..SHARDED_LOCK_COUNT]
                    }
                }
            };



            // Guards are stored contiguously at indices 0..initialized, in acquisition
            // order. On `drop` (including unwind mid-acquisition) we release exactly the
            // prefix we managed to take; release order is irrelevant.
            struct ShardsGuardDropGuard<'a, T> {
                shard_guards: &'a mut [MaybeUninit<T>],
                initialized: usize,
            }

            impl<T> Drop for ShardsGuardDropGuard<'_, T> {
                fn drop(&mut self) {
                    let initialized = self.initialized;
                    unsafe { std::hint::assert_unchecked(initialized <= SHARDED_LOCK_COUNT) }
                    for i in 0..initialized {
                        unsafe { self.shard_guards.get_unchecked_mut(i).assume_init_drop() }
                    }
                }
            }

            impl<T> ShardsGuardDropGuard<'_, T> {
                unsafe fn push(&mut self, value: T) {
                    unsafe {
                        let n = self.initialized;
                        self.shard_guards.get_unchecked_mut(n).write(value);
                        self.initialized = n.unchecked_add(1);
                    }
                }
            }

            let mut shard_guards_storage: [MaybeUninit<Op::Lock<'static>>; SHARDED_LOCK_COUNT]
                = const { [const { MaybeUninit::uninit() }; SHARDED_LOCK_COUNT] };


            let [range_a, range_b] = ranges.map(std::range::Range::from);
            let [slice_a, slice_b]
                = unsafe { shard_guards_storage.get_disjoint_unchecked_mut([range_a, range_b]) };

            let to_lock = [(slice_a, range_a), (slice_b, range_b)];
            let guards = to_lock.map(|(slice, range)| {
                let mut drop_guard = ShardsGuardDropGuard::<Op::Lock<'static>> {
                    shard_guards: slice,
                    initialized: 0,
                };

                for shard_idx in range {
                    let shard: &'static RwLock<()> = unsafe { ADDRESS_LOCKS.get_unchecked(shard_idx) };
                    let lock = Op::get_lock(shard);
                    unsafe { drop_guard.push(lock) };
                }

                drop_guard
            });

            let src = Op::src_ptr(vm_ptr, host_ptr);
            let dst = Op::dst_ptr(vm_ptr.cast_mut(), host_ptr);

            // All covering shards are held; do the whole copy in one shot under the locks.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, count) }

            drop(guards)
        }

        #[inline(never)]
        pub(crate) unsafe fn copy_non_overlapping_vm_to_host_inner(
            src: *const AtomicU8,
            dst: *mut u8,
            count: usize,
        ) {
            unsafe { bulk_mem_op::<VmToHost>(src, dst, count) }
        }

        #[inline(never)]
        pub(crate) unsafe fn copy_non_overlapping_host_to_vm_inner(
            src: *const u8,
            dst: *const AtomicU8,
            count: usize,
        ) {
            unsafe { bulk_mem_op::<HostToVm>(dst, src.cast_mut(), count) }
        }
    }

    _ => {
        use std::sync::atomic::Ordering;
        use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64};

        macro_rules! make_load_store_inner {
            ($($bits: tt)*) => {
                pastey::paste! {$(
                    #[inline(always)]
                    unsafe fn [<load $bits _ne_aligned_inner>](ptr: *const AtomicU8) -> [<u $bits>] {
                        unsafe { (*ptr.cast::<[<AtomicU $bits>]>()).load(Ordering::Relaxed) }
                    }

                    #[inline(always)]
                    unsafe fn [<store $bits _ne_aligned_inner>](ptr: *const AtomicU8, value: [<u $bits>]) {
                        unsafe { (*ptr.cast::<[<AtomicU $bits>]>()).store(value, Ordering::Relaxed) }
                    }
                )*}
            };
        }

        // Unaligned 64-bit little-endian access patterns:
        //
        // addr % 8 == 4:
        //   alignment:
        //     addr % 4 == 0
        //     addr + 4 is aligned to 8 and still aligned to 4
        //   layout:
        //     ptr: b0 b1 b2 b3 b4 b5 b6 b7
        //          └───lo────┘ └────hi───┘
        //     value = (hi << 32) | lo
        //   accesses exactly:
        //     [ptr + 0, ptr + 4)
        //     [ptr + 4, ptr + 8)
        //
        // addr % 8 == 2 or 6:
        //   alignment:
        //     addr is aligned to 2
        //     addr + 2 is aligned to 4
        //     addr + 6 is aligned to 2
        //   layout:
        //     ptr: b0 b1 b2 b3 b4 b5 b6 b7
        //          └w0─┘ └───mid───┘ └w3─┘
        //     value = (hi << 48) | (mid << 16) | lo
        //   accesses exactly:
        //     [ptr + 0, ptr + 2)
        //     [ptr + 2, ptr + 6)
        //     [ptr + 6, ptr + 8)
        //
        // addr % 8 == 1 or 5:
        //   alignment:
        //     addr is not aligned to 2
        //     addr + 1 is aligned to 2
        //     addr + 3 is aligned to 4
        //     addr + 7 is loaded/stored as one byte
        //   layout:
        //     ptr: b0 b1 b2 b3 b4 b5 b6 b7
        //          │  └w1─┘ └───mid───┘  │
        //          b0                    b7
        //     value = (b7 << 56) | (mid << 24) | (w1 << 8) | b0
        //   accesses exactly:
        //     [ptr + 0, ptr + 1)
        //     [ptr + 1, ptr + 3)
        //     [ptr + 3, ptr + 7)
        //     [ptr + 7, ptr + 8)
        //
        // addr % 8 == 3 or 7:
        //   alignment:
        //     addr is not aligned to 2
        //     addr + 1 is aligned to 4
        //     addr + 5 is aligned to 2
        //     addr + 7 is loaded/stored as one byte
        //   layout:
        //     ptr: b0 b1 b2 b3 b4 b5 b6 b7
        //          │  └───mid───┘ └w5─┘  │
        //          b0                    b7
        //     value = (b7 << 56) | (w5 << 40) | (mid << 8) | b0
        //   accesses exactly:
        //     [ptr + 0, ptr + 1)
        //     [ptr + 1, ptr + 5)
        //     [ptr + 5, ptr + 7)
        //     [ptr + 7, ptr + 8)

        #[inline(always)]
        unsafe fn load64_le_inner(ptr: *const AtomicU8) -> u64 {
            const { assert!(align_of::<AtomicU64>() == 8) }

            match ptr.addr() % 8 {
                0 => unsafe { load64_le_aligned(ptr) },

                4 => unsafe {
                    cold_path();

                    let lo = load32_le_aligned(ptr) as u64;
                    let hi = load32_le_aligned(ptr.byte_add(4)) as u64;

                    (hi << 32) | lo
                },

                2 | 6 => unsafe {
                    cold_path();

                    let lo = load16_le_aligned(ptr) as u64;
                    let mid = load32_le_aligned(ptr.byte_add(2)) as u64;
                    let hi = load16_le_aligned(ptr.byte_add(6)) as u64;

                    (hi << 48) | (mid << 16) | lo
                },

                1 | 5 => unsafe {
                    cold_path();

                    let b0 = load8(ptr) as u64;
                    let w1 = load16_le_aligned(ptr.byte_add(1)) as u64;
                    let mid = load32_le_aligned(ptr.byte_add(3)) as u64;
                    let b7 = load8(ptr.byte_add(7)) as u64;

                    (b7 << 56) | (mid << 24) | (w1 << 8) | b0
                },

                3 | 7 => unsafe {
                    cold_path();

                    let b0 = load8(ptr) as u64;
                    let mid = load32_le_aligned(ptr.byte_add(1)) as u64;
                    let w5 = load16_le_aligned(ptr.byte_add(5)) as u64;
                    let b7 = load8(ptr.byte_add(7)) as u64;

                    (b7 << 56) | (w5 << 40) | (mid << 8) | b0
                },
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        #[inline(always)]
        unsafe fn store64_le_inner(ptr: *const AtomicU8, value: u64) {
            const { assert!(align_of::<AtomicU64>() == 8) }

            match ptr.addr() % 8 {
                0 => unsafe { store64_le_aligned(ptr, value) },

                4 => unsafe {
                    cold_path();

                    let lo = value as u32;
                    let hi = (value >> 32) as u32;

                    store32_le_aligned(ptr, lo);
                    store32_le_aligned(ptr.byte_add(4), hi);
                },

                2 | 6 => unsafe {
                    cold_path();

                    let lo = value as u16;
                    let mid = (value >> 16) as u32;
                    let hi = (value >> 48) as u16;

                    store16_le_aligned(ptr, lo);
                    store32_le_aligned(ptr.byte_add(2), mid);
                    store16_le_aligned(ptr.byte_add(6), hi);
                },

                1 | 5 => unsafe {
                    cold_path();

                    let b0 = value as u8;
                    let w1 = (value >> 8) as u16;
                    let mid = (value >> 24) as u32;
                    let b7 = (value >> 56) as u8;

                    store8(ptr, b0);
                    store16_le_aligned(ptr.byte_add(1), w1);
                    store32_le_aligned(ptr.byte_add(3), mid);
                    store8(ptr.byte_add(7), b7);
                },

                3 | 7 => unsafe {
                    cold_path();

                    let b0 = value as u8;
                    let mid = (value >> 8) as u32;
                    let w5 = (value >> 40) as u16;
                    let b7 = (value >> 56) as u8;

                    store8(ptr, b0);
                    store32_le_aligned(ptr.byte_add(1), mid);
                    store16_le_aligned(ptr.byte_add(5), w5);
                    store8(ptr.byte_add(7), b7);
                },

                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        // Unaligned 32-bit little-endian access patterns:
        //
        // addr % 4 == 2:
        //   alignment:
        //     addr is aligned to 2
        //     addr + 2 is aligned to 4 and still aligned to 2
        //   layout:
        //     ptr: b0 b1 b2 b3
        //          └lo─┘ └─hi┘
        //     value = (hi << 16) | lo
        //   accesses exactly:
        //     [ptr + 0, ptr + 2)
        //     [ptr + 2, ptr + 4)
        //
        // addr % 4 == 1 or 3:
        //   alignment:
        //     addr is not aligned to 2
        //     addr + 1 is aligned to 2
        //     addr + 3 is loaded/stored as one byte
        //   layout:
        //     ptr: b0 b1 b2 b3
        //          │  └mid┘  │
        //          b0        b3
        //     value = (b3 << 24) | (mid << 8) | b0
        //   accesses exactly:
        //     [ptr + 0, ptr + 1)
        //     [ptr + 1, ptr + 3)
        //     [ptr + 3, ptr + 4)

        #[inline(always)]
        unsafe fn load32_le_inner(ptr: *const AtomicU8) -> u32 {
            const { assert!(align_of::<AtomicU32>() == 4) }

            match ptr.addr() % 4 {
                0 => unsafe { load32_le_aligned(ptr) },

                2 => unsafe {
                    cold_path();

                    let lo = load16_le_aligned(ptr) as u32;
                    let hi = load16_le_aligned(ptr.byte_add(2)) as u32;

                    (hi << 16) | lo
                },

                1 | 3 => unsafe {
                    cold_path();

                    let b0 = load8(ptr) as u32;
                    let mid = load16_le_aligned(ptr.byte_add(1)) as u32;
                    let b3 = load8(ptr.byte_add(3)) as u32;

                    (b3 << 24) | (mid << 8) | b0
                },

                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        #[inline(always)]
        unsafe fn store32_le_inner(ptr: *const AtomicU8, value: u32) {
            const { assert!(align_of::<AtomicU32>() == 4) }

            match ptr.addr() % 4 {
                0 => unsafe { store32_le_aligned(ptr, value) },

                2 => unsafe {
                    cold_path();

                    let lo = value as u16;
                    let hi = (value >> 16) as u16;

                    store16_le_aligned(ptr, lo);
                    store16_le_aligned(ptr.byte_add(2), hi);
                },

                1 | 3 => unsafe {
                    cold_path();

                    let b0 = value as u8;
                    let mid = (value >> 8) as u16;
                    let b3 = (value >> 24) as u8;

                    store8(ptr, b0);
                    store16_le_aligned(ptr.byte_add(1), mid);
                    store8(ptr.byte_add(3), b3);
                },

                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        // Unaligned 16-bit little-endian access pattern:
        //
        // addr % 2 == 1:
        //   alignment:
        //     both bytes are loaded/stored as single bytes, so alignment is always ok
        //   layout:
        //     ptr: b0 b1
        //          │  │
        //          b0 b1
        //     value = (b1 << 8) | b0
        //   accesses exactly:
        //     [ptr + 0, ptr + 1)
        //     [ptr + 1, ptr + 2)

        #[inline(always)]
        unsafe fn load16_le_inner(ptr: *const AtomicU8) -> u16 {
            const { assert!(align_of::<AtomicU16>() == 2) }

            match ptr.addr() % 2 {
                0 => unsafe { load16_le_aligned(ptr) },

                1 => unsafe {
                    cold_path();

                    let b0 = load8(ptr) as u16;
                    let b1 = load8(ptr.byte_add(1)) as u16;

                    (b1 << 8) | b0
                },

                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }

        #[inline(always)]
        unsafe fn store16_le_inner(ptr: *const AtomicU8, value: u16) {
            const { assert!(align_of::<AtomicU16>() == 2) }

            match ptr.addr() % 2 {
                0 => unsafe { store16_le_aligned(ptr, value) },

                1 => unsafe {
                    cold_path();

                    store8(ptr, value as u8);
                    store8(ptr.byte_add(1), (value >> 8) as u8);
                },

                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }


        #[inline(always)]
        unsafe fn load8(ptr: *const AtomicU8) -> u8 {
            const { assert!(align_of::<AtomicU8>() == 1) }

            unsafe { (*ptr).load(Ordering::Relaxed) }
        }

        #[inline(always)]
        unsafe fn store8(ptr: *const AtomicU8, value: u8) {
            const { assert!(align_of::<AtomicU8>() == 1) }

            unsafe { (*ptr).store(value, Ordering::Relaxed) }
        }

        #[inline(always)]
        pub(crate) unsafe fn copy_non_overlapping_vm_to_host_inner(
            src: *const AtomicU8,
            dst: *mut u8,
            count: usize,
        ) {
            for i in 0..count {
                unsafe {
                    let src_ptr = src.add(i);
                    let dst_ptr = dst.add(i);
                    let byte = load8(src_ptr);
                    std::ptr::write(dst_ptr, byte);
                }
            }
        }

        #[inline(always)]
        pub(crate) unsafe fn copy_non_overlapping_host_to_vm_inner(
            src: *const u8,
            dst: *const AtomicU8,
            count: usize,
        ) {
            for i in 0..count {
                unsafe {
                    let src_ptr = src.add(i);
                    let dst_ptr = dst.add(i);
                    let byte = std::ptr::read(src_ptr);
                    store8(dst_ptr, byte)
                }
            }
        }
    }
}

macro_rules! make_load_store {
    ($($bits: tt)*) => {
        make_load_store_inner! { $($bits)* }

        pastey::paste! {$(
            #[inline(always)]
            pub unsafe fn [<load $bits _ne_aligned>](ptr: *const AtomicU8) -> [<u $bits>] {
                unsafe { [<load $bits _ne_aligned_inner>](ptr) }
            }

            #[inline(always)]
            pub unsafe fn [<store $bits _ne_aligned>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_aligned_inner>](ptr, value) }
            }

            #[inline(always)]
            pub unsafe fn [<load $bits _le_aligned>](ptr: *const AtomicU8) -> [<u $bits>] {
                (unsafe { [<load $bits _ne_aligned_inner>](ptr) }).to_le()
            }

            #[inline(always)]
            pub unsafe fn [<store $bits _le_aligned>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_aligned_inner>](ptr, value.to_le()) }
            }

            #[inline(always)]
            pub unsafe fn [<load $bits _ne>](ptr: *const AtomicU8) -> [<u $bits>] {
                unsafe { [<load $bits _le_inner>](ptr) }
            }

            #[inline(always)]
            pub unsafe fn [<store $bits _ne>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _le_inner>](ptr, value) }
            }

            #[inline(always)]
            pub unsafe fn [<load $bits _le>](ptr: *const AtomicU8) -> [<u $bits>] {
                unsafe { [<load $bits _le_inner>](ptr) }
            }

            #[inline(always)]
            pub unsafe fn [<store $bits _le>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _le_inner>](ptr, value) }
            }
        )*}
    };
}

make_load_store! { 64 32 16 }

#[inline(always)]
pub unsafe fn load_byte(ptr: *const AtomicU8) -> u8 {
    unsafe { load8(ptr) }
}

#[inline(always)]
pub unsafe fn store_byte(ptr: *const AtomicU8, value: u8) {
    unsafe { store8(ptr, value) }
}

#[inline(always)]
pub unsafe fn copy_non_overlapping_vm_to_host(src: *const AtomicU8, dst: *mut u8, count: usize) {
    unsafe { copy_non_overlapping_vm_to_host_inner(src, dst, count) }
}

#[inline(always)]
pub unsafe fn copy_non_overlapping_host_to_vm(src: *const u8, dst: *const AtomicU8, count: usize) {
    unsafe { copy_non_overlapping_host_to_vm_inner(src, dst, count) }
}
