//! Atomic-backed access to guest ("VM") memory from the host.
//!
//! Aligned accesses through this module provide single-copy atomicity, in
//! the same sense as AArch64's memory model: a given aligned load or store
//! is guaranteed to be observed by other accessors as a whole, never as a
//! tear of some bytes from one access interleaved with bytes from another.
//! This guarantee does not extend to unaligned accesses, which this module
//! also supports but which decompose internally into multiple aligned
//! sub-accesses -- another accessor can observe those sub-accesses
//! individually. This module does not provide ordering guarantees beyond
//! single-copy atomicity for aligned accesses; it only rules out torn
//! reads/writes at that granularity.
//!
//! # Note
//! Guest memory must be accessed **only** through this module. Reading or
//! writing it by any other means -- casting the backing allocation to a
//! plain `&[u8]`/`&mut [u8]`, or accessing it through any path other than
//! the functions here -- is undefined behavior, full stop. It doesn't matter
//! whether that other access is itself "safe" Rust in isolation, or whether
//! it happens to use atomics of its own: this module makes no guarantee
//! about how it accesses memory internally, so nothing outside it can be
//! synchronized against.
//!
//! FIXME(SOUNDNESS):
//! This memory backend relies on mixed-size atomic accesses that are currently
//! UB under Rust's memory model, although current rustc/LLVM codegen preserves
//! the intended hardware behavior on supported targets as of rust 1.96.1.
//!
//! This is accepted for the initial 0.0.x emulator backend. The implementation
//! is intentionally isolated in `memops` so it can be replaced by a sound
//! backend if Rust does not legalize mixed-size atomics.

#![allow(
    clippy::cast_possible_truncation,
    reason = "all truncation here is actually handled,\
                  it used to split up one large integer to a bunch of smaller ones"
)]

use std::sync::atomic::AtomicU8;

cfg_select! {
    // PERF NOTE: this backend intentionally locks coarser than production for the
    // bulk-copy path (whole covering-shard-range for the duration of the copy,
    // vs. production's per-chunk/per-byte atomicity). Locking per-byte here was
    // measured at ~15000s for a single-byte-copy-loop test; range-locking brings
    // that to ~90s.
    //
    // Consequence 1 (atomicity): under miri, a bulk copy *appears* single-copy
    // atomic over the whole region, which production does NOT guarantee (see
    // `copy_nonoverlapping_{vm_to_host,host_to_vm}` docs' Atomicity section).
    // This is fine for miri's job -- catching UB/soundness bugs in this module's
    // own accesses -- but it means miri is NOT where the byte-level-tearing
    // contract itself gets validated; that's covered by the non-miri test run,
    // which uses the real backend.
    //
    // Consequence 2 (ordering): parking_lot's RwLock::read()/write() (and their
    // drop) carry acquire/release semantics, so under miri every load in this
    // module acts like an `acquire` and every store acts like a `release` -- TSO,
    // effectively. Production uses `Ordering::Relaxed` throughout, matching the
    // module's stated contract (single-copy atomicity only, no ordering
    // guarantee beyond that). So miri is also not where the *absence* of
    // ordering guarantees gets validated -- code that accidentally relies on
    // acquire/release visibility could pass under miri and still be broken
    // against production.
    //
    // If this backend (locking + RwLock) is ever promoted to production, the
    // locks should be switched to `Relaxed`-equivalent behavior -- e.g., by not
    // relying on RwLock's built-in acquire/release and instead fencing
    // explicitly only where the module's contract actually requires it.
    // And that should give us a tangible speed improvement
    // on weaker memory models.
    miri => {
        use crossbeam_utils::CachePadded;
        use parking_lot::RwLock;
        use std::mem::MaybeUninit;

        // use a prime number so that collisions are rarer
        const SHARDED_LOCK_COUNT: usize = 127;

        static ADDRESS_LOCKS: [CachePadded<RwLock<()>>; SHARDED_LOCK_COUNT] = {
            [const { CachePadded::new(RwLock::new(())) }; SHARDED_LOCK_COUNT]
        };

        const CACHE_LINE_SIZE: usize = emu_abi::memory::CACHE_LINE_SIZE;
        const CACHE_LINE_SHIFT: u8 = emu_abi::memory::CACHE_LINE_SHIFT;

        const _: () = assert!(CACHE_LINE_SIZE == (1 << CACHE_LINE_SHIFT));

        #[track_caller]
        const fn assert_can_load_and_store<T: Sized>() {
            let fits_in_cache = 1 <= size_of::<T>() && size_of::<T>() < CACHE_LINE_SIZE;
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
        // `bulk_mem_op` upholds this same invariant: it locks shards in ascending
        // index order too, explicitly handling the two cases where a naive walk over
        // cache *lines* (in line order) would NOT correspond to ascending shard index:
        //   - saturation, where the run covers >= SHARDED_LOCK_COUNT lines and every
        //     shard is touched (it collapses to "lock all shards 0..COUNT" instead of
        //     walking lines, which would double-lock a shard);
        //   - wraparound, where the line range crosses a multiple of COUNT (so
        //     `last_shard < first_shard` in line order); it locks the low group
        //     {0..=last_shard} before the high group {first_shard..=COUNT-1}, which
        //     stays globally ascending even though line order does not.
        // See `bulk_mem_op`'s own doc comment for the full argument. Both sites must
        // keep agreeing on "smaller shard index first, always" -- if one of them ever
        // stops, the two are no longer jointly deadlock-free even though each remains
        // deadlock-free considered alone.
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
        //     only happens after the second lock is acquired -- a permanent deadlock,
        //     not a transient stall.
        //
        // This is exactly the failure mode `bulk_mem_op` avoids by locking on shard
        // index rather than line order; see its comment for the mechanics.
        #[inline]
        unsafe fn get_cache_line_unaligned<T>(
            ptr: *const AtomicU8
        ) -> (&'static RwLock<()>, Option<&'static RwLock<()>>) {
            use std::num::NonZero;

            const { assert_can_load_and_store::<T>() }

            let Some(end_offset) = (const { NonZero::new(size_of::<T>().strict_sub(1)) }) else {
                return (unsafe { get_cache_line_aligned::<T>(ptr) }, None)
            };

            let i_addr = ptr.addr();
            let j_addr = (unsafe { ptr.byte_add(end_offset.get()) }).addr();
            let [i_line, j_line] = [i_addr, j_addr]
                .map(|addr| addr >> CACHE_LINE_SHIFT);

            if i_line == j_line {
                let idx = i_line % SHARDED_LOCK_COUNT;
                let lock = unsafe { ADDRESS_LOCKS.get_unchecked(idx) };
                return (lock, None)
            }

            std::hint::cold_path();

            unsafe { std::hint::assert_unchecked(j_line == i_line.unchecked_add(1)) }

            // we do this to avoid computing the expensive modulo twice for the common case
            let [i, j] = [i_line, j_line].map(|line| line % SHARDED_LOCK_COUNT);

            let sorted = match usize::cmp(&i, &j) {
                // SAFETY: `Ordering::Equal` is unreachable here.
                //
                // Reached only when `i_line != j_line`
                // (since the function would have already returned for the equal-line case).
                // `j_line - i_line` is exactly 1 whenever it's nonzero:
                // `assert_can_load_and_store::<T>()` bounds `size_of::<T>() <=
                // CACHE_LINE_SIZE`, and a byte range that long can cross at most one line
                // boundary -- this function's own return type (one lock, or optionally a
                // second) already assumes no `T` spans more than two lines, so that bound
                // isn't a new assumption introduced here.
                //
                // `i == j` would then require `COUNT | 1`, i.e. `COUNT == 1`, which
                // `const { assert!(SHARDED_LOCK_COUNT > 1) }` rules out at compile time.
                //
                // Both premises are compile-time facts for the concrete `T` and
                // `SHARDED_LOCK_COUNT` in use, and this module's miri suite exercises
                // exactly this path -- a violation here surfaces as a miri UB report, not
                // just a comment going stale silently.
                std::cmp::Ordering::Equal => unsafe { core::hint::unreachable_unchecked() },
                // lock the smaller shard index first, in both cases
                std::cmp::Ordering::Less => [i, j],
                std::cmp::Ordering::Greater => [j, i],
            };



            const { assert!(SHARDED_LOCK_COUNT > 1, "there is no lock sharding") }
            let [first, second] = sorted
                .map(|idx| unsafe { ADDRESS_LOCKS.get_unchecked(idx) });

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
                    unsafe fn [<load $bits _ne_aligned_inner>](ptr: *const AtomicU8) -> [<u $bits>] {
                        unsafe { read_aligned::<[<u $bits>]>(ptr) }
                    }

                    unsafe fn [<store $bits _ne_aligned_inner>](ptr: *const AtomicU8, value: [<u $bits>]) {
                        unsafe { write_aligned::<[<u $bits>]>(ptr, value) }
                    }

                    unsafe fn [<load $bits _ne_inner>](ptr: *const AtomicU8) -> [<u $bits>] {
                        unsafe { read_unaligned::<[<u $bits>]>(ptr) }
                    }

                    unsafe fn [<store $bits _ne_inner>](ptr: *const AtomicU8, value: [<u $bits>]) {
                        unsafe { write_unaligned::<[<u $bits>]>(ptr, value) }
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
        unsafe fn copy_nonoverlapping_vm_to_host_inner(
            src: *const AtomicU8,
            dst: *mut u8,
            count: usize,
        ) {
            unsafe { bulk_mem_op::<VmToHost>(src, dst, count) }
        }

        #[inline(never)]
        unsafe fn copy_nonoverlapping_host_to_vm_inner(
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
        use std::hint::cold_path;
        use std::num::NonZero;

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

        cfg_select! {
            target_endian = "little" => {
                const fn ne_part_shift_inner(offset: usize, _width: usize, _total: usize) -> usize {
                    offset
                }
            }
            target_endian = "big" => {
                const fn ne_part_shift_inner(offset: usize, width: usize, total: usize) -> usize {
                    total.strict_sub(offset).strict_sub(width)
                }
            }
        }


        const fn ne_part_shift<T>(offset: usize, width: usize) -> u32 {
            let total = size_of::<T>();
            let shift = ne_part_shift_inner(offset, width, total);
            assert!(shift < total);
            emu_abi::convert::usize_to_u32(shift.strict_mul(8)).unwrap()
        }


        const fn get_byte_count_from_load_fn<T>(_func: unsafe fn(*const AtomicU8) -> T) -> usize {
            size_of::<T>()
        }

        const fn get_byte_count_from_store_fn<T>(_func: unsafe fn(*const AtomicU8, T)) -> usize {
            size_of::<T>()
        }

        macro_rules! combine_ne_parts {
            ($ptr:expr, $ty:ty, [ $($load_fn:path),+ $(,)? ]) => {
                combine_ne_parts!(@fold $ptr, $ty, off: 0_usize, folded: [], todo: [ $($load_fn),+ ])
            };

            (
                @fold $ptr:expr, $ty:ty,
                off: $off:expr,
                folded: [ $(($f_load:path, $f_width:expr, $f_off:expr)),* ],
                todo: [ $load_fn:path $(, $rest:path)* $(,)? ]
            ) => {
                combine_ne_parts!(
                    @fold $ptr, $ty,
                    off: ($off.strict_add(get_byte_count_from_load_fn($load_fn))),
                    folded: [
                        $(($f_load, $f_width, $f_off),)*
                        ($load_fn, get_byte_count_from_load_fn($load_fn), $off)
                    ],
                    todo: [ $($rest),* ]
                )
            };

            (
                @fold $ptr:expr, $ty:ty,
                off: $off:expr,
                folded: [ $(($f_load:path, $f_width:expr, $f_off:expr)),+ ],
                todo: []
            ) => {{ #[allow(unused_unsafe)] {
                const TOTAL: usize = $off;
                const _: () = assert!(
                    TOTAL == size_of::<$ty>(),
                    "combine_ne_parts!: part widths do not sum to size_of::<$ty>()"
                );

                let ptr: *const AtomicU8 = $ptr;
                $(
                    ({
                        const OFFSET: usize = $f_off;

                        let limb = $f_load(ptr.byte_add(OFFSET)) as $ty;
                        unsafe {
                            limb.unchecked_shl(const { ne_part_shift::<$ty>(OFFSET, $f_width) })
                        }
                    })
                )|+
            }}};
        }

        macro_rules! split_ne_parts {
            ($ptr:expr, $value:expr, $ty:ty, [ $($store_fn:path),+ $(,)? ]) => {
                split_ne_parts!(
                    @fold $ptr, $value, $ty,
                    off: 0_usize,
                    folded: [],
                    todo: [ $($store_fn),+ ]
                )
            };

            (
                @fold $ptr:expr, $value:expr, $ty:ty,
                off: $off:expr,
                folded: [ $(($f_store:path, $f_width:expr, $f_off:expr)),* ],
                todo: [ $store_fn:path $(, $rest:path)* $(,)? ]
            ) => {
                split_ne_parts!(
                    @fold $ptr, $value, $ty,
                    off: ($off.strict_add(get_byte_count_from_store_fn($store_fn))),
                    folded: [
                        $(($f_store, $f_width, $f_off),)*
                        ($store_fn, get_byte_count_from_store_fn($store_fn), $off)
                    ],
                    todo: [ $($rest),* ]
                )
            };

            (
                @fold $ptr:expr, $value:expr, $ty:ty,
                off: $off:expr,
                folded: [ $(($f_store:path, $f_width:expr, $f_off:expr)),+ ],
                todo: []
            ) => {{ #[allow(unused_unsafe)] {
                const TOTAL: usize = $off;
                const _: () = assert!(
                    TOTAL == size_of::<$ty>(),
                    "split_ne_parts!: part widths do not sum to size_of::<$ty>()"
                );

                let ptr: *const AtomicU8 = $ptr;
                let value: $ty = $value;
                $({
                    const OFFSET: usize = $f_off;

                    $f_store(
                        ptr.byte_add(OFFSET),
                        unsafe {
                            let limb = value.unchecked_shr(const {
                                ne_part_shift::<$ty>(OFFSET, $f_width)
                            });

                            limb as _
                        },
                    )
                })+
            }}};
        }

        /// Generates the unaligned native-endian `load`/`store` pair for each listed
        /// type from ONE shared case table, so the two directions cannot drift apart.
        ///
        /// Per type, at compile time this checks:
        ///   - the backing atomic has `size == align == size_of::<$ty>()`
        ///   - every listed remainder is in `1..TOTAL`
        ///   - the remainder count is exactly `TOTAL - 1`
        ///   - (via `#[forbid(unreachable_patterns)]`) no remainder is duplicated
        ///   - each `(load, store)` pair at the same position has the same width
        ///
        /// The second and third checks, plus forbidden duplicates, mean the listed
        /// arms cover `1..TOTAL` exactly once (pigeonhole), `0` is the aligned arm,
        /// and `ptr.addr() % TOTAL` can produce nothing else -- which is what makes
        /// the trailing `unreachable_unchecked()` sound rather than merely
        /// hoped-for. Part offsets and the sum-to-`TOTAL` width check are handled
        /// inside `combine_ne_parts!` / `split_ne_parts!`.
        macro_rules! make_unaligned_ne_access {
            ($(
                $ty:ty {
                    atomic: $atomic:ty,
                    aligned: ($load_aligned:path, $store_aligned:path),
                    fns: ($load_name:ident, $store_name:ident),
                    cases: {
                        $(
                            [ $($pat:literal)|+ ] => [
                                $( ($load_part:path, $store_part:path) ),+ $(,)?
                            ]
                        ),+ $(,)?
                    } $(,)?
                }
            )+) => {$(
                const _: () = {
                    const TOTAL: usize = size_of::<$ty>();

                    assert!(align_of::<$atomic>() == TOTAL && size_of::<$atomic>() == TOTAL);

                    // every remainder is a real unaligned remainder of `addr % TOTAL`
                    let patterns: &[usize] = &[ $($($pat),+),+ ];

                    let mut patterns_iter_slice = patterns;
                    while let &[pattern, ref rest @ ..] = patterns_iter_slice {
                        assert!(pattern != 0 && pattern < TOTAL);
                        patterns_iter_slice = rest;
                    }

                    // ...and there are exactly TOTAL - 1 of them. Together with
                    // forbid(unreachable_patterns) on the fns (no duplicates), the
                    // arms are a bijection onto 1..TOTAL.
                    assert!(patterns.len() == TOTAL.strict_sub(1));

                    // load and store must decompose identically
                    $($(
                        assert!(
                            get_byte_count_from_load_fn($load_part)
                                == get_byte_count_from_store_fn($store_part)
                        );
                    )+)+
                };

                #[inline(always)]
                #[forbid(unreachable_patterns)]
                unsafe fn $load_name(ptr: *const AtomicU8) -> $ty {
                    const TOTAL: usize = size_of::<$ty>();

                    match ptr.addr() % TOTAL {
                        0 => unsafe { $load_aligned(ptr) },
                        $(
                            $($pat)|+ => {
                                cold_path();
                                unsafe {
                                    combine_ne_parts!(ptr, $ty, [ $($load_part),+ ])
                                }
                            }
                        )+
                        // sound: see the const block above
                        _ => unsafe { core::hint::unreachable_unchecked() },
                    }
                }

                #[inline(always)]
                #[forbid(unreachable_patterns)]
                unsafe fn $store_name(ptr: *const AtomicU8, value: $ty) {
                    const TOTAL: usize = size_of::<$ty>();

                    match ptr.addr() % TOTAL {
                        0 => unsafe { $store_aligned(ptr, value) },
                        $(
                            $($pat)|+ => {
                                cold_path();
                                unsafe {
                                    split_ne_parts!(ptr, value, $ty, [ $($store_part),+ ])
                                }
                            }
                        )+
                        // sound: see the const block above
                        _ => unsafe { core::hint::unreachable_unchecked() },
                    }
                }
            )+};
        }

        // Unaligned native-endian access patterns.
        //
        // Each case lists its parts in ADDRESS order (ascending offset); every part
        // is one properly aligned atomic access. Where a part lands in the assembled
        // value is decided by `ne_part_shift`, not by the table:
        //
        //   little-endian: part at offset o, width w -> bits [8*o, 8*(o+w))
        //                  (lower address = less significant)
        //   big-endian:    part at offset o, width w -> bits [8*(T-o-w), 8*(T-o))
        //                  (lower address = more significant)
        //
        // The alignment facts and "accesses exactly" ranges below are
        // endian-independent; only the `value` assembly differs, so both formulas
        // are given. `pN` names the part starting at ptr + N.
        make_unaligned_ne_access! {
            // addr % 8 == 4:
            //   alignment:
            //     addr % 4 == 0
            //     addr + 4 is aligned to 8 and still aligned to 4
            //   layout:
            //     ptr: b0 b1 b2 b3 b4 b5 b6 b7
            //          └───p0────┘ └────p4───┘
            //     value_le = (p4 << 32) | p0
            //     value_be = (p0 << 32) | p4
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
            //          └p0─┘ └────p2───┘ └p6─┘
            //     value_le = (p6 << 48) | (p2 << 16) | p0
            //     value_be = (p0 << 48) | (p2 << 16) | p6
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
            //          │  └p1─┘ └────p3───┘  │
            //          p0                    p7
            //     value_le = (p7 << 56) | (p3 << 24) | (p1 << 8) | p0
            //     value_be = (p0 << 56) | (p1 << 40) | (p3 << 8) | p7
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
            //          │  └────p1───┘ └p5─┘  │
            //          p0                    p7
            //     value_le = (p7 << 56) | (p5 << 40) | (p1 << 8) | p0
            //     value_be = (p0 << 56) | (p1 << 24) | (p5 << 8) | p7
            //   accesses exactly:
            //     [ptr + 0, ptr + 1)
            //     [ptr + 1, ptr + 5)
            //     [ptr + 5, ptr + 7)
            //     [ptr + 7, ptr + 8)
            u64 {
                atomic: AtomicU64,
                aligned: (load64_ne_aligned_inner, store64_ne_aligned_inner),
                fns: (load64_ne_inner, store64_ne_inner),
                cases: {
                    [4] => [
                        (load32_ne_aligned_inner, store32_ne_aligned_inner),
                        (load32_ne_aligned_inner, store32_ne_aligned_inner),
                    ],
                    [2 | 6] => [
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                        (load32_ne_aligned_inner, store32_ne_aligned_inner),
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                    ],
                    [1 | 5] => [
                        (load8, store8),
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                        (load32_ne_aligned_inner, store32_ne_aligned_inner),
                        (load8, store8),
                    ],
                    [3 | 7] => [
                        (load8, store8),
                        (load32_ne_aligned_inner, store32_ne_aligned_inner),
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                        (load8, store8),
                    ],
                }
            }

            // addr % 4 == 2:
            //   alignment:
            //     addr is aligned to 2
            //     addr + 2 is aligned to 4 and still aligned to 2
            //   layout:
            //     ptr: b0 b1 b2 b3
            //          └p0─┘ └p2─┘
            //     value_le = (p2 << 16) | p0
            //     value_be = (p0 << 16) | p2
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
            //          │  └p1─┘  │
            //          p0        p3
            //     value_le = (p3 << 24) | (p1 << 8) | p0
            //     value_be = (p0 << 24) | (p1 << 8) | p3
            //   accesses exactly:
            //     [ptr + 0, ptr + 1)
            //     [ptr + 1, ptr + 3)
            //     [ptr + 3, ptr + 4)
            u32 {
                atomic: AtomicU32,
                aligned: (load32_ne_aligned_inner, store32_ne_aligned_inner),
                fns: (load32_ne_inner, store32_ne_inner),
                cases: {
                    [2] => [
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                    ],
                    [1 | 3] => [
                        (load8, store8),
                        (load16_ne_aligned_inner, store16_ne_aligned_inner),
                        (load8, store8),
                    ],
                }
            }

            // addr % 2 == 1:
            //   alignment:
            //     both bytes are loaded/stored as single bytes, so alignment is always ok
            //   layout:
            //     ptr: b0 b1
            //          │  │
            //          p0 p1
            //     value_le = (p1 << 8) | p0
            //     value_be = (p0 << 8) | p1
            //   accesses exactly:
            //     [ptr + 0, ptr + 1)
            //     [ptr + 1, ptr + 2)
            u16 {
                atomic: AtomicU16,
                aligned: (load16_ne_aligned_inner, store16_ne_aligned_inner),
                fns: (load16_ne_inner, store16_ne_inner),
                cases: {
                    [1] => [
                        (load8, store8),
                        (load8, store8),
                    ],
                }
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


        type BulkMemChunk = u64;
        type BulkMemChunkAtomic = AtomicU64;

        const CHUNK_SIZE: NonZero<usize> = NonZero::new(size_of::<BulkMemChunk>()).unwrap();

        // `align_offset(CHUNK_SIZE)` below must produce alignment sufficient for the
        // atomic chunk access; assert the plain-int and atomic layouts agree so
        // CHUNK_SIZE is valid for both roles (offset stride AND atomic alignment).
        const _: () = assert!(
            size_of::<BulkMemChunkAtomic>() == CHUNK_SIZE.get()
                && align_of::<BulkMemChunkAtomic>() == CHUNK_SIZE.get()
        );

        /// # Safety
        /// `copy_chunk` requires `vm_ptr.add(i)` to be `CHUNK_SIZE`-aligned; both
        /// methods require `vm_ptr.add(i)` / `host_ptr.add(i)` valid for the width
        /// they access, in their respective direction.
        trait BulkByteChunkOp {
            unsafe fn copy_byte(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize);
            unsafe fn copy_chunk(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize);
        }

        enum VmToHost {}

        impl BulkByteChunkOp for VmToHost {
            #[inline(always)]
            unsafe fn copy_byte(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize) {
                unsafe { std::ptr::write::<u8>(host_ptr.add(i), load8(vm_ptr.add(i))) }
            }

            #[inline(always)]
            unsafe fn copy_chunk(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize) {
                unsafe {
                    let chunk: BulkMemChunk = load64_ne_aligned_inner(vm_ptr.add(i));
                    std::ptr::write_unaligned(host_ptr.add(i).cast::<BulkMemChunk>(), chunk)
                }
            }
        }

        enum HostToVm {}

        impl BulkByteChunkOp for HostToVm {
            #[inline(always)]
            unsafe fn copy_byte(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize) {
                unsafe { store8(vm_ptr.add(i), std::ptr::read::<u8>(host_ptr.add(i))) }
            }

            #[inline(always)]
            unsafe fn copy_chunk(vm_ptr: *const AtomicU8, host_ptr: *mut u8, i: usize) {
                unsafe {
                    let chunk = std::ptr::read_unaligned(host_ptr.add(i).cast::<BulkMemChunk>());
                    store64_ne_aligned_inner(vm_ptr.add(i), chunk)
                }
            }
        }

        #[inline(always)]
        unsafe fn bulk_byte_chunk_copy<Op: BulkByteChunkOp>(
            vm_ptr: *const AtomicU8,
            host_ptr: *mut u8,
            count: usize,
        ) {
            unsafe {
                let is_nonoverlapping = vm_ptr.addr().abs_diff(host_ptr.addr()) >= count;
                std::hint::assert_unchecked(is_nonoverlapping);

                let prefix = usize::min(vm_ptr.align_offset(CHUNK_SIZE.get()), count);

                for i in 0..prefix {
                    Op::copy_byte(vm_ptr, host_ptr, i);
                }

                // count >= prefix, so unchecked_sub never underflows;
                // mid_end <= count <= isize::MAX, so the add/mul never overflow.
                let mid_len = count.unchecked_sub(prefix) / CHUNK_SIZE;
                let mid_end = prefix.unchecked_add(mid_len.unchecked_mul(CHUNK_SIZE.get()));

                // mid_len * CHUNK_SIZE <= count - prefix
                // mid_end = prefix + mid_len * CHUNK_SIZE <= count
                std::hint::assert_unchecked(mid_end <= count);

                let mut i = prefix;
                while i < mid_end {
                    Op::copy_chunk(vm_ptr, host_ptr, i);
                    i = i.unchecked_add(CHUNK_SIZE.get());
                }

                for i in mid_end..count {
                    Op::copy_byte(vm_ptr, host_ptr, i);
                }
            }
        }

        #[inline(always)]
        unsafe fn copy_nonoverlapping_vm_to_host_inner(
            src: *const AtomicU8,
            dst: *mut u8,
            count: usize,
        ) {
            unsafe { bulk_byte_chunk_copy::<VmToHost>(src, dst, count) }
        }

        #[inline(always)]
        unsafe fn copy_nonoverlapping_host_to_vm_inner(
            src: *const u8,
            dst: *const AtomicU8,
            count: usize,
        ) {
            unsafe { bulk_byte_chunk_copy::<HostToVm>(dst, src.cast_mut(), count) }
        }
    }
}

macro_rules! make_load_store {
    ($(($bits:tt, $bytes:tt)),* $(,)?) => {
        make_load_store_inner! { $($bits)* }

        pastey::paste! {$(
            const _: () = assert!(
                $bits == $bytes * 8,
                concat!(
                    "make_load_store!: bits/bytes mismatch for u", stringify!($bits),
                    " (expected bytes == ", stringify!($bits), " / 8)"
                ),
            );

            #[doc = concat!(
                "Loads a `u", stringify!($bits), "` (", stringify!($bytes), " bytes) from `ptr` ",
                "in native-endian order.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for reads of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs).\n",
                "- `ptr` must be aligned to `", stringify!($bytes), "` bytes. Use ",
                "[`load", stringify!($bits), "_ne`] if `ptr` may be unaligned.",
            )]
            #[must_use]
            #[inline(always)]
            pub unsafe fn [<load $bits _ne_aligned>](ptr: *const AtomicU8) -> [<u $bits>] {
                unsafe { [<load $bits _ne_aligned_inner>](ptr) }
            }

            #[doc = concat!(
                "Stores `value` to `ptr` (", stringify!($bytes), " bytes) in native-endian ",
                "order.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for writes of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs).\n",
                "- `ptr` must be aligned to `", stringify!($bytes), "` bytes. Use ",
                "[`store", stringify!($bits), "_ne`] if `ptr` may be unaligned.",
            )]
            #[inline(always)]
            pub unsafe fn [<store $bits _ne_aligned>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_aligned_inner>](ptr, value) }
            }

            #[doc = concat!(
                "Loads a `u", stringify!($bits), "` (", stringify!($bytes), " bytes) from `ptr`, ",
                "interpreting the in-memory bytes as little-endian.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for reads of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs).\n",
                "- `ptr` must be aligned to `", stringify!($bytes), "` bytes. Use ",
                "[`load", stringify!($bits), "_le`] if `ptr` may be unaligned.",
            )]
            #[must_use]
            #[inline(always)]
            pub unsafe fn [<load $bits _le_aligned>](ptr: *const AtomicU8) -> [<u $bits>] {
                <[<u $bits>]>::from_le(unsafe { [<load $bits _ne_aligned_inner>](ptr) })
            }

            #[doc = concat!(
                "Stores `value` to `ptr` (", stringify!($bytes), " bytes) as little-endian ",
                "bytes.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for writes of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs).\n",
                "- `ptr` must be aligned to `", stringify!($bytes), "` bytes. Use ",
                "[`store", stringify!($bits), "_le`] if `ptr` may be unaligned.",
            )]
            #[inline(always)]
            pub unsafe fn [<store $bits _le_aligned>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_aligned_inner>](ptr, value.to_le()) }
            }

            #[doc = concat!(
                "Loads a `u", stringify!($bits), "` (", stringify!($bytes), " bytes) from `ptr` ",
                "in native-endian order. `ptr` need not be aligned.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for reads of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs). No alignment requirement.",
            )]
            #[must_use]
            #[inline(always)]
            pub unsafe fn [<load $bits _ne>](ptr: *const AtomicU8) -> [<u $bits>] {
                unsafe { [<load $bits _ne_inner>](ptr) }
            }

            #[doc = concat!(
                "Stores `value` to `ptr` (", stringify!($bytes), " bytes) in native-endian ",
                "order. `ptr` need not be aligned.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for writes of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs). No alignment requirement.",
            )]
            #[inline(always)]
            pub unsafe fn [<store $bits _ne>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_inner>](ptr, value) }
            }

            #[doc = concat!(
                "Loads a `u", stringify!($bits), "` (", stringify!($bytes), " bytes) from `ptr`, ",
                "interpreting the in-memory bytes as little-endian. `ptr` need not be ",
                "aligned.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for reads of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs). No alignment requirement.",
            )]
            #[must_use]
            #[inline(always)]
            pub unsafe fn [<load $bits _le>](ptr: *const AtomicU8) -> [<u $bits>] {
                <[<u $bits>]>::from_le(unsafe { [<load $bits _ne_inner>](ptr) })
            }

            #[doc = concat!(
                "Stores `value` to `ptr` (", stringify!($bytes), " bytes) as little-endian ",
                "bytes. `ptr` need not be aligned.\n\n",
                "# Safety\n",
                "- `ptr` must be valid for writes of ", stringify!($bytes), " bytes and must ",
                "point into guest memory reachable only through this module (see the ",
                "module docs). No alignment requirement.",
            )]
            #[inline(always)]
            pub unsafe fn [<store $bits _le>](ptr: *const AtomicU8, value: [<u $bits>]) {
                unsafe { [<store $bits _ne_inner>](ptr, value.to_le()) }
            }
        )*}
    };
}

make_load_store! { (64, 8), (32, 4), (16, 2) }

/// Loads a single byte from `ptr`.
///
/// # Safety
/// `ptr` must be valid for reads and must point into guest memory reachable
/// only through this module (see the module docs). Byte access has no
/// alignment requirement.
#[inline(always)]
pub unsafe fn load_byte(ptr: *const AtomicU8) -> u8 {
    unsafe { load8(ptr) }
}

/// Stores a single byte to `ptr`.
///
/// # Safety
/// `ptr` must be valid for writes and must point into guest memory reachable
/// only through this module (see the module docs). Byte access has no
/// alignment requirement.
#[inline(always)]
pub unsafe fn store_byte(ptr: *const AtomicU8, value: u8) {
    unsafe { store8(ptr, value) }
}

/// Copies `count` bytes from guest memory at `src` into host memory at `dst`.
///
/// # Atomicity
/// This provides **byte-level atomicity only**: each byte is read
/// exactly once via a single atomic access, and no byte is torn. It does
/// **not** provide single-copy atomicity for the copy as a whole -- a
/// concurrent writer to `src` can interleave with this copy at byte
/// granularity, so the destination may end up holding a mix of bytes from
/// before and after a concurrent write. Callers that need the whole region to
/// appear as an atomic unit must arrange their own higher-level
/// synchronization; this function does not provide it.
///
/// # Safety
/// - `src` must be valid for atomic reads of `count` bytes and must point
///   into guest memory reachable only through this module (see the module
///   docs).
/// - `dst` must be valid for writes of `count` bytes.
/// - `src` and `dst` must denote non-overlapping regions, per
///   [`std::ptr::copy_nonoverlapping`]'s contract.
#[inline(always)]
pub unsafe fn copy_nonoverlapping_vm_to_host(src: *const AtomicU8, dst: *mut u8, count: usize) {
    unsafe { copy_nonoverlapping_vm_to_host_inner(src, dst, count) }
}

/// Copies `count` bytes from host memory at `src` into guest memory at `dst`.
///
/// # Atomicity
/// This provides **byte-level atomicity only**: each byte is
/// written exactly once via a single atomic access, and no byte is torn. It
/// does **not** provide single-copy atomicity for the copy as a whole -- a
/// concurrent reader of `dst` can observe a partially written region, mixing
/// bytes from before and after this call. Callers that need the whole region
/// to appear as an atomic unit must arrange their own higher-level
/// synchronization; this function does not provide it.
///
/// # Safety
/// - `src` must be valid for reads of `count` bytes.
/// - `dst` must be valid for atomic writes of `count` bytes and must point
///   into guest memory reachable only through this module (see the module
///   docs).
/// - `src` and `dst` must denote non-overlapping regions, per
///   [`std::ptr::copy_nonoverlapping`]'s contract.
#[inline(always)]
pub unsafe fn copy_nonoverlapping_host_to_vm(src: *const u8, dst: *const AtomicU8, count: usize) {
    unsafe { copy_nonoverlapping_host_to_vm_inner(src, dst, count) }
}
