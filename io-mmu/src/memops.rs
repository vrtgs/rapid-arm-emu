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

use std::hint::cold_path;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

// example sound impl of memops
// use crossbeam_utils::CachePadded;
// // use a Mersenne prime so that collisions are more rare
// const SHARDED_LOCK_COUNT: usize = 127;
//
// static ADDRESS_LOCKS: [CachePadded<parking_lot::RwLock<()>>; SHARDED_LOCK_COUNT] = {
//     [const { CachePadded::new(parking_lot::RwLock::new(())) }; SHARDED_LOCK_COUNT]
// };
//
// const CACHE_LINE: usize = 64;
// const _: () = assert!(CACHE_LINE.is_power_of_two());
// const CACHE_LINE_SHIFT: u32 = CACHE_LINE.ilog2();
//
// // this actually works since all reads are truly aligned
// // and because all of our atomic operations are smaller than this simulated cache_line
//
//
// unsafe fn get_cache_line<T>(ptr: *const AtomicU8) -> &'static parking_lot::RwLock<()> {
//     const { assert!(size_of::<T>() <= CACHE_LINE && align_of::<T>() <= size_of::<T>()) }
//     unsafe { core::hint::assert_unchecked(ptr.addr().is_multiple_of(size_of::<T>())) }
//
//     let cache_line = ptr.addr() >> CACHE_LINE_SHIFT;
//     &ADDRESS_LOCKS[cache_line % SHARDED_LOCK_COUNT]
// }
//
// pub(crate) unsafe fn read_aligned<T>(ptr: *const AtomicU8) -> T {
//     let lock = unsafe { get_cache_line::<T>(ptr) }.read();
//     let ret = unsafe { std::ptr::read(ptr.cast::<T>()) };
//     drop(lock);
//     ret
// }
//
// pub(crate) unsafe fn write_aligned<T>(ptr: *const AtomicU8, value: T) {
//     let lock = unsafe { get_cache_line::<T>(ptr) }.write();
//     unsafe { std::ptr::write(ptr.cast_mut().cast::<T>(), value) }
//     drop(lock);
// }

#[inline(always)]
pub(crate) unsafe fn load64_le_aligned(ptr: *const AtomicU8) -> u64 {
    unsafe { (*ptr.cast::<AtomicU64>()).load(Ordering::Relaxed).to_le() }
}

#[inline(always)]
pub(crate) unsafe fn store64_le_aligned(ptr: *const AtomicU8, value: u64) {
    unsafe { (*ptr.cast::<AtomicU64>()).store(value.to_le(), Ordering::Relaxed) }
}

#[inline(always)]
pub(crate) unsafe fn load32_le_aligned(ptr: *const AtomicU8) -> u32 {
    unsafe { (*ptr.cast::<AtomicU32>()).load(Ordering::Relaxed).to_le() }
}

#[inline(always)]
pub(crate) unsafe fn store32_le_aligned(ptr: *const AtomicU8, value: u32) {
    unsafe { (*ptr.cast::<AtomicU32>()).store(value.to_le(), Ordering::Relaxed) }
}

#[inline(always)]
pub(crate) unsafe fn load16_le_aligned(ptr: *const AtomicU8) -> u16 {
    unsafe { (*ptr.cast::<AtomicU16>()).load(Ordering::Relaxed).to_le() }
}

#[inline(always)]
pub(crate) unsafe fn store16_le_aligned(ptr: *const AtomicU8, value: u16) {
    unsafe { (*ptr.cast::<AtomicU16>()).store(value.to_le(), Ordering::Relaxed) }
}

#[inline(always)]
pub(crate) unsafe fn load_8_unaligned(ptr: *const AtomicU8) -> u8 {
    const { assert!(align_of::<AtomicU8>() == 1) }

    unsafe { (*ptr).load(Ordering::Relaxed) }
}

#[inline(always)]
pub(crate) unsafe fn store_8_unaligned(ptr: *const AtomicU8, value: u8) {
    const { assert!(align_of::<AtomicU8>() == 1) }

    unsafe { (*ptr).store(value, Ordering::Relaxed) }
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
pub(super) unsafe fn load64_le(ptr: *const AtomicU8) -> u64 {
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

            let b0 = load_8_unaligned(ptr) as u64;
            let w1 = load16_le_aligned(ptr.byte_add(1)) as u64;
            let mid = load32_le_aligned(ptr.byte_add(3)) as u64;
            let b7 = load_8_unaligned(ptr.byte_add(7)) as u64;

            (b7 << 56) | (mid << 24) | (w1 << 8) | b0
        },

        3 | 7 => unsafe {
            cold_path();

            let b0 = load_8_unaligned(ptr) as u64;
            let mid = load32_le_aligned(ptr.byte_add(1)) as u64;
            let w5 = load16_le_aligned(ptr.byte_add(5)) as u64;
            let b7 = load_8_unaligned(ptr.byte_add(7)) as u64;

            (b7 << 56) | (w5 << 40) | (mid << 8) | b0
        },
        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[inline(always)]
pub(super) unsafe fn store64_le(ptr: *const AtomicU8, value: u64) {
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

            store_8_unaligned(ptr, b0);
            store16_le_aligned(ptr.byte_add(1), w1);
            store32_le_aligned(ptr.byte_add(3), mid);
            store_8_unaligned(ptr.byte_add(7), b7);
        },

        3 | 7 => unsafe {
            cold_path();

            let b0 = value as u8;
            let mid = (value >> 8) as u32;
            let w5 = (value >> 40) as u16;
            let b7 = (value >> 56) as u8;

            store_8_unaligned(ptr, b0);
            store32_le_aligned(ptr.byte_add(1), mid);
            store16_le_aligned(ptr.byte_add(5), w5);
            store_8_unaligned(ptr.byte_add(7), b7);
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
pub(crate) unsafe fn load32_le(ptr: *const AtomicU8) -> u32 {
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

            let b0 = load_8_unaligned(ptr) as u32;
            let mid = load16_le_aligned(ptr.byte_add(1)) as u32;
            let b3 = load_8_unaligned(ptr.byte_add(3)) as u32;

            (b3 << 24) | (mid << 8) | b0
        },

        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[inline(always)]
pub(crate) unsafe fn store32_le(ptr: *const AtomicU8, value: u32) {
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

            store_8_unaligned(ptr, b0);
            store16_le_aligned(ptr.byte_add(1), mid);
            store_8_unaligned(ptr.byte_add(3), b3);
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
pub(crate) unsafe fn load16_le(ptr: *const AtomicU8) -> u16 {
    const { assert!(align_of::<AtomicU16>() == 2) }

    match ptr.addr() % 2 {
        0 => unsafe { load16_le_aligned(ptr) },

        1 => unsafe {
            cold_path();

            let b0 = load_8_unaligned(ptr) as u16;
            let b1 = load_8_unaligned(ptr.byte_add(1)) as u16;

            (b1 << 8) | b0
        },

        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[inline(always)]
pub(crate) unsafe fn store16_le(ptr: *const AtomicU8, value: u16) {
    const { assert!(align_of::<AtomicU16>() == 2) }

    match ptr.addr() % 2 {
        0 => unsafe { store16_le_aligned(ptr, value) },

        1 => unsafe {
            cold_path();

            store_8_unaligned(ptr, value as u8);
            store_8_unaligned(ptr.byte_add(1), (value >> 8) as u8);
        },

        _ => unsafe { core::hint::unreachable_unchecked() },
    }
}

#[inline(always)]
pub(crate) unsafe fn load_byte(ptr: *const AtomicU8) -> u8 {
    unsafe { load_8_unaligned(ptr) }
}

#[inline(always)]
pub(crate) unsafe fn store_byte(ptr: *const AtomicU8, value: u8) {
    unsafe { store_8_unaligned(ptr, value) }
}
