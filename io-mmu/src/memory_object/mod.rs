//! Demand-paged backing stores for MMU mappings.
//!
//! A [`MemoryObject`] is the source of truth behind an
//! [`IoMMU::map_device`](crate::IoMMU::map_device) mapping: pages are read
//! from it on first access ("fault in") and dirty pages are written back to
//! it ("fault out"). This module defines the trait and its safety contract;
//! ready-made implementations live in [`objects`] and on [`std::fs::File`].

use crate::page_table::{copy_page_exclusive_to_shared, copy_page_shared_to_exclusive};
use emu_abi::memory::{PageNumber, PagePointer, UninitPage};
use std::ptr::NonNull;
use std::sync::atomic::AtomicU8;

mod r#impl;
/// Ready-made [`MemoryObject`] implementations provided by this crate.
pub mod objects;

/// A memory object that services demand-paging faults.
///
/// A `MemoryObject` is the source of truth for the contents of a range of pages identified by [`PageNumber`].
/// The paging layer calls into this trait for:
///  - *fault in* a page (read the object's data for some [`PageNumber`] into a host page)
///  - *fault out* a page (Write a host page's contents back to the object for some [`PageNumber`])
///
/// Each operation comes in two flavors:
///
/// * The **exclusive** variants ([`fault_in_exclusive`](Self::fault_in_exclusive) and
///   [`fault_out_exclusive`](Self::fault_out_exclusive)) operate on a page to which the
///   caller has exclusive access. The page is a plain `NonNull<u8>`, so the implementation
///   may use ordinary (non-atomic) loads and stores and, when faulting in, may treat the
///   page as uninitialized memory to be filled.
///
/// * The **shared** variants ([`fault_in`](Self::fault_in) and
///   [`fault_out`](Self::fault_out)) operate on a page that may be concurrently accessed
///   by other agents. The page is a `NonNull<AtomicU8>`, and only single-byte atomic
///   accesses are permitted. Default implementations are provided that bounce the data
///   through a private, exclusively owned scratch page and reuse the exclusive variants;
///   implementors may override them to perform the transfer directly against the shared
///   page when that can be done with single-byte atomics.
///
/// Implementors must provide [`fault_in_exclusive`](Self::fault_in_exclusive) and
/// [`fault_out_exclusive`](Self::fault_out_exclusive). The shared variants have defaults
/// but may be overridden.
///
/// # Safety
///
/// This trait is `unsafe` to implement because the paging layer — and the default
/// implementations of the shared variants — rely on the following behavioral guarantees
/// for memory safety. An implementation that violates any of them can cause undefined
/// behavior in otherwise-correct callers.
///
/// An implementor must guarantee all the following:
///
/// 1. **Fault-in fully initializes the page, or faults.** Whenever
///    [`fault_in_exclusive`](Self::fault_in_exclusive) returns `Ok(())`, every one of the
///    `PAGE_SIZE` bytes of the page at `page_ptr` has been initialized. If the
///    implementation cannot initialize the entire page, it must return `Err` instead of
///    reporting success. (On `Err`, the page may be left partially initialized; callers
///    must not assume any byte is initialized.) The default [`fault_in`](Self::fault_in)
///    reads the whole scratch page only after a successful return, so reporting `Ok` over
///    a partially initialized page is undefined behavior.
///
/// 2. **Shared variants never deinitialize and touch bytes only through [`memops`](crate::memops).**
///    Any implementation of [`fault_in`](Self::fault_in) or [`fault_out`](Self::fault_out)
///    (including overrides) may access the bytes of the shared page *only* through the
///    functions in [`memops`](crate::memops), and must never cause any byte to become
///    uninitialized. It must not reach for `AtomicU8`'s inherent methods, hand-rolled
///    atomic accesses, non-atomic accesses, accesses wider than one byte, or any `write`
///    that leaves a byte uninitialized. `memops` is the single audited interface for this
///    page representation: it is the only place the mixed-size-atomic access discipline is
///    known to be upheld, so going around it forfeits that guarantee. Other agents may be
///    performing concurrent accesses through `memops` to the same page; anything else is a
///    data race.
///
/// 3. **All access completes before returning.** Every access an implementation makes to
///    the page must finish before the method returns. The implementation must not retain
///    `page_ptr` or continue to access the page afterward (for example, via still-in-flight
///    asynchronous DMA), because the caller regains control of the page on return.
///
/// Implementors may assume that callers uphold the per-method preconditions documented in
/// each method's `# Safety` section (pointer validity, `PAGE_SIZE` alignment, exclusivity
/// for the exclusive variants, and the atomic-access discipline for the shared variants).
pub unsafe trait MemoryObject: 'static + Send + Sync {
    /// Faults in the page identified by `page_offset`, filling the exclusively owned host
    /// page at `page_ptr` with its contents.
    ///
    /// On success the entire page is initialized with the object's data for `page_offset`
    /// (see the trait-level `# Safety` contract). On failure the page may be left partially
    /// initialized, and an error describing the fault is returned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * `page_ptr` is [valid] for writes of `PAGE_SIZE` bytes and points to a single live
    ///   allocation of that size. Its bytes need not be initialized on entry.
    /// * `page_ptr` is aligned to `PAGE_SIZE`.
    /// * The caller has exclusive access to the page for the duration of the call: no other
    ///   thread or DMA agent may read or write it concurrently.
    ///
    /// `page_offset` need not be serviceable by the object; an out-of-range or otherwise
    /// unserviceable page must be reported as `Err` rather than causing undefined behavior.
    ///
    /// [valid]: core::ptr#safety
    unsafe fn fault_in_exclusive(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<u8>,
    ) -> anyhow::Result<()>;

    /// Faults out the exclusively owned host page at `page_ptr`, writing its contents back
    /// to the page identified by `page_offset`.
    ///
    /// The full page is read and persisted to the object; the page itself is treated as
    /// read-only. On failure an error describing the fault is returned, and whether the
    /// object-side state was partially updated is implementation-defined.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * `page_ptr` is [valid] for reads of `PAGE_SIZE` bytes and points to a single live
    ///   allocation of that size.
    /// * Every one of the `PAGE_SIZE` bytes of the page is initialized. This method reads
    ///   the entire page, so any uninitialized byte is undefined behavior.
    /// * `page_ptr` is aligned to `PAGE_SIZE`.
    /// * The caller has full read access to the page for the duration of the call: no other
    ///   thread or DMA agent may write it concurrently,
    ///   and that means normal non-atomic operations are allowed.
    ///
    /// As with [`fault_in_exclusive`](Self::fault_in_exclusive), an unserviceable
    /// `page_offset` must be reported as `Err`.
    ///
    /// [valid]: core::ptr#safety
    unsafe fn fault_out_exclusive(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<u8>,
    ) -> anyhow::Result<()>;

    /// Faults in the page identified by `page_offset` into the *shared* host page at
    /// `page_ptr`, using only single-byte atomic stores.
    ///
    /// Unlike [`fault_in_exclusive`](Self::fault_in_exclusive), the destination page may be
    /// concurrently accessed by other agents through single-byte atomic operations. The
    /// stores this method performs are therefore not coordinated as a unit: an agent reading
    /// the page while the fault is in progress may observe an arbitrary mix of prior and
    /// faulted-in bytes (each byte read atomically), the same tearing you would see from a
    /// concurrent `msync`. There is no page-granularity "publish". The only promises are the
    /// trait-level ones: atomic access exclusively, no byte ever left uninitialized, and all
    /// access complete before the method returns.
    ///
    /// On `Err` the page may be left holding a partial mix of prior and faulted-in bytes, so
    /// callers must not assume its earlier contents survive a failed fault-in. Every byte
    /// nevertheless remains initialized, so this is not undefined behavior.
    ///
    /// The default implementation faults into a private, exclusively owned scratch page via
    /// [`fault_in_exclusive`](Self::fault_in_exclusive) and copies it over using
    /// [`memops`](crate::memops); an override may instead transfer directly into the shared
    /// page, but every access it makes to the shared page must still go through
    /// [`memops`](crate::memops). Either way, the trait-level shared-access guarantees must
    /// hold (shared-page access through `memops` only, never deinitialize).    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * `page_ptr` points to a single live allocation of `PAGE_SIZE` consecutive
    ///   [`AtomicU8`] that is [valid] for atomic reads and writes.
    /// * `page_ptr` is aligned to `PAGE_SIZE`.
    /// * For the duration of the call, every access to the page from any agent (including
    ///   this call) must be single-byte atomic access; no party performs non-atomic access or
    ///   access wider than one byte.
    ///
    /// An unserviceable `page_offset` must be reported as `Err`.
    ///
    /// [valid]: core::ptr#safety
    unsafe fn fault_in(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<AtomicU8>,
    ) -> anyhow::Result<()> {
        let mut page = UninitPage::new();
        let shared = unsafe { PagePointer::new(page_ptr) };

        unsafe {
            self.fault_in_exclusive(
                page_offset,
                page.page_pointer_mut().as_non_null_ptr().cast::<u8>(),
            )?
        }

        unsafe { copy_page_exclusive_to_shared(&page, shared) }
        Ok(())
    }

    /// Faults out the *shared* host page at `page_ptr`, writing its contents back to the
    /// page identified by `page_offset`. The page is read using only single-byte atomic
    /// loads.
    ///
    /// Unlike [`fault_out_exclusive`](Self::fault_out_exclusive), the source page may be
    /// concurrently accessed by other agents through single-byte atomic operations. The
    /// loads are not coordinated as a unit: if another agent mutates the page while it is
    /// being read, the bytes persisted to the object are an arbitrary interleaving across
    /// time (each byte read atomically), the same tearing you would see from a concurrent
    /// `msync`. The page is only ever read, never written. The only promises are the
    /// trait-level ones: atomic access exclusively, and all access complete before the
    /// method returns.
    ///
    /// The default implementation reads the shared page into a private, exclusively owned
    /// scratch page using [`memops`](crate::memops) and faults that out via
    /// [`fault_out_exclusive`](Self::fault_out_exclusive); an override may instead read
    /// directly from the shared page, but every access it makes to the shared page must still
    /// go through [`memops`](crate::memops). Either way, the trait-level shared-access
    /// guarantees must hold (shared-page access through `memops` only, never deinitialize).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    ///
    /// * `page_ptr` points to a single live allocation of `PAGE_SIZE` consecutive
    ///   [`AtomicU8`] that is [valid] for atomic reads and writes.
    /// * `page_ptr` is aligned to `PAGE_SIZE`.
    /// * For the duration of the call, every access to the page from any agent (including
    ///   this call) must be single-byte atomic access; no party performs non-atomic access or
    ///   access wider than one byte.
    ///
    /// An unserviceable `page_offset` must be reported as `Err`.
    ///
    /// [valid]: core::ptr#safety
    unsafe fn fault_out(
        &self,
        page_offset: PageNumber,
        page_ptr: NonNull<AtomicU8>,
    ) -> anyhow::Result<()> {
        let mut page = UninitPage::new();
        let shared = unsafe { PagePointer::new(page_ptr) };
        unsafe { copy_page_shared_to_exclusive(shared, &mut page) }

        unsafe {
            self.fault_out_exclusive(
                page_offset,
                page.page_pointer_ref().as_non_null_ptr().cast::<u8>(),
            )
        }
    }
}
