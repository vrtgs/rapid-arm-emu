//! Address-translation caching for [`IoMMU`] lookups.
//!
//! Every memory access on an [`IoMMU`] is parameterized by a [`LookupCache`],
//! which resolves page numbers to live [`Page`] handles. Callers can pass
//! [`NoCache`] to hit the page table directly on every access, or a
//! [`Tlb`] to amortize translation cost across repeated accesses.

use crate::fault::{MemoryFault, ensure};
use crate::icache::ICache;
use crate::{IoMMU, PageTableAccess, div_rem_page_size};
use emu_abi::memory::{Page, PageNumber, Tlb};
use std::process::abort;

/// A cache layer that sits between the CPU and the page table, translating
/// [`PageNumber`]s into live [`Page`] handles.
///
/// Implementors may satisfy lookups from a fast local structure (e.g., a TLB)
/// or fall through to the page table on every call. Either way, the returned
/// [`Page`] must be a valid view into the [`IoMMU`]'s backing memory for the
/// requested page number.
// Note: callers rely on this to uphold memory safety across
//       the verify-then-access split in [`LookupCacheExt::access`]
///
/// # Safety
///
/// If `get_page` returns `Ok` for a given `page`, then every subsequent call
/// with the same `page` and the same [`IoMMU`] - without any intervening
/// mutation of that [`IoMMU`] - must also return `Ok`. Returning `Err` after
/// a prior `Ok` is **undefined behavior**: the cache's consistency guarantee
/// is a precondition that callers are permitted to assume without checking.
///
pub unsafe trait LookupCache {
    /// Resolves `page` to a [`Page`] handle valid for the lifetime of `io_mmu`.
    ///
    /// See the trait-level safety docs for the consistency requirement that
    /// implementations must uphold.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault>;
}

/// A [`LookupCache`] that performs no caching at all.
///
/// Every lookup goes straight to the page table. Useful when the overhead of
/// maintaining a TLB outweighs its benefit — e.g., one-off accesses where a
/// cached entry would never be reused.
pub struct NoCache;

unsafe impl LookupCache for NoCache {
    /// Bypasses any caching layer and faults directly to the page table.
    /// Useful when the overhead of a TLB lookup outweighs its benefit
    /// e.g., single-access patterns where a cached entry would never be reused.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        io_mmu.table.get_page(page)
    }
}

unsafe impl LookupCache for Tlb {
    /// Attempts a TLB hit before falling back to the page table, amortising
    /// translation cost across repeated accesses to the same page.
    ///
    /// The `get_ident` check ensures we don't serve a stale entry after an
    /// address-space switch - the TLB is logically scoped to one MMU identity.
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page_num: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        unsafe {
            self.lookup(page_num, io_mmu.get_ident(), |page_num| {
                io_mmu.table.get_page(page_num)
            })
        }
    }
}

unsafe impl<C: LookupCache> LookupCache for &mut C {
    #[inline(always)]
    fn get_page<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        page: PageNumber,
    ) -> Result<Page<'a>, MemoryFault> {
        <C as LookupCache>::get_page(*self, io_mmu, page)
    }
}

pub(crate) trait LookupCacheExt: LookupCache {
    /// Resolves a virtual address into its page number, in-page byte offset,
    /// and a handle to the backing page - all in one step so callers don't
    /// have to re-derive the page number from the address themselves.
    #[inline(always)]
    fn lookup_addr<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        vaddr: u64,
    ) -> Result<(PageNumber, usize, Page<'a>), MemoryFault> {
        let (page_num, offset) = div_rem_page_size(vaddr);
        let page = self.get_page(io_mmu, page_num)?;
        Ok((page_num, offset, page))
    }

    /// Verifies that every page in `pages` satisfies the caller's predicate,
    /// then calls `access` on each one.
    ///
    /// The split into two phases is intentional: all-or-nothing semantics.
    /// If any page fails verification, the access closure is never invoked,
    /// so callers don't need to reason about partially applied side effects.
    ///
    /// No ordering guarantee is made on `access`; the iteration order is an
    /// internal implementation detail and may change.
    fn access<'a, T: ?Sized + ICache>(
        &mut self,
        io_mmu: &'a IoMMU<T>,
        pages: PageTableAccess,
        mut verify: impl FnMut(PageNumber, Page<'a>) -> bool,
        mut access: impl FnMut(PageNumber, Page<'a>),
    ) -> Result<(), MemoryFault> {
        // Verify in reverse so the cache is hottest at the head of the range
        // when access begins - the forward access pass then walks into warm
        // entries rather than cold ones.
        for page_num in pages.iter().rev() {
            let page = self.get_page(io_mmu, page_num)?;
            ensure!(vaddr: page_num.vaddr_base(), verify(page_num, page))
        }

        // Forward walk gives the hardware prefetcher a predictable ascending
        // stride, which matters more on the access loop than verify because
        // this is where real work happens.
        //
        // Unwrap is sound here: verify already confirmed every page is
        // reachable, so a fault now would mean the page table was mutated
        // underneath us - unrecoverable either way.
        for page_num in pages.iter() {
            let page = self.get_page(io_mmu, page_num).unwrap_or_else(|_| abort());
            access(page_num, page)
        }

        Ok(())
    }
}

impl<C: LookupCache> LookupCacheExt for C {}
