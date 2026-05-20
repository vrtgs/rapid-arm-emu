use crate::{MemoryFault, PageTableAccess, ensure};
use emu_abi::abort::abort;
use emu_abi::convert::u64_to_usize;
use emu_abi::memory::{MemProt, PAGE_SIZE, Page, PageNumber, PagePointer, TaggedPagePtr};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const PAGE_LAYOUT: std::alloc::Layout = {
    match std::alloc::Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
        Ok(layout) => layout,
        Err(_) => panic!("page size too big"),
    }
};

/// # Safety
///
/// not actually unsafe, but care must be taken to ensure that the pointer isn't leaked
unsafe fn alloc_page() -> PagePointer {
    const { assert!(PAGE_SIZE != 0) }
    // Safety: layout is not zero
    let ptr = unsafe { std::alloc::alloc_zeroed(PAGE_LAYOUT) };
    match NonNull::new(ptr) {
        Some(ptr) => unsafe { PagePointer::new(ptr.cast()) },
        None => std::alloc::handle_alloc_error(PAGE_LAYOUT),
    }
}

unsafe fn dealloc_page(ptr: PagePointer) {
    unsafe { std::alloc::dealloc(ptr.as_non_null_ptr().cast().as_ptr(), PAGE_LAYOUT) }
}

struct BackingPage {
    allocated_page: Option<PagePointer>,
    dirty_page_flag: AtomicBool,
}

impl Drop for BackingPage {
    fn drop(&mut self) {
        if let Some(page) = self.allocated_page {
            unsafe { dealloc_page(page) }
        }
    }
}

// backing page is seperated
// this is to support COW in the future
#[derive(Clone)]
pub(super) struct PageEntry {
    ptr: TaggedPagePtr,
    backing: Arc<BackingPage>,
}

impl PageEntry {
    unsafe fn new_inner(is_allocated: bool, ptr: PagePointer, mem_prot: MemProt) -> Self {
        Self {
            ptr: TaggedPagePtr::new(ptr, mem_prot),
            backing: Arc::new(BackingPage {
                allocated_page: is_allocated.then_some(ptr),
                dirty_page_flag: AtomicBool::new(false),
            }),
        }
    }

    pub fn new_alloc(mem_prot: MemProt) -> Self {
        let ptr = unsafe { alloc_page() };
        let is_allocated = true;
        // Safety: ptr was just allocated
        unsafe { Self::new_inner(is_allocated, ptr, mem_prot) }
    }

    pub unsafe fn new_shared(ptr: PagePointer, mem_prot: MemProt) -> Self {
        let is_allocated = false;
        // Safety: ptr is not allocated, and is with shared memory
        unsafe { Self::new_inner(is_allocated, ptr, mem_prot) }
    }

    pub fn memprot(&mut self, new_prot: MemProt) {
        self.ptr = TaggedPagePtr::new(self.ptr.page_ptr(), new_prot);
    }

    pub fn prot(&self) -> MemProt {
        self.ptr.prot()
    }

    pub fn as_page(&self) -> Page<'_> {
        Page {
            ptr: self.ptr,
            insn_dirty: &self.backing.dirty_page_flag,
        }
    }
}

// SAFETY:
// PageEntry contains a raw page pointer, but all public access to the pointed-to
// memory is mediated through memops, which use atomic byte/scalar operations.
// For internally allocated pages, BackingPage owns the allocation and keeps it alive
// while any cloned PageEntry exists. For shared pages, pointer validity and aliasing
// obligations are required by map_shared. The dirty flag is AtomicBool.
unsafe impl Send for PageEntry {}
unsafe impl Sync for PageEntry {}

pub(super) struct PageTable {
    table: HashMap<PageNumber, PageEntry>,
}

impl PageTable {
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub unsafe fn map(
        &mut self,
        pages: PageTableAccess,
        mem_prot: MemProt,
        region_ptr: Option<PagePointer>,
    ) -> Result<(), MemoryFault> {
        for page in pages.iter() {
            ensure!(!self.table.contains_key(&page))
        }

        let start_page = pages.start_page;
        for page in pages.iter() {
            let mapped = match region_ptr {
                Some(ptr) => unsafe {
                    let page_offset = page.get().unchecked_sub(start_page.get());
                    let offset = u64_to_usize(page_offset).unwrap_unchecked();
                    let page_ptr = ptr.add_pages(offset);
                    PageEntry::new_shared(page_ptr, mem_prot)
                },
                None => PageEntry::new_alloc(mem_prot),
            };

            let old_page = self.table.insert(page, mapped);
            if old_page.is_some() {
                abort()
            }
        }

        Ok(())
    }

    pub fn unmap(&mut self, pages: PageTableAccess, mut removed: impl FnMut(Page<'_>)) {
        for page in pages.iter() {
            if let Entry::Occupied(page_entry) = self.table.entry(page) {
                removed(page_entry.get().as_page());
                page_entry.remove();
            }
        }
    }

    pub fn modify(
        &mut self,
        pages: PageTableAccess,
        mut modify: impl FnMut(PageNumber, &mut PageEntry),
    ) -> Result<(), MemoryFault> {
        for page in pages.iter() {
            ensure!(self.table.contains_key(&page))
        }

        for page in pages.iter() {
            let page_entry = self.table.get_mut(&page).unwrap_or_else(|| abort());
            modify(page, page_entry)
        }

        Ok(())
    }

    pub fn get_page(&self, page: PageNumber) -> Result<Page<'_>, MemoryFault> {
        self.table
            .get(&page)
            .ok_or_else(MemoryFault::fault)
            .map(PageEntry::as_page)
    }

    pub fn pages(&self) -> impl Iterator<Item = (PageNumber, Page<'_>)> {
        self.table
            .iter()
            .map(|(&page, entry)| (page, entry.as_page()))
    }
}
