use crate::cpu_fabric::object_manager::{FnCallback, ObjectSyncCallbacks};
use crate::cpu_fabric::{CpuFabric, CpuFabricWeak};
use crate::fault::{MemoryFault, ensure};
use crate::icache::ICache;
use crate::memory_object::MemoryObject;
use crate::{PageTableAccess, memops};
use crossbeam_utils::CachePadded;
use emu_abi::abort::{abort, panic_abort};
use emu_abi::convert::u64_to_usize;
use emu_abi::memory::{
    MemFlags, MemProt, PAGE_SIZE, Page, PageNumber, PagePointer, TaggedPagePtr, UninitPageMut,
};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::convert::Infallible;
use std::hint::cold_path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};

const PAGE_LAYOUT: std::alloc::Layout = {
    match std::alloc::Layout::from_size_align(PAGE_SIZE, PAGE_SIZE) {
        Ok(layout) => layout,
        Err(_) => panic!("page size too big"),
    }
};

/// # Safety
///
/// not actually unsafe, but care must be taken to ensure that the pointer isn't leaked
unsafe fn alloc_page_zeroed() -> PagePointer {
    const { assert!(PAGE_SIZE != 0) }
    // Safety: layout is not zero
    let ptr = unsafe { std::alloc::alloc_zeroed(PAGE_LAYOUT) };
    match NonNull::new(ptr) {
        Some(ptr) => unsafe { PagePointer::new(ptr.cast()) },
        None => std::alloc::handle_alloc_error(PAGE_LAYOUT),
    }
}

/// # Safety
///
/// not actually unsafe, but care must be taken to ensure that the pointer isn't leaked
/// and to also ensure no atomic ops happen on uninit data
unsafe fn alloc_page_uninit() -> PagePointer {
    const { assert!(PAGE_SIZE != 0) }
    // Safety: layout is not zero
    let ptr = unsafe { std::alloc::alloc(PAGE_LAYOUT) };
    match NonNull::new(ptr) {
        Some(ptr) => unsafe { PagePointer::new(ptr.cast()) },
        None => std::alloc::handle_alloc_error(PAGE_LAYOUT),
    }
}

unsafe fn dealloc_page(ptr: PagePointer) {
    unsafe { std::alloc::dealloc(ptr.as_non_null_ptr().cast().as_ptr(), PAGE_LAYOUT) }
}

pub(crate) struct MemoryBackedPage {
    allocated_page: PagePointer,
    cpu_fabric: CpuFabricWeak<dyn ICache>,
    should_dealloc: bool,
    dirty_page_flags: CachePadded<AtomicU8>,
}

impl MemoryBackedPage {
    unsafe fn try_make_new<E>(
        alloc: impl FnOnce() -> PagePointer,
        should_dealloc: bool,
        modify: impl FnOnce(&mut UninitPageMut) -> Result<(), E>,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Result<Self, E> {
        enum VoidCache {}

        impl ICache for VoidCache {
            fn invalidate(&self, _: PagePointer) {
                match *self {}
            }
        }

        let mut this = Self {
            allocated_page: alloc(),
            should_dealloc,
            dirty_page_flags: CachePadded::new(AtomicU8::new(0)),
            cpu_fabric: (const { CpuFabricWeak::<VoidCache>::new() }).into_dyn(),
        };

        modify(unsafe { UninitPageMut::from_ptr(this.allocated_page) })?;
        this.cpu_fabric = cpu_fabric();
        Ok(this)
    }

    fn alloc_zeroed(cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>) -> Self {
        let should_dealloc = true;
        let Ok(page) = unsafe {
            Self::try_make_new(
                || alloc_page_zeroed(),
                should_dealloc,
                |_| Ok::<(), Infallible>(()),
                cpu_fabric,
            )
        };

        page
    }

    /// # Safety
    /// must init the full page
    unsafe fn alloc_with_init<E>(
        init: impl FnOnce(&mut UninitPageMut) -> Result<(), E>,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Result<Self, E> {
        let should_dealloc = true;
        unsafe { Self::try_make_new(|| alloc_page_uninit(), should_dealloc, init, cpu_fabric) }
    }

    unsafe fn new_extern(
        allocated_page: PagePointer,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Self {
        // doing it this way because dropping this in the middle of
        // initializing the cpu_fabric does nothing, there is nothing to dealloc
        // and nothing to invalidate
        let cpu_fabric = cpu_fabric();
        Self {
            allocated_page,
            should_dealloc: false,
            dirty_page_flags: CachePadded::new(AtomicU8::new(0)),
            cpu_fabric,
        }
    }

    pub(crate) fn fault_object(
        object: &dyn MemoryObject,
        page_offset: PageNumber,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> anyhow::Result<Self> {
        unsafe {
            Self::alloc_with_init(
                |page| {
                    object.fault_in_exclusive(
                        page_offset,
                        page.page_pointer_mut().as_non_null_ptr().cast::<u8>(),
                    )
                },
                cpu_fabric,
            )
        }
    }

    pub(crate) fn page_pointer(&self) -> PagePointer {
        self.allocated_page
    }
}

impl Drop for MemoryBackedPage {
    fn drop(&mut self) {
        struct DeallocGuard(PagePointer);

        impl Drop for DeallocGuard {
            fn drop(&mut self) {
                unsafe { dealloc_page(self.0) }
            }
        }

        let page = self.allocated_page;

        // keep the page alive whilst we invalidate.
        // this ensures that there is no window where stale icache is available
        let _guard = self.should_dealloc.then(|| DeallocGuard(page));

        let Some(fabric) = self.cpu_fabric.upgrade() else {
            // this only ever really happens with the zero page
            // which is stored inside the fabric itself
            cold_path();
            return;
        };

        // this prevents 2 things
        // - memory leaks from code buffers accumulating
        // - preventing stale execution of a page that got
        //   reallocated and now happens to live
        //   at the old address, which causes some
        //   hard to debug UB
        fabric.icache().invalidate(page);
    }
}

const WORDS_IN_PAGE: usize = {
    let word_size = size_of::<u64>();
    assert!(PAGE_SIZE.is_multiple_of(word_size));
    PAGE_SIZE / word_size
};

#[inline]
pub(crate) unsafe fn copy_page_shared_to_exclusive(
    shared: PagePointer,
    exclusive: &mut UninitPageMut,
) {
    let dst_ptr = exclusive
        .page_pointer_mut()
        .as_non_null_ptr()
        .cast::<u64>()
        .as_ptr();
    let src_ptr = shared.as_non_null_ptr().as_ptr().cast::<AtomicU64>();

    for i in 0..WORDS_IN_PAGE {
        // load words with native endian and store with native endian;
        // that gives us maximal perf since load_le store_le cancel out
        let word_value = unsafe { memops::load64_ne_aligned(src_ptr.add(i).cast()) };
        // this is safe since we have unique access to this page
        unsafe { std::ptr::write(dst_ptr.add(i), word_value) }
    }
}

#[inline]
pub(crate) unsafe fn copy_page_exclusive_to_shared(exclusive: &UninitPageMut, shared: PagePointer) {
    let src_ptr = exclusive
        .page_pointer_ref()
        .as_non_null_ptr()
        .cast::<u64>()
        .as_ptr();
    let dst_ptr = shared.as_non_null_ptr().as_ptr().cast::<AtomicU64>();

    for i in 0..WORDS_IN_PAGE {
        // read the comments in the loop of `copy_page_shared_to_exclusive`
        // on why we use native endian ordering
        let word_value = unsafe { std::ptr::read(src_ptr.add(i)) };
        unsafe { memops::store64_ne_aligned(dst_ptr.add(i).cast(), word_value) }
    }
}

fn zero_page<T: ?Sized + ICache>(cpu_fabric: &CpuFabric<T>) -> Arc<MemoryBackedPage> {
    #[cold]
    #[inline(never)]
    fn init(cpu_fabric: CpuFabricWeak<dyn ICache>) -> Arc<MemoryBackedPage> {
        Arc::new(MemoryBackedPage::alloc_zeroed(move || cpu_fabric))
    }

    cpu_fabric
        .zero_page()
        .get_or_init(move || init(cpu_fabric.downgrade_dyn()))
        .clone()
}

impl Clone for MemoryBackedPage {
    fn clone(&self) -> Self {
        if let Some(fabric) = self.cpu_fabric.upgrade()
            && let Some(zero_page) = fabric.zero_page().get()
            && std::ptr::eq::<Self>(Arc::as_ptr(zero_page), self)
        {
            return Self::alloc_zeroed(|| self.cpu_fabric.clone());
        }

        // instantly create allocate page, and bind it to self
        // so that it gets dropped if anything panics
        let Ok(page) = unsafe {
            Self::alloc_with_init(
                |exclusive| {
                    let shared = self.allocated_page;
                    copy_page_shared_to_exclusive(shared, exclusive);
                    Ok::<(), Infallible>(())
                },
                || self.cpu_fabric.clone(),
            )
        };

        // this is a fresh page; therefore, it isn't even in the icache or DMA, so it's clean
        // *page.dirty_page_flag.get_mut() = self.dirty_page_flag.load(Ordering::Acquire);

        page
    }
}

pub(crate) struct ObjectSlot {
    pub(crate) page: once_cell::sync::OnceCell<Arc<MemoryBackedPage>>,
    pub(crate) object: Arc<dyn MemoryObject>,
    pub(crate) page_offset: PageNumber,
}

enum PageSource {
    Shared(Arc<MemoryBackedPage>),
    Private {
        page: Arc<MemoryBackedPage>,
        write_protected: bool,
    },

    Object {
        slot: Arc<ObjectSlot>,
        shared: bool,
    },
}

impl PageSource {
    fn new_private(page: MemoryBackedPage) -> Self {
        Self::Private {
            page: Arc::new(page),
            write_protected: false,
        }
    }

    fn new_private_cow(page: Arc<MemoryBackedPage>) -> Self {
        Self::Private {
            page,
            write_protected: true,
        }
    }

    fn zeroed_cow<T: ?Sized + ICache>(cpu_fabric: &CpuFabric<T>) -> Self {
        Self::new_private_cow(zero_page(cpu_fabric))
    }

    fn zeroed(cpu_fabric: &dyn Fn() -> CpuFabricWeak<dyn ICache>) -> Self {
        Self::new_private(MemoryBackedPage::alloc_zeroed(cpu_fabric))
    }

    fn zeroed_shared(cpu_fabric: &dyn Fn() -> CpuFabricWeak<dyn ICache>) -> Self {
        Self::Shared(Arc::new(MemoryBackedPage::alloc_zeroed(cpu_fabric)))
    }

    fn new_obj(object: Arc<dyn MemoryObject>, page_offset: PageNumber, shared: bool) -> Self {
        Self::Object {
            slot: Arc::new(ObjectSlot {
                page: once_cell::sync::OnceCell::new(),
                object,
                page_offset,
            }),
            shared,
        }
    }

    fn fork(&mut self) -> Self {
        match self {
            Self::Shared(page) => Self::Shared(Arc::clone(page)),

            &mut Self::Object {
                slot: ref page,
                shared: shared @ true,
            } => Self::Object {
                slot: Arc::clone(page),
                shared,
            },

            Self::Private {
                page,
                write_protected,
            } => {
                *write_protected = true;
                Self::Private {
                    page: Arc::clone(page),
                    write_protected: true,
                }
            }

            &mut Self::Object {
                ref slot,
                shared: shared @ false,
            } => match slot.page.get().cloned() {
                Some(page) => {
                    let page2 = Arc::clone(&page);
                    *self = Self::new_private_cow(page);
                    Self::new_private_cow(page2)
                }
                None => Self::new_obj(Arc::clone(&slot.object), slot.page_offset, shared),
            },
        }
    }
}

pub(super) struct PageEntry {
    source: PageSource,
    protections: MemProt,
}

impl PageEntry {
    fn new_inner(source: PageSource, prot: MemProt) -> Self {
        Self {
            source,
            protections: prot,
        }
    }

    fn new_zeroed_cow<T: ?Sized + ICache>(mem_prot: MemProt, cpu_fabric: &CpuFabric<T>) -> Self {
        Self::new_inner(PageSource::zeroed_cow(cpu_fabric), mem_prot)
    }

    fn new_zeroed(mem_prot: MemProt, cpu_fabric: &dyn Fn() -> CpuFabricWeak<dyn ICache>) -> Self {
        Self::new_inner(PageSource::zeroed(cpu_fabric), mem_prot)
    }

    fn new_zeroed_shared(
        mem_prot: MemProt,
        cpu_fabric: &dyn Fn() -> CpuFabricWeak<dyn ICache>,
    ) -> Self {
        Self::new_inner(PageSource::zeroed_shared(cpu_fabric), mem_prot)
    }

    unsafe fn new_extern(
        ptr: PagePointer,
        mem_prot: MemProt,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Self {
        let extern_page = unsafe { MemoryBackedPage::new_extern(ptr, cpu_fabric) };

        Self::new_inner(PageSource::new_private(extern_page), mem_prot)
    }

    unsafe fn new_extern_cow(
        ptr: PagePointer,
        mem_prot: MemProt,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Self {
        let extern_page = unsafe { MemoryBackedPage::new_extern(ptr, cpu_fabric) };

        Self::new_inner(PageSource::new_private_cow(Arc::new(extern_page)), mem_prot)
    }

    unsafe fn new_extern_shared(
        ptr: PagePointer,
        mem_prot: MemProt,
        cpu_fabric: impl FnOnce() -> CpuFabricWeak<dyn ICache>,
    ) -> Self {
        let extern_page = unsafe { MemoryBackedPage::new_extern(ptr, cpu_fabric) };

        Self::new_inner(PageSource::Shared(Arc::new(extern_page)), mem_prot)
    }

    fn new_obj(
        object: Arc<dyn MemoryObject>,
        page_offset: PageNumber,
        shared: bool,
        prot: MemProt,
    ) -> Self {
        let page = PageSource::new_obj(object, page_offset, shared);

        Self::new_inner(page, prot)
    }

    pub(super) fn memprot(&mut self, new_prot: MemProt) {
        self.protections = new_prot;
    }

    pub(super) fn prot(&self) -> MemProt {
        self.protections
    }

    fn as_page(&self) -> Option<Page<'_>> {
        let (mem_page, flags): (&MemoryBackedPage, MemFlags) = match &self.source {
            PageSource::Shared(page) => (page, MemFlags::from_prot(self.protections)),

            PageSource::Private {
                page,
                write_protected,
            } => {
                let flags = match *write_protected {
                    true => MemFlags::COW | (self.protections & (!MemProt::WRITE)),
                    false => MemFlags::from_prot(self.protections),
                };

                (page, flags)
            }

            &PageSource::Object { ref slot, shared } => {
                let page = slot.page.get()?;
                let flags = match shared {
                    true => MemFlags::DMA_DEV | self.protections,
                    false => MemFlags::from_prot(self.protections),
                };

                (page, flags)
            }
        };

        Some(Page {
            ptr: TaggedPagePtr::new(mem_page.allocated_page, flags | self.protections),
            dirty_flags: &mem_page.dirty_page_flags,
        })
    }

    // TODO(low priority) make this type erased by getting a DynCpuFabricRef type
    //                    this avoids cloning the `Arc`;
    //                    note this is helpful for non memory_object pages like `Shared` and `Private`
    //                    since it turns to just a cheap check and return
    fn flush_dyn(
        &self,
        make_cpu_fabric: &dyn Fn() -> CpuFabric<dyn ICache>,
        make_callback: Option<&mut dyn FnMut() -> Box<dyn ObjectSyncCallbacks>>,
    ) {
        let page: &MemoryBackedPage = match &self.source {
            PageSource::Shared(page) | PageSource::Private { page, .. } => page,

            PageSource::Object { slot, .. } if let Some(page) = slot.page.get() => page,

            PageSource::Object { .. } => return,
        };

        // `dirty_page_flags` is a 2-bit state machine, not a lock: 00 clean,
        // 01 DIRTY (unflushed write), 10 FLUSHING (epoch in progress), 11
        // DIRTY|FLUSHING (a `write` landed during an in-flight epoch).
        //
        // Writers only `fetch_or(DIRTY)` - they never touch FLUSHING, so a
        // `write` can move 00->01 or 10->11 but can never erase a pending
        // DIRTY or clobber an in-progress FLUSHING out from under a flusher.
        //
        // The CAS below is not a permission check - every thread that
        // observes pending work (DIRTY and/or FLUSHING) falls through and
        // runs `invalidate` + the trailing `fetch_and` regardless of whether
        // its own CAS won. Redundant invalidates across racing threads are
        // harmless (idempotent, and the DMA leg is just a hint anyway), so
        // there's nothing to serialize there. The CAS exists only to pick
        // exactly one thread to perform the 01->10 transition that opens a
        // flush epoch; `None` on every other observed state means "leave
        // the bits alone, I'm just going to help."
        //
        //   0           -> bail, nothing pending (the only `Err(0)` case)
        //   FLUSHING/11 -> None: an epoch already exists and is visible;
        //                  join it instead of re-claiming it
        //   DIRTY       -> Some(FLUSHING): claim the epoch
        //
        // `fetch_and(!FLUSHING)` afterward only ever masks FLUSHING, so a
        // DIRTY set mid-epoch (state 11) always survives to be picked up by
        // a future call - a `write` can never be dropped, only deferred.
        // `prev & FLUSHING` on that call doubles as a free, race-proof
        // "last one out" check: exactly one racing thread observes the bit
        // still set, so the DMA enqueue happens once per epoch, not once
        // per racing thread.
        //
        // Ordering:
        // Every operation on `dirty_page_flags` here is SeqCst,
        // this is a hard requirement on *all* of them, not just the CAS.
        // The flag is what's standing between a writer's plain store to the
        // page and a flusher concluding "nothing more to invalidate"; that
        // is a StoreLoad relationship (writer's own prior store vs. its own
        // subsequent flag check), and only SeqCst forecloses the reordering
        // where the store is still invisible when the flag read happens.
        // Acquire/Release do not provide StoreLoad and are not a safe
        // substitute on any of these ops - do not weaken any ordering here without re-confirming.
        let res = page
            .dirty_page_flags
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                const DIRTY_AND_FLUSHING: u8 = {
                    let bits = Page::DIRTY_FLAG_IS_DIRTY | Page::DIRTY_FLAG_FLUSHING;
                    assert!(bits.count_ones() == 2);
                    bits
                };

                match x {
                    0 => None,
                    Page::DIRTY_FLAG_FLUSHING | DIRTY_AND_FLUSHING => {
                        cold_path();
                        None
                    }
                    Page::DIRTY_FLAG_IS_DIRTY => {
                        cold_path();
                        Some(Page::DIRTY_FLAG_FLUSHING)
                    }
                    _ => panic_abort!("invalid page dirty state {x:02b}"),
                }
            });

        if let Err(0) = res {
            return;
        }

        let cpu_fabric = make_cpu_fabric();
        cpu_fabric.icache().invalidate(page.allocated_page);

        let prev = page
            .dirty_page_flags
            .fetch_and(!Page::DIRTY_FLAG_FLUSHING, Ordering::SeqCst);

        // if we won the epoch flush the DMA device
        // sinc the DMA flushing part of is just a hint
        // it doesn't need to be up to date
        if (prev & Page::DIRTY_FLAG_FLUSHING) != 0
            && let PageSource::Object { slot, shared: true } = &self.source
        {
            let cb = make_callback.map(|cb| cb());
            cpu_fabric.object_manager().enqueue_flush(slot, cb)
        }
    }

    pub(super) fn refresh<T: ?Sized + ICache>(&self, cpu_fabric: &CpuFabric<T>) {
        self.flush_dyn(&|| cpu_fabric.clone().into_dyn(), None)
    }

    fn make_sync_callback(
        rx: &mut Option<mpsc::Receiver<anyhow::Result<()>>>,
    ) -> Box<dyn ObjectSyncCallbacks> {
        assert!(rx.is_none());

        let (tx, new_rx) = mpsc::sync_channel(1);
        let callback = FnCallback::new_flush(move |res| {
            let _ = tx.send(res.map_err(|err| anyhow::Error::msg(format!("{err:#}"))));
        });

        *rx = Some(new_rx);

        Box::new(callback) as Box<dyn ObjectSyncCallbacks>
    }

    pub(super) fn flush_sync_inner<T: ?Sized + ICache>(
        &self,
        cpu_fabric: &CpuFabric<T>,
    ) -> Option<mpsc::Receiver<anyhow::Result<()>>> {
        let mut rx = None;
        let mut make_callback = || Self::make_sync_callback(&mut rx);
        self.flush_dyn(&|| cpu_fabric.clone().into_dyn(), Some(&mut make_callback));
        rx
    }

    // TODO(low priority) make this type erased by getting a DynCpuFabricRef type
    //                    this avoids cloning the `Arc`;
    //                    note this is helpful for non memory_object pages like `Shared` and `Private`
    //                    since it turns to just a cheap check and return
    fn init_page_dyn(
        &self,
        make_cpu_fabric: &dyn Fn() -> CpuFabric<dyn ICache>,
        make_callback: Option<&mut dyn FnMut() -> Box<dyn ObjectSyncCallbacks>>,
    ) {
        let PageSource::Object {
            ref slot,
            shared: _,
        } = self.source
        else {
            return;
        };

        if slot.page.get().is_some() {
            return;
        }

        let cpu = make_cpu_fabric();
        let cb = make_callback.map(|cb| cb());
        cpu.object_manager().enqueue_init(slot, cpu.downgrade(), cb);
    }

    pub(super) fn load_obj_memory_sync_inner<T: ?Sized + ICache>(
        &self,
        cpu_fabric: &CpuFabric<T>,
    ) -> Option<mpsc::Receiver<anyhow::Result<()>>> {
        let mut rx = None;
        let mut make_callback = || Self::make_sync_callback(&mut rx);
        self.init_page_dyn(&|| cpu_fabric.clone().into_dyn(), Some(&mut make_callback));
        rx
    }

    pub(super) fn fork(&mut self) -> Self {
        Self {
            source: self.source.fork(),
            protections: self.protections,
        }
    }

    pub(super) fn un_cow(&mut self) -> bool {
        match self.source {
            PageSource::Private {
                ref mut page,
                ref mut write_protected,
            } => {
                let was_cow = *write_protected;
                if was_cow {
                    let _: &mut MemoryBackedPage = Arc::make_mut(page);
                    *write_protected = false
                }
                was_cow
            }

            PageSource::Shared(_) | PageSource::Object { .. } => false,
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

#[derive(Copy, Clone, Eq, PartialEq)]
pub(super) enum MemMapFlags {
    Private,
    Shared,
    Cow,
}

pub(super) enum MapRegion {
    Object {
        shared: bool,
        object: Arc<dyn MemoryObject>,
    },
    Extern {
        flags: MemMapFlags,
        base_ptr: PagePointer,
    },
    Anon(MemMapFlags),
}

pub(super) struct PageTable {
    table: HashMap<PageNumber, PageEntry>,
}

impl PageTable {
    pub(super) fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    // TODO(low priority) make this type erased by getting a DynCpuFabricRef type
    //                    instead of strong_cpu_fabric and cpu_fabric closures
    unsafe fn map_inner(
        &mut self,
        cpu_fabric: &dyn Fn() -> CpuFabricWeak<dyn ICache>,
        strong_cpu_fabric: &dyn Fn() -> CpuFabric<dyn ICache>,
        pages: PageTableAccess,
        mem_prot: MemProt,
        region: MapRegion,
    ) -> Result<(), MemoryFault> {
        for page in pages.iter() {
            ensure!(vaddr: page.vaddr_base(), !self.table.contains_key(&page))
        }

        let strong_cpu_fabric = std::cell::LazyCell::new(strong_cpu_fabric);
        let start_page = pages.start_page;
        for page in pages.iter() {
            let page_offset = || unsafe { page.get().unchecked_sub(start_page.get()) };

            let mapped = match region {
                MapRegion::Object {
                    shared,
                    object: ref dev,
                } => PageEntry::new_obj(
                    Arc::clone(dev),
                    unsafe { PageNumber::from_page_number_unchecked(page_offset()) },
                    shared,
                    mem_prot,
                ),
                MapRegion::Extern { flags, base_ptr } => unsafe {
                    let offset = u64_to_usize(page_offset()).unwrap_unchecked();
                    let page_ptr = base_ptr.add_pages(offset);
                    match flags {
                        MemMapFlags::Private => {
                            PageEntry::new_extern(page_ptr, mem_prot, cpu_fabric)
                        }
                        MemMapFlags::Cow => {
                            PageEntry::new_extern_cow(page_ptr, mem_prot, cpu_fabric)
                        }
                        MemMapFlags::Shared => {
                            PageEntry::new_extern_shared(page_ptr, mem_prot, cpu_fabric)
                        }
                    }
                },
                MapRegion::Anon(flags) => match flags {
                    MemMapFlags::Private => PageEntry::new_zeroed(mem_prot, cpu_fabric),
                    MemMapFlags::Cow => PageEntry::new_zeroed_cow(mem_prot, &strong_cpu_fabric),
                    MemMapFlags::Shared => PageEntry::new_zeroed_shared(mem_prot, cpu_fabric),
                },
            };

            let old_page = self.table.insert(page, mapped);
            if old_page.is_some() {
                abort()
            }
        }

        Ok(())
    }

    pub(super) unsafe fn map<T: ?Sized + ICache>(
        &mut self,
        cpu_fabric: &CpuFabric<T>,
        pages: PageTableAccess,
        mem_prot: MemProt,
        region: MapRegion,
    ) -> Result<(), MemoryFault> {
        unsafe {
            self.map_inner(
                &move || cpu_fabric.downgrade_dyn(),
                &move || cpu_fabric.clone().into_dyn(),
                pages,
                mem_prot,
                region,
            )
        }
    }

    pub(super) fn unmap(&mut self, pages: PageTableAccess, mut removed: impl FnMut(PageEntry)) {
        for page in pages.iter() {
            if let Entry::Occupied(page_entry) = self.table.entry(page) {
                removed(page_entry.remove());
            }
        }
    }

    pub(super) fn modify(
        &mut self,
        pages: PageTableAccess,
        mut modify: impl FnMut(PageNumber, &mut PageEntry),
    ) -> Result<(), MemoryFault> {
        for page in pages.iter() {
            ensure!(vaddr: page.vaddr_base(), self.table.contains_key(&page))
        }

        for page in pages.iter() {
            let page_entry = self.table.get_mut(&page).unwrap_or_else(|| abort());
            modify(page, page_entry)
        }

        Ok(())
    }

    pub(super) fn get_page(&self, page_num: PageNumber) -> Result<Page<'_>, MemoryFault> {
        self.table
            .get(&page_num)
            .and_then(|page| page.as_page())
            .ok_or_else(|| MemoryFault::general_protection(page_num.vaddr_base()))
    }

    // TODO(low priority) have a set tracking all executable pages / memory_object pages
    //      meaning cache invalidation can happen faster;
    //      its low priority because cache invalidation doesn't happen often
    pub(super) fn pages(&self) -> impl Iterator<Item = (PageNumber, &PageEntry)> {
        self.table
            .iter()
            .map(|(&page_num, entry)| (page_num, entry))
    }

    pub(super) fn pages_mut(&mut self) -> impl Iterator<Item = (PageNumber, &mut PageEntry)> {
        self.table
            .iter_mut()
            .map(|(&page_num, entry)| (page_num, entry))
    }

    pub(super) fn fork(&mut self) -> Self {
        let table = self
            .table
            .iter_mut()
            .map(|(&page, entry)| (page, entry.fork()))
            .collect::<HashMap<PageNumber, PageEntry>>();

        Self { table }
    }
}
