//! Shared per-machine state connecting CPUs, MMUs, and memory.
//!
//! A [`CpuFabric`] bundles the state that every [`IoMMU`](crate::IoMMU) of an
//! emulated machine must agree on: the global exclusive monitor backing
//! load-/store-exclusive semantics, the asynchronous
//! [`MemoryObject`](crate::memory_object::MemoryObject) fault/flush worker(s),
//! the shared zero page, and the machine's
//! [`ICache`]. Cloning a `CpuFabric` is cheap and
//! yields another handle to the same underlying fabric.

use crate::cpu_fabric::exclusive_monitor::ExclusiveMonitor;
use crate::cpu_fabric::object_manager::ObjectManager;
use crate::icache::ICache;
use crate::page_table::MemoryBackedPage;
use emu_abi::abort::AbortGuard;
use emu_abi::internal_traits::InitInPlace;
use emu_abi::memory::PagePointer;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ops::Deref;
use std::sync::{Arc, Weak};

pub mod exclusive_monitor;
pub(crate) mod object_manager;

// dirty page manager
pub(crate) struct CpuFabricInner<T: ?Sized + ICache> {
    exclusive_monitor: ExclusiveMonitor,
    dma_async_flusher: ObjectManager,
    zero_page: std::sync::OnceLock<Arc<MemoryBackedPage>>,
    instruction_cache: T,
}

/// A cheaply clonable handle to the shared state of one emulated machine.
///
/// All [`IoMMU`](crate::IoMMU)s that belong to the same machine must be
/// created from clones of the same `CpuFabric`; see the [module
/// docs](self) for what the fabric contains. Two handles compare equal with
/// [`PartialEq`] iff they refer to the same underlying fabric.
#[repr(transparent)]
pub struct CpuFabric<T: ?Sized + ICache>(Arc<CpuFabricInner<T>>);

pub(crate) struct CpuFabricWeak(Weak<CpuFabricInner<dyn ICache>>);

pub(crate) struct DynCpuFabricRef<'a> {
    cache: ManuallyDrop<CpuFabric<dyn ICache>>,
    _life: PhantomData<&'a CpuFabric<dyn ICache>>,
}

impl CpuFabricWeak {
    #[inline(always)]
    pub(crate) const fn invalid() -> Self {
        enum VoidCache {}

        impl ICache for VoidCache {
            fn invalidate(&self, _: PagePointer) {
                match *self {}
            }
        }

        const { Self(Weak::<CpuFabricInner<VoidCache>>::new()) }
    }
}

impl CpuFabricWeak {
    pub(crate) fn is(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn upgrade(&self) -> Option<CpuFabric<dyn ICache>> {
        Some(CpuFabric(self.0.upgrade()?))
    }
}

impl Clone for CpuFabricWeak {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Deref for DynCpuFabricRef<'_> {
    type Target = CpuFabric<dyn ICache>;

    fn deref(&self) -> &Self::Target {
        &self.cache
    }
}

impl DynCpuFabricRef<'_> {
    pub(crate) fn downgrade(&self) -> CpuFabricWeak {
        CpuFabricWeak(Arc::downgrade(&self.0))
    }
}

macro_rules! into_dyn {
    (
        $(ref $({ $ref_tt: tt })?;)?
        generic_T: $T: ty,
        name: $func_name: ident,
        ret: $ret: path,
        into_raw: |$wrapper: ident| $into_raw: expr,
        from_raw: |$raw_wrapper: ident| $from_raw: expr,
        create_wrapper: |$this: ident| $make_wrapper: expr $(,)?
    ) => {
        pub(crate) fn $func_name($(& $($ref_tt)?)? self) -> $ret {
            let $this = self;

            #[cfg(any(false$(, true $($ref_tt)?)?))]
            let $wrapper = $make_wrapper;


            // Note: it is impossible for this call to panic
            // that is a safety requirement for implementing `get_metadata_ptr`
            let metadata_ptr = $this.0.instruction_cache.get_metadata_ptr();

            #[cfg(not(any(false$(, true $($ref_tt)?)?)))]
            let $wrapper = $make_wrapper;

            let guard = AbortGuard(());

            // FIXME(with_metadata_of) this is a hack at best to get around no metadata helpers
            let metadata = metadata_ptr.with_addr(0);

            let mut fat_ptr = unsafe {
                core::mem::transmute::<*const dyn ICache, [*const (); 2]>(metadata)
            };

            let data_ptr_index = {
                let [i0, i1] = fat_ptr;
                match (i0.is_null(), i1.is_null()) {
                    (true, false) => 0_usize,
                    (false, true) => 1_usize,
                    _ => emu_abi::abort::abort(),
                }
            };


            let this: *const () = $into_raw;

            unsafe { *fat_ptr.get_unchecked_mut(data_ptr_index) = this }

            let $raw_wrapper = unsafe {
                core::mem::transmute::<[*const (); 2], *const CpuFabricInner<dyn ICache>>(fat_ptr)
            };

            let new_arc = unsafe { $from_raw };

            guard.disarm();

            new_arc
        }
    };
}

impl<T: ?Sized + ICache> CpuFabric<T> {
    into_dyn!(
        generic_T: T,
        name: into_dyn,
        ret: CpuFabric<dyn ICache>,
        into_raw: |arc| Arc::into_raw(arc).cast::<()>(),
        from_raw: |ptr_with_meta| CpuFabric(Arc::from_raw(ptr_with_meta)),
        create_wrapper: |this| this.0,
    );

    into_dyn!(
        ref;
        generic_T: T,
        name: as_dyn_ref,
        ret: DynCpuFabricRef<'_>,
        into_raw: |arc| Arc::into_raw(ManuallyDrop::into_inner(arc)).cast::<()>(),
        from_raw: |ptr_with_meta| DynCpuFabricRef {
            cache: ManuallyDrop::new(CpuFabric(Arc::from_raw(ptr_with_meta))),
            _life: PhantomData
        },
        create_wrapper: |this| ManuallyDrop::new(unsafe { std::ptr::read(&this.0) }),
    );
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
    /// Creates a new [`CpuFabric`] object
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
                    let manually_drop: &mut ManuallyDrop<$ty> = unsafe {
                        &mut *(init_ref as *mut $ty as *mut ManuallyDrop<$ty>)
                    };

                    InitGuard(manually_drop)
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
            exclusive_monitor: ExclusiveMonitor,
            dma_async_flusher: ObjectManager,
            zero_page: std::sync::OnceLock<Arc<MemoryBackedPage>>,
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

impl<T: ?Sized + ICache> CpuFabric<T> {
    /// Returns the machine's instruction cache.
    pub fn icache(&self) -> &T {
        &self.0.instruction_cache
    }

    /// Returns the machine's instruction cache.
    pub fn exclusive_monitor(&self) -> &ExclusiveMonitor {
        &self.0.exclusive_monitor
    }

    pub(crate) fn zero_page(&self) -> &std::sync::OnceLock<Arc<MemoryBackedPage>> {
        &self.0.zero_page
    }

    pub(crate) fn object_manager(&self) -> &ObjectManager {
        &self.0.dma_async_flusher
    }
}
