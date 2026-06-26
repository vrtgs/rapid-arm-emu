use crate::cpu_fabric::exclusive_monitor::ExclusiveMonitor;
use crate::cpu_fabric::object_manager::ObjectManager;
use crate::icache::ICache;
use crate::page_table::MemoryBackedPage;
use emu_abi::abort::AbortGuard;
use emu_abi::internal_traits::InitInPlace;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::sync::{Arc, Weak};

pub mod exclusive_monitor;
pub(crate) mod object_manager;

// dirty page manager
pub(crate) struct CpuFabricInner<T: ?Sized + ICache> {
    monitor: ExclusiveMonitor,
    dma_async_flusher: ObjectManager,
    zero_page: std::sync::OnceLock<Arc<MemoryBackedPage>>,
    instruction_cache: T,
}

#[repr(transparent)]
pub struct CpuFabric<T: ?Sized + ICache>(Arc<CpuFabricInner<T>>);

pub(crate) struct CpuFabricWeak<T: ?Sized + ICache>(Weak<CpuFabricInner<T>>);

impl<T: Sized + ICache> CpuFabricWeak<T> {
    pub(crate) const fn new() -> Self {
        const { Self(Weak::new()) }
    }

    pub(crate) fn into_dyn<'a>(self) -> CpuFabricWeak<dyn ICache + 'a>
    where
        T: 'a,
    {
        CpuFabricWeak(self.0)
    }
}

impl<T: ?Sized + ICache> CpuFabricWeak<T> {
    pub(crate) fn is(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn upgrade(&self) -> Option<CpuFabric<T>> {
        Some(CpuFabric(self.0.upgrade()?))
    }
}

impl<T: ?Sized + ICache> Clone for CpuFabricWeak<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

macro_rules! into_dyn {
    (
        $(ref $({ $ref_tt: tt })?;)?
        generic_T: $T: ty,
        name: $func_name: ident,
        ret: $ret: ident,
        wrapper: $wrapper: ty,
        create_wrapper: |$this: ident| $make_wrapper: expr $(,)?
    ) => {
        pub(crate) fn $func_name<'a>($(& $($ref_tt)?)? self) -> $ret<dyn ICache + 'a>
            where $T: 'a
        {
            let $this = self;

            #[cfg(any(false$(, true $($ref_tt)?)?))]
            let expr = $make_wrapper;

            // FIXME(with_metadata_of) this is a hack at best to get around no metadata helpers
            let metadata = $this
                .0
                .instruction_cache
                .get_metadata_ptr()
                .with_addr(0);

            #[cfg(not(any(false$(, true $($ref_tt)?)?)))]
            let expr = $make_wrapper;

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

            let guard = AbortGuard(());

            let this = <$wrapper>::into_raw(expr).cast::<()>();

            unsafe { *fat_ptr.get_unchecked_mut(data_ptr_index) = this }

            let new_ptr = unsafe {
                core::mem::transmute::<[*const (); 2], *const CpuFabricInner<dyn ICache + 'a>>(fat_ptr)
            };

            let new_arc = unsafe { <$wrapper>::from_raw(new_ptr) };

            guard.disarm();

            $ret(new_arc)
        }
    };
}

impl<T: ?Sized + ICache> CpuFabric<T> {
    into_dyn!(
        generic_T: T,
        name: into_dyn,
        ret: CpuFabric,
        wrapper: Arc<_>,
        create_wrapper: |this| this.0,
    );

    into_dyn!(
        ref;
        generic_T: T,
        name: downgrade_dyn,
        ret: CpuFabricWeak,
        wrapper: Weak<_>,
        create_wrapper: |this| Arc::downgrade(&this.0),
    );

    pub(crate) fn downgrade(&self) -> CpuFabricWeak<T> {
        CpuFabricWeak(Arc::downgrade(&self.0))
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
            monitor: ExclusiveMonitor,
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
    pub fn icache(&self) -> &T {
        &self.0.instruction_cache
    }

    pub(crate) fn zero_page(&self) -> &std::sync::OnceLock<Arc<MemoryBackedPage>> {
        &self.0.zero_page
    }

    pub(crate) fn object_manager(&self) -> &ObjectManager {
        &self.0.dma_async_flusher
    }
}
