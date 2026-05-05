use std::fmt::Debug;
use std::hash::Hash;
use std::hint;
use std::hint::cold_path;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZero;
use std::ops::{Index, IndexMut};

cfg_select! {
    target_pointer_width = "16" => {
        type HandleInt = u32;
    }
    _ => {
        type HandleInt = u32;
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct RawHandle(NonZero<HandleInt>);

const _: () = assert!(HandleInt::BITS <= usize::BITS);
const _: () = assert!(HandleInt::MIN == 0);

#[allow(clippy::cast_possible_truncation)]
const fn usize_to_int(x: usize) -> Option<HandleInt> {
    if x >= HandleInt::MAX as usize {
        cold_path();
        return None
    }

    Some(x as HandleInt)
}

#[allow(clippy::cast_possible_truncation)]
const fn int_to_usize(int: HandleInt) -> usize {
    int as usize
}

impl RawHandle {
    pub const fn try_new(index: usize) -> Option<Self> {
        match usize_to_int(index) {
            Some(int) => match NonZero::new(int.wrapping_add(1)) {
                Some(nz) => Some(Self(nz)),
                None => None
            },
            None => None
        }
    }

    #[track_caller]
    pub const fn new(index: usize) -> Self {
        match Self::try_new(index) {
            Some(handle) => handle,
            None => panic!("SSA handle overflow")
        }
    }

    pub const fn inc(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(nz) => Some(Self(nz)),
            None => None
        }
    }

    pub const fn get(self) -> usize {
        int_to_usize(unsafe { self.0.get().unchecked_sub(1) })
    }
}


/// # Safety
///
/// `Self` must have the same layout as `RawHandle`
/// this can be achived with `#[repr(transparent)]`
/// like so:
/// ```rs
/// #[repr(transparent)]
/// struct Handle(RawHandle);
/// ```
pub unsafe trait Handle: Copy {}

const unsafe fn transmute_unckecked<T, U>(from: T) -> U {
    union Transmute<T, U> {
        from: ManuallyDrop<T>,
        to: ManuallyDrop<U>,
    }

    unsafe { ManuallyDrop::into_inner(Transmute { from: ManuallyDrop::new(from) }.to) }
}

pub const fn to_raw<H: Handle>(handle: H) -> RawHandle {
    unsafe { transmute_unckecked::<H, RawHandle>(handle) }
}

pub const fn from_raw<H: Handle>(handle: RawHandle) -> H {
    unsafe { transmute_unckecked::<RawHandle, H>(handle) }
}


pub trait Storable: Sized {
    type Handle: Handle;

    const INITIAL_VEC_LEN: usize = 0;

    fn initial_vec() -> Vec<Self> {
        vec![]
    }
}


pub struct Arena<S>(Vec<S>);


impl<S: Storable> Arena<S> {
    pub fn new() -> Self {
        let vec = S::initial_vec();
        assert_eq!(vec.len(), S::INITIAL_VEC_LEN);

        const {
            if S::INITIAL_VEC_LEN != 0 {
                assert!(RawHandle::try_new(S::INITIAL_VEC_LEN - 1).is_some())
            }
        }

        Self(vec)
    }

    fn assert_invariants(&self) {
        unsafe {
            hint::assert_unchecked(self.0.len() >= S::INITIAL_VEC_LEN);
            hint::assert_unchecked(self.0.len() <= int_to_usize(HandleInt::MAX));
        }
    }

    pub fn get(&self, handle: S::Handle) -> Option<&S> {
        self.assert_invariants();
        self.0.get(to_raw(handle).get())
    }

    pub fn get_mut(&mut self, handle: S::Handle) -> Option<&mut S> {
        self.assert_invariants();
        self.0.get_mut(to_raw(handle).get())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub struct Reservation<'a, S>(&'a mut Arena<S>);

impl<'a, S: Storable> Reservation<'a, S> {
    fn try_reserve(arena: &'a mut Arena<S>) -> Option<Self> {
        arena.assert_invariants();
        arena.0.try_reserve(1).ok()?;
        Some(Self(arena))
    }
    
    pub fn store(self, item: S) -> &'a mut S {
        unsafe { hint::assert_unchecked(self.0.0.len() < self.0.0.capacity()) }
        self.0.0.push_mut(item)
    }
}

impl<S: Storable> Arena<S> {
    pub fn reserve(&mut self) -> (S::Handle, Reservation<S>) {
        let handle = RawHandle::new(self.0.len());
        let handle = from_raw::<S::Handle>(handle);
        let reservation = Reservation::try_reserve(self)
            .expect("TODO: OOM handling");
        (handle, reservation)
    }
    
    pub fn store_mut(&mut self, item: S) -> (S::Handle, &mut S) {
        let (handle, reserve) = self.reserve();
        (handle, reserve.store(item))
    }

    pub fn store(&mut self, item: S) -> S::Handle {
        self.store_mut(item).0
    }
}



#[cold]
#[inline(never)]
#[track_caller]
fn indexing_handle_failed<T>() -> T {
    panic!("invalid handle used on arena")
}


impl<S: Storable> Index<S::Handle> for Arena<S> {
    type Output = S;

    #[track_caller]
    fn index(&self, index: S::Handle) -> &Self::Output {
        self.get(index).unwrap_or_else(indexing_handle_failed)
    }
}

impl<S: Storable> IndexMut<S::Handle> for Arena<S> {
    #[track_caller]
    fn index_mut(&mut self, index: S::Handle) -> &mut Self::Output {
        self.get_mut(index).unwrap_or_else(indexing_handle_failed)
    }
}

pub struct ArenaMap<K, V>(Vec<Option<V>>, PhantomData<K>);

impl<K: Handle, V> ArenaMap<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity), PhantomData)
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.0.get(to_raw(key).get()).map_or(None, Option::as_ref)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let index = to_raw(key).get();
        if self.0.len() <= index {
            cold_path();
            self.0.resize_with(index.strict_add(1), || None);
        }
        self.0[index].replace(value)
    }
}


#[cold]
#[inline(never)]
#[track_caller]
fn indexing_map_handle_failed<T>() -> T {
    panic!("key not found in map")
}

impl<K: Handle, V> Index<K> for ArenaMap<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        self.get(index).unwrap_or_else(indexing_map_handle_failed)
    }
}


macro_rules! make_handle {
    ($vis: vis $name: ident) => {
        #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
        #[repr(transparent)]
        $vis struct $name($crate::ir::arena::RawHandle);

        const _: () = {
            unsafe impl $crate::ir::arena::Handle for $name {}
        };
    };
}

macro_rules! handle_impl_helper {
    (
        impl const for $name: path {
            $(const $const_name: ident;)*
        }
    ) => {
        impl $name {
            $crate::ir::arena::handle_impl_helper! {
                @fold
                current_stack: [],
                munching: { $(const $const_name;)* }
            }
        }
    };
    
    (
        @fold
        current_stack: [$($tt:tt),*],
        munching: {
            const $const_name: ident;
            $($rest:tt)*
        }
    ) => {
        const $const_name: Self = {
            match $crate::ir::arena::RawHandle::try_new(<[()]>::len(&[$($tt),*])) {
                Some(x) => $crate::ir::arena::from_raw(x),
                None => panic!("too many constants declared")
            }
        };

        $crate::ir::arena::handle_impl_helper! {
            @fold
            current_stack: [$($tt,)* ()],
            munching: { $($rest)* }
        }
    };

    (
        @fold
        current_stack: [$($tt:tt),*],
        munching: {}
    ) => {};
}

macro_rules! impl_storable {
    (
        $ty: ty as $(impl $vis: vis $impl_name: ident)? $(($existing_handle: path))?$(;$(init: {
            $(const $const_name: ident = $init: expr;)*
        })?)?
    ) => {

        $($crate::ir::arena::make_handle!($vis $impl_name);)?


        $($crate::ir::arena::handle_impl_helper! {
            impl const for $impl_name {
                $($(const $const_name;)*)?
            }
        })?

        const _: () = { 
            #[allow(dead_code)]
            const LEN: usize = {
                #[allow(unused_variables, unused_mut)]
                let mut is_impl = false;
                $(
                let _ = ::core::stringify!($impl_name);
                is_impl |= true;
                )?

                0 $($(+ {
                    assert!(
                        is_impl,
                        "if storable is using an exisiting handle, it can't add const"
                    );

                    #[allow(non_snake_case, unused_variables)]
                    {
                        <[()]>::len(&[$({ let $const_name: (); }),*])
                    }
                })?)?
            };

            impl $crate::ir::arena::Storable for $ty {
                type Handle = $($existing_handle)? $($impl_name)?;

                const INITIAL_VEC_LEN: usize = LEN;

                fn initial_vec() -> Vec<Self> {
                    ::std::vec![$($($($init),*)?)?]
                }
            }
        };

    };
}


#[doc(hidden)]
#[allow(unused_imports)]
pub(super) use {handle_impl_helper};

pub(super) use {make_handle, impl_storable};
