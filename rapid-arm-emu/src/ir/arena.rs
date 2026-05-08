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

#[derive(Copy, Clone, Ord, PartialOrd, PartialEq, Eq, Hash)]
pub struct RawHandle(NonZero<HandleInt>);

impl Debug for RawHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawHandle")
            .field("index", &self.get())
            .finish()
    }
}

const _: () = assert!(HandleInt::BITS <= usize::BITS);
const _: () = assert!(HandleInt::MIN == 0);

#[allow(clippy::cast_possible_truncation)]
const fn usize_to_int(x: usize) -> Option<HandleInt> {
    if x >= HandleInt::MAX as usize {
        cold_path();
        return None;
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
                None => None,
            },
            None => None,
        }
    }

    #[track_caller]
    pub const fn new(index: usize) -> Self {
        match Self::try_new(index) {
            Some(handle) => handle,
            None => panic!("SSA handle overflow"),
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

    unsafe {
        ManuallyDrop::into_inner(
            Transmute {
                from: ManuallyDrop::new(from),
            }
            .to,
        )
    }
}

pub const fn to_raw<H: Handle>(handle: H) -> RawHandle {
    unsafe { transmute_unckecked::<H, RawHandle>(handle) }
}

pub const fn from_raw<H: Handle>(handle: RawHandle) -> H {
    unsafe { transmute_unckecked::<RawHandle, H>(handle) }
}

unsafe impl Handle for RawHandle {}

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

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (S::Handle, &S)> {
        self.assert_invariants();
        self.0
            .iter()
            .enumerate()
            .map(|(i, item)| (from_raw(RawHandle::new(i)), item))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = S::Handle> {
        self.assert_invariants();
        (0..self.len())
            .map(RawHandle::new)
            .map(from_raw::<S::Handle>)
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
    pub fn reserve(&mut self) -> (S::Handle, Reservation<'_, S>) {
        let handle = RawHandle::new(self.0.len());
        let handle = from_raw::<S::Handle>(handle);
        let reservation = Reservation::try_reserve(self).expect("TODO: OOM handling");
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

impl<S: Debug> Debug for Arena<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <[S] as Debug>::fmt(&self.0, f)
    }
}

pub struct ArenaMap<K, V>(Vec<Option<V>>, PhantomData<K>);

impl<K: Handle, V> ArenaMap<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(
            Vec::from_iter(std::iter::repeat_with(|| None).take(capacity)),
            PhantomData,
        )
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn get(&self, key: K) -> Option<&V> {
        self.0.get(to_raw(key).get()).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
        self.0.get_mut(to_raw(key).get()).and_then(Option::as_mut)
    }

    fn insertion_slot(&mut self, key: K) -> &mut Option<V> {
        let index = to_raw(key).get();
        if self.0.len() <= index {
            cold_path();
            self.0.resize_with(index.strict_add(1), || None);
        }

        &mut self.0[index]
    }

    pub fn get_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, with: F) -> &mut V {
        self.insertion_slot(key).get_or_insert_with(with)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.insertion_slot(key).replace(value)
    }

    pub fn insert_unique(&mut self, key: K, value: V) {
        let old = self.insert(key, value);
        assert!(old.is_none())
    }

    pub fn remove(&mut self, key: K) -> Option<V> {
        self.0.get_mut(to_raw(key).get()).and_then(Option::take)
    }

    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> {
        self.0.iter().enumerate().filter_map(|(i, val)| {
            val.as_ref()
                .map(|val| (from_raw::<K>(RawHandle::new(i)), val))
        })
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn indexing_map_handle_failed<T>() -> T {
    panic!("key not found in map")
}

// DO NOT IMPLEMENT IndexMut
impl<K: Handle, V> Index<K> for ArenaMap<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        self.get(index).unwrap_or_else(indexing_map_handle_failed)
    }
}

pub struct ArenaSet<K>(Vec<usize>, PhantomData<K>);

impl<K: Handle> Clone for ArenaSet<K> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), self.1)
    }

    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

impl<K: Handle> ArenaSet<K> {
    const BITS: NonZero<usize> = {
        let bits: u32 = usize::BITS;
        assert!(bits <= 256);
        NonZero::new(bits as usize).unwrap()
    };

    fn reserve(&mut self, capacity: usize) {
        let words = capacity.div_ceil(Self::BITS.get());
        if self.0.len() < words {
            self.0.resize(words, 0)
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut this = Self(vec![], PhantomData);
        this.reserve(capacity);
        this
    }

    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    fn index_and_mask(key: K) -> (usize, usize) {
        let key = to_raw(key).get();
        let index = key / Self::BITS;
        let bit = key % Self::BITS;
        (index, 1_usize << bit)
    }

    pub fn insert(&mut self, key: K) -> bool {
        let (index, mask) = Self::index_and_mask(key);
        self.reserve(index.strict_add(1));
        let word = &mut self.0[index];
        let was_there = (*word) & mask;
        *word |= mask;

        was_there == 0
    }

    pub fn contains(&self, key: K) -> bool {
        let (index, mask) = Self::index_and_mask(key);
        let row = self.0.get(index).copied().unwrap_or(0);
        (row & mask) != 0
    }

    pub fn remove(&mut self, key: K) -> bool {
        let (index, mask) = Self::index_and_mask(key);
        let Some(row) = self.0.get_mut(index) else {
            return false;
        };

        let old_row = *row;
        *row = old_row & !mask;
        (old_row & mask) != 0
    }

    pub fn iter(&self) -> impl Iterator<Item = K> {
        self.0
            .iter()
            .flat_map(|&bits| {
                std::array::from_fn::<_, { ArenaSet::<RawHandle>::BITS.get() }, _>(|i| {
                    bits & (1_usize << i)
                })
            })
            .enumerate()
            .filter_map(|(i, mask)| (mask != 0).then_some(i))
            .map(RawHandle::new)
            .map(from_raw::<K>)
    }
}

impl<K: Handle> FromIterator<K> for ArenaSet<K> {
    fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _upper) = iter.size_hint();
        let mut this = ArenaSet::with_capacity(lower);
        for key in iter {
            this.insert(key);
        }
        this
    }
}

impl<K: Handle + Debug> Debug for ArenaSet<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
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
        impl usize like for $name: path;
    ) => {
        impl $name {
            pub const fn new(index: usize) -> Self {
                let raw = $crate::ir::arena::RawHandle::new(index);
                $crate::ir::arena::from_raw(raw)
            }

            pub const fn get(self) -> usize {
                $crate::ir::arena::to_raw(self).get()
            }
        }
    };

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
    {
        $ty: ty as $(impl $vis: vis $impl_name: ident)? $(($existing_handle: path))?;
        $(init: {
            $(const $const_name: ident = $init: expr;)*
        })?
    } => {

        $($crate::ir::arena::make_handle!($vis $impl_name);)?


        $($crate::ir::arena::handle_impl_helper! {
            impl const for $impl_name {
                $(const $const_name;)*
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

                0 $(+ {
                    assert!(
                        is_impl,
                        "if storable is using an exisiting handle, it can't add const"
                    );

                    #[allow(non_snake_case, unused_variables)]
                    {
                        <[()]>::len(&[$({ let $const_name: (); }),*])
                    }
                })?
            };

            impl $crate::ir::arena::Storable for $ty {
                type Handle = $($existing_handle)? $($impl_name)?;

                const INITIAL_VEC_LEN: usize = LEN;

                fn initial_vec() -> Vec<Self> {
                    ::std::vec![$($($init),*)?]
                }
            }
        };

    };
}

#[doc(hidden)]
#[allow(unused_imports)]
pub(super) use handle_impl_helper;

pub(super) use {impl_storable, make_handle};
