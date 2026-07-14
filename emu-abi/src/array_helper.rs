use arrayvec::{ArrayVec, IntoIter};

/// Converts a fixed-size array `[T; N]` into an [`ArrayVec<T, M>`].
///
/// `N` must be <= `M`; this is verified at compile time.
#[inline]
pub fn from_arr<T, const N: usize, const M: usize>(array: [T; N]) -> ArrayVec<T, M> {
    const { assert!(N <= M) }

    let mut vec = ArrayVec::<T, M>::new_const();
    for item in array {
        unsafe { vec.push_unchecked(item) }
    }

    vec
}

/// Returns an [`IntoIter`] over the elements of a fixed-size array.
///
/// The underlying [`ArrayVec`] capacity is `M`; `N` must be <= `M`.
#[inline]
pub fn iter_from_arr<T, const N: usize, const M: usize>(array: [T; N]) -> IntoIter<T, M> {
    from_arr(array).into_iter()
}

/// Returns an empty [`ArrayVec<T, N>`].
#[inline(always)]
pub const fn empty<T, const N: usize>() -> ArrayVec<T, N> {
    ArrayVec::new_const()
}

/// Returns an empty [`IntoIter`] with capacity `N`.
#[inline(always)]
pub fn empty_iter<T, const N: usize>() -> IntoIter<T, N> {
    empty().into_iter()
}
