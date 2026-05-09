use arrayvec::{ArrayVec, IntoIter};

#[inline]
pub fn from_arr<T, const N: usize, const M: usize>(array: [T; N]) -> ArrayVec<T, M> {
    const { assert!(N <= M) }

    let mut vec = ArrayVec::<T, M>::new_const();
    for item in array {
        unsafe { vec.push_unchecked(item) }
    }

    vec
}

#[inline]
pub fn iter_from_arr<T, const N: usize, const M: usize>(array: [T; N]) -> IntoIter<T, M> {
    from_arr(array).into_iter()
}

#[inline(always)]
pub const fn empty<T, const N: usize>() -> ArrayVec<T, N> {
    ArrayVec::new_const()
}

#[inline(always)]
pub fn empty_iter<T, const N: usize>() -> IntoIter<T, N> {
    empty().into_iter()
}
