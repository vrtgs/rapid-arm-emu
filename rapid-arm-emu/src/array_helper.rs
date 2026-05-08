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

pub fn empty_iter<T, const N: usize>() -> IntoIter<T, N> {
    ArrayVec::new_const().into_iter()
}
