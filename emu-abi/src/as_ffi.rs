/// This allows exposing things only inbetween the internal crates
pub trait AsFFI {
    type Inetrface<'a>
    where
        Self: 'a;

    fn as_ffi(&self) -> Self::Inetrface<'_>;
}
