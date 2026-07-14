use std::hint::cold_path;

// FIXME feature(const_convert)
macro_rules! make_checked_usize_cast {
    ($from: ident => $to: ident) => {
        pastey::paste! {
            #[doc = concat!(
                "Converts [`", stringify!($from), "`] to [`", stringify!($to), "`].\n",
                "\n",
                "Returns `None` if the value does not fit in the target type."
            )]
            #[allow(
                clippy::cast_possible_truncation,
                reason = "this function ensures no truncation happens"
            )]
            #[inline(always)]
            pub const fn [<$from _to_ $to>](int: $from) -> Option<$to> {
                match $to::BITS >= $from::BITS {
                    true => Some(int as $to),
                    false => {
                        // this would be a widening cast
                        let max: $from = $to::MAX as $from;
                        if int > max {
                            cold_path();
                            return None
                        }
                        Some(int as $to)
                    },
                }
            }
        }
    };
}

macro_rules! make_unchecked_cast {
    ($from: ident => $to: ident) => {
        pastey::paste! {
            #[doc = concat!(
                "Converts [`", stringify!($from), "`] to [`", stringify!($to), "`].\n",
                "\n",
                "Returns `None` if the value does not fit in the target type."
            )]
            #[allow(
                clippy::cast_possible_truncation,
                reason = "this function ensures no truncation happens"
            )]
            #[inline(always)]
            pub const fn [<$from _to_ $to>](int: $from) -> $to {
                assert!($from::BITS <= $to::BITS);
                int as $to
            }
        }
    };
}

make_checked_usize_cast! { usize => u128 }
make_checked_usize_cast! { u128 => usize }

make_checked_usize_cast! { usize => u64 }
make_checked_usize_cast! { u64 => usize }

make_checked_usize_cast! { usize => u32 }
make_checked_usize_cast! { u32 => usize }

make_checked_usize_cast! { usize => u16 }
make_checked_usize_cast! { usize => u8 }

make_unchecked_cast! { u16 => usize }
make_unchecked_cast! { u8 => usize }

/// Adds a [`usize`] offset to a [`u64`], returning `None` if `x + y`
/// does not fit in a `u64`.
#[inline(always)]
pub const fn u64_add_usize(x: u64, y: usize) -> Option<u64> {
    let Some(y) = usize_to_u64(y) else {
        cold_path();
        return None;
    };
    x.checked_add(y)
}
