use std::hint::cold_path;

// FIXME feature(const_convert)
macro_rules! make_checked_usize_cast {
    ($from: ident => $to: ident) => {
        pastey::paste! {
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

make_checked_usize_cast! { u64 => usize }
make_checked_usize_cast! { usize => u64 }

#[inline(always)]
pub const fn u64_add_usize(x: u64, y: usize) -> Option<u64> {
    let Some(y) = usize_to_u64(y) else {
        cold_path();
        return None;
    };
    x.checked_add(y)
}
