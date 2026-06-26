use emu_abi::memory::Tlb;
use io_mmu::icache::ICache;
use std::mem::MaybeUninit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum IoMmuStatus {
    Ok = 0,
    Fault = 1,
}

type IoMMU<'a> = io_mmu::IoMMU<dyn ICache + 'a>;

macro_rules! impl_load_store {
    ($({ suffix: $suffix: tt, ty: $ty: ty $(,)? })*) => {
        pastey::paste! {$(
            pub(super) unsafe extern "C" fn [<io_mmu_load $suffix>](
                io_mmu: &IoMMU<'_>,
                tlb: &mut Tlb,
                addr: u64,
                out: &mut MaybeUninit<$ty>,
            ) -> IoMmuStatus {
                match io_mmu.[<load $suffix>](tlb, addr) {
                    Ok(value) => unsafe {
                        std::ptr::write(out.as_mut_ptr(), value);
                        IoMmuStatus::Ok
                    },
                    Err(_) => IoMmuStatus::Fault,
                }
            }

            pub(super) unsafe extern "C" fn [<io_mmu_store $suffix>](
                io_mmu: &IoMMU<'_>,
                tlb: &mut Tlb,
                addr: u64,
                value: $ty,
            ) -> IoMmuStatus {
                match io_mmu.[<store $suffix>](tlb, addr, value) {
                    Ok(()) => IoMmuStatus::Ok,
                    Err(_) => IoMmuStatus::Fault,
                }
            }
        )*}
    };
}

impl_load_store! {
    {
        suffix: 64_le,
        ty: u64
    }
    {
        suffix: 32_le,
        ty: u32
    }
    {
        suffix: 16_le,
        ty: u16
    }
    {
        suffix: _byte,
        ty: u8
    }
}
