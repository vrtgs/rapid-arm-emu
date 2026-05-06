use crate::io_mmu::IoMMU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IoMmuStatus {
    Ok = 0,
    Fault = 1,
}

macro_rules! impl_io_mmu_load_ints {
    ($({ func: $fun_name: ident, load_fn: $load_fn_name: ident, ty: $ty: ty })+) => {
        $(
            pub unsafe extern "C" fn $fun_name(
                io_mmu: *const IoMMU,
                addr: u64,
                out: *mut $ty,
            ) -> IoMmuStatus {
                unsafe {
                    match io_mmu.as_ref_unchecked().$load_fn_name(addr) {
                        Ok(byte) => {
                            std::ptr::write(out, byte);
                            IoMmuStatus::Ok
                        }
                        Err(_) => IoMmuStatus::Fault
                    }
                }
            }
        )+
    };
}

impl_io_mmu_load_ints!(
    {
        func: io_mmu_load64_le,
        load_fn: load64_le,
        ty: u64
    }
    {
        func: io_mmu_load32_le,
        load_fn: load32_le,
        ty: u32
    }
    {
        func: io_mmu_load16_le,
        load_fn: load16_le,
        ty: u16
    }
);

macro_rules! impl_io_mmu_store_ints {
    ($({ func: $fun_name: ident, store_fn: $store_fn_name: ident, ty: $ty: ty })+) => {
        $(
            pub unsafe extern "C" fn $fun_name(
                io_mmu: *const IoMMU,
                addr: u64,
                value: $ty,
            ) -> IoMmuStatus {
                unsafe {
                    match io_mmu.as_ref_unchecked().$store_fn_name(addr, value) {
                        Ok(()) => IoMmuStatus::Ok,
                        Err(_) => IoMmuStatus::Fault
                    }
                }
            }
        )+
    };
}


impl_io_mmu_store_ints!(
    {
        func: io_mmu_store64_le,
        store_fn: store64_le,
        ty: u64
    }
    {
        func: io_mmu_store32_le,
        store_fn: store32_le,
        ty: u32
    }
    {
        func: io_mmu_store16_le,
        store_fn: store16_le,
        ty: u16
    }
);
