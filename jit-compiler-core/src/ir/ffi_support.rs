use crate::exec_context::{ExecContext, MemOp};
use emu_abi::memory::{HostPointer, MemProt, Tlb};
use io_mmu::fault::MemoryFault;
use io_mmu::icache::ICache;
use std::hint::cold_path;
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
            pub(super) unsafe extern "C" fn [<load $suffix>](
                io_mmu: &IoMMU<'_>,
                tlb: &mut Tlb,
                context: &mut ExecContext,
                addr: u64,
                out: &mut MaybeUninit<$ty>,
            ) -> IoMmuStatus {
                match io_mmu.[<load $suffix>](tlb, addr) {
                    Ok(value) => unsafe {
                        std::ptr::write(out.as_mut_ptr(), value);
                        IoMmuStatus::Ok
                    },
                    Err(error) => {
                        // note: it is possible for `error.vaddr() != addr`
                        // for example, if the load crosses a page
                        context.current_mem_fault.set_fault(MemOp::Load, error);
                        IoMmuStatus::Fault
                    }
                }
            }

            pub(super) unsafe extern "C" fn [<store $suffix>](
                io_mmu: &IoMMU<'_>,
                tlb: &mut Tlb,
                context: &mut ExecContext,
                addr: u64,
                value: $ty,
            ) -> IoMmuStatus {
                match io_mmu.[<store $suffix>](tlb, addr, value) {
                    Ok(()) => IoMmuStatus::Ok,
                    Err(error) => {
                        // note: it is possible for `error.vaddr() != addr`
                        // for example, if the store crosses a page
                        context.current_mem_fault.set_fault(MemOp::Store, error);
                        IoMmuStatus::Fault
                    }
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

pub(super) unsafe extern "C" fn clrex(context: &mut ExecContext) {
    context.exclusive_monitor_reservation.take();
}

#[inline(always)]
fn resolve_addr_exclusive<const SIZE: usize>(
    io_mmu: &IoMMU<'_>,
    tlb: &mut Tlb,
    vaddr: u64,
    prot: MemProt,
) -> Result<HostPointer, MemoryFault> {
    io_mmu
        .resolve_aligned_scalar_access::<SIZE>(tlb, vaddr)
        .and_then(|(page, offset)| {
            if !page.ptr.flags().contains_any(prot.into()) {
                cold_path();
                return Err(MemoryFault::general_protection(vaddr));
            }

            Ok(HostPointer(unsafe { page.ptr.page_ptr().byte_add(offset) }))
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum StrexStatus {
    Stored = 0,
    Failed = 1,
    Fault = u8::MAX,
}

trait HasFault {
    const FAULT: Self;
}

impl HasFault for IoMmuStatus {
    const FAULT: Self = IoMmuStatus::Fault;
}

impl HasFault for StrexStatus {
    const FAULT: Self = StrexStatus::Fault;
}

macro_rules! resolve {
    ($ty:ty; $io_mmu: ident, $tlb: ident, $vaddr: ident, $context: ident; $flag:ident as $op_ty: ident) => {
        match resolve_addr_exclusive::<{ size_of::<$ty>() }>($io_mmu, $tlb, $vaddr, MemProt::$flag)
        {
            Ok(host_ptr) => host_ptr,
            Err(err) => {
                $context.current_mem_fault.set_fault(MemOp::$op_ty, err);
                return <_ as HasFault>::FAULT;
            }
        }
    };
}

macro_rules! atomic_ref {
    ($ptr: ident as $ty: ident) => {
        // FIXME(SOUNDNESS): same issue as with `memops` and mixed sized atomics
        unsafe {
            $ptr
                .0
                .as_ptr()
                .cast::<pastey::paste!(std::sync::atomic::[<Atomic $ty:upper>])>()
                .cast_const()
                .as_ref_unchecked()
        }
    };
}

macro_rules! make_ld_st_rex_pair {
    (load: $ldrex: ident, store: $strex: ident; $ty: ident) => {
// LDREX and LDAXR both use this; so always default to SeqCst
// and ldrex and strex are already expensive to emulate paying for a SeqCst barrier is nothing
pub(super) extern "C" fn $ldrex(
    io_mmu: &IoMMU<'_>,
    tlb: &mut Tlb,
    context: &mut ExecContext,
    vaddr: u64,
    out: &mut MaybeUninit<$ty>,
) -> IoMmuStatus {

    let ptr = resolve!($ty; io_mmu, tlb, vaddr, context; READ as Load);
    let (value, token) = io_mmu.get_fabric().exclusive_monitor().ldrex(
        ptr,
        // note that the reservation is made based on the reading of a native endian value
        || {
            let atomic_ref = atomic_ref!(ptr as $ty);
            atomic_ref.load(std::sync::atomic::Ordering::SeqCst)
        }
    );

    context.exclusive_monitor_reservation = Some(token);
    out.write(value.to_le());

    IoMmuStatus::Ok
}

pub(super) unsafe extern "C" fn $strex(
    io_mmu: &IoMMU<'_>,
    tlb: &mut Tlb,
    context: &mut ExecContext,
    vaddr: u64,
    value: $ty,
) -> StrexStatus {
    let Some(reservation) = context.exclusive_monitor_reservation.take() else {
        return StrexStatus::Failed;
    };


    use io_mmu::cpu_fabric::exclusive_monitor::{ExclusiveMonitorLoad as EML, ReservationLost};

    pastey::paste! {
        let EML::[<$ty:upper>](old_value) = reservation.value else {
            return StrexStatus::Failed;
        };
    }

    let ptr = resolve!($ty; io_mmu, tlb, vaddr, context; WRITE as Store);

    // convert value to little endian to store it back
    // note: `old_value` is stored in little endian byte order,
    // `new_value`, on the other hand, is stored as native endian byte order,
    // therefore, it needs to be converted before it can be used for a CAS
    let new_value = value.to_le();
    let strex_res = io_mmu
        .get_fabric()
        .exclusive_monitor()
        .strex(ptr, reservation.version, || {
            use std::sync::atomic::Ordering;

            let val = atomic_ref!(ptr as $ty);
            match val.compare_exchange(old_value, new_value, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => Ok(()),
                Err(_) => Err(ReservationLost),
            }
        });


    match strex_res {
        Ok(()) => StrexStatus::Stored,
        Err(ReservationLost {}) => StrexStatus::Failed
    }
}
    };
}

// FIXME: currently strex and ldrex don't have identical behaviour when paired with str
//        this is because if ldrex loads some value A and another thread (or current) does str
//        on the same location with the value of A, the reservation isn't lost and lingers
//        and the subsequent strex can succeed, preserving this behaviour is niche
//        and is delayed until the dynarec is more mature
make_ld_st_rex_pair!(load: ldrexb, store: strexb; u8);
make_ld_st_rex_pair!(load: ldrexh, store: strexh; u16);
make_ld_st_rex_pair!(load: ldrex, store: strex; u32);
make_ld_st_rex_pair!(load: ldrexd, store: strexd; u64);

// TODO implement LDXP and STXP for 64-bit pairs
