use std::mem::MaybeUninit;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(16))]
pub struct Vector(pub u128);

const _: () = assert!(align_of::<Vector>() == 16 && size_of::<Vector>() == 16);

pub const X_REGISTER_COUNT: u8 = 31;

#[derive(bytemuck::Zeroable, Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct PState(pub u32);

impl PState {
    pub const NEGATIVE: Self = Self(1 << 31);
    pub const ZERO: Self = Self(1 << 30);
    pub const CARRY: Self = Self(1 << 29);
    pub const OVERFLOW: Self = Self(1 << 28);

    pub const N: Self = Self::NEGATIVE;
    pub const Z: Self = Self::ZERO;
    pub const C: Self = Self::CARRY;
    pub const V: Self = Self::OVERFLOW;

    pub const NZCV_MASK: Self = Self(Self::N.0 | Self::Z.0 | Self::C.0 | Self::V.0);
}

#[derive(bytemuck::Zeroable)]
pub struct ProcessorState {
    pub sp: u64,
    pub pc: u64,
    pub x_registers: [u64; X_REGISTER_COUNT as usize],
    pub pstate: PState,
    pub fpsr: u32,
    pub fpcr: u32,
    pub vectors: [Vector; 32],
}

impl ProcessorState {
    #[inline(always)]
    pub const fn initial() -> Self {
        bytemuck::zeroed()
    }
}

#[derive(bytemuck::Zeroable)]
#[repr(C)]
pub struct ExecState {
    pub state: ProcessorState,
    pub trap_paylod: MaybeUninit<u64>,
}

impl ExecState {
    pub const fn initial() -> Self {
        const { Self::from_processor_state(ProcessorState::initial()) }
    }

    #[inline(always)]
    pub const fn from_processor_state(state: ProcessorState) -> Self {
        Self {
            state,
            trap_paylod: MaybeUninit::uninit(),
        }
    }
}
