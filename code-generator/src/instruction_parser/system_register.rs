use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter, Write};
use std::num::NonZero;
use std::ops::{BitOr, BitOrAssign};
use std::sync::LazyLock;
use compact_str::CompactString;
use tempfile::TempDir;
use crate::instruction_parser::{tar_ball, Interner, Symbol};
use crate::instruction_parser::isa::{Isa, IsaEnum};

type InnerBits = u32;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct UsageKind(NonZero<InnerBits>);

impl UsageKind {
    #[track_caller]
    fn from_index_inner(index: usize) -> Self {
        let kind = u32::try_from(index)
            .ok()
            .and_then(|index| NonZero::new((1 as InnerBits).unbounded_shl(index)))
            .expect("too many system register usages registered");

        Self(kind)
    }

    pub fn get_index(self) -> usize {
        self.0.trailing_zeros() as usize
    }
}

const USAGE_KINDS: u8 = 23;

const _: () = assert!(USAGE_KINDS as u32 <= InnerBits::BITS);


struct UsageKindRegistry {
    name_to_kind: HashMap<&'static str, UsageKind>,
    kind_to_str: [&'static str; USAGE_KINDS as usize],
}

impl UsageKindRegistry {
    pub fn new() -> Self {
        let usages = [
            "APAS",
            "AT",
            "BRB",
            "CFP",
            "COSP",
            "CPP",
            "DC",
            "DVP",
            "GCSPOPCX",
            "GCSPOPM",
            "GCSPOPX",
            "GCSPUSHM",
            "GCSPUSHX",
            "GCSSS1",
            "GCSSS2",
            "IC",
            "MRRS",
            "MRS",
            "MSR",
            "MSRR",
            "TLBI",
            "TLBIP",
            "TRCIT",
        ];


        let map = usages
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, UsageKind::from_index_inner(i)))
            .collect::<HashMap<_, _>>();

        assert_eq!(usages.len(), map.len());

        UsageKindRegistry {
            name_to_kind: map,
            kind_to_str: usages,
        }
    }
}

static USAGE_KIND_REGISTRY: LazyLock<UsageKindRegistry> = LazyLock::new(UsageKindRegistry::new);

impl UsageKind {
    fn get(usage_kind: &str) -> Option<Self> {
        let mut compact_str;
        let str = match usage_kind.bytes().any(|ch| ch.is_ascii_lowercase()) {
            true => {
                compact_str = CompactString::new(usage_kind);
                compact_str.make_ascii_uppercase();
                compact_str.as_str()
            },
            false => usage_kind,
        };

        USAGE_KIND_REGISTRY.name_to_kind.get(str).copied()
    }

    pub fn name(self) -> &'static str {
        USAGE_KIND_REGISTRY.kind_to_str[self.get_index()]
    }
}

impl Debug for UsageKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("UsageKind")
            .field("name", &self.name())
            .field("index", &self.get_index())
            .finish()
    }
}


#[derive(Copy, Clone, Eq, PartialEq)]
pub struct UsageKinds(InnerBits);

impl UsageKinds {
    pub const fn empty() -> Self {
        Self(0)
    }

    pub fn iter(self) -> impl Iterator<Item=UsageKind> {
        let mut current = self.0;
        core::iter::from_fn(move || {
            let isolate_lowest_one = |value: InnerBits| value & value.wrapping_neg();
            let ret = isolate_lowest_one(current);
            // ret.ones == 1 OR 0
            debug_assert!(ret.count_ones() <= 1);
            current ^= ret;
            NonZero::new(ret).map(UsageKind)
        })
    }

    pub const fn from_kind(kind: UsageKind) -> Self {
        Self(kind.0.get())
    }

    pub const fn with(self, kind: UsageKind) -> Self {
        self.union(Self::from_kind(kind))
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOr<UsageKind> for UsageKinds {
    type Output = Self;

    fn bitor(self, rhs: UsageKind) -> Self::Output {
        self.with(rhs)
    }
}

impl BitOr<UsageKinds> for UsageKinds {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl<Rhs> BitOrAssign<Rhs> for UsageKinds
    where Self: BitOr<Rhs, Output=Self>
{
    fn bitor_assign(&mut self, rhs: Rhs) {
        *self = (*self) | rhs
    }
}


impl Debug for UsageKinds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}


#[derive(Copy, Clone)]
pub struct SystemRegister {
    pub name: Symbol,
    pub description: Symbol,
    pub value: u16,
    pub usage_kinds: UsageKinds,
}

impl SystemRegister {
    pub fn bits_diplay(&self) -> impl Display {
        let value = self.value;
        core::fmt::from_fn(move |f| {
            // 0bxx_xxx_xxxx_xxxx_xxx
            f.write_str("0b")?;

            for i in (0..16_u32).rev() {
                if i == 13 || i == 10 || i == 6 || i == 2 {
                    f.write_char('_')?;
                }

                let char = b'0' + ((value >> i) & 1) as u8;
                f.write_char(char as char)?;
            }

            Ok(())
        })
    }

    pub fn debug(&self, interner: &Interner) -> impl Debug {
        core::fmt::from_fn(move |f| {
            f.debug_struct("SystemRegister")
                .field("name", &interner.resolve(self.name).as_str())
                .field("description", &interner.resolve(self.description).as_str())
                .field("value", &format_args!("{:04X}", self.value))
                .field("bits", &format_args!("{}", self.bits_diplay()))
                .field("usage_kinds", &self.usage_kinds)
                .finish()
        })
    }
}


pub struct SystemRegisters {
    system_register_usage_kinds: UsageKinds,
    system_register_values: HashSet<u16>,
    system_registers_not_handled: HashSet<Symbol>,
    system_registers_handled: HashSet<Symbol>,
    system_registers: HashMap<Symbol, SystemRegister>
}


const SYSTEM_REG_FOLDER: &str = "SysReg_xml_A_profile_2026-03_96-2026-03_rel";
const SYSTEM_REG_ARCHIVE: &str = "spec/SysReg_xml_A_profile-2026-03_96.tar.gz";

impl SystemRegisters {
    fn _load(
        _temp_dir: &TempDir,
        interner: &Interner,
        isa_name: &'static str,
    ) -> eyre::Result<Self> {
        println!("Processing System Registers");

        let archive = tar_ball::open_tar_gz_archive(SYSTEM_REG_ARCHIVE)?;
        for entry in archive {
            let entry = entry?;

        }

        todo!()
    }

    pub fn load<Arch: Isa>(temp_dir: &TempDir, interner: &Interner) -> eyre::Result<Self> {
        let isa_name = match Arch::AS_ENUM {
            IsaEnum::A64 => "AArch64",
        };
        Self::_load(temp_dir, interner, isa_name)
    }
}