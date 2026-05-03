use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::{BufRead, Read};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::atomic;
use std::sync::atomic::AtomicBool;
use compact_str::CompactString;
use eyre::{bail, ensure, ContextCompat, Result};
use rayon::iter::IntoParallelIterator;
use tempfile::TempDir;
use crate::instruction_parser::isa::{Isa, IsaEnum, A64};
use crate::instruction_parser::system_register::{SystemRegister, SystemRegisters};
use crate::instruction_parser::tar_ball::TarFileEntry;
use crate::interner::{Interner, Symbol};

mod tar_ball;
pub mod isa;
pub mod operand;
pub mod system_register;


const A64_ISA_ARCHIVE: &str = "spec/ISA_A64_xml_A_profile-2026-03_96.tar.gz";
const A64_ISA_XML_FOLDER: &str = "ISA_A64_xml_A_profile_2026-03_96-2026-03_rel";
const EXPECTED_ISA_IFORM_DTD: &[u8] = include_bytes!("expected-iform-p.dtd");
const IFORM_FILE_NAME: &str = "iform-p.dtd";


#[derive(Debug, Copy, Clone)]
pub struct AliasInfo {
    pub instruction_id: Symbol,
    pub condition: Symbol,
}

#[derive(Debug, Copy, Clone)]
pub struct FeatureExpression {
    pub id: Symbol,
    pub expression: Symbol,
    pub name: Symbol
}

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum InstructionClass {
    System,
    Other
}

#[derive(Debug, Copy, Clone)]
pub struct FeatureRequirement {
    expression: FeatureExpression,
    id_index: u32,
}

#[derive(Debug)]
pub struct BitRangeInfo {}

#[derive(Debug)]
pub struct Instruction<Arch: Isa> {
    pub id: Symbol,
    pub mnemonic: CompactString,
    pub name: CompactString,
    pub filename: CompactString,

    /// `true` if this instruction has been discarded by an alias that is preferred.
    pub is_discarded_by_preferred_alias: bool,
    /// `true` if this instruction is an alias with a dynamic condition.
    pub is_alias_with_dynamic_condition: bool,
    /// `true` if this instruction is aliased by other instructions
    /// from [`Self::alias_in`] and require a dynamic resolution.
    pub has_aliases_in_and_requiring_dynamic_resolution: bool,
    /// `true` if this instruction is an alias that is preferred by a more selective bit mask.
    pub is_alias_preferred_by_more_selective_bitmask: bool,

    /// When `Some`, this instruction instance is an alias for another instruction
    /// defined by [`AliasInfo::instruction_id`].
    /// when the given [`AliasInfo::condition`] is met.
    pub alias: Option<AliasInfo>,

    /// Gets the list of instructions this instruction is aliased from.
    pub alias_in: Vec<AliasInfo>,

    /// the syntax of the operands of this instruction.
    pub operands_syntax: Symbol,
    /// the full syntax of this instruction.
    // format!("{:<11} {}", self.mnemonic, self.operands_syntax)
    pub full_syntax: CompactString,

    pub summary: CompactString,
    pub class: InstructionClass,

    pub other_docvars: Option<HashMap<Symbol, Symbol>>,

    pub bitfield_mask: u32,
    pub bitfield_value: u32,
    pub not_bitfield_mask: u32,
    pub not_bitfield_value: u32,

    // merge of [JsonConverter(typeof(JsonUIntToHexConverter))]
    //  public uint BitfieldValueForTest { get; set; }
    //
    //  public bool IsBitFieldValueTestable { get; set; } = true;
    pub bitfield_value_for_test: Option<u32>,

    pub feature_requirement: Option<FeatureRequirement>,

    pub asm_template: CompactString,

    pub bit_ranges: Vec<BitRangeInfo>,
    pub operands: Vec<Arch::Operand>,
    pub use_operand_encoding_8bytes: bool,

    _marker: PhantomData<Arch>
}

impl<Arch: Isa> Instruction<Arch> {
    fn parse(interner: &Interner, filename: &str, file: impl BufRead) -> Result<Self> {
        let _ = (interner, filename, file);
        todo!()
    }
    
    pub fn gett_id(interner: &Interner, id: &str) -> Symbol {
        let id = id.trim_end_matches('_');
        
        let mut id = CompactString::new(id);
        match id.find('_') {
            Some(index) if index > 0 => {
                let (prefix, suffix) = id.split_at_mut(index);
                prefix.make_ascii_uppercase();
                suffix.make_ascii_lowercase();
            }
            _ => id.make_ascii_uppercase(),
        };
        
        interner.get_or_intern(&id)
    }
}


pub struct InstructionSet<Arch: Isa> {
    instruction_id_idx: HashMap<Symbol, usize>,
    instructions: Vec<Instruction<Arch>>,
    system_registers: SystemRegisters,
}

impl<Arch: Isa> InstructionSet<Arch> {
    pub fn instructions(&self) -> &[Instruction<Arch>] {
        &self.instructions
    }

    pub fn system_registers(&self) -> &SystemRegisters {
        &self.system_registers
    }
}


fn load_instruction_files_from_archive<Arch: Isa>(
    tempdir: &TempDir,
    interner: &Interner,
) -> Result<Vec<Instruction<Arch>>> {
    let arch_name = Arch::NAME;
    let (archive_path, xml_folder) = match Arch::AS_ENUM {
        IsaEnum::A64 => (A64_ISA_ARCHIVE, A64_ISA_XML_FOLDER)
    };

    let tar_stream = tar_ball::open_tar_gz_archive(archive_path)?;
    let found_iform_dtd = AtomicBool::new(false);
    let mut instructions = tar_stream
        .into_par_iter()
        .map(|entry| {
            let mut entry = entry?;
            let path =  entry.path();
            let Ok(file_name) = path.strip_prefix(xml_folder) else {
                return Ok(None)
            };

            let file_name = file_name
                .file_name()
                .map(Path::new)
                .filter(|&name| name == file_name);

            if file_name.is_some_and(|path| path.as_os_str() == OsStr::new(IFORM_FILE_NAME)) {
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                let was_found = found_iform_dtd.swap(true, atomic::Ordering::Relaxed);
                ensure!(!was_found, "duplicate {IFORM_FILE_NAME} files found in ISA folder");
                ensure!(data == EXPECTED_ISA_IFORM_DTD, "the {arch_name} ISA format changed!");
                return Ok(None)
            }

            let insn_name = file_name
                .filter(|file_name| file_name.extension() == Some(OsStr::new("xml")))
                .and_then(|name| name.file_stem())
                .map(|name| name.to_str().ok_or_else(|| {
                    eyre::eyre!("invalid instruction name `{}`", name.display())
                }))
                .transpose()?;

            insn_name
                .map(|name| name.trim().to_owned())
                .map(|name| Instruction::parse(interner, &name, entry))
                .transpose()
        })
        .filter_map(|res| res.transpose())
        .collect::<Result<Vec<Instruction<Arch>>>>()?;

    ensure!(
        found_iform_dtd.into_inner(),
        "there must be a `{IFORM_FILE_NAME}` file in ISA folder for {arch_name}"
    );

    instructions.sort_by(|a, b| {
        let (a_name, b_name) = (&a.filename, &b.filename);
        str::cmp(a_name, b_name)
    });

    let mut duplicates = None::<HashSet<&str>>;
    for [a, b] in instructions.array_windows::<2>() {
        if a.filename == b.filename {
            duplicates.get_or_insert_with(HashSet::new).insert(a.filename.as_str());
        }
    }

    if let Some(duplicates) = duplicates {
        bail!("duplicate instructions: {duplicates:?}")
    }

    for insn in &instructions {
        ensure!({
            let mut iter = insn.filename.bytes();
            iter.next().is_some_and(|x| x.is_ascii_lowercase())
                && iter.all(|x| matches!(x, b'a'..=b'z' | b'0'..=b'9' | b'_'))
        })
    }

    Ok(instructions)
}


pub struct InstructionSets {
    pub aarch64: InstructionSet<A64>,
}


pub fn load_instruction_sets(tempdir: &TempDir, interner: &Interner) -> Result<InstructionSets> {
    let a64_instruction_files = load_instruction_files_from_archive::<A64>(
        tempdir,
        interner
    )?;

    todo!()
}