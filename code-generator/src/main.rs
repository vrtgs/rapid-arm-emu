#![forbid(unsafe_code)]


use eyre::Result;
use crate::instruction_parser::InstructionSets;
use crate::interner::Interner;

mod interner;
mod instruction_parser;


fn main() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let interner = Interner::new();
    let InstructionSets { aarch64: _, .. } = instruction_parser::load_instruction_sets(
        &temp_dir,
        &interner
    )?;

    Ok(())
}
