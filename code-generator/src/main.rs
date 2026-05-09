#![forbid(unsafe_code)]
#![allow(dead_code, unused)]

use crate::instruction_parser::InstructionSets;
use crate::interner::Interner;
use eyre::Result;

mod instruction_parser;
mod interner;

fn main() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let interner = Interner::new();
    let InstructionSets { aarch64: _, .. } =
        instruction_parser::load_instruction_sets(&temp_dir, &interner)?;

    Ok(())
}
