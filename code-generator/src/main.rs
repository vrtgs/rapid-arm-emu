use std::collections::HashSet;
use eyre::Result;

mod instruction_parser;

fn main() -> Result<()> {
    let insns = instruction_parser::load_instruction_pages()?;

    let mut doc_var_combos = HashSet::new();

    for insn in insns.iter() {
        doc_var_combos.extend(insn.docvars.iter().filter(|(k, _)| ***k != *"mnemonic"))
    }

    println!("{:#?}", doc_var_combos);
    Ok(())
}
