use std::time::Instant;
use eyre::Result;

mod instruction_parser;


fn main() -> Result<()> {
    let start = Instant::now();
    let insns = instruction_parser::load_instruction_pages()?;

    let elapsed = start.elapsed();

    println!("{:#?}", insns[1]);

    println!("Took: {elapsed:?}");

    Ok(())
}
