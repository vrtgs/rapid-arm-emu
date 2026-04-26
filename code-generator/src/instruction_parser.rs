use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::io::{Read, Seek};
use std::path::Path;
use eyre::{bail, Context, ContextCompat, Result};

#[derive(Debug)]
pub struct InstructionPage {
    pub file_name: Box<str>,
    pub id: Box<str>,
    pub title: Box<str>,
    pub section_type: Box<str>,
    pub docvars: HashMap<Box<str>, Box<str>>,
    pub heading: Option<Box<str>>,
    pub brief: Option<Box<str>>,
    pub authored: Option<Box<str>>,
    pub operational_notes: Vec<Box<str>>,
}

fn attr(node: &roxmltree::Node, name: &str) -> Result<Box<str>> {
    node.attribute(name)
        .map(Box::<str>::from)
        .with_context(|| format!("missing attribute `{name}`"))
}

fn normalize_text(text: &str) -> Box<str> {
    text.split_whitespace().collect::<Vec<_>>().join(" ").into_boxed_str()
}

fn child_text(node: &roxmltree::Node, tag: &str) -> Option<Box<str>> {
    node.children()
        .find(|n| n.has_tag_name(tag))
        .and_then(|n| n.text())
        .map(normalize_text)
}

fn descendant_text(node: &roxmltree::Node, tag: &str) -> Option<Box<str>> {
    node.descendants()
        .find(|n| n.has_tag_name(tag))
        .and_then(|n| n.text())
        .map(normalize_text)
}

fn parse_instruction_section(file_name: Box<str>, xml: File) -> Result<Option<InstructionPage>> {
    let xml = {
        let mut file_contents = String::new();
        let mut xml = xml;
        xml.rewind()?;
        xml.read_to_string(&mut file_contents)?;
        file_contents
    };

    let xml = xml.as_str();

    let parse_options = roxmltree::ParsingOptions {
        allow_dtd: true,
        nodes_limit: u32::MAX,
        entity_resolver: Some(&|a, b| {
            dbg!(a, b);
            todo!()
        })
    };

    let doc = roxmltree::Document::parse_with_options(xml, parse_options)
        .context("failed to parse XML document")?;

    let root = doc
        .root()
        .children()
        .find(|node| node.has_tag_name("instructionsection"));

    let Some(root) = root else {
        return Ok(None)
    };

    let id = attr(&root, "id")?;
    let title = attr(&root, "title")?;
    let section_type = attr(&root, "type")?;

    let mut docvars_map = HashMap::new();

    let docvars = root
        .children()
        .find(|n| n.has_tag_name("docvars"))
        .into_iter()
        .flat_map(|a| a.children());

    for node in docvars {
        if let (Some(key), Some(value)) = (node.attribute("key"), node.attribute("value")) {
            if docvars_map.insert(Box::from(key), Box::from(value)).is_some() {
                bail!("duplicate docvar {key}")
            }
        }
    }

    let docvars = docvars_map;

    let heading = child_text(&root, "heading");

    let brief = root
        .descendants()
        .find(|n| n.has_tag_name("brief"))
        .and_then(|n| descendant_text(&n, "para"));

    let authored = root
        .descendants()
        .find(|n| n.has_tag_name("authored"))
        .and_then(|n| descendant_text(&n, "para"));

    let operational_notes = root
        .descendants()
        .filter(|n| n.has_tag_name("operationalnote_content"))
        .map(|n| normalize_text(n.text().unwrap_or_default()))
        .filter(|s| !s.is_empty())
        .collect();

    Ok(Some(InstructionPage {
        file_name,
        id,
        title,
        section_type,
        docvars,
        heading,
        brief,
        authored,
        operational_notes,
    }))
}

const ISA_ARCHIVE: &str = "spec/ISA_A64_xml_A_profile-2026-03_96.tar.gz";
const ISA_XML_FOLDER: &str = "ISA_A64_xml_A_profile_2026-03_96-2026-03_rel";
const EXPECTED_ISA_IFORM_DTD: &[u8] = include_bytes!("expected-iform-p.dtd");


const MAX_INSTRUCTIONS_DEBUG: usize = 128;

fn debug_instruction_list(instructions: &[InstructionPage]) {
    let mut debug_list = instructions
        .iter()
        .map(|insn| &insn.file_name)
        .take(MAX_INSTRUCTIONS_DEBUG)
        .map(|s| s as &dyn Debug)
        .collect::<Vec<&dyn Debug>>();

    let len = instructions.len();

    let overflow_list_trail = format_args!("...");

    if len > MAX_INSTRUCTIONS_DEBUG {
        debug_list.push(&overflow_list_trail);
    }

    assert!(len > 50, "i know there are at least 50 instructions in the ARM isa");

    println!("found {len} instructions: {debug_list:?}")
}


pub fn load_instruction_pages() -> Result<Vec<InstructionPage>> {
    let compressed_data_reader = File::open(ISA_ARCHIVE)?;
    let tar_reader = flate2::read::MultiGzDecoder::new(compressed_data_reader);
    let mut tar = tar::Archive::new(tar_reader);

    let mut iform_dtd = None;
    let mut instructions = Vec::<(Box<str>, File)>::new();
    for entry in tar.entries()? {
        let mut entry = entry?;
        let path =  entry.path()?;
        let Ok(file_name) = path.strip_prefix(ISA_XML_FOLDER) else {
            continue
        };

        let file_name = file_name
            .file_name()
            .map(Path::new)
            .filter(|&name| name == file_name);

        if file_name.is_some_and(|path| path.as_os_str() == OsStr::new("iform-p.dtd")) {
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            iform_dtd = Some(data);
            continue
        }

        let insn_name = file_name
            .filter(|file_name| file_name.extension() == Some(OsStr::new("xml")))
            .and_then(|name| name.file_stem())
            .map(|name| name.to_str().ok_or_else(|| {
                eyre::eyre!("invalid instruction name `{}`", name.display())
            }))
            .transpose()?;

        if let Some(insn) = insn_name {
            let name = Box::<str>::from(insn.trim());
            let mut file = tempfile::tempfile()?;
            io::copy(&mut entry, &mut file)?;
            instructions.push((name, file));
        }
    }

    let iform = iform_dtd.expect("there should always be a dtd file in the ISA folder");
    eyre::ensure!(iform == EXPECTED_ISA_IFORM_DTD);

    instructions.sort_unstable_by(|(a_name, _), (b_name, _)| {
        str::cmp(a_name, b_name)
    });

    let mut duplicates = None::<HashSet<&str>>;
    for [(a_name, _), (b_name, _)] in instructions.array_windows::<2>() {
        if a_name == b_name {
            duplicates.get_or_insert_with(HashSet::new).insert(a_name);
        }
    }

    if let Some(duplicates) = duplicates {
        bail!("duplicate instructions: {duplicates:?}")
    }

    for (insn_file_name, _) in &instructions {
        eyre::ensure!({
            let mut iter = insn_file_name.bytes();
            iter.next().is_some_and(|x| x.is_ascii_lowercase())
                && iter.all(|x| matches!(x, b'a'..=b'z' | b'0'..=b'9' | b'_'))
        })
    }

    let instructions = instructions;

    let instructions = instructions
        .into_iter()
        .flat_map(|(file_name, file)| {
            parse_instruction_section(file_name, file).transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    debug_instruction_list(&instructions);

    Ok(instructions)
}