use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::hash::Hash;
use std::io;
use std::io::{Read, Seek};
use std::path::Path;
use std::rc::Rc;
use eyre::{bail, ensure, Context, ContextCompat, Result};
use itertools::Itertools;
use markdown::mdast;
use roxmltree::NodeType;
use roxmltree::Node as XmlNode;


const ISA_ARCHIVE: &str = "spec/ISA_A64_xml_A_profile-2026-03_96.tar.gz";
const ISA_XML_FOLDER: &str = "ISA_A64_xml_A_profile_2026-03_96-2026-03_rel";
const EXPECTED_ISA_IFORM_DTD: &[u8] = include_bytes!("expected-iform-p.dtd");


struct ImageHandleInner {
    file: Rc<str>,
    data: RefCell<Option<Vec<u8>>>,
}

#[derive(Clone)]
pub struct ImageHandle(Rc<ImageHandleInner>);

impl Debug for ImageHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f
            .debug_struct("ImageHandle")
            .field("file", &self.0.file)
            .field("data_initialized", &self.0.data.borrow().is_some())
            .finish()
    }
}

struct ImageFactory {
    cache: HashMap<Rc<str>, ImageHandle>,
}

impl ImageFactory {
    fn new() -> Self {
        Self {
            cache: HashMap::new()
        }
    }

    fn make_handle(&mut self, file: &str) -> ImageHandle {
        self.cache.get(file).cloned().unwrap_or_else(|| {
            let file = Rc::<str>::from(file);
            let handle_ref = &*self.cache.entry(file)
                .or_insert_with_key(|file_name| {
                    ImageHandle(Rc::new(ImageHandleInner {
                        file: Rc::clone(file_name),
                        data: RefCell::new(None)
                    }))
                });
            handle_ref.clone()
        })
    }
}


#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Isa {
    A64,
    A32
}

#[derive(Debug)]
pub struct OperationalNote {
    titles: Vec<Box<str>>,
    content: MdSegment,
}


#[derive(Debug)]
enum BriefReliability {
    Verified,
    UnverifiedReliable,
    Unchecked,
}

#[derive(Debug)]
pub struct Brief {
    content: MdSegment,
    reliability: BriefReliability,
}

#[derive(Debug)]
pub struct Description {
    brief: Brief,
    authored: Option<MdSegment>,
    other: HashMap<Box<str>, String>,
}

#[derive(Debug)]
pub struct ExceptionGroup {
    pub group_name: Option<Box<str>>,
    pub exceptions: Vec<MdSegment>,
}

#[derive(Debug)]
pub struct InstructionPage {
    pub id: Box<str>,
    pub title: Box<str>,
    pub isa: Isa,
    pub docvars: HashMap<Box<str>, Box<str>>,
    pub heading: Box<str>,
    pub description: Description,
    pub operational_notes: Vec<OperationalNote>,
    pub exceptions: Vec<ExceptionGroup>,
}

fn attr<'a>(node: &XmlNode<'a, '_>, name: &str) -> Result<&'a str> {
    node.attribute(name).with_context(|| format!("missing attribute `{name}`"))
}

fn attr_owned(node: &XmlNode, name: &str) -> Result<Box<str>> {
    attr(node, name).map(Box::<str>::from)
}


fn filter_child_elements<'a, 'b: 'a>(parent: &XmlNode<'a, 'b>) -> impl DoubleEndedIterator<Item=Result<XmlNode<'a, 'b>>> {
    let name = match parent.node_type() {
        NodeType::Element => parent.tag_name().name(),
        NodeType::Comment => "comment",
        NodeType::Text => "text element",
        NodeType::PI => "processing instruction",
        NodeType::Root => "{root}"
    };
    parent.children().filter_map(move |n| {
        match n.node_type() {
            NodeType::Element => Some(Ok(n)),
            NodeType::Comment => None,
            NodeType::PI => {
                Some(Err(eyre::eyre!("unexpected processing instruction node")))
            },
            NodeType::Text => match n.text().unwrap().trim() {
                "" => None,
                text => {
                    Some(Err(eyre::eyre!("non empty text in {name} `{text}`")))
                }
            },
            NodeType::Root => {
                Some(Err(eyre::eyre!("unexpected root element in {name}")))
            }
        }
    })
}


const MAX_NESTING: u16 = 450;

struct MdParser<'a> {
    scratch_buffer: String,
    images: Vec<ImageHandle>,
    nesting_level: Rc<Cell<u16>>,
    image_factory: &'a mut ImageFactory,
}

#[derive(Debug)]
pub struct MdSegment {
    pub images: Vec<ImageHandle>,
    pub text: String,
}

impl<'images> MdParser<'images> {
    const LIST_SPREAD: bool = false;

    pub fn parse_pcdata_node<'a>(node: &XmlNode<'a, '_>) -> Result<&'a str> {
        match node.node_type() {
            NodeType::Text => {
                node.text().ok_or_else(|| eyre::eyre!("text node had no text"))
            },

            NodeType::Element => {
                let mut children = node.children();

                let child = children
                    .next()
                    .context("expected element with exactly one text child, got no children")?;

                ensure!(
                    children.next().is_none(),
                    "expected element with exactly one text child, got multiple children"
                );

                ensure!(
                    child.is_text(),
                    "expected element's only child to be text, got {:?}",
                    child.node_type()
                );

                child.text().ok_or_else(|| eyre::eyre!("text child had no text"))
            }

            other => bail!("expected text node or PCDATA element, got {:?}", other),
        }
    }

    fn parse_image_node(&mut self, node: &XmlNode) -> Result<mdast::Image> {
        ensure!(node.has_tag_name("image"));

        let unexpected_attr = node.attributes()
            .map(|attr| attr.name())
            .find(|&attr| !matches!(attr, "file" | "label"));

        if let Some(attr) = unexpected_attr {
            bail!("unexpected attribute {attr}")
        }

        let file = node.attribute("file").context("image node missing file")?;
        let label = node.attribute("label").context("image node missing label")?;

        let image_node = mdast::Image {
            position: None,
            alt: label.to_string(),
            url: file.to_string(),
            title: None,
        };

        let image = self.image_factory.make_handle(file);
        self.images.push(image);

        Ok(image_node)
    }

    fn parse_children(&mut self, node: &XmlNode) -> Result<Vec<mdast::Node>> {
        node.children()
            .flat_map(|child| self.parse_node(&child).transpose())
            .collect()
    }

    fn with_inner_xml<T>(&mut self, node: &XmlNode, op: impl FnOnce(&str) -> T) -> T {
        let mut iter = node.children();
        let inner_range = match iter.next() {
            Some(start) if let Some(end) = iter.next_back() => {
                start.range().start..end.range().end
            },
            Some(single) => single.range(),
            None => 0..0,
        };

        let input_xml = node.document().input_text();
        let inner_xml_content = input_xml[inner_range].trim();
        op(inner_xml_content)
    }

    fn with_scratch_buffer(&mut self, op: impl FnOnce(&mut String)) -> String {
        assert!(self.scratch_buffer.is_empty());
        op(&mut self.scratch_buffer);
        let str = Box::<str>::from(self.scratch_buffer.as_str()).into_string();
        self.scratch_buffer.clear();
        str
    }

    fn make_link(&mut self, label: &str, linkend: &str) -> mdast::Node {
        let url = self.with_scratch_buffer(|buf| {
            buf.push('#');
            *buf += linkend
        });

        mdast::Node::Link(mdast::Link {
            children: vec![mdast::Node::Text(mdast::Text {
                value: label.to_owned(),
                position: None,
            })],
            position: None,
            url,
            title: Some(linkend.to_owned()),
        })
    }

    fn parse_node(&mut self, node: &XmlNode) -> Result<Option<mdast::Node>> {
        ensure!(self.nesting_level.get() <= MAX_NESTING, "formated text went way too deep");
        self.nesting_level.update(|level| level.strict_add(1));

        struct DropNestingLvl(Rc<Cell<u16>>);
        impl Drop for DropNestingLvl {
            fn drop(&mut self) {
                self.0.update(|level| level.strict_sub(1))
            }
        }

        let _guard = DropNestingLvl(Rc::clone(&self.nesting_level));

        Ok(Some(match node.node_type() {
            NodeType::Text => {
                let text = node
                    .text()
                    .context("XML text node is missing text")?;


                let value = self.with_scratch_buffer(|buffer| {
                    let mut first_line = true;
                    for line in text.lines() {
                        if !core::mem::replace(&mut first_line, false) {
                            buffer.push('\n')
                        }

                        buffer.push_str(line.trim_start());
                    }
                });

                mdast::Node::Text(mdast::Text {
                    value,
                    position: None,
                })
            }

            NodeType::Element => match node.tag_name().name() {
                // Block-ish
                "para" | "content" => {
                    mdast::Node::Paragraph(mdast::Paragraph {
                        children: self.parse_children(node)?,
                        position: None,
                    })
                },

                "note" => {
                    mdast::Node::Blockquote(mdast::Blockquote {
                        children: self.parse_children(node)?,
                        position: None,
                    })
                }

                "image" => mdast::Node::Image(self.parse_image_node(node)?),

                "br" => mdast::Node::Break(mdast::Break {
                    position: None
                }),

                "list" => mdast::Node::List(self.push_list_node(node)?),

                "listitem" => {
                    let mut iter = filter_child_elements(node);
                    let content = iter
                        .next_back()
                        .context("list item missing content")
                        .and_then(|content_node| {
                            let content_node = content_node?;
                            ensure!(
                                content_node.has_tag_name("content"),
                                "misformatted list item"
                            );
                            Ok(content_node)
                        })?;

                    let mut children = Vec::with_capacity(1);

                    for child in iter {
                        let child = child?;
                        ensure!(matches!(child.tag_name().name(), "term" | "param"));
                        if let Some(node) = self.parse_node(&child)? {
                            children.push(node);
                        }
                    }

                    if let Some(node) = self.parse_node(&content)? {
                        children.push(node);
                    }

                    mdast::Node::ListItem(mdast::ListItem {
                        children,
                        position: None,
                        spread: Self::LIST_SPREAD,
                        checked: None,
                    })
                },


                // Inline formatting
                "b" => mdast::Node::Strong(mdast::Strong {
                    children: self.parse_children(node)?,
                    position: None,
                }),

                tag_name @ ("sub" | "sup") => {
                    let html_modified_children = self.parse_children(node)?;

                    if html_modified_children.is_empty() {
                        return Ok(None)
                    }

                    let html_content = mdast::Node::Root(mdast::Root {
                        children: html_modified_children,
                        position: None,
                    });

                    let html_inner = markdown::to_html(&html_content.to_string());

                    mdast::Node::Html(mdast::Html {
                        value: format!("<{tag_name}>{html_inner}</{tag_name}>"),
                        position: None,
                    })
                }

                // Inline code-ish
                "asm-code"
                | "instruction"
                | "literal"
                | "parameter"
                | "binarynumber"
                | "hexnumber"
                | "syntax"
                | "field"
                | "value"
                | "function"
                | "enum"
                | "enumvalue"
                | "arm-defined-word" => {
                    let code_str = self.with_inner_xml(node, str::to_owned);
                    match code_str.contains('\n') {
                        true => mdast::Node::Code(mdast::Code {
                            value: code_str,
                            position: None,
                            lang: None,
                            meta: None,
                        }),
                        false => mdast::Node::InlineCode(mdast::InlineCode {
                            value: code_str,
                            position: None
                        })
                    }
                }

                "xref" => {
                    let linkend = node.attribute("linkend")
                        .context("xref missing linkend")?;

                    let label = Self::parse_pcdata_node(node)?;

                    self.make_link(label, linkend)
                }

                "register_link" => {
                    let id = node.attribute("id")
                        .context("register_link missing id")?;
                    let label = Self::parse_pcdata_node(node).unwrap_or(id);
                    self.make_link(label, id)
                }

                "url" => {
                    let url = Self::parse_pcdata_node(node)?;
                    self.make_link(url, url)
                }

                // TODO
                "table" => {
                    self.with_inner_xml(node, |text| {
                        mdast::Node::Code(mdast::Code {
                            value: text.to_owned(),
                            position: None,
                            lang: Some("arm_table".to_owned()),
                            meta: None,
                        })
                    })
                },

                other => bail!("unsupported formatted text element <{other}>")
            },

            NodeType::Comment | NodeType::PI => return Ok(None),

            NodeType::Root => bail!("XML root can't be parsed as formatted text")
        }))
    }

    fn push_list_node(&mut self, node: &XmlNode) -> Result<mdast::List> {
        ensure!(node.has_tag_name("list"));

        let list_type = node.attribute("type")
            .context("list missing type")?;

        let start = match list_type {
            "ordered" => Some(1_u32),
            "unordered" | "var" | "param" => None,
            other => bail!("unsupported list type {other:?}"),
        };

        let list = filter_child_elements(node)
            .map(|res| res.and_then(|item| {
                ensure!(item.has_tag_name("listitem"));
                Ok(item)
            }));

        let children = list
            .flat_map(|res| match res {
                Ok(res) => self.parse_node(&res).transpose(),
                Err(err) => Some(Err(err))
            })
            .collect::<Result<Vec<mdast::Node>>>()?;

        Ok(mdast::List {
            children,
            position: None,
            ordered: start.is_some(),
            start,
            spread: false,
        })
    }


    fn parse_formatted_text_node(
        image_factory: &'images mut ImageFactory,
        node: &XmlNode,
    ) -> Result<MdSegment> {
        let mut this = MdParser {
            scratch_buffer: String::new(),
            images: vec![],
            nesting_level: Rc::new(Cell::new(0)),
            image_factory
        };

        let node_as_md_node = mdast::Node::Paragraph(mdast::Paragraph {
            children: this.parse_children(node)?,
            position: None,
        });

        let md_doc = mdast::Node::Root(mdast::Root {
            children: vec![
                node_as_md_node,
                mdast::Node::Break(mdast::Break { position: None }),
                mdast::Node::Text(mdast::Text {
                    value: "original AML:".to_owned(),
                    position: None
                }),
                mdast::Node::Break(mdast::Break { position: None }),
                mdast::Node::Code(mdast::Code {
                    value: this.with_inner_xml(node, str::to_owned),
                    position: None,
                    lang: None,
                    meta: None,
                })
            ],
            position: None,
        });

        debug_assert_eq!(this.nesting_level.get(), 0);

        // FIXME propper mdast to md conversion
        Ok(MdSegment {
            images: this.images,
            text: md_doc.to_string(),
        })
    }
}


fn parse_decription(img_factory: &mut ImageFactory, xml: XmlNode) -> Result<Description> {
    let mut iter = filter_child_elements(&xml).peekable();
    let brief = iter.next().context("no brief node in description")??;
    ensure!(brief.has_tag_name("brief"), "first element in the description node must be a brief");

    let checked = brief.attribute("checked").map_or(
        Ok(true),
        |checked| match checked {
            "yes" => Ok(true),
            "no" => Ok(false),
            checked_state => bail!("unknown brief check state checked=`{checked_state}`")
        }
    )?;

    let single_synthesis = brief.attribute("synth")
        .is_none_or(|synth| synth == "single");

    let brief_reliability = match checked {
        true => BriefReliability::Verified,
        false if single_synthesis => BriefReliability::UnverifiedReliable,
        false => BriefReliability::Unchecked
    };


    let brief = MdParser::parse_formatted_text_node(img_factory, &brief)?;
    let authored = iter.next_if_map(|node| match node {
        Ok(node) if node.has_tag_name("authored") => Ok(node),
        res => Err(res)
    });

    let authored = authored
        .map(|node| {
            MdParser::parse_formatted_text_node(img_factory, &node)
        })
        .transpose()?;


    let mut other = HashMap::new();

    for node in iter {
        let node = node?;
        let text = MdParser::parse_formatted_text_node(img_factory, &node)?.text;
        other.insert(node.tag_name().name(), text);
    }

    Ok(Description {
        brief: Brief {
            content: brief,
            reliability: brief_reliability
        },
        authored,
        other: Default::default(),
    })
}

fn parse_exceptions(
    img_factory: &mut ImageFactory,
    xml: XmlNode,
) -> Result<Vec<ExceptionGroup>> {
    filter_child_elements(&xml)
        .map(|group| {
            let group = group?;
            ensure!(group.has_tag_name("exception_group"));

            let group_name = group.attribute("group_name").map(Box::<str>::from);

            let exceptions = filter_child_elements(&group)
                .map(|exception| {
                    let exception = exception?;
                    ensure!(exception.has_tag_name("exception"));

                    MdParser::parse_formatted_text_node(img_factory, &exception)
                })
                .collect::<Result<Vec<_>>>()?;

            ensure!(
                !exceptions.is_empty(),
                "exception_group must contain at least one exception"
            );

            Ok(ExceptionGroup {
                group_name,
                exceptions,
            })
        })
        .collect()
}

fn parse_instruction_section(
    image_factory: &mut ImageFactory,
    xml: File
) -> Result<Option<InstructionPage>> {
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
        entity_resolver: None
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

    let section_type = attr(&root, "type")?;

    if *section_type != *"instruction" {
        if !matches!(section_type, "alias" | "pseudocode") {
            eprintln!("WARNING: unknown instruction page type `{section_type}`");
        }
        return Ok(None)
    }


    let id = attr_owned(&root, "id")?;
    let title = attr_owned(&root, "title")?;

    let maybe_find_from_root_elem = |name: &str| {
        let mut iter = root.children().filter(|n| n.has_tag_name(name));

        match iter.next() {
            Some(elem) => match iter.next() {
                None => Ok(Some(elem)),
                Some(_) => bail!("found too many `{name}` elements"),
            }
            None => Ok(None),
        }
    };

    let find_from_root_elem = |name: &str|{
        match maybe_find_from_root_elem(name) {
            Ok(Some(x)) => Ok(x),
            Ok(None) => bail!("element `{name}` was not found"),
            Err(err) => Err(err)
        }
    };

    let docvars = find_from_root_elem("docvars")?;
    let mut docvars_map = HashMap::new();
    let mut isa = None::<Isa>;

    for node in docvars.children() {
        if let (Some(key), Some(value)) = (node.attribute("key"), node.attribute("value")) {
            let duplicate_key = match key {
                "isa" => {
                    let isa_value = match value {
                        "A64" => Isa::A64,
                        "A32" => Isa::A32,
                        isa => bail!("unknown isa `{isa}`"),
                    };
                    isa.replace(isa_value).is_some()
                },
                key => docvars_map.insert(Box::from(key), Box::from(value)).is_some()
            };

            if duplicate_key {
                bail!("duplicate docvar {key}")
            }
        }
    }

    let isa = isa.context("missing isa docvar")?;
    let docvars = docvars_map;

    let heading = find_from_root_elem("heading")?
        .text()
        .context("invalid heading text")?;

    let description = parse_decription(image_factory, find_from_root_elem("desc")?)?;


    let operational_notes = root
        .children()
        .find(|n| n.has_tag_name("operationalnotes"))
        .map(|notes_elem| {
            let iter = filter_child_elements(&notes_elem).map(|note_elem| {
                let note_elem = note_elem?;
                ensure!(note_elem.has_tag_name("operationalnote"));


                let mut titles = vec![];
                let mut content = None::<MdSegment>;

                for n in filter_child_elements(&note_elem) {
                    let node = n?;
                    match node.tag_name().name() {
                        "operationalnote_content" => {
                            ensure!(
                                content.is_none(),
                                "multiple operational note contents in one note"
                            );

                            let content_md = MdParser::parse_formatted_text_node(
                                image_factory,
                                &node
                            )?;

                            content = Some(content_md)
                        },
                        "operationalnote_titles" => {
                            ensure!(content.is_none(), "trailing operational note titles");
                            for child in filter_child_elements(&node) {
                                let child = child?;
                                ensure!(child.has_tag_name("operationalnote_title"));
                                let text = MdParser::parse_pcdata_node(&child)?;
                                titles.push(Box::<str>::from(text))
                            }
                        }
                        name => bail!("unknown operational note field `{name}`")
                    }
                }

                Ok(OperationalNote {
                    titles,
                    content: content.context("operational note with no content")?
                })
            });

            iter.collect::<Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or(vec![]);

    let exceptions = maybe_find_from_root_elem("exceptions")?
        .map(|node| parse_exceptions(image_factory, node))
        .transpose()?
        .unwrap_or(vec![]);


    Ok(Some(InstructionPage {
        id,
        title,
        isa,
        docvars,
        description,
        heading: Box::<str>::from(heading),
        operational_notes,
        exceptions
    }))
}


const MAX_INSTRUCTIONS_DEBUG: usize = 128;

fn debug_instruction_list(instructions: &[InstructionPage]) {
    let mut debug_list = instructions
        .iter()
        .map(|insn| &insn.id)
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

fn open_tar_archive() -> io::Result<tar::Archive<impl Read>> {
    let compressed_data_reader = File::open(ISA_ARCHIVE)?;
    let tar_reader = flate2::read::MultiGzDecoder::new(compressed_data_reader);
    Ok(tar::Archive::new(tar_reader))
}

pub fn load_instruction_pages() -> Result<Vec<InstructionPage>> {
    let mut tar = open_tar_archive()?;
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


    drop(tar);

    let iform = iform_dtd.expect("there should always be a dtd file in the ISA folder");
    ensure!(iform == EXPECTED_ISA_IFORM_DTD);

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
        ensure!({
            let mut iter = insn_file_name.bytes();
            iter.next().is_some_and(|x| x.is_ascii_lowercase())
                && iter.all(|x| matches!(x, b'a'..=b'z' | b'0'..=b'9' | b'_'))
        })
    }

    let instructions = instructions;

    let mut image_factory = ImageFactory::new();

    let instructions = instructions
        .into_iter()
        .flat_map(|(file_name, file)| {
            parse_instruction_section(&mut image_factory, file)
                .map_err(|err| {
                    err.wrap_err(format!("failed to parse instruction page `{file_name}`"))
                })
                .transpose()
        })
        .collect::<Result<Vec<_>>>()?;

    let mut tar = open_tar_archive()?;

    for entry in tar.entries()? {
        let mut entry = entry?;
        let path =  entry.path()?;
        let Ok(file_name) = path.strip_prefix(ISA_XML_FOLDER) else {
            continue
        };

        let Some(file_path) = file_name.to_str() else {
            continue
        };

        if let Some(handle) = image_factory.cache.remove(file_path) {
            let mut image_data = Vec::new();
            entry.read_to_end(&mut image_data)?;
            let mut data = handle.0.data.borrow_mut();
            assert!(data.is_none());
            *data = Some(image_data);
        }

        if image_factory.cache.is_empty() {
            break
        }
    }

    drop(tar);

    if !image_factory.cache.is_empty() {
        let names = image_factory.cache.into_keys().collect_vec();
        println!("could not find the following images: `{names:?}`")
    }

    debug_instruction_list(&instructions);

    Ok(instructions)
}