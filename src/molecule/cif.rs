//! PDBx/mmCIF reading shared by the text, binary and XML encodings.
//!
//! Each encoding exposes its categories through [`CifDocument`]; [`structure_from_cif`]
//! then reads coordinates, connectivity, assemblies, symmetry and secondary structure.

use std::{borrow::Cow, collections::HashMap};

use glam::Vec3;

use super::{
    AnnotatedStructure, Assembly, AssemblyGenerator, Atom, BondKind, BondOrder, CrystalInfo,
    Element, SecondaryAnnotation, StructureError, StructureInfo, SymmetryOperator, UnitCell,
    info::parse_operator_expression,
    pdb::infer_element,
    records::{AtomRecord, BondRecord, SiteKey, assemble},
};

/// Read access to one category (table) of a data block.
pub(crate) trait CifCategory {
    fn row_count(&self) -> usize;
    /// Column index for a lower-case item name.
    fn column(&self, name: &str) -> Option<usize>;
    /// Cell text, or `None` for missing values (`.`/`?` or masked).
    fn text(&self, column: usize, row: usize) -> Option<Cow<'_, str>>;
    fn number(&self, column: usize, row: usize) -> Option<f64> {
        self.text(column, row)?.trim().parse().ok()
    }
}

pub(crate) trait CifDocument {
    /// Category by lower-case name without the leading underscore.
    fn category(&self, name: &str) -> Option<&dyn CifCategory>;
}

/// Convenience accessor resolving columns once per category.
struct Table<'a> {
    category: &'a dyn CifCategory,
}

impl<'a> Table<'a> {
    fn new(document: &'a dyn CifDocument, name: &str) -> Option<Self> {
        document
            .category(name)
            .filter(|category| category.row_count() > 0)
            .map(|category| Self { category })
    }

    fn rows(&self) -> usize {
        self.category.row_count()
    }

    fn column(&self, names: &[&str]) -> Option<usize> {
        names.iter().find_map(|name| self.category.column(name))
    }

    fn text(&self, column: Option<usize>, row: usize) -> Option<Cow<'a, str>> {
        self.category.text(column?, row)
    }

    fn string(&self, names: &[&str], row: usize) -> Option<String> {
        names.iter().find_map(|name| {
            let column = self.category.column(name)?;
            self.category
                .text(column, row)
                .map(|value| value.trim().to_string())
        })
    }

    fn number(&self, column: Option<usize>, row: usize) -> Option<f64> {
        self.category.number(column?, row)
    }
}

pub(crate) struct CifStructure {
    pub molecule: super::Molecule,
    pub frames: Vec<Vec<Vec3>>,
}

/// Reads a structure from the categories of a PDBx/mmCIF data block.
pub(crate) fn structure_from_cif(
    document: &dyn CifDocument,
    format: &'static str,
) -> Result<CifStructure, StructureError> {
    let no_coordinates = || StructureError::NoCoordinates(format_file_label(format));
    let atom_site = Table::new(document, "atom_site").ok_or_else(no_coordinates)?;
    let (records, label_chains) = read_atom_site(&atom_site, format)?;
    if records.is_empty() {
        return Err(no_coordinates());
    }
    let mut info = StructureInfo {
        id: Table::new(document, "entry").and_then(|table| table.string(&["id"], 0)),
        title: Table::new(document, "struct").and_then(|table| table.string(&["title"], 0)),
        ..StructureInfo::default()
    };
    read_symmetry(document, &mut info);
    read_assemblies(document, &label_chains, &mut info);
    read_secondary(document, &mut info);
    let bonds = read_struct_conn(document);
    let assembled = assemble(records, &bonds, info).ok_or_else(no_coordinates)?;
    Ok(CifStructure {
        molecule: assembled.molecule,
        frames: assembled.frames,
    })
}

fn format_file_label(format: &'static str) -> &'static str {
    match format {
        "BinaryCIF" => "BinaryCIF file",
        "PDBML/XML" => "PDBML/XML file",
        _ => "mmCIF file",
    }
}

type LabelChains = HashMap<String, String>;

fn read_atom_site(
    table: &Table<'_>,
    format: &'static str,
) -> Result<(Vec<AtomRecord>, LabelChains), StructureError> {
    let group = table.column(&["group_pdb"]);
    let id = table.column(&["id"]);
    let symbol = table.column(&["type_symbol"]);
    let atom_name = table.column(&["auth_atom_id", "label_atom_id"]);
    let label_atom_name = table.column(&["label_atom_id"]);
    let alt = table.column(&["label_alt_id", "auth_alt_id"]);
    let residue = table.column(&["auth_comp_id", "label_comp_id"]);
    let label_residue = table.column(&["label_comp_id"]);
    let auth_asym = table.column(&["auth_asym_id"]);
    let label_asym = table.column(&["label_asym_id"]);
    let auth_seq = table.column(&["auth_seq_id"]);
    let label_seq = table.column(&["label_seq_id"]);
    let insertion = table.column(&["pdbx_pdb_ins_code"]);
    let (x, y, z) = (
        table.column(&["cartn_x"]),
        table.column(&["cartn_y"]),
        table.column(&["cartn_z"]),
    );
    let occupancy = table.column(&["occupancy"]);
    let b_factor = table.column(&["b_iso_or_equiv"]);
    let charge = table.column(&["pdbx_formal_charge"]);
    let model = table.column(&["pdbx_pdb_model_num"]);

    let mut records = Vec::with_capacity(table.rows());
    let mut label_chains = LabelChains::new();
    for row in 0..table.rows() {
        let error = |message: String| StructureError::AtomSite {
            format,
            row: row + 1,
            message,
        };
        let name = table
            .text(atom_name, row)
            .or_else(|| table.text(label_atom_name, row))
            .ok_or_else(|| error("missing atom name".into()))?
            .trim()
            .to_ascii_uppercase();
        let residue_name = table
            .text(residue, row)
            .or_else(|| table.text(label_residue, row))
            .ok_or_else(|| error("missing residue name".into()))?
            .trim()
            .to_ascii_uppercase();
        let residue_text = table
            .text(auth_seq, row)
            .or_else(|| table.text(label_seq, row))
            .ok_or_else(|| error("missing residue sequence number".into()))?;
        let residue_number = residue_text
            .trim()
            .parse::<i64>()
            .ok()
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| error(format!("invalid residue sequence number '{residue_text}'")))?;
        let coordinate = |column: Option<usize>, label: &str| -> Result<f32, StructureError> {
            table
                .number(column, row)
                .map(|value| value as f32)
                .ok_or_else(|| match table.text(column, row) {
                    Some(value) => error(format!("invalid {label} value '{value}'")),
                    None => error(format!("missing {label}")),
                })
        };
        let position = Vec3::new(
            coordinate(x, "x coordinate")?,
            coordinate(y, "y coordinate")?,
            coordinate(z, "z coordinate")?,
        );
        let chain_id = table
            .text(auth_asym, row)
            .or_else(|| table.text(label_asym, row))
            .map(|value| value.trim().to_string())
            .unwrap_or_default();
        if let Some(label) = table.text(label_asym, row) {
            label_chains
                .entry(label.trim().to_string())
                .or_insert_with(|| chain_id.clone());
        }
        let given = table
            .text(symbol, row)
            .and_then(|value| value.parse::<Element>().ok());
        let element_inferred = given.is_none();
        let element = given.unwrap_or_else(|| infer_element(&name));
        let alt_loc = table
            .text(alt, row)
            .and_then(|value| value.trim().chars().next());
        let atom = Atom {
            serial: table
                .number(id, row)
                .filter(|value| *value >= 0.0 && *value <= u32::MAX as f64)
                .map_or(row as u32 + 1, |value| value as u32),
            name,
            element,
            residue_name,
            residue_number,
            insertion_code: table
                .text(insertion, row)
                .and_then(|value| value.trim().chars().next()),
            chain_id,
            position,
            occupancy: table
                .number(occupancy, row)
                .map_or(1.0, |value| value as f32),
            b_factor: table
                .number(b_factor, row)
                .map_or(0.0, |value| value as f32),
            hetero: table
                .text(group, row)
                .is_some_and(|value| value.trim().eq_ignore_ascii_case("HETATM")),
            alt_loc,
            formal_charge: table
                .number(charge, row)
                .map_or(0, |value| value.clamp(-128.0, 127.0) as i8),
        };
        let model = table.number(model, row).map_or(1, |value| value as i64);
        records.push(AtomRecord {
            atom,
            model,
            element_inferred,
        });
    }
    Ok((records, label_chains))
}

fn read_symmetry(document: &dyn CifDocument, info: &mut StructureInfo) {
    let Some(cell) = Table::new(document, "cell") else {
        return;
    };
    let value = |name: &str| cell.number(cell.column(&[name]), 0);
    let (Some(a), Some(b), Some(c), Some(alpha), Some(beta), Some(gamma)) = (
        value("length_a"),
        value("length_b"),
        value("length_c"),
        value("angle_alpha"),
        value("angle_beta"),
        value("angle_gamma"),
    ) else {
        return;
    };
    let space_group = Table::new(document, "symmetry")
        .and_then(|table| table.string(&["space_group_name_h-m", "int_tables_number"], 0))
        .or_else(|| {
            Table::new(document, "space_group")
                .and_then(|table| table.string(&["name_h-m_alt", "it_number"], 0))
        })
        .filter(|group| !group.is_empty())
        .unwrap_or_else(|| "P 1".into());
    info.crystal = Some(CrystalInfo {
        cell: UnitCell {
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
        },
        space_group,
    });
}

fn read_assemblies(
    document: &dyn CifDocument,
    label_chains: &LabelChains,
    info: &mut StructureInfo,
) {
    let (Some(operators), Some(generators)) = (
        Table::new(document, "pdbx_struct_oper_list"),
        Table::new(document, "pdbx_struct_assembly_gen"),
    ) else {
        return;
    };
    let mut operator_index = HashMap::<String, usize>::new();
    for row in 0..operators.rows() {
        let Some(id) = operators.string(&["id"], row) else {
            continue;
        };
        let matrix = |i: usize, j: usize| {
            operators
                .number(operators.column(&[&format!("matrix[{i}][{j}]")]), row)
                .unwrap_or(if i == j { 1.0 } else { 0.0 })
        };
        let vector = |i: usize| {
            operators
                .number(operators.column(&[&format!("vector[{i}]")]), row)
                .unwrap_or(0.0)
        };
        let operator = SymmetryOperator {
            id: id.clone(),
            name: operators
                .string(&["name", "type"], row)
                .unwrap_or_else(|| id.clone()),
            rotation: [
                [matrix(1, 1), matrix(1, 2), matrix(1, 3)],
                [matrix(2, 1), matrix(2, 2), matrix(2, 3)],
                [matrix(3, 1), matrix(3, 2), matrix(3, 3)],
            ],
            translation: [vector(1), vector(2), vector(3)],
        };
        operator_index.insert(id, info.operators.len());
        info.operators.push(operator);
    }
    let details = Table::new(document, "pdbx_struct_assembly");
    let mut assemblies: Vec<Assembly> = Vec::new();
    if let Some(details) = &details {
        for row in 0..details.rows() {
            let Some(id) = details.string(&["id"], row) else {
                continue;
            };
            assemblies.push(Assembly {
                id,
                details: details
                    .string(&["oligomeric_details", "details"], row)
                    .unwrap_or_default(),
                oligomeric_count: details
                    .number(details.column(&["oligomeric_count"]), row)
                    .map(|value| value as u32),
                generators: Vec::new(),
            });
        }
    }
    for row in 0..generators.rows() {
        let (Some(assembly_id), Some(expression), Some(chains)) = (
            generators.string(&["assembly_id"], row),
            generators.string(&["oper_expression"], row),
            generators.string(&["asym_id_list"], row),
        ) else {
            continue;
        };
        let Ok(products) = parse_operator_expression(&expression) else {
            info.notes.push(format!(
                "assembly {assembly_id}: unreadable operators '{expression}'"
            ));
            continue;
        };
        let Some(products) = products
            .into_iter()
            .map(|product| {
                product
                    .iter()
                    .map(|id| operator_index.get(id).copied())
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()
        else {
            info.notes.push(format!(
                "assembly {assembly_id} refers to an undefined operator"
            ));
            continue;
        };
        // Generators list label_asym_id values; the molecule uses author chain IDs.
        let mut chain_ids = Vec::<String>::new();
        for label in chains
            .split(',')
            .map(str::trim)
            .filter(|label| !label.is_empty())
        {
            if let Some(chain) = label_chains.get(label)
                && !chain_ids.contains(chain)
            {
                chain_ids.push(chain.clone());
            }
        }
        if chain_ids.is_empty() {
            continue;
        }
        let index = match assemblies
            .iter()
            .position(|assembly| assembly.id == assembly_id)
        {
            Some(index) => index,
            None => {
                assemblies.push(Assembly {
                    id: assembly_id,
                    details: String::new(),
                    oligomeric_count: None,
                    generators: Vec::new(),
                });
                assemblies.len() - 1
            }
        };
        assemblies[index].generators.push(AssemblyGenerator {
            chains: chain_ids,
            products,
        });
    }
    info.assemblies = assemblies
        .into_iter()
        .filter(|assembly| !assembly.generators.is_empty())
        .collect();
}

fn read_secondary(document: &dyn CifDocument, info: &mut StructureInfo) {
    let mut read = |name: &str, kind_of: &dyn Fn(Option<String>) -> Option<AnnotatedStructure>| {
        let Some(table) = Table::new(document, name) else {
            return;
        };
        for row in 0..table.rows() {
            let Some(kind) = kind_of(table.string(&["conf_type_id"], row)) else {
                continue;
            };
            let position = |prefix: &str| -> Option<(String, i32, Option<char>)> {
                let chain = table.string(
                    &[
                        &format!("{prefix}_auth_asym_id"),
                        &format!("{prefix}_label_asym_id"),
                    ],
                    row,
                )?;
                let number = table
                    .string(
                        &[
                            &format!("{prefix}_auth_seq_id"),
                            &format!("{prefix}_label_seq_id"),
                        ],
                        row,
                    )?
                    .parse()
                    .ok()?;
                let code = table
                    .string(&[&format!("pdbx_{prefix}_pdb_ins_code")], row)
                    .and_then(|value| value.chars().next());
                Some((chain, number, code))
            };
            if let (Some(start), Some(end)) = (position("beg"), position("end"))
                && start.0 == end.0
            {
                info.secondary.push(SecondaryAnnotation {
                    kind,
                    chain: start.0,
                    start: (start.1, start.2),
                    end: (end.1, end.2),
                });
            }
        }
    };
    read("struct_conf", &|kind| {
        let kind = kind?.to_ascii_uppercase();
        if kind.starts_with("HELX_RH_3T") {
            Some(AnnotatedStructure::Helix310)
        } else if kind.starts_with("HELX_RH_PI") {
            Some(AnnotatedStructure::HelixPi)
        } else if kind.starts_with("HELX") {
            Some(AnnotatedStructure::Helix)
        } else if kind.starts_with("STRN") {
            Some(AnnotatedStructure::Strand)
        } else if kind.starts_with("TURN") {
            Some(AnnotatedStructure::Turn)
        } else {
            None
        }
    });
    read("struct_sheet_range", &|_| Some(AnnotatedStructure::Strand));
}

fn read_struct_conn(document: &dyn CifDocument) -> Vec<BondRecord> {
    let Some(table) = Table::new(document, "struct_conn") else {
        return Vec::new();
    };
    let mut bonds = Vec::new();
    for row in 0..table.rows() {
        let kind = match table
            .string(&["conn_type_id"], row)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "disulf" => BondKind::Disulfide,
            "metalc" => BondKind::MetalCoordination,
            kind if kind.starts_with("covale") => BondKind::Covalent,
            _ => continue,
        };
        let partner = |index: u8| -> Option<SiteKey> {
            let prefix = format!("ptnr{index}");
            Some(SiteKey {
                chain: table.string(
                    &[
                        &format!("{prefix}_auth_asym_id"),
                        &format!("{prefix}_label_asym_id"),
                    ],
                    row,
                )?,
                residue_number: table
                    .string(
                        &[
                            &format!("{prefix}_auth_seq_id"),
                            &format!("{prefix}_label_seq_id"),
                        ],
                        row,
                    )?
                    .parse()
                    .ok()?,
                insertion_code: table
                    .string(&[&format!("pdbx_{prefix}_pdb_ins_code")], row)
                    .and_then(|value| value.chars().next()),
                atom_name: table
                    .string(
                        &[
                            &format!("{prefix}_auth_atom_id"),
                            &format!("{prefix}_label_atom_id"),
                        ],
                        row,
                    )?
                    .to_ascii_uppercase(),
            })
        };
        let order = match table
            .string(&["pdbx_value_order"], row)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "doub" => Some(BondOrder::Double),
            "trip" => Some(BondOrder::Triple),
            "arom" => Some(BondOrder::Aromatic),
            "sing" => Some(BondOrder::Single),
            _ => None,
        };
        if let (Some(a), Some(b)) = (partner(1), partner(2)) {
            bonds.push(BondRecord::Site { a, b, kind, order });
        }
    }
    bonds
}

/// A text CIF data block: category name to table of borrowed values.
pub(crate) struct TextDocument<'a> {
    categories: HashMap<String, TextCategory<'a>>,
}

struct TextCategory<'a> {
    names: HashMap<String, usize>,
    width: usize,
    values: Vec<&'a str>,
}

impl CifCategory for TextCategory<'_> {
    fn row_count(&self) -> usize {
        self.values.len() / self.width.max(1)
    }

    fn column(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    fn text(&self, column: usize, row: usize) -> Option<Cow<'_, str>> {
        let value = *self.values.get(row * self.width + column)?;
        (!matches!(value, "." | "?")).then_some(Cow::Borrowed(value))
    }
}

impl CifDocument for TextDocument<'_> {
    fn category(&self, name: &str) -> Option<&dyn CifCategory> {
        self.categories
            .get(name)
            .map(|category| category as &dyn CifCategory)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Token<'a> {
    Value(&'a str),
    /// Quoted and text-field values are never keywords, even when they read like one.
    Quoted(&'a str),
}

impl<'a> Token<'a> {
    fn text(self) -> &'a str {
        match self {
            Self::Value(value) | Self::Quoted(value) => value,
        }
    }

    fn keyword(self) -> Option<&'a str> {
        match self {
            Self::Value(value) => Some(value),
            Self::Quoted(_) => None,
        }
    }
}

/// Parses the first data block of a text CIF document.
pub(crate) fn parse_text(input: &str) -> Result<TextDocument<'_>, StructureError> {
    let tokens = tokenize(input)?;
    let mut categories = HashMap::<String, TextCategory<'_>>::new();
    let mut cursor = 0;
    let mut blocks = 0;
    let is_control = |token: Token<'_>| {
        token.keyword().is_some_and(|value| {
            value.starts_with('_')
                || value.eq_ignore_ascii_case("loop_")
                || value.eq_ignore_ascii_case("stop_")
                || value.eq_ignore_ascii_case("global_")
                || value.get(..5).is_some_and(|prefix| {
                    prefix.eq_ignore_ascii_case("data_") || prefix.eq_ignore_ascii_case("save_")
                })
        })
    };
    while cursor < tokens.len() {
        let token = tokens[cursor];
        let keyword = token.keyword().unwrap_or("");
        if keyword
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data_"))
        {
            blocks += 1;
            if blocks > 1 {
                break;
            }
            cursor += 1;
        } else if keyword.eq_ignore_ascii_case("loop_") {
            cursor += 1;
            let mut tags = Vec::new();
            while let Some(tag) = tokens.get(cursor).and_then(|token| token.keyword())
                && tag.starts_with('_')
            {
                tags.push(tag);
                cursor += 1;
            }
            let start = cursor;
            while cursor < tokens.len() && !is_control(tokens[cursor]) {
                cursor += 1;
            }
            if tags.is_empty() {
                continue;
            }
            let values: Vec<&str> = tokens[start..cursor]
                .iter()
                .map(|token| token.text())
                .collect();
            if !values.len().is_multiple_of(tags.len()) {
                let category = split_tag(tags[0]).0;
                return Err(StructureError::Mmcif(format!(
                    "{category} loop has {} values for {} columns",
                    values.len(),
                    tags.len()
                )));
            }
            let category_name = split_tag(tags[0]).0;
            let names = tags
                .iter()
                .enumerate()
                .map(|(index, tag)| (split_tag(tag).1, index))
                .collect();
            categories.insert(
                category_name,
                TextCategory {
                    names,
                    width: tags.len(),
                    values,
                },
            );
        } else if keyword.starts_with('_') {
            let (category, item) = split_tag(keyword);
            let value = tokens
                .get(cursor + 1)
                .filter(|token| !is_control(**token))
                .map(|token| token.text());
            cursor += if value.is_some() { 2 } else { 1 };
            let Some(value) = value else {
                continue;
            };
            // Key-value pairs of one category form a single-row table.
            let entry = categories.entry(category).or_insert_with(|| TextCategory {
                names: HashMap::new(),
                width: 0,
                values: Vec::new(),
            });
            if entry.row_count() <= 1 && !entry.names.contains_key(&item) {
                entry.names.insert(item, entry.width);
                entry.width += 1;
                entry.values.push(value);
            }
        } else {
            cursor += 1;
        }
    }
    Ok(TextDocument { categories })
}

fn split_tag(tag: &str) -> (String, String) {
    let tag = tag.trim_start_matches('_').to_ascii_lowercase();
    match tag.split_once('.') {
        Some((category, item)) => (category.to_string(), item.to_string()),
        None => (tag.clone(), String::new()),
    }
}

fn tokenize(input: &str) -> Result<Vec<Token<'_>>, StructureError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut line_start = true;
    while cursor < bytes.len() {
        match bytes[cursor] {
            byte if byte.is_ascii_whitespace() => {
                line_start = byte == b'\n' || (line_start && byte == b'\r');
                cursor += 1;
            }
            b'#' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
            }
            b';' if line_start => {
                cursor += 1;
                let start = cursor;
                let mut end = None;
                while cursor < bytes.len() {
                    if bytes[cursor] == b'\n' && bytes.get(cursor + 1) == Some(&b';') {
                        end = Some(cursor);
                        cursor += 2;
                        break;
                    }
                    cursor += 1;
                }
                let end = end.ok_or_else(|| {
                    StructureError::Mmcif("unterminated semicolon-delimited value".into())
                })?;
                tokens.push(Token::Quoted(
                    input[start..end]
                        .trim_start_matches(['\r', '\n'])
                        .trim_end_matches('\r'),
                ));
                line_start = false;
            }
            quote @ (b'\'' | b'"') => {
                line_start = false;
                cursor += 1;
                let start = cursor;
                // A quote closes the value only when followed by whitespace or the end.
                loop {
                    if cursor >= bytes.len() {
                        return Err(StructureError::Mmcif("unterminated quoted value".into()));
                    }
                    if bytes[cursor] == quote
                        && bytes
                            .get(cursor + 1)
                            .is_none_or(|next| next.is_ascii_whitespace())
                    {
                        break;
                    }
                    if bytes[cursor] == b'\n' {
                        return Err(StructureError::Mmcif("unterminated quoted value".into()));
                    }
                    cursor += 1;
                }
                tokens.push(Token::Quoted(&input[start..cursor]));
                cursor += 1;
            }
            _ => {
                line_start = false;
                let start = cursor;
                while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                tokens.push(Token::Value(&input[start..cursor]));
            }
        }
    }
    Ok(tokens)
}

/// A category with owned values, used by the XML reader.
#[derive(Default)]
pub(crate) struct OwnedCategory {
    pub names: HashMap<String, usize>,
    pub rows: Vec<Vec<Option<String>>>,
}

impl OwnedCategory {
    pub fn set(&mut self, row: usize, name: &str, value: String) {
        let width = self.names.len();
        let column = *self.names.entry(name.to_string()).or_insert(width);
        let values = &mut self.rows[row];
        if values.len() <= column {
            values.resize(column + 1, None);
        }
        match &mut values[column] {
            Some(existing) => existing.push_str(&value),
            slot => *slot = Some(value),
        }
    }
}

impl CifCategory for OwnedCategory {
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn column(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    fn text(&self, column: usize, row: usize) -> Option<Cow<'_, str>> {
        self.rows
            .get(row)?
            .get(column)?
            .as_deref()
            .filter(|value| !matches!(*value, "." | "?"))
            .map(Cow::Borrowed)
    }
}

#[derive(Default)]
pub(crate) struct OwnedDocument {
    pub categories: HashMap<String, OwnedCategory>,
}

impl CifDocument for OwnedDocument {
    fn category(&self, name: &str) -> Option<&dyn CifCategory> {
        self.categories
            .get(name)
            .map(|category| category as &dyn CifCategory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MMCIF: &str = r#"data_1ABC
_entry.id 1ABC
_struct.title 'Two chains; one "quoted" title'
_cell.length_a 40.0
_cell.length_b 50.0
_cell.length_c 60.0
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 21 21 21'
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.pdbx_PDB_model_num
ATOM 1 S SG . CYS A A 3 ? 0.0 0.0 0.0 1.0 10.0 ? 1
ATOM 2 S SG . CYS B B 7 ? 2.05 0.0 0.0 1.0 10.0 ? 1
HETATM 3 ZN ZN . ZN C A 101 ? 5.0 0.0 0.0 1.0 10.0 2 1
ATOM 1 S SG . CYS A A 3 ? 0.5 0.0 0.0 1.0 10.0 ? 2
ATOM 2 S SG . CYS B B 7 ? 2.55 0.0 0.0 1.0 10.0 ? 2
HETATM 3 ZN ZN . ZN C A 101 ? 5.5 0.0 0.0 1.0 10.0 2 2
#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_label_atom_id
disulf1 disulf A 3 SG B 7 SG
metalc1 metalc A 3 SG A 101 ZN
#
loop_
_pdbx_struct_assembly.id
_pdbx_struct_assembly.details
_pdbx_struct_assembly.oligomeric_count
1 author_defined_assembly 2
_pdbx_struct_assembly_gen.assembly_id 1
_pdbx_struct_assembly_gen.oper_expression '(1,2)'
_pdbx_struct_assembly_gen.asym_id_list A,C
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.type
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[1]
1 'identity operation' 1 1 1 0
2 'crystal symmetry operation' -1 -1 1 20.0
loop_
_struct_conf.conf_type_id
_struct_conf.beg_auth_asym_id
_struct_conf.beg_auth_seq_id
_struct_conf.end_auth_asym_id
_struct_conf.end_auth_seq_id
HELX_P A 1 A 3
HELX_RH_3T_P B 5 B 7
_struct_keywords.text
;
text field value
spanning lines
;
"#;

    #[test]
    fn reads_models_bonds_assemblies_symmetry_and_annotations() {
        let document = parse_text(MMCIF).unwrap();
        let structure = structure_from_cif(&document, "PDBx/mmCIF").unwrap();
        let molecule = &structure.molecule;
        assert_eq!(molecule.atoms.len(), 3);
        assert_eq!(structure.frames.len(), 1);
        assert_eq!(structure.frames[0][0], Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(molecule.atoms[2].formal_charge, 2);
        assert_eq!(molecule.atoms[2].chain_id, "A");
        let info = &molecule.info;
        assert_eq!(info.id.as_deref(), Some("1ABC"));
        assert_eq!(
            info.title.as_deref(),
            Some("Two chains; one \"quoted\" title")
        );
        assert_eq!(info.crystal.as_ref().unwrap().space_group, "P 21 21 21");
        assert_eq!(info.assemblies.len(), 1);
        assert_eq!(info.assemblies[0].generators[0].chains, vec!["A"]);
        assert_eq!(
            info.assemblies[0].generators[0].products,
            vec![vec![0], vec![1]]
        );
        assert_eq!(info.operators[1].rotation[0][0], -1.0);
        assert_eq!(info.operators[1].translation, [20.0, 0.0, 0.0]);
        assert_eq!(info.secondary.len(), 2);
        assert_eq!(info.secondary[1].kind, AnnotatedStructure::Helix310);
        let disulfide = molecule
            .bonds
            .iter()
            .find(|b| (b.a, b.b) == (0, 1))
            .unwrap();
        assert_eq!(disulfide.kind, BondKind::Disulfide);
        let metal = molecule
            .bonds
            .iter()
            .find(|b| (b.a, b.b) == (0, 2))
            .unwrap();
        assert_eq!(metal.kind, BondKind::MetalCoordination);
    }

    #[test]
    fn quotes_close_only_before_whitespace() {
        let tokens = tokenize("'O5'' \"a b\" c").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Quoted("O5'"),
                Token::Quoted("a b"),
                Token::Value("c")
            ]
        );
    }

    #[test]
    fn reports_misaligned_loops_and_unterminated_values() {
        assert!(parse_text("data_x\nloop_\n_a.b\n_a.c\n1 2 3\n").is_err());
        assert!(parse_text("data_x\n_a.b 'open\n").is_err());
        assert!(parse_text("data_x\n_a.b\n;open\n").is_err());
    }
}
