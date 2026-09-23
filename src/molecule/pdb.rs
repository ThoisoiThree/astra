//! Legacy PDB format reader (wwPDB format 3.3, including hybrid-36 serial numbers).

use std::str::FromStr;

use glam::Vec3;
use thiserror::Error;

use super::{
    AnnotatedStructure, Assembly, AssemblyGenerator, Atom, BondKind, CrystalInfo, Element,
    Molecule, SecondaryAnnotation, StructureInfo, SymmetryOperator, UnitCell,
    records::{AtomRecord, BondRecord, SiteKey, assemble, serial_bond_records},
};

#[derive(Debug, Error, PartialEq)]
pub enum PdbError {
    #[error("PDB line {line}: missing {field}")]
    MissingField { line: usize, field: &'static str },
    #[error("PDB line {line}: invalid {field} value '{value}'")]
    InvalidField {
        line: usize,
        field: &'static str,
        value: String,
    },
    #[error("PDB contains no ATOM or HETATM records")]
    NoAtoms,
}

/// A PDB document: the first model as topology plus later models as coordinate frames.
#[derive(Debug, Clone)]
pub struct PdbDocument {
    pub molecule: Molecule,
    pub frames: Vec<Vec<Vec3>>,
}

/// Parses the first model of a PDB document with its connectivity and annotations.
pub fn parse_pdb(input: &str) -> Result<Molecule, PdbError> {
    parse_pdb_document(input).map(|document| document.molecule)
}

/// Parses every model; models after the first become coordinate frames.
pub fn parse_pdb_document(input: &str) -> Result<PdbDocument, PdbError> {
    let mut records = Vec::new();
    let mut conect = Vec::new();
    let mut bonds = Vec::new();
    let mut info = StructureInfo::default();
    let mut remark350 = Remark350::default();
    let mut model: i64 = 1;
    let mut model_seen = false;
    let mut title = String::new();

    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let record = field(line, 0, 6).trim_end();
        match record {
            "HEADER" => {
                let id = field(line, 62, 66).trim();
                if !id.is_empty() {
                    info.id = Some(id.to_string());
                }
            }
            "TITLE" => {
                let text = field(line, 10, 80).trim();
                if !text.is_empty() {
                    if !title.is_empty() {
                        title.push(' ');
                    }
                    title.push_str(text);
                }
            }
            "MODEL" => {
                // Model serials are optional in some writers; count models in that case.
                let next = field(line, 10, 14).trim().parse::<i64>().ok();
                model = match (next, model_seen) {
                    (Some(number), _) => number,
                    (None, true) => model + 1,
                    (None, false) => 1,
                };
                model_seen = true;
            }
            "ATOM" | "HETATM" => {
                let atom = parse_atom(line, line_number, record == "HETATM", records.len())?;
                let element_inferred = Element::from_str(field(line, 76, 78)).is_err();
                records.push(AtomRecord {
                    atom,
                    model,
                    element_inferred,
                });
            }
            "CONECT" => {
                if let Some(source) = hybrid36(field(line, 6, 11), 5) {
                    for start in [11, 16, 21, 26] {
                        if let Some(target) = hybrid36(field(line, start, start + 5), 5) {
                            conect.push((source as u32, target as u32));
                        }
                    }
                }
            }
            "SSBOND" => {
                let first = site(line, 15, 17, 21, "SG");
                let second = site(line, 29, 31, 35, "SG");
                if let (Some(a), Some(b)) = (first, second) {
                    bonds.push(BondRecord::Site {
                        a,
                        b,
                        kind: BondKind::Disulfide,
                        order: None,
                    });
                }
            }
            "LINK" | "LINKR" => {
                let first = named_site(line, 12, 21, 22, 26);
                let second = named_site(line, 42, 51, 52, 56);
                if let (Some(a), Some(b)) = (first, second) {
                    bonds.push(BondRecord::Site {
                        a,
                        b,
                        kind: BondKind::Covalent,
                        order: None,
                    });
                }
            }
            "HELIX" => {
                let class = field(line, 38, 40).trim().parse::<u32>().unwrap_or(1);
                let kind = match class {
                    3 => AnnotatedStructure::HelixPi,
                    5 => AnnotatedStructure::Helix310,
                    _ => AnnotatedStructure::Helix,
                };
                if let Some(annotation) = annotation(line, kind, 19, 21, 25, 31, 33, 37) {
                    info.secondary.push(annotation);
                }
            }
            "SHEET" => {
                if let Some(annotation) =
                    annotation(line, AnnotatedStructure::Strand, 21, 22, 26, 32, 33, 37)
                {
                    info.secondary.push(annotation);
                }
            }
            "CRYST1" => {
                let number = |start, end| field(line, start, end).trim().parse::<f64>().ok();
                if let (Some(a), Some(b), Some(c), Some(alpha), Some(beta), Some(gamma)) = (
                    number(6, 15),
                    number(15, 24),
                    number(24, 33),
                    number(33, 40),
                    number(40, 47),
                    number(47, 54),
                ) {
                    info.crystal = Some(CrystalInfo {
                        cell: UnitCell {
                            a,
                            b,
                            c,
                            alpha,
                            beta,
                            gamma,
                        },
                        // A cell without a space group is taken as P 1.
                        space_group: Some(field(line, 55, 66).trim())
                            .filter(|group| !group.is_empty())
                            .unwrap_or("P 1")
                            .to_string(),
                    });
                }
            }
            "REMARK" if field(line, 7, 10).trim() == "350" => remark350.line(line),
            _ => {}
        }
    }

    if records.is_empty() {
        return Err(PdbError::NoAtoms);
    }
    if !title.is_empty() {
        info.title = Some(title);
    }
    remark350.finish(&mut info);
    bonds.extend(serial_bond_records(&conect));
    let assembled = assemble(records, &bonds, info).ok_or(PdbError::NoAtoms)?;
    Ok(PdbDocument {
        molecule: assembled.molecule,
        frames: assembled.frames,
    })
}

/// Parses a PQR file (PDB2PQR, APBS): whitespace-separated ATOM records ending with the
/// partial charge and radius. Charges are stored as B-factors so they can be colored.
pub fn parse_pqr_document(input: &str) -> Result<PdbDocument, PdbError> {
    let mut records = Vec::new();
    let mut model: i64 = 1;
    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("MODEL") => {
                model = tokens
                    .get(1)
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(model + 1);
            }
            Some(record @ ("ATOM" | "HETATM")) => {
                if tokens.len() < 10 {
                    return Err(PdbError::MissingField {
                        line: line_number,
                        field: "PQR atom fields",
                    });
                }
                let number = |index: usize, field: &'static str| -> Result<f32, PdbError> {
                    tokens[index].parse().map_err(|_| PdbError::InvalidField {
                        line: line_number,
                        field,
                        value: tokens[index].to_string(),
                    })
                };
                let n = tokens.len();
                let position = Vec3::new(
                    number(n - 5, "x coordinate")?,
                    number(n - 4, "y coordinate")?,
                    number(n - 3, "z coordinate")?,
                );
                let charge = number(n - 2, "charge")?;
                // Between the residue name and x: [chain] residue number [insertion code].
                let middle = &tokens[4..n - 5];
                let residue_token = match middle {
                    [chain, residue, ..] if chain.parse::<i32>().is_err() => *residue,
                    [residue, ..] => *residue,
                    [] => {
                        return Err(PdbError::MissingField {
                            line: line_number,
                            field: "residue number",
                        });
                    }
                };
                let chain = match middle {
                    [chain, _, ..] if chain.parse::<i32>().is_err() => *chain,
                    _ => "",
                };
                let digits = residue_token.trim_end_matches(|c: char| c.is_ascii_alphabetic());
                let residue_number = digits.parse().map_err(|_| PdbError::InvalidField {
                    line: line_number,
                    field: "residue number",
                    value: residue_token.to_string(),
                })?;
                let insertion_code = residue_token[digits.len()..].chars().next();
                let name = tokens[2].to_ascii_uppercase();
                records.push(AtomRecord {
                    atom: Atom {
                        serial: tokens[1].parse().unwrap_or(records.len() as u32 + 1),
                        element: infer_element(&format!(" {name:<3}")),
                        name,
                        residue_name: tokens[3].to_ascii_uppercase(),
                        residue_number,
                        insertion_code,
                        chain_id: chain.to_string(),
                        position,
                        occupancy: 1.0,
                        b_factor: charge,
                        hetero: record == "HETATM",
                        ..Atom::default()
                    },
                    model,
                    element_inferred: true,
                });
            }
            _ => {}
        }
    }
    if records.is_empty() {
        return Err(PdbError::NoAtoms);
    }
    let info = StructureInfo {
        notes: vec!["PQR partial charges are stored as B-factors".into()],
        ..StructureInfo::default()
    };
    let assembled = assemble(records, &[], info).ok_or(PdbError::NoAtoms)?;
    Ok(PdbDocument {
        molecule: assembled.molecule,
        frames: assembled.frames,
    })
}

/// Chains of one `APPLY THE FOLLOWING TO CHAINS` block with its BIOMT operators.
type Remark350Generator = (Vec<String>, Vec<SymmetryOperator>);

/// Collects REMARK 350 biomolecule definitions: id, oligomer count, description, generators.
#[derive(Default)]
struct Remark350 {
    assemblies: Vec<(String, Option<u32>, String, Vec<Remark350Generator>)>,
    expecting_chains: bool,
    pending_rows: Vec<(usize, [f64; 4])>,
}

impl Remark350 {
    fn line(&mut self, line: &str) {
        let text = field(line, 10, 80).trim();
        if let Some(id) = text.strip_prefix("BIOMOLECULE:") {
            self.assemblies
                .push((id.trim().to_string(), None, String::new(), Vec::new()));
            self.expecting_chains = false;
            return;
        }
        let Some(assembly) = self.assemblies.last_mut() else {
            return;
        };
        if let Some(rest) = text
            .strip_prefix("AUTHOR DETERMINED BIOLOGICAL UNIT:")
            .or_else(|| text.strip_prefix("SOFTWARE DETERMINED QUATERNARY STRUCTURE:"))
        {
            let rest = rest.trim();
            if assembly.2.is_empty() {
                assembly.2 = rest.to_ascii_lowercase();
            }
            if assembly.1.is_none() {
                assembly.1 = oligomer_count(rest);
            }
        } else if let Some(chains) = text.strip_prefix("APPLY THE FOLLOWING TO CHAINS:") {
            assembly.3.push((parse_chain_list(chains), Vec::new()));
            self.expecting_chains = true;
        } else if let Some(chains) = text.strip_prefix("AND CHAINS:") {
            if self.expecting_chains
                && let Some(generator) = assembly.3.last_mut()
            {
                generator.0.extend(parse_chain_list(chains));
            }
        } else if let Some(rest) = text.strip_prefix("BIOMT") {
            self.expecting_chains = false;
            let mut parts = rest.split_whitespace();
            let row = parts
                .next()
                .and_then(|row| row.parse::<usize>().ok())
                .filter(|row| (1..=3).contains(row));
            let _serial = parts.next();
            let values: Vec<f64> = parts.filter_map(|value| value.parse().ok()).collect();
            if let (Some(row), 4) = (row, values.len()) {
                self.pending_rows
                    .push((row, [values[0], values[1], values[2], values[3]]));
                if row == 3 && self.pending_rows.len() >= 3 {
                    let rows = std::mem::take(&mut self.pending_rows);
                    let rows = &rows[rows.len() - 3..];
                    if let Some(generator) = assembly.3.last_mut() {
                        let index = generator.1.len() + 1;
                        generator.1.push(SymmetryOperator {
                            id: index.to_string(),
                            name: format!("BIOMT {index}"),
                            rotation: [
                                [rows[0].1[0], rows[0].1[1], rows[0].1[2]],
                                [rows[1].1[0], rows[1].1[1], rows[1].1[2]],
                                [rows[2].1[0], rows[2].1[1], rows[2].1[2]],
                            ],
                            translation: [rows[0].1[3], rows[1].1[3], rows[2].1[3]],
                        });
                    }
                }
            }
        }
    }

    fn finish(self, info: &mut StructureInfo) {
        for (id, count, details, generators) in self.assemblies {
            let mut assembly = Assembly {
                id,
                details,
                oligomeric_count: count,
                generators: Vec::new(),
            };
            for (chains, operators) in generators {
                if chains.is_empty() || operators.is_empty() {
                    continue;
                }
                let products = operators
                    .into_iter()
                    .map(|operator| {
                        info.operators.push(operator);
                        vec![info.operators.len() - 1]
                    })
                    .collect();
                assembly
                    .generators
                    .push(AssemblyGenerator { chains, products });
            }
            if !assembly.generators.is_empty() {
                info.assemblies.push(assembly);
            }
        }
    }
}

fn parse_chain_list(text: &str) -> Vec<String> {
    text.split([',', ' '])
        .map(str::trim)
        .filter(|chain| !chain.is_empty())
        .map(str::to_string)
        .collect()
}

fn oligomer_count(text: &str) -> Option<u32> {
    let text = text.trim().to_ascii_uppercase();
    let known = [
        ("MONOMERIC", 1),
        ("DIMERIC", 2),
        ("TRIMERIC", 3),
        ("TETRAMERIC", 4),
        ("PENTAMERIC", 5),
        ("HEXAMERIC", 6),
        ("HEPTAMERIC", 7),
        ("OCTAMERIC", 8),
        ("DECAMERIC", 10),
        ("DODECAMERIC", 12),
        ("TETRADECAMERIC", 14),
        ("ICOSAHEDRAL", 60),
    ];
    known
        .iter()
        .find(|(name, _)| text == *name)
        .map(|(_, count)| *count)
        .or_else(|| {
            text.strip_suffix("-MERIC")
                .and_then(|number| number.parse().ok())
        })
}

fn site(line: &str, chain: usize, start: usize, code: usize, name: &str) -> Option<SiteKey> {
    Some(SiteKey {
        chain: field(line, chain, chain + 1).trim().to_string(),
        residue_number: hybrid36(field(line, start, start + 4), 4)? as i32,
        insertion_code: insertion(field(line, code, code + 1)),
        atom_name: name.to_string(),
    })
}

fn named_site(line: &str, name: usize, chain: usize, start: usize, code: usize) -> Option<SiteKey> {
    let atom_name = field(line, name, name + 4).trim().to_ascii_uppercase();
    if atom_name.is_empty() {
        return None;
    }
    Some(SiteKey {
        chain: field(line, chain, chain + 1).trim().to_string(),
        residue_number: hybrid36(field(line, start, start + 4), 4)? as i32,
        insertion_code: insertion(field(line, code, code + 1)),
        atom_name,
    })
}

#[allow(clippy::too_many_arguments)]
fn annotation(
    line: &str,
    kind: AnnotatedStructure,
    start_chain: usize,
    start_number: usize,
    start_code: usize,
    end_chain: usize,
    end_number: usize,
    end_code: usize,
) -> Option<SecondaryAnnotation> {
    let chain = field(line, start_chain, start_chain + 1).trim().to_string();
    let end_chain = field(line, end_chain, end_chain + 1).trim();
    if !end_chain.is_empty() && end_chain != chain {
        return None;
    }
    Some(SecondaryAnnotation {
        kind,
        chain,
        start: (
            hybrid36(field(line, start_number, start_number + 4), 4)? as i32,
            insertion(field(line, start_code, start_code + 1)),
        ),
        end: (
            hybrid36(field(line, end_number, end_number + 4), 4)? as i32,
            insertion(field(line, end_code, end_code + 1)),
        ),
    })
}

fn insertion(value: &str) -> Option<char> {
    value.chars().next().filter(|c| !c.is_ascii_whitespace())
}

fn parse_atom(
    line: &str,
    line_number: usize,
    hetero: bool,
    ordinal: usize,
) -> Result<Atom, PdbError> {
    // Serials that overflow the field ("*****") are only needed for CONECT.
    let serial = hybrid36(field(line, 6, 11), 5)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(ordinal as u32 + 1);
    let raw_name = field(line, 12, 16);
    let name = raw_name.trim().to_string();
    if name.is_empty() {
        return Err(PdbError::MissingField {
            line: line_number,
            field: "atom name",
        });
    }
    let alt_loc = insertion(field(line, 16, 17));
    // Column 21 is unused by the standard but carries a fourth residue-name character in
    // some large-ligand files.
    let residue_name = format!("{}{}", field(line, 17, 20), field(line, 20, 21))
        .trim()
        .to_ascii_uppercase();
    if residue_name.is_empty() {
        return Err(PdbError::MissingField {
            line: line_number,
            field: "residue name",
        });
    }
    let residue_field = field(line, 22, 26);
    let residue_number = hybrid36(residue_field, 4).ok_or_else(|| {
        if residue_field.trim().is_empty() {
            PdbError::MissingField {
                line: line_number,
                field: "residue number",
            }
        } else {
            PdbError::InvalidField {
                line: line_number,
                field: "residue number",
                value: residue_field.trim().to_string(),
            }
        }
    })? as i32;
    let insertion_code = insertion(field(line, 26, 27));
    let chain_id = field(line, 21, 22).trim().to_string();
    let x = required::<f32>(line, line_number, 30, 38, "x coordinate")?;
    let y = required::<f32>(line, line_number, 38, 46, "y coordinate")?;
    let z = required::<f32>(line, line_number, 46, 54, "z coordinate")?;
    let occupancy = optional_number(line, line_number, 54, 60, "occupancy", 1.0)?;
    let b_factor = optional_number(line, line_number, 60, 66, "B-factor", 0.0)?;
    let element = Element::from_str(field(line, 76, 78)).unwrap_or_else(|()| {
        let guess = infer_element(raw_name);
        // Polymer (ATOM) records never hold metal ions, which come as HETATM residues named
        // after their element; "CA  " in an ATOM record is an alpha carbon.
        if guess.is_metal() && !hetero && residue_name != name.to_ascii_uppercase() {
            name.chars()
                .next()
                .and_then(|first| Element::from_str(&first.to_string()).ok())
                .unwrap_or(guess)
        } else {
            guess
        }
    });
    let formal_charge = parse_charge(field(line, 78, 80));

    Ok(Atom {
        serial,
        name: name.to_ascii_uppercase(),
        element,
        residue_name,
        residue_number,
        insertion_code,
        chain_id,
        position: Vec3::new(x, y, z),
        occupancy,
        b_factor,
        hetero,
        alt_loc,
        formal_charge,
    })
}

/// Parses a PDB charge such as `2+` or `1-`.
fn parse_charge(value: &str) -> i8 {
    let value = value.trim();
    let (digits, sign) = if let Some(digits) = value.strip_suffix('+') {
        (digits, 1)
    } else if let Some(digits) = value.strip_suffix('-') {
        (digits, -1)
    } else {
        return value.parse().unwrap_or(0);
    };
    digits
        .trim()
        .parse::<i8>()
        .map_or(0, |magnitude| sign * magnitude)
}

/// Decodes a hybrid-36 number field of the given width, which extends decimal columns
/// with base-36 digits (`A0000` = 100000 for width 5).
pub(crate) fn hybrid36(value: &str, width: u32) -> Option<i64> {
    let trimmed = value.trim();
    let first = trimmed.chars().next()?;
    if first.is_ascii_digit() || first == '-' || first == '+' {
        return trimmed.parse().ok();
    }
    if trimmed.len() != width as usize {
        return None;
    }
    let base = 36_i64.pow(width - 1);
    let decimal_limit = 10_i64.pow(width);
    let decode = |uppercase: bool| {
        trimmed.chars().try_fold(0_i64, |total, character| {
            let digit = match character {
                '0'..='9' => character as i64 - '0' as i64,
                'A'..='Z' if uppercase => character as i64 - 'A' as i64 + 10,
                'a'..='z' if !uppercase => character as i64 - 'a' as i64 + 10,
                _ => return None,
            };
            Some(total * 36 + digit)
        })
    };
    if first.is_ascii_uppercase() {
        Some(decode(true)? - 10 * base + decimal_limit)
    } else if first.is_ascii_lowercase() {
        Some(decode(false)? + 16 * base + decimal_limit)
    } else {
        None
    }
}

/// Infers the element from the aligned 4-character PDB atom name field.
pub(crate) fn infer_element(raw_name: &str) -> Element {
    let bytes = raw_name.as_bytes();
    let trimmed = raw_name.trim();
    if bytes.first().is_some_and(u8::is_ascii_whitespace) {
        return trimmed
            .chars()
            .next()
            .and_then(|c| Element::from_str(&c.to_string()).ok())
            .unwrap_or(Element::Unknown);
    }
    // Four-character names of hydrogens start in column 13 (HG12, HD21).
    if raw_name.len() >= 4 && !raw_name.ends_with(' ') && trimmed.starts_with(['H', 'h']) {
        return Element::H;
    }
    let letters: String = trimmed
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .take(2)
        .collect();
    if letters.len() == 2
        && let Ok(element) = Element::from_str(&letters)
        && (element.is_metal() || matches!(element, Element::Cl | Element::Br | Element::Se))
    {
        return element;
    }
    letters
        .chars()
        .next()
        .and_then(|c| Element::from_str(&c.to_string()).ok())
        .unwrap_or(Element::Unknown)
}

fn required<T: FromStr>(
    line: &str,
    line_number: usize,
    start: usize,
    end: usize,
    name: &'static str,
) -> Result<T, PdbError> {
    let value = field(line, start, end).trim();
    if value.is_empty() {
        return Err(PdbError::MissingField {
            line: line_number,
            field: name,
        });
    }
    value.parse().map_err(|_| PdbError::InvalidField {
        line: line_number,
        field: name,
        value: value.to_string(),
    })
}

fn optional_number(
    line: &str,
    line_number: usize,
    start: usize,
    end: usize,
    name: &'static str,
    default: f32,
) -> Result<f32, PdbError> {
    let value = field(line, start, end).trim();
    if value.is_empty() {
        Ok(default)
    } else {
        value.parse().map_err(|_| PdbError::InvalidField {
            line: line_number,
            field: name,
            value: value.to_string(),
        })
    }
}

fn field(line: &str, start: usize, end: usize) -> &str {
    if start >= line.len() {
        return "";
    }
    let end = end.min(line.len());
    line.get(start..end).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::{Bond, BondOrder};

    const PDB: &str = concat!(
        "ATOM      1  N   ALA A  10      11.104  13.207   9.110  1.00 20.00           N  \n",
        "ATOM      2  CA AALA A  10      12.500  13.100   9.000  0.60 21.00              \n",
        "ATOM      3  C  BALA A  10      13.000  14.500   9.100  0.40 22.00           C  \n",
        "HETATM    4  O   HOH B 201      20.000  20.000  20.000  1.00 10.00           O  \n",
        "CONECT    1    2\n",
    );

    #[test]
    fn parses_atom_hetatm_fields_altloc_and_conect() {
        let molecule = parse_pdb(PDB).unwrap();
        // The residue's more occupied conformer (A) is kept.
        assert_eq!(molecule.atoms.len(), 3);
        let alpha = &molecule.atoms[1];
        assert_eq!(alpha.name, "CA");
        assert_eq!(alpha.alt_loc, Some('A'));
        assert_eq!(alpha.residue_name, "ALA");
        assert_eq!(alpha.residue_number, 10);
        assert_eq!(alpha.chain_id, "A");
        assert_eq!(alpha.position, Vec3::new(12.5, 13.1, 9.0));
        assert_eq!(alpha.element, Element::C);
        assert!(molecule.atoms[2].hetero);
        assert!(molecule.bonds.contains(&Bond::new(0, 1).unwrap()));
    }

    #[test]
    fn parses_insertion_code_charge_and_defaults_optional_numbers() {
        let line =
            "ATOM      8  O   GLY A  42B      1.000   2.000   3.000                       O1-";
        let atom = parse_atom(line, 1, false, 0).unwrap();
        assert_eq!(atom.insertion_code, Some('B'));
        assert_eq!(atom.occupancy, 1.0);
        assert_eq!(atom.b_factor, 0.0);
        assert_eq!(atom.formal_charge, -1);
    }

    #[test]
    fn decodes_hybrid36_fields() {
        assert_eq!(hybrid36("99999", 5), Some(99_999));
        assert_eq!(hybrid36("A0000", 5), Some(100_000));
        assert_eq!(hybrid36("A0001", 5), Some(100_001));
        assert_eq!(hybrid36("ZZZZZ", 5), Some(100_000 + 26 * 36_i64.pow(4) - 1));
        assert_eq!(hybrid36("a0000", 5), Some(100_000 + 26 * 36_i64.pow(4)));
        assert_eq!(hybrid36("A000", 4), Some(10_000));
        assert_eq!(hybrid36(" -12", 4), Some(-12));
        assert_eq!(hybrid36("A0-0", 4), None);
        let line =
            "ATOM  A0000  CA  GLY AA000       0.000   0.000   0.000  1.00  0.00           C  ";
        let atom = parse_atom(line, 1, false, 0).unwrap();
        assert_eq!(atom.serial, 100_000);
        assert_eq!(atom.residue_number, 10_000);
    }

    #[test]
    fn reports_malformed_required_fields() {
        let error =
            parse_pdb("ATOM      1  C   GLY A   1       0.000   X.000   0.000").unwrap_err();
        assert!(matches!(
            error,
            PdbError::InvalidField {
                field: "y coordinate",
                ..
            }
        ));
    }

    #[test]
    fn later_models_become_frames() {
        let pdb = concat!(
            "MODEL        1\n",
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00  0.00           C  \n",
            "ENDMDL\nMODEL        2\n",
            "ATOM      1  C   GLY A   1       1.000   0.000   0.000  1.00  0.00           C  \n",
            "ENDMDL\n",
        );
        let document = parse_pdb_document(pdb).unwrap();
        assert_eq!(document.molecule.atoms.len(), 1);
        assert_eq!(document.frames, vec![vec![Vec3::new(1.0, 0.0, 0.0)]]);
        assert_eq!(document.molecule.info.model_count, 2);
    }

    #[test]
    fn reads_annotations_symmetry_and_assemblies() {
        let pdb = concat!(
            "HEADER    HYDROLASE                               01-JAN-00   1ABC              \n",
            "TITLE     A TEST\n",
            "TITLE    2 STRUCTURE\n",
            "HELIX    1   1 ALA A    2  ALA A    9  1                                   8    \n",
            "SHEET    1   A 2 GLY B  10  GLY B  14  0                                        \n",
            "CRYST1   61.200   61.200   97.100  90.00  90.00 120.00 P 31 2 1      6          \n",
            "REMARK 350 BIOMOLECULE: 1\n",
            "REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: DIMERIC\n",
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A,\n",
            "REMARK 350                    AND CHAINS: B\n",
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000\n",
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000\n",
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000\n",
            "REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000       10.00000\n",
            "REMARK 350   BIOMT2   2  0.000000 -1.000000  0.000000        0.00000\n",
            "REMARK 350   BIOMT3   2  0.000000  0.000000  1.000000        0.00000\n",
            "ATOM      1  CA  ALA A   2       0.000   0.000   0.000  1.00  0.00           C  \n",
            "ATOM      2  CA  GLY B  10       5.000   0.000   0.000  1.00  0.00           C  \n",
        );
        let molecule = parse_pdb(pdb).unwrap();
        let info = &molecule.info;
        assert_eq!(info.id.as_deref(), Some("1ABC"));
        assert_eq!(info.title.as_deref(), Some("A TEST STRUCTURE"));
        assert_eq!(info.secondary.len(), 2);
        assert_eq!(info.secondary[0].kind, AnnotatedStructure::Helix);
        assert_eq!(info.secondary[1].start, (10, None));
        let crystal = info.crystal.as_ref().unwrap();
        assert_eq!(crystal.space_group, "P 31 2 1");
        assert_eq!(crystal.cell.gamma, 120.0);
        assert_eq!(info.assemblies.len(), 1);
        let assembly = &info.assemblies[0];
        assert_eq!(assembly.oligomeric_count, Some(2));
        assert_eq!(assembly.generators[0].chains, vec!["A", "B"]);
        assert_eq!(assembly.generators[0].products.len(), 2);
        assert_eq!(info.operators[1].translation, [10.0, 0.0, 0.0]);
    }

    #[test]
    fn ssbond_and_link_records_create_bonds() {
        let pdb = concat!(
            "SSBOND   1 CYS A    1    CYS A    5                          1555   1555  2.04  \n",
            "LINK         O   HOH A 100                ZN    ZN A 101     1555   1555  2.10  \n",
            "ATOM      1  SG  CYS A   1       0.000   0.000   0.000  1.00  0.00           S  \n",
            "ATOM      2  SG  CYS A   5       2.500   0.000   0.000  1.00  0.00           S  \n",
            "HETATM    3  O   HOH A 100      10.000   0.000   0.000  1.00  0.00           O  \n",
            "HETATM    4 ZN    ZN A 101      12.100   0.000   0.000  1.00  0.00          ZN  \n",
        );
        let molecule = parse_pdb(pdb).unwrap();
        assert_eq!(molecule.atoms[3].element, Element::Zn);
        let disulfide = molecule
            .bonds
            .iter()
            .find(|b| b.a == 0 && b.b == 1)
            .unwrap();
        assert_eq!(disulfide.kind, BondKind::Disulfide);
        let metal = molecule
            .bonds
            .iter()
            .find(|b| b.a == 2 && b.b == 3)
            .unwrap();
        assert_eq!(metal.kind, BondKind::MetalCoordination);
        assert_eq!(metal.order, BondOrder::Single);
    }

    #[test]
    fn reads_whitespace_separated_pqr() {
        let pqr = concat!(
            "ATOM      1  N    ILE    16       5.007   -9.234   18.432 -0.3000 1.8500
",
            "ATOM      2  CA   ILE A  17A      4.405   -8.908   19.756  0.2100 2.2750
",
        );
        let molecule = parse_pqr_document(pqr).unwrap().molecule;
        assert_eq!(molecule.atoms[0].residue_number, 16);
        assert_eq!(molecule.atoms[0].b_factor, -0.3);
        assert_eq!(molecule.atoms[1].chain_id, "A");
        assert_eq!(molecule.atoms[1].insertion_code, Some('A'));
        assert_eq!(molecule.atoms[1].element, Element::C);
    }

    #[test]
    fn infers_elements_from_atom_names() {
        assert_eq!(infer_element(" CA "), Element::C);
        assert_eq!(infer_element("CA  "), Element::Ca);
        assert_eq!(infer_element("HG12"), Element::H);
        assert_eq!(infer_element("FE1 "), Element::Fe);
        assert_eq!(infer_element("SE  "), Element::Se);
        assert_eq!(infer_element("CL1 "), Element::Cl);
        assert_eq!(infer_element("1HB "), Element::H);
    }

    #[test]
    fn bundled_example_is_valid() {
        let molecule = parse_pdb(include_str!("../../examples/minimal.pdb")).unwrap();
        assert_eq!(molecule.atoms.len(), 13);
        assert!(molecule.bonds.len() >= 10);
        assert_eq!(molecule.atoms[9].element, Element::Zn);
    }
}
