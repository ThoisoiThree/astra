use std::{collections::HashMap, str::FromStr};

use glam::Vec3;
use thiserror::Error;

use super::{Atom, Bond, Element, Molecule, infer_bonds};

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

/// Parses the first model of a PDB document and supplements explicit connectivity.
pub fn parse_pdb(input: &str) -> Result<Molecule, PdbError> {
    let mut atoms = Vec::new();
    let mut serial_to_index = HashMap::new();
    let mut conect_serials = Vec::new();
    let mut saw_model = false;
    let mut inside_first_model = true;

    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let record = field(line, 0, 6).trim();
        match record {
            "MODEL" => {
                if saw_model {
                    inside_first_model = false;
                } else {
                    saw_model = true;
                    inside_first_model = true;
                }
            }
            "ENDMDL" if saw_model && inside_first_model => inside_first_model = false,
            "ATOM" | "HETATM" if inside_first_model => {
                let altloc = field(line, 16, 17).chars().next().unwrap_or(' ');
                if !matches!(altloc, ' ' | 'A') {
                    continue;
                }
                let atom = parse_atom(line, line_number, record == "HETATM")?;
                serial_to_index.insert(atom.serial, atoms.len());
                atoms.push(atom);
            }
            "CONECT" => {
                let source = parse_optional::<u32>(field(line, 6, 11));
                if let Some(source) = source {
                    for start in (11..line.len()).step_by(5) {
                        if let Some(target) = parse_optional::<u32>(field(line, start, start + 5)) {
                            conect_serials.push((source, target));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if atoms.is_empty() {
        return Err(PdbError::NoAtoms);
    }

    let mut explicit = Vec::new();
    for (source, target) in conect_serials {
        if let (Some(&a), Some(&b)) = (serial_to_index.get(&source), serial_to_index.get(&target))
            && let Some(bond) = Bond::new(a, b)
            && !explicit.contains(&bond)
        {
            explicit.push(bond);
        }
    }
    explicit.sort_by_key(|bond| (bond.a, bond.b));
    let bonds = infer_bonds(&atoms, &explicit);

    Ok(Molecule { atoms, bonds })
}

fn parse_atom(line: &str, line_number: usize, hetero: bool) -> Result<Atom, PdbError> {
    let serial = required::<u32>(line, line_number, 6, 11, "atom serial")?;
    let raw_name = field(line, 12, 16);
    let name = raw_name.trim().to_string();
    if name.is_empty() {
        return Err(PdbError::MissingField {
            line: line_number,
            field: "atom name",
        });
    }
    let residue_name = field(line, 17, 20).trim().to_ascii_uppercase();
    if residue_name.is_empty() {
        return Err(PdbError::MissingField {
            line: line_number,
            field: "residue name",
        });
    }
    let residue_number = required::<i32>(line, line_number, 22, 26, "residue number")?;
    let insertion_code = field(line, 26, 27)
        .chars()
        .next()
        .filter(|character| !character.is_ascii_whitespace());
    let chain_id = field(line, 21, 22).trim().to_string();
    let x = required::<f32>(line, line_number, 30, 38, "x coordinate")?;
    let y = required::<f32>(line, line_number, 38, 46, "y coordinate")?;
    let z = required::<f32>(line, line_number, 46, 54, "z coordinate")?;
    let occupancy = optional_number(line, line_number, 54, 60, "occupancy", 1.0)?;
    let b_factor = optional_number(line, line_number, 60, 66, "B-factor", 0.0)?;
    let element =
        Element::from_str(field(line, 76, 78)).unwrap_or_else(|()| infer_element(raw_name));

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
    })
}

pub(crate) fn infer_element(raw_name: &str) -> Element {
    let bytes = raw_name.as_bytes();
    let candidate = if bytes.first().is_some_and(u8::is_ascii_whitespace) {
        raw_name.trim().chars().next().map(|c| c.to_string())
    } else {
        let letters: String = raw_name
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .chars()
            .take(2)
            .collect();
        Element::from_str(&letters)
            .ok()
            .filter(|element| {
                matches!(
                    element,
                    Element::Cl
                        | Element::Br
                        | Element::Na
                        | Element::Mg
                        | Element::Ca
                        | Element::Fe
                        | Element::Zn
                        | Element::Li
                        | Element::Rb
                        | Element::Cs
                        | Element::Be
                        | Element::Sr
                        | Element::Ba
                )
            })
            .map(|element| element.symbol().to_string())
            .or_else(|| letters.chars().next().map(|c| c.to_string()))
    };
    candidate
        .as_deref()
        .and_then(|value| Element::from_str(value).ok())
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

fn parse_optional<T: FromStr>(value: &str) -> Option<T> {
    value.trim().parse().ok()
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

    const PDB: &str = concat!(
        "ATOM      1  N   ALA A  10      11.104  13.207   9.110  1.00 20.00           N  \n",
        "ATOM      2  CA AALA A  10      12.500  13.100   9.000  0.50 21.00              \n",
        "ATOM      3  C  BALA A  10      13.000  14.500   9.100  0.50 22.00           C  \n",
        "HETATM    4  O   HOH B 201      20.000  20.000  20.000  1.00 10.00           O  \n",
        "CONECT    1    2\n",
    );

    #[test]
    fn parses_atom_hetatm_fields_altloc_and_conect() {
        let molecule = parse_pdb(PDB).unwrap();
        assert_eq!(molecule.atoms.len(), 3);
        let alpha = &molecule.atoms[1];
        assert_eq!(alpha.name, "CA");
        assert_eq!(alpha.residue_name, "ALA");
        assert_eq!(alpha.residue_number, 10);
        assert_eq!(alpha.chain_id, "A");
        assert_eq!(alpha.position, Vec3::new(12.5, 13.1, 9.0));
        assert_eq!(alpha.element, Element::C);
        assert!(molecule.atoms[2].hetero);
        assert!(molecule.bonds.contains(&Bond { a: 0, b: 1 }));
    }

    #[test]
    fn parses_insertion_code_and_defaults_optional_numbers() {
        let line =
            "ATOM      8  O   GLY A  42B      1.000   2.000   3.000                      O  ";
        let atom = parse_atom(line, 1, false).unwrap();
        assert_eq!(atom.insertion_code, Some('B'));
        assert_eq!(atom.occupancy, 1.0);
        assert_eq!(atom.b_factor, 0.0);
    }

    #[test]
    fn reports_malformed_required_fields() {
        let error =
            parse_pdb("ATOM      X  C   GLY A   1       0.000   0.000   0.000").unwrap_err();
        assert!(matches!(
            error,
            PdbError::InvalidField {
                field: "atom serial",
                ..
            }
        ));
    }

    #[test]
    fn reads_only_first_model() {
        let pdb = concat!(
            "MODEL        1\n",
            "ATOM      1  C   GLY A   1       0.000   0.000   0.000  1.00  0.00           C  \n",
            "ENDMDL\nMODEL        2\n",
            "ATOM      2  O   GLY A   1       1.000   0.000   0.000  1.00  0.00           O  \n",
            "ENDMDL\n",
        );
        assert_eq!(parse_pdb(pdb).unwrap().atoms.len(), 1);
    }

    #[test]
    fn bundled_example_is_valid() {
        let molecule = parse_pdb(include_str!("../../examples/minimal.pdb")).unwrap();
        assert_eq!(molecule.atoms.len(), 13);
        assert!(molecule.bonds.len() >= 10);
        assert_eq!(molecule.atoms[9].element, Element::Zn);
    }
}
