//! Simple coordinate formats that also carry a minimal topology: GROMACS GRO and XYZ.
//! Both may contain several frames; later frames become trajectory frames.

use glam::Vec3;

use super::{
    Atom, Element, Molecule, StructureError, StructureInfo, bonds::build_bonds, pdb::infer_element,
};

type Parsed = (Molecule, Vec<Vec<Vec3>>);

const ION_RESIDUES: &[&str] = &[
    "NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "BR", "IOD", "F", "MN", "FE", "CU", "CO",
    "NI", "CD", "SR", "BA",
];
const SOLVENT_RESIDUES: &[&str] = &["SOL", "HOH", "WAT", "TIP3", "TIP4", "TIP5", "SPC", "T3P"];

fn coordinates_error(
    format: &'static str,
    line: usize,
    message: impl Into<String>,
) -> StructureError {
    StructureError::Coordinates {
        format,
        line,
        message: message.into(),
    }
}

/// Parses a GROMACS `.gro` file (coordinates in nm; converted to Å).
pub(crate) fn parse_gro(input: &str) -> Result<Parsed, StructureError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut cursor = 0;
    let mut topology: Option<Vec<Atom>> = None;
    let mut frames = Vec::new();
    let mut info = StructureInfo::default();
    while cursor < lines.len() {
        if lines[cursor].trim().is_empty() {
            cursor += 1;
            continue;
        }
        let title = lines[cursor].trim();
        let count_line = cursor + 1;
        let count: usize = lines
            .get(count_line)
            .and_then(|line| line.trim().parse().ok())
            .ok_or_else(|| coordinates_error("GRO", count_line + 1, "missing atom count"))?;
        if cursor + 2 + count > lines.len() {
            return Err(coordinates_error(
                "GRO",
                lines.len(),
                format!("frame declares {count} atoms but the file ends early"),
            ));
        }
        let mut atoms = Vec::with_capacity(count);
        let mut previous_residue: Option<i32> = None;
        let mut chain_index = 0_usize;
        for offset in 0..count {
            let line_number = cursor + 2 + offset;
            let line = lines[line_number];
            let atom = parse_gro_atom(line, line_number + 1, atoms.len())?;
            // GRO has no chain identifiers: a new chain starts whenever numbering restarts,
            // which also keeps residues distinct after the 99999 wrap.
            if let Some(previous) = previous_residue
                && atom.residue_number < previous
            {
                chain_index += 1;
            }
            previous_residue = Some(atom.residue_number);
            atoms.push(Atom {
                chain_id: chain_name(chain_index),
                ..atom
            });
        }
        cursor += 2 + count;
        // The box vector line follows each frame.
        if cursor < lines.len() && !lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        match &topology {
            None => {
                if !title.is_empty() {
                    info.title = Some(title.to_string());
                }
                topology = Some(atoms);
            }
            Some(existing) if existing.len() == atoms.len() => {
                frames.push(atoms.into_iter().map(|atom| atom.position).collect());
            }
            Some(existing) => {
                return Err(coordinates_error(
                    "GRO",
                    count_line + 1,
                    format!(
                        "frame has {} atoms, the first frame has {}",
                        atoms.len(),
                        existing.len()
                    ),
                ));
            }
        }
    }
    let atoms = topology.ok_or(StructureError::NoCoordinates("GRO file"))?;
    info.model_count = frames.len() + 1;
    let bonds = build_bonds(&atoms, &[]);
    Ok((Molecule { atoms, bonds, info }, frames))
}

fn chain_name(index: usize) -> String {
    const LETTERS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if index < LETTERS.len() {
        char::from(LETTERS[index]).to_string()
    } else {
        format!("C{}", index + 1)
    }
}

fn parse_gro_atom(line: &str, line_number: usize, ordinal: usize) -> Result<Atom, StructureError> {
    let field = |start: usize, end: usize| line.get(start..end.min(line.len())).unwrap_or("");
    let residue_number = field(0, 5)
        .trim()
        .parse::<i32>()
        .map_err(|_| coordinates_error("GRO", line_number, "invalid residue number"))?;
    let residue_name = field(5, 10).trim().to_ascii_uppercase();
    let name = field(10, 15).trim().to_ascii_uppercase();
    if name.is_empty() || residue_name.is_empty() {
        return Err(coordinates_error(
            "GRO",
            line_number,
            "missing residue or atom name",
        ));
    }
    // Precision is variable: the distance between the first two decimal points gives the
    // field width.
    let coordinates = line.get(20..).unwrap_or("");
    let dots: Vec<usize> = coordinates
        .match_indices('.')
        .map(|(index, _)| index)
        .take(2)
        .collect();
    let width = if dots.len() == 2 {
        dots[1] - dots[0]
    } else {
        8
    };
    let value = |index: usize| -> Result<f32, StructureError> {
        coordinates
            .get(index * width..(index + 1) * width)
            .and_then(|text| text.trim().parse::<f32>().ok())
            .map(|nm| nm * 10.0)
            .ok_or_else(|| coordinates_error("GRO", line_number, "invalid coordinate"))
    };
    let position = Vec3::new(value(0)?, value(1)?, value(2)?);
    let element = gro_element(&residue_name, &name);
    Ok(Atom {
        serial: field(15, 20).trim().parse().unwrap_or(ordinal as u32 + 1),
        hetero: SOLVENT_RESIDUES.contains(&residue_name.as_str())
            || ION_RESIDUES.contains(&residue_name.as_str()),
        name,
        element,
        residue_name,
        residue_number,
        position,
        occupancy: 1.0,
        ..Atom::default()
    })
}

fn gro_element(residue: &str, name: &str) -> Element {
    let bare = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '+' || c == '-');
    if ION_RESIDUES.contains(&residue.trim_end_matches(['+', '-']))
        && let Ok(element) = bare.parse::<Element>()
    {
        return element;
    }
    infer_element(&format!(" {name:<3}"))
}

/// Parses an XYZ file with one or more frames of equal size.
pub(crate) fn parse_xyz(input: &str) -> Result<Parsed, StructureError> {
    let lines: Vec<&str> = input.lines().collect();
    let mut cursor = 0;
    let mut topology: Option<Vec<Atom>> = None;
    let mut frames = Vec::new();
    let mut info = StructureInfo::default();
    while cursor < lines.len() {
        if lines[cursor].trim().is_empty() {
            cursor += 1;
            continue;
        }
        let count: usize = lines[cursor]
            .trim()
            .parse()
            .map_err(|_| coordinates_error("XYZ", cursor + 1, "expected an atom count"))?;
        let comment = lines.get(cursor + 1).copied().unwrap_or_default().trim();
        if cursor + 2 + count > lines.len() {
            return Err(coordinates_error(
                "XYZ",
                lines.len(),
                format!("frame declares {count} atoms but the file ends early"),
            ));
        }
        let mut atoms = Vec::with_capacity(count);
        for offset in 0..count {
            let line_number = cursor + 2 + offset;
            let parts: Vec<&str> = lines[line_number].split_whitespace().collect();
            if parts.len() < 4 {
                return Err(coordinates_error(
                    "XYZ",
                    line_number + 1,
                    "expected an element and three coordinates",
                ));
            }
            let element = parts[0]
                .parse::<Element>()
                .ok()
                .or_else(|| {
                    parts[0]
                        .parse::<u32>()
                        .ok()
                        .and_then(Element::from_atomic_number)
                })
                .unwrap_or(Element::Unknown);
            let number = |index: usize| {
                parts[index]
                    .parse::<f32>()
                    .map_err(|_| coordinates_error("XYZ", line_number + 1, "invalid coordinate"))
            };
            atoms.push(Atom {
                serial: offset as u32 + 1,
                name: format!("{}{}", element.symbol().to_ascii_uppercase(), offset + 1),
                element,
                residue_name: "UNL".into(),
                residue_number: 1,
                chain_id: "A".into(),
                position: Vec3::new(number(1)?, number(2)?, number(3)?),
                occupancy: 1.0,
                hetero: true,
                ..Atom::default()
            });
        }
        cursor += 2 + count;
        match &topology {
            None => {
                if !comment.is_empty() {
                    info.title = Some(comment.to_string());
                }
                topology = Some(atoms);
            }
            Some(existing) if existing.len() == atoms.len() => {
                frames.push(atoms.into_iter().map(|atom| atom.position).collect());
            }
            Some(existing) => {
                return Err(coordinates_error(
                    "XYZ",
                    cursor - count - 1,
                    format!(
                        "frame has {} atoms, the first frame has {}",
                        atoms.len(),
                        existing.len()
                    ),
                ));
            }
        }
    }
    let atoms = topology.ok_or(StructureError::NoCoordinates("XYZ file"))?;
    info.model_count = frames.len() + 1;
    let bonds = build_bonds(&atoms, &[]);
    Ok((Molecule { atoms, bonds, info }, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRO: &str = "Water and ion t=   0.00000\n    4\n    1SOL     OW    1   0.126   1.624   1.679\n    1SOL    HW1    2   0.190   1.661   1.747\n    1SOL    HW2    3   0.177   1.568   1.613\n    2NA      NA    4   1.000   1.000   1.000\n   3.00000   3.00000   3.00000\nWater and ion t=   1.00000\n    4\n    1SOL     OW    1   0.226   1.624   1.679\n    1SOL    HW1    2   0.290   1.661   1.747\n    1SOL    HW2    3   0.277   1.568   1.613\n    2NA      NA    4   1.100   1.000   1.000\n   3.00000   3.00000   3.00000\n";

    #[test]
    fn reads_gro_frames_in_angstrom() {
        let (molecule, frames) = parse_gro(GRO).unwrap();
        assert_eq!(molecule.atoms.len(), 4);
        assert_eq!(molecule.atoms[0].element, Element::O);
        assert_eq!(molecule.atoms[1].element, Element::H);
        assert_eq!(molecule.atoms[3].element, Element::Na);
        assert!((molecule.atoms[0].position.x - 1.26).abs() < 1e-5);
        assert_eq!(frames.len(), 1);
        assert!((frames[0][3].x - 11.0).abs() < 1e-5);
        assert_eq!(
            molecule.bonds.iter().filter(|bond| bond.a == 0).count(),
            2,
            "water oxygen bonds both hydrogens"
        );
        assert!(molecule.bonds.iter().all(|bond| bond.b != 3));
    }

    #[test]
    fn gro_chains_follow_residue_numbering_restarts() {
        let gro = "t\n    2\n    5ALA     CA    1   0.000   0.000   0.000\n    1ALA     CA    2   1.000   0.000   0.000\n 1 1 1\n";
        let (molecule, _) = parse_gro(gro).unwrap();
        assert_eq!(molecule.atoms[0].chain_id, "A");
        assert_eq!(molecule.atoms[1].chain_id, "B");
    }

    #[test]
    fn reads_multi_frame_xyz() {
        let xyz = "3\nwater\nO 0 0 0\nH 0.96 0 0\n1 -0.24 0.93 0\n3\n\nO 0 0 1\nH 0.96 0 1\nH -0.24 0.93 1\n";
        let (molecule, frames) = parse_xyz(xyz).unwrap();
        assert_eq!(molecule.atoms[2].element, Element::H);
        assert_eq!(molecule.bonds.len(), 2);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0][0], Vec3::new(0.0, 0.0, 1.0));
        assert!(parse_xyz("2\n\nO 0 0 0\n").is_err());
    }
}
