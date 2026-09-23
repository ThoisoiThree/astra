use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    bitset::AtomMask,
    molecule::{Atom, Molecule},
};

use super::SelectionExpr;

/// Renderer-independent selection result. The representation can later become a bitset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    flags: AtomMask,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SelectionEvaluationError {
    #[error("named selection '{0}' does not exist")]
    UnknownNamedSelection(String),
}

impl Selection {
    pub fn from_flags(flags: Vec<bool>) -> Self {
        Self {
            flags: AtomMask::from_bools(flags),
        }
    }

    pub fn from_mask(flags: AtomMask) -> Self {
        Self { flags }
    }

    pub fn flags(&self) -> &AtomMask {
        &self.flags
    }

    pub fn count(&self) -> usize {
        self.flags.count()
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.flags.indices()
    }
}

pub fn evaluate(expression: &SelectionExpr, molecule: &Molecule) -> Selection {
    let named = BTreeMap::new();
    Selection {
        flags: AtomMask::from_bools(
            molecule
                .atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| matches_atom(expression, atom, index, &named)),
        ),
    }
}

pub fn evaluate_with_named(
    expression: &SelectionExpr,
    molecule: &Molecule,
    named: &BTreeMap<String, Selection>,
) -> Result<Selection, SelectionEvaluationError> {
    validate_named(expression, named)?;
    Ok(Selection {
        flags: AtomMask::from_bools(
            molecule
                .atoms
                .iter()
                .enumerate()
                .map(|(index, atom)| matches_atom(expression, atom, index, named)),
        ),
    })
}

fn validate_named(
    expression: &SelectionExpr,
    named: &BTreeMap<String, Selection>,
) -> Result<(), SelectionEvaluationError> {
    match expression {
        SelectionExpr::Named(name) if find_named(named, name).is_none() => Err(
            SelectionEvaluationError::UnknownNamedSelection(name.clone()),
        ),
        SelectionExpr::Not(inner) => validate_named(inner, named),
        SelectionExpr::And(left, right)
        | SelectionExpr::Xor(left, right)
        | SelectionExpr::Or(left, right) => {
            validate_named(left, named)?;
            validate_named(right, named)
        }
        _ => Ok(()),
    }
}

fn matches_atom(
    expression: &SelectionExpr,
    atom: &Atom,
    atom_index: usize,
    named: &BTreeMap<String, Selection>,
) -> bool {
    match expression {
        SelectionExpr::All => true,
        SelectionExpr::None => false,
        SelectionExpr::Element(element) => atom.element == *element,
        SelectionExpr::AtomName(name) => atom.name.eq_ignore_ascii_case(name),
        SelectionExpr::AtomNamePattern(pattern) => glob_matches(pattern, &atom.name),
        SelectionExpr::ResidueName(name) => atom.residue_name.eq_ignore_ascii_case(name),
        SelectionExpr::ResidueNamePattern(pattern) => glob_matches(pattern, &atom.residue_name),
        SelectionExpr::ResidueNumber(number) => atom.residue_number == *number,
        SelectionExpr::ResidueRange(start, end) => (*start..=*end).contains(&atom.residue_number),
        SelectionExpr::Chain(chain) => atom.chain_id.eq_ignore_ascii_case(chain),
        SelectionExpr::ChainPattern(pattern) => glob_matches(pattern, &atom.chain_id),
        SelectionExpr::Serial(serial) => atom.serial == *serial,
        SelectionExpr::Named(name) => find_named(named, name)
            .and_then(|selection| selection.flags.get(atom_index))
            .unwrap_or(false),
        SelectionExpr::Hetatm => atom.hetero,
        SelectionExpr::Polymer => !atom.hetero,
        SelectionExpr::Not(inner) => !matches_atom(inner, atom, atom_index, named),
        SelectionExpr::And(left, right) => {
            matches_atom(left, atom, atom_index, named)
                && matches_atom(right, atom, atom_index, named)
        }
        SelectionExpr::Xor(left, right) => {
            matches_atom(left, atom, atom_index, named)
                != matches_atom(right, atom, atom_index, named)
        }
        SelectionExpr::Or(left, right) => {
            matches_atom(left, atom, atom_index, named)
                || matches_atom(right, atom, atom_index, named)
        }
    }
}

fn glob_matches(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let (mut pattern_index, mut value_index) = (0, 0);
    let (mut star_index, mut star_value_index) = (None, 0);
    while value_index < value.len() {
        if pattern_index < pattern.len()
            && (pattern[pattern_index] == b'?'
                || pattern[pattern_index].eq_ignore_ascii_case(&value[value_index]))
        {
            pattern_index += 1;
            value_index += 1;
        } else if pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
            star_index = Some(pattern_index);
            pattern_index += 1;
            star_value_index = value_index;
        } else if let Some(star) = star_index {
            pattern_index = star + 1;
            star_value_index += 1;
            value_index = star_value_index;
        } else {
            return false;
        }
    }
    while pattern_index < pattern.len() && pattern[pattern_index] == b'*' {
        pattern_index += 1;
    }
    pattern_index == pattern.len()
}

fn find_named<'a>(named: &'a BTreeMap<String, Selection>, name: &str) -> Option<&'a Selection> {
    named.get(name).or_else(|| {
        named
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value)
    })
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::{
        molecule::{Atom, Element},
        selection::parse_selection,
    };

    fn molecule() -> Molecule {
        Molecule {
            atoms: vec![
                atom(1, "CA", Element::C, "ALA", 10, "A", false),
                atom(2, "N", Element::N, "ALA", 10, "A", false),
                atom(3, "O", Element::O, "HOH", 20, "B", true),
                atom(4, "H1", Element::H, "HOH", 20, "B", true),
            ],
            bonds: Vec::new(),
            info: Default::default(),
        }
    }

    fn atom(
        serial: u32,
        name: &str,
        element: crate::molecule::Element,
        residue_name: &str,
        residue_number: i32,
        chain_id: &str,
        hetero: bool,
    ) -> Atom {
        Atom {
            serial,
            name: name.into(),
            element,
            residue_name: residue_name.into(),
            residue_number,
            insertion_code: None,
            chain_id: chain_id.into(),
            position: Vec3::ZERO,
            occupancy: 1.0,
            b_factor: 0.0,
            hetero,
            alt_loc: None,
            formal_charge: 0,
        }
    }

    #[test]
    fn evaluates_exact_indices() {
        let molecule = molecule();
        let expression = parse_selection("chain A and (element C or element O)").unwrap();
        assert_eq!(
            evaluate(&expression, &molecule)
                .indices()
                .collect::<Vec<_>>(),
            vec![0]
        );

        let expression = parse_selection("hetatm and not element H").unwrap();
        assert_eq!(
            evaluate(&expression, &molecule)
                .indices()
                .collect::<Vec<_>>(),
            vec![2]
        );

        let expression = parse_selection("resi 10-20 and not element N").unwrap();
        assert_eq!(
            evaluate(&expression, &molecule)
                .indices()
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
    }

    #[test]
    fn evaluates_named_selections_and_reports_unknown_names() {
        let molecule = molecule();
        let mut named = BTreeMap::new();
        named.insert(
            "protein".into(),
            evaluate(&parse_selection("chain A").unwrap(), &molecule),
        );
        let expression = parse_selection("selection protein and element N").unwrap();
        assert_eq!(
            evaluate_with_named(&expression, &molecule, &named)
                .unwrap()
                .indices()
                .collect::<Vec<_>>(),
            vec![1]
        );
        let unknown = parse_selection("selection missing").unwrap();
        assert!(matches!(
            evaluate_with_named(&unknown, &molecule, &named),
            Err(SelectionEvaluationError::UnknownNamedSelection(name)) if name == "missing"
        ));
    }

    #[test]
    fn evaluates_chain_path_masks_and_wildcards() {
        let molecule = molecule();
        assert_eq!(
            evaluate(&parse_selection("Chain A/AL*").unwrap(), &molecule)
                .indices()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            evaluate(
                &parse_selection("Chain B/[20:22, 70:71]").unwrap(),
                &molecule
            )
            .indices()
            .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(
            evaluate(
                &parse_selection("../AL* AND [10:30, 45:50]").unwrap(),
                &molecule
            )
            .indices()
            .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            evaluate(
                &parse_selection("Chain A/AL* XOR Chain B/HOH*").unwrap(),
                &molecule
            )
            .indices()
            .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            evaluate(&parse_selection("../AL* XOR [10:10]").unwrap(), &molecule)
                .indices()
                .collect::<Vec<_>>(),
            Vec::<usize>::new()
        );
    }
}
