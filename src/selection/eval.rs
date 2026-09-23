use std::{
    cell::OnceCell,
    collections::{BTreeMap, HashMap, HashSet},
    ops::Range,
};

use glam::Vec3;
use thiserror::Error;

use crate::{
    bitset::AtomMask,
    molecule::{
        Atom, Molecule,
        classify::{ResidueClass, classify_atoms, is_backbone_atom, residue_ranges},
    },
};

use super::{SelectionExpr, milli_to_distance};

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
        flags: Evaluator::new(molecule, &named).mask(expression),
    }
}

pub fn evaluate_with_named(
    expression: &SelectionExpr,
    molecule: &Molecule,
    named: &BTreeMap<String, Selection>,
) -> Result<Selection, SelectionEvaluationError> {
    validate_named(expression, named)?;
    Ok(Selection {
        flags: Evaluator::new(molecule, named).mask(expression),
    })
}

fn validate_named(
    expression: &SelectionExpr,
    named: &BTreeMap<String, Selection>,
) -> Result<(), SelectionEvaluationError> {
    if let SelectionExpr::Named(name) = expression
        && find_named(named, name).is_none()
    {
        return Err(SelectionEvaluationError::UnknownNamedSelection(
            name.clone(),
        ));
    }
    expression
        .children()
        .into_iter()
        .try_for_each(|child| validate_named(child, named))
}

/// Evaluates a selection tree bottom-up into packed masks. Chemical classes and residue
/// ranges are computed at most once per evaluation.
struct Evaluator<'a> {
    molecule: &'a Molecule,
    named: &'a BTreeMap<String, Selection>,
    classes: OnceCell<Vec<ResidueClass>>,
    residues: OnceCell<Vec<Range<usize>>>,
}

impl<'a> Evaluator<'a> {
    fn new(molecule: &'a Molecule, named: &'a BTreeMap<String, Selection>) -> Self {
        Self {
            molecule,
            named,
            classes: OnceCell::new(),
            residues: OnceCell::new(),
        }
    }

    fn len(&self) -> usize {
        self.molecule.atoms.len()
    }

    fn classes(&self) -> &[ResidueClass] {
        self.classes.get_or_init(|| classify_atoms(self.molecule))
    }

    fn residues(&self) -> &[Range<usize>] {
        self.residues.get_or_init(|| residue_ranges(self.molecule))
    }

    fn atoms(&self, predicate: impl Fn(&Atom) -> bool) -> AtomMask {
        let atoms = &self.molecule.atoms;
        AtomMask::from_fn(atoms.len(), |index| predicate(&atoms[index]))
    }

    fn class_mask(&self, predicate: impl Fn(&Atom, ResidueClass) -> bool) -> AtomMask {
        let classes = self.classes();
        let atoms = &self.molecule.atoms;
        AtomMask::from_fn(atoms.len(), |index| {
            predicate(&atoms[index], classes[index])
        })
    }

    fn mask(&self, expression: &SelectionExpr) -> AtomMask {
        match expression {
            SelectionExpr::All => AtomMask::new(self.len(), true),
            SelectionExpr::None => AtomMask::new(self.len(), false),
            SelectionExpr::Element(element) => self.atoms(|atom| atom.element == *element),
            SelectionExpr::AtomName(name) => {
                self.atoms(|atom| atom.name.eq_ignore_ascii_case(name))
            }
            SelectionExpr::AtomNamePattern(pattern) => {
                self.atoms(|atom| glob_matches(pattern, &atom.name))
            }
            SelectionExpr::ResidueName(name) => {
                self.atoms(|atom| atom.residue_name.eq_ignore_ascii_case(name))
            }
            SelectionExpr::ResidueNamePattern(pattern) => {
                self.atoms(|atom| glob_matches(pattern, &atom.residue_name))
            }
            SelectionExpr::ResidueNumber(number) => {
                self.atoms(|atom| atom.residue_number == *number)
            }
            SelectionExpr::ResidueRange(start, end) => {
                self.atoms(|atom| (*start..=*end).contains(&atom.residue_number))
            }
            SelectionExpr::Chain(chain) => {
                self.atoms(|atom| atom.chain_id.eq_ignore_ascii_case(chain))
            }
            SelectionExpr::ChainPattern(pattern) => {
                self.atoms(|atom| glob_matches(pattern, &atom.chain_id))
            }
            SelectionExpr::Serial(serial) => self.atoms(|atom| atom.serial == *serial),
            SelectionExpr::Named(name) => match find_named(self.named, name) {
                Some(selection) if selection.flags.len() == self.len() => selection.flags.clone(),
                Some(selection) => AtomMask::from_fn(self.len(), |index| {
                    selection.flags.get(index).unwrap_or(false)
                }),
                None => AtomMask::new(self.len(), false),
            },
            SelectionExpr::Hetatm => self.atoms(|atom| atom.hetero),
            SelectionExpr::Polymer => self.atoms(|atom| !atom.hetero),
            SelectionExpr::Protein => self.class_mask(|_, class| class == ResidueClass::Protein),
            SelectionExpr::Nucleic => self.class_mask(|_, class| class == ResidueClass::Nucleic),
            SelectionExpr::Water => self.class_mask(|_, class| class == ResidueClass::Water),
            SelectionExpr::Ion => self.class_mask(|_, class| class == ResidueClass::Ion),
            SelectionExpr::Ligand => self.class_mask(|_, class| class == ResidueClass::Ligand),
            SelectionExpr::Backbone => self.class_mask(is_backbone_atom),
            SelectionExpr::Sidechain => self.class_mask(|atom, class| {
                class == ResidueClass::Protein && !is_backbone_atom(atom, class)
            }),
            SelectionExpr::Hydrogen => self.atoms(|atom| atom.element.is_hydrogen()),
            SelectionExpr::Within(distance, inner) => {
                self.within(milli_to_distance(*distance), &self.mask(inner))
            }
            SelectionExpr::Around(distance, inner) => {
                let seeds = self.mask(inner);
                let mut result = self.within(milli_to_distance(*distance), &seeds);
                let mut outside = seeds;
                outside.invert();
                result.intersect_with(&outside);
                result
            }
            SelectionExpr::ByResidue(inner) => self.expand_residues(&self.mask(inner)),
            SelectionExpr::ByChain(inner) => self.expand_chains(&self.mask(inner)),
            SelectionExpr::Not(inner) => {
                let mut mask = self.mask(inner);
                mask.invert();
                mask
            }
            SelectionExpr::And(left, right) => {
                let mut mask = self.mask(left);
                if mask.count() > 0 {
                    mask.intersect_with(&self.mask(right));
                }
                mask
            }
            SelectionExpr::Xor(left, right) => {
                let mut mask = self.mask(left);
                mask.symmetric_difference_with(&self.mask(right));
                mask
            }
            SelectionExpr::Or(left, right) => {
                let mut mask = self.mask(left);
                mask.union_with(&self.mask(right));
                mask
            }
        }
    }

    /// All atoms whose center lies within `distance` of a seed atom, using a uniform grid
    /// over the seeds so the cost is proportional to the number of atoms, not their product.
    fn within(&self, distance: f32, seeds: &AtomMask) -> AtomMask {
        let atoms = &self.molecule.atoms;
        let seed_count = seeds.count();
        if seed_count == 0 {
            return AtomMask::new(atoms.len(), false);
        }
        let cell = distance.max(0.5);
        let key = |position: Vec3| -> (i32, i32, i32) {
            let scaled = (position / cell).floor();
            (scaled.x as i32, scaled.y as i32, scaled.z as i32)
        };
        let mut grid = HashMap::<(i32, i32, i32), Vec<Vec3>>::with_capacity(seed_count);
        for index in seeds.indices() {
            let position = atoms[index].position;
            if position.is_finite() {
                grid.entry(key(position)).or_default().push(position);
            }
        }
        let squared = distance * distance;
        AtomMask::from_fn(atoms.len(), |index| {
            if seeds.contains(index) {
                return true;
            }
            let position = atoms[index].position;
            if !position.is_finite() {
                return false;
            }
            let (x, y, z) = key(position);
            (-1..=1).any(|dx| {
                (-1..=1).any(|dy| {
                    (-1..=1).any(|dz| {
                        grid.get(&(x + dx, y + dy, z + dz)).is_some_and(|points| {
                            points
                                .iter()
                                .any(|point| point.distance_squared(position) <= squared)
                        })
                    })
                })
            })
        })
    }

    fn expand_residues(&self, seeds: &AtomMask) -> AtomMask {
        let mut result = AtomMask::new(self.len(), false);
        for range in self.residues() {
            if range.clone().any(|index| seeds.contains(index)) {
                for index in range.clone() {
                    result.set(index, true);
                }
            }
        }
        result
    }

    fn expand_chains(&self, seeds: &AtomMask) -> AtomMask {
        let atoms = &self.molecule.atoms;
        let chains = seeds
            .indices()
            .map(|index| atoms[index].chain_id.as_str())
            .collect::<HashSet<_>>();
        AtomMask::from_fn(atoms.len(), |index| {
            chains.contains(atoms[index].chain_id.as_str())
        })
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

    fn placed(
        serial: u32,
        name: &str,
        element: Element,
        residue: (&str, i32),
        chain: &str,
        x: f32,
    ) -> Atom {
        Atom {
            position: Vec3::new(x, 0.0, 0.0),
            ..atom(serial, name, element, residue.0, residue.1, chain, false)
        }
    }

    fn complex() -> Molecule {
        Molecule::new(
            vec![
                placed(1, "N", Element::N, ("ALA", 1), "A", 0.0),
                placed(2, "CA", Element::C, ("ALA", 1), "A", 1.5),
                placed(3, "CB", Element::C, ("ALA", 1), "A", 2.5),
                placed(4, "HA", Element::H, ("ALA", 1), "A", 1.5),
                placed(5, "CA", Element::C, ("GLY", 2), "A", 10.0),
                placed(6, "C1", Element::C, ("LIG", 3), "B", 5.0),
                placed(7, "O", Element::O, ("HOH", 4), "W", 7.9),
                placed(8, "NA", Element::Na, ("NA", 5), "W", 30.0),
            ],
            Vec::new(),
        )
    }

    fn indices(source: &str, molecule: &Molecule) -> Vec<usize> {
        evaluate(&parse_selection(source).unwrap(), molecule)
            .indices()
            .collect()
    }

    #[test]
    fn evaluates_chemical_keywords() {
        let molecule = complex();
        assert_eq!(indices("protein", &molecule), vec![0, 1, 2, 3, 4]);
        assert_eq!(indices("backbone", &molecule), vec![0, 1, 3, 4]);
        assert_eq!(indices("sidechain", &molecule), vec![2]);
        assert_eq!(indices("ligand", &molecule), vec![5]);
        assert_eq!(indices("water", &molecule), vec![6]);
        assert_eq!(indices("ion", &molecule), vec![7]);
        assert_eq!(indices("hydrogen", &molecule), vec![3]);
        assert_eq!(indices("not protein and not water", &molecule), vec![5, 7]);
    }

    #[test]
    fn evaluates_distance_and_expansion_operators() {
        let molecule = complex();
        // The ligand at x = 5 reaches CB (2.5 Å) and the water (2.9 Å) within 3 Å.
        assert_eq!(indices("within 3 of ligand", &molecule), vec![2, 5, 6]);
        assert_eq!(indices("around 3 of ligand", &molecule), vec![2, 6]);
        assert_eq!(
            indices("around 2.4 of ligand", &molecule),
            Vec::<usize>::new()
        );
        assert_eq!(
            indices("byres around 3 of ligand", &molecule),
            vec![0, 1, 2, 3, 6]
        );
        assert_eq!(indices("bychain serial 5", &molecule), vec![0, 1, 2, 3, 4]);
        assert_eq!(
            indices("same residue as name CB", &molecule),
            vec![0, 1, 2, 3]
        );
        assert_eq!(indices("within 0 of ion", &molecule), vec![7]);
        assert_eq!(
            indices("within 100 of none", &molecule),
            Vec::<usize>::new()
        );
        assert_eq!(
            indices("protein and within 4 of ligand and not hydrogen", &molecule),
            vec![1, 2]
        );
    }

    #[test]
    fn within_matches_brute_force_on_random_points() {
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 20_000) as f32 / 1000.0
        };
        let atoms = (0..600)
            .map(|index| Atom {
                position: Vec3::new(next(), next(), next()),
                ..atom(
                    index + 1,
                    "C",
                    Element::C,
                    "LIG",
                    (index / 10) as i32,
                    "A",
                    true,
                )
            })
            .collect::<Vec<_>>();
        let molecule = Molecule::new(atoms, Vec::new());
        for distance in [0.7_f32, 2.0, 3.3, 9.0] {
            let expression = parse_selection(&format!("within {distance} of resi 0-4")).unwrap();
            let fast = evaluate(&expression, &molecule);
            for (index, atom) in molecule.atoms.iter().enumerate() {
                let expected = molecule.atoms[..50]
                    .iter()
                    .any(|seed| seed.position.distance(atom.position) <= distance);
                assert_eq!(
                    fast.flags().contains(index),
                    expected,
                    "atom {index} at {distance}"
                );
            }
        }
    }
}
