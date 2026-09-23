use std::collections::HashMap;

use super::Molecule;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidueId {
    pub name: String,
    pub number: i32,
    pub insertion_code: Option<char>,
}

impl ResidueId {
    pub fn label(&self) -> String {
        match self.insertion_code {
            Some(code) => format!("{} {}{}", self.name, self.number, code),
            None => format!("{} {}", self.name, self.number),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResidueGroup {
    pub id: ResidueId,
    pub atom_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ChainGroup {
    pub id: String,
    pub residues: Vec<ResidueGroup>,
    pub atom_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MoleculeHierarchy {
    pub chains: Vec<ChainGroup>,
    atom_paths: Vec<Option<AtomHierarchyPath>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomHierarchyPath {
    pub chain_index: usize,
    pub residue_index: usize,
}

impl MoleculeHierarchy {
    pub fn from_molecule(molecule: &Molecule) -> Self {
        let mut chains = Vec::<ChainGroup>::new();
        let mut chain_indices = HashMap::<String, usize>::new();
        let mut residue_indices = Vec::<HashMap<ResidueId, usize>>::new();

        let mut atom_paths = vec![None; molecule.atoms.len()];
        for (atom_index, atom) in molecule.atoms.iter().enumerate() {
            let chain_index = if let Some(&index) = chain_indices.get(&atom.chain_id) {
                index
            } else {
                let index = chains.len();
                chain_indices.insert(atom.chain_id.clone(), index);
                chains.push(ChainGroup {
                    id: atom.chain_id.clone(),
                    residues: Vec::new(),
                    atom_count: 0,
                });
                residue_indices.push(HashMap::new());
                index
            };
            let residue_id = ResidueId {
                name: atom.residue_name.clone(),
                number: atom.residue_number,
                insertion_code: atom.insertion_code,
            };
            let residue_index = if let Some(&index) = residue_indices[chain_index].get(&residue_id)
            {
                index
            } else {
                let index = chains[chain_index].residues.len();
                residue_indices[chain_index].insert(residue_id.clone(), index);
                chains[chain_index].residues.push(ResidueGroup {
                    id: residue_id,
                    atom_indices: Vec::new(),
                });
                index
            };
            chains[chain_index].residues[residue_index]
                .atom_indices
                .push(atom_index);
            chains[chain_index].atom_count += 1;
            atom_paths[atom_index] = Some(AtomHierarchyPath {
                chain_index,
                residue_index,
            });
        }

        Self { chains, atom_paths }
    }

    pub fn residue(&self, chain_index: usize, residue_index: usize) -> Option<&ResidueGroup> {
        self.chains.get(chain_index)?.residues.get(residue_index)
    }

    pub fn atom_path(&self, atom_index: usize) -> Option<AtomHierarchyPath> {
        self.atom_paths.get(atom_index).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element};

    fn atom(chain: &str, residue: &str, number: i32, insertion_code: Option<char>) -> Atom {
        Atom {
            serial: 1,
            name: "CA".into(),
            element: Element::C,
            residue_name: residue.into(),
            residue_number: number,
            insertion_code,
            chain_id: chain.into(),
            position: Vec3::ZERO,
            occupancy: 1.0,
            b_factor: 0.0,
            hetero: false,
            alt_loc: None,
            formal_charge: 0,
        }
    }

    #[test]
    fn groups_atoms_by_chain_residue_and_insertion_code() {
        let molecule = Molecule {
            atoms: vec![
                atom("A", "GLY", 1, None),
                atom("A", "GLY", 1, None),
                atom("A", "SER", 2, Some('A')),
                atom("B", "HOH", 1, None),
            ],
            bonds: Vec::new(),
            info: Default::default(),
        };
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        assert_eq!(hierarchy.chains.len(), 2);
        assert_eq!(hierarchy.chains[0].atom_count, 3);
        assert_eq!(hierarchy.chains[0].residues.len(), 2);
        assert_eq!(hierarchy.chains[0].residues[0].atom_indices, vec![0, 1]);
        assert_eq!(hierarchy.chains[0].residues[1].id.insertion_code, Some('A'));
    }
}
