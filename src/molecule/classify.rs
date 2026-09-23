//! Chemical classification of residues and atoms used by selections and representations.

use std::collections::HashMap;

use super::{
    Atom, Element, Molecule,
    ccd::{ComponentCache, ComponentKind, canonical_atom_name, canonical_component},
};

/// Broad chemical class of the residue an atom belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidueClass {
    Protein,
    Nucleic,
    Water,
    Ion,
    /// Everything else: cofactors, drugs, saccharides, buffer molecules, lipids.
    Ligand,
}

/// Classifies residues once per residue name; chemically identical names share a result.
pub struct ResidueClassifier {
    components: ComponentCache,
    by_name: HashMap<(String, usize), ResidueClass>,
}

impl Default for ResidueClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidueClassifier {
    pub fn new() -> Self {
        Self {
            components: ComponentCache::default(),
            by_name: HashMap::new(),
        }
    }

    /// `residue_atoms` is the atoms of the residue; it disambiguates single-atom ions whose
    /// names are not in the dictionary (for example force-field specific ion names).
    pub fn classify(&mut self, residue_name: &str, residue_atoms: &[&Atom]) -> ResidueClass {
        let heavy_atoms = residue_atoms
            .iter()
            .filter(|atom| !atom.element.is_hydrogen())
            .count();
        let key = (residue_name.to_ascii_uppercase(), heavy_atoms.min(2));
        if let Some(&class) = self.by_name.get(&key) {
            return class;
        }
        let class = self.classify_uncached(&key.0, residue_atoms, heavy_atoms);
        self.by_name.insert(key, class);
        class
    }

    fn classify_uncached(
        &mut self,
        residue_name: &str,
        residue_atoms: &[&Atom],
        heavy_atoms: usize,
    ) -> ResidueClass {
        let canonical = canonical_component(residue_name);
        if matches!(canonical, "HOH" | "DOD" | "H2O" | "OH2") {
            return ResidueClass::Water;
        }
        if let Some(component) = self.components.get(canonical) {
            match component.kind {
                ComponentKind::Peptide => return ResidueClass::Protein,
                ComponentKind::Dna | ComponentKind::Rna => return ResidueClass::Nucleic,
                _ => {}
            }
            let heavy = component
                .atoms
                .iter()
                .filter(|atom| !atom.element.is_hydrogen())
                .collect::<Vec<_>>();
            if let [atom] = heavy.as_slice()
                && is_ion_element(atom.element)
            {
                return ResidueClass::Ion;
            }
            return ResidueClass::Ligand;
        }
        if heavy_atoms == 1
            && residue_atoms
                .iter()
                .any(|atom| !atom.element.is_hydrogen() && is_ion_element(atom.element))
        {
            return ResidueClass::Ion;
        }
        ResidueClass::Ligand
    }
}

fn is_ion_element(element: Element) -> bool {
    element.is_metal() || matches!(element, Element::F | Element::Cl | Element::Br | Element::I)
}

/// Per-atom residue classes, computed residue by residue.
pub fn classify_atoms(molecule: &Molecule) -> Vec<ResidueClass> {
    let mut classifier = ResidueClassifier::new();
    let mut classes = vec![ResidueClass::Ligand; molecule.atoms.len()];
    for range in residue_ranges(molecule) {
        let atoms = molecule.atoms[range.clone()].iter().collect::<Vec<_>>();
        let class = classifier.classify(&atoms[0].residue_name, &atoms);
        classes[range].fill(class);
    }
    classes
}

/// Consecutive atom ranges that share chain, residue number, insertion code and name.
pub fn residue_ranges(molecule: &Molecule) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for index in 1..=molecule.atoms.len() {
        let boundary = index == molecule.atoms.len() || {
            let (previous, current) = (&molecule.atoms[index - 1], &molecule.atoms[index]);
            previous.chain_id != current.chain_id
                || previous.residue_number != current.residue_number
                || previous.insertion_code != current.insertion_code
                || previous.residue_name != current.residue_name
        };
        if boundary && index > start {
            ranges.push(start..index);
            start = index;
        }
    }
    ranges
}

/// Main-chain atoms of amino acids (including their hydrogens) and the sugar-phosphate
/// backbone of nucleotides.
pub fn is_backbone_atom(atom: &Atom, class: ResidueClass) -> bool {
    let residue = canonical_component(&atom.residue_name);
    let name = canonical_atom_name(residue, &atom.name);
    match class {
        ResidueClass::Protein => matches!(
            name.as_ref(),
            "N" | "CA"
                | "C"
                | "O"
                | "OXT"
                | "H"
                | "HN"
                | "H1"
                | "H2"
                | "H3"
                | "HT1"
                | "HT2"
                | "HT3"
                | "HA"
                | "HA2"
                | "HA3"
                | "HXT"
        ),
        ResidueClass::Nucleic => {
            let name = name.replace('*', "'");
            matches!(
                name.as_str(),
                "P" | "OP1"
                    | "OP2"
                    | "OP3"
                    | "O1P"
                    | "O2P"
                    | "O3P"
                    | "O5'"
                    | "C5'"
                    | "C4'"
                    | "O4'"
                    | "C3'"
                    | "O3'"
                    | "C2'"
                    | "O2'"
                    | "C1'"
                    | "H5'"
                    | "H5''"
                    | "H4'"
                    | "H3'"
                    | "H2'"
                    | "H2''"
                    | "HO2'"
                    | "H1'"
                    | "HO3'"
                    | "HO5'"
                    | "HOP2"
                    | "HOP3"
            )
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn atom(name: &str, element: Element, residue: &str, number: i32) -> Atom {
        Atom {
            name: name.into(),
            element,
            residue_name: residue.into(),
            residue_number: number,
            chain_id: "A".into(),
            position: Vec3::ZERO,
            ..Default::default()
        }
    }

    #[test]
    fn classifies_common_residue_families() {
        let molecule = Molecule::new(
            vec![
                atom("CA", Element::C, "ALA", 1),
                atom("CA", Element::C, "MSE", 2),
                atom("P", Element::P, "DA", 3),
                atom("OW", Element::O, "SOL", 4),
                atom("NA", Element::Na, "SOD", 5),
                atom("ZN", Element::Zn, "ZN", 6),
                atom("FE", Element::Fe, "HEM", 7),
                atom("C1", Element::C, "XYZ", 8),
                atom("CL", Element::Cl, "CL", 9),
            ],
            Vec::new(),
        );
        assert_eq!(
            classify_atoms(&molecule),
            vec![
                ResidueClass::Protein,
                ResidueClass::Protein,
                ResidueClass::Nucleic,
                ResidueClass::Water,
                ResidueClass::Ion,
                ResidueClass::Ion,
                ResidueClass::Ligand,
                ResidueClass::Ligand,
                ResidueClass::Ion,
            ]
        );
    }

    #[test]
    fn backbone_covers_main_chain_and_sugar_phosphate() {
        assert!(is_backbone_atom(
            &atom("CA", Element::C, "ALA", 1),
            ResidueClass::Protein
        ));
        assert!(is_backbone_atom(
            &atom("HN", Element::H, "ALA", 1),
            ResidueClass::Protein
        ));
        assert!(!is_backbone_atom(
            &atom("CB", Element::C, "ALA", 1),
            ResidueClass::Protein
        ));
        assert!(is_backbone_atom(
            &atom("C4*", Element::C, "DA", 1),
            ResidueClass::Nucleic
        ));
        assert!(!is_backbone_atom(
            &atom("N9", Element::N, "DA", 1),
            ResidueClass::Nucleic
        ));
    }
}
