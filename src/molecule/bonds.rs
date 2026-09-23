//! Covalent connectivity from chemical component templates, polymer linkage, explicit file
//! records and, for atoms without a template, interatomic distances.

use std::collections::{HashMap, HashSet};

use super::{
    Atom, Bond, BondKind, BondOrder, Element,
    ccd::{ComponentCache, ComponentKind, canonical_atom_name, canonical_component},
};

const CELL_SIZE: f32 = 3.0;
const TOLERANCE: f32 = 0.45;
const MIN_DISTANCE: f32 = 0.35;
const MAX_DISTANCE: f32 = 3.0;
/// Template bonds are accepted up to this far beyond the covalent-radius sum, which tolerates
/// poorly refined geometry without joining atoms that are clearly apart.
const TEMPLATE_SLACK: f32 = 0.75;
const PEPTIDE_LINK: f32 = 1.75;
const PHOSPHODIESTER_LINK: f32 = 1.9;
const DISULFIDE: f32 = 2.3;

type Cell = (i32, i32, i32);

/// A bond stated by the file (CONECT, LINK, SSBOND or `struct_conn`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExplicitBond {
    pub a: usize,
    pub b: usize,
    pub kind: BondKind,
    pub order: Option<BondOrder>,
}

/// Builds the bond list of a structure.
///
/// 1. Residues whose component is in the chemical component dictionary receive exactly the
///    template bonds, including bond orders, between atoms present in the file.
/// 2. Consecutive amino acids and nucleotides are linked through C–N and O3'–P.
/// 3. Explicit bonds from the file are added (metal coordination only comes from here).
/// 4. Atoms without a template, and hydrogens, are connected by covalent radii.
pub fn build_bonds(atoms: &[Atom], explicit: &[ExplicitBond]) -> Vec<Bond> {
    let residues = residue_groups(atoms);
    let mut cache = ComponentCache::default();
    let mut bonds: HashMap<(usize, usize), Bond> = HashMap::new();
    let mut covered = vec![false; atoms.len()];
    let mut polymer = vec![ResidueClass::Other; residues.len()];
    let mut residue_of = vec![usize::MAX; atoms.len()];

    for (residue_index, residue) in residues.iter().enumerate() {
        for &atom in residue {
            residue_of[atom] = residue_index;
        }
        let name = canonical_component(&atoms[residue[0]].residue_name);
        let component = cache.get(name);
        polymer[residue_index] = classify(atoms, residue, component.as_ref().map(|c| c.kind));
        let Some(component) = component else {
            continue;
        };
        let mut by_name = HashMap::<std::borrow::Cow<'_, str>, usize>::new();
        for &atom in residue {
            by_name
                .entry(canonical_atom_name(name, &atoms[atom].name))
                .or_insert(atom);
        }
        let mut template_atoms = Vec::with_capacity(component.atoms.len());
        for template in &component.atoms {
            let atom = by_name.get(template.name.as_str()).copied();
            if let Some(atom) = atom {
                covered[atom] = true;
            }
            template_atoms.push(atom);
        }
        for template in &component.bonds {
            let (Some(a), Some(b)) = (template_atoms[template.a], template_atoms[template.b])
            else {
                continue;
            };
            if within_covalent_reach(&atoms[a], &atoms[b], TEMPLATE_SLACK)
                && let Some(bond) = Bond::with_order(a, b, template.order, BondKind::Covalent)
            {
                bonds.insert((bond.a, bond.b), bond);
            }
        }
    }

    link_polymers(atoms, &residues, &polymer, &mut bonds);

    for bond in explicit {
        if bond.a >= atoms.len() || bond.b >= atoms.len() {
            continue;
        }
        let metal = atoms[bond.a].element.is_metal() || atoms[bond.b].element.is_metal();
        let kind = if metal {
            BondKind::MetalCoordination
        } else if bond.kind == BondKind::Covalent
            && atoms[bond.a].element == Element::S
            && atoms[bond.b].element == Element::S
        {
            BondKind::Disulfide
        } else {
            bond.kind
        };
        let existing = bonds.get(&ordered(bond.a, bond.b)).map(|bond| bond.order);
        let order = bond.order.or(existing).unwrap_or_default();
        if let Some(bond) = Bond::with_order(bond.a, bond.b, order, kind) {
            bonds.insert((bond.a, bond.b), bond);
        }
    }

    infer_remaining(atoms, &covered, &residue_of, &polymer, &mut bonds);

    let mut result: Vec<Bond> = bonds.into_values().collect();
    result.sort_by_key(|bond| (bond.a, bond.b));
    result
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidueClass {
    AminoAcid,
    Nucleotide,
    Other,
}

fn classify(atoms: &[Atom], residue: &[usize], kind: Option<ComponentKind>) -> ResidueClass {
    let has = |name: &str| residue.iter().any(|&index| atoms[index].name == name);
    match kind {
        Some(kind) if kind.is_amino_acid() => ResidueClass::AminoAcid,
        Some(kind) if kind.is_nucleotide() => ResidueClass::Nucleotide,
        _ if has("N") && has("CA") && has("C") => ResidueClass::AminoAcid,
        _ if has("P") && has("O3'") && has("C4'") => ResidueClass::Nucleotide,
        _ => ResidueClass::Other,
    }
}

/// Residues in first-appearance order; atoms of one residue need not be contiguous.
fn residue_groups(atoms: &[Atom]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut lookup = HashMap::<(&str, i32, Option<char>, &str), usize>::new();
    for (index, atom) in atoms.iter().enumerate() {
        let key = (
            atom.chain_id.as_str(),
            atom.residue_number,
            atom.insertion_code,
            atom.residue_name.as_str(),
        );
        let group = *lookup.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(index);
    }
    groups
}

fn link_polymers(
    atoms: &[Atom],
    residues: &[Vec<usize>],
    classes: &[ResidueClass],
    bonds: &mut HashMap<(usize, usize), Bond>,
) {
    let find = |residue: &[usize], name: &str| {
        residue
            .iter()
            .copied()
            .find(|&index| atoms[index].name == name)
    };
    for index in 1..residues.len() {
        let (previous, current) = (&residues[index - 1], &residues[index]);
        if atoms[previous[0]].chain_id != atoms[current[0]].chain_id
            || classes[index - 1] != classes[index]
        {
            continue;
        }
        let link = match classes[index] {
            ResidueClass::AminoAcid => find(previous, "C")
                .zip(find(current, "N"))
                .map(|pair| (pair, PEPTIDE_LINK)),
            ResidueClass::Nucleotide => find(previous, "O3'")
                .or_else(|| find(previous, "O3*"))
                .zip(find(current, "P"))
                .map(|pair| (pair, PHOSPHODIESTER_LINK)),
            ResidueClass::Other => None,
        };
        if let Some(((a, b), limit)) = link
            && atoms[a].position.distance(atoms[b].position) <= limit
            && let Some(bond) = Bond::new(a, b)
        {
            bonds.insert((bond.a, bond.b), bond);
        }
    }
}

fn within_covalent_reach(a: &Atom, b: &Atom, slack: f32) -> bool {
    let distance = a.position.distance(b.position);
    distance >= MIN_DISTANCE
        && distance <= a.element.covalent_radius() + b.element.covalent_radius() + slack
}

fn infer_remaining(
    atoms: &[Atom],
    covered: &[bool],
    residue_of: &[usize],
    classes: &[ResidueClass],
    bonds: &mut HashMap<(usize, usize), Bond>,
) {
    let mut cells: HashMap<Cell, Vec<usize>> = HashMap::new();
    for (index, atom) in atoms.iter().enumerate() {
        cells.entry(cell_for(atom)).or_default().push(index);
    }
    for (index, atom) in atoms.iter().enumerate() {
        // Metals bond only through explicit records; atoms of unknown element are usually
        // virtual sites (TIP4P M, lone pairs) and never bond.
        if atom.element.is_metal() || atom.element == Element::Unknown {
            continue;
        }
        let cell = cell_for(atom);
        let hydrogen = atom.element.is_hydrogen();
        // Hydrogens bond to their nearest heavy atom; only an isolated H2 pairs two hydrogens.
        let mut nearest_heavy: Option<(f32, usize)> = None;
        let mut nearest_hydrogen: Option<(f32, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(candidates) = cells.get(&(cell.0 + dx, cell.1 + dy, cell.2 + dz))
                    else {
                        continue;
                    };
                    for &other in candidates {
                        if other == index {
                            continue;
                        }
                        let partner = &atoms[other];
                        if partner.element.is_metal() || partner.element == Element::Unknown {
                            continue;
                        }
                        let distance = atom.position.distance(partner.position);
                        let limit = (atom.element.covalent_radius()
                            + partner.element.covalent_radius()
                            + TOLERANCE)
                            .min(MAX_DISTANCE);
                        if !(MIN_DISTANCE..=limit).contains(&distance) {
                            continue;
                        }
                        if hydrogen {
                            let slot = if partner.element.is_hydrogen() {
                                &mut nearest_hydrogen
                            } else {
                                &mut nearest_heavy
                            };
                            if slot.is_none_or(|(best, _)| distance < best) {
                                *slot = Some((distance, other));
                            }
                            continue;
                        }
                        if partner.element.is_hydrogen() || other < index {
                            continue;
                        }
                        let same_residue = residue_of[index] == residue_of[other];
                        let accept = if !covered[index] || !covered[other] {
                            true
                        } else if same_residue {
                            // Both atoms are described by the template, which is authoritative.
                            false
                        } else {
                            let disulfide = atom.element == Element::S
                                && partner.element == Element::S
                                && distance <= DISULFIDE;
                            let ligand = classes[residue_of[index]] == ResidueClass::Other
                                || classes[residue_of[other]] == ResidueClass::Other;
                            disulfide
                                || (ligand
                                    && distance
                                        <= atom.element.covalent_radius()
                                            + partner.element.covalent_radius()
                                            + 0.2)
                        };
                        if accept && let Some(bond) = Bond::new(index, other) {
                            let kind = if atom.element == Element::S
                                && partner.element == Element::S
                                && !same_residue
                            {
                                BondKind::Disulfide
                            } else {
                                BondKind::Covalent
                            };
                            bonds
                                .entry((bond.a, bond.b))
                                .or_insert(Bond { kind, ..bond });
                        }
                    }
                }
            }
        }
        if let Some((_, partner)) = nearest_heavy.or(nearest_hydrogen)
            && let Some(bond) = Bond::new(index, partner)
        {
            bonds.entry((bond.a, bond.b)).or_insert(bond);
        }
    }
}

/// Distance-only connectivity, kept for callers that construct geometry without residue
/// templates. Existing bonds are preserved.
pub fn infer_bonds(atoms: &[Atom], existing: &[Bond]) -> Vec<Bond> {
    let mut cells: HashMap<Cell, Vec<usize>> = HashMap::new();
    let mut result = existing.to_vec();
    let mut seen: HashSet<(usize, usize)> = existing.iter().map(|bond| (bond.a, bond.b)).collect();

    for (index, atom) in atoms.iter().enumerate() {
        let cell = cell_for(atom);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                    if let Some(candidates) = cells.get(&neighbor) {
                        for &other_index in candidates {
                            let other = &atoms[other_index];
                            let distance = atom.position.distance(other.position);
                            let limit = (atom.element.covalent_radius()
                                + other.element.covalent_radius()
                                + TOLERANCE)
                                .min(MAX_DISTANCE);
                            if (MIN_DISTANCE..=limit).contains(&distance)
                                && seen.insert((other_index, index))
                                && let Some(bond) = Bond::new(other_index, index)
                            {
                                result.push(bond);
                            }
                        }
                    }
                }
            }
        }
        cells.entry(cell).or_default().push(index);
    }

    result.sort_by_key(|bond| (bond.a, bond.b));
    result
}

fn cell_for(atom: &Atom) -> Cell {
    let position = atom.position / CELL_SIZE;
    (
        position.x.floor() as i32,
        position.y.floor() as i32,
        position.z.floor() as i32,
    )
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::Element;

    fn atom(name: &str, element: Element, residue: &str, number: i32, position: Vec3) -> Atom {
        Atom {
            serial: 1,
            name: name.into(),
            element,
            residue_name: residue.into(),
            residue_number: number,
            chain_id: "A".into(),
            position,
            occupancy: 1.0,
            ..Atom::default()
        }
    }

    fn has(bonds: &[Bond], a: usize, b: usize) -> Option<Bond> {
        let (a, b) = ordered(a, b);
        bonds
            .iter()
            .copied()
            .find(|bond| bond.a == a && bond.b == b)
    }

    #[test]
    fn infers_near_but_not_distant_bonds() {
        let atoms = [
            atom("C1", Element::C, "LIG", 1, Vec3::ZERO),
            atom("C2", Element::C, "LIG", 1, Vec3::new(1.45, 0.0, 0.0)),
            atom("C3", Element::C, "LIG", 1, Vec3::new(8.0, 0.0, 0.0)),
        ];
        assert_eq!(infer_bonds(&atoms, &[]), vec![Bond::new(0, 1).unwrap()]);
        let bonds = build_bonds(&atoms, &[]);
        assert_eq!(bonds, vec![Bond::new(0, 1).unwrap()]);
    }

    #[test]
    fn keeps_explicit_bonds_without_duplicates() {
        let atoms = [
            atom("C1", Element::C, "LIG", 1, Vec3::ZERO),
            atom("C2", Element::C, "LIG", 1, Vec3::new(1.4, 0.0, 0.0)),
        ];
        let explicit = [Bond::new(0, 1).unwrap()];
        assert_eq!(infer_bonds(&atoms, &explicit), explicit);
    }

    /// Glycine with ideal geometry followed by the N of the next residue.
    fn glycine_pair() -> Vec<Atom> {
        vec![
            atom("N", Element::N, "GLY", 1, Vec3::new(0.0, 0.0, 0.0)),
            atom("CA", Element::C, "GLY", 1, Vec3::new(1.46, 0.0, 0.0)),
            atom("C", Element::C, "GLY", 1, Vec3::new(2.0, 1.42, 0.0)),
            atom("O", Element::O, "GLY", 1, Vec3::new(1.25, 2.39, 0.0)),
            atom("N", Element::N, "GLY", 2, Vec3::new(3.33, 1.55, 0.0)),
            atom("CA", Element::C, "GLY", 2, Vec3::new(3.95, 2.87, 0.0)),
            atom("H", Element::H, "GLY", 2, Vec3::new(3.9, 0.7, 0.0)),
        ]
    }

    #[test]
    fn templates_supply_bond_orders_and_polymer_links() {
        let atoms = glycine_pair();
        let bonds = build_bonds(&atoms, &[]);
        assert_eq!(has(&bonds, 2, 3).unwrap().order, BondOrder::Double);
        assert_eq!(has(&bonds, 0, 1).unwrap().order, BondOrder::Single);
        assert!(has(&bonds, 2, 4).is_some(), "peptide bond");
        assert!(
            has(&bonds, 6, 4).is_some(),
            "hydrogen joins its nearest atom"
        );
        assert_eq!(bonds.iter().filter(|b| b.a == 6 || b.b == 6).count(), 1);
        assert!(has(&bonds, 0, 2).is_none());
    }

    #[test]
    fn metals_bond_only_through_explicit_records() {
        let atoms = [
            atom("ZN", Element::Zn, "ZN", 1, Vec3::ZERO),
            atom("O1", Element::O, "LIG", 2, Vec3::new(2.0, 0.0, 0.0)),
        ];
        assert!(build_bonds(&atoms, &[]).is_empty());
        let explicit = [ExplicitBond {
            a: 0,
            b: 1,
            kind: BondKind::Covalent,
            order: None,
        }];
        let bonds = build_bonds(&atoms, &explicit);
        assert_eq!(bonds[0].kind, BondKind::MetalCoordination);
    }

    #[test]
    fn detects_disulfides_between_cysteines() {
        let atoms = [
            atom("SG", Element::S, "CYS", 1, Vec3::ZERO),
            atom("SG", Element::S, "CYS", 2, Vec3::new(2.04, 0.0, 0.0)),
        ];
        let bonds = build_bonds(&atoms, &[]);
        assert_eq!(bonds.len(), 1);
        assert_eq!(bonds[0].kind, BondKind::Disulfide);
    }
}
