//! Biological assemblies and crystal packing built from structure annotations.

use std::{
    collections::{BTreeMap, HashMap},
    sync::OnceLock,
};

use glam::{DMat3, DVec3, Vec3};
use thiserror::Error;

use super::{Atom, Bond, CrystalInfo, Molecule, StructureInfo, SymmetryOperator, UnitCell};

/// Hard limit for generated copies; beyond it rendering and selection become impractical.
pub const MAX_GENERATED_ATOMS: usize = 20_000_000;

#[derive(Debug, Error, PartialEq)]
pub enum SymmetryError {
    #[error("assembly {0} is not defined in this structure")]
    UnknownAssembly(String),
    #[error("assembly refers to undefined operator '{0}'")]
    UnknownOperator(String),
    #[error("the structure has no crystallographic unit cell")]
    NoCell,
    #[error("space group '{0}' is not recognized")]
    UnknownSpaceGroup(String),
    #[error("the generated structure would contain {0} atoms, above the supported limit")]
    TooLarge(usize),
    #[error("the generated structure contains no atoms")]
    Empty,
}

/// One copy of a set of chains under an operator.
struct ChainCopy<'a> {
    chains: &'a [String],
    operator: SymmetryOperator,
}

/// Builds the biological assembly with the given id as a new molecule. Chains created by
/// the identity operator keep their identifiers; other copies are suffixed with the
/// operator name (`A-2`).
pub fn build_assembly(molecule: &Molecule, assembly_id: &str) -> Result<Molecule, SymmetryError> {
    let assembly = molecule
        .info
        .assemblies
        .iter()
        .find(|assembly| assembly.id == assembly_id)
        .ok_or_else(|| SymmetryError::UnknownAssembly(assembly_id.into()))?;
    let mut copies = Vec::new();
    for generator in &assembly.generators {
        for product in &generator.products {
            let mut operator: Option<SymmetryOperator> = None;
            for &index in product {
                let next = molecule
                    .info
                    .operators
                    .get(index)
                    .ok_or_else(|| SymmetryError::UnknownOperator(index.to_string()))?;
                operator = Some(match operator {
                    None => next.clone(),
                    Some(current) => current.compose(next),
                });
            }
            let mut operator = operator.unwrap_or_else(|| SymmetryOperator::identity("1"));
            operator.id = product
                .iter()
                .map(|&index| molecule.info.operators[index].id.clone())
                .collect::<Vec<_>>()
                .join("x");
            copies.push(ChainCopy {
                chains: &generator.chains,
                operator,
            });
        }
    }
    let mut result = replicate(molecule, &copies)?;
    result.info.title = molecule.info.title.clone();
    result.info.id = molecule.info.id.clone();
    result.info.notes = vec![format!("{} of {}", assembly.label(), source_name(molecule))];
    Ok(result)
}

/// Builds crystal packing: every space-group copy whose atoms come within `radius` Å of the
/// asymmetric unit, including translations to neighboring cells.
pub fn build_symmetry_mates(molecule: &Molecule, radius: f64) -> Result<Molecule, SymmetryError> {
    let crystal = molecule
        .info
        .crystal
        .as_ref()
        .ok_or(SymmetryError::NoCell)?;
    let operators = crystal_operators(crystal)?;
    let chains = chain_ids(molecule);
    let grid = AtomGrid::new(molecule, radius.max(1.0));
    let asu_bounds = molecule.bounds().map(|(minimum, maximum)| {
        (
            minimum.as_dvec3() - DVec3::splat(radius),
            maximum.as_dvec3() + DVec3::splat(radius),
        )
    });
    let chain_atoms = atoms_by_chain(molecule);
    let mut copies = Vec::new();
    let identity_index = operators.iter().position(SymmetryOperator::is_identity);
    for (operator_index, operator) in operators.iter().enumerate() {
        for tx in -1..=1 {
            for ty in -1..=1 {
                for tz in -1..=1 {
                    let is_identity =
                        Some(operator_index) == identity_index && tx == 0 && ty == 0 && tz == 0;
                    let shift = DVec3::new(tx as f64, ty as f64, tz as f64);
                    let mut shifted = operator.clone();
                    shifted.translation = (operator.translation()
                        + crystal.cell.orthogonalization() * shift)
                        .to_array();
                    shifted.id = format!("{}_{}{}{}", operator_index + 1, 5 + tx, 5 + ty, 5 + tz);
                    shifted.name = shifted.id.clone();
                    for chain in &chains {
                        let atoms = &chain_atoms[chain];
                        let near = is_identity
                            || copy_is_near(molecule, atoms, &shifted, &grid, asu_bounds, radius);
                        if near {
                            copies.push((chain.clone(), shifted.clone(), is_identity));
                        }
                    }
                }
            }
        }
    }
    // The asymmetric unit comes first so its chains keep their original position.
    copies.sort_by_key(|(_, _, identity)| !identity);
    finish_crystal_copies(
        molecule,
        copies,
        crystal,
        &format!("Symmetry mates within {radius} Å"),
    )
}

/// Builds the full unit cell: each chain copy is translated so its centroid lies in the cell.
pub fn build_unit_cell(molecule: &Molecule) -> Result<Molecule, SymmetryError> {
    let crystal = molecule
        .info
        .crystal
        .as_ref()
        .ok_or(SymmetryError::NoCell)?;
    let operators = crystal_operators(crystal)?;
    let orthogonal = crystal.cell.orthogonalization();
    let fractional = orthogonal.inverse();
    let chain_atoms = atoms_by_chain(molecule);
    let mut copies = Vec::new();
    for (operator_index, operator) in operators.iter().enumerate() {
        for (chain, atoms) in &chain_atoms {
            let centroid = atoms
                .iter()
                .map(|&index| operator.apply(molecule.atoms[index].position.as_dvec3()))
                .sum::<DVec3>()
                / atoms.len().max(1) as f64;
            let shift = -(fractional * centroid).floor();
            let mut shifted = operator.clone();
            shifted.translation = (operator.translation() + orthogonal * shift).to_array();
            shifted.id = format!(
                "{}_{}{}{}",
                operator_index + 1,
                5 + shift.x as i64,
                5 + shift.y as i64,
                5 + shift.z as i64
            );
            shifted.name = shifted.id.clone();
            let identity = shifted.is_identity();
            copies.push((chain.clone(), shifted, identity));
        }
    }
    finish_crystal_copies(molecule, copies, crystal, "Unit cell")
}

fn finish_crystal_copies(
    molecule: &Molecule,
    copies: Vec<(String, SymmetryOperator, bool)>,
    crystal: &CrystalInfo,
    description: &str,
) -> Result<Molecule, SymmetryError> {
    let chain_lists: Vec<Vec<String>> = copies
        .iter()
        .map(|(chain, _, _)| vec![chain.clone()])
        .collect();
    let copies: Vec<ChainCopy<'_>> = copies
        .into_iter()
        .zip(&chain_lists)
        .map(|((_, mut operator, identity), chains)| {
            if identity {
                operator.rotation = SymmetryOperator::identity("1").rotation;
                operator.translation = [0.0; 3];
                operator.id = "1".into();
            }
            ChainCopy { chains, operator }
        })
        .collect();
    let mut result = replicate(molecule, &copies)?;
    result.info.title = molecule.info.title.clone();
    result.info.id = molecule.info.id.clone();
    result.info.crystal = Some(crystal.clone());
    result.info.notes = vec![format!(
        "{} of {} ({})",
        description,
        source_name(molecule),
        crystal.space_group
    )];
    Ok(result)
}

fn source_name(molecule: &Molecule) -> String {
    molecule
        .info
        .id
        .clone()
        .unwrap_or_else(|| "the structure".into())
}

fn chain_ids(molecule: &Molecule) -> Vec<String> {
    let mut seen = Vec::<String>::new();
    for atom in &molecule.atoms {
        if seen.last() != Some(&atom.chain_id) && !seen.contains(&atom.chain_id) {
            seen.push(atom.chain_id.clone());
        }
    }
    seen
}

fn atoms_by_chain(molecule: &Molecule) -> BTreeMap<String, Vec<usize>> {
    let mut map = BTreeMap::<String, Vec<usize>>::new();
    for (index, atom) in molecule.atoms.iter().enumerate() {
        map.entry(atom.chain_id.clone()).or_default().push(index);
    }
    map
}

fn copy_is_near(
    molecule: &Molecule,
    atoms: &[usize],
    operator: &SymmetryOperator,
    grid: &AtomGrid,
    bounds: Option<(DVec3, DVec3)>,
    radius: f64,
) -> bool {
    let Some((minimum, maximum)) = bounds else {
        return false;
    };
    let transformed: Vec<DVec3> = atoms
        .iter()
        .map(|&index| operator.apply(molecule.atoms[index].position.as_dvec3()))
        .collect();
    let (copy_min, copy_max) = transformed.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(low, high), point| (low.min(*point), high.max(*point)),
    );
    if copy_min.cmpgt(maximum).any() || copy_max.cmplt(minimum).any() {
        return false;
    }
    transformed
        .iter()
        .any(|point| grid.any_within(molecule, *point, radius))
}

/// Uniform grid over the asymmetric unit for neighbor tests.
struct AtomGrid {
    cell: f64,
    cells: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl AtomGrid {
    fn new(molecule: &Molecule, cell: f64) -> Self {
        let mut cells = HashMap::<(i64, i64, i64), Vec<usize>>::new();
        for (index, atom) in molecule.atoms.iter().enumerate() {
            cells
                .entry(Self::key(atom.position.as_dvec3(), cell))
                .or_default()
                .push(index);
        }
        Self { cell, cells }
    }

    fn key(point: DVec3, cell: f64) -> (i64, i64, i64) {
        let scaled = (point / cell).floor();
        (scaled.x as i64, scaled.y as i64, scaled.z as i64)
    }

    fn any_within(&self, molecule: &Molecule, point: DVec3, radius: f64) -> bool {
        let (x, y, z) = Self::key(point, self.cell);
        let reach = (radius / self.cell).ceil() as i64;
        let radius_squared = radius * radius;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                for dz in -reach..=reach {
                    if let Some(indices) = self.cells.get(&(x + dx, y + dy, z + dz))
                        && indices.iter().any(|&index| {
                            molecule.atoms[index]
                                .position
                                .as_dvec3()
                                .distance_squared(point)
                                <= radius_squared
                        })
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

fn replicate(molecule: &Molecule, copies: &[ChainCopy<'_>]) -> Result<Molecule, SymmetryError> {
    let chain_atoms = atoms_by_chain(molecule);
    let total: usize = copies
        .iter()
        .map(|copy| {
            copy.chains
                .iter()
                .map(|chain| chain_atoms.get(chain).map_or(0, Vec::len))
                .sum::<usize>()
        })
        .sum();
    if total > MAX_GENERATED_ATOMS {
        return Err(SymmetryError::TooLarge(total));
    }
    if total == 0 {
        return Err(SymmetryError::Empty);
    }
    let mut bonds_by_atom = vec![Vec::new(); molecule.atoms.len()];
    for bond in &molecule.bonds {
        bonds_by_atom[bond.a].push(*bond);
    }
    let mut atoms = Vec::with_capacity(total);
    let mut bonds = Vec::new();
    let mut used_names = HashMap::<String, usize>::new();
    let mut chain_renames = Vec::new();
    for copy in copies {
        let identity = copy.operator.is_identity();
        // Atoms of this copy, mapped from source index to output index for bonds.
        let mut mapping = HashMap::<usize, usize>::new();
        for chain in copy.chains {
            let Some(indices) = chain_atoms.get(chain) else {
                continue;
            };
            let base = if identity {
                chain.clone()
            } else {
                format!("{chain}-{}", copy.operator.id)
            };
            // Distinct copies must never merge into one hierarchy chain.
            let count = used_names.entry(base.clone()).or_insert(0);
            let name = if *count == 0 {
                base.clone()
            } else {
                format!("{base}.{}", *count + 1)
            };
            *count += 1;
            chain_renames.push((chain.clone(), name.clone()));
            for &index in indices {
                let source = &molecule.atoms[index];
                let position = copy.operator.apply(source.position.as_dvec3());
                mapping.insert(index, atoms.len());
                atoms.push(Atom {
                    serial: atoms.len() as u32 + 1,
                    chain_id: name.clone(),
                    position: Vec3::new(position.x as f32, position.y as f32, position.z as f32),
                    ..source.clone()
                });
            }
        }
        for (&source, &target) in &mapping {
            for bond in &bonds_by_atom[source] {
                if let Some(&other) = mapping.get(&bond.b)
                    && let Some(bond) = Bond::with_order(target, other, bond.order, bond.kind)
                {
                    bonds.push(bond);
                }
            }
        }
    }
    bonds.sort_by_key(|bond| (bond.a, bond.b));
    bonds.dedup_by_key(|bond| (bond.a, bond.b));
    let mut info = StructureInfo {
        model_count: 1,
        ..StructureInfo::default()
    };
    for (source, target) in chain_renames {
        for annotation in &molecule.info.secondary {
            if annotation.chain == source {
                let mut copied = annotation.clone();
                copied.chain = target.clone();
                info.secondary.push(copied);
            }
        }
    }
    Ok(Molecule { atoms, bonds, info })
}

/// A space group with its general-position operators in fractional coordinates.
#[derive(Debug, Clone)]
pub struct SpaceGroup {
    pub number: u32,
    pub hermann_mauguin: String,
    pub extended: String,
    pub operations: Vec<(DMat3, DVec3)>,
}

struct SpaceGroupEntry {
    number: u32,
    hm: String,
    xhm: String,
    short: String,
    operators: String,
}

fn space_group_table() -> &'static [SpaceGroupEntry] {
    static TABLE: OnceLock<Vec<SpaceGroupEntry>> = OnceLock::new();
    TABLE.get_or_init(|| {
        include_str!("../../data/symmetry/spacegroups.tsv")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
            .filter_map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                (fields.len() == 6).then(|| SpaceGroupEntry {
                    number: fields[0].parse().unwrap_or(0),
                    hm: fields[1].to_string(),
                    xhm: fields[2].to_string(),
                    short: fields[3].to_string(),
                    operators: fields[5].to_string(),
                })
            })
            .collect()
    })
}

fn normalize_symbol(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '\'' && *c != '"')
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Finds a space group by Hermann–Mauguin symbol (full, short or PDB `H` notation) or by
/// number. The cell decides between hexagonal and rhombohedral settings.
pub fn find_space_group(symbol: &str, cell: Option<&UnitCell>) -> Option<SpaceGroup> {
    let wanted = normalize_symbol(symbol);
    if wanted.is_empty() {
        return None;
    }
    let table = space_group_table();
    let hexagonal_cell = cell.is_none_or(|cell| (cell.gamma - 120.0).abs() < 1.0);
    let matches: Vec<&SpaceGroupEntry> = table
        .iter()
        .filter(|entry| {
            normalize_symbol(&entry.hm) == wanted
                || normalize_symbol(&entry.xhm) == wanted
                || normalize_symbol(&entry.short) == wanted
                || entry.number.to_string() == wanted
        })
        .collect();
    let entry = if matches.len() > 1 {
        let setting = if hexagonal_cell { ":H" } else { ":R" };
        matches
            .iter()
            .find(|entry| entry.xhm.ends_with(setting))
            .or_else(|| matches.first())
            .copied()
    } else {
        matches.first().copied()
    }?;
    let operations = entry
        .operators
        .split(';')
        .map(parse_triplet)
        .collect::<Option<Vec<_>>>()?;
    Some(SpaceGroup {
        number: entry.number,
        hermann_mauguin: entry.hm.clone(),
        extended: entry.xhm.clone(),
        operations,
    })
}

/// Cartesian operators for the crystal's space group.
pub fn crystal_operators(crystal: &CrystalInfo) -> Result<Vec<SymmetryOperator>, SymmetryError> {
    if !crystal.cell.is_crystallographic() {
        return Err(SymmetryError::NoCell);
    }
    let group = find_space_group(&crystal.space_group, Some(&crystal.cell))
        .ok_or_else(|| SymmetryError::UnknownSpaceGroup(crystal.space_group.clone()))?;
    let orthogonal = crystal.cell.orthogonalization();
    let fractional = orthogonal.inverse();
    Ok(group
        .operations
        .iter()
        .enumerate()
        .map(|(index, (rotation, translation))| {
            let cartesian = orthogonal * *rotation * fractional;
            let shift = orthogonal * *translation;
            SymmetryOperator {
                id: (index + 1).to_string(),
                name: format!("{}_555", index + 1),
                rotation: cartesian.transpose().to_cols_array_2d(),
                translation: shift.to_array(),
            }
        })
        .collect())
}

/// Parses a coordinate triplet such as `-x+y,-x,z+1/3` into a rotation and translation.
pub fn parse_triplet(triplet: &str) -> Option<(DMat3, DVec3)> {
    let parts: Vec<&str> = triplet.split(',').collect();
    if parts.len() != 3 {
        return None;
    }
    let mut rows = [[0.0; 3]; 3];
    let mut translation = [0.0; 3];
    for (row, part) in parts.iter().enumerate() {
        let text: String = part
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_ascii_lowercase();
        let mut index = 0;
        let bytes = text.as_bytes();
        while index < bytes.len() {
            let mut sign = 1.0;
            if bytes[index] == b'+' || bytes[index] == b'-' {
                sign = if bytes[index] == b'-' { -1.0 } else { 1.0 };
                index += 1;
            }
            let start = index;
            while index < bytes.len() && bytes[index] != b'+' && bytes[index] != b'-' {
                index += 1;
            }
            let term = &text[start..index];
            if term.is_empty() {
                return None;
            }
            match term {
                "x" => rows[row][0] += sign,
                "y" => rows[row][1] += sign,
                "z" => rows[row][2] += sign,
                number => {
                    let value = if let Some((numerator, denominator)) = number.split_once('/') {
                        numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?
                    } else {
                        number.parse::<f64>().ok()?
                    };
                    translation[row] += sign * value;
                }
            }
        }
    }
    Some((
        DMat3::from_cols_array_2d(&rows).transpose(),
        DVec3::from_array(translation),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::{Assembly, AssemblyGenerator, Element};

    fn atom(chain: &str, position: Vec3) -> Atom {
        Atom {
            serial: 1,
            name: "CA".into(),
            element: Element::C,
            residue_name: "ALA".into(),
            residue_number: 1,
            chain_id: chain.into(),
            position,
            occupancy: 1.0,
            ..Atom::default()
        }
    }

    #[test]
    fn parses_symmetry_triplets() {
        let (rotation, translation) = parse_triplet("-x+y,-x,z+1/3").unwrap();
        let point = rotation * DVec3::new(1.0, 2.0, 3.0) + translation;
        assert!((point - DVec3::new(1.0, -1.0, 3.0 + 1.0 / 3.0)).length() < 1e-12);
        assert!(parse_triplet("x,y").is_none());
        assert!(parse_triplet("x,y,q").is_none());
    }

    #[test]
    fn finds_space_groups_by_common_notations() {
        assert_eq!(find_space_group("P 21 21 21", None).unwrap().number, 19);
        assert_eq!(find_space_group("P 1 21 1", None).unwrap().number, 4);
        assert_eq!(find_space_group("P21", None).unwrap().number, 4);
        let hexagonal = find_space_group("H 3", None).unwrap();
        assert_eq!(hexagonal.number, 146);
        assert_eq!(hexagonal.operations.len(), 9);
        let cell = UnitCell {
            a: 50.0,
            b: 50.0,
            c: 50.0,
            alpha: 80.0,
            beta: 80.0,
            gamma: 80.0,
        };
        let rhombohedral = find_space_group("R 3", Some(&cell)).unwrap();
        assert!(rhombohedral.extended.ends_with(":R"));
        assert_eq!(rhombohedral.operations.len(), 3);
        assert!(find_space_group("Q 99", None).is_none());
    }

    #[test]
    fn assembly_copies_chains_and_renames_non_identity_copies() {
        let mut molecule = Molecule::new(
            vec![atom("A", Vec3::new(1.0, 0.0, 0.0)), atom("B", Vec3::ZERO)],
            Vec::new(),
        );
        let mut rotation = SymmetryOperator::identity("2");
        rotation.rotation = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
        molecule.info.operators = vec![SymmetryOperator::identity("1"), rotation];
        molecule.info.assemblies = vec![Assembly {
            id: "1".into(),
            details: String::new(),
            oligomeric_count: Some(2),
            generators: vec![AssemblyGenerator {
                chains: vec!["A".into()],
                products: vec![vec![0], vec![1]],
            }],
        }];
        let assembly = build_assembly(&molecule, "1").unwrap();
        assert_eq!(assembly.atoms.len(), 2);
        assert_eq!(assembly.atoms[0].chain_id, "A");
        assert_eq!(assembly.atoms[1].chain_id, "A-2");
        assert!((assembly.atoms[1].position - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-6);
        assert_eq!(
            build_assembly(&molecule, "9"),
            Err(SymmetryError::UnknownAssembly("9".into()))
        );
    }

    #[test]
    fn unit_cell_places_every_copy_inside_the_cell() {
        let mut molecule = Molecule::new(vec![atom("A", Vec3::new(5.0, 6.0, 7.0))], Vec::new());
        let cell = UnitCell {
            a: 40.0,
            b: 50.0,
            c: 60.0,
            alpha: 90.0,
            beta: 90.0,
            gamma: 90.0,
        };
        molecule.info.crystal = Some(CrystalInfo {
            cell,
            space_group: "P 21 21 21".into(),
        });
        let packed = build_unit_cell(&molecule).unwrap();
        assert_eq!(packed.atoms.len(), 4);
        for atom in &packed.atoms {
            let p = atom.position;
            assert!(p.x >= -1e-3 && p.x <= 40.0 && p.y >= -1e-3 && p.y <= 50.0);
            assert!(p.z >= -1e-3 && p.z <= 60.0);
        }
        let mates = build_symmetry_mates(&molecule, 100.0).unwrap();
        assert!(mates.atoms.len() > 4);
        assert_eq!(mates.atoms[0].chain_id, "A");
    }
}
