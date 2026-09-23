use glam::Vec3;

use super::{AnnotatedStructure, Molecule, MoleculeHierarchy, ResidueGroup, dssp::DsspCode};

/// Three-state protein assignment plus turns and nucleic-acid backbones for rendering.
/// Helix combines DSSP H/G/I and strand combines E/B; coils and turns remain visually distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryStructure {
    Coil,
    Turn,
    Helix,
    Strand,
    Nucleic,
}

#[derive(Clone, Copy, Default)]
struct ProteinBackbone {
    n: Option<Vec3>,
    ca: Option<Vec3>,
    c: Option<Vec3>,
}

/// Where the displayed secondary structure comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SecondarySource {
    /// DSSP from backbone hydrogen bonds, consistent across every file format.
    #[default]
    Dssp,
    /// HELIX/SHEET or struct_conf/struct_sheet_range records deposited with the file.
    File,
}

impl SecondarySource {
    pub const ALL: [Self; 2] = [Self::Dssp, Self::File];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Dssp => "DSSP (computed)",
            Self::File => "From file",
        }
    }
}

/// Assigns secondary structure with DSSP. Chains without complete backbones (Cα traces) use a
/// P-SEA-style geometric method so that coarse models still show helices and strands.
pub fn assign_secondary_structure(
    molecule: &Molecule,
    hierarchy: &MoleculeHierarchy,
) -> Vec<Vec<SecondaryStructure>> {
    assign_secondary_structure_from(molecule, hierarchy, SecondarySource::Dssp)
}

/// Assigns secondary structure from the requested source. `File` falls back to DSSP when the
/// structure carries no annotations.
pub fn assign_secondary_structure_from(
    molecule: &Molecule,
    hierarchy: &MoleculeHierarchy,
    source: SecondarySource,
) -> Vec<Vec<SecondaryStructure>> {
    if source == SecondarySource::File && !molecule.info.secondary.is_empty() {
        return from_annotations(molecule, hierarchy);
    }
    let dssp = super::dssp::assign(molecule, hierarchy);
    hierarchy
        .chains
        .iter()
        .zip(dssp)
        .map(|(chain, codes)| {
            let complete = codes.iter().filter(|code| code.is_some()).count();
            let with_ca = chain
                .residues
                .iter()
                .filter(|residue| protein_backbone(molecule, residue).ca.is_some())
                .count();
            if with_ca > 0 && complete * 2 < with_ca {
                return assign_chain(molecule, &chain.residues);
            }
            chain
                .residues
                .iter()
                .zip(codes)
                .map(|(residue, code)| match code {
                    Some(DsspCode::AlphaHelix | DsspCode::Helix310 | DsspCode::PiHelix) => {
                        SecondaryStructure::Helix
                    }
                    Some(DsspCode::Strand) => SecondaryStructure::Strand,
                    Some(DsspCode::Turn) => SecondaryStructure::Turn,
                    Some(DsspCode::Bridge | DsspCode::Loop) => SecondaryStructure::Coil,
                    None if has_nucleic_anchor(molecule, residue) => SecondaryStructure::Nucleic,
                    None => SecondaryStructure::Coil,
                })
                .collect()
        })
        .collect()
}

fn from_annotations(
    molecule: &Molecule,
    hierarchy: &MoleculeHierarchy,
) -> Vec<Vec<SecondaryStructure>> {
    hierarchy
        .chains
        .iter()
        .map(|chain| {
            let mut states: Vec<SecondaryStructure> = chain
                .residues
                .iter()
                .map(|residue| {
                    if has_nucleic_anchor(molecule, residue) {
                        SecondaryStructure::Nucleic
                    } else {
                        SecondaryStructure::Coil
                    }
                })
                .collect();
            let position = |(number, code): (i32, Option<char>)| {
                chain.residues.iter().position(|residue| {
                    residue.id.number == number && residue.id.insertion_code == code
                })
            };
            for annotation in molecule
                .info
                .secondary
                .iter()
                .filter(|annotation| annotation.chain == chain.id)
            {
                let (Some(start), Some(end)) =
                    (position(annotation.start), position(annotation.end))
                else {
                    continue;
                };
                let state = match annotation.kind {
                    AnnotatedStructure::Helix
                    | AnnotatedStructure::Helix310
                    | AnnotatedStructure::HelixPi => SecondaryStructure::Helix,
                    AnnotatedStructure::Strand => SecondaryStructure::Strand,
                    AnnotatedStructure::Turn => SecondaryStructure::Turn,
                };
                for slot in &mut states[start.min(end)..=start.max(end)] {
                    if *slot != SecondaryStructure::Nucleic
                        && !(state == SecondaryStructure::Turn && *slot != SecondaryStructure::Coil)
                    {
                        *slot = state;
                    }
                }
            }
            states
        })
        .collect()
}

fn assign_chain(molecule: &Molecule, residues: &[ResidueGroup]) -> Vec<SecondaryStructure> {
    let backbone: Vec<_> = residues
        .iter()
        .map(|residue| protein_backbone(molecule, residue))
        .collect();
    let mut assignment = residues
        .iter()
        .map(|residue| {
            if has_nucleic_anchor(molecule, residue) {
                SecondaryStructure::Nucleic
            } else {
                SecondaryStructure::Coil
            }
        })
        .collect::<Vec<_>>();

    for index in 1..residues.len().saturating_sub(1) {
        let (Some(previous_c), Some(n), Some(ca), Some(c), Some(next_n)) = (
            backbone[index - 1].c,
            backbone[index].n,
            backbone[index].ca,
            backbone[index].c,
            backbone[index + 1].n,
        ) else {
            continue;
        };
        if previous_c.distance(n) > 2.0 || c.distance(next_n) > 2.0 {
            continue;
        }
        let phi = dihedral_degrees(previous_c, n, ca, c);
        let psi = dihedral_degrees(n, ca, c, next_n);
        assignment[index] = classify_ramachandran(phi, psi);
    }

    assign_ca_distance_windows(&backbone, &mut assignment);
    fill_single_residue_gaps(&mut assignment, SecondaryStructure::Helix);
    fill_single_residue_gaps(&mut assignment, SecondaryStructure::Strand);
    remove_short_runs(&mut assignment, SecondaryStructure::Helix, 3);
    remove_short_runs(&mut assignment, SecondaryStructure::Strand, 2);
    assign_turns(&backbone, &mut assignment);
    assignment
}

/// P-SEA-style Cα masks recover regular elements at their termini and in structures whose
/// peptide atoms are incomplete. The tolerances include the reported terminal variation.
fn assign_ca_distance_windows(backbone: &[ProteinBackbone], assignment: &mut [SecondaryStructure]) {
    for start in 0..backbone.len().saturating_sub(4) {
        let Some(points) = ca_window(backbone, start) else {
            continue;
        };
        let d2 = points[0].distance(points[2]);
        let d3 = points[0].distance(points[3]);
        let d4 = points[0].distance(points[4]);
        let helix = within(d2, 5.5, 0.65) && within(d3, 5.3, 0.75) && within(d4, 6.4, 0.90);
        let strand = within(d2, 6.7, 0.75) && within(d3, 9.9, 1.10) && within(d4, 12.4, 1.30);
        if helix {
            for state in &mut assignment[start..=start + 4] {
                if *state != SecondaryStructure::Nucleic {
                    *state = SecondaryStructure::Helix;
                }
            }
        } else if strand {
            for state in &mut assignment[start..=start + 4] {
                if !matches!(
                    *state,
                    SecondaryStructure::Nucleic | SecondaryStructure::Helix
                ) {
                    *state = SecondaryStructure::Strand;
                }
            }
        }
    }
}

fn ca_window(backbone: &[ProteinBackbone], start: usize) -> Option<[Vec3; 5]> {
    let points = [
        backbone.get(start)?.ca?,
        backbone.get(start + 1)?.ca?,
        backbone.get(start + 2)?.ca?,
        backbone.get(start + 3)?.ca?,
        backbone.get(start + 4)?.ca?,
    ];
    points
        .windows(2)
        .all(|pair| (2.8..=4.5).contains(&pair[0].distance(pair[1])))
        .then_some(points)
}

fn within(value: f32, center: f32, tolerance: f32) -> bool {
    (value - center).abs() <= tolerance
}

fn protein_backbone(molecule: &Molecule, residue: &ResidueGroup) -> ProteinBackbone {
    let mut result = ProteinBackbone::default();
    for atom in residue
        .atom_indices
        .iter()
        .filter_map(|index| molecule.atoms.get(*index))
    {
        match atom.name.as_str() {
            "N" => result.n = Some(atom.position),
            "CA" => result.ca = Some(atom.position),
            "C" => result.c = Some(atom.position),
            _ => {}
        }
    }
    result
}

fn has_nucleic_anchor(molecule: &Molecule, residue: &ResidueGroup) -> bool {
    residue.atom_indices.iter().any(|index| {
        molecule
            .atoms
            .get(*index)
            .is_some_and(|atom| !atom.hetero && atom.name == "P")
    })
}

fn classify_ramachandran(phi: f32, psi: f32) -> SecondaryStructure {
    let right_handed_helix = (-120.0..=-20.0).contains(&phi) && (-90.0..=35.0).contains(&psi);
    let left_handed_helix = (20.0..=120.0).contains(&phi) && (-35.0..=95.0).contains(&psi);
    if right_handed_helix || left_handed_helix {
        SecondaryStructure::Helix
    } else if (-180.0..=-55.0).contains(&phi) && (psi >= 45.0 || psi <= -140.0) {
        SecondaryStructure::Strand
    } else {
        SecondaryStructure::Coil
    }
}

fn fill_single_residue_gaps(assignment: &mut [SecondaryStructure], kind: SecondaryStructure) {
    if assignment.len() < 3 {
        return;
    }
    for index in 1..assignment.len() - 1 {
        if assignment[index] == SecondaryStructure::Coil
            && assignment[index - 1] == kind
            && assignment[index + 1] == kind
        {
            assignment[index] = kind;
        }
    }
}

fn remove_short_runs(
    assignment: &mut [SecondaryStructure],
    kind: SecondaryStructure,
    minimum_length: usize,
) {
    let mut start = 0;
    while start < assignment.len() {
        if assignment[start] != kind {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < assignment.len() && assignment[end] == kind {
            end += 1;
        }
        if end - start < minimum_length {
            assignment[start..end].fill(SecondaryStructure::Coil);
        }
        start = end;
    }
}

fn assign_turns(backbone: &[ProteinBackbone], assignment: &mut [SecondaryStructure]) {
    for start in 0..backbone.len().saturating_sub(3) {
        let (Some(first), Some(last)) = (backbone[start].ca, backbone[start + 3].ca) else {
            continue;
        };
        if first.distance(last) <= 7.0 {
            for state in &mut assignment[start + 1..=start + 2] {
                if *state == SecondaryStructure::Coil {
                    *state = SecondaryStructure::Turn;
                }
            }
        }
    }

    for index in 2..backbone.len().saturating_sub(2) {
        let (Some(before), Some(center), Some(after)) = (
            backbone[index - 2].ca,
            backbone[index].ca,
            backbone[index + 2].ca,
        ) else {
            continue;
        };
        let incoming = (center - before).normalize_or_zero();
        let outgoing = (after - center).normalize_or_zero();
        if incoming != Vec3::ZERO
            && outgoing != Vec3::ZERO
            && incoming.dot(outgoing).clamp(-1.0, 1.0).acos().to_degrees() >= 70.0
            && assignment[index] == SecondaryStructure::Coil
        {
            assignment[index] = SecondaryStructure::Turn;
        }
    }
}

fn dihedral_degrees(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f32 {
    let axis = (c - b).normalize_or_zero();
    if axis == Vec3::ZERO {
        return 0.0;
    }
    let first_bond = a - b;
    let first = first_bond - axis * first_bond.dot(axis);
    let second = (d - c) - axis * (d - c).dot(axis);
    let first = first.normalize_or_zero();
    let second = second.normalize_or_zero();
    if first == Vec3::ZERO || second == Vec3::ZERO {
        return 0.0;
    }
    axis.dot(first.cross(second))
        .atan2(first.dot(second))
        .to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramachandran_regions_cover_helices_strands_and_coils() {
        assert_eq!(
            classify_ramachandran(-57.0, -47.0),
            SecondaryStructure::Helix
        );
        assert_eq!(
            classify_ramachandran(-130.0, 130.0),
            SecondaryStructure::Strand
        );
        assert_eq!(
            classify_ramachandran(170.0, 170.0),
            SecondaryStructure::Coil
        );
    }

    #[test]
    fn short_regular_fragments_are_reduced_to_coil() {
        let mut assignment = vec![
            SecondaryStructure::Coil,
            SecondaryStructure::Helix,
            SecondaryStructure::Helix,
            SecondaryStructure::Coil,
            SecondaryStructure::Strand,
            SecondaryStructure::Coil,
        ];
        remove_short_runs(&mut assignment, SecondaryStructure::Helix, 3);
        remove_short_runs(&mut assignment, SecondaryStructure::Strand, 2);
        assert!(
            assignment
                .iter()
                .all(|state| *state == SecondaryStructure::Coil)
        );
    }

    #[test]
    fn ca_distance_mask_recovers_a_canonical_alpha_helix() {
        let backbone = (0..7)
            .map(|index| {
                let angle = index as f32 * 100.0_f32.to_radians();
                ProteinBackbone {
                    ca: Some(Vec3::new(
                        2.3 * angle.cos(),
                        2.3 * angle.sin(),
                        index as f32 * 1.5,
                    )),
                    ..Default::default()
                }
            })
            .collect::<Vec<_>>();
        let mut assignment = vec![SecondaryStructure::Coil; backbone.len()];
        assign_ca_distance_windows(&backbone, &mut assignment);
        assert!(
            assignment
                .iter()
                .all(|state| *state == SecondaryStructure::Helix)
        );
    }
}
