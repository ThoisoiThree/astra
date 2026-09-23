//! DSSP secondary-structure assignment (Kabsch & Sander, Biopolymers 22, 2577, 1983).
//!
//! Backbone hydrogen bonds use the electrostatic energy model with the amide hydrogen placed
//! along the preceding C=O bisector. Each NH keeps its two strongest acceptors, as in DSSP.
//! Helices (H, G, I), bridges and ladders (B, E) and turns (T) follow the original rules.

use std::collections::HashMap;

use glam::Vec3;

use super::{Molecule, MoleculeHierarchy};

/// DSSP one-letter states. Bends (S) and polyproline (P) are not assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsspCode {
    AlphaHelix,
    Helix310,
    PiHelix,
    Strand,
    Bridge,
    Turn,
    Loop,
}

impl DsspCode {
    pub const fn letter(self) -> char {
        match self {
            Self::AlphaHelix => 'H',
            Self::Helix310 => 'G',
            Self::PiHelix => 'I',
            Self::Strand => 'E',
            Self::Bridge => 'B',
            Self::Turn => 'T',
            Self::Loop => '-',
        }
    }
}

const COUPLING: f32 = 0.084 * 332.0;
const MAX_HBOND_ENERGY: f32 = -0.5;
const MIN_ENERGY: f32 = -9.9;
const CA_CUTOFF: f32 = 9.0;
const PEPTIDE_BREAK: f32 = 2.5;

struct Residue {
    chain: usize,
    residue: usize,
    n: Vec3,
    ca: Vec3,
    c: Vec3,
    o: Vec3,
    h: Option<Vec3>,
    /// True when the residue is not covalently linked to the previous entry.
    break_before: bool,
    /// The two strongest acceptors of this residue's NH: (acceptor index, energy).
    acceptors: [(usize, f32); 2],
}

/// Assigns DSSP states to every residue with a complete N, CA, C, O backbone. The result is
/// indexed like `hierarchy.chains[chain].residues[residue]`; other residues are `None`.
pub fn assign(molecule: &Molecule, hierarchy: &MoleculeHierarchy) -> Vec<Vec<Option<DsspCode>>> {
    let mut result: Vec<Vec<Option<DsspCode>>> = hierarchy
        .chains
        .iter()
        .map(|chain| vec![None; chain.residues.len()])
        .collect();
    let mut residues = collect_backbone(molecule, hierarchy);
    if residues.is_empty() {
        return result;
    }
    compute_hydrogen_bonds(&mut residues);
    let codes = classify(&residues);
    for (residue, code) in residues.iter().zip(codes) {
        result[residue.chain][residue.residue] = Some(code);
    }
    result
}

fn collect_backbone(molecule: &Molecule, hierarchy: &MoleculeHierarchy) -> Vec<Residue> {
    let mut residues: Vec<Residue> = Vec::new();
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let mut previous_in_chain: Option<usize> = None;
        for (residue_index, group) in chain.residues.iter().enumerate() {
            let mut atoms: [Option<Vec3>; 4] = [None; 4];
            let mut proline = false;
            for &index in &group.atom_indices {
                let atom = &molecule.atoms[index];
                proline |= atom.residue_name == "PRO";
                let slot = match atom.name.as_str() {
                    "N" => 0,
                    "CA" => 1,
                    "C" => 2,
                    "O" => 3,
                    _ => continue,
                };
                atoms[slot].get_or_insert(atom.position);
            }
            let [Some(n), Some(ca), Some(c), Some(o)] = atoms else {
                previous_in_chain = None;
                continue;
            };
            let previous = previous_in_chain.map(|index| &residues[index]);
            let break_before = previous.is_none_or(|previous| {
                previous.c.distance(n) > PEPTIDE_BREAK || previous.residue + 1 != residue_index
            });
            let h = match previous {
                Some(previous) if !break_before && !proline => {
                    let direction = (previous.c - previous.o).normalize_or_zero();
                    (direction != Vec3::ZERO).then_some(n + direction)
                }
                _ => None,
            };
            previous_in_chain = Some(residues.len());
            residues.push(Residue {
                chain: chain_index,
                residue: residue_index,
                n,
                ca,
                c,
                o,
                h,
                break_before,
                acceptors: [(usize::MAX, 0.0); 2],
            });
        }
    }
    residues
}

fn energy(donor: &Residue, acceptor: &Residue) -> f32 {
    let Some(h) = donor.h else {
        return 0.0;
    };
    let on = acceptor.o.distance(donor.n);
    let ch = acceptor.c.distance(h);
    let oh = acceptor.o.distance(h);
    let cn = acceptor.c.distance(donor.n);
    if on < 0.5 || ch < 0.5 || oh < 0.5 || cn < 0.5 {
        return MIN_ENERGY;
    }
    (COUPLING * (1.0 / on + 1.0 / ch - 1.0 / oh - 1.0 / cn)).max(MIN_ENERGY)
}

fn compute_hydrogen_bonds(residues: &mut [Residue]) {
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    let cell = |point: Vec3| {
        let scaled = (point / CA_CUTOFF).floor();
        (scaled.x as i32, scaled.y as i32, scaled.z as i32)
    };
    for (index, residue) in residues.iter().enumerate() {
        cells.entry(cell(residue.ca)).or_default().push(index);
    }
    for donor in 0..residues.len() {
        if residues[donor].h.is_none() {
            continue;
        }
        let (x, y, z) = cell(residues[donor].ca);
        let mut best = [(usize::MAX, 0.0_f32); 2];
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(candidates) = cells.get(&(x + dx, y + dy, z + dz)) else {
                        continue;
                    };
                    for &acceptor in candidates {
                        if acceptor == donor
                            || residues[donor].ca.distance(residues[acceptor].ca) >= CA_CUTOFF
                        {
                            continue;
                        }
                        let value = energy(&residues[donor], &residues[acceptor]);
                        if value < best[0].1 {
                            best[1] = best[0];
                            best[0] = (acceptor, value);
                        } else if value < best[1].1 {
                            best[1] = (acceptor, value);
                        }
                    }
                }
            }
        }
        residues[donor].acceptors = best;
    }
}

/// NH of `donor` is hydrogen-bonded to CO of `acceptor`.
fn bonded(residues: &[Residue], donor: usize, acceptor: usize) -> bool {
    residues.get(donor).is_some_and(|residue| {
        residue
            .acceptors
            .iter()
            .any(|&(index, energy)| index == acceptor && energy < MAX_HBOND_ENERGY)
    })
}

/// No chain break between residues `first` and `last` inclusive.
fn continuous(residues: &[Residue], first: usize, last: usize) -> bool {
    last < residues.len() && (first + 1..=last).all(|index| !residues[index].break_before)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BridgeKind {
    Parallel,
    Antiparallel,
}

fn classify(residues: &[Residue]) -> Vec<DsspCode> {
    let count = residues.len();
    let mut codes = vec![DsspCode::Loop; count];

    // n-turns: CO(i) accepts from NH(i+n).
    let mut turns = [vec![false; count], vec![false; count], vec![false; count]];
    for (turn, n) in turns.iter_mut().zip([3_usize, 4, 5]) {
        for (i, slot) in turn.iter_mut().enumerate().take(count.saturating_sub(n)) {
            *slot = continuous(residues, i, i + n) && bonded(residues, i + n, i);
        }
    }

    // Bridges between residues i and j (|i - j| > 2) with both neighbors present.
    let mut partners: Vec<Vec<(usize, BridgeKind)>> = vec![Vec::new(); count];
    for i in 1..count.saturating_sub(1) {
        if !continuous(residues, i - 1, i + 1) {
            continue;
        }
        for j in i + 3..count.saturating_sub(1) {
            if !continuous(residues, j - 1, j + 1) {
                continue;
            }
            let parallel = (bonded(residues, i + 1, j) && bonded(residues, j, i - 1))
                || (bonded(residues, j + 1, i) && bonded(residues, i, j - 1));
            let antiparallel = (bonded(residues, i, j) && bonded(residues, j, i))
                || (bonded(residues, i + 1, j - 1) && bonded(residues, j + 1, i - 1));
            let kind = if parallel {
                Some(BridgeKind::Parallel)
            } else if antiparallel {
                Some(BridgeKind::Antiparallel)
            } else {
                None
            };
            if let Some(kind) = kind {
                partners[i].push((j, kind));
                partners[j].push((i, kind));
            }
        }
    }

    // Ladders: consecutive bridges of one kind; bulges join ladders separated by at most
    // one residue on one strand and four on the other.
    let mut strand = vec![false; count];
    let mut bridge = vec![false; count];
    for i in 0..count {
        for &(j, kind) in &partners[i] {
            if j < i {
                continue;
            }
            bridge[i] = true;
            bridge[j] = true;
            // Look for the next bridge of the same ladder: one residue further on both
            // strands, or a bulge of up to four residues on one strand and one on the other.
            for step_i in 1..=4_usize {
                let i2 = i + step_i;
                if i2 >= count {
                    break;
                }
                for &(j2, kind2) in &partners[i2] {
                    if kind2 != kind || j2 <= i2 {
                        continue;
                    }
                    let step_j = match kind {
                        BridgeKind::Parallel => j2.checked_sub(j),
                        BridgeKind::Antiparallel => j.checked_sub(j2),
                    };
                    let Some(step_j) = step_j.filter(|step| *step >= 1) else {
                        continue;
                    };
                    let bulge_ok = (step_i <= 1 && step_j <= 4) || (step_i <= 4 && step_j <= 1);
                    if bulge_ok {
                        strand[i..=i2].fill(true);
                        let (low, high) = (j.min(j2), j.max(j2));
                        strand[low..=high].fill(true);
                    }
                }
            }
        }
    }

    // Priority H > B > E > G > I > T.
    let helix = |slot: usize, n: usize, code: DsspCode, codes: &mut [DsspCode]| {
        for i in 1..count.saturating_sub(n) {
            if !(turns[slot][i - 1] && turns[slot][i]) {
                continue;
            }
            let span = i..i + n;
            if code != DsspCode::AlphaHelix
                && span.clone().any(|index| codes[index] != DsspCode::Loop)
            {
                continue;
            }
            for index in span {
                if codes[index] == DsspCode::Loop || code == DsspCode::AlphaHelix {
                    codes[index] = code;
                }
            }
        }
    };
    helix(1, 4, DsspCode::AlphaHelix, &mut codes);
    for index in 0..count {
        if codes[index] == DsspCode::Loop {
            if strand[index] {
                codes[index] = DsspCode::Strand;
            } else if bridge[index] {
                codes[index] = DsspCode::Bridge;
            }
        }
    }
    helix(0, 3, DsspCode::Helix310, &mut codes);
    helix(2, 5, DsspCode::PiHelix, &mut codes);
    for (slot, n) in [3_usize, 4, 5].into_iter().enumerate() {
        for i in 0..count.saturating_sub(n) {
            if turns[slot][i] {
                for code in &mut codes[i + 1..i + n] {
                    if *code == DsspCode::Loop {
                        *code = DsspCode::Turn;
                    }
                }
            }
        }
    }
    codes
}

#[cfg(test)]
pub(crate) mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element};

    /// Places atom D from A, B, C with bond length, angle and torsion (NeRF).
    fn place(a: Vec3, b: Vec3, c: Vec3, length: f32, angle: f32, torsion: f32) -> Vec3 {
        let bc = (c - b).normalize();
        let normal = (b - a).cross(bc).normalize();
        let m = [bc, normal.cross(bc), normal];
        let (angle, torsion) = (angle.to_radians(), torsion.to_radians());
        let d2 = Vec3::new(
            -length * angle.cos(),
            length * angle.sin() * torsion.cos(),
            length * angle.sin() * torsion.sin(),
        );
        c + m[0] * d2.x + m[1] * d2.y + m[2] * d2.z
    }

    /// Builds an ideal backbone (N, CA, C, O per residue) from phi/psi angles.
    pub(crate) fn backbone(dihedrals: &[(f32, f32)], chain: &str, first_number: i32) -> Vec<Atom> {
        let mut n = Vec3::new(0.0, 1.458, 0.0);
        let mut ca = Vec3::ZERO;
        let mut c = place(Vec3::new(-1.0, 1.5, 0.0), n, ca, 1.525, 111.0, -60.0);
        let mut atoms = Vec::new();
        let atom = |name: &str, element, position, number| Atom {
            serial: 1,
            name: name.into(),
            element,
            residue_name: "ALA".into(),
            residue_number: number,
            chain_id: chain.into(),
            position,
            occupancy: 1.0,
            ..Atom::default()
        };
        for (index, &(phi, psi)) in dihedrals.iter().enumerate() {
            if index > 0 {
                let previous_psi = dihedrals[index - 1].1;
                let next_n = place(n, ca, c, 1.329, 116.2, previous_psi);
                let next_ca = place(ca, c, next_n, 1.458, 121.7, 180.0);
                let next_c = place(c, next_n, next_ca, 1.525, 111.0, phi);
                n = next_n;
                ca = next_ca;
                c = next_c;
            }
            let o = place(n, ca, c, 1.231, 120.5, psi + 180.0);
            let number = first_number + index as i32;
            atoms.push(atom("N", Element::N, n, number));
            atoms.push(atom("CA", Element::C, ca, number));
            atoms.push(atom("C", Element::C, c, number));
            atoms.push(atom("O", Element::O, o, number));
        }
        atoms
    }

    fn codes(atoms: Vec<Atom>) -> String {
        let molecule = Molecule::new(atoms, Vec::new());
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        assign(&molecule, &hierarchy)
            .into_iter()
            .flatten()
            .map(|code| code.map_or('?', DsspCode::letter))
            .collect()
    }

    #[test]
    fn ideal_alpha_helix_is_assigned_h() {
        let result = codes(backbone(&[(-57.0, -47.0); 16], "A", 1));
        assert_eq!(result.len(), 16);
        // DSSP leaves the first and last residues of a helix outside the H segment.
        assert!(result[1..15].chars().all(|code| code == 'H'), "{result}");
    }

    #[test]
    fn extended_chain_alone_has_no_regular_structure() {
        let result = codes(backbone(&[(-120.0, 130.0); 10], "A", 1));
        assert!(result.chars().all(|code| code == '-'), "{result}");
    }

    #[test]
    fn beta_hairpin_forms_antiparallel_strands() {
        // Two extended strands joined by a type I' turn.
        let mut dihedrals = vec![(-120.0, 130.0); 6];
        dihedrals.extend([(60.0, 30.0), (90.0, 0.0)]);
        dihedrals.extend(vec![(-120.0, 130.0); 6]);
        let result = codes(backbone(&dihedrals, "A", 1));
        // The strands pair up next to the turn; ideal torsions splay them further out.
        assert_eq!(&result[4..10], "EETTEE", "{result}");
    }
}
