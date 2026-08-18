use std::collections::{HashMap, HashSet};

use super::{Atom, Bond};

const CELL_SIZE: f32 = 3.0;
const TOLERANCE: f32 = 0.45;
const MIN_DISTANCE: f32 = 0.35;
const MAX_DISTANCE: f32 = 3.0;

type Cell = (i32, i32, i32);

/// Infers connectivity with a uniform spatial grid, avoiding global all-pairs work.
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
                            if (MIN_DISTANCE..=limit).contains(&distance) {
                                let key = (other_index, index);
                                if seen.insert(key) {
                                    result.push(Bond {
                                        a: other_index,
                                        b: index,
                                    });
                                }
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

    fn atom(position: Vec3) -> Atom {
        Atom {
            serial: 1,
            name: "C".into(),
            element: Element::C,
            residue_name: "GLY".into(),
            residue_number: 1,
            insertion_code: None,
            chain_id: "A".into(),
            position,
            occupancy: 1.0,
            b_factor: 0.0,
            hetero: false,
        }
    }

    #[test]
    fn infers_near_but_not_distant_bonds() {
        let atoms = [
            atom(Vec3::ZERO),
            atom(Vec3::new(1.45, 0.0, 0.0)),
            atom(Vec3::new(8.0, 0.0, 0.0)),
        ];
        assert_eq!(infer_bonds(&atoms, &[]), vec![Bond { a: 0, b: 1 }]);
    }

    #[test]
    fn keeps_explicit_bonds_without_duplicates() {
        let atoms = [atom(Vec3::ZERO), atom(Vec3::new(1.4, 0.0, 0.0))];
        let explicit = [Bond { a: 0, b: 1 }];
        assert_eq!(infer_bonds(&atoms, &explicit), explicit);
    }
}
