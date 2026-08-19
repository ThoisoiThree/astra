use glam::Vec3;

use crate::molecule::Molecule;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Returns the closest atom intersected by the ray using display-sized atom spheres.
pub fn pick_atom(molecule: &Molecule, ray: Ray) -> Option<usize> {
    pick_atom_filtered(molecule, ray, |_| true)
}

/// Returns the closest intersected atom accepted by the display predicate.
pub fn pick_atom_filtered(
    molecule: &Molecule,
    ray: Ray,
    is_visible: impl FnMut(usize) -> bool,
) -> Option<usize> {
    pick_atom_filtered_with_radius(molecule, ray, is_visible, |_, atom| {
        (atom.element.van_der_waals_radius() * 0.32).max(0.32)
    })
}

/// Returns the closest atom using the radius supplied by the active display mode.
pub fn pick_atom_filtered_with_radius(
    molecule: &Molecule,
    ray: Ray,
    mut is_visible: impl FnMut(usize) -> bool,
    mut radius: impl FnMut(usize, &crate::molecule::Atom) -> f32,
) -> Option<usize> {
    let direction = ray.direction.normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, _)| is_visible(*index))
        .filter_map(|(index, atom)| {
            let radius = radius(index, atom).max(0.001);
            ray_sphere_distance(ray.origin, direction, atom.position, radius)
                .map(|distance| (index, distance))
        })
        .min_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

fn ray_sphere_distance(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let to_center = center - origin;
    let projected = to_center.dot(direction);
    let perpendicular_squared = to_center.length_squared() - projected * projected;
    let radius_squared = radius * radius;
    if perpendicular_squared > radius_squared {
        return None;
    }
    let offset = (radius_squared - perpendicular_squared).sqrt();
    let near = projected - offset;
    let far = projected + offset;
    if near >= 0.0 {
        Some(near)
    } else if far >= 0.0 {
        Some(far)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element};

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
    fn picks_closest_intersected_atom() {
        let molecule = Molecule {
            atoms: vec![
                atom(Vec3::new(0.0, 0.0, -4.0)),
                atom(Vec3::new(0.0, 0.0, -2.0)),
                atom(Vec3::new(4.0, 0.0, -1.0)),
            ],
            bonds: Vec::new(),
        };
        let picked = pick_atom(
            &molecule,
            Ray {
                origin: Vec3::ZERO,
                direction: -Vec3::Z,
            },
        );
        assert_eq!(picked, Some(1));
    }

    #[test]
    fn misses_atoms_outside_ray() {
        let molecule = Molecule {
            atoms: vec![atom(Vec3::new(3.0, 0.0, -2.0))],
            bonds: Vec::new(),
        };
        assert_eq!(
            pick_atom(
                &molecule,
                Ray {
                    origin: Vec3::ZERO,
                    direction: -Vec3::Z,
                }
            ),
            None
        );
    }

    #[test]
    fn ignores_atoms_rejected_by_visibility_filter() {
        let molecule = Molecule {
            atoms: vec![
                atom(Vec3::new(0.0, 0.0, -2.0)),
                atom(Vec3::new(0.0, 0.0, -4.0)),
            ],
            bonds: Vec::new(),
        };
        let picked = pick_atom_filtered(
            &molecule,
            Ray {
                origin: Vec3::ZERO,
                direction: -Vec3::Z,
            },
            |index| index == 1,
        );
        assert_eq!(picked, Some(1));
    }
}
