use glam::Vec3;

use crate::molecule::Molecule;

const LEAF_SIZE: usize = 12;

#[derive(Debug, Clone, Default)]
pub struct AtomBvh {
    nodes: Vec<BvhNode>,
    atom_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct BvhNode {
    minimum: Vec3,
    maximum: Vec3,
    start: usize,
    count: usize,
    left: Option<usize>,
    right: Option<usize>,
}

impl AtomBvh {
    pub fn build(molecule: &Molecule) -> Self {
        let mut result = Self {
            nodes: Vec::new(),
            atom_indices: (0..molecule.atoms.len()).collect(),
        };
        if !result.atom_indices.is_empty() {
            result.build_node(molecule, 0, result.atom_indices.len());
        }
        result
    }

    pub fn pick_with_radius(
        &self,
        molecule: &Molecule,
        ray: Ray,
        mut is_visible: impl FnMut(usize) -> bool,
        mut radius: impl FnMut(usize, &crate::molecule::Atom) -> f32,
    ) -> Option<usize> {
        let direction = ray.direction.normalize_or_zero();
        if direction == Vec3::ZERO || self.nodes.is_empty() {
            return None;
        }
        let inverse = Vec3::new(
            reciprocal_or_infinity(direction.x),
            reciprocal_or_infinity(direction.y),
            reciprocal_or_infinity(direction.z),
        );
        let mut stack = vec![0_usize];
        let mut closest = None;
        let mut closest_distance = f32::INFINITY;
        while let Some(node_index) = stack.pop() {
            let node = self.nodes[node_index];
            if ray_aabb_distance(ray.origin, inverse, node.minimum, node.maximum)
                .is_none_or(|distance| distance > closest_distance)
            {
                continue;
            }
            if let (Some(left), Some(right)) = (node.left, node.right) {
                stack.push(right);
                stack.push(left);
                continue;
            }
            for &atom_index in &self.atom_indices[node.start..node.start + node.count] {
                if !is_visible(atom_index) {
                    continue;
                }
                let Some(atom) = molecule.atoms.get(atom_index) else {
                    continue;
                };
                let radius = radius(atom_index, atom).max(0.001);
                if let Some(distance) =
                    ray_sphere_distance(ray.origin, direction, atom.position, radius)
                    && distance < closest_distance
                {
                    closest_distance = distance;
                    closest = Some(atom_index);
                }
            }
        }
        closest
    }

    fn build_node(&mut self, molecule: &Molecule, start: usize, count: usize) -> usize {
        let (minimum, maximum) = bounds_for_indices(
            molecule,
            &self.atom_indices[start..start.saturating_add(count)],
        );
        let node_index = self.nodes.len();
        self.nodes.push(BvhNode {
            minimum,
            maximum,
            start,
            count,
            left: None,
            right: None,
        });
        if count <= LEAF_SIZE {
            return node_index;
        }
        let extent = maximum - minimum;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        self.atom_indices[start..start + count].sort_unstable_by(|left, right| {
            molecule.atoms[*left].position[axis].total_cmp(&molecule.atoms[*right].position[axis])
        });
        let left_count = count / 2;
        let left = self.build_node(molecule, start, left_count);
        let right = self.build_node(molecule, start + left_count, count - left_count);
        self.nodes[node_index].left = Some(left);
        self.nodes[node_index].right = Some(right);
        node_index
    }
}

fn reciprocal_or_infinity(value: f32) -> f32 {
    if value.abs() <= f32::EPSILON {
        f32::INFINITY.copysign(value)
    } else {
        value.recip()
    }
}

fn bounds_for_indices(molecule: &Molecule, indices: &[usize]) -> (Vec3, Vec3) {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for &index in indices {
        let atom = &molecule.atoms[index];
        let radius = atom.element.van_der_waals_radius() * 1.2;
        minimum = minimum.min(atom.position - Vec3::splat(radius));
        maximum = maximum.max(atom.position + Vec3::splat(radius));
    }
    (minimum, maximum)
}

fn ray_aabb_distance(origin: Vec3, inverse: Vec3, minimum: Vec3, maximum: Vec3) -> Option<f32> {
    let first = (minimum - origin) * inverse;
    let second = (maximum - origin) * inverse;
    let near = first.min(second).max_element();
    let far = first.max(second).min_element();
    (far >= near.max(0.0)).then_some(near.max(0.0))
}

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
            alt_loc: None,
            formal_charge: 0,
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
            info: Default::default(),
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
            info: Default::default(),
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
            info: Default::default(),
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

    #[test]
    fn bvh_matches_linear_picker() {
        let molecule = Molecule {
            atoms: (0..100)
                .map(|index| atom(Vec3::new(index as f32 * 0.2 - 10.0, 0.0, -4.0)))
                .collect(),
            bonds: Vec::new(),
            info: Default::default(),
        };
        let ray = Ray {
            origin: Vec3::ZERO,
            direction: -Vec3::Z,
        };
        let linear = pick_atom(&molecule, ray);
        let bvh = AtomBvh::build(&molecule);
        let accelerated = bvh.pick_with_radius(
            &molecule,
            ray,
            |_| true,
            |_, atom| (atom.element.van_der_waals_radius() * 0.32).max(0.32),
        );
        assert_eq!(accelerated, linear);
    }
}
