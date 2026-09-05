use glam::Vec3;

use crate::{
    DisplayMode, DisplayState,
    molecule::{
        Molecule, MoleculeHierarchy, ResidueGroup, SecondaryStructure, assign_secondary_structure,
    },
};

use super::instances::{CartoonVertex, semantic_ids_from_hierarchy};

#[derive(Clone, Copy)]

struct BackboneAnchor {
    atom_index: usize,
    residue_index: usize,
    position: Vec3,
    guide: Vec3,
    nucleic: bool,
    structure: SecondaryStructure,
}

#[derive(Default)]
pub(super) struct CartoonRenderData {
    pub(super) vertices: Vec<CartoonVertex>,
    pub(super) indices: Vec<u32>,
    pub(super) standard_atomic: Vec<bool>,
    pub(super) semantic_ids: Vec<[u32; 4]>,
}

pub(super) fn cartoon_render_data(
    molecule: &Molecule,
    display: &DisplayState,
) -> CartoonRenderData {
    let hierarchy = MoleculeHierarchy::from_molecule(molecule);
    let assignments = assign_secondary_structure(molecule, &hierarchy);
    cartoon_render_data_cached(molecule, display, &hierarchy, &assignments)
}

pub(super) fn cartoon_render_data_cached(
    molecule: &Molecule,
    display: &DisplayState,
    hierarchy: &MoleculeHierarchy,
    assignments: &[Vec<SecondaryStructure>],
) -> CartoonRenderData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let semantic_ids = semantic_ids_from_hierarchy(molecule.atoms.len(), hierarchy);
    let quality = display.ambient_occlusion.quality;
    let samples_per_residue = quality.cartoon_samples_per_residue();
    let width_segments = quality.cartoon_width_segments();
    let ring_segments = width_segments.max(6) * 2;
    let mut cartoon_residue_atoms = vec![false; molecule.atoms.len()];
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let mut anchors = Vec::new();
        for (residue_index, residue) in chain.residues.iter().enumerate() {
            let Some((atom_index, nucleic)) = backbone_atom(molecule, residue) else {
                continue;
            };
            for &index in &residue.atom_indices {
                if let Some(value) = cartoon_residue_atoms.get_mut(index) {
                    *value = true;
                }
            }
            let position = molecule.atoms[atom_index].position;
            let guide = residue
                .atom_indices
                .iter()
                .filter_map(|index| molecule.atoms.get(*index))
                .find(|atom| {
                    if nucleic {
                        matches!(atom.name.as_str(), "C4'" | "C4*")
                    } else {
                        atom.name == "O"
                    }
                })
                .map_or(Vec3::ZERO, |atom| atom.position - position);
            anchors.push(BackboneAnchor {
                atom_index,
                residue_index,
                position,
                guide,
                nucleic,
                structure: assignments
                    .get(chain_index)
                    .and_then(|chain| chain.get(residue_index))
                    .copied()
                    .unwrap_or(if nucleic {
                        SecondaryStructure::Nucleic
                    } else {
                        SecondaryStructure::Coil
                    }),
            });
        }

        let mut run = Vec::new();
        for anchor in anchors {
            let drawable = display.visible.get(anchor.atom_index).unwrap_or(false)
                && display.modes.get(anchor.atom_index) == Some(&DisplayMode::Cartoon);
            let continuous = run.last().is_none_or(|previous: &BackboneAnchor| {
                anchor.residue_index == previous.residue_index + 1
                    && anchor.nucleic == previous.nucleic
                    && anchor.position.distance(previous.position)
                        <= if anchor.nucleic { 8.5 } else { 5.0 }
            });
            if !drawable || !continuous {
                append_ribbon_run(
                    &run,
                    display,
                    samples_per_residue,
                    width_segments,
                    ring_segments,
                    &mut vertices,
                    &mut indices,
                );
                run.clear();
            }
            if drawable {
                run.push(anchor);
            }
        }
        append_ribbon_run(
            &run,
            display,
            samples_per_residue,
            width_segments,
            ring_segments,
            &mut vertices,
            &mut indices,
        );
    }

    let standard_atomic = display
        .modes
        .iter()
        .enumerate()
        .map(|(index, mode)| match mode {
            DisplayMode::BallAndStick => true,
            DisplayMode::Cartoon => !cartoon_residue_atoms[index],
            DisplayMode::Toon => false,
        })
        .collect();
    CartoonRenderData {
        vertices,
        indices,
        standard_atomic,
        semantic_ids,
    }
}

fn backbone_atom(molecule: &Molecule, residue: &ResidueGroup) -> Option<(usize, bool)> {
    residue
        .atom_indices
        .iter()
        .copied()
        .find(|index| {
            molecule
                .atoms
                .get(*index)
                .is_some_and(|atom| !atom.hetero && atom.name == "CA")
        })
        .map(|index| (index, false))
        .or_else(|| {
            residue
                .atom_indices
                .iter()
                .copied()
                .find(|index| {
                    molecule
                        .atoms
                        .get(*index)
                        .is_some_and(|atom| !atom.hetero && atom.name == "P")
                })
                .map(|index| (index, true))
        })
}

fn append_ribbon_run(
    run: &[BackboneAnchor],
    display: &DisplayState,
    samples_per_residue: usize,
    width_segments: u32,
    ring_segments: u32,
    vertices: &mut Vec<CartoonVertex>,
    indices: &mut Vec<u32>,
) {
    if run.len() < 2 {
        return;
    }
    let mut guides: Vec<Vec3> = run
        .iter()
        .map(|anchor| anchor.guide.normalize_or_zero())
        .collect();
    for index in 0..guides.len() {
        if guides[index] == Vec3::ZERO {
            guides[index] = guides
                .get(index.wrapping_sub(1))
                .copied()
                .filter(|guide| *guide != Vec3::ZERO)
                .unwrap_or(Vec3::X);
        }
        if index > 0 && guides[index].dot(guides[index - 1]) < 0.0 {
            guides[index] = -guides[index];
        }
    }

    let sample_count = (run.len() - 1) * samples_per_residue + 1;
    let mut positions = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let segment = (index / samples_per_residue).min(run.len() - 2);
        let t = if index + 1 == sample_count {
            1.0
        } else {
            (index % samples_per_residue) as f32 / samples_per_residue as f32
        };
        positions.push(centripetal_catmull_rom(run, segment, t));
    }

    let mut sides = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let before = positions[index.saturating_sub(1)];
        let after = positions[(index + 1).min(sample_count - 1)];
        let tangent = (after - before).normalize_or_zero();
        let segment = (index / samples_per_residue).min(run.len() - 2);
        let t = if index + 1 == sample_count {
            1.0
        } else {
            (index % samples_per_residue) as f32 / samples_per_residue as f32
        };
        let guide = guides[segment].lerp(guides[segment + 1], t);
        let target = project_side(guide, tangent);
        let side = if let Some(previous) = sides.last().copied() {
            let transported = project_side(previous, tangent);
            if target == Vec3::ZERO {
                transported
            } else if transported == Vec3::ZERO {
                target
            } else {
                let target = if target.dot(transported) < 0.0 {
                    -target
                } else {
                    target
                };
                transported.lerp(target, 0.18).normalize_or_zero()
            }
        } else if target != Vec3::ZERO {
            target
        } else {
            fallback_side(tangent)
        };
        sides.push(if side == Vec3::ZERO {
            fallback_side(tangent)
        } else {
            side
        });
    }

    let mut samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let before = positions[index.saturating_sub(1)];
        let after = positions[(index + 1).min(sample_count - 1)];
        let tangent = (after - before).normalize_or_zero();
        let mut side = project_side(sides[index], tangent);
        if side == Vec3::ZERO {
            side = fallback_side(tangent);
        }
        let normal = side.cross(tangent).normalize_or_zero();
        side = tangent.cross(normal).normalize_or_zero();
        let segment = (index / samples_per_residue).min(run.len() - 2);
        let t = if index + 1 == sample_count {
            1.0
        } else {
            (index % samples_per_residue) as f32 / samples_per_residue as f32
        };
        let structure = if t < 0.5 {
            run[segment].structure
        } else {
            run[segment + 1].structure
        };
        let (mut width, thickness) = cartoon_profile(run[segment].structure)
            .lerp(cartoon_profile(run[segment + 1].structure), smoothstep(t));
        if structure == SecondaryStructure::Strand {
            width *= strand_arrow_scale(run, segment, t);
        }
        let left = display.colors[run[segment].atom_index];
        let right = display.colors[run[segment + 1].atom_index];
        let color = [
            left[0] + (right[0] - left[0]) * t,
            left[1] + (right[1] - left[1]) * t,
            left[2] + (right[2] - left[2]) * t,
            1.0,
        ];
        let selected = display.selection[run[segment].atom_index]
            || display.selection[run[segment + 1].atom_index];
        let atom_index = if t < 0.5 {
            run[segment].atom_index
        } else {
            run[segment + 1].atom_index
        };
        samples.push(CartoonSample {
            atom_index,
            position: positions[index],
            tangent,
            side,
            normal,
            width,
            thickness,
            color,
            selected,
            tube: matches!(
                structure,
                SecondaryStructure::Coil | SecondaryStructure::Turn
            ),
        });
    }

    let mut first_edge = 0;
    while first_edge + 1 < samples.len() {
        let tube = samples[first_edge].tube && samples[first_edge + 1].tube;
        let mut last_edge = first_edge;
        while last_edge + 2 < samples.len()
            && (samples[last_edge + 1].tube && samples[last_edge + 2].tube) == tube
        {
            last_edge += 1;
        }
        let strip = &samples[first_edge..=last_edge + 1];
        if tube {
            append_tube_strip(strip, ring_segments, vertices, indices);
        } else {
            append_rectangular_strip(strip, width_segments, vertices, indices);
        }
        first_edge = last_edge + 1;
    }
}

#[derive(Clone, Copy)]
struct CartoonSample {
    atom_index: usize,
    position: Vec3,
    tangent: Vec3,
    side: Vec3,
    normal: Vec3,
    width: f32,
    thickness: f32,
    color: [f32; 4],
    selected: bool,
    tube: bool,
}

fn append_rectangular_strip(
    samples: &[CartoonSample],
    width_segments: u32,
    vertices: &mut Vec<CartoonVertex>,
    indices: &mut Vec<u32>,
) {
    // A single quad across the full ribbon width becomes strongly non-planar on tight
    // bends. Its implicit diagonal then leaks into both diffuse and specular lighting.
    // A small transverse grid follows the ruled surface closely without changing its
    // silhouette, while analytical surface normals keep the bend visually continuous.
    let face_row = width_segments + 1;
    let vertices_per_ring = face_row * 2 + 4;
    let base = vertices.len() as u32;
    for (ring, sample) in samples.iter().copied().enumerate() {
        for top in [true, false] {
            for width_index in 0..=width_segments {
                let across = width_index as f32 / width_segments as f32 - 0.5;
                let position = ribbon_surface_position(sample, across, top);
                let before = ribbon_surface_position(samples[ring.saturating_sub(1)], across, top);
                let after = ribbon_surface_position(
                    samples[(ring + 1).min(samples.len() - 1)],
                    across,
                    top,
                );
                let longitudinal = (after - before).normalize_or_zero();
                let expected = if top { sample.normal } else { -sample.normal };
                let candidate = sample.side.cross(longitudinal).normalize_or_zero();
                let mut surface_normal = if top { candidate } else { -candidate };
                if surface_normal == Vec3::ZERO {
                    surface_normal = expected;
                } else if surface_normal.dot(expected) < 0.0 {
                    surface_normal = -surface_normal;
                }
                vertices.push(CartoonVertex::new(
                    position,
                    surface_normal,
                    sample.color,
                    sample.selected,
                    sample.atom_index,
                ));
            }
        }

        let left_bottom = ribbon_surface_position(sample, -0.5, false);
        let left_top = ribbon_surface_position(sample, -0.5, true);
        let right_top = ribbon_surface_position(sample, 0.5, true);
        let right_bottom = ribbon_surface_position(sample, 0.5, false);
        vertices.extend_from_slice(&[
            CartoonVertex::new(
                left_bottom,
                -sample.side,
                sample.color,
                sample.selected,
                sample.atom_index,
            ),
            CartoonVertex::new(
                left_top,
                -sample.side,
                sample.color,
                sample.selected,
                sample.atom_index,
            ),
            CartoonVertex::new(
                right_top,
                sample.side,
                sample.color,
                sample.selected,
                sample.atom_index,
            ),
            CartoonVertex::new(
                right_bottom,
                sample.side,
                sample.color,
                sample.selected,
                sample.atom_index,
            ),
        ]);
    }
    for ring in 0..samples.len().saturating_sub(1) as u32 {
        let current = base + ring * vertices_per_ring;
        let next = current + vertices_per_ring;
        for width_index in 0..width_segments {
            let a = current + width_index;
            let b = a + 1;
            let next_a = next + width_index;
            let next_b = next_a + 1;
            indices.extend_from_slice(&[a, b, next_a, next_a, b, next_b]);

            let a = current + face_row + width_index;
            let b = a + 1;
            let next_a = next + face_row + width_index;
            let next_b = next_a + 1;
            indices.extend_from_slice(&[a, next_a, b, next_a, next_b, b]);
        }

        let left = current + face_row * 2;
        let next_left = next + face_row * 2;
        indices.extend_from_slice(&[
            left,
            left + 1,
            next_left,
            next_left,
            left + 1,
            next_left + 1,
        ]);
        let right = left + 2;
        let next_right = next_left + 2;
        indices.extend_from_slice(&[
            right,
            right + 1,
            next_right,
            next_right,
            right + 1,
            next_right + 1,
        ]);
    }
    append_rectangular_cap(samples[0], -samples[0].tangent, vertices, indices);
    let last = samples[samples.len() - 1];
    append_rectangular_cap(last, last.tangent, vertices, indices);
}

fn ribbon_surface_position(sample: CartoonSample, across: f32, top: bool) -> Vec3 {
    let height = if top { 0.5 } else { -0.5 };
    sample.position
        + sample.side * (sample.width * across)
        + sample.normal * (sample.thickness * height)
}

fn append_rectangular_cap(
    sample: CartoonSample,
    cap_normal: Vec3,
    vertices: &mut Vec<CartoonVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    let half_side = sample.side * sample.width * 0.5;
    let half_normal = sample.normal * sample.thickness * 0.5;
    for position in [
        sample.position - half_side - half_normal,
        sample.position + half_side - half_normal,
        sample.position + half_side + half_normal,
        sample.position - half_side + half_normal,
    ] {
        vertices.push(CartoonVertex::new(
            position,
            cap_normal,
            sample.color,
            sample.selected,
            sample.atom_index,
        ));
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn append_tube_strip(
    samples: &[CartoonSample],
    ring_segments: u32,
    vertices: &mut Vec<CartoonVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;
    for sample in samples {
        let radius = sample.width * 0.5;
        for segment in 0..ring_segments {
            let angle = segment as f32 / ring_segments as f32 * std::f32::consts::TAU;
            let radial = (sample.side * angle.cos() + sample.normal * angle.sin()).normalize();
            vertices.push(CartoonVertex::new(
                sample.position + radial * radius,
                radial,
                sample.color,
                sample.selected,
                sample.atom_index,
            ));
        }
    }
    for ring in 0..samples.len().saturating_sub(1) as u32 {
        let current = base + ring * ring_segments;
        let next = current + ring_segments;
        for segment in 0..ring_segments {
            let following = (segment + 1) % ring_segments;
            indices.extend_from_slice(&[
                current + segment,
                current + following,
                next + segment,
                next + segment,
                current + following,
                next + following,
            ]);
        }
    }
}

#[derive(Clone, Copy)]
struct CartoonProfile {
    width: f32,
    thickness: f32,
}

impl CartoonProfile {
    fn lerp(self, other: Self, t: f32) -> (f32, f32) {
        (
            self.width + (other.width - self.width) * t,
            self.thickness + (other.thickness - self.thickness) * t,
        )
    }
}

fn cartoon_profile(structure: SecondaryStructure) -> CartoonProfile {
    match structure {
        SecondaryStructure::Helix => CartoonProfile {
            width: 1.42,
            thickness: 0.30,
        },
        SecondaryStructure::Strand => CartoonProfile {
            width: 1.18,
            thickness: 0.13,
        },
        SecondaryStructure::Nucleic => CartoonProfile {
            width: 1.18,
            thickness: 0.24,
        },
        SecondaryStructure::Turn => CartoonProfile {
            width: 0.19,
            thickness: 0.19,
        },
        SecondaryStructure::Coil => CartoonProfile {
            width: 0.16,
            thickness: 0.16,
        },
    }
}

fn strand_arrow_scale(run: &[BackboneAnchor], segment: usize, t: f32) -> f32 {
    let mut end = segment;
    while end + 1 < run.len() && run[end + 1].structure == SecondaryStructure::Strand {
        end += 1;
    }
    let arrow_start = end.saturating_sub(1) as f32;
    let position = segment as f32 + t;
    if position < arrow_start {
        return 1.0;
    }
    let u = (position - arrow_start).clamp(0.0, 1.0);
    if u < 0.35 {
        1.0 + u / 0.35 * 0.55
    } else {
        1.55 + (0.08 - 1.55) * ((u - 0.35) / 0.65)
    }
}

fn project_side(vector: Vec3, tangent: Vec3) -> Vec3 {
    (vector - tangent * vector.dot(tangent)).normalize_or_zero()
}

fn fallback_side(tangent: Vec3) -> Vec3 {
    let fallback = if tangent.x.abs() < 0.8 {
        Vec3::X
    } else {
        Vec3::Z
    };
    project_side(fallback, tangent)
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn centripetal_catmull_rom(run: &[BackboneAnchor], segment: usize, t: f32) -> Vec3 {
    let p0 = run[segment.saturating_sub(1)].position;
    let p1 = run[segment].position;
    let p2 = run[segment + 1].position;
    let p3 = run[(segment + 2).min(run.len() - 1)].position;
    let d01 = p0.distance(p1).sqrt().max(0.001);
    let d12 = p1.distance(p2).sqrt().max(0.001);
    let d23 = p2.distance(p3).sqrt().max(0.001);
    let tangent1 = (p2 - p0) * (d12 / (d01 + d12));
    let tangent2 = (p3 - p1) * (d12 / (d12 + d23));
    let t2 = t * t;
    let t3 = t2 * t;
    p1 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + tangent1 * (t3 - 2.0 * t2 + t)
        + p2 * (-2.0 * t3 + 3.0 * t2)
        + tangent2 * (t3 - t2)
}
