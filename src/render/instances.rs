//! CPU-side construction of GPU instance and per-atom buffers.
//!
//! Instances reference atoms by index; positions, colors and selection live in per-atom
//! storage buffers. A trajectory frame therefore only rewrites positions.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    DisplayMode, DisplayState, RepresentationMask, SrgbColor,
    measurement::MeasurementLine,
    molecule::{BondKind, BondOrder, Molecule, MoleculeHierarchy},
};

use super::mesh;

/// Ball-and-stick sphere radius as a fraction of the van der Waals radius.
const BALL_RADIUS_SCALE: f32 = 0.28;
const STICK_RADIUS: f32 = 0.11;
const LICORICE_RADIUS: f32 = 0.2;
const COORDINATION_RADIUS: f32 = 0.055;
const NO_REFERENCE: u32 = u32::MAX;
pub(super) const STYLE_TOON: u32 = 1;
const BOND_DASHED: u32 = 1;

/// A triangle-mesh vertex (ribbons, bases, surfaces) colored through its atom.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(super) struct MeshVertex {
    pub(super) position: [f32; 3],
    pub(super) normal: [f32; 3],
    pub(super) atom: u32,
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Uint32
    ];

    pub(super) fn new(position: Vec3, normal: Vec3, atom: usize) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            atom: atom as u32,
        }
    }

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(super) struct SphereInstance {
    pub(super) atom: u32,
    pub(super) radius: f32,
    pub(super) style: u32,
}

impl SphereInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Uint32,
        1 => Float32,
        2 => Uint32
    ];

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(super) struct BondInstance {
    pub(super) atoms: [u32; 2],
    /// Neighboring atom that orients multiple-bond offsets, or `u32::MAX`.
    pub(super) reference: u32,
    pub(super) style: u32,
    pub(super) radius: f32,
    /// Signed offset toward the reference atom, in Å.
    pub(super) offset: f32,
}

impl BondInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Uint32x2,
        1 => Uint32,
        2 => Uint32,
        3 => Float32,
        4 => Float32
    ];

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// A transformed mesh instance for annotations such as measurement dashes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct AnnotationInstance {
    pub(super) model: [[f32; 4]; 4],
    pub(super) color: [f32; 4],
}

impl AnnotationInstance {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
        6 => Float32x4
    ];

    fn new(model: Mat4, color: [f32; 4]) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            color: SrgbColor(color).to_linear().0,
        }
    }

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub(super) struct GpuMesh {
    pub(super) vertices: wgpu::Buffer,
    pub(super) indices: wgpu::Buffer,
    pub(super) index_count: u32,
    pub(super) estimated_bytes: u64,
}

impl GpuMesh {
    pub(super) fn new(device: &wgpu::Device, label: &str, mesh: mesh::Mesh) -> Self {
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} vertices")),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} indices")),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            estimated_bytes: (mesh.vertices.len() * std::mem::size_of::<mesh::Vertex>()
                + mesh.indices.len() * std::mem::size_of::<u32>())
                as u64,
        }
    }
}

/// Grow-only GPU storage for frequently changing geometry and display data.
/// Small edits stay on the same allocation and are uploaded with `queue.write_buffer`.
pub(super) struct ReusableBuffer {
    pub(super) buffer: wgpu::Buffer,
    capacity_bytes: u64,
    len_bytes: u64,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl ReusableBuffer {
    const MIN_CAPACITY: u64 = 256;

    pub(super) fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let capacity_bytes = Self::MIN_CAPACITY;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity_bytes,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity_bytes,
            len_bytes: 0,
            label,
            usage,
        }
    }

    /// Uploads `values`, growing the allocation when needed. Returns `true` when the
    /// buffer was replaced, which invalidates bind groups that reference it.
    pub(super) fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[T],
    ) -> bool {
        let bytes = bytemuck::cast_slice(values);
        let required = bytes.len() as u64;
        let mut replaced = false;
        if required > self.capacity_bytes {
            self.capacity_bytes = required.next_power_of_two().max(Self::MIN_CAPACITY);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.capacity_bytes,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            replaced = true;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        self.len_bytes = required;
        replaced
    }

    pub(super) fn estimated_bytes(&self) -> u64 {
        self.capacity_bytes
    }
}

pub(super) fn measurement_instances(lines: &[MeasurementLine]) -> Vec<AnnotationInstance> {
    let mut instances = Vec::new();
    for line in lines.iter().filter(|line| line.is_visible()) {
        for (first, second) in line.segments() {
            let vector = second - first;
            let length = vector.length();
            if length <= f32::EPSILON {
                continue;
            }
            let direction = vector / length;
            let rotation = Quat::from_rotation_arc(Vec3::Y, direction);
            let half_label_gap = (length * 0.1).clamp(0.18, 0.45).min(length * 0.35);
            let middle = length * 0.5;
            for (start, end) in [
                (0.0, middle - half_label_gap),
                (middle + half_label_gap, length),
            ] {
                append_measurement_dashes(
                    &mut instances,
                    first,
                    direction,
                    rotation,
                    start,
                    end,
                    line.effective_color(),
                    line.thickness,
                );
            }
        }
    }
    instances
}

#[allow(clippy::too_many_arguments)]
fn append_measurement_dashes(
    instances: &mut Vec<AnnotationInstance>,
    origin: Vec3,
    direction: Vec3,
    rotation: Quat,
    start: f32,
    end: f32,
    color: [f32; 4],
    thickness: f32,
) {
    const DASH_LENGTH: f32 = 0.30;
    const DASH_GAP: f32 = 0.18;
    let mut cursor = start.max(0.0);
    while cursor < end {
        let dash_end = (cursor + DASH_LENGTH).min(end);
        let length = dash_end - cursor;
        if length > 0.025 {
            let midpoint = origin + direction * ((cursor + dash_end) * 0.5);
            let model = Mat4::from_scale_rotation_translation(
                Vec3::new(thickness, length, thickness),
                rotation,
                midpoint,
            );
            instances.push(AnnotationInstance::new(model, color));
        }
        cursor += DASH_LENGTH + DASH_GAP;
    }
}

/// Per-atom `[atom + 1, residue id, chain id, 1]`; residue and chain ids start at 1.
pub(super) fn semantic_ids_from_hierarchy(
    atom_count: usize,
    hierarchy: &MoleculeHierarchy,
) -> Vec<[u32; 4]> {
    let mut result = vec![[0; 4]; atom_count];
    let mut residue_id = 1u32;
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let chain_id = (chain_index as u32).saturating_add(1);
        for residue in &chain.residues {
            for &atom_index in &residue.atom_indices {
                if let Some(ids) = result.get_mut(atom_index) {
                    *ids = [
                        (atom_index as u32).saturating_add(1),
                        residue_id,
                        chain_id,
                        1,
                    ];
                }
            }
            residue_id = residue_id.saturating_add(1);
        }
    }
    result
}

pub(super) fn atom_positions(molecule: &Molecule) -> Vec<[f32; 4]> {
    molecule
        .atoms
        .iter()
        .map(|atom| atom.position.extend(0.0).to_array())
        .collect()
}

pub(super) fn atom_colors(display: &DisplayState) -> Vec<[f32; 4]> {
    display
        .colors
        .iter()
        .map(|color| SrgbColor(*color).to_linear().0)
        .collect()
}

/// `[atom + 1, chain << 20 | residue, flags, 0]`, matching the semantic render target.
pub(super) fn atom_meta(display: &DisplayState, semantic_ids: &[[u32; 4]]) -> Vec<[u32; 4]> {
    (0..display.colors.len())
        .map(|index| {
            let ids = semantic_ids.get(index).copied().unwrap_or([0; 4]);
            let atom_id = if ids[0] == 0 {
                index as u32 + 1
            } else {
                ids[0]
            };
            [
                atom_id,
                (ids[2] << 20) | (ids[1] & 0xF_FFFF),
                u32::from(display.selection.get(index).unwrap_or(false)),
                0,
            ]
        })
        .collect()
}

/// How an atom is drawn with atomic geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AtomicStyle {
    Hidden,
    BallAndStick,
    Licorice,
    Spacefill,
    Toon,
}

impl AtomicStyle {
    const fn stick_radius(self) -> Option<f32> {
        match self {
            Self::BallAndStick => Some(STICK_RADIUS),
            Self::Licorice => Some(LICORICE_RADIUS),
            _ => None,
        }
    }
}

/// `atomic` marks atoms that are not represented by a ribbon.
pub(super) fn atomic_styles(display: &DisplayState, atomic: &[bool]) -> Vec<AtomicStyle> {
    (0..display.modes.len())
        .map(|index| {
            if !display.visible.get(index).unwrap_or(false)
                || !atomic.get(index).copied().unwrap_or(false)
            {
                return AtomicStyle::Hidden;
            }
            match display.modes[index] {
                DisplayMode::Cartoon | DisplayMode::BallAndStick => AtomicStyle::BallAndStick,
                DisplayMode::Licorice => AtomicStyle::Licorice,
                DisplayMode::Spacefill => AtomicStyle::Spacefill,
                DisplayMode::Toon => AtomicStyle::Toon,
            }
        })
        .collect()
}

pub(super) fn sphere_instances(
    molecule: &Molecule,
    display: &DisplayState,
    styles: &[AtomicStyle],
) -> Vec<SphereInstance> {
    let mut instances = Vec::new();
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let representation = display.representations[index];
        let spheres = representation.contains(RepresentationMask::SPHERES);
        let sticks = representation.contains(RepresentationMask::STICKS);
        let vdw = atom.element.van_der_waals_radius();
        let (radius, style) = match styles[index] {
            AtomicStyle::Hidden => continue,
            AtomicStyle::BallAndStick if spheres => (vdw * BALL_RADIUS_SCALE, 0),
            // Joints of uniform sticks, and atoms without bonds, need a sphere.
            AtomicStyle::Licorice if spheres || sticks => (LICORICE_RADIUS, 0),
            AtomicStyle::Spacefill if spheres || sticks => (vdw, 0),
            AtomicStyle::Toon if spheres => (vdw, STYLE_TOON),
            _ => continue,
        };
        instances.push(SphereInstance {
            atom: index as u32,
            radius,
            style,
        });
    }
    instances
}

pub(super) fn bond_instances(
    molecule: &Molecule,
    display: &DisplayState,
    styles: &[AtomicStyle],
) -> Vec<BondInstance> {
    let show_orders = display.bond_orders;
    let mut neighbors: Vec<Vec<(usize, BondOrder)>> = Vec::new();
    if show_orders {
        neighbors = vec![Vec::new(); molecule.atoms.len()];
        for bond in &molecule.bonds {
            neighbors[bond.a].push((bond.b, bond.order));
            neighbors[bond.b].push((bond.a, bond.order));
        }
    }
    let reference = |a: usize, b: usize, aromatic: bool| -> u32 {
        let candidates = |atom: usize, other: usize| {
            neighbors[atom]
                .iter()
                .filter(move |(neighbor, _)| *neighbor != other)
                .filter(|(neighbor, _)| !molecule.atoms[*neighbor].element.is_hydrogen())
        };
        let preferred = |atom: usize, other: usize| {
            candidates(atom, other)
                .find(|(_, order)| !aromatic || *order == BondOrder::Aromatic)
                .or_else(|| candidates(atom, other).next())
                .map(|(neighbor, _)| *neighbor as u32)
        };
        preferred(a, b)
            .or_else(|| preferred(b, a))
            .unwrap_or(NO_REFERENCE)
    };
    let mut instances = Vec::with_capacity(molecule.bonds.len());
    for bond in &molecule.bonds {
        let (a, b) = (bond.a, bond.b);
        let (Some(radius_a), Some(radius_b)) = (styles[a].stick_radius(), styles[b].stick_radius())
        else {
            continue;
        };
        let atoms = [a as u32, b as u32];
        if bond.kind == BondKind::MetalCoordination {
            instances.push(BondInstance {
                atoms,
                reference: NO_REFERENCE,
                style: BOND_DASHED,
                radius: COORDINATION_RADIUS,
                offset: 0.0,
            });
            continue;
        }
        if !display.representations[a].contains(RepresentationMask::STICKS)
            || !display.representations[b].contains(RepresentationMask::STICKS)
        {
            continue;
        }
        let radius = radius_a.min(radius_b);
        let single = |radius: f32, offset: f32, reference: u32, style: u32| BondInstance {
            atoms,
            reference,
            style,
            radius,
            offset,
        };
        match bond.order {
            BondOrder::Double if show_orders => {
                let reference = reference(a, b, false);
                instances.push(single(radius * 0.55, radius, reference, 0));
                instances.push(single(radius * 0.55, -radius, reference, 0));
            }
            BondOrder::Triple if show_orders => {
                let reference = reference(a, b, false);
                instances.push(single(radius * 0.42, 0.0, reference, 0));
                instances.push(single(radius * 0.42, radius * 1.2, reference, 0));
                instances.push(single(radius * 0.42, -radius * 1.2, reference, 0));
            }
            BondOrder::Aromatic if show_orders => {
                instances.push(single(radius, 0.0, NO_REFERENCE, 0));
                let reference = reference(a, b, true);
                if reference != NO_REFERENCE {
                    instances.push(single(radius * 0.4, radius * 2.4, reference, BOND_DASHED));
                }
            }
            _ => instances.push(single(radius, 0.0, NO_REFERENCE, 0)),
        }
    }
    instances
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Bond, Element};

    fn molecule() -> Molecule {
        let atom = |name: &str, element, x: f32| Atom {
            name: name.into(),
            element,
            residue_name: "LIG".into(),
            residue_number: 1,
            chain_id: "A".into(),
            position: Vec3::new(x, 0.0, 0.0),
            occupancy: 1.0,
            hetero: true,
            ..Atom::default()
        };
        Molecule::new(
            vec![
                atom("C1", Element::C, 0.0),
                atom("O1", Element::O, 1.2),
                atom("C2", Element::C, -1.5),
                atom("ZN", Element::Zn, 3.2),
            ],
            vec![
                Bond::with_order(0, 1, BondOrder::Double, BondKind::Covalent).unwrap(),
                Bond::new(0, 2).unwrap(),
                Bond::with_order(1, 3, BondOrder::Single, BondKind::MetalCoordination).unwrap(),
            ],
        )
    }

    #[test]
    fn double_bonds_split_into_offset_sticks_oriented_by_a_neighbor() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_global_mode(DisplayMode::BallAndStick);
        let styles = atomic_styles(&display, &[true; 4]);
        let bonds = bond_instances(&molecule, &display, &styles);
        let double: Vec<_> = bonds.iter().filter(|bond| bond.atoms == [0, 1]).collect();
        assert_eq!(double.len(), 2);
        assert_eq!(double[0].reference, 2);
        assert_eq!(double[0].offset, -double[1].offset);
        let metal = bonds.iter().find(|bond| bond.atoms == [1, 3]).unwrap();
        assert_eq!(metal.style, BOND_DASHED);

        display.bond_orders = false;
        let plain = bond_instances(&molecule, &display, &styles);
        assert_eq!(plain.iter().filter(|bond| bond.atoms == [0, 1]).count(), 1);
    }

    #[test]
    fn mode_radii_follow_the_representation() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        for (mode, radius) in [
            (
                DisplayMode::BallAndStick,
                Element::C.van_der_waals_radius() * BALL_RADIUS_SCALE,
            ),
            (DisplayMode::Licorice, LICORICE_RADIUS),
            (DisplayMode::Spacefill, Element::C.van_der_waals_radius()),
        ] {
            display.set_global_mode(mode);
            let styles = atomic_styles(&display, &[true; 4]);
            let spheres = sphere_instances(&molecule, &display, &styles);
            assert_eq!(spheres[0].radius, radius, "{mode:?}");
        }
        display.set_global_mode(DisplayMode::Spacefill);
        let styles = atomic_styles(&display, &[true; 4]);
        assert!(bond_instances(&molecule, &display, &styles).is_empty());
    }
}
