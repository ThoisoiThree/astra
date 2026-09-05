use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    SrgbColor,
    measurement::MeasurementLine,
    molecule::{Molecule, MoleculeHierarchy},
};

use super::mesh;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceRaw {
    pub(super) model: [[f32; 4]; 4],
    pub(super) color: [f32; 4],
    pub(super) highlight: [f32; 4],
    pub(super) semantic_ids: [u32; 4],
}

impl InstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4,
        8 => Uint32x4
    ];

    pub(super) fn new(model: Mat4, color: [f32; 4], highlighted: bool) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            color: SrgbColor(color).to_linear().0,
            highlight: [f32::from(highlighted), 0.0, 0.0, 0.0],
            semantic_ids: [0; 4],
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceTopologyRaw {
    model: [[f32; 4]; 4],
    semantic_ids: [u32; 4],
}

impl InstanceTopologyRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
        8 => Uint32x4
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
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceDisplayRaw {
    color: [f32; 4],
    highlight: [f32; 4],
}

impl InstanceDisplayRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![6 => Float32x4, 7 => Float32x4];

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CartoonVertex {
    pub(super) position: [f32; 3],
    pub(super) normal: [f32; 3],
    pub(super) semantic_ids: [u32; 4],
}

impl CartoonVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Uint32x4
    ];

    pub(super) fn new(
        position: Vec3,
        normal: Vec3,
        _color: [f32; 4],
        _highlighted: bool,
        atom_index: usize,
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            semantic_ids: [(atom_index as u32).saturating_add(1), 0, 0, 1],
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
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CartoonDisplayRaw {
    color: [f32; 4],
    highlight: [f32; 4],
}

impl CartoonDisplayRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![3 => Float32x4, 4 => Float32x4];

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ToonTopologyRaw {
    center_radius: [f32; 4],
    semantic_ids: [u32; 4],
}

impl ToonTopologyRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x4, 2 => Uint32x4];

    pub(super) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct ToonDisplayRaw {
    color: [f32; 4],
    highlight: [f32; 4],
}

impl ToonDisplayRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![1 => Float32x4, 3 => Float32x4];

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

    pub(super) fn write<T: Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[T],
    ) {
        let bytes = bytemuck::cast_slice(values);
        let required = bytes.len() as u64;
        if required > self.capacity_bytes {
            self.capacity_bytes = required.next_power_of_two().max(Self::MIN_CAPACITY);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.capacity_bytes,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        self.len_bytes = required;
    }

    pub(super) fn estimated_bytes(&self) -> u64 {
        self.capacity_bytes
    }
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

pub(super) fn measurement_instances(lines: &[MeasurementLine]) -> Vec<InstanceRaw> {
    let mut instances = Vec::new();
    for line in lines.iter().filter(|line| line.is_visible()) {
        let vector = line.second.position - line.first.position;
        let length = vector.length();
        if length <= f32::EPSILON {
            continue;
        }
        let direction = vector / length;
        let rotation = Quat::from_rotation_arc(Vec3::Y, direction);
        let half_label_gap = (length * 0.1).clamp(0.18, 0.45).min(length * 0.35);
        let middle = length * 0.5;
        append_measurement_dashes(
            &mut instances,
            line.first.position,
            direction,
            rotation,
            0.0,
            middle - half_label_gap,
            line.effective_color(),
            line.thickness,
        );
        append_measurement_dashes(
            &mut instances,
            line.first.position,
            direction,
            rotation,
            middle + half_label_gap,
            length,
            line.effective_color(),
            line.thickness,
        );
    }
    instances
}

#[allow(clippy::too_many_arguments)]
fn append_measurement_dashes(
    instances: &mut Vec<InstanceRaw>,
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
            instances.push(InstanceRaw::new(model, color, false));
        }
        cursor += DASH_LENGTH + DASH_GAP;
    }
}

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

pub(super) struct DisplayTopology {
    pub(super) atom_topology: Vec<InstanceTopologyRaw>,
    pub(super) bond_topology: Vec<InstanceTopologyRaw>,
    pub(super) toon_topology: Vec<ToonTopologyRaw>,
}

pub(super) struct DisplayAttributes {
    pub(super) atom_display: Vec<InstanceDisplayRaw>,
    pub(super) bond_display: Vec<InstanceDisplayRaw>,
    pub(super) toon_display: Vec<ToonDisplayRaw>,
}

fn standard_atom_visible(
    index: usize,
    display: &crate::DisplayState,
    standard_atomic: &[bool],
) -> bool {
    display.visible[index]
        && standard_atomic.get(index).copied().unwrap_or(false)
        && display.representations[index].contains(crate::RepresentationMask::SPHERES)
}

fn standard_bond_visible(
    bond: &crate::molecule::Bond,
    display: &crate::DisplayState,
    standard_atomic: &[bool],
) -> bool {
    display.visible[bond.a]
        && display.visible[bond.b]
        && standard_atomic.get(bond.a).copied().unwrap_or(false)
        && standard_atomic.get(bond.b).copied().unwrap_or(false)
        && display.representations[bond.a].contains(crate::RepresentationMask::STICKS)
        && display.representations[bond.b].contains(crate::RepresentationMask::STICKS)
}

fn toon_atom_visible(index: usize, display: &crate::DisplayState) -> bool {
    display.visible[index]
        && display.modes.get(index) == Some(&crate::DisplayMode::Toon)
        && display.representations[index].contains(crate::RepresentationMask::SPHERES)
}

pub(super) fn display_topology(
    molecule: &Molecule,
    display: &crate::DisplayState,
    standard_atomic: &[bool],
    semantic_ids: &[[u32; 4]],
) -> DisplayTopology {
    let atom_topology = molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, _)| standard_atom_visible(*index, display, standard_atomic))
        .map(|(index, atom)| {
            let radius = atom.element.van_der_waals_radius() * 0.28;
            InstanceTopologyRaw {
                model: Mat4::from_scale_rotation_translation(
                    Vec3::splat(radius),
                    Quat::IDENTITY,
                    atom.position,
                )
                .to_cols_array_2d(),
                semantic_ids: semantic_ids.get(index).copied().unwrap_or([0; 4]),
            }
        })
        .collect();
    let bond_topology = molecule
        .bonds
        .iter()
        .filter(|bond| standard_bond_visible(bond, display, standard_atomic))
        .filter_map(|bond| {
            let start = molecule.atoms[bond.a].position;
            let end = molecule.atoms[bond.b].position;
            let vector = end - start;
            let length = vector.length();
            if length <= f32::EPSILON {
                return None;
            }
            let rotation = Quat::from_rotation_arc(Vec3::Y, vector / length);
            let model = Mat4::from_scale_rotation_translation(
                Vec3::new(0.11, length, 0.11),
                rotation,
                (start + end) * 0.5,
            );
            Some(InstanceTopologyRaw {
                model: model.to_cols_array_2d(),
                semantic_ids: [0; 4],
            })
        })
        .collect();
    let toon_topology = molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, _)| toon_atom_visible(*index, display))
        .map(|(index, atom)| ToonTopologyRaw {
            center_radius: atom
                .position
                .extend(atom.element.van_der_waals_radius())
                .to_array(),
            semantic_ids: semantic_ids.get(index).copied().unwrap_or([0; 4]),
        })
        .collect();

    DisplayTopology {
        atom_topology,
        bond_topology,
        toon_topology,
    }
}

pub(super) fn display_attributes(
    molecule: &Molecule,
    display: &crate::DisplayState,
    standard_atomic: &[bool],
) -> DisplayAttributes {
    let atom_display = molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, _)| standard_atom_visible(*index, display, standard_atomic))
        .map(|(index, _)| {
            let selected = display.selection[index];
            InstanceDisplayRaw {
                color: SrgbColor(display.colors[index]).to_linear().0,
                highlight: [
                    f32::from(selected),
                    if selected { 0.16 } else { 0.0 },
                    0.0,
                    0.0,
                ],
            }
        })
        .collect();
    let bond_display = molecule
        .bonds
        .iter()
        .filter(|bond| standard_bond_visible(bond, display, standard_atomic))
        .map(|bond| {
            let color_a = display.colors[bond.a];
            let color_b = display.colors[bond.b];
            InstanceDisplayRaw {
                color: SrgbColor([
                    (color_a[0] + color_b[0]) * 0.5,
                    (color_a[1] + color_b[1]) * 0.5,
                    (color_a[2] + color_b[2]) * 0.5,
                    1.0,
                ])
                .to_linear()
                .0,
                highlight: [
                    f32::from(display.selection[bond.a] || display.selection[bond.b]),
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        })
        .collect();
    let toon_display = molecule
        .atoms
        .iter()
        .enumerate()
        .filter(|(index, _)| toon_atom_visible(*index, display))
        .map(|(index, _)| ToonDisplayRaw {
            color: SrgbColor(display.colors[index]).to_linear().0,
            highlight: [f32::from(display.selection[index]), 0.0, 0.0, 0.0],
        })
        .collect();

    DisplayAttributes {
        atom_display,
        bond_display,
        toon_display,
    }
}

pub(super) fn cartoon_display_attributes(
    vertices: &[CartoonVertex],
    display: &crate::DisplayState,
) -> Vec<CartoonDisplayRaw> {
    let mut attributes = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let Some(atom_index) = vertex.semantic_ids[0]
            .checked_sub(1)
            .map(|index| index as usize)
        else {
            attributes.push(CartoonDisplayRaw {
                color: [0.0; 4],
                highlight: [0.0; 4],
            });
            continue;
        };
        let color = display
            .colors
            .get(atom_index)
            .copied()
            .map_or([0.0; 4], |color| SrgbColor(color).to_linear().0);
        attributes.push(CartoonDisplayRaw {
            color,
            highlight: [
                f32::from(display.selection.get(atom_index).unwrap_or(false)),
                0.0,
                0.0,
                0.0,
            ],
        });
    }
    attributes
}
