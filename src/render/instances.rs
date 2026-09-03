use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::{
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
            color,
            highlight: [f32::from(highlighted), 0.0, 0.0, 0.0],
            semantic_ids: [0; 4],
        }
    }

    pub(super) fn with_semantic_ids(mut self, semantic_ids: [u32; 4]) -> Self {
        self.semantic_ids = semantic_ids;
        self
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
pub(super) struct CartoonVertex {
    pub(super) position: [f32; 3],
    pub(super) normal: [f32; 3],
    pub(super) color: [f32; 4],
    pub(super) highlight: [f32; 4],
    pub(super) semantic_ids: [u32; 4],
}

impl CartoonVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x4,
        4 => Uint32x4
    ];

    pub(super) fn new(
        position: Vec3,
        normal: Vec3,
        color: [f32; 4],
        highlighted: bool,
        atom_index: usize,
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            color,
            highlight: [f32::from(highlighted), 0.0, 0.0, 0.0],
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
pub(super) struct ToonInstanceRaw {
    pub(super) center_radius: [f32; 4],
    pub(super) color: [f32; 4],
    pub(super) semantic_ids: [u32; 4],
    pub(super) highlight: [f32; 4],
}

impl ToonInstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Uint32x4,
        3 => Float32x4
    ];

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

pub(super) fn toon_semantic_ids(molecule: &Molecule) -> Vec<[u32; 4]> {
    let hierarchy = MoleculeHierarchy::from_molecule(molecule);
    let mut result = vec![[0; 4]; molecule.atoms.len()];
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

pub(super) fn instance_buffer(
    device: &wgpu::Device,
    label: &str,
    instances: &[InstanceRaw],
) -> wgpu::Buffer {
    if instances.is_empty() {
        return empty_instance_buffer(device, label);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

pub(super) fn empty_instance_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<InstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

pub(super) fn cartoon_vertex_buffer(
    device: &wgpu::Device,
    vertices: &[CartoonVertex],
) -> wgpu::Buffer {
    if vertices.is_empty() {
        return empty_cartoon_vertex_buffer(device);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("continuous cartoon vertices"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

pub(super) fn empty_cartoon_vertex_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("continuous cartoon vertices"),
        size: std::mem::size_of::<CartoonVertex>() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

pub(super) fn cartoon_index_buffer(device: &wgpu::Device, indices: &[u32]) -> wgpu::Buffer {
    if indices.is_empty() {
        return empty_cartoon_index_buffer(device);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("continuous cartoon indices"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

pub(super) fn empty_cartoon_index_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("continuous cartoon indices"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    })
}

pub(super) fn toon_instance_buffer(
    device: &wgpu::Device,
    instances: &[ToonInstanceRaw],
) -> wgpu::Buffer {
    if instances.is_empty() {
        return empty_toon_instance_buffer(device);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("toon sphere instances"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

pub(super) fn empty_toon_instance_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("toon sphere instances"),
        size: std::mem::size_of::<ToonInstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}
