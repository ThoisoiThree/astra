use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn cylinder(segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for segment in 0..=segments {
        let angle = segment as f32 / segments as f32 * std::f32::consts::TAU;
        let normal = [angle.cos(), 0.0, angle.sin()];
        vertices.push(Vertex {
            position: [normal[0], -0.5, normal[2]],
            normal,
        });
        vertices.push(Vertex {
            position: [normal[0], 0.5, normal[2]],
            normal,
        });
    }
    for segment in 0..segments {
        let bottom = segment * 2;
        indices.extend_from_slice(&[
            bottom,
            bottom + 1,
            bottom + 2,
            bottom + 2,
            bottom + 1,
            bottom + 3,
        ]);
    }
    let bottom_center = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -0.5, 0.0],
        normal: [0.0, -1.0, 0.0],
    });
    let top_center = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, 0.5, 0.0],
        normal: [0.0, 1.0, 0.0],
    });
    let bottom_ring = vertices.len() as u32;
    for segment in 0..segments {
        let angle = segment as f32 / segments as f32 * std::f32::consts::TAU;
        let point = [angle.cos(), angle.sin()];
        vertices.push(Vertex {
            position: [point[0], -0.5, point[1]],
            normal: [0.0, -1.0, 0.0],
        });
        vertices.push(Vertex {
            position: [point[0], 0.5, point[1]],
            normal: [0.0, 1.0, 0.0],
        });
    }
    for segment in 0..segments {
        let next = (segment + 1) % segments;
        indices.extend_from_slice(&[
            bottom_center,
            bottom_ring + next * 2,
            bottom_ring + segment * 2,
            top_center,
            bottom_ring + segment * 2 + 1,
            bottom_ring + next * 2 + 1,
        ]);
    }
    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cylinder_has_flat_top_and_bottom_caps() {
        let mesh = cylinder(8);
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.position == [0.0, -0.5, 0.0] && vertex.normal == [0.0, -1.0, 0.0]
        }));
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.position == [0.0, 0.5, 0.0] && vertex.normal == [0.0, 1.0, 0.0]
        }));
        assert_eq!(mesh.indices.len(), 8 * 12);
    }
}
