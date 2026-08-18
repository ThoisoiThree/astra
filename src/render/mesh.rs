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

pub fn uv_sphere(latitude_segments: u32, longitude_segments: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for latitude in 0..=latitude_segments {
        let v = latitude as f32 / latitude_segments as f32;
        let phi = v * std::f32::consts::PI;
        for longitude in 0..=longitude_segments {
            let u = longitude as f32 / longitude_segments as f32;
            let theta = u * std::f32::consts::TAU;
            let normal = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            vertices.push(Vertex {
                position: normal,
                normal,
            });
        }
    }
    let row = longitude_segments + 1;
    for latitude in 0..latitude_segments {
        for longitude in 0..longitude_segments {
            let top_left = latitude * row + longitude;
            let bottom_left = (latitude + 1) * row + longitude;
            indices.extend_from_slice(&[
                top_left,
                bottom_left,
                top_left + 1,
                top_left + 1,
                bottom_left,
                bottom_left + 1,
            ]);
        }
    }
    Mesh { vertices, indices }
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
    Mesh { vertices, indices }
}
