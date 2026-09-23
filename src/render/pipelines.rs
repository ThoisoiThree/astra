use super::{
    instances::{AnnotationInstance, BondInstance, MeshVertex, SphereInstance},
    mesh::Vertex,
    renderer::{DEPTH_FORMAT, SCENE_FORMAT, SEMANTIC_FORMAT},
};

pub(super) fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    fragment_entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Measurement annotations drawn over the composed image.
pub(super) fn create_annotation_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let vertex_layouts = [Some(Vertex::layout()), Some(AnnotationInstance::layout())];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("annotation pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex_annotation"),
            compilation_options: Default::default(),
            buffers: &vertex_layouts,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // Reusable meshes have different seam winding at poles and end caps.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fragment_annotation"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Which targets a scene pipeline writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SceneTarget {
    /// Color and semantic IDs for picking, AO and Toon contours.
    Scene,
    /// Color only: the first depth-of-field layer.
    Color,
    /// Color only, rejecting fragments already captured by the previous layer.
    Peel,
}

impl SceneTarget {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Scene => "scene",
            Self::Color => "color",
            Self::Peel => "peel",
        }
    }
}

/// Every pipeline that draws molecular geometry for one kind of target.
pub(super) struct ScenePipelines {
    pub(super) spheres: wgpu::RenderPipeline,
    pub(super) bonds: wgpu::RenderPipeline,
    pub(super) meshes: wgpu::RenderPipeline,
}

impl ScenePipelines {
    pub(super) fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        impostor_shader: &wgpu::ShaderModule,
        mesh_shader: &wgpu::ShaderModule,
        target: SceneTarget,
    ) -> Self {
        let sphere_layouts = [Some(SphereInstance::layout())];
        let bond_layouts = [Some(BondInstance::layout())];
        let mesh_layouts = [Some(MeshVertex::layout())];
        Self {
            spheres: scene_pipeline(
                device,
                layout,
                impostor_shader,
                (
                    "vertex_sphere",
                    &format!("fragment_sphere_{}", target.suffix()),
                ),
                &sphere_layouts,
                None,
                target,
            ),
            // Capsules rasterize the back faces of their bounding box, which covers the
            // silhouette from outside and still works when the camera is inside the box.
            bonds: scene_pipeline(
                device,
                layout,
                impostor_shader,
                ("vertex_bond", &format!("fragment_bond_{}", target.suffix())),
                &bond_layouts,
                Some(wgpu::Face::Front),
                target,
            ),
            meshes: scene_pipeline(
                device,
                layout,
                mesh_shader,
                ("vertex_mesh", &format!("fragment_mesh_{}", target.suffix())),
                &mesh_layouts,
                None,
                target,
            ),
        }
    }
}

fn scene_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    (vertex_entry, fragment_entry): (&str, &str),
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    cull_mode: Option<wgpu::Face>,
    target: SceneTarget,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment_entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                (target == SceneTarget::Scene).then_some(wgpu::ColorTargetState {
                    format: SEMANTIC_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}
