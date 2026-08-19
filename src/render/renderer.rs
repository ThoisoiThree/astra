use std::{borrow::Cow, sync::Arc};

use bytemuck::{Pod, Zeroable};
use egui::TexturesDelta;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use glam::{Mat4, Quat, Vec3};
use thiserror::Error;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    DisplayMode, DisplayState, RepresentationMask,
    camera::{OrbitCamera, Viewport},
    measurement::MeasurementLine,
    molecule::{Molecule, MoleculeHierarchy, ResidueGroup},
};

use super::mesh::{self, Vertex};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const SEMANTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("could not create a rendering surface: {0}")]
    SurfaceCreation(#[from] wgpu::CreateSurfaceError),
    #[error("no compatible graphics adapter was found: {0}")]
    Adapter(#[from] wgpu::RequestAdapterError),
    #[error("could not create the graphics device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("the rendering surface has no compatible configuration")]
    SurfaceConfiguration,
    #[error("could not acquire the next frame: {0}")]
    Surface(#[from] SurfaceIssue),
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceIssue {
    #[error("surface acquisition timed out")]
    Timeout,
    #[error("window is occluded")]
    Occluded,
    #[error("surface configuration is outdated")]
    Outdated,
    #[error("surface was lost")]
    Lost,
    #[error("surface acquisition failed validation")]
    Validation,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [f32; 16],
    inverse_view_projection: [f32; 16],
    eye_position: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    inverse_view_projection: [f32; 16],
    eye_position: [f32; 4],
    optical_axis: [f32; 4],
    lens: [f32; 4],
    aperture: [f32; 4],
    ao: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    highlight: [f32; 4],
}

impl InstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4,
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4
    ];

    fn new(model: Mat4, color: [f32; 4], highlighted: bool) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            color,
            highlight: [f32::from(highlighted), 0.0, 0.0, 0.0],
        }
    }

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ToonInstanceRaw {
    center_radius: [f32; 4],
    color: [f32; 4],
    semantic_ids: [u32; 4],
    highlight: [f32; 4],
}

impl ToonInstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Uint32x4,
        3 => Float32x4
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn new(device: &wgpu::Device, label: &str, mesh: mesh::Mesh) -> Self {
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

struct DepthTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DepthTarget {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("molecule depth texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

struct ColorTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl ColorTarget {
    fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

struct SemanticTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl SemanticTarget {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("toon semantic IDs"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SEMANTIC_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

struct PostProcess {
    scene: ColorTarget,
    semantic: SemanticTarget,
    ao_raw: ColorTarget,
    ao_filtered: ColorTarget,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    ao_bind_group_layout: wgpu::BindGroupLayout,
    ao_raw_bind_group: wgpu::BindGroup,
    ao_blur_bind_group: wgpu::BindGroup,
    ao_raw_pipeline: wgpu::RenderPipeline,
    ao_blur_pipeline: wgpu::RenderPipeline,
}

impl PostProcess {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
    ) -> Self {
        let scene = ColorTarget::new(device, "molecule scene color", width, height, SCENE_FORMAT);
        let semantic = SemanticTarget::new(device, width, height);
        let ao_raw = ColorTarget::new(device, "raw ambient occlusion", width, height, AO_FORMAT);
        let ao_filtered = ColorTarget::new(
            device,
            "filtered ambient occlusion",
            width,
            height,
            AO_FORMAT,
        );
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DOF postprocess uniform"),
            contents: bytemuck::bytes_of(&PostUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene color sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF postprocess bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = create_post_bind_group(
            device,
            &bind_group_layout,
            &scene.view,
            &sampler,
            depth_view,
            &uniform,
            &ao_filtered.view,
            &semantic.view,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DOF postprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("postprocess.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF postprocess pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DOF postprocess pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let ao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ambient occlusion bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let ao_raw_bind_group = create_ao_bind_group(
            device,
            &ao_bind_group_layout,
            depth_view,
            &uniform,
            &ao_filtered.view,
        );
        let ao_blur_bind_group = create_ao_bind_group(
            device,
            &ao_bind_group_layout,
            depth_view,
            &uniform,
            &ao_raw.view,
        );
        let ao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ambient occlusion shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ambient_occlusion.wgsl"))),
        });
        let ao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ambient occlusion pipeline layout"),
            bind_group_layouts: &[Some(&ao_bind_group_layout)],
            immediate_size: 0,
        });
        let ao_raw_pipeline = create_fullscreen_pipeline(
            device,
            "raw ambient occlusion pipeline",
            &ao_pipeline_layout,
            &ao_shader,
            "raw_ao_main",
            AO_FORMAT,
        );
        let ao_blur_pipeline = create_fullscreen_pipeline(
            device,
            "bilateral ambient occlusion pipeline",
            &ao_pipeline_layout,
            &ao_shader,
            "blur_ao_main",
            AO_FORMAT,
        );
        Self {
            scene,
            semantic,
            ao_raw,
            ao_filtered,
            uniform,
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
            ao_bind_group_layout,
            ao_raw_bind_group,
            ao_blur_bind_group,
            ao_raw_pipeline,
            ao_blur_pipeline,
        }
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
    ) {
        self.scene = ColorTarget::new(device, "molecule scene color", width, height, SCENE_FORMAT);
        self.semantic = SemanticTarget::new(device, width, height);
        self.ao_raw = ColorTarget::new(device, "raw ambient occlusion", width, height, AO_FORMAT);
        self.ao_filtered = ColorTarget::new(
            device,
            "filtered ambient occlusion",
            width,
            height,
            AO_FORMAT,
        );
        self.bind_group = create_post_bind_group(
            device,
            &self.bind_group_layout,
            &self.scene.view,
            &self.sampler,
            depth_view,
            &self.uniform,
            &self.ao_filtered.view,
            &self.semantic.view,
        );
        self.ao_raw_bind_group = create_ao_bind_group(
            device,
            &self.ao_bind_group_layout,
            depth_view,
            &self.uniform,
            &self.ao_filtered.view,
        );
        self.ao_blur_bind_group = create_ao_bind_group(
            device,
            &self.ao_bind_group_layout,
            depth_view,
            &self.uniform,
            &self.ao_raw.view,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn create_post_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    depth_view: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    ao_view: &wgpu::TextureView,
    semantic_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("DOF postprocess bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(ao_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(semantic_view),
            },
        ],
    })
}

fn create_ao_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
    ao_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ambient occlusion bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(ao_view),
            },
        ],
    })
}

fn create_fullscreen_pipeline(
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

fn create_geometry_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    depth_write_enabled: bool,
) -> wgpu::RenderPipeline {
    let vertex_layouts = [Some(Vertex::layout()), Some(InstanceRaw::layout())];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &vertex_layouts,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // Reusable meshes have different seam winding at poles and end caps.
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fragment_main"),
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

fn create_scene_geometry_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let vertex_layouts = [Some(Vertex::layout()), Some(InstanceRaw::layout())];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("molecule scene pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &vertex_layouts,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            front_face: wgpu::FrontFace::Ccw,
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
            entry_point: Some("fragment_scene"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
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

fn create_toon_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let vertex_layouts = [Some(ToonInstanceRaw::layout())];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("analytic toon sphere pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &vertex_layouts,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
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
            entry_point: Some("fragment_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
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

pub struct Renderer {
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    toon_pipeline: wgpu::RenderPipeline,
    annotation_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    sphere: GpuMesh,
    cylinder: GpuMesh,
    ribbon: GpuMesh,
    atom_instances: wgpu::Buffer,
    bond_instances: wgpu::Buffer,
    cartoon_instances: wgpu::Buffer,
    toon_instances: wgpu::Buffer,
    measurement_instances: wgpu::Buffer,
    atom_instance_count: u32,
    bond_instance_count: u32,
    cartoon_instance_count: u32,
    toon_instance_count: u32,
    measurement_instance_count: u32,
    depth: DepthTarget,
    post_process: PostProcess,
    egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self, RenderError> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("molview device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(RenderError::SurfaceConfiguration)?;
        surface.configure(&device, &config);

        let camera_uniform = CameraUniform::zeroed();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molecule shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molecule pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline = create_scene_geometry_pipeline(&device, &pipeline_layout, &shader);
        let toon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("analytic toon sphere shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("toon_sphere.wgsl"))),
        });
        let toon_pipeline = create_toon_pipeline(&device, &pipeline_layout, &toon_shader);
        let annotation_pipeline = create_geometry_pipeline(
            &device,
            "annotation pipeline",
            &pipeline_layout,
            &shader,
            config.format,
            false,
        );

        let sphere = GpuMesh::new(&device, "sphere", mesh::uv_sphere(14, 22));
        let cylinder = GpuMesh::new(&device, "cylinder", mesh::cylinder(16));
        let ribbon = GpuMesh::new(&device, "cartoon ribbon segment", mesh::cube());
        let atom_instances = empty_instance_buffer(&device, "atom instances");
        let bond_instances = empty_instance_buffer(&device, "bond instances");
        let cartoon_instances = empty_instance_buffer(&device, "cartoon instances");
        let toon_instances = empty_toon_instance_buffer(&device);
        let measurement_instances = empty_instance_buffer(&device, "measurement instances");
        let depth = DepthTarget::new(&device, config.width, config.height);
        let post_process = PostProcess::new(
            &device,
            config.format,
            config.width,
            config.height,
            &depth.view,
        );
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, config.format, RendererOptions::default());

        Ok(Self {
            instance,
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            toon_pipeline,
            annotation_pipeline,
            camera_buffer,
            camera_bind_group,
            sphere,
            cylinder,
            ribbon,
            atom_instances,
            bond_instances,
            cartoon_instances,
            toon_instances,
            measurement_instances,
            atom_instance_count: 0,
            bond_instance_count: 0,
            cartoon_instance_count: 0,
            toon_instance_count: 0,
            measurement_instance_count: 0,
            depth,
            post_process,
            egui_renderer,
        })
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth = DepthTarget::new(&self.device, size.width, size.height);
        self.post_process
            .resize(&self.device, size.width, size.height, &self.depth.view);
    }

    pub fn recover_surface(&mut self) -> Result<(), RenderError> {
        self.surface = self.instance.create_surface(self.window.clone())?;
        self.surface.configure(&self.device, &self.config);
        Ok(())
    }

    pub fn update_instances(&mut self, molecule: &Molecule, display: &DisplayState) {
        let (cartoons, standard_atomic) = cartoon_render_data(molecule, display);
        let atoms: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                display.visible[*index]
                    && standard_atomic[*index]
                    && display.representations[*index].contains(RepresentationMask::SPHERES)
            })
            .map(|(index, atom)| {
                let selected = display.selection[index];
                let radius = atom.element.van_der_waals_radius() * 0.28;
                let scale = if selected { radius * 1.16 } else { radius };
                InstanceRaw::new(
                    Mat4::from_scale_rotation_translation(
                        Vec3::splat(scale),
                        Quat::IDENTITY,
                        atom.position,
                    ),
                    display.colors[index],
                    selected,
                )
            })
            .collect();

        let bonds: Vec<_> = molecule
            .bonds
            .iter()
            .filter(|bond| {
                display.visible[bond.a]
                    && display.visible[bond.b]
                    && standard_atomic[bond.a]
                    && standard_atomic[bond.b]
                    && display.representations[bond.a].contains(RepresentationMask::STICKS)
                    && display.representations[bond.b].contains(RepresentationMask::STICKS)
            })
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
                let color_a = display.colors[bond.a];
                let color_b = display.colors[bond.b];
                let color = [
                    (color_a[0] + color_b[0]) * 0.5,
                    (color_a[1] + color_b[1]) * 0.5,
                    (color_a[2] + color_b[2]) * 0.5,
                    1.0,
                ];
                Some(InstanceRaw::new(
                    model,
                    color,
                    display.selection[bond.a] || display.selection[bond.b],
                ))
            })
            .collect();

        let semantic_ids = toon_semantic_ids(molecule);
        let toon_atoms: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                display.visible[*index]
                    && display.modes.get(*index) == Some(&DisplayMode::Toon)
                    && display.representations[*index].contains(RepresentationMask::SPHERES)
            })
            .map(|(index, atom)| {
                let selected = display.selection[index];
                let radius = atom.element.van_der_waals_radius();
                ToonInstanceRaw {
                    center_radius: atom
                        .position
                        .extend(if selected { radius * 1.04 } else { radius })
                        .to_array(),
                    color: display.colors[index],
                    semantic_ids: semantic_ids[index],
                    highlight: [f32::from(selected), 0.0, 0.0, 0.0],
                }
            })
            .collect();

        self.atom_instances = instance_buffer(&self.device, "atom instances", &atoms);
        self.bond_instances = instance_buffer(&self.device, "bond instances", &bonds);
        self.cartoon_instances = instance_buffer(&self.device, "cartoon instances", &cartoons);
        self.toon_instances = toon_instance_buffer(&self.device, &toon_atoms);
        self.atom_instance_count = atoms.len() as u32;
        self.bond_instance_count = bonds.len() as u32;
        self.cartoon_instance_count = cartoons.len() as u32;
        self.toon_instance_count = toon_atoms.len() as u32;
    }

    pub fn update_measurements(&mut self, lines: &[MeasurementLine]) {
        let instances = measurement_instances(lines);
        self.measurement_instances =
            instance_buffer(&self.device, "measurement instances", &instances);
        self.measurement_instance_count = instances.len() as u32;
    }

    pub fn render(
        &mut self,
        camera: &OrbitCamera,
        display: Option<&DisplayState>,
        viewport: Viewport,
        paint_jobs: &[egui::ClippedPrimitive],
        textures_delta: &TexturesDelta,
        pixels_per_point: f32,
    ) -> Result<(), RenderError> {
        let ambient_occlusion = display
            .map(|display| display.ambient_occlusion)
            .unwrap_or_default();
        let global_mode = display.map_or(DisplayMode::Cartoon, |display| display.global_mode);
        let view_projection = camera.view_projection();
        let forward = camera.optical_axis();
        let mut camera_right = forward.cross(Vec3::Y).normalize_or_zero();
        if camera_right == Vec3::ZERO {
            camera_right = Vec3::X;
        }
        let camera_up = camera_right.cross(forward).normalize_or_zero();
        let camera_uniform = CameraUniform {
            view_projection: view_projection.to_cols_array(),
            inverse_view_projection: view_projection.inverse().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
            camera_right: camera_right.extend(0.0).to_array(),
            camera_up: camera_up.extend(0.0).to_array(),
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
        let focal_length = camera.depth_of_field.focal_length_mm
            / camera.depth_of_field.sensor_height_mm.max(0.001);
        let post_uniform = PostUniform {
            inverse_view_projection: view_projection.inverse().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
            optical_axis: camera.optical_axis().extend(0.0).to_array(),
            lens: [
                camera.focus_depth(),
                camera.depth_of_field.f_stop,
                camera.depth_of_field.max_coc_pixels,
                f32::from(camera.depth_of_field.enabled),
            ],
            aperture: [
                focal_length,
                camera.depth_of_field.blade_count as f32,
                camera.depth_of_field.blade_rotation,
                0.0,
            ],
            ao: [
                ambient_occlusion.strength.clamp(0.0, 3.0),
                ambient_occlusion.radius.clamp(0.05, 10.0),
                ambient_occlusion.bias.clamp(0.0, 0.3),
                if ambient_occlusion.enabled {
                    ambient_occlusion.quality.sample_count() as f32
                } else {
                    0.0
                },
            ],
        };
        self.queue.write_buffer(
            &self.post_process.uniform,
            0,
            bytemuck::bytes_of(&post_uniform),
        );
        for (id, deltas) in &textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(output)
            | wgpu::CurrentSurfaceTexture::Suboptimal(output) => output,
            wgpu::CurrentSurfaceTexture::Timeout => return Err(SurfaceIssue::Timeout.into()),
            wgpu::CurrentSurfaceTexture::Occluded => return Err(SurfaceIssue::Occluded.into()),
            wgpu::CurrentSurfaceTexture::Outdated => return Err(SurfaceIssue::Outdated.into()),
            wgpu::CurrentSurfaceTexture::Lost => return Err(SurfaceIssue::Lost.into()),
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(SurfaceIssue::Validation.into());
            }
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molview frame encoder"),
            });
        let screen = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point,
        };
        let callback_buffers = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            paint_jobs,
            &screen,
        );
        let scene_background = if global_mode == DisplayMode::Toon {
            // Linear RGB corresponding approximately to warm paper #F7F6F1.
            wgpu::Color {
                r: 0.930,
                g: 0.922,
                b: 0.880,
                a: 1.0,
            }
        } else {
            wgpu::Color {
                // sRGB #1D2123 converted to linear RGB for Rgba16Float.
                r: 0.012_286,
                g: 0.015_209,
                b: 0.016_807,
                a: 1.0,
            }
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("molecule scene pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.post_process.scene.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(scene_background),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.post_process.semantic.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_viewport(
                viewport.x,
                viewport.y,
                viewport.width.max(1.0),
                viewport.height.max(1.0),
                0.0,
                1.0,
            );
            let scissor_x = viewport.x.max(0.0) as u32;
            let scissor_y = viewport.y.max(0.0) as u32;
            let scissor_width = viewport
                .width
                .max(1.0)
                .min(self.config.width.saturating_sub(scissor_x) as f32)
                as u32;
            let scissor_height = viewport
                .height
                .max(1.0)
                .min(self.config.height.saturating_sub(scissor_y) as f32)
                as u32;
            pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);

            pass.set_vertex_buffer(0, self.ribbon.vertices.slice(..));
            pass.set_vertex_buffer(1, self.cartoon_instances.slice(..));
            pass.set_index_buffer(self.ribbon.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..self.ribbon.index_count,
                0,
                0..self.cartoon_instance_count,
            );

            pass.set_vertex_buffer(0, self.cylinder.vertices.slice(..));
            pass.set_vertex_buffer(1, self.bond_instances.slice(..));
            pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.cylinder.index_count, 0, 0..self.bond_instance_count);

            pass.set_vertex_buffer(0, self.sphere.vertices.slice(..));
            pass.set_vertex_buffer(1, self.atom_instances.slice(..));
            pass.set_index_buffer(self.sphere.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.sphere.index_count, 0, 0..self.atom_instance_count);

            pass.set_pipeline(&self.toon_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_vertex_buffer(0, self.toon_instances.slice(..));
            pass.draw(0..6, 0..self.toon_instance_count);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raw ambient occlusion pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.ao_raw.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.ao_raw_pipeline);
            pass.set_bind_group(0, &self.post_process.ao_raw_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bilateral ambient occlusion pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.ao_filtered.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.ao_blur_pipeline);
            pass.set_bind_group(0, &self.post_process.ao_blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("AO and DOF composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.pipeline);
            pass.set_bind_group(0, &self.post_process.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("annotation pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.annotation_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_viewport(
                viewport.x,
                viewport.y,
                viewport.width.max(1.0),
                viewport.height.max(1.0),
                0.0,
                1.0,
            );
            let scissor_x = viewport.x.max(0.0) as u32;
            let scissor_y = viewport.y.max(0.0) as u32;
            let scissor_width = viewport
                .width
                .max(1.0)
                .min(self.config.width.saturating_sub(scissor_x) as f32)
                as u32;
            let scissor_height = viewport
                .height
                .max(1.0)
                .min(self.config.height.saturating_sub(scissor_y) as f32)
                as u32;
            pass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);
            pass.set_vertex_buffer(0, self.cylinder.vertices.slice(..));
            pass.set_vertex_buffer(1, self.measurement_instances.slice(..));
            pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..self.cylinder.index_count,
                0,
                0..self.measurement_instance_count,
            );
        }
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("UI pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    // egui's pipeline has no depth-stencil target, so it must not share
                    // the depth-tested annotation pass.
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, paint_jobs, &screen);
        }

        self.queue
            .submit(callback_buffers.into_iter().chain([encoder.finish()]));
        self.queue.present(output);
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        Ok(())
    }
}

fn measurement_instances(lines: &[MeasurementLine]) -> Vec<InstanceRaw> {
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

#[derive(Clone, Copy)]
struct BackboneAnchor {
    atom_index: usize,
    residue_index: usize,
    position: Vec3,
    guide: Vec3,
    nucleic: bool,
}

fn cartoon_render_data(
    molecule: &Molecule,
    display: &DisplayState,
) -> (Vec<InstanceRaw>, Vec<bool>) {
    let hierarchy = MoleculeHierarchy::from_molecule(molecule);
    let mut cartoons = Vec::new();
    let mut cartoon_residue_atoms = vec![false; molecule.atoms.len()];
    for chain in &hierarchy.chains {
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
            });
        }

        let mut run = Vec::new();
        for anchor in anchors {
            let drawable = display
                .visible
                .get(anchor.atom_index)
                .copied()
                .unwrap_or(false)
                && display.modes.get(anchor.atom_index) == Some(&DisplayMode::Cartoon);
            let continuous = run.last().is_none_or(|previous: &BackboneAnchor| {
                anchor.residue_index == previous.residue_index + 1
                    && anchor.nucleic == previous.nucleic
                    && anchor.position.distance(previous.position)
                        <= if anchor.nucleic { 8.5 } else { 5.0 }
            });
            if !drawable || !continuous {
                append_ribbon_run(&run, display, &mut cartoons);
                run.clear();
            }
            if drawable {
                run.push(anchor);
            }
        }
        append_ribbon_run(&run, display, &mut cartoons);
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
    (cartoons, standard_atomic)
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
    instances: &mut Vec<InstanceRaw>,
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

    const SAMPLES_PER_RESIDUE: usize = 6;
    for segment in 0..run.len() - 1 {
        for sample in 0..SAMPLES_PER_RESIDUE {
            let t0 = sample as f32 / SAMPLES_PER_RESIDUE as f32;
            let t1 = (sample + 1) as f32 / SAMPLES_PER_RESIDUE as f32;
            let start = catmull_rom(run, segment, t0);
            let end = catmull_rom(run, segment, t1);
            let tangent = (end - start).normalize_or_zero();
            if tangent == Vec3::ZERO {
                continue;
            }
            let midpoint_t = (t0 + t1) * 0.5;
            let guide = guides[segment]
                .lerp(guides[segment + 1], midpoint_t)
                .normalize_or_zero();
            let mut side = (guide - tangent * guide.dot(tangent)).normalize_or_zero();
            if side == Vec3::ZERO {
                let fallback = if tangent.x.abs() < 0.8 {
                    Vec3::X
                } else {
                    Vec3::Z
                };
                side = (fallback - tangent * fallback.dot(tangent)).normalize_or_zero();
            }
            let normal = side.cross(tangent).normalize_or_zero();
            side = tangent.cross(normal).normalize_or_zero();
            let length = start.distance(end);
            let width = if run[segment].nucleic { 1.05 } else { 0.82 };
            let thickness = if run[segment].nucleic { 0.22 } else { 0.16 };
            let model = Mat4::from_cols(
                (side * width).extend(0.0),
                (tangent * length).extend(0.0),
                (normal * thickness).extend(0.0),
                ((start + end) * 0.5).extend(1.0),
            );
            let left = display.colors[run[segment].atom_index];
            let right = display.colors[run[segment + 1].atom_index];
            let color = [
                left[0] + (right[0] - left[0]) * midpoint_t,
                left[1] + (right[1] - left[1]) * midpoint_t,
                left[2] + (right[2] - left[2]) * midpoint_t,
                1.0,
            ];
            let selected = display.selection[run[segment].atom_index]
                || display.selection[run[segment + 1].atom_index];
            instances.push(InstanceRaw::new(model, color, selected));
        }
    }
}

fn catmull_rom(run: &[BackboneAnchor], segment: usize, t: f32) -> Vec3 {
    let p0 = run[segment.saturating_sub(1)].position;
    let p1 = run[segment].position;
    let p2 = run[segment + 1].position;
    let p3 = run[(segment + 2).min(run.len() - 1)].position;
    let t2 = t * t;
    let t3 = t2 * t;
    (p1 * 2.0
        + (p2 - p0) * t
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * t2
        + (-p0 + p1 * 3.0 - p2 * 3.0 + p3) * t3)
        * 0.5
}

fn toon_semantic_ids(molecule: &Molecule) -> Vec<[u32; 4]> {
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

fn instance_buffer(device: &wgpu::Device, label: &str, instances: &[InstanceRaw]) -> wgpu::Buffer {
    if instances.is_empty() {
        return empty_instance_buffer(device, label);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn empty_instance_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<InstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

fn toon_instance_buffer(device: &wgpu::Device, instances: &[ToonInstanceRaw]) -> wgpu::Buffer {
    if instances.is_empty() {
        return empty_toon_instance_buffer(device);
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("toon sphere instances"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn empty_toon_instance_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("toon sphere instances"),
        size: std::mem::size_of::<ToonInstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DisplayLevel, ModeOverride,
        molecule::{Atom, Element},
    };

    fn atom(serial: u32, name: &str, residue: i32, position: Vec3, hetero: bool) -> Atom {
        Atom {
            serial,
            name: name.into(),
            element: if name == "O" { Element::O } else { Element::C },
            residue_name: if hetero { "LIG".into() } else { "ALA".into() },
            residue_number: residue,
            insertion_code: None,
            chain_id: "A".into(),
            position,
            occupancy: 1.0,
            b_factor: 0.0,
            hetero,
        }
    }

    fn backbone_molecule() -> Molecule {
        let mut atoms = Vec::new();
        for residue in 0..3 {
            let x = residue as f32 * 3.8;
            atoms.push(atom(
                atoms.len() as u32 + 1,
                "CA",
                residue + 1,
                Vec3::new(x, 0.0, 0.0),
                false,
            ));
            atoms.push(atom(
                atoms.len() as u32 + 1,
                "O",
                residue + 1,
                Vec3::new(x, 1.0, 0.0),
                false,
            ));
        }
        atoms.push(atom(7, "C1", 10, Vec3::new(0.0, 4.0, 0.0), true));
        Molecule {
            atoms,
            bonds: Vec::new(),
        }
    }

    #[test]
    fn cartoon_builds_smooth_ribbon_and_keeps_ligands_atomic() {
        let molecule = backbone_molecule();
        let display = DisplayState::for_molecule(&molecule);
        let (cartoons, atomic) = cartoon_render_data(&molecule, &display);
        assert_eq!(cartoons.len(), 12);
        assert_eq!(&atomic[..6], &[false; 6]);
        assert!(atomic[6]);
    }

    #[test]
    fn hierarchy_mode_override_switches_a_residue_to_ball_and_stick() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_mode_override(&[2, 3], DisplayLevel::Residue, ModeOverride::BallAndStick);
        let (cartoons, atomic) = cartoon_render_data(&molecule, &display);
        assert!(cartoons.is_empty());
        assert!(atomic[2] && atomic[3]);
        assert!(!atomic[0] && !atomic[4]);
    }

    #[test]
    fn toon_uses_semantic_spheres_instead_of_standard_atomic_meshes() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_global_mode(DisplayMode::Toon);
        let (cartoons, standard_atomic) = cartoon_render_data(&molecule, &display);
        let ids = toon_semantic_ids(&molecule);
        assert!(cartoons.is_empty());
        assert!(standard_atomic.iter().all(|standard| !standard));
        assert_eq!(ids.len(), molecule.atoms.len());
        assert!(ids.iter().all(|id| id[0] != 0 && id[1] != 0 && id[2] != 0));
        assert_ne!(ids[0][1], ids[2][1]);
    }

    #[test]
    fn measurement_line_builds_dashes_with_a_center_label_gap() {
        let mut line = MeasurementLine::new(
            1,
            crate::measurement::MeasurementEndpoint {
                position: Vec3::ZERO,
                description: "A".into(),
            },
            crate::measurement::MeasurementEndpoint {
                position: Vec3::new(0.0, 0.0, 5.0),
                description: "B".into(),
            },
        );
        let instances = measurement_instances(std::slice::from_ref(&line));
        assert!(instances.len() >= 8);
        assert!(
            instances
                .iter()
                .all(|instance| instance.color == line.effective_color())
        );

        line.visibility = crate::VisibilityOverride::Hide;
        assert!(measurement_instances(&[line]).is_empty());
    }
}
