use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{
    pipelines::create_fullscreen_pipeline,
    renderer::{AO_FORMAT, SCENE_FORMAT},
    targets::{ColorTarget, SemanticTarget},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PostUniform {
    pub(super) inverse_view_projection: [f32; 16],
    pub(super) eye_position: [f32; 4],
    pub(super) optical_axis: [f32; 4],
    pub(super) lens: [f32; 4],
    pub(super) aperture: [f32; 4],
    pub(super) ao: [f32; 4],
}

pub(super) struct PostProcess {
    pub(super) scene: ColorTarget,
    pub(super) semantic: SemanticTarget,
    pub(super) ao_raw: ColorTarget,
    pub(super) ao_filtered: ColorTarget,
    pub(super) uniform: wgpu::Buffer,
    pub(super) sampler: wgpu::Sampler,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pub(super) bind_group: wgpu::BindGroup,
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) ao_bind_group_layout: wgpu::BindGroupLayout,
    pub(super) ao_raw_bind_group: wgpu::BindGroup,
    pub(super) ao_blur_bind_group: wgpu::BindGroup,
    pub(super) ao_raw_pipeline: wgpu::RenderPipeline,
    pub(super) ao_blur_pipeline: wgpu::RenderPipeline,
}

impl PostProcess {
    pub(super) fn new(
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

    pub(super) fn resize(
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
