use std::{borrow::Cow, sync::Arc};

use bytemuck::{Pod, Zeroable};
use egui::TexturesDelta;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use glam::{Mat4, Quat, Vec3};
use thiserror::Error;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    DisplayState, RepresentationMask,
    camera::{OrbitCamera, Viewport},
    molecule::Molecule,
};

use super::mesh::{self, Vertex};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

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
    eye_position: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    inverse_view_projection: [f32; 16],
    eye_position: [f32; 4],
    optical_axis: [f32; 4],
    lens: [f32; 4],
    aperture: [f32; 4],
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
    fn new(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("molecule scene color"),
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

struct PostProcess {
    scene: ColorTarget,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl PostProcess {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
    ) -> Self {
        let scene = ColorTarget::new(device, width, height, SCENE_FORMAT);
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
            ],
        });
        let bind_group = create_post_bind_group(
            device,
            &bind_group_layout,
            &scene.view,
            &sampler,
            depth_view,
            &uniform,
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
        Self {
            scene,
            uniform,
            sampler,
            bind_group_layout,
            bind_group,
            pipeline,
        }
    }

    fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        depth_view: &wgpu::TextureView,
    ) {
        self.scene = ColorTarget::new(device, width, height, SCENE_FORMAT);
        self.bind_group = create_post_bind_group(
            device,
            &self.bind_group_layout,
            &self.scene.view,
            &self.sampler,
            depth_view,
            &self.uniform,
        );
    }
}

fn create_post_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    depth_view: &wgpu::TextureView,
    uniform: &wgpu::Buffer,
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
        ],
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
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    sphere: GpuMesh,
    cylinder: GpuMesh,
    atom_instances: wgpu::Buffer,
    bond_instances: wgpu::Buffer,
    atom_instance_count: u32,
    bond_instance_count: u32,
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
        let vertex_layouts = [Some(Vertex::layout()), Some(InstanceRaw::layout())];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("molecule pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &vertex_layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // The two reusable meshes use different parametric seam winding at poles/ends.
                // Two-sided rasterization keeps those low-resolution boundary triangles robust.
                cull_mode: None,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let sphere = GpuMesh::new(&device, "sphere", mesh::uv_sphere(14, 22));
        let cylinder = GpuMesh::new(&device, "cylinder", mesh::cylinder(16));
        let atom_instances = empty_instance_buffer(&device, "atom instances");
        let bond_instances = empty_instance_buffer(&device, "bond instances");
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
            camera_buffer,
            camera_bind_group,
            sphere,
            cylinder,
            atom_instances,
            bond_instances,
            atom_instance_count: 0,
            bond_instance_count: 0,
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
        let atoms: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                display.visible[*index]
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

        self.atom_instances = instance_buffer(&self.device, "atom instances", &atoms);
        self.bond_instances = instance_buffer(&self.device, "bond instances", &bonds);
        self.atom_instance_count = atoms.len() as u32;
        self.bond_instance_count = bonds.len() as u32;
    }

    pub fn render(
        &mut self,
        camera: &OrbitCamera,
        viewport: Viewport,
        paint_jobs: &[egui::ClippedPrimitive],
        textures_delta: &TexturesDelta,
        pixels_per_point: f32,
    ) -> Result<(), RenderError> {
        let camera_uniform = CameraUniform {
            view_projection: camera.view_projection().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
        let focal_length = camera.depth_of_field.focal_length_mm
            / camera.depth_of_field.sensor_height_mm.max(0.001);
        let post_uniform = PostUniform {
            inverse_view_projection: camera.view_projection().inverse().to_cols_array(),
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
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("molecule scene pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.scene.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.032,
                            b: 0.045,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
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

            pass.set_vertex_buffer(0, self.cylinder.vertices.slice(..));
            pass.set_vertex_buffer(1, self.bond_instances.slice(..));
            pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.cylinder.index_count, 0, 0..self.bond_instance_count);

            pass.set_vertex_buffer(0, self.sphere.vertices.slice(..));
            pass.set_vertex_buffer(1, self.atom_instances.slice(..));
            pass.set_index_buffer(self.sphere.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.sphere.index_count, 0, 0..self.atom_instance_count);
        }
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DOF composition and UI pass"),
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
                })
                .forget_lifetime();
            pass.set_pipeline(&self.post_process.pipeline);
            pass.set_bind_group(0, &self.post_process.bind_group, &[]);
            pass.draw(0..3, 0..1);
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
