use std::{
    borrow::Cow,
    sync::{Arc, mpsc},
};

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
    molecule::Molecule,
};

use super::{
    cartoon::{CartoonRenderData, cartoon_render_data},
    instances::{
        GpuMesh, InstanceRaw, ToonInstanceRaw, cartoon_index_buffer, cartoon_vertex_buffer,
        empty_cartoon_index_buffer, empty_cartoon_vertex_buffer, empty_instance_buffer,
        empty_toon_instance_buffer, instance_buffer, measurement_instances, toon_instance_buffer,
        toon_semantic_ids,
    },
    mesh,
    pipelines::{
        create_cartoon_pipeline, create_geometry_pipeline, create_scene_geometry_pipeline,
        create_toon_pipeline,
    },
    postprocess::{PostProcess, PostUniform},
    targets::{DepthTarget, PendingPickReadback},
};

pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub(super) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(super) const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
pub(super) const SEMANTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

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

pub struct Renderer {
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    cartoon_pipeline: wgpu::RenderPipeline,
    toon_pipeline: wgpu::RenderPipeline,
    annotation_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    sphere: GpuMesh,
    cylinder: GpuMesh,
    atom_instances: wgpu::Buffer,
    bond_instances: wgpu::Buffer,
    cartoon_vertices: wgpu::Buffer,
    cartoon_indices: wgpu::Buffer,
    toon_instances: wgpu::Buffer,
    measurement_instances: wgpu::Buffer,
    atom_instance_count: u32,
    bond_instance_count: u32,
    cartoon_index_count: u32,
    toon_instance_count: u32,
    measurement_instance_count: u32,
    depth: DepthTarget,
    post_process: PostProcess,
    egui_renderer: egui_wgpu::Renderer,
    requested_pick: Option<(u64, u32, u32)>,
    pending_pick: Option<PendingPickReadback>,
}

pub struct PreparedCartoon(CartoonRenderData);

pub fn prepare_cartoon(molecule: &Molecule, display: &DisplayState) -> PreparedCartoon {
    PreparedCartoon(cartoon_render_data(molecule, display))
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
        let cartoon_pipeline = create_cartoon_pipeline(&device, &pipeline_layout, &shader);
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
        let atom_instances = empty_instance_buffer(&device, "atom instances");
        let bond_instances = empty_instance_buffer(&device, "bond instances");
        let cartoon_vertices = empty_cartoon_vertex_buffer(&device);
        let cartoon_indices = empty_cartoon_index_buffer(&device);
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
            cartoon_pipeline,
            toon_pipeline,
            annotation_pipeline,
            camera_buffer,
            camera_bind_group,
            sphere,
            cylinder,
            atom_instances,
            bond_instances,
            cartoon_vertices,
            cartoon_indices,
            toon_instances,
            measurement_instances,
            atom_instance_count: 0,
            bond_instance_count: 0,
            cartoon_index_count: 0,
            toon_instance_count: 0,
            measurement_instance_count: 0,
            depth,
            post_process,
            egui_renderer,
            requested_pick: None,
            pending_pick: None,
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

    pub fn request_pick(&mut self, request_id: u64, x: u32, y: u32) {
        if x < self.config.width && y < self.config.height {
            self.requested_pick = Some((request_id, x, y));
        }
    }

    pub fn poll_pick(&mut self) -> Option<(u64, Result<Option<usize>, ()>)> {
        let _ = self.device.poll(wgpu::PollType::Poll);
        let ready = self
            .pending_pick
            .as_ref()
            .and_then(|pending| pending.receiver.try_recv().ok())?;
        let pending = self.pending_pick.take()?;
        let atom = if ready.is_ok() {
            match pending.buffer.get_mapped_range(0..16) {
                Ok(mapped) => {
                    let value = mapped
                        .get(..4)
                        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
                        .map(u32::from_le_bytes)
                        .unwrap_or(0);
                    drop(mapped);
                    pending.buffer.unmap();
                    Ok(value.checked_sub(1).map(|index| index as usize))
                }
                Err(_) => Err(()),
            }
        } else {
            Err(())
        };
        Some((pending.request_id, atom))
    }

    pub fn update_instances(&mut self, molecule: &Molecule, display: &DisplayState) {
        self.update_prepared_cartoon(molecule, display, prepare_cartoon(molecule, display));
    }

    pub fn update_prepared_cartoon(
        &mut self,
        molecule: &Molecule,
        display: &DisplayState,
        prepared: PreparedCartoon,
    ) {
        let cartoon = prepared.0;
        let semantic_ids = toon_semantic_ids(molecule);
        let atoms: Vec<_> = molecule
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                display.visible[*index]
                    && cartoon.standard_atomic[*index]
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
                .with_semantic_ids(semantic_ids[index])
            })
            .collect();

        let bonds: Vec<_> = molecule
            .bonds
            .iter()
            .filter(|bond| {
                display.visible[bond.a]
                    && display.visible[bond.b]
                    && cartoon.standard_atomic[bond.a]
                    && cartoon.standard_atomic[bond.b]
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
        self.cartoon_vertices = cartoon_vertex_buffer(&self.device, &cartoon.vertices);
        self.cartoon_indices = cartoon_index_buffer(&self.device, &cartoon.indices);
        self.toon_instances = toon_instance_buffer(&self.device, &toon_atoms);
        self.atom_instance_count = atoms.len() as u32;
        self.bond_instance_count = bonds.len() as u32;
        self.cartoon_index_count = cartoon.indices.len() as u32;
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
                f32::from(global_mode == DisplayMode::Toon),
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

            pass.set_pipeline(&self.cartoon_pipeline);
            pass.set_vertex_buffer(0, self.cartoon_vertices.slice(..));
            pass.set_index_buffer(self.cartoon_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.cartoon_index_count, 0, 0..1);

            pass.set_pipeline(&self.pipeline);

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
        let pick_readback = self.requested_pick.take().map(|(request_id, x, y)| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molecule ID pick readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.post_process.semantic.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            (request_id, buffer)
        });
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
        if let Some((request_id, buffer)) = pick_readback {
            let (sender, receiver) = mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read, 0..16, move |result| {
                let _ = sender.send(result);
            });
            self.pending_pick = Some(PendingPickReadback {
                request_id,
                buffer,
                receiver,
            });
        }
        self.queue.present(output);
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        Ok(())
    }
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
        let cartoon = cartoon_render_data(&molecule, &display);
        assert_eq!(cartoon.vertices.len(), 21 * 16);
        assert_eq!(cartoon.indices.len(), 20 * 16 * 6);
        assert_eq!(&cartoon.standard_atomic[..6], &[false; 6]);
        assert!(cartoon.standard_atomic[6]);
    }

    #[test]
    fn hierarchy_mode_override_switches_a_residue_to_ball_and_stick() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_mode_override(&[2, 3], DisplayLevel::Residue, ModeOverride::BallAndStick);
        let cartoon = cartoon_render_data(&molecule, &display);
        assert!(cartoon.vertices.is_empty() && cartoon.indices.is_empty());
        assert!(cartoon.standard_atomic[2] && cartoon.standard_atomic[3]);
        assert!(!cartoon.standard_atomic[0] && !cartoon.standard_atomic[4]);
    }

    #[test]
    fn toon_uses_semantic_spheres_instead_of_standard_atomic_meshes() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_global_mode(DisplayMode::Toon);
        let cartoon = cartoon_render_data(&molecule, &display);
        let ids = toon_semantic_ids(&molecule);
        assert!(cartoon.vertices.is_empty() && cartoon.indices.is_empty());
        assert!(cartoon.standard_atomic.iter().all(|standard| !standard));
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
