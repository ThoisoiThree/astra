use std::{
    sync::{Arc, mpsc},
    time::Instant,
};

use bytemuck::{Pod, Zeroable};
use egui::TexturesDelta;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use glam::Vec3;
use thiserror::Error;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    DisplayMode, DisplayState,
    camera::{OrbitCamera, Viewport},
    measurement::MeasurementLine,
    molecule::{Molecule, MoleculeHierarchy, SecondaryStructure},
};

#[cfg(test)]
use super::instances::semantic_ids_from_hierarchy;

use super::{
    cartoon::{CartoonRenderData, cartoon_render_data, cartoon_render_data_cached},
    dof::{DepthOfField, scene_shader},
    instances::{
        GpuMesh, ReusableBuffer, cartoon_display_attributes, display_attributes, display_topology,
        measurement_instances,
    },
    mesh,
    pipelines::{
        create_cartoon_pipeline, create_geometry_pipeline, create_scene_geometry_pipeline,
        create_toon_pipeline,
    },
    postprocess::{PostProcess, PostUniform},
    profiling::{GpuProfiler, PASS_COUNT, ProfilePass},
    targets::{DepthTarget, PendingPickReadback},
    viewport_cache::ViewportCache,
};

pub(super) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub(super) const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(super) const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
// Atom ID plus packed residue/chain ID. This is half the memory of Rgba32Uint while retaining
// every hierarchy boundary required by picking and toon outlines.
pub(super) const SEMANTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

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

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub cpu_frame_ms: f32,
    pub gpu_pass_ms: [f32; PASS_COUNT],
    pub gpu_memory_bytes: u64,
    pub atom_instances: u32,
    pub bond_instances: u32,
    pub cartoon_triangles: u32,
    pub toon_triangles: u32,
    pub gpu_timestamps_supported: bool,
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
    camera_layout: wgpu::BindGroupLayout,
    dof: Option<DepthOfField>,
    sphere: GpuMesh,
    cylinder: GpuMesh,
    atom_topology: ReusableBuffer,
    atom_display: ReusableBuffer,
    bond_topology: ReusableBuffer,
    bond_display: ReusableBuffer,
    cartoon_vertices: ReusableBuffer,
    cartoon_display: ReusableBuffer,
    cartoon_indices: ReusableBuffer,
    toon_topology: ReusableBuffer,
    toon_display: ReusableBuffer,
    measurement_instances: ReusableBuffer,
    atom_instance_count: u32,
    bond_instance_count: u32,
    cartoon_index_count: u32,
    toon_instance_count: u32,
    measurement_instance_count: u32,
    cartoon_data: CartoonRenderData,
    depth: DepthTarget,
    post_process: PostProcess,
    viewport_cache: ViewportCache,
    egui_renderer: egui_wgpu::Renderer,
    requested_pick: Option<(u64, u32, u32)>,
    pending_pick: Option<PendingPickReadback>,
    profiler: Option<GpuProfiler>,
    cpu_frame_ms: f32,
}

pub struct PreparedCartoon(CartoonRenderData);

pub fn prepare_cartoon(molecule: &Molecule, display: &DisplayState) -> PreparedCartoon {
    PreparedCartoon(cartoon_render_data(molecule, display))
}

pub fn prepare_cartoon_cached(
    molecule: &Molecule,
    display: &DisplayState,
    hierarchy: &MoleculeHierarchy,
    secondary_structure: &[Vec<SecondaryStructure>],
) -> PreparedCartoon {
    PreparedCartoon(cartoon_render_data_cached(
        molecule,
        display,
        hierarchy,
        secondary_structure,
    ))
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
        let optional_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Astra device"),
                required_features: optional_features,
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
        let shader = scene_shader(&device, include_str!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molecule pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline =
            create_scene_geometry_pipeline(&device, &pipeline_layout, &shader, false, false);
        let cartoon_pipeline =
            create_cartoon_pipeline(&device, &pipeline_layout, &shader, false, false);
        let toon_shader = scene_shader(&device, include_str!("toon_sphere.wgsl"));
        let toon_pipeline =
            create_toon_pipeline(&device, &pipeline_layout, &toon_shader, false, false);
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
        let atom_topology =
            ReusableBuffer::new(&device, "atom topology", wgpu::BufferUsages::VERTEX);
        let atom_display = ReusableBuffer::new(
            &device,
            "atom display attributes",
            wgpu::BufferUsages::VERTEX,
        );
        let bond_topology =
            ReusableBuffer::new(&device, "bond topology", wgpu::BufferUsages::VERTEX);
        let bond_display = ReusableBuffer::new(
            &device,
            "bond display attributes",
            wgpu::BufferUsages::VERTEX,
        );
        let cartoon_vertices = ReusableBuffer::new(
            &device,
            "continuous cartoon vertices",
            wgpu::BufferUsages::VERTEX,
        );
        let cartoon_display = ReusableBuffer::new(
            &device,
            "continuous cartoon display attributes",
            wgpu::BufferUsages::VERTEX,
        );
        let cartoon_indices = ReusableBuffer::new(
            &device,
            "continuous cartoon indices",
            wgpu::BufferUsages::INDEX,
        );
        let toon_topology =
            ReusableBuffer::new(&device, "toon sphere topology", wgpu::BufferUsages::VERTEX);
        let toon_display = ReusableBuffer::new(
            &device,
            "toon sphere display attributes",
            wgpu::BufferUsages::VERTEX,
        );
        let measurement_instances =
            ReusableBuffer::new(&device, "measurement instances", wgpu::BufferUsages::VERTEX);
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
        let profiler = GpuProfiler::new(&device, &queue);
        let viewport_cache =
            ViewportCache::new(&device, config.width, config.height, config.format);

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
            camera_layout,
            dof: None,
            sphere,
            cylinder,
            atom_topology,
            atom_display,
            bond_topology,
            bond_display,
            cartoon_vertices,
            cartoon_display,
            cartoon_indices,
            toon_topology,
            toon_display,
            measurement_instances,
            atom_instance_count: 0,
            bond_instance_count: 0,
            cartoon_index_count: 0,
            toon_instance_count: 0,
            measurement_instance_count: 0,
            cartoon_data: CartoonRenderData::default(),
            depth,
            post_process,
            viewport_cache,
            egui_renderer,
            requested_pick: None,
            pending_pick: None,
            profiler,
            cpu_frame_ms: 0.0,
        })
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    pub fn stats(&self) -> RenderStats {
        let dynamic_bytes = self.atom_topology.estimated_bytes()
            + self.atom_display.estimated_bytes()
            + self.bond_topology.estimated_bytes()
            + self.bond_display.estimated_bytes()
            + self.cartoon_vertices.estimated_bytes()
            + self.cartoon_display.estimated_bytes()
            + self.cartoon_indices.estimated_bytes()
            + self.toon_topology.estimated_bytes()
            + self.toon_display.estimated_bytes()
            + self.measurement_instances.estimated_bytes();
        let pixel_count = self.config.width as u64 * self.config.height as u64;
        let dof_width = u64::from(self.post_process.dof_color._texture.width());
        let dof_height = u64::from(self.post_process.dof_color._texture.height());
        // scene RGBA16F + semantic RG32U + depth32F + two R16F AO targets + DOF RGBA16F.
        let target_bytes = pixel_count * (8 + 8 + 4 + 2 + 2 + 4)
            + dof_width * dof_height * 8
            + self.dof.as_ref().map_or(0, DepthOfField::estimated_bytes);
        RenderStats {
            cpu_frame_ms: self.cpu_frame_ms,
            gpu_pass_ms: self
                .profiler
                .as_ref()
                .map_or([0.0; PASS_COUNT], GpuProfiler::latest_ms),
            gpu_memory_bytes: dynamic_bytes
                + self.sphere.estimated_bytes
                + self.cylinder.estimated_bytes
                + target_bytes,
            atom_instances: self.atom_instance_count,
            bond_instances: self.bond_instance_count,
            cartoon_triangles: self.cartoon_index_count / 3,
            toon_triangles: self.toon_instance_count * 2,
            gpu_timestamps_supported: self.profiler.is_some(),
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth = DepthTarget::new(&self.device, size.width, size.height);
        self.dof = None;
        self.viewport_cache =
            ViewportCache::new(&self.device, size.width, size.height, self.config.format);
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
        self.cartoon_data = prepared.0;
        self.cartoon_indices
            .write(&self.device, &self.queue, &self.cartoon_data.indices);
        self.cartoon_vertices
            .write(&self.device, &self.queue, &self.cartoon_data.vertices);
        self.cartoon_index_count = self.cartoon_data.indices.len() as u32;
        self.update_instance_buffers(molecule, display, true);
    }

    /// Refreshes visibility, color, and selection data without rebuilding ribbon topology or
    /// recomputing secondary structure.
    pub fn update_display_attributes(&mut self, molecule: &Molecule, display: &DisplayState) {
        self.update_instance_buffers(molecule, display, false);
    }

    fn update_instance_buffers(
        &mut self,
        molecule: &Molecule,
        display: &DisplayState,
        topology_changed: bool,
    ) {
        self.viewport_cache.revision.invalidate();
        let attributes = display_attributes(molecule, display, &self.cartoon_data.standard_atomic);
        let cartoon_display = cartoon_display_attributes(&self.cartoon_data.vertices, display);
        if topology_changed {
            let topology = display_topology(
                molecule,
                display,
                &self.cartoon_data.standard_atomic,
                &self.cartoon_data.semantic_ids,
            );
            self.atom_topology
                .write(&self.device, &self.queue, &topology.atom_topology);
            self.bond_topology
                .write(&self.device, &self.queue, &topology.bond_topology);
            self.toon_topology
                .write(&self.device, &self.queue, &topology.toon_topology);
        }
        self.atom_display
            .write(&self.device, &self.queue, &attributes.atom_display);
        self.bond_display
            .write(&self.device, &self.queue, &attributes.bond_display);
        self.cartoon_display
            .write(&self.device, &self.queue, &cartoon_display);
        self.toon_display
            .write(&self.device, &self.queue, &attributes.toon_display);
        self.atom_instance_count = attributes.atom_display.len() as u32;
        self.bond_instance_count = attributes.bond_display.len() as u32;
        self.toon_instance_count = attributes.toon_display.len() as u32;
    }

    pub fn update_measurements(&mut self, lines: &[MeasurementLine]) {
        self.viewport_cache.revision.invalidate();
        let instances = measurement_instances(lines);
        self.measurement_instances
            .write(&self.device, &self.queue, &instances);
        self.measurement_instance_count = instances.len() as u32;
    }

    fn draw_molecules(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        geometry_pipeline: &wgpu::RenderPipeline,
        cartoon_pipeline: &wgpu::RenderPipeline,
        toon_pipeline: &wgpu::RenderPipeline,
    ) {
        pass.set_pipeline(cartoon_pipeline);
        pass.set_vertex_buffer(0, self.cartoon_vertices.buffer.slice(..));
        pass.set_vertex_buffer(1, self.cartoon_display.buffer.slice(..));
        pass.set_index_buffer(
            self.cartoon_indices.buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        pass.draw_indexed(0..self.cartoon_index_count, 0, 0..1);

        pass.set_pipeline(geometry_pipeline);

        pass.set_vertex_buffer(0, self.cylinder.vertices.slice(..));
        pass.set_vertex_buffer(1, self.bond_topology.buffer.slice(..));
        pass.set_vertex_buffer(2, self.bond_display.buffer.slice(..));
        pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.cylinder.index_count, 0, 0..self.bond_instance_count);

        pass.set_vertex_buffer(0, self.sphere.vertices.slice(..));
        pass.set_vertex_buffer(1, self.atom_topology.buffer.slice(..));
        pass.set_vertex_buffer(2, self.atom_display.buffer.slice(..));
        pass.set_index_buffer(self.sphere.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.sphere.index_count, 0, 0..self.atom_instance_count);

        pass.set_pipeline(toon_pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.toon_topology.buffer.slice(..));
        pass.set_vertex_buffer(1, self.toon_display.buffer.slice(..));
        pass.draw(0..6, 0..self.toon_instance_count);
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
        let cpu_start = Instant::now();
        if let Some(profiler) = &mut self.profiler {
            profiler.poll(&self.device);
            profiler.begin_frame();
        }
        let ambient_occlusion = display
            .map(|display| display.ambient_occlusion)
            .unwrap_or_default();
        let global_mode = display.map_or(DisplayMode::Cartoon, |display| display.global_mode);
        let quality = ambient_occlusion.quality;
        let previous_dof_scale = self.post_process.dof_scale;
        self.post_process.set_dof_scale(
            &self.device,
            self.config.width,
            self.config.height,
            &self.depth.view,
            quality.dof_resolution_scale(),
        );
        let dof_enabled =
            camera.depth_of_field.enabled && camera.depth_of_field.max_coc_pixels > 0.0;
        let dof_size = [
            self.post_process.dof_color._texture.width(),
            self.post_process.dof_color._texture.height(),
        ];
        if dof_enabled
            && (previous_dof_scale != self.post_process.dof_scale
                || self.dof.as_ref().is_none_or(|dof| dof.size != dof_size))
        {
            self.dof = None;
            self.dof = Some(DepthOfField::new(
                &self.device,
                &self.camera_layout,
                &self.post_process,
                &self.depth.view,
            ));
        } else if !dof_enabled {
            self.dof = None;
        }
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
        let post_uniform = PostUniform {
            inverse_view_projection: view_projection.inverse().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
            optical_axis: camera.optical_axis().extend(viewport.height).to_array(),
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
                f32::from(self.toon_instance_count > 0),
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
            viewport: [
                viewport.x / self.config.width as f32,
                viewport.y / self.config.height as f32,
                viewport.width / self.config.width as f32,
                viewport.height / self.config.height as f32,
            ],
            background: [
                scene_background.r as f32,
                scene_background.g as f32,
                scene_background.b as f32,
                1.0,
            ],
            quality: [
                quality.dof_layer_count() as f32,
                dof_size[1] as f32 / self.config.height as f32,
                dof_size[0] as f32,
                dof_size[1] as f32,
            ],
        };
        self.queue.write_buffer(
            &self.post_process.uniform,
            0,
            bytemuck::bytes_of(&post_uniform),
        );
        let scene_key = [
            bytemuck::bytes_of(&camera_uniform),
            bytemuck::bytes_of(&post_uniform),
        ]
        .concat();
        let scene_changed = self.viewport_cache.revision.changed(&scene_key);
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
                label: Some("Astra frame encoder"),
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
        if scene_changed {
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
                timestamp_writes: self
                    .profiler
                    .as_ref()
                    .map(|profiler| profiler.writes(ProfilePass::Scene)),
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

            self.draw_molecules(
                &mut pass,
                &self.pipeline,
                &self.cartoon_pipeline,
                &self.toon_pipeline,
            );
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
        if scene_changed && ambient_occlusion.enabled {
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
                timestamp_writes: self
                    .profiler
                    .as_ref()
                    .map(|profiler| profiler.writes(ProfilePass::AoRaw)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.ao_raw_pipeline);
            pass.set_bind_group(0, &self.post_process.ao_raw_bind_group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);

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
                timestamp_writes: self
                    .profiler
                    .as_ref()
                    .map(|profiler| profiler.writes(ProfilePass::AoBlur)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.ao_blur_pipeline);
            pass.set_bind_group(0, &self.post_process.ao_blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if scene_changed
            && dof_enabled
            && let Some(dof) = &self.dof
        {
            let mut timestamps = self.profiler.as_ref().map(|p| p.writes(ProfilePass::Dof));
            let end_timestamps = timestamps
                .as_ref()
                .map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: None,
                    end_of_pass_write_index: t.end_of_pass_write_index,
                });
            if let Some(t) = &mut timestamps {
                t.end_of_pass_write_index = None;
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DOF first geometry layer"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: dof.layer_color(0),
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(scene_background),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        None,
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: dof.layer_depth(0),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: timestamps,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let sx = dof.size[0] as f32 / self.config.width as f32;
                let sy = dof.size[1] as f32 / self.config.height as f32;
                pass.set_viewport(
                    viewport.x * sx,
                    viewport.y * sy,
                    viewport.width * sx,
                    viewport.height * sy,
                    0.0,
                    1.0,
                );
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.draw_molecules(
                    &mut pass,
                    &dof.first_geometry_pipeline,
                    &dof.first_cartoon_pipeline,
                    &dof.first_toon_pipeline,
                );
            }
            dof.encode_mask(&mut encoder);
            for layer in 1..quality.dof_layer_count() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DOF partial depth peeling"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: dof.layer_color(layer),
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(scene_background),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        None,
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: dof.layer_depth(layer),
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
                let sx = dof.size[0] as f32 / self.config.width as f32;
                let sy = dof.size[1] as f32 / self.config.height as f32;
                pass.set_viewport(
                    viewport.x * sx,
                    viewport.y * sy,
                    viewport.width * sx,
                    viewport.height * sy,
                    0.0,
                    1.0,
                );
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, dof.peel_binding(layer), &[]);
                self.draw_molecules(
                    &mut pass,
                    &dof.geometry_pipeline,
                    &dof.cartoon_pipeline,
                    &dof.toon_pipeline,
                );
            }
            dof.encode_splat(&mut encoder, end_timestamps);
        }
        if scene_changed {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("AO and DOF composition pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.viewport_cache.color.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self
                    .profiler
                    .as_ref()
                    .map(|profiler| profiler.writes(ProfilePass::Compose)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_process.pipeline);
            pass.set_bind_group(0, &self.post_process.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if scene_changed {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("annotation pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.viewport_cache.color.view,
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
                timestamp_writes: self
                    .profiler
                    .as_ref()
                    .map(|profiler| profiler.writes(ProfilePass::Annotations)),
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
            pass.set_vertex_buffer(1, self.measurement_instances.buffer.slice(..));
            pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(
                0..self.cylinder.index_count,
                0,
                0..self.measurement_instance_count,
            );
        }
        self.viewport_cache.blit(&mut encoder, &view);
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
                    timestamp_writes: self
                        .profiler
                        .as_ref()
                        .map(|profiler| profiler.writes(ProfilePass::Ui)),
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, paint_jobs, &screen);
        }

        let profile_readback = self
            .profiler
            .as_mut()
            .and_then(|profiler| profiler.encode_readback(&self.device, &mut encoder));
        self.queue
            .submit(callback_buffers.into_iter().chain([encoder.finish()]));
        if let (Some(profiler), Some(buffer)) = (&mut self.profiler, profile_readback) {
            profiler.begin_readback(buffer);
        }
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
        self.viewport_cache.revision.commit(scene_key);
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        let elapsed_ms = cpu_start.elapsed().as_secs_f32() * 1_000.0;
        self.cpu_frame_ms = if self.cpu_frame_ms == 0.0 {
            elapsed_ms
        } else {
            self.cpu_frame_ms * 0.9 + elapsed_ms * 0.1
        };
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
        assert_eq!(cartoon.vertices.len(), 11 * 12);
        assert_eq!(cartoon.indices.len(), 10 * 12 * 6);
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
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let ids = semantic_ids_from_hierarchy(molecule.atoms.len(), &hierarchy);
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
        let expected = crate::SrgbColor(line.effective_color()).to_linear().0;
        assert!(instances.iter().all(|instance| instance.color == expected));

        line.visibility = crate::VisibilityOverride::Hide;
        assert!(measurement_instances(&[line]).is_empty());
    }
}
