use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
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
    AmbientOcclusionQuality, AmbientOcclusionSettings, DisplayMode, DisplayState,
    camera::{OrbitCamera, Viewport},
    diagnostics::StartupTrace,
    measurement::MeasurementLine,
    molecule::{Molecule, MoleculeHierarchy, SecondaryStructure},
    surface::{SurfaceMesh, compute_surface_cached},
};

#[cfg(test)]
use super::instances::semantic_ids_from_hierarchy;

use super::{
    cartoon::{CartoonRenderData, cartoon_render_data, cartoon_render_data_cached},
    dof::{DepthOfField, scene_shader},
    instances::{
        AtomicStyle, BondInstance, GpuMesh, MeshVertex, ReusableBuffer, STYLE_TOON, SphereInstance,
        atom_colors, atom_meta, atom_positions, atomic_styles, bond_instances,
        measurement_instances, sphere_instances,
    },
    mesh,
    pipelines::{ScenePipelines, SceneTarget, create_annotation_pipeline},
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
/// Exported images are written as 8-bit sRGB with straight alpha.
const EXPORT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const MAX_RENDER_SCALE: u32 = 4;

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
    #[error("Could not initialize DoF: {0}")]
    DepthOfField(String),
    #[error("image export failed: {0}")]
    Export(String),
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
    pub present_mode: wgpu::PresentMode,
    pub fps: f32,
    pub cpu_frame_ms: f32,
    /// Wall times for preparation, encoding, surface acquisition, submission and presentation.
    pub cpu_stages_ms: [f32; 5],
    pub gpu_pass_ms: [f32; PASS_COUNT],
    pub gpu_frame_ms: f32,
    pub gpu_dof_ms: f32,
    pub gpu_memory_bytes: u64,
    pub atom_instances: u32,
    pub bond_instances: u32,
    pub cartoon_triangles: u32,
    pub surface_triangles: u32,
    pub toon_triangles: u32,
    pub render_scale: u32,
    pub gpu_timestamps_supported: bool,
}

/// Background of an exported image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportBackground {
    /// The viewport's own background.
    #[default]
    Scene,
    White,
    Black,
    Transparent,
}

impl ExportBackground {
    pub const ALL: [Self; 4] = [Self::Scene, Self::White, Self::Black, Self::Transparent];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Scene => "Viewport",
            Self::White => "White",
            Self::Black => "Black",
            Self::Transparent => "Transparent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageExport {
    pub width: u32,
    pub height: u32,
    /// Samples per output pixel along each axis (1–4); reduced if the scene would exceed
    /// the device's texture size.
    pub supersampling: u32,
    pub background: ExportBackground,
}

/// An exported image: 8-bit sRGB, straight (non-premultiplied) alpha, rows top to bottom.
#[derive(Debug, Clone)]
pub struct RenderedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// The supersampling factor that was actually used.
    pub supersampling: u32,
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

impl CameraUniform {
    fn new(camera: &OrbitCamera) -> Self {
        let view_projection = camera.view_projection();
        let forward = camera.optical_axis();
        let mut camera_right = forward.cross(Vec3::Y).normalize_or_zero();
        if camera_right == Vec3::ZERO {
            camera_right = Vec3::X;
        }
        let camera_up = camera_right.cross(forward).normalize_or_zero();
        Self {
            view_projection: view_projection.to_cols_array(),
            inverse_view_projection: view_projection.inverse().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
            camera_right: camera_right.extend(0.0).to_array(),
            camera_up: camera_up.extend(0.0).to_array(),
        }
    }
}

/// Render targets for one output image: scene targets at `scale` times the output size.
struct FrameTargets {
    /// Output size in pixels.
    width: u32,
    height: u32,
    scale: u32,
    depth: DepthTarget,
    post: PostProcess,
    dof: Option<DepthOfField>,
}

impl FrameTargets {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        scale: u32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let (scene_width, scene_height) = (width.max(1) * scale, height.max(1) * scale);
        let depth = DepthTarget::new(device, scene_width, scene_height);
        let post = PostProcess::new(
            device,
            output_format,
            scene_width,
            scene_height,
            &depth.view,
        );
        Self {
            width: width.max(1),
            height: height.max(1),
            scale,
            depth,
            post,
            dof: None,
        }
    }

    fn scene_size(&self) -> (u32, u32) {
        (self.width * self.scale, self.height * self.scale)
    }

    fn estimated_bytes(&self) -> u64 {
        let (width, height) = self.scene_size();
        let pixels = u64::from(width) * u64::from(height);
        let dof_width = u64::from(self.post.dof_color._texture.width());
        let dof_height = u64::from(self.post.dof_color._texture.height());
        // scene RGBA16F + overlay RGBA16F + semantic RG32U + depth32F + two R16F AO + DoF.
        pixels * (8 + 8 + 8 + 4 + 2 + 2)
            + dof_width * dof_height * 8
            + self.dof.as_ref().map_or(0, DepthOfField::estimated_bytes)
    }
}

/// Everything a frame's scene passes need besides the geometry.
#[derive(Clone, Copy)]
struct FrameParameters {
    /// Molecular viewport in output pixels.
    viewport: Viewport,
    /// Linear RGB background; alpha 0 exports a transparent background.
    background: [f32; 4],
    ambient_occlusion: AmbientOcclusionSettings,
    quality: AmbientOcclusionQuality,
    dof_enabled: bool,
    fxaa: bool,
    toon: bool,
}

/// Linear RGB for sRGB #1D2123 and for warm paper #F7F6F1 behind Toon.
const DARK_BACKGROUND: [f32; 4] = [0.012_286, 0.015_209, 0.016_807, 1.0];
const PAPER_BACKGROUND: [f32; 4] = [0.930, 0.922, 0.880, 1.0];

pub struct Renderer {
    presented_frames: std::collections::VecDeque<Instant>,
    adapter_info: wgpu::AdapterInfo,
    instance: wgpu::Instance,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene_layout: wgpu::BindGroupLayout,
    scene_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    atom_positions: ReusableBuffer,
    atom_colors: ReusableBuffer,
    atom_meta: ReusableBuffer,
    pipelines: ScenePipelines,
    annotation_pipeline: wgpu::RenderPipeline,
    cylinder: GpuMesh,
    spheres: ReusableBuffer,
    bonds: ReusableBuffer,
    cartoon_vertices: ReusableBuffer,
    cartoon_indices: ReusableBuffer,
    surface_vertices: ReusableBuffer,
    surface_indices: ReusableBuffer,
    measurement_instances: ReusableBuffer,
    sphere_count: u32,
    toon_count: u32,
    bond_count: u32,
    cartoon_index_count: u32,
    surface_index_count: u32,
    surface_mesh: Arc<SurfaceMesh>,
    measurement_instance_count: u32,
    cartoon_data: CartoonRenderData,
    frame: FrameTargets,
    render_scale: u32,
    viewport_cache: ViewportCache,
    egui_renderer: egui_wgpu::Renderer,
    requested_pick: Option<(u64, u32, u32)>,
    pending_pick: Option<PendingPickReadback>,
    profiler: Option<GpuProfiler>,
    cpu_frame_ms: f32,
    cpu_stages_ms: [f32; 5],
}

/// Geometry built off the render thread: ribbons, bases and the molecular surface.
pub struct PreparedCartoon {
    cartoon: CartoonRenderData,
    surface: Arc<SurfaceMesh>,
}

pub fn prepare_cartoon(molecule: &Molecule, display: &DisplayState) -> PreparedCartoon {
    PreparedCartoon {
        cartoon: cartoon_render_data(molecule, display),
        surface: prepare_surface(molecule, display, &AtomicBool::new(false)).unwrap_or_default(),
    }
}

/// Like [`prepare_cartoon`], reusing cached ribbons and surfaces; `None` if cancelled.
pub fn prepare_cartoon_cached(
    molecule: &Molecule,
    display: &DisplayState,
    hierarchy: &MoleculeHierarchy,
    secondary_structure: &[Vec<SecondaryStructure>],
    cancel: &AtomicBool,
) -> Option<PreparedCartoon> {
    let cartoon = cartoon_render_data_cached(molecule, display, hierarchy, secondary_structure);
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    Some(PreparedCartoon {
        cartoon,
        surface: prepare_surface(molecule, display, cancel)?,
    })
}

/// Surface over the visible atoms with the surface representation; empty when there are
/// none. `None` if cancelled.
pub fn prepare_surface(
    molecule: &Molecule,
    display: &DisplayState,
    cancel: &AtomicBool,
) -> Option<Arc<SurfaceMesh>> {
    let spheres = display.surface_spheres(molecule);
    if spheres.is_empty() {
        return Some(Arc::default());
    }
    compute_surface_cached(&spheres, &display.surface, cancel)
}

fn scene_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1),
            storage(2),
            storage(3),
        ],
    })
}

fn scene_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera: &wgpu::Buffer,
    buffers: [&ReusableBuffer; 3],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers[0].buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers[1].buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers[2].buffer.as_entire_binding(),
            },
        ],
    })
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self, RenderError> {
        let trace = StartupTrace::new("gpu-init");
        let size = window.inner_size();
        let descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        #[cfg(target_os = "windows")]
        let descriptor = {
            let mut descriptor = descriptor;
            descriptor.backends = super::backend::WindowsBackend::load().backends();
            #[cfg(all(target_env = "msvc", not(target_arch = "aarch64")))]
            {
                // DoF requires a modern HLSL compiler; do not silently fall back to FXC.
                descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::StaticDxc;
            }
            descriptor
        };
        trace.mark(format_args!(
            "BEGIN instance: backends={:?}, dx12_compiler={:?}",
            descriptor.backends, descriptor.backend_options.dx12.shader_compiler
        ));
        crate::diagnostics::write_graphics_log(format_args!(
            "wgpu {}: backends={:?}",
            env!("ASTRA_WGPU_VERSION"),
            descriptor.backends,
        ));
        let instance = wgpu::Instance::new(descriptor);
        trace.mark("END instance; BEGIN surface creation");
        let surface = instance.create_surface(window.clone())?;
        trace.mark("END surface creation; BEGIN adapter selection");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await?;
        crate::diagnostics::write_graphics_log(format_args!("Adapter: {:?}", adapter.get_info()));
        trace.mark(format_args!(
            "END adapter selection: {:?}; BEGIN device creation",
            adapter.get_info()
        ));
        let optional_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        // Request the adapter's texture size so large exports can supersample.
        let required_limits = wgpu::Limits {
            max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Astra device"),
                required_features: optional_features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        trace.mark("END device creation; BEGIN surface configuration");
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(RenderError::SurfaceConfiguration)?;
        #[cfg(target_os = "windows")]
        let config = {
            let mut config = config;
            let modes = surface.get_capabilities(&adapter).present_modes;
            // Mailbox accepts newer frames without waiting for the FIFO queue
            // to drain at vblank, while avoiding Immediate's tearing.
            config.present_mode = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
                .into_iter()
                .find(|mode| modes.contains(mode))
                .unwrap_or(config.present_mode);
            crate::diagnostics::write_graphics_log(format_args!(
                "Presentation: {:?}; supported={modes:?}",
                config.present_mode
            ));
            config
        };
        surface.configure(&device, &config);
        trace.mark(format_args!(
            "END surface configuration: {config:?}; BEGIN scene pipelines"
        ));

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let storage = wgpu::BufferUsages::STORAGE;
        let atom_positions = ReusableBuffer::new(&device, "atom positions", storage);
        let atom_colors = ReusableBuffer::new(&device, "atom colors", storage);
        let atom_meta = ReusableBuffer::new(&device, "atom semantic ids and flags", storage);
        let scene_layout = scene_layout(&device);
        let scene_group = scene_group(
            &device,
            &scene_layout,
            &camera_buffer,
            [&atom_positions, &atom_colors, &atom_meta],
        );
        let impostor_shader = scene_shader(&device, include_str!("impostor.wgsl"));
        let mesh_shader = scene_shader(&device, include_str!("shader.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molecule pipeline layout"),
            bind_group_layouts: &[Some(&scene_layout)],
            immediate_size: 0,
        });
        let pipelines = ScenePipelines::new(
            &device,
            &pipeline_layout,
            &impostor_shader,
            &mesh_shader,
            SceneTarget::Scene,
        );
        let annotation_pipeline =
            create_annotation_pipeline(&device, &pipeline_layout, &mesh_shader, SCENE_FORMAT);
        trace.mark("END scene pipelines; BEGIN buffers and frame targets");

        let cylinder = GpuMesh::new(&device, "cylinder", mesh::cylinder(16));
        let vertex = wgpu::BufferUsages::VERTEX;
        let index = wgpu::BufferUsages::INDEX;
        let frame = FrameTargets::new(&device, config.width, config.height, 1, config.format);
        trace.mark("END buffers and frame targets; BEGIN egui renderer");
        let egui_renderer =
            egui_wgpu::Renderer::new(&device, config.format, RendererOptions::default());
        trace.mark("END egui renderer; BEGIN profiler and viewport cache");
        let profiler = GpuProfiler::new(&device, &queue);
        let viewport_cache =
            ViewportCache::new(&device, config.width, config.height, config.format);
        trace.mark("END profiler and viewport cache; renderer ready");

        Ok(Self {
            presented_frames: std::collections::VecDeque::new(),
            adapter_info: adapter.get_info(),
            instance,
            window,
            surface,
            spheres: ReusableBuffer::new(&device, "sphere impostors", vertex),
            bonds: ReusableBuffer::new(&device, "bond impostors", vertex),
            cartoon_vertices: ReusableBuffer::new(&device, "cartoon vertices", vertex),
            cartoon_indices: ReusableBuffer::new(&device, "cartoon indices", index),
            surface_vertices: ReusableBuffer::new(&device, "surface vertices", vertex),
            surface_indices: ReusableBuffer::new(&device, "surface indices", index),
            measurement_instances: ReusableBuffer::new(&device, "measurement instances", vertex),
            device,
            queue,
            config,
            scene_layout,
            scene_group,
            camera_buffer,
            atom_positions,
            atom_colors,
            atom_meta,
            pipelines,
            annotation_pipeline,
            cylinder,
            sphere_count: 0,
            toon_count: 0,
            bond_count: 0,
            cartoon_index_count: 0,
            surface_index_count: 0,
            surface_mesh: Arc::default(),
            measurement_instance_count: 0,
            cartoon_data: CartoonRenderData::default(),
            frame,
            render_scale: 1,
            viewport_cache,
            egui_renderer,
            requested_pick: None,
            pending_pick: None,
            profiler,
            cpu_frame_ms: 0.0,
            cpu_stages_ms: [0.0; 5],
        })
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    pub fn max_texture_dimension(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub fn stats(&self) -> RenderStats {
        let dynamic_bytes = [
            &self.atom_positions,
            &self.atom_colors,
            &self.atom_meta,
            &self.spheres,
            &self.bonds,
            &self.cartoon_vertices,
            &self.cartoon_indices,
            &self.surface_vertices,
            &self.surface_indices,
            &self.measurement_instances,
        ]
        .iter()
        .map(|buffer| buffer.estimated_bytes())
        .sum::<u64>();
        let viewport_bytes = u64::from(self.config.width) * u64::from(self.config.height) * 4;
        RenderStats {
            present_mode: self.config.present_mode,
            fps: self
                .presented_frames
                .iter()
                .filter(|frame| frame.elapsed().as_secs_f32() < 1.0)
                .count() as f32,
            cpu_frame_ms: self.cpu_frame_ms,
            cpu_stages_ms: self.cpu_stages_ms,
            gpu_pass_ms: self
                .profiler
                .as_ref()
                .map_or([0.0; PASS_COUNT], GpuProfiler::latest_ms),
            gpu_frame_ms: self
                .profiler
                .as_ref()
                .map_or(0.0, GpuProfiler::latest_frame_ms),
            gpu_dof_ms: self
                .profiler
                .as_ref()
                .map_or(0.0, GpuProfiler::latest_dof_ms),
            gpu_memory_bytes: dynamic_bytes
                + self.cylinder.estimated_bytes
                + self.frame.estimated_bytes()
                + viewport_bytes,
            atom_instances: self.sphere_count,
            bond_instances: self.bond_count,
            cartoon_triangles: self.cartoon_index_count / 3,
            surface_triangles: self.surface_index_count / 3,
            toon_triangles: self.toon_count * 2,
            render_scale: self.render_scale,
            gpu_timestamps_supported: self.profiler.is_some(),
        }
    }

    /// Largest supersampling factor the device supports for the current window.
    fn clamp_scale(&self, width: u32, height: u32, requested: u32) -> u32 {
        let limit = self.max_texture_dimension();
        let mut scale = requested.clamp(1, MAX_RENDER_SCALE);
        while scale > 1 && (width.max(1) * scale > limit || height.max(1) * scale > limit) {
            scale -= 1;
        }
        scale
    }

    /// Sets viewport supersampling: the scene renders at `scale` times the window size and
    /// is box-filtered down, which antialiases every edge.
    pub fn set_render_scale(&mut self, scale: u32) {
        let scale = self.clamp_scale(self.config.width, self.config.height, scale);
        if scale != self.render_scale {
            self.render_scale = scale;
            self.frame = FrameTargets::new(
                &self.device,
                self.config.width,
                self.config.height,
                scale,
                self.config.format,
            );
            self.viewport_cache.revision.invalidate();
        }
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.render_scale = self.clamp_scale(size.width, size.height, self.render_scale);
        self.frame = FrameTargets::new(
            &self.device,
            size.width,
            size.height,
            self.render_scale,
            self.config.format,
        );
        self.viewport_cache =
            ViewportCache::new(&self.device, size.width, size.height, self.config.format);
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

    /// Uploads ribbon geometry and rebuilds every atomic instance and per-atom buffer.
    pub fn update_prepared_cartoon(
        &mut self,
        molecule: &Molecule,
        display: &DisplayState,
        prepared: PreparedCartoon,
    ) {
        self.cartoon_data = prepared.cartoon;
        self.upload_cartoon_geometry();
        self.upload_surface(prepared.surface);
        self.update_atom_buffers(molecule, display, true);
    }

    /// Replaces only the ribbon mesh, for example after a new trajectory frame.
    pub fn update_cartoon_geometry(&mut self, prepared: PreparedCartoon) {
        let atomic = std::mem::take(&mut self.cartoon_data.atomic);
        let semantic_ids = std::mem::take(&mut self.cartoon_data.semantic_ids);
        self.cartoon_data = prepared.cartoon;
        self.upload_surface(prepared.surface);
        // The ribbon/atomic split and semantic ids depend only on the topology.
        if self.cartoon_data.atomic.len() != atomic.len() {
            self.cartoon_data.atomic = atomic;
        }
        if self.cartoon_data.semantic_ids.len() != semantic_ids.len() {
            self.cartoon_data.semantic_ids = semantic_ids;
        }
        self.upload_cartoon_geometry();
    }

    fn upload_cartoon_geometry(&mut self) {
        self.cartoon_indices
            .write(&self.device, &self.queue, &self.cartoon_data.indices);
        self.cartoon_vertices
            .write(&self.device, &self.queue, &self.cartoon_data.vertices);
        self.cartoon_index_count = self.cartoon_data.indices.len() as u32;
        self.viewport_cache.revision.invalidate();
    }

    /// Refreshes colors and selection without rebuilding geometry.
    pub fn update_display_attributes(&mut self, molecule: &Molecule, display: &DisplayState) {
        self.update_atom_buffers(molecule, display, false);
    }

    /// Uploads new coordinates for every atom. Impostor geometry follows automatically;
    /// ribbons and surfaces are rebuilt separately.
    pub fn update_positions(&mut self, molecule: &Molecule) {
        let replaced =
            self.atom_positions
                .write(&self.device, &self.queue, &atom_positions(molecule));
        if replaced {
            self.rebuild_scene_group();
        }
        self.viewport_cache.revision.invalidate();
    }

    /// Replaces the molecular surface mesh unless it is the one already uploaded.
    fn upload_surface(&mut self, surface: Arc<SurfaceMesh>) {
        if Arc::ptr_eq(&surface, &self.surface_mesh) {
            return;
        }
        let vertices: Vec<MeshVertex> = surface
            .positions
            .iter()
            .zip(&surface.normals)
            .zip(&surface.atoms)
            .map(|((position, normal), atom)| MeshVertex::new(*position, *normal, *atom as usize))
            .collect();
        self.surface_vertices
            .write(&self.device, &self.queue, &vertices);
        self.surface_indices
            .write(&self.device, &self.queue, &surface.indices);
        self.surface_index_count = surface.indices.len() as u32;
        self.surface_mesh = surface;
        self.viewport_cache.revision.invalidate();
    }

    fn rebuild_scene_group(&mut self) {
        self.scene_group = scene_group(
            &self.device,
            &self.scene_layout,
            &self.camera_buffer,
            [&self.atom_positions, &self.atom_colors, &self.atom_meta],
        );
    }

    fn update_atom_buffers(
        &mut self,
        molecule: &Molecule,
        display: &DisplayState,
        topology_changed: bool,
    ) {
        self.viewport_cache.revision.invalidate();
        let mut replaced = false;
        if topology_changed {
            let styles: Vec<AtomicStyle> = atomic_styles(display, &self.cartoon_data.atomic);
            let spheres: Vec<SphereInstance> = sphere_instances(molecule, display, &styles);
            let bonds: Vec<BondInstance> = bond_instances(molecule, display, &styles);
            self.spheres.write(&self.device, &self.queue, &spheres);
            self.bonds.write(&self.device, &self.queue, &bonds);
            self.sphere_count = spheres.len() as u32;
            self.toon_count = spheres
                .iter()
                .filter(|sphere| sphere.style == STYLE_TOON)
                .count() as u32;
            self.bond_count = bonds.len() as u32;
            replaced |=
                self.atom_positions
                    .write(&self.device, &self.queue, &atom_positions(molecule));
        }
        replaced |= self
            .atom_colors
            .write(&self.device, &self.queue, &atom_colors(display));
        replaced |= self.atom_meta.write(
            &self.device,
            &self.queue,
            &atom_meta(display, &self.cartoon_data.semantic_ids),
        );
        if replaced {
            self.rebuild_scene_group();
        }
    }

    pub fn update_measurements(&mut self, lines: &[MeasurementLine]) {
        self.viewport_cache.revision.invalidate();
        let instances = measurement_instances(lines);
        self.measurement_instances
            .write(&self.device, &self.queue, &instances);
        self.measurement_instance_count = instances.len() as u32;
    }

    fn draw_molecules(&self, pass: &mut wgpu::RenderPass<'_>, pipelines: &ScenePipelines) {
        pass.set_bind_group(0, &self.scene_group, &[]);
        pass.set_pipeline(&pipelines.meshes);
        for (vertices, indices, count) in [
            (
                &self.cartoon_vertices,
                &self.cartoon_indices,
                self.cartoon_index_count,
            ),
            (
                &self.surface_vertices,
                &self.surface_indices,
                self.surface_index_count,
            ),
        ] {
            if count > 0 {
                pass.set_vertex_buffer(0, vertices.buffer.slice(..));
                pass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }
        if self.bond_count > 0 {
            pass.set_pipeline(&pipelines.bonds);
            pass.set_vertex_buffer(0, self.bonds.buffer.slice(..));
            pass.draw(0..36, 0..self.bond_count);
        }
        if self.sphere_count > 0 {
            pass.set_pipeline(&pipelines.spheres);
            pass.set_vertex_buffer(0, self.spheres.buffer.slice(..));
            pass.draw(0..6, 0..self.sphere_count);
        }
    }

    fn frame_parameters(
        &self,
        camera: &OrbitCamera,
        display: Option<&DisplayState>,
        viewport: Viewport,
    ) -> FrameParameters {
        let ambient_occlusion = display
            .map(|display| display.ambient_occlusion)
            .unwrap_or_default();
        let global_mode = display.map_or(DisplayMode::Cartoon, |display| display.global_mode);
        FrameParameters {
            viewport,
            background: if global_mode == DisplayMode::Toon {
                PAPER_BACKGROUND
            } else {
                DARK_BACKGROUND
            },
            ambient_occlusion,
            quality: ambient_occlusion.quality,
            dof_enabled: camera.depth_of_field.enabled
                && camera.depth_of_field.max_coc_pixels > 0.0,
            fxaa: true,
            toon: self.toon_count > 0,
        }
    }

    /// Creates or replaces depth-of-field resources for `targets` when enabled.
    fn prepare_depth_of_field(
        device: &wgpu::Device,
        scene_layout: &wgpu::BindGroupLayout,
        targets: &mut FrameTargets,
        parameters: &FrameParameters,
        adapter: &wgpu::AdapterInfo,
    ) -> Result<(), RenderError> {
        let (width, height) = targets.scene_size();
        let previous_scale = targets.post.dof_scale;
        targets.post.set_dof_scale(
            device,
            width,
            height,
            &targets.depth.view,
            parameters.quality.dof_resolution_scale(),
        );
        let dof_size = [
            targets.post.dof_color._texture.width(),
            targets.post.dof_color._texture.height(),
        ];
        if previous_scale != targets.post.dof_scale {
            // Bind groups reference the output texture, even if rounded dimensions match.
            targets.dof = None;
        }
        if !parameters.dof_enabled || targets.dof.as_ref().is_some_and(|dof| dof.size == dof_size) {
            return Ok(());
        }
        targets.dof = None;
        crate::diagnostics::write_graphics_log(format_args!(
            "BEGIN DoF initialization: {adapter:?}, size={dof_size:?}"
        ));
        let memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let dof = DepthOfField::new(device, scene_layout, &targets.post, &targets.depth.view);
        // These scopes only contain synchronous resource/pipeline creation, no submitted GPU
        // work. Pop every scope even when the first reports an error.
        let errors = [
            pollster::block_on(validation_scope.pop()),
            pollster::block_on(internal_scope.pop()),
            pollster::block_on(memory_scope.pop()),
        ]
        .into_iter()
        .flatten()
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
        if !errors.is_empty() {
            let message = errors.join("\n");
            crate::diagnostics::write_graphics_log(&message);
            return Err(RenderError::DepthOfField(message));
        }
        crate::diagnostics::write_graphics_log("END DoF initialization");
        targets.dof = Some(dof);
        Ok(())
    }

    fn post_uniform(
        camera: &OrbitCamera,
        targets: &FrameTargets,
        parameters: &FrameParameters,
    ) -> PostUniform {
        let scale = targets.scale as f32;
        let (scene_width, scene_height) = targets.scene_size();
        let dof_size = [
            targets.post.dof_color._texture.width(),
            targets.post.dof_color._texture.height(),
        ];
        let view_projection = camera.view_projection();
        let focal_length = camera.depth_of_field.focal_length_mm
            / camera.depth_of_field.sensor_height_mm.max(0.001);
        let ambient_occlusion = parameters.ambient_occlusion;
        let viewport = parameters.viewport;
        PostUniform {
            inverse_view_projection: view_projection.inverse().to_cols_array(),
            eye_position: camera.eye().extend(1.0).to_array(),
            // Lens quantities are expressed in scene (supersampled) pixels.
            optical_axis: camera
                .optical_axis()
                .extend(viewport.height * scale)
                .to_array(),
            lens: [
                camera.focus_depth(),
                camera.depth_of_field.f_stop,
                camera.depth_of_field.max_coc_pixels * scale,
                f32::from(camera.depth_of_field.enabled),
            ],
            aperture: [
                focal_length,
                camera.depth_of_field.blade_count as f32,
                camera.depth_of_field.blade_rotation,
                f32::from(parameters.toon),
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
            quality: [
                parameters.quality.dof_layer_count() as f32,
                dof_size[1] as f32 / scene_height as f32,
                dof_size[0] as f32,
                dof_size[1] as f32,
            ],
            background: parameters.background,
            viewport: [
                viewport.x * scale / scene_width as f32,
                viewport.y * scale / scene_height as f32,
                viewport.width * scale / scene_width as f32,
                viewport.height * scale / scene_height as f32,
            ],
            output: [
                scale,
                f32::from(parameters.background[3] < 1.0),
                f32::from(parameters.fxaa && targets.scale == 1),
                0.0,
            ],
        }
    }

    /// Encodes the molecular scene, AO, depth of field, annotations and composition into
    /// `output` (an image of the targets' output size).
    fn encode_scene(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        parameters: &FrameParameters,
        output: &wgpu::TextureView,
        profile: bool,
    ) {
        let profiler = self.profiler.as_ref().filter(|_| profile);
        let scale = targets.scale as f32;
        let viewport = parameters.viewport;
        let (scene_width, scene_height) = targets.scene_size();
        let background = wgpu::Color {
            r: f64::from(parameters.background[0]),
            g: f64::from(parameters.background[1]),
            b: f64::from(parameters.background[2]),
            a: f64::from(parameters.background[3]),
        };
        let scene_rect = (
            viewport.x * scale,
            viewport.y * scale,
            (viewport.width * scale).max(1.0),
            (viewport.height * scale).max(1.0),
        );
        let scissor = {
            let x = scene_rect.0.max(0.0) as u32;
            let y = scene_rect.1.max(0.0) as u32;
            (
                x,
                y,
                (scene_rect.2.min(scene_width.saturating_sub(x) as f32) as u32).max(1),
                (scene_rect.3.min(scene_height.saturating_sub(y) as f32) as u32).max(1),
            )
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("molecule scene pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.post.scene.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(background),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &targets.post.semantic.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: profiler.map(|profiler| profiler.writes(ProfilePass::Scene)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(
                scene_rect.0,
                scene_rect.1,
                scene_rect.2,
                scene_rect.3,
                0.0,
                1.0,
            );
            pass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
            self.draw_molecules(&mut pass, &self.pipelines);
        }
        if parameters.ambient_occlusion.enabled {
            for (label, view, pipeline, bindings, profile_pass) in [
                (
                    "raw ambient occlusion pass",
                    &targets.post.ao_raw.view,
                    &targets.post.ao_raw_pipeline,
                    &targets.post.ao_raw_bind_group,
                    ProfilePass::AoRaw,
                ),
                (
                    "bilateral ambient occlusion pass",
                    &targets.post.ao_filtered.view,
                    &targets.post.ao_blur_pipeline,
                    &targets.post.ao_blur_bind_group,
                    ProfilePass::AoBlur,
                ),
            ] {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: profiler.map(|profiler| profiler.writes(profile_pass)),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bindings, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        if parameters.dof_enabled
            && let Some(dof) = &targets.dof
        {
            self.encode_depth_of_field(encoder, targets, dof, parameters, background, profiler);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("annotation pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.post.overlay.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: profiler
                    .map(|profiler| profiler.writes(ProfilePass::Annotations)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.measurement_instance_count > 0 {
                pass.set_pipeline(&self.annotation_pipeline);
                pass.set_bind_group(0, &self.scene_group, &[]);
                pass.set_viewport(
                    scene_rect.0,
                    scene_rect.1,
                    scene_rect.2,
                    scene_rect.3,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
                pass.set_vertex_buffer(0, self.cylinder.vertices.slice(..));
                pass.set_vertex_buffer(1, self.measurement_instances.buffer.slice(..));
                pass.set_index_buffer(self.cylinder.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(
                    0..self.cylinder.index_count,
                    0,
                    0..self.measurement_instance_count,
                );
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("AO, DOF and annotation composition pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: profiler.map(|profiler| profiler.writes(ProfilePass::Compose)),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&targets.post.pipeline);
        pass.set_bind_group(0, &targets.post.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn encode_depth_of_field(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &FrameTargets,
        dof: &DepthOfField,
        parameters: &FrameParameters,
        background: wgpu::Color,
        profiler: Option<&GpuProfiler>,
    ) {
        let quality = parameters.quality;
        let viewport = parameters.viewport;
        let (scene_width, scene_height) = targets.scene_size();
        let scale = targets.scale as f32;
        let sx = dof.size[0] as f32 / scene_width as f32 * scale;
        let sy = dof.size[1] as f32 / scene_height as f32 * scale;
        let mut timestamps = profiler.map(|p| p.writes(ProfilePass::Dof));
        let end_timestamps = timestamps
            .as_ref()
            .map(|t| wgpu::RenderPassTimestampWrites {
                query_set: t.query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: t.end_of_pass_write_index,
            });
        if let Some(t) = &mut timestamps
            && quality.dof_layer_count() > 1
        {
            t.end_of_pass_write_index = None;
        }
        for layer in 0..quality.dof_layer_count() {
            let timestamp_writes = if layer == 0 {
                timestamps.take()
            } else {
                end_timestamps
                    .as_ref()
                    .filter(|_| layer + 1 == quality.dof_layer_count())
                    .map(|t| wgpu::RenderPassTimestampWrites {
                        query_set: t.query_set,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: t.end_of_pass_write_index,
                    })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(if layer == 0 {
                    "DOF first geometry layer"
                } else {
                    "DOF partial depth peeling"
                }),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: dof.layer_color(layer),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(background),
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
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(
                viewport.x * sx,
                viewport.y * sy,
                (viewport.width * sx).max(1.0),
                (viewport.height * sy).max(1.0),
                0.0,
                1.0,
            );
            if layer == 0 {
                self.draw_molecules(&mut pass, &dof.first_pipelines);
            } else {
                pass.set_bind_group(1, dof.peel_binding(layer), &[]);
                self.draw_molecules(&mut pass, &dof.peel_pipelines);
            }
            drop(pass);
            if layer == 0 {
                dof.encode_mask(encoder);
            }
        }
        let compute_timestamps = |kind| {
            profiler.map(|p| {
                let t = p.writes(kind);
                wgpu::ComputePassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: t.beginning_of_pass_write_index,
                    end_of_pass_write_index: t.end_of_pass_write_index,
                }
            })
        };
        dof.encode_splat_profiled(
            encoder,
            compute_timestamps(ProfilePass::DofReduce),
            compute_timestamps(ProfilePass::DofSplat),
        );
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
        let trace = StartupTrace::new("gpu-frame");
        trace.mark("BEGIN frame preparation");
        if let Some(profiler) = &mut self.profiler {
            profiler.poll(&self.device);
            profiler.begin_frame();
        }
        let parameters = self.frame_parameters(camera, display, viewport);
        Self::prepare_depth_of_field(
            &self.device,
            &self.scene_layout,
            &mut self.frame,
            &parameters,
            &self.adapter_info,
        )?;
        // Keep resources while disabled: toggling the lens must not recompile
        // every shader. Resize or a changed target size still replaces them.
        let camera_uniform = CameraUniform::new(camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
        let post_uniform = Self::post_uniform(camera, &self.frame, &parameters);
        self.queue.write_buffer(
            &self.frame.post.uniform,
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

        trace.mark("END frame preparation; BEGIN surface acquisition");
        let prepare_ms = cpu_start.elapsed().as_secs_f32() * 1_000.0;
        let acquire_start = Instant::now();
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
        let acquire_ms = acquire_start.elapsed().as_secs_f32() * 1_000.0;
        let encode_start = Instant::now();
        trace.mark("END surface acquisition; BEGIN command encoding");
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
            self.encode_scene(
                &mut encoder,
                &self.frame,
                &parameters,
                &self.viewport_cache.color.view,
                true,
            );
        }
        let pick_readback = self.requested_pick.take().map(|(request_id, x, y)| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molecule ID pick readback"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            // The semantic target is supersampled; read the sample at the pixel center.
            let scale = self.frame.scale;
            let (scene_width, scene_height) = self.frame.scene_size();
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.frame.post.semantic.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: (x * scale + scale / 2).min(scene_width - 1),
                        y: (y * scale + scale / 2).min(scene_height - 1),
                        z: 0,
                    },
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
        trace.mark("END command encoding; BEGIN queue submission");
        let command_buffer = encoder.finish();
        let encode_ms = encode_start.elapsed().as_secs_f32() * 1_000.0;
        let submit_start = Instant::now();
        self.queue
            .submit(callback_buffers.into_iter().chain([command_buffer]));
        let submit_ms = submit_start.elapsed().as_secs_f32() * 1_000.0;
        trace.mark("END queue submission");
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
        trace.mark("BEGIN presentation");
        let present_start = Instant::now();
        self.queue.present(output);
        let present_ms = present_start.elapsed().as_secs_f32() * 1_000.0;
        let now = Instant::now();
        self.presented_frames.push_back(now);
        while self
            .presented_frames
            .front()
            .is_some_and(|frame| now.duration_since(*frame).as_secs_f32() >= 1.0)
        {
            self.presented_frames.pop_front();
        }
        trace.mark("END presentation");
        self.viewport_cache.revision.commit(scene_key);
        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        let elapsed_ms = cpu_start.elapsed().as_secs_f32() * 1_000.0;
        for (smoothed, sample) in self
            .cpu_stages_ms
            .iter_mut()
            .zip([prepare_ms, encode_ms, acquire_ms, submit_ms, present_ms])
        {
            *smoothed = if self.cpu_frame_ms == 0.0 {
                sample
            } else {
                *smoothed * 0.9 + sample * 0.1
            };
        }
        self.cpu_frame_ms = if self.cpu_frame_ms == 0.0 {
            elapsed_ms
        } else {
            self.cpu_frame_ms * 0.9 + elapsed_ms * 0.1
        };
        Ok(())
    }

    /// Renders the current scene offscreen at an arbitrary size with supersampling. The
    /// camera keeps its orientation and framing; only its aspect ratio follows the image.
    pub fn render_image(
        &mut self,
        camera: &OrbitCamera,
        display: Option<&DisplayState>,
        export: &ImageExport,
    ) -> Result<RenderedImage, RenderError> {
        let limit = self.max_texture_dimension();
        let (width, height) = (export.width, export.height);
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(RenderError::Export(format!(
                "image size must be between 1 and {limit} pixels per side"
            )));
        }
        let scale = self.clamp_scale(width, height, export.supersampling);
        let mut camera = camera.clone();
        camera.aspect = width as f32 / height as f32;
        let viewport = Viewport::full(width, height);
        let mut parameters = self.frame_parameters(&camera, display, viewport);
        parameters.fxaa = scale == 1;
        match export.background {
            ExportBackground::Scene => {}
            ExportBackground::White => parameters.background = [1.0, 1.0, 1.0, 1.0],
            ExportBackground::Black => parameters.background = [0.0, 0.0, 0.0, 1.0],
            ExportBackground::Transparent => {
                parameters.background[3] = 0.0;
                // Depth of field blends with an opaque background; a transparent image
                // renders the pinhole view.
                parameters.dof_enabled = false;
            }
        }
        let mut targets = FrameTargets::new(&self.device, width, height, scale, EXPORT_FORMAT);
        Self::prepare_depth_of_field(
            &self.device,
            &self.scene_layout,
            &mut targets,
            &parameters,
            &self.adapter_info,
        )?;
        let image = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("exported image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: EXPORT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let image_view = image.create_view(&wgpu::TextureViewDescriptor::default());
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(&camera)),
        );
        self.queue.write_buffer(
            &targets.post.uniform,
            0,
            bytemuck::bytes_of(&Self::post_uniform(&camera, &targets, &parameters)),
        );
        let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exported image readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("image export encoder"),
            });
        self.encode_scene(&mut encoder, &targets, &parameters, &image_view, false);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &image,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        // The window's next frame must redraw with its own camera.
        self.viewport_cache.revision.invalidate();
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(scope.pop()),
        ]
        .into_iter()
        .flatten()
        .map(|error| error.to_string())
        .collect::<Vec<_>>();
        if !errors.is_empty() {
            return Err(RenderError::Export(errors.join("\n")));
        }
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| RenderError::Export(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| RenderError::Export(error.to_string()))?
            .map_err(|error| RenderError::Export(error.to_string()))?;
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|error| RenderError::Export(error.to_string()))?;
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for row in 0..height as usize {
            let start = row * bytes_per_row as usize;
            rgba.extend_from_slice(&mapped[start..start + width as usize * 4]);
        }
        drop(mapped);
        readback.unmap();
        if export.background == ExportBackground::Transparent {
            unpremultiply_srgb(&mut rgba);
        }
        Ok(RenderedImage {
            width,
            height,
            rgba,
            supersampling: scale,
        })
    }
}

/// Converts sRGB-encoded premultiplied color to straight alpha, in linear light.
fn unpremultiply_srgb(rgba: &mut [u8]) {
    let decode = |value: u8| {
        let value = f32::from(value) / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    let encode = |value: f32| {
        let value = value.clamp(0.0, 1.0);
        let encoded = if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u8
    };
    for pixel in rgba.as_chunks_mut::<4>().0 {
        let alpha = f32::from(pixel[3]) / 255.0;
        if alpha <= 0.0 {
            pixel[..3].fill(0);
        } else if alpha < 1.0 {
            for channel in &mut pixel[..3] {
                *channel = encode(decode(*channel) / alpha);
            }
        }
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
            alt_loc: None,
            formal_charge: 0,
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
            info: Default::default(),
        }
    }

    #[test]
    fn cartoon_builds_smooth_ribbon_and_keeps_ligands_atomic() {
        let molecule = backbone_molecule();
        let display = DisplayState::for_molecule(&molecule);
        let cartoon = cartoon_render_data(&molecule, &display);
        // Three residues form two spline segments; each ring has twice the width segments
        // (at least six per side).
        let quality = display.ambient_occlusion.quality;
        let rows = 2 * quality.cartoon_samples_per_residue() + 1;
        let ring = quality.cartoon_width_segments().max(6) as usize * 2;
        assert_eq!(cartoon.vertices.len(), rows * ring);
        assert_eq!(cartoon.indices.len(), (rows - 1) * ring * 6);
        assert_eq!(&cartoon.atomic[..6], &[false; 6]);
        assert!(cartoon.atomic[6]);
    }

    #[test]
    fn hierarchy_mode_override_switches_a_residue_to_ball_and_stick() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_mode_override(&[2, 3], DisplayLevel::Residue, ModeOverride::BallAndStick);
        let cartoon = cartoon_render_data(&molecule, &display);
        assert!(cartoon.vertices.is_empty() && cartoon.indices.is_empty());
        assert!(cartoon.atomic[2] && cartoon.atomic[3]);
        assert!(!cartoon.atomic[0] && !cartoon.atomic[4]);
    }

    #[test]
    fn toon_uses_semantic_spheres_instead_of_ribbons() {
        let molecule = backbone_molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_global_mode(DisplayMode::Toon);
        let cartoon = cartoon_render_data(&molecule, &display);
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let ids = semantic_ids_from_hierarchy(molecule.atoms.len(), &hierarchy);
        assert!(cartoon.vertices.is_empty() && cartoon.indices.is_empty());
        assert!(cartoon.atomic.iter().all(|atomic| *atomic));
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
