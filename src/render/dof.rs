//! Partial multilayer generation and tiled, depth-ordered splatting (Franke et al., 2018).
use std::borrow::Cow;

use super::{
    pipelines::{create_cartoon_pipeline, create_scene_geometry_pipeline, create_toon_pipeline},
    postprocess::{PostProcess, create_post_bind_group},
    renderer::{DEPTH_FORMAT, SCENE_FORMAT},
    targets::ColorTarget,
};

const MAX_LAYERS: u32 = 5;

struct ArrayTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    layers: Vec<wgpu::TextureView>,
}

impl ArrayTarget {
    fn new(device: &wgpu::Device, size: [u32; 2], format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Franke DOF layered target"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: MAX_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layers = (0..MAX_LAYERS)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        Self {
            _texture: texture,
            view,
            layers,
        }
    }
}

struct ComputeStage {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
}

pub(super) struct DepthOfField {
    pub(super) size: [u32; 2],
    color: ArrayTarget,
    depth: ArrayTarget,
    pub(super) reduced: wgpu::Texture,
    shaded_first: ColorTarget,
    shade_bindings: wgpu::BindGroup,
    shade_pipeline: wgpu::RenderPipeline,
    _mask: [wgpu::Texture; 2],
    pub(super) first_geometry_pipeline: wgpu::RenderPipeline,
    pub(super) first_cartoon_pipeline: wgpu::RenderPipeline,
    pub(super) first_toon_pipeline: wgpu::RenderPipeline,
    pub(super) geometry_pipeline: wgpu::RenderPipeline,
    pub(super) cartoon_pipeline: wgpu::RenderPipeline,
    pub(super) toon_pipeline: wgpu::RenderPipeline,
    peel_bindings: Vec<wgpu::BindGroup>,
    stages: Vec<ComputeStage>,
}

impl DepthOfField {
    pub(super) fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        post: &PostProcess,
        scene_depth: &wgpu::TextureView,
    ) -> Self {
        let size = [
            post.dof_color._texture.width(),
            post.dof_color._texture.height(),
        ];
        let color = ArrayTarget::new(device, size, SCENE_FORMAT);
        let depth = ArrayTarget::new(device, size, DEPTH_FORMAT);
        let reduced = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF reduced fragment records"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: MAX_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let reduced_view = reduced.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shaded_first = ColorTarget::new(
            device,
            "DOF shaded visible layer",
            size[0],
            size[1],
            SCENE_FORMAT,
        );
        let shade_bindings = create_post_bind_group(
            device,
            &post.bind_group_layout,
            &color.layers[0],
            &post.sampler,
            scene_depth,
            &post.uniform,
            &post.ao_filtered.view,
            &post.semantic.view,
            &post.scene.view,
        );
        let mask = std::array::from_fn(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("DOF disocclusion mask"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        });
        let mask_views: [_; 2] = std::array::from_fn(|i| mask[i].create_view(&Default::default()));
        let peel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF depth peeling inputs"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Depth),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let peel_bindings = (1..MAX_LAYERS)
            .map(|layer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("DOF previous layer and disocclusion"),
                    layout: &peel_layout,
                    entries: &[
                        view_entry(0, &depth.layers[layer as usize - 1]),
                        view_entry(1, &mask_views[0]),
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: post.uniform.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF peeling pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&peel_layout)],
            immediate_size: 0,
        });
        let geometry_shader = scene_shader(device, include_str!("shader.wgsl"));
        let toon_shader = scene_shader(device, include_str!("toon_sphere.wgsl"));
        // An explicit layout requires every listed group at draw time, even when
        // the entry point does not read it. Layer zero must not bind its own
        // depth attachment as a peeling input.
        let first_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF first layer pipeline layout"),
            bind_group_layouts: &[Some(camera_layout)],
            immediate_size: 0,
        });
        // Every layer is rasterized at the same resolution and sample positions.
        // The first-layer variants omit peeling but share the same vertex paths and shading.
        let first_geometry_pipeline =
            create_scene_geometry_pipeline(device, &first_layout, &geometry_shader, false, true);
        let first_cartoon_pipeline =
            create_cartoon_pipeline(device, &first_layout, &geometry_shader, false, true);
        let first_toon_pipeline =
            create_toon_pipeline(device, &first_layout, &toon_shader, false, true);
        let geometry_pipeline =
            create_scene_geometry_pipeline(device, &layout, &geometry_shader, true, true);
        let cartoon_pipeline =
            create_cartoon_pipeline(device, &layout, &geometry_shader, true, true);
        let toon_pipeline = create_toon_pipeline(device, &layout, &toon_shader, true, true);
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Franke tiled splatting"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                [
                    include_str!("optics.wgsl"),
                    include_str!("dof.wgsl"),
                    include_str!("dof_reduce.wgsl"),
                ]
                .concat(),
            )),
        });
        let mut stages = Vec::new();
        for (entry, inputs) in [
            ("edges", vec![(2, &depth.view), (4, &mask_views[0])]),
            ("dilate_x", vec![(3, &mask_views[0]), (4, &mask_views[1])]),
            ("dilate_y", vec![(3, &mask_views[1]), (4, &mask_views[0])]),
            (
                "reduce",
                vec![
                    (1, &color.view),
                    (2, &depth.view),
                    (3, &mask_views[0]),
                    (6, &shaded_first.view),
                    (7, &reduced_view),
                ],
            ),
            (
                "splat",
                vec![
                    (1, &color.view),
                    (2, &depth.view),
                    (3, &mask_views[0]),
                    (5, &post.dof_color.view),
                    (6, &shaded_first.view),
                    (8, &reduced_view),
                ],
            ),
            (
                "splat_reference",
                vec![
                    (1, &color.view),
                    (2, &depth.view),
                    (3, &mask_views[0]),
                    (5, &post.dof_color.view),
                    (6, &shaded_first.view),
                    (8, &reduced_view),
                ],
            ),
        ] {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &compute_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: post.uniform.as_entire_binding(),
            }];
            entries.extend(
                inputs
                    .into_iter()
                    .map(|(binding, view)| view_entry(binding, view)),
            );
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            stages.push(ComputeStage { pipeline, bindings });
        }
        Self {
            size,
            color,
            depth,
            reduced,
            shaded_first,
            shade_bindings,
            shade_pipeline: post.dof_shade_pipeline.clone(),
            _mask: mask,
            first_geometry_pipeline,
            first_cartoon_pipeline,
            first_toon_pipeline,
            geometry_pipeline,
            cartoon_pipeline,
            toon_pipeline,
            peel_bindings,
            stages,
        }
    }

    pub(super) fn layer_color(&self, layer: u32) -> &wgpu::TextureView {
        &self.color.layers[layer as usize]
    }
    pub(super) fn layer_depth(&self, layer: u32) -> &wgpu::TextureView {
        &self.depth.layers[layer as usize]
    }
    pub(super) fn peel_binding(&self, layer: u32) -> &wgpu::BindGroup {
        &self.peel_bindings[layer as usize - 1]
    }

    pub(super) fn encode_mask(&self, encoder: &mut wgpu::CommandEncoder) {
        for stage in &self.stages[..3] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DOF disocclusion"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bindings, &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        }
    }

    pub(super) fn encode_splat(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.encode_accumulation(encoder, timestamps, true);
    }

    // Kept for numerical/visual comparisons; the application uses reduction.
    #[allow(dead_code)]
    pub(super) fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_accumulation(encoder, None, false);
    }

    fn encode_accumulation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
        reduced: bool,
    ) {
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DOF shade visible fragments"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shaded_first.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.shade_pipeline);
            pass.set_bind_group(0, &self.shade_bindings, &[]);
            pass.draw(0..3, 0..1);
        }
        if reduced {
            let stage = &self.stages[3];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DOF list merging and umbra trimming"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, &stage.bindings, &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(32), self.size[1].div_ceil(32), 1);
        }
        let stage = &self.stages[if reduced { 4 } else { 5 }];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DOF tile sort and accumulation"),
            timestamp_writes: timestamps,
        });
        pass.set_pipeline(&stage.pipeline);
        pass.set_bind_group(0, &stage.bindings, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(16), self.size[1].div_ceil(16), 1);
    }

    pub(super) fn estimated_bytes(&self) -> u64 {
        u64::from(self.size[0]) * u64::from(self.size[1]) * (u64::from(MAX_LAYERS) * (8 + 4) + 16)
            + u64::from(self.reduced.width())
                * u64::from(self.reduced.height())
                * u64::from(MAX_LAYERS)
                * 16
    }
}

pub(super) fn scene_shader(device: &wgpu::Device, source: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("molecular shading with optional depth peeling"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(
            [
                include_str!("optics.wgsl"),
                include_str!("peel.wgsl"),
                source,
            ]
            .concat(),
        )),
    })
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}
