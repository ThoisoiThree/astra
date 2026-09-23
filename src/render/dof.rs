//! Partial multilayer generation and tiled, depth-ordered splatting (Franke et al., 2018).
use std::borrow::Cow;

use super::{
    pipelines::{ScenePipelines, SceneTarget},
    postprocess::{PostProcess, create_post_bind_group},
    renderer::{DEPTH_FORMAT, SCENE_FORMAT},
    targets::ColorTarget,
};

pub(super) const MAX_DOF_LAYERS: u32 = 5;
const ORACLE_MAX_DIMENSION: u32 = 128;

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
                depth_or_array_layers: MAX_DOF_LAYERS,
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
        let layers = (0..MAX_DOF_LAYERS)
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
    block_counts: wgpu::Texture,
    block_sources: wgpu::Texture,
    shaded_first: ColorTarget,
    shade_bindings: wgpu::BindGroup,
    shade_pipeline: wgpu::RenderPipeline,
    _mask: [wgpu::Texture; 2],
    /// First layer: plain color output at the peeling resolution.
    pub(super) first_pipelines: ScenePipelines,
    /// Later layers reject fragments captured by the previous layer.
    pub(super) peel_pipelines: ScenePipelines,
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
        Self::new_internal(device, camera_layout, post, scene_depth, false)
    }

    /// Validation tools opt into the expensive comparison pipelines.
    #[allow(dead_code)]
    pub(super) fn new_validation(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        post: &PostProcess,
        scene_depth: &wgpu::TextureView,
    ) -> Self {
        Self::new_internal(device, camera_layout, post, scene_depth, true)
    }

    fn new_internal(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        post: &PostProcess,
        scene_depth: &wgpu::TextureView,
        validation: bool,
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
                depth_or_array_layers: MAX_DOF_LAYERS,
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
        // A 4x4 source block has at most 16 * MAX_DOF_LAYERS surviving
        // fragments. Four IDs per texel, in a 2x2 array footprint, store that
        // exact upper bound without a global append buffer or overflow drops.
        let blocks = [size[0].div_ceil(4), size[1].div_ceil(4)];
        let block_counts = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF compact block counts"),
            size: wgpu::Extent3d {
                width: blocks[0],
                height: blocks[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let block_sources = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF compact block source IDs"),
            size: wgpu::Extent3d {
                width: blocks[0] * 2,
                height: blocks[1] * 2,
                depth_or_array_layers: MAX_DOF_LAYERS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let block_counts_view = block_counts.create_view(&Default::default());
        let block_sources_view = block_sources.create_view(&wgpu::TextureViewDescriptor {
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
            &post.overlay.view,
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
        let peel_bindings = (1..MAX_DOF_LAYERS)
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
        let impostor_shader = scene_shader(device, include_str!("impostor.wgsl"));
        let mesh_shader = scene_shader(device, include_str!("shader.wgsl"));
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
        let first_pipelines = ScenePipelines::new(
            device,
            &first_layout,
            &impostor_shader,
            &mesh_shader,
            SceneTarget::Color,
        );
        let peel_pipelines = ScenePipelines::new(
            device,
            &layout,
            &impostor_shader,
            &mesh_shader,
            SceneTarget::Peel,
        );
        // Inject the host-side layer bound into WGSL so texture allocation and shader
        // traversal cannot silently diverge. post.quality.x may request fewer layers,
        // but the shader clamps it to this allocation bound.
        let compute_source = format!(
            "const HOST_MAX_DOF_LAYERS: u32 = {MAX_DOF_LAYERS}u;\n{}",
            [
                include_str!("optics.wgsl"),
                include_str!("dof.wgsl"),
                include_str!("dof_reduce.wgsl"),
            ]
            .concat(),
        );
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Franke tiled splatting"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(compute_source)),
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
                    (11, &block_counts_view),
                    (12, &block_sources_view),
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
                    (9, &block_counts_view),
                    (10, &block_sources_view),
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
                    (9, &block_counts_view),
                    (10, &block_sources_view),
                ],
            ),
            (
                "splat_oracle",
                vec![
                    (1, &color.view),
                    (2, &depth.view),
                    (3, &mask_views[0]),
                    (5, &post.dof_color.view),
                    (6, &shaded_first.view),
                ],
            ),
            (
                "splat_dense",
                vec![
                    (1, &color.view),
                    (2, &depth.view),
                    (3, &mask_views[0]),
                    (5, &post.dof_color.view),
                    (6, &shaded_first.view),
                    (8, &reduced_view),
                    (9, &block_counts_view),
                    (10, &block_sources_view),
                ],
            ),
        ]
        .into_iter()
        .take(if validation { 8 } else { 5 })
        {
            crate::diagnostics::write_graphics_log(format_args!(
                "BEGIN DoF compute pipeline: {entry}"
            ));
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &compute_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            crate::diagnostics::write_graphics_log(format_args!(
                "END DoF compute pipeline: {entry}"
            ));
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
            block_counts,
            block_sources,
            shaded_first,
            shade_bindings,
            shade_pipeline: post.dof_shade_pipeline.clone(),
            _mask: mask,
            first_pipelines,
            peel_pipelines,
            peel_bindings,
            stages,
        }
    }

    pub(super) fn layer_color(&self, layer: u32) -> &wgpu::TextureView {
        assert!(
            layer < MAX_DOF_LAYERS,
            "DOF color layer {layer} exceeds allocation"
        );
        &self.color.layers[layer as usize]
    }
    pub(super) fn layer_depth(&self, layer: u32) -> &wgpu::TextureView {
        assert!(
            layer < MAX_DOF_LAYERS,
            "DOF depth layer {layer} exceeds allocation"
        );
        &self.depth.layers[layer as usize]
    }
    pub(super) fn peel_binding(&self, layer: u32) -> &wgpu::BindGroup {
        assert!(
            (1..MAX_DOF_LAYERS).contains(&layer),
            "DOF peel binding is only valid for layers 1..{}",
            MAX_DOF_LAYERS - 1
        );
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

    #[allow(dead_code)]
    pub(super) fn encode_splat(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.encode_accumulation(encoder, timestamps, None, true);
    }

    pub(super) fn encode_splat_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        reduction: Option<wgpu::ComputePassTimestampWrites<'_>>,
        splat: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        self.encode_accumulation(encoder, splat, reduction, true);
    }

    // No-reduction comparison path. This is useful for validating fragment reduction,
    // but it is not an independent accumulation oracle because it shares sorting/compositing.
    #[allow(dead_code)]
    pub(super) fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_accumulation(encoder, None, None, false);
    }

    /// Run after encode_splat, reusing its shaded/reduced textures unchanged.
    #[allow(dead_code)]
    pub(super) fn encode_dense_comparison(&self, encoder: &mut wgpu::CommandEncoder) {
        let stage = &self.stages[7];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DOF dense compaction baseline"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&stage.pipeline);
        pass.set_bind_group(0, &stage.bindings, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(16), self.size[1].div_ceil(16), 1);
    }

    /// Extremely slow but ordering-independent validation path. It deliberately
    /// bypasses reduction, tile candidate partitioning and bitonic sorting.
    /// Returns false rather than accidentally dispatching the O(N^2) oracle on
    /// a production-sized render target.
    #[allow(dead_code)]
    pub(super) fn encode_ordering_oracle(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if self.size[0] > ORACLE_MAX_DIMENSION || self.size[1] > ORACLE_MAX_DIMENSION {
            return false;
        }

        self.encode_visible_shading(encoder, None);
        let stage = &self.stages[6];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("DOF independent ordering oracle"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&stage.pipeline);
        pass.set_bind_group(0, &stage.bindings, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        true
    }

    fn encode_visible_shading(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DOF shade visible fragments"),
            timestamp_writes: timestamps,
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

    fn encode_accumulation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
        mut reduction_timestamps: Option<wgpu::ComputePassTimestampWrites<'_>>,
        reduced: bool,
    ) {
        let shading_timestamps =
            reduction_timestamps
                .as_ref()
                .map(|t| wgpu::RenderPassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: t.beginning_of_pass_write_index,
                    end_of_pass_write_index: None,
                });
        if let Some(t) = &mut reduction_timestamps {
            t.beginning_of_pass_write_index = None;
        }
        self.encode_visible_shading(encoder, shading_timestamps);
        if reduced {
            let stage = &self.stages[3];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DOF hierarchical fragment merging"),
                timestamp_writes: reduction_timestamps,
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
        u64::from(self.size[0])
            * u64::from(self.size[1])
            * (u64::from(MAX_DOF_LAYERS) * (8 + 4) + 16)
            + u64::from(self.reduced.width())
                * u64::from(self.reduced.height())
                * u64::from(MAX_DOF_LAYERS)
                * 16
            + u64::from(self.block_counts.width()) * u64::from(self.block_counts.height()) * 4
            + u64::from(self.block_sources.width())
                * u64::from(self.block_sources.height())
                * u64::from(MAX_DOF_LAYERS)
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
                include_str!("scene_common.wgsl"),
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

#[cfg(test)]
mod tests {
    #[test]
    fn merged_opacity_matches_supplemental_geometric_sum() {
        for radius in [1.0_f32, 2.0, 4.0, 8.0, 16.0] {
            let alpha = (1.0 / (radius * radius)).min(1.0);
            for mass in [1, 4, 16] {
                let sum: f32 = (0..mass).map(|k| alpha * (1.0 - alpha).powi(k)).sum();
                let closed = 1.0 - (1.0 - alpha).powi(mass);
                assert!((sum - closed).abs() < 1e-6);
            }
        }
    }
}
