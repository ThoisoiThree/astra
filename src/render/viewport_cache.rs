use super::{pipelines::create_fullscreen_pipeline, targets::ColorTarget};

#[derive(Default)]
pub(super) struct SceneRevision {
    key: Vec<u8>,
    dirty: bool,
}

impl SceneRevision {
    pub(super) fn invalidate(&mut self) {
        self.dirty = true;
    }
    pub(super) fn changed(&self, key: &[u8]) -> bool {
        self.dirty || self.key != key
    }
    pub(super) fn commit(&mut self, key: Vec<u8>) {
        self.key = key;
        self.dirty = false;
    }
}

pub(super) struct ViewportCache {
    pub(super) color: ColorTarget,
    pub(super) revision: SceneRevision,
    pipeline: wgpu::RenderPipeline,
    bindings: wgpu::BindGroup,
}

impl ViewportCache {
    pub(super) fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let color = ColorTarget::new(device, "cached molecular viewport", width, height, format);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cached viewport blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("viewport_cache.wgsl").into()),
        });
        let bindings_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings_layout)],
            immediate_size: 0,
        });
        let pipeline = create_fullscreen_pipeline(
            device,
            "cached viewport blit",
            &layout,
            &shader,
            "fragment_main",
            format,
        );
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bindings_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color.view),
            }],
        });
        Self {
            color,
            revision: SceneRevision::default(),
            pipeline,
            bindings,
        }
    }

    pub(super) fn blit(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("restore cached molecular viewport"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::SceneRevision;

    #[test]
    fn ui_frames_reuse_scene_but_camera_and_geometry_changes_invalidate_it() {
        let mut revision = SceneRevision::default();
        let camera_and_lens = vec![1, 2, 3];
        assert!(revision.changed(&camera_and_lens));
        revision.commit(camera_and_lens.clone());
        for _ in 0..10 {
            assert!(!revision.changed(&camera_and_lens));
        }
        assert!(revision.changed(&[1, 2, 4]));
        revision.invalidate();
        assert!(revision.changed(&camera_and_lens));
        // Failed/occluded presentations must not commit a new revision.
        assert!(revision.changed(&camera_and_lens));
        revision.commit(camera_and_lens.clone());
        assert!(!revision.changed(&camera_and_lens));
    }
}
