//! Structural checks for section 6, independent of the color accumulation model.
use super::*;

pub(super) fn validate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera: &wgpu::BindGroupLayout,
) -> Result<()> {
    let size = 8;
    let depth = targets::DepthTarget::new(device, size, size);
    let mut post = PostProcess::new(device, renderer::SCENE_FORMAT, size, size, &depth.view);
    post.set_dof_scale(device, size, size, &depth.view, 1.0);
    let dof = dof::DepthOfField::new(device, camera, &post, &depth.view);
    let projection =
        glam::camera::rh::proj::directx::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 101.0);
    for case in 0..5 {
        let uniform = PostUniform {
            inverse_view_projection: projection.inverse().to_cols_array(),
            eye_position: [0.0; 4],
            optical_axis: [0.0, 0.0, -1.0, size as f32],
            lens: [
                if case == 2 { 20.0 } else { 5.0 },
                if case == 3 { 0.8 } else { 0.7 },
                64.0,
                1.0,
            ],
            aperture: [2.5, 0.0, 0.0, 0.0],
            ao: [0.0; 4],
            quality: [5.0, 1.0, size as f32, size as f32],
            background: [0.0; 4],
            viewport: [0.0, 0.0, 1.0, 1.0],
        };
        queue.write_buffer(&post.uniform, 0, bytemuck::bytes_of(&uniform));
        let mut encoder = device.create_command_encoder(&Default::default());
        for layer in 0..5 {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("section 6 fixture"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        r#"
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let p = array<vec2<f32>,3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    return vec4<f32>(p[i], 0, 1);
}}
struct Out {{ @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 }};
@fragment fn fragment_main(@builtin(position) p: vec4<f32>) -> Out {{
    var o: Out;
    o.color = vec4<f32>(0,0,0,1); o.depth = 1.0;
    if {case}u == 4u {{
        // Environment has radiance but no physical footprint to merge.
        o.color = vec4<f32>(0.2,0.4,0.6,1);
    }} else if {case}u == 3u {{
        if p.x < 4.0 && p.y < 4.0 {{
            if {layer}u == 0u {{ o.color = vec4<f32>(0,0,1,1); o.depth = 0.19; }}
            if {layer}u == 1u {{ o.color = vec4<f32>(0,1,0,1); o.depth = 0.79; }}
        }} else if {layer}u == 0u {{ o.color = vec4<f32>(1,0,0,1); o.depth = 0.09; }}
    }} else if {case}u != 1u {{
        if {layer}u == 0u {{ o.color = vec4<f32>(0,0,1,1); o.depth = 0.19; }}
    }} else if p.x < 2.0 && p.y < 2.0 {{
        // The same blue surface is in layer 0 or 1 depending on the pixel.
        let occluder = ((u32(p.x) + u32(p.y)) % 2u) == 0u;
        let surface = i32({layer}) - select(0,1,occluder);
        if surface == -1 {{ o.color = vec4<f32>(1,0,0,1); o.depth = 0.09; }}
        if surface == 0 {{ o.color = vec4<f32>(0,0,1,1); o.depth = 0.19; }}
        if surface == 1 {{ o.color = vec4<f32>(0,1,0,1); o.depth = 0.27; }}
        if surface == 2 {{ o.color = vec4<f32>(1,1,0,1); o.depth = 0.79; }}
    }}
    return o;
}}
"#
                    )
                    .into(),
                ),
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                multisample: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: renderer::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: renderer::SCENE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("section 6 prepared layer"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dof.layer_color(layer),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dof.layer_depth(layer),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        dof.encode_mask(&mut encoder);
        dof.encode_splat(&mut encoder, None);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("section 6 records"),
            size: 256 * 8 * 5,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &dof.reduced,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 5,
            },
        );
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;
        let bytes = buffer.get_mapped_range(..)?;
        let mut masses = Vec::new();
        let mut depths = Vec::new();
        for layer in 0..5 {
            for y in 0..8 {
                for x in 0..8 {
                    let offset = (layer * 8 + y) * 256 + x * 16;
                    let mass =
                        decode_half(u16::from_le_bytes([bytes[offset + 6], bytes[offset + 7]]));
                    if mass == 0.0 {
                        continue;
                    }
                    masses.push(mass);
                    depths.push(f32::from_le_bytes(
                        bytes[offset + 8..offset + 12].try_into()?,
                    ));
                }
            }
        }
        match case {
            0 => ensure!(
                masses == vec![16.0; 4],
                "constant 8x8 surface must collapse to four 4x4 records: {masses:?}"
            ),
            1 => {
                let blue: Vec<_> = depths
                    .iter()
                    .zip(&masses)
                    .filter(|(z, _)| (20.0..20.01).contains(*z))
                    .collect();
                ensure!(
                    blue.len() == 1 && *blue[0].1 == 4.0,
                    "merge must match heads across different layer numbers"
                );
                ensure!(
                    !depths.iter().any(|z| (27.99..28.01).contains(z)),
                    "merged umbra must discard the green surface"
                );
                ensure!(
                    depths.iter().any(|z| (79.99..80.01).contains(z)),
                    "umbra must preserve the distant surface"
                );
            }
            2 => ensure!(
                masses == vec![1.0; 64],
                "focused fragments must remain individual pixels"
            ),
            3 => {
                ensure!(
                    depths
                        .iter()
                        .zip(&masses)
                        .any(|(z, m)| (20.0..20.01).contains(z) && *m == 16.0),
                    "4x4 blue surface must merge"
                );
                ensure!(
                    !depths.iter().any(|z| (79.99..80.01).contains(z)),
                    "4x4 footprint larger than aperture must cast infinite shadow"
                );
            }
            _ => ensure!(
                masses == vec![1.0; 64],
                "environment samples must not merge"
            ),
        }
        println!("section 6 case {case}: {} surviving records", masses.len());
    }
    Ok(())
}
