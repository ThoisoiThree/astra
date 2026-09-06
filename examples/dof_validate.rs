//! Manual GPU regression check: cargo run --example dof_validate
//! Kept outside unit tests so `cargo test` never requires an adapter or a window.
#![allow(dead_code)]
pub use astra::*;
#[path = "../src/render/cartoon.rs"]
mod cartoon;
#[path = "../src/render/dof.rs"]
mod dof;
#[path = "../src/render/instances.rs"]
mod instances;
#[path = "../src/render/mesh.rs"]
mod mesh;
#[path = "dof_validate/molecular.rs"]
mod molecular;
#[path = "../src/render/pipelines.rs"]
mod pipelines;
#[path = "../src/render/postprocess.rs"]
mod postprocess;
#[path = "dof_validate/reduction.rs"]
mod reduction;
#[path = "../src/render/targets.rs"]
mod targets;
mod renderer {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const SEMANTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
    pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
}

use anyhow::{Context, Result, ensure};
use glam::Vec3;
use postprocess::{PostProcess, PostUniform};
use wgpu::util::DeviceExt;

fn main() -> Result<()> {
    pollster::block_on(run())
}

async fn run() -> Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    println!("GPU: {}", adapter.get_info().name);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
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
    if std::env::var_os("ASTRA_DOF_MOLECULAR_ONLY").is_some() {
        return molecular::validate(&device, &queue, &camera_layout);
    }
    for (width, height, scale, blades, rotation, layers, oracle) in [
        (65, 49, 1.0, 7.0, 0.37, 5, false),
        (65, 49, 0.5, 0.0, 0.0, 4, false),
        (65, 49, 1.0, 3.0, 1.2, 3, false),
        (1, 1, 0.5, 12.0, -0.8, 5, false),
        (17, 13, 1.0, 0.0, 0.0, 5, true),
    ] {
        let depth = targets::DepthTarget::new(&device, width, height);
        let mut post = PostProcess::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            width,
            height,
            &depth.view,
        );
        post.set_dof_scale(&device, width, height, &depth.view, scale);
        let dof = dof::DepthOfField::new(&device, &camera_layout, &post, &depth.view);
        let projection =
            glam::camera::rh::proj::directx::orthographic(-1.0, 1.0, -1.0, 1.0, 1.0, 101.0);
        let mut uniform = PostUniform {
            inverse_view_projection: projection.inverse().to_cols_array(),
            eye_position: [0.0; 4],
            optical_axis: [0.0, 0.0, -1.0, height as f32],
            lens: [5.0, 0.7, 64.0, 1.0],
            aperture: [2.5, blades, rotation, 0.0],
            ao: [0.0; 4],
            quality: [
                layers as f32,
                dof.size[1] as f32 / height as f32,
                dof.size[0] as f32,
                dof.size[1] as f32,
            ],
            background: [0.0, 1.0, 0.0, 1.0],
            viewport: [0.0, 0.0, 1.0, 1.0],
        };
        for mode in 0..6 {
            uniform.lens[0] = if matches!(mode, 2 | 3 | 5) { 20.0 } else { 5.0 };
            let patterned = mode == 1 || mode == 2;
            queue.write_buffer(&post.uniform, 0, bytemuck::bytes_of(&uniform));
            let fixture = fixture_pipeline(&device, mode, dof.size[0]);
            let peel = peel_fixture_pipeline(&device, &camera_layout, &dof);
            let dummy_camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &[0u8; 176],
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let camera_binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &camera_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dummy_camera.as_entire_binding(),
                }],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            for layer in 0..5 {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DOF regression fixture"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dof.layer_color(layer),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.25,
                                g: 0.5,
                                b: 0.75,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: dof.layer_depth(layer),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(if layer == 0 { 0.19 } else { 1.0 }),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if layer == 0 {
                    pass.set_pipeline(&fixture);
                    pass.draw(0..3, 0..1);
                }
            }
            dof.encode_mask(&mut encoder);
            if mode == 3 {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DOF hidden blue plane"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: dof.layer_color(1),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: dof.layer_depth(1),
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
                pass.set_pipeline(&peel);
                pass.set_bind_group(0, &camera_binding, &[]);
                pass.set_bind_group(1, dof.peel_binding(1), &[]);
                pass.draw(0..3, 0..1);
            }
            if oracle {
                ensure!(
                    dof.encode_ordering_oracle(&mut encoder),
                    "oracle target too large"
                );
            } else {
                dof.encode_splat(&mut encoder, None);
            }
            let bytes_per_row = (dof.size[0] * 8).div_ceil(256) * 256;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DOF regression readback"),
                size: u64::from(bytes_per_row * dof.size[1]),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &post.dof_color._texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: dof.size[0],
                    height: dof.size[1],
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = sender.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely())?;
            receiver.recv().context("GPU readback callback")??;
            let bytes = buffer.get_mapped_range(..)?;
            let mut maximum_error = 0.0_f32;
            for y in 0..dof.size[1] {
                for x in 0..dof.size[0] {
                    let offset = (y * bytes_per_row + x * 8) as usize;
                    let actual = Vec3::from_array(std::array::from_fn(|c| {
                        decode_half(u16::from_le_bytes([
                            bytes[offset + c * 2],
                            bytes[offset + c * 2 + 1],
                        ]))
                    }));
                    ensure!(actual.is_finite(), "nonfinite DOF at {x},{y}");
                    let expected = if mode >= 4 {
                        gradient_reference(x, y, dof.size, &uniform)
                    } else if mode == 3 {
                        partial_reference(x, y, dof.size, &uniform)
                    } else if mode == 2 {
                        if (x + y).is_multiple_of(2) {
                            Vec3::X
                        } else {
                            Vec3::Z
                        }
                    } else if patterned {
                        reference_pixel(x, y, dof.size, &uniform)
                    } else {
                        Vec3::new(0.25, 0.5, 0.75)
                    };
                    maximum_error = maximum_error.max((actual - expected).abs().max_element());
                }
            }
            ensure!(
                maximum_error < 0.008,
                "DOF differs from exhaustive ordered reference by {maximum_error}"
            );
            println!(
                "{width}x{height}, scale {scale}, iris {blades}, mode {mode}: max error {maximum_error:.6}"
            );
            drop(bytes);
            buffer.unmap();
        }
    }
    reduction::validate(&device, &queue, &camera_layout)?;
    molecular::validate(&device, &queue, &camera_layout)?;
    Ok(())
}

fn fixture_pipeline(device: &wgpu::Device, mode: u32, width: u32) -> wgpu::RenderPipeline {
    let source = format!(
        r#"
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    return vec4<f32>(p[i], 0.19, 1.0);
}}
struct Output {{ @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 }};
@fragment fn fragment_main(@builtin(position) p: vec4<f32>) -> Output {{
    var o: Output;
    o.depth = 0.19;
    o.color = vec4<f32>(0.25, 0.5, 0.75, 1);
    if {mode}u == 1u || {mode}u == 2u {{ o.color = select(vec4<f32>(0,0,1,1), vec4<f32>(1,0,0,1), ((u32(p.x) + u32(p.y)) % 2u) == 0u); }}
    if {mode}u == 3u {{
        o.color = vec4<f32>(0,0,1,1);
        if u32(p.x) >= {left}u && u32(p.x) < {right}u {{
            o.color = vec4<f32>(select(0.8, 1.0, ((u32(p.x) + u32(p.y)) % 2u) == 0u), 0, 0, 1);
            o.depth = 0.09;
        }}
    }}
    if {mode}u >= 4u {{
        // Descending depth makes the depth order oppose source-pixel order.
        // Color changes are small enough to expose fixed-grid merge artifacts.
        let z = 16.0 + f32({width}u - 1u - u32(p.x)) * 0.125;
        o.depth = (z - 1.0) / 100.0;
        o.color = vec4<f32>(0.25 + f32(u32(p.x)) / 1024.0,
            0.5 + f32(u32(p.y)) / 1024.0, 0.75, 1.0);
    }}
    return o;
}}
"#,
        left = width * 2 / 5,
        right = width * 3 / 5
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
    })
}

// Exhaustive CPU reference has no tiling, reduction, list capacity or radix partitioning.
fn reference_pixel(x: u32, y: u32, size: [u32; 2], post: &PostUniform) -> Vec3 {
    let f = post.aperture[0];
    let r = (0.5 * f * f * (1.0 - post.lens[0] / 20.0) / (post.lens[1] * (post.lens[0] - f))
        * post.optical_axis[3])
        .abs()
        .min(post.lens[2])
        * post.quality[1];
    let mut rgb = Vec3::ZERO;
    let mut transmission = 1.0;
    for sy in 0..size[1] {
        for sx in 0..size[0] {
            let dx = x as f32 - sx as f32;
            let dy = y as f32 - sy as f32;
            let distance = aperture_distance(dx, dy, post);
            let t =
                ((distance - (r - 0.5).max(0.0)) / (r + 0.5 - (r - 0.5).max(0.0))).clamp(0.0, 1.0);
            let coverage = 1.0 - t * t * (3.0 - 2.0 * t);
            let alpha = (1.0 / (r * r)).min(1.0) * coverage;
            let color = if (sx + sy).is_multiple_of(2) {
                Vec3::X
            } else {
                Vec3::Z
            };
            let weight = transmission * alpha;
            rgb += weight * color;
            transmission -= weight;
        }
    }
    rgb / (1.0 - transmission).max(1e-6)
}

// A smooth sloped surface exercises small color/depth differences, exact depth
// ordering and the transition through subpixel CoC. The CPU enumerates sources
// in known physical depth order, independently of the GPU radix traversal.
fn gradient_reference(x: u32, y: u32, size: [u32; 2], post: &PostUniform) -> Vec3 {
    let mut rgb = Vec3::ZERO;
    let mut transmission = 1.0;
    let f = post.aperture[0];
    for sx in (0..size[0]).rev() {
        let z = 16.0 + (size[0] - 1 - sx) as f32 * 0.125;
        let r = (0.5 * f * f * (1.0 - post.lens[0] / z) / (post.lens[1] * (post.lens[0] - f))
            * post.optical_axis[3])
            .abs()
            .min(post.lens[2])
            * post.quality[1];
        for sy in 0..size[1] {
            let coverage = if r < 0.5 {
                f32::from(sx == x && sy == y)
            } else {
                let distance = aperture_distance(x as f32 - sx as f32, y as f32 - sy as f32, post);
                let t = ((distance - (r - 0.5)) / 1.0).clamp(0.0, 1.0);
                1.0 - t * t * (3.0 - 2.0 * t)
            };
            let alpha = (1.0 / (r * r).max(0.25)).min(1.0) * coverage;
            let color = Vec3::new(0.25 + sx as f32 / 1024.0, 0.5 + sy as f32 / 1024.0, 0.75);
            rgb += transmission * alpha * color;
            transmission *= 1.0 - alpha;
        }
    }
    rgb / (1.0 - transmission).max(1e-6)
}

fn decode_half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let mantissa = f32::from(bits & 1023) / 1024.0;
    if exponent == 0 {
        sign * mantissa * 2.0_f32.powi(-14)
    } else if exponent == 31 {
        f32::NAN
    } else {
        sign * (1.0 + mantissa) * 2.0_f32.powi(i32::from(exponent) - 15)
    }
}

fn peel_fixture_pipeline(
    device: &wgpu::Device,
    camera: &wgpu::BindGroupLayout,
    dof: &dof::DepthOfField,
) -> wgpu::RenderPipeline {
    let source = [
        include_str!("../src/render/optics.wgsl"),
        include_str!("../src/render/peel.wgsl"),
        r#"
@vertex fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
    return vec4<f32>(p[i], 0.19, 1.0);
}
@fragment fn fragment_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    reject_peeled_fragment(p.xy, p.z);
    return vec4<f32>(0, 0, 1, 1);
}
"#,
    ]
    .concat();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[
            Some(camera),
            Some(&dof.geometry_pipeline.get_bind_group_layout(1)),
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&layout),
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
            depth_compare: Some(wgpu::CompareFunction::Less),
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
    })
}

fn partial_reference(x: u32, y: u32, size: [u32; 2], post: &PostUniform) -> Vec3 {
    let f = post.aperture[0];
    let r = (0.5 * f * f * (1.0 - post.lens[0] / 10.0) / (post.lens[1] * (post.lens[0] - f))
        * post.optical_axis[3])
        .abs()
        .min(post.lens[2])
        * post.quality[1];
    let mut rgb = Vec3::ZERO;
    let mut transmission = 1.0;
    for sy in 0..size[1] {
        for sx in (size[0] * 2 / 5)..(size[0] * 3 / 5) {
            let dx = x as f32 - sx as f32;
            let dy = y as f32 - sy as f32;
            let distance = aperture_distance(dx, dy, post);
            let t =
                ((distance - (r - 0.5).max(0.0)) / (r + 0.5 - (r - 0.5).max(0.0))).clamp(0.0, 1.0);
            let alpha = (1.0 / (r * r)).min(1.0) * (1.0 - t * t * (3.0 - 2.0 * t));
            let red = if (sx + sy).is_multiple_of(2) {
                1.0
            } else {
                decode_half(0x3a66)
            };
            let weight = transmission * alpha;
            rgb.x += weight * red;
            transmission -= weight;
        }
    }
    rgb + transmission * Vec3::Z
}

fn aperture_distance(dx: f32, dy: f32, post: &PostUniform) -> f32 {
    if post.aperture[1] < 3.0 {
        return dx.hypot(dy);
    }
    let angle = dy.atan2(dx) - post.aperture[2];
    let sector = std::f32::consts::TAU / post.aperture[1];
    let local = angle - sector * (angle / sector + 0.5).floor();
    let boundary = (std::f32::consts::PI / post.aperture[1]).cos() / local.cos();
    dx.hypot(dy) / boundary
}
