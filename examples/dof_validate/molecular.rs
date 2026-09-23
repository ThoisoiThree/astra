//! Actual molecular draws: pipeline creation alone cannot validate render-pass bindings.
use std::{fs, io::Write};

use super::*;

pub(super) fn validate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera_layout: &wgpu::BindGroupLayout,
) -> Result<()> {
    for (name, source, mode, scale) in [
        (
            "4r8p",
            include_str!("../4R8P.pdb"),
            DisplayMode::Cartoon,
            1.0,
        ),
        (
            "4r8p-preview",
            include_str!("../4R8P.pdb"),
            DisplayMode::Cartoon,
            0.5,
        ),
        (
            "atoms",
            include_str!("../minimal.pdb"),
            DisplayMode::BallAndStick,
            1.0,
        ),
        (
            "toon",
            include_str!("../minimal.pdb"),
            DisplayMode::Toon,
            1.0,
        ),
    ] {
        let molecule = molecule::parse_pdb(source)?;
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_global_mode(mode);
        let data = cartoon::cartoon_render_data(&molecule, &display);
        let styles = instances::atomic_styles(&display, &data.atomic);
        let spheres = instances::sphere_instances(&molecule, &display, &styles);
        let bonds = instances::bond_instances(&molecule, &display, &styles);
        let vertices = vertex_buffer(device, &data.vertices);
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molecular regression indices"),
            contents: if data.indices.is_empty() {
                &[0; 4]
            } else {
                bytemuck::cast_slice(&data.indices)
            },
            usage: wgpu::BufferUsages::INDEX,
        });
        let sphere_buffer = vertex_buffer(device, &spheres);
        let bond_buffer = vertex_buffer(device, &bonds);
        let positions = storage_buffer(device, &instances::atom_positions(&molecule));
        let colors = storage_buffer(device, &instances::atom_colors(&display));
        let meta = storage_buffer(device, &instances::atom_meta(&display, &data.semantic_ids));
        let (width, height) = (641, 513);
        let depth = targets::DepthTarget::new(device, width, height);
        let mut post = PostProcess::new(device, renderer::SCENE_FORMAT, width, height, &depth.view);
        post.set_dof_scale(device, width, height, &depth.view, scale);
        let dof = dof::DepthOfField::new_validation(device, camera_layout, &post, &depth.view);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molecular regression scene layout"),
            bind_group_layouts: &[Some(camera_layout)],
            immediate_size: 0,
        });
        let mesh_shader = dof::scene_shader(device, include_str!("../../src/render/shader.wgsl"));
        let impostor_shader =
            dof::scene_shader(device, include_str!("../../src/render/impostor.wgsl"));
        let scene_pipelines = pipelines::ScenePipelines::new(
            device,
            &layout,
            &impostor_shader,
            &mesh_shader,
            pipelines::SceneTarget::Scene,
        );
        let composed = targets::ColorTarget::new_storage(
            device,
            "molecular regression composition",
            width,
            height,
            renderer::SCENE_FORMAT,
        );
        // Like the application's side panel and tab bar, this moves the camera
        // viewport away from the texture origin. Odd sizes also exercise rounded
        // Preview dimensions rather than an assumed exact 0.5 scale.
        let viewport = [19.0, 11.0, width as f32 - 37.0, height as f32 - 29.0];
        let mut camera = camera::OrbitCamera::new(viewport[2] / viewport[3]);
        let (minimum, maximum) = molecule.bounds().context("empty molecular fixture")?;
        camera.fit_bounds(minimum, maximum);
        camera.distance *= 0.7;
        camera.depth_of_field.focus_point =
            camera.target + camera.optical_axis() * (maximum - minimum).length() * 0.25;
        let forward = camera.optical_axis();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();
        let projection = camera.view_projection();
        let camera_data = [
            projection.to_cols_array().to_vec(),
            projection.inverse().to_cols_array().to_vec(),
            camera.eye().extend(1.0).to_array().to_vec(),
            right.extend(0.0).to_array().to_vec(),
            up.extend(0.0).to_array().to_vec(),
        ]
        .concat();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molecular regression camera"),
            contents: bytemuck::cast_slice(&camera_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let binding = scene_binding(
            device,
            camera_layout,
            &camera_buffer,
            [&positions, &colors, &meta],
        );
        let background = wgpu::Color {
            r: 0.012286,
            g: 0.015209,
            b: 0.016807,
            a: 1.0,
        };
        let mut sharp = Vec::new();
        let mut reduced: Vec<Vec3> = Vec::new();
        for (label, f_stop) in [
            ("sharp", 1e6),
            ("ao", 1e6),
            ("dof", 0.7),
            ("reference", 0.7),
        ] {
            let uniform = PostUniform {
                inverse_view_projection: projection.inverse().to_cols_array(),
                eye_position: camera.eye().extend(1.0).to_array(),
                optical_axis: forward.extend(viewport[3]).to_array(),
                lens: [camera.focus_depth(), f_stop, 24.0, 1.0],
                aperture: [2.5, 7.0, 0.0, f32::from(mode == DisplayMode::Toon)],
                ao: [0.0, 0.0, 0.0, f32::from(label == "ao")],
                quality: [
                    4.0,
                    dof.size[1] as f32 / height as f32,
                    dof.size[0] as f32,
                    dof.size[1] as f32,
                ],
                background: [
                    background.r as f32,
                    background.g as f32,
                    background.b as f32,
                    1.0,
                ],
                viewport: [
                    viewport[0] / width as f32,
                    viewport[1] / height as f32,
                    viewport[2] / width as f32,
                    viewport[3] / height as f32,
                ],
                output: [1.0, 0.0, 1.0, 0.0],
            };
            queue.write_buffer(&post.uniform, 0, bytemuck::bytes_of(&uniform));
            let mut encoder = device.create_command_encoder(&Default::default());
            // A constant AO field tests that source shading is applied exactly
            // once, before splatting, rather than again in final composition.
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("regression half-intensity AO"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &post.ao_filtered.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 0.5,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
            for layer in -1..4 {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("molecular regression layer"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: if layer < 0 {
                                &post.scene.view
                            } else {
                                dof.layer_color(layer as u32)
                            },
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(background),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        (layer < 0).then_some(wgpu::RenderPassColorAttachment {
                            view: &post.semantic.view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: if layer < 0 {
                            &depth.view
                        } else {
                            dof.layer_depth(layer as u32)
                        },
                        stencil_ops: None,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }),
                    ..Default::default()
                });
                pass.set_bind_group(0, &binding, &[]);
                let sx = if layer < 0 {
                    1.0
                } else {
                    dof.size[0] as f32 / width as f32
                };
                let sy = if layer < 0 {
                    1.0
                } else {
                    dof.size[1] as f32 / height as f32
                };
                pass.set_viewport(
                    viewport[0] * sx,
                    viewport[1] * sy,
                    viewport[2] * sx,
                    viewport[3] * sy,
                    0.0,
                    1.0,
                );
                if layer > 0 {
                    pass.set_bind_group(1, dof.peel_binding(layer as u32), &[]);
                }
                let pipelines = if layer < 0 {
                    &scene_pipelines
                } else if layer == 0 {
                    &dof.first_pipelines
                } else {
                    &dof.peel_pipelines
                };
                pass.set_pipeline(&pipelines.meshes);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..data.indices.len() as u32, 0, 0..1);
                pass.set_pipeline(&pipelines.bonds);
                pass.set_vertex_buffer(0, bond_buffer.slice(..));
                pass.draw(0..36, 0..bonds.len() as u32);
                pass.set_pipeline(&pipelines.spheres);
                pass.set_vertex_buffer(0, sphere_buffer.slice(..));
                pass.draw(0..6, 0..spheres.len() as u32);
                drop(pass);
                if layer == 0 {
                    dof.encode_mask(&mut encoder);
                }
            }
            if label == "reference" {
                dof.encode_reference(&mut encoder);
            } else {
                dof.encode_splat(&mut encoder, None);
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("molecular regression final composition"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &composed.view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(background),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                pass.set_pipeline(&post.pipeline);
                pass.set_bind_group(0, &post.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            let pixels = readback(device, queue, encoder, &composed._texture)?;
            ensure!(
                pixels.iter().all(|p| p.is_finite()),
                "nonfinite molecular DOF"
            );
            ensure!(
                pixels.iter().filter(|p| p.max_element() > 0.05).count() > 100,
                "empty molecular draw: {name}"
            );
            if label == "sharp" {
                sharp.clone_from(&pixels);
            } else if label == "ao" {
                let energy =
                    |image: &[Vec3]| image.iter().map(|p| p.element_sum() as f64).sum::<f64>();
                let ratio = energy(&pixels) / energy(&sharp);
                ensure!(
                    (0.48..0.52).contains(&ratio),
                    "AO must apply exactly once: {name}, energy ratio {ratio}"
                );
                println!("{name}: AO energy ratio {ratio:.4}");
            } else if label == "reference" {
                let errors: Vec<f32> = pixels
                    .iter()
                    .zip(&reduced)
                    .map(|(a, b)| (*a - *b).abs().max_element())
                    .collect();
                let mean = errors.iter().sum::<f32>() / errors.len() as f32;
                let max = errors.iter().copied().fold(0.0_f32, f32::max);
                let large = errors.iter().filter(|e| **e > 0.05).count();
                println!(
                    "{name}: {large} pixels differ by more than 0.05, max at {:?}",
                    errors
                        .iter()
                        .position(|e| *e == max)
                        .map(|i| (i % width as usize, i / width as usize))
                );
                ensure!(
                    mean < 0.002 && max < 0.05,
                    "excessive reduction error: {name}, mean {mean}, max {max}"
                );
                println!("{name}: reduction error mean {mean:.6}, max {max:.6}");
            } else {
                reduced.clone_from(&pixels);
                let changed = pixels
                    .iter()
                    .zip(&sharp)
                    .filter(|(a, b)| (**a - **b).abs().max_element() > 0.002)
                    .count();
                ensure!(
                    changed > 100,
                    "DOF did not affect molecular fixture: {name}"
                );
                println!("{name}: {changed} pixels changed with finite aperture");
            }
            // Optional local artifacts, kept out of the unit-test and application paths.
            if let Some(directory) = std::env::var_os("ASTRA_DOF_ARTIFACTS") {
                fs::create_dir_all(&directory)?;
                let path = std::path::Path::new(&directory).join(format!("{name}-{label}.ppm"));
                let mut file = fs::File::create(path)?;
                write!(file, "P6\n{width} {height}\n255\n")?;
                for p in &pixels {
                    for c in p.to_array() {
                        let srgb = if c <= 0.0031308 {
                            12.92 * c
                        } else {
                            1.055 * c.powf(1.0 / 2.4) - 0.055
                        };
                        file.write_all(&[(srgb.clamp(0.0, 1.0) * 255.0).round() as u8])?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn vertex_buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("molecular regression vertices"),
        contents: if values.is_empty() {
            &[0; 256]
        } else {
            bytemuck::cast_slice(values)
        },
        usage: wgpu::BufferUsages::VERTEX,
    })
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> Result<Vec<Vec3>> {
    let (width, height) = (texture.width(), texture.height());
    let stride = (width * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molecular regression readback"),
        size: u64::from(stride * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver.recv().context("molecular GPU readback")??;
    let bytes = buffer.get_mapped_range(..)?;
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let offset = (y * stride + x * 8) as usize;
            pixels.push(Vec3::from_array(std::array::from_fn(|c| {
                decode_half(u16::from_le_bytes([
                    bytes[offset + 2 * c],
                    bytes[offset + 2 * c + 1],
                ]))
            })));
        }
    }
    Ok(pixels)
}
