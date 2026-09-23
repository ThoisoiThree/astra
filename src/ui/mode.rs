use super::*;

pub(super) fn performance_overlay(
    context: &egui::Context,
    viewport: egui::Rect,
    stats: astra::render::RenderStats,
) {
    context.request_repaint_after(std::time::Duration::from_millis(250));
    let position = egui::pos2(
        (viewport.right() - 238.0).max(viewport.left()),
        viewport.top() + 8.0,
    );
    egui::Area::new(egui::Id::new("render performance overlay"))
        .fixed_pos(position)
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.strong("Renderer");
                ui.monospace(format!("FPS        {:>7.0}", stats.fps))
                    .on_hover_text("Presented frames in the last second; idle rendering is limited to UI updates.");
                ui.monospace(format!("Frame wall {:>7.2} ms", stats.cpu_frame_ms))
                    .on_hover_text("Renderer elapsed time, including driver/GPU waits; not CPU utilization or the frame interval.");
                for (label, time) in ["Prepare", "Encode", "Acquire", "Submit", "Present"]
                    .into_iter().zip(stats.cpu_stages_ms)
                {
                    ui.monospace(format!("{label:<10} {time:>7.2} ms"));
                }
                ui.separator();
                if stats.gpu_timestamps_supported {
                    ui.monospace(format!("GPU span   {:>7.2} ms", stats.gpu_frame_ms))
                        .on_hover_text("First measured start to last measured end in one sampled frame. Pass intervals below may overlap or include dependency waits; do not sum them.");
                    for (label, milliseconds) in [
                        ("Scene", stats.gpu_pass_ms[0]),
                        ("AO raw", stats.gpu_pass_ms[1]),
                        ("AO blur", stats.gpu_pass_ms[2]),
                        ("DOF span", stats.gpu_dof_ms),
                        ("DOF layers", stats.gpu_pass_ms[3]),
                        ("DOF merge", stats.gpu_pass_ms[7]),
                        ("DOF splat", stats.gpu_pass_ms[8]),
                        ("Compose", stats.gpu_pass_ms[4]),
                        ("Labels", stats.gpu_pass_ms[5]),
                        ("UI", stats.gpu_pass_ms[6]),
                    ] {
                        ui.monospace(format!("{label:<9} {milliseconds:>7.2} ms"));
                    }
                } else {
                    ui.weak("GPU timestamps unavailable");
                }
                ui.separator();
                ui.monospace(format!(
                    "GPU memory ~{}",
                    format_bytes(stats.gpu_memory_bytes)
                ));
                ui.monospace(format!(
                    "Atoms {} · bonds {}",
                    stats.atom_instances, stats.bond_instances
                ));
                ui.monospace(format!(
                    "Cartoon {} tris · toon {} quads",
                    stats.cartoon_triangles, stats.toon_triangles
                ));
                ui.monospace(format!(
                    "Surface {} tris · scale {}×",
                    stats.surface_triangles, stats.render_scale
                ));
            });
        });
}

impl UiState {
    pub(super) fn mode_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.mode_open {
            return;
        }
        let mut open = self.mode_open;
        egui::Window::new("Mode")
            .id(egui::Id::new("display mode window"))
            .open(&mut open)
            .default_width(280.0)
            .resizable(true)
            .show(context, |ui| {
                let Some(display) = info.display else {
                    ui.weak("Open a structure to choose its display mode");
                    return;
                };
                let mut mode = display.global_mode;
                egui::ComboBox::from_label("Global mode")
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        for candidate in DisplayMode::ALL {
                            ui.selectable_value(&mut mode, candidate, candidate.label());
                        }
                    });
                if mode != display.global_mode {
                    actions.manager = Some(ManagerAction::SetGlobalMode(mode));
                }
                if mode == DisplayMode::Spacefill {
                    ui.small("Van der Waals spheres");
                } else if mode == DisplayMode::Licorice {
                    ui.small("Uniform sticks with rounded joints");
                }
                if mode == DisplayMode::Toon {
                    ui.small(
                        "Analytic space-filling spheres · atom/residue/chain outlines · paper background",
                    );
                }
                ui.separator();
                ui.small(
                    "Hierarchy overrides have priority: atom > residue > chain > named selection > global.",
                );
                ui.separator();
                ui.heading("Render quality");
                let mut ao = display.ambient_occlusion;
                let mut changed = false;
                egui::ComboBox::from_label("Quality")
                    .selected_text(ao.quality.label())
                    .show_ui(ui, |ui| {
                        for quality in astra::AmbientOcclusionQuality::ALL {
                            changed |= ui
                                .selectable_value(&mut ao.quality, quality, quality.label())
                                .changed();
                        }
                    });
                ui.small("Controls AO samples, DOF resolution/layers, and cartoon tessellation");
                let current_scale = info.render_stats.render_scale.max(1);
                let label = |scale: u32| {
                    if scale == 1 {
                        "FXAA".to_string()
                    } else {
                        format!("Supersampling {scale}×")
                    }
                };
                let mut scale = current_scale;
                egui::ComboBox::from_label("Antialiasing")
                    .selected_text(label(scale))
                    .show_ui(ui, |ui| {
                        for candidate in 1..=astra::render::MAX_RENDER_SCALE {
                            ui.selectable_value(&mut scale, candidate, label(candidate));
                        }
                    });
                if scale != current_scale {
                    actions.render_scale = Some(scale);
                }
                ui.small("Supersampling renders N×N samples per pixel; use it for presentation");
                let mut bond_orders = display.bond_orders;
                if ui
                    .checkbox(&mut bond_orders, "Show bond orders")
                    .on_hover_text("Double, triple and aromatic bonds in stick representations")
                    .changed()
                {
                    actions.manager = Some(ManagerAction::SetBondOrders(bond_orders));
                }
                ui.checkbox(&mut self.performance_overlay, "Performance overlay");
                ui.separator();
                ui.heading("Ambient occlusion");
                changed |= ui.checkbox(&mut ao.enabled, "Enabled").changed();
                ui.add_enabled_ui(ao.enabled, |ui| {
                    changed |= ui
                        .add(egui::Slider::new(&mut ao.strength, 0.0..=3.0).text("Strength"))
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut ao.radius, 0.1..=8.0)
                                .logarithmic(true)
                                .text("Radius, Å"),
                        )
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut ao.bias, 0.0..=0.25).text("Bias"))
                        .changed();
                });
                ui.small("Depth-aware bilateral SSAO · annotations are excluded");
                if changed {
                    actions.manager = Some(ManagerAction::SetAmbientOcclusion(ao));
                }
            });
        self.mode_open = open;
    }
}
