use super::*;

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
            .resizable(false)
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
                ui.heading("Ambient occlusion");
                let mut ao = display.ambient_occlusion;
                let mut changed = ui.checkbox(&mut ao.enabled, "Enabled").changed();
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
                    egui::ComboBox::from_label("Quality")
                        .selected_text(ao.quality.label())
                        .show_ui(ui, |ui| {
                            for quality in molview::AmbientOcclusionQuality::ALL {
                                changed |= ui
                                    .selectable_value(&mut ao.quality, quality, quality.label())
                                    .changed();
                            }
                        });
                });
                ui.small("Depth-aware bilateral SSAO · annotations are excluded");
                if changed {
                    actions.manager = Some(ManagerAction::SetAmbientOcclusion(ao));
                }
            });
        self.mode_open = open;
    }
}
