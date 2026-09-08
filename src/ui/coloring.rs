use super::*;

impl UiState {
    pub(super) fn coloring_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.coloring_open {
            return;
        }
        let mut open = self.coloring_open;
        egui::Window::new("Coloring")
            .id(egui::Id::new("coloring settings window"))
            .open(&mut open)
            .default_width(320.0)
            .resizable(true)
            .show(context, |ui| {
                let Some(display) = info.display else {
                    ui.weak("Open a PDB file to configure coloring");
                    return;
                };
                let mut mode = display.coloring_mode;
                egui::ComboBox::from_label("Color scheme")
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        for candidate in ColoringMode::ALL {
                            ui.selectable_value(&mut mode, candidate, candidate.label());
                        }
                    });
                if mode != display.coloring_mode {
                    actions.manager = Some(ManagerAction::SetColoringMode(mode));
                }
                ui.add_space(6.0);
                match mode {
                    ColoringMode::Element => {
                        ui.label("Standard element / CPK colors.");
                    }
                    ColoringMode::Chain => {
                        ui.label("A categorical color for each chain.");
                    }
                    ColoringMode::Residue => {
                        ui.label("A categorical color for each residue instance.");
                    }
                    ColoringMode::ResidueType => {
                        ui.label("Equal residue names share the same color.");
                    }
                    ColoringMode::SecondaryStructure => {
                        ui.label("Helix red · Sheet yellow · Turn blue");
                        ui.label("Coil gray · Nucleic acid violet");
                        ui.small("Ligands, solvent, and ions retain element colors.");
                    }
                    ColoringMode::BFactor => {
                        ui.label("Blue → cyan → yellow → red B-factor gradient.");
                    }
                    ColoringMode::Uniform => {
                        ui.label("Uniform base color");
                        let mut hsva = hsva_from_color(display.uniform_color);
                        if egui::color_picker::color_picker_hsva_2d(
                            ui,
                            &mut hsva,
                            egui::color_picker::Alpha::Opaque,
                        ) {
                            actions.manager =
                                Some(ManagerAction::SetUniformColor(color_from_hsva(hsva)));
                        }
                    }
                };
                ui.separator();
                ui.small("Hierarchy colors override this scheme: atom > residue > chain > base.");
            });
        self.coloring_open = open;
    }

    pub(super) fn color_editor_window(&mut self, context: &egui::Context, actions: &mut UiActions) {
        let Some(mut editor) = self.color_editor.take() else {
            return;
        };
        let mut open = true;
        let mut restore_default = false;
        egui::Window::new(format!("HSV color · {}", editor.label))
            .id(egui::Id::new("hierarchy HSV color editor"))
            .open(&mut open)
            .resizable(true)
            .show(context, |ui| {
                if egui::color_picker::color_picker_hsva_2d(
                    ui,
                    &mut editor.hsva,
                    egui::color_picker::Alpha::Opaque,
                ) {
                    actions.manager = Some(ManagerAction::SetColor {
                        targets: editor.targets.clone(),
                        color: Some(color_from_hsva(editor.hsva)),
                    });
                }
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "H {:.0}°  S {:.0}%  V {:.0}%",
                        editor.hsva.h * 360.0,
                        editor.hsva.s * 100.0,
                        editor.hsva.v * 100.0
                    ));
                    if ui
                        .button("Default")
                        .on_hover_text("Restore inheritance")
                        .clicked()
                    {
                        restore_default = true;
                    }
                });
            });
        if restore_default {
            actions.manager = Some(ManagerAction::SetColor {
                targets: editor.targets.clone(),
                color: None,
            });
        } else if open {
            self.color_editor = Some(editor);
        }
    }

    pub(super) fn named_color_editor_window(
        &mut self,
        context: &egui::Context,
        actions: &mut UiActions,
    ) {
        let Some(mut editor) = self.named_color_editor.take() else {
            return;
        };
        let mut open = true;
        let mut restore_default = false;
        egui::Window::new(format!("HSV color · selection {}", editor.name))
            .id(egui::Id::new("named selection HSV color editor"))
            .open(&mut open)
            .resizable(true)
            .show(context, |ui| {
                if egui::color_picker::color_picker_hsva_2d(
                    ui,
                    &mut editor.hsva,
                    egui::color_picker::Alpha::Opaque,
                ) {
                    actions.manager = Some(ManagerAction::SetNamedColor {
                        name: editor.name.clone(),
                        color: Some(color_from_hsva(editor.hsva)),
                    });
                }
                if ui.button("Default").clicked() {
                    restore_default = true;
                }
            });
        if restore_default {
            actions.manager = Some(ManagerAction::SetNamedColor {
                name: editor.name,
                color: None,
            });
        } else if open {
            self.named_color_editor = Some(editor);
        }
    }

    pub(super) fn measurement_color_editor_window(
        &mut self,
        context: &egui::Context,
        actions: &mut UiActions,
    ) {
        let Some(mut editor) = self.measurement_color_editor.take() else {
            return;
        };
        let mut open = true;
        let mut restore_default = false;
        egui::Window::new(format!("HSV color · line #{}", editor.id))
            .id(egui::Id::new("measurement line HSV color editor"))
            .open(&mut open)
            .resizable(true)
            .show(context, |ui| {
                if egui::color_picker::color_picker_hsva_2d(
                    ui,
                    &mut editor.hsva,
                    egui::color_picker::Alpha::Opaque,
                ) {
                    actions.manager = Some(ManagerAction::SetMeasurementColor {
                        id: editor.id,
                        color: Some(color_from_hsva(editor.hsva)),
                    });
                }
                if ui.button("Default").clicked() {
                    restore_default = true;
                }
            });
        if restore_default {
            actions.manager = Some(ManagerAction::SetMeasurementColor {
                id: editor.id,
                color: None,
            });
        } else if open {
            self.measurement_color_editor = Some(editor);
        }
    }
}
