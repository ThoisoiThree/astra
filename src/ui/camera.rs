use super::*;

pub(super) fn molecule_summary(ui: &mut egui::Ui, info: UiInfo<'_>) {
    ui.horizontal(|ui| {
        ui.heading("Molecule");
        if let Some(id) = info.molecule_id {
            ui.monospace(id);
        }
    });
    egui::Grid::new("molecule stats")
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            for (label, value) in [
                ("Atoms", info.atom_count),
                ("Bonds", info.bond_count),
                ("Selected", info.selection_count),
                (
                    "Chains",
                    info.hierarchy.map_or(0, |hierarchy| hierarchy.chains.len()),
                ),
            ] {
                ui.label(label);
                ui.monospace(value.to_string());
                ui.end_row();
            }
        });
}

pub(super) fn focus_chain_number_fields(
    ui: &mut egui::Ui,
    chain: &mut String,
    residue_number: &mut String,
) {
    ui.horizontal(|ui| {
        ui.label("Chain ID");
        ui.text_edit_singleline(chain);
    });
    ui.horizontal(|ui| {
        ui.label("Residue number");
        ui.text_edit_singleline(residue_number);
    });
}
impl UiState {
    pub(super) fn camera_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.camera_open {
            return;
        }
        let mut open = self.camera_open;
        let mut near = info.camera.near;
        let mut far = info.camera.far;
        let mut enabled = info.camera.depth_of_field.enabled;
        let mut focal_length = info.camera.depth_of_field.focal_length_mm;
        let mut sensor_height = info.camera.depth_of_field.sensor_height_mm;
        let mut f_stop = info.camera.depth_of_field.f_stop;
        let mut blade_count = info.camera.depth_of_field.blade_count;
        let mut blade_rotation = info.camera.depth_of_field.blade_rotation.to_degrees();
        let mut max_coc = info.camera.depth_of_field.max_coc_pixels;
        let mut changed = false;
        egui::Window::new("Camera")
            .id(egui::Id::new("camera settings window"))
            .open(&mut open)
            .default_width(330.0)
            .resizable(false)
            .show(context, |ui| {
                ui.heading("Clipping planes");
                egui::Grid::new("camera clipping settings")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Near");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut near)
                                    .speed(0.01)
                                    .range(0.001..=100_000.0),
                            )
                            .changed();
                        ui.end_row();
                        ui.label("Far");
                        changed |= ui
                            .add(
                                egui::DragValue::new(&mut far)
                                    .speed(10.0)
                                    .range(0.002..=1_000_000.0),
                            )
                            .changed();
                        ui.end_row();
                    });
                ui.separator();
                ui.heading("Depth of field");
                changed |= ui.checkbox(&mut enabled, "Enabled").changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut focal_length, 5.0..=400.0)
                            .logarithmic(true)
                            .text("Focal length, mm"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut sensor_height, 4.0..=70.0).text("Sensor, mm"))
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut f_stop, 0.1..=22.0)
                            .logarithmic(true)
                            .max_decimals(2)
                            .text("F-stop"),
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut max_coc, 1.0..=64.0).text("Max CoC radius, px"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut blade_count, 0..=12).text("Iris blades"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut blade_rotation, 0.0..=360.0).text("Iris rotation"))
                    .changed();
                ui.small("0 blades = circular iris; 3–12 = polygonal optical bokeh");
                ui.label(format!("Focus: {}", info.focus_description));
                ui.label(format!("Distance: {:.3} Å", info.camera.focus_distance()));
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.focus_kind, FocusKind::Chain, "Chain");
                    ui.selectable_value(&mut self.focus_kind, FocusKind::Residue, "Residue");
                    ui.selectable_value(&mut self.focus_kind, FocusKind::Base, "Base");
                    ui.selectable_value(&mut self.focus_kind, FocusKind::Atom, "Atom");
                });
                match self.focus_kind {
                    FocusKind::Chain => {
                        ui.horizontal(|ui| {
                            ui.label("Chain ID");
                            ui.text_edit_singleline(&mut self.focus_chain);
                        });
                    }
                    FocusKind::Residue => {
                        focus_chain_number_fields(
                            ui,
                            &mut self.focus_chain,
                            &mut self.focus_residue_number,
                        );
                    }
                    FocusKind::Base => {
                        focus_chain_number_fields(
                            ui,
                            &mut self.focus_chain,
                            &mut self.focus_residue_number,
                        );
                        ui.horizontal(|ui| {
                            ui.label("Base name");
                            ui.text_edit_singleline(&mut self.focus_base_name);
                        });
                    }
                    FocusKind::Atom => {
                        ui.horizontal(|ui| {
                            ui.label("PDB serial");
                            ui.text_edit_singleline(&mut self.focus_atom_serial);
                        });
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("Set focus").clicked() {
                        actions.focus_request = Some(match self.focus_kind {
                            FocusKind::Chain => FocusRequest::Chain(self.focus_chain.clone()),
                            FocusKind::Residue => FocusRequest::Residue {
                                chain: self.focus_chain.clone(),
                                number: self.focus_residue_number.clone(),
                            },
                            FocusKind::Base => FocusRequest::Base {
                                chain: self.focus_chain.clone(),
                                name: self.focus_base_name.clone(),
                                number: self.focus_residue_number.clone(),
                            },
                            FocusKind::Atom => {
                                FocusRequest::AtomSerial(self.focus_atom_serial.clone())
                            }
                        });
                    }
                    if ui
                        .add_enabled(
                            info.inspection.is_some(),
                            egui::Button::new("Use inspected"),
                        )
                        .clicked()
                    {
                        actions.focus_request = Some(FocusRequest::Inspected);
                    }
                });
                ui.separator();
                ui.heading("Orbit pivot");
                ui.label(format!("Pivot: {}", info.pivot_description));
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            info.inspection.is_some(),
                            egui::Button::new("Set pivot from inspected"),
                        )
                        .clicked()
                    {
                        actions.pivot_request = Some(PivotRequest::Inspected);
                    }
                    if ui
                        .add_enabled(info.molecule.is_some(), egui::Button::new("Reset pivot"))
                        .clicked()
                    {
                        actions.pivot_request = Some(PivotRequest::Reset);
                    }
                });
            });
        self.camera_open = open;
        if changed {
            actions.camera_update = Some(CameraUpdate {
                near,
                far,
                dof_enabled: enabled,
                focal_length_mm: focal_length,
                sensor_height_mm: sensor_height,
                f_stop,
                blade_count,
                blade_rotation: blade_rotation.to_radians(),
                max_coc_pixels: max_coc,
            });
        }
    }
}
