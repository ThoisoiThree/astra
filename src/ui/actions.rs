use super::*;

pub(super) fn measurement_selection_combo(
    ui: &mut egui::Ui,
    label: &str,
    names: &[String],
    selected: &mut String,
) {
    egui::ComboBox::from_label(label)
        .selected_text(if selected.is_empty() {
            "No selection"
        } else {
            selected.as_str()
        })
        .show_ui(ui, |ui| {
            for name in names {
                ui.selectable_value(selected, name.clone(), name);
            }
        });
}

pub(super) fn measurement_selection_summary(
    molecule: Option<&Molecule>,
    selection: Option<&Selection>,
) -> Option<String> {
    let (molecule, selection) = (molecule?, selection?);
    let indices: Vec<_> = selection.indices().collect();
    let first = molecule.atoms.get(*indices.first()?)?;
    if indices.len() == 1 {
        return Some(format!("one atom · #{} {}", first.serial, first.name));
    }
    let one_residue = indices.iter().all(|index| {
        molecule.atoms.get(*index).is_some_and(|atom| {
            atom.chain_id == first.chain_id
                && atom.residue_name == first.residue_name
                && atom.residue_number == first.residue_number
                && atom.insertion_code == first.insertion_code
        })
    });
    if one_residue {
        Some(format!(
            "one residue/base · {} {} / chain {}",
            first.residue_name,
            first.residue_number,
            display_chain(&first.chain_id)
        ))
    } else {
        Some(format!("invalid · {} atoms across residues", indices.len()))
    }
}

pub(super) fn measurement_lines(
    ui: &mut egui::Ui,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<MeasurementColorEditor>,
    rename_editor: &mut Option<RenameEditor>,
) {
    ui.heading(format!("Measurements · {}", info.measurement_lines.len()));
    for line in info.measurement_lines {
        ui.horizontal(|ui| {
            let color = line.effective_color();
            let color_response = color_square(ui, color, line.color.is_some())
                .on_hover_text("Line color (click to edit in HSV)");
            color_response.context_menu(|ui| {
                if ui.button("Reset to default").clicked() {
                    actions.manager = Some(ManagerAction::SetMeasurementColor {
                        id: line.id,
                        color: None,
                    });
                    ui.close();
                }
            });
            if color_response.clicked() {
                *color_editor = Some(MeasurementColorEditor {
                    id: line.id,
                    hsva: hsva_from_color(color),
                });
            }

            let visibility_response = visibility_button(ui, line.visibility);
            visibility_response.context_menu(|ui| {
                if ui.button("Reset to default").clicked() {
                    actions.manager = Some(ManagerAction::SetMeasurementVisibility {
                        id: line.id,
                        state: VisibilityOverride::Inherit,
                    });
                    ui.close();
                }
            });
            if visibility_response.clicked() {
                actions.manager = Some(ManagerAction::SetMeasurementVisibility {
                    id: line.id,
                    state: line.visibility.next(),
                });
            }

            let name_response = ui
                .label(match &line.hydrogen_bonds {
                    Some(report) => format!(
                        "{} · {}/{}",
                        line.name,
                        report.displayed().count(),
                        report.candidates.len()
                    ),
                    None => format!("{} · {:.2} Å", line.name, line.distance()),
                })
                .on_hover_text(format!(
                    "{} ↔ {}",
                    line.first.description, line.second.description
                ));
            name_response.context_menu(|ui| {
                if ui.button("Rename").clicked() {
                    *rename_editor = Some(RenameEditor {
                        target: RenameTarget::Measurement(line.id),
                        name: line.name.clone(),
                    });
                    ui.close();
                }
            });
            let mut size = line.label_size;
            if ui
                .add(
                    egui::DragValue::new(&mut size)
                        .range(8.0..=48.0)
                        .speed(0.25)
                        .suffix(" pt"),
                )
                .on_hover_text("Distance label size")
                .changed()
            {
                actions.manager =
                    Some(ManagerAction::SetMeasurementLabelSize { id: line.id, size });
            }
            if ui.small_button("×").on_hover_text("Remove line").clicked() {
                actions.manager = Some(ManagerAction::RemoveMeasurement(line.id));
            }
        });
        if let Some(report) = &line.hydrogen_bonds {
            egui::CollapsingHeader::new("AMOEBA energies & parameterization")
                .id_salt(("amoeba report", line.id)).show(ui, |ui| {
                    ui.small(format!("{} · selection: {}", report.force_field, report.selection));
                    ui.small("ΔE = E(full) − E(A↔B decoupled), kJ/mol; negative is stabilizing.");
                    ui.small("Groups are template-derived AMOEBA polarization domains. Scores are not additive.");
                    let mut threshold = report.energy_threshold;
                    ui.horizontal(|ui| {
                        ui.label("Show ΔE <");
                        if ui.add(egui::DragValue::new(&mut threshold).speed(0.1).suffix(" kJ/mol")).changed() {
                            actions.manager = Some(ManagerAction::SetHydrogenBondThreshold { id: line.id, threshold });
                        }
                    });
                    ui.small(format!("SCF: {} iterations · residual {:.2e} D", report.scf_iterations, report.scf_residual_debye));
                    for residue in &report.residues {
                        if residue.status == "unparameterized" {
                            ui.colored_label(egui::Color32::YELLOW, format!("{}: unparameterized — {}", residue.residue, residue.reason));
                        }
                    }
                    egui::ScrollArea::vertical().max_height(220.0).id_salt(("amoeba candidates", line.id)).show(ui, |ui| {
                        for c in &report.candidates {
                            let atom_label = |index: usize| info.molecule.and_then(|m| m.atoms.get(index))
                                .map(|a| format!("{} {}{}:{} #{}", a.chain_id, a.residue_name, a.residue_number, a.name, a.serial))
                                .unwrap_or_else(|| format!("atom index {index}"));
                            ui.label(format!("{} → {} · ΔE {:.3} kJ/mol", atom_label(c.donor), atom_label(c.acceptor), c.delta_energy))
                                .on_hover_text(format!("H: {}\nPermanent: {:.6}\nvdW: {:.6}\nPolarization response: {:.6} kJ/mol\nD–A {:.3} Å · H–A {:.3} Å · D–H–A {:.2}°\nDonor group indices: {:?}\nAcceptor group indices: {:?}\nStatus: {}", atom_label(c.hydrogen), c.permanent, c.vdw, c.polarization, c.da_distance, c.ha_distance, c.dha_angle, c.donor_group, c.acceptor_group, c.parameterization_status));
                        }
                    });
                });
        }
        ui.horizontal(|ui| {
            ui.add_space(39.0);
            ui.small("Thickness");
            let mut thickness = line.thickness;
            if ui
                .add(
                    egui::DragValue::new(&mut thickness)
                        .range(0.01..=MAX_MEASUREMENT_THICKNESS)
                        .speed(0.05)
                        .fixed_decimals(3)
                        .suffix(" Å"),
                )
                .on_hover_text("Width of the dashed line segments")
                .changed()
            {
                actions.manager = Some(ManagerAction::SetMeasurementThickness {
                    id: line.id,
                    thickness,
                });
            }
        });
    }
}

pub(super) fn measurement_labels(painter: &egui::Painter, viewport: egui::Rect, info: UiInfo<'_>) {
    if !viewport.is_positive() {
        return;
    }
    let view_projection = info.camera.view_projection();
    for line in info
        .measurement_lines
        .iter()
        .filter(|line| line.is_visible())
    {
        for (first, second) in line.segments() {
            let clip = view_projection * ((first + second) * 0.5).extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = clip.truncate() / clip.w;
            if !(-1.0..=1.0).contains(&ndc.x)
                || !(-1.0..=1.0).contains(&ndc.y)
                || !(0.0..=1.0).contains(&ndc.z)
            {
                continue;
            }
            let center = egui::pos2(
                viewport.left() + (ndc.x + 1.0) * 0.5 * viewport.width(),
                viewport.top() + (1.0 - ndc.y) * 0.5 * viewport.height(),
            );
            let galley = painter.layout_no_wrap(
                format!("{:.2} Å", first.distance(second)),
                egui::FontId::proportional(line.label_size),
                egui::Color32::WHITE,
            );
            let rect = egui::Rect::from_center_size(center, galley.size() + egui::vec2(12.0, 6.0));
            painter.rect(
                rect,
                4.0,
                egui::Color32::from_rgb(29, 33, 35),
                egui::Stroke::new(1.0, color32(line.effective_color())),
                egui::StrokeKind::Outside,
            );
            painter.galley(
                rect.center() - galley.size() * 0.5,
                galley,
                egui::Color32::WHITE,
            );
        }
    }
}
impl UiState {
    pub(super) fn command_editor(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        ui.heading("Expression");
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.command_input)
                .hint_text("select active_site: Chain A/LEU*")
                .desired_width(f32::INFINITY),
        );
        let enter =
            response.lost_focus() && ui.ctx().input(|input| input.key_pressed(egui::Key::Enter));
        let (up, down) = if response.has_focus() {
            ui.ctx().input(|input| {
                (
                    input.key_pressed(egui::Key::ArrowUp),
                    input.key_pressed(egui::Key::ArrowDown),
                )
            })
        } else {
            (false, false)
        };
        if up {
            self.history_previous();
        } else if down {
            self.history_next();
        }
        if ui.button("Select!").clicked() || enter {
            let command = self.command_input.trim().to_string();
            if !command.is_empty() {
                self.history.push(command.clone());
                self.history_cursor = None;
                if let Some(name) = current_selection_name(&command) {
                    actions.manager = Some(ManagerAction::SaveCurrentSelection(name.to_owned()));
                } else {
                    actions.execute = Some(command);
                }
            }
        }
        ui.small("Enter only a name to save the current selection");
        ui.small(
            "Click inspect · drag orbit · RMB pan · Ctrl/Cmd+drag pan · Ctrl/Cmd+Z undo · Ctrl/Cmd+R redo",
        );
    }

    pub(super) fn history_previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let current = self.history_cursor.unwrap_or(self.history.len());
        let previous = current.saturating_sub(1);
        self.history_cursor = Some(previous);
        self.command_input.clone_from(&self.history[previous]);
    }

    pub(super) fn history_next(&mut self) {
        let Some(current) = self.history_cursor else {
            return;
        };
        if current + 1 < self.history.len() {
            self.history_cursor = Some(current + 1);
            self.command_input.clone_from(&self.history[current + 1]);
        } else {
            self.history_cursor = None;
            self.command_input.clear();
        }
    }
}

impl UiState {
    pub(super) fn actions_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.actions_open {
            return;
        }
        let names: Vec<_> = info.named_selections.keys().cloned().collect();
        if !names.contains(&self.measurement_first) {
            self.measurement_first = names.first().cloned().unwrap_or_default();
        }
        if !names.contains(&self.measurement_second) {
            self.measurement_second = names
                .iter()
                .find(|name| **name != self.measurement_first)
                .cloned()
                .unwrap_or_default();
        }

        if !names.contains(&self.hydrogen_bond_selection) {
            self.hydrogen_bond_selection = names.first().cloned().unwrap_or_default();
        }
        if !names.contains(&self.preparation_selection) {
            self.preparation_selection = names.first().cloned().unwrap_or_default();
        }
        let mut open = self.actions_open;
        egui::Window::new("Actions")
            .id(egui::Id::new("actions window"))
            .open(&mut open)
            .default_width(340.0)
            .resizable(true)
            .show(context, |ui| {
                egui::CollapsingHeader::new("Prepare structure · pH").default_open(true).show(ui, |ui| {
                    use astra::molecule::amoeba::protonation::{HistidineTautomer,PKA_TABLE};
                    measurement_selection_combo(ui, "Named selection", &names, &mut self.preparation_selection);
                    ui.small("Restore protein side chains and rebuild H in whole residues touched by the selection. Existing heavy atoms stay fixed. Existing H in prepared residues are replaced; Undo restores all changes.");
                    ui.checkbox(&mut self.preparation_settings.restore_heavy,"Restore missing heavy atoms first");
                    ui.horizontal(|ui|{ui.label("pH");ui.add(egui::DragValue::new(&mut self.preparation_settings.ph).range(0.0..=14.0).speed(0.1));});
                    ui.horizontal(|ui|{ui.label("Neutral histidine");ui.selectable_value(&mut self.preparation_settings.histidine,HistidineTautomer::Epsilon,"HIE (NE2–H)");ui.selectable_value(&mut self.preparation_settings.histidine,HistidineTautomer::Delta,"HID (ND1–H)");});
                    ui.small("Tabulated pKa is an approximation; local environment shifts are not predicted. Unsupported states and incomplete residues are skipped.");
                    egui::CollapsingHeader::new("pKa table · EMBOSS").show(ui,|ui|{for (site,pka) in PKA_TABLE{ui.label(format!("{site}: {pka:.1}"));}});
                    if ui.add_enabled(!names.is_empty(), egui::Button::new("Prepare structure at this pH")).clicked(){actions.manager=Some(ManagerAction::PrepareHydrogens { selection: self.preparation_selection.clone(), settings: self.preparation_settings });}
                    if let Some(report)=&self.preparation_report {
                        if let Some(repair)=&report.repair {
                            ui.label(format!("Heavy atoms restored: {}",repair.added));
                            egui::CollapsingHeader::new("Heavy-atom reconstruction report").show(ui,|ui|{egui::ScrollArea::vertical().id_salt("heavy repair report").max_height(180.).show(ui,|ui|{for r in &repair.residues{ui.label(format!("{}: {} · {}",r.residue,r.status,r.atoms.join(", ")));ui.small(&r.reason);}});});
                        }
                        ui.label(format!("Last preparation · pH {:.2}: {} H built, {} previous H removed",report.ph,report.added,report.removed));
                        egui::ScrollArea::vertical().id_salt("protonation report").max_height(180.0).show(ui,|ui|{for r in &report.residues {
                            ui.label(format!("{}: {}{}",r.residue,r.status,r.template.as_ref().map(|t|format!(" → {t}")).unwrap_or_default())).on_hover_text(&r.reason);
                            for site in &r.sites {ui.small(format!("{}: pKa {:.1}, protonated {:.0}%{}",site.name,site.pka,site.protonated_fraction*100.,if site.ambiguous{" · ambiguous near pKa"}else{""}));}
                            if r.status=="skipped"{ui.small(&r.reason);}
                        }});
                    }
                });
                ui.separator();
                ui.heading("Hydrogen bonds · AMOEBA 2018");
                measurement_selection_combo(ui, "Named selection", &names, &mut self.hydrogen_bond_selection);
                ui.small("Requires explicit H. D, H and A must belong to the selection; the full parameterized structure supplies the environment.");
                ui.small("Energy scores include self-consistent polarization response. Unsupported residues are reported as unparameterized.");
                egui::CollapsingHeader::new("Calculation settings").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Neighbor radius");
                        ui.add(egui::DragValue::new(&mut self.hydrogen_bond_settings.cutoff).range(5.0..=12.0).speed(0.1).suffix(" Å"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("SCF tolerance (D)");
                        ui.add(egui::DragValue::new(&mut self.hydrogen_bond_settings.tolerance).range(1e-10..=1e-5).speed(1e-7).max_decimals(10));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Maximum SCF iterations");
                        ui.add(egui::DragValue::new(&mut self.hydrogen_bond_settings.max_iterations).range(1..=10000));
                    });
                });
                if ui.add_enabled(!self.hydrogen_bond_selection.is_empty(), egui::Button::new("Calculate AMOEBA hydrogen bonds")).clicked() {
                    actions.manager = Some(ManagerAction::CreateHydrogenBonds { selection: self.hydrogen_bond_selection.clone(), settings: self.hydrogen_bond_settings });
                }
                ui.separator();
                ui.heading("Distance line");
                ui.label("Create from two named selections:");
                measurement_selection_combo(
                    ui,
                    "Endpoint A",
                    &names,
                    &mut self.measurement_first,
                );
                measurement_selection_combo(
                    ui,
                    "Endpoint B",
                    &names,
                    &mut self.measurement_second,
                );
                if let Some(summary) = measurement_selection_summary(
                    info.molecule,
                    info.named_selections.get(&self.measurement_first),
                ) {
                    ui.small(format!("A: {summary}"));
                }
                if let Some(summary) = measurement_selection_summary(
                    info.molecule,
                    info.named_selections.get(&self.measurement_second),
                ) {
                    ui.small(format!("B: {summary}"));
                }
                let can_create = !self.measurement_first.is_empty()
                    && !self.measurement_second.is_empty()
                    && self.measurement_first != self.measurement_second;
                if ui
                    .add_enabled(can_create, egui::Button::new("Create distance line"))
                    .clicked()
                {
                    actions.manager = Some(ManagerAction::CreateMeasurement {
                        first_selection: self.measurement_first.clone(),
                        second_selection: self.measurement_second.clone(),
                    });
                }
                ui.separator();
                ui.small(
                    "Each endpoint must select one atom or atoms belonging to exactly one residue/base.",
                );
            });
        self.actions_open = open;
    }
}
