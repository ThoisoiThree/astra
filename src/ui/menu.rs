//! Shared menu presentation. Use native egui menus for focus, dismissal and
//! submenu navigation; keep application actions in the calling UI code.
use super::*;

const MENU_WIDTH: f32 = 260.0;

fn style(style: &mut egui::Style) {
    style.spacing.interact_size.y = 28.0;
    style.spacing.item_spacing = egui::vec2(0.0, 2.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.visuals.window_fill = egui::Color32::from_rgb(27, 27, 27);
    style.visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(55));
    style.visuals.window_corner_radius = egui::CornerRadius::ZERO;
    style.visuals.menu_corner_radius = egui::CornerRadius::ZERO;
    style.visuals.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
    style.visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_gray(72);
    style.visuals.widgets.active.weak_bg_fill = egui::Color32::from_gray(80);
    style.visuals.widgets.open.weak_bg_fill = egui::Color32::from_gray(72);
    for widget in [
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        widget.corner_radius = egui::CornerRadius::ZERO;
        widget.bg_stroke = egui::Stroke::NONE;
    }
}

pub(super) fn section(ui: &mut egui::Ui, title: &str) {
    ui.set_min_width(MENU_WIDTH);
    ui.add_space(7.0);
    ui.label(
        egui::RichText::new(title)
            .size(11.0)
            .strong()
            .color(egui::Color32::from_gray(175)),
    );
}

/// Reserve a leading mark column and a trailing shortcut column in every row.
/// Shortcut text is only a label; callers must provide a working binding.
pub(super) fn item(
    ui: &mut egui::Ui,
    label: &str,
    shortcut: &str,
    enabled: bool,
    selected: bool,
) -> bool {
    let response = ui.add_enabled(
        enabled,
        egui::Button::new(format!("     {label}"))
            .shortcut_text(shortcut)
            .min_size(egui::vec2(MENU_WIDTH, 28.0))
            .frame(false),
    );
    if selected {
        let center = egui::pos2(response.rect.left() + 12.0, response.rect.center().y);
        let stroke = egui::Stroke::new(1.5, ui.visuals().text_color());
        ui.painter().line_segment(
            [
                center + egui::vec2(-4.0, 0.0),
                center + egui::vec2(-1.0, 3.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                center + egui::vec2(-1.0, 3.0),
                center + egui::vec2(5.0, -4.0),
            ],
            stroke,
        );
    }
    if response.clicked() {
        ui.close();
        true
    } else {
        false
    }
}

pub(super) fn submenu(ui: &mut egui::Ui, label: &str, content: impl FnOnce(&mut egui::Ui)) {
    ui.menu_button(format!("     {label}"), |ui| {
        ui.set_min_width(MENU_WIDTH);
        content(ui);
    });
}

impl UiState {
    pub(super) fn toolbar(&mut self, ui: &mut egui::Ui, info: UiInfo<'_>, actions: &mut UiActions) {
        egui::MenuBar::new()
            .style(style)
            .config(egui::containers::menu::MenuConfig::new().style(style))
            .ui(ui, |ui| {
                let has_structure = info.molecule.is_some();
                ui.menu_button("File", |ui| {
                    section(ui, "STRUCTURES");
                    actions.open |= item(ui, "Open…", "", true, false);
                    if item(ui, "Fetch from PDB…", "", true, false) {
                        self.fetch_open = true;
                    }
                    ui.separator();
                    section(ui, "TRAJECTORY");
                    if item(ui, "Load trajectory…", "", has_structure, false) {
                        actions.trajectory = Some(TrajectoryAction::Load);
                    }
                    if item(
                        ui,
                        "Unload trajectory",
                        "",
                        info.trajectory.is_some(),
                        false,
                    ) {
                        actions.trajectory = Some(TrajectoryAction::Unload);
                    }
                    ui.separator();
                    section(ui, "SCENE");
                    actions.save |= item(ui, "Save", "", has_structure, false);
                    actions.save_as |= item(ui, "Save as…", "", has_structure, false);
                    ui.separator();
                    section(ui, "IMAGE");
                    if item(ui, "Export image…", "", has_structure, false) {
                        self.open_export();
                    }
                });
                ui.menu_button("Structure", |ui| {
                    self.structure_menu(ui, info, actions);
                });
                ui.menu_button("Mode", |ui| {
                    section(ui, "REPRESENTATION");
                    if let Some(display) = info.display {
                        for mode in DisplayMode::ALL {
                            if item(ui, mode.label(), "", true, display.global_mode == mode) {
                                actions.manager = Some(ManagerAction::SetGlobalMode(mode));
                            }
                        }
                        ui.separator();
                        submenu(ui, "Render quality", |ui| {
                            for quality in astra::AmbientOcclusionQuality::ALL {
                                if item(
                                    ui,
                                    quality.label(),
                                    "",
                                    true,
                                    display.ambient_occlusion.quality == quality,
                                ) {
                                    let mut ao = display.ambient_occlusion;
                                    ao.quality = quality;
                                    actions.manager = Some(ManagerAction::SetAmbientOcclusion(ao));
                                }
                            }
                        });
                    }
                    if item(ui, "Render settings…", "", has_structure, false) {
                        self.mode_open = true;
                    }
                    if item(ui, "Surface & labels…", "", has_structure, false) {
                        self.representations_open = true;
                    }
                });
                ui.menu_button("Coloring", |ui| {
                    section(ui, "COLOR SCHEME");
                    if let Some(display) = info.display {
                        for mode in ColoringMode::ALL {
                            if item(ui, mode.label(), "", true, display.coloring_mode == mode) {
                                actions.manager = Some(ManagerAction::SetColoringMode(mode));
                            }
                        }
                    }
                    ui.separator();
                    actions.reset_colors |= item(ui, "Reset colors", "", has_structure, false);
                    if item(ui, "Color settings…", "", has_structure, false) {
                        self.coloring_open = true;
                    }
                });
                ui.menu_button("Camera", |ui| {
                    section(ui, "VIEW");
                    actions.fit |= item(ui, "Fit molecule", "", has_structure, false);
                    ui.separator();
                    section(ui, "LENS & FOCUS");
                    if item(ui, "Camera settings…", "", true, false) {
                        self.camera_open = true;
                    }
                });
                ui.menu_button("Actions", |ui| {
                    section(ui, "TOOLS");
                    if item(ui, "Actions & measurements…", "", has_structure, false) {
                        self.actions_open = true;
                    }
                    ui.separator();
                    section(ui, "DIAGNOSTICS");
                    if item(
                        ui,
                        "Performance overlay",
                        "",
                        true,
                        self.performance_overlay,
                    ) {
                        self.performance_overlay = !self.performance_overlay;
                    }
                });
                if ui.selectable_label(self.info_open, "Info").clicked() {
                    self.info_open = !self.info_open;
                }
                ui.separator();
                ui.label(info.filename.unwrap_or("No molecule loaded"));
                if let (Some(job_id), Some(stage)) = (info.background_job_id, info.background_stage)
                {
                    ui.spinner();
                    ui.add(
                        egui::ProgressBar::new(info.background_progress)
                            .desired_width(110.0)
                            .text(stage),
                    );
                    if ui.small_button("Cancel").clicked() {
                        actions.cancel_background_job = Some(job_id);
                    }
                }
            });
    }
}

impl UiState {
    fn structure_menu(&mut self, ui: &mut egui::Ui, info: UiInfo<'_>, actions: &mut UiActions) {
        let Some(molecule) = info.molecule else {
            section(ui, "STRUCTURE");
            ui.weak("     Open a structure first");
            return;
        };
        let structure = &molecule.info;
        section(ui, "BIOLOGICAL ASSEMBLY");
        if structure.assemblies.is_empty() {
            ui.weak("     The file defines no assemblies");
        }
        for assembly in &structure.assemblies {
            let label = format!(
                "{} · {} chain copies",
                assembly.label(),
                assembly.copy_count()
            );
            if item(ui, &label, "", true, false) {
                actions.structure_request = Some(StructureRequest::Assembly(assembly.id.clone()));
            }
        }
        ui.separator();
        section(ui, "CRYSTAL");
        let crystal = structure
            .crystal
            .as_ref()
            .filter(|crystal| crystal.cell.is_crystallographic());
        let label = crystal.map_or_else(
            || "No crystallographic cell".to_string(),
            |crystal| {
                format!(
                    "{} · {:.1} {:.1} {:.1} Å",
                    crystal.space_group, crystal.cell.a, crystal.cell.b, crystal.cell.c
                )
            },
        );
        ui.weak(format!("     {label}"));
        if item(ui, "Unit cell", "", crystal.is_some(), false) {
            actions.structure_request = Some(StructureRequest::UnitCell);
        }
        if item(
            ui,
            "Symmetry mates within 10 Å",
            "",
            crystal.is_some(),
            false,
        ) {
            actions.structure_request = Some(StructureRequest::SymmetryMates(10.0));
        }
        if item(
            ui,
            "Symmetry mates within 20 Å",
            "",
            crystal.is_some(),
            false,
        ) {
            actions.structure_request = Some(StructureRequest::SymmetryMates(20.0));
        }
        ui.separator();
        section(ui, "SECONDARY STRUCTURE");
        if let Some(display) = info.display {
            for source in astra::molecule::SecondarySource::ALL {
                let available = source != astra::molecule::SecondarySource::File
                    || !structure.secondary.is_empty();
                if item(
                    ui,
                    source.label(),
                    "",
                    available,
                    display.secondary_source == source,
                ) {
                    actions.manager = Some(ManagerAction::SetSecondarySource(source));
                }
            }
        }
    }
}
