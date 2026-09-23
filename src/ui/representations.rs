//! Molecular surface and label settings.

use astra::{
    RepresentationMask,
    labels::{LabelContent, LabelSettings},
    surface::{SurfaceKind, SurfaceSettings},
};

use super::*;

/// Which atoms a show/hide button applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationScope {
    Selection,
    All,
}

const RESOLUTIONS: [(f32, &str); 5] = [
    (1.0, "Draft · 1.0 Å"),
    (0.7, "Medium · 0.7 Å"),
    (0.5, "Fine · 0.5 Å"),
    (0.35, "Very fine · 0.35 Å"),
    (0.25, "Publication · 0.25 Å"),
];

impl UiState {
    pub(super) fn representations_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.representations_open {
            return;
        }
        let mut open = self.representations_open;
        egui::Window::new("Surface & labels")
            .id(egui::Id::new("representations window"))
            .open(&mut open)
            .default_width(300.0)
            .resizable(true)
            .show(context, |ui| {
                let Some(display) = info.display else {
                    ui.weak("Open a structure to add surfaces and labels");
                    return;
                };
                let has_selection = info.selection_count > 0;
                ui.heading("Molecular surface");
                visibility_buttons(ui, RepresentationMask::SURFACE, has_selection, actions);
                self.surface_settings(ui, display.surface, actions);
                if info.render_stats.surface_triangles > 0 {
                    ui.small(format!("{} triangles", info.render_stats.surface_triangles));
                }
                ui.small(
                    "The surface encloses the visible atoms that have it. Commands: \
                     show surface, protein · hide surface, all",
                );
                ui.separator();
                ui.heading("Labels");
                visibility_buttons(ui, RepresentationMask::LABEL, has_selection, actions);
                label_settings(ui, display.labels, actions);
                ui.small("Commands: show labels, byres within 4 of ligand · hide labels, all");
            });
        self.representations_open = open;
    }

    fn surface_settings(
        &mut self,
        ui: &mut egui::Ui,
        current: SurfaceSettings,
        actions: &mut UiActions,
    ) {
        // Dragging edits a draft; the surface is rebuilt when the drag ends.
        let draft = self.surface_draft.get_or_insert(current);
        let mut commit = false;
        egui::ComboBox::from_label("Surface type")
            .selected_text(draft.kind.label())
            .show_ui(ui, |ui| {
                for kind in SurfaceKind::ALL {
                    commit |= ui
                        .selectable_value(&mut draft.kind, kind, kind.label())
                        .changed();
                }
            });
        let resolution_label = RESOLUTIONS
            .iter()
            .find(|(value, _)| (value - draft.resolution).abs() < 1e-3)
            .map_or_else(
                || format!("{:.2} Å", draft.resolution),
                |(_, label)| (*label).to_owned(),
            );
        egui::ComboBox::from_label("Grid spacing")
            .selected_text(resolution_label)
            .show_ui(ui, |ui| {
                for (value, label) in RESOLUTIONS {
                    commit |= ui
                        .selectable_value(&mut draft.resolution, value, label)
                        .changed();
                }
            });
        ui.add_enabled_ui(draft.kind != SurfaceKind::VanDerWaals, |ui| {
            let response = ui.add(
                egui::Slider::new(&mut draft.probe_radius, SurfaceSettings::PROBE_RANGE)
                    .step_by(0.05)
                    .text("Probe radius, Å"),
            );
            commit |= response.drag_stopped() || (response.changed() && !response.dragged());
            commit |= ui
                .checkbox(&mut draft.cavities, "Internal cavities")
                .on_hover_text(
                    "Also draw surfaces around voids the probe cannot reach from outside",
                )
                .changed();
        });
        ui.small("Large structures coarsen the grid automatically to bound memory");
        if commit && *draft != current {
            actions.manager = Some(ManagerAction::SetSurfaceSettings(*draft));
        }
        if !commit && self.surface_draft.is_some_and(|draft| draft == current) {
            self.surface_draft = None;
        }
    }
}

fn visibility_buttons(
    ui: &mut egui::Ui,
    representation: RepresentationMask,
    has_selection: bool,
    actions: &mut UiActions,
) {
    ui.horizontal_wrapped(|ui| {
        for (label, scope, shown, enabled) in [
            (
                "Show for selection",
                RepresentationScope::Selection,
                true,
                has_selection,
            ),
            (
                "Hide for selection",
                RepresentationScope::Selection,
                false,
                has_selection,
            ),
            ("Show all", RepresentationScope::All, true, true),
            ("Hide all", RepresentationScope::All, false, true),
        ] {
            if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
                actions.manager = Some(ManagerAction::SetRepresentation {
                    representation,
                    scope,
                    shown,
                });
            }
        }
    });
}

fn label_settings(ui: &mut egui::Ui, current: LabelSettings, actions: &mut UiActions) {
    let mut settings = current;
    egui::ComboBox::from_label("Label text")
        .selected_text(settings.content.label())
        .show_ui(ui, |ui| {
            for content in LabelContent::ALL {
                ui.selectable_value(&mut settings.content, content, content.label());
            }
        });
    ui.add(egui::Slider::new(&mut settings.size, LabelSettings::SIZE_RANGE).text("Size, pt"));
    ui.horizontal(|ui| {
        ui.color_edit_button_srgb(&mut settings.color);
        ui.label("Text color");
        ui.checkbox(&mut settings.background, "Background");
    });
    if settings != current {
        actions.manager = Some(ManagerAction::SetLabelSettings(settings));
    }
}
