use std::collections::{BTreeMap, BTreeSet};

use molview::{
    ColoringMode, DisplayColor, DisplayLevel, DisplayMode, DisplayState, ModeOverride,
    NamedSelectionStyle, VisibilityOverride,
    camera::OrbitCamera,
    molecule::{Atom, Molecule, MoleculeHierarchy, ResidueGroup},
    selection::Selection,
};

#[derive(Debug, Default)]
pub struct UiState {
    pub command_input: String,
    pub latest_error: Option<String>,
    history: Vec<String>,
    history_cursor: Option<usize>,
    camera_open: bool,
    coloring_open: bool,
    mode_open: bool,
    color_editor: Option<ColorEditor>,
    named_color_editor: Option<NamedColorEditor>,
    named_expression_editor: Option<NamedExpressionEditor>,
    focus_kind: FocusKind,
    focus_chain: String,
    focus_residue_number: String,
    focus_base_name: String,
    focus_atom_serial: String,
}

#[derive(Debug, Clone)]
struct ColorEditor {
    targets: Vec<InspectionTarget>,
    label: String,
    hsva: egui::ecolor::Hsva,
}

#[derive(Debug, Clone)]
struct NamedColorEditor {
    name: String,
    hsva: egui::ecolor::Hsva,
}

#[derive(Debug, Clone)]
struct NamedExpressionEditor {
    name: String,
    expression: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum FocusKind {
    #[default]
    Chain,
    Residue,
    Base,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraUpdate {
    pub near: f32,
    pub far: f32,
    pub dof_enabled: bool,
    pub focal_length_mm: f32,
    pub sensor_height_mm: f32,
    pub f_stop: f32,
    pub blade_count: u32,
    pub blade_rotation: f32,
    pub max_coc_pixels: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FocusRequest {
    Chain(String),
    Residue {
        chain: String,
        number: String,
    },
    Base {
        chain: String,
        name: String,
        number: String,
    },
    AtomSerial(String),
    Inspected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotRequest {
    Inspected,
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InspectionTarget {
    Chain(usize),
    Residue {
        chain_index: usize,
        residue_index: usize,
    },
    Atom(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchySelectionGesture {
    Replace,
    Range,
    Toggle,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManagerAction {
    SelectHierarchy {
        target: InspectionTarget,
        gesture: HierarchySelectionGesture,
    },
    SetColor {
        targets: Vec<InspectionTarget>,
        color: Option<DisplayColor>,
    },
    SetVisibility {
        targets: Vec<InspectionTarget>,
        state: VisibilityOverride,
    },
    SetMode {
        targets: Vec<InspectionTarget>,
        state: ModeOverride,
    },
    PropagateColor(Vec<InspectionTarget>),
    PropagateVisibility(Vec<InspectionTarget>),
    PropagateMode(Vec<InspectionTarget>),
    SetNamedColor {
        name: String,
        color: Option<DisplayColor>,
    },
    SetNamedVisibility {
        name: String,
        state: VisibilityOverride,
    },
    SetNamedMode {
        name: String,
        state: ModeOverride,
    },
    PropagateNamedColor {
        name: String,
        color: DisplayColor,
    },
    PropagateNamedVisibility {
        name: String,
        state: VisibilityOverride,
    },
    PropagateNamedMode {
        name: String,
        state: ModeOverride,
    },
    SelectNamedSubset {
        indices: Vec<usize>,
        inspection: InspectionTarget,
    },
    SetColoringMode(ColoringMode),
    SetGlobalMode(DisplayMode),
    SetUniformColor(DisplayColor),
    ActivateNamed(String),
    UpdateNamedExpression {
        name: String,
        expression: String,
    },
    RemoveNamed(String),
}

#[derive(Debug)]
pub struct UiActions {
    pub open: bool,
    pub fit: bool,
    pub reset_colors: bool,
    pub execute: Option<String>,
    pub manager: Option<ManagerAction>,
    pub camera_update: Option<CameraUpdate>,
    pub focus_request: Option<FocusRequest>,
    pub pivot_request: Option<PivotRequest>,
    pub viewport: egui::Rect,
}

impl Default for UiActions {
    fn default() -> Self {
        Self {
            open: false,
            fit: false,
            reset_colors: false,
            execute: None,
            manager: None,
            camera_update: None,
            focus_request: None,
            pivot_request: None,
            viewport: egui::Rect::NOTHING,
        }
    }
}

#[derive(Clone, Copy)]
pub struct UiInfo<'a> {
    pub filename: Option<&'a str>,
    pub atom_count: usize,
    pub bond_count: usize,
    pub selection_count: usize,
    pub molecule: Option<&'a Molecule>,
    pub hierarchy: Option<&'a MoleculeHierarchy>,
    pub display: Option<&'a DisplayState>,
    pub named_selections: &'a BTreeMap<String, Selection>,
    pub named_selection_expressions: &'a BTreeMap<String, String>,
    pub named_selection_styles: &'a BTreeMap<String, NamedSelectionStyle>,
    pub inspection: Option<InspectionTarget>,
    pub hierarchy_selection: &'a BTreeSet<InspectionTarget>,
    pub camera: &'a OrbitCamera,
    pub focus_description: &'a str,
    pub pivot_description: &'a str,
}

impl UiState {
    pub fn show(&mut self, root: &mut egui::Ui, info: UiInfo<'_>) -> UiActions {
        let mut actions = UiActions::default();
        egui::Panel::top("toolbar").show(root, |ui| {
            ui.horizontal(|ui| {
                actions.open = ui.button("Open structure").clicked();
                actions.fit = ui.button("Fit").clicked();
                actions.reset_colors = ui.button("Reset colors").clicked();
                if ui.selectable_label(self.mode_open, "Mode").clicked() {
                    self.mode_open = !self.mode_open;
                }
                if ui
                    .selectable_label(self.coloring_open, "Coloring")
                    .clicked()
                {
                    self.coloring_open = !self.coloring_open;
                }
                if ui.selectable_label(self.camera_open, "Camera").clicked() {
                    self.camera_open = !self.camera_open;
                }
                ui.separator();
                ui.strong(info.filename.unwrap_or("No molecule loaded"));
            });
        });

        egui::Panel::left("molecule manager")
            .default_size(350.0)
            .size_range(280.0..=520.0)
            .resizable(true)
            .show(root, |ui| {
                molecule_summary(ui, info);
                ui.add_space(10.0);
                self.command_editor(ui, &mut actions);
                if let Some(error) = &self.latest_error {
                    ui.add_space(6.0);
                    ui.colored_label(egui::Color32::from_rgb(245, 95, 95), error);
                }
                ui.separator();
                named_selections(
                    ui,
                    info,
                    &mut actions,
                    &mut self.named_color_editor,
                    &mut self.named_expression_editor,
                );
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("molecule hierarchy scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        inspector(ui, info);
                        if info.inspection.is_some() {
                            ui.separator();
                        }
                        hierarchy_tree(ui, info, &mut actions, &mut self.color_editor);
                    });
            });
        self.mode_window(root.ctx(), info, &mut actions);
        self.coloring_window(root.ctx(), info, &mut actions);
        self.camera_window(root.ctx(), info, &mut actions);
        self.color_editor_window(root.ctx(), &mut actions);
        self.named_color_editor_window(root.ctx(), &mut actions);
        self.named_expression_editor_window(root.ctx(), &mut actions);
        actions.viewport = root.available_rect_before_wrap();
        actions
    }

    fn mode_window(&mut self, context: &egui::Context, info: UiInfo<'_>, actions: &mut UiActions) {
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
                ui.separator();
                ui.small(
                    "Hierarchy overrides have priority: atom > residue > chain > named selection > global.",
                );
            });
        self.mode_open = open;
    }

    fn camera_window(
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
                    .add(egui::Slider::new(&mut f_stop, 0.7..=22.0).text("F-stop"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut max_coc, 1.0..=64.0).text("Max CoC, px"))
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

    fn coloring_window(
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
            .resizable(false)
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

    fn color_editor_window(&mut self, context: &egui::Context, actions: &mut UiActions) {
        let Some(mut editor) = self.color_editor.take() else {
            return;
        };
        let mut open = true;
        let mut restore_default = false;
        egui::Window::new(format!("HSV color · {}", editor.label))
            .id(egui::Id::new("hierarchy HSV color editor"))
            .open(&mut open)
            .resizable(false)
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

    fn named_color_editor_window(&mut self, context: &egui::Context, actions: &mut UiActions) {
        let Some(mut editor) = self.named_color_editor.take() else {
            return;
        };
        let mut open = true;
        let mut restore_default = false;
        egui::Window::new(format!("HSV color · selection {}", editor.name))
            .id(egui::Id::new("named selection HSV color editor"))
            .open(&mut open)
            .resizable(false)
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

    fn named_expression_editor_window(&mut self, context: &egui::Context, actions: &mut UiActions) {
        let Some(mut editor) = self.named_expression_editor.take() else {
            return;
        };
        let mut open = true;
        let mut close = false;
        egui::Window::new(format!("Edit expression · {}", editor.name))
            .id(egui::Id::new("named selection expression editor"))
            .open(&mut open)
            .default_width(440.0)
            .resizable(true)
            .show(context, |ui| {
                ui.label("Selection expression");
                ui.add(
                    egui::TextEdit::multiline(&mut editor.expression)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                ui.small("Operators: NOT, AND, XOR, OR");
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        actions.manager = Some(ManagerAction::UpdateNamedExpression {
                            name: editor.name.clone(),
                            expression: editor.expression.trim().to_string(),
                        });
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
        if open && !close {
            self.named_expression_editor = Some(editor);
        }
    }

    fn command_editor(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        ui.heading("Selection / Command");
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
        if ui.button("Execute").clicked() || enter {
            let command = self.command_input.trim().to_string();
            if !command.is_empty() {
                self.history.push(command.clone());
                self.history_cursor = None;
                actions.execute = Some(command);
            }
        }
        ui.small(
            "Click inspect · drag orbit · RMB pan · Ctrl/Cmd+drag pan · Ctrl/Cmd+Z undo · Ctrl/Cmd+R redo",
        );
    }

    fn history_previous(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let current = self.history_cursor.unwrap_or(self.history.len());
        let previous = current.saturating_sub(1);
        self.history_cursor = Some(previous);
        self.command_input.clone_from(&self.history[previous]);
    }

    fn history_next(&mut self) {
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

fn molecule_summary(ui: &mut egui::Ui, info: UiInfo<'_>) {
    ui.heading("Molecule");
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

fn focus_chain_number_fields(ui: &mut egui::Ui, chain: &mut String, residue_number: &mut String) {
    ui.horizontal(|ui| {
        ui.label("Chain ID");
        ui.text_edit_singleline(chain);
    });
    ui.horizontal(|ui| {
        ui.label("Residue number");
        ui.text_edit_singleline(residue_number);
    });
}

fn named_selections(
    ui: &mut egui::Ui,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<NamedColorEditor>,
    expression_editor: &mut Option<NamedExpressionEditor>,
) {
    ui.heading("Named selections");
    if info.named_selections.is_empty() {
        ui.weak("Create with: select name: Chain A/LEU*");
        return;
    }
    let (Some(molecule), Some(hierarchy), Some(display)) =
        (info.molecule, info.hierarchy, info.display)
    else {
        return;
    };
    for (name, selection) in info.named_selections {
        let indices: Vec<_> = selection.indices().collect();
        let Some(first_atom) = indices.first().copied() else {
            ui.horizontal(|ui| {
                let response = ui.weak(format!("{name}  (empty)"));
                named_expression_menu(
                    response,
                    name,
                    info.named_selection_expressions.get(name),
                    expression_editor,
                );
                if ui.small_button("×").clicked() {
                    actions.manager = Some(ManagerAction::RemoveNamed(name.clone()));
                }
            });
            continue;
        };
        let style = info
            .named_selection_styles
            .get(name)
            .copied()
            .unwrap_or_default();
        let color = style.color.unwrap_or_else(|| {
            display
                .colors
                .get(first_atom)
                .copied()
                .unwrap_or([0.5, 0.5, 0.5, 1.0])
        });
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("named selection", name)),
            false,
        );
        state
            .show_header(ui, |ui| {
                let effective_mode = style.mode.mode().unwrap_or(display.global_mode);
                let mode_response = mode_button(ui, effective_mode, style.mode)
                    .on_hover_text("Named selection display mode");
                mode_response.context_menu(|ui| {
                    if ui.button("Reset to default").clicked() {
                        actions.manager = Some(ManagerAction::SetNamedMode {
                            name: name.clone(),
                            state: ModeOverride::Inherit,
                        });
                        ui.close();
                    }
                    if ui.button("Set to children").clicked() {
                        actions.manager = Some(ManagerAction::PropagateNamedMode {
                            name: name.clone(),
                            state: ModeOverride::from_mode(effective_mode),
                        });
                        ui.close();
                    }
                });
                if mode_response.clicked() {
                    actions.manager = Some(ManagerAction::SetNamedMode {
                        name: name.clone(),
                        state: next_mode_override(style.mode, display.global_mode),
                    });
                }

                let color_response = color_square(ui, color, style.color.is_some())
                    .on_hover_text("Named selection color");
                color_response.context_menu(|ui| {
                    if ui.button("Reset to default").clicked() {
                        actions.manager = Some(ManagerAction::SetNamedColor {
                            name: name.clone(),
                            color: None,
                        });
                        ui.close();
                    }
                    if ui.button("Set to children").clicked() {
                        actions.manager = Some(ManagerAction::PropagateNamedColor {
                            name: name.clone(),
                            color,
                        });
                        ui.close();
                    }
                });
                if color_response.clicked() {
                    *color_editor = Some(NamedColorEditor {
                        name: name.clone(),
                        hsva: hsva_from_color(color),
                    });
                }

                let visibility_response = visibility_button(ui, style.visibility);
                visibility_response.context_menu(|ui| {
                    if ui.button("Reset to default").clicked() {
                        actions.manager = Some(ManagerAction::SetNamedVisibility {
                            name: name.clone(),
                            state: VisibilityOverride::Inherit,
                        });
                        ui.close();
                    }
                    if ui.button("Set to children").clicked() {
                        let effective = match style.visibility {
                            VisibilityOverride::Inherit => {
                                if display.visible.get(first_atom).copied().unwrap_or(true) {
                                    VisibilityOverride::Show
                                } else {
                                    VisibilityOverride::Hide
                                }
                            }
                            state => state,
                        };
                        actions.manager = Some(ManagerAction::PropagateNamedVisibility {
                            name: name.clone(),
                            state: effective,
                        });
                        ui.close();
                    }
                });
                if visibility_response.clicked() {
                    actions.manager = Some(ManagerAction::SetNamedVisibility {
                        name: name.clone(),
                        state: style.visibility.next(),
                    });
                }

                let label = format!("{name}  ({})", selection.count());
                let name_response = ui
                    .selectable_label(false, label)
                    .on_hover_text("Activate selection · right-click to edit expression");
                if name_response.clicked() {
                    actions.manager = Some(ManagerAction::ActivateNamed(name.clone()));
                }
                named_expression_menu(
                    name_response,
                    name,
                    info.named_selection_expressions.get(name),
                    expression_editor,
                );
                if ui
                    .small_button("×")
                    .on_hover_text("Remove selection")
                    .clicked()
                {
                    actions.manager = Some(ManagerAction::RemoveNamed(name.clone()));
                }
            })
            .body(|ui| {
                named_selection_hierarchy(ui, name, selection, molecule, hierarchy, info, actions);
            });
    }
}

fn named_expression_menu(
    response: egui::Response,
    name: &str,
    expression: Option<&String>,
    editor: &mut Option<NamedExpressionEditor>,
) {
    response.context_menu(|ui| {
        if ui.button("Edit expression").clicked() {
            *editor = Some(NamedExpressionEditor {
                name: name.to_string(),
                expression: expression.cloned().unwrap_or_default(),
            });
            ui.close();
        }
    });
}

fn named_selection_hierarchy(
    ui: &mut egui::Ui,
    selection_name: &str,
    selection: &Selection,
    molecule: &Molecule,
    hierarchy: &MoleculeHierarchy,
    info: UiInfo<'_>,
    actions: &mut UiActions,
) {
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let chain_indices: Vec<_> = chain
            .residues
            .iter()
            .flat_map(|residue| residue.atom_indices.iter().copied())
            .filter(|index| selection.flags().get(*index).copied().unwrap_or(false))
            .collect();
        if chain_indices.is_empty() {
            continue;
        }
        let target = InspectionTarget::Chain(chain_index);
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("named chain", selection_name, chain_index)),
            false,
        );
        state
            .show_header(ui, |ui| {
                named_subset_label(
                    ui,
                    format!(
                        "Chain {} · {} atoms",
                        display_chain(&chain.id),
                        chain_indices.len()
                    ),
                    chain_indices.clone(),
                    target,
                    info.inspection == Some(target),
                    actions,
                );
            })
            .body(|ui| {
                for (residue_index, residue) in chain.residues.iter().enumerate() {
                    let residue_indices: Vec<_> = residue
                        .atom_indices
                        .iter()
                        .copied()
                        .filter(|index| selection.flags().get(*index).copied().unwrap_or(false))
                        .collect();
                    if residue_indices.is_empty() {
                        continue;
                    }
                    let residue_target = InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    };
                    let residue_state =
                        egui::collapsing_header::CollapsingState::load_with_default_open(
                            ui.ctx(),
                            ui.make_persistent_id((
                                "named residue",
                                selection_name,
                                chain_index,
                                residue_index,
                            )),
                            false,
                        );
                    residue_state
                        .show_header(ui, |ui| {
                            named_subset_label(
                                ui,
                                format!("{} · {} atoms", residue.id.label(), residue_indices.len()),
                                residue_indices.clone(),
                                residue_target,
                                info.inspection == Some(residue_target),
                                actions,
                            );
                        })
                        .body(|ui| {
                            for &atom_index in &residue_indices {
                                if let Some(atom) = molecule.atoms.get(atom_index) {
                                    let target = InspectionTarget::Atom(atom_index);
                                    named_subset_label(
                                        ui,
                                        format!(
                                            "#{:<5} {:<4} · {}",
                                            atom.serial,
                                            atom.name,
                                            atom.element.symbol()
                                        ),
                                        vec![atom_index],
                                        target,
                                        info.inspection == Some(target),
                                        actions,
                                    );
                                }
                            }
                        });
                }
            });
    }
}

fn named_subset_label(
    ui: &mut egui::Ui,
    label: String,
    indices: Vec<usize>,
    inspection: InspectionTarget,
    selected: bool,
    actions: &mut UiActions,
) {
    if ui.selectable_label(selected, label).clicked() {
        actions.manager = Some(ManagerAction::SelectNamedSubset {
            indices,
            inspection,
        });
    }
}

fn inspector(ui: &mut egui::Ui, info: UiInfo<'_>) {
    let (Some(molecule), Some(hierarchy), Some(target)) =
        (info.molecule, info.hierarchy, info.inspection)
    else {
        return;
    };
    ui.heading("Inspector");
    match target {
        InspectionTarget::Chain(chain_index) => {
            if let Some(chain) = hierarchy.chains.get(chain_index) {
                property_grid(
                    ui,
                    &[
                        ("Type", "Chain".into()),
                        ("ID", display_chain(&chain.id).into()),
                        ("Residues", chain.residues.len().to_string()),
                        ("Atoms", chain.atom_count.to_string()),
                    ],
                );
            }
        }
        InspectionTarget::Residue {
            chain_index,
            residue_index,
        } => {
            if let Some(residue) = hierarchy.residue(chain_index, residue_index) {
                let chain = hierarchy
                    .chains
                    .get(chain_index)
                    .map_or("?", |chain| display_chain(&chain.id));
                residue_inspector(ui, chain, residue);
            }
        }
        InspectionTarget::Atom(atom_index) => {
            if let Some(atom) = molecule.atoms.get(atom_index) {
                atom_inspector(ui, atom_index, atom);
            }
        }
    }
}

fn residue_inspector(ui: &mut egui::Ui, chain: &str, residue: &ResidueGroup) {
    property_grid(
        ui,
        &[
            ("Type", "Residue".into()),
            ("Chain", chain.into()),
            ("Name", residue.id.name.clone()),
            ("Number", residue.id.number.to_string()),
            (
                "Insertion",
                residue
                    .id
                    .insertion_code
                    .map_or("—".into(), |code| code.to_string()),
            ),
            ("Atoms", residue.atom_indices.len().to_string()),
        ],
    );
}

fn atom_inspector(ui: &mut egui::Ui, atom_index: usize, atom: &Atom) {
    property_grid(
        ui,
        &[
            ("Type", if atom.hetero { "HETATM" } else { "ATOM" }.into()),
            ("Index", atom_index.to_string()),
            ("Serial", atom.serial.to_string()),
            ("Name", atom.name.clone()),
            ("Element", atom.element.symbol().into()),
            ("Chain", display_chain(&atom.chain_id).into()),
            (
                "Residue",
                format!("{} {}", atom.residue_name, atom.residue_number),
            ),
            (
                "Position",
                format!(
                    "{:.3}, {:.3}, {:.3}",
                    atom.position.x, atom.position.y, atom.position.z
                ),
            ),
            ("Occupancy", format!("{:.2}", atom.occupancy)),
            ("B-factor", format!("{:.2}", atom.b_factor)),
        ],
    );
}

fn property_grid(ui: &mut egui::Ui, properties: &[(&str, String)]) {
    egui::Grid::new("inspector properties")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            for (name, value) in properties {
                ui.label(*name);
                ui.monospace(value);
                ui.end_row();
            }
        });
}

fn hierarchy_tree(
    ui: &mut egui::Ui,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<ColorEditor>,
) {
    ui.heading("Hierarchy");
    ui.small("Shift: select range · Ctrl/Cmd: toggle one object");
    let (Some(molecule), Some(hierarchy), Some(display)) =
        (info.molecule, info.hierarchy, info.display)
    else {
        ui.weak("Open a PDB file to browse chains and residues");
        return;
    };
    let inspected_atom_path = info.inspection.and_then(|inspection| {
        let InspectionTarget::Atom(atom_index) = inspection else {
            return None;
        };
        hierarchy
            .chains
            .iter()
            .enumerate()
            .find_map(|(chain_index, chain)| {
                chain
                    .residues
                    .iter()
                    .enumerate()
                    .find(|(_, residue)| residue.atom_indices.contains(&atom_index))
                    .map(|(residue_index, _)| (chain_index, residue_index, atom_index))
            })
    });
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let chain_label = format!(
            "Chain {} · {} residues · {} atoms",
            display_chain(&chain.id),
            chain.residues.len(),
            chain.atom_count
        );
        let chain_target = InspectionTarget::Chain(chain_index);
        let first_atom = chain
            .residues
            .iter()
            .find_map(|residue| residue.atom_indices.first().copied());
        let chain_on_inspected_path = inspected_atom_path
            .is_some_and(|(inspected_chain, _, _)| inspected_chain == chain_index);
        let mut chain_state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("chain", chain_index)),
            false,
        );
        if chain_on_inspected_path {
            chain_state.set_open(true);
        }
        chain_state
            .show_header(ui, |ui| {
                hierarchy_row(
                    ui,
                    display,
                    chain_target,
                    DisplayLevel::Chain,
                    first_atom,
                    &chain_label,
                    info.hierarchy_selection.contains(&chain_target) || chain_on_inspected_path,
                    info.hierarchy_selection,
                    actions,
                    color_editor,
                );
            })
            .body(|ui| {
                for (residue_index, residue) in chain.residues.iter().enumerate() {
                    let residue_label = format!(
                        "{} · {} atoms",
                        residue.id.label(),
                        residue.atom_indices.len()
                    );
                    let residue_target = InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    };
                    let residue_on_inspected_path = inspected_atom_path.is_some_and(
                        |(inspected_chain, inspected_residue, _)| {
                            inspected_chain == chain_index && inspected_residue == residue_index
                        },
                    );
                    let mut residue_state =
                        egui::collapsing_header::CollapsingState::load_with_default_open(
                            ui.ctx(),
                            ui.make_persistent_id(("residue", chain_index, residue_index)),
                            false,
                        );
                    if residue_on_inspected_path {
                        residue_state.set_open(true);
                    }
                    residue_state
                        .show_header(ui, |ui| {
                            hierarchy_row(
                                ui,
                                display,
                                residue_target,
                                DisplayLevel::Residue,
                                residue.atom_indices.first().copied(),
                                &residue_label,
                                info.hierarchy_selection.contains(&residue_target)
                                    || residue_on_inspected_path,
                                info.hierarchy_selection,
                                actions,
                                color_editor,
                            );
                        })
                        .body(|ui| {
                            for &atom_index in &residue.atom_indices {
                                if let Some(atom) = molecule.atoms.get(atom_index) {
                                    let label = format!(
                                        "#{:<5} {:<4} · {}",
                                        atom.serial,
                                        atom.name,
                                        atom.element.symbol()
                                    );
                                    let target = InspectionTarget::Atom(atom_index);
                                    ui.horizontal(|ui| {
                                        hierarchy_row(
                                            ui,
                                            display,
                                            target,
                                            DisplayLevel::Atom,
                                            Some(atom_index),
                                            &label,
                                            info.hierarchy_selection.contains(&target),
                                            info.hierarchy_selection,
                                            actions,
                                            color_editor,
                                        );
                                    });
                                }
                            }
                        });
                }
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn hierarchy_row(
    ui: &mut egui::Ui,
    display: &DisplayState,
    target: InspectionTarget,
    level: DisplayLevel,
    first_atom: Option<usize>,
    label: &str,
    selected: bool,
    hierarchy_selection: &BTreeSet<InspectionTarget>,
    actions: &mut UiActions,
    color_editor: &mut Option<ColorEditor>,
) {
    let Some(atom_index) = first_atom else {
        ui.weak(label);
        return;
    };
    let color = display.color_at_level(atom_index, level);
    let overridden = display.color_is_overridden(atom_index, level);
    let attribute_targets = if hierarchy_selection.contains(&target) {
        hierarchy_selection.iter().copied().collect::<Vec<_>>()
    } else {
        vec![target]
    };
    let direct_mode = display.mode_override(atom_index, level);
    let effective_mode = display.mode_at_level(atom_index, level);
    let mode_response = mode_button(ui, effective_mode, direct_mode);
    mode_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetMode {
                targets: attribute_targets.clone(),
                state: ModeOverride::Inherit,
            });
            ui.close();
        }
        if ui
            .add_enabled(
                attribute_targets
                    .iter()
                    .any(|target| target_has_children(*target)),
                egui::Button::new("Set to children"),
            )
            .clicked()
        {
            actions.manager = Some(ManagerAction::PropagateMode(attribute_targets.clone()));
            ui.close();
        }
    });
    if mode_response.clicked() {
        actions.manager = Some(ManagerAction::SetMode {
            targets: attribute_targets.clone(),
            state: next_mode_override(direct_mode, display.global_mode),
        });
    }
    let color_response = color_square(ui, color, overridden).on_hover_text(if overridden {
        "Custom HSV color (click to edit; orange border = override)"
    } else {
        "Inherited/default color (click to override in HSV)"
    });
    color_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetColor {
                targets: attribute_targets.clone(),
                color: None,
            });
            ui.close();
        }
        if ui
            .add_enabled(
                attribute_targets
                    .iter()
                    .any(|target| target_has_children(*target)),
                egui::Button::new("Set to children"),
            )
            .clicked()
        {
            actions.manager = Some(ManagerAction::PropagateColor(attribute_targets.clone()));
            ui.close();
        }
    });
    if color_response.clicked() {
        let editor_label = if attribute_targets.len() > 1 {
            format!("{} selected objects", attribute_targets.len())
        } else {
            label.to_owned()
        };
        *color_editor = Some(ColorEditor {
            targets: attribute_targets.clone(),
            label: editor_label,
            hsva: hsva_from_color(color),
        });
    }
    let visibility = display.visibility_override(atom_index, level);
    let visibility_response = visibility_button(ui, visibility);
    visibility_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetVisibility {
                targets: attribute_targets.clone(),
                state: VisibilityOverride::Inherit,
            });
            ui.close();
        }
        if ui
            .add_enabled(
                attribute_targets
                    .iter()
                    .any(|target| target_has_children(*target)),
                egui::Button::new("Set to children"),
            )
            .clicked()
        {
            actions.manager = Some(ManagerAction::PropagateVisibility(
                attribute_targets.clone(),
            ));
            ui.close();
        }
    });
    if visibility_response.clicked() {
        actions.manager = Some(ManagerAction::SetVisibility {
            targets: attribute_targets,
            state: visibility.next(),
        });
    }
    if ui.selectable_label(selected, label).clicked() {
        let modifiers = ui.ctx().input(|input| input.modifiers);
        let gesture = if modifiers.shift {
            HierarchySelectionGesture::Range
        } else if modifiers.command || modifiers.ctrl || modifiers.mac_cmd {
            HierarchySelectionGesture::Toggle
        } else {
            HierarchySelectionGesture::Replace
        };
        actions.manager = Some(ManagerAction::SelectHierarchy { target, gesture });
    }
}

fn target_has_children(target: InspectionTarget) -> bool {
    !matches!(target, InspectionTarget::Atom(_))
}

fn next_mode_override(current: ModeOverride, global: DisplayMode) -> ModeOverride {
    if current == ModeOverride::Inherit {
        ModeOverride::from_mode(match global {
            DisplayMode::Cartoon => DisplayMode::BallAndStick,
            DisplayMode::BallAndStick => DisplayMode::Cartoon,
        })
    } else {
        ModeOverride::Inherit
    }
}

fn mode_button(ui: &mut egui::Ui, effective: DisplayMode, direct: ModeOverride) -> egui::Response {
    let label = match effective {
        DisplayMode::Cartoon => "C",
        DisplayMode::BallAndStick => "B",
    };
    let color = if direct == ModeOverride::Inherit {
        egui::Color32::from_gray(125)
    } else {
        egui::Color32::from_rgb(255, 172, 55)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(label).strong().color(color))
            .min_size(egui::vec2(20.0, 17.0)),
    )
    .on_hover_text(match (effective, direct) {
        (DisplayMode::Cartoon, ModeOverride::Inherit) => {
            "Mode: inherited Cartoon (click → Ball & stick override)"
        }
        (DisplayMode::BallAndStick, ModeOverride::Inherit) => {
            "Mode: inherited Ball & stick (click → Cartoon override)"
        }
        (DisplayMode::Cartoon, _) => "Mode override: Cartoon (click → inherit)",
        (DisplayMode::BallAndStick, _) => "Mode override: Ball & stick (click → inherit)",
    })
}

fn color_square(ui: &mut egui::Ui, color: DisplayColor, overridden: bool) -> egui::Response {
    let size = egui::vec2(15.0, 15.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let fill = color32(color);
    let stroke = if overridden {
        egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 172, 55))
    } else {
        egui::Stroke::new(1.0, egui::Color32::from_gray(115))
    };
    ui.painter().rect(
        rect.shrink(1.0),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Inside,
    );
    response
}

fn visibility_button(ui: &mut egui::Ui, state: VisibilityOverride) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(20.0, 17.0), egui::Sense::click());
    let color = match state {
        VisibilityOverride::Inherit => egui::Color32::from_gray(125),
        VisibilityOverride::Show => egui::Color32::from_rgb(60, 205, 105),
        VisibilityOverride::Hide => egui::Color32::from_rgb(235, 70, 70),
    };
    let center = rect.center();
    let left = egui::pos2(rect.left() + 2.0, center.y);
    let right = egui::pos2(rect.right() - 2.0, center.y);
    let top = egui::pos2(center.x, rect.top() + 3.0);
    let bottom = egui::pos2(center.x, rect.bottom() - 3.0);
    let stroke = egui::Stroke::new(1.5, color);
    ui.painter().line_segment([left, top], stroke);
    ui.painter().line_segment([top, right], stroke);
    ui.painter().line_segment([right, bottom], stroke);
    ui.painter().line_segment([bottom, left], stroke);
    ui.painter().circle_filled(center, 2.5, color);
    response.on_hover_text(match state {
        VisibilityOverride::Inherit => "Visibility: inherited (click → force visible)",
        VisibilityOverride::Show => "Visibility: forced visible (click → hide)",
        VisibilityOverride::Hide => "Visibility: hidden (click → inherit)",
    })
}

fn hsva_from_color(color: DisplayColor) -> egui::ecolor::Hsva {
    egui::ecolor::Hsva::from(egui::Rgba::from_rgba_unmultiplied(
        color[0], color[1], color[2], 1.0,
    ))
}

fn color_from_hsva(hsva: egui::ecolor::Hsva) -> DisplayColor {
    let mut color = hsva.to_rgba_unmultiplied();
    color[3] = 1.0;
    color
}

fn color32(color: DisplayColor) -> egui::Color32 {
    egui::Color32::from_rgb(
        (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

fn display_chain(chain: &str) -> &str {
    if chain.is_empty() { "(blank)" } else { chain }
}
