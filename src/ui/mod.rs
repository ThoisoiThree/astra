use std::collections::BTreeMap;

use molview::{
    ColoringMode, DisplayColor, DisplayLevel, DisplayState, VisibilityOverride,
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
    color_editor: Option<ColorEditor>,
    focus_kind: FocusKind,
    focus_chain: String,
    focus_residue_number: String,
    focus_base_name: String,
    focus_atom_serial: String,
}

#[derive(Debug, Clone)]
struct ColorEditor {
    target: InspectionTarget,
    label: String,
    hsva: egui::ecolor::Hsva,
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
pub enum InspectionTarget {
    Chain(usize),
    Residue {
        chain_index: usize,
        residue_index: usize,
    },
    Atom(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ManagerAction {
    SelectChain(usize),
    SelectResidue {
        chain_index: usize,
        residue_index: usize,
    },
    SelectAtom(usize),
    SetColor {
        target: InspectionTarget,
        color: Option<DisplayColor>,
    },
    CycleVisibility(InspectionTarget),
    SetColoringMode(ColoringMode),
    SetUniformColor(DisplayColor),
    ActivateNamed(String),
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
    pub inspection: Option<InspectionTarget>,
    pub camera: &'a OrbitCamera,
    pub focus_description: &'a str,
}

impl UiState {
    pub fn show(&mut self, root: &mut egui::Ui, info: UiInfo<'_>) -> UiActions {
        let mut actions = UiActions::default();
        egui::Panel::top("toolbar").show(root, |ui| {
            ui.horizontal(|ui| {
                actions.open = ui.button("Open PDB").clicked();
                actions.fit = ui.button("Fit").clicked();
                actions.reset_colors = ui.button("Reset colors").clicked();
                if ui.selectable_label(self.camera_open, "Camera").clicked() {
                    self.camera_open = !self.camera_open;
                }
                if ui
                    .selectable_label(self.coloring_open, "Coloring")
                    .clicked()
                {
                    self.coloring_open = !self.coloring_open;
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
                named_selections(ui, info.named_selections, &mut actions);
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
        self.camera_window(root.ctx(), info, &mut actions);
        self.coloring_window(root.ctx(), info, &mut actions);
        self.color_editor_window(root.ctx(), &mut actions);
        actions.viewport = root.available_rect_before_wrap();
        actions
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
                        target: editor.target,
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
                target: editor.target,
                color: None,
            });
        } else if open {
            self.color_editor = Some(editor);
        }
    }

    fn command_editor(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        ui.heading("Selection / Command");
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.command_input)
                .hint_text("select active_site, chain A")
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
        ui.small("Click atom to inspect · drag to orbit · right/Shift+drag to pan");
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
    selections: &BTreeMap<String, Selection>,
    actions: &mut UiActions,
) {
    ui.heading("Named selections");
    if selections.is_empty() {
        ui.weak("Create with: select name, expression");
        return;
    }
    for (name, selection) in selections {
        ui.horizontal(|ui| {
            let label = format!("{name}  ({})", selection.count());
            if ui.selectable_label(false, label).clicked() {
                actions.manager = Some(ManagerAction::ActivateNamed(name.clone()));
            }
            if ui
                .small_button("×")
                .on_hover_text("Remove selection")
                .clicked()
            {
                actions.manager = Some(ManagerAction::RemoveNamed(name.clone()));
            }
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
    let (Some(molecule), Some(hierarchy), Some(display)) =
        (info.molecule, info.hierarchy, info.display)
    else {
        ui.weak("Open a PDB file to browse chains and residues");
        return;
    };
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
        let chain_state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("chain", chain_index)),
            false,
        );
        chain_state
            .show_header(ui, |ui| {
                hierarchy_row(
                    ui,
                    display,
                    chain_target,
                    DisplayLevel::Chain,
                    first_atom,
                    &chain_label,
                    info.inspection == Some(chain_target),
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
                    let residue_state =
                        egui::collapsing_header::CollapsingState::load_with_default_open(
                            ui.ctx(),
                            ui.make_persistent_id(("residue", chain_index, residue_index)),
                            false,
                        );
                    residue_state
                        .show_header(ui, |ui| {
                            hierarchy_row(
                                ui,
                                display,
                                residue_target,
                                DisplayLevel::Residue,
                                residue.atom_indices.first().copied(),
                                &residue_label,
                                info.inspection == Some(residue_target),
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
                                            info.inspection == Some(target),
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
    actions: &mut UiActions,
    color_editor: &mut Option<ColorEditor>,
) {
    let Some(atom_index) = first_atom else {
        ui.weak(label);
        return;
    };
    let color = display.color_at_level(atom_index, level);
    let overridden = display.color_is_overridden(atom_index, level);
    if color_square(ui, color, overridden)
        .on_hover_text(if overridden {
            "Custom HSV color (click to edit; orange border = override)"
        } else {
            "Inherited/default color (click to override in HSV)"
        })
        .clicked()
    {
        *color_editor = Some(ColorEditor {
            target,
            label: label.to_owned(),
            hsva: hsva_from_color(color),
        });
    }
    let visibility = display.visibility_override(atom_index, level);
    if visibility_button(ui, visibility).clicked() {
        actions.manager = Some(ManagerAction::CycleVisibility(target));
    }
    if ui.selectable_label(selected, label).clicked() {
        actions.manager = Some(match target {
            InspectionTarget::Chain(index) => ManagerAction::SelectChain(index),
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => ManagerAction::SelectResidue {
                chain_index,
                residue_index,
            },
            InspectionTarget::Atom(index) => ManagerAction::SelectAtom(index),
        });
    }
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
