use std::collections::{BTreeMap, BTreeSet};

use molview::{
    AmbientOcclusionSettings, ColoringMode, DisplayColor, DisplayLevel, DisplayMode, DisplayState,
    ModeOverride, NamedSelectionStyle, VisibilityOverride,
    camera::OrbitCamera,
    measurement::{MAX_MEASUREMENT_THICKNESS, MeasurementLine},
    molecule::{Atom, Molecule, MoleculeHierarchy, ResidueGroup},
    render::RenderStats,
    selection::{Selection, SelectionStatus},
};

mod actions;
mod camera;
mod coloring;
mod file;
mod hierarchy;
mod mode;
mod selections;

use actions::*;
use camera::*;
use hierarchy::*;
use mode::*;
use selections::*;

#[derive(Debug, Default)]
pub struct UiState {
    pub command_input: String,
    pub latest_error: Option<String>,
    history: Vec<String>,
    history_cursor: Option<usize>,
    file_open: bool,
    fetch_open: bool,
    fetch_id: String,
    camera_open: bool,
    coloring_open: bool,
    mode_open: bool,
    actions_open: bool,
    performance_overlay: bool,
    measurement_first: String,
    measurement_second: String,
    color_editor: Option<ColorEditor>,
    named_color_editor: Option<NamedColorEditor>,
    measurement_color_editor: Option<MeasurementColorEditor>,
    named_expression_editor: Option<NamedExpressionEditor>,
    rename_editor: Option<RenameEditor>,
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
struct MeasurementColorEditor {
    id: u64,
    hsva: egui::ecolor::Hsva,
}

#[derive(Debug, Clone)]
struct NamedExpressionEditor {
    name: String,
    expression: String,
}

#[derive(Debug, Clone)]
struct RenameEditor {
    target: RenameTarget,
    name: String,
}

#[derive(Debug, Clone)]
enum RenameTarget {
    Hierarchy(InspectionTarget),
    NamedSelection(String),
    Measurement(u64),
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
    SetAmbientOcclusion(AmbientOcclusionSettings),
    SetUniformColor(DisplayColor),
    ActivateNamed(String),
    UpdateNamedExpression {
        name: String,
        expression: String,
    },
    RemoveNamed(String),
    CreateMeasurement {
        first_selection: String,
        second_selection: String,
    },
    SetMeasurementColor {
        id: u64,
        color: Option<DisplayColor>,
    },
    SetMeasurementVisibility {
        id: u64,
        state: VisibilityOverride,
    },
    SetMeasurementLabelSize {
        id: u64,
        size: f32,
    },
    SetMeasurementThickness {
        id: u64,
        thickness: f32,
    },
    RemoveMeasurement(u64),
    SaveCurrentSelection(String),
    RenameHierarchy {
        target: InspectionTarget,
        name: Option<String>,
    },
    RenameNamedSelection {
        old_name: String,
        new_name: String,
    },
    RenameMeasurement {
        id: u64,
        name: String,
    },
}

#[derive(Debug)]
pub struct UiActions {
    pub open: bool,
    pub fetch: Option<String>,
    pub cancel_fetch: bool,
    pub cancel_background_job: Option<u64>,
    pub activate_session: Option<u64>,
    pub close_session: Option<u64>,
    pub save: bool,
    pub save_as: bool,
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
            fetch: None,
            cancel_fetch: false,
            cancel_background_job: None,
            activate_session: None,
            close_session: None,
            save: false,
            save_as: false,
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

#[derive(Debug, Clone)]
pub struct SessionTab {
    pub id: u64,
    pub label: String,
    pub dirty: bool,
}

#[derive(Clone, Copy)]
pub struct UiInfo<'a> {
    pub filename: Option<&'a str>,
    pub molecule_id: Option<&'a str>,
    pub session_tabs: &'a [SessionTab],
    pub active_session_id: Option<u64>,
    pub atom_count: usize,
    pub bond_count: usize,
    pub selection_count: usize,
    pub molecule: Option<&'a Molecule>,
    pub hierarchy: Option<&'a MoleculeHierarchy>,
    pub display: Option<&'a DisplayState>,
    pub named_selections: &'a BTreeMap<String, Selection>,
    pub named_selection_expressions: &'a BTreeMap<String, String>,
    pub named_selection_styles: &'a BTreeMap<String, NamedSelectionStyle>,
    pub named_selection_statuses: &'a BTreeMap<String, SelectionStatus>,
    pub measurement_lines: &'a [MeasurementLine],
    pub hierarchy_names: &'a BTreeMap<InspectionTarget, String>,
    pub inspection: Option<InspectionTarget>,
    pub hierarchy_selection: &'a BTreeSet<InspectionTarget>,
    pub camera: &'a OrbitCamera,
    pub focus_description: &'a str,
    pub pivot_description: &'a str,
    pub fetching_pdb_id: Option<&'a str>,
    pub fetch_downloaded_bytes: u64,
    pub fetch_total_bytes: Option<u64>,
    pub fetch_bytes_per_second: f64,
    pub background_job_id: Option<u64>,
    pub background_stage: Option<&'a str>,
    pub background_progress: f32,
    pub render_stats: RenderStats,
}

impl UiState {
    pub fn show(&mut self, root: &mut egui::Ui, info: UiInfo<'_>) -> UiActions {
        let mut actions = UiActions::default();
        let mut file_button = None;
        egui::Panel::top("toolbar").show(root, |ui| {
            ui.horizontal(|ui| {
                let response = ui.selectable_label(self.file_open, "File");
                if response.clicked() {
                    self.file_open = !self.file_open;
                }
                file_button = Some(response);
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
                if ui.selectable_label(self.actions_open, "Actions").clicked() {
                    self.actions_open = !self.actions_open;
                }
                ui.separator();
                ui.strong(info.filename.unwrap_or("No molecule loaded"));
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
        });
        if let Some(response) = &file_button {
            self.file_panel(response, info, &mut actions);
        }

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
                    &mut self.rename_editor,
                );
                ui.separator();
                inspector(ui, info);
                if info.inspection.is_some() {
                    ui.separator();
                }
                if !info.measurement_lines.is_empty() {
                    measurement_lines(
                        ui,
                        info,
                        &mut actions,
                        &mut self.measurement_color_editor,
                        &mut self.rename_editor,
                    );
                    ui.separator();
                }
                hierarchy_tree(
                    ui,
                    info,
                    &mut actions,
                    &mut self.color_editor,
                    &mut self.rename_editor,
                );
            });
        if !info.session_tabs.is_empty() {
            egui::Panel::top("document tabs")
                .exact_size(32.0)
                .show(root, |ui| {
                    egui::ScrollArea::horizontal()
                        .id_salt("document tabs scroll")
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                for tab in info.session_tabs {
                                    ui.group(|ui| {
                                        ui.horizontal(|ui| {
                                            let selected = info.active_session_id == Some(tab.id);
                                            if ui
                                                .selectable_label(
                                                    selected,
                                                    format!(
                                                        "◉ {}{}",
                                                        tab.label,
                                                        if tab.dirty { " ●" } else { "" }
                                                    ),
                                                )
                                                .clicked()
                                            {
                                                actions.activate_session = Some(tab.id);
                                            }
                                            if ui.small_button("×").clicked() {
                                                actions.close_session = Some(tab.id);
                                            }
                                        });
                                    });
                                }
                            });
                        });
                });
        }
        let viewport = root.available_rect_before_wrap();
        measurement_labels(root.painter(), viewport, info);
        if self.performance_overlay {
            performance_overlay(root.ctx(), viewport, info.render_stats);
        }
        self.mode_window(root.ctx(), info, &mut actions);
        self.coloring_window(root.ctx(), info, &mut actions);
        self.camera_window(root.ctx(), info, &mut actions);
        self.actions_window(root.ctx(), info, &mut actions);
        self.color_editor_window(root.ctx(), &mut actions);
        self.named_color_editor_window(root.ctx(), &mut actions);
        self.measurement_color_editor_window(root.ctx(), &mut actions);
        self.named_expression_editor_window(root.ctx(), &mut actions);
        self.rename_window(root.ctx(), &mut actions);
        self.fetch_window(root.ctx(), info, &mut actions);
        actions.viewport = viewport;
        actions
    }
}

fn target_has_children(target: InspectionTarget) -> bool {
    !matches!(target, InspectionTarget::Atom(_))
}

fn next_mode_override(current: ModeOverride, global: DisplayMode) -> ModeOverride {
    let next = current.mode().unwrap_or(global).next();
    if next == global {
        ModeOverride::Inherit
    } else {
        ModeOverride::from_mode(next)
    }
}

fn mode_button(ui: &mut egui::Ui, effective: DisplayMode, direct: ModeOverride) -> egui::Response {
    let label = match effective {
        DisplayMode::Cartoon => "C",
        DisplayMode::BallAndStick => "B",
        DisplayMode::Toon => "T",
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
        (mode, ModeOverride::Inherit) => match mode {
            DisplayMode::Cartoon => "Mode: inherited Cartoon (click → Ball & stick override)",
            DisplayMode::BallAndStick => "Mode: inherited Ball & stick (click → Toon override)",
            DisplayMode::Toon => "Mode: inherited Toon (click → Cartoon override)",
        },
        (DisplayMode::Cartoon, _) => "Mode override: Cartoon (click → Ball & stick)",
        (DisplayMode::BallAndStick, _) => "Mode override: Ball & stick (click → Toon)",
        (DisplayMode::Toon, _) => "Mode override: Toon (click → Cartoon/inherit)",
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

fn is_simple_selection_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
}

fn current_selection_name(input: &str) -> Option<&str> {
    let input = input.trim();
    if is_simple_selection_name(input) {
        return Some(input);
    }
    let (keyword, remainder) = input.split_once(char::is_whitespace)?;
    if !keyword.eq_ignore_ascii_case("select") {
        return None;
    }
    let name = remainder.trim().strip_suffix(':')?.trim();
    is_simple_selection_name(name).then_some(name)
}

fn display_chain(chain: &str) -> &str {
    if chain.is_empty() { "(blank)" } else { chain }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_speed(bytes_per_second: f64) -> String {
    if bytes_per_second <= 0.0 {
        "Waiting for data…".into()
    } else {
        format!("{}/s", format_bytes(bytes_per_second.round() as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_button_distinguishes_current_name_from_full_command() {
        assert_eq!(current_selection_name("active_site"), Some("active_site"));
        assert_eq!(
            current_selection_name("select active_site:"),
            Some("active_site")
        );
        assert_eq!(current_selection_name("select all"), None);
        assert_eq!(current_selection_name("chain A"), None);
    }

    #[test]
    fn hierarchy_mode_button_cycles_through_toon_and_back_to_inherit() {
        let global = DisplayMode::Cartoon;
        let ball = next_mode_override(ModeOverride::Inherit, global);
        let toon = next_mode_override(ball, global);
        let inherited = next_mode_override(toon, global);
        assert_eq!(ball, ModeOverride::BallAndStick);
        assert_eq!(toon, ModeOverride::Toon);
        assert_eq!(inherited, ModeOverride::Inherit);
    }
}
