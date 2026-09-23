//! Trajectory timeline below the viewport.

use astra::molecule::trajectory::{LoopMode, PlaybackSettings};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrajectoryAction {
    Load,
    Unload,
    Play,
    Pause,
    Seek(usize),
    Step(i64),
    Settings(PlaybackSettings),
}

/// What the timeline shows about the active trajectory.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryInfo<'a> {
    pub label: &'a str,
    pub format: Option<&'static str>,
    /// Zero while the file is being indexed.
    pub frame_count: usize,
    pub current: usize,
    pub playing: bool,
    pub settings: PlaybackSettings,
    pub time: Option<f64>,
    pub cell: Option<[f64; 6]>,
}

impl UiState {
    pub(super) fn trajectory_timeline(
        &mut self,
        root: &mut egui::Ui,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        let Some(trajectory) = info.trajectory else {
            return;
        };
        egui::Panel::bottom("trajectory timeline").show(root, |ui| {
            ui.add_space(4.0);
            if trajectory.frame_count == 0 {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Indexing {}…", trajectory.label));
                    if ui.button("Cancel").clicked() {
                        actions.trajectory = Some(TrajectoryAction::Unload);
                    }
                });
                ui.add_space(4.0);
                return;
            }
            let last = trajectory.frame_count - 1;
            // Space plays or pauses, arrow keys step, unless a text field has focus.
            let context = ui.ctx().clone();
            if context.memory(|memory| memory.focused().is_none()) {
                context.input(|input| {
                    if input.key_pressed(egui::Key::Space) {
                        actions.trajectory = Some(if trajectory.playing {
                            TrajectoryAction::Pause
                        } else {
                            TrajectoryAction::Play
                        });
                    } else if input.key_pressed(egui::Key::ArrowRight) {
                        actions.trajectory = Some(TrajectoryAction::Step(1));
                    } else if input.key_pressed(egui::Key::ArrowLeft) {
                        actions.trajectory = Some(TrajectoryAction::Step(-1));
                    }
                });
            }
            ui.horizontal(|ui| {
                let button = |ui: &mut egui::Ui, text: &str, hint: &str| {
                    ui.add(egui::Button::new(text).min_size(egui::vec2(28.0, 0.0)))
                        .on_hover_text(hint)
                        .clicked()
                };
                if button(ui, "⏮", "First frame") {
                    actions.trajectory = Some(TrajectoryAction::Seek(0));
                }
                if button(ui, "◀", "Previous frame (←)") {
                    actions.trajectory = Some(TrajectoryAction::Step(-1));
                }
                if trajectory.playing {
                    if button(ui, "⏸", "Pause (Space)") {
                        actions.trajectory = Some(TrajectoryAction::Pause);
                    }
                } else if button(ui, "▶", "Play (Space)") {
                    actions.trajectory = Some(TrajectoryAction::Play);
                }
                if button(ui, "▶|", "Next frame (→)") {
                    actions.trajectory = Some(TrajectoryAction::Step(1));
                }
                if button(ui, "⏭", "Last frame") {
                    actions.trajectory = Some(TrajectoryAction::Seek(last));
                }
                let mut frame = trajectory.current + 1;
                let slider_width = (ui.available_width() - 330.0).max(120.0);
                ui.spacing_mut().slider_width = slider_width;
                let response = ui.add(
                    egui::Slider::new(&mut frame, 1..=trajectory.frame_count)
                        .text(format!("/ {}", trajectory.frame_count)),
                );
                if response.changed() {
                    actions.trajectory = Some(TrajectoryAction::Seek(frame - 1));
                }
                if let Some(time) = trajectory.time {
                    ui.monospace(format_time(time));
                }
            });
            ui.horizontal(|ui| {
                ui.weak(match trajectory.format {
                    Some(format) => format!("{} · {format}", trajectory.label),
                    None => trajectory.label.to_owned(),
                });
                let mut settings = trajectory.settings;
                ui.separator();
                ui.add(
                    egui::DragValue::new(&mut settings.fps)
                        .range(PlaybackSettings::FPS_RANGE)
                        .speed(0.5)
                        .suffix(" fps"),
                );
                ui.add(
                    egui::DragValue::new(&mut settings.stride)
                        .range(1..=last.max(1))
                        .prefix("step "),
                )
                .on_hover_text("Frames advanced per playback step");
                egui::ComboBox::from_id_salt("trajectory loop mode")
                    .selected_text(settings.loop_mode.label())
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        for mode in LoopMode::ALL {
                            ui.selectable_value(&mut settings.loop_mode, mode, mode.label());
                        }
                    });
                if settings != trajectory.settings {
                    actions.trajectory = Some(TrajectoryAction::Settings(settings));
                }
                if let Some([a, b, c, ..]) = trajectory.cell {
                    ui.separator();
                    ui.weak(format!("cell {a:.1} × {b:.1} × {c:.1} Å"));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Unload").clicked() {
                        actions.trajectory = Some(TrajectoryAction::Unload);
                    }
                });
            });
            ui.add_space(2.0);
        });
    }
}

fn format_time(picoseconds: f64) -> String {
    if picoseconds.abs() >= 1000.0 {
        format!("{:.3} ns", picoseconds / 1000.0)
    } else {
        format!("{picoseconds:.2} ps")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn times_switch_to_nanoseconds() {
        assert_eq!(super::format_time(12.5), "12.50 ps");
        assert_eq!(super::format_time(2500.0), "2.500 ns");
    }
}
