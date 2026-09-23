use super::*;

use astra::render::{ExportBackground, ImageExport, MAX_RENDER_SCALE};

impl UiState {
    pub(super) fn open_export(&mut self) {
        self.export_open = true;
    }

    pub(super) fn export_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.export_open {
            return;
        }
        let [viewport_width, viewport_height] = info.viewport_pixels;
        if !self.export.customized && viewport_width > 0 && viewport_height > 0 {
            // Twice the viewport keeps its framing at print-friendly resolution.
            self.export.width = (viewport_width * 2).min(info.max_texture_dimension);
            self.export.height = (viewport_height * 2).min(info.max_texture_dimension);
        }
        let mut open = self.export_open;
        egui::Window::new("Export image")
            .id(egui::Id::new("image export window"))
            .open(&mut open)
            .resizable(false)
            .default_width(320.0)
            .show(context, |ui| {
                let form = &mut self.export;
                let limit = info.max_texture_dimension.max(1);
                egui::Grid::new("export size").num_columns(2).show(ui, |ui| {
                    ui.label("Width");
                    form.customized |= ui
                        .add(egui::DragValue::new(&mut form.width).range(16..=limit).suffix(" px"))
                        .changed();
                    ui.end_row();
                    ui.label("Height");
                    form.customized |= ui
                        .add(egui::DragValue::new(&mut form.height).range(16..=limit).suffix(" px"))
                        .changed();
                    ui.end_row();
                });
                ui.horizontal_wrapped(|ui| {
                    let mut preset = |ui: &mut egui::Ui, label: &str, width: u32, height: u32| {
                        if ui.small_button(label).clicked() {
                            form.width = width.clamp(16, limit);
                            form.height = height.clamp(16, limit);
                            form.customized = true;
                        }
                    };
                    preset(ui, "Viewport", viewport_width, viewport_height);
                    preset(ui, "2× viewport", viewport_width * 2, viewport_height * 2);
                    preset(ui, "Full HD", 1920, 1080);
                    preset(ui, "4K", 3840, 2160);
                    preset(ui, "Square 3000", 3000, 3000);
                });
                ui.add(
                    egui::Slider::new(&mut form.supersampling, 1..=MAX_RENDER_SCALE)
                        .text("Supersampling")
                        .suffix("×"),
                )
                .on_hover_text(
                    "Renders each output pixel from N×N samples, antialiasing every edge. \
                     It is reduced automatically when the scene would exceed the GPU texture limit.",
                );
                egui::ComboBox::from_label("Background")
                    .selected_text(form.background.label())
                    .show_ui(ui, |ui| {
                        for background in ExportBackground::ALL {
                            ui.selectable_value(&mut form.background, background, background.label());
                        }
                    });
                if form.background == ExportBackground::Transparent && info.camera.depth_of_field.enabled {
                    ui.small("Depth of field needs an opaque background and is omitted.");
                }
                ui.add(egui::DragValue::new(&mut form.dots_per_inch).range(36..=2400).prefix("Resolution ").suffix(" dpi"));
                let inches = (form.width as f32 / form.dots_per_inch as f32, form.height as f32 / form.dots_per_inch as f32);
                ui.small(format!(
                    "Print size {:.1} × {:.1} cm",
                    inches.0 * 2.54,
                    inches.1 * 2.54
                ));
                ui.separator();
                let enabled = info.molecule.is_some();
                if ui
                    .add_enabled(enabled, egui::Button::new("Export PNG…"))
                    .clicked()
                {
                    actions.export_image = Some(ExportRequest {
                        image: ImageExport {
                            width: form.width,
                            height: form.height,
                            supersampling: form.supersampling,
                            background: form.background,
                        },
                        dots_per_inch: form.dots_per_inch,
                    });
                }
            });
        self.export_open = open;
    }
}
