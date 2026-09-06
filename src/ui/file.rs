use super::*;

impl UiState {
    pub(super) fn recovery_window(
        &self,
        context: &egui::Context,
        path: Option<&std::path::Path>,
        actions: &mut UiActions,
    ) {
        let Some(path) = path else {
            return;
        };
        let mut open = true;
        egui::Window::new("Recover autosaved scene")
            .id(egui::Id::new("scene recovery"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(context, |ui| {
                ui.label("An autosaved scene is available.");
                let name = path
                    .file_name()
                    .unwrap_or(path.as_os_str())
                    .to_string_lossy();
                ui.label(name).on_hover_text(path.display().to_string());
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Restore").clicked() {
                        actions.recovery = Some(RecoveryAction::Restore);
                    }
                    if ui
                        .button("Discard")
                        .on_hover_text("Delete this autosave")
                        .clicked()
                    {
                        actions.recovery = Some(RecoveryAction::Discard);
                    }
                    if ui
                        .button("Later")
                        .on_hover_text("Keep autosaves for the next launch")
                        .clicked()
                    {
                        actions.recovery = Some(RecoveryAction::Later);
                    }
                });
            });
        if !open {
            actions.recovery = Some(RecoveryAction::Later);
        }
    }

    pub(super) fn file_panel(
        &mut self,
        button: &egui::Response,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        let mut open = self.file_open;
        let mut close = false;
        egui::Popup::from_response(button)
            .open_bool(&mut open)
            .width(180.0)
            .show(|ui| {
                if ui.button("Open").clicked() {
                    actions.open = true;
                    close = true;
                }
                if ui.button("Fetch").clicked() {
                    self.fetch_open = true;
                    close = true;
                }
                let has_structure = info.molecule.is_some();
                if ui
                    .add_enabled(has_structure, egui::Button::new("Save as…"))
                    .clicked()
                {
                    actions.save_as = true;
                    close = true;
                }
                if ui
                    .add_enabled(has_structure, egui::Button::new("Save"))
                    .clicked()
                {
                    actions.save = true;
                    close = true;
                }
            });
        self.file_open = open && !close;
    }

    pub(super) fn fetch_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        actions: &mut UiActions,
    ) {
        if !self.fetch_open {
            return;
        }
        let response =
            egui::Modal::new(egui::Id::new("fetch PDB structure modal")).show(context, |ui| {
                ui.set_min_width(360.0);
                ui.heading("Fetch");
                ui.label("Download a structure from RCSB PDB");
                ui.add_space(6.0);

                if let Some(id) = info.fetching_pdb_id {
                    ui.label(format!("Downloading {id}"));
                    let progress = info.fetch_total_bytes.map_or(0.0, |total| {
                        if total == 0 {
                            0.0
                        } else {
                            info.fetch_downloaded_bytes as f32 / total as f32
                        }
                    });
                    let progress_text = info.fetch_total_bytes.map_or_else(
                        || format_bytes(info.fetch_downloaded_bytes),
                        |total| {
                            format!(
                                "{} / {}",
                                format_bytes(info.fetch_downloaded_bytes),
                                format_bytes(total)
                            )
                        },
                    );
                    ui.add(
                        egui::ProgressBar::new(progress)
                            .desired_width(340.0)
                            .text(progress_text)
                            .animate(info.fetch_total_bytes.is_none()),
                    );
                    ui.label(format!(
                        "{} · saved to ~/downloads/pdb/",
                        format_speed(info.fetch_bytes_per_second)
                    ));
                    ui.add_space(6.0);
                    if ui.button("Cancel").clicked() {
                        actions.cancel_fetch = true;
                        self.fetch_open = false;
                    }
                } else {
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.fetch_id)
                            .hint_text("PDB ID, e.g. 4R8P")
                            .desired_width(340.0),
                    );
                    let enter = edit.lost_focus()
                        && context.input(|input| input.key_pressed(egui::Key::Enter));
                    ui.small("The downloaded file will be saved to ~/downloads/pdb/");
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        let can_fetch = !self.fetch_id.trim().is_empty();
                        if ui
                            .add_enabled(can_fetch, egui::Button::new("Fetch"))
                            .clicked()
                            || (enter && can_fetch)
                        {
                            actions.fetch = Some(self.fetch_id.trim().to_owned());
                        }
                        if ui.button("Cancel").clicked() {
                            self.fetch_open = false;
                        }
                    });
                }
            });
        if response.should_close() && info.fetching_pdb_id.is_none() {
            self.fetch_open = false;
        }
    }

    pub fn close_fetch(&mut self) {
        self.fetch_open = false;
    }

    pub fn document_changed(&mut self) {
        self.color_editor = None;
        self.named_color_editor = None;
        self.measurement_color_editor = None;
        self.named_expression_editor = None;
        self.rename_editor = None;
    }
}
