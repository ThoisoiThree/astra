use super::*;

fn detail(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.weak(label);
    ui.label(if value.is_empty() {
        "Not reported"
    } else {
        value
    });
    ui.end_row();
}

impl UiState {
    pub(super) fn info_window(
        &mut self,
        context: &egui::Context,
        info: UiInfo<'_>,
        _actions: &mut UiActions,
    ) {
        if !self.info_open {
            return;
        }
        // No external commands or system probes on the UI thread.
        static CPU_THREADS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            std::thread::available_parallelism()
                .map(|count| count.to_string())
                .unwrap_or_else(|_| "Not reported".into())
        });
        egui::Window::new("Info")
            .id(egui::Id::new("application info"))
            .open(&mut self.info_open)
            .default_width(420.0)
            .resizable(true)
            .show(context, |ui| {
                ui.heading("Astra");
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.label("© Thoisoi Three");
                ui.label(format!("License: {}", env!("CARGO_PKG_LICENSE")));
                ui.add_enabled(false, egui::Button::new("Repository — coming soon"))
                    .on_disabled_hover_text("Repository link will be added later");
                ui.separator();
                ui.checkbox(
                    &mut self.experimental_features,
                    "Enable experimental features",
                );
                ui.separator();
                ui.strong("System");
                egui::Grid::new("application system info")
                    .num_columns(2)
                    .spacing(egui::vec2(18.0, 6.0))
                    .show(ui, |ui| {
                        detail(ui, "OS", std::env::consts::OS);
                        detail(ui, "Architecture", std::env::consts::ARCH);
                        detail(ui, "Available CPU threads", CPU_THREADS.as_str());
                        detail(
                            ui,
                            "Build",
                            if cfg!(debug_assertions) {
                                "Debug"
                            } else {
                                "Release"
                            },
                        );
                    });
                ui.separator();
                ui.strong("Graphics");
                let adapter = info.adapter_info;
                egui::Grid::new("application graphics info")
                    .num_columns(2)
                    .spacing(egui::vec2(18.0, 6.0))
                    .show(ui, |ui| {
                        detail(ui, "wgpu version", env!("ASTRA_WGPU_VERSION"));
                        detail(ui, "Active backend", &format!("{:?}", adapter.backend));
                        detail(
                            ui,
                            "Presentation",
                            &format!("{:?}", info.render_stats.present_mode),
                        );
                        detail(ui, "GPU", &adapter.name);
                        detail(ui, "Device type", &format!("{:?}", adapter.device_type));
                        detail(ui, "Driver", &adapter.driver);
                        detail(ui, "Driver details / version", &adapter.driver_info);
                    });
                #[cfg(target_os = "windows")]
                {
                    use astra::render::backend::WindowsBackend;
                    ui.add_space(8.0);
                    let mut selected = info.windows_backend;
                    egui::ComboBox::from_label("Backend on next launch")
                        .selected_text(selected.label())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut selected,
                                WindowsBackend::Vulkan,
                                "Vulkan (default)",
                            );
                            ui.selectable_value(&mut selected, WindowsBackend::Dx12, "DX12");
                        });
                    if selected != info.windows_backend {
                        _actions.windows_backend = Some(selected);
                    }
                    ui.small("Backend changes require restarting Astra.");
                    if let Some(error) = &self.latest_error {
                        ui.colored_label(ui.visuals().error_fg_color, error);
                    }
                }
            });
    }
}
