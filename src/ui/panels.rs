/// A sidebar section with a persistent height and a draggable bottom divider.
/// Scrolling keeps large contents from forcing the section past its chosen size.
pub(super) fn section(
    ui: &mut egui::Ui,
    id: &'static str,
    default_height: f32,
    contents: impl FnOnce(&mut egui::Ui),
) {
    // Leave room for the following sections and the hierarchy below them.
    let max_height = (ui.available_height() * 0.65).max(44.0);
    egui::Panel::top(id)
        .default_size(default_height.min(max_height))
        .size_range(44.0..=max_height)
        .resizable(true)
        .show_separator_line(true)
        .show(ui, |ui| {
            egui::ScrollArea::both()
                .id_salt((id, "contents"))
                .auto_shrink([false, false])
                .show(ui, contents);
        });
}
