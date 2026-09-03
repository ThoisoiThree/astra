use super::*;

pub(super) fn named_selections(
    ui: &mut egui::Ui,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<NamedColorEditor>,
    expression_editor: &mut Option<NamedExpressionEditor>,
    rename_editor: &mut Option<RenameEditor>,
) {
    ui.heading("Selections");
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
        let status = info
            .named_selection_statuses
            .get(name)
            .unwrap_or(&SelectionStatus::Valid);
        let indices: Vec<_> = selection.indices().collect();
        let Some(first_atom) = indices.first().copied() else {
            ui.horizontal(|ui| {
                selection_status_badge(ui, status);
                let response = ui.weak(format!("{name}  (empty · {})", status.label()));
                named_expression_menu(
                    response,
                    name,
                    info.named_selection_expressions.get(name),
                    expression_editor,
                    rename_editor,
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
                selection_status_badge(ui, status);
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
                                if display.visible.get(first_atom).unwrap_or(true) {
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

                let label = format!("{name}  ({} · {})", selection.count(), status.label());
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
                    rename_editor,
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

pub(super) fn selection_status_badge(ui: &mut egui::Ui, status: &SelectionStatus) {
    let (color, detail) = match status {
        SelectionStatus::Valid => (
            egui::Color32::from_rgb(70, 190, 105),
            "Expression is valid".into(),
        ),
        SelectionStatus::Broken(names) => (
            egui::Color32::from_rgb(225, 90, 75),
            format!("Missing dependencies: {}", names.join(", ")),
        ),
        SelectionStatus::Cyclic => (
            egui::Color32::from_rgb(215, 80, 190),
            "Cyclic selection dependency".into(),
        ),
        SelectionStatus::Stale(error) => (
            egui::Color32::from_rgb(225, 165, 55),
            format!("Using the last valid result: {error}"),
        ),
    };
    ui.colored_label(color, "●").on_hover_text(detail);
}

pub(super) fn named_expression_menu(
    response: egui::Response,
    name: &str,
    expression: Option<&String>,
    editor: &mut Option<NamedExpressionEditor>,
    rename_editor: &mut Option<RenameEditor>,
) {
    response.context_menu(|ui| {
        if ui.button("Rename").clicked() {
            *rename_editor = Some(RenameEditor {
                target: RenameTarget::NamedSelection(name.to_owned()),
                name: name.to_owned(),
            });
            ui.close();
        }
        if ui.button("Edit expression").clicked() {
            *editor = Some(NamedExpressionEditor {
                name: name.to_string(),
                expression: expression.cloned().unwrap_or_default(),
            });
            ui.close();
        }
    });
}

pub(super) fn named_selection_hierarchy(
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
            .filter(|index| selection.flags().get(*index).unwrap_or(false))
            .collect();
        if chain_indices.is_empty() {
            continue;
        }
        let target = InspectionTarget::Chain(chain_index);
        let chain_name = info
            .hierarchy_names
            .get(&target)
            .cloned()
            .unwrap_or_else(|| format!("Chain {}", display_chain(&chain.id)));
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("named chain", selection_name, chain_index)),
            false,
        );
        state
            .show_header(ui, |ui| {
                named_subset_label(
                    ui,
                    format!("{} · {} atoms", chain_name, chain_indices.len()),
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
                        .filter(|index| selection.flags().get(*index).unwrap_or(false))
                        .collect();
                    if residue_indices.is_empty() {
                        continue;
                    }
                    let residue_target = InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    };
                    let residue_name = info
                        .hierarchy_names
                        .get(&residue_target)
                        .cloned()
                        .unwrap_or_else(|| residue.id.label());
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
                                format!("{} · {} atoms", residue_name, residue_indices.len()),
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
                                    let atom_name =
                                        info.hierarchy_names.get(&target).cloned().unwrap_or_else(
                                            || format!("#{} {}", atom.serial, atom.name),
                                        );
                                    named_subset_label(
                                        ui,
                                        format!("{} · {}", atom_name, atom.element.symbol()),
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

pub(super) fn named_subset_label(
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
impl UiState {
    pub(super) fn named_expression_editor_window(
        &mut self,
        context: &egui::Context,
        actions: &mut UiActions,
    ) {
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

    pub(super) fn rename_window(&mut self, context: &egui::Context, actions: &mut UiActions) {
        let Some(mut editor) = self.rename_editor.take() else {
            return;
        };
        let mut open = true;
        let mut apply = false;
        egui::Window::new("Rename object")
            .id(egui::Id::new("rename object window"))
            .open(&mut open)
            .resizable(false)
            .show(context, |ui| {
                ui.label("Display name");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut editor.name)
                        .desired_width(280.0)
                        .hint_text("Object name"),
                );
                let enter = response.lost_focus()
                    && ui.ctx().input(|input| input.key_pressed(egui::Key::Enter));
                apply = ui
                    .add_enabled(!editor.name.trim().is_empty(), egui::Button::new("Rename"))
                    .clicked()
                    || (enter && !editor.name.trim().is_empty());
            });
        if apply {
            let name = editor.name.trim().to_owned();
            actions.manager = Some(match editor.target {
                RenameTarget::Hierarchy(target) => ManagerAction::RenameHierarchy {
                    target,
                    name: Some(name),
                },
                RenameTarget::NamedSelection(old_name) => ManagerAction::RenameNamedSelection {
                    old_name,
                    new_name: name,
                },
                RenameTarget::Measurement(id) => ManagerAction::RenameMeasurement { id, name },
            });
        } else if open {
            self.rename_editor = Some(editor);
        }
    }
}
