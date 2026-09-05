use super::*;

#[derive(Debug, Clone, Copy)]
enum SelectionListRow<'a> {
    Selection(&'a str),
    Chain {
        selection_name: &'a str,
        chain_index: usize,
        atom_count: usize,
    },
    Residue {
        selection_name: &'a str,
        chain_index: usize,
        residue_index: usize,
        atom_count: usize,
    },
    Atom {
        selection_name: &'a str,
        atom_index: usize,
    },
}

fn selection_tree_open(ui: &egui::Ui, id: egui::Id) -> bool {
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false).is_open()
}

fn selection_disclosure_button(ui: &mut egui::Ui, id: egui::Id, open: bool) {
    let symbol = if open { "▾" } else { "▸" };
    let response = ui
        .push_id(id.with("toggle"), |ui| {
            ui.add_sized(
                [ui.spacing().indent, ui.spacing().interact_size.y],
                egui::Button::new(symbol).frame(false),
            )
        })
        .inner;
    if response.clicked() {
        let mut state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        state.toggle(ui);
        state.store(ui.ctx());
    }
}

fn visible_selection_rows<'a>(
    ui: &egui::Ui,
    selections: &'a BTreeMap<String, Selection>,
    hierarchy: &MoleculeHierarchy,
) -> Vec<SelectionListRow<'a>> {
    let mut rows = Vec::new();
    for (name, selection) in selections {
        let selection_name = name.as_str();
        rows.push(SelectionListRow::Selection(selection_name));
        if selection.count() == 0
            || !selection_tree_open(
                ui,
                ui.make_persistent_id(("named selection", selection_name)),
            )
        {
            continue;
        }
        for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
            let atom_count = chain
                .residues
                .iter()
                .flat_map(|residue| &residue.atom_indices)
                .filter(|&&atom_index| selection.flags().get(atom_index).unwrap_or(false))
                .count();
            if atom_count == 0 {
                continue;
            }
            rows.push(SelectionListRow::Chain {
                selection_name,
                chain_index,
                atom_count,
            });
            if !selection_tree_open(
                ui,
                ui.make_persistent_id(("named chain", selection_name, chain_index)),
            ) {
                continue;
            }
            for (residue_index, residue) in chain.residues.iter().enumerate() {
                let residue_atom_count = residue
                    .atom_indices
                    .iter()
                    .filter(|&&atom_index| selection.flags().get(atom_index).unwrap_or(false))
                    .count();
                if residue_atom_count == 0 {
                    continue;
                }
                rows.push(SelectionListRow::Residue {
                    selection_name,
                    chain_index,
                    residue_index,
                    atom_count: residue_atom_count,
                });
                if selection_tree_open(
                    ui,
                    ui.make_persistent_id((
                        "named residue",
                        selection_name,
                        chain_index,
                        residue_index,
                    )),
                ) {
                    rows.extend(
                        residue
                            .atom_indices
                            .iter()
                            .copied()
                            .filter(|&atom_index| {
                                selection.flags().get(atom_index).unwrap_or(false)
                            })
                            .map(|atom_index| SelectionListRow::Atom {
                                selection_name,
                                atom_index,
                            }),
                    );
                }
            }
        }
    }
    rows
}

#[allow(clippy::too_many_arguments)]
fn named_selection_header(
    ui: &mut egui::Ui,
    name: &str,
    selection: &Selection,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<NamedColorEditor>,
    expression_editor: &mut Option<NamedExpressionEditor>,
    rename_editor: &mut Option<RenameEditor>,
) {
    let status = info
        .named_selection_statuses
        .get(name)
        .unwrap_or(&SelectionStatus::Valid);
    let Some(first_atom) = selection.indices().next() else {
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
            actions.manager = Some(ManagerAction::RemoveNamed(name.to_owned()));
        }
        return;
    };
    let Some(display) = info.display else {
        return;
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

    selection_status_badge(ui, status);
    let effective_mode = style.mode.mode().unwrap_or(display.global_mode);
    let mode_response =
        mode_button(ui, effective_mode, style.mode).on_hover_text("Named selection display mode");
    mode_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetNamedMode {
                name: name.to_owned(),
                state: ModeOverride::Inherit,
            });
            ui.close();
        }
        if ui.button("Set to children").clicked() {
            actions.manager = Some(ManagerAction::PropagateNamedMode {
                name: name.to_owned(),
                state: ModeOverride::from_mode(effective_mode),
            });
            ui.close();
        }
    });
    if mode_response.clicked() {
        actions.manager = Some(ManagerAction::SetNamedMode {
            name: name.to_owned(),
            state: next_mode_override(style.mode, display.global_mode),
        });
    }

    let color_response =
        color_square(ui, color, style.color.is_some()).on_hover_text("Named selection color");
    color_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetNamedColor {
                name: name.to_owned(),
                color: None,
            });
            ui.close();
        }
        if ui.button("Set to children").clicked() {
            actions.manager = Some(ManagerAction::PropagateNamedColor {
                name: name.to_owned(),
                color,
            });
            ui.close();
        }
    });
    if color_response.clicked() {
        *color_editor = Some(NamedColorEditor {
            name: name.to_owned(),
            hsva: hsva_from_color(color),
        });
    }

    let visibility_response = visibility_button(ui, style.visibility);
    visibility_response.context_menu(|ui| {
        if ui.button("Reset to default").clicked() {
            actions.manager = Some(ManagerAction::SetNamedVisibility {
                name: name.to_owned(),
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
                name: name.to_owned(),
                state: effective,
            });
            ui.close();
        }
    });
    if visibility_response.clicked() {
        actions.manager = Some(ManagerAction::SetNamedVisibility {
            name: name.to_owned(),
            state: style.visibility.next(),
        });
    }

    let label = format!("{name}  ({} · {})", selection.count(), status.label());
    let response = ui
        .selectable_label(false, label)
        .on_hover_text("Activate selection · right-click to edit expression");
    if response.clicked() {
        actions.manager = Some(ManagerAction::ActivateNamed(name.to_owned()));
    }
    named_expression_menu(
        response,
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
        actions.manager = Some(ManagerAction::RemoveNamed(name.to_owned()));
    }
}

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
    let (Some(molecule), Some(hierarchy)) = (info.molecule, info.hierarchy) else {
        return;
    };
    let rows = visible_selection_rows(ui, info.named_selections, hierarchy);
    let row_height = ui.spacing().interact_size.y;
    let max_height = (ui.available_height() * 0.34).clamp(row_height * 3.0, 260.0);
    egui::ScrollArea::vertical()
        .id_salt("virtual named selections")
        .max_height(max_height)
        .auto_shrink([false, true])
        .show_rows(ui, row_height, rows.len(), |ui, visible_range| {
            for row in &rows[visible_range] {
                ui.horizontal(|ui| match *row {
                    SelectionListRow::Selection(name) => {
                        let Some(selection) = info.named_selections.get(name) else {
                            return;
                        };
                        if selection.count() == 0 {
                            ui.add_space(ui.spacing().indent);
                        } else {
                            let id = ui.make_persistent_id(("named selection", name));
                            selection_disclosure_button(ui, id, selection_tree_open(ui, id));
                        }
                        named_selection_header(
                            ui,
                            name,
                            selection,
                            info,
                            actions,
                            color_editor,
                            expression_editor,
                            rename_editor,
                        );
                    }
                    SelectionListRow::Chain {
                        selection_name,
                        chain_index,
                        atom_count,
                    } => {
                        let (Some(selection), Some(chain)) = (
                            info.named_selections.get(selection_name),
                            hierarchy.chains.get(chain_index),
                        ) else {
                            return;
                        };
                        ui.add_space(ui.spacing().indent);
                        let id =
                            ui.make_persistent_id(("named chain", selection_name, chain_index));
                        selection_disclosure_button(ui, id, selection_tree_open(ui, id));
                        let target = InspectionTarget::Chain(chain_index);
                        let chain_name = info
                            .hierarchy_names
                            .get(&target)
                            .cloned()
                            .unwrap_or_else(|| format!("Chain {}", display_chain(&chain.id)));
                        let indices = chain
                            .residues
                            .iter()
                            .flat_map(|residue| &residue.atom_indices)
                            .copied()
                            .filter(|&atom_index| {
                                selection.flags().get(atom_index).unwrap_or(false)
                            })
                            .collect();
                        named_subset_label(
                            ui,
                            format!("{chain_name} · {atom_count} atoms"),
                            indices,
                            target,
                            info.inspection == Some(target),
                            actions,
                        );
                    }
                    SelectionListRow::Residue {
                        selection_name,
                        chain_index,
                        residue_index,
                        atom_count,
                    } => {
                        let (Some(selection), Some(residue)) = (
                            info.named_selections.get(selection_name),
                            hierarchy.residue(chain_index, residue_index),
                        ) else {
                            return;
                        };
                        ui.add_space(ui.spacing().indent * 2.0);
                        let id = ui.make_persistent_id((
                            "named residue",
                            selection_name,
                            chain_index,
                            residue_index,
                        ));
                        selection_disclosure_button(ui, id, selection_tree_open(ui, id));
                        let target = InspectionTarget::Residue {
                            chain_index,
                            residue_index,
                        };
                        let residue_name = info
                            .hierarchy_names
                            .get(&target)
                            .cloned()
                            .unwrap_or_else(|| residue.id.label());
                        let indices = residue
                            .atom_indices
                            .iter()
                            .copied()
                            .filter(|&atom_index| {
                                selection.flags().get(atom_index).unwrap_or(false)
                            })
                            .collect();
                        named_subset_label(
                            ui,
                            format!("{residue_name} · {atom_count} atoms"),
                            indices,
                            target,
                            info.inspection == Some(target),
                            actions,
                        );
                    }
                    SelectionListRow::Atom {
                        selection_name,
                        atom_index,
                    } => {
                        let Some(atom) = molecule.atoms.get(atom_index) else {
                            return;
                        };
                        ui.add_space(ui.spacing().indent * 4.0);
                        let target = InspectionTarget::Atom(atom_index);
                        let atom_name = info
                            .hierarchy_names
                            .get(&target)
                            .cloned()
                            .unwrap_or_else(|| format!("#{} {}", atom.serial, atom.name));
                        ui.push_id((selection_name, atom_index), |ui| {
                            named_subset_label(
                                ui,
                                format!("{} · {}", atom_name, atom.element.symbol()),
                                vec![atom_index],
                                target,
                                info.inspection == Some(target),
                                actions,
                            );
                        });
                    }
                });
            }
        });
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

#[cfg(test)]
mod tests {
    use astra::molecule::{Atom, Element, Molecule};
    use glam::Vec3;

    use super::*;

    fn selected_hierarchy() -> (MoleculeHierarchy, BTreeMap<String, Selection>) {
        let atoms = [1, 2]
            .into_iter()
            .enumerate()
            .map(|(index, residue_number)| Atom {
                serial: index as u32 + 1,
                name: "CA".into(),
                element: Element::C,
                residue_name: "GLY".into(),
                residue_number,
                insertion_code: None,
                chain_id: "A".into(),
                position: Vec3::new(index as f32, 0.0, 0.0),
                occupancy: 1.0,
                b_factor: 0.0,
                hetero: false,
            })
            .collect();
        let hierarchy = MoleculeHierarchy::from_molecule(&Molecule {
            atoms,
            bonds: Vec::new(),
        });
        let selections = BTreeMap::from([("all".into(), Selection::from_flags(vec![true, true]))]);
        (hierarchy, selections)
    }

    #[test]
    fn virtual_selection_tree_flattens_only_open_branches() {
        let (hierarchy, selections) = selected_hierarchy();
        egui::__run_test_ui(|ui| {
            assert_eq!(visible_selection_rows(ui, &selections, &hierarchy).len(), 1);

            for id in [
                ui.make_persistent_id(("named selection", "all")),
                ui.make_persistent_id(("named chain", "all", 0)),
                ui.make_persistent_id(("named residue", "all", 0, 0)),
            ] {
                let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    id,
                    false,
                );
                state.set_open(true);
                state.store(ui.ctx());
            }
            assert_eq!(visible_selection_rows(ui, &selections, &hierarchy).len(), 5);
        });
    }
}
