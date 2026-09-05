use super::*;

pub(super) fn inspector(ui: &mut egui::Ui, info: UiInfo<'_>) {
    let (Some(molecule), Some(hierarchy), Some(target)) =
        (info.molecule, info.hierarchy, info.inspection)
    else {
        return;
    };
    ui.heading("Inspector");
    if let Some(name) = info.hierarchy_names.get(&target) {
        ui.strong(name);
    }
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

pub(super) fn residue_inspector(ui: &mut egui::Ui, chain: &str, residue: &ResidueGroup) {
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

pub(super) fn atom_inspector(ui: &mut egui::Ui, atom_index: usize, atom: &Atom) {
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

pub(super) fn property_grid(ui: &mut egui::Ui, properties: &[(&str, String)]) {
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

#[derive(Debug, Clone, Copy)]
enum HierarchyListRow {
    Chain(usize),
    Residue {
        chain_index: usize,
        residue_index: usize,
    },
    Atom {
        atom_index: usize,
        depth: usize,
    },
}

fn hierarchy_open(ui: &egui::Ui, id: egui::Id, force_open: bool) -> bool {
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    if force_open && !state.is_open() {
        state.set_open(true);
        state.store(ui.ctx());
    }
    state.is_open()
}

fn disclosure_button(ui: &mut egui::Ui, id: egui::Id, open: bool) {
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

fn visible_hierarchy_rows(
    ui: &egui::Ui,
    hierarchy: &MoleculeHierarchy,
    inspected_atom_path: Option<(usize, usize, usize)>,
) -> Vec<HierarchyListRow> {
    let mut rows = Vec::new();
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        rows.push(HierarchyListRow::Chain(chain_index));
        let chain_on_path = inspected_atom_path
            .is_some_and(|(inspected_chain, _, _)| inspected_chain == chain_index);
        let chain_id = ui.make_persistent_id(("chain", chain_index));
        if !hierarchy_open(ui, chain_id, chain_on_path) {
            continue;
        }
        for (residue_index, residue) in chain.residues.iter().enumerate() {
            rows.push(HierarchyListRow::Residue {
                chain_index,
                residue_index,
            });
            let residue_on_path =
                inspected_atom_path.is_some_and(|(inspected_chain, inspected_residue, _)| {
                    inspected_chain == chain_index && inspected_residue == residue_index
                });
            let residue_id = ui.make_persistent_id(("residue", chain_index, residue_index));
            if hierarchy_open(ui, residue_id, residue_on_path) {
                rows.extend(residue.atom_indices.iter().copied().map(|atom_index| {
                    HierarchyListRow::Atom {
                        atom_index,
                        depth: 2,
                    }
                }));
            }
        }
    }
    rows
}

pub(super) fn hierarchy_tree(
    ui: &mut egui::Ui,
    info: UiInfo<'_>,
    actions: &mut UiActions,
    color_editor: &mut Option<ColorEditor>,
    rename_editor: &mut Option<RenameEditor>,
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
            .atom_path(atom_index)
            .map(|path| (path.chain_index, path.residue_index, atom_index))
    });
    let rows = visible_hierarchy_rows(ui, hierarchy, inspected_atom_path);
    let row_height = ui.spacing().interact_size.y;
    egui::ScrollArea::vertical()
        .id_salt("virtual molecule hierarchy")
        .auto_shrink([false, false])
        .show_rows(ui, row_height, rows.len(), |ui, visible_range| {
            for row in &rows[visible_range] {
                ui.horizontal(|ui| match *row {
                    HierarchyListRow::Chain(chain_index) => {
                        let Some(chain) = hierarchy.chains.get(chain_index) else {
                            return;
                        };
                        let target = InspectionTarget::Chain(chain_index);
                        let default_name = format!("Chain {}", display_chain(&chain.id));
                        let name = info.hierarchy_names.get(&target).unwrap_or(&default_name);
                        let label = format!(
                            "{} · {} residues · {} atoms",
                            name,
                            chain.residues.len(),
                            chain.atom_count
                        );
                        let first_atom = chain
                            .residues
                            .iter()
                            .find_map(|residue| residue.atom_indices.first().copied());
                        let on_path = inspected_atom_path
                            .is_some_and(|(inspected_chain, _, _)| inspected_chain == chain_index);
                        let id = ui.make_persistent_id(("chain", chain_index));
                        disclosure_button(ui, id, hierarchy_open(ui, id, on_path));
                        hierarchy_row(
                            ui,
                            display,
                            target,
                            DisplayLevel::Chain,
                            first_atom,
                            &label,
                            info.hierarchy_selection.contains(&target) || on_path,
                            info.hierarchy_selection,
                            actions,
                            color_editor,
                            rename_editor,
                            &default_name,
                            info.hierarchy_names.get(&target),
                        );
                    }
                    HierarchyListRow::Residue {
                        chain_index,
                        residue_index,
                    } => {
                        let Some(residue) = hierarchy.residue(chain_index, residue_index) else {
                            return;
                        };
                        ui.add_space(ui.spacing().indent);
                        let target = InspectionTarget::Residue {
                            chain_index,
                            residue_index,
                        };
                        let default_name = residue.id.label();
                        let name = info.hierarchy_names.get(&target).unwrap_or(&default_name);
                        let label = format!("{} · {} atoms", name, residue.atom_indices.len());
                        let on_path = inspected_atom_path.is_some_and(
                            |(inspected_chain, inspected_residue, _)| {
                                inspected_chain == chain_index && inspected_residue == residue_index
                            },
                        );
                        let id = ui.make_persistent_id(("residue", chain_index, residue_index));
                        disclosure_button(ui, id, hierarchy_open(ui, id, on_path));
                        hierarchy_row(
                            ui,
                            display,
                            target,
                            DisplayLevel::Residue,
                            residue.atom_indices.first().copied(),
                            &label,
                            info.hierarchy_selection.contains(&target) || on_path,
                            info.hierarchy_selection,
                            actions,
                            color_editor,
                            rename_editor,
                            &default_name,
                            info.hierarchy_names.get(&target),
                        );
                    }
                    HierarchyListRow::Atom { atom_index, depth } => {
                        let Some(atom) = molecule.atoms.get(atom_index) else {
                            return;
                        };
                        ui.add_space(ui.spacing().indent * (depth as f32 + 1.0));
                        let target = InspectionTarget::Atom(atom_index);
                        let default_name = format!("#{} {}", atom.serial, atom.name);
                        let name = info.hierarchy_names.get(&target).unwrap_or(&default_name);
                        let label = format!("{} · {}", name, atom.element.symbol());
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
                            rename_editor,
                            &default_name,
                            info.hierarchy_names.get(&target),
                        );
                    }
                });
            }
        });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn hierarchy_row(
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
    rename_editor: &mut Option<RenameEditor>,
    default_name: &str,
    custom_name: Option<&String>,
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
    let label_response = ui.selectable_label(selected, label);
    label_response.context_menu(|ui| {
        if ui.button("Rename").clicked() {
            *rename_editor = Some(RenameEditor {
                target: RenameTarget::Hierarchy(target),
                name: custom_name.map_or_else(|| default_name.to_owned(), Clone::clone),
            });
            ui.close();
        }
        if ui
            .add_enabled(custom_name.is_some(), egui::Button::new("Reset name"))
            .clicked()
        {
            actions.manager = Some(ManagerAction::RenameHierarchy { target, name: None });
            ui.close();
        }
    });
    if label_response.clicked() {
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

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use molview::molecule::{Atom, Element, Molecule};

    use super::*;

    fn two_residue_hierarchy() -> MoleculeHierarchy {
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
        MoleculeHierarchy::from_molecule(&Molecule {
            atoms,
            bonds: Vec::new(),
        })
    }

    #[test]
    fn virtual_hierarchy_flattens_only_open_branches() {
        let hierarchy = two_residue_hierarchy();
        egui::__run_test_ui(|ui| {
            assert_eq!(visible_hierarchy_rows(ui, &hierarchy, None).len(), 1);

            let chain_id = ui.make_persistent_id(("chain", 0));
            let mut chain_state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                chain_id,
                false,
            );
            chain_state.set_open(true);
            chain_state.store(ui.ctx());
            assert_eq!(visible_hierarchy_rows(ui, &hierarchy, None).len(), 3);

            let residue_id = ui.make_persistent_id(("residue", 0, 0));
            let mut residue_state =
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    residue_id,
                    false,
                );
            residue_state.set_open(true);
            residue_state.store(ui.ctx());
            assert_eq!(visible_hierarchy_rows(ui, &hierarchy, None).len(), 4);
        });
    }
}
