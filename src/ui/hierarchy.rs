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
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        let chain_target = InspectionTarget::Chain(chain_index);
        let chain_default_name = format!("Chain {}", display_chain(&chain.id));
        let chain_name = info
            .hierarchy_names
            .get(&chain_target)
            .unwrap_or(&chain_default_name);
        let chain_label = format!(
            "{} · {} residues · {} atoms",
            chain_name,
            chain.residues.len(),
            chain.atom_count
        );
        let first_atom = chain
            .residues
            .iter()
            .find_map(|residue| residue.atom_indices.first().copied());
        let chain_on_inspected_path = inspected_atom_path
            .is_some_and(|(inspected_chain, _, _)| inspected_chain == chain_index);
        let mut chain_state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("chain", chain_index)),
            false,
        );
        if chain_on_inspected_path {
            chain_state.set_open(true);
        }
        chain_state
            .show_header(ui, |ui| {
                hierarchy_row(
                    ui,
                    display,
                    chain_target,
                    DisplayLevel::Chain,
                    first_atom,
                    &chain_label,
                    info.hierarchy_selection.contains(&chain_target) || chain_on_inspected_path,
                    info.hierarchy_selection,
                    actions,
                    color_editor,
                    rename_editor,
                    &chain_default_name,
                    info.hierarchy_names.get(&chain_target),
                );
            })
            .body(|ui| {
                for (residue_index, residue) in chain.residues.iter().enumerate() {
                    let residue_target = InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    };
                    let residue_default_name = residue.id.label();
                    let residue_name = info
                        .hierarchy_names
                        .get(&residue_target)
                        .unwrap_or(&residue_default_name);
                    let residue_label =
                        format!("{} · {} atoms", residue_name, residue.atom_indices.len());
                    let residue_on_inspected_path = inspected_atom_path.is_some_and(
                        |(inspected_chain, inspected_residue, _)| {
                            inspected_chain == chain_index && inspected_residue == residue_index
                        },
                    );
                    let mut residue_state =
                        egui::collapsing_header::CollapsingState::load_with_default_open(
                            ui.ctx(),
                            ui.make_persistent_id(("residue", chain_index, residue_index)),
                            false,
                        );
                    if residue_on_inspected_path {
                        residue_state.set_open(true);
                    }
                    residue_state
                        .show_header(ui, |ui| {
                            hierarchy_row(
                                ui,
                                display,
                                residue_target,
                                DisplayLevel::Residue,
                                residue.atom_indices.first().copied(),
                                &residue_label,
                                info.hierarchy_selection.contains(&residue_target)
                                    || residue_on_inspected_path,
                                info.hierarchy_selection,
                                actions,
                                color_editor,
                                rename_editor,
                                &residue_default_name,
                                info.hierarchy_names.get(&residue_target),
                            );
                        })
                        .body(|ui| {
                            for &atom_index in &residue.atom_indices {
                                if let Some(atom) = molecule.atoms.get(atom_index) {
                                    let target = InspectionTarget::Atom(atom_index);
                                    let atom_default_name =
                                        format!("#{} {}", atom.serial, atom.name);
                                    let atom_name = info
                                        .hierarchy_names
                                        .get(&target)
                                        .unwrap_or(&atom_default_name);
                                    let label =
                                        format!("{} · {}", atom_name, atom.element.symbol());
                                    ui.horizontal(|ui| {
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
                                            &atom_default_name,
                                            info.hierarchy_names.get(&target),
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
