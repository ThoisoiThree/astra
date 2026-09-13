use super::*;

impl Runtime {
    pub(super) fn apply_command(&mut self, command: Command) -> Result<()> {
        if self.molecule.is_none() || self.display.is_none() {
            anyhow::bail!("load a PDB file before executing commands");
        }
        let topology_changed = match command {
            Command::Select {
                name,
                source,
                selection,
            } => {
                let selection = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(name) = name {
                    validate_unique_name(&self.named_selections, &name, None)
                        .map_err(anyhow::Error::msg)?;
                    self.named_selections
                        .insert(name.clone(), selection.clone());
                    self.named_selection_expressions
                        .insert(name.clone(), source);
                    self.named_selection_styles.entry(name.clone()).or_default();
                    self.recalculate_named_selections();
                }
                if let Some(display) = &mut self.display {
                    display.set_selection(selection.flags().clone());
                }
                self.inspection = None;
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
                false
            }
            Command::Color { color, selection } => {
                let indices: Vec<_> = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?
                .indices()
                .collect();
                if let Some(display) = &mut self.display {
                    display.set_color_override(&indices, DisplayLevel::Atom, Some(color.0));
                }
                false
            }
            Command::Show {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                let selected = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(display) = &mut self.display {
                    display.set_representation(selected.indices(), mask, true);
                }
                true
            }
            Command::Hide {
                representation,
                selection,
            } => {
                let mask = representation_mask(representation);
                let selected = evaluate_with_named(
                    &selection,
                    self.molecule.as_ref().unwrap_or_else(|| unreachable!()),
                    &self.named_selections,
                )?;
                if let Some(display) = &mut self.display {
                    display.set_representation(selected.indices(), mask, false);
                }
                true
            }
        };
        if topology_changed {
            self.refresh_instances();
        } else {
            self.refresh_display_attributes();
        }
        Ok(())
    }

    pub(super) fn update_viewport(&mut self, rect: egui::Rect, pixels_per_point: f32) {
        if !rect.is_positive() {
            return;
        }
        let size = self.renderer.size();
        let x = (rect.min.x * pixels_per_point).clamp(0.0, size.width as f32);
        let y = (rect.min.y * pixels_per_point).clamp(0.0, size.height as f32);
        let width = (rect.width() * pixels_per_point)
            .max(1.0)
            .min(size.width as f32 - x);
        let height = (rect.height() * pixels_per_point)
            .max(1.0)
            .min(size.height as f32 - y);
        self.viewport = Viewport {
            x,
            y,
            width,
            height,
        };
        self.camera.set_viewport(self.viewport);
    }

    pub(super) fn request_gpu_pick(&mut self, position: PhysicalPosition<f64>) {
        let point = Vec2::new(position.x as f32, position.y as f32);
        if !self.viewport.contains(point) || self.molecule.is_none() {
            return;
        }
        let cpu_fallback = self
            .camera
            .screen_ray(point, self.viewport)
            .and_then(|ray| {
                self.molecule.as_ref().and_then(|molecule| {
                    let display = self.display.as_ref();
                    self.atom_bvh.as_ref()?.pick_with_radius(
                        molecule,
                        ray,
                        |index| {
                            display.is_none_or(|display| {
                                let Some(atom) = molecule.atoms.get(index) else {
                                    return false;
                                };
                                display.visible.get(index).unwrap_or(false)
                                    && (matches!(
                                        display.modes.get(index),
                                        Some(DisplayMode::BallAndStick | DisplayMode::Toon)
                                    ) || atom.hetero
                                        || matches!(atom.name.as_str(), "CA" | "P"))
                            })
                        },
                        |index, atom| {
                            if display.and_then(|display| display.modes.get(index))
                                == Some(&DisplayMode::Toon)
                            {
                                atom.element.van_der_waals_radius()
                            } else {
                                (atom.element.van_der_waals_radius() * 0.32).max(0.32)
                            }
                        },
                    )
                })
            });
        let request_id = self.next_pick_request_id;
        self.next_pick_request_id = self.next_pick_request_id.wrapping_add(1);
        self.renderer.request_pick(
            request_id,
            position.x.max(0.0) as u32,
            position.y.max(0.0) as u32,
        );
        self.pending_pick = Some(PendingPick {
            request_id,
            session_id: self.active_session_id,
            document_version: self.document_version,
            cpu_fallback,
        });
    }

    pub(super) fn poll_pick_result(&mut self) {
        let Some((request_id, result)) = self.renderer.poll_pick() else {
            return;
        };
        let Some(pending) = self.pending_pick.take() else {
            return;
        };
        if pending.request_id != request_id
            || pending.session_id != self.active_session_id
            || pending.document_version != self.document_version
        {
            return;
        }
        let picked = result.unwrap_or(pending.cpu_fallback);
        if picked.is_some_and(|index| {
            self.molecule
                .as_ref()
                .is_none_or(|molecule| index >= molecule.atoms.len())
        }) {
            return;
        }
        let before = self.begin_edit();
        match picked {
            Some(atom_index) => {
                let target = InspectionTarget::Atom(atom_index);
                self.hierarchy_selection.clear();
                self.hierarchy_selection.insert(target);
                self.hierarchy_selection_anchor = Some(target);
                self.select_indices(vec![atom_index], Some(target));
            }
            None => {
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
                self.select_indices(Vec::new(), None);
            }
        }
        self.commit_edit(before);
        self.window.request_redraw();
    }

    pub(super) fn handle_manager_action(&mut self, action: ManagerAction) {
        match action {
            ManagerAction::SelectHierarchy { target, gesture } => {
                self.select_hierarchy(target, gesture);
            }
            ManagerAction::SetColor { targets, color } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .collect();
                if let Some(display) = &mut self.display {
                    for (indices, level) in operations {
                        display.set_color_override(&indices, level, color);
                    }
                }
                self.refresh_display_attributes();
            }
            ManagerAction::SetVisibility { targets, state } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .map(|(indices, level)| (indices, level, state))
                    .collect();
                if let Some(display) = &mut self.display {
                    display.set_visibility_overrides(&operations);
                }
                self.refresh_instances();
            }
            ManagerAction::SetMode { targets, state } => {
                let operations: Vec<_> = targets
                    .into_iter()
                    .map(|target| self.display_target(target))
                    .map(|(indices, level)| (indices, level, state))
                    .collect();
                if let Some(display) = &mut self.display {
                    display.set_mode_overrides(&operations);
                }
                self.refresh_instances();
            }
            ManagerAction::PropagateColor(targets) => self.propagate_color(&targets),
            ManagerAction::PropagateVisibility(targets) => self.propagate_visibility(&targets),
            ManagerAction::PropagateMode(targets) => self.propagate_mode(&targets),
            ManagerAction::SetNamedColor { name, color } => {
                if self.named_selections.contains_key(&name) {
                    self.named_selection_styles.entry(name).or_default().color = color;
                    self.rebuild_named_display_attributes();
                }
            }
            ManagerAction::SetNamedVisibility { name, state } => {
                if self.named_selections.contains_key(&name) {
                    self.named_selection_styles
                        .entry(name)
                        .or_default()
                        .visibility = state;
                    self.rebuild_named_display_layers();
                }
            }
            ManagerAction::SetNamedMode { name, state } => {
                if self.named_selections.contains_key(&name) {
                    let global = self
                        .display
                        .as_ref()
                        .map_or(DisplayMode::Cartoon, |display| display.global_mode);
                    self.named_selection_styles.entry(name).or_default().mode =
                        if state.mode() == Some(global) {
                            ModeOverride::Inherit
                        } else {
                            state
                        };
                    self.rebuild_named_display_layers();
                }
            }
            ManagerAction::PropagateNamedColor { name, color } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_color_overrides_forced(&[(indices, DisplayLevel::Atom, color)]);
                }
                self.refresh_display_attributes();
            }
            ManagerAction::PropagateNamedVisibility { name, state } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_visibility_override(&indices, DisplayLevel::Atom, state);
                }
                self.refresh_instances();
            }
            ManagerAction::PropagateNamedMode { name, state } => {
                let indices = self.named_selection_indices(&name);
                if let Some(display) = &mut self.display {
                    display.set_mode_override(&indices, DisplayLevel::Atom, state);
                }
                self.refresh_instances();
            }
            ManagerAction::SelectNamedSubset {
                indices,
                inspection,
            } => {
                self.hierarchy_selection.clear();
                self.hierarchy_selection_anchor = None;
                self.select_indices(indices, Some(inspection));
            }
            ManagerAction::SetColoringMode(mode) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_coloring_mode(molecule, mode);
                }
                self.refresh_display_attributes();
            }
            ManagerAction::SetGlobalMode(mode) => {
                if let Some(display) = &mut self.display {
                    display.set_global_mode(mode);
                }
                let redundant = ModeOverride::from_mode(mode);
                for style in self.named_selection_styles.values_mut() {
                    if style.mode == redundant {
                        style.mode = ModeOverride::Inherit;
                    }
                }
                self.rebuild_named_display_layers();
            }
            ManagerAction::SetAmbientOcclusion(settings) => {
                let quality_changed = self
                    .display
                    .as_ref()
                    .is_some_and(|display| display.ambient_occlusion.quality != settings.quality);
                if let Some(display) = &mut self.display {
                    display.set_ambient_occlusion(settings);
                }
                if quality_changed {
                    self.refresh_instances();
                }
            }
            ManagerAction::SetUniformColor(color) => {
                if let (Some(molecule), Some(display)) = (&self.molecule, &mut self.display) {
                    display.set_uniform_color(molecule, color);
                }
                self.refresh_display_attributes();
            }
            ManagerAction::ActivateNamed(name) => {
                if let Some(selection) = self.named_selections.get(&name) {
                    self.hierarchy_selection.clear();
                    self.hierarchy_selection_anchor = None;
                    let flags = selection.flags().clone();
                    if let Some(display) = &mut self.display {
                        display.set_selection(flags);
                    }
                    self.inspection = None;
                    self.refresh_display_attributes();
                }
            }
            ManagerAction::UpdateNamedExpression { name, expression } => {
                let result = (|| -> Result<()> {
                    self.molecule
                        .as_ref()
                        .context("load a PDB file before editing a selection")?;
                    if !self.named_selections.contains_key(&name) {
                        anyhow::bail!("named selection '{name}' does not exist");
                    }
                    self.named_selection_expressions
                        .insert(name.clone(), expression);
                    self.named_selection_styles.entry(name).or_default();
                    self.recalculate_named_selections();
                    self.hierarchy_selection.clear();
                    self.hierarchy_selection_anchor = None;
                    self.inspection = None;
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RemoveNamed(name) => {
                self.named_selections.remove(&name);
                self.named_selection_expressions.remove(&name);
                self.named_selection_styles.remove(&name);
                self.named_selection_statuses.remove(&name);
                self.recalculate_named_selections();
            }
            ManagerAction::PrepareHydrogens {
                selection,
                settings,
            } => {
                if let Err(error) = self.start_protonation(&selection, settings) {
                    self.ui.latest_error = Some(error.to_string());
                }
            }
            ManagerAction::CreateHydrogenBonds {
                selection,
                settings,
            } => {
                if let Err(error) = self.start_hydrogen_bonds(&selection, settings) {
                    self.ui.latest_error = Some(format!("{error:#}"));
                }
            }
            ManagerAction::SetHydrogenBondThreshold { id, threshold } => {
                if threshold.is_finite()
                    && let Some(report) = self
                        .measurement_lines
                        .iter_mut()
                        .find(|line| line.id == id)
                        .and_then(|line| line.hydrogen_bonds.as_mut())
                {
                    report.energy_threshold = threshold;
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::CreateMeasurement {
                first_selection,
                second_selection,
            } => match self.create_measurement(&first_selection, &second_selection) {
                Ok(()) => self.ui.latest_error = None,
                Err(error) => self.ui.latest_error = Some(error.to_string()),
            },
            ManagerAction::SetMeasurementColor { id, color } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.color = color;
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::SetMeasurementVisibility { id, state } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.visibility = state;
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::SetMeasurementLabelSize { id, size } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.set_label_size(size);
                }
            }
            ManagerAction::SetMeasurementThickness { id, thickness } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.set_thickness(thickness);
                    self.renderer.update_measurements(&self.measurement_lines);
                }
            }
            ManagerAction::RemoveMeasurement(id) => {
                self.measurement_lines.retain(|line| line.id != id);
                self.renderer.update_measurements(&self.measurement_lines);
            }
            ManagerAction::SaveCurrentSelection(name) => {
                let result = (|| -> Result<()> {
                    validate_unique_name(&self.named_selections, &name, None)
                        .map_err(anyhow::Error::msg)?;
                    let flags = self
                        .display
                        .as_ref()
                        .context("load a structure before saving a selection")?
                        .selection
                        .clone();
                    if !flags.iter().any(|selected| selected) {
                        anyhow::bail!("select at least one atom first");
                    }
                    self.named_selections
                        .insert(name.clone(), Selection::from_mask(flags));
                    self.named_selection_styles.entry(name).or_default();
                    self.recalculate_named_selections();
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RenameHierarchy { target, name } => {
                if let Some(name) = name {
                    self.hierarchy_names.insert(target, name);
                } else {
                    self.hierarchy_names.remove(&target);
                }
            }
            ManagerAction::RenameNamedSelection { old_name, new_name } => {
                let result = (|| -> Result<()> {
                    if old_name == new_name {
                        return Ok(());
                    }
                    validate_unique_name(&self.named_selections, &new_name, Some(&old_name))
                        .map_err(anyhow::Error::msg)?;
                    let selection = self
                        .named_selections
                        .remove(&old_name)
                        .with_context(|| format!("named selection '{old_name}' does not exist"))?;
                    self.named_selections.insert(new_name.clone(), selection);
                    if let Some(expression) = self.named_selection_expressions.remove(&old_name) {
                        self.named_selection_expressions
                            .insert(new_name.clone(), expression);
                    }
                    if let Some(style) = self.named_selection_styles.remove(&old_name) {
                        self.named_selection_styles.insert(new_name.clone(), style);
                    }
                    for expression in self.named_selection_expressions.values_mut() {
                        if let Ok(renamed) =
                            rename_named_reference(expression, &old_name, &new_name)
                        {
                            *expression = renamed;
                        }
                    }
                    self.named_selection_statuses.remove(&old_name);
                    self.recalculate_named_selections();
                    Ok(())
                })();
                match result {
                    Ok(()) => self.ui.latest_error = None,
                    Err(error) => self.ui.latest_error = Some(error.to_string()),
                }
            }
            ManagerAction::RenameMeasurement { id, name } => {
                if let Some(line) = self.measurement_lines.iter_mut().find(|line| line.id == id) {
                    line.name = name;
                }
            }
        }
    }

    pub(super) fn start_hydrogen_bonds(
        &mut self,
        selection: &str,
        settings: astra::molecule::amoeba::AnalysisSettings,
    ) -> Result<()> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a structure before AMOEBA analysis")?;
        let selected = self
            .named_selections
            .get(selection)
            .with_context(|| format!("named selection '{selection}' does not exist"))?
            .indices()
            .collect::<Vec<_>>();
        if selected.is_empty() {
            anyhow::bail!("named selection '{selection}' is empty");
        }
        if self
            .background_jobs
            .values()
            .any(|j| j.kind == JobKind::HydrogenBonds && j.session_id == self.active_session_id)
        {
            anyhow::bail!("AMOEBA analysis is already running in this tab");
        }
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.submit_job(JobRequest::HydrogenBonds {
            id,
            molecule: Box::new(molecule.clone()),
            selection: selection.to_owned(),
            settings,
            selected,
            cancel: cancel.clone(),
        })?;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: self.document_version,
                kind: JobKind::HydrogenBonds,
                stage: "Queued AMOEBA analysis".into(),
                progress: 0.0,
                cancel,
            },
        );
        self.ui.latest_error = None;
        Ok(())
    }

    pub(super) fn create_measurement(&mut self, first_name: &str, second_name: &str) -> Result<()> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a structure before creating a distance line")?;
        let first = measurement_endpoint(
            molecule,
            first_name,
            self.named_selections
                .get(first_name)
                .with_context(|| format!("named selection '{first_name}' does not exist"))?,
        )?;
        let second = measurement_endpoint(
            molecule,
            second_name,
            self.named_selections
                .get(second_name)
                .with_context(|| format!("named selection '{second_name}' does not exist"))?,
        )?;
        if first.position.distance(second.position) <= f32::EPSILON {
            anyhow::bail!("the two measurement endpoints occupy the same point");
        }
        self.measurement_lines.push(MeasurementLine::new(
            self.next_measurement_id,
            first,
            second,
        ));
        self.next_measurement_id += 1;
        self.renderer.update_measurements(&self.measurement_lines);
        Ok(())
    }

    pub(super) fn select_hierarchy(
        &mut self,
        target: InspectionTarget,
        gesture: HierarchySelectionGesture,
    ) {
        let range = self
            .hierarchy_selection_anchor
            .and_then(|anchor| self.hierarchy_range(anchor, target));
        let inspection = update_hierarchy_selection(
            &mut self.hierarchy_selection,
            &mut self.hierarchy_selection_anchor,
            target,
            gesture,
            range.as_deref(),
        );
        let targets: Vec<_> = self.hierarchy_selection.iter().copied().collect();
        let indices = targets
            .into_iter()
            .flat_map(|target| self.display_target(target).0)
            .collect();
        self.select_indices(indices, inspection);
    }

    pub(super) fn named_selection_indices(&self, name: &str) -> Vec<usize> {
        self.named_selections
            .get(name)
            .map(|selection| selection.indices().collect())
            .unwrap_or_default()
    }

    pub(super) fn rebuild_named_display_layers(&mut self) {
        let layers = named_display_layers(&self.named_selections, &self.named_selection_styles);
        if let Some(display) = &mut self.display {
            display.replace_named_layers(&layers);
        }
        self.refresh_instances();
    }

    pub(super) fn recalculate_named_selections(&mut self) {
        let Some(molecule) = &self.molecule else {
            self.named_selection_statuses.clear();
            return;
        };
        let resolution = resolve_named_expressions(
            molecule,
            &self.named_selection_expressions,
            &self.named_selections,
        );
        self.named_selections = resolution.selections;
        self.named_selection_statuses = resolution.statuses;
        self.rebuild_named_display_layers();
    }

    pub(super) fn hierarchy_range(
        &self,
        anchor: InspectionTarget,
        target: InspectionTarget,
    ) -> Option<Vec<InspectionTarget>> {
        if std::mem::discriminant(&anchor) != std::mem::discriminant(&target) {
            return None;
        }
        let hierarchy = self.hierarchy.as_ref()?;
        let ordered: Vec<InspectionTarget> = match target {
            InspectionTarget::Chain(_) => (0..hierarchy.chains.len())
                .map(InspectionTarget::Chain)
                .collect(),
            InspectionTarget::Residue { .. } => hierarchy
                .chains
                .iter()
                .enumerate()
                .flat_map(|(chain_index, chain)| {
                    (0..chain.residues.len()).map(move |residue_index| InspectionTarget::Residue {
                        chain_index,
                        residue_index,
                    })
                })
                .collect(),
            InspectionTarget::Atom(_) => hierarchy
                .chains
                .iter()
                .flat_map(|chain| &chain.residues)
                .flat_map(|residue| residue.atom_indices.iter().copied())
                .map(InspectionTarget::Atom)
                .collect(),
        };
        inclusive_target_range(&ordered, anchor, target)
    }

    pub(super) fn propagate_color(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, DisplayColor)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(color) = self
                .display
                .as_ref()
                .map(|display| display.color_at_level(source_atom, source_level))
            else {
                continue;
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, color));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_color_overrides_forced(&operations);
        }
        self.refresh_display_attributes();
    }

    pub(super) fn rebuild_named_display_attributes(&mut self) {
        let layers = named_display_layers(&self.named_selections, &self.named_selection_styles);
        if let Some(display) = &mut self.display {
            display.replace_named_layers(&layers);
        }
        self.refresh_display_attributes();
    }

    pub(super) fn propagate_visibility(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, VisibilityOverride)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(display) = &self.display else {
                continue;
            };
            let direct = display.visibility_override(source_atom, source_level);
            let state = match direct {
                VisibilityOverride::Inherit => {
                    if display.visible.get(source_atom).unwrap_or(true) {
                        VisibilityOverride::Show
                    } else {
                        VisibilityOverride::Hide
                    }
                }
                state => state,
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, state));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_visibility_overrides(&operations);
        }
        self.refresh_instances();
    }

    pub(super) fn propagate_mode(&mut self, targets: &[InspectionTarget]) {
        let mut operations = Vec::<(Vec<usize>, DisplayLevel, ModeOverride)>::new();
        for &target in targets {
            let (source_indices, source_level) = self.display_target(target);
            let Some(source_atom) = source_indices.first().copied() else {
                continue;
            };
            let Some(mode) = self
                .display
                .as_ref()
                .map(|display| display.mode_at_level(source_atom, source_level))
            else {
                continue;
            };
            for child in self.descendant_targets(target) {
                let (indices, level) = self.display_target(child);
                operations.push((indices, level, ModeOverride::from_mode(mode)));
            }
        }
        if let Some(display) = &mut self.display {
            display.set_mode_overrides(&operations);
        }
        self.refresh_instances();
    }

    pub(super) fn descendant_targets(&self, target: InspectionTarget) -> Vec<InspectionTarget> {
        match target {
            InspectionTarget::Chain(chain_index) => self
                .hierarchy
                .as_ref()
                .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                .map(|chain| {
                    chain
                        .residues
                        .iter()
                        .enumerate()
                        .flat_map(|(residue_index, residue)| {
                            std::iter::once(InspectionTarget::Residue {
                                chain_index,
                                residue_index,
                            })
                            .chain(
                                residue
                                    .atom_indices
                                    .iter()
                                    .copied()
                                    .map(InspectionTarget::Atom),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => self
                .hierarchy
                .as_ref()
                .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                .map(|residue| {
                    residue
                        .atom_indices
                        .iter()
                        .copied()
                        .map(InspectionTarget::Atom)
                        .collect()
                })
                .unwrap_or_default(),
            InspectionTarget::Atom(_) => Vec::new(),
        }
    }

    pub(super) fn select_indices(
        &mut self,
        indices: Vec<usize>,
        inspection: Option<InspectionTarget>,
    ) {
        let atom_count = self
            .molecule
            .as_ref()
            .map_or(0, |molecule| molecule.atoms.len());
        let mut flags = vec![false; atom_count];
        for index in indices {
            if let Some(selected) = flags.get_mut(index) {
                *selected = true;
            }
        }
        self.set_selection(flags, inspection);
    }

    pub(super) fn set_selection(&mut self, flags: Vec<bool>, inspection: Option<InspectionTarget>) {
        let Some(display) = &mut self.display else {
            return;
        };
        display.set_selection_from_bools(flags);
        self.inspection = inspection;
        self.refresh_display_attributes();
    }

    pub(super) fn display_target(&self, target: InspectionTarget) -> (Vec<usize>, DisplayLevel) {
        match target {
            InspectionTarget::Chain(chain_index) => (
                self.hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.chains.get(chain_index))
                    .map(|chain| {
                        chain
                            .residues
                            .iter()
                            .flat_map(|residue| residue.atom_indices.iter().copied())
                            .collect()
                    })
                    .unwrap_or_default(),
                DisplayLevel::Chain,
            ),
            InspectionTarget::Residue {
                chain_index,
                residue_index,
            } => (
                self.hierarchy
                    .as_ref()
                    .and_then(|hierarchy| hierarchy.residue(chain_index, residue_index))
                    .map(|residue| residue.atom_indices.clone())
                    .unwrap_or_default(),
                DisplayLevel::Residue,
            ),
            InspectionTarget::Atom(index) => (vec![index], DisplayLevel::Atom),
        }
    }

    pub(super) fn refresh_instances(&mut self) {
        self.needs_cartoon_refresh = true;
        self.renderer.update_measurements(&self.measurement_lines);
        self.window.request_redraw();
    }

    pub(super) fn refresh_display_attributes(&mut self) {
        if let (Some(molecule), Some(display)) = (&self.molecule, &self.display) {
            self.renderer.update_display_attributes(molecule, display);
        }
        self.renderer.update_measurements(&self.measurement_lines);
        self.window.request_redraw();
    }
}

pub(super) fn molecule_path(mut path: PathBuf) -> PathBuf {
    let has_extension = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case(SCENE_EXTENSION));
    if !has_extension {
        path.set_extension(SCENE_EXTENSION);
    }
    path
}

pub(super) fn scene_target(target: InspectionTarget) -> SceneHierarchyTarget {
    match target {
        InspectionTarget::Chain(index) => SceneHierarchyTarget::Chain(index),
        InspectionTarget::Residue {
            chain_index,
            residue_index,
        } => SceneHierarchyTarget::Residue {
            chain_index,
            residue_index,
        },
        InspectionTarget::Atom(index) => SceneHierarchyTarget::Atom(index),
    }
}

pub(super) fn inspection_target(target: SceneHierarchyTarget) -> InspectionTarget {
    match target {
        SceneHierarchyTarget::Chain(index) => InspectionTarget::Chain(index),
        SceneHierarchyTarget::Residue {
            chain_index,
            residue_index,
        } => InspectionTarget::Residue {
            chain_index,
            residue_index,
        },
        SceneHierarchyTarget::Atom(index) => InspectionTarget::Atom(index),
    }
}

pub(super) fn parse_residue_number(value: &str) -> Result<i32> {
    value
        .trim()
        .parse::<i32>()
        .with_context(|| format!("'{}' is not a valid residue number", value.trim()))
}

pub(super) fn matching_residue_indices(
    molecule: &Molecule,
    chain: &str,
    number: i32,
    residue_name: Option<&str>,
) -> Vec<usize> {
    molecule
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(index, atom)| {
            (atom.chain_id.eq_ignore_ascii_case(chain)
                && atom.residue_number == number
                && residue_name.is_none_or(|name| atom.residue_name.eq_ignore_ascii_case(name)))
            .then_some(index)
        })
        .collect()
}

pub(super) fn centroid(molecule: &Molecule, indices: &[usize]) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0_u32;
    for &index in indices {
        if let Some(atom) = molecule.atoms.get(index) {
            sum += atom.position;
            count += 1;
        }
    }
    (count > 0).then(|| sum / count as f32)
}

pub(super) fn measurement_endpoint(
    molecule: &Molecule,
    selection_name: &str,
    selection: &Selection,
) -> Result<MeasurementEndpoint> {
    let indices: Vec<_> = selection.indices().collect();
    let first_index = *indices
        .first()
        .with_context(|| format!("named selection '{selection_name}' is empty"))?;
    let first = molecule
        .atoms
        .get(first_index)
        .with_context(|| format!("named selection '{selection_name}' refers to a missing atom"))?;
    if indices.len() == 1 {
        return Ok(MeasurementEndpoint {
            position: first.position,
            description: format!("{selection_name}: atom #{} {}", first.serial, first.name),
        });
    }

    let one_residue = indices.iter().all(|index| {
        molecule.atoms.get(*index).is_some_and(|atom| {
            atom.chain_id == first.chain_id
                && atom.residue_name == first.residue_name
                && atom.residue_number == first.residue_number
                && atom.insertion_code == first.insertion_code
        })
    });
    if !one_residue {
        anyhow::bail!(
            "named selection '{selection_name}' must contain one atom or atoms from exactly one residue/base"
        );
    }
    Ok(MeasurementEndpoint {
        position: centroid(molecule, &indices).with_context(|| {
            format!("named selection '{selection_name}' contains no valid atoms")
        })?,
        description: format!(
            "{selection_name}: {} {} / chain {}",
            first.residue_name, first.residue_number, first.chain_id
        ),
    })
}

pub(super) fn representation_mask(representation: Representation) -> RepresentationMask {
    match representation {
        Representation::Spheres => RepresentationMask::SPHERES,
        Representation::Sticks => RepresentationMask::STICKS,
    }
}

pub(super) fn named_display_layers(
    selections: &BTreeMap<String, Selection>,
    styles: &BTreeMap<String, NamedSelectionStyle>,
) -> Vec<(Vec<usize>, NamedSelectionStyle)> {
    selections
        .iter()
        .map(|(name, selection)| {
            (
                selection.indices().collect(),
                styles.get(name).copied().unwrap_or_default(),
            )
        })
        .collect()
}

pub(super) fn update_hierarchy_selection(
    selection: &mut BTreeSet<InspectionTarget>,
    anchor: &mut Option<InspectionTarget>,
    target: InspectionTarget,
    gesture: HierarchySelectionGesture,
    range: Option<&[InspectionTarget]>,
) -> Option<InspectionTarget> {
    match gesture {
        HierarchySelectionGesture::Replace => {
            selection.clear();
            selection.insert(target);
            *anchor = Some(target);
        }
        HierarchySelectionGesture::Range => {
            selection.clear();
            if let Some(range) = range {
                selection.extend(range.iter().copied());
            } else {
                selection.insert(target);
                *anchor = Some(target);
            }
        }
        HierarchySelectionGesture::Toggle => {
            if !selection.insert(target) {
                selection.remove(&target);
            }
            if anchor.is_none() {
                *anchor = Some(target);
            }
        }
    }
    if selection.contains(&target) {
        Some(target)
    } else {
        selection.iter().next_back().copied()
    }
}

pub(super) fn inclusive_target_range(
    ordered: &[InspectionTarget],
    anchor: InspectionTarget,
    target: InspectionTarget,
) -> Option<Vec<InspectionTarget>> {
    let anchor_index = ordered.iter().position(|candidate| *candidate == anchor)?;
    let target_index = ordered.iter().position(|candidate| *candidate == target)?;
    let start = anchor_index.min(target_index);
    let end = anchor_index.max(target_index);
    Some(ordered[start..=end].to_vec())
}
