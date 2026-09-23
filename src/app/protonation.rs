use super::*;
use astra::molecule::amoeba::protonation::{Prepared, Settings};

impl Runtime {
    pub(super) fn start_protonation(&mut self, selection: &str, settings: Settings) -> Result<()> {
        let molecule = self
            .molecule
            .as_ref()
            .context("load a structure before preparing atoms")?;
        if self
            .background_jobs
            .values()
            .any(|j| j.kind == JobKind::Protonation && j.session_id == self.active_session_id)
        {
            anyhow::bail!("structure preparation is already running in this tab");
        }
        let selected: Vec<_> = self
            .named_selections
            .get(selection)
            .with_context(|| format!("unknown named selection: {selection}"))?
            .indices()
            .collect();
        anyhow::ensure!(
            !selected.is_empty(),
            "preparation selection is empty: {selection}"
        );
        let id = self.next_job_id;
        self.next_job_id = self.next_job_id.wrapping_add(1);
        let cancel = Arc::new(AtomicBool::new(false));
        self.submit_job(JobRequest::Protonate {
            id,
            molecule: Box::new(molecule.clone()),
            settings,
            selected,
            cancel: cancel.clone(),
        })?;
        self.background_jobs.insert(
            id,
            BackgroundJob {
                session_id: self.active_session_id,
                version: self.document_version,
                kind: JobKind::Protonation,
                stage: "Queued structure preparation".into(),
                progress: 0.,
                cancel,
            },
        );
        self.ui.latest_error = None;
        Ok(())
    }
    pub(super) fn replace_molecular_geometry(&mut self, molecule: Molecule) {
        self.cartoon_generation = self.cartoon_generation.wrapping_add(1);
        for job in self
            .background_jobs
            .values()
            .filter(|j| j.session_id == self.active_session_id && j.kind == JobKind::Cartoon)
        {
            job.cancel.store(true, Ordering::Relaxed);
        }
        self.pending_pick = None;
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        self.secondary_structure = Some(assign_secondary_structure_from(
            &molecule,
            &hierarchy,
            self.display
                .as_ref()
                .map_or_else(Default::default, |display| display.secondary_source),
        ));
        self.atom_bvh = Some(AtomBvh::build(&molecule));
        self.hierarchy = Some(hierarchy);
        self.display = Some(DisplayState::for_molecule(&molecule));
        self.molecule = Some(molecule);
    }
    pub(super) fn apply_protonation(&mut self, prepared: Prepared) {
        if prepared.report.added == 0
            && prepared.report.removed == 0
            && prepared.restored_atoms.is_empty()
        {
            self.ui.preparation_report = Some(prepared.report);
            return;
        }
        let Some(old) = self.molecule.clone() else {
            return;
        };
        let before = self.begin_edit();
        let display = before
            .display
            .clone()
            .map(|s| s.remapped(&old, &prepared.molecule, &prepared.old_to_new));
        let targets = hierarchy_remap(&old, &prepared.molecule, &prepared.old_to_new);
        for selection in self.named_selections.values_mut() {
            *selection = remap_selection(selection, &prepared);
        }
        self.hierarchy_names = self
            .hierarchy_names
            .iter()
            .filter_map(|(k, v)| targets.get(k).map(|&k| (k, v.clone())))
            .collect();
        self.inspection = self.inspection.and_then(|i| targets.get(&i).copied());
        self.hierarchy_selection = self
            .hierarchy_selection
            .iter()
            .filter_map(|i| targets.get(i).copied())
            .collect();
        self.hierarchy_selection_anchor = self
            .hierarchy_selection_anchor
            .and_then(|i| targets.get(&i).copied());
        // Previously scored energies describe a different protonation state.
        // Ordinary geometric measurements retain their fixed endpoints.
        self.measurement_lines
            .retain(|line| line.hydrogen_bonds.is_none());
        let mut generated = vec![false; prepared.molecule.atoms.len()];
        for &(h, _) in &prepared.added_parents {
            generated[h] = true;
        }
        let base = format!("prepared_h_ph_{:.2}", prepared.report.ph).replace('.', "_");
        let mut name = base.clone();
        let mut suffix = 2;
        while self
            .named_selections
            .keys()
            .any(|k| k.eq_ignore_ascii_case(&name))
        {
            name = format!("{base}_{suffix}");
            suffix += 1;
        }
        self.named_selections
            .insert(name, Selection::from_flags(generated));
        if !prepared.restored_atoms.is_empty() {
            let mut flags = vec![false; prepared.molecule.atoms.len()];
            for &(i, _) in &prepared.restored_atoms {
                flags[i] = true;
            }
            let mut name = "restored_heavy".to_owned();
            let mut suffix = 2;
            while self
                .named_selections
                .keys()
                .any(|k| k.eq_ignore_ascii_case(&name))
            {
                name = format!("restored_heavy_{suffix}");
                suffix += 1;
            }
            self.named_selections
                .insert(name, Selection::from_flags(flags));
        }
        let new = prepared.molecule.clone();
        self.replace_molecular_geometry(prepared.molecule);
        if let (Some(state), Some(molecule), Some(display)) =
            (display, &self.molecule, &mut self.display)
        {
            display.restore_edit_state(molecule, state);
        }
        self.recalculate_named_selections();
        self.ui.document_changed();
        self.ui.preparation_report = Some(prepared.report);
        let operation = EditOperation::molecular(before, self.begin_edit(), old, new);
        push_history(&mut self.undo_history, operation);
        self.redo_history.clear();
        self.mark_dirty();
        if let (Some(m), Some(d)) = (&self.molecule, &self.display) {
            self.renderer.update_instances(m, d);
        }
        self.refresh_labels();
        self.renderer.update_measurements(&self.measurement_lines);
        self.needs_cartoon_refresh = false;
        self.window.request_redraw();
        self.ui.latest_error = None;
    }
}
fn remap_selection(selection: &Selection, prepared: &Prepared) -> Selection {
    let mut flags = vec![false; prepared.molecule.atoms.len()];
    for i in selection.indices() {
        if let Some(Some(j)) = prepared.old_to_new.get(i) {
            flags[*j] = true;
        }
    }
    for &(i, parent) in &prepared.restored_atoms {
        flags[i] = flags[parent];
    }
    for &(h, parent) in &prepared.added_parents {
        flags[h] = flags[parent];
    }
    Selection::from_flags(flags)
}
fn hierarchy_remap(
    old: &Molecule,
    new: &Molecule,
    map: &[Option<usize>],
) -> BTreeMap<InspectionTarget, InspectionTarget> {
    let before = MoleculeHierarchy::from_molecule(old);
    let after = MoleculeHierarchy::from_molecule(new);
    let mut result = BTreeMap::new();
    for (i, j) in map.iter().enumerate() {
        if let Some(j) = j
            && let (Some(a), Some(b)) = (before.atom_path(i), after.atom_path(*j))
        {
            result.insert(InspectionTarget::Atom(i), InspectionTarget::Atom(*j));
            result.insert(
                InspectionTarget::Chain(a.chain_index),
                InspectionTarget::Chain(b.chain_index),
            );
            result.insert(
                InspectionTarget::Residue {
                    chain_index: a.chain_index,
                    residue_index: a.residue_index,
                },
                InspectionTarget::Residue {
                    chain_index: b.chain_index,
                    residue_index: b.residue_index,
                },
            );
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use astra::molecule::{amoeba::protonation, parse_pdb};
    fn molecule() -> Molecule {
        parse_pdb(include_str!("../../tests/fixtures/amoeba_water_dimer.pdb")).unwrap()
    }
    #[test]
    fn static_selection_and_hierarchy_targets_follow_surviving_atoms() {
        let mol = molecule();
        let p = protonation::prepare(&mol, Default::default(), &AtomicBool::new(false)).unwrap();
        let selected = remap_selection(
            &Selection::from_flags(vec![false, false, false, true, true, true]),
            &p,
        );
        assert_eq!(selected.count(), 3);
        assert!(
            selected
                .indices()
                .all(|i| p.molecule.atoms[i].residue_number == 2)
        );
        let map = hierarchy_remap(&mol, &p.molecule, &p.old_to_new);
        assert_eq!(map[&InspectionTarget::Atom(3)], InspectionTarget::Atom(1));
        assert!(!map.contains_key(&InspectionTarget::Atom(4)));
    }
    #[test]
    fn restored_atoms_follow_the_selected_residue_anchor() {
        let old = parse_pdb(include_str!("../../tests/fixtures/incomplete_alanine.pdb")).unwrap();
        let p = protonation::prepare(&old, Default::default(), &AtomicBool::new(false)).unwrap();
        assert_eq!(p.restored_atoms.len(), 2);
        let selected = remap_selection(&Selection::from_flags(vec![false, true, false, false]), &p);
        assert_eq!(selected.count(), 3);
        for &(i, _) in &p.restored_atoms {
            assert!(selected.indices().any(|j| j == i));
        }
        let unselected =
            remap_selection(&Selection::from_flags(vec![true, false, false, false]), &p);
        assert_eq!(unselected.count(), 1);
    }
    #[test]
    fn molecular_undo_retains_identical_style_states_in_both_directions() {
        for old in [
            molecule(),
            parse_pdb(include_str!("../../tests/fixtures/incomplete_alanine.pdb")).unwrap(),
        ] {
            verify_molecular_undo(old);
        }
    }
    fn verify_molecular_undo(old: Molecule) {
        let p = protonation::prepare(&old, Default::default(), &AtomicBool::new(false)).unwrap();
        let mut display = DisplayState::for_molecule(&old);
        display.set_global_mode(DisplayMode::BallAndStick);
        let snapshot = EditTransaction {
            display: Some(display.edit_state()),
            named_selections: BTreeMap::new(),
            measurements: BTreeMap::new(),
            hierarchy_names: BTreeMap::new(),
            workspace: WorkspaceSelection {
                inspection: None,
                hierarchy_selection: BTreeSet::new(),
                hierarchy_selection_anchor: None,
            },
        };
        let operation =
            EditOperation::molecular(snapshot.clone(), snapshot, old.clone(), p.molecule.clone());
        for direction in [HistoryDirection::Undo, HistoryDirection::Redo] {
            let mut mol = Molecule::default();
            let mut restored = None;
            for change in &operation.changes {
                match change {
                    EditChange::Molecule(c) => {
                        mol = history_value(c, direction).clone();
                        restored = Some(DisplayState::for_molecule(&mol));
                    }
                    EditChange::Display(c) => {
                        restored
                            .as_mut()
                            .unwrap()
                            .restore_edit_state(&mol, history_value(c, direction).clone().unwrap());
                    }
                    _ => {}
                }
            }
            assert_eq!(restored.unwrap().global_mode, DisplayMode::BallAndStick);
            assert_eq!(
                mol,
                match direction {
                    HistoryDirection::Undo => old.clone(),
                    HistoryDirection::Redo => p.molecule.clone(),
                }
            );
        }
    }
}
