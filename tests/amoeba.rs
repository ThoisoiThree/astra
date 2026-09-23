use std::{collections::BTreeMap, sync::atomic::AtomicBool};

use astra::{
    DisplayState, VisibilityOverride,
    camera::OrbitCamera,
    measurement::{MeasurementEndpoint, MeasurementLine},
    molecule::{
        Molecule,
        amoeba::{self, AnalysisSettings, HydrogenBondReport},
        parse_pdb,
    },
    scene::{SceneDocument, decode, encode},
};
use glam::Vec3;

fn molecule() -> Molecule {
    parse_pdb(include_str!("fixtures/amoeba_water_dimer.pdb")).unwrap()
}

fn report() -> HydrogenBondReport {
    serde_json::from_str(include_str!("fixtures/amoeba_water_dimer.json")).unwrap()
}

#[test]
fn scored_object_round_trips_with_style_and_all_candidates() {
    let molecule = molecule();
    let mut report = report();
    report.validate(molecule.atoms.len()).unwrap();
    report.energy_threshold = -1.0;
    let endpoint = MeasurementEndpoint {
        position: Vec3::ZERO,
        description: "AMOEBA group".into(),
    };
    let mut line = MeasurementLine::new(41, endpoint.clone(), endpoint);
    line.name = "Water H-bonds".into();
    line.color = Some([0.2, 0.7, 0.9, 1.0]);
    line.visibility = VisibilityOverride::Show;
    line.set_thickness(0.12);
    line.set_label_size(24.0);
    line.hydrogen_bonds = Some(report.clone());
    assert_eq!(line.segments().count(), 1);
    assert!(line.segments().all(|(a, b)| a.distance(b) > 2.0));
    let document = SceneDocument {
        trajectory: None,
        source_name: "water".into(),
        display: DisplayState::for_molecule(&molecule),
        molecule,
        named_selections: BTreeMap::new(),
        named_selection_expressions: BTreeMap::new(),
        named_selection_styles: BTreeMap::new(),
        measurement_lines: vec![line.clone()],
        hierarchy_names: BTreeMap::new(),
        inspection: None,
        hierarchy_selection: Vec::new(),
        hierarchy_selection_anchor: None,
        focus_description: String::new(),
        pivot_description: String::new(),
        camera: OrbitCamera::new(1.0),
    };
    let restored = decode(&encode(&document).unwrap()).unwrap();
    assert_eq!(restored.measurement_lines, vec![line]);
}

#[test]
fn validation_rejects_bad_indices_overlap_and_inconsistent_decomposition() {
    let mut bad = report();
    bad.candidates[0].acceptor = 100;
    assert!(bad.validate(6).is_err());
    let mut bad = report();
    bad.candidates[0].donor_group.push(3);
    assert!(bad.validate(6).is_err());
    let mut bad = report();
    bad.candidates[0].polarization += 1.;
    assert!(bad.validate(6).is_err());
    let mut bad = report();
    bad.candidates[0].scf_residual_debye = 1.;
    assert!(bad.validate(6).is_err());
}

#[test]
fn display_filter_keeps_the_complete_energy_report() {
    let mut report = report();
    let total = report.candidates.len();
    report.energy_threshold = -1e9;
    assert_eq!(report.displayed().count(), 0);
    report.energy_threshold = 1e9;
    assert_eq!(report.displayed().count(), total);
    assert_eq!(report.candidates.len(), 4);
}

#[test]
fn cancellation_does_not_require_python() {
    assert!(matches!(
        amoeba::analyze(
            &molecule(),
            "water",
            &[0, 1, 2],
            AnalysisSettings::default(),
            &AtomicBool::new(true)
        ),
        Err(amoeba::AmoebaError::Cancelled)
    ));
}

#[test]
fn native_engine_matches_the_openmm_validated_fixture() {
    let actual = amoeba::analyze(
        &molecule(),
        "water",
        &[0, 1, 2, 3, 4, 5],
        AnalysisSettings::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let expected = report();
    assert_eq!(actual.candidates.len(), expected.candidates.len());
    for (a, b) in actual.candidates.iter().zip(&expected.candidates) {
        assert_eq!(
            (a.donor, a.hydrogen, a.acceptor),
            (b.donor, b.hydrogen, b.acceptor)
        );
        assert!((a.delta_energy - b.delta_energy).abs() < 1e-4);
        assert!((a.polarization - b.polarization).abs() < 1e-4);
    }
}

#[test]
fn preparation_preserves_sparse_style_across_hydrogen_reindexing_and_scene_save() {
    use astra::molecule::amoeba::protonation;
    use astra::{DisplayLevel, DisplayMode};
    let old = molecule();
    let p = protonation::prepare(&old, Default::default(), &AtomicBool::new(false)).unwrap();
    let mut display = DisplayState::for_molecule(&old);
    display.set_global_mode(DisplayMode::BallAndStick);
    let first = [0.1, 0.9, 0.2, 1.];
    let second = [0.8, 0.1, 0.2, 1.];
    display.set_color_override(&[0], DisplayLevel::Residue, Some(first));
    display.set_color_override(&[3], DisplayLevel::Atom, Some(second));
    display.set_visibility_override(&[3], DisplayLevel::Residue, VisibilityOverride::Hide);
    let state = display
        .edit_state()
        .remapped(&old, &p.molecule, &p.old_to_new);
    let mut restored = DisplayState::for_molecule(&p.molecule);
    restored.restore_edit_state(&p.molecule, state);
    assert_eq!(restored.global_mode, DisplayMode::BallAndStick);
    assert_eq!(restored.colors[p.old_to_new[3].unwrap()], second);
    for &(h, parent) in &p.added_parents {
        if parent == p.old_to_new[0].unwrap() {
            assert_eq!(restored.colors[h], first);
        } else {
            assert!(!restored.visible[h]);
        }
    }
    let document = SceneDocument {
        trajectory: None,
        source_name: "prepared water".into(),
        molecule: p.molecule.clone(),
        display: restored.clone(),
        named_selections: BTreeMap::new(),
        named_selection_expressions: BTreeMap::new(),
        named_selection_styles: BTreeMap::new(),
        measurement_lines: vec![],
        hierarchy_names: BTreeMap::new(),
        inspection: None,
        hierarchy_selection: vec![],
        hierarchy_selection_anchor: None,
        focus_description: String::new(),
        pivot_description: String::new(),
        camera: OrbitCamera::new(1.),
    };
    let loaded = decode(&encode(&document).unwrap()).unwrap();
    assert_eq!(loaded.molecule, p.molecule);
    assert_eq!(loaded.display, restored);
}

#[test]
fn restored_heavy_atoms_and_their_selection_survive_scene_save() {
    use astra::{DisplayLevel, molecule::amoeba::protonation, selection::Selection};
    let old = parse_pdb(include_str!("fixtures/incomplete_alanine.pdb")).unwrap();
    let p = protonation::prepare(&old, Default::default(), &AtomicBool::new(false)).unwrap();
    assert_eq!(p.restored_atoms.len(), 2);
    let mut display = DisplayState::for_molecule(&old);
    let color = [0.2, 0.7, 0.1, 1.0];
    display.set_color_override(&[1], DisplayLevel::Residue, Some(color));
    let state = display
        .edit_state()
        .remapped(&old, &p.molecule, &p.old_to_new);
    let mut display = DisplayState::for_molecule(&p.molecule);
    display.restore_edit_state(&p.molecule, state);
    let mut flags = vec![false; p.molecule.atoms.len()];
    for &(i, _) in &p.restored_atoms {
        flags[i] = true;
        assert_eq!(display.colors[i], color);
    }
    let document = SceneDocument {
        trajectory: None,
        source_name: "repaired alanine".into(),
        molecule: p.molecule,
        display,
        named_selections: BTreeMap::from([("restored_heavy".into(), Selection::from_flags(flags))]),
        named_selection_expressions: BTreeMap::new(),
        named_selection_styles: BTreeMap::new(),
        measurement_lines: vec![],
        hierarchy_names: BTreeMap::new(),
        inspection: None,
        hierarchy_selection: vec![],
        hierarchy_selection_anchor: None,
        focus_description: String::new(),
        pivot_description: String::new(),
        camera: OrbitCamera::new(1.),
    };
    let loaded = decode(&encode(&document).unwrap()).unwrap();
    assert_eq!(loaded.molecule, document.molecule);
    assert_eq!(loaded.display, document.display);
    assert_eq!(loaded.named_selections, document.named_selections);
}

#[test]
fn selected_preparation_keeps_unselected_hydrogens_and_rejects_empty_scope() {
    use astra::molecule::amoeba::protonation;
    let old = molecule();
    let cancel = AtomicBool::new(false);
    let p = protonation::prepare_selected(&old, Default::default(), &[1], &cancel).unwrap();
    assert_eq!(p.report.residues.len(), 1);
    assert_eq!(p.report.removed, 2);
    assert_eq!(p.report.added, 2);
    for i in 3..6 {
        assert_eq!(old.atoms[i], p.molecule.atoms[p.old_to_new[i].unwrap()]);
    }
    assert!(protonation::prepare_selected(&old, Default::default(), &[], &cancel).is_err());
    assert!(
        protonation::prepare_selected(&old, Default::default(), &[old.atoms.len()], &cancel)
            .is_err()
    );
}

#[test]
fn heavy_repair_is_limited_to_selected_residues() {
    use astra::molecule::amoeba::protonation;
    let mut old = parse_pdb(include_str!("fixtures/incomplete_alanine.pdb")).unwrap();
    let copy = old.atoms.clone();
    for mut atom in copy {
        atom.chain_id = "B".into();
        atom.serial += 4;
        atom.position += Vec3::new(20., 0., 0.);
        old.atoms.push(atom);
    }
    let p = protonation::prepare_selected(&old, Default::default(), &[1], &AtomicBool::new(false))
        .unwrap();
    assert_eq!(p.restored_atoms.len(), 2);
    assert_eq!(p.report.repair.as_ref().unwrap().residues.len(), 1);
    for i in 4..8 {
        assert_eq!(old.atoms[i], p.molecule.atoms[p.old_to_new[i].unwrap()]);
    }
    assert!(
        p.restored_atoms
            .iter()
            .all(|&(i, _)| p.molecule.atoms[i].chain_id == "A")
    );
}
