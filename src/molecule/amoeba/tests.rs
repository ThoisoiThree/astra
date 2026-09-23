use super::*;
use crate::molecule::{Atom, Bond, Element};
use glam::{DMat3, DVec3, Vec3};
use serde::Deserialize;
#[derive(Deserialize)]
struct InputAtom {
    name: String,
    element: String,
    chain: String,
    residue_number: i32,
    residue_name: String,
    position: [f32; 3],
}
#[derive(Deserialize)]
struct Oracle {
    name: String,
    atoms: Vec<InputAtom>,
    bonds: Vec<[usize; 2]>,
    permanent: f64,
    vdw: f64,
    polarization: f64,
    q: Vec<f64>,
    mu: Vec<[f64; 3]>,
    quad: Vec<[[f64; 3]; 3]>,
    alpha: Vec<f64>,
    #[serde(default)]
    decoupling: Vec<Decoupling>,
}
#[derive(Deserialize)]
struct Decoupling {
    groups: [Vec<usize>; 2],
    permanent: f64,
    vdw: f64,
    polarization: f64,
    delta_energy: f64,
}
impl Oracle {
    fn molecule(&self) -> Molecule {
        Molecule {
            atoms: self
                .atoms
                .iter()
                .enumerate()
                .map(|(i, a)| Atom {
                    serial: i as u32 + 1,
                    name: a.name.clone(),
                    element: a.element.parse::<Element>().unwrap(),
                    residue_name: a.residue_name.clone(),
                    residue_number: a.residue_number,
                    insertion_code: None,
                    chain_id: a.chain.clone(),
                    position: Vec3::from_array(a.position),
                    occupancy: 1.,
                    b_factor: 0.,
                    hetero: false,
                    alt_loc: None,
                    formal_charge: 0,
                })
                .collect(),
            bonds: self
                .bonds
                .iter()
                .map(|&[a, b]| Bond {
                    a,
                    b,
                    order: Default::default(),
                    kind: Default::default(),
                })
                .collect(),
            info: Default::default(),
        }
    }
}
fn oracles() -> Vec<Oracle> {
    serde_json::from_str(include_str!("../../../tests/fixtures/amoeba_oracles.json")).unwrap()
}
fn close(name: &str, term: &str, actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "{name} {term}: actual {actual}, expected {expected}, difference {}",
        actual - expected
    );
}
#[test]
fn all_134_templates_and_full_nonbonded_energies_match_openmm() {
    let ff = parameters::ForceField::builtin().unwrap();
    let cancel = AtomicBool::new(false);
    let cases = oracles();
    assert_eq!(cases.len(), 137);
    for o in cases {
        let mol = o.molecule();
        let (t, status) = topology::prepare(&mol, &ff, &cancel).unwrap();
        assert_eq!(
            t.original.len(),
            mol.atoms.len(),
            "{}: {:?}",
            o.name,
            status
        );
        let ev = energy::Evaluator::new(
            &t,
            &ff,
            AnalysisSettings {
                tolerance: 1e-9,
                ..Default::default()
            },
            &cancel,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", o.name));
        for (i, p) in ev.particles.iter().enumerate() {
            close(&o.name, "charge", p.q, o.q[i], 1e-10);
            close(&o.name, "alpha", p.alpha, o.alpha[i], 1e-12);
            assert!(
                (p.mu - DVec3::from_array(o.mu[i])).length() < 1e-10,
                "{} dipole {i}",
                o.name
            );
            let q = DMat3::from_cols_array_2d(&o.quad[i]);
            assert!(
                (p.quad - q).to_cols_array().iter().all(|v| v.abs() < 1e-10),
                "{} quadrupole {i}",
                o.name
            );
        }
        let (perm, vdw) = ev.pairwise(None).unwrap();
        let pol = ev
            .polarization(None)
            .unwrap_or_else(|e| panic!("{}: {e}", o.name));
        close(&o.name, "permanent", perm, o.permanent, 1e-7);
        close(&o.name, "vdw", vdw, o.vdw, 1e-5);
        close(&o.name, "polarization", pol.energy, o.polarization, 2e-5);
        for dec in &o.decoupling {
            let key = (t.groups[dec.groups[0][0]], t.groups[dec.groups[1][0]]);
            let (p, v) = ev.pairwise(Some(key)).unwrap();
            let response = pol.energy - ev.polarization(Some(key)).unwrap().energy;
            close(&o.name, "cross permanent", p, dec.permanent, 1e-6);
            close(&o.name, "cross vdw", v, dec.vdw, 1e-6);
            close(
                &o.name,
                "environment response",
                response,
                dec.polarization,
                1e-6,
            );
            close(&o.name, "delta", p + v + response, dec.delta_energy, 1e-6);
        }
    }
}
#[test]
fn interleaving_preserves_indices_groups_and_environment_score() {
    let mol = oracles().remove(0).molecule();
    let settings = AnalysisSettings {
        tolerance: 1e-9,
        ..Default::default()
    };
    let cancel = AtomicBool::new(false);
    let original = analyze(
        &mol,
        "water",
        &(0..9).collect::<Vec<_>>(),
        settings,
        &cancel,
    )
    .unwrap();
    let order = [0, 3, 6, 1, 4, 7, 2, 5, 8];
    let mut inverse = [0; 9];
    for (new, &old) in order.iter().enumerate() {
        inverse[old] = new;
    }
    let reordered = Molecule {
        atoms: order.iter().map(|&i| mol.atoms[i].clone()).collect(),
        bonds: mol
            .bonds
            .iter()
            .map(|b| Bond {
                a: inverse[b.a],
                b: inverse[b.b],
                order: Default::default(),
                kind: Default::default(),
            })
            .collect(),
        info: Default::default(),
    };
    let actual = analyze(
        &reordered,
        "water",
        &(0..9).collect::<Vec<_>>(),
        settings,
        &cancel,
    )
    .unwrap();
    assert_eq!(actual.candidates.len(), original.candidates.len());
    for a in actual.candidates {
        let b = original
            .candidates
            .iter()
            .find(|b| {
                (b.donor, b.hydrogen, b.acceptor)
                    == (order[a.donor], order[a.hydrogen], order[a.acceptor])
            })
            .unwrap();
        close("interleaved", "delta", a.delta_energy, b.delta_energy, 1e-6);
        let mut group: Vec<_> = a.donor_group.iter().map(|&i| order[i]).collect();
        group.sort();
        assert_eq!(group, b.donor_group);
    }
    let selected = analyze(
        &mol,
        "first two",
        &(0..6).collect::<Vec<_>>(),
        settings,
        &cancel,
    )
    .unwrap();
    assert_eq!(selected.candidates.len(), 4);
    close(
        "selection",
        "full environment",
        selected.full_polarization,
        original.full_polarization,
        1e-8,
    );
    let mut dimer = mol.clone();
    dimer.atoms.truncate(6);
    dimer.bonds.retain(|b| b.a < 6 && b.b < 6);
    let isolated = analyze(
        &dimer,
        "two",
        &(0..6).collect::<Vec<_>>(),
        settings,
        &cancel,
    )
    .unwrap();
    assert!(
        (isolated.candidates[0].polarization - selected.candidates[0].polarization).abs() > 0.01
    );
}
#[test]
fn missing_atoms_and_unknown_residues_never_fall_back_to_geometry() {
    let ff = parameters::ForceField::builtin().unwrap();
    let cancel = AtomicBool::new(false);
    let mut mol = oracles().remove(0).molecule();
    mol.atoms.truncate(1);
    mol.bonds.clear();
    let r = analyze(&mol, "oxygen", &[0], Default::default(), &cancel).unwrap();
    assert!(r.candidates.is_empty());
    assert_eq!(r.residues[0].status, "unparameterized");
    mol.atoms[0].residue_name = "LIG".into();
    let (_, r) = topology::prepare(&mol, &ff, &cancel).unwrap();
    assert!(r[0].reason.contains("import parameters"));
    let mut mol = oracles()
        .into_iter()
        .find(|o| o.name == "ALA")
        .unwrap()
        .molecule();
    for a in &mut mol.atoms {
        if a.residue_name == "ALA" {
            a.residue_name = "LYS".into();
        }
    }
    let (t, r) = topology::prepare(&mol, &ff, &cancel).unwrap();
    assert!(t.original.is_empty());
    assert!(
        r.iter()
            .any(|r| r.reason.contains("CG, CD, CE") || r.reason.contains("Missing heavy atoms"))
    );
}
#[test]
fn nonconvergence_fails_and_angles_do_not_filter_candidates() {
    let mol = oracles().remove(0).molecule();
    let cancel = AtomicBool::new(false);
    assert!(
        analyze(
            &mol,
            "all",
            &(0..9).collect::<Vec<_>>(),
            AnalysisSettings {
                max_iterations: 1,
                tolerance: 1e-10,
                ..Default::default()
            },
            &cancel
        )
        .is_err()
    );
    let r = analyze(
        &mol,
        "all",
        &(0..9).collect::<Vec<_>>(),
        Default::default(),
        &cancel,
    )
    .unwrap();
    assert!(r.candidates.iter().any(|c| c.dha_angle < 90.));
}

#[test]
fn ion_coordination_is_not_covalent_and_unknown_boundaries_are_excluded() {
    let ff = parameters::ForceField::builtin().unwrap();
    let cancel = AtomicBool::new(false);
    let mut mol = oracles()
        .into_iter()
        .find(|o| o.name == "water_ions")
        .unwrap()
        .molecule();
    let chloride = mol
        .atoms
        .iter()
        .position(|a| a.element == Element::Cl)
        .unwrap();
    mol.bonds.push(Bond {
        a: 0,
        b: chloride,
        order: Default::default(),
        kind: Default::default(),
    });
    let r = analyze(
        &mol,
        "water chloride",
        &[0, 1, 2, chloride],
        Default::default(),
        &cancel,
    )
    .unwrap();
    assert_eq!(r.candidates.len(), 2);
    assert!(r.candidates.iter().all(|c| c.acceptor == chloride));
    assert!(r.residues.iter().all(|r| r.status == "parameterized"));
    mol.atoms[chloride].element = Element::C;
    mol.atoms[chloride].residue_name = "LIG".into();
    mol.atoms[chloride].position = Vec3::new(1.5, 0., 0.);
    let (top, status) = topology::prepare(&mol, &ff, &cancel).unwrap();
    assert!(!top.original.contains(&0));
    assert!(!top.original.contains(&chloride));
    assert!(
        status
            .iter()
            .filter(|r| r.atoms.contains(&0) || r.atoms.contains(&chloride))
            .all(|r| r.status == "unparameterized")
    );
}

#[test]
fn parameter_provider_rejects_invalid_references_and_handles_unicode_names() {
    let json = include_str!("../../../data/amoeba/amoeba2018.json");
    let mut data: serde_json::Value = serde_json::from_str(json).unwrap();
    data["residues"][0]["bonds"][0][0] = serde_json::json!(usize::MAX);
    assert!(parameters::ForceField::from_json(&data.to_string()).is_err());
    let ff = parameters::ForceField::builtin().unwrap();
    assert_eq!(ff.canonical("RЖ"), "RЖ");
}

#[test]
fn default_settings_converge_for_a_protein_group_in_its_full_environment() {
    let mol = oracles()
        .into_iter()
        .find(|o| o.name == "villin")
        .unwrap()
        .molecule();
    let cancel = AtomicBool::new(false);
    let ff = parameters::ForceField::builtin().unwrap();
    let (t, _) = topology::prepare(&mol, &ff, &cancel).unwrap();
    let mut selected = None;
    'donor: for d in 0..t.original.len() {
        if t.elements[d] != "N" {
            continue;
        }
        for &h in &t.bonds[d] {
            if t.elements[h] != "H" {
                continue;
            }
            for a in 0..t.original.len() {
                if t.elements[a] == "O"
                    && t.groups[a] != t.groups[d]
                    && !t.bonds[d].contains(&a)
                    && t.xyz[d].distance(t.xyz[a]) < 0.6
                {
                    selected = Some([t.original[d], t.original[h], t.original[a]]);
                    break 'donor;
                }
            }
        }
    }
    let result = analyze(
        &mol,
        "protein pair",
        &selected.unwrap(),
        Default::default(),
        &cancel,
    )
    .unwrap();
    assert_eq!(result.candidates.len(), 1);
    assert!(result.scf_iterations > 0);
    assert!(result.candidates[0].scf_iterations > 0);
    assert!(result.candidates[0].donor_group.len() > 1);
}

fn heavy_only(mol: &Molecule) -> Molecule {
    let mut map = vec![None; mol.atoms.len()];
    let mut atoms = Vec::new();
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.element != Element::H {
            map[i] = Some(atoms.len());
            atoms.push(a.clone());
        }
    }
    Molecule {
        atoms,
        bonds: mol
            .bonds
            .iter()
            .filter_map(|b| {
                Some(Bond {
                    a: map[b.a]?,
                    b: map[b.b]?,
                    order: Default::default(),
                    kind: Default::default(),
                })
            })
            .collect(),
        info: Default::default(),
    }
}
#[test]
fn protonation_builds_complete_protein_and_water_compatible_with_amoeba() {
    let ff = parameters::ForceField::builtin().unwrap();
    let cancel = AtomicBool::new(false);
    for name in ["villin", "water_trimer"] {
        let original = oracles()
            .into_iter()
            .find(|o| o.name == name)
            .unwrap()
            .molecule();
        let heavy = heavy_only(&original);
        let prepared = protonation::prepare(&heavy, Default::default(), &cancel).unwrap();
        assert!(prepared.report.added > 0);
        assert!(
            prepared
                .report
                .residues
                .iter()
                .all(|r| r.status == "prepared"),
            "{name}: {:?}",
            prepared.report.residues
        );
        for (i, a) in heavy.atoms.iter().enumerate() {
            assert_eq!(
                a.position,
                prepared.molecule.atoms[prepared.old_to_new[i].unwrap()].position
            );
        }
        let (t, status) = topology::prepare(&prepared.molecule, &ff, &cancel).unwrap();
        assert_eq!(
            t.original.len(),
            prepared.molecule.atoms.len(),
            "{name}: {status:?}"
        );
        let r = analyze(
            &prepared.molecule,
            "prepared",
            &[0],
            Default::default(),
            &cancel,
        )
        .unwrap();
        assert!(r.full_polarization.is_finite());
        for &(h, parent) in &prepared.added_parents {
            let d = prepared.molecule.atoms[h]
                .position
                .distance(prepared.molecule.atoms[parent].position);
            assert!((0.9..1.4).contains(&d));
        }
    }
}
#[test]
fn ph_switches_supported_sidechains_and_tautomers_without_moving_heavy_atoms() {
    let cancel = AtomicBool::new(false);
    let cases = oracles();
    for (name, low, high) in [
        ("ASP", "ASH", "ASP"),
        ("GLU", "GLH", "GLU"),
        ("LYS", "LYS", "LYD"),
        ("CYS", "CYS", "CYD"),
        ("TYR", "TYR", "TYD"),
        ("HIS", "HIS", "HIE"),
    ] {
        let mol = heavy_only(&cases.iter().find(|o| o.name == name).unwrap().molecule());
        for (ph, expected) in [(2., low), (12., high)] {
            let p = protonation::prepare(
                &mol,
                protonation::Settings {
                    ph,
                    ..Default::default()
                },
                &cancel,
            )
            .unwrap();
            assert!(
                p.report
                    .residues
                    .iter()
                    .any(|r| r.template.as_deref() == Some(expected)),
                "{name} at {ph}: {:?}",
                p.report
            );
            let (_, status) = topology::prepare(
                &p.molecule,
                &parameters::ForceField::builtin().unwrap(),
                &cancel,
            )
            .unwrap();
            assert!(
                status.iter().all(|r| r.status == "parameterized"),
                "{name}: {status:?}"
            );
        }
    }
    let mol = heavy_only(&cases.iter().find(|o| o.name == "HIS").unwrap().molecule());
    let p = protonation::prepare(
        &mol,
        protonation::Settings {
            ph: 7.,
            histidine: protonation::HistidineTautomer::Delta,
            ..Default::default()
        },
        &cancel,
    )
    .unwrap();
    assert!(
        p.report
            .residues
            .iter()
            .any(|r| r.template.as_deref() == Some("HID"))
    );
}
#[test]
fn preparation_skips_incomplete_and_unsupported_states_and_is_deterministic() {
    let cases = oracles();
    let cancel = AtomicBool::new(false);
    let mut mol = heavy_only(&cases.iter().find(|o| o.name == "LYS").unwrap().molecule());
    let missing = mol.atoms.iter().position(|a| a.name == "NZ").unwrap();
    mol.atoms[missing].name = "UNKNOWN".into();
    let p = protonation::prepare(&mol, Default::default(), &cancel).unwrap();
    assert!(
        p.report
            .residues
            .iter()
            .any(|r| r.status == "skipped" && r.reason.contains("NZ"))
    );
    let mol = heavy_only(&cases.iter().find(|o| o.name == "NALA").unwrap().molecule());
    let p = protonation::prepare(
        &mol,
        protonation::Settings {
            ph: 12.,
            ..Default::default()
        },
        &cancel,
    )
    .unwrap();
    assert!(
        p.report
            .residues
            .iter()
            .any(|r| r.status == "skipped" && r.reason.contains("neutral terminus"))
    );
    let mol = heavy_only(
        &cases
            .iter()
            .find(|o| o.name == "water_trimer")
            .unwrap()
            .molecule(),
    );
    let a = protonation::prepare(&mol, Default::default(), &cancel).unwrap();
    let b = protonation::prepare(&mol, Default::default(), &cancel).unwrap();
    assert_eq!(a.molecule, b.molecule);
    let c = protonation::prepare(&a.molecule, Default::default(), &cancel).unwrap();
    assert_eq!(c.report.added, a.report.added);
    assert_eq!(c.report.removed, a.report.added);
    assert_eq!(c.molecule.atoms.len(), a.molecule.atoms.len());
    assert!(
        protonation::prepare(
            &mol,
            protonation::Settings {
                ph: f64::NAN,
                ..Default::default()
            },
            &cancel
        )
        .is_err()
    );
    assert!(matches!(
        protonation::prepare(&mol, Default::default(), &AtomicBool::new(true)),
        Err(AmoebaError::Cancelled)
    ));
}

#[test]
fn repairs_real_protein_sidechain_before_protonation_and_preserves_original_indices() {
    let original = oracles()
        .into_iter()
        .find(|o| o.name == "villin")
        .unwrap()
        .molecule();
    let heavy = heavy_only(&original);
    let target = heavy
        .atoms
        .iter()
        .find(|a| a.residue_name == "LYS")
        .unwrap()
        .residue_number;
    let mut map = vec![None; heavy.atoms.len()];
    let mut atoms = Vec::new();
    for (i, a) in heavy.atoms.iter().enumerate() {
        if a.residue_number == target && matches!(a.name.as_str(), "CG" | "CD" | "CE" | "NZ") {
            continue;
        }
        map[i] = Some(atoms.len());
        atoms.push(a.clone());
    }
    let damaged = Molecule {
        atoms,
        bonds: heavy
            .bonds
            .iter()
            .filter_map(|b| {
                Some(Bond {
                    a: map[b.a]?,
                    b: map[b.b]?,
                    order: Default::default(),
                    kind: Default::default(),
                })
            })
            .collect(),
        info: Default::default(),
    };
    let selected = damaged
        .atoms
        .iter()
        .position(|a| a.residue_number == target && a.name == "CA")
        .unwrap();
    let scoped = protonation::prepare_selected(
        &damaged,
        Default::default(),
        &[selected],
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(scoped.report.residues.len(), 1);
    assert_eq!(scoped.report.residues[0].status, "prepared");
    assert_eq!(scoped.restored_atoms.len(), 4);
    assert!(
        scoped
            .added_parents
            .iter()
            .all(|&(i, _)| scoped.molecule.atoms[i].residue_number == target)
    );
    assert!(
        !scoped
            .molecule
            .atoms
            .iter()
            .any(|a| a.residue_number == target && a.name == "OXT")
    );
    for (i, atom) in damaged
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.residue_number != target)
    {
        assert_eq!(atom, &scoped.molecule.atoms[scoped.old_to_new[i].unwrap()]);
    }
    let repaired =
        protonation::prepare(&damaged, Default::default(), &AtomicBool::new(false)).unwrap();
    assert_eq!(
        repaired.restored_atoms.len(),
        4,
        "{:?}",
        repaired.report.repair
    );
    for (i, a) in damaged.atoms.iter().enumerate() {
        assert_eq!(
            a.position,
            repaired.molecule.atoms[repaired.old_to_new[i].unwrap()].position
        );
    }
    let (top, status) = topology::prepare(
        &repaired.molecule,
        &parameters::ForceField::builtin().unwrap(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(
        top.original.len(),
        repaired.molecule.atoms.len(),
        "{status:?}"
    );
}
