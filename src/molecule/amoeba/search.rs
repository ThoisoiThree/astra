use super::{
    AmoebaError, AnalysisSettings, HydrogenBondCandidate, HydrogenBondReport,
    energy::Evaluator,
    parameters::ForceField,
    topology::{self, Topology, check_cancel},
};
use crate::molecule::Molecule;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};
fn accepts(i: usize, t: &Topology, q: f64) -> bool {
    let symbol = |j: usize| t.elements[j].as_str();
    let h = t.bonds[i].iter().any(|&j| symbol(j) == "H");
    match symbol(i) {
        "F" | "Cl" | "Br" | "I" => t.bonds[i].is_empty() && q < 0.,
        "O" => {
            !(h && t.bonds[i].iter().any(|&j| {
                symbol(j) == "P"
                    || (symbol(j) == "C"
                        && t.bonds[j].iter().filter(|&&k| symbol(k) == "O").count() >= 2)
            }))
        }
        "S" => t.bonds[i].len() <= 2,
        "N" => {
            let template = &t.templates[i];
            let residue = if template.len() == 4
                && (template.starts_with('N') || template.starts_with('C'))
            {
                &template[1..]
            } else {
                template.as_str()
            };
            let name = t.names[i].as_str();
            if template.starts_with('R') || template.starts_with('D') {
                let base = template.as_bytes().get(1).copied();
                return !h
                    && match base {
                        Some(b'A') => matches!(name, "N1" | "N3" | "N7"),
                        Some(b'G') => matches!(name, "N3" | "N7"),
                        Some(b'C') => name == "N3",
                        _ => false,
                    };
            }
            if matches!(residue, "HID" | "HIE" | "HIS" | "HIP") && matches!(name, "ND1" | "NE2") {
                return !h;
            }
            residue == "LYD" && name == "NZ" && t.bonds[i].len() < 4
        }
        _ => false,
    }
}
pub(super) fn run(
    mol: &Molecule,
    selection: &str,
    selected: &[usize],
    settings: AnalysisSettings,
    cancel: &AtomicBool,
    ff: &ForceField,
) -> Result<HydrogenBondReport, AmoebaError> {
    let (t, residues) = topology::prepare(mol, ff, cancel)?;
    let evaluator = Evaluator::new(&t, ff, settings, cancel)?;
    let (full_permanent, full_vdw) = evaluator.pairwise(None)?;
    let full = evaluator.polarization(None)?;
    let mut report = HydrogenBondReport {
        version: 1,
        selection: selection.into(),
        force_field: "AMOEBA 2018 / native Rust".into(),
        cutoff: settings.cutoff,
        tolerance: settings.tolerance,
        max_iterations: settings.max_iterations,
        energy_threshold: 0.,
        candidates: Vec::new(),
        residues,
        full_permanent,
        full_vdw,
        full_polarization: full.energy,
        scf_iterations: full.iterations,
        scf_residual_debye: full.residual,
    };
    let selected: BTreeSet<_> = selected.iter().copied().collect();
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &g) in t.groups.iter().enumerate() {
        groups.entry(g).or_default().push(t.original[i]);
    }
    let cell = |i: usize| {
        let p = (t.xyz[i] * 10. / settings.cutoff).floor();
        (p.x as i64, p.y as i64, p.z as i64)
    };
    let mut cells: BTreeMap<(i64, i64, i64), Vec<usize>> = BTreeMap::new();
    for i in 0..t.original.len() {
        if selected.contains(&t.original[i]) && accepts(i, &t, evaluator.particles[i].q) {
            cells.entry(cell(i)).or_default().push(i);
        }
    }
    let mut scores = BTreeMap::new();
    for d in 0..t.original.len() {
        check_cancel(cancel)?;
        if !selected.contains(&t.original[d]) || !matches!(t.elements[d].as_str(), "N" | "O" | "S")
        {
            continue;
        }
        let c = cell(d);
        let mut nearby = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(v) = cells.get(&(
                        c.0.saturating_add(x),
                        c.1.saturating_add(y),
                        c.2.saturating_add(z),
                    )) {
                        nearby.extend(v);
                    }
                }
            }
        }
        nearby.sort_unstable();
        nearby.dedup();
        for &h in &t.bonds[d] {
            if t.elements[h] != "H" || !selected.contains(&t.original[h]) {
                continue;
            }
            for &a in &nearby {
                if a == d
                    || t.bonds[d].contains(&a)
                    || t.bonds[h].contains(&a)
                    || t.groups[a] == t.groups[d]
                {
                    continue;
                }
                let da = t.xyz[d].distance(t.xyz[a]) * 10.;
                if da > settings.cutoff {
                    continue;
                }
                // Domains must contain the complete donor-H functional group.
                if t.groups[h] != t.groups[d] {
                    return Err(AmoebaError::Invalid(
                        "donor and its hydrogen occupy different polarization domains".into(),
                    ));
                }
                let key = (t.groups[d].min(t.groups[a]), t.groups[d].max(t.groups[a]));
                if let std::collections::btree_map::Entry::Vacant(e) = scores.entry(key) {
                    let (perm, vdw) = evaluator.pairwise(Some(key))?;
                    let dec = evaluator.polarization(Some(key))?;
                    e.insert((perm, vdw, full.energy - dec.energy, dec));
                }
                let &(permanent, vdw, polarization, scf) = &scores[&key];
                let dh = t.xyz[d] - t.xyz[h];
                let ah = t.xyz[a] - t.xyz[h];
                let angle = (dh.normalize().dot(ah.normalize()))
                    .clamp(-1., 1.)
                    .acos()
                    .to_degrees();
                report.candidates.push(HydrogenBondCandidate {
                    donor: t.original[d],
                    hydrogen: t.original[h],
                    acceptor: t.original[a],
                    donor_group: groups[&t.groups[d]].clone(),
                    acceptor_group: groups[&t.groups[a]].clone(),
                    donor_position: mol.atoms[t.original[d]].position.to_array(),
                    acceptor_position: mol.atoms[t.original[a]].position.to_array(),
                    delta_energy: permanent + vdw + polarization,
                    permanent,
                    vdw,
                    polarization,
                    da_distance: da,
                    ha_distance: ah.length() * 10.,
                    dha_angle: angle,
                    parameterization_status: "parameterized".into(),
                    scf_iterations: scf.iterations,
                    scf_residual_debye: scf.residual,
                });
            }
        }
    }
    report.validate(mol.atoms.len())?;
    Ok(report)
}
