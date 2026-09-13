//! Fixed-anchor reconstruction from wwPDB CCD ideal heavy-atom geometry.
//! Scores are geometric restraints/steric penalties, not AMOEBA energies.
use super::{
    AmoebaError,
    parameters::ForceField,
    topology::{ChemicalGraph, check_cancel, chemical_graph},
};
use crate::molecule::{Atom, Bond, Element, Molecule};
use glam::{DMat3, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};
#[derive(Deserialize)]
struct IdealAtom {
    name: String,
    element: String,
    position: [f64; 3],
}
#[derive(Deserialize)]
struct IdealBond {
    atoms: [usize; 2],
    order: String,
}
#[derive(Deserialize)]
struct Template {
    atoms: Vec<IdealAtom>,
    bonds: Vec<IdealBond>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueReport {
    pub residue: String,
    pub status: String,
    pub atoms: Vec<String>,
    pub reason: String,
    pub max_bond_error: f64,
    pub minimum_contact: Option<f64>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Report {
    pub added: usize,
    pub residues: Vec<ResidueReport>,
}
pub struct Repaired {
    pub molecule: Molecule,
    /// New atom and an existing CA anchor; original indices never change.
    pub added_parents: Vec<(usize, usize)>,
    pub report: Report,
}
fn err(s: impl Into<String>) -> AmoebaError {
    AmoebaError::Invalid(s.into())
}
fn unit(v: DVec3) -> Result<DVec3, AmoebaError> {
    if v.length() < 1e-6 {
        Err(err("degenerate backbone anchors"))
    } else {
        Ok(v.normalize())
    }
}
fn frame(n: DVec3, ca: DVec3, c: DVec3) -> Result<DMat3, AmoebaError> {
    let z = unit(c - ca)?;
    let x = unit(n - ca - z * (n - ca).dot(z))?;
    Ok(DMat3::from_cols(x, z.cross(x), z))
}

/// Restore supported residue fragments with known N, CA and C. No residues,
/// backbone traces, caps or cross-residue covalent links are synthesized.
pub fn restore(mol: &Molecule, cancel: &AtomicBool) -> Result<Repaired, AmoebaError> {
    restore_selected(mol, None, cancel)
}

pub(super) fn restore_selected(
    mol: &Molecule,
    selected: Option<&[usize]>,
    cancel: &AtomicBool,
) -> Result<Repaired, AmoebaError> {
    check_cancel(cancel)?;
    if mol.atoms.iter().any(|a| !a.position.is_finite())
        || mol
            .bonds
            .iter()
            .any(|b| b.a >= mol.atoms.len() || b.b >= mol.atoms.len() || b.a == b.b)
    {
        return Err(err("invalid molecular coordinates or bonds"));
    }
    let templates: BTreeMap<String, Template> =
        serde_json::from_str(include_str!("../../../data/amoeba/heavy_templates.json"))?;
    let selected = selected.map(|indices| indices.iter().copied().collect::<BTreeSet<_>>());
    let ff = ForceField::builtin()?;
    let g = chemical_graph(mol, &ff, cancel)?;
    let mut output = Repaired {
        molecule: mol.clone(),
        added_parents: Vec::new(),
        report: Report::default(),
    };
    let mut serial = mol.atoms.iter().map(|a| a.serial).max().unwrap_or(0);
    let mut bonds: BTreeSet<_> = mol
        .bonds
        .iter()
        .map(|b| (b.a.min(b.b), b.a.max(b.b)))
        .collect();
    let mut last_protein = BTreeMap::new();
    for (ri, r) in g.residues.iter().enumerate() {
        if templates.contains_key(&ff.family(&r.canonical)) {
            last_protein.insert(&r.chain, ri);
        }
    }
    for (ri, r) in g.residues.iter().enumerate() {
        check_cancel(cancel)?;
        if selected
            .as_ref()
            .is_some_and(|s| !r.indices.iter().any(|i| s.contains(i)))
        {
            continue;
        }
        let Some(t) = templates.get(&ff.family(&r.canonical)) else {
            continue;
        };
        let old: BTreeMap<_, _> = r
            .indices
            .iter()
            .copied()
            .filter(|&i| mol.atoms[i].element != Element::H)
            .map(|i| (g.names[i].as_str(), i))
            .collect();
        let external_c = old
            .get("C")
            .is_some_and(|&i| g.bonds[i].iter().any(|&j| g.atom_res[j] != ri));
        // OXT is only inferred at the end of an observed protein chain, never
        // at an internal gap or at an explicitly capped/linked carbonyl carbon.
        let terminal =
            old.contains_key("OXT") || last_protein.get(&r.chain) == Some(&ri) && !external_c;
        let wanted: Vec<_> = (0..t.atoms.len())
            .filter(|&i| t.atoms[i].name != "OXT" || terminal)
            .collect();
        let missing: Vec<_> = wanted
            .iter()
            .copied()
            .filter(|&i| !old.contains_key(t.atoms[i].name.as_str()))
            .collect();
        if missing.is_empty() {
            continue;
        }
        let mut row = ResidueReport {
            residue: r.label.clone(),
            status: "skipped".into(),
            atoms: missing.iter().map(|&i| t.atoms[i].name.clone()).collect(),
            reason: String::new(),
            max_bond_error: 0.,
            minimum_contact: None,
        };
        if old.len()
            != r.indices
                .iter()
                .filter(|&&i| mol.atoms[i].element != Element::H)
                .count()
            || old
                .keys()
                .any(|name| !t.atoms.iter().any(|a| a.name == *name))
        {
            row.reason = "Duplicate or nonstandard heavy-atom names; residue left unchanged".into();
            output.report.residues.push(row);
            continue;
        }
        if ["N", "CA", "C"].iter().any(|a| !old.contains_key(a)) {
            row.reason="Requires existing N, CA and C backbone anchors; backbone gaps are not reconstructed".into();
            output.report.residues.push(row);
            continue;
        }
        if old.iter().any(|(name, &i)| {
            t.atoms.iter().find(|a| a.name == *name).is_none_or(|a| {
                !a.element
                    .eq_ignore_ascii_case(mol.atoms[i].element.symbol())
            })
        }) {
            row.reason = "Heavy-atom element does not match CCD template".into();
            output.report.residues.push(row);
            continue;
        }
        match reconstruct(
            t,
            &wanted,
            &missing,
            &old,
            &g,
            mol,
            &output.molecule,
            cancel,
        ) {
            Ok((coordinates, bond_error, contact)) => {
                let parent = old["CA"];
                let mut mapping: BTreeMap<usize, usize> = wanted
                    .iter()
                    .filter_map(|&i| old.get(t.atoms[i].name.as_str()).map(|&j| (i, j)))
                    .collect();
                for &i in &missing {
                    let mut atom: Atom = mol.atoms[parent].clone();
                    serial = serial
                        .checked_add(1)
                        .ok_or_else(|| err("atom serial number overflow"))?;
                    atom.serial = serial;
                    atom.name = t.atoms[i].name.clone();
                    atom.element = t.atoms[i]
                        .element
                        .parse()
                        .map_err(|_| err("unsupported CCD element"))?;
                    atom.position = coordinates[i].as_vec3();
                    atom.occupancy = 1.;
                    let index = output.molecule.atoms.len();
                    output.molecule.atoms.push(atom);
                    mapping.insert(i, index);
                    output.added_parents.push((index, parent));
                }
                for b in &t.bonds {
                    if let (Some(&a), Some(&b)) =
                        (mapping.get(&b.atoms[0]), mapping.get(&b.atoms[1]))
                    {
                        bonds.insert((a.min(b), a.max(b)));
                    }
                }
                row.status = "restored".into();
                row.max_bond_error = bond_error;
                row.minimum_contact = contact;
                row.reason="Modelled CCD geometry; existing atoms fixed; torsion search and restrained clash refinement".into();
                output.report.added += missing.len();
            }
            Err(AmoebaError::Cancelled) => return Err(AmoebaError::Cancelled),
            Err(e) => {
                row.reason = e.to_string();
            }
        }
        output.report.residues.push(row);
    }
    output.molecule.bonds = bonds.into_iter().map(|(a, b)| Bond { a, b }).collect();
    Ok(output)
}
struct Distance {
    a: usize,
    b: usize,
    target: f64,
    weight: f64,
}
struct Volume {
    center: usize,
    neighbors: [usize; 3],
    target: f64,
}
struct Geometry {
    restraints: Vec<Distance>,
    volumes: Vec<Volume>,
    contacts: Vec<(usize, DVec3, f64)>,
    nonbonded: Vec<(usize, usize, f64)>,
}
impl Geometry {
    fn evaluate(&self, p: &[DVec3]) -> (f64, Vec<DVec3>) {
        let mut score = 0.;
        let mut grad = vec![DVec3::ZERO; p.len()];
        let mut distance =
            |a: usize, b: Option<usize>, q: DVec3, target: f64, w: f64, repulsive: bool| {
                let v = p[a] - q;
                let d = v.length().max(1e-8);
                let delta = d - target;
                if !repulsive || delta < 0. {
                    score += w * delta * delta;
                    let f = v * (2. * w * delta / d);
                    grad[a] += f;
                    if let Some(b) = b {
                        grad[b] -= f;
                    }
                }
            };
        for r in &self.restraints {
            distance(r.a, Some(r.b), p[r.b], r.target, r.weight, false);
        }
        for &(i, q, r) in &self.contacts {
            distance(i, None, q, r, 4., true);
        }
        for &(i, j, r) in &self.nonbonded {
            distance(i, Some(j), p[j], r, 4., true);
        }
        for v in &self.volumes {
            let [a, b, c] = v.neighbors;
            let x = p[a] - p[v.center];
            let y = p[b] - p[v.center];
            let z = p[c] - p[v.center];
            let delta = x.dot(y.cross(z)) - v.target;
            score += 2. * delta * delta;
            let fa = 4. * delta * y.cross(z);
            let fb = 4. * delta * z.cross(x);
            let fc = 4. * delta * x.cross(y);
            grad[a] += fa;
            grad[b] += fb;
            grad[c] += fc;
            grad[v.center] -= fa + fb + fc;
        }
        (score, grad)
    }
    fn refine(
        &self,
        mut p: Vec<DVec3>,
        missing: &[usize],
        cancel: &AtomicBool,
    ) -> Result<Vec<DVec3>, AmoebaError> {
        for _ in 0..250 {
            check_cancel(cancel)?;
            let (score, gradient) = self.evaluate(&p);
            if missing
                .iter()
                .map(|&i| gradient[i].length_squared())
                .sum::<f64>()
                < 1e-8
            {
                break;
            }
            let mut step = 0.01;
            let mut accepted = false;
            for _ in 0..16 {
                let mut next = p.clone();
                for &i in missing {
                    next[i] -= gradient[i] * step;
                }
                if self.evaluate(&next).0 < score {
                    p = next;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        Ok(p)
    }
}
fn radius(element: &str) -> f64 {
    match element {
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.8,
        "H" => 1.1,
        _ => 1.7,
    }
}
fn component(edges: &[Vec<usize>], start: usize, cut: (usize, usize)) -> BTreeSet<usize> {
    let mut seen = BTreeSet::from([start]);
    let mut stack = vec![start];
    while let Some(i) = stack.pop() {
        for &j in &edges[i] {
            if (i, j) == cut || (j, i) == cut {
                continue;
            }
            if seen.insert(j) {
                stack.push(j);
            }
        }
    }
    seen
}
#[allow(clippy::too_many_arguments)]
fn reconstruct(
    t: &Template,
    wanted: &[usize],
    missing: &[usize],
    old: &BTreeMap<&str, usize>,
    g: &ChemicalGraph,
    mol: &Molecule,
    environment: &Molecule,
    cancel: &AtomicBool,
) -> Result<(Vec<DVec3>, f64, Option<f64>), AmoebaError> {
    let index = |name: &str| {
        t.atoms
            .iter()
            .position(|a| a.name == name)
            .ok_or_else(|| err("CCD backbone anchor missing"))
    };
    let n = index("N")?;
    let ca = index("CA")?;
    let c = index("C")?;
    let ideal: Vec<_> = t
        .atoms
        .iter()
        .map(|a| DVec3::from_array(a.position))
        .collect();
    let target_n = mol.atoms[old["N"]].position.as_dvec3();
    let target_ca = mol.atoms[old["CA"]].position.as_dvec3();
    let target_c = mol.atoms[old["C"]].position.as_dvec3();
    let rotation =
        frame(target_n, target_ca, target_c)? * frame(ideal[n], ideal[ca], ideal[c])?.transpose();
    let mut initial: Vec<_> = ideal
        .iter()
        .map(|&p| target_ca + rotation * (p - ideal[ca]))
        .collect();
    for (name, &i) in old {
        initial[index(name)?] = mol.atoms[i].position.as_dvec3();
    }
    let mut fixed_carbonyl = BTreeSet::new();
    for name in ["O", "OXT"] {
        if let Ok(i) = index(name)
            && missing.contains(&i)
        {
            let partner = if name == "O" {
                g.bonds[old["C"]]
                    .iter()
                    .copied()
                    .find(|&j| {
                        g.atom_res[j] != g.atom_res[old["C"]] && mol.atoms[j].element == Element::N
                    })
                    .or_else(|| old.get("OXT").copied())
            } else {
                old.get("O").copied()
            };
            if let Some(partner) = partner {
                let away = -unit(
                    unit(target_ca - target_c)?
                        + unit(mol.atoms[partner].position.as_dvec3() - target_c)?,
                )?;
                initial[i] = target_c + away * ideal[i].distance(ideal[c]);
                fixed_carbonyl.insert(i);
            }
        }
    }
    let free: Vec<_> = missing
        .iter()
        .copied()
        .filter(|i| !fixed_carbonyl.contains(i))
        .collect();
    let mut edges = vec![Vec::new(); ideal.len()];
    let mut constraints = BTreeMap::new();
    for b in &t.bonds {
        let [a, b] = b.atoms;
        if wanted.contains(&a) && wanted.contains(&b) {
            edges[a].push(b);
            edges[b].push(a);
            constraints.insert((a.min(b), a.max(b)), 80.);
        }
    }
    for neighbors in &edges {
        for &a in neighbors {
            for &b in neighbors {
                if a < b {
                    constraints.entry((a, b)).or_insert(20.);
                }
            }
        }
    }
    // Preserve complete ring shape, including fused aromatics and proline's
    // nonplanar ring, rather than independently constructing ring bonds.
    let mut ring_edges = vec![Vec::new(); ideal.len()];
    for b in &t.bonds {
        let [a, b] = b.atoms;
        if wanted.contains(&a) && wanted.contains(&b) && component(&edges, a, (a, b)).contains(&b) {
            ring_edges[a].push(b);
            ring_edges[b].push(a);
        }
    }
    for i in 0..ideal.len() {
        if ring_edges[i].is_empty() {
            continue;
        }
        let ring = component(&ring_edges, i, (usize::MAX, usize::MAX));
        for &a in &ring {
            for &b in &ring {
                if a < b {
                    constraints.entry((a, b)).or_insert(20.);
                }
            }
        }
    }
    let mut geometry = Geometry {
        restraints: constraints
            .iter()
            .map(|(&(a, b), &weight)| Distance {
                a,
                b,
                target: ideal[a].distance(ideal[b]),
                weight,
            })
            .collect(),
        volumes: Vec::new(),
        contacts: Vec::new(),
        nonbonded: Vec::new(),
    };
    for &center in wanted {
        if t.atoms[center].element == "C"
            && edges[center].len() == 3
            && t.bonds
                .iter()
                .filter(|b| b.atoms.contains(&center))
                .all(|b| b.order == "SING")
        {
            let neighbors = [edges[center][0], edges[center][1], edges[center][2]];
            let [a, b, c] = neighbors;
            geometry.volumes.push(Volume {
                center,
                neighbors,
                target: (ideal[a] - ideal[center])
                    .dot((ideal[b] - ideal[center]).cross(ideal[c] - ideal[center])),
            });
        }
    }
    let own: BTreeSet<_> = g.residues[g.atom_res[old["CA"]]]
        .indices
        .iter()
        .copied()
        .collect();
    let nearby: Vec<_> = environment
        .atoms
        .iter()
        .enumerate()
        .filter(|(j, a)| !own.contains(j) && a.position.as_dvec3().distance(target_ca) < 13.)
        .collect();
    for &i in missing {
        let excluded: BTreeSet<_> = edges[i]
            .iter()
            .filter_map(|&k| old.get(t.atoms[k].name.as_str()))
            .flat_map(|&k| g.bonds[k].iter().copied())
            .collect();
        for &(j, a) in &nearby {
            if !excluded.contains(&j) {
                geometry.contacts.push((
                    i,
                    a.position.as_dvec3(),
                    0.8 * (radius(&t.atoms[i].element) + radius(a.element.symbol())),
                ));
            }
        }
        for &j in wanted {
            if i != j
                && !constraints.contains_key(&(i.min(j), i.max(j)))
                && (!missing.contains(&j) || i < j)
            {
                geometry.nonbonded.push((
                    i,
                    j,
                    0.8 * (radius(&t.atoms[i].element) + radius(&t.atoms[j].element)),
                ));
            }
        }
    }
    let mut variants = vec![initial.clone()];
    for b in &t.bonds {
        if b.order != "SING" {
            continue;
        }
        let [mut a, mut b] = b.atoms;
        if !wanted.contains(&a) || !wanted.contains(&b) {
            continue;
        }
        let mut side = component(&edges, b, (a, b));
        if side.contains(&ca) {
            std::mem::swap(&mut a, &mut b);
            side = component(&edges, b, (a, b));
        }
        if side.contains(&a)
            || side.len() < 2
            || side.iter().any(|&i| i != b && !missing.contains(&i))
            || side
                .iter()
                .any(|&i| matches!(t.atoms[i].name.as_str(), "N" | "CA" | "C" | "O" | "OXT"))
        {
            continue;
        }
        let mut expanded = Vec::new();
        for p in &variants {
            for angle in [
                0.,
                std::f64::consts::TAU / 3.,
                2. * std::f64::consts::TAU / 3.,
            ] {
                let axis = unit(p[b] - p[a])?;
                let q = DQuat::from_axis_angle(axis, angle);
                let mut v = p.clone();
                for &i in &side {
                    if missing.contains(&i) {
                        v[i] = p[a] + q * (p[i] - p[a]);
                    }
                }
                expanded.push(v);
            }
        }
        // A deterministic beam bounds cost for long side chains. There is no
        // claim that the chosen state is a statistically likely rotamer.
        expanded.sort_by(|a, b| geometry.evaluate(a).0.total_cmp(&geometry.evaluate(b).0));
        expanded.truncate(24);
        variants = expanded;
    }
    variants.sort_by(|a, b| geometry.evaluate(a).0.total_cmp(&geometry.evaluate(b).0));
    variants.truncate(6);
    let mut best = None;
    for p in variants {
        let p = geometry.refine(p, &free, cancel)?;
        let score = geometry.evaluate(&p).0;
        if best.as_ref().is_none_or(|(s, _)| score < *s) {
            best = Some((score, p));
        }
    }
    let (_, p) = best.ok_or_else(|| err("no reconstruction conformer"))?;
    let max_bond = t
        .bonds
        .iter()
        .filter(|b| {
            b.atoms.iter().any(|i| missing.contains(i))
                && b.atoms.iter().all(|i| wanted.contains(i))
        })
        .map(|b| {
            let [a, b] = b.atoms;
            (p[a].distance(p[b]) - ideal[a].distance(ideal[b])).abs()
        })
        .fold(0., f64::max);
    let contact = geometry
        .contacts
        .iter()
        .map(|&(i, q, _)| p[i].distance(q))
        .chain(
            geometry
                .nonbonded
                .iter()
                .map(|&(i, j, _)| p[i].distance(p[j])),
        )
        .reduce(f64::min);
    if !p.iter().all(|p| p.is_finite()) || max_bond > 0.15 {
        return Err(err(format!(
            "fixed anchors cannot be reconciled with ideal bonds (maximum error {max_bond:.3} Å)"
        )));
    }
    if contact.is_some_and(|d| d < 1.2) {
        return Err(err(
            "unresolved severe steric overlap; residue left unchanged",
        ));
    }
    for v in &geometry.volumes {
        let [a, b, c] = v.neighbors;
        let actual = (p[a] - p[v.center]).dot((p[b] - p[v.center]).cross(p[c] - p[v.center]));
        if actual * v.target <= 0. {
            return Err(err(
                "fixed anchors incompatible with template stereochemistry",
            ));
        }
    }
    // Verify all existing coordinates stayed exact even through refinement.
    for (name, &i) in old {
        if p[index(name)?] != mol.atoms[i].position.as_dvec3() {
            return Err(err("reconstruction moved an existing atom"));
        }
    }
    Ok((p, max_bond, contact))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn templates() -> BTreeMap<String, Template> {
        serde_json::from_str(include_str!("../../../data/amoeba/heavy_templates.json")).unwrap()
    }
    fn backbone(name: &str, t: &Template) -> Molecule {
        let atoms = t
            .atoms
            .iter()
            .filter(|a| matches!(a.name.as_str(), "N" | "CA" | "C" | "O"))
            .enumerate()
            .map(|(i, a)| Atom {
                serial: i as u32 + 1,
                name: a.name.clone(),
                element: a.element.parse().unwrap(),
                residue_name: name.into(),
                residue_number: 1,
                insertion_code: None,
                chain_id: "A".into(),
                position: DVec3::from_array(a.position).as_vec3(),
                occupancy: 1.,
                b_factor: 0.,
                hetero: false,
            })
            .collect();
        Molecule {
            atoms,
            bonds: Vec::new(),
        }
    }
    #[test]
    fn restores_all_twenty_sidechains_including_rings_without_moving_anchors() {
        let templates = templates();
        assert_eq!(templates.len(), 20);
        let cancel = AtomicBool::new(false);
        for (name, t) in templates {
            let mol = backbone(&name, &t);
            let r = restore(&mol, &cancel).unwrap();
            assert_eq!(
                r.molecule.atoms.len(),
                t.atoms.len(),
                "{name}: {:?}",
                r.report
            );
            assert_eq!(&r.molecule.atoms[..mol.atoms.len()], mol.atoms.as_slice());
            assert!(
                r.report.residues.iter().all(|r| r.status == "restored"),
                "{name}: {:?}",
                r.report
            );
            for b in &t.bonds {
                let [i, j] = b.atoms;
                let a = r
                    .molecule
                    .atoms
                    .iter()
                    .find(|a| a.name == t.atoms[i].name)
                    .unwrap();
                let b = r
                    .molecule
                    .atoms
                    .iter()
                    .find(|a| a.name == t.atoms[j].name)
                    .unwrap();
                let ideal = DVec3::from_array(t.atoms[i].position)
                    .distance(DVec3::from_array(t.atoms[j].position));
                assert!(
                    (a.position.distance(b.position) as f64 - ideal).abs() < 0.15,
                    "{name}: {} {}",
                    a.name,
                    b.name
                );
            }
            let again = restore(&r.molecule, &cancel).unwrap();
            assert_eq!(again.report.added, 0);
            assert_eq!(again.molecule, r.molecule);
        }
    }
    #[test]
    fn refuses_backbone_gaps_and_cancels_without_partial_mutation() {
        let t = templates();
        let mut mol = backbone("LYS", &t["LYS"]);
        mol.atoms.retain(|a| a.name != "CA");
        let r = restore(&mol, &AtomicBool::new(false)).unwrap();
        assert_eq!(r.molecule, mol);
        assert!(r.report.residues[0].reason.contains("backbone anchors"));
        assert!(matches!(
            restore(&mol, &AtomicBool::new(true)),
            Err(AmoebaError::Cancelled)
        ));
    }
    #[test]
    fn conflicting_fixed_stereochemistry_is_not_silently_overwritten() {
        let t = templates();
        let mut mol = backbone("ILE", &t["ILE"]);
        let mut cb = mol.atoms[1].clone();
        cb.name = "CB".into();
        cb.position = mol.atoms[1].position + (mol.atoms[0].position - mol.atoms[1].position) * 0.1;
        mol.atoms.push(cb);
        let r = restore(&mol, &AtomicBool::new(false)).unwrap();
        assert_eq!(r.report.added, 0);
        assert_eq!(r.molecule, mol);
    }
}
