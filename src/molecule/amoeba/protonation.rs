//! Deterministic, template-based hydrogen construction with independent-site
//! EMBOSS Epk.dat pKa values. This is not a structure-dependent pKa predictor.
use super::{
    AmoebaError,
    parameters::{ForceField, Template},
    topology::{ChemicalGraph, check_cancel, chemical_graph},
};
use crate::molecule::{Atom, Bond, Element, Molecule};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistidineTautomer {
    Delta,
    #[default]
    Epsilon,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub ph: f64,
    pub histidine: HistidineTautomer,
    pub restore_heavy: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            ph: 7.,
            histidine: HistidineTautomer::Epsilon,
            restore_heavy: true,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Site {
    pub name: String,
    pub pka: f64,
    pub protonated_fraction: f64,
    pub protonated: bool,
    pub ambiguous: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueReport {
    pub residue: String,
    pub template: Option<String>,
    pub status: String,
    pub reason: String,
    pub sites: Vec<Site>,
    pub added: usize,
    pub removed: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    #[serde(default)]
    pub repair: Option<super::repair::Report>,
    pub ph: f64,
    pub method: String,
    pub residues: Vec<ResidueReport>,
    pub added: usize,
    pub removed: usize,
}
#[derive(Debug, Clone)]
pub struct Prepared {
    pub molecule: Molecule,
    pub old_to_new: Vec<Option<usize>>,
    pub added_parents: Vec<(usize, usize)>,
    pub restored_atoms: Vec<(usize, usize)>,
    pub report: Report,
}
/// EMBOSS Epk.dat, https://emboss.bioinformatics.nl/cgi-bin/emboss/help/iep
pub const PKA_TABLE: [(&str, f64); 9] = [
    ("ASP", 3.9),
    ("GLU", 4.1),
    ("HIS", 6.5),
    ("CYS", 8.5),
    ("TYR", 10.1),
    ("LYS", 10.8),
    ("ARG", 12.5),
    ("N-terminus", 8.6),
    ("C-terminus", 3.6),
];
fn site(name: &str, pka: f64, ph: f64) -> Site {
    Site {
        name: name.into(),
        pka,
        protonated_fraction: 1. / (1. + 10_f64.powf(ph - pka)),
        protonated: ph <= pka,
        ambiguous: (ph - pka).abs() <= 1.,
    }
}
fn error(message: impl Into<String>) -> AmoebaError {
    AmoebaError::Invalid(message.into())
}

fn choose(
    ri: usize,
    g: &ChemicalGraph,
    mol: &Molecule,
    ff: &ForceField,
    settings: Settings,
) -> Result<(&'static str, Vec<Site>), String> {
    let r = &g.residues[ri];
    let family = ff.family(&r.canonical);
    let mut sites = Vec::new();
    if family == "HOH" {
        return Ok(("HOH", sites));
    }
    let disulfide = r.indices.iter().any(|&i| {
        g.names[i] == "SG"
            && g.bonds[i]
                .iter()
                .any(|&j| g.atom_res[j] != ri && g.names[j] == "SG")
    });
    let state = match family.as_str() {
        "ASP" | "GLU" | "HIS" | "CYS" | "TYR" | "LYS" | "ARG" => {
            if family == "CYS" && disulfide {
                return Ok(("CYX", sites));
            }
            let pka = PKA_TABLE
                .iter()
                .find(|(n, _)| *n == family)
                .map(|(_, p)| *p)
                .ok_or("missing pKa")?;
            let s = site(&family, pka, settings.ph);
            let protonated = s.protonated;
            sites.push(s);
            match (family.as_str(), protonated) {
                ("ASP", true) => "ASH",
                ("ASP", false) => "ASP",
                ("GLU", true) => "GLH",
                ("GLU", false) => "GLU",
                ("HIS", true) => "HIS",
                ("HIS", false) => match settings.histidine {
                    HistidineTautomer::Delta => "HID",
                    HistidineTautomer::Epsilon => "HIE",
                },
                ("CYS", true) => "CYS",
                ("CYS", false) => "CYD",
                ("TYR", true) => "TYR",
                ("TYR", false) => "TYD",
                ("LYS", true) => "LYS",
                ("LYS", false) => "LYD",
                ("ARG", true) => "ARG",
                _ => {
                    return Err(
                        "Neutral ARG has no AMOEBA 2018 template; residue left unchanged".into(),
                    );
                }
            }
        }
        "ALA" => "ALA",
        "ASN" => "ASN",
        "GLN" => "GLN",
        "GLY" => "GLY",
        "ILE" => "ILE",
        "LEU" => "LEU",
        "MET" => "MET",
        "PHE" => "PHE",
        "PRO" => "PRO",
        "SER" => "SER",
        "THR" => "THR",
        "TRP" => "TRP",
        "VAL" => "VAL",
        "ACE" => "ACE",
        "NME" => "NME",
        _ => {
            return Err(
                if r.indices.len() == 1 && mol.atoms[r.indices[0]].element != Element::H {
                    "Monatomic species left unchanged"
                } else {
                    "No tabulated protein protonation model for this residue; left unchanged"
                }
                .into(),
            );
        }
    };
    Ok((state, sites))
}
fn template_mapping(
    ri: usize,
    g: &ChemicalGraph,
    mol: &Molecule,
    ff: &ForceField,
    t: &Template,
) -> Option<Vec<Option<usize>>> {
    let r = &g.residues[ri];
    let heavy: Vec<_> = r
        .indices
        .iter()
        .copied()
        .filter(|&i| mol.atoms[i].element != Element::H)
        .collect();
    if heavy.len()
        != t.atoms
            .iter()
            .filter(|a| ff.types[&a.r#type].element != "H")
            .count()
    {
        return None;
    }
    let mut map = vec![None; t.atoms.len()];
    let mut used = BTreeSet::new();
    for (k, a) in t.atoms.iter().enumerate() {
        if ff.types[&a.r#type].element == "H" {
            continue;
        }
        let i = heavy.iter().copied().find(|&i| {
            g.names[i] == a.name
                && mol.atoms[i]
                    .element
                    .symbol()
                    .eq_ignore_ascii_case(&ff.types[&a.r#type].element)
        })?;
        if !used.insert(i) {
            return None;
        }
        map[k] = Some(i);
    }
    for (k, oi) in map.iter().enumerate() {
        if let Some(i) = oi {
            let expected = t.external.iter().filter(|&&e| e == k).count();
            let actual = g.bonds[*i]
                .iter()
                .filter(|&&j| g.atom_res[j] != ri && mol.atoms[j].element != Element::H)
                .count();
            if expected != actual {
                return None;
            }
            let expected: BTreeSet<_> = t
                .bonds
                .iter()
                .filter_map(|&[a, b]| {
                    if a == k {
                        map[b]
                    } else if b == k {
                        map[a]
                    } else {
                        None
                    }
                })
                .collect();
            let actual: BTreeSet<_> = g.bonds[*i]
                .iter()
                .copied()
                .filter(|&j| g.atom_res[j] == ri && mol.atoms[j].element != Element::H)
                .collect();
            if expected != actual {
                return None;
            }
        }
    }
    Some(map)
}

/// Rebuild hydrogens only in complete, supported residues. Heavy coordinates
/// never move. Skipped residues retain all their original atoms and bonds.
pub fn prepare(
    mol: &Molecule,
    settings: Settings,
    cancel: &AtomicBool,
) -> Result<Prepared, AmoebaError> {
    prepare_scoped(mol, settings, None, cancel)
}

/// Prepare whole residues touched by the named selection. The full molecule
/// remains available for covalent topology, termini and steric surroundings.
/// Empty or invalid selections are errors, never an implicit whole-structure scope.
pub fn prepare_selected(
    mol: &Molecule,
    settings: Settings,
    selected: &[usize],
    cancel: &AtomicBool,
) -> Result<Prepared, AmoebaError> {
    if selected.is_empty() || selected.iter().any(|&i| i >= mol.atoms.len()) {
        return Err(error(
            "preparation requires a nonempty selection of valid atoms",
        ));
    }
    prepare_scoped(mol, settings, Some(selected), cancel)
}

fn prepare_scoped(
    mol: &Molecule,
    settings: Settings,
    selected: Option<&[usize]>,
    cancel: &AtomicBool,
) -> Result<Prepared, AmoebaError> {
    if !settings.ph.is_finite() || !(0. ..=14.).contains(&settings.ph) {
        return Err(error("pH must be between 0 and 14"));
    }
    if settings.restore_heavy {
        let repaired = super::repair::restore_selected(mol, selected, cancel)?;
        let mut prepared = prepare_hydrogens(&repaired.molecule, settings, selected, cancel)?;
        for (i, parent) in repaired.added_parents {
            if let (Some(i), Some(parent)) = (prepared.old_to_new[i], prepared.old_to_new[parent]) {
                prepared.restored_atoms.push((i, parent));
            }
        }
        prepared.old_to_new.truncate(mol.atoms.len());
        prepared.report.repair = Some(repaired.report);
        Ok(prepared)
    } else {
        prepare_hydrogens(mol, settings, selected, cancel)
    }
}

fn prepare_hydrogens(
    mol: &Molecule,
    settings: Settings,
    selected: Option<&[usize]>,
    cancel: &AtomicBool,
) -> Result<Prepared, AmoebaError> {
    check_cancel(cancel)?;
    if !settings.ph.is_finite() || !(0. ..=14.).contains(&settings.ph) {
        return Err(error("pH must be between 0 and 14"));
    }
    if mol.atoms.iter().any(|a| !a.position.is_finite())
        || mol
            .bonds
            .iter()
            .any(|b| b.a >= mol.atoms.len() || b.b >= mol.atoms.len() || b.a == b.b)
    {
        return Err(error("invalid molecular coordinates or bonds"));
    }
    let selected = selected.map(|indices| indices.iter().copied().collect::<BTreeSet<_>>());
    let ff = ForceField::builtin()?;
    let g = chemical_graph(mol, &ff, cancel)?;
    let mut replacements: BTreeMap<usize, (String, Vec<(Atom, usize)>)> = BTreeMap::new();
    let mut reports = Vec::new();
    for (ri, r) in g.residues.iter().enumerate() {
        check_cancel(cancel)?;
        if selected
            .as_ref()
            .is_some_and(|s| !r.indices.iter().any(|i| s.contains(i)))
        {
            continue;
        }
        let mut row = ResidueReport {
            residue: r.label.clone(),
            template: None,
            status: "skipped".into(),
            reason: String::new(),
            sites: Vec::new(),
            added: 0,
            removed: 0,
        };
        let (state, sites) = match choose(ri, &g, mol, &ff, settings) {
            Ok(v) => v,
            Err(e) => {
                if let Some((name, pka)) = PKA_TABLE
                    .iter()
                    .find(|(name, _)| *name == ff.family(&r.canonical))
                {
                    row.sites.push(site(name, *pka, settings.ph));
                }
                row.reason = e;
                reports.push(row);
                continue;
            }
        };
        row.sites = sites;
        let candidates: Vec<_> = ff
            .residues
            .iter()
            .filter(|t| {
                t.name == state || t.name == format!("N{state}") || t.name == format!("C{state}")
            })
            .collect();
        let matched = candidates
            .iter()
            .find_map(|&t| template_mapping(ri, &g, mol, &ff, t).map(|m| (t, m)));
        let Some((t, map)) = matched else {
            let missing = candidates
                .iter()
                .map(|t| {
                    t.atoms
                        .iter()
                        .filter(|a| {
                            ff.types[&a.r#type].element != "H"
                                && !r.indices.iter().any(|&i| g.names[i] == a.name)
                        })
                        .map(|a| a.name.clone())
                        .collect::<Vec<_>>()
                })
                .min_by_key(Vec::len)
                .unwrap_or_default();
            row.reason = if missing.is_empty() {
                "No complete template matches heavy-atom connectivity and termini; left unchanged"
                    .into()
            } else {
                format!(
                    "Missing heavy atoms: {}; left unchanged",
                    missing.join(", ")
                )
            };
            reports.push(row);
            continue;
        };
        // AMOEBA 2018 supplies charged protein termini only. Never silently
        // use that template when the pH model requests its neutral alternative.
        if t.name.len() == 4 && (t.name.starts_with('N') || t.name.starts_with('C')) {
            let nterm = t.name.starts_with('N');
            let s = site(
                if nterm { "N-terminus" } else { "C-terminus" },
                if nterm { 8.6 } else { 3.6 },
                settings.ph,
            );
            let unsupported = if nterm { !s.protonated } else { s.protonated };
            row.sites.push(s);
            if unsupported {
                row.reason =
                    "Requested neutral terminus has no AMOEBA 2018 template; left unchanged".into();
                reports.push(row);
                continue;
            }
        }
        match build_hydrogens(&g, mol, &ff, t, &map, cancel) {
            Ok(atoms) => {
                row.template = Some(t.name.clone());
                row.status = "prepared".into();
                row.added = atoms.len();
                row.removed = r
                    .indices
                    .iter()
                    .filter(|&&i| mol.atoms[i].element == Element::H)
                    .count();
                row.reason = if state == "HID" || state == "HIE" {
                    format!(
                        "{} tautomer chosen explicitly; pH alone does not resolve neutral histidine",
                        state
                    )
                } else {
                    "Independent-site tabulated pKa; ideal hydrogen geometry with torsion clash search".into()
                };
                replacements.insert(ri, (t.name.clone(), atoms));
            }
            Err(AmoebaError::Cancelled) => return Err(AmoebaError::Cancelled),
            Err(e) => {
                row.reason = format!("{e}; residue left unchanged");
            }
        }
        reports.push(row);
    }
    let mut molecule = Molecule {
        info: mol.info.clone(),
        ..Molecule::default()
    };
    let mut old_to_new = vec![None; mol.atoms.len()];
    for (i, a) in mol.atoms.iter().enumerate() {
        if replacements.contains_key(&g.atom_res[i]) && a.element == Element::H {
            continue;
        }
        let mut a = a.clone();
        if let Some((template, _)) = replacements.get(&g.atom_res[i]) {
            a.name = g.names[i].clone();
            a.residue_name = template.clone();
        }
        old_to_new[i] = Some(molecule.atoms.len());
        molecule.atoms.push(a);
    }
    let mut bonds = BTreeSet::new();
    for b in &mol.bonds {
        if let (Some(a), Some(b)) = (old_to_new[b.a], old_to_new[b.b]) {
            bonds.insert((a.min(b), a.max(b)));
        }
    }
    for (i, neighbors) in g.bonds.iter().enumerate() {
        for &j in neighbors {
            if (replacements.contains_key(&g.atom_res[i])
                || replacements.contains_key(&g.atom_res[j]))
                && let (Some(a), Some(b)) = (old_to_new[i], old_to_new[j])
            {
                bonds.insert((a.min(b), a.max(b)));
            }
        }
    }
    let mut next_serial = mol.atoms.iter().map(|a| a.serial).max().unwrap_or(0);
    let mut added_parents = Vec::new();
    for (_, (_, atoms)) in replacements {
        for (mut atom, old_parent) in atoms {
            next_serial = next_serial
                .checked_add(1)
                .ok_or_else(|| error("atom serial number overflow"))?;
            atom.serial = next_serial;
            let parent = old_to_new[old_parent].ok_or_else(|| error("missing hydrogen parent"))?;
            let i = molecule.atoms.len();
            molecule.atoms.push(atom);
            bonds.insert((parent, i));
            added_parents.push((i, parent));
        }
    }
    let original: std::collections::HashMap<(usize, usize), Bond> = mol
        .bonds
        .iter()
        .filter_map(|bond| {
            let (a, b) = (old_to_new[bond.a]?, old_to_new[bond.b]?);
            Some(((a.min(b), a.max(b)), *bond))
        })
        .collect();
    molecule.bonds = bonds
        .into_iter()
        .filter_map(|(a, b)| match original.get(&(a, b)) {
            Some(bond) => Bond::with_order(a, b, bond.order, bond.kind),
            None => Bond::new(a, b),
        })
        .collect();
    let report = Report {
        repair: None,
        ph: settings.ph,
        method: "EMBOSS Epk.dat; independent sites; pH <= pKa selects protonated state".into(),
        added: reports.iter().map(|r| r.added).sum(),
        removed: reports.iter().map(|r| r.removed).sum(),
        residues: reports,
    };
    Ok(Prepared {
        restored_atoms: Vec::new(),
        molecule,
        old_to_new,
        added_parents,
        report,
    })
}

fn normalized(v: DVec3) -> Result<DVec3, AmoebaError> {
    if v.length() < 1e-6 {
        Err(error("degenerate heavy-atom geometry"))
    } else {
        Ok(v.normalize())
    }
}
fn perpendicular(axis: DVec3) -> DVec3 {
    let v = if axis.x.abs() < 0.8 {
        DVec3::X
    } else {
        DVec3::Y
    };
    (v - axis * v.dot(axis)).normalize()
}
fn build_hydrogens(
    g: &ChemicalGraph,
    mol: &Molecule,
    ff: &ForceField,
    t: &Template,
    map: &[Option<usize>],
    cancel: &AtomicBool,
) -> Result<Vec<(Atom, usize)>, AmoebaError> {
    let mut attached: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, a) in t.atoms.iter().enumerate() {
        if ff.types[&a.r#type].element != "H" {
            continue;
        }
        let parents: Vec<_> = t
            .bonds
            .iter()
            .filter_map(|&[a, b]| {
                if a == k {
                    map[b]
                } else if b == k {
                    map[a]
                } else {
                    None
                }
            })
            .collect();
        if parents.len() != 1 {
            return Err(error("hydrogen template must have one heavy-atom parent"));
        }
        attached.entry(parents[0]).or_default().push(k);
    }
    let mut result: Vec<(Atom, usize)> = Vec::new();
    for (parent, hs) in attached {
        check_cancel(cancel)?;
        let p = &mol.atoms[parent];
        let pos = p.position.as_dvec3();
        let neighbors: Vec<_> = g.bonds[parent]
            .iter()
            .copied()
            .filter(|&i| mol.atoms[i].element != Element::H)
            .collect();
        let dirs: Vec<_> = neighbors
            .iter()
            .map(|&i| normalized(mol.atoms[i].position.as_dvec3() - pos))
            .collect::<Result<_, _>>()?;
        let planar = match p.element {
            Element::C => dirs.len() + hs.len() == 3,
            Element::N => {
                !(g.names[parent] == "NZ"
                    || t.name.len() == 4 && t.name.starts_with('N') && g.names[parent] == "N")
            }
            _ => false,
        };
        let length = match p.element {
            Element::C => 1.09,
            Element::N => 1.01,
            Element::O => 0.96,
            Element::S => 1.34,
            _ => return Err(error("unsupported hydrogen parent element")),
        };
        let mut alternatives: Vec<Vec<DVec3>> = Vec::new();
        match dirs.len() {
            0 if p.element == Element::O && hs.len() == 2 => {
                for axis in [
                    DVec3::X,
                    -DVec3::X,
                    DVec3::Y,
                    -DVec3::Y,
                    DVec3::Z,
                    -DVec3::Z,
                ] {
                    let u = perpendicular(axis);
                    let v = axis.cross(u);
                    for step in 0..12 {
                        let theta = step as f64 * std::f64::consts::TAU / 12.;
                        let sideways = u * theta.cos() + v * theta.sin();
                        let half = 104.52_f64.to_radians() / 2.;
                        alternatives.push(vec![
                            axis * half.cos() + sideways * half.sin(),
                            axis * half.cos() - sideways * half.sin(),
                        ]);
                    }
                }
            }
            1 => {
                let axis = dirs[0];
                let mut u = perpendicular(axis);
                if let Some(&neighbor) = neighbors.first()
                    && let Some(&anchor) = g.bonds[neighbor]
                        .iter()
                        .find(|&&i| i != parent && mol.atoms[i].element != Element::H)
                {
                    let q = mol.atoms[anchor].position.as_dvec3()
                        - mol.atoms[neighbor].position.as_dvec3();
                    let projected = q - axis * q.dot(axis);
                    if projected.length() > 1e-6 {
                        u = projected.normalize();
                    }
                }
                let v = axis.cross(u);
                let cos = if planar {
                    -0.5
                } else if p.element == Element::O {
                    104.52_f64.to_radians().cos()
                } else {
                    -1. / 3.
                };
                let sin = (1. - cos * cos).sqrt();
                let steps = if planar { 2 } else { 24 };
                for step in 0..steps {
                    let phase = std::f64::consts::TAU * step as f64 / steps as f64;
                    let mut positions = Vec::new();
                    for k in 0..hs.len() {
                        let offset = if planar && hs.len() == 2 {
                            std::f64::consts::PI * k as f64
                        } else {
                            std::f64::consts::TAU * k as f64 / 3.
                        };
                        let theta = phase + offset;
                        positions.push(axis * cos + (u * theta.cos() + v * theta.sin()) * sin);
                    }
                    alternatives.push(positions);
                }
            }
            2 if hs.len() == 2 && !planar => {
                let bisector = normalized(dirs[0] + dirs[1])?;
                let normal = normalized(dirs[0].cross(dirs[1]))?;
                let c = (-1. / 3. / dirs[0].dot(bisector)).clamp(-0.95, 0.95);
                let s = (1. - c * c).sqrt();
                alternatives.push(vec![bisector * c + normal * s, bisector * c - normal * s]);
            }
            2 | 3 if hs.len() == 1 => {
                alternatives.push(vec![normalized(-dirs.iter().copied().sum::<DVec3>())?]);
            }
            _ => return Err(error("unsupported hydrogen valence geometry")),
        }
        // This optimizes only steric placement of ideal H geometries. It is not
        // AMOEBA minimization, a pKa correction, or an H-bond existence test.
        let excluded: BTreeSet<_> = std::iter::once(parent)
            .chain(neighbors.iter().copied())
            .collect();
        let nearby: Vec<_> = mol
            .atoms
            .iter()
            .enumerate()
            .filter(|(i, a)| {
                a.element != Element::H
                    && !excluded.contains(i)
                    && a.position.as_dvec3().distance(pos) < 4.
            })
            .map(|(_, a)| a.position.as_dvec3())
            .chain(
                result
                    .iter()
                    .filter(|(_, p)| *p != parent)
                    .map(|(a, _)| a.position.as_dvec3()),
            )
            .collect();
        let score = |directions: &Vec<DVec3>| {
            directions
                .iter()
                .map(|&d| {
                    let h = pos + length * d;
                    nearby
                        .iter()
                        .map(|q| {
                            let penetration = (1.7 - h.distance(*q)).max(0.);
                            penetration * penetration
                        })
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
        let best = alternatives
            .into_iter()
            .min_by(|a, b| score(a).total_cmp(&score(b)))
            .ok_or_else(|| error("no hydrogen geometry"))?;
        for (&k, d) in hs.iter().zip(best) {
            let mut h = p.clone();
            h.element = Element::H;
            h.name = t.atoms[k].name.clone();
            h.residue_name = t.name.clone();
            h.position = (pos + length * d).as_vec3();
            h.occupancy = 1.;
            result.push((h, parent));
        }
    }
    Ok(result)
}
