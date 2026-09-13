//! Template data and the prepared-parameter import boundary. Units: nm, e, kJ/mol.
use super::AmoebaError;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Clone, Deserialize)]
pub struct AtomType {
    pub class: usize,
    pub element: String,
}
#[derive(Clone, Deserialize)]
pub struct TemplateAtom {
    pub name: String,
    pub r#type: usize,
}
#[derive(Clone, Deserialize)]
pub struct Template {
    pub name: String,
    pub atoms: Vec<TemplateAtom>,
    pub bonds: Vec<[usize; 2]>,
    pub external: Vec<usize>,
}
#[derive(Clone, Deserialize)]
pub struct Multipole {
    pub r#type: usize,
    pub axes: [i32; 3],
    pub charge: f64,
    pub dipole: [f64; 3],
    pub quadrupole: [f64; 9],
}
#[derive(Clone, Deserialize)]
pub struct Polar {
    pub alpha: f64,
    pub thole: f64,
    pub groups: Vec<usize>,
}
#[derive(Clone, Deserialize)]
pub struct Vdw {
    pub radius: f64,
    pub epsilon: f64,
    pub reduction: f64,
}
#[derive(Clone, Deserialize)]
pub struct Pair {
    pub classes: [usize; 2],
    pub radius: f64,
    pub epsilon: f64,
}
/// Complete template provider. A future Poltype adapter can produce this data
/// format; parameter assignment and energy evaluation share no UI dependencies.
#[derive(Clone, Deserialize)]
pub struct ForceField {
    pub version: u32,
    pub types: BTreeMap<usize, AtomType>,
    pub residues: Vec<Template>,
    pub multipoles: Vec<Multipole>,
    pub polar: BTreeMap<usize, Polar>,
    pub vdw: BTreeMap<usize, Vdw>,
    pub pairs: Vec<Pair>,
    pub standard_bonds: BTreeMap<String, Vec<[String; 2]>>,
    pub residue_aliases: BTreeMap<String, String>,
    pub atom_aliases: BTreeMap<String, BTreeMap<String, String>>,
}
impl ForceField {
    pub fn builtin() -> Result<Self, AmoebaError> {
        Self::from_json(include_str!("../../../data/amoeba/amoeba2018.json"))
    }
    pub fn from_json(json: &str) -> Result<Self, AmoebaError> {
        let mut ff: Self = serde_json::from_str(json)?;
        for i in 0..ff.residues.len() {
            let canonical = ff.canonical(&ff.residues[i].name);
            if let Some(aliases) = ff.atom_aliases.get(&canonical) {
                for atom in &mut ff.residues[i].atoms {
                    if let Some(name) = aliases.get(&atom.name) {
                        atom.name = name.clone();
                    }
                }
            }
        }
        let invalid = || AmoebaError::Invalid("invalid AMOEBA parameter provider".into());
        if ff.version != 1 || ff.types.is_empty() {
            return Err(invalid());
        }
        for (id, t) in &ff.types {
            let p = ff.polar.get(id).ok_or_else(invalid)?;
            let v = ff.vdw.get(&t.class).ok_or_else(invalid)?;
            if ![p.alpha, p.thole, v.radius, v.epsilon, v.reduction]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.)
                || p.groups.iter().any(|g| !ff.types.contains_key(g))
                || !ff.multipoles.iter().any(|m| m.r#type == *id)
            {
                return Err(invalid());
            }
        }
        for r in &ff.residues {
            if r.atoms.is_empty()
                || r.atoms.iter().any(|a| !ff.types.contains_key(&a.r#type))
                || r.bonds
                    .iter()
                    .flatten()
                    .chain(&r.external)
                    .any(|&i| i >= r.atoms.len())
            {
                return Err(invalid());
            }
        }
        for m in &ff.multipoles {
            if !m
                .dipole
                .iter()
                .chain(&m.quadrupole)
                .chain([&m.charge])
                .all(|v| v.is_finite())
                || !ff.types.contains_key(&m.r#type)
                || m.axes.iter().any(|a| {
                    *a == i32::MIN
                        || (*a != 0 && !ff.types.contains_key(&(a.unsigned_abs() as usize)))
                })
            {
                return Err(invalid());
            }
        }
        for pair in &ff.pairs {
            if pair.classes.iter().any(|c| !ff.vdw.contains_key(c))
                || ![pair.radius, pair.epsilon]
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.)
            {
                return Err(invalid());
            }
        }
        Ok(ff)
    }
    pub fn canonical(&self, name: &str) -> String {
        let name = self
            .residue_aliases
            .get(name)
            .map(String::as_str)
            .unwrap_or(name);
        if !name.is_ascii() {
            return name.into();
        }
        let name = match name {
            "CYD" => "CYS",
            "LYD" => "LYS",
            "TYD" => "TYR",
            _ => name,
        };
        if name.len() == 4
            && (name.starts_with('N') || name.starts_with('C'))
            && self.residues.iter().any(|r| r.name == name)
        {
            return self.canonical(&name[1..]);
        }
        if name.starts_with('R')
            && name.len() >= 2
            && matches!(&name[1..2], "A" | "C" | "G" | "U")
            && matches!(&name[2..], "" | "N" | "3" | "5")
        {
            return name[1..2].into();
        }
        if name.starts_with('D') && name.len() == 3 && matches!(&name[2..], "N" | "3" | "5") {
            return name[..2].into();
        }
        name.into()
    }
    pub fn family(&self, name: &str) -> String {
        let n = self.canonical(name);
        match n.as_str() {
            "HID" | "HIE" | "HIP" => "HIS".into(),
            "ASH" => "ASP".into(),
            "GLH" => "GLU".into(),
            "CYX" => "CYS".into(),
            _ => n,
        }
    }
}
