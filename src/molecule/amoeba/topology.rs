use super::{
    AmoebaError, ResidueParameterization,
    parameters::{ForceField, Template},
};
use crate::molecule::Molecule;
use glam::DVec3;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
};

pub(super) fn check_cancel(cancel: &AtomicBool) -> Result<(), AmoebaError> {
    if cancel.load(Ordering::Relaxed) {
        Err(AmoebaError::Cancelled)
    } else {
        Ok(())
    }
}
pub(super) struct Topology {
    pub original: Vec<usize>,
    pub xyz: Vec<DVec3>,
    pub names: Vec<String>,
    pub elements: Vec<String>,
    pub templates: Vec<String>,
    pub types: Vec<usize>,
    pub bonds: Vec<Vec<usize>>,
    pub shells: Vec<[BTreeSet<usize>; 4]>,
    pub groups: Vec<usize>,
}
pub(super) struct Residue {
    pub indices: Vec<usize>,
    pub canonical: String,
    pub label: String,
    pub chain: String,
}

// Match the complete element/connectivity graph within a named chemical family.
// Atom names guide symmetric choices, but numbered H aliases need not be identical.
fn match_template(
    r: &Residue,
    t: &Template,
    names: &[String],
    mol: &Molecule,
    bonds: &[BTreeSet<usize>],
    ff: &ForceField,
) -> Option<Vec<usize>> {
    if r.indices.len() != t.atoms.len() {
        return None;
    }
    let n = t.atoms.len();
    let mut tb = vec![BTreeSet::new(); n];
    let mut ext = vec![0; n];
    for &[a, b] in &t.bonds {
        tb[a].insert(b);
        tb[b].insert(a);
    }
    for &a in &t.external {
        ext[a] += 1;
    }
    let members: BTreeSet<_> = r.indices.iter().copied().collect();
    let mut options = Vec::new();
    for &i in &r.indices {
        let element = mol.atoms[i].element.symbol();
        let outside = bonds[i].iter().filter(|j| !members.contains(j)).count();
        let mut choices: Vec<_> = (0..n)
            .filter(|&j| {
                ff.types[&t.atoms[j].r#type]
                    .element
                    .eq_ignore_ascii_case(element)
                    && tb[j].len() + ext[j] == bonds[i].len()
                    && ext[j] == outside
                    && (n == 1 || element == "H" || t.atoms[j].name == names[i])
            })
            .collect();
        choices.sort_by_key(|&j| (t.atoms[j].name != names[i], j));
        if choices.is_empty() {
            return None;
        }
        options.push(choices);
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&i| options[i].len());
    struct Search<'a> {
        order: &'a [usize],
        options: &'a [Vec<usize>],
        indices: &'a [usize],
        bonds: &'a [BTreeSet<usize>],
        tb: &'a [BTreeSet<usize>],
    }
    impl Search<'_> {
        fn run(&self, k: usize, map: &mut [usize], used: &mut [bool]) -> bool {
            if k == self.order.len() {
                return true;
            }
            let i = self.order[k];
            for &j in &self.options[i] {
                if used[j] {
                    continue;
                }
                if self.order[..k].iter().any(|&a| {
                    self.bonds[self.indices[i]].contains(&self.indices[a])
                        != self.tb[j].contains(&map[a])
                }) {
                    continue;
                }
                map[i] = j;
                used[j] = true;
                if self.run(k + 1, map, used) {
                    return true;
                }
                used[j] = false;
            }
            false
        }
    }
    let mut map = vec![0; n];
    if (Search {
        order: &order,
        options: &options,
        indices: &r.indices,
        bonds,
        tb: &tb,
    })
    .run(0, &mut map, &mut vec![false; n])
    {
        Some(map)
    } else {
        None
    }
}

pub(super) struct ChemicalGraph {
    pub residues: Vec<Residue>,
    pub atom_res: Vec<usize>,
    pub names: Vec<String>,
    pub bonds: Vec<BTreeSet<usize>>,
}
pub(super) fn chemical_graph(
    mol: &Molecule,
    ff: &ForceField,
    cancel: &AtomicBool,
) -> Result<ChemicalGraph, AmoebaError> {
    let n = mol.atoms.len();
    let mut residues: Vec<Residue> = Vec::new();
    let mut lookup = BTreeMap::new();
    let mut atom_res = vec![0; n];
    let mut names = vec![String::new(); n];
    for (i, a) in mol.atoms.iter().enumerate() {
        let key = (
            a.chain_id.clone(),
            a.residue_number,
            a.insertion_code,
            a.residue_name.clone(),
        );
        let next = residues.len();
        let ri = *lookup.entry(key).or_insert_with(|| {
            residues.push(Residue {
                indices: Vec::new(),
                canonical: ff.canonical(&a.residue_name),
                label: format!(
                    "{}:{}{}:{}",
                    a.chain_id,
                    a.residue_number,
                    a.insertion_code.map(|c| c.to_string()).unwrap_or_default(),
                    a.residue_name
                ),
                chain: a.chain_id.clone(),
            });
            next
        });
        residues[ri].indices.push(i);
        atom_res[i] = ri;
        names[i] = ff
            .atom_aliases
            .get(&residues[ri].canonical)
            .and_then(|m| m.get(&a.name))
            .unwrap_or(&a.name)
            .clone();
    }
    let mut chains: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, r) in residues.iter().enumerate() {
        chains.entry(&r.chain).or_default().push(i);
    }
    let mut bonds = vec![BTreeSet::new(); n];
    for chain in chains.values() {
        for (k, &ri) in chain.iter().enumerate() {
            check_cancel(cancel)?;
            let r = &residues[ri];
            if let Some(defs) = ff.standard_bonds.get(&r.canonical) {
                for ends in defs {
                    let resolve = |name: &str| -> Option<usize> {
                        let (res, name) = if let Some(s) = name.strip_prefix('-') {
                            (*chain.get(k.checked_sub(1)?)?, s)
                        } else if let Some(s) = name.strip_prefix('+') {
                            (*chain.get(k + 1)?, s)
                        } else {
                            (ri, name)
                        };
                        residues[res]
                            .indices
                            .iter()
                            .copied()
                            .find(|&i| names[i] == name)
                    };
                    if let (Some(a), Some(b)) = (resolve(&ends[0]), resolve(&ends[1]))
                        && a != b
                        && (atom_res[a] == atom_res[b]
                            || mol.atoms[a].position.distance(mol.atoms[b].position) <= 2.5)
                    {
                        bonds[a].insert(b);
                        bonds[b].insert(a);
                    }
                }
            }
        }
    }
    for b in &mol.bonds {
        let a = b.a;
        let j = b.b;
        let ion = |i: usize| {
            residues[atom_res[i]].indices.len() == 1
                && !matches!(
                    mol.atoms[i].element.symbol(),
                    "H" | "C" | "N" | "O" | "P" | "S"
                )
        };
        if ion(a) || ion(j) {
            continue;
        }
        let unknown = !ff
            .standard_bonds
            .contains_key(&residues[atom_res[a]].canonical)
            || !ff
                .standard_bonds
                .contains_key(&residues[atom_res[j]].canonical);
        if (unknown || (names[a] == "SG" && names[j] == "SG" && atom_res[a] != atom_res[j]))
            && (atom_res[a] == atom_res[j]
                || mol.atoms[a].position.distance(mol.atoms[j].position) <= 2.5)
        {
            bonds[a].insert(j);
            bonds[j].insert(a);
        }
    }
    Ok(ChemicalGraph {
        residues,
        atom_res,
        names,
        bonds,
    })
}

pub(super) fn prepare(
    mol: &Molecule,
    ff: &ForceField,
    cancel: &AtomicBool,
) -> Result<(Topology, Vec<ResidueParameterization>), AmoebaError> {
    let n = mol.atoms.len();
    let ChemicalGraph {
        residues,
        atom_res,
        names,
        bonds,
    } = chemical_graph(mol, ff, cancel)?;
    let mut matches = Vec::new();
    let mut excluded = vec![false; residues.len()];
    for (ri, r) in residues.iter().enumerate() {
        check_cancel(cancel)?;
        let matched = ff
            .residues
            .iter()
            .filter(|t| ff.family(&t.name) == ff.family(&r.canonical))
            .find_map(|t| match_template(r, t, &names, mol, &bonds, ff).map(|m| (t, m)));
        excluded[ri] = matched.is_none();
        matches.push(matched);
    }
    // Exclude entire covalent components: dropping a residue must not fabricate
    // a terminus or change neighboring local-frame/protonation assignments.
    let mut queue: VecDeque<_> = excluded
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| x.then_some(i))
        .collect();
    while let Some(ri) = queue.pop_front() {
        for &a in &residues[ri].indices {
            for &b in &bonds[a] {
                let rj = atom_res[b];
                if !excluded[rj] {
                    excluded[rj] = true;
                    queue.push_back(rj);
                }
            }
        }
    }
    let mut statuses = Vec::new();
    let mut assigned = vec![0; n];
    let mut templates = vec![String::new(); n];
    for (ri, r) in residues.iter().enumerate() {
        let reason = if excluded[ri] {
            if matches[ri].is_some() {
                "Covalently connected to an unparameterized residue; boundary not capped".into()
            } else {
                let family: Vec<_> = ff
                    .residues
                    .iter()
                    .filter(|t| ff.family(&t.name) == ff.family(&r.canonical))
                    .collect();
                if family.is_empty() {
                    "No AMOEBA 2018 template for this residue; import parameters required".into()
                } else {
                    let missing = family
                        .iter()
                        .map(|t| {
                            t.atoms
                                .iter()
                                .filter(|a| {
                                    ff.types[&a.r#type].element != "H"
                                        && !r.indices.iter().any(|&i| names[i] == a.name)
                                })
                                .map(|a| a.name.clone())
                                .collect::<Vec<_>>()
                        })
                        .min_by_key(Vec::len)
                        .unwrap_or_default();
                    if !missing.is_empty() {
                        format!(
                            "Missing heavy atoms: {}; explicit H and a complete template are required",
                            missing.join(", ")
                        )
                    } else {
                        "No exact template: missing explicit H, unresolved protonation, or incompatible covalent connectivity".into()
                    }
                }
            }
        } else if let Some((t, m)) = &matches[ri] {
            for (k, &i) in r.indices.iter().enumerate() {
                assigned[i] = t.atoms[m[k]].r#type;
                templates[i] = t.name.clone();
            }
            t.name.clone()
        } else {
            return Err(AmoebaError::Invalid(
                "inconsistent template assignment".into(),
            ));
        };
        statuses.push(ResidueParameterization {
            residue: r.label.clone(),
            atoms: r.indices.clone(),
            status: if excluded[ri] {
                "unparameterized"
            } else {
                "parameterized"
            }
            .into(),
            reason,
        });
    }
    let original: Vec<_> = (0..n).filter(|&i| !excluded[atom_res[i]]).collect();
    let mut remap = vec![usize::MAX; n];
    for (i, &j) in original.iter().enumerate() {
        remap[j] = i;
    }
    let edges: Vec<Vec<usize>> = original
        .iter()
        .map(|&i| {
            bonds[i]
                .iter()
                .filter_map(|&j| (remap[j] != usize::MAX).then_some(remap[j]))
                .collect()
        })
        .collect();
    let types: Vec<_> = original.iter().map(|&i| assigned[i]).collect();
    let count = original.len();
    let mut shells = Vec::new();
    for i in 0..count {
        let mut seen = BTreeSet::from([i]);
        let mut front = BTreeSet::from([i]);
        let mut ss: [BTreeSet<usize>; 4] = Default::default();
        for shell in &mut ss {
            let next: BTreeSet<_> = front
                .iter()
                .flat_map(|&j| edges[j].iter().copied())
                .filter(|j| !seen.contains(j))
                .collect();
            seen.extend(&next);
            *shell = next.clone();
            front = next;
        }
        shells.push(ss);
    }
    let mut pg = vec![Vec::new(); count];
    for i in 0..count {
        for &j in &edges[i] {
            if ff.polar[&types[i]].groups.contains(&types[j]) {
                pg[i].push(j);
                pg[j].push(i);
            }
        }
    }
    let mut groups = vec![usize::MAX; count];
    for i in 0..count {
        if groups[i] != usize::MAX {
            continue;
        }
        groups[i] = i;
        let mut q = vec![i];
        while let Some(j) = q.pop() {
            for &k in &pg[j] {
                if groups[k] == usize::MAX {
                    groups[k] = i;
                    q.push(k);
                }
            }
        }
    }
    Ok((
        Topology {
            xyz: original
                .iter()
                .map(|&i| mol.atoms[i].position.as_dvec3() * 0.1)
                .collect(),
            names: original.iter().map(|&i| names[i].clone()).collect(),
            elements: original
                .iter()
                .map(|&i| mol.atoms[i].element.symbol().into())
                .collect(),
            templates: original.iter().map(|&i| templates[i].clone()).collect(),
            original,
            types,
            bonds: edges,
            shells,
            groups,
        },
        statuses,
    ))
}
