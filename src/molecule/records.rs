//! Format-independent assembly of parsed atom records into a molecule and its model frames.

use std::collections::HashMap;

use glam::Vec3;

use super::{
    Atom, BondKind, BondOrder, Molecule, StructureInfo,
    bonds::{ExplicitBond, build_bonds},
};

/// One atom site as read from a file, before alternate locations are resolved.
#[derive(Debug, Clone)]
pub(crate) struct AtomRecord {
    pub atom: Atom,
    /// Model identifier as written in the file; models are ordered by first appearance.
    pub model: i64,
}

/// Identifies an atom by its residue and name, as LINK, SSBOND and `struct_conn` do.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SiteKey {
    pub chain: String,
    pub residue_number: i32,
    pub insertion_code: Option<char>,
    pub atom_name: String,
}

#[derive(Debug, Clone)]
pub(crate) enum BondRecord {
    /// CONECT by serial numbers; `count` repeats encode the bond order in some writers.
    Serial { a: u32, b: u32, count: u8 },
    Site {
        a: SiteKey,
        b: SiteKey,
        kind: BondKind,
        order: Option<BondOrder>,
    },
}

pub(crate) struct Assembled {
    pub molecule: Molecule,
    /// Coordinates of models after the first, in topology atom order.
    pub frames: Vec<Vec<Vec3>>,
}

/// Resolves alternate locations, splits models into topology plus frames and builds bonds.
pub(crate) fn assemble(
    records: Vec<AtomRecord>,
    bond_records: &[BondRecord],
    mut info: StructureInfo,
) -> Option<Assembled> {
    let mut model_order = Vec::<i64>::new();
    let mut by_model = HashMap::<i64, Vec<Atom>>::new();
    for record in records {
        if !by_model.contains_key(&record.model) {
            model_order.push(record.model);
        }
        by_model.entry(record.model).or_default().push(record.atom);
    }
    let first = *model_order.first()?;
    let (atoms, serials) = select_alternate_locations(by_model.remove(&first)?);
    if atoms.is_empty() {
        return None;
    }
    info.model_count = model_order.len();

    let mut frames = Vec::new();
    let mut skipped = Vec::new();
    for model in &model_order[1..] {
        let Some(model_atoms) = by_model.remove(model) else {
            continue;
        };
        match frame_for_model(&atoms, model_atoms) {
            Some(frame) => frames.push(frame),
            None => skipped.push(model.to_string()),
        }
    }
    if !skipped.is_empty() {
        info.notes.push(format!(
            "{} model(s) with a different atom inventory were not loaded as frames ({})",
            skipped.len(),
            abbreviate(&skipped)
        ));
    }

    let explicit = resolve_bonds(&atoms, &serials, bond_records);
    let bonds = build_bonds(&atoms, &explicit);
    Some(Assembled {
        molecule: Molecule { atoms, bonds, info },
        frames,
    })
}

fn abbreviate(values: &[String]) -> String {
    if values.len() <= 6 {
        values.join(", ")
    } else {
        format!("{}, … {}", values[..3].join(", "), values[values.len() - 1])
    }
}

type ResidueKey = (String, i32, Option<char>);

/// Keeps atoms without an alternate location plus, per residue, the conformer with the
/// highest mean occupancy (the first label wins ties). Returns the kept atoms and their
/// original serial numbers.
pub(crate) fn select_alternate_locations(atoms: Vec<Atom>) -> (Vec<Atom>, Vec<u32>) {
    let mut totals = HashMap::<ResidueKey, Vec<(char, f32, usize)>>::new();
    for atom in &atoms {
        let Some(label) = atom.alt_loc else {
            continue;
        };
        let entries = totals
            .entry((
                atom.chain_id.clone(),
                atom.residue_number,
                atom.insertion_code,
            ))
            .or_default();
        match entries
            .iter_mut()
            .find(|(candidate, _, _)| *candidate == label)
        {
            Some((_, sum, count)) => {
                *sum += atom.occupancy;
                *count += 1;
            }
            None => entries.push((label, atom.occupancy, 1)),
        }
    }
    let chosen: HashMap<ResidueKey, char> = totals
        .into_iter()
        .filter_map(|(key, entries)| {
            let mut best: Option<(char, f32)> = None;
            for (label, sum, count) in entries {
                let mean = sum / count.max(1) as f32;
                if best.is_none_or(|(_, value)| mean > value + 1e-4) {
                    best = Some((label, mean));
                }
            }
            best.map(|(label, _)| (key, label))
        })
        .collect();
    let mut kept = Vec::with_capacity(atoms.len());
    let mut serials = Vec::with_capacity(atoms.len());
    let mut seen_names = HashMap::<(ResidueKey, String), ()>::new();
    for atom in atoms {
        let key = (
            atom.chain_id.clone(),
            atom.residue_number,
            atom.insertion_code,
        );
        if let Some(label) = atom.alt_loc
            && chosen.get(&key) != Some(&label)
        {
            continue;
        }
        // A site listed both without a label and with the chosen label keeps one copy.
        if atom.alt_loc.is_some() && seen_names.insert((key, atom.name.clone()), ()).is_some() {
            continue;
        }
        serials.push(atom.serial);
        kept.push(atom);
    }
    (kept, serials)
}

fn atom_key(atom: &Atom) -> (String, i32, Option<char>, String, String) {
    (
        atom.chain_id.clone(),
        atom.residue_number,
        atom.insertion_code,
        atom.residue_name.clone(),
        atom.name.clone(),
    )
}

/// Matches another model's atoms to the topology by chain, residue and atom name, using the
/// same conformer label where one exists.
fn frame_for_model(topology: &[Atom], atoms: Vec<Atom>) -> Option<Vec<Vec3>> {
    let mut exact = HashMap::new();
    let mut any = HashMap::new();
    for atom in &atoms {
        exact.insert((atom_key(atom), atom.alt_loc), atom.position);
        any.entry(atom_key(atom)).or_insert(atom.position);
    }
    topology
        .iter()
        .map(|atom| {
            let key = atom_key(atom);
            exact
                .get(&(key.clone(), atom.alt_loc))
                .or_else(|| any.get(&key))
                .copied()
        })
        .collect()
}

fn resolve_bonds(atoms: &[Atom], serials: &[u32], records: &[BondRecord]) -> Vec<ExplicitBond> {
    let serial_index: HashMap<u32, usize> = serials
        .iter()
        .enumerate()
        .map(|(index, serial)| (*serial, index))
        .collect();
    let mut site_index = HashMap::<SiteKey, usize>::new();
    for (index, atom) in atoms.iter().enumerate() {
        site_index
            .entry(SiteKey {
                chain: atom.chain_id.clone(),
                residue_number: atom.residue_number,
                insertion_code: atom.insertion_code,
                atom_name: atom.name.clone(),
            })
            .or_insert(index);
    }
    let mut result = Vec::new();
    for record in records {
        match record {
            BondRecord::Serial { a, b, count } => {
                if let (Some(&a), Some(&b)) = (serial_index.get(a), serial_index.get(b)) {
                    let order = match count {
                        2 => Some(BondOrder::Double),
                        3.. => Some(BondOrder::Triple),
                        _ => None,
                    };
                    result.push(ExplicitBond {
                        a,
                        b,
                        kind: BondKind::Covalent,
                        order,
                    });
                }
            }
            BondRecord::Site { a, b, kind, order } => {
                if let (Some(&a), Some(&b)) = (site_index.get(a), site_index.get(b)) {
                    result.push(ExplicitBond {
                        a,
                        b,
                        kind: *kind,
                        order: *order,
                    });
                }
            }
        }
    }
    result
}

/// Collapses CONECT pairs, counting repeated listings of the same bond from one atom.
pub(crate) fn serial_bond_records(pairs: &[(u32, u32)]) -> Vec<BondRecord> {
    let mut counts = HashMap::<(u32, u32), u8>::new();
    let mut order = Vec::new();
    for &(source, target) in pairs {
        if source == target {
            continue;
        }
        let count = counts.entry((source, target)).or_insert_with(|| {
            order.push((source, target));
            0
        });
        *count = count.saturating_add(1);
    }
    let mut seen = HashMap::<(u32, u32), ()>::new();
    let mut records = Vec::new();
    for (source, target) in order {
        let key = (source.min(target), source.max(target));
        if seen.insert(key, ()).is_some() {
            continue;
        }
        let forward = counts.get(&(source, target)).copied().unwrap_or(1);
        let backward = counts.get(&(target, source)).copied().unwrap_or(0);
        records.push(BondRecord::Serial {
            a: key.0,
            b: key.1,
            count: forward.max(backward),
        });
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Element;

    fn atom(name: &str, residue: i32, alt: Option<char>, occupancy: f32, x: f32) -> Atom {
        Atom {
            serial: residue as u32 * 10 + x as u32,
            name: name.into(),
            element: Element::C,
            residue_name: "SER".into(),
            residue_number: residue,
            chain_id: "A".into(),
            position: Vec3::new(x, 0.0, 0.0),
            occupancy,
            alt_loc: alt,
            ..Atom::default()
        }
    }

    #[test]
    fn keeps_the_most_occupied_conformer_per_residue() {
        let atoms = vec![
            atom("CA", 1, None, 1.0, 0.0),
            atom("CB", 1, Some('A'), 0.3, 1.0),
            atom("OG", 1, Some('A'), 0.3, 2.0),
            atom("CB", 1, Some('B'), 0.7, 3.0),
            atom("OG", 1, Some('B'), 0.7, 4.0),
            atom("CB", 2, Some('B'), 0.5, 5.0),
            atom("CB", 2, Some('C'), 0.5, 6.0),
        ];
        let (kept, _) = select_alternate_locations(atoms);
        let labels: Vec<_> = kept
            .iter()
            .map(|atom| (atom.residue_number, atom.alt_loc))
            .collect();
        assert_eq!(
            labels,
            vec![(1, None), (1, Some('B')), (1, Some('B')), (2, Some('B'))]
        );
    }

    #[test]
    fn later_models_become_frames_and_mismatched_models_are_reported() {
        let mut records = Vec::new();
        for model in 1..=3 {
            records.push(AtomRecord {
                atom: atom("CA", 1, None, 1.0, model as f32),
                model,
            });
            if model != 3 {
                records.push(AtomRecord {
                    atom: atom("CB", 1, None, 1.0, 10.0 + model as f32),
                    model,
                });
            }
        }
        let assembled = assemble(records, &[], StructureInfo::default()).unwrap();
        assert_eq!(assembled.molecule.atoms.len(), 2);
        assert_eq!(assembled.frames.len(), 1);
        assert_eq!(assembled.frames[0][1], Vec3::new(12.0, 0.0, 0.0));
        assert_eq!(assembled.molecule.info.model_count, 3);
        assert_eq!(assembled.molecule.info.notes.len(), 1);
    }

    #[test]
    fn repeated_conect_pairs_encode_bond_order() {
        let records = serial_bond_records(&[(1, 2), (1, 2), (2, 1), (2, 1), (1, 3), (3, 1)]);
        assert!(matches!(
            records[0],
            BondRecord::Serial {
                a: 1,
                b: 2,
                count: 2
            }
        ));
        assert!(matches!(
            records[1],
            BondRecord::Serial {
                a: 1,
                b: 3,
                count: 1
            }
        ));
    }
}
