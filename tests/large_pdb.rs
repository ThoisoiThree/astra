use std::{error::Error, fs, path::PathBuf};

use molview::molecule::{
    MoleculeHierarchy, SecondaryStructure, assign_secondary_structure, parse_pdb,
};
use molview::selection::{evaluate, parse_selection};

#[test]
fn supplied_4r8p_builds_expected_hierarchy() -> Result<(), Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/4R8P.pdb");
    let contents = fs::read_to_string(path)?;
    let molecule = parse_pdb(&contents)?;
    let hierarchy = MoleculeHierarchy::from_molecule(&molecule);

    assert_eq!(molecule.atoms.len(), 17_134);
    assert_eq!(hierarchy.chains.len(), 14);
    assert_eq!(
        hierarchy.chains.first().map(|chain| chain.id.as_str()),
        Some("A")
    );
    assert_eq!(
        hierarchy.chains.last().map(|chain| chain.id.as_str()),
        Some("N")
    );
    assert!(
        hierarchy
            .chains
            .iter()
            .all(|chain| !chain.residues.is_empty())
    );
    let secondary = assign_secondary_structure(&molecule, &hierarchy);
    let helix_count = secondary
        .iter()
        .flatten()
        .filter(|state| **state == SecondaryStructure::Helix)
        .count();
    let strand_count = secondary
        .iter()
        .flatten()
        .filter(|state| **state == SecondaryStructure::Strand)
        .count();
    assert!(
        helix_count > 300,
        "expected protein helices, got {helix_count}; strands {strand_count}"
    );
    assert!(
        strand_count > 50,
        "expected beta strands, got {strand_count}"
    );

    let leucines = evaluate(&parse_selection("Chain A/LEU*")?, &molecule);
    assert!(leucines.count() > 0);
    assert!(leucines.indices().all(|index| {
        molecule.atoms[index].chain_id == "A"
            && molecule.atoms[index].residue_name.starts_with("LEU")
    }));
    let ranges = evaluate(&parse_selection("Chain B/[20:22, 70:71]")?, &molecule);
    assert!(ranges.indices().all(|index| {
        let atom = &molecule.atoms[index];
        atom.chain_id == "B"
            && ((20..=22).contains(&atom.residue_number)
                || (70..=71).contains(&atom.residue_number))
    }));
    let combined = evaluate(&parse_selection("../LEU* AND [20:30, 45:50]")?, &molecule);
    assert!(combined.indices().all(|index| {
        let atom = &molecule.atoms[index];
        atom.residue_name.starts_with("LEU")
            && ((20..=30).contains(&atom.residue_number)
                || (45..=50).contains(&atom.residue_number))
    }));
    Ok(())
}
