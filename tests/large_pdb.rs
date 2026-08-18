use std::{error::Error, fs, path::PathBuf};

use molview::molecule::{MoleculeHierarchy, parse_pdb};

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
    Ok(())
}
