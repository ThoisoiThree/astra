//! Measures structure parsing and connectivity on a file: `cargo run --release --example
//! parse_timing -- path/to/structure`.

use std::time::Instant;

use astra::molecule::{BondOrder, MoleculeHierarchy, assign_secondary_structure, parse_structure};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/4R8P.pdb".into());
    let bytes = std::fs::read(&path)?;
    let start = Instant::now();
    let parsed = parse_structure(&bytes, &path)?;
    let parsed_at = start.elapsed();
    let hierarchy = MoleculeHierarchy::from_molecule(&parsed.molecule);
    let secondary = assign_secondary_structure(&parsed.molecule, &hierarchy);
    let molecule = &parsed.molecule;
    println!(
        "{path}: {} atoms, {} bonds ({} multiple/aromatic), {} extra frames, {} chains",
        molecule.atoms.len(),
        molecule.bonds.len(),
        molecule
            .bonds
            .iter()
            .filter(|bond| bond.order != BondOrder::Single)
            .count(),
        parsed.frames.len(),
        secondary.len()
    );
    println!(
        "parse + bonds {:.1} ms, total with DSSP {:.1} ms",
        parsed_at.as_secs_f64() * 1e3,
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
