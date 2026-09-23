//! Agreement of Astra's DSSP with reference three-state assignments.

use std::{io::Read, path::PathBuf};

use astra::molecule::{
    MoleculeHierarchy,
    dssp::{self, DsspCode},
    parse_pdb,
};

fn three_state(code: DsspCode) -> char {
    match code {
        DsspCode::AlphaHelix | DsspCode::Helix310 | DsspCode::PiHelix => 'H',
        DsspCode::Strand | DsspCode::Bridge => 'E',
        DsspCode::Turn | DsspCode::Loop => '-',
    }
}

#[test]
fn dssp_agrees_with_reference_assignments() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dssp");
    let (mut total, mut agree) = (0, 0);
    for name in ["1ahsA", "2cviA", "2xcjA", "3aqgA"] {
        let mut text = String::new();
        flate2::read::GzDecoder::new(
            std::fs::File::open(directory.join(format!("{name}.pdb.gz"))).unwrap(),
        )
        .read_to_string(&mut text)
        .unwrap();
        let reference: Vec<char> =
            std::fs::read_to_string(directory.join(format!("{name}.pdb.dssp")))
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .chars()
                .collect();
        let molecule = parse_pdb(&text).unwrap();
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let computed: Vec<char> = dssp::assign(&molecule, &hierarchy)
            .into_iter()
            .flatten()
            .map(|code| three_state(code.expect("complete backbone")))
            .collect();
        assert_eq!(computed.len(), reference.len(), "{name}");
        let same = computed
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a == b)
            .count();
        let fraction = same as f64 / reference.len() as f64;
        assert!(fraction > 0.9, "{name}: {fraction:.3}");
        total += reference.len();
        agree += same;
    }
    let overall = agree as f64 / total as f64;
    assert!(overall > 0.94, "overall agreement {overall:.3}");
}
