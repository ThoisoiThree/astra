use std::collections::BTreeMap;

use astra::{
    DisplayState,
    camera::OrbitCamera,
    molecule::parse_structure,
    scene::{SceneDocument, decode, encode},
};

#[test]
fn supplied_4r8p_round_trips_into_a_smaller_self_contained_scene() {
    let source = include_bytes!("../examples/4R8P.pdb");
    let molecule = parse_structure(source, "4R8P.pdb").unwrap().molecule;
    let display = DisplayState::for_molecule(&molecule);
    let document = SceneDocument {
        source_name: "4R8P.pdb".into(),
        molecule: molecule.clone(),
        display: display.clone(),
        named_selections: BTreeMap::new(),
        named_selection_expressions: BTreeMap::new(),
        named_selection_styles: BTreeMap::new(),
        measurement_lines: Vec::new(),
        hierarchy_names: BTreeMap::new(),
        inspection: None,
        hierarchy_selection: Vec::new(),
        hierarchy_selection_anchor: None,
        focus_description: "Molecule center".into(),
        pivot_description: "Molecule center".into(),
        camera: OrbitCamera::new(16.0 / 9.0),
    };

    let encoded = encode(&document).unwrap();
    assert!(encoded.len() < source.len() / 2);

    let restored = decode(&encoded).unwrap();
    assert_eq!(restored.molecule.atoms, molecule.atoms);
    assert_eq!(restored.molecule.bonds, molecule.bonds);
    assert_eq!(restored.display, display);
}
