use std::{hint::black_box, time::Instant};

use astra::{
    DisplayLevel, DisplayState,
    molecule::{Atom, Element, Molecule},
};
use glam::Vec3;

fn molecule(atom_count: usize) -> Molecule {
    let template = Atom {
        serial: 1,
        name: "CA".into(),
        element: Element::C,
        residue_name: "GLY".into(),
        residue_number: 1,
        insertion_code: None,
        chain_id: "A".into(),
        position: Vec3::ZERO,
        occupancy: 1.0,
        b_factor: 0.0,
        hetero: false,
    };
    Molecule {
        atoms: vec![template; atom_count],
        bonds: Vec::new(),
    }
}

fn measure(atom_count: usize) {
    let molecule = molecule(atom_count);
    let started = Instant::now();
    let mut display = DisplayState::for_molecule(&molecule);
    display.set_color_override(&[0], DisplayLevel::Chain, Some([0.2, 0.4, 0.8, 1.0]));
    let build_time = started.elapsed();
    let edit_bytes = display.estimated_edit_state_bytes();
    black_box(&display);
    println!(
        "{atom_count:>8} atoms: editable state ~{:.2} MiB, build {:?}",
        edit_bytes as f64 / (1024.0 * 1024.0),
        build_time
    );
}

fn main() {
    measure(100_000);
    measure(1_000_000);
}
