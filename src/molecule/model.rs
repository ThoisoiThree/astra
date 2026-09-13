use glam::Vec3;

use super::Element;

#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    pub serial: u32,
    pub name: String,
    pub element: Element,
    pub residue_name: String,
    pub residue_number: i32,
    pub insertion_code: Option<char>,
    pub chain_id: String,
    pub position: Vec3,
    pub occupancy: f32,
    pub b_factor: f32,
    pub hetero: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
}

impl Bond {
    pub fn new(a: usize, b: usize) -> Option<Self> {
        (a != b).then(|| {
            let (a, b) = if a < b { (a, b) } else { (b, a) };
            Self { a, b }
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

impl Molecule {
    pub fn bounds(&self) -> Option<(Vec3, Vec3)> {
        let first = self.atoms.first()?.position;
        let mut minimum = first;
        let mut maximum = first;
        for atom in &self.atoms[1..] {
            minimum = minimum.min(atom.position);
            maximum = maximum.max(atom.position);
        }
        Some((minimum, maximum))
    }
}
