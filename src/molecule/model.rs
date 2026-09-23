use glam::Vec3;

use super::{Element, StructureInfo};

/// One atom of the molecular topology together with its coordinates in the active frame.
///
/// Everything except `position` is topology: it is fixed for the lifetime of a document.
/// Trajectory playback replaces only positions (see [`super::trajectory`]).
#[derive(Debug, Clone, PartialEq, Default)]
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
    /// Alternate-location label of the conformer that was kept, if the site had several.
    pub alt_loc: Option<char>,
    pub formal_charge: i8,
}

/// Chemical bond order. Aromatic bonds are delocalized and drawn as such.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondOrder {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondOrder {
    pub const fn code(self) -> u8 {
        match self {
            Self::Single => 1,
            Self::Double => 2,
            Self::Triple => 3,
            Self::Aromatic => 4,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Single),
            2 => Some(Self::Double),
            3 => Some(Self::Triple),
            4 => Some(Self::Aromatic),
            _ => None,
        }
    }
}

/// How a connection was established; coordination bonds are drawn differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondKind {
    #[default]
    Covalent,
    Disulfide,
    MetalCoordination,
}

impl BondKind {
    pub const fn code(self) -> u8 {
        match self {
            Self::Covalent => 0,
            Self::Disulfide => 1,
            Self::MetalCoordination => 2,
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Covalent),
            1 => Some(Self::Disulfide),
            2 => Some(Self::MetalCoordination),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
    pub order: BondOrder,
    pub kind: BondKind,
}

impl Bond {
    /// A single covalent bond with normalized endpoint order, or `None` for a self-bond.
    pub fn new(a: usize, b: usize) -> Option<Self> {
        Self::with_order(a, b, BondOrder::Single, BondKind::Covalent)
    }

    pub fn with_order(a: usize, b: usize, order: BondOrder, kind: BondKind) -> Option<Self> {
        (a != b).then(|| {
            let (a, b) = if a < b { (a, b) } else { (b, a) };
            Self { a, b, order, kind }
        })
    }

    pub const fn endpoints(&self) -> (usize, usize) {
        (self.a, self.b)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    /// File-level annotations: assemblies, crystal symmetry and deposited secondary structure.
    pub info: StructureInfo,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>, bonds: Vec<Bond>) -> Self {
        Self {
            atoms,
            bonds,
            info: StructureInfo::default(),
        }
    }

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

    pub fn positions(&self) -> Vec<Vec3> {
        self.atoms.iter().map(|atom| atom.position).collect()
    }

    /// Replaces the active coordinates. Returns `false` without changes when the frame does
    /// not have one position per atom.
    pub fn set_positions(&mut self, positions: &[Vec3]) -> bool {
        if positions.len() != self.atoms.len() {
            return false;
        }
        for (atom, position) in self.atoms.iter_mut().zip(positions) {
            atom.position = *position;
        }
        true
    }
}
