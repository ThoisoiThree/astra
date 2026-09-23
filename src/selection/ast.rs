use crate::molecule::Element;

/// Parsed selection. Distances are stored in milliångström so the tree stays `Eq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionExpr {
    All,
    None,
    Element(Element),
    AtomName(String),
    AtomNamePattern(String),
    ResidueName(String),
    ResidueNamePattern(String),
    ResidueNumber(i32),
    ResidueRange(i32, i32),
    Chain(String),
    ChainPattern(String),
    Serial(u32),
    Named(String),
    Hetatm,
    Polymer,
    Protein,
    Nucleic,
    Water,
    Ion,
    Ligand,
    Backbone,
    Sidechain,
    Hydrogen,
    /// Atoms within a distance of any atom of the operand, including the operand itself.
    Within(u32, Box<Self>),
    /// Like [`Self::Within`] but excluding the operand.
    Around(u32, Box<Self>),
    /// Complete residues that contain at least one atom of the operand.
    ByResidue(Box<Self>),
    /// Complete chains that contain at least one atom of the operand.
    ByChain(Box<Self>),
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Xor(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

/// Largest distance accepted by `within` and `around`, in ångström.
pub const MAX_SELECTION_DISTANCE: f32 = 1000.0;

pub fn distance_to_milli(distance: f32) -> u32 {
    (distance.clamp(0.0, MAX_SELECTION_DISTANCE) * 1000.0).round() as u32
}

pub fn milli_to_distance(milli: u32) -> f32 {
    milli as f32 / 1000.0
}

pub fn format_distance(milli: u32) -> String {
    let text = format!("{:.3}", milli_to_distance(milli));
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

impl SelectionExpr {
    /// Direct child expressions.
    pub fn children(&self) -> Vec<&Self> {
        match self {
            Self::Within(_, inner)
            | Self::Around(_, inner)
            | Self::ByResidue(inner)
            | Self::ByChain(inner)
            | Self::Not(inner) => vec![inner],
            Self::And(left, right) | Self::Xor(left, right) | Self::Or(left, right) => {
                vec![left, right]
            }
            _ => Vec::new(),
        }
    }

    pub fn children_mut(&mut self) -> Vec<&mut Self> {
        match self {
            Self::Within(_, inner)
            | Self::Around(_, inner)
            | Self::ByResidue(inner)
            | Self::ByChain(inner)
            | Self::Not(inner) => vec![inner],
            Self::And(left, right) | Self::Xor(left, right) | Self::Or(left, right) => {
                vec![left, right]
            }
            _ => Vec::new(),
        }
    }

    /// True when the result depends on coordinates and must be refreshed when atoms move.
    pub fn is_spatial(&self) -> bool {
        matches!(self, Self::Within(..) | Self::Around(..))
            || self.children().into_iter().any(Self::is_spatial)
    }
}
