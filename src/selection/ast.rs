use crate::molecule::Element;

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
    Not(Box<Self>),
    And(Box<Self>, Box<Self>),
    Xor(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}
