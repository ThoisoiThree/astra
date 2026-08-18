use std::{fmt, str::FromStr};

/// Chemical element subset needed by common biomolecular PDB files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    H,
    C,
    N,
    O,
    P,
    S,
    F,
    Cl,
    Br,
    I,
    Na,
    Mg,
    K,
    Ca,
    Fe,
    Zn,
    Unknown,
}

impl Element {
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::H => "H",
            Self::C => "C",
            Self::N => "N",
            Self::O => "O",
            Self::P => "P",
            Self::S => "S",
            Self::F => "F",
            Self::Cl => "Cl",
            Self::Br => "Br",
            Self::I => "I",
            Self::Na => "Na",
            Self::Mg => "Mg",
            Self::K => "K",
            Self::Ca => "Ca",
            Self::Fe => "Fe",
            Self::Zn => "Zn",
            Self::Unknown => "?",
        }
    }

    pub const fn cpk_color(self) -> [f32; 4] {
        let rgb = match self {
            Self::H => [0.92, 0.92, 0.92],
            Self::C => [0.32, 0.34, 0.38],
            Self::N => [0.18, 0.31, 0.97],
            Self::O => [0.95, 0.12, 0.10],
            Self::P => [1.00, 0.50, 0.05],
            Self::S => [0.95, 0.82, 0.13],
            Self::F | Self::Cl => [0.12, 0.78, 0.20],
            Self::Br => [0.60, 0.13, 0.06],
            Self::I => [0.45, 0.10, 0.65],
            Self::Na => [0.45, 0.40, 0.95],
            Self::Mg => [0.13, 0.62, 0.13],
            Self::K => [0.56, 0.25, 0.83],
            Self::Ca => [0.25, 0.95, 0.25],
            Self::Fe => [0.88, 0.40, 0.20],
            Self::Zn => [0.49, 0.50, 0.69],
            Self::Unknown => [0.75, 0.45, 0.75],
        };
        [rgb[0], rgb[1], rgb[2], 1.0]
    }

    pub const fn covalent_radius(self) -> f32 {
        match self {
            Self::H => 0.31,
            Self::C => 0.76,
            Self::N => 0.71,
            Self::O => 0.66,
            Self::P => 1.07,
            Self::S => 1.05,
            Self::F => 0.57,
            Self::Cl => 1.02,
            Self::Br => 1.20,
            Self::I => 1.39,
            Self::Na => 1.66,
            Self::Mg => 1.41,
            Self::K => 2.03,
            Self::Ca => 1.76,
            Self::Fe => 1.32,
            Self::Zn => 1.22,
            Self::Unknown => 0.77,
        }
    }

    pub const fn van_der_waals_radius(self) -> f32 {
        match self {
            Self::H => 1.20,
            Self::C => 1.70,
            Self::N => 1.55,
            Self::O => 1.52,
            Self::P => 1.80,
            Self::S => 1.80,
            Self::F => 1.47,
            Self::Cl => 1.75,
            Self::Br => 1.85,
            Self::I => 1.98,
            Self::Na => 2.27,
            Self::Mg => 1.73,
            Self::K => 2.75,
            Self::Ca => 2.31,
            Self::Fe => 2.00,
            Self::Zn => 2.10,
            Self::Unknown => 1.70,
        }
    }
}

impl FromStr for Element {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let element = match value.trim().to_ascii_uppercase().as_str() {
            "H" => Self::H,
            "C" => Self::C,
            "N" => Self::N,
            "O" => Self::O,
            "P" => Self::P,
            "S" => Self::S,
            "F" => Self::F,
            "CL" => Self::Cl,
            "BR" => Self::Br,
            "I" => Self::I,
            "NA" => Self::Na,
            "MG" => Self::Mg,
            "K" => Self::K,
            "CA" => Self::Ca,
            "FE" => Self::Fe,
            "ZN" => Self::Zn,
            _ => return Err(()),
        };
        Ok(element)
    }
}

impl fmt::Display for Element {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}
