use std::{fmt, str::FromStr};

/// Chemical elements of the periodic table (Z = 1…118) plus an unknown placeholder.
///
/// Covalent radii follow Cordero et al. (2008), van der Waals radii Bondi (1964) with the
/// Mantina et al. (2009) main-group additions (2.0 Å where no value is tabulated), and
/// colors the Jmol scheme with Astra's adjustments for common biomolecular elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Element {
    #[default]
    Unknown = 0,
    H = 1,
    He = 2,
    Li = 3,
    Be = 4,
    B = 5,
    C = 6,
    N = 7,
    O = 8,
    F = 9,
    Ne = 10,
    Na = 11,
    Mg = 12,
    Al = 13,
    Si = 14,
    P = 15,
    S = 16,
    Cl = 17,
    Ar = 18,
    K = 19,
    Ca = 20,
    Sc = 21,
    Ti = 22,
    V = 23,
    Cr = 24,
    Mn = 25,
    Fe = 26,
    Co = 27,
    Ni = 28,
    Cu = 29,
    Zn = 30,
    Ga = 31,
    Ge = 32,
    As = 33,
    Se = 34,
    Br = 35,
    Kr = 36,
    Rb = 37,
    Sr = 38,
    Y = 39,
    Zr = 40,
    Nb = 41,
    Mo = 42,
    Tc = 43,
    Ru = 44,
    Rh = 45,
    Pd = 46,
    Ag = 47,
    Cd = 48,
    In = 49,
    Sn = 50,
    Sb = 51,
    Te = 52,
    I = 53,
    Xe = 54,
    Cs = 55,
    Ba = 56,
    La = 57,
    Ce = 58,
    Pr = 59,
    Nd = 60,
    Pm = 61,
    Sm = 62,
    Eu = 63,
    Gd = 64,
    Tb = 65,
    Dy = 66,
    Ho = 67,
    Er = 68,
    Tm = 69,
    Yb = 70,
    Lu = 71,
    Hf = 72,
    Ta = 73,
    W = 74,
    Re = 75,
    Os = 76,
    Ir = 77,
    Pt = 78,
    Au = 79,
    Hg = 80,
    Tl = 81,
    Pb = 82,
    Bi = 83,
    Po = 84,
    At = 85,
    Rn = 86,
    Fr = 87,
    Ra = 88,
    Ac = 89,
    Th = 90,
    Pa = 91,
    U = 92,
    Np = 93,
    Pu = 94,
    Am = 95,
    Cm = 96,
    Bk = 97,
    Cf = 98,
    Es = 99,
    Fm = 100,
    Md = 101,
    No = 102,
    Lr = 103,
    Rf = 104,
    Db = 105,
    Sg = 106,
    Bh = 107,
    Hs = 108,
    Mt = 109,
    Ds = 110,
    Rg = 111,
    Cn = 112,
    Nh = 113,
    Fl = 114,
    Mc = 115,
    Lv = 116,
    Ts = 117,
    Og = 118,
}

const SYMBOLS: [&str; 119] = [
    "?", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

const COVALENT_RADII: [f32; 119] = [
    0.77, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07,
    1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22,
    1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45,
    1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98,
    1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36,
    1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80,
    1.69, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60,
    1.60, 1.60, 1.60, 1.60, 1.60, 1.60, 1.60,
];

const VAN_DER_WAALS_RADII: [f32; 119] = [
    1.70, 1.20, 1.40, 2.12, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 1.84, 2.10, 1.80,
    1.80, 1.75, 1.88, 2.75, 2.31, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.40, 2.10, 1.87,
    2.11, 1.85, 1.90, 1.85, 2.02, 3.03, 2.49, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.72,
    1.58, 1.93, 2.17, 2.06, 2.06, 1.98, 2.16, 3.43, 2.68, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.75, 1.66,
    1.55, 1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.00, 2.00, 2.00, 1.86, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00,
];

const COLORS: [[f32; 3]; 119] = [
    [0.750, 0.450, 0.750],
    [0.920, 0.920, 0.920],
    [0.851, 1.000, 1.000],
    [0.450, 0.400, 0.950],
    [0.130, 0.620, 0.130],
    [1.000, 0.710, 0.710],
    [0.320, 0.340, 0.380],
    [0.180, 0.310, 0.970],
    [0.950, 0.120, 0.100],
    [0.120, 0.780, 0.200],
    [0.702, 0.890, 0.961],
    [0.450, 0.400, 0.950],
    [0.130, 0.620, 0.130],
    [0.749, 0.651, 0.651],
    [0.941, 0.784, 0.627],
    [1.000, 0.500, 0.050],
    [0.950, 0.820, 0.130],
    [0.120, 0.780, 0.200],
    [0.502, 0.820, 0.890],
    [0.560, 0.250, 0.830],
    [0.250, 0.950, 0.250],
    [0.902, 0.902, 0.902],
    [0.749, 0.761, 0.780],
    [0.651, 0.651, 0.671],
    [0.541, 0.600, 0.780],
    [0.612, 0.478, 0.780],
    [0.880, 0.400, 0.200],
    [0.941, 0.565, 0.627],
    [0.314, 0.816, 0.314],
    [0.784, 0.502, 0.200],
    [0.490, 0.500, 0.690],
    [0.761, 0.561, 0.561],
    [0.400, 0.561, 0.561],
    [0.741, 0.502, 0.890],
    [1.000, 0.631, 0.000],
    [0.600, 0.130, 0.060],
    [0.361, 0.722, 0.820],
    [0.450, 0.400, 0.950],
    [0.130, 0.620, 0.130],
    [0.580, 1.000, 1.000],
    [0.580, 0.878, 0.878],
    [0.451, 0.761, 0.788],
    [0.329, 0.710, 0.710],
    [0.231, 0.620, 0.620],
    [0.141, 0.561, 0.561],
    [0.039, 0.490, 0.549],
    [0.000, 0.412, 0.522],
    [0.753, 0.753, 0.753],
    [1.000, 0.851, 0.561],
    [0.651, 0.459, 0.451],
    [0.400, 0.502, 0.502],
    [0.620, 0.388, 0.710],
    [0.831, 0.478, 0.000],
    [0.450, 0.100, 0.650],
    [0.259, 0.620, 0.690],
    [0.450, 0.400, 0.950],
    [0.130, 0.620, 0.130],
    [0.439, 0.831, 1.000],
    [1.000, 1.000, 0.780],
    [0.851, 1.000, 0.780],
    [0.780, 1.000, 0.780],
    [0.639, 1.000, 0.780],
    [0.561, 1.000, 0.780],
    [0.380, 1.000, 0.780],
    [0.271, 1.000, 0.780],
    [0.188, 1.000, 0.780],
    [0.122, 1.000, 0.780],
    [0.000, 1.000, 0.612],
    [0.000, 0.902, 0.459],
    [0.000, 0.831, 0.322],
    [0.000, 0.749, 0.220],
    [0.000, 0.671, 0.141],
    [0.302, 0.761, 1.000],
    [0.302, 0.651, 1.000],
    [0.129, 0.580, 0.839],
    [0.149, 0.490, 0.671],
    [0.149, 0.400, 0.588],
    [0.090, 0.329, 0.529],
    [0.816, 0.816, 0.878],
    [1.000, 0.820, 0.137],
    [0.722, 0.722, 0.816],
    [0.651, 0.329, 0.302],
    [0.341, 0.349, 0.380],
    [0.620, 0.310, 0.710],
    [0.671, 0.361, 0.000],
    [0.459, 0.310, 0.271],
    [0.259, 0.510, 0.588],
    [0.259, 0.000, 0.400],
    [0.000, 0.490, 0.000],
    [0.439, 0.671, 0.980],
    [0.000, 0.729, 1.000],
    [0.000, 0.631, 1.000],
    [0.000, 0.561, 1.000],
    [0.000, 0.502, 1.000],
    [0.000, 0.420, 1.000],
    [0.329, 0.361, 0.949],
    [0.471, 0.361, 0.890],
    [0.541, 0.310, 0.890],
    [0.631, 0.212, 0.831],
    [0.702, 0.122, 0.831],
    [0.702, 0.122, 0.729],
    [0.702, 0.051, 0.651],
    [0.741, 0.051, 0.529],
    [0.780, 0.000, 0.400],
    [0.800, 0.000, 0.349],
    [0.820, 0.000, 0.310],
    [0.851, 0.000, 0.271],
    [0.878, 0.000, 0.220],
    [0.902, 0.000, 0.180],
    [0.922, 0.000, 0.149],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
    [1.000, 0.078, 0.576],
];

const ELEMENTS: [Element; 119] = [
    Element::Unknown,
    Element::H,
    Element::He,
    Element::Li,
    Element::Be,
    Element::B,
    Element::C,
    Element::N,
    Element::O,
    Element::F,
    Element::Ne,
    Element::Na,
    Element::Mg,
    Element::Al,
    Element::Si,
    Element::P,
    Element::S,
    Element::Cl,
    Element::Ar,
    Element::K,
    Element::Ca,
    Element::Sc,
    Element::Ti,
    Element::V,
    Element::Cr,
    Element::Mn,
    Element::Fe,
    Element::Co,
    Element::Ni,
    Element::Cu,
    Element::Zn,
    Element::Ga,
    Element::Ge,
    Element::As,
    Element::Se,
    Element::Br,
    Element::Kr,
    Element::Rb,
    Element::Sr,
    Element::Y,
    Element::Zr,
    Element::Nb,
    Element::Mo,
    Element::Tc,
    Element::Ru,
    Element::Rh,
    Element::Pd,
    Element::Ag,
    Element::Cd,
    Element::In,
    Element::Sn,
    Element::Sb,
    Element::Te,
    Element::I,
    Element::Xe,
    Element::Cs,
    Element::Ba,
    Element::La,
    Element::Ce,
    Element::Pr,
    Element::Nd,
    Element::Pm,
    Element::Sm,
    Element::Eu,
    Element::Gd,
    Element::Tb,
    Element::Dy,
    Element::Ho,
    Element::Er,
    Element::Tm,
    Element::Yb,
    Element::Lu,
    Element::Hf,
    Element::Ta,
    Element::W,
    Element::Re,
    Element::Os,
    Element::Ir,
    Element::Pt,
    Element::Au,
    Element::Hg,
    Element::Tl,
    Element::Pb,
    Element::Bi,
    Element::Po,
    Element::At,
    Element::Rn,
    Element::Fr,
    Element::Ra,
    Element::Ac,
    Element::Th,
    Element::Pa,
    Element::U,
    Element::Np,
    Element::Pu,
    Element::Am,
    Element::Cm,
    Element::Bk,
    Element::Cf,
    Element::Es,
    Element::Fm,
    Element::Md,
    Element::No,
    Element::Lr,
    Element::Rf,
    Element::Db,
    Element::Sg,
    Element::Bh,
    Element::Hs,
    Element::Mt,
    Element::Ds,
    Element::Rg,
    Element::Cn,
    Element::Nh,
    Element::Fl,
    Element::Mc,
    Element::Lv,
    Element::Ts,
    Element::Og,
];

const METALS: [bool; 119] = [
    false, false, false, true, true, false, false, false, false, false, false, true, true, true,
    false, false, false, false, false, true, true, true, true, true, true, true, true, true, true,
    true, true, true, false, false, false, false, false, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, false, false, false, false, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, false, false, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true, true, true, true, false, false,
];

impl Element {
    pub const fn atomic_number(self) -> u8 {
        self as u8
    }

    pub fn from_atomic_number(number: u32) -> Option<Self> {
        ELEMENTS.get(number as usize).copied()
    }

    pub const fn symbol(self) -> &'static str {
        SYMBOLS[self as usize]
    }

    pub const fn cpk_color(self) -> [f32; 4] {
        let [r, g, b] = COLORS[self as usize];
        [r, g, b, 1.0]
    }

    pub const fn covalent_radius(self) -> f32 {
        COVALENT_RADII[self as usize]
    }

    pub const fn van_der_waals_radius(self) -> f32 {
        VAN_DER_WAALS_RADII[self as usize]
    }

    /// Metals form coordination rather than covalent bonds in biomolecular structures; they
    /// are never connected by distance alone.
    pub const fn is_metal(self) -> bool {
        METALS[self as usize]
    }

    pub const fn is_hydrogen(self) -> bool {
        matches!(self, Self::H)
    }
}

impl FromStr for Element {
    type Err = ();

    /// Parses a case-insensitive element symbol. Deuterium and tritium map to hydrogen.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() || value.len() > 2 {
            return Err(());
        }
        if value.eq_ignore_ascii_case("D") || value.eq_ignore_ascii_case("T") {
            return Ok(Self::H);
        }
        SYMBOLS[1..]
            .iter()
            .position(|symbol| symbol.eq_ignore_ascii_case(value))
            .map(|index| ELEMENTS[index + 1])
            .ok_or(())
    }
}

impl fmt::Display for Element {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.symbol())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_indexed_by_atomic_number() {
        for number in 1..=118 {
            let element = Element::from_atomic_number(number).unwrap();
            assert_eq!(u32::from(element.atomic_number()), number);
            assert_eq!(element.symbol().parse::<Element>(), Ok(element));
        }
        assert_eq!(Element::Og.atomic_number(), 118);
        assert_eq!("se".parse(), Ok(Element::Se));
        assert_eq!("D".parse(), Ok(Element::H));
        assert!("Xx".parse::<Element>().is_err());
    }

    #[test]
    fn metals_are_flagged() {
        assert!(Element::Zn.is_metal());
        assert!(Element::Mg.is_metal());
        assert!(!Element::C.is_metal());
        assert!(!Element::Se.is_metal());
    }
}
