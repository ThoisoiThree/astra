//! File-level structure annotations that are not per-atom topology.

use glam::{DMat3, DVec3};

/// Annotations read from the structure file. Empty for formats that carry none.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StructureInfo {
    /// Entry identifier, for example a PDB ID.
    pub id: Option<String>,
    pub title: Option<String>,
    /// Operators referenced by assembly generators.
    pub operators: Vec<SymmetryOperator>,
    pub assemblies: Vec<Assembly>,
    pub crystal: Option<CrystalInfo>,
    /// Deposited secondary structure (HELIX/SHEET or struct_conf/struct_sheet_range).
    pub secondary: Vec<SecondaryAnnotation>,
    /// Number of models in the source file; later models become trajectory frames.
    pub model_count: usize,
    /// Conditions the reader handled by fallback, shown to the user after loading.
    pub notes: Vec<String>,
}

/// A rigid transformation `x' = rotation * x + translation` in Cartesian Å.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryOperator {
    pub id: String,
    pub name: String,
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl SymmetryOperator {
    pub fn identity(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: "identity".into(),
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    pub fn matrix(&self) -> DMat3 {
        // DMat3 is column-major; rows are stored here.
        DMat3::from_cols_array_2d(&self.rotation).transpose()
    }

    pub fn translation(&self) -> DVec3 {
        DVec3::from_array(self.translation)
    }

    pub fn apply(&self, point: DVec3) -> DVec3 {
        self.matrix() * point + self.translation()
    }

    pub fn is_identity(&self) -> bool {
        let rotation = self.matrix();
        (rotation - DMat3::IDENTITY)
            .to_cols_array()
            .iter()
            .all(|value| value.abs() < 1e-6)
            && self.translation().length() < 1e-4
    }

    /// `self ∘ other`: apply `other` first.
    pub fn compose(&self, other: &Self) -> Self {
        let rotation = self.matrix() * other.matrix();
        let translation = self.matrix() * other.translation() + self.translation();
        Self {
            id: format!("{}x{}", self.id, other.id),
            name: format!("{} · {}", self.name, other.name),
            rotation: rotation.transpose().to_cols_array_2d(),
            translation: translation.to_array(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assembly {
    pub id: String,
    pub details: String,
    pub oligomeric_count: Option<u32>,
    pub generators: Vec<AssemblyGenerator>,
}

impl Assembly {
    pub fn label(&self) -> String {
        let mut label = format!("Assembly {}", self.id);
        if let Some(count) = self.oligomeric_count {
            label.push_str(&format!(" · {}", oligomer_name(count)));
        }
        if !self.details.is_empty() {
            label.push_str(&format!(" ({})", self.details));
        }
        label
    }

    /// Number of chain copies the assembly creates.
    pub fn copy_count(&self) -> usize {
        self.generators
            .iter()
            .map(|generator| generator.chains.len() * generator.products.len())
            .sum()
    }
}

fn oligomer_name(count: u32) -> String {
    match count {
        1 => "monomer".into(),
        2 => "dimer".into(),
        3 => "trimer".into(),
        4 => "tetramer".into(),
        5 => "pentamer".into(),
        6 => "hexamer".into(),
        8 => "octamer".into(),
        12 => "dodecamer".into(),
        24 => "24-mer".into(),
        60 => "60-mer".into(),
        count => format!("{count}-mer"),
    }
}

/// Applies each operator product to the listed chains (author chain identifiers).
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyGenerator {
    pub chains: Vec<String>,
    /// Each entry lists operator indices; they compose left to right, so the last operator
    /// is applied first, following the mmCIF `oper_expression` convention.
    pub products: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitCell {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl UnitCell {
    /// Whether the cell is a real crystal cell rather than the 1 Å cube placeholder that
    /// cryo-EM and NMR entries carry.
    pub fn is_crystallographic(&self) -> bool {
        let placeholder = (self.a - 1.0).abs() < 1e-3
            && (self.b - 1.0).abs() < 1e-3
            && (self.c - 1.0).abs() < 1e-3;
        !placeholder
            && [self.a, self.b, self.c]
                .iter()
                .all(|length| length.is_finite() && *length > 1.5)
            && [self.alpha, self.beta, self.gamma]
                .iter()
                .all(|angle| angle.is_finite() && *angle > 1.0 && *angle < 179.0)
    }

    /// PDB convention: a along x, b in the xy plane (columns are the cell vectors).
    pub fn orthogonalization(&self) -> DMat3 {
        let (alpha, beta, gamma) = (
            self.alpha.to_radians(),
            self.beta.to_radians(),
            self.gamma.to_radians(),
        );
        let (cos_alpha, cos_beta, cos_gamma) = (alpha.cos(), beta.cos(), gamma.cos());
        let sin_gamma = gamma.sin();
        let volume_factor =
            (1.0 - cos_alpha * cos_alpha - cos_beta * cos_beta - cos_gamma * cos_gamma
                + 2.0 * cos_alpha * cos_beta * cos_gamma)
                .max(0.0)
                .sqrt();
        DMat3::from_cols(
            DVec3::new(self.a, 0.0, 0.0),
            DVec3::new(self.b * cos_gamma, self.b * sin_gamma, 0.0),
            DVec3::new(
                self.c * cos_beta,
                self.c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                self.c * volume_factor / sin_gamma,
            ),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrystalInfo {
    pub cell: UnitCell,
    /// Hermann–Mauguin symbol as written in the file.
    pub space_group: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnotatedStructure {
    Helix,
    Helix310,
    HelixPi,
    Strand,
    Turn,
}

/// An inclusive residue range with a deposited secondary-structure element.
#[derive(Debug, Clone, PartialEq)]
pub struct SecondaryAnnotation {
    pub kind: AnnotatedStructure,
    pub chain: String,
    pub start: (i32, Option<char>),
    pub end: (i32, Option<char>),
}

/// Parses an mmCIF `oper_expression` such as `1`, `1,2`, `(1-60)` or `(1-5)(X0)` into the
/// list of operator-id products it denotes.
pub fn parse_operator_expression(expression: &str) -> Result<Vec<Vec<String>>, String> {
    let expression: String = expression.chars().filter(|c| !c.is_whitespace()).collect();
    if expression.is_empty() {
        return Err("empty operator expression".into());
    }
    let mut groups = Vec::new();
    if expression.starts_with('(') {
        let mut rest = expression.as_str();
        while !rest.is_empty() {
            let Some(inner) = rest.strip_prefix('(') else {
                return Err(format!(
                    "unexpected text in operator expression '{expression}'"
                ));
            };
            let end = inner
                .find(')')
                .ok_or_else(|| format!("unbalanced parentheses in '{expression}'"))?;
            groups.push(expand_operator_list(&inner[..end])?);
            rest = &inner[end + 1..];
        }
    } else {
        groups.push(expand_operator_list(&expression)?);
    }
    let mut products: Vec<Vec<String>> = vec![Vec::new()];
    for group in groups {
        let mut next = Vec::with_capacity(products.len() * group.len());
        for product in &products {
            for id in &group {
                let mut extended = product.clone();
                extended.push(id.clone());
                next.push(extended);
            }
        }
        products = next;
        if products.len() > 1_000_000 {
            return Err("operator expression expands to too many copies".into());
        }
    }
    Ok(products)
}

fn expand_operator_list(list: &str) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();
    for item in list.split(',').filter(|item| !item.is_empty()) {
        if let Some((start, end)) = item.split_once('-')
            && let (Ok(start), Ok(end)) = (start.parse::<i64>(), end.parse::<i64>())
        {
            if end < start || end - start > 100_000 {
                return Err(format!("invalid operator range '{item}'"));
            }
            ids.extend((start..=end).map(|id| id.to_string()));
        } else {
            ids.push(item.to_string());
        }
    }
    if ids.is_empty() {
        return Err(format!("empty operator list '{list}'"));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_operator_expressions() {
        assert_eq!(parse_operator_expression("1").unwrap(), vec![vec!["1"]]);
        assert_eq!(
            parse_operator_expression("1,2").unwrap(),
            vec![vec!["1"], vec!["2"]]
        );
        assert_eq!(parse_operator_expression("(1-3)").unwrap().len(), 3);
        let products = parse_operator_expression("(1-2)(X0, 7)").unwrap();
        assert_eq!(
            products,
            vec![
                vec!["1", "X0"],
                vec!["1", "7"],
                vec!["2", "X0"],
                vec!["2", "7"]
            ]
        );
        assert!(parse_operator_expression("(1-2").is_err());
        assert!(parse_operator_expression("").is_err());
    }

    #[test]
    fn composition_applies_the_right_operator_first() {
        let mut shift = SymmetryOperator::identity("1");
        shift.translation = [1.0, 0.0, 0.0];
        let mut rotate = SymmetryOperator::identity("2");
        rotate.rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let combined = rotate.compose(&shift);
        let point = combined.apply(DVec3::ZERO);
        assert!((point - DVec3::new(0.0, 1.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn orthogonalization_matches_cell_lengths_and_angles() {
        let cell = UnitCell {
            a: 50.0,
            b: 60.0,
            c: 70.0,
            alpha: 80.0,
            beta: 95.0,
            gamma: 105.0,
        };
        let matrix = cell.orthogonalization();
        let (a, b, c) = (matrix.x_axis, matrix.y_axis, matrix.z_axis);
        assert!((a.length() - 50.0).abs() < 1e-9);
        assert!((b.length() - 60.0).abs() < 1e-9);
        assert!((c.length() - 70.0).abs() < 1e-9);
        let angle = |u: DVec3, v: DVec3| u.angle_between(v).to_degrees();
        assert!((angle(b, c) - 80.0).abs() < 1e-9);
        assert!((angle(a, c) - 95.0).abs() < 1e-9);
        assert!((angle(a, b) - 105.0).abs() < 1e-9);
        assert!(
            !UnitCell {
                a: 1.0,
                b: 1.0,
                c: 1.0,
                alpha: 90.0,
                beta: 90.0,
                gamma: 90.0
            }
            .is_crystallographic()
        );
    }
}
