pub mod camera;
pub mod command;
pub mod molecule;
pub mod picking;
pub mod render;
pub mod selection;

use std::collections::HashMap;

use molecule::Molecule;

pub type DisplayColor = [f32; 4];

const DEFAULT_UNIFORM_COLOR: DisplayColor = [0.55, 0.67, 0.82, 1.0];

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NamedSelectionStyle {
    pub color: Option<DisplayColor>,
    pub visibility: VisibilityOverride,
    pub mode: ModeOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplayMode {
    #[default]
    Cartoon,
    BallAndStick,
}

impl DisplayMode {
    pub const ALL: [Self; 2] = [Self::Cartoon, Self::BallAndStick];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cartoon => "Cartoon",
            Self::BallAndStick => "Ball & stick",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModeOverride {
    #[default]
    Inherit,
    Cartoon,
    BallAndStick,
}

impl ModeOverride {
    pub const fn from_mode(mode: DisplayMode) -> Self {
        match mode {
            DisplayMode::Cartoon => Self::Cartoon,
            DisplayMode::BallAndStick => Self::BallAndStick,
        }
    }

    pub const fn mode(self) -> Option<DisplayMode> {
        match self {
            Self::Inherit => None,
            Self::Cartoon => Some(DisplayMode::Cartoon),
            Self::BallAndStick => Some(DisplayMode::BallAndStick),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColoringMode {
    #[default]
    Element,
    Chain,
    Residue,
    ResidueType,
    BFactor,
    Uniform,
}

impl ColoringMode {
    pub const ALL: [Self; 6] = [
        Self::Element,
        Self::Chain,
        Self::Residue,
        Self::ResidueType,
        Self::BFactor,
        Self::Uniform,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Element => "Element / CPK",
            Self::Chain => "Chain based",
            Self::Residue => "Residue identity",
            Self::ResidueType => "Residue type",
            Self::BFactor => "B-factor",
            Self::Uniform => "Uniform",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayLevel {
    Chain,
    Residue,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VisibilityOverride {
    #[default]
    Inherit,
    Show,
    Hide,
}

impl VisibilityOverride {
    pub const fn next(self) -> Self {
        match self {
            Self::Inherit => Self::Show,
            Self::Show => Self::Hide,
            Self::Hide => Self::Inherit,
        }
    }
}

/// Per-atom visual state, kept separate from immutable chemical data.
///
/// Hierarchical overrides are stored per atom so the renderer receives flat,
/// contiguous arrays while the application remains responsible for mapping
/// hierarchy nodes to atom indices.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayState {
    pub colors: Vec<DisplayColor>,
    pub visible: Vec<bool>,
    pub representations: Vec<RepresentationMask>,
    pub selection: Vec<bool>,
    pub modes: Vec<DisplayMode>,
    pub global_mode: DisplayMode,
    pub coloring_mode: ColoringMode,
    pub uniform_color: DisplayColor,
    base_colors: Vec<DisplayColor>,
    named_colors: Vec<Option<DisplayColor>>,
    chain_colors: Vec<Option<DisplayColor>>,
    residue_colors: Vec<Option<DisplayColor>>,
    atom_colors: Vec<Option<DisplayColor>>,
    chain_visibility: Vec<VisibilityOverride>,
    named_visibility: Vec<VisibilityOverride>,
    residue_visibility: Vec<VisibilityOverride>,
    atom_visibility: Vec<VisibilityOverride>,
    named_modes: Vec<ModeOverride>,
    chain_modes: Vec<ModeOverride>,
    residue_modes: Vec<ModeOverride>,
    atom_modes: Vec<ModeOverride>,
}

impl DisplayState {
    pub fn for_molecule(molecule: &Molecule) -> Self {
        let atom_count = molecule.atoms.len();
        let base_colors = element_colors(molecule);
        Self {
            colors: base_colors.clone(),
            visible: vec![true; atom_count],
            representations: molecule
                .atoms
                .iter()
                .map(|atom| {
                    if atom.element == molecule::Element::H {
                        RepresentationMask::STICKS
                    } else {
                        RepresentationMask::SPHERES | RepresentationMask::STICKS
                    }
                })
                .collect(),
            selection: vec![false; atom_count],
            modes: vec![DisplayMode::Cartoon; atom_count],
            global_mode: DisplayMode::Cartoon,
            coloring_mode: ColoringMode::Element,
            uniform_color: DEFAULT_UNIFORM_COLOR,
            base_colors,
            named_colors: vec![None; atom_count],
            chain_colors: vec![None; atom_count],
            residue_colors: vec![None; atom_count],
            atom_colors: vec![None; atom_count],
            chain_visibility: vec![VisibilityOverride::Inherit; atom_count],
            named_visibility: vec![VisibilityOverride::Inherit; atom_count],
            residue_visibility: vec![VisibilityOverride::Inherit; atom_count],
            atom_visibility: vec![VisibilityOverride::Inherit; atom_count],
            named_modes: vec![ModeOverride::Inherit; atom_count],
            chain_modes: vec![ModeOverride::Inherit; atom_count],
            residue_modes: vec![ModeOverride::Inherit; atom_count],
            atom_modes: vec![ModeOverride::Inherit; atom_count],
        }
    }

    pub fn set_coloring_mode(&mut self, molecule: &Molecule, mode: ColoringMode) {
        self.coloring_mode = mode;
        self.base_colors = match mode {
            ColoringMode::Element => element_colors(molecule),
            ColoringMode::Chain => {
                indexed_categorical_colors(molecule, |atom| atom.chain_id.clone())
            }
            ColoringMode::Residue => indexed_categorical_colors(molecule, |atom| {
                format!(
                    "{}:{}:{}:{:?}",
                    atom.chain_id, atom.residue_name, atom.residue_number, atom.insertion_code
                )
            }),
            ColoringMode::ResidueType => molecule
                .atoms
                .iter()
                .map(|atom| categorical_color(&atom.residue_name))
                .collect(),
            ColoringMode::BFactor => b_factor_colors(molecule),
            ColoringMode::Uniform => vec![self.uniform_color; molecule.atoms.len()],
        };
        self.recompute_colors();
    }

    pub fn set_uniform_color(&mut self, molecule: &Molecule, color: DisplayColor) {
        self.uniform_color = opaque(color);
        if self.coloring_mode == ColoringMode::Uniform {
            self.base_colors = vec![self.uniform_color; molecule.atoms.len()];
            self.recompute_colors();
        }
    }

    /// Rebuilds named-selection layers in the supplied deterministic order.
    /// Hierarchy overrides remain higher priority than these parent-level layers.
    pub fn replace_named_layers(&mut self, layers: &[(Vec<usize>, NamedSelectionStyle)]) {
        self.named_colors.fill(None);
        self.named_visibility.fill(VisibilityOverride::Inherit);
        self.named_modes.fill(ModeOverride::Inherit);
        for (indices, style) in layers {
            for &index in indices {
                if let Some(value) = self.named_colors.get_mut(index)
                    && let Some(color) = style.color
                {
                    *value = Some(opaque(color));
                }
                if let Some(value) = self.named_visibility.get_mut(index)
                    && style.visibility != VisibilityOverride::Inherit
                {
                    *value = style.visibility;
                }
                if let Some(value) = self.named_modes.get_mut(index)
                    && style.mode != ModeOverride::Inherit
                    && style.mode.mode() != Some(self.global_mode)
                {
                    *value = style.mode;
                }
            }
        }
        self.recompute_colors();
        self.recompute_visibility();
        self.recompute_modes();
    }

    pub fn set_color_override(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        color: Option<DisplayColor>,
    ) {
        let color = color.map(opaque);
        let differs_from_inherited = color.is_some_and(|requested| {
            indices.iter().copied().any(|index| {
                !colors_approximately_equal(requested, self.inherited_color(index, level))
            })
        });
        let color = color.filter(|_| differs_from_inherited);
        self.write_color_override(indices, level, color);
    }

    /// Writes an explicit override even when it currently matches inheritance.
    /// This is used by the hierarchy's "Set to children" operation.
    pub fn set_color_override_forced(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        color: DisplayColor,
    ) {
        self.write_color_override(indices, level, Some(opaque(color)));
    }

    pub fn set_color_overrides_forced(
        &mut self,
        operations: &[(Vec<usize>, DisplayLevel, DisplayColor)],
    ) {
        for (indices, level, color) in operations {
            let overrides = match level {
                DisplayLevel::Chain => &mut self.chain_colors,
                DisplayLevel::Residue => &mut self.residue_colors,
                DisplayLevel::Atom => &mut self.atom_colors,
            };
            for &index in indices {
                if let Some(value) = overrides.get_mut(index) {
                    *value = Some(opaque(*color));
                }
            }
        }
        self.recompute_colors();
    }

    fn write_color_override(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        color: Option<DisplayColor>,
    ) {
        let overrides = match level {
            DisplayLevel::Chain => &mut self.chain_colors,
            DisplayLevel::Residue => &mut self.residue_colors,
            DisplayLevel::Atom => &mut self.atom_colors,
        };
        for &index in indices {
            if let Some(value) = overrides.get_mut(index) {
                *value = color;
            }
        }
        self.recompute_colors();
    }

    pub fn color_at_level(&self, atom_index: usize, level: DisplayLevel) -> DisplayColor {
        let base = self
            .base_colors
            .get(atom_index)
            .copied()
            .unwrap_or([0.5, 0.5, 0.5, 1.0]);
        let base = self
            .named_colors
            .get(atom_index)
            .copied()
            .flatten()
            .unwrap_or(base);
        let chain = self
            .chain_colors
            .get(atom_index)
            .copied()
            .flatten()
            .unwrap_or(base);
        match level {
            DisplayLevel::Chain => chain,
            DisplayLevel::Residue => self
                .residue_colors
                .get(atom_index)
                .copied()
                .flatten()
                .unwrap_or(chain),
            DisplayLevel::Atom => self.colors.get(atom_index).copied().unwrap_or(chain),
        }
    }

    pub fn color_is_overridden(&self, atom_index: usize, level: DisplayLevel) -> bool {
        match level {
            DisplayLevel::Chain => self.chain_colors.get(atom_index),
            DisplayLevel::Residue => self.residue_colors.get(atom_index),
            DisplayLevel::Atom => self.atom_colors.get(atom_index),
        }
        .is_some_and(Option::is_some)
    }

    pub fn cycle_visibility(&mut self, indices: &[usize], level: DisplayLevel) {
        let current = indices
            .first()
            .map_or(VisibilityOverride::Inherit, |&index| {
                self.visibility_override(index, level)
            });
        self.set_visibility_override(indices, level, current.next());
    }

    pub fn set_visibility_override(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        state: VisibilityOverride,
    ) {
        let overrides = match level {
            DisplayLevel::Chain => &mut self.chain_visibility,
            DisplayLevel::Residue => &mut self.residue_visibility,
            DisplayLevel::Atom => &mut self.atom_visibility,
        };
        for &index in indices {
            if let Some(value) = overrides.get_mut(index) {
                *value = state;
            }
        }
        self.recompute_visibility();
    }

    pub fn set_visibility_overrides(
        &mut self,
        operations: &[(Vec<usize>, DisplayLevel, VisibilityOverride)],
    ) {
        for (indices, level, state) in operations {
            let overrides = match level {
                DisplayLevel::Chain => &mut self.chain_visibility,
                DisplayLevel::Residue => &mut self.residue_visibility,
                DisplayLevel::Atom => &mut self.atom_visibility,
            };
            for &index in indices {
                if let Some(value) = overrides.get_mut(index) {
                    *value = *state;
                }
            }
        }
        self.recompute_visibility();
    }

    pub fn visibility_override(
        &self,
        atom_index: usize,
        level: DisplayLevel,
    ) -> VisibilityOverride {
        match level {
            DisplayLevel::Chain => self.chain_visibility.get(atom_index),
            DisplayLevel::Residue => self.residue_visibility.get(atom_index),
            DisplayLevel::Atom => self.atom_visibility.get(atom_index),
        }
        .copied()
        .unwrap_or_default()
    }

    pub fn set_global_mode(&mut self, mode: DisplayMode) {
        self.global_mode = mode;
        let redundant = ModeOverride::from_mode(mode);
        for overrides in [
            &mut self.named_modes,
            &mut self.chain_modes,
            &mut self.residue_modes,
            &mut self.atom_modes,
        ] {
            for value in overrides {
                if *value == redundant {
                    *value = ModeOverride::Inherit;
                }
            }
        }
        self.recompute_modes();
    }

    pub fn set_mode_override(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        state: ModeOverride,
    ) {
        let state = if state.mode() == Some(self.global_mode) {
            ModeOverride::Inherit
        } else {
            state
        };
        let overrides = match level {
            DisplayLevel::Chain => &mut self.chain_modes,
            DisplayLevel::Residue => &mut self.residue_modes,
            DisplayLevel::Atom => &mut self.atom_modes,
        };
        for &index in indices {
            if let Some(value) = overrides.get_mut(index) {
                *value = state;
            }
        }
        self.recompute_modes();
    }

    pub fn set_mode_overrides(&mut self, operations: &[(Vec<usize>, DisplayLevel, ModeOverride)]) {
        for (indices, level, state) in operations {
            let state = if state.mode() == Some(self.global_mode) {
                ModeOverride::Inherit
            } else {
                *state
            };
            let overrides = match level {
                DisplayLevel::Chain => &mut self.chain_modes,
                DisplayLevel::Residue => &mut self.residue_modes,
                DisplayLevel::Atom => &mut self.atom_modes,
            };
            for &index in indices {
                if let Some(value) = overrides.get_mut(index) {
                    *value = state;
                }
            }
        }
        self.recompute_modes();
    }

    pub fn mode_override(&self, atom_index: usize, level: DisplayLevel) -> ModeOverride {
        match level {
            DisplayLevel::Chain => self.chain_modes.get(atom_index),
            DisplayLevel::Residue => self.residue_modes.get(atom_index),
            DisplayLevel::Atom => self.atom_modes.get(atom_index),
        }
        .copied()
        .unwrap_or_default()
    }

    pub fn mode_at_level(&self, atom_index: usize, level: DisplayLevel) -> DisplayMode {
        let named = self
            .named_modes
            .get(atom_index)
            .copied()
            .and_then(ModeOverride::mode)
            .unwrap_or(self.global_mode);
        let chain = self
            .chain_modes
            .get(atom_index)
            .copied()
            .and_then(ModeOverride::mode)
            .unwrap_or(named);
        match level {
            DisplayLevel::Chain => chain,
            DisplayLevel::Residue => self
                .residue_modes
                .get(atom_index)
                .copied()
                .and_then(ModeOverride::mode)
                .unwrap_or(chain),
            DisplayLevel::Atom => self.modes.get(atom_index).copied().unwrap_or(chain),
        }
    }

    pub fn reset_colors(&mut self, molecule: &Molecule) {
        self.chain_colors.fill(None);
        self.residue_colors.fill(None);
        self.atom_colors.fill(None);
        self.uniform_color = DEFAULT_UNIFORM_COLOR;
        self.set_coloring_mode(molecule, ColoringMode::Element);
    }

    pub fn selection_count(&self) -> usize {
        self.selection.iter().filter(|selected| **selected).count()
    }

    fn recompute_colors(&mut self) {
        for index in 0..self.colors.len() {
            self.colors[index] = self.atom_colors[index]
                .or(self.residue_colors[index])
                .or(self.chain_colors[index])
                .or(self.named_colors[index])
                .unwrap_or(self.base_colors[index]);
        }
    }

    fn inherited_color(&self, atom_index: usize, level: DisplayLevel) -> DisplayColor {
        let base = self
            .base_colors
            .get(atom_index)
            .copied()
            .unwrap_or([0.5, 0.5, 0.5, 1.0]);
        let base = self
            .named_colors
            .get(atom_index)
            .copied()
            .flatten()
            .unwrap_or(base);
        match level {
            DisplayLevel::Chain => base,
            DisplayLevel::Residue => self
                .chain_colors
                .get(atom_index)
                .copied()
                .flatten()
                .unwrap_or(base),
            DisplayLevel::Atom => self
                .residue_colors
                .get(atom_index)
                .copied()
                .flatten()
                .or_else(|| self.chain_colors.get(atom_index).copied().flatten())
                .unwrap_or(base),
        }
    }

    fn recompute_visibility(&mut self) {
        for index in 0..self.visible.len() {
            let state = [
                self.named_visibility[index],
                self.chain_visibility[index],
                self.residue_visibility[index],
                self.atom_visibility[index],
            ]
            .into_iter()
            .rev()
            .find(|state| *state != VisibilityOverride::Inherit)
            .unwrap_or(VisibilityOverride::Show);
            self.visible[index] = state == VisibilityOverride::Show;
        }
    }

    fn recompute_modes(&mut self) {
        for index in 0..self.modes.len() {
            self.modes[index] = [
                self.named_modes[index],
                self.chain_modes[index],
                self.residue_modes[index],
                self.atom_modes[index],
            ]
            .into_iter()
            .rev()
            .find_map(ModeOverride::mode)
            .unwrap_or(self.global_mode);
        }
    }
}

fn opaque(mut color: DisplayColor) -> DisplayColor {
    color[3] = 1.0;
    color
}

fn colors_approximately_equal(left: DisplayColor, right: DisplayColor) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| (left - right).abs() <= 1.0 / 255.0)
}

fn element_colors(molecule: &Molecule) -> Vec<DisplayColor> {
    molecule
        .atoms
        .iter()
        .map(|atom| atom.element.cpk_color())
        .collect()
}

const CATEGORICAL_PALETTE: [DisplayColor; 12] = [
    [0.30, 0.62, 0.98, 1.0],
    [0.96, 0.48, 0.25, 1.0],
    [0.34, 0.78, 0.48, 1.0],
    [0.75, 0.42, 0.95, 1.0],
    [0.96, 0.76, 0.24, 1.0],
    [0.20, 0.78, 0.80, 1.0],
    [0.94, 0.36, 0.58, 1.0],
    [0.55, 0.72, 0.26, 1.0],
    [0.48, 0.50, 0.95, 1.0],
    [0.88, 0.60, 0.28, 1.0],
    [0.28, 0.68, 0.62, 1.0],
    [0.78, 0.45, 0.66, 1.0],
];

fn categorical_color(key: &str) -> DisplayColor {
    let hash = key.as_bytes().iter().fold(2_166_136_261_u32, |hash, byte| {
        (hash ^ u32::from(*byte)).wrapping_mul(16_777_619)
    });
    CATEGORICAL_PALETTE[hash as usize % CATEGORICAL_PALETTE.len()]
}

fn indexed_categorical_colors(
    molecule: &Molecule,
    mut key: impl FnMut(&molecule::Atom) -> String,
) -> Vec<DisplayColor> {
    let mut categories = HashMap::<String, usize>::new();
    molecule
        .atoms
        .iter()
        .map(|atom| {
            let key = key(atom);
            let next_index = categories.len();
            let index = *categories.entry(key).or_insert(next_index);
            CATEGORICAL_PALETTE[index % CATEGORICAL_PALETTE.len()]
        })
        .collect()
}

fn b_factor_colors(molecule: &Molecule) -> Vec<DisplayColor> {
    let minimum = molecule
        .atoms
        .iter()
        .map(|atom| atom.b_factor)
        .min_by(f32::total_cmp)
        .unwrap_or(0.0);
    let maximum = molecule
        .atoms
        .iter()
        .map(|atom| atom.b_factor)
        .max_by(f32::total_cmp)
        .unwrap_or(minimum);
    let span = (maximum - minimum).max(f32::EPSILON);
    molecule
        .atoms
        .iter()
        .map(|atom| {
            let t = ((atom.b_factor - minimum) / span).clamp(0.0, 1.0);
            let (r, g, b) = if t < 0.5 {
                let local = t * 2.0;
                (0.1 + 0.1 * local, 0.25 + 0.65 * local, 1.0 - 0.15 * local)
            } else {
                let local = (t - 0.5) * 2.0;
                (0.2 + 0.8 * local, 0.9 - 0.72 * local, 0.85 - 0.75 * local)
            };
            [r, g, b, 1.0]
        })
        .collect()
}

/// A compact mask allows atoms to take part in multiple representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepresentationMask(u8);

impl RepresentationMask {
    pub const SPHERES: Self = Self(1 << 0);
    pub const STICKS: Self = Self(1 << 1);

    pub const fn contains(self, representation: Self) -> bool {
        self.0 & representation.0 != 0
    }

    pub fn insert(&mut self, representation: Self) {
        self.0 |= representation.0;
    }

    pub fn remove(&mut self, representation: Self) {
        self.0 &= !representation.0;
    }
}

impl std::ops::BitOr for RepresentationMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[cfg(test)]
mod display_tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::{Atom, Element};

    fn molecule() -> Molecule {
        Molecule {
            atoms: vec![
                Atom {
                    serial: 1,
                    name: "C".into(),
                    element: Element::C,
                    residue_name: "GLY".into(),
                    residue_number: 1,
                    insertion_code: None,
                    chain_id: "A".into(),
                    position: Vec3::ZERO,
                    occupancy: 1.0,
                    b_factor: 10.0,
                    hetero: false,
                },
                Atom {
                    serial: 2,
                    name: "O".into(),
                    element: Element::O,
                    residue_name: "GLY".into(),
                    residue_number: 1,
                    insertion_code: None,
                    chain_id: "A".into(),
                    position: Vec3::X,
                    occupancy: 1.0,
                    b_factor: 20.0,
                    hetero: false,
                },
            ],
            bonds: Vec::new(),
        }
    }

    #[test]
    fn default_and_reset_colors_use_element_coloring() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let element_colors = element_colors(&molecule);
        assert_eq!(display.coloring_mode, ColoringMode::Element);
        assert_eq!(display.colors, element_colors);

        display.set_coloring_mode(&molecule, ColoringMode::Uniform);
        display.set_uniform_color(&molecule, [1.0, 0.0, 0.0, 1.0]);
        display.set_color_override(&[0], DisplayLevel::Atom, Some([0.0, 1.0, 0.0, 1.0]));
        display.reset_colors(&molecule);

        assert_eq!(display.coloring_mode, ColoringMode::Element);
        assert_eq!(display.uniform_color, DEFAULT_UNIFORM_COLOR);
        assert_eq!(display.colors, element_colors);
    }

    #[test]
    fn lower_color_override_has_priority_and_can_inherit_again() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_color_override(&[0, 1], DisplayLevel::Chain, Some([1.0, 0.0, 0.0, 1.0]));
        display.set_color_override(&[0], DisplayLevel::Atom, Some([0.0, 1.0, 0.0, 1.0]));
        assert_eq!(display.colors[0], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(display.colors[1], [1.0, 0.0, 0.0, 1.0]);
        display.set_color_override(&[0], DisplayLevel::Atom, None);
        assert_eq!(display.colors[0], [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn shown_descendant_overrides_hidden_parent() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_visibility_override(&[0, 1], DisplayLevel::Chain, VisibilityOverride::Hide);
        display.set_visibility_override(&[0], DisplayLevel::Atom, VisibilityOverride::Show);
        assert_eq!(display.visible, vec![true, false]);
    }

    #[test]
    fn color_equal_to_inherited_value_is_not_an_override() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let inherited = display.color_at_level(0, DisplayLevel::Atom);
        display.set_color_override(&[0], DisplayLevel::Atom, Some(inherited));
        assert!(!display.color_is_overridden(0, DisplayLevel::Atom));
    }

    #[test]
    fn forced_child_color_remains_an_override_when_equal_to_parent() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let color = [0.2, 0.4, 0.8, 1.0];
        display.set_color_override(&[0, 1], DisplayLevel::Chain, Some(color));
        display.set_color_overrides_forced(&[(vec![0], DisplayLevel::Atom, color)]);
        assert!(display.color_is_overridden(0, DisplayLevel::Atom));
    }

    #[test]
    fn hierarchy_children_override_named_selection_style() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let named_color = [0.8, 0.2, 0.6, 1.0];
        display.replace_named_layers(&[(
            vec![0, 1],
            NamedSelectionStyle {
                color: Some(named_color),
                visibility: VisibilityOverride::Hide,
                mode: ModeOverride::BallAndStick,
            },
        )]);
        assert_eq!(display.colors, vec![named_color; 2]);
        assert_eq!(display.visible, vec![false, false]);
        assert_eq!(display.modes, vec![DisplayMode::BallAndStick; 2]);

        let child_color = [0.1, 0.9, 0.2, 1.0];
        display.set_color_override(&[0], DisplayLevel::Atom, Some(child_color));
        display.set_visibility_override(&[0], DisplayLevel::Atom, VisibilityOverride::Show);
        assert_eq!(display.colors[0], child_color);
        assert!(display.visible[0]);
        assert!(!display.visible[1]);
    }

    #[test]
    fn lower_mode_override_wins_and_global_matches_inherit() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        assert_eq!(display.global_mode, DisplayMode::Cartoon);
        display.set_mode_override(&[0, 1], DisplayLevel::Chain, ModeOverride::BallAndStick);
        display.set_mode_override(&[0], DisplayLevel::Atom, ModeOverride::Cartoon);
        assert_eq!(
            display.mode_override(0, DisplayLevel::Atom),
            ModeOverride::Inherit
        );
        assert_eq!(display.modes, vec![DisplayMode::BallAndStick; 2]);

        display.set_global_mode(DisplayMode::BallAndStick);
        assert_eq!(display.modes, vec![DisplayMode::BallAndStick; 2]);
        assert_eq!(
            display.mode_override(0, DisplayLevel::Chain),
            ModeOverride::Inherit
        );
    }
}
