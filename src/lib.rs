pub mod bitset;
pub mod camera;
pub mod command;
pub mod diagnostics;
pub mod measurement;
pub mod molecule;
pub mod paths;
pub mod picking;
pub mod render;
pub mod scene;
pub mod selection;

use std::collections::HashMap;

use bitset::AtomMask;
use molecule::Molecule;

/// UI and scene-file color. Components are encoded in the sRGB transfer function.
pub type DisplayColor = [f32; 4];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrgbColor(pub DisplayColor);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearColor(pub [f32; 4]);

impl SrgbColor {
    pub fn to_linear(self) -> LinearColor {
        let [red, green, blue, alpha] = self.0;
        LinearColor([
            srgb_channel_to_linear(red),
            srgb_channel_to_linear(green),
            srgb_channel_to_linear(blue),
            alpha.clamp(0.0, 1.0),
        ])
    }
}

fn srgb_channel_to_linear(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    fn srgb_transfer_preserves_endpoints_and_alpha() {
        assert_eq!(
            SrgbColor([0.0, 1.0, 0.0, 0.4]).to_linear(),
            LinearColor([0.0, 1.0, 0.0, 0.4])
        );
    }

    #[test]
    fn srgb_midpoint_is_converted_to_linear_light() {
        let linear = SrgbColor([0.5, 0.5, 0.5, 1.0]).to_linear().0;
        for channel in &linear[..3] {
            assert!((*channel - 0.214_041_14).abs() < 1.0e-6);
        }
    }
}

const DEFAULT_UNIFORM_COLOR: DisplayColor = [0.55, 0.67, 0.82, 1.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmbientOcclusionQuality {
    #[default]
    Preview,
    Medium,
    High,
}

impl AmbientOcclusionQuality {
    pub const ALL: [Self; 3] = [Self::Preview, Self::Medium, Self::High];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Preview => "Preview",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }

    pub const fn sample_count(self) -> u32 {
        match self {
            Self::Preview => 6,
            Self::Medium => 12,
            Self::High => 32,
        }
    }

    pub const fn dof_layer_count(self) -> u32 {
        match self {
            Self::Preview => 2,
            Self::Medium => 3,
            Self::High => 4,
        }
    }

    pub const fn dof_resolution_scale(self) -> f32 {
        match self {
            Self::Preview => 0.25,
            Self::Medium => 0.5,
            Self::High => 1.0,
        }
    }

    pub const fn cartoon_samples_per_residue(self) -> usize {
        match self {
            Self::Preview => 3,
            Self::Medium => 5,
            Self::High => 10,
        }
    }

    pub const fn cartoon_width_segments(self) -> u32 {
        match self {
            Self::Preview => 2,
            Self::Medium => 4,
            Self::High => 8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientOcclusionSettings {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub bias: f32,
    pub quality: AmbientOcclusionQuality,
}

impl Default for AmbientOcclusionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 1.0,
            radius: 2.0,
            bias: 0.05,
            quality: AmbientOcclusionQuality::Preview,
        }
    }
}

impl AmbientOcclusionSettings {
    pub const fn ball_and_stick_default() -> Self {
        Self {
            enabled: true,
            strength: 1.4,
            radius: 2.3,
            bias: 0.05,
            quality: AmbientOcclusionQuality::Preview,
        }
    }

    pub const fn toon_default() -> Self {
        Self {
            enabled: true,
            strength: 0.15,
            radius: 2.0,
            bias: 0.05,
            quality: AmbientOcclusionQuality::Preview,
        }
    }
}

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
    Toon,
}

impl DisplayMode {
    pub const ALL: [Self; 3] = [Self::Cartoon, Self::BallAndStick, Self::Toon];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cartoon => "Cartoon",
            Self::BallAndStick => "Ball & stick",
            Self::Toon => "Toon",
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Cartoon => Self::BallAndStick,
            Self::BallAndStick => Self::Toon,
            Self::Toon => Self::Cartoon,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModeOverride {
    #[default]
    Inherit,
    Cartoon,
    BallAndStick,
    Toon,
}

impl ModeOverride {
    pub const fn from_mode(mode: DisplayMode) -> Self {
        match mode {
            DisplayMode::Cartoon => Self::Cartoon,
            DisplayMode::BallAndStick => Self::BallAndStick,
            DisplayMode::Toon => Self::Toon,
        }
    }

    pub const fn mode(self) -> Option<DisplayMode> {
        match self {
            Self::Inherit => None,
            Self::Cartoon => Some(DisplayMode::Cartoon),
            Self::BallAndStick => Some(DisplayMode::BallAndStick),
            Self::Toon => Some(DisplayMode::Toon),
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
    SecondaryStructure,
    BFactor,
    Uniform,
}

impl ColoringMode {
    pub const ALL: [Self; 7] = [
        Self::Element,
        Self::Chain,
        Self::Residue,
        Self::ResidueType,
        Self::SecondaryStructure,
        Self::BFactor,
        Self::Uniform,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Element => "Element / CPK",
            Self::Chain => "Chain based",
            Self::Residue => "Residue identity",
            Self::ResidueType => "Residue type",
            Self::SecondaryStructure => "Secondary structure",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HierarchyMembership {
    chain: u32,
    residue: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct HierarchyOverrides<T> {
    chains: HashMap<u32, T>,
    residues: HashMap<u32, T>,
    atoms: HashMap<usize, T>,
}

impl<T> Default for HierarchyOverrides<T> {
    fn default() -> Self {
        Self {
            chains: HashMap::new(),
            residues: HashMap::new(),
            atoms: HashMap::new(),
        }
    }
}

impl<T: Copy> HierarchyOverrides<T> {
    fn get(
        &self,
        atom_index: usize,
        level: DisplayLevel,
        membership: &[HierarchyMembership],
    ) -> Option<T> {
        match level {
            DisplayLevel::Chain => membership
                .get(atom_index)
                .and_then(|path| self.chains.get(&path.chain)),
            DisplayLevel::Residue => membership
                .get(atom_index)
                .and_then(|path| self.residues.get(&path.residue)),
            DisplayLevel::Atom => self.atoms.get(&atom_index),
        }
        .copied()
    }

    fn set(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        value: Option<T>,
        membership: &[HierarchyMembership],
    ) {
        for &index in indices {
            match level {
                DisplayLevel::Chain => {
                    if let Some(path) = membership.get(index) {
                        set_sparse(&mut self.chains, path.chain, value);
                    }
                }
                DisplayLevel::Residue => {
                    if let Some(path) = membership.get(index) {
                        set_sparse(&mut self.residues, path.residue, value);
                    }
                }
                DisplayLevel::Atom => set_sparse(&mut self.atoms, index, value),
            }
        }
    }

    fn clear(&mut self) {
        self.chains.clear();
        self.residues.clear();
        self.atoms.clear();
    }
}

fn set_sparse<K: Eq + std::hash::Hash, T>(map: &mut HashMap<K, T>, key: K, value: Option<T>) {
    if let Some(value) = value {
        map.insert(key, value);
    } else {
        map.remove(&key);
    }
}

/// Authoritative, history-safe portion of a display state. Dense renderer
/// caches and immutable hierarchy membership are intentionally excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayStateData {
    global_mode: DisplayMode,
    coloring_mode: ColoringMode,
    uniform_color: DisplayColor,
    ambient_occlusion: AmbientOcclusionSettings,
    ambient_occlusion_customized: bool,
    selection: AtomMask,
    representation_overrides: HashMap<usize, RepresentationMask>,
    colors: HierarchyOverrides<DisplayColor>,
    visibility: HierarchyOverrides<VisibilityOverride>,
    modes: HierarchyOverrides<ModeOverride>,
}

impl DisplayStateData {
    /// Carry sparse formatting across an explicit atom remap. Hierarchy ids are
    /// derived afresh because protonation can rename residues and remove H.
    pub fn remapped(mut self, old: &Molecule, new: &Molecule, mapping: &[Option<usize>]) -> Self {
        let old_members = hierarchy_membership(old);
        let new_members = hierarchy_membership(new);
        let mut chains = HashMap::new();
        let mut residues = HashMap::new();
        for (i, j) in mapping.iter().enumerate() {
            if let Some(j) = j
                && let (Some(a), Some(b)) = (old_members.get(i), new_members.get(*j))
            {
                chains.insert(a.chain, b.chain);
                residues.insert(a.residue, b.residue);
            }
        }
        fn atoms<T: Copy>(values: HashMap<usize, T>, map: &[Option<usize>]) -> HashMap<usize, T> {
            values
                .into_iter()
                .filter_map(|(i, v)| map.get(i).copied().flatten().map(|j| (j, v)))
                .collect()
        }
        fn hierarchy<T: Copy>(
            values: HierarchyOverrides<T>,
            map: &[Option<usize>],
            chains: &HashMap<u32, u32>,
            residues: &HashMap<u32, u32>,
        ) -> HierarchyOverrides<T> {
            HierarchyOverrides {
                atoms: atoms(values.atoms, map),
                chains: values
                    .chains
                    .into_iter()
                    .filter_map(|(i, v)| chains.get(&i).map(|&j| (j, v)))
                    .collect(),
                residues: values
                    .residues
                    .into_iter()
                    .filter_map(|(i, v)| residues.get(&i).map(|&j| (j, v)))
                    .collect(),
            }
        }
        let mut selection = vec![false; new.atoms.len()];
        for i in self.selection.indices() {
            if let Some(Some(j)) = mapping.get(i) {
                selection[*j] = true;
            }
        }
        self.selection = AtomMask::from_bools(selection);
        self.representation_overrides = atoms(self.representation_overrides, mapping);
        self.colors = hierarchy(self.colors, mapping, &chains, &residues);
        self.visibility = hierarchy(self.visibility, mapping, &chains, &residues);
        self.modes = hierarchy(self.modes, mapping, &chains, &residues);
        self
    }
}

/// Per-document display state. Hierarchy edits are sparse; the public flat
/// arrays are a derived renderer cache rebuilt from the authoritative state.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayState {
    pub colors: Vec<DisplayColor>,
    pub visible: AtomMask,
    pub representations: Vec<RepresentationMask>,
    pub selection: AtomMask,
    pub modes: Vec<DisplayMode>,
    pub global_mode: DisplayMode,
    pub coloring_mode: ColoringMode,
    pub uniform_color: DisplayColor,
    pub ambient_occlusion: AmbientOcclusionSettings,
    ambient_occlusion_customized: bool,
    base_colors: Vec<DisplayColor>,
    named_colors: Vec<Option<DisplayColor>>,
    named_visibility: Vec<VisibilityOverride>,
    named_modes: Vec<ModeOverride>,
    membership: Vec<HierarchyMembership>,
    default_representations: Vec<RepresentationMask>,
    representation_overrides: HashMap<usize, RepresentationMask>,
    hierarchy_colors: HierarchyOverrides<DisplayColor>,
    hierarchy_visibility: HierarchyOverrides<VisibilityOverride>,
    hierarchy_modes: HierarchyOverrides<ModeOverride>,
}

impl DisplayState {
    pub fn for_molecule(molecule: &Molecule) -> Self {
        let atom_count = molecule.atoms.len();
        let base_colors = indexed_categorical_colors(molecule, |atom| atom.chain_id.clone());
        let default_representations = default_representations(molecule);
        Self {
            colors: base_colors.clone(),
            visible: AtomMask::new(atom_count, true),
            representations: default_representations.clone(),
            selection: AtomMask::new(atom_count, false),
            modes: vec![DisplayMode::Cartoon; atom_count],
            global_mode: DisplayMode::Cartoon,
            coloring_mode: ColoringMode::Chain,
            uniform_color: DEFAULT_UNIFORM_COLOR,
            ambient_occlusion: AmbientOcclusionSettings::default(),
            ambient_occlusion_customized: false,
            base_colors,
            named_colors: vec![None; atom_count],
            named_visibility: vec![VisibilityOverride::Inherit; atom_count],
            named_modes: vec![ModeOverride::Inherit; atom_count],
            membership: hierarchy_membership(molecule),
            default_representations,
            representation_overrides: HashMap::new(),
            hierarchy_colors: HierarchyOverrides::default(),
            hierarchy_visibility: HierarchyOverrides::default(),
            hierarchy_modes: HierarchyOverrides::default(),
        }
    }

    pub fn edit_state(&self) -> DisplayStateData {
        DisplayStateData {
            global_mode: self.global_mode,
            coloring_mode: self.coloring_mode,
            uniform_color: self.uniform_color,
            ambient_occlusion: self.ambient_occlusion,
            ambient_occlusion_customized: self.ambient_occlusion_customized,
            selection: self.selection.clone(),
            representation_overrides: self.representation_overrides.clone(),
            colors: self.hierarchy_colors.clone(),
            visibility: self.hierarchy_visibility.clone(),
            modes: self.hierarchy_modes.clone(),
        }
    }

    pub fn restore_edit_state(&mut self, molecule: &Molecule, state: DisplayStateData) {
        self.global_mode = state.global_mode;
        self.coloring_mode = state.coloring_mode;
        self.uniform_color = state.uniform_color;
        self.ambient_occlusion = state.ambient_occlusion;
        self.ambient_occlusion_customized = state.ambient_occlusion_customized;
        self.selection = state.selection;
        self.representation_overrides = state.representation_overrides;
        self.hierarchy_colors = state.colors;
        self.hierarchy_visibility = state.visibility;
        self.hierarchy_modes = state.modes;
        self.base_colors = base_colors(molecule, self.coloring_mode, self.uniform_color);
        self.recompute_all();
    }

    pub fn set_selection(&mut self, selection: AtomMask) {
        if selection.len() == self.selection.len() {
            self.selection = selection;
        }
    }

    pub fn set_selection_from_bools(&mut self, selection: Vec<bool>) {
        self.set_selection(AtomMask::from_bools(selection));
    }

    pub fn set_representation(
        &mut self,
        indices: impl IntoIterator<Item = usize>,
        representation: RepresentationMask,
        shown: bool,
    ) {
        for index in indices {
            let Some(default) = self.default_representations.get(index).copied() else {
                continue;
            };
            let mut value = self
                .representation_overrides
                .get(&index)
                .copied()
                .unwrap_or(default);
            if shown {
                value.insert(representation);
            } else {
                value.remove(representation);
            }
            set_sparse(
                &mut self.representation_overrides,
                index,
                (value != default).then_some(value),
            );
            self.representations[index] = value;
        }
    }

    pub fn load_representations(&mut self, values: Vec<RepresentationMask>) {
        self.representation_overrides.clear();
        for (index, value) in values.into_iter().enumerate() {
            if self
                .default_representations
                .get(index)
                .is_some_and(|default| *default != value)
            {
                self.representation_overrides.insert(index, value);
            }
        }
        self.recompute_representations();
    }

    pub fn set_coloring_mode(&mut self, molecule: &Molecule, mode: ColoringMode) {
        self.coloring_mode = mode;
        self.base_colors = base_colors(molecule, mode, self.uniform_color);
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
            self.hierarchy_colors
                .set(indices, *level, Some(opaque(*color)), &self.membership);
        }
        self.recompute_colors();
    }

    fn write_color_override(
        &mut self,
        indices: &[usize],
        level: DisplayLevel,
        color: Option<DisplayColor>,
    ) {
        self.hierarchy_colors
            .set(indices, level, color, &self.membership);
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
            .hierarchy_colors
            .get(atom_index, DisplayLevel::Chain, &self.membership)
            .unwrap_or(base);
        match level {
            DisplayLevel::Chain => chain,
            DisplayLevel::Residue => self
                .hierarchy_colors
                .get(atom_index, DisplayLevel::Residue, &self.membership)
                .unwrap_or(chain),
            DisplayLevel::Atom => self.colors.get(atom_index).copied().unwrap_or(chain),
        }
    }

    pub fn color_is_overridden(&self, atom_index: usize, level: DisplayLevel) -> bool {
        self.hierarchy_colors
            .get(atom_index, level, &self.membership)
            .is_some()
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
        self.hierarchy_visibility.set(
            indices,
            level,
            (state != VisibilityOverride::Inherit).then_some(state),
            &self.membership,
        );
        self.recompute_visibility();
    }

    pub fn set_visibility_overrides(
        &mut self,
        operations: &[(Vec<usize>, DisplayLevel, VisibilityOverride)],
    ) {
        for (indices, level, state) in operations {
            self.hierarchy_visibility.set(
                indices,
                *level,
                (*state != VisibilityOverride::Inherit).then_some(*state),
                &self.membership,
            );
        }
        self.recompute_visibility();
    }

    pub fn visibility_override(
        &self,
        atom_index: usize,
        level: DisplayLevel,
    ) -> VisibilityOverride {
        self.hierarchy_visibility
            .get(atom_index, level, &self.membership)
            .unwrap_or_default()
    }

    pub fn set_global_mode(&mut self, mode: DisplayMode) {
        if !self.ambient_occlusion_customized {
            self.ambient_occlusion = match mode {
                DisplayMode::Cartoon => AmbientOcclusionSettings::default(),
                DisplayMode::BallAndStick => AmbientOcclusionSettings::ball_and_stick_default(),
                DisplayMode::Toon => AmbientOcclusionSettings::toon_default(),
            };
        }
        self.global_mode = mode;
        let redundant = ModeOverride::from_mode(mode);
        self.hierarchy_modes
            .chains
            .retain(|_, value| *value != redundant);
        self.hierarchy_modes
            .residues
            .retain(|_, value| *value != redundant);
        self.hierarchy_modes
            .atoms
            .retain(|_, value| *value != redundant);
        self.recompute_modes();
    }

    pub fn set_ambient_occlusion(&mut self, settings: AmbientOcclusionSettings) {
        self.ambient_occlusion = settings;
        self.ambient_occlusion_customized = true;
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
        self.hierarchy_modes.set(
            indices,
            level,
            (state != ModeOverride::Inherit).then_some(state),
            &self.membership,
        );
        self.recompute_modes();
    }

    pub fn set_mode_overrides(&mut self, operations: &[(Vec<usize>, DisplayLevel, ModeOverride)]) {
        for (indices, level, state) in operations {
            let state = if state.mode() == Some(self.global_mode) {
                ModeOverride::Inherit
            } else {
                *state
            };
            self.hierarchy_modes.set(
                indices,
                *level,
                (state != ModeOverride::Inherit).then_some(state),
                &self.membership,
            );
        }
        self.recompute_modes();
    }

    pub fn mode_override(&self, atom_index: usize, level: DisplayLevel) -> ModeOverride {
        self.hierarchy_modes
            .get(atom_index, level, &self.membership)
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
            .hierarchy_modes
            .get(atom_index, DisplayLevel::Chain, &self.membership)
            .and_then(ModeOverride::mode)
            .unwrap_or(named);
        match level {
            DisplayLevel::Chain => chain,
            DisplayLevel::Residue => self
                .hierarchy_modes
                .get(atom_index, DisplayLevel::Residue, &self.membership)
                .and_then(ModeOverride::mode)
                .unwrap_or(chain),
            DisplayLevel::Atom => self.modes.get(atom_index).copied().unwrap_or(chain),
        }
    }

    pub fn reset_colors(&mut self, molecule: &Molecule) {
        self.hierarchy_colors.clear();
        self.uniform_color = DEFAULT_UNIFORM_COLOR;
        self.set_coloring_mode(molecule, ColoringMode::Chain);
    }

    pub fn selection_count(&self) -> usize {
        self.selection.count()
    }

    fn recompute_colors(&mut self) {
        for index in 0..self.colors.len() {
            self.colors[index] = self
                .hierarchy_colors
                .get(index, DisplayLevel::Atom, &self.membership)
                .or_else(|| {
                    self.hierarchy_colors
                        .get(index, DisplayLevel::Residue, &self.membership)
                })
                .or_else(|| {
                    self.hierarchy_colors
                        .get(index, DisplayLevel::Chain, &self.membership)
                })
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
                .hierarchy_colors
                .get(atom_index, DisplayLevel::Chain, &self.membership)
                .unwrap_or(base),
            DisplayLevel::Atom => self
                .hierarchy_colors
                .get(atom_index, DisplayLevel::Residue, &self.membership)
                .or_else(|| {
                    self.hierarchy_colors
                        .get(atom_index, DisplayLevel::Chain, &self.membership)
                })
                .unwrap_or(base),
        }
    }

    fn recompute_visibility(&mut self) {
        for index in 0..self.visible.len() {
            let state = [
                self.named_visibility[index],
                self.hierarchy_visibility
                    .get(index, DisplayLevel::Chain, &self.membership)
                    .unwrap_or_default(),
                self.hierarchy_visibility
                    .get(index, DisplayLevel::Residue, &self.membership)
                    .unwrap_or_default(),
                self.hierarchy_visibility
                    .get(index, DisplayLevel::Atom, &self.membership)
                    .unwrap_or_default(),
            ]
            .into_iter()
            .rev()
            .find(|state| *state != VisibilityOverride::Inherit)
            .unwrap_or(VisibilityOverride::Show);
            self.visible.set(index, state == VisibilityOverride::Show);
        }
    }

    fn recompute_modes(&mut self) {
        for index in 0..self.modes.len() {
            self.modes[index] = [
                self.named_modes[index],
                self.hierarchy_modes
                    .get(index, DisplayLevel::Chain, &self.membership)
                    .unwrap_or_default(),
                self.hierarchy_modes
                    .get(index, DisplayLevel::Residue, &self.membership)
                    .unwrap_or_default(),
                self.hierarchy_modes
                    .get(index, DisplayLevel::Atom, &self.membership)
                    .unwrap_or_default(),
            ]
            .into_iter()
            .rev()
            .find_map(ModeOverride::mode)
            .unwrap_or(self.global_mode);
        }
    }

    fn recompute_representations(&mut self) {
        self.representations
            .clone_from(&self.default_representations);
        for (&index, &value) in &self.representation_overrides {
            if let Some(slot) = self.representations.get_mut(index) {
                *slot = value;
            }
        }
    }

    fn recompute_all(&mut self) {
        self.recompute_representations();
        self.recompute_colors();
        self.recompute_visibility();
        self.recompute_modes();
    }

    pub fn color_override_values(&self, level: DisplayLevel) -> Vec<Option<DisplayColor>> {
        (0..self.colors.len())
            .map(|index| self.hierarchy_colors.get(index, level, &self.membership))
            .collect()
    }

    pub fn visibility_override_values(&self, level: DisplayLevel) -> Vec<VisibilityOverride> {
        (0..self.colors.len())
            .map(|index| self.visibility_override(index, level))
            .collect()
    }

    pub fn mode_override_values(&self, level: DisplayLevel) -> Vec<ModeOverride> {
        (0..self.colors.len())
            .map(|index| self.mode_override(index, level))
            .collect()
    }

    pub fn load_color_override_values(
        &mut self,
        level: DisplayLevel,
        values: Vec<Option<DisplayColor>>,
    ) {
        for (index, value) in values.into_iter().enumerate() {
            self.hierarchy_colors
                .set(&[index], level, value.map(opaque), &self.membership);
        }
        self.recompute_colors();
    }

    pub fn load_visibility_override_values(
        &mut self,
        level: DisplayLevel,
        values: Vec<VisibilityOverride>,
    ) {
        for (index, value) in values.into_iter().enumerate() {
            self.hierarchy_visibility.set(
                &[index],
                level,
                (value != VisibilityOverride::Inherit).then_some(value),
                &self.membership,
            );
        }
        self.recompute_visibility();
    }

    pub fn load_mode_override_values(&mut self, level: DisplayLevel, values: Vec<ModeOverride>) {
        for (index, value) in values.into_iter().enumerate() {
            self.hierarchy_modes.set(
                &[index],
                level,
                (value != ModeOverride::Inherit).then_some(value),
                &self.membership,
            );
        }
        self.recompute_modes();
    }

    pub fn estimated_edit_state_bytes(&self) -> usize {
        self.selection.estimated_heap_bytes()
            + self.representation_overrides.capacity()
                * (size_of::<usize>() + size_of::<RepresentationMask>())
            + override_bytes(&self.hierarchy_colors)
            + override_bytes(&self.hierarchy_visibility)
            + override_bytes(&self.hierarchy_modes)
    }
}

fn override_bytes<T>(overrides: &HierarchyOverrides<T>) -> usize {
    overrides.chains.capacity() * (size_of::<u32>() + size_of::<T>())
        + overrides.residues.capacity() * (size_of::<u32>() + size_of::<T>())
        + overrides.atoms.capacity() * (size_of::<usize>() + size_of::<T>())
}

fn default_representations(molecule: &Molecule) -> Vec<RepresentationMask> {
    molecule
        .atoms
        .iter()
        .map(|atom| {
            if atom.element == molecule::Element::H {
                RepresentationMask::STICKS
            } else {
                RepresentationMask::SPHERES | RepresentationMask::STICKS
            }
        })
        .collect()
}

fn hierarchy_membership(molecule: &Molecule) -> Vec<HierarchyMembership> {
    let mut chain_ids = HashMap::<String, u32>::new();
    let mut residue_ids = HashMap::<(String, String, i32, Option<char>), u32>::new();
    let mut next_chain = 0_u32;
    let mut next_residue = 0_u32;
    molecule
        .atoms
        .iter()
        .map(|atom| {
            let chain = *chain_ids.entry(atom.chain_id.clone()).or_insert_with(|| {
                let id = next_chain;
                next_chain = next_chain.saturating_add(1);
                id
            });
            let residue_key = (
                atom.chain_id.clone(),
                atom.residue_name.clone(),
                atom.residue_number,
                atom.insertion_code,
            );
            let residue = *residue_ids.entry(residue_key).or_insert_with(|| {
                let id = next_residue;
                next_residue = next_residue.saturating_add(1);
                id
            });
            HierarchyMembership { chain, residue }
        })
        .collect()
}

fn base_colors(
    molecule: &Molecule,
    mode: ColoringMode,
    uniform_color: DisplayColor,
) -> Vec<DisplayColor> {
    match mode {
        ColoringMode::Element => element_colors(molecule),
        ColoringMode::Chain => indexed_categorical_colors(molecule, |atom| atom.chain_id.clone()),
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
        ColoringMode::SecondaryStructure => secondary_structure_colors(molecule),
        ColoringMode::BFactor => b_factor_colors(molecule),
        ColoringMode::Uniform => vec![uniform_color; molecule.atoms.len()],
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

fn secondary_structure_colors(molecule: &Molecule) -> Vec<DisplayColor> {
    use molecule::{MoleculeHierarchy, SecondaryStructure, assign_secondary_structure};

    let hierarchy = MoleculeHierarchy::from_molecule(molecule);
    let assignments = assign_secondary_structure(molecule, &hierarchy);
    // Ligands, solvent, and ions have no secondary structure; retaining CPK colors makes
    // that distinction explicit instead of incorrectly presenting them as protein coils.
    let mut colors = element_colors(molecule);
    for (chain_index, chain) in hierarchy.chains.iter().enumerate() {
        for (residue_index, residue) in chain.residues.iter().enumerate() {
            let structure = assignments
                .get(chain_index)
                .and_then(|chain| chain.get(residue_index))
                .copied()
                .unwrap_or(SecondaryStructure::Coil);
            let color = match structure {
                SecondaryStructure::Helix => [0.92, 0.25, 0.36, 1.0],
                SecondaryStructure::Strand => [0.98, 0.78, 0.20, 1.0],
                SecondaryStructure::Turn => [0.25, 0.72, 0.92, 1.0],
                SecondaryStructure::Coil => [0.72, 0.76, 0.82, 1.0],
                SecondaryStructure::Nucleic => [0.68, 0.43, 0.90, 1.0],
            };
            for &atom_index in &residue.atom_indices {
                if molecule
                    .atoms
                    .get(atom_index)
                    .is_some_and(|atom| !atom.hetero)
                    && let Some(atom_color) = colors.get_mut(atom_index)
                {
                    *atom_color = color;
                }
            }
        }
    }
    colors
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

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & (Self::SPHERES.0 | Self::STICKS.0))
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
    fn default_and_reset_colors_use_chain_coloring() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let chain_colors = indexed_categorical_colors(&molecule, |atom| atom.chain_id.clone());
        assert_eq!(display.coloring_mode, ColoringMode::Chain);
        assert_eq!(display.colors, chain_colors);
        assert!(display.ambient_occlusion.enabled);
        assert_eq!(
            display.ambient_occlusion.quality,
            AmbientOcclusionQuality::default()
        );

        display.set_coloring_mode(&molecule, ColoringMode::Uniform);
        display.set_uniform_color(&molecule, [1.0, 0.0, 0.0, 1.0]);
        display.set_color_override(&[0], DisplayLevel::Atom, Some([0.0, 1.0, 0.0, 1.0]));
        display.reset_colors(&molecule);

        assert_eq!(display.coloring_mode, ColoringMode::Chain);
        assert_eq!(display.uniform_color, DEFAULT_UNIFORM_COLOR);
        assert_eq!(display.colors, chain_colors);
    }

    #[test]
    fn secondary_structure_coloring_marks_polymers_and_keeps_ligands_element_colored() {
        let mut atoms = (0..7)
            .map(|index| {
                let angle = index as f32 * 100.0_f32.to_radians();
                Atom {
                    serial: index + 1,
                    name: "CA".into(),
                    element: Element::C,
                    residue_name: "ALA".into(),
                    residue_number: index as i32 + 1,
                    insertion_code: None,
                    chain_id: "A".into(),
                    position: Vec3::new(2.3 * angle.cos(), 2.3 * angle.sin(), index as f32 * 1.5),
                    occupancy: 1.0,
                    b_factor: 0.0,
                    hetero: false,
                }
            })
            .collect::<Vec<_>>();
        atoms.push(Atom {
            serial: 8,
            name: "O".into(),
            element: Element::O,
            residue_name: "HOH".into(),
            residue_number: 1,
            insertion_code: None,
            chain_id: "B".into(),
            position: Vec3::new(10.0, 0.0, 0.0),
            occupancy: 1.0,
            b_factor: 0.0,
            hetero: true,
        });
        let molecule = Molecule {
            atoms,
            bonds: Vec::new(),
        };
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_coloring_mode(&molecule, ColoringMode::SecondaryStructure);

        assert!(
            display.colors[..7]
                .iter()
                .all(|color| *color == [0.92, 0.25, 0.36, 1.0])
        );
        assert_eq!(display.colors[7], Element::O.cpk_color());
    }

    #[test]
    fn modes_have_distinct_ao_defaults_without_overwriting_manual_settings() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);

        assert_eq!(
            display.ambient_occlusion.quality,
            AmbientOcclusionQuality::Preview
        );

        display.set_global_mode(DisplayMode::BallAndStick);
        assert_eq!(
            display.ambient_occlusion,
            AmbientOcclusionSettings::ball_and_stick_default()
        );
        assert_eq!(
            display.ambient_occlusion.quality,
            AmbientOcclusionQuality::Preview
        );

        display.set_global_mode(DisplayMode::Toon);
        assert_eq!(
            display.ambient_occlusion,
            AmbientOcclusionSettings::toon_default()
        );
        assert_eq!(
            display.ambient_occlusion.quality,
            AmbientOcclusionQuality::Preview
        );

        let custom = AmbientOcclusionSettings {
            strength: 0.73,
            ..display.ambient_occlusion
        };
        display.set_ambient_occlusion(custom);
        display.set_global_mode(DisplayMode::Cartoon);
        display.set_global_mode(DisplayMode::Toon);
        assert_eq!(display.ambient_occlusion, custom);
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
        assert_eq!(
            display.visible.iter().collect::<Vec<_>>(),
            vec![true, false]
        );
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
        assert_eq!(
            display.visible.iter().collect::<Vec<_>>(),
            vec![false, false]
        );
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

    #[test]
    fn edit_state_restores_sparse_authoritative_data_and_dense_cache() {
        let molecule = molecule();
        let mut display = DisplayState::for_molecule(&molecule);
        let before = display.edit_state();

        display.set_selection_from_bools(vec![true, false]);
        display.set_color_override(&[0, 1], DisplayLevel::Chain, Some([0.2, 0.4, 0.8, 1.0]));
        display.set_visibility_override(&[0], DisplayLevel::Atom, VisibilityOverride::Hide);
        display.set_mode_override(&[1], DisplayLevel::Atom, ModeOverride::Toon);
        display.set_representation([0], RepresentationMask::SPHERES, false);
        let after = display.edit_state();

        display.restore_edit_state(&molecule, before);
        assert_eq!(display.selection_count(), 0);
        assert!(display.visible[0]);
        assert_eq!(display.global_mode, DisplayMode::Cartoon);
        assert!(display.representations[0].contains(RepresentationMask::SPHERES));

        display.restore_edit_state(&molecule, after);
        assert_eq!(display.selection_count(), 1);
        assert_eq!(display.colors, vec![[0.2, 0.4, 0.8, 1.0]; 2]);
        assert!(!display.visible[0]);
        assert_eq!(display.modes[1], DisplayMode::Toon);
        assert!(!display.representations[0].contains(RepresentationMask::SPHERES));
    }
}
