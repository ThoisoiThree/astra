use crate::{
    AmbientOcclusionQuality, ColoringMode, DisplayLevel, DisplayMode, ModeOverride,
    VisibilityOverride, molecule::Element,
};

use super::{SceneDocument, SceneError, validation::invalid};

pub(super) fn minimum_reader_version(document: &SceneDocument) -> u32 {
    // Reader 4 understands `atomic_numbers` and the Spacefill and Licorice modes; older
    // readers would silently turn other elements into unknown atoms or reject the modes.
    let new_mode =
        |mode: ModeOverride| matches!(mode, ModeOverride::Spacefill | ModeOverride::Licorice);
    let display = &document.display;
    if document
        .molecule
        .atoms
        .iter()
        .any(|a| a.element != Element::Unknown && element_code(a.element) == 0)
        || matches!(
            display.global_mode,
            DisplayMode::Spacefill | DisplayMode::Licorice
        )
        || [
            DisplayLevel::Chain,
            DisplayLevel::Residue,
            DisplayLevel::Atom,
        ]
        .into_iter()
        .any(|level| {
            display
                .mode_override_values(level)
                .into_iter()
                .any(new_mode)
        })
        || document
            .named_selection_styles
            .values()
            .any(|style| new_mode(style.mode))
        || display
            .representations
            .iter()
            .any(|mask| mask.bits() & !0b11 != 0)
    {
        4
    } else if document
        .measurement_lines
        .iter()
        .any(|line| line.hydrogen_bonds.is_some())
        || document
            .molecule
            .atoms
            .iter()
            .any(|a| element_code(a.element) >= 17)
    {
        3
    } else if document.display.coloring_mode == ColoringMode::SecondaryStructure {
        2
    } else {
        1
    }
}

/// Element codes of the reader-1 `Element` wire enum. Other elements are written as
/// `ELEMENT_UNKNOWN` there and carried exactly by the `atomic_numbers` field.
const LEGACY_ELEMENTS: [Element; 23] = [
    Element::Unknown,
    Element::H,
    Element::C,
    Element::N,
    Element::O,
    Element::P,
    Element::S,
    Element::F,
    Element::Cl,
    Element::Br,
    Element::I,
    Element::Na,
    Element::Mg,
    Element::K,
    Element::Ca,
    Element::Fe,
    Element::Zn,
    Element::Li,
    Element::Rb,
    Element::Cs,
    Element::Be,
    Element::Sr,
    Element::Ba,
];

pub(super) fn element_code(element: Element) -> i32 {
    LEGACY_ELEMENTS
        .iter()
        .position(|candidate| *candidate == element)
        .unwrap_or(0) as i32
}

pub(super) fn element_from_code(code: i32) -> Result<Element, SceneError> {
    usize::try_from(code)
        .ok()
        .and_then(|index| LEGACY_ELEMENTS.get(index).copied())
        .ok_or_else(|| invalid(format!("unknown element code {code}")))
}

pub(super) fn display_mode_code(mode: DisplayMode) -> i32 {
    match mode {
        DisplayMode::Cartoon => 0,
        DisplayMode::BallAndStick => 1,
        DisplayMode::Toon => 2,
        DisplayMode::Spacefill => 3,
        DisplayMode::Licorice => 4,
    }
}

pub(super) fn display_mode_from_code(code: i32) -> Result<DisplayMode, SceneError> {
    match code {
        0 => Ok(DisplayMode::Cartoon),
        1 => Ok(DisplayMode::BallAndStick),
        2 => Ok(DisplayMode::Toon),
        3 => Ok(DisplayMode::Spacefill),
        4 => Ok(DisplayMode::Licorice),
        value => Err(invalid(format!("unknown display mode {value}"))),
    }
}

pub(super) fn coloring_mode_code(mode: ColoringMode) -> i32 {
    match mode {
        ColoringMode::Element => 0,
        ColoringMode::Chain => 1,
        ColoringMode::Residue => 2,
        ColoringMode::ResidueType => 3,
        ColoringMode::BFactor => 4,
        ColoringMode::Uniform => 5,
        ColoringMode::SecondaryStructure => 6,
    }
}

pub(super) fn coloring_mode_from_code(code: i32) -> Result<ColoringMode, SceneError> {
    match code {
        0 => Ok(ColoringMode::Element),
        1 => Ok(ColoringMode::Chain),
        2 => Ok(ColoringMode::Residue),
        3 => Ok(ColoringMode::ResidueType),
        4 => Ok(ColoringMode::BFactor),
        5 => Ok(ColoringMode::Uniform),
        6 => Ok(ColoringMode::SecondaryStructure),
        value => Err(invalid(format!("unknown coloring mode {value}"))),
    }
}

pub(super) fn visibility_code(state: VisibilityOverride) -> u32 {
    match state {
        VisibilityOverride::Inherit => 0,
        VisibilityOverride::Show => 1,
        VisibilityOverride::Hide => 2,
    }
}

pub(super) fn visibility_from_code(code: u32) -> Result<VisibilityOverride, SceneError> {
    match code {
        0 => Ok(VisibilityOverride::Inherit),
        1 => Ok(VisibilityOverride::Show),
        2 => Ok(VisibilityOverride::Hide),
        value => Err(invalid(format!("unknown visibility override {value}"))),
    }
}

pub(super) fn mode_override_code(state: ModeOverride) -> u32 {
    match state {
        ModeOverride::Inherit => 0,
        ModeOverride::Cartoon => 1,
        ModeOverride::BallAndStick => 2,
        ModeOverride::Toon => 3,
        ModeOverride::Spacefill => 4,
        ModeOverride::Licorice => 5,
    }
}

pub(super) fn mode_override_from_code(code: u32) -> Result<ModeOverride, SceneError> {
    match code {
        0 => Ok(ModeOverride::Inherit),
        1 => Ok(ModeOverride::Cartoon),
        2 => Ok(ModeOverride::BallAndStick),
        3 => Ok(ModeOverride::Toon),
        4 => Ok(ModeOverride::Spacefill),
        5 => Ok(ModeOverride::Licorice),
        value => Err(invalid(format!("unknown mode override {value}"))),
    }
}

pub(super) fn ao_quality_code(quality: AmbientOcclusionQuality) -> i32 {
    match quality {
        AmbientOcclusionQuality::Preview => 0,
        AmbientOcclusionQuality::Medium => 1,
        AmbientOcclusionQuality::High => 2,
    }
}

pub(super) fn ao_quality_from_code(code: i32) -> Result<AmbientOcclusionQuality, SceneError> {
    match code {
        0 => Ok(AmbientOcclusionQuality::Preview),
        1 => Ok(AmbientOcclusionQuality::Medium),
        2 => Ok(AmbientOcclusionQuality::High),
        value => Err(invalid(format!("unknown AO quality {value}"))),
    }
}

pub(super) fn enum_code(code: i32, label: &str) -> Result<u32, SceneError> {
    u32::try_from(code).map_err(|_| invalid(format!("{label} has negative value {code}")))
}
