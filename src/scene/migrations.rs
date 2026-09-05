use crate::{
    AmbientOcclusionQuality, ColoringMode, DisplayMode, ModeOverride, VisibilityOverride,
    molecule::Element,
};

use super::{SceneDocument, SceneError, validation::invalid};

pub(super) fn minimum_reader_version(document: &SceneDocument) -> u32 {
    if document.display.coloring_mode == ColoringMode::SecondaryStructure {
        2
    } else {
        1
    }
}

pub(super) fn element_code(element: Element) -> i32 {
    match element {
        Element::Unknown => 0,
        Element::H => 1,
        Element::C => 2,
        Element::N => 3,
        Element::O => 4,
        Element::P => 5,
        Element::S => 6,
        Element::F => 7,
        Element::Cl => 8,
        Element::Br => 9,
        Element::I => 10,
        Element::Na => 11,
        Element::Mg => 12,
        Element::K => 13,
        Element::Ca => 14,
        Element::Fe => 15,
        Element::Zn => 16,
    }
}

pub(super) fn element_from_code(code: i32) -> Result<Element, SceneError> {
    match code {
        0 => Ok(Element::Unknown),
        1 => Ok(Element::H),
        2 => Ok(Element::C),
        3 => Ok(Element::N),
        4 => Ok(Element::O),
        5 => Ok(Element::P),
        6 => Ok(Element::S),
        7 => Ok(Element::F),
        8 => Ok(Element::Cl),
        9 => Ok(Element::Br),
        10 => Ok(Element::I),
        11 => Ok(Element::Na),
        12 => Ok(Element::Mg),
        13 => Ok(Element::K),
        14 => Ok(Element::Ca),
        15 => Ok(Element::Fe),
        16 => Ok(Element::Zn),
        value => Err(invalid(format!("unknown element code {value}"))),
    }
}

pub(super) fn display_mode_code(mode: DisplayMode) -> i32 {
    match mode {
        DisplayMode::Cartoon => 0,
        DisplayMode::BallAndStick => 1,
        DisplayMode::Toon => 2,
    }
}

pub(super) fn display_mode_from_code(code: i32) -> Result<DisplayMode, SceneError> {
    match code {
        0 => Ok(DisplayMode::Cartoon),
        1 => Ok(DisplayMode::BallAndStick),
        2 => Ok(DisplayMode::Toon),
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
    }
}

pub(super) fn mode_override_from_code(code: u32) -> Result<ModeOverride, SceneError> {
    match code {
        0 => Ok(ModeOverride::Inherit),
        1 => Ok(ModeOverride::Cartoon),
        2 => Ok(ModeOverride::BallAndStick),
        3 => Ok(ModeOverride::Toon),
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
