//! Portable, versioned scene documents.
//!
//! The wire representation is deliberately separate from the runtime model. This keeps file
//! compatibility independent of Rust layout, wgpu, egui, and implementation-only cached state.

use std::collections::{BTreeMap, HashMap};

#[cfg(test)]
use std::io::{Cursor, Read, Write};

use glam::Vec3;
#[cfg(test)]
use prost::Message;
use thiserror::Error;

use crate::{
    AmbientOcclusionSettings, DisplayLevel, DisplayState, ModeOverride, NamedSelectionStyle,
    RepresentationMask, VisibilityOverride,
    camera::{DepthOfField, OrbitCamera},
    measurement::{MeasurementEndpoint, MeasurementLine},
    molecule::{Atom, Bond, Molecule, MoleculeHierarchy},
    selection::Selection,
};

#[cfg(test)]
use crate::{AmbientOcclusionQuality, ColoringMode, DisplayMode, molecule::Element};

pub const SCENE_EXTENSION: &str = "mol";
pub const SCENE_FORMAT_NAME: &str = "Molecule 1.0";
pub const SCENE_MIME_TYPE: &str = "application/vnd.molview.molecule";
pub const SCENE_SCHEMA_VERSION: u32 = 1;
pub const SCENE_READER_VERSION: u32 = 2;

mod container;
#[cfg(test)]
use container::{COMPRESSION_ZSTD, CONTAINER_VERSION, HEADER_LEN, MAGIC, MAX_DECOMPRESSED_SIZE};
pub use container::{decode, encode, is_scene_document};
mod validation;
use validation::*;
mod migrations;
use migrations::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SceneHierarchyTarget {
    Chain(usize),
    Residue {
        chain_index: usize,
        residue_index: usize,
    },
    Atom(usize),
}

#[derive(Debug, Clone)]
pub struct SceneDocument {
    pub source_name: String,
    pub molecule: Molecule,
    pub display: DisplayState,
    pub named_selections: BTreeMap<String, Selection>,
    pub named_selection_expressions: BTreeMap<String, String>,
    pub named_selection_styles: BTreeMap<String, NamedSelectionStyle>,
    pub measurement_lines: Vec<MeasurementLine>,
    pub hierarchy_names: BTreeMap<SceneHierarchyTarget, String>,
    pub inspection: Option<SceneHierarchyTarget>,
    pub hierarchy_selection: Vec<SceneHierarchyTarget>,
    pub hierarchy_selection_anchor: Option<SceneHierarchyTarget>,
    pub focus_description: String,
    pub pivot_description: String,
    pub camera: OrbitCamera,
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("not a Molecule 1.0 document")]
    InvalidMagic,
    #[error("scene header is truncated")]
    TruncatedHeader,
    #[error("unsupported scene container version {0}")]
    UnsupportedContainer(u16),
    #[error("unsupported scene compression codec {0}")]
    UnsupportedCompression(u16),
    #[error("scene schema {found} requires a reader newer than {supported}")]
    UnsupportedSchema { found: u32, supported: u32 },
    #[error(
        "scene requires Molecule reader {required}, but this application supports reader {supported}"
    )]
    UnsupportedReader { required: u32, supported: u32 },
    #[error("scene payload exceeds the {0} byte safety limit")]
    TooLarge(u64),
    #[error("scene payload length mismatch: expected {expected}, decoded {actual}")]
    LengthMismatch { expected: u64, actual: u64 },
    #[error("invalid scene data: {0}")]
    InvalidData(String),
    #[error("scene compression failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("could not decode scene schema: {0}")]
    Decode(#[from] prost::DecodeError),
}

mod wire;
#[cfg(test)]
use wire::legacy_wire_v1;

impl wire::SceneV1 {
    fn from_document(document: &SceneDocument) -> Result<Self, SceneError> {
        let atom_count = document.molecule.atoms.len();
        if document.display.colors.len() != atom_count {
            return Err(invalid("display state does not match molecule atom count"));
        }
        let selections = document
            .named_selections
            .iter()
            .map(|(name, selection)| {
                if selection.flags().len() != atom_count {
                    return Err(invalid(format!(
                        "selection '{name}' does not match molecule atom count"
                    )));
                }
                let style = document
                    .named_selection_styles
                    .get(name)
                    .copied()
                    .unwrap_or_default();
                Ok(wire::NamedSelectionV1 {
                    name: name.clone(),
                    expression: document.named_selection_expressions.get(name).cloned(),
                    flags: pack_atom_mask(selection.flags()),
                    style: Some(selection_style_to_wire(style)),
                })
            })
            .collect::<Result<Vec<_>, SceneError>>()?;

        Ok(Self {
            schema_version: SCENE_SCHEMA_VERSION,
            minimum_reader_version: minimum_reader_version(document),
            source_name: document.source_name.clone(),
            molecule: Some(molecule_to_wire(&document.molecule)?),
            display: Some(display_to_wire(&document.display)?),
            selections,
            measurements: document
                .measurement_lines
                .iter()
                .map(measurement_to_wire)
                .collect(),
            camera: Some(camera_to_wire(&document.camera)),
            hierarchy_names: document
                .hierarchy_names
                .iter()
                .map(|(target, name)| wire::HierarchyNameV1 {
                    target: Some(hierarchy_target_to_wire(*target)),
                    name: name.clone(),
                })
                .collect(),
            inspection: document.inspection.map(hierarchy_target_to_wire),
            hierarchy_selection: document
                .hierarchy_selection
                .iter()
                .copied()
                .map(hierarchy_target_to_wire)
                .collect(),
            hierarchy_selection_anchor: document
                .hierarchy_selection_anchor
                .map(hierarchy_target_to_wire),
            focus_description: document.focus_description.clone(),
            pivot_description: document.pivot_description.clone(),
        })
    }

    fn into_document(self) -> Result<SceneDocument, SceneError> {
        if self.schema_version != SCENE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedSchema {
                found: self.schema_version,
                supported: SCENE_SCHEMA_VERSION,
            });
        }
        if self.minimum_reader_version > SCENE_READER_VERSION {
            return Err(SceneError::UnsupportedReader {
                required: self.minimum_reader_version,
                supported: SCENE_READER_VERSION,
            });
        }
        let molecule = molecule_from_wire(
            self.molecule
                .ok_or_else(|| invalid("molecule section is missing"))?,
        )?;
        let hierarchy = MoleculeHierarchy::from_molecule(&molecule);
        let mut display = display_from_wire(
            self.display
                .ok_or_else(|| invalid("display section is missing"))?,
            &molecule,
        )?;
        let atom_count = molecule.atoms.len();

        let mut named_selections = BTreeMap::new();
        let mut named_selection_expressions = BTreeMap::new();
        let mut named_selection_styles = BTreeMap::new();
        for selection in self.selections {
            if selection.name.trim().is_empty() {
                return Err(invalid("selection name cannot be empty"));
            }
            if named_selections.contains_key(&selection.name) {
                return Err(invalid(format!(
                    "selection '{}' occurs more than once",
                    selection.name
                )));
            }
            let flags = unpack_flags(&selection.flags, atom_count, "selection")?;
            let style = selection_style_from_wire(selection.style.unwrap_or_default())?;
            if let Some(expression) = selection.expression {
                named_selection_expressions.insert(selection.name.clone(), expression);
            }
            named_selection_styles.insert(selection.name.clone(), style);
            named_selections.insert(selection.name, Selection::from_flags(flags));
        }
        let named_layers = named_selections
            .iter()
            .map(|(name, selection)| {
                (
                    selection.indices().collect(),
                    named_selection_styles
                        .get(name)
                        .copied()
                        .unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        display.replace_named_layers(&named_layers);

        let mut measurement_ids = std::collections::BTreeSet::new();
        let mut measurement_lines = Vec::with_capacity(self.measurements.len());
        for measurement in self.measurements {
            let line = measurement_from_wire(measurement)?;
            if !measurement_ids.insert(line.id) {
                return Err(invalid(format!(
                    "measurement line id {} occurs more than once",
                    line.id
                )));
            }
            measurement_lines.push(line);
        }

        let mut hierarchy_names = BTreeMap::new();
        for entry in self.hierarchy_names {
            let target = hierarchy_target_from_wire(
                entry
                    .target
                    .ok_or_else(|| invalid("hierarchy name target is missing"))?,
                &hierarchy,
                atom_count,
            )?;
            hierarchy_names.insert(target, entry.name);
        }
        let inspection = self
            .inspection
            .map(|target| hierarchy_target_from_wire(target, &hierarchy, atom_count))
            .transpose()?;
        let hierarchy_selection = self
            .hierarchy_selection
            .into_iter()
            .map(|target| hierarchy_target_from_wire(target, &hierarchy, atom_count))
            .collect::<Result<Vec<_>, _>>()?;
        let hierarchy_selection_anchor = self
            .hierarchy_selection_anchor
            .map(|target| hierarchy_target_from_wire(target, &hierarchy, atom_count))
            .transpose()?;

        Ok(SceneDocument {
            source_name: self.source_name,
            molecule,
            display,
            named_selections,
            named_selection_expressions,
            named_selection_styles,
            measurement_lines,
            hierarchy_names,
            inspection,
            hierarchy_selection,
            hierarchy_selection_anchor,
            focus_description: self.focus_description,
            pivot_description: self.pivot_description,
            camera: camera_from_wire(
                self.camera
                    .ok_or_else(|| invalid("camera section is missing"))?,
            )?,
        })
    }
}

fn molecule_to_wire(molecule: &Molecule) -> Result<wire::MoleculeV1, SceneError> {
    let atom_count = u32::try_from(molecule.atoms.len())
        .map_err(|_| invalid("molecule has more than u32::MAX atoms"))?;
    let mut strings = Vec::new();
    let mut string_indices = HashMap::<String, u32>::new();
    let mut name_indices = Vec::with_capacity(molecule.atoms.len());
    let mut residue_name_indices = Vec::with_capacity(molecule.atoms.len());
    let mut chain_indices = Vec::with_capacity(molecule.atoms.len());
    let mut positions = Vec::with_capacity(molecule.atoms.len() * 3);

    for atom in &molecule.atoms {
        name_indices.push(intern_string(
            &atom.name,
            &mut strings,
            &mut string_indices,
        )?);
        residue_name_indices.push(intern_string(
            &atom.residue_name,
            &mut strings,
            &mut string_indices,
        )?);
        chain_indices.push(intern_string(
            &atom.chain_id,
            &mut strings,
            &mut string_indices,
        )?);
        if !atom.position.is_finite() {
            return Err(invalid(format!(
                "atom #{} contains a non-finite position",
                atom.serial
            )));
        }
        positions.extend_from_slice(&atom.position.to_array());
    }

    let mut bond_endpoints = Vec::with_capacity(molecule.bonds.len() * 2);
    for bond in &molecule.bonds {
        bond_endpoints
            .push(u32::try_from(bond.a).map_err(|_| invalid("bond atom index exceeds u32::MAX"))?);
        bond_endpoints
            .push(u32::try_from(bond.b).map_err(|_| invalid("bond atom index exceeds u32::MAX"))?);
    }

    Ok(wire::MoleculeV1 {
        atom_count,
        strings,
        serials: molecule.atoms.iter().map(|atom| atom.serial).collect(),
        name_indices,
        elements: molecule
            .atoms
            .iter()
            .map(|atom| element_code(atom.element))
            .collect(),
        residue_name_indices,
        residue_numbers: molecule
            .atoms
            .iter()
            .map(|atom| atom.residue_number)
            .collect(),
        insertion_codes: molecule
            .atoms
            .iter()
            .map(|atom| atom.insertion_code.map_or(0, |code| u32::from(code) + 1))
            .collect(),
        chain_indices,
        positions,
        occupancies: molecule.atoms.iter().map(|atom| atom.occupancy).collect(),
        b_factors: molecule.atoms.iter().map(|atom| atom.b_factor).collect(),
        hetero_flags: pack_flags(
            &molecule
                .atoms
                .iter()
                .map(|atom| atom.hetero)
                .collect::<Vec<_>>(),
        ),
        bond_endpoints,
    })
}

fn molecule_from_wire(molecule: wire::MoleculeV1) -> Result<Molecule, SceneError> {
    let atom_count = usize::try_from(molecule.atom_count)
        .map_err(|_| invalid("atom count does not fit this platform"))?;
    require_len(&molecule.serials, atom_count, "atom serials")?;
    require_len(&molecule.name_indices, atom_count, "atom names")?;
    require_len(&molecule.elements, atom_count, "atom elements")?;
    require_len(&molecule.residue_name_indices, atom_count, "residue names")?;
    require_len(&molecule.residue_numbers, atom_count, "residue numbers")?;
    require_len(&molecule.insertion_codes, atom_count, "insertion codes")?;
    require_len(&molecule.chain_indices, atom_count, "chain ids")?;
    require_len(&molecule.positions, atom_count * 3, "atom positions")?;
    require_len(&molecule.occupancies, atom_count, "atom occupancies")?;
    require_len(&molecule.b_factors, atom_count, "atom B-factors")?;
    let hetero_flags = unpack_flags(&molecule.hetero_flags, atom_count, "hetero")?;

    let mut atoms = Vec::with_capacity(atom_count);
    for (index, &hetero) in hetero_flags.iter().enumerate() {
        let position = Vec3::from_array([
            molecule.positions[index * 3],
            molecule.positions[index * 3 + 1],
            molecule.positions[index * 3 + 2],
        ]);
        if !position.is_finite()
            || !molecule.occupancies[index].is_finite()
            || !molecule.b_factors[index].is_finite()
        {
            return Err(invalid(format!(
                "atom at index {index} contains a non-finite number"
            )));
        }
        let insertion_code = match molecule.insertion_codes[index] {
            0 => None,
            value => Some(
                char::from_u32(value - 1)
                    .ok_or_else(|| invalid(format!("invalid insertion code at atom {index}")))?,
            ),
        };
        atoms.push(Atom {
            serial: molecule.serials[index],
            name: table_string(&molecule.strings, molecule.name_indices[index], "atom name")?,
            element: element_from_code(molecule.elements[index])?,
            residue_name: table_string(
                &molecule.strings,
                molecule.residue_name_indices[index],
                "residue name",
            )?,
            residue_number: molecule.residue_numbers[index],
            insertion_code,
            chain_id: table_string(&molecule.strings, molecule.chain_indices[index], "chain id")?,
            position,
            occupancy: molecule.occupancies[index],
            b_factor: molecule.b_factors[index],
            hetero,
        });
    }

    let chunks = molecule.bond_endpoints.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(invalid("bond endpoint list has an odd length"));
    }
    let mut bonds = Vec::with_capacity(molecule.bond_endpoints.len() / 2);
    for pair in chunks {
        let a = pair[0] as usize;
        let b = pair[1] as usize;
        if a >= atom_count || b >= atom_count {
            return Err(invalid("bond refers to an atom outside the molecule"));
        }
        bonds.push(Bond::new(a, b).ok_or_else(|| invalid("bond connects an atom to itself"))?);
    }
    Ok(Molecule { atoms, bonds })
}

fn display_to_wire(display: &DisplayState) -> Result<wire::DisplayV1, SceneError> {
    Ok(wire::DisplayV1 {
        global_mode: display_mode_code(display.global_mode),
        coloring_mode: coloring_mode_code(display.coloring_mode),
        uniform_color: display.uniform_color.to_vec(),
        ambient_occlusion: Some(ambient_occlusion_to_wire(display.ambient_occlusion)),
        ambient_occlusion_customized: display.ambient_occlusion_customized,
        representation_masks: display
            .representations
            .iter()
            .map(|mask| u32::from(mask.bits()))
            .collect(),
        selection_flags: pack_atom_mask(&display.selection),
        chain_colors: color_runs(&display.color_override_values(DisplayLevel::Chain))?,
        residue_colors: color_runs(&display.color_override_values(DisplayLevel::Residue))?,
        atom_colors: color_runs(&display.color_override_values(DisplayLevel::Atom))?,
        chain_visibility: state_runs(
            &display.visibility_override_values(DisplayLevel::Chain),
            visibility_code,
        )?,
        residue_visibility: state_runs(
            &display.visibility_override_values(DisplayLevel::Residue),
            visibility_code,
        )?,
        atom_visibility: state_runs(
            &display.visibility_override_values(DisplayLevel::Atom),
            visibility_code,
        )?,
        chain_modes: state_runs(
            &display.mode_override_values(DisplayLevel::Chain),
            mode_override_code,
        )?,
        residue_modes: state_runs(
            &display.mode_override_values(DisplayLevel::Residue),
            mode_override_code,
        )?,
        atom_modes: state_runs(
            &display.mode_override_values(DisplayLevel::Atom),
            mode_override_code,
        )?,
    })
}

fn display_from_wire(
    wire: wire::DisplayV1,
    molecule: &Molecule,
) -> Result<DisplayState, SceneError> {
    let atom_count = molecule.atoms.len();
    require_len(
        &wire.representation_masks,
        atom_count,
        "representation masks",
    )?;
    let mut display = DisplayState::for_molecule(molecule);
    display.global_mode = display_mode_from_code(wire.global_mode)?;
    display.uniform_color = checked_color(&wire.uniform_color, "uniform color")?;
    display.set_coloring_mode(molecule, coloring_mode_from_code(wire.coloring_mode)?);
    display.ambient_occlusion = ambient_occlusion_from_wire(
        wire.ambient_occlusion
            .ok_or_else(|| invalid("ambient occlusion settings are missing"))?,
    )?;
    display.ambient_occlusion_customized = wire.ambient_occlusion_customized;
    let representations = wire
        .representation_masks
        .into_iter()
        .map(|bits| {
            if bits & !0b11 != 0 {
                Err(invalid(format!(
                    "unknown atom representation mask bits {bits:#x}"
                )))
            } else {
                Ok(RepresentationMask::from_bits(bits as u8))
            }
        })
        .collect::<Result<Vec<_>, SceneError>>()?;
    display.load_representations(representations);
    display.set_selection_from_bools(unpack_flags(
        &wire.selection_flags,
        atom_count,
        "current selection",
    )?);
    display.load_color_override_values(
        DisplayLevel::Chain,
        colors_from_runs(&wire.chain_colors, atom_count, "chain colors")?,
    );
    display.load_color_override_values(
        DisplayLevel::Residue,
        colors_from_runs(&wire.residue_colors, atom_count, "residue colors")?,
    );
    display.load_color_override_values(
        DisplayLevel::Atom,
        colors_from_runs(&wire.atom_colors, atom_count, "atom colors")?,
    );
    display.load_visibility_override_values(
        DisplayLevel::Chain,
        states_from_runs(
            &wire.chain_visibility,
            atom_count,
            VisibilityOverride::Inherit,
            visibility_from_code,
            "chain visibility",
        )?,
    );
    display.load_visibility_override_values(
        DisplayLevel::Residue,
        states_from_runs(
            &wire.residue_visibility,
            atom_count,
            VisibilityOverride::Inherit,
            visibility_from_code,
            "residue visibility",
        )?,
    );
    display.load_visibility_override_values(
        DisplayLevel::Atom,
        states_from_runs(
            &wire.atom_visibility,
            atom_count,
            VisibilityOverride::Inherit,
            visibility_from_code,
            "atom visibility",
        )?,
    );
    display.load_mode_override_values(
        DisplayLevel::Chain,
        states_from_runs(
            &wire.chain_modes,
            atom_count,
            ModeOverride::Inherit,
            mode_override_from_code,
            "chain modes",
        )?,
    );
    display.load_mode_override_values(
        DisplayLevel::Residue,
        states_from_runs(
            &wire.residue_modes,
            atom_count,
            ModeOverride::Inherit,
            mode_override_from_code,
            "residue modes",
        )?,
    );
    display.load_mode_override_values(
        DisplayLevel::Atom,
        states_from_runs(
            &wire.atom_modes,
            atom_count,
            ModeOverride::Inherit,
            mode_override_from_code,
            "atom modes",
        )?,
    );
    display.recompute_colors();
    display.recompute_visibility();
    display.recompute_modes();
    Ok(display)
}

fn selection_style_to_wire(style: NamedSelectionStyle) -> wire::SelectionStyleV1 {
    wire::SelectionStyleV1 {
        color: style.color.map_or_else(Vec::new, |color| color.to_vec()),
        visibility: visibility_code(style.visibility) as i32,
        mode: mode_override_code(style.mode) as i32,
    }
}

fn selection_style_from_wire(
    style: wire::SelectionStyleV1,
) -> Result<NamedSelectionStyle, SceneError> {
    Ok(NamedSelectionStyle {
        color: (!style.color.is_empty())
            .then(|| checked_color(&style.color, "selection color"))
            .transpose()?,
        visibility: visibility_from_code(enum_code(style.visibility, "visibility override")?)?,
        mode: mode_override_from_code(enum_code(style.mode, "mode override")?)?,
    })
}

fn measurement_to_wire(line: &MeasurementLine) -> wire::MeasurementLineV1 {
    wire::MeasurementLineV1 {
        id: line.id,
        name: line.name.clone(),
        first: Some(measurement_endpoint_to_wire(&line.first)),
        second: Some(measurement_endpoint_to_wire(&line.second)),
        color: line.color.map_or_else(Vec::new, |color| color.to_vec()),
        visibility: visibility_code(line.visibility) as i32,
        label_size: line.label_size,
        thickness: line.thickness,
    }
}

fn measurement_endpoint_to_wire(endpoint: &MeasurementEndpoint) -> wire::MeasurementEndpointV1 {
    wire::MeasurementEndpointV1 {
        position: Some(vec3(endpoint.position)),
        description: endpoint.description.clone(),
    }
}

fn measurement_from_wire(line: wire::MeasurementLineV1) -> Result<MeasurementLine, SceneError> {
    let first = measurement_endpoint_from_wire(
        line.first
            .ok_or_else(|| invalid("measurement first endpoint is missing"))?,
        "measurement first endpoint",
    )?;
    let second = measurement_endpoint_from_wire(
        line.second
            .ok_or_else(|| invalid("measurement second endpoint is missing"))?,
        "measurement second endpoint",
    )?;
    let mut result = MeasurementLine::new(line.id, first, second);
    result.name = line.name;
    result.color = (!line.color.is_empty())
        .then(|| checked_color(&line.color, "measurement color"))
        .transpose()?;
    result.visibility = visibility_from_code(enum_code(line.visibility, "visibility override")?)?;
    result.set_label_size(checked_finite(line.label_size, "measurement label size")?);
    result.set_thickness(checked_finite(line.thickness, "measurement thickness")?);
    Ok(result)
}

fn measurement_endpoint_from_wire(
    endpoint: wire::MeasurementEndpointV1,
    label: &str,
) -> Result<MeasurementEndpoint, SceneError> {
    Ok(MeasurementEndpoint {
        position: checked_vec3(endpoint.position, label)?,
        description: endpoint.description,
    })
}

fn camera_to_wire(camera: &OrbitCamera) -> wire::CameraV1 {
    wire::CameraV1 {
        target: Some(vec3(camera.target)),
        distance: camera.distance,
        yaw: camera.yaw,
        pitch: camera.pitch,
        field_of_view_y: camera.field_of_view_y,
        near: camera.near,
        far: camera.far,
        depth_of_field: Some(wire::DepthOfFieldV1 {
            enabled: camera.depth_of_field.enabled,
            focus_point: Some(vec3(camera.depth_of_field.focus_point)),
            focal_length_mm: camera.depth_of_field.focal_length_mm,
            sensor_height_mm: camera.depth_of_field.sensor_height_mm,
            f_stop: camera.depth_of_field.f_stop,
            blade_count: camera.depth_of_field.blade_count,
            blade_rotation: camera.depth_of_field.blade_rotation,
            max_coc_pixels: camera.depth_of_field.max_coc_pixels,
        }),
    }
}

fn camera_from_wire(camera: wire::CameraV1) -> Result<OrbitCamera, SceneError> {
    let depth = camera
        .depth_of_field
        .ok_or_else(|| invalid("depth of field settings are missing"))?;
    let distance = checked_finite(camera.distance, "camera distance")?;
    let yaw = checked_finite(camera.yaw, "camera yaw")?;
    let pitch = checked_finite(camera.pitch, "camera pitch")?;
    let field_of_view_y = checked_finite(camera.field_of_view_y, "camera field of view")?;
    let near = checked_finite(camera.near, "near clipping plane")?;
    let far = checked_finite(camera.far, "far clipping plane")?;
    if distance <= 0.0
        || !(0.0..std::f32::consts::PI).contains(&field_of_view_y)
        || near <= 0.0
        || far <= near
    {
        return Err(invalid(
            "camera projection parameters are outside valid ranges",
        ));
    }
    let focal_length_mm = checked_finite(depth.focal_length_mm, "focal length")?;
    let sensor_height_mm = checked_finite(depth.sensor_height_mm, "sensor height")?;
    let f_stop = checked_finite(depth.f_stop, "f-stop")?;
    let blade_rotation = checked_finite(depth.blade_rotation, "blade rotation")?;
    let max_coc_pixels = checked_finite(depth.max_coc_pixels, "maximum CoC")?;
    if focal_length_mm <= 0.0 || sensor_height_mm <= 0.0 || f_stop <= 0.0 || max_coc_pixels <= 0.0 {
        return Err(invalid(
            "depth of field parameters are outside valid ranges",
        ));
    }
    let mut result = OrbitCamera::new(1.0);
    result.target = checked_vec3(camera.target, "camera target")?;
    result.distance = distance;
    result.yaw = yaw;
    result.pitch = pitch;
    result.field_of_view_y = field_of_view_y;
    result.near = near;
    result.far = far;
    result.depth_of_field = DepthOfField {
        enabled: depth.enabled,
        focus_point: checked_vec3(depth.focus_point, "depth of field focus point")?,
        focal_length_mm,
        sensor_height_mm,
        f_stop,
        blade_count: depth.blade_count,
        blade_rotation,
        max_coc_pixels,
    };
    Ok(result)
}

fn hierarchy_target_to_wire(target: SceneHierarchyTarget) -> wire::HierarchyTargetV1 {
    match target {
        SceneHierarchyTarget::Chain(index) => wire::HierarchyTargetV1 {
            kind: 0,
            first: index as u32,
            second: 0,
        },
        SceneHierarchyTarget::Residue {
            chain_index,
            residue_index,
        } => wire::HierarchyTargetV1 {
            kind: 1,
            first: chain_index as u32,
            second: residue_index as u32,
        },
        SceneHierarchyTarget::Atom(index) => wire::HierarchyTargetV1 {
            kind: 2,
            first: index as u32,
            second: 0,
        },
    }
}

fn hierarchy_target_from_wire(
    target: wire::HierarchyTargetV1,
    hierarchy: &MoleculeHierarchy,
    atom_count: usize,
) -> Result<SceneHierarchyTarget, SceneError> {
    let first = target.first as usize;
    let second = target.second as usize;
    match target.kind {
        0 if hierarchy.chains.get(first).is_some() => Ok(SceneHierarchyTarget::Chain(first)),
        1 if hierarchy.residue(first, second).is_some() => Ok(SceneHierarchyTarget::Residue {
            chain_index: first,
            residue_index: second,
        }),
        2 if first < atom_count => Ok(SceneHierarchyTarget::Atom(first)),
        0..=2 => Err(invalid("hierarchy target is outside the molecule")),
        kind => Err(invalid(format!("unknown hierarchy target kind {kind}"))),
    }
}

fn ambient_occlusion_to_wire(settings: AmbientOcclusionSettings) -> wire::AmbientOcclusionV1 {
    wire::AmbientOcclusionV1 {
        enabled: settings.enabled,
        strength: settings.strength,
        radius: settings.radius,
        bias: settings.bias,
        quality: ao_quality_code(settings.quality),
    }
}

fn ambient_occlusion_from_wire(
    settings: wire::AmbientOcclusionV1,
) -> Result<AmbientOcclusionSettings, SceneError> {
    Ok(AmbientOcclusionSettings {
        enabled: settings.enabled,
        strength: checked_finite(settings.strength, "AO strength")?,
        radius: checked_finite(settings.radius, "AO radius")?,
        bias: checked_finite(settings.bias, "AO bias")?,
        quality: ao_quality_from_code(settings.quality)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DisplayLevel;

    fn document() -> SceneDocument {
        let molecule = Molecule {
            atoms: vec![
                Atom {
                    serial: 10,
                    name: "CA".into(),
                    element: Element::C,
                    residue_name: "GLY".into(),
                    residue_number: 1,
                    insertion_code: None,
                    chain_id: "A".into(),
                    position: Vec3::new(1.0, 2.0, 3.0),
                    occupancy: 1.0,
                    b_factor: 12.5,
                    hetero: false,
                },
                Atom {
                    serial: 11,
                    name: "O".into(),
                    element: Element::O,
                    residue_name: "GLY".into(),
                    residue_number: 1,
                    insertion_code: Some('A'),
                    chain_id: "A".into(),
                    position: Vec3::new(2.0, 2.0, 3.0),
                    occupancy: 0.75,
                    b_factor: 18.0,
                    hetero: true,
                },
            ],
            bonds: vec![Bond::new(0, 1).unwrap()],
        };
        let mut display = DisplayState::for_molecule(&molecule);
        display.set_coloring_mode(&molecule, ColoringMode::SecondaryStructure);
        display.set_global_mode(DisplayMode::BallAndStick);
        display.set_color_override(&[0, 1], DisplayLevel::Chain, Some([0.2, 0.4, 0.8, 1.0]));
        display.set_color_override(&[1], DisplayLevel::Atom, Some([0.9, 0.2, 0.1, 1.0]));
        display.set_visibility_override(&[1], DisplayLevel::Atom, VisibilityOverride::Hide);
        display.set_mode_override(&[0], DisplayLevel::Atom, ModeOverride::Toon);
        display.set_selection_from_bools(vec![true, false]);
        display.set_ambient_occlusion(AmbientOcclusionSettings {
            enabled: true,
            strength: 1.4,
            radius: 2.3,
            bias: 0.05,
            quality: AmbientOcclusionQuality::Medium,
        });

        let named_selection = Selection::from_flags(vec![true, false]);
        let named_style = NamedSelectionStyle {
            color: Some([0.3, 0.8, 0.5, 1.0]),
            visibility: VisibilityOverride::Show,
            mode: ModeOverride::Toon,
        };
        display.replace_named_layers(&[(vec![0], named_style)]);
        let line = MeasurementLine {
            id: 7,
            name: "Catalytic distance".into(),
            first: MeasurementEndpoint {
                position: molecule.atoms[0].position,
                description: "first".into(),
            },
            second: MeasurementEndpoint {
                position: molecule.atoms[1].position,
                description: "second".into(),
            },
            color: Some([1.0, 0.5, 0.1, 1.0]),
            visibility: VisibilityOverride::Show,
            label_size: 20.0,
            thickness: 0.2,
        };
        let mut camera = OrbitCamera::new(16.0 / 9.0);
        camera.target = Vec3::new(1.5, 2.0, 3.0);
        camera.distance = 25.0;
        camera.yaw = 0.4;
        camera.pitch = -0.2;
        camera.depth_of_field.enabled = true;
        camera.depth_of_field.focus_point = molecule.atoms[0].position;

        SceneDocument {
            source_name: "example.cif".into(),
            molecule,
            display,
            named_selections: BTreeMap::from([("active".into(), named_selection)]),
            named_selection_expressions: BTreeMap::from([("active".into(), "Chain A/GLY*".into())]),
            named_selection_styles: BTreeMap::from([("active".into(), named_style)]),
            measurement_lines: vec![line],
            hierarchy_names: BTreeMap::from([(
                SceneHierarchyTarget::Atom(0),
                "Catalytic carbon".into(),
            )]),
            inspection: Some(SceneHierarchyTarget::Atom(0)),
            hierarchy_selection: vec![SceneHierarchyTarget::Atom(0)],
            hierarchy_selection_anchor: Some(SceneHierarchyTarget::Atom(0)),
            focus_description: "Atom #10 CA".into(),
            pivot_description: "Residue GLY 1".into(),
            camera,
        }
    }

    #[test]
    fn scene_round_trip_preserves_geometry_formatting_and_workspace() {
        let original = document();
        let bytes = encode(&original).unwrap();
        assert_eq!(SCENE_EXTENSION, "mol");
        assert_eq!(SCENE_FORMAT_NAME, "Molecule 1.0");
        assert_eq!(&bytes[..8], b"MOLECULE");
        assert!(is_scene_document(&bytes));
        let restored = decode(&bytes).unwrap();

        assert_eq!(restored.source_name, original.source_name);
        assert_eq!(restored.molecule.atoms, original.molecule.atoms);
        assert_eq!(restored.molecule.bonds, original.molecule.bonds);
        assert_eq!(restored.display, original.display);
        assert_eq!(restored.named_selections, original.named_selections);
        assert_eq!(
            restored.named_selection_expressions,
            original.named_selection_expressions
        );
        assert_eq!(
            restored.named_selection_styles,
            original.named_selection_styles
        );
        assert_eq!(restored.measurement_lines, original.measurement_lines);
        assert_eq!(restored.hierarchy_names, original.hierarchy_names);
        assert_eq!(restored.inspection, original.inspection);
        assert_eq!(restored.hierarchy_selection, original.hierarchy_selection);
        assert_eq!(
            restored.hierarchy_selection_anchor,
            original.hierarchy_selection_anchor
        );
        assert_eq!(restored.camera.target, original.camera.target);
        assert_eq!(restored.camera.distance, original.camera.distance);
        assert_eq!(
            restored.camera.depth_of_field,
            original.camera.depth_of_field
        );
    }

    #[test]
    fn future_schema_is_rejected_before_decompression() {
        let mut bytes = encode(&document()).unwrap();
        bytes[12..16].copy_from_slice(&(SCENE_SCHEMA_VERSION + 1).to_le_bytes());
        assert!(matches!(
            decode(&bytes),
            Err(SceneError::UnsupportedSchema { .. })
        ));
    }

    #[test]
    fn newer_required_reader_has_a_clear_error() {
        let mut message = wire::SceneV1::from_document(&document()).unwrap();
        message.minimum_reader_version = SCENE_READER_VERSION + 1;
        let bytes = test_container(message.encode_to_vec());
        assert!(matches!(
            decode(&bytes),
            Err(SceneError::UnsupportedReader {
                required,
                supported: SCENE_READER_VERSION
            }) if required == SCENE_READER_VERSION + 1
        ));
    }

    #[test]
    fn corrupted_compressed_payload_is_rejected() {
        let mut bytes = encode(&document()).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x80;
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn reader_1_golden_opens_in_current_reader() {
        let restored =
            decode(include_bytes!("../tests/fixtures/molecule_1_0_reader1.mol")).unwrap();
        assert_eq!(restored.source_name, "example.cif");
        assert_eq!(restored.display.coloring_mode, ColoringMode::Chain);
    }

    #[test]
    fn compatible_current_file_opens_in_frozen_reader_1() {
        let mut compatible = document();
        compatible
            .display
            .set_coloring_mode(&compatible.molecule, ColoringMode::Chain);
        let message = legacy_reader_message(&encode(&compatible).unwrap()).unwrap();
        assert_eq!(message.schema_version, 1);
        assert_eq!(message.minimum_reader_version, 1);
        assert_eq!(message.display.unwrap().coloring_mode, 1);
    }

    #[test]
    fn incompatible_current_file_is_rejected_by_frozen_reader_1() {
        let bytes = encode(&document()).unwrap();
        let error = legacy_reader_message(&bytes).unwrap_err();
        assert!(error.contains("requires reader 2"));
    }

    #[test]
    fn every_display_and_coloring_mode_round_trips() {
        let display_modes = [
            DisplayMode::Cartoon,
            DisplayMode::BallAndStick,
            DisplayMode::Toon,
        ];
        let coloring_modes = [
            ColoringMode::Element,
            ColoringMode::Chain,
            ColoringMode::Residue,
            ColoringMode::ResidueType,
            ColoringMode::BFactor,
            ColoringMode::Uniform,
            ColoringMode::SecondaryStructure,
        ];
        for display_mode in display_modes {
            for coloring_mode in coloring_modes {
                let mut original = document();
                original.display.set_global_mode(display_mode);
                original
                    .display
                    .set_coloring_mode(&original.molecule, coloring_mode);
                let restored = decode(&encode(&original).unwrap()).unwrap();
                assert_eq!(restored.display.global_mode, display_mode);
                assert_eq!(restored.display.coloring_mode, coloring_mode);
            }
        }
    }

    #[test]
    fn oversized_payload_is_rejected_from_header() {
        let mut bytes = encode(&document()).unwrap();
        bytes[16..24].copy_from_slice(&(MAX_DECOMPRESSED_SIZE + 1).to_le_bytes());
        assert!(matches!(decode(&bytes), Err(SceneError::TooLarge(_))));
    }

    fn legacy_reader_message(bytes: &[u8]) -> Result<legacy_wire_v1::SceneV1, String> {
        if !bytes.starts_with(MAGIC) || bytes.len() < HEADER_LEN {
            return Err("invalid Molecule container".into());
        }
        let expected = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let decoder = zstd::stream::read::Decoder::new(Cursor::new(&bytes[HEADER_LEN..]))
            .map_err(|error| error.to_string())?;
        let mut payload = Vec::new();
        decoder
            .take(expected as u64 + 1)
            .read_to_end(&mut payload)
            .map_err(|error| error.to_string())?;
        if payload.len() != expected {
            return Err("invalid payload length".into());
        }
        let message = legacy_wire_v1::SceneV1::decode(payload.as_slice())
            .map_err(|error| error.to_string())?;
        if message.minimum_reader_version > 1 {
            return Err(format!(
                "file requires reader {} but frozen reader supports 1",
                message.minimum_reader_version
            ));
        }
        Ok(message)
    }

    fn test_container(payload: Vec<u8>) -> Vec<u8> {
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
        bytes.extend_from_slice(&COMPRESSION_ZSTD.to_le_bytes());
        bytes.extend_from_slice(&SCENE_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&compressed);
        bytes
    }
}
