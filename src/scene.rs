//! Portable, versioned scene documents.
//!
//! The wire representation is deliberately separate from the runtime model. This keeps file
//! compatibility independent of Rust layout, wgpu, egui, and implementation-only cached state.

use std::{
    collections::{BTreeMap, HashMap},
    io::{Cursor, Read, Write},
};

use glam::Vec3;
use prost::Message;
use thiserror::Error;

use crate::{
    AmbientOcclusionQuality, AmbientOcclusionSettings, ColoringMode, DisplayColor, DisplayMode,
    DisplayState, ModeOverride, NamedSelectionStyle, RepresentationMask, VisibilityOverride,
    camera::{DepthOfField, OrbitCamera},
    measurement::{MeasurementEndpoint, MeasurementLine},
    molecule::{Atom, Bond, Element, Molecule, MoleculeHierarchy},
    selection::Selection,
};

pub const SCENE_EXTENSION: &str = "mol";
pub const SCENE_FORMAT_NAME: &str = "Molecule 1.0";
pub const SCENE_SCHEMA_VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"MOLECULE";
const CONTAINER_VERSION: u16 = 1;
const COMPRESSION_ZSTD: u16 = 1;
const HEADER_LEN: usize = 24;
const MAX_DECOMPRESSED_SIZE: u64 = 2 * 1024 * 1024 * 1024;

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

pub fn is_scene_document(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

pub fn encode(document: &SceneDocument) -> Result<Vec<u8>, SceneError> {
    let message = wire::SceneV1::from_document(document)?;
    let payload = message.encode_to_vec();
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| SceneError::InvalidData("scene payload does not fit in u64".into()))?;
    if payload_len > MAX_DECOMPRESSED_SIZE {
        return Err(SceneError::TooLarge(payload_len));
    }

    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 7)?;
    encoder.include_checksum(true)?;
    encoder.include_contentsize(true)?;
    encoder.set_pledged_src_size(Some(payload_len))?;
    encoder.write_all(&payload)?;
    let compressed = encoder.finish()?;

    let mut output = Vec::with_capacity(HEADER_LEN + compressed.len());
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
    output.extend_from_slice(&COMPRESSION_ZSTD.to_le_bytes());
    output.extend_from_slice(&SCENE_SCHEMA_VERSION.to_le_bytes());
    output.extend_from_slice(&payload_len.to_le_bytes());
    output.extend_from_slice(&compressed);
    Ok(output)
}

pub fn decode(bytes: &[u8]) -> Result<SceneDocument, SceneError> {
    if bytes.len() < MAGIC.len() {
        return Err(SceneError::TruncatedHeader);
    }
    if !is_scene_document(bytes) {
        return Err(SceneError::InvalidMagic);
    }
    if bytes.len() < HEADER_LEN {
        return Err(SceneError::TruncatedHeader);
    }
    let container_version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if container_version != CONTAINER_VERSION {
        return Err(SceneError::UnsupportedContainer(container_version));
    }
    let compression = u16::from_le_bytes([bytes[10], bytes[11]]);
    if compression != COMPRESSION_ZSTD {
        return Err(SceneError::UnsupportedCompression(compression));
    }
    let schema_version = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| {
        SceneError::InvalidData("schema version is missing from the header".into())
    })?);
    if schema_version > SCENE_SCHEMA_VERSION || schema_version == 0 {
        return Err(SceneError::UnsupportedSchema {
            found: schema_version,
            supported: SCENE_SCHEMA_VERSION,
        });
    }
    let expected_len = u64::from_le_bytes(bytes[16..24].try_into().map_err(|_| {
        SceneError::InvalidData("payload length is missing from the header".into())
    })?);
    if expected_len > MAX_DECOMPRESSED_SIZE {
        return Err(SceneError::TooLarge(expected_len));
    }

    let decoder = zstd::stream::read::Decoder::new(Cursor::new(&bytes[HEADER_LEN..]))?;
    let mut limited = decoder.take(expected_len.saturating_add(1));
    let mut payload = Vec::with_capacity(usize::try_from(expected_len).unwrap_or(0));
    limited.read_to_end(&mut payload)?;
    let actual_len = payload.len() as u64;
    if actual_len != expected_len {
        return Err(SceneError::LengthMismatch {
            expected: expected_len,
            actual: actual_len,
        });
    }

    match schema_version {
        1 => wire::SceneV1::decode(payload.as_slice())?.into_document(),
        version => Err(SceneError::UnsupportedSchema {
            found: version,
            supported: SCENE_SCHEMA_VERSION,
        }),
    }
}

fn invalid(message: impl Into<String>) -> SceneError {
    SceneError::InvalidData(message.into())
}

fn pack_flags(flags: &[bool]) -> Vec<u8> {
    let mut bytes = vec![0; flags.len().div_ceil(8)];
    for (index, flag) in flags.iter().copied().enumerate() {
        if flag {
            bytes[index / 8] |= 1 << (index % 8);
        }
    }
    bytes
}

fn unpack_flags(bytes: &[u8], count: usize, label: &str) -> Result<Vec<bool>, SceneError> {
    let expected = count.div_ceil(8);
    if bytes.len() != expected {
        return Err(invalid(format!(
            "{label} bitset has {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    Ok((0..count)
        .map(|index| bytes[index / 8] & (1 << (index % 8)) != 0)
        .collect())
}

fn checked_vec3(vector: Option<wire::Vec3V1>, label: &str) -> Result<Vec3, SceneError> {
    let vector = vector.ok_or_else(|| invalid(format!("{label} is missing")))?;
    if !vector.x.is_finite() || !vector.y.is_finite() || !vector.z.is_finite() {
        return Err(invalid(format!("{label} contains a non-finite coordinate")));
    }
    Ok(Vec3::new(vector.x, vector.y, vector.z))
}

fn vec3(vector: Vec3) -> wire::Vec3V1 {
    wire::Vec3V1 {
        x: vector.x,
        y: vector.y,
        z: vector.z,
    }
}

fn checked_color(values: &[f32], label: &str) -> Result<DisplayColor, SceneError> {
    let values: [f32; 4] = values
        .try_into()
        .map_err(|_| invalid(format!("{label} must contain four components")))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{label} contains a non-finite component")));
    }
    Ok(values.map(|value| value.clamp(0.0, 1.0)))
}

fn checked_finite(value: f32, label: &str) -> Result<f32, SceneError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| invalid(format!("{label} is not finite")))
}

mod wire {
    use super::*;

    #[derive(Clone, PartialEq, Message)]
    pub struct SceneV1 {
        #[prost(uint32, tag = "1")]
        pub schema_version: u32,
        #[prost(uint32, tag = "2")]
        pub minimum_reader_version: u32,
        #[prost(string, tag = "3")]
        pub source_name: String,
        #[prost(message, optional, tag = "4")]
        pub molecule: Option<MoleculeV1>,
        #[prost(message, optional, tag = "5")]
        pub display: Option<DisplayV1>,
        #[prost(message, repeated, tag = "6")]
        pub selections: Vec<NamedSelectionV1>,
        #[prost(message, repeated, tag = "7")]
        pub measurements: Vec<MeasurementLineV1>,
        #[prost(message, optional, tag = "8")]
        pub camera: Option<CameraV1>,
        #[prost(message, repeated, tag = "9")]
        pub hierarchy_names: Vec<HierarchyNameV1>,
        #[prost(message, optional, tag = "10")]
        pub inspection: Option<HierarchyTargetV1>,
        #[prost(message, repeated, tag = "11")]
        pub hierarchy_selection: Vec<HierarchyTargetV1>,
        #[prost(message, optional, tag = "12")]
        pub hierarchy_selection_anchor: Option<HierarchyTargetV1>,
        #[prost(string, tag = "13")]
        pub focus_description: String,
        #[prost(string, tag = "14")]
        pub pivot_description: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct MoleculeV1 {
        #[prost(uint32, tag = "1")]
        pub atom_count: u32,
        #[prost(string, repeated, tag = "2")]
        pub strings: Vec<String>,
        #[prost(uint32, repeated, packed = "true", tag = "3")]
        pub serials: Vec<u32>,
        #[prost(uint32, repeated, packed = "true", tag = "4")]
        pub name_indices: Vec<u32>,
        #[prost(uint32, repeated, packed = "true", tag = "5")]
        pub elements: Vec<u32>,
        #[prost(uint32, repeated, packed = "true", tag = "6")]
        pub residue_name_indices: Vec<u32>,
        #[prost(sint32, repeated, packed = "true", tag = "7")]
        pub residue_numbers: Vec<i32>,
        #[prost(uint32, repeated, packed = "true", tag = "8")]
        pub insertion_codes: Vec<u32>,
        #[prost(uint32, repeated, packed = "true", tag = "9")]
        pub chain_indices: Vec<u32>,
        #[prost(float, repeated, packed = "true", tag = "10")]
        pub positions: Vec<f32>,
        #[prost(float, repeated, packed = "true", tag = "11")]
        pub occupancies: Vec<f32>,
        #[prost(float, repeated, packed = "true", tag = "12")]
        pub b_factors: Vec<f32>,
        #[prost(bytes = "vec", tag = "13")]
        pub hetero_flags: Vec<u8>,
        #[prost(uint32, repeated, packed = "true", tag = "14")]
        pub bond_endpoints: Vec<u32>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct DisplayV1 {
        #[prost(uint32, tag = "1")]
        pub global_mode: u32,
        #[prost(uint32, tag = "2")]
        pub coloring_mode: u32,
        #[prost(float, repeated, packed = "true", tag = "3")]
        pub uniform_color: Vec<f32>,
        #[prost(message, optional, tag = "4")]
        pub ambient_occlusion: Option<AmbientOcclusionV1>,
        #[prost(bool, tag = "5")]
        pub ambient_occlusion_customized: bool,
        #[prost(uint32, repeated, packed = "true", tag = "6")]
        pub representation_masks: Vec<u32>,
        #[prost(bytes = "vec", tag = "7")]
        pub selection_flags: Vec<u8>,
        #[prost(message, repeated, tag = "10")]
        pub chain_colors: Vec<ColorRunV1>,
        #[prost(message, repeated, tag = "11")]
        pub residue_colors: Vec<ColorRunV1>,
        #[prost(message, repeated, tag = "12")]
        pub atom_colors: Vec<ColorRunV1>,
        #[prost(message, repeated, tag = "20")]
        pub chain_visibility: Vec<StateRunV1>,
        #[prost(message, repeated, tag = "21")]
        pub residue_visibility: Vec<StateRunV1>,
        #[prost(message, repeated, tag = "22")]
        pub atom_visibility: Vec<StateRunV1>,
        #[prost(message, repeated, tag = "30")]
        pub chain_modes: Vec<StateRunV1>,
        #[prost(message, repeated, tag = "31")]
        pub residue_modes: Vec<StateRunV1>,
        #[prost(message, repeated, tag = "32")]
        pub atom_modes: Vec<StateRunV1>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct AmbientOcclusionV1 {
        #[prost(bool, tag = "1")]
        pub enabled: bool,
        #[prost(float, tag = "2")]
        pub strength: f32,
        #[prost(float, tag = "3")]
        pub radius: f32,
        #[prost(float, tag = "4")]
        pub bias: f32,
        #[prost(uint32, tag = "5")]
        pub quality: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct ColorRunV1 {
        #[prost(uint32, tag = "1")]
        pub start: u32,
        #[prost(uint32, tag = "2")]
        pub length: u32,
        #[prost(float, repeated, packed = "true", tag = "3")]
        pub rgba: Vec<f32>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct StateRunV1 {
        #[prost(uint32, tag = "1")]
        pub start: u32,
        #[prost(uint32, tag = "2")]
        pub length: u32,
        #[prost(uint32, tag = "3")]
        pub value: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct NamedSelectionV1 {
        #[prost(string, tag = "1")]
        pub name: String,
        #[prost(string, optional, tag = "2")]
        pub expression: Option<String>,
        #[prost(bytes = "vec", tag = "3")]
        pub flags: Vec<u8>,
        #[prost(message, optional, tag = "4")]
        pub style: Option<SelectionStyleV1>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct SelectionStyleV1 {
        #[prost(float, repeated, packed = "true", tag = "1")]
        pub color: Vec<f32>,
        #[prost(uint32, tag = "2")]
        pub visibility: u32,
        #[prost(uint32, tag = "3")]
        pub mode: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct MeasurementLineV1 {
        #[prost(uint64, tag = "1")]
        pub id: u64,
        #[prost(string, tag = "2")]
        pub name: String,
        #[prost(message, optional, tag = "3")]
        pub first: Option<MeasurementEndpointV1>,
        #[prost(message, optional, tag = "4")]
        pub second: Option<MeasurementEndpointV1>,
        #[prost(float, repeated, packed = "true", tag = "5")]
        pub color: Vec<f32>,
        #[prost(uint32, tag = "6")]
        pub visibility: u32,
        #[prost(float, tag = "7")]
        pub label_size: f32,
        #[prost(float, tag = "8")]
        pub thickness: f32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct MeasurementEndpointV1 {
        #[prost(message, optional, tag = "1")]
        pub position: Option<Vec3V1>,
        #[prost(string, tag = "2")]
        pub description: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct CameraV1 {
        #[prost(message, optional, tag = "1")]
        pub target: Option<Vec3V1>,
        #[prost(float, tag = "2")]
        pub distance: f32,
        #[prost(float, tag = "3")]
        pub yaw: f32,
        #[prost(float, tag = "4")]
        pub pitch: f32,
        #[prost(float, tag = "5")]
        pub field_of_view_y: f32,
        #[prost(float, tag = "6")]
        pub near: f32,
        #[prost(float, tag = "7")]
        pub far: f32,
        #[prost(message, optional, tag = "8")]
        pub depth_of_field: Option<DepthOfFieldV1>,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct DepthOfFieldV1 {
        #[prost(bool, tag = "1")]
        pub enabled: bool,
        #[prost(message, optional, tag = "2")]
        pub focus_point: Option<Vec3V1>,
        #[prost(float, tag = "3")]
        pub focal_length_mm: f32,
        #[prost(float, tag = "4")]
        pub sensor_height_mm: f32,
        #[prost(float, tag = "5")]
        pub f_stop: f32,
        #[prost(uint32, tag = "6")]
        pub blade_count: u32,
        #[prost(float, tag = "7")]
        pub blade_rotation: f32,
        #[prost(float, tag = "8")]
        pub max_coc_pixels: f32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct HierarchyNameV1 {
        #[prost(message, optional, tag = "1")]
        pub target: Option<HierarchyTargetV1>,
        #[prost(string, tag = "2")]
        pub name: String,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct HierarchyTargetV1 {
        #[prost(uint32, tag = "1")]
        pub kind: u32,
        #[prost(uint32, tag = "2")]
        pub first: u32,
        #[prost(uint32, tag = "3")]
        pub second: u32,
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Vec3V1 {
        #[prost(float, tag = "1")]
        pub x: f32,
        #[prost(float, tag = "2")]
        pub y: f32,
        #[prost(float, tag = "3")]
        pub z: f32,
    }
}

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
                    flags: pack_flags(selection.flags()),
                    style: Some(selection_style_to_wire(style)),
                })
            })
            .collect::<Result<Vec<_>, SceneError>>()?;

        Ok(Self {
            schema_version: SCENE_SCHEMA_VERSION,
            minimum_reader_version: 1,
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
        if self.minimum_reader_version > SCENE_SCHEMA_VERSION {
            return Err(SceneError::UnsupportedSchema {
                found: self.minimum_reader_version,
                supported: SCENE_SCHEMA_VERSION,
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
        selection_flags: pack_flags(&display.selection),
        chain_colors: color_runs(&display.chain_colors)?,
        residue_colors: color_runs(&display.residue_colors)?,
        atom_colors: color_runs(&display.atom_colors)?,
        chain_visibility: state_runs(&display.chain_visibility, visibility_code)?,
        residue_visibility: state_runs(&display.residue_visibility, visibility_code)?,
        atom_visibility: state_runs(&display.atom_visibility, visibility_code)?,
        chain_modes: state_runs(&display.chain_modes, mode_override_code)?,
        residue_modes: state_runs(&display.residue_modes, mode_override_code)?,
        atom_modes: state_runs(&display.atom_modes, mode_override_code)?,
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
    display.representations = wire
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
    display.selection = unpack_flags(&wire.selection_flags, atom_count, "current selection")?;
    display.chain_colors = colors_from_runs(&wire.chain_colors, atom_count, "chain colors")?;
    display.residue_colors = colors_from_runs(&wire.residue_colors, atom_count, "residue colors")?;
    display.atom_colors = colors_from_runs(&wire.atom_colors, atom_count, "atom colors")?;
    display.chain_visibility = states_from_runs(
        &wire.chain_visibility,
        atom_count,
        VisibilityOverride::Inherit,
        visibility_from_code,
        "chain visibility",
    )?;
    display.residue_visibility = states_from_runs(
        &wire.residue_visibility,
        atom_count,
        VisibilityOverride::Inherit,
        visibility_from_code,
        "residue visibility",
    )?;
    display.atom_visibility = states_from_runs(
        &wire.atom_visibility,
        atom_count,
        VisibilityOverride::Inherit,
        visibility_from_code,
        "atom visibility",
    )?;
    display.chain_modes = states_from_runs(
        &wire.chain_modes,
        atom_count,
        ModeOverride::Inherit,
        mode_override_from_code,
        "chain modes",
    )?;
    display.residue_modes = states_from_runs(
        &wire.residue_modes,
        atom_count,
        ModeOverride::Inherit,
        mode_override_from_code,
        "residue modes",
    )?;
    display.atom_modes = states_from_runs(
        &wire.atom_modes,
        atom_count,
        ModeOverride::Inherit,
        mode_override_from_code,
        "atom modes",
    )?;
    display.recompute_colors();
    display.recompute_visibility();
    display.recompute_modes();
    Ok(display)
}

fn selection_style_to_wire(style: NamedSelectionStyle) -> wire::SelectionStyleV1 {
    wire::SelectionStyleV1 {
        color: style.color.map_or_else(Vec::new, |color| color.to_vec()),
        visibility: visibility_code(style.visibility),
        mode: mode_override_code(style.mode),
    }
}

fn selection_style_from_wire(
    style: wire::SelectionStyleV1,
) -> Result<NamedSelectionStyle, SceneError> {
    Ok(NamedSelectionStyle {
        color: (!style.color.is_empty())
            .then(|| checked_color(&style.color, "selection color"))
            .transpose()?,
        visibility: visibility_from_code(style.visibility)?,
        mode: mode_override_from_code(style.mode)?,
    })
}

fn measurement_to_wire(line: &MeasurementLine) -> wire::MeasurementLineV1 {
    wire::MeasurementLineV1 {
        id: line.id,
        name: line.name.clone(),
        first: Some(measurement_endpoint_to_wire(&line.first)),
        second: Some(measurement_endpoint_to_wire(&line.second)),
        color: line.color.map_or_else(Vec::new, |color| color.to_vec()),
        visibility: visibility_code(line.visibility),
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
    result.visibility = visibility_from_code(line.visibility)?;
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

fn color_runs(values: &[Option<DisplayColor>]) -> Result<Vec<wire::ColorRunV1>, SceneError> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < values.len() {
        let Some(color) = values[index] else {
            index += 1;
            continue;
        };
        let start = index;
        index += 1;
        while index < values.len() && values[index] == Some(color) {
            index += 1;
        }
        runs.push(wire::ColorRunV1 {
            start: u32::try_from(start).map_err(|_| invalid("color run start exceeds u32::MAX"))?,
            length: u32::try_from(index - start)
                .map_err(|_| invalid("color run length exceeds u32::MAX"))?,
            rgba: color.to_vec(),
        });
    }
    Ok(runs)
}

fn colors_from_runs(
    runs: &[wire::ColorRunV1],
    count: usize,
    label: &str,
) -> Result<Vec<Option<DisplayColor>>, SceneError> {
    let mut values = vec![None; count];
    for run in runs {
        let range = checked_run(run.start, run.length, count, label)?;
        let color = checked_color(&run.rgba, label)?;
        values[range].fill(Some(color));
    }
    Ok(values)
}

fn state_runs<T: Copy + PartialEq>(
    values: &[T],
    code: fn(T) -> u32,
) -> Result<Vec<wire::StateRunV1>, SceneError> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < values.len() {
        if code(values[index]) == 0 {
            index += 1;
            continue;
        }
        let value = values[index];
        let start = index;
        index += 1;
        while index < values.len() && values[index] == value {
            index += 1;
        }
        runs.push(wire::StateRunV1 {
            start: u32::try_from(start).map_err(|_| invalid("state run start exceeds u32::MAX"))?,
            length: u32::try_from(index - start)
                .map_err(|_| invalid("state run length exceeds u32::MAX"))?,
            value: code(value),
        });
    }
    Ok(runs)
}

fn states_from_runs<T: Copy>(
    runs: &[wire::StateRunV1],
    count: usize,
    default: T,
    decode: fn(u32) -> Result<T, SceneError>,
    label: &str,
) -> Result<Vec<T>, SceneError> {
    let mut values = vec![default; count];
    for run in runs {
        let range = checked_run(run.start, run.length, count, label)?;
        values[range].fill(decode(run.value)?);
    }
    Ok(values)
}

fn checked_run(
    start: u32,
    length: u32,
    count: usize,
    label: &str,
) -> Result<std::ops::Range<usize>, SceneError> {
    if length == 0 {
        return Err(invalid(format!("{label} contains an empty run")));
    }
    let start = start as usize;
    let end = start
        .checked_add(length as usize)
        .filter(|end| *end <= count)
        .ok_or_else(|| invalid(format!("{label} run exceeds atom count")))?;
    Ok(start..end)
}

fn require_len<T>(values: &[T], expected: usize, label: &str) -> Result<(), SceneError> {
    if values.len() != expected {
        return Err(invalid(format!(
            "{label} has {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn intern_string(
    value: &str,
    strings: &mut Vec<String>,
    indices: &mut HashMap<String, u32>,
) -> Result<u32, SceneError> {
    if let Some(index) = indices.get(value) {
        return Ok(*index);
    }
    let index = u32::try_from(strings.len())
        .map_err(|_| invalid("molecule string table exceeds u32::MAX entries"))?;
    strings.push(value.to_owned());
    indices.insert(value.to_owned(), index);
    Ok(index)
}

fn table_string(strings: &[String], index: u32, label: &str) -> Result<String, SceneError> {
    strings
        .get(index as usize)
        .cloned()
        .ok_or_else(|| invalid(format!("{label} string table index {index} is invalid")))
}

fn element_code(element: Element) -> u32 {
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

fn element_from_code(code: u32) -> Result<Element, SceneError> {
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

fn display_mode_code(mode: DisplayMode) -> u32 {
    match mode {
        DisplayMode::Cartoon => 0,
        DisplayMode::BallAndStick => 1,
        DisplayMode::Toon => 2,
    }
}

fn display_mode_from_code(code: u32) -> Result<DisplayMode, SceneError> {
    match code {
        0 => Ok(DisplayMode::Cartoon),
        1 => Ok(DisplayMode::BallAndStick),
        2 => Ok(DisplayMode::Toon),
        value => Err(invalid(format!("unknown display mode {value}"))),
    }
}

fn coloring_mode_code(mode: ColoringMode) -> u32 {
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

fn coloring_mode_from_code(code: u32) -> Result<ColoringMode, SceneError> {
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

fn visibility_code(state: VisibilityOverride) -> u32 {
    match state {
        VisibilityOverride::Inherit => 0,
        VisibilityOverride::Show => 1,
        VisibilityOverride::Hide => 2,
    }
}

fn visibility_from_code(code: u32) -> Result<VisibilityOverride, SceneError> {
    match code {
        0 => Ok(VisibilityOverride::Inherit),
        1 => Ok(VisibilityOverride::Show),
        2 => Ok(VisibilityOverride::Hide),
        value => Err(invalid(format!("unknown visibility override {value}"))),
    }
}

fn mode_override_code(state: ModeOverride) -> u32 {
    match state {
        ModeOverride::Inherit => 0,
        ModeOverride::Cartoon => 1,
        ModeOverride::BallAndStick => 2,
        ModeOverride::Toon => 3,
    }
}

fn mode_override_from_code(code: u32) -> Result<ModeOverride, SceneError> {
    match code {
        0 => Ok(ModeOverride::Inherit),
        1 => Ok(ModeOverride::Cartoon),
        2 => Ok(ModeOverride::BallAndStick),
        3 => Ok(ModeOverride::Toon),
        value => Err(invalid(format!("unknown mode override {value}"))),
    }
}

fn ao_quality_code(quality: AmbientOcclusionQuality) -> u32 {
    match quality {
        AmbientOcclusionQuality::Low => 0,
        AmbientOcclusionQuality::Medium => 1,
        AmbientOcclusionQuality::High => 2,
    }
}

fn ao_quality_from_code(code: u32) -> Result<AmbientOcclusionQuality, SceneError> {
    match code {
        0 => Ok(AmbientOcclusionQuality::Low),
        1 => Ok(AmbientOcclusionQuality::Medium),
        2 => Ok(AmbientOcclusionQuality::High),
        value => Err(invalid(format!("unknown AO quality {value}"))),
    }
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
        display.selection = vec![true, false];
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
    fn corrupted_compressed_payload_is_rejected() {
        let mut bytes = encode(&document()).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x80;
        assert!(decode(&bytes).is_err());
    }
}
