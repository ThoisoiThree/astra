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

pub const SCENE_EXTENSION: &str = "molscene";
pub const SCENE_SCHEMA_VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"MOLSCENE";
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
    #[error("not a molview scene document")]
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
