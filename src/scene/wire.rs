#![allow(clippy::enum_variant_names)]

#[cfg(test)]
use prost::Message;

// Frozen reader-1 wire declarations are retained only for compatibility tests. Production
// serialization is generated from schemas/molecule_1_0.proto below, preventing schema drift.
#[cfg(test)]
pub(super) mod legacy_wire_v1 {
    #![allow(dead_code)]
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

include!(concat!(env!("OUT_DIR"), "/molecule.v1.rs"));

pub type SceneV1 = Scene;
pub type MoleculeV1 = Molecule;
pub type DisplayV1 = Display;
pub type AmbientOcclusionV1 = AmbientOcclusion;
pub type ColorRunV1 = ColorRun;
pub type StateRunV1 = StateRun;
pub type NamedSelectionV1 = NamedSelection;
pub type SelectionStyleV1 = SelectionStyle;
pub type MeasurementLineV1 = MeasurementLine;
pub type MeasurementEndpointV1 = MeasurementEndpoint;
pub type CameraV1 = Camera;
pub type DepthOfFieldV1 = DepthOfField;
pub type HierarchyNameV1 = HierarchyName;
pub type HierarchyTargetV1 = HierarchyTarget;
pub type Vec3V1 = Vec3;
