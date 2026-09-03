use std::io::{Cursor, Read, Write};

use prost::Message;

use super::{SCENE_SCHEMA_VERSION, SceneDocument, SceneError, wire};

pub(super) const MAGIC: &[u8; 8] = b"MOLECULE";
pub(super) const CONTAINER_VERSION: u16 = 1;
pub(super) const COMPRESSION_ZSTD: u16 = 1;
pub(super) const HEADER_LEN: usize = 24;
pub(super) const MAX_DECOMPRESSED_SIZE: u64 = 512 * 1024 * 1024;

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
    // Do not trust the header enough to reserve the complete advertised size up front.
    let initial_capacity = usize::try_from(expected_len.min(8 * 1024 * 1024)).unwrap_or(0);
    let mut payload = Vec::with_capacity(initial_capacity);
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
