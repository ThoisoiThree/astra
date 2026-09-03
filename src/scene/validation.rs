use std::collections::HashMap;

use glam::Vec3;

use crate::DisplayColor;

use super::{SceneError, wire};

pub(super) fn invalid(message: impl Into<String>) -> SceneError {
    SceneError::InvalidData(message.into())
}

pub(super) fn pack_flags(flags: &[bool]) -> Vec<u8> {
    let mut bytes = vec![0; flags.len().div_ceil(8)];
    for (index, flag) in flags.iter().copied().enumerate() {
        if flag {
            bytes[index / 8] |= 1 << (index % 8);
        }
    }
    bytes
}

pub(super) fn pack_atom_mask(flags: &crate::bitset::AtomMask) -> Vec<u8> {
    let mut bytes = vec![0_u8; flags.len().div_ceil(8)];
    for index in flags.indices() {
        bytes[index / 8] |= 1 << (index % 8);
    }
    bytes
}

pub(super) fn unpack_flags(
    bytes: &[u8],
    count: usize,
    label: &str,
) -> Result<Vec<bool>, SceneError> {
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

pub(super) fn checked_vec3(vector: Option<wire::Vec3V1>, label: &str) -> Result<Vec3, SceneError> {
    let vector = vector.ok_or_else(|| invalid(format!("{label} is missing")))?;
    if !vector.x.is_finite() || !vector.y.is_finite() || !vector.z.is_finite() {
        return Err(invalid(format!("{label} contains a non-finite coordinate")));
    }
    Ok(Vec3::new(vector.x, vector.y, vector.z))
}

pub(super) fn vec3(vector: Vec3) -> wire::Vec3V1 {
    wire::Vec3V1 {
        x: vector.x,
        y: vector.y,
        z: vector.z,
    }
}

pub(super) fn checked_color(values: &[f32], label: &str) -> Result<DisplayColor, SceneError> {
    let values: [f32; 4] = values
        .try_into()
        .map_err(|_| invalid(format!("{label} must contain four components")))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{label} contains a non-finite component")));
    }
    Ok(values.map(|value| value.clamp(0.0, 1.0)))
}

pub(super) fn checked_finite(value: f32, label: &str) -> Result<f32, SceneError> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| invalid(format!("{label} is not finite")))
}
pub(super) fn color_runs(
    values: &[Option<DisplayColor>],
) -> Result<Vec<wire::ColorRunV1>, SceneError> {
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

pub(super) fn colors_from_runs(
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

pub(super) fn state_runs<T: Copy + PartialEq>(
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

pub(super) fn states_from_runs<T: Copy>(
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

pub(super) fn checked_run(
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

pub(super) fn require_len<T>(values: &[T], expected: usize, label: &str) -> Result<(), SceneError> {
    if values.len() != expected {
        return Err(invalid(format!(
            "{label} has {} values, expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

pub(super) fn intern_string(
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

pub(super) fn table_string(
    strings: &[String],
    index: u32,
    label: &str,
) -> Result<String, SceneError> {
    strings
        .get(index as usize)
        .cloned()
        .ok_or_else(|| invalid(format!("{label} string table index {index} is invalid")))
}
