//! BinaryCIF (MessagePack-encoded PDBx/mmCIF) decoding.

use std::{borrow::Cow, collections::HashMap, io::Cursor};

use rmpv::Value;

use super::{
    StructureError,
    cif::{CifCategory, CifDocument},
    structure::MAX_DECOMPRESSED_STRUCTURE_SIZE,
};

const MAX_BINARY_CIF_VALUES: usize = MAX_DECOMPRESSED_STRUCTURE_SIZE as usize / 8;

/// Categories the structure reader uses; others are skipped without decoding.
const WANTED_CATEGORIES: &[&str] = &[
    "atom_site",
    "entry",
    "struct",
    "cell",
    "symmetry",
    "space_group",
    "struct_conn",
    "struct_conf",
    "struct_sheet_range",
    "pdbx_struct_assembly",
    "pdbx_struct_assembly_gen",
    "pdbx_struct_oper_list",
];

struct BinaryColumn {
    values: Decoded,
    mask: Option<Decoded>,
}

pub(crate) struct BinaryCategory {
    rows: usize,
    names: HashMap<String, usize>,
    columns: Vec<BinaryColumn>,
}

impl CifCategory for BinaryCategory {
    fn row_count(&self) -> usize {
        self.rows
    }

    fn column(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }

    fn text(&self, column: usize, row: usize) -> Option<Cow<'_, str>> {
        let column = self.columns.get(column)?;
        if column
            .mask
            .as_ref()
            .and_then(|mask| mask.integer(row))
            .is_some_and(|value| value != 0)
        {
            return None;
        }
        match &column.values {
            Decoded::Strings(values) => values
                .get(row)
                .map(String::as_str)
                .filter(|value| !matches!(*value, "." | "?"))
                .map(Cow::Borrowed),
            values => values.text(row).map(Cow::Owned),
        }
    }

    fn number(&self, column: usize, row: usize) -> Option<f64> {
        let data = self.columns.get(column)?;
        if data
            .mask
            .as_ref()
            .and_then(|mask| mask.integer(row))
            .is_some_and(|value| value != 0)
        {
            return None;
        }
        match &data.values {
            Decoded::Floats(values) => values.get(row).copied(),
            Decoded::Integers(values) => values.get(row).map(|value| *value as f64),
            Decoded::Bytes(values) => values.get(row).map(|value| f64::from(*value)),
            Decoded::Strings(values) => values.get(row)?.trim().parse().ok(),
        }
    }
}

pub(crate) struct BinaryDocument {
    categories: HashMap<String, BinaryCategory>,
}

impl CifDocument for BinaryDocument {
    fn category(&self, name: &str) -> Option<&dyn CifCategory> {
        self.categories
            .get(name)
            .map(|category| category as &dyn CifCategory)
    }
}

/// Decodes the categories of the first data block that contains `atom_site`.
pub(crate) fn parse_binary(input: &[u8]) -> Result<BinaryDocument, StructureError> {
    let root = rmpv::decode::read_value(&mut Cursor::new(input))
        .map_err(|error| StructureError::BinaryCif(error.to_string()))?;
    let blocks = map_get(&root, "dataBlocks")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("missing dataBlocks array".into()))?;
    let category_name = |category: &Value| {
        map_get(category, "name")
            .and_then(Value::as_str)
            .map(|name| name.trim_start_matches('_').to_ascii_lowercase())
    };
    let block = blocks
        .iter()
        .find(|block| {
            map_get(block, "categories")
                .and_then(Value::as_array)
                .is_some_and(|categories| {
                    categories
                        .iter()
                        .any(|category| category_name(category).as_deref() == Some("atom_site"))
                })
        })
        .ok_or(StructureError::NoCoordinates("BinaryCIF file"))?;
    let mut categories = HashMap::new();
    for category in map_get(block, "categories")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(name) = category_name(category) else {
            continue;
        };
        if !WANTED_CATEGORIES.contains(&name.as_str()) {
            continue;
        }
        let rows = map_get(category, "rowCount")
            .and_then(value_usize)
            .ok_or_else(|| StructureError::BinaryCif(format!("{name} has no rowCount")))?;
        check_binary_value_count(rows, "rowCount")?;
        let mut names = HashMap::new();
        let mut columns = Vec::new();
        for column in map_get(category, "columns")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(column_name) = map_get(column, "name").and_then(Value::as_str) else {
                continue;
            };
            let data = map_get(column, "data").ok_or_else(|| {
                StructureError::BinaryCif(format!("column '{column_name}' has no data"))
            })?;
            let values = decode_binary_data(data)?;
            let mask = map_get(column, "mask")
                .filter(|value| !value.is_nil())
                .map(decode_binary_data)
                .transpose()?;
            names.insert(column_name.to_ascii_lowercase(), columns.len());
            columns.push(BinaryColumn { values, mask });
        }
        categories.insert(
            name,
            BinaryCategory {
                rows,
                names,
                columns,
            },
        );
    }
    Ok(BinaryDocument { categories })
}

fn map_get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_map()?.iter().find_map(|(candidate, value)| {
        candidate
            .as_str()
            .is_some_and(|candidate| candidate == key)
            .then_some(value)
    })
}

fn value_usize(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| value.as_i64().and_then(|value| usize::try_from(value).ok()))
}

#[derive(Debug)]
enum Decoded {
    Bytes(Vec<u8>),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

impl Decoded {
    fn integer(&self, index: usize) -> Option<i64> {
        match self {
            Self::Integers(values) => values.get(index).copied(),
            Self::Bytes(values) => values.get(index).map(|value| i64::from(*value)),
            _ => None,
        }
    }

    fn text(&self, index: usize) -> Option<String> {
        match self {
            Self::Integers(values) => values.get(index).map(ToString::to_string),
            Self::Floats(values) => values.get(index).map(ToString::to_string),
            Self::Strings(values) => values.get(index).cloned(),
            Self::Bytes(values) => values.get(index).map(ToString::to_string),
        }
    }
}

fn decode_binary_data(data: &Value) -> Result<Decoded, StructureError> {
    let bytes = map_get(data, "data")
        .and_then(value_bytes)
        .ok_or_else(|| StructureError::BinaryCif("encoded column has no byte data".into()))?;
    let encodings = map_get(data, "encoding")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("encoded column has no encoding array".into()))?;
    decode_with_encodings(bytes.to_vec(), encodings)
}

fn value_bytes(value: &Value) -> Option<&[u8]> {
    match value {
        Value::Binary(bytes) => Some(bytes),
        _ => None,
    }
}

fn decode_with_encodings(bytes: Vec<u8>, encodings: &[Value]) -> Result<Decoded, StructureError> {
    let mut decoded = Decoded::Bytes(bytes);
    for encoding in encodings.iter().rev() {
        let kind = map_get(encoding, "kind")
            .and_then(Value::as_str)
            .ok_or_else(|| StructureError::BinaryCif("encoding has no kind".into()))?;
        decoded = match kind {
            "ByteArray" => decode_byte_array(decoded, encoding)?,
            "IntegerPacking" => decode_integer_packing(decoded, encoding)?,
            "RunLength" => decode_run_length(decoded)?,
            "Delta" => decode_delta(decoded, encoding)?,
            "FixedPoint" => decode_fixed_point(decoded, encoding)?,
            "IntervalQuantization" => decode_interval(decoded, encoding)?,
            "StringArray" => decode_string_array(decoded, encoding)?,
            other => {
                return Err(StructureError::BinaryCif(format!(
                    "unsupported encoding '{other}'"
                )));
            }
        };
        check_decoded_size(&decoded, kind)?;
    }
    Ok(decoded)
}

fn check_binary_value_count(count: usize, label: &str) -> Result<(), StructureError> {
    if count > MAX_BINARY_CIF_VALUES {
        Err(StructureError::BinaryCif(format!(
            "{label} requests {count} values, exceeding the safety limit of {MAX_BINARY_CIF_VALUES}"
        )))
    } else {
        Ok(())
    }
}

fn check_binary_allocation<T>(count: usize, label: &str) -> Result<(), StructureError> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>().max(1))
        .ok_or_else(|| StructureError::BinaryCif(format!("{label} allocation overflow")))?;
    if bytes > MAX_DECOMPRESSED_STRUCTURE_SIZE as usize {
        Err(StructureError::BinaryCif(format!(
            "{label} requests {bytes} bytes, exceeding the {MAX_DECOMPRESSED_STRUCTURE_SIZE} byte safety limit"
        )))
    } else {
        Ok(())
    }
}

fn check_decoded_size(decoded: &Decoded, label: &str) -> Result<(), StructureError> {
    match decoded {
        Decoded::Bytes(values) => check_binary_allocation::<u8>(values.len(), label),
        Decoded::Integers(values) => check_binary_allocation::<i64>(values.len(), label),
        Decoded::Floats(values) => check_binary_allocation::<f64>(values.len(), label),
        Decoded::Strings(values) => check_binary_allocation::<String>(values.len(), label),
    }
}

fn decode_byte_array(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let Decoded::Bytes(bytes) = value else {
        return Err(StructureError::BinaryCif(
            "ByteArray expected raw bytes".into(),
        ));
    };
    let data_type = map_get(encoding, "type")
        .and_then(Value::as_i64)
        .ok_or_else(|| StructureError::BinaryCif("ByteArray has no type".into()))?;
    let exact_chunks = |size: usize| {
        (bytes.len() % size == 0)
            .then_some(bytes.chunks_exact(size))
            .ok_or_else(|| StructureError::BinaryCif("misaligned ByteArray data".into()))
    };
    let output_count = match data_type {
        1 | 4 => bytes.len(),
        2 | 5 => bytes.len() / 2,
        3 | 6 | 32 => bytes.len() / 4,
        33 => bytes.len() / 8,
        _ => 0,
    };
    if data_type == 32 || data_type == 33 {
        check_binary_allocation::<f64>(output_count, "ByteArray output")?;
    } else {
        check_binary_allocation::<i64>(output_count, "ByteArray output")?;
    }
    match data_type {
        1 => Ok(Decoded::Integers(
            bytes.iter().map(|value| i64::from(*value as i8)).collect(),
        )),
        2 => Ok(Decoded::Integers(
            exact_chunks(2)?
                .map(|chunk| i64::from(i16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
        )),
        3 => Ok(Decoded::Integers(
            exact_chunks(4)?
                .map(|chunk| i64::from(i32::from_le_bytes(chunk.try_into().unwrap_or_default())))
                .collect(),
        )),
        4 => Ok(Decoded::Integers(
            bytes.iter().map(|value| i64::from(*value)).collect(),
        )),
        5 => Ok(Decoded::Integers(
            exact_chunks(2)?
                .map(|chunk| i64::from(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
        )),
        6 => Ok(Decoded::Integers(
            exact_chunks(4)?
                .map(|chunk| i64::from(u32::from_le_bytes(chunk.try_into().unwrap_or_default())))
                .collect(),
        )),
        32 => Ok(Decoded::Floats(
            exact_chunks(4)?
                .map(|chunk| f64::from(f32::from_le_bytes(chunk.try_into().unwrap_or_default())))
                .collect(),
        )),
        33 => Ok(Decoded::Floats(
            exact_chunks(8)?
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or_default()))
                .collect(),
        )),
        _ => Err(StructureError::BinaryCif(format!(
            "unknown ByteArray type {data_type}"
        ))),
    }
}

fn integers(value: Decoded, kind: &str) -> Result<Vec<i64>, StructureError> {
    match value {
        Decoded::Integers(values) => Ok(values),
        _ => Err(StructureError::BinaryCif(format!(
            "{kind} expected integer input"
        ))),
    }
}

fn decode_integer_packing(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let packed = integers(value, "IntegerPacking")?;
    let byte_count = map_get(encoding, "byteCount")
        .and_then(Value::as_i64)
        .ok_or_else(|| StructureError::BinaryCif("IntegerPacking has no byteCount".into()))?;
    let unsigned = map_get(encoding, "isUnsigned")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let upper = match (byte_count, unsigned) {
        (1, true) => 0xff,
        (2, true) => 0xffff,
        (1, false) => 0x7f,
        (2, false) => 0x7fff,
        _ => {
            return Err(StructureError::BinaryCif(format!(
                "unsupported IntegerPacking byteCount {byte_count}"
            )));
        }
    };
    let lower = if unsigned { i64::MIN } else { -upper - 1 };
    let mut values = Vec::new();
    values.try_reserve(packed.len()).map_err(|_| {
        StructureError::BinaryCif("IntegerPacking output allocation is too large".into())
    })?;
    let mut cursor = 0;
    while cursor < packed.len() {
        let mut value = 0_i64;
        loop {
            let part = *packed.get(cursor).ok_or_else(|| {
                StructureError::BinaryCif("truncated IntegerPacking value".into())
            })?;
            cursor += 1;
            value = value
                .checked_add(part)
                .ok_or_else(|| StructureError::BinaryCif("IntegerPacking value overflow".into()))?;
            if part != upper && part != lower {
                break;
            }
        }
        values.push(value);
    }
    Ok(Decoded::Integers(values))
}

fn decode_run_length(value: Decoded) -> Result<Decoded, StructureError> {
    let packed = integers(value, "RunLength")?;
    if packed.len() % 2 != 0 {
        return Err(StructureError::BinaryCif(
            "RunLength data has an incomplete pair".into(),
        ));
    }
    let total = packed
        .as_chunks::<2>()
        .0
        .iter()
        .try_fold(0_usize, |total, pair| {
            let count = usize::try_from(pair[1])
                .map_err(|_| StructureError::BinaryCif("negative RunLength count".into()))?;
            total
                .checked_add(count)
                .ok_or_else(|| StructureError::BinaryCif("RunLength size overflow".into()))
        })?;
    check_binary_value_count(total, "RunLength output")?;
    let mut values = Vec::new();
    values.try_reserve_exact(total).map_err(|_| {
        StructureError::BinaryCif("RunLength output allocation is too large".into())
    })?;
    for pair in packed.as_chunks::<2>().0 {
        let count = usize::try_from(pair[1])
            .map_err(|_| StructureError::BinaryCif("negative RunLength count".into()))?;
        values.extend(std::iter::repeat_n(pair[0], count));
    }
    Ok(Decoded::Integers(values))
}

fn decode_delta(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let deltas = integers(value, "Delta")?;
    let mut running = map_get(encoding, "origin")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let mut values = Vec::new();
    values
        .try_reserve_exact(deltas.len())
        .map_err(|_| StructureError::BinaryCif("Delta output allocation is too large".into()))?;
    for delta in deltas {
        running = running
            .checked_add(delta)
            .ok_or_else(|| StructureError::BinaryCif("Delta value overflow".into()))?;
        values.push(running);
    }
    Ok(Decoded::Integers(values))
}

fn decode_fixed_point(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let values = integers(value, "FixedPoint")?;
    let factor = map_get(encoding, "factor")
        .and_then(value_f64)
        .ok_or_else(|| StructureError::BinaryCif("FixedPoint has no factor".into()))?;
    if !factor.is_finite() || factor == 0.0 {
        return Err(StructureError::BinaryCif(
            "FixedPoint factor must be finite and non-zero".into(),
        ));
    }
    Ok(Decoded::Floats(
        values
            .into_iter()
            .map(|value| value as f64 / factor)
            .collect(),
    ))
}

fn decode_interval(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let values = integers(value, "IntervalQuantization")?;
    let minimum = map_get(encoding, "min")
        .and_then(value_f64)
        .ok_or_else(|| StructureError::BinaryCif("IntervalQuantization has no min".into()))?;
    let maximum = map_get(encoding, "max")
        .and_then(value_f64)
        .ok_or_else(|| StructureError::BinaryCif("IntervalQuantization has no max".into()))?;
    let steps = map_get(encoding, "numSteps")
        .and_then(value_f64)
        .ok_or_else(|| StructureError::BinaryCif("IntervalQuantization has no numSteps".into()))?;
    if !minimum.is_finite() || !maximum.is_finite() || !steps.is_finite() || steps < 2.0 {
        return Err(StructureError::BinaryCif(
            "IntervalQuantization parameters must be finite and numSteps must be at least 2".into(),
        ));
    }
    let delta = (maximum - minimum) / (steps - 1.0);
    Ok(Decoded::Floats(
        values
            .into_iter()
            .map(|value| minimum + value as f64 * delta)
            .collect(),
    ))
}

fn value_f64(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|value| value as f64))
        .or_else(|| value.as_u64().map(|value| value as f64))
}

fn decode_string_array(value: Decoded, encoding: &Value) -> Result<Decoded, StructureError> {
    let Decoded::Bytes(data) = value else {
        return Err(StructureError::BinaryCif(
            "StringArray expected raw bytes".into(),
        ));
    };
    let data_encoding = map_get(encoding, "dataEncoding")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("StringArray has no dataEncoding".into()))?;
    let indices = integers(
        decode_with_encodings(data, data_encoding)?,
        "StringArray indices",
    )?;
    let offsets = map_get(encoding, "offsets")
        .and_then(value_bytes)
        .ok_or_else(|| StructureError::BinaryCif("StringArray has no offsets".into()))?;
    let offset_encoding = map_get(encoding, "offsetEncoding")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("StringArray has no offsetEncoding".into()))?;
    let offsets = integers(
        decode_with_encodings(offsets.to_vec(), offset_encoding)?,
        "StringArray offsets",
    )?;
    let string_data = map_get(encoding, "stringData")
        .and_then(Value::as_str)
        .ok_or_else(|| StructureError::BinaryCif("StringArray has no stringData".into()))?;
    if offsets
        .windows(2)
        .any(|pair| pair[0] < 0 || pair[1] < pair[0])
    {
        return Err(StructureError::BinaryCif(
            "StringArray offsets must be non-negative and monotonic".into(),
        ));
    }
    check_binary_allocation::<String>(indices.len(), "StringArray output")?;
    let mut values = Vec::new();
    values.try_reserve_exact(indices.len()).map_err(|_| {
        StructureError::BinaryCif("StringArray output allocation is too large".into())
    })?;
    let mut output_text_bytes = 0_usize;
    for index in indices {
        if index < 0 {
            values.push(String::new());
            continue;
        }
        let index = usize::try_from(index)
            .map_err(|_| StructureError::BinaryCif("invalid StringArray index".into()))?;
        let start = offsets
            .get(index)
            .and_then(|value| usize::try_from(*value).ok())
            .ok_or_else(|| StructureError::BinaryCif("StringArray offset is missing".into()))?;
        let end = offsets
            .get(index + 1)
            .and_then(|value| usize::try_from(*value).ok())
            .ok_or_else(|| StructureError::BinaryCif("StringArray offset is missing".into()))?;
        let value = string_data
            .get(start..end)
            .ok_or_else(|| StructureError::BinaryCif("invalid StringArray range".into()))?;
        output_text_bytes = output_text_bytes
            .checked_add(value.len())
            .ok_or_else(|| StructureError::BinaryCif("StringArray text size overflow".into()))?;
        check_binary_allocation::<u8>(output_text_bytes, "StringArray text")?;
        values.push(value.to_string());
    }
    Ok(Decoded::Strings(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(values: Vec<(&str, Value)>) -> Value {
        Value::Map(
            values
                .into_iter()
                .map(|(name, value)| (Value::from(name), value))
                .collect(),
        )
    }

    #[test]
    fn binary_cif_rejects_unbounded_run_length_before_allocation() {
        let count = i64::try_from(MAX_BINARY_CIF_VALUES).unwrap() + 1;
        let error = decode_run_length(Decoded::Integers(vec![7, count])).unwrap_err();
        assert!(error.to_string().contains("safety limit"));
    }

    #[test]
    fn binary_cif_rejects_delta_overflow() {
        let encoding = object(vec![("origin", Value::from(i64::MAX))]);
        let error = decode_delta(Decoded::Integers(vec![1]), &encoding).unwrap_err();
        assert!(error.to_string().contains("overflow"));
    }
}
