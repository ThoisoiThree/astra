use std::{
    borrow::Cow,
    collections::HashMap,
    io::{Cursor, Read},
    str::FromStr,
};

use flate2::read::GzDecoder;
use glam::Vec3;
use rmpv::Value;
use thiserror::Error;

use super::{Atom, Element, Molecule, PdbError, infer_bonds, parse_pdb, pdb::infer_element};

pub const MAX_DECOMPRESSED_STRUCTURE_SIZE: u64 = 512 * 1024 * 1024;
const MAX_BINARY_CIF_VALUES: usize = MAX_DECOMPRESSED_STRUCTURE_SIZE as usize / 8;
const MAX_GZIP_EXPANSION_RATIO: u64 = 200;
const GZIP_RATIO_CHECK_THRESHOLD: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureFormat {
    Pdb,
    Mmcif,
    BinaryCif,
    Pdbml,
}

impl StructureFormat {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pdb => "legacy PDB",
            Self::Mmcif => "PDBx/mmCIF",
            Self::BinaryCif => "BinaryCIF",
            Self::Pdbml => "PDBML/XML",
        }
    }
}

#[derive(Debug, Error)]
pub enum StructureError {
    #[error("could not decompress gzip data: {0}")]
    Gzip(#[source] std::io::Error),
    #[error("structure input exceeds the {limit} byte safety limit (decoded {actual} bytes)")]
    TooLarge { actual: u64, limit: u64 },
    #[error("gzip expansion ratio is suspicious ({decoded} decoded bytes from {compressed} bytes)")]
    SuspiciousCompressionRatio { compressed: u64, decoded: u64 },
    #[error("{format} input is not valid UTF-8: {source}")]
    Utf8 {
        format: &'static str,
        #[source]
        source: std::str::Utf8Error,
    },
    #[error(transparent)]
    Pdb(#[from] PdbError),
    #[error("mmCIF syntax error: {0}")]
    Mmcif(String),
    #[error("BinaryCIF decoding error: {0}")]
    BinaryCif(String),
    #[error("PDBML/XML syntax error: {0}")]
    Pdbml(String),
    #[error("{format} atom_site row {row}: {message}")]
    AtomSite {
        format: &'static str,
        row: usize,
        message: String,
    },
    #[error("{0} contains no atom_site coordinates and cannot be displayed as a structure")]
    NoCoordinates(&'static str),
    #[error("PDF validation reports contain no molecular coordinates and cannot be displayed")]
    ValidationPdf,
    #[error("unsupported structure format; expected PDB, mmCIF, BinaryCIF, or PDBML/XML")]
    Unsupported,
}

/// Parses a coordinate file, transparently handling gzip by its magic bytes.
/// `filename` is only used as a format hint; content sniffing remains the fallback.
pub fn parse_structure(
    bytes: &[u8],
    filename: &str,
) -> Result<(Molecule, StructureFormat), StructureError> {
    let bytes = decompress_if_needed(bytes)?;
    if bytes.starts_with(b"%PDF-") {
        return Err(StructureError::ValidationPdf);
    }
    let hint = filename_hint(filename);
    let format = hint
        .or_else(|| sniff_format(&bytes))
        .ok_or(StructureError::Unsupported)?;
    let molecule = match format {
        StructureFormat::Pdb => parse_pdb(text(&bytes, format)?)?,
        StructureFormat::Mmcif => parse_mmcif(text(&bytes, format)?)?,
        StructureFormat::BinaryCif => parse_binary_cif(&bytes)?,
        StructureFormat::Pdbml => parse_pdbml(text(&bytes, format)?)?,
    };
    Ok((molecule, format))
}

fn decompress_if_needed(bytes: &[u8]) -> Result<Cow<'_, [u8]>, StructureError> {
    if !bytes.starts_with(&[0x1f, 0x8b]) {
        check_structure_size(bytes.len() as u64, MAX_DECOMPRESSED_STRUCTURE_SIZE)?;
        return Ok(Cow::Borrowed(bytes));
    }
    decompress_gzip_with_limit(bytes, MAX_DECOMPRESSED_STRUCTURE_SIZE).map(Cow::Owned)
}

fn decompress_gzip_with_limit(bytes: &[u8], limit: u64) -> Result<Vec<u8>, StructureError> {
    let mut decoded = Vec::new();
    GzDecoder::new(bytes)
        .take(limit.saturating_add(1))
        .read_to_end(&mut decoded)
        .map_err(StructureError::Gzip)?;
    check_structure_size(decoded.len() as u64, limit)?;
    let compressed = bytes.len().max(1) as u64;
    let decoded_len = decoded.len() as u64;
    if decoded_len >= GZIP_RATIO_CHECK_THRESHOLD
        && decoded_len > compressed.saturating_mul(MAX_GZIP_EXPANSION_RATIO)
    {
        return Err(StructureError::SuspiciousCompressionRatio {
            compressed,
            decoded: decoded_len,
        });
    }
    Ok(decoded)
}

fn check_structure_size(actual: u64, limit: u64) -> Result<(), StructureError> {
    if actual > limit {
        Err(StructureError::TooLarge { actual, limit })
    } else {
        Ok(())
    }
}

fn filename_hint(filename: &str) -> Option<StructureFormat> {
    let lower = filename.to_ascii_lowercase();
    let lower = lower.strip_suffix(".gz").unwrap_or(&lower);
    if lower.ends_with(".bcif") {
        Some(StructureFormat::BinaryCif)
    } else if lower.ends_with(".cif") || lower.ends_with(".mmcif") {
        Some(StructureFormat::Mmcif)
    } else if lower.ends_with(".xml") {
        Some(StructureFormat::Pdbml)
    } else if lower.ends_with(".pdb") || lower.ends_with(".ent") {
        Some(StructureFormat::Pdb)
    } else {
        None
    }
}

fn sniff_format(bytes: &[u8]) -> Option<StructureFormat> {
    let prefix = bytes.get(..bytes.len().min(4096)).unwrap_or(bytes);
    let text = std::str::from_utf8(prefix)
        .ok()?
        .trim_start_matches('\u{feff}')
        .trim_start();
    if text.starts_with("data_") || text.starts_with("global_") {
        Some(StructureFormat::Mmcif)
    } else if text.starts_with("<?xml") || text.starts_with('<') {
        Some(StructureFormat::Pdbml)
    } else if text.lines().any(|line| {
        matches!(
            line.get(..line.len().min(6)).unwrap_or(line).trim(),
            "ATOM" | "HETATM" | "HEADER" | "MODEL"
        )
    }) {
        Some(StructureFormat::Pdb)
    } else if bytes.first().is_some_and(|byte| byte & 0x80 != 0) {
        Some(StructureFormat::BinaryCif)
    } else {
        None
    }
}

fn text(bytes: &[u8], format: StructureFormat) -> Result<&str, StructureError> {
    std::str::from_utf8(bytes).map_err(|source| StructureError::Utf8 {
        format: format.label(),
        source,
    })
}

fn parse_mmcif(input: &str) -> Result<Molecule, StructureError> {
    let tokens = tokenize_cif(input)?;
    let mut atoms = Vec::new();
    let mut cursor = 0;
    let mut first_model = None;
    let mut found_atom_site = false;
    while cursor < tokens.len() {
        if !tokens[cursor].eq_ignore_ascii_case("loop_") {
            cursor += 1;
            continue;
        }
        cursor += 1;
        let mut tags = Vec::new();
        while cursor < tokens.len() && tokens[cursor].starts_with('_') {
            tags.push(tokens[cursor].to_ascii_lowercase());
            cursor += 1;
        }
        if tags.is_empty() {
            continue;
        }
        let value_start = cursor;
        while cursor < tokens.len() && !is_cif_control_token(&tokens[cursor]) {
            cursor += 1;
        }
        let values = &tokens[value_start..cursor];
        if !tags.iter().any(|tag| tag.starts_with("_atom_site.")) {
            continue;
        }
        found_atom_site = true;
        if values.len() % tags.len() != 0 {
            return Err(StructureError::Mmcif(format!(
                "atom_site loop has {} values for {} columns",
                values.len(),
                tags.len()
            )));
        }
        let columns: HashMap<_, _> = tags
            .iter()
            .enumerate()
            .filter_map(|(index, tag)| tag.split_once('.').map(|(_, name)| (name, index)))
            .collect();
        for (row_index, row) in values.chunks(tags.len()).enumerate() {
            let fields = |names: &[&str]| column_text(row, &columns, names);
            if let Some(atom) = atom_from_fields(&fields, "mmCIF", row_index + 1, &mut first_model)?
            {
                atoms.push(atom);
            }
        }
    }
    if !found_atom_site || atoms.is_empty() {
        return Err(StructureError::NoCoordinates("mmCIF file"));
    }
    Ok(molecule_from_atoms(atoms))
}

fn is_cif_control_token(value: &str) -> bool {
    value.starts_with('_')
        || value.eq_ignore_ascii_case("loop_")
        || value.eq_ignore_ascii_case("stop_")
        || value.eq_ignore_ascii_case("global_")
        || value.get(..5).is_some_and(|prefix| {
            prefix.eq_ignore_ascii_case("data_") || prefix.eq_ignore_ascii_case("save_")
        })
}

fn tokenize_cif(input: &str) -> Result<Vec<String>, StructureError> {
    let bytes = input.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut line_start = true;
    while cursor < bytes.len() {
        match bytes[cursor] {
            byte if byte.is_ascii_whitespace() => {
                line_start = byte == b'\n' || (line_start && byte == b'\r');
                cursor += 1;
            }
            b'#' => {
                while cursor < bytes.len() && bytes[cursor] != b'\n' {
                    cursor += 1;
                }
                line_start = true;
            }
            b';' if line_start => {
                cursor += 1;
                let start = cursor;
                let mut end = None;
                while cursor < bytes.len() {
                    if bytes[cursor] == b'\n'
                        && bytes.get(cursor + 1) == Some(&b';')
                        && bytes
                            .get(cursor + 2)
                            .is_none_or(|byte| matches!(byte, b'\r' | b'\n'))
                    {
                        end = Some(cursor);
                        cursor += 2;
                        while cursor < bytes.len() && bytes[cursor] != b'\n' {
                            cursor += 1;
                        }
                        break;
                    }
                    cursor += 1;
                }
                let end = end.ok_or_else(|| {
                    StructureError::Mmcif("unterminated semicolon-delimited value".into())
                })?;
                tokens.push(
                    input[start..end]
                        .trim_start_matches(['\r', '\n'])
                        .to_string(),
                );
                line_start = true;
            }
            quote @ (b'\'' | b'"') => {
                line_start = false;
                cursor += 1;
                let start = cursor;
                while cursor < bytes.len() && bytes[cursor] != quote {
                    cursor += 1;
                }
                if cursor == bytes.len() {
                    return Err(StructureError::Mmcif("unterminated quoted value".into()));
                }
                tokens.push(input[start..cursor].to_string());
                cursor += 1;
            }
            _ => {
                line_start = false;
                let start = cursor;
                while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
                    if bytes[cursor] == b'#' && cursor == start {
                        break;
                    }
                    cursor += 1;
                }
                tokens.push(input[start..cursor].to_string());
            }
        }
    }
    Ok(tokens)
}

fn column_text<'a>(
    row: &'a [String],
    columns: &HashMap<&str, usize>,
    names: &[&str],
) -> Option<&'a str> {
    names.iter().find_map(|name| {
        columns
            .get(name)
            .and_then(|index| row.get(*index))
            .map(String::as_str)
            .filter(|value| !matches!(*value, "." | "?"))
    })
}

fn parse_pdbml(input: &str) -> Result<Molecule, StructureError> {
    let mut atoms = Vec::new();
    let mut cursor = 0;
    let mut text_start = 0;
    let mut record: Option<HashMap<String, String>> = None;
    let mut field_name: Option<String> = None;
    let mut first_model = None;
    while let Some(relative) = input[cursor..].find('<') {
        let tag_start = cursor + relative;
        if let (Some(record), Some(field)) = (&mut record, &field_name) {
            let value = xml_unescape(input[text_start..tag_start].trim())?;
            if !value.is_empty() {
                record.entry(field.clone()).or_default().push_str(&value);
            }
        }
        let Some(relative_end) = input[tag_start..].find('>') else {
            return Err(StructureError::Pdbml("unterminated XML tag".into()));
        };
        let tag_end = tag_start + relative_end;
        let raw = input[tag_start + 1..tag_end].trim();
        cursor = tag_end + 1;
        text_start = cursor;
        if raw.starts_with(['?', '!']) {
            continue;
        }
        let closing = raw.starts_with('/');
        let self_closing = raw.ends_with('/');
        let body = raw.trim_start_matches('/').trim_end_matches('/').trim();
        let qualified_name = body.split_ascii_whitespace().next().unwrap_or_default();
        let name = qualified_name
            .rsplit_once(':')
            .map_or(qualified_name, |(_, local)| local)
            .to_ascii_lowercase();
        if closing {
            if name == "atom_site"
                && let Some(record) = record.take()
            {
                let fields = |names: &[&str]| record_text(&record, names);
                if let Some(atom) =
                    atom_from_fields(&fields, "PDBML/XML", atoms.len() + 1, &mut first_model)?
                {
                    atoms.push(atom);
                }
            }
            field_name = None;
        } else if name == "atom_site" {
            let mut values = HashMap::new();
            if let Some(id) = xml_attribute(body, "id") {
                values.insert("id".into(), id);
            }
            record = Some(values);
            field_name = None;
        } else if record.is_some() {
            field_name = Some(name);
            if self_closing {
                field_name = None;
            }
        }
    }
    if atoms.is_empty() {
        return Err(StructureError::NoCoordinates("PDBML/XML file"));
    }
    Ok(molecule_from_atoms(atoms))
}

fn xml_attribute(tag: &str, wanted: &str) -> Option<String> {
    tag.split_ascii_whitespace().skip(1).find_map(|part| {
        let (name, value) = part.split_once('=')?;
        name.eq_ignore_ascii_case(wanted)
            .then(|| value.trim_matches(['"', '\'', '/']).to_string())
    })
}

fn xml_unescape(value: &str) -> Result<String, StructureError> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(index) = rest.find('&') {
        output.push_str(&rest[..index]);
        let entity = &rest[index..];
        let Some(end) = entity.find(';') else {
            return Err(StructureError::Pdbml("unterminated XML entity".into()));
        };
        let name = &entity[1..end];
        let decoded = match name {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            hexadecimal if hexadecimal.starts_with("#x") => {
                u32::from_str_radix(&hexadecimal[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| {
                        StructureError::Pdbml(format!("invalid XML entity '&{name};'"))
                    })?
            }
            decimal if decimal.starts_with('#') => decimal[1..]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .ok_or_else(|| StructureError::Pdbml(format!("invalid XML entity '&{name};'")))?,
            _ => {
                return Err(StructureError::Pdbml(format!(
                    "unknown XML entity '&{name};'"
                )));
            }
        };
        output.push(decoded);
        rest = &entity[end + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

fn record_text<'a>(record: &'a HashMap<String, String>, names: &[&str]) -> Option<&'a str> {
    names
        .iter()
        .find_map(|name| record.get(*name))
        .map(String::as_str)
        .filter(|value| !matches!(*value, "." | "?"))
}

fn parse_binary_cif(input: &[u8]) -> Result<Molecule, StructureError> {
    let root = rmpv::decode::read_value(&mut Cursor::new(input))
        .map_err(|error| StructureError::BinaryCif(error.to_string()))?;
    let blocks = map_get(&root, "dataBlocks")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("missing dataBlocks array".into()))?;
    let mut atom_category = None;
    for block in blocks {
        let Some(categories) = map_get(block, "categories").and_then(Value::as_array) else {
            continue;
        };
        atom_category = categories.iter().find(|category| {
            map_get(category, "name")
                .and_then(Value::as_str)
                .is_some_and(|name| {
                    name.trim_start_matches('_')
                        .eq_ignore_ascii_case("atom_site")
                })
        });
        if atom_category.is_some() {
            break;
        }
    }
    let category = atom_category.ok_or(StructureError::NoCoordinates("BinaryCIF file"))?;
    let row_count = map_get(category, "rowCount")
        .and_then(value_usize)
        .ok_or_else(|| StructureError::BinaryCif("atom_site has no rowCount".into()))?;
    check_binary_allocation::<Atom>(row_count, "atom_site rowCount")?;
    let columns = map_get(category, "columns")
        .and_then(Value::as_array)
        .ok_or_else(|| StructureError::BinaryCif("atom_site has no columns".into()))?;
    let mut decoded = HashMap::new();
    for column in columns {
        let Some(name) = map_get(column, "name").and_then(Value::as_str) else {
            continue;
        };
        if !wanted_atom_column(name) {
            continue;
        }
        let data = map_get(column, "data")
            .ok_or_else(|| StructureError::BinaryCif(format!("column '{name}' has no data")))?;
        let values = decode_binary_data(data)?;
        let mask = map_get(column, "mask")
            .filter(|value| !value.is_nil())
            .map(decode_binary_data)
            .transpose()?;
        decoded.insert(name.to_ascii_lowercase(), BinaryColumn { values, mask });
    }
    let mut atoms = Vec::new();
    let mut first_model = None;
    for row in 0..row_count {
        let fields = |names: &[&str]| binary_column_text(&decoded, names, row);
        if let Some(atom) = atom_from_fields(&fields, "BinaryCIF", row + 1, &mut first_model)? {
            atoms.push(atom);
        }
    }
    if atoms.is_empty() {
        return Err(StructureError::NoCoordinates("BinaryCIF file"));
    }
    Ok(molecule_from_atoms(atoms))
}

fn wanted_atom_column(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "group_pdb"
            | "id"
            | "type_symbol"
            | "label_atom_id"
            | "auth_atom_id"
            | "label_alt_id"
            | "label_comp_id"
            | "auth_comp_id"
            | "label_asym_id"
            | "auth_asym_id"
            | "label_seq_id"
            | "auth_seq_id"
            | "pdbx_pdb_ins_code"
            | "cartn_x"
            | "cartn_y"
            | "cartn_z"
            | "occupancy"
            | "b_iso_or_equiv"
            | "pdbx_pdb_model_num"
    )
}

struct BinaryColumn {
    values: Decoded,
    mask: Option<Decoded>,
}

fn binary_column_text(
    columns: &HashMap<String, BinaryColumn>,
    names: &[&str],
    row: usize,
) -> Option<String> {
    names.iter().find_map(|name| {
        let column = columns.get(*name)?;
        if column
            .mask
            .as_ref()
            .and_then(|mask| mask.integer(row))
            .is_some_and(|value| value != 0)
        {
            return None;
        }
        column.values.text(row)
    })
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

fn atom_from_fields<F, S>(
    fields: &F,
    format: &'static str,
    row: usize,
    first_model: &mut Option<String>,
) -> Result<Option<Atom>, StructureError>
where
    F: Fn(&[&str]) -> Option<S>,
    S: AsRef<str>,
{
    let value = |names: &[&str]| fields(names).map(|value| value.as_ref().to_string());
    let model = value(&["pdbx_pdb_model_num"]).unwrap_or_else(|| "1".into());
    if let Some(first) = first_model {
        if model != *first {
            return Ok(None);
        }
    } else {
        *first_model = Some(model);
    }
    let altloc = value(&["label_alt_id"]).unwrap_or_default();
    if !matches!(altloc.as_str(), "" | "." | "?" | "A") {
        return Ok(None);
    }
    let required = |names: &[&str], label: &str| {
        value(names).ok_or_else(|| StructureError::AtomSite {
            format,
            row,
            message: format!("missing {label}"),
        })
    };
    let number = |names: &[&str], label: &str| -> Result<f32, StructureError> {
        let raw = required(names, label)?;
        raw.parse::<f32>().map_err(|_| StructureError::AtomSite {
            format,
            row,
            message: format!("invalid {label} value '{raw}'"),
        })
    };
    let name = required(&["auth_atom_id", "label_atom_id"], "atom name")?;
    let residue_name = required(&["auth_comp_id", "label_comp_id"], "residue name")?;
    let residue_number_raw = required(&["auth_seq_id", "label_seq_id"], "residue sequence number")?;
    let residue_number =
        residue_number_raw
            .parse::<i32>()
            .map_err(|_| StructureError::AtomSite {
                format,
                row,
                message: format!("invalid residue sequence number '{residue_number_raw}'"),
            })?;
    let chain_id = value(&["auth_asym_id", "label_asym_id"]).unwrap_or_default();
    let serial = value(&["id"])
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(row as u32);
    let element = value(&["type_symbol"])
        .and_then(|value| Element::from_str(&value).ok())
        .unwrap_or_else(|| infer_element(&name));
    let insertion_code = value(&["pdbx_pdb_ins_code"])
        .and_then(|value| value.chars().next())
        .filter(|value| !matches!(*value, '.' | '?' | ' '));
    let occupancy = value(&["occupancy"])
        .and_then(|value| value.parse().ok())
        .unwrap_or(1.0);
    let b_factor = value(&["b_iso_or_equiv"])
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.0);
    Ok(Some(Atom {
        serial,
        name: name.to_ascii_uppercase(),
        element,
        residue_name: residue_name.to_ascii_uppercase(),
        residue_number,
        insertion_code,
        chain_id,
        position: Vec3::new(
            number(&["cartn_x"], "x coordinate")?,
            number(&["cartn_y"], "y coordinate")?,
            number(&["cartn_z"], "z coordinate")?,
        ),
        occupancy,
        b_factor,
        hetero: value(&["group_pdb"]).is_some_and(|value| value.eq_ignore_ascii_case("HETATM")),
    }))
}

fn molecule_from_atoms(atoms: Vec<Atom>) -> Molecule {
    let bonds = infer_bonds(&atoms, &[]);
    Molecule { atoms, bonds }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};

    use super::*;

    const MMCIF: &str = "data_demo\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\nATOM 1 N N . GLY A 7 ? 1.0 2.0 3.0 1.0 12.0 1\nHETATM 2 O O . HOH B 8 ? 4.0 5.0 6.0 1.0 20.0 1\n#\n";

    const PDBML: &str = r#"<?xml version="1.0"?>
<PDBx:datablock xmlns:PDBx="http://pdbml.pdb.org/schema/pdbx-v50.xsd">
<PDBx:atom_siteCategory><PDBx:atom_site id="1">
<PDBx:group_PDB>ATOM</PDBx:group_PDB><PDBx:type_symbol>C</PDBx:type_symbol>
<PDBx:label_atom_id>CA</PDBx:label_atom_id><PDBx:label_comp_id>ALA</PDBx:label_comp_id>
<PDBx:auth_asym_id>A</PDBx:auth_asym_id><PDBx:auth_seq_id>10</PDBx:auth_seq_id>
<PDBx:Cartn_x>1.25</PDBx:Cartn_x><PDBx:Cartn_y>2.5</PDBx:Cartn_y><PDBx:Cartn_z>3.75</PDBx:Cartn_z>
</PDBx:atom_site></PDBx:atom_siteCategory></PDBx:datablock>"#;

    #[test]
    fn parses_mmcif_atom_site() {
        let (molecule, format) = parse_structure(MMCIF.as_bytes(), "demo.cif").unwrap();
        assert_eq!(format, StructureFormat::Mmcif);
        assert_eq!(molecule.atoms.len(), 2);
        assert_eq!(molecule.atoms[0].residue_number, 7);
        assert!(molecule.atoms[1].hetero);
    }

    #[test]
    fn parses_gzipped_mmcif_by_magic_bytes() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(MMCIF.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        let (molecule, format) = parse_structure(&compressed, "demo.cif.gz").unwrap();
        assert_eq!(format, StructureFormat::Mmcif);
        assert_eq!(molecule.atoms.len(), 2);
    }

    #[test]
    fn gzip_decoder_stops_at_the_configured_limit() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&vec![b'A'; 2048]).unwrap();
        let compressed = encoder.finish().unwrap();
        assert!(matches!(
            decompress_gzip_with_limit(&compressed, 1024),
            Err(StructureError::TooLarge {
                actual: 1025,
                limit: 1024
            })
        ));
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

    #[test]
    fn parses_pdbml_atom_site() {
        let (molecule, format) = parse_structure(PDBML.as_bytes(), "demo.xml.gz").unwrap();
        assert_eq!(format, StructureFormat::Pdbml);
        assert_eq!(molecule.atoms.len(), 1);
        assert_eq!(molecule.atoms[0].name, "CA");
        assert_eq!(molecule.atoms[0].position, Vec3::new(1.25, 2.5, 3.75));
    }

    #[test]
    fn reports_non_coordinate_cif_and_validation_pdf() {
        assert!(matches!(
            parse_structure(b"data_sf\n_refln.index_h 1\n", "demo-sf.cif"),
            Err(StructureError::NoCoordinates(_))
        ));
        assert!(matches!(
            parse_structure(b"%PDF-1.7", "validation.pdf.gz"),
            Err(StructureError::ValidationPdf)
        ));
    }

    #[test]
    fn parses_binary_cif_atom_site() {
        let columns = vec![
            string_column("group_PDB", "ATOM"),
            string_column("id", "1"),
            string_column("type_symbol", "C"),
            string_column("auth_atom_id", "CA"),
            string_column("auth_comp_id", "ALA"),
            string_column("auth_asym_id", "A"),
            string_column("auth_seq_id", "7"),
            float_column("Cartn_x", 1.25),
            float_column("Cartn_y", 2.5),
            float_column("Cartn_z", 3.75),
        ];
        let root = object(vec![(
            "dataBlocks",
            Value::Array(vec![object(vec![(
                "categories",
                Value::Array(vec![object(vec![
                    ("name", Value::from("_atom_site")),
                    ("rowCount", Value::from(1)),
                    ("columns", Value::Array(columns)),
                ])]),
            )])]),
        )]);
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, &root).unwrap();
        let (molecule, format) = parse_structure(&bytes, "demo.bcif").unwrap();
        assert_eq!(format, StructureFormat::BinaryCif);
        assert_eq!(molecule.atoms.len(), 1);
        assert_eq!(molecule.atoms[0].position, Vec3::new(1.25, 2.5, 3.75));
    }

    fn object(values: Vec<(&str, Value)>) -> Value {
        Value::Map(
            values
                .into_iter()
                .map(|(name, value)| (Value::from(name), value))
                .collect(),
        )
    }

    fn byte_array_encoding(data_type: i64) -> Value {
        object(vec![
            ("kind", Value::from("ByteArray")),
            ("type", Value::from(data_type)),
        ])
    }

    fn string_column(name: &str, value: &str) -> Value {
        let indices = 0_i32.to_le_bytes().to_vec();
        let mut offsets = Vec::new();
        offsets.extend_from_slice(&0_i32.to_le_bytes());
        offsets.extend_from_slice(&(value.len() as i32).to_le_bytes());
        let string_encoding = object(vec![
            ("kind", Value::from("StringArray")),
            ("dataEncoding", Value::Array(vec![byte_array_encoding(3)])),
            ("stringData", Value::from(value)),
            ("offsetEncoding", Value::Array(vec![byte_array_encoding(3)])),
            ("offsets", Value::Binary(offsets)),
        ]);
        object(vec![
            ("name", Value::from(name)),
            (
                "data",
                object(vec![
                    ("data", Value::Binary(indices)),
                    ("encoding", Value::Array(vec![string_encoding])),
                ]),
            ),
            ("mask", Value::Nil),
        ])
    }

    fn float_column(name: &str, value: f32) -> Value {
        object(vec![
            ("name", Value::from(name)),
            (
                "data",
                object(vec![
                    ("data", Value::Binary(value.to_le_bytes().to_vec())),
                    ("encoding", Value::Array(vec![byte_array_encoding(32)])),
                ]),
            ),
            ("mask", Value::Nil),
        ])
    }
}
