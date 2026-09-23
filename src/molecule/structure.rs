//! Structure file entry point: decompression, format detection and dispatch.

use std::{borrow::Cow, io::Read};

use flate2::read::GzDecoder;
use glam::Vec3;
use thiserror::Error;

use super::{
    Molecule, PdbError, bcif,
    cif::{self, OwnedCategory, OwnedDocument},
    coordinates, pdb,
};

pub const MAX_DECOMPRESSED_STRUCTURE_SIZE: u64 = 512 * 1024 * 1024;
const MAX_GZIP_EXPANSION_RATIO: u64 = 200;
const GZIP_RATIO_CHECK_THRESHOLD: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureFormat {
    Pdb,
    Mmcif,
    BinaryCif,
    Pdbml,
    Gro,
    Xyz,
    Pqr,
}

impl StructureFormat {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pdb => "legacy PDB",
            Self::Mmcif => "PDBx/mmCIF",
            Self::BinaryCif => "BinaryCIF",
            Self::Pdbml => "PDBML/XML",
            Self::Gro => "GROMACS GRO",
            Self::Xyz => "XYZ",
            Self::Pqr => "PQR",
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
    #[error("{format} line {line}: {message}")]
    Coordinates {
        format: &'static str,
        line: usize,
        message: String,
    },
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
    #[error("unsupported structure format; expected PDB, mmCIF, BinaryCIF, PDBML/XML, GRO or XYZ")]
    Unsupported,
}

/// A parsed structure: topology with the first model's coordinates plus later frames.
#[derive(Debug, Clone)]
pub struct ParsedStructure {
    pub molecule: Molecule,
    pub format: StructureFormat,
    /// Coordinates of further models or frames in the file, one position per atom.
    pub frames: Vec<Vec<Vec3>>,
}

/// Parses a coordinate file, transparently handling gzip by its magic bytes.
/// `filename` is only used as a format hint; content sniffing remains the fallback.
pub fn parse_structure(bytes: &[u8], filename: &str) -> Result<ParsedStructure, StructureError> {
    let bytes = decompress_if_needed(bytes)?;
    if bytes.starts_with(b"%PDF-") {
        return Err(StructureError::ValidationPdf);
    }
    let format = filename_hint(filename)
        .or_else(|| sniff_format(&bytes))
        .ok_or(StructureError::Unsupported)?;
    let (molecule, frames) = match format {
        StructureFormat::Pdb => {
            let document = pdb::parse_pdb_document(text(&bytes, format)?)?;
            (document.molecule, document.frames)
        }
        StructureFormat::Mmcif => {
            let document = cif::parse_text(text(&bytes, format)?)?;
            let structure = cif::structure_from_cif(&document, "PDBx/mmCIF")?;
            (structure.molecule, structure.frames)
        }
        StructureFormat::BinaryCif => {
            let document = bcif::parse_binary(&bytes)?;
            let structure = cif::structure_from_cif(&document, "BinaryCIF")?;
            (structure.molecule, structure.frames)
        }
        StructureFormat::Pdbml => {
            let document = parse_pdbml(text(&bytes, format)?)?;
            let structure = cif::structure_from_cif(&document, "PDBML/XML")?;
            (structure.molecule, structure.frames)
        }
        StructureFormat::Pqr => {
            let document = pdb::parse_pqr_document(text(&bytes, format)?)?;
            (document.molecule, document.frames)
        }
        StructureFormat::Gro => coordinates::parse_gro(text(&bytes, format)?)?,
        StructureFormat::Xyz => coordinates::parse_xyz(text(&bytes, format)?)?,
    };
    Ok(ParsedStructure {
        molecule,
        format,
        frames,
    })
}

pub(crate) fn decompress_if_needed(bytes: &[u8]) -> Result<Cow<'_, [u8]>, StructureError> {
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
    } else if lower.ends_with(".cif") || lower.ends_with(".mmcif") || lower.ends_with(".pdbx") {
        Some(StructureFormat::Mmcif)
    } else if lower.ends_with(".xml") {
        Some(StructureFormat::Pdbml)
    } else if lower.ends_with(".pdb") || lower.ends_with(".ent") {
        Some(StructureFormat::Pdb)
    } else if lower.ends_with(".pqr") {
        Some(StructureFormat::Pqr)
    } else if lower.ends_with(".gro") {
        Some(StructureFormat::Gro)
    } else if lower.ends_with(".xyz") {
        Some(StructureFormat::Xyz)
    } else {
        None
    }
}

/// Whether a file name denotes a structure format this reader supports.
pub fn is_structure_filename(filename: &str) -> bool {
    filename_hint(filename).is_some()
}

fn sniff_format(bytes: &[u8]) -> Option<StructureFormat> {
    let prefix = bytes.get(..bytes.len().min(4096)).unwrap_or(bytes);
    let text = std::str::from_utf8(prefix)
        .ok()
        .map(|text| text.trim_start_matches('\u{feff}').trim_start());
    if let Some(text) = text {
        // CIF files may open with comment lines before the first data block.
        let first_content = text
            .lines()
            .map(str::trim_start)
            .find(|line| !line.is_empty() && !line.starts_with('#'))
            .unwrap_or_default();
        if first_content.starts_with("data_") || first_content.starts_with("global_") {
            return Some(StructureFormat::Mmcif);
        }
        if text.starts_with("<?xml") || text.starts_with('<') {
            return Some(StructureFormat::Pdbml);
        }
        if text.lines().any(|line| {
            matches!(
                line.get(..line.len().min(6)).unwrap_or(line).trim(),
                "ATOM" | "HETATM" | "HEADER" | "MODEL" | "CRYST1"
            )
        }) {
            return Some(StructureFormat::Pdb);
        }
        let mut lines = text.lines();
        let first = lines.next().unwrap_or_default().trim();
        let second = lines.next().unwrap_or_default().trim();
        if first.parse::<usize>().is_ok() {
            return Some(StructureFormat::Xyz);
        }
        if second.parse::<usize>().is_ok() {
            return Some(StructureFormat::Gro);
        }
    }
    bytes
        .first()
        .is_some_and(|byte| byte & 0x80 != 0)
        .then_some(StructureFormat::BinaryCif)
}

fn text(bytes: &[u8], format: StructureFormat) -> Result<&str, StructureError> {
    std::str::from_utf8(bytes).map_err(|source| StructureError::Utf8 {
        format: format.label(),
        source,
    })
}

/// Reads every `<PDBx:xxxCategory>` of a PDBML document into generic tables. Row
/// attributes (such as `id`) and child elements both become columns.
fn parse_pdbml(input: &str) -> Result<OwnedDocument, StructureError> {
    let mut document = OwnedDocument::default();
    let mut cursor = 0;
    let mut text_start = 0;
    let mut category: Option<String> = None;
    let mut row_open = false;
    let mut field_name: Option<String> = None;
    while let Some(relative) = input[cursor..].find('<') {
        let tag_start = cursor + relative;
        if row_open && let (Some(name), Some(field)) = (&category, &field_name) {
            let value = xml_unescape(input[text_start..tag_start].trim())?;
            if !value.is_empty()
                && let Some(table) = document.categories.get_mut(name)
            {
                let row = table.rows.len() - 1;
                table.set(row, field, value);
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
        if let Some(table_name) = name.strip_suffix("category") {
            category = (!closing && !self_closing).then(|| table_name.to_string());
            row_open = false;
            continue;
        }
        let Some(current) = category.clone() else {
            continue;
        };
        if name == current {
            if closing {
                row_open = false;
            } else {
                let table = document
                    .categories
                    .entry(current.clone())
                    .or_insert_with(OwnedCategory::default);
                table.rows.push(Vec::new());
                let row = table.rows.len() - 1;
                for (attribute, value) in xml_attributes(body) {
                    table.set(row, &attribute, xml_unescape(&value)?);
                }
                row_open = !self_closing;
            }
            field_name = None;
        } else if row_open {
            if closing || self_closing {
                field_name = None;
            } else {
                field_name = Some(name);
            }
        }
    }
    if document
        .categories
        .get("atom_site")
        .is_none_or(|table| table.rows.is_empty())
    {
        return Err(StructureError::NoCoordinates("PDBML/XML file"));
    }
    Ok(document)
}

fn xml_attributes(tag: &str) -> Vec<(String, String)> {
    let mut attributes = Vec::new();
    let mut rest = tag
        .split_once(char::is_whitespace)
        .map_or("", |(_, rest)| rest);
    while let Some((name, after)) = rest.split_once('=') {
        let name = name.trim().to_ascii_lowercase();
        let after = after.trim_start();
        let Some(quote) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            break;
        };
        let Some(end) = after[1..].find(quote) else {
            break;
        };
        if !name.contains(':') {
            attributes.push((name, after[1..1 + end].to_string()));
        }
        rest = &after[end + 2..];
    }
    attributes
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};
    use rmpv::Value;

    use super::*;

    const MMCIF: &str = "data_demo\n#\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\nATOM 1 N N . GLY A 7 ? 1.0 2.0 3.0 1.0 12.0 1\nHETATM 2 O O . HOH B 8 ? 4.0 5.0 6.0 1.0 20.0 1\n#\n";

    const PDBML: &str = r#"<?xml version="1.0"?>
<PDBx:datablock xmlns:PDBx="http://pdbml.pdb.org/schema/pdbx-v50.xsd">
<PDBx:cellCategory><PDBx:cell entry_id="1ABC"><PDBx:length_a>40.0</PDBx:length_a>
<PDBx:length_b>50.0</PDBx:length_b><PDBx:length_c>60.0</PDBx:length_c>
<PDBx:angle_alpha>90</PDBx:angle_alpha><PDBx:angle_beta>90</PDBx:angle_beta>
<PDBx:angle_gamma>90</PDBx:angle_gamma></PDBx:cell></PDBx:cellCategory>
<PDBx:atom_siteCategory><PDBx:atom_site id="1">
<PDBx:group_PDB>ATOM</PDBx:group_PDB><PDBx:type_symbol>C</PDBx:type_symbol>
<PDBx:label_atom_id>CA</PDBx:label_atom_id><PDBx:label_comp_id>ALA</PDBx:label_comp_id>
<PDBx:auth_asym_id>A</PDBx:auth_asym_id><PDBx:auth_seq_id>10</PDBx:auth_seq_id>
<PDBx:Cartn_x>1.25</PDBx:Cartn_x><PDBx:Cartn_y>2.5</PDBx:Cartn_y><PDBx:Cartn_z>3.75</PDBx:Cartn_z>
<PDBx:pdbx_PDB_ins_code xsi:nil="true" />
</PDBx:atom_site></PDBx:atom_siteCategory></PDBx:datablock>"#;

    #[test]
    fn parses_mmcif_atom_site() {
        let parsed = parse_structure(MMCIF.as_bytes(), "demo.cif").unwrap();
        assert_eq!(parsed.format, StructureFormat::Mmcif);
        assert_eq!(parsed.molecule.atoms.len(), 2);
        assert_eq!(parsed.molecule.atoms[0].residue_number, 7);
        assert!(parsed.molecule.atoms[1].hetero);
    }

    #[test]
    fn parses_gzipped_mmcif_by_magic_bytes() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(MMCIF.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        let parsed = parse_structure(&compressed, "demo.cif.gz").unwrap();
        assert_eq!(parsed.format, StructureFormat::Mmcif);
        assert_eq!(parsed.molecule.atoms.len(), 2);
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
    fn parses_pdbml_categories() {
        let parsed = parse_structure(PDBML.as_bytes(), "demo.xml.gz").unwrap();
        assert_eq!(parsed.format, StructureFormat::Pdbml);
        let molecule = parsed.molecule;
        assert_eq!(molecule.atoms.len(), 1);
        assert_eq!(molecule.atoms[0].name, "CA");
        assert_eq!(molecule.atoms[0].serial, 1);
        assert_eq!(molecule.atoms[0].position, Vec3::new(1.25, 2.5, 3.75));
        assert_eq!(molecule.info.crystal.unwrap().cell.c, 60.0);
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
    fn sniffs_formats_without_extensions() {
        assert_eq!(
            sniff_format(b"3\ncomment\nO 0 0 0\nH 1 0 0\nH 0 1 0\n"),
            Some(StructureFormat::Xyz)
        );
        assert_eq!(
            sniff_format(b"Water\n    1\n    1SOL     OW    1   0.126   1.624   1.679\n1 1 1\n"),
            Some(StructureFormat::Gro)
        );
        assert_eq!(sniff_format(MMCIF.as_bytes()), Some(StructureFormat::Mmcif));
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
        let parsed = parse_structure(&bytes, "demo.bcif").unwrap();
        assert_eq!(parsed.format, StructureFormat::BinaryCif);
        assert_eq!(parsed.molecule.atoms.len(), 1);
        assert_eq!(
            parsed.molecule.atoms[0].position,
            Vec3::new(1.25, 2.5, 3.75)
        );
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
