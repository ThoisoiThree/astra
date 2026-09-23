//! AMBER NetCDF trajectories (NetCDF-3 classic and 64-bit offset formats).
//!
//! Only the parts of the NetCDF-3 header needed to locate the `coordinates`, `time`,
//! `cell_lengths` and `cell_angles` record variables are decoded.

use std::path::Path;

use super::{BinaryFile, Endian, Frame, FrameReader, TrajectoryError, TrajectoryFormat};

const FORMAT: &str = "NetCDF";
const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;
const STREAMING: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NcType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
}

impl NcType {
    fn from_code(code: u32) -> Result<Self, TrajectoryError> {
        Ok(match code {
            1 => Self::Byte,
            2 => Self::Char,
            3 => Self::Short,
            4 => Self::Int,
            5 => Self::Float,
            6 => Self::Double,
            _ => {
                return Err(TrajectoryError::format(
                    FORMAT,
                    format!("unknown type {code}"),
                ));
            }
        })
    }

    const fn size(self) -> u64 {
        match self {
            Self::Byte | Self::Char => 1,
            Self::Short => 2,
            Self::Int | Self::Float => 4,
            Self::Double => 8,
        }
    }
}

#[derive(Debug, Clone)]
struct Variable {
    name: String,
    dimensions: Vec<usize>,
    kind: NcType,
    begin: u64,
    scale: Option<f64>,
}

#[derive(Debug, Clone)]
struct Dimension {
    length: u64,
}

pub struct NetCdfReader {
    file: BinaryFile,
    atom_count: usize,
    frame_count: usize,
    record_size: u64,
    coordinates: Variable,
    time: Option<Variable>,
    cell_lengths: Option<Variable>,
    cell_angles: Option<Variable>,
}

impl NetCdfReader {
    pub fn open(path: &Path) -> Result<Self, TrajectoryError> {
        let error = |message: String| TrajectoryError::format(FORMAT, message);
        let mut file = BinaryFile::open(path, Endian::Big)?;
        let magic = file.bytes(4)?;
        let offset_64 = match magic.as_slice() {
            b"CDF\x01" => false,
            b"CDF\x02" => true,
            _ => return Err(error("not a NetCDF-3 file".into())),
        };
        let record_count = file.u32()?;

        let dimensions = read_list(&mut file, NC_DIMENSION, |file| {
            read_name(file)?;
            Ok(Dimension {
                length: u64::from(file.u32()?),
            })
        })?;
        let record_dimension = dimensions
            .iter()
            .position(|dimension| dimension.length == 0);
        skip_attributes(&mut file)?;
        let variables = read_list(&mut file, NC_VARIABLE, |file| {
            let name = read_name(file)?;
            let count = file.u32()? as usize;
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                ids.push(file.u32()? as usize);
            }
            let scale = read_scale_factor(file)?;
            let kind = NcType::from_code(file.u32()?)?;
            let _size = file.u32()?;
            let begin = if offset_64 {
                file.i64()? as u64
            } else {
                u64::from(file.u32()?)
            };
            Ok(Variable {
                name,
                dimensions: ids,
                kind,
                begin,
                scale,
            })
        })?;
        if variables
            .iter()
            .flat_map(|variable| &variable.dimensions)
            .any(|id| *id >= dimensions.len())
        {
            return Err(error("a variable refers to an unknown dimension".into()));
        }

        let find = |name: &str| {
            variables
                .iter()
                .find(|variable| variable.name == name)
                .cloned()
        };
        let coordinates = find("coordinates").ok_or_else(|| {
            error("no 'coordinates' variable; is this an AMBER trajectory?".into())
        })?;
        let is_record = |variable: &Variable| {
            record_dimension.is_some_and(|record| variable.dimensions.first() == Some(&record))
        };
        if !is_record(&coordinates) || coordinates.dimensions.len() != 3 {
            return Err(error(
                "'coordinates' must have dimensions (frame, atom, spatial)".into(),
            ));
        }
        if dimensions[coordinates.dimensions[2]].length != 3 {
            return Err(error(
                "'coordinates' must have three spatial components".into(),
            ));
        }
        if !matches!(coordinates.kind, NcType::Float | NcType::Double) {
            return Err(error("'coordinates' must be floating point".into()));
        }
        let atom_count = dimensions[coordinates.dimensions[1]].length as usize;

        // Every record holds one slice of each record variable, each padded to 4 bytes,
        // except that a single record variable is stored without padding.
        let slice_size = |variable: &Variable| {
            variable.dimensions[1..]
                .iter()
                .map(|id| dimensions[*id].length)
                .product::<u64>()
                * variable.kind.size()
        };
        let record_variables: Vec<&Variable> = variables.iter().filter(|v| is_record(v)).collect();
        let record_size = if record_variables.len() == 1 {
            slice_size(record_variables[0])
        } else {
            record_variables
                .iter()
                .map(|variable| slice_size(variable).next_multiple_of(4))
                .sum()
        };
        let frame_count = if record_count == STREAMING {
            let first = record_variables
                .iter()
                .map(|variable| variable.begin)
                .min()
                .unwrap_or(0);
            (file.len.saturating_sub(first) / record_size.max(1)) as usize
        } else {
            record_count as usize
        };
        let optional = |name: &str, components: u64| {
            find(name).filter(|variable| {
                is_record(variable)
                    && variable.dimensions[1..]
                        .iter()
                        .map(|id| dimensions[*id].length)
                        .product::<u64>()
                        == components
            })
        };
        let time = optional("time", 1);
        let cell_lengths = optional("cell_lengths", 3);
        let cell_angles = optional("cell_angles", 3);
        Ok(Self {
            file,
            atom_count,
            frame_count,
            record_size,
            coordinates,
            time,
            cell_lengths,
            cell_angles,
        })
    }

    fn values(
        &mut self,
        variable: &Variable,
        frame: usize,
        count: usize,
    ) -> Result<Vec<f64>, TrajectoryError> {
        self.file
            .seek(variable.begin + frame as u64 * self.record_size)?;
        let values: Vec<f64> = match variable.kind {
            NcType::Float => self
                .file
                .f32_array(count)?
                .into_iter()
                .map(f64::from)
                .collect(),
            NcType::Double => self.file.f64_array(count)?,
            NcType::Int => (0..count)
                .map(|_| self.file.i32().map(f64::from))
                .collect::<Result<_, _>>()?,
            _ => {
                return Err(TrajectoryError::format(
                    FORMAT,
                    format!("unsupported type for '{}'", variable.name),
                ));
            }
        };
        let scale = variable.scale.unwrap_or(1.0);
        Ok(if scale == 1.0 {
            values
        } else {
            values.into_iter().map(|value| value * scale).collect()
        })
    }
}

fn read_name(file: &mut BinaryFile) -> Result<String, TrajectoryError> {
    let length = file.u32()? as usize;
    if length > 1 << 16 {
        return Err(TrajectoryError::format(FORMAT, "name too long"));
    }
    let bytes = file.bytes(length.next_multiple_of(4))?;
    Ok(String::from_utf8_lossy(&bytes[..length]).into_owned())
}

/// Reads a tagged list; `ABSENT` is encoded as two zero words.
fn read_list<T>(
    file: &mut BinaryFile,
    tag: u32,
    mut item: impl FnMut(&mut BinaryFile) -> Result<T, TrajectoryError>,
) -> Result<Vec<T>, TrajectoryError> {
    let found = file.u32()?;
    let count = file.u32()? as usize;
    if found == 0 && count == 0 {
        return Ok(Vec::new());
    }
    if found != tag {
        return Err(TrajectoryError::format(
            FORMAT,
            format!("expected header tag {tag:#x}, found {found:#x}"),
        ));
    }
    if count > 1 << 16 {
        return Err(TrajectoryError::format(
            FORMAT,
            "implausibly long header list",
        ));
    }
    (0..count).map(|_| item(file)).collect()
}

fn skip_attributes(file: &mut BinaryFile) -> Result<(), TrajectoryError> {
    read_list(file, NC_ATTRIBUTE, |file| {
        read_name(file)?;
        let kind = NcType::from_code(file.u32()?)?;
        let count = u64::from(file.u32()?);
        file.skip((count * kind.size()).next_multiple_of(4))?;
        Ok(())
    })
    .map(|_| ())
}

/// Returns a variable's `scale_factor` attribute, skipping the others.
fn read_scale_factor(file: &mut BinaryFile) -> Result<Option<f64>, TrajectoryError> {
    let mut scale = None;
    read_list(file, NC_ATTRIBUTE, |file| {
        let name = read_name(file)?;
        let kind = NcType::from_code(file.u32()?)?;
        let count = u64::from(file.u32()?);
        if name == "scale_factor" && count == 1 && matches!(kind, NcType::Float | NcType::Double) {
            scale = Some(if kind == NcType::Float {
                f64::from(file.f32()?)
            } else {
                file.f64()?
            });
        } else {
            file.skip((count * kind.size()).next_multiple_of(4))?;
        }
        Ok(())
    })?;
    Ok(scale)
}

impl FrameReader for NetCdfReader {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::NetCdf
    }

    fn atom_count(&self) -> usize {
        self.atom_count
    }

    fn len(&self) -> usize {
        self.frame_count
    }

    fn read_frame(&mut self, index: usize) -> Result<Frame, TrajectoryError> {
        if index >= self.frame_count {
            return Err(TrajectoryError::FrameOutOfRange {
                index,
                count: self.frame_count,
            });
        }
        let coordinates = self.coordinates.clone();
        let values = self.values(&coordinates, index, self.atom_count * 3)?;
        let positions = values
            .as_chunks::<3>()
            .0
            .iter()
            .map(|[x, y, z]| glam::Vec3::new(*x as f32, *y as f32, *z as f32))
            .collect();
        let time = match self.time.clone() {
            Some(variable) => self.values(&variable, index, 1)?.first().copied(),
            None => None,
        };
        let cell = match (self.cell_lengths.clone(), self.cell_angles.clone()) {
            (Some(lengths), Some(angles)) => {
                let lengths = self.values(&lengths, index, 3)?;
                let angles = self.values(&angles, index, 3)?;
                (lengths.iter().all(|value| *value > 0.0)).then(|| {
                    [
                        lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2],
                    ]
                })
            }
            _ => None,
        };
        Ok(Frame {
            positions,
            time,
            cell,
        })
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::molecule::trajectory::test_data;

    fn close(actual: glam::Vec3, expected: [f32; 3]) -> bool {
        actual.distance(glam::Vec3::from(expected)) < 1e-3
    }

    #[test]
    fn matches_mdanalysis_reference_values() {
        if let Some(path) = test_data::file("Amber/posfor.ncdf") {
            let mut reader = NetCdfReader::open(&path).unwrap();
            let frame = reader.read_frame(0).unwrap();
            assert!(close(
                frame.positions[0],
                [-0.11980818, 18.70524979, 11.6477766]
            ));
            assert!(close(
                frame.positions[2],
                [-0.60952115, 19.47885513, 11.22137547]
            ));
            assert!((frame.time.unwrap() - 35.02).abs() < 1e-3);
            assert!(frame.cell.is_none());
            let frame = reader.read_frame(1).unwrap();
            assert!(close(
                frame.positions[1],
                [-0.46643803, 18.60186768, 12.646698]
            ));
            assert!((frame.time.unwrap() - 35.04).abs() < 1e-3);
        }
        if let Some(path) = test_data::file("Amber/ace_tip3p.nc") {
            let mut reader = NetCdfReader::open(&path).unwrap();
            let frame = reader.read_frame(0).unwrap();
            assert!(close(frame.positions[0], [15.249873, 12.578178, 15.191731]));
            let cell = frame.cell.unwrap();
            assert!((cell[0] - 28.81876287).abs() < 1e-3 && (cell[5] - 90.0).abs() < 1e-6);
            let frame = reader.read_frame(8).unwrap();
            assert!(close(frame.positions[2], [16.03358, 16.183628, 14.02995]));
            assert!((frame.cell.unwrap()[1] - 26.55555665).abs() < 1e-3);
            assert!(reader.read_frame(reader.len()).is_err());
        }
    }
}
