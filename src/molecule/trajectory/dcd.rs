//! CHARMM/NAMD/X-PLOR DCD trajectories: Fortran unformatted records in either byte order.

use std::path::Path;

use glam::Vec3;

use super::{BinaryFile, Endian, Frame, FrameReader, TrajectoryError, TrajectoryFormat};

const FORMAT: &str = "DCD";
/// CHARMM time unit (AKMA) in picoseconds.
const AKMA_PS: f64 = 0.048_888_21;

pub struct DcdReader {
    file: BinaryFile,
    atom_count: usize,
    /// Indices of moving atoms when some atoms are fixed; later frames store only these.
    free_atoms: Option<Vec<usize>>,
    /// The complete first frame, which supplies the fixed atoms of later frames.
    first_frame: Option<Vec<Vec3>>,
    has_cell: bool,
    has_fourth_dimension: bool,
    charmm_version: i32,
    frames_offset: u64,
    first_frame_bytes: u64,
    frame_bytes: u64,
    frame_count: usize,
    first_step: f64,
    step_interval: f64,
    time_step_ps: f64,
}

impl DcdReader {
    pub fn open(path: &Path) -> Result<Self, TrajectoryError> {
        let error = |message: &str| TrajectoryError::format(FORMAT, message);
        let mut probe = BinaryFile::open(path, Endian::Little)?;
        let marker = probe.bytes(4)?;
        let endian = if i32::from_le_bytes(marker[..4].try_into().unwrap_or_default()) == 84 {
            Endian::Little
        } else if i32::from_be_bytes(marker[..4].try_into().unwrap_or_default()) == 84 {
            Endian::Big
        } else {
            return Err(error(
                "the header record has an unexpected size (64-bit record markers are not supported)",
            ));
        };
        let mut file = BinaryFile::open(path, endian)?;
        let header = read_record(&mut file, Some(84))?;
        if &header[..4] != b"CORD" {
            return Err(error(
                "velocity DCD files (VELD) do not contain coordinates",
            ));
        }
        let int = |index: usize| {
            let bytes: [u8; 4] = header[4 + index * 4..8 + index * 4]
                .try_into()
                .unwrap_or_default();
            match endian {
                Endian::Little => i32::from_le_bytes(bytes),
                Endian::Big => i32::from_be_bytes(bytes),
            }
        };
        let charmm = int(19) != 0;
        let fixed_count =
            usize::try_from(int(8)).map_err(|_| error("negative fixed-atom count"))?;
        let has_cell = charmm && int(10) != 0;
        let has_fourth_dimension = charmm && int(11) != 0;
        let delta = if charmm {
            f64::from(f32::from_bits(int(9) as u32))
        } else {
            // X-PLOR files store the time step as a double over two fields.
            let bytes: Vec<u8> = header[40..48].to_vec();
            let bytes: [u8; 8] = bytes.try_into().unwrap_or_default();
            match endian {
                Endian::Little => f64::from_le_bytes(bytes),
                Endian::Big => f64::from_be_bytes(bytes),
            }
        };
        // Title record.
        read_record(&mut file, None)?;
        let atoms = read_record(&mut file, Some(4))?;
        let atom_count = {
            let bytes: [u8; 4] = atoms[..4].try_into().unwrap_or_default();
            match endian {
                Endian::Little => i32::from_le_bytes(bytes),
                Endian::Big => i32::from_be_bytes(bytes),
            }
        };
        let atom_count = usize::try_from(atom_count).map_err(|_| error("negative atom count"))?;
        if fixed_count > atom_count {
            return Err(error("more fixed atoms than atoms"));
        }
        let free_atoms = if fixed_count > 0 {
            let record = read_record(&mut file, Some((atom_count - fixed_count) * 4))?;
            let indices = record
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| {
                    let value = match endian {
                        Endian::Little => i32::from_le_bytes(*bytes),
                        Endian::Big => i32::from_be_bytes(*bytes),
                    };
                    // Fortran indices start at 1.
                    usize::try_from(value - 1)
                        .ok()
                        .filter(|index| *index < atom_count)
                        .ok_or_else(|| error("free-atom index out of range"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(indices)
        } else {
            None
        };
        let frames_offset = file.position()?;
        let coordinate_bytes = |atoms: usize| (atoms as u64 * 4 + 8) * 3;
        let extra = |atoms: usize| {
            (if has_cell { 48 + 8 } else { 0 })
                + if has_fourth_dimension {
                    atoms as u64 * 4 + 8
                } else {
                    0
                }
        };
        let first_frame_bytes = coordinate_bytes(atom_count) + extra(atom_count);
        let moving = free_atoms.as_ref().map_or(atom_count, Vec::len);
        let frame_bytes = coordinate_bytes(moving) + extra(moving);
        let available = file.len.saturating_sub(frames_offset);
        // The frame count in the header is often stale; the file size is authoritative.
        let frame_count = if available < first_frame_bytes {
            0
        } else {
            1 + ((available - first_frame_bytes) / frame_bytes) as usize
        };
        let mut reader = Self {
            file,
            atom_count,
            free_atoms,
            first_frame: None,
            has_cell,
            has_fourth_dimension,
            charmm_version: int(19),
            frames_offset,
            first_frame_bytes,
            frame_bytes,
            frame_count,
            first_step: f64::from(int(1)),
            step_interval: f64::from(int(2).max(1)),
            time_step_ps: delta * AKMA_PS,
        };
        if reader.free_atoms.is_some() && frame_count > 0 {
            let first = reader.read_coordinates(0, atom_count)?;
            reader.first_frame = Some(first.0);
        }
        Ok(reader)
    }

    fn read_coordinates(
        &mut self,
        index: usize,
        atoms: usize,
    ) -> Result<(Vec<Vec3>, Option<[f64; 6]>), TrajectoryError> {
        let offset = if index == 0 {
            self.frames_offset
        } else {
            self.frames_offset + self.first_frame_bytes + (index as u64 - 1) * self.frame_bytes
        };
        self.file.seek(offset)?;
        let cell = if self.has_cell {
            let record = read_record(&mut self.file, Some(48))?;
            let values: Vec<f64> = record
                .as_chunks::<8>()
                .0
                .iter()
                .map(|bytes| match self.file.endian {
                    Endian::Little => f64::from_le_bytes(*bytes),
                    Endian::Big => f64::from_be_bytes(*bytes),
                })
                .collect();
            charmm_cell(&values, self.charmm_version)
        } else {
            None
        };
        let mut axes = [Vec::new(), Vec::new(), Vec::new()];
        for axis in &mut axes {
            let length = self.file.i32()?;
            if length as usize != atoms * 4 {
                return Err(TrajectoryError::format(
                    FORMAT,
                    format!("frame {index} has a coordinate record of {length} bytes"),
                ));
            }
            *axis = self.file.f32_array(atoms)?;
            self.file.i32()?;
        }
        if self.has_fourth_dimension {
            read_record(&mut self.file, Some(atoms * 4))?;
        }
        let positions = (0..atoms)
            .map(|atom| Vec3::new(axes[0][atom], axes[1][atom], axes[2][atom]))
            .collect();
        Ok((positions, cell))
    }
}

/// Periodic cell of a CHARMM/NAMD frame. Three layouts exist:
/// - angle cosines `A, cos gamma, B, cos beta, cos alpha, C` (NAMD 2.5+);
/// - the symmetric shape matrix `XX, XY, YY, XZ, YZ, ZZ` (CHARMM c36 and later);
/// - degrees `A, gamma, B, beta, alpha, C` (older writers).
fn charmm_cell(values: &[f64], charmm_version: i32) -> Option<[f64; 6]> {
    let [a, first, b, beta, second, c] = values.try_into().ok()?;
    let cosines = [first, beta, second]
        .iter()
        .all(|value| (-1.0..=1.0).contains(value));
    if cosines {
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return None;
        }
        let angle = |cosine: f64| 90.0 - cosine.asin().to_degrees();
        return Some([a, b, c, angle(second), angle(beta), angle(first)]);
    }
    if charmm_version >= 36 {
        let [xx, xy, yy, xz, yz, zz] = [a, first, b, beta, second, c];
        return super::cell_from_vectors([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]);
    }
    (a > 0.0 && b > 0.0 && c > 0.0).then_some([a, b, c, second, beta, first])
}

fn read_record(file: &mut BinaryFile, expected: Option<usize>) -> Result<Vec<u8>, TrajectoryError> {
    let length = file.i32()?;
    let length = usize::try_from(length)
        .map_err(|_| TrajectoryError::format(FORMAT, "negative record length"))?;
    if expected.is_some_and(|expected| expected != length) {
        return Err(TrajectoryError::format(
            FORMAT,
            format!("unexpected record length {length}"),
        ));
    }
    let data = file.bytes(length)?;
    if file.i32()? as usize != length {
        return Err(TrajectoryError::format(FORMAT, "mismatched record markers"));
    }
    Ok(data)
}

impl FrameReader for DcdReader {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::Dcd
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
        let (positions, cell) = match (&self.free_atoms, index) {
            (Some(free), 1..) => {
                let free = free.clone();
                let (moving, cell) = self.read_coordinates(index, free.len())?;
                let mut positions = self.first_frame.clone().unwrap_or_default();
                for (atom, position) in free.into_iter().zip(moving) {
                    positions[atom] = position;
                }
                (positions, cell)
            }
            _ => self.read_coordinates(index, self.atom_count)?,
        };
        let time = (self.time_step_ps > 0.0)
            .then_some((self.first_step + index as f64 * self.step_interval) * self.time_step_ps);
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

    #[test]
    fn charmm_cells_accept_angles_and_cosines() {
        let degrees = charmm_cell(&[10.0, 80.0, 20.0, 85.0, 70.0, 30.0], 24).unwrap();
        assert_eq!(degrees, [10.0, 20.0, 30.0, 70.0, 85.0, 80.0]);
        let cosines = charmm_cell(&[10.0, 0.5, 20.0, 0.0, 0.0, 30.0], 24).unwrap();
        assert!((cosines[5] - 60.0).abs() < 1e-9);
        assert!((cosines[3] - 90.0).abs() < 1e-9);
    }

    fn assert_cell(actual: [f64; 6], expected: [f64; 6]) {
        for (value, reference) in actual.iter().zip(expected) {
            assert!(
                (value - reference).abs() < 1e-3,
                "{actual:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn matches_mdanalysis_reference_values() {
        let Some(path) = test_data::file("adk_dims.dcd") else {
            return;
        };
        let mut reader = DcdReader::open(&path).unwrap();
        assert_eq!(reader.atom_count(), 3341);
        assert_eq!(reader.len(), 98);
        // legacy_DCD_adk_coords.npy: frames 5 and 29.
        let frame = reader.read_frame(5).unwrap();
        assert!(frame.positions[0].distance(Vec3::new(12.975_403, 8.556_466, -8.790_868)) < 1e-4);
        assert!(
            frame.positions[3340].distance(Vec3::new(6.213_671_7, 17.115_02, -6.094_524_4)) < 1e-4
        );
        let frame = reader.read_frame(29).unwrap();
        assert!(frame.positions[0].distance(Vec3::new(13.933_175, 6.668_233_4, -8.638_411)) < 1e-4);
        assert!(reader.read_frame(97).is_ok());
        assert!(reader.read_frame(98).is_err());

        if let Some(path) = test_data::file("tip125_tric_C36.dcd") {
            let mut reader = DcdReader::open(&path).unwrap();
            assert_eq!(reader.len(), 10);
            let frame = reader.read_frame(0).unwrap();
            assert_cell(
                frame.cell.unwrap(),
                [35.44604, 35.06156, 34.1585, 91.32802, 61.73521, 44.40703],
            );
            assert_cell(
                reader.read_frame(9).unwrap().cell.unwrap(),
                [31.99748, 30.21518, 35.24292, 95.85821, 71.08429, 31.85939],
            );
        }
        if let Some(path) = test_data::file("SiN_tric_namd.dcd") {
            let mut reader = DcdReader::open(&path).unwrap();
            assert_cell(
                reader.read_frame(0).unwrap().cell.unwrap(),
                [38.426594, 38.393101, 44.7598, 90.0, 90.0, 60.028915],
            );
        }
    }
}
