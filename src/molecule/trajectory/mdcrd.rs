//! AMBER ASCII trajectories (`mdcrd`/`trj`): a title line, then `10F8.3` coordinates for
//! every frame, optionally followed by a `3F8.3` box line. The file does not record the
//! atom count, so it comes from the structure.

use std::{
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
};

use super::{Frame, FrameReader, TrajectoryError, TrajectoryFormat};

const FORMAT: &str = "AMBER mdcrd";

pub struct MdcrdReader {
    reader: BufReader<File>,
    atom_count: usize,
    lines_per_frame: usize,
    has_box: bool,
    offsets: Vec<u64>,
}

impl MdcrdReader {
    pub fn open(path: &Path, atom_count: usize) -> Result<Self, TrajectoryError> {
        if atom_count == 0 {
            return Err(TrajectoryError::format(
                FORMAT,
                "the structure has no atoms",
            ));
        }
        let mut reader = BufReader::with_capacity(1 << 16, File::open(path)?);
        let lines_per_frame = (atom_count * 3).div_ceil(10);
        let mut line = Vec::new();
        let mut offset = reader.read_until(b'\n', &mut line)? as u64;
        let mut offsets = Vec::new();
        let mut has_box = None;
        'frames: loop {
            let start = offset;
            for _ in 0..lines_per_frame {
                line.clear();
                let read = reader.read_until(b'\n', &mut line)?;
                if read == 0 || line.iter().all(u8::is_ascii_whitespace) {
                    break 'frames;
                }
                offset += read as u64;
            }
            offsets.push(start);
            // A three-value line after the coordinates is the periodic box.
            let expect_box = match has_box {
                Some(value) => value,
                None => {
                    let position = reader.stream_position()?;
                    line.clear();
                    reader.read_until(b'\n', &mut line)?;
                    reader.seek(SeekFrom::Start(position))?;
                    let value = fields(&line).len() == 3 && atom_count * 3 > 3;
                    has_box = Some(value);
                    value
                }
            };
            if expect_box {
                line.clear();
                let read = reader.read_until(b'\n', &mut line)?;
                if read == 0 {
                    break;
                }
                offset += read as u64;
            }
        }
        Ok(Self {
            reader,
            atom_count,
            lines_per_frame,
            has_box: has_box.unwrap_or(false),
            offsets,
        })
    }
}

/// Splits fixed-width `F8.3` fields, which may run together for large values; falls back
/// to whitespace for files written with other widths.
fn fields(line: &[u8]) -> Vec<f32> {
    let text = String::from_utf8_lossy(line);
    let text = text.trim_end_matches(['\r', '\n']);
    let trimmed = text.trim_end();
    if trimmed.len().is_multiple_of(8)
        && let Some(values) = trimmed
            .as_bytes()
            .chunks(8)
            .map(|chunk| std::str::from_utf8(chunk).ok()?.trim().parse::<f32>().ok())
            .collect::<Option<Vec<_>>>()
    {
        return values;
    }
    trimmed
        .split_whitespace()
        .filter_map(|value| value.parse::<f32>().ok())
        .collect()
}

impl FrameReader for MdcrdReader {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::Mdcrd
    }

    fn atom_count(&self) -> usize {
        self.atom_count
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }

    fn read_frame(&mut self, index: usize) -> Result<Frame, TrajectoryError> {
        let offset = *self
            .offsets
            .get(index)
            .ok_or(TrajectoryError::FrameOutOfRange {
                index,
                count: self.offsets.len(),
            })?;
        self.reader.seek(SeekFrom::Start(offset))?;
        let mut values = Vec::with_capacity(self.atom_count * 3);
        let mut line = Vec::new();
        for _ in 0..self.lines_per_frame {
            line.clear();
            self.reader.read_until(b'\n', &mut line)?;
            values.extend(fields(&line));
        }
        if values.len() != self.atom_count * 3 {
            return Err(TrajectoryError::format(
                FORMAT,
                format!(
                    "frame {index} has {} values; expected {} for {} atoms",
                    values.len(),
                    self.atom_count * 3,
                    self.atom_count
                ),
            ));
        }
        let cell = if self.has_box {
            line.clear();
            self.reader.read_until(b'\n', &mut line)?;
            match fields(&line).as_slice() {
                [a, b, c] if *a > 0.0 && *b > 0.0 && *c > 0.0 => Some([
                    f64::from(*a),
                    f64::from(*b),
                    f64::from(*c),
                    90.0,
                    90.0,
                    90.0,
                ]),
                _ => None,
            }
        } else {
            None
        };
        Ok(Frame {
            positions: super::positions_from_xyz(&values, 1.0),
            time: None,
            cell,
        })
    }
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::molecule::trajectory::test_data;

    #[test]
    fn splits_fixed_width_fields() {
        assert_eq!(
            fields(b"  32.555-124.652  14.213\n"),
            vec![32.555, -124.652, 14.213]
        );
        assert_eq!(fields(b"1.5 2.5 3.5\n"), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn detects_box_lines() {
        let directory = std::env::temp_dir().join(format!("astra-mdcrd-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("box.mdcrd");
        // Two atoms: six values per frame on one line, then a box line.
        std::fs::write(
            &path,
            "title\n   1.000   2.000   3.000   4.000   5.000   6.000\n  10.000  11.000  12.000\n   \
             1.500   2.000   3.000   4.000   5.000   6.000\n  10.000  11.000  12.000\n",
        )
        .unwrap();
        let mut reader = MdcrdReader::open(&path, 2).unwrap();
        assert_eq!(reader.len(), 2);
        let frame = reader.read_frame(1).unwrap();
        assert_eq!(frame.positions[0], Vec3::new(1.5, 2.0, 3.0));
        assert_eq!(frame.cell.unwrap()[2], 12.0);
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn matches_vmd_reference_centroids() {
        let Some(path) = test_data::file("Amber/ache.mdcrd") else {
            return;
        };
        let mut reader = MdcrdReader::open(&path, 252).unwrap();
        assert_eq!(reader.len(), 11);
        // Sum over frames of the geometric center, summed over x, y and z (VMD).
        let total: f64 = (0..reader.len())
            .map(|index| {
                let frame = reader.read_frame(index).unwrap();
                let center = frame.positions.iter().copied().sum::<Vec3>() / 252.0;
                f64::from(center.x + center.y + center.z)
            })
            .sum();
        assert!((total - 472.259_215_950_965_9).abs() < 1e-2, "{total}");
    }
}
