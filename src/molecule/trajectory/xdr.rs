//! GROMACS XTC (compressed) and TRR (full-precision) trajectories, both XDR encoded.

use std::path::Path;

use glam::Vec3;

use super::{
    BinaryFile, Endian, Frame, FrameReader, TrajectoryError, TrajectoryFormat, cell_from_vectors,
    positions_from_xyz,
};

/// GROMACS lengths are in nanometers.
const NM_TO_ANGSTROM: f32 = 10.0;
const XTC_MAGIC: i32 = 1995;
const TRR_MAGIC: i32 = 1993;

fn box_cell(values: &[f64]) -> Option<[f64; 6]> {
    let vectors = [
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ]
    .map(|row| row.map(|value| value * f64::from(NM_TO_ANGSTROM)));
    cell_from_vectors(vectors)
}

fn out_of_range(index: usize, count: usize) -> TrajectoryError {
    TrajectoryError::FrameOutOfRange { index, count }
}

// ---------------------------------------------------------------------------------------
// XTC

pub struct XtcReader {
    file: BinaryFile,
    atom_count: usize,
    offsets: Vec<u64>,
}

impl XtcReader {
    pub fn open(path: &Path) -> Result<Self, TrajectoryError> {
        let mut file = BinaryFile::open(path, Endian::Big)?;
        let mut offsets = Vec::new();
        let mut atom_count = None;
        let mut offset = 0_u64;
        // Index frames by walking their headers; each frame records its compressed size.
        while offset + 4 * 14 <= file.len {
            file.seek(offset)?;
            if file.i32()? != XTC_MAGIC {
                return Err(TrajectoryError::format(
                    "XTC",
                    format!("bad frame magic at byte {offset}"),
                ));
            }
            let atoms = usize::try_from(file.i32()?)
                .map_err(|_| TrajectoryError::format("XTC", "negative atom count"))?;
            if *atom_count.get_or_insert(atoms) != atoms {
                return Err(TrajectoryError::format(
                    "XTC",
                    "the atom count changes between frames",
                ));
            }
            // step, time, box[9], natoms again
            file.skip(4 + 4 + 36 + 4)?;
            let data_bytes = if atoms <= 9 {
                atoms as u64 * 12
            } else {
                // precision, minint[3], maxint[3], smallidx
                file.skip(4 * 8)?;
                let bytes = u64::from(file.u32()?);
                4 * 8 + 4 + bytes.next_multiple_of(4)
            };
            let end = offset + 4 * 14 + data_bytes;
            if end > file.len {
                // A truncated last frame from an interrupted run.
                break;
            }
            offsets.push(offset);
            offset = end;
        }
        Ok(Self {
            file,
            atom_count: atom_count.unwrap_or(0),
            offsets,
        })
    }
}

impl FrameReader for XtcReader {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::Xtc
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
            .ok_or_else(|| out_of_range(index, self.offsets.len()))?;
        let file = &mut self.file;
        file.seek(offset + 8)?;
        let _step = file.i32()?;
        let time = file.f32()?;
        let box_values: Vec<f64> = file.f32_array(9)?.into_iter().map(f64::from).collect();
        let atoms = usize::try_from(file.i32()?).unwrap_or_default();
        if atoms != self.atom_count {
            return Err(TrajectoryError::format("XTC", "inconsistent atom count"));
        }
        let coordinates = if atoms <= 9 {
            file.f32_array(atoms * 3)?
        } else {
            let precision = file.f32()?;
            let mut header = [0_i32; 7];
            for value in &mut header {
                *value = file.i32()?;
            }
            let bytes = file.u32()? as usize;
            let data = file.bytes(bytes)?;
            decompress(
                &data,
                atoms,
                precision,
                [header[0], header[1], header[2]],
                [header[3], header[4], header[5]],
                header[6],
            )?
        };
        Ok(Frame {
            positions: positions_from_xyz(&coordinates, NM_TO_ANGSTROM),
            time: Some(f64::from(time)),
            cell: box_cell(&box_values),
        })
    }
}

const MAGIC_INTS: [i32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 101, 128, 161, 203, 256,
    322, 406, 512, 645, 812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321,
    13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031, 131072, 165140, 208063,
    262144, 330280, 416127, 524287, 660561, 832255, 1048576, 1321122, 1664510, 2097152, 2642245,
    3329021, 4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];
const FIRST_INDEX: usize = 9;

/// Reads bit fields from the compressed stream, most significant bit first.
struct BitReader<'a> {
    data: &'a [u8],
    count: usize,
    last_bits: u32,
    last_byte: u32,
}

impl BitReader<'_> {
    fn byte(&mut self) -> Result<u32, TrajectoryError> {
        let value = *self
            .data
            .get(self.count)
            .ok_or_else(|| TrajectoryError::format("XTC", "compressed coordinates end early"))?;
        self.count += 1;
        Ok(u32::from(value))
    }

    fn bits(&mut self, mut count: u32) -> Result<i32, TrajectoryError> {
        let mask: u64 = (1_u64 << count) - 1;
        let mut value: u64 = 0;
        while count >= 8 {
            self.last_byte = (self.last_byte << 8) | self.byte()?;
            value |= u64::from(self.last_byte >> self.last_bits) << (count - 8);
            count -= 8;
        }
        if count > 0 {
            if self.last_bits < count {
                self.last_bits += 8;
                self.last_byte = (self.last_byte << 8) | self.byte()?;
            }
            self.last_bits -= count;
            value |= u64::from((self.last_byte >> self.last_bits) & ((1_u32 << count) - 1));
        }
        Ok((value & mask) as i32)
    }

    /// Three integers packed into one mixed-radix number of `bit_count` bits.
    fn ints(&mut self, mut bit_count: u32, sizes: [u32; 3]) -> Result<[i32; 3], TrajectoryError> {
        let mut bytes = [0_u32; 32];
        let mut byte_count = 0;
        while bit_count > 8 {
            bytes[byte_count] = self.bits(8)? as u32;
            byte_count += 1;
            bit_count -= 8;
        }
        if bit_count > 0 {
            bytes[byte_count] = self.bits(bit_count)? as u32;
            byte_count += 1;
        }
        let mut numbers = [0_i32; 3];
        for index in (1..3).rev() {
            let mut number: u64 = 0;
            for byte in bytes[..byte_count].iter_mut().rev() {
                number = (number << 8) | u64::from(*byte);
                let quotient = number / u64::from(sizes[index].max(1));
                *byte = quotient as u32;
                number -= quotient * u64::from(sizes[index].max(1));
            }
            numbers[index] = number as i32;
        }
        numbers[0] = (bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)) as i32;
        Ok(numbers)
    }
}

fn size_of_int(size: u32) -> u32 {
    let (mut number, mut bits) = (1_u64, 0);
    while u64::from(size) >= number && bits < 32 {
        bits += 1;
        number <<= 1;
    }
    bits
}

fn size_of_ints(sizes: [u32; 3]) -> u32 {
    let mut bytes = [0_u32; 32];
    bytes[0] = 1;
    let mut byte_count = 1;
    for size in sizes {
        let mut carry: u64 = 0;
        let mut index = 0;
        while index < byte_count {
            carry += u64::from(bytes[index]) * u64::from(size);
            bytes[index] = (carry & 0xff) as u32;
            carry >>= 8;
            index += 1;
        }
        while carry != 0 && index < bytes.len() {
            bytes[index] = (carry & 0xff) as u32;
            carry >>= 8;
            index += 1;
        }
        byte_count = index;
    }
    let mut bits = 0;
    let mut number = 1_u32;
    byte_count -= 1;
    while bytes[byte_count] >= number {
        bits += 1;
        number *= 2;
    }
    bits + byte_count as u32 * 8
}

/// The GROMACS `xtc3dfcoord` decompression: coordinates are integers relative to a
/// bounding box, and runs of nearby atoms (typically water) are delta coded with an
/// adaptive bit width.
fn decompress(
    data: &[u8],
    atoms: usize,
    precision: f32,
    min: [i32; 3],
    max: [i32; 3],
    small_index: i32,
) -> Result<Vec<f32>, TrajectoryError> {
    let error = |message: &str| TrajectoryError::format("XTC", message);
    if precision <= 0.0 || !precision.is_finite() {
        return Err(error("invalid precision"));
    }
    let sizes: [u32; 3] = std::array::from_fn(|axis| {
        (i64::from(max[axis]) - i64::from(min[axis]) + 1).clamp(0, i64::from(u32::MAX)) as u32
    });
    let (bit_size, axis_bits) = if (sizes[0] | sizes[1] | sizes[2]) > 0xff_ffff {
        (0, sizes.map(size_of_int))
    } else {
        (size_of_ints(sizes), [0; 3])
    };
    let mut small_index = usize::try_from(small_index)
        .ok()
        .filter(|index| (FIRST_INDEX..MAGIC_INTS.len()).contains(index))
        .ok_or_else(|| error("invalid small-integer index"))?;
    let mut smaller = MAGIC_INTS[(small_index - 1).max(FIRST_INDEX)] / 2;
    let mut small_number = MAGIC_INTS[small_index] / 2;
    let mut small_sizes = [MAGIC_INTS[small_index] as u32; 3];

    let inverse = 1.0 / precision;
    let mut output = Vec::with_capacity(atoms * 3);
    let push = |coordinate: [i32; 3], output: &mut Vec<f32>| {
        output.extend(coordinate.map(|value| value as f32 * inverse));
    };
    let mut reader = BitReader {
        data,
        count: 0,
        last_bits: 0,
        last_byte: 0,
    };
    let mut run = 0_i32;
    let mut atom = 0;
    while atom < atoms {
        let mut this = if bit_size == 0 {
            [
                reader.bits(axis_bits[0])?,
                reader.bits(axis_bits[1])?,
                reader.bits(axis_bits[2])?,
            ]
        } else {
            reader.ints(bit_size, sizes)?
        };
        atom += 1;
        for axis in 0..3 {
            this[axis] = this[axis].wrapping_add(min[axis]);
        }
        let mut previous = this;
        let flag = reader.bits(1)?;
        let mut is_smaller = 0;
        if flag == 1 {
            run = reader.bits(5)?;
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }
        if run > 0 {
            if atom + (run as usize) / 3 > atoms {
                return Err(error("a run extends past the last atom"));
            }
            for k in (0..run).step_by(3) {
                let mut next = reader.ints(small_index as u32, small_sizes)?;
                atom += 1;
                for axis in 0..3 {
                    next[axis] = next[axis]
                        .wrapping_add(previous[axis])
                        .wrapping_sub(small_number);
                }
                if k == 0 {
                    // The first two atoms of a run are swapped (water: O before H).
                    std::mem::swap(&mut next, &mut previous);
                    push(previous, &mut output);
                } else {
                    previous = next;
                }
                push(next, &mut output);
            }
        } else {
            push(this, &mut output);
        }
        let new_index = small_index as i32 + is_smaller;
        small_index = usize::try_from(new_index)
            .ok()
            .filter(|index| *index < MAGIC_INTS.len())
            .ok_or_else(|| error("invalid small-integer index"))?;
        if is_smaller < 0 {
            small_number = smaller;
            smaller = if small_index > FIRST_INDEX {
                MAGIC_INTS[small_index - 1] / 2
            } else {
                0
            };
        } else if is_smaller > 0 {
            smaller = small_number;
            small_number = MAGIC_INTS[small_index] / 2;
        }
        small_sizes = [MAGIC_INTS[small_index] as u32; 3];
    }
    output.truncate(atoms * 3);
    Ok(output)
}

// ---------------------------------------------------------------------------------------
// TRR

struct TrrFrame {
    offset: u64,
    double: bool,
    box_present: bool,
    /// Bytes between the header and the coordinates (virial and pressure blocks).
    skip_before_positions: u64,
}

pub struct TrrReader {
    file: BinaryFile,
    atom_count: usize,
    frames: Vec<TrrFrame>,
}

impl TrrReader {
    pub fn open(path: &Path) -> Result<Self, TrajectoryError> {
        let error = |message: String| TrajectoryError::format("TRR", message);
        let mut file = BinaryFile::open(path, Endian::Big)?;
        let mut frames = Vec::new();
        let mut atom_count = None;
        let mut offset = 0_u64;
        while offset + 84 <= file.len {
            file.seek(offset)?;
            if file.i32()? != TRR_MAGIC {
                return Err(error(format!("bad frame magic at byte {offset}")));
            }
            // Version string: its length + 1, then an XDR string.
            let _ = file.i32()?;
            let length = u64::from(file.u32()?);
            file.skip(length.next_multiple_of(4))?;
            let mut sizes = [0_u64; 10];
            for size in &mut sizes {
                *size = u64::from(file.u32()?);
            }
            let [
                ir,
                e,
                box_size,
                virial,
                pressure,
                topology,
                symmetry,
                x,
                v,
                f,
            ] = sizes;
            let atoms =
                usize::try_from(file.i32()?).map_err(|_| error("negative atom count".into()))?;
            let _step = file.i32()?;
            let _energies = file.i32()?;
            let real = if box_size > 0 {
                box_size / 9
            } else if x > 0 {
                x / (atoms as u64 * 3).max(1)
            } else if v > 0 {
                v / (atoms as u64 * 3).max(1)
            } else if f > 0 {
                f / (atoms as u64 * 3).max(1)
            } else {
                4
            };
            if real != 4 && real != 8 {
                return Err(error(format!("unsupported real size {real}")));
            }
            let header_end = file.position()? + 2 * real;
            let end = header_end
                + ir
                + e
                + box_size
                + virial
                + pressure
                + topology
                + symmetry
                + x
                + v
                + f;
            if end > file.len {
                break;
            }
            if x > 0 {
                if *atom_count.get_or_insert(atoms) != atoms {
                    return Err(error("the atom count changes between frames".into()));
                }
                frames.push(TrrFrame {
                    offset,
                    double: real == 8,
                    box_present: box_size > 0,
                    skip_before_positions: ir + e + virial + pressure + topology + symmetry,
                });
            }
            offset = end;
        }
        Ok(Self {
            file,
            atom_count: atom_count.unwrap_or(0),
            frames,
        })
    }
}

impl FrameReader for TrrReader {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::Trr
    }

    fn atom_count(&self) -> usize {
        self.atom_count
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn read_frame(&mut self, index: usize) -> Result<Frame, TrajectoryError> {
        let frame = self
            .frames
            .get(index)
            .ok_or_else(|| out_of_range(index, self.frames.len()))?;
        let file = &mut self.file;
        file.seek(frame.offset + 8)?;
        let length = u64::from(file.u32()?);
        file.skip(length.next_multiple_of(4) + 13 * 4)?;
        let time = if frame.double {
            let time = file.f64()?;
            file.f64()?;
            time
        } else {
            let time = f64::from(file.f32()?);
            file.f32()?;
            time
        };
        let reals = |file: &mut BinaryFile, count: usize| -> std::io::Result<Vec<f64>> {
            if frame.double {
                file.f64_array(count)
            } else {
                Ok(file.f32_array(count)?.into_iter().map(f64::from).collect())
            }
        };
        // Blocks follow in GROMACS order: ir, e, box, vir, pres, top, sym, x, v, f.
        let cell = if frame.box_present {
            box_cell(&reals(file, 9)?)
        } else {
            None
        };
        file.skip(frame.skip_before_positions)?;
        let values = reals(file, self.atom_count * 3)?;
        let positions = values
            .as_chunks::<3>()
            .0
            .iter()
            .map(|[x, y, z]| Vec3::new(*x as f32, *y as f32, *z as f32) * NM_TO_ANGSTROM)
            .collect();
        Ok(Frame {
            positions,
            time: Some(time),
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
    fn integer_size_helpers_match_gromacs() {
        assert_eq!(size_of_int(0), 0);
        assert_eq!(size_of_int(1), 1);
        assert_eq!(size_of_int(255), 8);
        assert_eq!(size_of_int(256), 9);
        // 100 * 100 * 100 = 10^6 needs 20 bits.
        assert_eq!(size_of_ints([100, 100, 100]), 20);
        assert_eq!(size_of_ints([2, 2, 2]), 4);
    }

    #[test]
    fn small_files_decode_frames_and_times() {
        let (Some(xtc), Some(trr)) = (
            test_data::file("xtc_test_only_10_frame_10_atoms.xtc"),
            test_data::file("trr_test_only_10_frame_10_atoms.trr"),
        ) else {
            return;
        };
        let mut xtc = XtcReader::open(&xtc).unwrap();
        let mut trr = TrrReader::open(&trr).unwrap();
        assert_eq!((xtc.len(), xtc.atom_count()), (10, 10));
        assert_eq!((trr.len(), trr.atom_count()), (10, 10));
        for index in 0..10 {
            let (a, b) = (
                xtc.read_frame(index).unwrap(),
                trr.read_frame(index).unwrap(),
            );
            // The XTC stores the same trajectory at 0.001 nm precision.
            for (p, q) in a.positions.iter().zip(&b.positions) {
                assert!(p.distance(*q) < 0.02, "frame {index}: {p} vs {q}");
            }
            assert_eq!(a.time, b.time);
        }
    }

    #[test]
    fn protein_trajectories_agree_between_formats() {
        let (Some(xtc), Some(trr)) = (
            test_data::file("adk_oplsaa.xtc"),
            test_data::file("adk_oplsaa.trr"),
        ) else {
            return;
        };
        let mut xtc = XtcReader::open(&xtc).unwrap();
        let mut trr = TrrReader::open(&trr).unwrap();
        assert_eq!(xtc.atom_count(), 47681);
        assert_eq!(xtc.len(), 10);
        assert_eq!(trr.len(), 10);
        for index in [0, 4, 9] {
            let (a, b) = (
                xtc.read_frame(index).unwrap(),
                trr.read_frame(index).unwrap(),
            );
            let worst = a
                .positions
                .iter()
                .zip(&b.positions)
                .map(|(p, q)| p.distance(*q))
                .fold(0.0_f32, f32::max);
            assert!(worst < 0.01, "frame {index}: {worst} Å");
            let cell = a.cell.unwrap();
            assert!(cell[0] > 70.0 && cell[0] < 90.0, "{cell:?}");
        }
    }
}
