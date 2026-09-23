//! Coordinate trajectories: a sequence of frames for a fixed topology.
//!
//! A [`Molecule`](super::Molecule) is the topology together with the coordinates of the
//! active frame. Trajectories supply further frames; playback replaces only positions, so
//! bonds, selections, colors and representations stay attached to the same atoms.
//!
//! File-backed readers index frame offsets when they are opened and read one frame at a
//! time, so trajectories larger than memory play back without loading them whole.

mod dcd;
mod mdcrd;
mod netcdf;
mod xdr;

use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom},
    path::Path,
};

use glam::Vec3;
use thiserror::Error;

pub use dcd::DcdReader;
pub use mdcrd::MdcrdReader;
pub use netcdf::NetCdfReader;
pub use xdr::{TrrReader, XtcReader};

/// One frame of coordinates in Å.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub positions: Vec<Vec3>,
    /// Simulation time in picoseconds, when the format records it.
    pub time: Option<f64>,
    /// Periodic cell as `a, b, c` in Å and `alpha, beta, gamma` in degrees.
    pub cell: Option<[f64; 6]>,
}

impl Frame {
    pub fn new(positions: Vec<Vec3>) -> Self {
        Self {
            positions,
            time: None,
            cell: None,
        }
    }
}

#[derive(Debug, Error)]
pub enum TrajectoryError {
    #[error("could not read the trajectory: {0}")]
    Io(#[from] io::Error),
    #[error("invalid {format} trajectory: {message}")]
    Format {
        format: &'static str,
        message: String,
    },
    #[error("the trajectory has {found} atoms but the structure has {expected}")]
    AtomCount { expected: usize, found: usize },
    #[error("unsupported trajectory: {0}")]
    Unsupported(String),
    #[error("frame {index} is out of range; the trajectory has {count} frames")]
    FrameOutOfRange { index: usize, count: usize },
}

impl TrajectoryError {
    pub(crate) fn format(format: &'static str, message: impl Into<String>) -> Self {
        Self::Format {
            format,
            message: message.into(),
        }
    }
}

/// Trajectory file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryFormat {
    /// CHARMM, NAMD and X-PLOR binary trajectories.
    Dcd,
    /// GROMACS compressed trajectories.
    Xtc,
    /// GROMACS full-precision trajectories.
    Trr,
    /// AMBER NetCDF (classic and 64-bit offset NetCDF-3).
    NetCdf,
    /// AMBER ASCII trajectories.
    Mdcrd,
    /// Models or frames of a structure file (PDB, mmCIF, GRO, XYZ, ...), held in memory.
    Models,
}

impl TrajectoryFormat {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dcd => "DCD",
            Self::Xtc => "XTC",
            Self::Trr => "TRR",
            Self::NetCdf => "AMBER NetCDF",
            Self::Mdcrd => "AMBER mdcrd",
            Self::Models => "structure models",
        }
    }

    /// File extensions offered by the trajectory dialog.
    pub const EXTENSIONS: [&'static str; 8] =
        ["dcd", "xtc", "trr", "nc", "ncdf", "netcdf", "mdcrd", "trj"];
}

/// Random access to the frames of a trajectory.
pub trait FrameReader: Send {
    fn format(&self) -> TrajectoryFormat;
    fn atom_count(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn read_frame(&mut self, index: usize) -> Result<Frame, TrajectoryError>;
}

/// Frames that are already in memory, such as the models of an NMR ensemble.
pub struct InMemoryFrames {
    frames: Vec<Frame>,
    atom_count: usize,
}

impl InMemoryFrames {
    /// All frames must have the same number of atoms.
    pub fn new(frames: Vec<Frame>) -> Result<Self, TrajectoryError> {
        let atom_count = frames.first().map_or(0, |frame| frame.positions.len());
        if let Some(frame) = frames
            .iter()
            .find(|frame| frame.positions.len() != atom_count)
        {
            return Err(TrajectoryError::AtomCount {
                expected: atom_count,
                found: frame.positions.len(),
            });
        }
        Ok(Self { frames, atom_count })
    }

    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }
}

impl FrameReader for InMemoryFrames {
    fn format(&self) -> TrajectoryFormat {
        TrajectoryFormat::Models
    }

    fn atom_count(&self) -> usize {
        self.atom_count
    }

    fn len(&self) -> usize {
        self.frames.len()
    }

    fn read_frame(&mut self, index: usize) -> Result<Frame, TrajectoryError> {
        self.frames
            .get(index)
            .cloned()
            .ok_or(TrajectoryError::FrameOutOfRange {
                index,
                count: self.frames.len(),
            })
    }
}

/// What happens when playback reaches the last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopMode {
    /// Stop at the end.
    Once,
    /// Jump back to the first frame.
    #[default]
    Loop,
    /// Reverse direction at either end.
    Bounce,
}

impl LoopMode {
    pub const ALL: [Self; 3] = [Self::Once, Self::Loop, Self::Bounce];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Once => "Once",
            Self::Loop => "Loop",
            Self::Bounce => "Bounce",
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::Once => 0,
            Self::Loop => 1,
            Self::Bounce => 2,
        }
    }

    pub const fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Once,
            2 => Self::Bounce,
            _ => Self::Loop,
        }
    }
}

/// Playback speed and order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSettings {
    /// Frames shown per second.
    pub fps: f32,
    pub loop_mode: LoopMode,
    /// Frames advanced per step.
    pub stride: usize,
}

impl Default for PlaybackSettings {
    fn default() -> Self {
        Self {
            fps: 24.0,
            loop_mode: LoopMode::Loop,
            stride: 1,
        }
    }
}

impl PlaybackSettings {
    pub const FPS_RANGE: std::ops::RangeInclusive<f32> = 1.0..=120.0;

    pub fn sanitized(self) -> Self {
        Self {
            fps: if self.fps.is_finite() {
                self.fps
                    .clamp(*Self::FPS_RANGE.start(), *Self::FPS_RANGE.end())
            } else {
                Self::default().fps
            },
            loop_mode: self.loop_mode,
            stride: self.stride.clamp(1, 10_000),
        }
    }

    /// The frame after `current` when moving in `direction` (+1 or -1), with the new
    /// direction, or `None` when playback should stop.
    pub fn next_frame(
        &self,
        current: usize,
        frame_count: usize,
        direction: i8,
    ) -> Option<(usize, i8)> {
        if frame_count < 2 {
            return None;
        }
        let last = frame_count - 1;
        let stride = self.stride.max(1);
        let step = |from: usize, direction: i8| -> Option<usize> {
            if direction >= 0 {
                from.checked_add(stride).filter(|next| *next <= last)
            } else {
                from.checked_sub(stride)
            }
        };
        if let Some(next) = step(current, direction) {
            return Some((next, direction));
        }
        match self.loop_mode {
            LoopMode::Once => None,
            LoopMode::Loop => Some(if direction >= 0 { (0, 1) } else { (last, -1) }),
            LoopMode::Bounce => {
                let reversed = if direction >= 0 { -1 } else { 1 };
                step(current, reversed)
                    .map(|next| (next, reversed))
                    .or(Some((if reversed > 0 { 0 } else { last }, reversed)))
            }
        }
    }
}

/// Detects the format from the file contents, falling back to the extension.
pub fn detect_format(path: &Path) -> Result<TrajectoryFormat, TrajectoryError> {
    let mut magic = [0_u8; 8];
    let read = File::open(path)?.read(&mut magic)?;
    let magic = &magic[..read];
    if magic.starts_with(b"CDF\x01") || magic.starts_with(b"CDF\x02") {
        return Ok(TrajectoryFormat::NetCdf);
    }
    if magic.starts_with(b"\x89HDF") {
        return Err(TrajectoryError::Unsupported(
            "NetCDF-4/HDF5 files are not supported; convert with `ncks -3` or write NetCDF-3 \
             (the AMBER default)"
                .into(),
        ));
    }
    if magic.len() >= 8 && (&magic[4..8] == b"CORD" || &magic[4..8] == b"VELD") {
        return Ok(TrajectoryFormat::Dcd);
    }
    if magic.len() >= 4 {
        match i32::from_be_bytes([magic[0], magic[1], magic[2], magic[3]]) {
            1995 => return Ok(TrajectoryFormat::Xtc),
            1993 => return Ok(TrajectoryFormat::Trr),
            _ => {}
        }
    }
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "mdcrd" | "trj" | "crd" | "x" => Ok(TrajectoryFormat::Mdcrd),
        _ if super::is_structure_filename(&path.to_string_lossy()) => Ok(TrajectoryFormat::Models),
        _ => Err(TrajectoryError::Unsupported(format!(
            "{} is not a DCD, XTC, TRR, NetCDF, mdcrd or multi-model structure file",
            path.display()
        ))),
    }
}

/// Opens a trajectory for a structure with `atom_count` atoms.
pub fn open_trajectory(
    path: &Path,
    atom_count: usize,
) -> Result<Box<dyn FrameReader>, TrajectoryError> {
    let reader: Box<dyn FrameReader> = match detect_format(path)? {
        TrajectoryFormat::Dcd => Box::new(DcdReader::open(path)?),
        TrajectoryFormat::Xtc => Box::new(XtcReader::open(path)?),
        TrajectoryFormat::Trr => Box::new(TrrReader::open(path)?),
        TrajectoryFormat::NetCdf => Box::new(NetCdfReader::open(path)?),
        TrajectoryFormat::Mdcrd => Box::new(MdcrdReader::open(path, atom_count)?),
        TrajectoryFormat::Models => Box::new(structure_models(path)?),
    };
    if reader.atom_count() != atom_count {
        return Err(TrajectoryError::AtomCount {
            expected: atom_count,
            found: reader.atom_count(),
        });
    }
    if reader.is_empty() {
        return Err(TrajectoryError::Unsupported(format!(
            "{} contains no frames",
            path.display()
        )));
    }
    Ok(reader)
}

/// Every model of a structure file as a frame, including the first.
fn structure_models(path: &Path) -> Result<InMemoryFrames, TrajectoryError> {
    let bytes = std::fs::read(path)?;
    let parsed = super::parse_structure(&bytes, &path.to_string_lossy())
        .map_err(|error| TrajectoryError::Unsupported(error.to_string()))?;
    let mut frames = vec![Frame::new(parsed.molecule.positions())];
    frames.extend(parsed.frames.into_iter().map(Frame::new));
    InMemoryFrames::new(frames)
}

/// Byte order of a binary trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Endian {
    Little,
    Big,
}

/// Seekable binary input with explicit byte order.
pub(crate) struct BinaryFile {
    reader: BufReader<File>,
    pub(crate) endian: Endian,
    pub(crate) len: u64,
}

impl BinaryFile {
    pub(crate) fn open(path: &Path, endian: Endian) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            reader: BufReader::with_capacity(1 << 16, file),
            endian,
            len,
        })
    }

    pub(crate) fn seek(&mut self, offset: u64) -> io::Result<()> {
        self.reader.seek(SeekFrom::Start(offset)).map(|_| ())
    }

    pub(crate) fn position(&mut self) -> io::Result<u64> {
        self.reader.stream_position()
    }

    pub(crate) fn skip(&mut self, bytes: u64) -> io::Result<()> {
        self.reader
            .seek_relative(i64::try_from(bytes).map_err(io::Error::other)?)
    }

    pub(crate) fn bytes(&mut self, count: usize) -> io::Result<Vec<u8>> {
        let mut buffer = vec![0; count];
        self.reader.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn array<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut buffer = [0; N];
        self.reader.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    pub(crate) fn i32(&mut self) -> io::Result<i32> {
        let bytes = self.array::<4>()?;
        Ok(match self.endian {
            Endian::Little => i32::from_le_bytes(bytes),
            Endian::Big => i32::from_be_bytes(bytes),
        })
    }

    pub(crate) fn u32(&mut self) -> io::Result<u32> {
        self.i32().map(|value| value as u32)
    }

    pub(crate) fn i64(&mut self) -> io::Result<i64> {
        let bytes = self.array::<8>()?;
        Ok(match self.endian {
            Endian::Little => i64::from_le_bytes(bytes),
            Endian::Big => i64::from_be_bytes(bytes),
        })
    }

    pub(crate) fn f32(&mut self) -> io::Result<f32> {
        self.u32().map(f32::from_bits)
    }

    pub(crate) fn f64(&mut self) -> io::Result<f64> {
        self.i64().map(|value| f64::from_bits(value as u64))
    }

    /// Decodes `count` floats from one bulk read.
    pub(crate) fn f32_array(&mut self, count: usize) -> io::Result<Vec<f32>> {
        let bytes = self.bytes(count * 4)?;
        let endian = self.endian;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| match endian {
                Endian::Little => f32::from_le_bytes(*chunk),
                Endian::Big => f32::from_be_bytes(*chunk),
            })
            .collect())
    }

    pub(crate) fn f64_array(&mut self, count: usize) -> io::Result<Vec<f64>> {
        let bytes = self.bytes(count * 8)?;
        let endian = self.endian;
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| match endian {
                Endian::Little => f64::from_le_bytes(*chunk),
                Endian::Big => f64::from_be_bytes(*chunk),
            })
            .collect())
    }
}

/// Cell lengths and angles from three box vectors (rows), in the vectors' length unit.
pub(crate) fn cell_from_vectors(vectors: [[f64; 3]; 3]) -> Option<[f64; 6]> {
    let length = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let [a, b, c] = vectors.map(length);
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return None;
    }
    let angle = |u: [f64; 3], v: [f64; 3], lu: f64, lv: f64| {
        ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    Some([
        a,
        b,
        c,
        angle(vectors[1], vectors[2], b, c),
        angle(vectors[0], vectors[2], a, c),
        angle(vectors[0], vectors[1], a, b),
    ])
}

pub(crate) fn positions_from_xyz(values: &[f32], scale: f32) -> Vec<Vec3> {
    values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|[x, y, z]| Vec3::new(*x, *y, *z) * scale)
        .collect()
}

#[cfg(test)]
pub(crate) mod test_data {
    use std::path::PathBuf;

    /// MDAnalysis test files, when present (`MDANALYSIS_TEST_DATA` or the scratch copy
    /// used in development). Tests that need them are skipped otherwise.
    pub fn file(name: &str) -> Option<PathBuf> {
        let roots = std::env::var_os("MDANALYSIS_TEST_DATA")
            .map(PathBuf::from)
            .into_iter()
            .chain([PathBuf::from(
                "/tmp/claude-0/mdat/mdanalysistests-2.10.0/MDAnalysisTests/data",
            )]);
        roots
            .map(|root| root.join(name))
            .find(|path| path.is_file())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_frames_require_equal_atom_counts() {
        let frames = vec![
            Frame::new(vec![Vec3::ZERO; 2]),
            Frame::new(vec![Vec3::ONE; 2]),
        ];
        let mut reader = InMemoryFrames::new(frames).unwrap();
        assert_eq!(reader.len(), 2);
        assert_eq!(reader.read_frame(1).unwrap().positions[0], Vec3::ONE);
        assert!(matches!(
            reader.read_frame(2),
            Err(TrajectoryError::FrameOutOfRange { index: 2, count: 2 })
        ));
        assert!(
            InMemoryFrames::new(vec![
                Frame::new(vec![Vec3::ZERO; 2]),
                Frame::new(vec![Vec3::ZERO; 3]),
            ])
            .is_err()
        );
    }

    #[test]
    fn cell_parameters_follow_from_box_vectors() {
        let cell =
            cell_from_vectors([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]]).unwrap();
        for (value, expected) in cell.iter().zip([10.0, 20.0, 30.0, 90.0, 90.0, 90.0]) {
            assert!((value - expected).abs() < 1e-9);
        }
        assert!(cell_from_vectors([[0.0; 3]; 3]).is_none());
    }

    #[test]
    fn opens_every_format_and_checks_atom_counts() {
        for (name, atoms, format) in [
            ("adk_dims.dcd", 3341, TrajectoryFormat::Dcd),
            ("adk_oplsaa.xtc", 47681, TrajectoryFormat::Xtc),
            ("adk_oplsaa.trr", 47681, TrajectoryFormat::Trr),
            ("Amber/ace_tip3p.nc", 1398, TrajectoryFormat::NetCdf),
            ("Amber/ache.mdcrd", 252, TrajectoryFormat::Mdcrd),
        ] {
            let Some(path) = test_data::file(name) else {
                continue;
            };
            assert_eq!(detect_format(&path).unwrap(), format, "{name}");
            if format != TrajectoryFormat::Mdcrd {
                assert!(matches!(
                    open_trajectory(&path, atoms + 1),
                    Err(TrajectoryError::AtomCount { expected, found })
                        if expected == atoms + 1 && found == atoms
                ));
            }
            let mut reader = open_trajectory(&path, atoms).unwrap();
            assert_eq!(reader.format(), format);
            let last = reader.len() - 1;
            assert_eq!(reader.read_frame(last).unwrap().positions.len(), atoms);
        }
        if let Some(path) = test_data::file("nmr_neopetrosiamide.pdb") {
            let bytes = std::fs::read(&path).unwrap();
            let parsed = crate::molecule::parse_structure(&bytes, "nmr.pdb").unwrap();
            let mut reader = open_trajectory(&path, parsed.molecule.atoms.len()).unwrap();
            assert_eq!(reader.format(), TrajectoryFormat::Models);
            assert_eq!(reader.len(), parsed.frames.len() + 1);
            assert_eq!(
                reader.read_frame(0).unwrap().positions,
                parsed.molecule.positions()
            );
        }
    }

    #[test]
    fn playback_advances_loops_and_bounces() {
        let mut settings = PlaybackSettings::default();
        assert_eq!(settings.next_frame(0, 5, 1), Some((1, 1)));
        assert_eq!(settings.next_frame(4, 5, 1), Some((0, 1)));
        settings.loop_mode = LoopMode::Once;
        assert_eq!(settings.next_frame(4, 5, 1), None);
        settings.loop_mode = LoopMode::Bounce;
        assert_eq!(settings.next_frame(4, 5, 1), Some((3, -1)));
        assert_eq!(settings.next_frame(0, 5, -1), Some((1, 1)));
        settings.stride = 3;
        assert_eq!(settings.next_frame(3, 5, 1), Some((0, -1)));
        settings.loop_mode = LoopMode::Loop;
        assert_eq!(settings.next_frame(3, 5, 1), Some((0, 1)));
        assert_eq!(settings.next_frame(0, 1, 1), None);
        assert_eq!(
            PlaybackSettings {
                fps: f32::NAN,
                stride: 0,
                ..settings
            }
            .sanitized(),
            PlaybackSettings {
                fps: 24.0,
                stride: 1,
                ..settings
            }
        );
    }
}
