//! Molecular surfaces: van der Waals, solvent-accessible and solvent-excluded.
//!
//! Each surface is the zero level set of a scalar field sampled on a regular grid and
//! extracted with surface nets. The solvent-excluded surface follows the probe-sphere
//! construction used by molecular viewers: the probe rolls over every accessible position
//! of the solvent-accessible surface, and the region it cannot reach is the molecule.

mod field;
mod nets;

use std::{
    hash::{Hash, Hasher},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use glam::Vec3;

pub use field::MAX_GRID_POINTS;

/// Which molecular surface to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SurfaceKind {
    /// Union of van der Waals spheres.
    VanDerWaals,
    /// Surface traced by the center of a solvent probe (Lee–Richards).
    SolventAccessible,
    /// Surface the probe touches (Connolly); smooth reentrant patches fill crevices.
    #[default]
    SolventExcluded,
}

impl SurfaceKind {
    pub const ALL: [Self; 3] = [
        Self::SolventExcluded,
        Self::SolventAccessible,
        Self::VanDerWaals,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::VanDerWaals => "van der Waals",
            Self::SolventAccessible => "Solvent accessible",
            Self::SolventExcluded => "Solvent excluded",
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::SolventExcluded => 0,
            Self::SolventAccessible => 1,
            Self::VanDerWaals => 2,
        }
    }

    pub const fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::SolventExcluded),
            1 => Some(Self::SolventAccessible),
            2 => Some(Self::VanDerWaals),
            _ => None,
        }
    }
}

/// User-facing surface parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSettings {
    pub kind: SurfaceKind,
    /// Solvent probe radius in Å (1.4 Å approximates water).
    pub probe_radius: f32,
    /// Grid spacing in Å. Large structures coarsen it to stay within the memory budget.
    pub resolution: f32,
    /// Keep surfaces around internal cavities that the probe cannot reach from outside.
    pub cavities: bool,
}

impl Default for SurfaceSettings {
    fn default() -> Self {
        Self {
            kind: SurfaceKind::SolventExcluded,
            probe_radius: 1.4,
            resolution: 0.5,
            cavities: false,
        }
    }
}

impl SurfaceSettings {
    pub const PROBE_RANGE: std::ops::RangeInclusive<f32> = 0.5..=3.0;
    pub const RESOLUTION_RANGE: std::ops::RangeInclusive<f32> = 0.2..=2.0;

    /// Clamps values from files or the user interface into the supported range.
    pub fn sanitized(self) -> Self {
        let clamp = |value: f32, range: &std::ops::RangeInclusive<f32>, fallback: f32| {
            if value.is_finite() {
                value.clamp(*range.start(), *range.end())
            } else {
                fallback
            }
        };
        let defaults = Self::default();
        Self {
            kind: self.kind,
            probe_radius: clamp(self.probe_radius, &Self::PROBE_RANGE, defaults.probe_radius),
            resolution: clamp(
                self.resolution,
                &Self::RESOLUTION_RANGE,
                defaults.resolution,
            ),
            cavities: self.cavities,
        }
    }

    /// Probe radius that actually enters the construction for this kind.
    fn effective_probe(self) -> f32 {
        match self.kind {
            SurfaceKind::VanDerWaals => 0.0,
            _ => self.probe_radius,
        }
    }
}

/// One atom contributing to a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSphere {
    pub center: Vec3,
    /// Van der Waals radius in Å.
    pub radius: f32,
    /// Atom index used to color and pick the surface.
    pub atom: u32,
}

/// Indexed triangle mesh with per-vertex normals and owning atoms.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub atoms: Vec<u32>,
    pub indices: Vec<u32>,
    /// Grid spacing that was actually used, after any coarsening for large structures.
    pub spacing: f32,
}

impl SurfaceMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Total triangle area in Å².
    pub fn area(&self) -> f32 {
        self.indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|triangle| {
                let [a, b, c] = [0, 1, 2].map(|corner| self.positions[triangle[corner] as usize]);
                (b - a).cross(c - a).length() * 0.5
            })
            .sum()
    }
}

/// Builds a surface, or returns `None` if `cancel` was raised.
pub fn compute_surface(
    spheres: &[SurfaceSphere],
    settings: &SurfaceSettings,
    cancel: &AtomicBool,
) -> Option<SurfaceMesh> {
    let settings = settings.sanitized();
    let spheres: Vec<SurfaceSphere> = spheres
        .iter()
        .copied()
        .filter(|sphere| sphere.center.is_finite() && sphere.radius.is_finite())
        .map(|sphere| SurfaceSphere {
            radius: sphere.radius.max(0.1),
            ..sphere
        })
        .collect();
    if spheres.is_empty() {
        return Some(SurfaceMesh {
            spacing: settings.resolution,
            ..SurfaceMesh::default()
        });
    }
    let field = field::build(&spheres, &settings, cancel)?;
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let mut mesh = nets::extract(&field.grid, cancel)?;
    mesh.spacing = field.grid.spacing;
    let owner_padding = match settings.kind {
        SurfaceKind::SolventAccessible => settings.probe_radius,
        _ => 0.0,
    };
    mesh.atoms = field::assign_atoms(&mesh.positions, &spheres, owner_padding);
    Some(mesh)
}

/// Surface for the most recent input, so display edits that leave the surface atoms and
/// coordinates untouched do not rebuild it.
pub fn compute_surface_cached(
    spheres: &[SurfaceSphere],
    settings: &SurfaceSettings,
    cancel: &AtomicBool,
) -> Option<Arc<SurfaceMesh>> {
    static CACHE: Mutex<Option<(u64, Arc<SurfaceMesh>)>> = Mutex::new(None);
    let key = input_key(spheres, settings);
    if let Ok(cache) = CACHE.lock()
        && let Some((cached_key, mesh)) = cache.as_ref()
        && *cached_key == key
    {
        return Some(mesh.clone());
    }
    let started = std::time::Instant::now();
    let mesh = Arc::new(compute_surface(spheres, settings, cancel)?);
    log::info!(
        "{} surface: {} atoms, {} triangles, spacing {:.2} Å, {:.0} ms",
        settings.kind.label(),
        spheres.len(),
        mesh.triangle_count(),
        mesh.spacing,
        started.elapsed().as_secs_f64() * 1000.0
    );
    if let Ok(mut cache) = CACHE.lock() {
        *cache = Some((key, mesh.clone()));
    }
    Some(mesh)
}

fn input_key(spheres: &[SurfaceSphere], settings: &SurfaceSettings) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let settings = settings.sanitized();
    settings.kind.hash(&mut hasher);
    settings.probe_radius.to_bits().hash(&mut hasher);
    settings.resolution.to_bits().hash(&mut hasher);
    settings.cavities.hash(&mut hasher);
    spheres.len().hash(&mut hasher);
    for sphere in spheres {
        sphere.atom.hash(&mut hasher);
        sphere.radius.to_bits().hash(&mut hasher);
        sphere.center.to_array().map(f32::to_bits).hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    fn sphere(center: Vec3, radius: f32, atom: u32) -> SurfaceSphere {
        SurfaceSphere {
            center,
            radius,
            atom,
        }
    }

    fn build(spheres: &[SurfaceSphere], settings: SurfaceSettings) -> SurfaceMesh {
        compute_surface(spheres, &settings, &AtomicBool::new(false)).unwrap()
    }

    fn settings(kind: SurfaceKind, resolution: f32) -> SurfaceSettings {
        SurfaceSettings {
            kind,
            resolution,
            ..SurfaceSettings::default()
        }
    }

    #[test]
    fn single_atom_surfaces_match_analytic_spheres() {
        let atom = [sphere(Vec3::new(0.3, -0.2, 0.1), 1.7, 7)];
        for (kind, radius) in [
            (SurfaceKind::VanDerWaals, 1.7),
            (SurfaceKind::SolventAccessible, 3.1),
            (SurfaceKind::SolventExcluded, 1.7),
        ] {
            let mesh = build(&atom, settings(kind, 0.25));
            assert!(!mesh.is_empty(), "{kind:?}");
            for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
                let offset = *position - atom[0].center;
                assert!(
                    (offset.length() - radius).abs() < 0.05,
                    "{kind:?}: vertex at {} Å",
                    offset.length()
                );
                assert!(normal.dot(offset.normalize()) > 0.95, "{kind:?} normal");
            }
            let expected = 4.0 * PI * radius * radius;
            let area = mesh.area();
            assert!(
                (area - expected).abs() / expected < 0.03,
                "{kind:?}: area {area} vs {expected}"
            );
            assert!(mesh.atoms.iter().all(|&owner| owner == 7));
        }
    }

    #[test]
    fn solvent_excluded_surface_fills_the_crevice_between_two_atoms() {
        // Two touching carbon spheres. The probe (1.4 Å) cannot enter the groove at the
        // contact circle, so the reentrant surface lies outside both spheres there.
        let atoms = [
            sphere(Vec3::new(-1.6, 0.0, 0.0), 1.7, 0),
            sphere(Vec3::new(1.6, 0.0, 0.0), 1.7, 1),
        ];
        let mesh = build(&atoms, settings(SurfaceKind::SolventExcluded, 0.2));
        // Analytic radius of the reentrant torus at the midplane:
        // the probe touching both spheres sits at x = 0, distance sqrt((1.7+1.4)^2 - 1.6^2).
        let probe_center = ((1.7_f32 + 1.4).powi(2) - 1.6_f32.powi(2)).sqrt();
        let expected = probe_center - 1.4;
        let midplane: Vec<f32> = mesh
            .positions
            .iter()
            .filter(|position| position.x.abs() < 0.1)
            .map(|position| position.truncate().y.hypot(position.z))
            .collect();
        assert!(!midplane.is_empty());
        for radius in midplane {
            assert!(
                (radius - expected).abs() < 0.08,
                "midplane radius {radius} vs {expected}"
            );
        }
        // Vertices on the outer caps belong to the nearer atom.
        for (position, atom) in mesh.positions.iter().zip(&mesh.atoms) {
            if position.x < -2.0 {
                assert_eq!(*atom, 0);
            } else if position.x > 2.0 {
                assert_eq!(*atom, 1);
            }
        }
    }

    #[test]
    fn internal_cavities_are_optional() {
        // A closed shell of atoms around an empty interior large enough for the probe.
        let mut atoms = Vec::new();
        let count = 400;
        for index in 0..count {
            let t = (index as f32 + 0.5) / count as f32;
            let polar = (1.0 - 2.0 * t).acos();
            let azimuth = PI * (1.0 + 5.0_f32.sqrt()) * index as f32;
            let direction = Vec3::new(
                polar.sin() * azimuth.cos(),
                polar.sin() * azimuth.sin(),
                polar.cos(),
            );
            atoms.push(sphere(direction * 8.0, 1.7, index));
        }
        let closed = build(&atoms, settings(SurfaceKind::SolventExcluded, 0.5));
        let open = build(
            &atoms,
            SurfaceSettings {
                cavities: true,
                ..settings(SurfaceKind::SolventExcluded, 0.5)
            },
        );
        let inner = |mesh: &SurfaceMesh| {
            mesh.positions
                .iter()
                .filter(|position| position.length() < 7.5)
                .count()
        };
        assert_eq!(inner(&closed), 0);
        assert!(inner(&open) > 100);
    }

    #[test]
    fn large_inputs_are_coarsened_and_cancellation_is_honored() {
        let atoms: Vec<_> = (0..64)
            .map(|index| {
                sphere(
                    Vec3::new((index % 4) as f32, (index / 16) as f32, 0.0) * 150.0,
                    1.7,
                    index,
                )
            })
            .collect();
        let mesh = build(&atoms, settings(SurfaceKind::VanDerWaals, 0.2));
        assert!(mesh.spacing > 0.2);
        assert!(!mesh.is_empty());
        assert!(
            compute_surface(&atoms, &SurfaceSettings::default(), &AtomicBool::new(true)).is_none()
        );
        assert!(build(&[], SurfaceSettings::default()).is_empty());
    }
}
