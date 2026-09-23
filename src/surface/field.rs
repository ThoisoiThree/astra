//! Scalar fields whose zero level set is the requested surface (positive inside).

use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use glam::Vec3;

use super::{SurfaceKind, SurfaceSettings, SurfaceSphere};

/// Upper bound on grid points (4 bytes each plus a byte of flood-fill state). Larger
/// structures use a coarser spacing instead of exhausting memory.
pub const MAX_GRID_POINTS: usize = 24_000_000;

/// Spacing of accessible probe positions on each expanded atom sphere, in Å, for a water
/// probe. The reentrant surface deviates from the exact probe envelope by about
/// `spacing² / (8 · probe)`, which is 0.04 Å here.
const PROBE_POINT_SPACING: f32 = 0.7;

/// Regular grid of field values, x fastest.
pub(super) struct Grid {
    pub origin: Vec3,
    pub spacing: f32,
    pub dims: [usize; 3],
    pub values: Vec<f32>,
}

impl Grid {
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.dims[0] * (y + self.dims[1] * z)
    }

    pub fn value(&self, x: usize, y: usize, z: usize) -> f32 {
        self.values[self.index(x, y, z)]
    }

    pub fn point(&self, x: f32, y: f32, z: f32) -> Vec3 {
        self.origin + Vec3::new(x, y, z) * self.spacing
    }
}

pub(super) struct Field {
    pub grid: Grid,
}

/// A spherical stamp that lowers the field to `|p - center| - radius` within `reach`.
#[derive(Clone, Copy)]
struct Stamp {
    center: Vec3,
    radius: f32,
    reach: f32,
}

pub(super) fn build(
    spheres: &[SurfaceSphere],
    settings: &SurfaceSettings,
    cancel: &AtomicBool,
) -> Option<Field> {
    let probe = settings.effective_probe();
    let max_radius = spheres
        .iter()
        .map(|sphere| sphere.radius)
        .fold(0.0_f32, f32::max);
    let (low, high) = spheres.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(low, high), sphere| (low.min(sphere.center), high.max(sphere.center)),
    );

    let mut spacing = settings.resolution;
    let (dims, padding) = loop {
        let padding = max_radius + probe + 3.0 * spacing;
        let extent = high - low + Vec3::splat(2.0 * padding);
        let dims = [extent.x, extent.y, extent.z].map(|axis| (axis / spacing).ceil() as usize + 1);
        let points = dims.iter().product::<usize>();
        if points <= MAX_GRID_POINTS {
            break (dims, padding);
        }
        spacing *= (points as f32 / MAX_GRID_POINTS as f32).cbrt() * 1.02;
    };
    if spacing > settings.resolution * 1.001 {
        log::info!(
            "surface grid coarsened from {:.2} Å to {spacing:.2} Å to stay within {MAX_GRID_POINTS} points",
            settings.resolution
        );
    }
    let mut grid = Grid {
        origin: low - Vec3::splat(padding),
        spacing,
        dims,
        values: Vec::new(),
    };

    // Signed distance to the union of expanded spheres (negative inside). Only values near
    // zero matter, so each atom writes a shell a little wider than its sphere.
    let band = 2.0 * spacing;
    grid.values = vec![band; dims.iter().product()];
    let atom_stamps: Vec<Stamp> = spheres
        .iter()
        .map(|sphere| Stamp {
            center: sphere.center,
            radius: sphere.radius + probe,
            reach: sphere.radius + probe + band,
        })
        .collect();
    stamp_min(&mut grid, atom_stamps, cancel)?;

    let reachable = exterior_reachable(&grid, cancel)?;
    let outside =
        |index: usize, value: f32| value >= 0.0 && (settings.cavities || reachable[index]);

    if settings.kind != SurfaceKind::SolventExcluded || probe <= 0.0 {
        for (index, value) in grid.values.iter_mut().enumerate() {
            *value = if *value < 0.0 || outside(index, *value) {
                -*value
            } else {
                // A filled cavity.
                0.5 * spacing
            };
        }
        return Some(Field { grid });
    }

    let probes = accessible_probes(spheres, probe, &grid, &reachable, settings.cavities, cancel)?;
    for (index, value) in grid.values.iter_mut().enumerate() {
        *value = if outside(index, *value) {
            -probe
        } else {
            probe
        };
    }
    // Inside is everything farther than one probe radius from every accessible probe center.
    let probe_stamps: Vec<Stamp> = probes
        .into_iter()
        .map(|center| Stamp {
            center,
            radius: probe,
            reach: probe + band,
        })
        .collect();
    stamp_min(&mut grid, probe_stamps, cancel)?;
    Some(Field { grid })
}

/// Lowers grid values to each stamp's signed distance, in parallel over z slabs.
fn stamp_min(grid: &mut Grid, mut stamps: Vec<Stamp>, cancel: &AtomicBool) -> Option<()> {
    stamps.sort_by(|a, b| a.center.z.total_cmp(&b.center.z));
    let max_reach = stamps
        .iter()
        .map(|stamp| stamp.reach)
        .fold(0.0_f32, f32::max);
    let [nx, ny, nz] = grid.dims;
    let plane = nx * ny;
    let threads = worker_count().min(nz.max(1));
    let slab = nz.div_ceil(threads);
    let (origin, spacing) = (grid.origin, grid.spacing);
    let stamps = &stamps;
    thread::scope(|scope| {
        for (chunk_index, chunk) in grid.values.chunks_mut(slab * plane).enumerate() {
            scope.spawn(move || {
                let z_begin = chunk_index * slab;
                let z_end = z_begin + chunk.len() / plane;
                let world_low = origin.z + z_begin as f32 * spacing - max_reach;
                let world_high = origin.z + (z_end - 1) as f32 * spacing + max_reach;
                let first = stamps.partition_point(|stamp| stamp.center.z < world_low);
                for (count, stamp) in stamps[first..].iter().enumerate() {
                    if stamp.center.z > world_high {
                        break;
                    }
                    if count % 2048 == 0 && cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let low = ((stamp.center - Vec3::splat(stamp.reach) - origin) / spacing).ceil();
                    let high =
                        ((stamp.center + Vec3::splat(stamp.reach) - origin) / spacing).floor();
                    let range = |low: f32, high: f32, begin: usize, end: usize| {
                        let low = (low.max(0.0) as usize).max(begin);
                        let high = if high < 0.0 {
                            0
                        } else {
                            (high as usize + 1).min(end)
                        };
                        low..high.max(low)
                    };
                    let reach_squared = stamp.reach * stamp.reach;
                    for z in range(low.z, high.z, z_begin, z_end) {
                        let dz = origin.z + z as f32 * spacing - stamp.center.z;
                        let dz2 = dz * dz;
                        let slice = &mut chunk[(z - z_begin) * plane..(z - z_begin + 1) * plane];
                        for y in range(low.y, high.y, 0, ny) {
                            let dy = origin.y + y as f32 * spacing - stamp.center.y;
                            let dyz2 = dy * dy + dz2;
                            if dyz2 > reach_squared {
                                continue;
                            }
                            let row = &mut slice[y * nx..(y + 1) * nx];
                            let half_width = (reach_squared - dyz2).sqrt();
                            let row_low =
                                ((stamp.center.x - half_width - origin.x) / spacing).ceil();
                            let row_high =
                                ((stamp.center.x + half_width - origin.x) / spacing).floor();
                            for x in range(row_low, row_high, 0, nx) {
                                let dx = origin.x + x as f32 * spacing - stamp.center.x;
                                let squared = dx * dx + dyz2;
                                // Compare squared distances first; most points are already
                                // lower than this stamp and need no square root.
                                let limit = row[x] + stamp.radius;
                                if limit > 0.0 && squared < limit * limit {
                                    row[x] = squared.sqrt() - stamp.radius;
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    (!cancel.load(Ordering::Relaxed)).then_some(())
}

/// Grid points outside the expanded spheres that connect to the grid boundary, found with
/// a scanline flood fill so each run of open points along x is handled once.
fn exterior_reachable(grid: &Grid, cancel: &AtomicBool) -> Option<Vec<bool>> {
    let [nx, ny, nz] = grid.dims;
    let open = |index: usize| grid.values[index] >= 0.0;
    let mut reachable = vec![false; grid.values.len()];
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for z in 0..nz {
        for y in 0..ny {
            if y == 0 || z == 0 || y == ny - 1 || z == nz - 1 {
                stack.push((0, y, z));
            } else {
                stack.push((0, y, z));
                stack.push((nx - 1, y, z));
            }
        }
    }
    let mut spans = 0_usize;
    while let Some((x, y, z)) = stack.pop() {
        let row = grid.index(0, y, z);
        if reachable[row + x] || !open(row + x) {
            continue;
        }
        spans += 1;
        if spans.is_multiple_of(1 << 16) && cancel.load(Ordering::Relaxed) {
            return None;
        }
        let mut low = x;
        while low > 0 && !reachable[row + low - 1] && open(row + low - 1) {
            low -= 1;
        }
        let mut high = x;
        while high + 1 < nx && !reachable[row + high + 1] && open(row + high + 1) {
            high += 1;
        }
        reachable[row + low..=row + high].fill(true);
        let neighbors = [
            (y > 0).then(|| (y - 1, z)),
            (y + 1 < ny).then_some((y + 1, z)),
            (z > 0).then(|| (y, z - 1)),
            (z + 1 < nz).then_some((y, z + 1)),
        ];
        for (next_y, next_z) in neighbors.into_iter().flatten() {
            let next_row = grid.index(0, next_y, next_z);
            let mut in_run = false;
            for next_x in low..=high {
                let index = next_row + next_x;
                let candidate = !reachable[index] && open(index);
                if candidate && !in_run {
                    stack.push((next_x, next_y, next_z));
                }
                in_run = candidate;
            }
        }
    }
    Some(reachable)
}

/// Probe centers on the solvent-accessible surface: points of each expanded sphere that
/// no other expanded sphere covers.
fn accessible_probes(
    spheres: &[SurfaceSphere],
    probe: f32,
    grid: &Grid,
    reachable: &[bool],
    cavities: bool,
    cancel: &AtomicBool,
) -> Option<Vec<Vec3>> {
    let expanded = |sphere: &SurfaceSphere| sphere.radius + probe;
    let max_expanded = spheres.iter().map(expanded).fold(0.0_f32, f32::max);
    let centers: Vec<Vec3> = spheres.iter().map(|sphere| sphere.center).collect();
    let hash = CellIndex::new(&centers, 2.0 * max_expanded);
    let point_spacing =
        (PROBE_POINT_SPACING * (probe / 1.4).sqrt()).clamp(0.3, PROBE_POINT_SPACING);
    // A probe inside an enclosed cavity has no reachable grid point around it.
    let accessible_from_outside = |point: Vec3| {
        if cavities {
            return true;
        }
        let cell = ((point - grid.origin) / grid.spacing).floor();
        let [nx, ny, nz] = grid.dims;
        (0..8).any(|corner| {
            let x = cell.x as isize + (corner & 1) as isize;
            let y = cell.y as isize + ((corner >> 1) & 1) as isize;
            let z = cell.z as isize + ((corner >> 2) & 1) as isize;
            (0..nx as isize).contains(&x)
                && (0..ny as isize).contains(&y)
                && (0..nz as isize).contains(&z)
                && reachable[grid.index(x as usize, y as usize, z as usize)]
        })
    };

    let threads = worker_count();
    let chunk = spheres.len().div_ceil(threads).max(1);
    let results: Vec<Option<Vec<Vec3>>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..spheres.len())
            .step_by(chunk)
            .map(|begin| {
                let hash = &hash;
                let accessible_from_outside = &accessible_from_outside;
                scope.spawn(move || {
                    let mut points = Vec::new();
                    let mut neighbors: Vec<(usize, Vec3, f32)> = Vec::new();
                    let mut circle_blockers: Vec<(usize, Vec3, f32)> = Vec::new();
                    for index in begin..(begin + chunk).min(spheres.len()) {
                        if index % 256 == 0 && cancel.load(Ordering::Relaxed) {
                            return None;
                        }
                        let sphere = &spheres[index];
                        let radius = expanded(sphere);
                        neighbors.clear();
                        hash.visit(sphere.center, 1, |other| {
                            if other != index {
                                let neighbor = &spheres[other];
                                let reach = radius + expanded(neighbor);
                                if neighbor.center.distance_squared(sphere.center) < reach * reach {
                                    neighbors.push((other, neighbor.center, expanded(neighbor)));
                                }
                            }
                        });
                        let blocked = |point: Vec3, blockers: &[(usize, Vec3, f32)]| {
                            blockers.iter().any(|(_, center, radius)| {
                                let own = radius - 1e-3;
                                center.distance_squared(point) < own * own
                            })
                        };
                        let count = ((4.0 * std::f32::consts::PI * radius * radius)
                            / (point_spacing * point_spacing))
                            .ceil()
                            .max(12.0) as usize;
                        let mut last_blocker = 0;
                        for direction in fibonacci_sphere(count) {
                            let point = sphere.center + direction * radius;
                            // The neighbor that covered the previous point usually covers
                            // the next one too.
                            if neighbors.get(last_blocker).is_some_and(|blocker| {
                                blocked(point, std::slice::from_ref(blocker))
                            }) {
                                continue;
                            }
                            if let Some(blocker) = neighbors
                                .iter()
                                .position(|blocker| blocked(point, std::slice::from_ref(blocker)))
                            {
                                last_blocker = blocker;
                                continue;
                            }
                            if accessible_from_outside(point) {
                                points.push(point);
                            }
                        }
                        // Probe positions along the arcs where two expanded spheres meet.
                        // They define the bottoms of the grooves between atoms, which
                        // sphere sampling alone would only approximate.
                        for &(other, center, other_radius) in &neighbors {
                            if other < index {
                                continue;
                            }
                            let axis = center - sphere.center;
                            let distance = axis.length();
                            if distance <= (radius - other_radius).abs() || distance < 1e-4 {
                                continue;
                            }
                            let along = (distance * distance + radius * radius
                                - other_radius * other_radius)
                                / (2.0 * distance);
                            let circle_squared = radius * radius - along * along;
                            if circle_squared <= 0.0 {
                                continue;
                            }
                            let circle_radius = circle_squared.sqrt();
                            let axis = axis / distance;
                            let circle_center = sphere.center + axis * along;
                            circle_blockers.clear();
                            circle_blockers.extend(neighbors.iter().copied().filter(
                                |&(candidate, blocker_center, blocker_radius)| {
                                    candidate != other
                                        && blocker_center.distance(circle_center)
                                            < blocker_radius + circle_radius
                                },
                            ));
                            let (first, second) = axis.any_orthonormal_pair();
                            let count = ((std::f32::consts::TAU * circle_radius) / point_spacing)
                                .ceil()
                                .max(8.0) as usize;
                            for step in 0..count {
                                let angle = std::f32::consts::TAU * step as f32 / count as f32;
                                let point = circle_center
                                    + (first * angle.cos() + second * angle.sin()) * circle_radius;
                                if !blocked(point, &circle_blockers)
                                    && accessible_from_outside(point)
                                {
                                    points.push(point);
                                }
                            }
                        }
                    }
                    Some(points)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(None))
            .collect()
    });
    let mut probes = Vec::new();
    for result in results {
        probes.extend(result?);
    }
    Some(probes)
}

fn fibonacci_sphere(count: usize) -> impl Iterator<Item = Vec3> {
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    (0..count).map(move |index| {
        let z = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
        let radius = (1.0 - z * z).max(0.0).sqrt();
        let angle = golden_angle * index as f32;
        Vec3::new(radius * angle.cos(), radius * angle.sin(), z)
    })
}

/// Atom owning each surface vertex: the one whose (expanded) sphere is closest.
pub(super) fn assign_atoms(
    positions: &[Vec3],
    spheres: &[SurfaceSphere],
    padding: f32,
) -> Vec<u32> {
    let max_radius = spheres
        .iter()
        .map(|sphere| sphere.radius)
        .fold(0.0_f32, f32::max);
    // Every vertex lies within one probe radius (plus grid error) of an atom surface.
    let search = max_radius + padding.max(1.4) + 1.0;
    let rings = 2;
    let centers: Vec<Vec3> = spheres.iter().map(|sphere| sphere.center).collect();
    let index = CellIndex::new(&centers, search / rings as f32);
    let mut owners = vec![0_u32; positions.len()];
    let chunk = positions.len().div_ceil(worker_count()).max(1024);
    thread::scope(|scope| {
        for (positions, owners) in positions.chunks(chunk).zip(owners.chunks_mut(chunk)) {
            let index = &index;
            scope.spawn(move || {
                for (position, owner) in positions.iter().zip(owners) {
                    let mut best = (f32::INFINITY, usize::MAX);
                    index.visit(*position, rings, |candidate| {
                        let sphere = &spheres[candidate];
                        let distance = sphere.center.distance(*position) - sphere.radius - padding;
                        if distance < best.0 {
                            best = (distance, candidate);
                        }
                    });
                    if best.1 == usize::MAX {
                        best.1 = spheres
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                (a.center.distance(*position) - a.radius)
                                    .total_cmp(&(b.center.distance(*position) - b.radius))
                            })
                            .map_or(0, |(index, _)| index);
                    }
                    *owner = spheres[best.1].atom;
                }
            });
        }
    });
    owners
}

/// Dense uniform cell index over points in compressed-row form; `visit` reports every point
/// in the cells within `rings` cells of the query point.
struct CellIndex {
    origin: Vec3,
    cell: f32,
    dims: [usize; 3],
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl CellIndex {
    fn new(points: &[Vec3], cell: f32) -> Self {
        let cell = cell.max(0.1);
        let (low, high) = points.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(low, high), point| (low.min(*point), high.max(*point)),
        );
        let (low, high) = if points.is_empty() {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (low, high)
        };
        let dims = ((high - low) / cell)
            .floor()
            .to_array()
            .map(|axis| axis as usize + 1);
        let mut index = Self {
            origin: low,
            cell,
            dims,
            starts: vec![0; dims.iter().product::<usize>() + 1],
            items: vec![0; points.len()],
        };
        let cells: Vec<usize> = points.iter().map(|point| index.cell_of(*point)).collect();
        for &cell in &cells {
            index.starts[cell + 1] += 1;
        }
        for cell in 1..index.starts.len() {
            index.starts[cell] += index.starts[cell - 1];
        }
        let mut fill = index.starts.clone();
        for (point, &cell) in cells.iter().enumerate() {
            index.items[fill[cell] as usize] = point as u32;
            fill[cell] += 1;
        }
        index
    }

    fn coordinates(&self, point: Vec3) -> [isize; 3] {
        ((point - self.origin) / self.cell)
            .floor()
            .to_array()
            .map(|axis| axis as isize)
    }

    fn cell_of(&self, point: Vec3) -> usize {
        let [x, y, z] = self.coordinates(point).map(|axis| axis.max(0) as usize);
        let [nx, ny, nz] = self.dims;
        x.min(nx - 1) + nx * (y.min(ny - 1) + ny * z.min(nz - 1))
    }

    fn visit(&self, point: Vec3, rings: isize, mut visit: impl FnMut(usize)) {
        let [cx, cy, cz] = self.coordinates(point);
        let [nx, ny, nz] = self.dims.map(|axis| axis as isize);
        let range =
            |center: isize, limit: isize| (center - rings).max(0)..(center + rings + 1).min(limit);
        for z in range(cz, nz) {
            for y in range(cy, ny) {
                let xs = range(cx, nx);
                if xs.is_empty() {
                    continue;
                }
                let row = (nx * (y + ny * z)) as usize;
                let begin = self.starts[row + xs.start as usize] as usize;
                let end = self.starts[row + xs.end as usize] as usize;
                for &item in &self.items[begin..end] {
                    visit(item as usize);
                }
            }
        }
    }
}

fn worker_count() -> usize {
    thread::available_parallelism()
        .map_or(4, |count| count.get())
        .clamp(1, 16)
}
