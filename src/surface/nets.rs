//! Naive surface nets: one vertex per grid cell that the surface crosses, placed at the
//! mean of the edge crossings, and one quad per crossed grid edge. The result is watertight
//! wherever the field is, has well-shaped triangles and needs no case tables.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec3;

use super::{SurfaceMesh, field::Grid};

const NONE: u32 = u32::MAX;

/// Corner offsets in cell-local order: bit 0 → x, bit 1 → y, bit 2 → z.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// The twelve cell edges as corner pairs.
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

pub(super) fn extract(grid: &Grid, cancel: &AtomicBool) -> Option<SurfaceMesh> {
    let [nx, ny, nz] = grid.dims;
    let mut mesh = SurfaceMesh::default();
    if nx < 2 || ny < 2 || nz < 2 {
        return Some(mesh);
    }
    let (cells_x, cells_y) = (nx - 1, ny - 1);
    let mut previous = vec![NONE; cells_x * cells_y];
    let mut current = vec![NONE; cells_x * cells_y];
    let inside = |x: usize, y: usize, z: usize| grid.value(x, y, z) > 0.0;

    for z in 0..nz - 1 {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
        for y in 0..cells_y {
            for x in 0..cells_x {
                current[x + y * cells_x] = cell_vertex(grid, [x, y, z], &mut mesh);
            }
        }
        let cell = |layer: &[u32], x: usize, y: usize| layer[x + y * cells_x];

        // Edges along z between grid levels z and z + 1: the four cells share this layer.
        for y in 1..cells_y {
            for x in 1..cells_x {
                let (start, end) = (inside(x, y, z), inside(x, y, z + 1));
                if start != end {
                    emit_quad(
                        &mut mesh,
                        [
                            cell(&current, x - 1, y - 1),
                            cell(&current, x, y - 1),
                            cell(&current, x, y),
                            cell(&current, x - 1, y),
                        ],
                        start,
                    );
                }
            }
        }
        if z == 0 {
            continue;
        }
        // Edges along x and y on grid level z: cells from this layer and the one below.
        for y in 1..cells_y {
            for x in 0..cells_x {
                let (start, end) = (inside(x, y, z), inside(x + 1, y, z));
                if start != end {
                    emit_quad(
                        &mut mesh,
                        [
                            cell(&previous, x, y - 1),
                            cell(&previous, x, y),
                            cell(&current, x, y),
                            cell(&current, x, y - 1),
                        ],
                        start,
                    );
                }
            }
        }
        for y in 0..cells_y {
            for x in 1..cells_x {
                let (start, end) = (inside(x, y, z), inside(x, y + 1, z));
                if start != end {
                    emit_quad(
                        &mut mesh,
                        [
                            cell(&previous, x - 1, y),
                            cell(&current, x - 1, y),
                            cell(&current, x, y),
                            cell(&previous, x, y),
                        ],
                        start,
                    );
                }
            }
        }
    }
    Some(mesh)
}

/// Adds the vertex of a crossed cell and returns its index.
fn cell_vertex(grid: &Grid, [x, y, z]: [usize; 3], mesh: &mut SurfaceMesh) -> u32 {
    // Field values never exceed the distance to the surface, so a corner farther than the
    // cell diagonal proves the cell is not crossed.
    if grid.value(x, y, z).abs() > grid.spacing * 1.75 {
        return NONE;
    }
    let values = CORNERS.map(|[dx, dy, dz]| grid.value(x + dx, y + dy, z + dz));
    let inside_count = values.iter().filter(|value| **value > 0.0).count();
    if inside_count == 0 || inside_count == 8 {
        return NONE;
    }
    let mut sum = Vec3::ZERO;
    let mut crossings = 0.0;
    for [a, b] in EDGES {
        let (va, vb) = (values[a], values[b]);
        if (va > 0.0) != (vb > 0.0) {
            let t = (va / (va - vb)).clamp(0.0, 1.0);
            let pa = Vec3::from(CORNERS[a].map(|offset| offset as f32));
            let pb = Vec3::from(CORNERS[b].map(|offset| offset as f32));
            sum += pa.lerp(pb, t);
            crossings += 1.0;
        }
    }
    let mut local = sum / crossings;
    let mut gradient = interpolated_gradient(grid, [x, y, z], local);
    // One Newton step along the gradient moves the vertex from the chord average onto the
    // interpolated level set, which removes the inward bias on curved surfaces.
    let length_squared = gradient.length_squared();
    if length_squared > 1e-12 {
        let value = trilinear(&values, local);
        let step = (-gradient * value / length_squared).clamp_length_max(0.5);
        local = (local + step).clamp(Vec3::ZERO, Vec3::ONE);
        gradient = interpolated_gradient(grid, [x, y, z], local);
    }
    // The field increases inward, so the outward normal is the negative gradient.
    let normal = (-gradient).try_normalize().unwrap_or(Vec3::Z);
    let index = mesh.positions.len() as u32;
    mesh.positions
        .push(grid.point(x as f32 + local.x, y as f32 + local.y, z as f32 + local.z));
    mesh.normals.push(normal);
    index
}

fn trilinear(values: &[f32; 8], local: Vec3) -> f32 {
    CORNERS
        .iter()
        .zip(values)
        .map(|(&[dx, dy, dz], value)| corner_weight([dx, dy, dz], local) * value)
        .sum()
}

fn corner_weight([dx, dy, dz]: [usize; 3], local: Vec3) -> f32 {
    (if dx == 1 { local.x } else { 1.0 - local.x })
        * (if dy == 1 { local.y } else { 1.0 - local.y })
        * (if dz == 1 { local.z } else { 1.0 - local.z })
}

/// Central-difference gradients at the cell corners, trilinearly interpolated.
fn interpolated_gradient(grid: &Grid, [x, y, z]: [usize; 3], local: Vec3) -> Vec3 {
    let [nx, ny, nz] = grid.dims;
    let difference = |x: usize, y: usize, z: usize| {
        let axis = |position: usize, limit: usize, sample: &dyn Fn(usize) -> f32| {
            let low = position.saturating_sub(1);
            let high = (position + 1).min(limit - 1);
            (sample(high) - sample(low)) / (high - low).max(1) as f32
        };
        Vec3::new(
            axis(x, nx, &|value| grid.value(value, y, z)),
            axis(y, ny, &|value| grid.value(x, value, z)),
            axis(z, nz, &|value| grid.value(x, y, value)),
        )
    };
    CORNERS
        .iter()
        .map(|&[dx, dy, dz]| {
            difference(x + dx, y + dy, z + dz) * corner_weight([dx, dy, dz], local)
        })
        .sum()
}

/// Two triangles for the quad around a crossed edge, wound counter-clockwise when seen
/// from outside and split along the shorter diagonal.
fn emit_quad(mesh: &mut SurfaceMesh, mut quad: [u32; 4], start_inside: bool) {
    if quad.contains(&NONE) {
        return;
    }
    if !start_inside {
        quad.reverse();
    }
    let [a, b, c, d] = quad;
    let position = |index: u32| mesh.positions[index as usize];
    if position(a).distance_squared(position(c)) <= position(b).distance_squared(position(d)) {
        mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
    } else {
        mesh.indices.extend_from_slice(&[a, b, d, b, c, d]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangles_face_outward() {
        let dims = [24, 24, 24];
        let center = Vec3::splat(11.5);
        let mut values = Vec::new();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    values.push(7.0 - Vec3::new(x as f32, y as f32, z as f32).distance(center));
                }
            }
        }
        let grid = Grid {
            origin: Vec3::ZERO,
            spacing: 1.0,
            dims,
            values,
        };
        let mesh = extract(&grid, &AtomicBool::new(false)).unwrap();
        assert!(mesh.triangle_count() > 500);
        for triangle in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = [0, 1, 2].map(|corner| mesh.positions[triangle[corner] as usize]);
            let face = (b - a).cross(c - a);
            let centroid = (a + b + c) / 3.0;
            assert!(face.dot(centroid - center) > 0.0);
        }
        // Closed surface: every edge is shared by exactly two triangles.
        let mut edges = std::collections::HashMap::<(u32, u32), i32>::new();
        for triangle in mesh.indices.as_chunks::<3>().0 {
            for corner in 0..3 {
                let (a, b) = (triangle[corner], triangle[(corner + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        assert!(edges.values().all(|count| *count == 2));
    }
}
