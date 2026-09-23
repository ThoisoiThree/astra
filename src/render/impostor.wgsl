// Ray-cast sphere and capsule impostors. Geometry is exact at every zoom level and writes
// its true depth, so impostors intersect correctly with each other and with meshes.

const STYLE_TOON: u32 = 1u;
const BOND_DASHED: u32 = 1u;
const DASH_PERIOD: f32 = 0.36;
const DASH_DUTY: f32 = 0.55;

struct SphereInput {
    @location(0) atom: u32,
    @location(1) radius: f32,
    @location(2) style: u32,
};

struct SphereVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) center_radius: vec4<f32>,
    @location(2) @interpolate(flat) atom: u32,
    @location(3) @interpolate(flat) style: u32,
};

@vertex
fn vertex_sphere(input: SphereInput, @builtin(vertex_index) vertex_index: u32) -> SphereVarying {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let selected = atom_selected(input.atom);
    let growth = select(0.16, 0.04, input.style == STYLE_TOON);
    let radius = input.radius * (1.0 + selected * growth);
    let center = atom_positions[input.atom].xyz;
    let to_eye = camera.eye_position.xyz - center;
    let distance = length(to_eye);
    var output: SphereVarying;
    output.center_radius = vec4<f32>(center, radius);
    output.atom = input.atom;
    output.style = input.style;
    if distance <= radius * 1.001 {
        // The camera is inside the sphere: emit a degenerate triangle.
        output.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        output.world_position = center;
        return output;
    }
    // A quad on the sphere's front tangent plane, sized to the exact silhouette cone.
    let direction = to_eye / distance;
    let basis = perpendicular_basis(direction);
    let half_size = (distance - radius) * radius / sqrt(distance * distance - radius * radius);
    let corner = corners[vertex_index] * 1.02;
    let world = center + direction * radius
        + (basis[0] * corner.x + basis[1] * corner.y) * half_size;
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    output.world_position = world;
    return output;
}

struct Hit {
    position: vec3<f32>,
    normal: vec3<f32>,
    // Fraction along a bond axis; unused for spheres.
    axial: f32,
};

fn trace_sphere(input: SphereVarying) -> Hit {
    let origin = camera.eye_position.xyz;
    let direction = normalize(input.world_position - origin);
    let offset = origin - input.center_radius.xyz;
    let b = dot(offset, direction);
    let c = dot(offset, offset) - input.center_radius.w * input.center_radius.w;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        discard;
    }
    let distance = -b - sqrt(discriminant);
    if distance <= 0.0 {
        discard;
    }
    let position = origin + direction * distance;
    var hit: Hit;
    hit.position = position;
    hit.normal = (position - input.center_radius.xyz) / input.center_radius.w;
    hit.axial = 0.0;
    return hit;
}

fn sphere_color(input: SphereVarying, hit: Hit) -> vec4<f32> {
    let base = atom_colors[input.atom];
    let selected = atom_selected(input.atom);
    if input.style == STYLE_TOON {
        return vec4<f32>(shade_toon(hit.normal, base.rgb, selected), base.a);
    }
    return vec4<f32>(shade_surface(hit.normal, hit.position, base.rgb, selected), base.a);
}

@fragment
fn fragment_sphere_scene(input: SphereVarying) -> SceneFragmentOutput {
    let hit = trace_sphere(input);
    var output: SceneFragmentOutput;
    output.color = sphere_color(input, hit);
    output.semantic_ids = atom_semantic(input.atom);
    output.depth = clip_depth(hit.position);
    return output;
}

@fragment
fn fragment_sphere_color(input: SphereVarying) -> ColorFragmentOutput {
    let hit = trace_sphere(input);
    var output: ColorFragmentOutput;
    output.color = sphere_color(input, hit);
    output.depth = clip_depth(hit.position);
    return output;
}

@fragment
fn fragment_sphere_peel(input: SphereVarying) -> ColorFragmentOutput {
    let hit = trace_sphere(input);
    var output: ColorFragmentOutput;
    output.depth = clip_depth(hit.position);
    reject_peeled_fragment(input.clip_position.xy, output.depth);
    output.color = sphere_color(input, hit);
    return output;
}

struct BondInput {
    @location(0) atoms: vec2<u32>,
    @location(1) reference: u32,
    @location(2) style: u32,
    @location(3) radius: f32,
    @location(4) offset: f32,
};

struct BondVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) start: vec3<f32>,
    @location(2) end_radius: vec4<f32>,
    @location(3) @interpolate(flat) atoms: vec2<u32>,
    @location(4) @interpolate(flat) style: u32,
};

// Twelve outward-facing triangles of the unit cube [0,1]^3, counter-clockwise.
fn cube_corner(index: u32) -> vec3<f32> {
    let faces = array<u32, 36>(
        0u, 2u, 1u, 1u, 2u, 3u, // -z
        4u, 5u, 6u, 5u, 7u, 6u, // +z
        0u, 1u, 4u, 1u, 5u, 4u, // -y
        2u, 6u, 3u, 3u, 6u, 7u, // +y
        0u, 4u, 2u, 2u, 4u, 6u, // -x
        1u, 3u, 5u, 3u, 7u, 5u, // +x
    );
    let corner = faces[index];
    return vec3<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u), f32((corner >> 2u) & 1u));
}

@vertex
fn vertex_bond(input: BondInput, @builtin(vertex_index) vertex_index: u32) -> BondVarying {
    var start = atom_positions[input.atoms.x].xyz;
    var end = atom_positions[input.atoms.y].xyz;
    let axis_vector = end - start;
    let length = length(axis_vector);
    var output: BondVarying;
    output.atoms = input.atoms;
    output.style = input.style;
    if length < 1e-4 {
        output.clip_position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        output.world_position = start;
        output.start = start;
        output.end_radius = vec4<f32>(end, input.radius);
        return output;
    }
    let axis = axis_vector / length;
    // Multiple bonds are offset in the plane of a neighboring atom, or toward the viewer's
    // side when the bond has no neighbor.
    if input.offset != 0.0 {
        var side = vec3<f32>(0.0);
        if input.reference != 0xffffffffu {
            let toward = atom_positions[input.reference].xyz - start;
            side = toward - axis * dot(toward, axis);
        }
        if dot(side, side) < 1e-8 {
            side = cross(axis, camera.eye_position.xyz - (start + end) * 0.5);
        }
        let shift = normalize(side) * input.offset;
        start += shift;
        end += shift;
    }
    let radius = input.radius * (1.0 + 0.10 * max(atom_selected(input.atoms.x), atom_selected(input.atoms.y)));
    let basis = perpendicular_basis(axis);
    let unit = cube_corner(vertex_index);
    let world = start - axis * radius
        + axis * (unit.x * (length + 2.0 * radius))
        + basis[0] * ((unit.y * 2.0 - 1.0) * radius)
        + basis[1] * ((unit.z * 2.0 - 1.0) * radius);
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    output.world_position = world;
    output.start = start;
    output.end_radius = vec4<f32>(end, radius);
    return output;
}

// Ray–capsule intersection (Inigo Quilez), returning the entry distance or -1.
fn capsule_distance(origin: vec3<f32>, direction: vec3<f32>, a: vec3<f32>, b: vec3<f32>, radius: f32) -> f32 {
    let ba = b - a;
    let oa = origin - a;
    let baba = dot(ba, ba);
    let bard = dot(ba, direction);
    let baoa = dot(ba, oa);
    let rdoa = dot(direction, oa);
    let oaoa = dot(oa, oa);
    let qa = baba - bard * bard;
    if qa > 1e-8 {
        let qb = baba * rdoa - baoa * bard;
        let qc = baba * oaoa - baoa * baoa - radius * radius * baba;
        let h = qb * qb - qa * qc;
        if h >= 0.0 {
            let t = (-qb - sqrt(h)) / qa;
            let y = baoa + t * bard;
            if y > 0.0 && y < baba {
                return t;
            }
        }
    }
    // Hemispherical caps.
    var best = -1.0;
    for (var cap = 0u; cap < 2u; cap += 1u) {
        let center = select(a, b, cap == 1u);
        let oc = origin - center;
        let cb = dot(direction, oc);
        let cc = dot(oc, oc) - radius * radius;
        let ch = cb * cb - cc;
        if ch > 0.0 {
            let t = -cb - sqrt(ch);
            if t > 0.0 && (best < 0.0 || t < best) {
                best = t;
            }
        }
    }
    return best;
}

fn trace_bond(input: BondVarying) -> Hit {
    let origin = camera.eye_position.xyz;
    let direction = normalize(input.world_position - origin);
    let a = input.start;
    let b = input.end_radius.xyz;
    let radius = input.end_radius.w;
    let distance = capsule_distance(origin, direction, a, b, radius);
    if distance <= 0.0 {
        discard;
    }
    let position = origin + direction * distance;
    let ba = b - a;
    let along = clamp(dot(position - a, ba) / dot(ba, ba), 0.0, 1.0);
    if (input.style & BOND_DASHED) != 0u {
        let travelled = along * length(ba);
        if fract(travelled / DASH_PERIOD) > DASH_DUTY {
            discard;
        }
    }
    var hit: Hit;
    hit.position = position;
    hit.normal = (position - (a + along * ba)) / radius;
    hit.axial = along;
    return hit;
}

// Each half of a bond takes the color of its atom.
fn bond_atom(input: BondVarying, hit: Hit) -> u32 {
    return select(input.atoms.y, input.atoms.x, hit.axial < 0.5);
}

fn bond_color(input: BondVarying, hit: Hit) -> vec4<f32> {
    let atom = bond_atom(input, hit);
    let base = atom_colors[atom];
    let selected = max(atom_selected(input.atoms.x), atom_selected(input.atoms.y));
    return vec4<f32>(shade_surface(hit.normal, hit.position, base.rgb, selected), base.a);
}

@fragment
fn fragment_bond_scene(input: BondVarying) -> SceneFragmentOutput {
    let hit = trace_bond(input);
    var output: SceneFragmentOutput;
    output.color = bond_color(input, hit);
    output.semantic_ids = atom_semantic(bond_atom(input, hit));
    output.depth = clip_depth(hit.position);
    return output;
}

@fragment
fn fragment_bond_color(input: BondVarying) -> ColorFragmentOutput {
    let hit = trace_bond(input);
    var output: ColorFragmentOutput;
    output.color = bond_color(input, hit);
    output.depth = clip_depth(hit.position);
    return output;
}

@fragment
fn fragment_bond_peel(input: BondVarying) -> ColorFragmentOutput {
    let hit = trace_bond(input);
    var output: ColorFragmentOutput;
    output.depth = clip_depth(hit.position);
    reject_peeled_fragment(input.clip_position.xy, output.depth);
    output.color = bond_color(input, hit);
    return output;
}
