// Shared scene bindings: camera plus per-atom storage indexed by atom number.
// Trajectory playback rewrites only `atom_positions`; display edits rewrite colors and meta.

struct Camera {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;
// xyz position in Å; w unused.
@group(0) @binding(1)
var<storage, read> atom_positions: array<vec4<f32>>;
// Linear RGB and alpha.
@group(0) @binding(2)
var<storage, read> atom_colors: array<vec4<f32>>;
// x: atom index + 1, y: chain << 20 | residue, z: flags (bit 0 selected), w: unused.
@group(0) @binding(3)
var<storage, read> atom_meta: array<vec4<u32>>;

const HIGHLIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.84, 0.08);

fn atom_selected(atom: u32) -> f32 {
    return select(0.0, 1.0, (atom_meta[atom].z & 1u) != 0u);
}

fn atom_semantic(atom: u32) -> vec2<u32> {
    let info = atom_meta[atom];
    return vec2<u32>(info.x, info.y);
}

// Key light, cool fill light and a soft ambient term; specular stays neutral.
fn shade_surface(
    normal_in: vec3<f32>,
    world_position: vec3<f32>,
    base_color: vec3<f32>,
    highlight: f32,
) -> vec3<f32> {
    let normal = normalize(normal_in);
    let view_direction = normalize(camera.eye_position.xyz - world_position);
    let key = normalize(vec3<f32>(0.35, 0.65, 0.70));
    let fill = normalize(vec3<f32>(-0.55, -0.25, 0.45));
    let key_diffuse = max(dot(normal, key), 0.0);
    let fill_diffuse = max(dot(normal, fill), 0.0);
    let half_vector = normalize(key + view_direction);
    let specular = pow(max(dot(normal, half_vector), 0.0), 36.0) * 0.30;
    let rim = pow(1.0 - max(dot(normal, view_direction), 0.0), 3.0) * 0.08;
    let color = mix(base_color, HIGHLIGHT_COLOR, highlight * 0.90);
    let lit = color * (0.24 + 0.66 * key_diffuse + 0.14 * fill_diffuse)
        + vec3<f32>(specular + rim);
    return lit + HIGHLIGHT_COLOR * highlight * 0.24;
}

// Broad, quantized diffuse bands reproduce a flat pastel illustration style while AO and
// semantic contours carry most of the shape information.
fn shade_toon(normal_in: vec3<f32>, base_color: vec3<f32>, highlight: f32) -> vec3<f32> {
    let normal = normalize(normal_in);
    let light_direction = normalize(vec3<f32>(0.30, 0.55, 0.78));
    let diffuse = dot(normal, light_direction);
    let broad_light = smoothstep(-0.35, 0.72, diffuse);
    let light_band = mix(0.82, 1.0, smoothstep(0.20, 0.82, broad_light));
    let pastel = mix(base_color, vec3<f32>(1.0), 0.12);
    let selected = mix(pastel, vec3<f32>(1.0, 0.87, 0.12), highlight * 0.88);
    return selected * light_band;
}

fn clip_depth(world_position: vec3<f32>) -> f32 {
    let clip = camera.view_projection * vec4<f32>(world_position, 1.0);
    return clamp(clip.z / clip.w, 0.0, 1.0);
}

// Orthonormal vectors perpendicular to `axis`.
fn perpendicular_basis(axis: vec3<f32>) -> mat2x3<f32> {
    var helper = camera.camera_up.xyz;
    if abs(dot(helper, axis)) > 0.95 {
        helper = camera.camera_right.xyz;
    }
    let first = normalize(cross(axis, helper));
    let second = cross(axis, first);
    return mat2x3<f32>(first, second);
}

struct SceneFragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) semantic_ids: vec2<u32>,
    @builtin(frag_depth) depth: f32,
};

struct ColorFragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};
