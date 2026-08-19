struct PostUniform {
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    optical_axis: vec4<f32>,
    lens: vec4<f32>,
    aperture: vec4<f32>,
    // strength, world-space radius, normal bias, sample count (zero disables)
    ao: vec4<f32>,
};

@group(0) @binding(0)
var scene_depth: texture_depth_2d;
@group(0) @binding(1)
var<uniform> post: PostUniform;
@group(0) @binding(2)
var raw_ao: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn world_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = post.inverse_view_projection * clip;
    return world_h.xyz / world_h.w;
}

fn depth_at(pixel: vec2<i32>, dimensions: vec2<u32>) -> f32 {
    let maximum = vec2<i32>(dimensions) - 1;
    return textureLoad(scene_depth, clamp(pixel, vec2<i32>(0), maximum), 0);
}

fn optical_depth(uv: vec2<f32>, depth: f32) -> f32 {
    if depth >= 0.999999 {
        return 1e20;
    }
    let world = world_position(uv, depth);
    return max(dot(world - post.eye_position.xyz, post.optical_axis.xyz), 1e-4);
}

fn reconstructed_normal(
    pixel: vec2<i32>,
    uv: vec2<f32>,
    center: vec3<f32>,
    dimensions: vec2<u32>,
) -> vec3<f32> {
    let inverse_dimensions = 1.0 / vec2<f32>(dimensions);
    let left = world_position(
        uv - vec2<f32>(inverse_dimensions.x, 0.0),
        depth_at(pixel + vec2<i32>(-1, 0), dimensions),
    );
    let right = world_position(
        uv + vec2<f32>(inverse_dimensions.x, 0.0),
        depth_at(pixel + vec2<i32>(1, 0), dimensions),
    );
    let top = world_position(
        uv - vec2<f32>(0.0, inverse_dimensions.y),
        depth_at(pixel + vec2<i32>(0, -1), dimensions),
    );
    let bottom = world_position(
        uv + vec2<f32>(0.0, inverse_dimensions.y),
        depth_at(pixel + vec2<i32>(0, 1), dimensions),
    );
    let dx = select(center - left, right - center, distance(center, left) > distance(center, right));
    let dy = select(center - top, bottom - center, distance(center, top) > distance(center, bottom));
    var normal = normalize(cross(dy, dx));
    if dot(normal, post.eye_position.xyz - center) < 0.0 {
        normal = -normal;
    }
    return normal;
}

@fragment
fn raw_ao_main(input: VertexOutput) -> @location(0) f32 {
    let dimensions_u = textureDimensions(scene_depth);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel = vec2<i32>(clamp(input.position.xy, vec2<f32>(0.0), dimensions - 1.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    let sample_count = u32(round(post.ao.w));
    if sample_count == 0u || center_depth >= 0.999999 {
        return 1.0;
    }

    let uv = (vec2<f32>(pixel) + 0.5) / dimensions;
    let center = world_position(uv, center_depth);
    let normal = reconstructed_normal(pixel, uv, center, dimensions_u);
    let view_depth = optical_depth(uv, center_depth);
    let radius = max(post.ao.y, 0.05);
    let radius_pixels = clamp(radius / view_depth * dimensions.y, 2.0, 128.0);

    // A stable 4x4 rotation pattern preserves detail; the following bilateral pass
    // removes the residual pattern without bleeding across molecular silhouettes.
    let tile = vec2<u32>(pixel) & vec2<u32>(3u);
    let rotation_index = tile.x + tile.y * 4u;
    let rotation = 6.28318530718 * (f32(rotation_index) + 0.5) / 16.0;
    var obscurance = 0.0;
    for (var index = 0u; index < 48u; index += 1u) {
        if index >= sample_count {
            break;
        }
        let fraction = (f32(index) + 0.5) / f32(sample_count);
        let angle = f32(index) * 2.39996322973 + rotation;
        let offset = vec2<f32>(cos(angle), sin(angle)) * sqrt(fraction) * radius_pixels;
        let sample_pixel = clamp(
            pixel + vec2<i32>(round(offset)),
            vec2<i32>(0),
            vec2<i32>(dimensions_u) - 1,
        );
        let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
        if sample_depth >= 0.999999 {
            continue;
        }
        let sample_uv = (vec2<f32>(sample_pixel) + 0.5) / dimensions;
        let delta = world_position(sample_uv, sample_depth) - center;
        let sample_distance = length(delta);
        if sample_distance <= 1e-4 || sample_distance >= radius {
            continue;
        }
        let cosine = max(dot(normal, delta / sample_distance) - post.ao.z, 0.0);
        let falloff = 1.0 - smoothstep(radius * 0.12, radius, sample_distance);
        obscurance += cosine * falloff;
    }
    let normalized = obscurance * (2.5 / f32(sample_count));
    return clamp(1.0 - post.ao.x * normalized, 0.18, 1.0);
}

@fragment
fn blur_ao_main(input: VertexOutput) -> @location(0) f32 {
    let dimensions_u = textureDimensions(raw_ao);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel = vec2<i32>(clamp(input.position.xy, vec2<f32>(0.0), dimensions - 1.0));
    let center_depth = textureLoad(scene_depth, pixel, 0);
    if post.ao.w < 0.5 || center_depth >= 0.999999 {
        return 1.0;
    }

    let center_uv = (vec2<f32>(pixel) + 0.5) / dimensions;
    let center_view_depth = optical_depth(center_uv, center_depth);
    let depth_sigma = max(post.ao.y * 0.075, 0.015);
    var accumulated = 0.0;
    var accumulated_weight = 0.0;
    for (var y: i32 = -2; y <= 2; y += 1) {
        for (var x: i32 = -2; x <= 2; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_pixel = clamp(pixel + offset, vec2<i32>(0), vec2<i32>(dimensions_u) - 1);
            let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
            if sample_depth >= 0.999999 {
                continue;
            }
            let sample_uv = (vec2<f32>(sample_pixel) + 0.5) / dimensions;
            let depth_delta = abs(optical_depth(sample_uv, sample_depth) - center_view_depth);
            let spatial_distance_squared = f32(x * x + y * y);
            let spatial_weight = exp(-spatial_distance_squared * 0.32);
            let edge_weight = exp(-(depth_delta * depth_delta) / (2.0 * depth_sigma * depth_sigma));
            let weight = spatial_weight * edge_weight;
            accumulated += textureLoad(raw_ao, sample_pixel, 0).r * weight;
            accumulated_weight += weight;
        }
    }
    return accumulated / max(accumulated_weight, 1e-4);
}
