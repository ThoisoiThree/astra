struct PostUniform {
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    optical_axis: vec4<f32>,
    // focus distance, f-number, maximum CoC radius in pixels, enabled
    lens: vec4<f32>,
    // focal length for a unit-height sensor, diaphragm blades, rotation, reserved
    aperture: vec4<f32>,
};

@group(0) @binding(0)
var scene_color: texture_2d<f32>;
@group(0) @binding(1)
var scene_sampler: sampler;
@group(0) @binding(2)
var scene_depth: texture_depth_2d;
@group(0) @binding(3)
var<uniform> post: PostUniform;

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

fn optical_depth(uv: vec2<f32>, depth: f32) -> f32 {
    if depth >= 0.999999 {
        return 1e20;
    }
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = post.inverse_view_projection * clip;
    let world = world_h.xyz / world_h.w;
    return max(dot(world - post.eye_position.xyz, post.optical_axis.xyz), 1e-4);
}

// Signed thin-lens circle of confusion. Scene and focal length use the same
// virtual unit-height sensor scale, so the result is a sensor-height fraction.
fn circle_of_confusion(distance: f32, image_height: f32) -> f32 {
    if distance >= 1e19 {
        return post.lens.z;
    }
    let focus = max(post.lens.x, post.aperture.x + 1e-4);
    let object_distance = max(distance, post.aperture.x + 1e-4);
    let focal_length = post.aperture.x;
    let f_number = max(post.lens.y, 0.1);
    let sensor_coc = focal_length * focal_length * (object_distance - focus)
        / (f_number * object_distance * (focus - focal_length));
    return clamp(sensor_coc * image_height, -post.lens.z, post.lens.z);
}

fn aperture_boundary(angle: f32) -> f32 {
    let blades = floor(post.aperture.y + 0.5);
    if blades < 3.0 {
        return 1.0;
    }
    let sector = 6.28318530718 / blades;
    let local_angle = (angle + 3.14159265359) % sector - sector * 0.5;
    return cos(3.14159265359 / blades) / max(cos(local_angle), 1e-4);
}

fn aperture_sample(index: f32, count: f32) -> vec2<f32> {
    let angle = index * 2.39996322973 + post.aperture.z;
    let disk_radius = sqrt((index + 0.5) / count);
    let radius = disk_radius * aperture_boundary(angle);
    return vec2<f32>(cos(angle), sin(angle)) * radius;
}

fn aperture_metric(point: vec2<f32>) -> f32 {
    let angle = atan2(point.y, point.x) - post.aperture.z;
    return length(point) / max(aperture_boundary(angle), 1e-4);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(scene_color);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel = vec2<i32>(clamp(input.position.xy, vec2<f32>(0.0), dimensions - 1.0));
    let uv = input.position.xy / dimensions;
    let center = textureSampleLevel(scene_color, scene_sampler, uv, 0.0);
    if post.lens.w < 0.5 || post.lens.z <= 0.0 {
        return center;
    }

    let center_depth = textureLoad(scene_depth, pixel, 0);
    let center_coc = circle_of_confusion(optical_depth(uv, center_depth), dimensions.y);

    // A source-based aperture gather approximates optical scatter. Coverage and
    // near/far classification are deliberately continuous: binary CoC tests make
    // small atoms and highlights pop as the camera crosses a sample boundary.
    const SAMPLE_COUNT = 64u;
    var accumulated = center;
    var accumulated_weight = 1.0;
    for (var index = 0u; index < SAMPLE_COUNT; index += 1u) {
        let aperture_point = aperture_sample(f32(index), f32(SAMPLE_COUNT));
        let offset_pixels = aperture_point * post.lens.z;
        let sample_uv = clamp(uv + offset_pixels / dimensions, vec2<f32>(0.0), vec2<f32>(1.0));
        let sample_pixel = vec2<i32>(clamp(sample_uv * dimensions, vec2<f32>(0.0), dimensions - 1.0));
        let sample_depth = textureLoad(scene_depth, sample_pixel, 0);
        let sample_coc = circle_of_confusion(
            optical_depth(sample_uv, sample_depth),
            dimensions.y,
        );
        let required_radius = aperture_metric(aperture_point) * post.lens.z;
        let coc_radius = abs(sample_coc);
        let coverage = 1.0 - smoothstep(
            max(coc_radius - 1.25, 0.0),
            coc_radius + 1.25,
            required_radius,
        );
        let near_strength = smoothstep(0.0, 1.5, -sample_coc);
        let far_source_strength = smoothstep(0.0, 1.5, sample_coc);
        let far_receiver_strength = smoothstep(0.0, 1.5, center_coc);
        let layer_strength = near_strength
            + (1.0 - near_strength) * far_source_strength * far_receiver_strength;
        let rim_weight = 0.8 + 0.4 * aperture_metric(aperture_point);
        let weight = coverage * layer_strength * rim_weight;
        let sample_color = textureSampleLevel(scene_color, scene_sampler, sample_uv, 0.0);
        accumulated += sample_color * weight;
        accumulated_weight += weight;
    }
    return accumulated / accumulated_weight;
}
