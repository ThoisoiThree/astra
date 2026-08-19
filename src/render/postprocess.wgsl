struct PostUniform {
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    optical_axis: vec4<f32>,
    // focus distance, f-number, maximum CoC radius in pixels, enabled
    lens: vec4<f32>,
    // focal length for a unit-height sensor, diaphragm blades, rotation, reserved
    aperture: vec4<f32>,
    // AO strength, world-space radius, normal bias, sample count (zero disables)
    ao: vec4<f32>,
};

@group(0) @binding(0)
var scene_color: texture_2d<f32>;
@group(0) @binding(1)
var scene_sampler: sampler;
@group(0) @binding(2)
var scene_depth: texture_depth_2d;
@group(0) @binding(3)
var<uniform> post: PostUniform;
@group(0) @binding(4)
var filtered_ao: texture_2d<f32>;
@group(0) @binding(5)
var semantic_ids: texture_2d<u32>;

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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// FXAA is applied only to the molecular scene. UI and measurement labels are
// composited later, so their glyphs remain pixel-sharp.
fn antialiased_scene(uv: vec2<f32>, dimensions: vec2<f32>) -> vec4<f32> {
    let inverse_dimensions = 1.0 / dimensions;
    let northwest = textureSampleLevel(
        scene_color,
        scene_sampler,
        uv + vec2<f32>(-1.0, -1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let northeast = textureSampleLevel(
        scene_color,
        scene_sampler,
        uv + vec2<f32>(1.0, -1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let southwest = textureSampleLevel(
        scene_color,
        scene_sampler,
        uv + vec2<f32>(-1.0, 1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let southeast = textureSampleLevel(
        scene_color,
        scene_sampler,
        uv + vec2<f32>(1.0, 1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let center = textureSampleLevel(scene_color, scene_sampler, uv, 0.0);

    let luma_northwest = luminance(northwest);
    let luma_northeast = luminance(northeast);
    let luma_southwest = luminance(southwest);
    let luma_southeast = luminance(southeast);
    let luma_center = luminance(center.rgb);
    let luma_minimum = min(
        luma_center,
        min(min(luma_northwest, luma_northeast), min(luma_southwest, luma_southeast)),
    );
    let luma_maximum = max(
        luma_center,
        max(max(luma_northwest, luma_northeast), max(luma_southwest, luma_southeast)),
    );
    if luma_maximum - luma_minimum < max(0.025, luma_maximum * 0.08) {
        return center;
    }

    var direction = vec2<f32>(
        -((luma_northwest + luma_northeast) - (luma_southwest + luma_southeast)),
        (luma_northwest + luma_southwest) - (luma_northeast + luma_southeast),
    );
    let direction_reduce = max(
        (luma_northwest + luma_northeast + luma_southwest + luma_southeast) * 0.0078125,
        0.0009765625,
    );
    let reciprocal_minimum = 1.0 / (min(abs(direction.x), abs(direction.y)) + direction_reduce);
    direction = clamp(direction * reciprocal_minimum, vec2<f32>(-8.0), vec2<f32>(8.0))
        * inverse_dimensions;

    let first = 0.5 * (
        textureSampleLevel(scene_color, scene_sampler, uv + direction * (-1.0 / 6.0), 0.0).rgb
        + textureSampleLevel(scene_color, scene_sampler, uv + direction * (1.0 / 6.0), 0.0).rgb
    );
    let second = first * 0.5 + 0.25 * (
        textureSampleLevel(scene_color, scene_sampler, uv + direction * -0.5, 0.0).rgb
        + textureSampleLevel(scene_color, scene_sampler, uv + direction * 0.5, 0.0).rgb
    );
    let second_luminance = luminance(second);
    let resolved = select(second, first, second_luminance < luma_minimum || second_luminance > luma_maximum);
    return vec4<f32>(resolved, center.a);
}

fn semantic_at(pixel: vec2<i32>, dimensions: vec2<u32>) -> vec4<u32> {
    return textureLoad(
        semantic_ids,
        clamp(pixel, vec2<i32>(0), vec2<i32>(dimensions) - 1),
        0,
    );
}

fn toon_outline(pixel: vec2<i32>, dimensions: vec2<u32>) -> f32 {
    let center = semantic_at(pixel, dimensions);
    let dimensions_f = vec2<f32>(dimensions);
    let center_depth = textureLoad(scene_depth, pixel, 0);
    let center_uv = (vec2<f32>(pixel) + 0.5) / dimensions_f;
    var opacity = 0.0;
    for (var y: i32 = -2; y <= 2; y += 1) {
        for (var x: i32 = -2; x <= 2; x += 1) {
            if x == 0 && y == 0 {
                continue;
            }
            let neighbor = semantic_at(pixel + vec2<i32>(x, y), dimensions);
            var radius = 0.0;
            var edge_strength = 1.0;
            if center.w == 0u && neighbor.w != 0u {
                radius = 2.4;
            } else if center.w != 0u && neighbor.w == 0u {
                radius = 2.4;
            } else if center.w != 0u && neighbor.w != 0u {
                if center.z != neighbor.z {
                    radius = 2.25;
                } else if center.y != neighbor.y {
                    radius = 1.85;
                    let neighbor_pixel = clamp(
                        pixel + vec2<i32>(x, y),
                        vec2<i32>(0),
                        vec2<i32>(dimensions) - 1,
                    );
                    let neighbor_depth = textureLoad(scene_depth, neighbor_pixel, 0);
                    let neighbor_uv = (vec2<f32>(neighbor_pixel) + 0.5) / dimensions_f;
                    let depth_difference = abs(
                        optical_depth(center_uv, center_depth)
                        - optical_depth(neighbor_uv, neighbor_depth)
                    );
                    edge_strength = smoothstep(0.16, 0.72, depth_difference);
                } else if center.x != neighbor.x {
                    radius = 1.45;
                    let neighbor_pixel = clamp(
                        pixel + vec2<i32>(x, y),
                        vec2<i32>(0),
                        vec2<i32>(dimensions) - 1,
                    );
                    let neighbor_depth = textureLoad(scene_depth, neighbor_pixel, 0);
                    let neighbor_uv = (vec2<f32>(neighbor_pixel) + 0.5) / dimensions_f;
                    let depth_difference = abs(
                        optical_depth(center_uv, center_depth)
                        - optical_depth(neighbor_uv, neighbor_depth)
                    );
                    // Intersecting atomic spheres should read as one molecular mass.
                    // Ink appears only where one atom clearly occludes another.
                    edge_strength = smoothstep(0.22, 0.85, depth_difference);
                }
            }
            if radius > 0.0 {
                let sample_distance = length(vec2<f32>(f32(x), f32(y)));
                let sample_opacity = edge_strength
                    * (1.0 - smoothstep(radius * 0.72, radius, sample_distance));
                opacity = max(opacity, sample_opacity);
            }
        }
    }
    return opacity;
}

fn compose_toon(
    color: vec4<f32>,
    ao: f32,
    pixel: vec2<i32>,
    dimensions: vec2<u32>,
) -> vec4<f32> {
    let outline = toon_outline(pixel, dimensions);
    let ink = vec3<f32>(0.0086, 0.0103, 0.0123);
    return vec4<f32>(mix(color.rgb * ao, ink, outline), color.a);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(scene_color);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel = vec2<i32>(clamp(input.position.xy, vec2<f32>(0.0), dimensions - 1.0));
    let uv = input.position.xy / dimensions;
    let center = antialiased_scene(uv, dimensions);
    let center_depth = textureLoad(scene_depth, pixel, 0);
    let ao = textureLoad(filtered_ao, pixel, 0).r;
    if post.lens.w < 0.5 || post.lens.z <= 0.0 {
        return compose_toon(center, ao, pixel, dimensions_u);
    }

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
    let resolved = accumulated / accumulated_weight;
    return compose_toon(resolved, ao, pixel, dimensions_u);
}
