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
@group(0) @binding(6)
var dof_color: texture_2d<f32>;

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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// FXAA is applied only to the molecular scene. UI and measurement labels are
// composited later, so their glyphs remain pixel-sharp.
fn antialiased_scene(source: texture_2d<f32>, uv: vec2<f32>, dimensions: vec2<f32>) -> vec4<f32> {
    let inverse_dimensions = 1.0 / dimensions;
    let northwest = textureSampleLevel(
        source,
        scene_sampler,
        uv + vec2<f32>(-1.0, -1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let northeast = textureSampleLevel(
        source,
        scene_sampler,
        uv + vec2<f32>(1.0, -1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let southwest = textureSampleLevel(
        source,
        scene_sampler,
        uv + vec2<f32>(-1.0, 1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let southeast = textureSampleLevel(
        source,
        scene_sampler,
        uv + vec2<f32>(1.0, 1.0) * inverse_dimensions,
        0.0,
    ).rgb;
    let center = textureSampleLevel(source, scene_sampler, uv, 0.0);

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
        textureSampleLevel(source, scene_sampler, uv + direction * (-1.0 / 6.0), 0.0).rgb
        + textureSampleLevel(source, scene_sampler, uv + direction * (1.0 / 6.0), 0.0).rgb
    );
    let second = first * 0.5 + 0.25 * (
        textureSampleLevel(source, scene_sampler, uv + direction * -0.5, 0.0).rgb
        + textureSampleLevel(source, scene_sampler, uv + direction * 0.5, 0.0).rgb
    );
    let second_luminance = luminance(second);
    let resolved = select(second, first, second_luminance < luma_minimum || second_luminance > luma_maximum);
    return vec4<f32>(resolved, center.a);
}

fn semantic_at(pixel: vec2<i32>, dimensions: vec2<u32>) -> vec2<u32> {
    return textureLoad(
        semantic_ids,
        clamp(pixel, vec2<i32>(0), vec2<i32>(dimensions) - 1),
        0,
    ).xy;
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
            let center_present = center.x != 0u;
            let neighbor_present = neighbor.x != 0u;
            let center_chain = center.y >> 20u;
            let neighbor_chain = neighbor.y >> 20u;
            let center_residue = center.y & 0xFFFFFu;
            let neighbor_residue = neighbor.y & 0xFFFFFu;
            if !center_present && neighbor_present {
                radius = 2.4;
            } else if center_present && !neighbor_present {
                radius = 2.4;
            } else if center_present && neighbor_present {
                if center_chain != neighbor_chain {
                    radius = 2.25;
                } else if center_residue != neighbor_residue {
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
    if post.aperture.w < 0.5 {
        return vec4<f32>(color.rgb * ao, color.a);
    }
    let outline = toon_outline(pixel, dimensions);
    let ink = vec3<f32>(0.0086, 0.0103, 0.0123);
    return vec4<f32>(mix(color.rgb * ao, ink, outline), color.a);
}

@fragment
fn dof_shade(input: VertexOutput) -> @location(0) vec4<f32> {
    // Color comes from the exact rasterized DOF pixel, not a filtered pinhole
    // image. Only screen-space shading is mapped from the visible scene.
    let uv = input.position.xy / post.quality.zw;
    let dimensions = textureDimensions(semantic_ids);
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), vec2<i32>(dimensions) - 1);
    let color = textureLoad(scene_color, vec2<i32>(input.position.xy), 0);
    let ao = select(1.0, textureLoad(filtered_ao, pixel, 0).r, post.ao.w > 0.5);
    return compose_toon(color, ao, pixel, dimensions);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions_u = textureDimensions(scene_color);
    let dimensions = vec2<f32>(dimensions_u);
    let pixel = vec2<i32>(clamp(input.position.xy, vec2<f32>(0.0), dimensions - 1.0));
    let uv = input.position.xy / dimensions;
    if post.lens.w >= 0.5 && post.lens.z > 0.0 {
        // Rasterized in-focus splats still need edge antialiasing. Apply it to
        // the accumulated image, so color filtering cannot corrupt peel depths.
        return antialiased_scene(dof_color, uv, vec2<f32>(textureDimensions(dof_color)));
    }
    let resolved = antialiased_scene(scene_color, uv, dimensions);
    let ao = select(1.0, textureLoad(filtered_ao, pixel, 0).r, post.ao.w > 0.5);
    return compose_toon(resolved, ao, pixel, dimensions_u);
}
