struct Camera {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) center_radius: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) semantic_ids: vec4<u32>,
    @location(3) highlight: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) billboard_position: vec3<f32>,
    @location(1) center_radius: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) @interpolate(flat) semantic_ids: vec4<u32>,
    @location(4) highlight: f32,
};

@vertex
fn vertex_main(input: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[vertex_index];
    let extent = input.center_radius.w * 1.08;
    let world = input.center_radius.xyz
        + (camera.camera_right.xyz * corner.x + camera.camera_up.xyz * corner.y) * extent;
    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(world, 1.0);
    output.billboard_position = world;
    output.center_radius = input.center_radius;
    output.color = input.color;
    output.semantic_ids = input.semantic_ids;
    output.highlight = input.highlight.x;
    return output;
}

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) semantic_ids: vec4<u32>,
};

@fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput {
    let ray_direction = normalize(input.billboard_position - camera.eye_position.xyz);
    let eye_to_center = camera.eye_position.xyz - input.center_radius.xyz;
    let projected = dot(eye_to_center, ray_direction);
    let discriminant = projected * projected
        - (dot(eye_to_center, eye_to_center) - input.center_radius.w * input.center_radius.w);
    if discriminant < 0.0 {
        discard;
    }
    let distance = -projected - sqrt(discriminant);
    if distance <= 0.0 {
        discard;
    }

    let world_position = camera.eye_position.xyz + ray_direction * distance;
    let normal = normalize(world_position - input.center_radius.xyz);
    let clip = camera.view_projection * vec4<f32>(world_position, 1.0);

    // Broad, quantized diffuse bands reproduce the flat pastel illustration
    // style while AO and semantic contours carry most of the shape information.
    let light_direction = normalize(vec3<f32>(0.30, 0.55, 0.78));
    let diffuse = dot(normal, light_direction);
    let broad_light = smoothstep(-0.35, 0.72, diffuse);
    let light_band = mix(0.82, 1.0, smoothstep(0.20, 0.82, broad_light));
    let pastel = mix(input.color.rgb, vec3<f32>(1.0), 0.12);
    let highlight_color = vec3<f32>(1.0, 0.87, 0.12);
    let selected = mix(pastel, highlight_color, input.highlight * 0.88);

    var output: FragmentOutput;
    output.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    output.color = vec4<f32>(selected * light_band, input.color.a);
    output.semantic_ids = input.semantic_ids;
    return output;
}
