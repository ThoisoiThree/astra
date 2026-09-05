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
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
    @location(7) highlight: vec4<f32>,
    @location(8) semantic_ids: vec4<u32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) highlight: f32,
    @location(4) @interpolate(flat) semantic_ids: vec4<u32>,
};

struct CartoonVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) semantic_ids: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) highlight: vec4<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let selected_scale = 1.0 + input.highlight.y;
    let world = model * vec4<f32>(input.position * selected_scale, 1.0);
    var output: VertexOutput;
    output.clip_position = camera.view_projection * world;
    output.world_position = world.xyz;
    output.normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color;
    output.highlight = input.highlight.x;
    output.semantic_ids = input.semantic_ids;
    return output;
}

@vertex
fn cartoon_vertex_main(input: CartoonVertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.world_position = input.position;
    output.normal = normalize(input.normal);
    output.color = input.color;
    output.highlight = input.highlight.x;
    output.semantic_ids = input.semantic_ids;
    return output;
}

fn shaded_color(input: VertexOutput) -> vec4<f32> {
    let normal = normalize(input.normal);
    let light_direction = normalize(vec3<f32>(0.35, 0.65, 0.70));
    let diffuse = max(dot(normal, light_direction), 0.0);
    let view_direction = normalize(camera.eye_position.xyz - input.world_position);
    let half_vector = normalize(light_direction + view_direction);
    let specular = pow(max(dot(normal, half_vector), 0.0), 28.0) * 0.32;
    let highlight_color = vec3<f32>(1.0, 0.84, 0.08);
    let selected_color = mix(input.color.rgb, highlight_color, input.highlight * 0.90);
    let lit = selected_color * (0.28 + 0.72 * diffuse) + vec3<f32>(specular);
    let highlight_emission = highlight_color * input.highlight * 0.24;
    return vec4<f32>(lit + highlight_emission, input.color.a);
}

struct SceneFragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) semantic_ids: vec2<u32>,
};

@fragment
fn fragment_scene(input: VertexOutput) -> SceneFragmentOutput {
    var output: SceneFragmentOutput;
    output.color = shaded_color(input);
    output.semantic_ids = vec2<u32>(
        input.semantic_ids.x,
        (input.semantic_ids.z << 20u) | (input.semantic_ids.y & 0xFFFFFu),
    );
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return shaded_color(input);
}

@fragment
fn fragment_peel(input: VertexOutput) -> @location(0) vec4<f32> {
    reject_peeled_fragment(input.clip_position.xy, input.clip_position.z);
    return shaded_color(input);
}
