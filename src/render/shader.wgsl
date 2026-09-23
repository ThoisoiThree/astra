// Triangle meshes: ribbons and molecular surfaces colored through their nearest atom, and
// instanced annotation geometry for measurements.

struct MeshInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) atom: u32,
};

struct MeshVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) atom: u32,
};

@vertex
fn vertex_mesh(input: MeshInput) -> MeshVarying {
    var output: MeshVarying;
    output.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.world_position = input.position;
    output.normal = input.normal;
    output.atom = input.atom;
    return output;
}

fn mesh_color(input: MeshVarying, front_facing: bool) -> vec4<f32> {
    let base = atom_colors[input.atom];
    // Ribbons are two-sided; flip the normal on back faces.
    let normal = select(-input.normal, input.normal, front_facing);
    return vec4<f32>(
        shade_surface(normal, input.world_position, base.rgb, atom_selected(input.atom)),
        base.a,
    );
}

struct MeshSceneOutput {
    @location(0) color: vec4<f32>,
    @location(1) semantic_ids: vec2<u32>,
};

@fragment
fn fragment_mesh_scene(
    input: MeshVarying,
    @builtin(front_facing) front_facing: bool,
) -> MeshSceneOutput {
    var output: MeshSceneOutput;
    output.color = mesh_color(input, front_facing);
    // Meshes are pickable by atom but draw no residue contours in Toon outlines.
    output.semantic_ids = vec2<u32>(atom_meta[input.atom].x, 0u);
    return output;
}

@fragment
fn fragment_mesh_color(
    input: MeshVarying,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    return mesh_color(input, front_facing);
}

@fragment
fn fragment_mesh_peel(
    input: MeshVarying,
    @builtin(front_facing) front_facing: bool,
) -> @location(0) vec4<f32> {
    reject_peeled_fragment(input.clip_position.xy, input.clip_position.z);
    return mesh_color(input, front_facing);
}

struct AnnotationInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct AnnotationVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vertex_annotation(input: AnnotationInput) -> AnnotationVarying {
    let model = mat4x4<f32>(input.model_0, input.model_1, input.model_2, input.model_3);
    let world = model * vec4<f32>(input.position, 1.0);
    var output: AnnotationVarying;
    output.clip_position = camera.view_projection * world;
    output.world_position = world.xyz;
    output.normal = normalize((model * vec4<f32>(input.normal, 0.0)).xyz);
    output.color = input.color;
    return output;
}

@fragment
fn fragment_annotation(input: AnnotationVarying) -> @location(0) vec4<f32> {
    return vec4<f32>(
        shade_surface(input.normal, input.world_position, input.color.rgb, 0.0),
        input.color.a,
    );
}
