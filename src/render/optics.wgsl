struct PostUniform {
    inverse_view_projection: mat4x4<f32>,
    eye_position: vec4<f32>,
    // Optical axis (xyz), molecular viewport height in full-resolution pixels (w).
    optical_axis: vec4<f32>,
    // focus distance, f-number, maximum CoC radius in pixels, enabled
    lens: vec4<f32>,
    // focal length for a unit-height sensor, diaphragm blades, rotation, Toon contours enabled
    aperture: vec4<f32>,
    // AO strength, world-space radius, normal bias, sample count (zero disables)
    ao: vec4<f32>,
    // layer count, actual vertical resolution scale, target width, target height
    quality: vec4<f32>,
    background: vec4<f32>,
    // Normalized molecular viewport origin and extent within the window.
    viewport: vec4<f32>,
};

fn optical_depth(uv: vec2<f32>, depth: f32) -> f32 {
    if depth >= 0.999999 {
        return 1e20;
    }
    let local_uv = (uv - post.viewport.xy) / post.viewport.zw;
    let clip = vec4<f32>(local_uv.x * 2.0 - 1.0, 1.0 - local_uv.y * 2.0, depth, 1.0);
    let world_h = post.inverse_view_projection * clip;
    let world = world_h.xyz / world_h.w;
    return max(dot(world - post.eye_position.xyz, post.optical_axis.xyz), 1e-4);
}

// Signed thin-lens circle of confusion. Scene and focal length use the same
// virtual unit-height sensor scale, so the result is a sensor-height fraction.
fn circle_of_confusion(distance: f32, image_height: f32) -> f32 {
    let focus = max(post.lens.x, post.aperture.x + 1e-4);
    let object_distance = max(distance, post.aperture.x + 1e-4);
    let focal_length = post.aperture.x;
    let f_number = max(post.lens.y, 0.1);
    // The thin-lens equation gives a diameter; splatting uses its radius.
    let sensor_coc = 0.5 * focal_length * focal_length * (1.0 - focus / object_distance)
        / (f_number * (focus - focal_length));
    return clamp(sensor_coc * image_height, -post.lens.z, post.lens.z);
}

fn aperture_boundary(angle: f32) -> f32 {
    let blades = floor(post.aperture.y + 0.5);
    if blades < 3.0 {
        return 1.0;
    }
    let sector = 6.28318530718 / blades;
    let local_angle = angle - sector * floor(angle / sector + 0.5);
    return cos(3.14159265359 / blades) / max(cos(local_angle), 1e-4);
}

fn aperture_metric(point: vec2<f32>) -> f32 {
    let angle = atan2(point.y, point.x) - post.aperture.z;
    return length(point) / max(aperture_boundary(angle), 1e-4);
}

