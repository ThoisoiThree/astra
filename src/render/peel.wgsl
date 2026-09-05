@group(1) @binding(0) var previous_depth: texture_depth_2d;
@group(1) @binding(1) var disocclusion: texture_2d<f32>;
@group(1) @binding(2) var<uniform> post: PostUniform;

// Franke et al., section 4.1: the previous pixel acts as an occluder
// of the finite lens. Its umbra ends at z * aperture / (aperture - footprint).
fn reject_peeled_fragment(position: vec2<f32>, depth: f32) {
    let pixel = vec2<i32>(position);
    if textureLoad(disocclusion, pixel, 0).r < 0.5 {
        discard;
    }
    let previous = textureLoad(previous_depth, pixel, 0);
    if previous >= 0.999999 || depth <= previous + 1e-7 {
        discard;
    }
    let uv = position / post.quality.zw;
    let z = optical_depth(uv, previous);
    let aperture = post.aperture.x / max(post.lens.y, 0.1);
    let footprint = z / (post.aperture.x * post.optical_axis.w * post.quality.y);
    if footprint >= aperture {
        discard;
    }
    if optical_depth(uv, depth) < z * aperture / (aperture - footprint) {
        discard;
    }
}
