// Section 6: merge the heads of four depth-sorted lists, then repeat for 4x4.
// Output occupies surviving source slots, so positions need no lossy encoding.
@group(0) @binding(7) var reduced_target: texture_storage_2d_array<rgba32uint, write>;
@group(0) @binding(8) var reduced_source: texture_2d_array<u32>;

struct ReducedFragment {
    color: vec3<f32>,
    z: f32,
    coc: f32,
    mass: f32,
    source: u32,
};

fn fragment_center(f: ReducedFragment) -> vec2<f32> {
    return vec2<f32>(fragment_pixel(f.source)) + 0.5 * sqrt(f.mass);
}

fn unpack_fragment(p: vec2<i32>, layer: i32) -> ReducedFragment {
    let packed = textureLoad(reduced_source, p, layer, 0);
    let rg = unpack2x16float(packed.x);
    let bm = unpack2x16float(packed.y);
    let size = vec2<u32>(post.quality.zw);
    return ReducedFragment(vec3<f32>(rg, bm.x), bitcast<f32>(packed.z),
        bitcast<f32>(packed.w), bm.y, (u32(layer) * size.y + u32(p.y)) * size.x + u32(p.x));
}

fn emit_fragment(f: ReducedFragment) {
    let p = fragment_pixel(f.source);
    let layer = i32(f.source / u32(post.quality.z * post.quality.w));
    textureStore(reduced_target, p, layer, vec4<u32>(
        pack2x16float(f.color.rg), pack2x16float(vec2<f32>(f.color.b, f.mass)),
        bitcast<u32>(f.z), bitcast<u32>(f.coc)));
}

fn merge_candidate(heads: array<ReducedFragment, 4>, mass: f32) -> ReducedFragment {
    var result = heads[0];
    result.mass = 0.0;
    var min_z = heads[0].z;
    var max_z = min_z;
    var min_color = heads[0].color;
    var max_color = min_color;
    var color = vec3<f32>(0.0);
    var min_coc = heads[0].coc;
    for (var i = 0u; i < 4u; i++) {
        if heads[i].mass != mass { return result; }
        min_z = min(min_z, heads[i].z);
        max_z = max(max_z, heads[i].z);
        min_color = min(min_color, heads[i].color);
        max_color = max(max_color, heads[i].color);
        color += heads[i].color * 0.25;
        min_coc = min(min_coc, heads[i].coc);
    }
    // The paper's similarity/focus gates expressed in relative depth and pixel
    // CoC instead of scene-unit constants. Require blur well beyond the footprint
    // and check every RGB channel, including equal-luminance chain colors.
    if max_z - min_z > max(0.01 * min_z, 1e-4)
        || any(max_color - min_color > vec3<f32>(0.02))
        || min_coc < 4.0 * sqrt(mass)
        || (min_z < post.lens.x && max_z >= post.lens.x) { return result; }
    result.mass = 4.0 * mass;
    result.color = color;
    // Preserve front-to-back ordering; never assign the nearest child's depth.
    result.z = max_z + max(1e-6 * max_z, 1e-5);
    let center = fragment_center(result);
    result.coc = 0.0;
    for (var i = 0u; i < 4u; i++) {
        result.coc = max(result.coc, heads[i].coc + aperture_metric(fragment_center(heads[i]) - center));
    }
    return result;
}

fn merged_umbra(f: ReducedFragment) -> f32 {
    let aperture = post.aperture.x / max(post.lens.y, 0.1);
    // Use an inset footprint as in the supplemental umbra heuristic (1.5
    // pixels for the 2x2 merge); larger merged fragments cast longer shadows.
    let width = sqrt(f.mass) - 0.5;
    let footprint = width * f.z / (post.aperture.x * post.optical_axis.w * post.quality.y);
    if footprint >= aperture { return 1e30; }
    return f.z * aperture / (aperture - footprint);
}

@compute @workgroup_size(8, 8)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let origin = vec2<i32>(id.xy * 4u);
    let size = vec2<i32>(post.quality.zw);
    if any(origin >= size) { return; }
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let p = origin + vec2<i32>(x,y);
            if any(p >= size) { continue; }
            for (var layer = 0; layer < 5; layer++) {
                textureStore(reduced_target, p, layer, vec4<u32>(0u));
            }
        }
    }
    var quadrants: array<ReducedFragment, 80>;
    var quadrant_counts = vec4<u32>(0u);
    for (var step = 0u; step < 5u; step++) {
        var input: array<ReducedFragment, 80>;
        var counts = vec4<u32>(0u);
        var stride = 5u;
        var expected_mass = 1.0;
        if step < 4u {
            let base = origin + vec2<i32>(i32(step % 2u), i32(step / 2u)) * 2;
            for (var child = 0u; child < 4u; child++) {
                let p = base + vec2<i32>(i32(child % 2u), i32(child / 2u));
                if any(p >= size) { continue; }
                for (var layer = 0u; layer < u32(post.quality.x); layer++) {
                    let z = layer_depth(p, i32(layer));
                    if layer > 0u && z >= 1e19 && (textureLoad(mask_source, p, 0).r < 0.5
                        || layer_depth(p, i32(layer) - 1) >= 1e19) { break; }
                    var color = textureLoad(layers_color, p, i32(layer), 0).rgb;
                    if layer == 0u { color = textureLoad(shaded_first, p, 0).rgb; }
                    let source = (layer * u32(size.y) + u32(p.y)) * u32(size.x) + u32(p.x);
                    input[child * stride + counts[child]] = ReducedFragment(color, z, radius(z), 1.0, source);
                    counts[child]++;
                }
            }
        } else {
            input = quadrants;
            counts = quadrant_counts;
            stride = 20u;
            expected_mass = 4.0;
        }
        var cursors = vec4<u32>(0u);
        var output_count = 0u;
        loop {
            var heads: array<ReducedFragment, 4>;
            var nearest = 4u;
            for (var i = 0u; i < 4u; i++) {
                if cursors[i] >= counts[i] { continue; }
                heads[i] = input[i * stride + cursors[i]];
                if nearest == 4u { nearest = i; }
                else if heads[i].z < heads[nearest].z
                    || (heads[i].z == heads[nearest].z && heads[i].source < heads[nearest].source) { nearest = i; }
            }
            if nearest == 4u { break; }
            var output = merge_candidate(heads, expected_mass);
            if output.mass > 0.0 {
                cursors += vec4<u32>(1u);
                let shadow_end = merged_umbra(output);
                for (var i = 0u; i < 4u; i++) {
                    while cursors[i] < counts[i] {
                        let z = input[i * stride + cursors[i]].z;
                        if z <= output.z || z >= shadow_end { break; }
                        cursors[i]++;
                    }
                }
            } else {
                output = heads[nearest];
                cursors[nearest]++;
            }
            if step < 4u {
                // max-depth placement can move a merge past a later list head.
                // Keep the lists sorted for the second four-way traversal.
                var at = output_count;
                while at > 0u && quadrants[step * 20u + at - 1u].z > output.z {
                    quadrants[step * 20u + at] = quadrants[step * 20u + at - 1u];
                    at--;
                }
                quadrants[step * 20u + at] = output;
                output_count++;
            } else { emit_fragment(output); }
        }
        if step < 4u { quadrant_counts[step] = output_count; }
    }
}
