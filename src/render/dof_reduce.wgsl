// Section 6: merge the heads of four depth-sorted lists, then repeat for 4x4.
// Output occupies surviving source slots, so positions need no lossy encoding.
@group(0) @binding(7) var reduced_target: texture_storage_2d_array<rgba32uint, write>;
@group(0) @binding(8) var reduced_source: texture_2d_array<u32>;
@group(0) @binding(9) var block_counts: texture_2d<u32>;
@group(0) @binding(10) var block_sources: texture_2d_array<u32>;
@group(0) @binding(11) var block_counts_target: texture_storage_2d<r32uint, write>;
@group(0) @binding(12) var block_sources_target: texture_storage_2d_array<rgba32uint, write>;

fn store_block_sources(block: vec2<u32>, slot: u32, sources: vec4<u32>) {
    let texel = (slot % 16u) / 4u;
    let p = vec2<i32>(block * 2u + vec2<u32>(texel % 2u, texel / 2u));
    textureStore(block_sources_target, p, i32(slot / 16u), sources);
}

fn block_source(block: vec2<i32>, slot: u32) -> u32 {
    let texel = (slot % 16u) / 4u;
    let p = block * 2 + vec2<i32>(i32(texel % 2u), i32(texel / 2u));
    return textureLoad(block_sources, p, i32(slot / 16u), 0)[slot % 4u];
}

struct ReducedFragment {
    color: vec3<f32>,
    z: f32,
    coc: f32,
    mass: f32,
    source: u32,
};

// The paper specifies merge predicates qualitatively; these thresholds are
// implementation parameters kept in one place instead of being hidden magic values.
const MERGE_RELATIVE_DEPTH_THRESHOLD: f32 = 0.001;
const MERGE_COLOR_THRESHOLD: f32 = 0.01;
const MERGE_MIN_COC_PER_LINEAR_FOOTPRINT: f32 = 4.0;

const MAX_QUADRANT_FRAGMENTS: u32 = 4u * HOST_MAX_DOF_LAYERS;
const MAX_REDUCTION_FRAGMENTS: u32 = 4u * MAX_QUADRANT_FRAGMENTS;

fn reduced_less(a: ReducedFragment, b: ReducedFragment) -> bool {
    return a.z < b.z || (a.z == b.z && a.source < b.source);
}

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
        // Clear-color samples close exposed lists but have no physical occluder
        // footprint. Merging them would spread environment through nearby geometry.
        if heads[i].mass != mass || !valid_dof_depth(heads[i].z) { return result; }
        min_z = min(min_z, heads[i].z);
        max_z = max(max_z, heads[i].z);
        min_color = min(min_color, heads[i].color);
        max_color = max(max_color, heads[i].color);
        color += heads[i].color * 0.25;
        min_coc = min(min_coc, heads[i].coc);
    }
    // Implementation-specific conservative similarity/focus gates. The paper
    // specifies the criteria qualitatively but not these exact numeric thresholds.
    // Keep them isolated so they can be tuned or replaced without changing the
    // reduction topology.
    if max_z - min_z > max(MERGE_RELATIVE_DEPTH_THRESHOLD * min_z, 1e-4)
        || any(max_color - min_color > vec3<f32>(MERGE_COLOR_THRESHOLD))
        || min_coc < MERGE_MIN_COC_PER_LINEAR_FOOTPRINT * sqrt(mass)
        || (min_z < post.lens.x && max_z >= post.lens.x) { return result; }
    result.mass = 4.0 * mass;
    result.color = color;
    // Place behind all children, as in the authors' supplemental merge kernels.
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
    // Inset the square footprint by half a pixel (supplemental 2x2 heuristic).
    let footprint = (sqrt(f.mass) - 0.5) * f.z
        / (post.aperture.x * post.optical_axis.w * post.quality.y);
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
            // Splat traversal only reads active layers; newly enabled layers are
            // cleared on the next reduction before any of their records are read.
            for (var layer = 0u; layer < active_layer_count(); layer++) {
                textureStore(reduced_target, p, i32(layer), vec4<u32>(0u));
            }
        }
    }
    var quadrants: array<ReducedFragment, MAX_REDUCTION_FRAGMENTS>;
    var quadrant_counts = vec4<u32>(0u);
    for (var step = 0u; step < 5u; step++) {
        var input: array<ReducedFragment, MAX_REDUCTION_FRAGMENTS>;
        var counts = vec4<u32>(0u);
        var stride = HOST_MAX_DOF_LAYERS;
        var expected_mass = 1.0;
        if step < 4u {
            let base = origin + vec2<i32>(i32(step % 2u), i32(step / 2u)) * 2;
            for (var child = 0u; child < 4u; child++) {
                let p = base + vec2<i32>(i32(child % 2u), i32(child / 2u));
                if any(p >= size) { continue; }
                for (var layer = 0u; layer < active_layer_count(); layer++) {
                    if layer > 0u && textureLoad(mask_source, p, 0).r < 0.5 { break; }
                    let z = layer_depth(p, i32(layer));
                    // Depth-peeling lists are ordered; once the first missing layer is
                    // reached, all following layers are absent as well.
                    if !raw_layer_valid(p, layer, z) { break; }
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
            stride = MAX_QUADRANT_FRAGMENTS;
            expected_mass = 4.0;
        }
        var cursors = vec4<u32>(0u);
        var output_count = 0u;
        var source_ids = vec4<u32>(0u);
        loop {
            var heads: array<ReducedFragment, 4>;
            var nearest = 4u;
            for (var i = 0u; i < 4u; i++) {
                if cursors[i] >= counts[i] { continue; }
                heads[i] = input[i * stride + cursors[i]];
                if nearest == 4u { nearest = i; }
                else if reduced_less(heads[i], heads[nearest]) { nearest = i; }
            }
            if nearest == 4u { break; }
            var output = merge_candidate(heads, expected_mass);
            if output.mass > 0.0 {
                cursors += vec4<u32>(1u);
                if valid_dof_depth(output.z) {
                    let shadow_end = merged_umbra(output);
                    for (var i = 0u; i < 4u; i++) {
                        while cursors[i] < counts[i] {
                            let z = input[i * stride + cursors[i]].z;
                            if z <= output.z || z >= shadow_end { break; }
                            cursors[i]++;
                        }
                    }
                }
            } else {
                output = heads[nearest];
                cursors[nearest]++;
            }
            if step < 4u {
                // A representative merged depth can still interleave with later list
                // heads; keep each quadrant list sorted for the second traversal.
                var at = output_count;
                while at > 0u && reduced_less(output, quadrants[step * MAX_QUADRANT_FRAGMENTS + at - 1u]) {
                    quadrants[step * MAX_QUADRANT_FRAGMENTS + at] = quadrants[step * MAX_QUADRANT_FRAGMENTS + at - 1u];
                    at--;
                }
                quadrants[step * MAX_QUADRANT_FRAGMENTS + at] = output;
                output_count++;
            } else {
                emit_fragment(output);
                source_ids[output_count % 4u] = output.source;
                if output_count % 4u == 3u {
                    store_block_sources(id.xy, output_count, source_ids);
                }
                output_count++;
            }
        }
        if step < 4u { quadrant_counts[step] = output_count; }
        else {
            if output_count % 4u != 0u {
                store_block_sources(id.xy, output_count - 1u, source_ids);
            }
            // Slots beyond the count are never consumed; no full-list clear is needed.
            textureStore(block_counts_target, vec2<i32>(id.xy), vec4<u32>(output_count));
        }
    }
}
