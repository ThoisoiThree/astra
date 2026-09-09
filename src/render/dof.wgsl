// Franke et al. (2018), sections 4–6. All coordinates below are DOF target pixels.
@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var layers_color: texture_2d_array<f32>;
@group(0) @binding(2) var layers_depth: texture_depth_2d_array;
@group(0) @binding(3) var mask_source: texture_2d<f32>;
@group(0) @binding(4) var mask_target: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var result: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var shaded_first: texture_2d<f32>;

// Injected by dof.rs so Rust and WGSL share the same bounded layer count.
// HOST_MAX_DOF_LAYERS is declared before this source at shader-module creation.
const INVALID_DOF_DEPTH: f32 = 1e19;

fn active_layer_count() -> u32 {
    return min(max(u32(max(post.quality.x, 1.0)), 1u), HOST_MAX_DOF_LAYERS);
}

fn valid_dof_depth(z: f32) -> bool {
    // optical_depth is expected to be positive and monotonic with camera distance.
    // NaN also fails these ordered comparisons.
    return z > 0.0 && z < INVALID_DOF_DEPTH;
}

fn layer_depth(p: vec2<i32>, layer: i32) -> f32 {
    let pixel = clamp(p, vec2<i32>(0), vec2<i32>(post.quality.zw) - 1);
    return optical_depth((vec2<f32>(pixel) + 0.5) / post.quality.zw,
        textureLoad(layers_depth, pixel, layer, 0));
}

fn radius(z: f32) -> f32 {
    return abs(circle_of_confusion(z, post.optical_axis.w)) * post.quality.y;
}

@compute @workgroup_size(8, 8)
fn edges(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(post.quality.zw)) { return; }
    let p = vec2<i32>(id.xy);
    let z = layer_depth(p, 0);
    let z_valid = valid_dof_depth(z);
    var edge_radius = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 { continue; }
            let neighbor = layer_depth(p + vec2<i32>(x, y), 0);
            let neighbor_valid = valid_dof_depth(neighbor);
            let silhouette = z_valid != neighbor_valid;
            let depth_edge = z_valid && neighbor_valid
                && abs(z - neighbor) > max(0.02 * min(z, neighbor), 0.01);
            if silhouette || depth_edge {
                // Partial occlusion is caused by the front-most surface at a depth
                // discontinuity. Do not dilate by a far-background CoC.
                var foreground_radius = 0.0;
                if z_valid && neighbor_valid {
                    foreground_radius = radius(min(z, neighbor));
                } else if z_valid {
                    foreground_radius = radius(z);
                } else if neighbor_valid {
                    foreground_radius = radius(neighbor);
                }
                if foreground_radius > 0.5 {
                    edge_radius = max(edge_radius, ceil(foreground_radius) + 1.0);
                }
            }
        }
    }
    textureStore(mask_target, p, vec4<f32>(edge_radius));
}

@compute @workgroup_size(8, 8)
fn dilate_x(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(post.quality.zw)) { return; }
    let p = vec2<i32>(id.xy);
    let extent = i32(ceil(post.lens.z * post.quality.y)) + 1;
    var vertical_reach = 0.0;
    for (var x = -extent; x <= extent; x++) {
        let q = clamp(p + vec2<i32>(x, 0), vec2<i32>(0), vec2<i32>(post.quality.zw) - 1);
        let seed_radius = textureLoad(mask_source, q, 0).r;
        let dx = f32(abs(x));
        if seed_radius > 0.0 && dx <= seed_radius {
            // Store the remaining vertical radius of a circular CoC. The following
            // Y pass therefore produces a circle rather than a conservative square.
            vertical_reach = max(vertical_reach,
                sqrt(max(seed_radius * seed_radius - dx * dx, 0.0)));
        }
    }
    textureStore(mask_target, p, vec4<f32>(vertical_reach));
}

@compute @workgroup_size(8, 8)
fn dilate_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(post.quality.zw)) { return; }
    let p = vec2<i32>(id.xy);
    let extent = i32(ceil(post.lens.z * post.quality.y)) + 1;
    var marked = 0.0;
    for (var y = -extent; y <= extent; y++) {
        let q = clamp(p + vec2<i32>(0, y), vec2<i32>(0), vec2<i32>(post.quality.zw) - 1);
        let remaining_radius = textureLoad(mask_source, q, 0).r;
        if remaining_radius > 0.0 && f32(abs(y)) <= remaining_radius {
            marked = 1.0;
            break;
        }
    }
    textureStore(mask_target, p, vec4<f32>(marked));
}

struct Fragment {
    color: vec4<f32>,
    // Eye-space depth, conservative CoC radius, represented source-pixel count,
    // and opacity cached for all output pixels sharing this fragment.
    shape: vec4<f32>,
};

// Supplemental material, precompute_for_blur and Apply.
const DOF_SATURATION_TRANSMISSION: f32 = 0.01;

fn fragment_coverage(offset: vec2<f32>, coc_radius: f32) -> f32 {
    if coc_radius <= 0.5 {
        return select(0.0, 1.0, all(abs(offset) < vec2<f32>(0.5)));
    }
    return 1.0 - smoothstep(max(coc_radius - 0.5, 0.0), coc_radius + 0.5,
        aperture_metric(offset));
}

fn fragment_alpha(coc_radius: f32, mass: f32, coverage: f32) -> f32 {
    if coverage <= 0.0 { return 0.0; }
    if coc_radius <= 0.5 { return coverage; }

    return clamp(fragment_opacity(coc_radius, mass) * coverage, 0.0, 1.0);
}

fn fragment_opacity(coc_radius: f32, mass: f32) -> f32 {
    if coc_radius <= 0.5 { return 1.0; }
    let single_alpha = min(1.0, 1.0 / (coc_radius * coc_radius));
    return 1.0 - pow(1.0 - single_alpha, max(mass, 1.0));
}

fn raw_layer_valid(p: vec2<i32>, layer: u32, z: f32) -> bool {
    if layer >= active_layer_count() { return false; }
    if layer > 0u && textureLoad(mask_source, p, 0).r < 0.5 { return false; }
    // One environment fragment closes each exposed list. Further empty layers
    // must not count as additional radiance samples.
    if valid_dof_depth(z) { return true; }
    return layer == 0u || valid_dof_depth(layer_depth(p, i32(layer) - 1));
}

fn raw_layer_color(p: vec2<i32>, layer: i32) -> vec4<f32> {
    if layer == 0 { return textureLoad(shaded_first, p, 0); }
    return textureLoad(layers_color, p, layer, 0);
}

// A tile's list lives in workgroup memory. Overfull lists are partitioned by the
// exact (32-bit depth, source ID) radix key and traversed in front-to-back order.
// This bounds memory without dropping splats, quantizing depth, or capping CoC.
const CAPACITY: u32 = 1024u;
var<workgroup> keys: array<vec2<u32>, 1024>;
var<workgroup> count: atomic<u32>;
var<workgroup> key_and: array<atomic<u32>, 2>;
var<workgroup> key_or: array<atomic<u32>, 2>;
var<workgroup> prefix: vec2<u32>;
var<workgroup> prefix_bits: u32;
var<workgroup> stack: array<vec2<u32>, 64>;
var<workgroup> stack_bits: array<u32, 64>;
var<workgroup> stack_size: u32;
var<workgroup> done: u32;
var<workgroup> unsaturated: atomic<u32>;
var<workgroup> chunk: array<Fragment, 32>;
var<workgroup> centers: array<vec2<f32>, 32>;
fn less(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

fn prefix_matches(key: vec2<u32>) -> bool {
    if prefix_bits == 0u { return true; }
    if prefix_bits <= 32u {
        return (key.x >> (32u - prefix_bits)) == (prefix.x >> (32u - prefix_bits));
    }
    return key.x == prefix.x && (key.y >> (64u - prefix_bits)) == (prefix.y >> (64u - prefix_bits));
}

fn fragment_pixel(index: u32) -> vec2<i32> {
    let size = vec2<u32>(post.quality.zw);
    return vec2<i32>(i32(index % size.x), i32((index / size.x) % size.y));
}

fn append_tile_candidate(z: f32, coc: f32, center: vec2<f32>, source_id: u32,
    tile_min: vec2<f32>, tile_max: vec2<f32>) {
    let key = vec2<u32>(bitcast<u32>(z), source_id);
    if !prefix_matches(key) { return; }
    let delta = center - clamp(center, tile_min, tile_max);
    if length(delta) > max(coc, 0.5) + 0.5 { return; }
    atomicAnd(&key_and[0], key.x);
    atomicAnd(&key_and[1], key.y);
    atomicOr(&key_or[0], key.x);
    atomicOr(&key_or[1], key.y);
    let slot = atomicAdd(&count, 1u);
    if slot < CAPACITY { keys[slot] = key; }
}

@compute @workgroup_size(16, 16)
fn splat(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) id: vec3<u32>) {
    accumulate_tile(tile, lane, id, true, true);
}

@compute @workgroup_size(16, 16)
fn splat_reference(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) id: vec3<u32>) {
    accumulate_tile(tile, lane, id, false, false);
}

// Validation baseline for compaction: identical reduced fragments, dense traversal.
@compute @workgroup_size(16, 16)
fn splat_dense(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) id: vec3<u32>) {
    accumulate_tile(tile, lane, id, true, false);
}

// Independent ordering oracle for small validation images. It performs no tile
// list construction, radix partitioning, bitonic sorting, or fragment reduction.
// Instead, every output pixel repeatedly scans all potentially contributing raw
// fragments and selects the next exact (depth, source-id) key. This is deliberately
// O(N^2) and must only be dispatched by the guarded Rust debug path.
@compute @workgroup_size(8, 8)
fn splat_oracle(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(post.quality.zw);
    if any(id.xy >= vec2<u32>(size)) { return; }

    let pixel = vec2<f32>(id.xy) + 0.5;
    let margin = i32(ceil(post.lens.z * post.quality.y)) + 1;
    let begin = max(vec2<i32>(id.xy) - margin, vec2<i32>(0));
    let end = min(vec2<i32>(id.xy) + margin + 1, size);

    var rgb = vec3<f32>(0.0);
    var transmission = 1.0;
    var last_key = vec2<u32>(0u);
    var have_last = false;

    loop {
        var found = false;
        var best_key = vec2<u32>(0xffffffffu);
        var best_p = vec2<i32>(0);
        var best_layer = 0i;
        var best_z = 0.0;
        var best_radius = 0.0;

        for (var y = begin.y; y < end.y; y++) {
            for (var x = begin.x; x < end.x; x++) {
                let p = vec2<i32>(x, y);
                for (var layer = 0u; layer < active_layer_count(); layer++) {
                    if layer > 0u && textureLoad(mask_source, p, 0).r < 0.5 { break; }
                    let z = layer_depth(p, i32(layer));
                    // The depth-peeling list is ordered, so later entries are absent too.
                    if !raw_layer_valid(p, layer, z) { break; }
                    let coc_radius = radius(z);
                    let coverage = fragment_coverage(pixel - (vec2<f32>(p) + 0.5), coc_radius);
                    if coverage <= 0.0 { continue; }
                    let source_id = (layer * u32(size.y) + u32(p.y)) * u32(size.x) + u32(p.x);
                    let key = vec2<u32>(bitcast<u32>(z), source_id);
                    if have_last && !less(last_key, key) { continue; }
                    if !found || less(key, best_key) {
                        found = true;
                        best_key = key;
                        best_p = p;
                        best_layer = i32(layer);
                        best_z = z;
                        best_radius = coc_radius;
                    }
                }
            }
        }

        if !found { break; }
        let coverage = fragment_coverage(pixel - (vec2<f32>(best_p) + 0.5), best_radius);
        let alpha = fragment_alpha(best_radius, 1.0, coverage);
        let weight = transmission * alpha;
        rgb += weight * raw_layer_color(best_p, best_layer).rgb;
        transmission -= weight;
        last_key = best_key;
        have_last = true;
        if transmission <= DOF_SATURATION_TRANSMISSION { break; }
    }

    let resolved = resolve_splats(rgb, transmission);
    textureStore(result, vec2<i32>(id.xy), vec4<f32>(resolved, 1.0));
}

fn accumulate_tile(tile: vec3<u32>, lane: u32, id: vec3<u32>, reduced: bool, compact: bool) {
    let size = vec2<i32>(post.quality.zw);
    let valid_pixel = all(id.xy < vec2<u32>(size));
    let pixel = vec2<f32>(id.xy) + 0.5;
    let tile_min = vec2<f32>(tile.xy * 16u) + 0.5;
    let tile_max = min(tile_min + 15.0, vec2<f32>(size) - 0.5);
    // Include centroid displacement and union support of a 4x4 merge, including
    // the triangular iris's largest directional extent.
    let margin = i32(ceil(post.lens.z * post.quality.y)) + select(1, 8, reduced);
    let begin = max(vec2<i32>(tile.xy * 16u) - margin, vec2<i32>(0));
    let end = min(vec2<i32>(tile.xy * 16u + 16u) + margin, size);
    let extent = end - begin;
    let area = u32(extent.x * extent.y);
    var rgb = vec3<f32>(0.0);
    var transmission = 1.0;
    if lane == 0u {
        prefix = vec2<u32>(0u);
        prefix_bits = 0u;
        stack_size = 0u;
        done = 0u;
    }
    workgroupBarrier();
    loop {
        if lane == 0u {
            atomicStore(&count, 0u);
            atomicStore(&key_and[0], 0xffffffffu);
            atomicStore(&key_and[1], 0xffffffffu);
            atomicStore(&key_or[0], 0u);
            atomicStore(&key_or[1], 0u);
        }
        for (var i = lane; i < CAPACITY; i += 256u) { keys[i] = vec2<u32>(0xffffffffu); }
        workgroupBarrier();
        if compact {
            let first_block = begin / 4;
            let block_extent = (end + 3) / 4 - first_block;
            let block_area = u32(block_extent.x * block_extent.y);
            // Eight lanes traverse each compact 4x4 source block. Unlike the
            // texture scan, this never visits holes left by merging/umbra culling.
            for (var b = lane / 8u; b < block_area; b += 32u) {
                let block = first_block + vec2<i32>(i32(b % u32(block_extent.x)), i32(b / u32(block_extent.x)));
                let surviving = textureLoad(block_counts, block, 0).r;
                for (var slot = lane % 8u; slot < surviving; slot += 8u) {
                    let source_id = block_source(block, slot);
                    let p = fragment_pixel(source_id);
                    // Boundary blocks straddle the old candidate rectangle;
                    // keep exactly its source set before the usual circle test.
                    if any(p < begin) || any(p >= end) { continue; }
                    let layer = i32(source_id / u32(size.x * size.y));
                    let fragment = unpack_fragment(p, layer);
                    append_tile_candidate(fragment.z, fragment.coc, fragment_center(fragment),
                        source_id, tile_min, tile_max);
                }
            }
        } else {
            for (var i = lane; i < area * active_layer_count(); i += 256u) {
                let layer = i / area;
                let offset = i % area;
                let p = begin + vec2<i32>(i32(offset % u32(extent.x)), i32(offset / u32(extent.x)));
                let source_id = (layer * u32(size.y) + u32(p.y)) * u32(size.x) + u32(p.x);
                if reduced {
                    let fragment = unpack_fragment(p, i32(layer));
                    if fragment.mass == 0.0 { continue; }
                    append_tile_candidate(fragment.z, fragment.coc, fragment_center(fragment),
                        source_id, tile_min, tile_max);
                } else {
                    let z = layer_depth(p, i32(layer));
                    if !raw_layer_valid(p, layer, z) { continue; }
                    append_tile_candidate(z, radius(z), vec2<f32>(p) + 0.5, source_id, tile_min, tile_max);
                }
            }
        }
        workgroupBarrier();
        let length = atomicLoad(&count);
        if length > CAPACITY {
            if lane == 0u {
                prefix = vec2<u32>(atomicLoad(&key_and[0]), atomicLoad(&key_and[1]));
                let differences = vec2<u32>(atomicLoad(&key_or[0]), atomicLoad(&key_or[1])) ^ prefix;
                // Skip common radix bits, especially for planar or clear-color layers.
                var far = prefix;
                if differences.x != 0u {
                    prefix_bits = countLeadingZeros(differences.x) + 1u;
                    far.x |= 1u << (32u - prefix_bits);
                } else {
                    prefix_bits = countLeadingZeros(differences.y) + 33u;
                    far.y |= 1u << (64u - prefix_bits);
                }
                stack[stack_size] = far;
                stack_bits[stack_size] = prefix_bits;
                stack_size++;
            }
            workgroupBarrier();
            continue;
        }
        var sort_length = 1u;
        while sort_length < length { sort_length *= 2u; }
        for (var width = 2u; width <= sort_length; width *= 2u) {
            for (var stride = width / 2u; stride > 0u; stride /= 2u) {
                for (var i = lane; i < sort_length; i += 256u) {
                    let other = i ^ stride;
                    if other > i {
                        let a = keys[i];
                        let b = keys[other];
                        if select(less(a, b), less(b, a), (i & width) == 0u) {
                            keys[i] = b;
                            keys[other] = a;
                        }
                    }
                }
                workgroupBarrier();
            }
        }
        for (var base = 0u; base < length; base += 32u) {
            if lane < 32u && base + lane < length {
                let index = keys[base + lane].y;
                let p = fragment_pixel(index);
                let layer = i32(index / u32(size.x * size.y));
                if reduced {
                    let fragment = unpack_fragment(p, layer);
                    chunk[lane].color = vec4<f32>(fragment.color, 1.0);
                    chunk[lane].shape = vec4<f32>(fragment.z, fragment.coc, fragment.mass, 0.0);
                    centers[lane] = fragment_center(fragment);
                } else {
                    let z = layer_depth(p, layer);
                    chunk[lane].color = raw_layer_color(p, layer);
                    chunk[lane].shape = vec4<f32>(z, radius(z), 1.0, 0.0);
                    centers[lane] = vec2<f32>(p) + 0.5;
                }
                // Radius and represented mass are identical for all 256 output
                // pixels. Evaluate the power once per loaded fragment, not per pixel.
                if compact {
                    chunk[lane].shape.w = fragment_opacity(chunk[lane].shape.y, chunk[lane].shape.z);
                }
            }
            workgroupBarrier();
            if valid_pixel && transmission > DOF_SATURATION_TRANSMISSION {
                for (var i = 0u; i < min(32u, length - base); i++) {
                    let f = chunk[i];
                    let offset = pixel - centers[i];
                    let coverage = fragment_coverage(offset, f.shape.y);
                    var alpha = clamp(f.shape.w * coverage, 0.0, 1.0);
                    if !compact {
                        // Keep the old per-pixel expression in the validation
                        // paths so the comparison also covers opacity hoisting.
                        alpha = fragment_alpha(f.shape.y, f.shape.z, coverage);
                    }
                    let weight = transmission * alpha;
                    rgb += weight * f.color.rgb;
                    transmission -= weight;
                }
            }
            workgroupBarrier();
        }
        if lane == 0u { atomicStore(&unsaturated, 0u); }
        workgroupBarrier();
        if valid_pixel && transmission > DOF_SATURATION_TRANSMISSION { atomicAdd(&unsaturated, 1u); }
        workgroupBarrier();
        if lane == 0u {
            if stack_size == 0u || atomicLoad(&unsaturated) == 0u { done = 1u; }
            else {
                stack_size--;
                prefix = stack[stack_size];
                prefix_bits = stack_bits[stack_size];
            }
        }
        workgroupBarrier();
        if done != 0u { break; }
    }
    if valid_pixel {
        let resolved = resolve_splats(rgb, transmission);
        textureStore(result, vec2<i32>(id.xy), vec4<f32>(resolved, 1.0));
    }
}

fn resolve_splats(rgb: vec3<f32>, transmission: f32) -> vec3<f32> {
    let opacity = 1.0 - transmission;
    return select(post.background.rgb, rgb / max(opacity, 1e-6), opacity > 1e-6);
}
