// Franke et al. (2018), sections 4–6. All coordinates below are DOF target pixels.
@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var layers_color: texture_2d_array<f32>;
@group(0) @binding(2) var layers_depth: texture_depth_2d_array;
@group(0) @binding(3) var mask_source: texture_2d<f32>;
@group(0) @binding(4) var mask_target: texture_storage_2d<r32float, write>;
@group(0) @binding(5) var result: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var shaded_first: texture_2d<f32>;

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
    var edge_radius = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = layer_depth(p + vec2<i32>(x, y), 0);
            // Relative eye-space threshold works across molecule sizes and zoom levels.
            if abs(z - neighbor) > max(0.02 * min(z, neighbor), 0.01) {
                edge_radius = max(edge_radius, ceil(max(radius(z), radius(neighbor))) + 1.0);
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
    var r = 0.0;
    for (var x = -extent; x <= extent; x++) {
        let q = clamp(p + vec2<i32>(x, 0), vec2<i32>(0), vec2<i32>(post.quality.zw) - 1);
        let sample_radius = textureLoad(mask_source, q, 0).r;
        if f32(abs(x)) <= sample_radius { r = max(r, sample_radius); }
    }
    textureStore(mask_target, p, vec4<f32>(r));
}

@compute @workgroup_size(8, 8)
fn dilate_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= vec2<u32>(post.quality.zw)) { return; }
    let p = vec2<i32>(id.xy);
    let extent = i32(ceil(post.lens.z * post.quality.y)) + 1;
    var marked = 0.0;
    for (var y = -extent; y <= extent; y++) {
        let q = clamp(p + vec2<i32>(0, y), vec2<i32>(0), vec2<i32>(post.quality.zw) - 1);
        let r = textureLoad(mask_source, q, 0).r;
        if r > 0.0 && f32(abs(y)) <= r { marked = 1.0; }
    }
    textureStore(mask_target, p, vec4<f32>(marked));
}

struct Fragment {
    color: vec4<f32>,
    // Eye-space depth, conservative CoC radius and represented source-pixel count.
    shape: vec4<f32>,
};

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

@compute @workgroup_size(16, 16)
fn splat(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) id: vec3<u32>) {
    accumulate_tile(tile, lane, id, true);
}

@compute @workgroup_size(16, 16)
fn splat_reference(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) id: vec3<u32>) {
    accumulate_tile(tile, lane, id, false);
}

fn accumulate_tile(tile: vec3<u32>, lane: u32, id: vec3<u32>, reduced: bool) {
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
        for (var i = lane; i < area * u32(post.quality.x); i += 256u) {
            let layer = i / area;
            let offset = i % area;
            let p = begin + vec2<i32>(i32(offset % u32(extent.x)), i32(offset / u32(extent.x)));
            var z = layer_depth(p, i32(layer));
            if !reduced && layer > 0u && z >= 1e19
                && (textureLoad(mask_source, p, 0).r < 0.5
                    || layer_depth(p, i32(layer) - 1) >= 1e19) {
                continue;
            }
            var fragment_radius = radius(z);
            var center = vec2<f32>(p) + 0.5;
            if reduced {
                let fragment = unpack_fragment(p, i32(layer));
                if fragment.mass == 0.0 { continue; }
                z = fragment.z;
                fragment_radius = fragment.coc;
                center = fragment_center(fragment);
            }
            let delta = center - clamp(center, tile_min, tile_max);
            if length(delta) > max(fragment_radius, 0.5) + 0.5 { continue; }
            let source_id = (layer * u32(size.y) + u32(p.y)) * u32(size.x) + u32(p.x);
            let key = vec2<u32>(bitcast<u32>(z), source_id);
            if !prefix_matches(key) { continue; }
            atomicAnd(&key_and[0], key.x);
            atomicAnd(&key_and[1], key.y);
            atomicOr(&key_or[0], key.x);
            atomicOr(&key_or[1], key.y);
            let slot = atomicAdd(&count, 1u);
            if slot < CAPACITY { keys[slot] = key; }
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
                let z = layer_depth(p, layer);
                chunk[lane].color = textureLoad(layers_color, p, layer, 0);
                if layer == 0 { chunk[lane].color = textureLoad(shaded_first, p, 0); }
                chunk[lane].shape = vec4<f32>(z, radius(z), 1.0, 0.0);
                centers[lane] = vec2<f32>(p) + 0.5;
                if reduced {
                    let fragment = unpack_fragment(p, layer);
                    chunk[lane].color = vec4<f32>(fragment.color, 1.0);
                    chunk[lane].shape = vec4<f32>(fragment.z, fragment.coc, fragment.mass, 0.0);
                    centers[lane] = fragment_center(fragment);
                }
            }
            workgroupBarrier();
            if valid_pixel && transmission > 0.01 {
                for (var i = 0u; i < min(32u, length - base); i++) {
                    let f = chunk[i];
                    let offset = pixel - centers[i];
                    let r = max(f.shape.y, 0.5);
                    var coverage = 1.0 - smoothstep(max(r - 0.5, 0.0), r + 0.5, aperture_metric(offset));
                    if f.shape.y < 0.5 {
                        coverage = select(0.0, 1.0, all(abs(offset) < vec2<f32>(0.5)));
                    }
                    let single_alpha = min(1.0, 1.0 / (r * r));
                    let alpha = (1.0 - pow(1.0 - single_alpha, f.shape.z)) * coverage;
                    let weight = transmission * alpha;
                    rgb += weight * f.color.rgb;
                    transmission -= weight;
                }
            }
            workgroupBarrier();
        }
        if lane == 0u { atomicStore(&unsaturated, 0u); }
        workgroupBarrier();
        if valid_pixel && transmission > 0.01 { atomicAdd(&unsaturated, 1u); }
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
        // Normalize the finite splat coverage as in section 5.3. This preserves a
        // constant radiance field instead of darkening it by residual transmittance.
        let opacity = 1.0 - transmission;
        let resolved = select(post.background.rgb, rgb / max(opacity, 1e-6), opacity > 1e-6);
        textureStore(result, vec2<i32>(id.xy), vec4<f32>(resolved, 1.0));
    }
}
