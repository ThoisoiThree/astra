# Depth of field: Franke et al. (2018)

The main DOF path implements the partial multilayer tiled splatting algorithm in
**Linus Franke, Nikolai Hofmann, Marc Stamminger, Kai Selgrad: Multi-Layer Depth
of Field Rendering with Tiled Splatting**, PACM CGIT 1(1), Article 6, 2018,
[DOI: 10.1145/3203200](https://doi.org/10.1145/3203200).
The supplied paper is `reference/franke2018.pdf`. The authors' [supplemental
material](https://selgrad.org/publications/2018_i3d_FHSS_suppl.pdf) clarifies
umbra separation and opacity accumulation.

## Paper analysis

The contribution is a complete pipeline, not a different aperture sampling
pattern. A single pinhole image lacks surfaces that become visible through a
blurred foreground. More samples of that image cannot recover this information.
Likewise, averaging visible neighbors cannot reproduce ordered occlusion.

Sections 4–6 address these problems in four phases:

1. **Generation:** collect several depth layers. Keep hidden fragments near depth
   discontinuities dilated by CoC; discard fragments in the previous fragment's
   umbra, where the finite aperture cannot see them.
2. **Reduction:** merge four similarly shaded, similarly deep, defocused
   fragments; repeat to form footprints up to 4×4. Keep unmergeable fragments.
3. **Tiling:** put every fragment into every screen tile touched by its footprint.
4. **Accumulation:** sort each tile by depth and alpha-composite contributions
   front to back, terminating saturated pixels.

This avoids statistical sampling noise and supports partial occlusion. The paper
also identifies a remaining bias: splatting can overestimate defocused silhouette
width because opacity saturates sooner than in lens ray tracing. Finite layer
counts cannot reproduce arbitrarily complex occlusion.

## Implementation

`src/render/dof.rs` owns the additional GPU resources and pipelines;
`dof.wgsl` implements mask generation, sorting and accumulation.
`peel.wgsl` is shared by ordinary geometry, cartoons and analytic Toon spheres.
`optics.wgsl` shares camera reconstruction and the uniform layout with the other
postprocessing shaders. Chemistry, selections and molecular display data remain
independent of the DOF implementation.

Every layer, including the visible first layer, is rasterized by the molecular
geometry pipelines at the DOF target resolution. This keeps color, depth, rays,
viewport coordinates and pixel centers aligned. A relative eye-space
discontinuity test creates a radius field; two separable dilation passes create
a conservative circular disocclusion mask. Successive geometry passes collect the
next visible surface only in this mask.
The umbra endpoint for preceding depth `z`, projected pixel width `s` and aperture
diameter `A` is `z*A/(A-s)`. When `s >= A`, no further layer is retained.

Section 6 reduction runs before splatting. Each invocation merges the heads of
four depth-ordered lists, first over 2×2 pixels and then over 4×4. A merge requires
all four heads to have the same footprint, similar depth and shaded RGB, and a
sufficiently large CoC. Unmerged fragments remain in the lists and cannot merge
at the next level. The output averages color, moves the center to the footprint
center, places depth just behind the deepest child, and conservatively encloses
the children's CoCs. Intermediate lists are sorted again before the second step.
After each merge, fragments inside its umbra are discarded; a footprint wider
than the aperture casts an infinite shadow. The inset footprint is 1.5 pixels
at 2×2 and 3.5 at 4×4. Both steps run in one compute dispatch.

Merge thresholds are conservative implementation parameters: 0.1% relative depth,
0.01 maximum RGB difference, and minimum CoC of 4/8 pixels for the two steps.
They adapt the paper's qualitative criteria to molecular scene units, rather
than copying the supplemental scene-dependent numeric thresholds. RGB comparison
preserves boundaries between differently colored chains of similar luminance.
Artificial environment samples are not merged: they close exposed lists but do
not represent physical occluders with a mergeable footprint.

Each 16×16 compute workgroup constructs its tile list from the source fragments
in the surrounding CoC extent, then bitonic-sorts exact `(f32 depth, source ID)`
keys. This fuses binning, sorting and traversal without global duplicated lists.
A 1024-entry workgroup list is a **batch size, not a truncation limit**. Overfull
lists split at the first differing bit of the full key; near partitions are
consumed before far partitions. Common key bits are skipped. Thus large radii,
equal-depth surfaces and dense occlusion retain all relevant fragments with
bounded workgroup storage. Dense tiles may require additional traversal work.

Each fragment uses `alpha = min(1, 1/r²)`. For a merged fragment representing
`n` samples, opacity is `1-(1-alpha)^n`, the closed form of the supplemental
geometric sum.
Iris-shaped coverage modulates opacity. Front-to-back accumulation is normalized
by total coverage; one clear-color layer represents environment exposed by
peeling. Stable source IDs resolve equal-depth ties. Subpixel in-focus fragments
keep only their own pixel. The final composition always consumes the accumulated
DOF image when DOF is active, so there is no coverage threshold that can create
hard transitions between sharp and blurred regions. AO and semantic Toon
contours are applied to the visible layer's color before splatting, using the
full-resolution scene buffers. FXAA runs on the accumulated image afterwards;
it does not mix colors across different depths before layer generation.

The thin-lens equation gives a **diameter**; both CPU and GPU now convert this to
a radius before applying the existing maximum-CoC clamp. The infinity limit is
evaluated from the lens equation rather than forced to maximum blur. The focal
point, lens controls and saved scene representation are preserved; correcting
the diameter/radius distinction halves the previous unclamped blur extent.
Actual rounded target dimensions and normalized viewport coordinates are used
for depth reconstruction and image mapping.

| Quality | Depth layers | DOF resolution |
| --- | ---: | ---: |
| Preview | 3 | Half |
| Medium | 4 | Full |
| High | 5 | Full |

DOF remains opt-in in the Camera panel. This is the sole DOF algorithm once
enabled; the former stochastic aperture gather is removed. UI and measurement
annotations are rendered afterwards. The GPU profiler's DOF interval includes
first-layer generation, mask construction, peeling, visible-layer shading and splatting.

## Deliberate differences and limits

- Layers and the mask are rebuilt in the current frame. This uses the
  non-temporal option discussed in section 4.3 instead of geometry-shader
  amplification and temporal reprojection; it costs extra scene draws.
- The portable WGSL implementation uses bounded radix partitions instead of
  CUDA list-size scheduling and global-memory overflow sorting. No subgroup
  extensions or additional device features are required.
- Umbra rejection is applied during peeling and after both merge levels. The
  4×4 footprint scales the supplemental 2×2 inset heuristic with fragment size.
- Hidden layers use their normal material shading; they do not have independent
  AO or semantic Toon contours. Those effects come from the visible layer.
- Preview can lose subpixel geometry and soften transitions.
  Screen-space data cannot reconstruct geometry outside the viewport or beyond
  the selected layer count. Polygonal irises use analytic coverage with a
  half-pixel soft rim instead of the circular support in the paper.
- Layer textures are allocated lazily while DOF is enabled and released when it
  is disabled. The five color/depth layers, two masks and shaded visible-layer
  target cost 76 bytes per DOF pixel; packed reduction records add 80 bytes.
  Total DOF storage is 156 bytes per pixel (about 77 MiB at half-resolution 1080p
  or 309 MiB at full-resolution 1080p), in addition to scene/output targets. The memory counter
  includes them. No frame-rate equivalence with the paper's CUDA implementation
  is claimed.

## Validation

`cargo test` validates all assembled WGSL modules with Naga, checks their uniform
layout against Rust, and tests signed CoC, its physical radius, infinity limit,
and clamp without requiring a GPU.

`cargo run --example dof_validate` is a separate manual headless GPU check.
It creates the actual DOF pipelines and compares readback to an exhaustive CPU
reference with no tiling, reduction or list capacity. Fixtures cover constant
HDR radiance, equal-depth checkerboards exceeding tile capacity, exact focus,
a peeled background behind a defocused foreground stripe, and a smooth sloped
surface with depth ordering opposite to source order and CoC crossing the focal
plane. Odd sizes, half resolution and 1×1 images are included.
An independent GPU ordering oracle scans raw lists without tiling, bitonic sort,
or reduction on small fixtures. Structural GPU readbacks verify 4×4 collapse,
matching surfaces across different layer numbers, finite and infinite umbra
trimming, and preservation of focused pixels. Molecular images are additionally
compared with the unreduced path to bound reduction error.

The renderer caches the composed viewport. Camera, lens, display, geometry,
annotations, and target-size changes invalidate it; UI-only frames reuse it.
The event loop honors egui repaint deadlines and does not request another frame
merely because it received `RedrawRequested`.

The same example also rasterizes actual `4R8P` cartoon geometry, atomic meshes
and analytic Toon spheres through the scene, first-layer, peeling, source-shading
and final-composition pipelines. These draw calls catch missing bind groups that
pipeline creation alone cannot detect. An offset viewport at odd resolutions
exercises full and Preview resolution. Readback checks finite, nonempty results,
a visible aperture response, and a constant AO field applying exactly once.
Set `ASTRA_DOF_ARTIFACTS=/tmp/astra-dof-check` to save sharp, AO and DOF PPM images.
This check was run on Apple M3 Pro, including visual inspection of the molecular
readbacks. It requires an available GPU; normal unit tests do not. Interactive
frame rate and temporal stability while orbiting remain separate evaluations.
