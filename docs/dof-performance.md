# DoF performance investigation

This is a source-level investigation, not a measured speedup report. Builds,
benchmarks and image comparisons were not run at the user's request.

## What the old CPU number measured

`Renderer::render` measured wall time from entry through `queue.present`.
It included pipeline compilation, command encoding, surface acquisition and
driver submission/presentation. In particular, acquiring the next surface image
can wait for earlier GPU work. A slow DoF pass can therefore inflate this number
without expensive CPU computation. It also excluded the preceding egui/app work,
so it was neither total application CPU time nor the frame interval.

The overlay now labels this value `Frame wall` and separates `Prepare`, `Encode`,
`Acquire`, `Submit` and `Present`. These remain wall-clock measurements, not
CPU utilization. GPU timestamps separately show `DOF layers` (rasterization and
the intervening mask generation), `DOF merge` (visible-layer shading and fragment
reduction), and `DOF splat` (candidate gathering, sorting and accumulation).
`GPU span` and `DOF span` use the earliest measured start and latest measured
end in the relevant set, rather than adding potentially overlapping intervals.
GPU results are asynchronously sampled, so they need
not correspond to the CPU frame currently displayed.

## Changes that preserve the rendering algorithm

- Retain DoF resources when the effect is disabled. Toggling it back on reuses
  pipelines and textures. Resizing or replacing the output texture invalidates
  this cache. The tradeoff is retaining the allocated DoF GPU memory while off.
- Compile the reference and quadratic ordering-oracle pipelines only through
  `DepthOfField::new_validation`, used by the validation examples. The ordinary
  renderer creates only its five compute pipelines.
- In the reduced splat path, load depth, CoC and color from reduced records
  directly, without first loading raw values that are immediately overwritten.
  A driver compiler might already remove some of those redundant operations;
  the magnitude of any benefit needs measurement.
- Clear only active reduced-texture layers. Accumulation only reads active
  layers, and a newly activated layer is cleared before being consumed.

Layer count, resolution, merge predicates, depth ordering, aperture shape,
opacity and saturation thresholds have not been lowered or approximated.

## Remaining likely bottlenecks and next steps

The user's subsequent screenshot showed approximately 0.04 ms preparation,
0.44 ms encoding, 798.70 ms acquisition, 66.11 ms DoF merge and 828.96 ms splat.
This points to GPU throughput, not expensive application CPU work. Compose and
annotation intervals were both about 895 ms; they must not be interpreted as
another independent 895 ms of work each. Timestamp ordering/dependency effects
need a native GPU trace to distinguish precisely.

### Compact source traversal

Reduction now emits an exact compact list of surviving source IDs per 4x4 block
alongside the existing reduced records. The capacity is the full source bound:
16 pixels times the allocated layer count. Merging and umbra culling can only
reduce that count, so no fragment is discarded to fit the list. Counts are
rewritten every dispatch; list texels beyond the count need no clearing.

Each tile traverses these compact block lists rather than rereading every empty
texture slot. Source positions at boundary blocks are clipped to the original
candidate rectangle. The circle test, full depth/source key, sorting and radix
overflow traversal are unchanged. Radix overflow still repeats traversal, but
over compact source IDs. Opacity depending only on fragment CoC and mass is now
calculated once when loading a shared fragment chunk, instead of per output
pixel. The coverage and compositing formulas are unchanged.

The additional texture allocation is approximately 20.25 bytes per DoF target
pixel at five allocated layers, including block counts (rounded up at image
edges). It trades modest extra GPU memory for fewer empty-slot visits and avoids
a large global per-output-tile append buffer.

`splat_dense` retains the reduced-texture scan as a validation-only baseline.
It also retains the per-pixel opacity calculation, so the comparison covers
both compaction and moving that calculation into the shared fragment load.
The manual validation example now compares compact and dense results in addition
to the existing exhaustive reference, including odd dimensions and multiple iris
shapes. These comparisons have been added but **not executed**. No before/after
performance numbers are available yet.

The splat shader examines a neighborhood around every 16×16 output tile. Its
initial candidate count scales roughly as
`tiles × (16 + 2 × margin)² × layers`. The margin follows maximum CoC, including
the merged-fragment footprint. When more than 1024 candidates survive, radix
partitioning scans the neighborhood again for the next partitions. Large blur
radii and dense, multilayer molecular geometry make this expensive on every
backend. This is a code-derived cost model, not proof that it dominates a
particular scene.

### Rolled-back experiment

The per-output-tile candidate cache and face-normal iris evaluation were removed
following the user's report of a performance regression. Splat uses the earlier
compact source-block traversal and original trigonometric aperture formula.

The render quality presets were shifted: Medium uses the former Preview settings,
High uses the former Medium settings, and Preview is now lighter:

| Preset | DoF scale per axis | Layers | AO samples | Cartoon samples / width segments |
| --- | --- | --- | --- | --- |
| Preview | 0.25 | 2 | 6 | 3 / 2 |
| Medium | 0.5 | 3 | 12 | 5 / 4 |
| High | 1.0 | 4 | 32 | 10 / 8 |

The new Preview trades image detail for responsiveness. Its DoF output has four
times fewer pixels than the previous Preview, before accounting for fewer layers
and smaller blur neighborhoods in target pixels. This is not a measured speedup.

If `DOF merge` dominates, investigate register spilling from the two private
arrays of up to 80 fragment records per invocation. A separate 2×2/4×4 traversal
or a different workgroup size may help, but extra memory traffic can negate it.
If `DOF layers` dominates, cached render bundles for repeated molecular draws
and shared-memory caching in the mask dilation passes are candidates.

If only `Prepare` spikes on viewport resize, separate reusable pipelines from
size-dependent targets: the current resize path still rebuilds both. If
`Acquire`/`Submit`/`Present` dominates while GPU DoF times are high, optimizing
the shader workload is more useful than treating the wall-time number as
application CPU computation.

Before accepting these larger changes, compare the existing reference/oracle
outputs on edges, exposed background, overlapping surfaces, multiple iris
shapes and maximum CoC. Benchmark steady rotation separately from first enable,
resize, and idle cached UI frames on both Vulkan and DX12.
