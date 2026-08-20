# molview

`molview` is an early, usable native molecular viewer written in Rust. This MVP
loads PDB, PDBx/mmCIF, BinaryCIF, and PDBML/XML structures, builds renderer-independent molecular topology, evaluates a
composable selection language, and displays atoms and bonds through batched wgpu
instancing with a small egui interface.

> Screenshot placeholder — a real project screenshot will be added after UI polish.

## Current features

- PDB, PDBx/mmCIF, BinaryCIF, and PDBML/XML coordinate parsing, including gzip
- Conservative element inference and common biological element properties
- Spatial-grid covalent bond inference (no global all-pairs scan)
- Depth-tested, lit instanced spheres and sticks
- Orbit, pan, zoom, automatic framing, resize handling, and Fit
- Click-to-pick atoms with atom/residue/chain property inspection
- Chain → residue → atom hierarchy with independent selection/expansion controls
- Hierarchical HSV color overrides and inherited tri-state visibility controls
- Global and hierarchical Cartoon/Ball & stick/Toon display modes
- Named selections with color/visibility attributes and a preserved internal hierarchy
- Portable Molecule 1.0 (`.mol`) project files with embedded geometry, selections, styling, camera state,
  schema versioning, and Zstandard compression
- Dashed distance measurements with Å labels and editable line styles
- Camera panel with clipping and thin-lens optical bokeh controls
- Element/CPK, chain, residue, residue-type, B-factor, and uniform color schemes
- Open/Fetch/Save file menu, RCSB PDB download by ID, command-line path, and drag-and-drop
- Boolean selection AST with useful position-bearing syntax errors
- Per-atom color, sphere/stick visibility, and non-destructive selection highlight
- Command history with Up/Down while the command field is focused
- 50-step non-camera undo/redo history with Ctrl/Cmd+Z and Ctrl/Cmd+R

## Requirements

- Current stable Rust with edition 2024 support
- A desktop supported by winit/wgpu: Windows (D3D12), macOS (Metal), or Linux
  (Vulkan; an OpenGL fallback may be available)
- On Linux, the normal graphics/window-system development packages required by
  winit and wgpu

## Build and run

Run the bundled molecule immediately:

```bash
cargo run -- examples/minimal.pdb
cargo run -- examples/4R8P.pdb
```

Or start without a file and use **File → Open**:

```bash
cargo run
```

Optimized build:

```bash
cargo build --release
```

Controls: click an atom to select and inspect it, left-drag orbits, right-drag or
Shift+left-drag pans, and the mouse wheel/trackpad zooms. Ctrl/Cmd with either
right-drag or macOS three-finger drag translates the camera and pivot in screen
space, so the gesture always follows visible left/right/up/down. The
hierarchy manager can select whole chains, residues, or individual atoms. **Fit**
reframes the molecule and **Reset colors** restores chain coloring.

Supported coordinate inputs are `.pdb`/`.ent`, `.cif`/`.mmcif`, `.bcif`, and
PDBML `.xml`; each can be gzip-compressed. Biological assembly files use these
same readers. Structure-factor and validation CIF/XML files are recognized, but
if they contain no `atom_site` coordinates the viewer reports that they are
non-displayable data rather than treating them as a broken structure. Validation
PDF reports likewise have no molecular coordinates.

The **File** menu contains **Open**, **Fetch**, **Save as**, and **Save**. **Fetch** opens a
modal PDB ID dialog with cancellable download progress and transfer speed. Large files use
four HTTP byte-range workers when RCSB advertises range support; small files use one stream
to avoid connection overhead. Downloads are stored under `~/downloads/pdb/` and opened
without blocking the UI. Save as
creates a Molecule 1.0 `.mol` project; subsequent Save operations update that file.

Ctrl/Cmd+Z undoes edits to selections, named selections, colors, visibility,
representations, and other display state. Ctrl/Cmd+R reapplies them. The newest
50 edits are retained; camera changes and loading a different structure are not
part of this history.

### Display mode

The top toolbar menus are ordered **Mode**, **Coloring**, **Camera**. Cartoon is
the default and draws a smoothed backbone ribbon through protein `CA` atoms and
nucleic-acid `P` atoms; disconnected residues and chains are never bridged.
Ligands and residues without a cartoon backbone remain in Ball & stick. The
second global mode is **Ball & stick**, matching the viewer's original rendering.
**Toon** renders space-filling atoms as analytic ray/sphere impostors, writes
atom/residue/chain IDs, and adds depth-gated illustrative outlines over a warm
paper background. Smoothly intersecting atoms merge visually instead of receiving
an unconditional circle around every sphere.

All modes use Chain coloring by default; Element/CPK remains available in the
Coloring menu. Picking an
atom in the viewport reveals its hierarchy path and highlights the atom, its
residue, and its chain without adding the ancestors to the editable
multi-selection.

Every chain, residue, and atom has a mode badge before its color and visibility
attributes. A gray badge inherits; an orange badge is a local override. Clicking
cycles through the three modes and back to inheritance. Right-click
provides **Reset to default** and **Set to children**. Effective priority is atom
→ residue → chain → named selection → global; a value equal to the global mode is
stored as inheritance rather than as an unnecessary override.

The **Mode** window also contains ambient-occlusion controls: enable/disable,
strength, world-space radius, surface bias, and Low/Medium/High quality (16/32/48
samples). The implementation reconstructs positions and normals from molecular
depth, uses a rotated low-discrepancy screen-space kernel, and applies a depth-aware
bilateral filter. It applies to Cartoon, Ball & stick, and Toon.

### Camera and optical depth of field

Open **Camera** in the top toolbar to edit the near and far clipping planes,
which default to 1 and 1000. The
DOF renderer uses a thin-lens circle-of-confusion equation and a 64-sample
source-aware aperture gather instead of a generic radial blur. Focal length,
sensor height, f-stop, maximum CoC, iris blade count, and iris rotation are
editable. Zero blades produces a circular iris; 3–12 blades produce the matching
polygonal optical bokeh shape. Near and far samples are depth-aware, so foreground
bokeh can cross a background edge while far blur does not bleed over focused
foreground geometry.
The focus point can be resolved from:

- a chain ID (focuses its atom centroid);
- a residue by chain and residue number;
- a nucleic-acid base by chain, residue number, and PDB residue name;
- an atom by PDB serial number;
- the chain, residue, or atom currently shown in the Inspector.

Focus points remain attached to molecular coordinates while the camera orbits.
**Set pivot from inspected** moves the orbit pivot to the selected atom or to the
centroid of the selected residue/chain without moving the eye. **Reset pivot**
returns it to the molecule center.

### Coloring and hierarchy overrides

Open **Coloring** in the top toolbar to choose Element/CPK, Chain, Residue
identity, Residue type, B-factor, or Uniform coloring. Every chain, residue, and
atom row has a color square. Clicking it opens an HSV editor; **Default** removes
the local override. Effective priority is atom → residue → chain → active base
scheme, and a local color participates in that priority only while it is a real
override (an inherited/default match is discarded).

The eye beside each hierarchy row cycles gray → green → red → gray: inherit,
force visible, hide, inherit. A lower-level state wins, so a green residue or atom
remains visible inside a red chain. Clicking a row selects/highlights it without
opening it; only its disclosure triangle expands or collapses the node.

Shift-click selects the inclusive same-level range from the last normal-click
anchor: chains between chains, residues between residues, or atoms between
atoms. Ctrl/Cmd-click toggles one row without moving that anchor. Editing the
color square or eye of any selected row applies the new value to the whole set.
Right-click either attribute for **Reset to default** or **Set to children**. The
latter recursively forces the row's effective value onto its residues and atoms.

Named selections use the same color square, visibility eye, reset, and propagation
controls. Expanding one shows only its selected atoms while preserving their
chain → residue → atom hierarchy. A named-selection style is a parent layer, so
explicit chain, residue, and atom overrides still take priority.

### Distance lines

Create two named selections with the **Select!** button, then open **Actions** and
choose them as endpoints A and B. Each endpoint may contain one atom or atoms
from exactly one residue/nucleic-acid base; residue and base endpoints use their
atom centroid. **Create distance line** adds a depth-tested dashed line with a
centered distance label in ångströms.

Dashed segments have closed flat end caps. Lines are rendered in a separate
depth-tested annotation pass after molecular AO and DOF, so measurements and
future markup do not alter molecular depth, ambient occlusion, or shading.

Every measurement is an independent object in the **Lines** folder. Its HSV
color, gray/green/red visibility state, line thickness (0.01–10 Å), label size
(8–48 pt), and lifetime can be edited there. Measurement creation, styling,
visibility changes, resizing, and deletion participate in the 50-step Undo/Redo
history.

Right-click a chain, residue/base, atom, named selection, or measurement line and
choose **Rename** to assign a user-facing name. Molecular aliases never modify
the original structure identifiers; **Reset name** restores their generated PDB/
mmCIF label. Renaming a named selection updates stored `selection …` references.

## Selection language

Keywords are ASCII case-insensitive. Boolean precedence is `not`, `and`, `xor`,
then `or`; parentheses override precedence.

```text
all                     none
element C               name CA
resn ALA                resi 42
resi 10-30              chain A
serial 123              hetatm
polymer
selection active_site

not <expression>
<expression> and <expression>
<expression> xor <expression>
<expression> or <expression>
(<expression>)

Chain A/LEU*                 residue-name wildcard
Chain B/[20:22, 70:71]       residue list and inclusive ranges
Chain A/LEU*/C*              optional atom-name wildcard
../LEU* AND [20:30, 45:50]   combine masks and residue ranges
```

Path masks are case-insensitive. `*` matches any sequence, `?` matches one
character, and `..` means any chain. Commas inside `[]` do not conflict with the
colon separating a named selection from its expression.

## Commands

Selection expressions and commands are separate typed parsers:

```text
select <expression>
select <name>: <expression>
color <name-or-#RRGGBB>, <expression>
show spheres|sticks, <expression>
hide spheres|sticks, <expression>
```

To save the atoms currently selected in the viewport or hierarchy, enter only a
new name such as `active_site` and press **Select!**. The explicit
`select active_site:` form is also accepted. Full commands such as `select all`
retain their existing meaning.

Try these with `examples/minimal.pdb`:

```text
select chain A and resi 1-2
select active_site: chain A and resi 1-2
select leucines: Chain A/LEU*
select loops: Chain B/[20:22, 70:71]
select leucine_loops: ../LEU* AND [20:30, 45:50]
color magenta, selection active_site
select hetatm and not element H
color red, element O
color #33aaff, chain A and element C
show spheres, hetatm
hide spheres, element H
hide sticks, chain B
```

Named colors currently include red, green, blue, yellow, orange, magenta, cyan,
white, and gray/grey.

The legacy comma form for named selections remains accepted. Right-click a named
selection in the manager and choose **Edit expression** to reevaluate it while
keeping its color and visibility settings.

## Architecture

The project deliberately remains one Cargo package with strict state boundaries:

```text
PDB/mmCIF/BCIF/PDBML -> Molecule (atoms/topology) -> selection AST/evaluation
                                                    -> DisplayState
CameraState + Molecule + DisplayState -> instanced wgpu Renderer
HDR scene color + depth -> thin-lens aperture gather -> egui overlay
egui UiState -> typed Command -> DisplayState mutation
```

`molecule`, `selection`, and `command` have no dependency on egui or application
GPU objects. `DisplayState` owns colors, representation masks, and the current
selection; the renderer derives GPU instance buffers only when that state changes.
See `AGENTS.md` for the contributor contract.

## Current limitations

- Only the first model and altloc blank/A are loaded
- Connectivity has no bond order and uses approximate distance perception
- One molecular object at a time
- No labels, measurements, or saved sessions
- Spheres and sticks only; rendering favors responsiveness over publication quality
- No browser build yet

## Roadmap

1. Multiple molecular objects
2. Improved bond perception
3. Sequence viewer
4. Cartoon/ribbon representation
5. Molecular surfaces
6. Labels and measurements
7. Trajectory support
8. Scripting/API
9. WASM/browser target

## Development checks

```bash
cargo fmt --check
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## License

GNU General Public License v3.0 only. See `LICENSE`.
