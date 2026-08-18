# molview

`molview` is an early, usable native molecular viewer written in Rust. This MVP
loads PDB files, builds renderer-independent molecular topology, evaluates a
composable selection language, and displays atoms and bonds through batched wgpu
instancing with a small egui interface.

> Screenshot placeholder — a real project screenshot will be added after UI polish.

## Current features

- Fixed-column `ATOM`, `HETATM`, `CONECT`, first-`MODEL`, and altloc A/blank parsing
- Conservative element inference and common biological element properties
- Spatial-grid covalent bond inference (no global all-pairs scan)
- Depth-tested, lit instanced spheres and sticks
- Orbit, pan, zoom, automatic framing, resize handling, and Fit
- Click-to-pick atoms with atom/residue/chain property inspection
- Chain → residue → atom hierarchy with independent selection/expansion controls
- Hierarchical HSV color overrides and inherited tri-state visibility controls
- Named selections that can be reactivated, deleted, and reused in expressions
- Camera panel with clipping and thin-lens optical bokeh controls
- Element/CPK, chain, residue, residue-type, B-factor, and uniform color schemes
- Native open dialog, command-line path, and `.pdb` drag-and-drop
- Boolean selection AST with useful position-bearing syntax errors
- Per-atom color, sphere/stick visibility, and non-destructive selection highlight
- Command history with Up/Down while the command field is focused

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

Or start without a file and use **Open PDB**:

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
reframes the molecule and **Reset colors** restores element/CPK coloring.

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

## Selection language

Keywords are ASCII case-insensitive. `not` binds most tightly, then `and`, then
`or`; parentheses override precedence.

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
<expression> or <expression>
(<expression>)
```

## Commands

Selection expressions and commands are separate typed parsers:

```text
select <expression>
select <name>, <expression>
color <name-or-#RRGGBB>, <expression>
show spheres|sticks, <expression>
hide spheres|sticks, <expression>
```

Try these with `examples/minimal.pdb`:

```text
select chain A and resi 1-2
select active_site, chain A and resi 1-2
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

## Architecture

The project deliberately remains one Cargo package with strict state boundaries:

```text
PDB -> Molecule (atoms/topology) -> selection AST/evaluation
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

- PDB only; only the first model and altloc blank/A are loaded
- Connectivity has no bond order and uses approximate distance perception
- One molecular object at a time
- No labels, measurements, undo, or saved sessions
- Spheres and sticks only; rendering favors responsiveness over publication quality
- No browser build yet

## Roadmap

1. mmCIF
2. Multiple molecular objects
3. Improved bond perception
4. Sequence viewer
5. Cartoon/ribbon representation
6. Molecular surfaces
7. Labels and measurements
8. Trajectory support
9. Scripting/API
10. WASM/browser target

## Development checks

```bash
cargo fmt --check
cargo check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## License

GNU General Public License v3.0 only. See `LICENSE`.
