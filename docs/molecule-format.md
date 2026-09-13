# Molecule 1.0

Molecule 1.0 (`.mol`) is a self-contained, portable project file. It stores molecular geometry and bonds,
all selections and expressions, hierarchy and selection styling, representations, measurements,
camera projection, clipping, pivot, and depth of field. GPU resources, tessellated meshes, viewport
pixels, undo history, and transient dialogs are intentionally excluded and rebuilt on load.

The media type is `application/vnd.astra.molecule`. The `.mol` suffix is also traditionally used
by MDL Molfile, so the suffix alone never identifies this format. Astra always checks the leading
ASCII magic `MOLECULE`; a `.mol` file without it is reported explicitly as an unsupported MDL
Molfile instead of being passed to the Molecule decoder.

## Why Protobuf plus Zstandard

The payload uses Protocol Buffers because numbered fields provide a language-neutral schema and a
well-defined compatibility discipline. Unlike serializing Rust structs with bincode, Postcard, or
rkyv, changing an in-memory type does not silently change the file contract. FlatBuffers and Cap'n
Proto offer excellent zero-copy access, but a scene is loaded once and then converted to renderer
batches, so their alignment overhead and mandatory code-generation toolchain do not buy enough here.
glTF is useful for rendered mesh exchange, but it cannot natively preserve molecular topology,
selection expressions, or hierarchical style inheritance.

The Protobuf payload is compressed as one Zstandard frame at level 7 with content size and checksum.
This is lossless and portable. Molecular data compresses especially well after columnar encoding:
atom properties are packed arrays, repeated strings use one string table, boolean selections use
bitsets, and sparse style overrides use run-length records.

## Container

All integers in the fixed header are little-endian.

| Offset | Size | Meaning |
|---:|---:|---|
| 0 | 8 | ASCII magic `MOLECULE` |
| 8 | 2 | Container version, currently `1` |
| 10 | 2 | Compression codec, `1` = Zstandard |
| 12 | 4 | Payload schema version, currently `1` |
| 16 | 8 | Uncompressed payload length |
| 24 | ... | One Zstandard frame containing the Protobuf payload |

Readers reject unknown container/codec versions, payloads larger than the configured safety limit,
length mismatches, checksum errors, non-finite geometry, invalid indices, and unsupported schemas.

## Versioning policy

The authoritative payload contract is [`schemas/molecule_1_0.proto`](../schemas/molecule_1_0.proto).

- Existing field numbers are never reused. Removed fields remain `reserved`.
- Compatible optional fields may be added with new numbers while keeping the schema version.
- A semantic change or a required new field creates a new schema version and an explicit migration
  from the previous version into the current runtime model.
- `minimum_reader_version` is raised whenever an older reader cannot reproduce the scene faithfully.
- Writers always emit the newest schema. Readers validate and migrate older supported schemas before
  constructing runtime state.
- Derived caches and GPU data never enter the format; this prevents renderer changes from becoming
  file-format changes.

The current implementation writes schema `1` and implements reader version `3`. Files using only the
reader-1 feature set declare `minimum_reader_version = 1`; secondary-structure coloring declares
version `2`. A reader rejects a higher requirement before constructing runtime state. Both the local
container and decompressed payload are limited to 512 MiB, and decompression never reserves the full
advertised size up front.

## Save and recovery

Save encodes to a uniquely named temporary file in the destination directory, flushes and synchronizes
it, then atomically renames it over the destination. A failed write leaves the previous project intact.
Changed tabs show a dirty indicator and require an explicit Save, Discard, or Cancel decision on close.

Dirty documents are autosaved after a short idle delay into Astra's per-user recovery directory.
Recovery files never replace the user's project and are offered for restore at the next start. A
successful Save or an explicit Discard removes the corresponding recovery file.

## Reproducibility boundary

Floating-point coordinates and camera values are stored as IEEE-754 binary32, matching the runtime.
Opening a scene restores the same semantic scene and camera. Exact output pixels can still vary with
viewport aspect ratio, GPU shader precision, font rasterization, and future renderer improvements;
those platform-specific artifacts are intentionally not serialized.

AMOEBA measurement objects require reader 3. `MeasurementLine.hydrogen_bonds` (field 9)
is a versioned UTF-8 JSON report containing candidate atom/group indices, continuous
energies and their decomposition, diagnostic geometry, template status and SCF settings.
It also stores the energy display threshold; ordinary line style fields apply to the
whole object. Older readers reject these files instead of silently losing the analysis.
Reader 3 additionally supports element codes 17–22 (Li, Rb, Cs, Be, Sr, Ba).
