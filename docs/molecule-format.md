# Molecule 1.0

Molecule 1.0 (`.mol`) is a self-contained, portable project file. It stores molecular geometry and bonds,
all selections and expressions, hierarchy and selection styling, representations, measurements,
camera projection, clipping, pivot, and depth of field. GPU resources, tessellated meshes, viewport
pixels, undo history, and transient dialogs are intentionally excluded and rebuilt on load.

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

The current implementation writes schema `1`, requires reader `1`, and applies a 2 GiB upper bound to
the decompressed payload to limit decompression-bomb and allocation risks.

## Reproducibility boundary

Floating-point coordinates and camera values are stored as IEEE-754 binary32, matching the runtime.
Opening a scene restores the same semantic scene and camera. Exact output pixels can still vary with
viewport aspect ratio, GPU shader precision, font rasterization, and future renderer improvements;
those platform-specific artifacts are intentionally not serialized.
