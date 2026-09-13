# Heavy-atom template provenance

`heavy_templates.json` contains ideal heavy-atom coordinates (Å), element names
and heavy-atom bonds for the 20 standard amino acids, extracted from the
[wwPDB Chemical Component Dictionary](https://west.wwpdb.org/data/ccd).
Original component CIF files are served by [RCSB PDB](https://www.rcsb.org/downloads/ligands)
at `https://files.rcsb.org/ligands/download/{COMPONENT}.cif`.
Each entry records its source URL and the SHA-256 digest of the downloaded CIF.

Regenerate with `python3 tests/reference/amoeba/export_heavy_templates.py`.
The exporter uses only the Python standard library and is a development utility;
the application embeds the exported data and reconstructs coordinates in Rust.

These geometry templates are distinct from the AMOEBA force-field parameters in
`amoeba2018.json`. The adjacent OpenMM license pertains to those AMOEBA parameters
and the OpenMM reference implementation, not to attribution of the CCD data.
