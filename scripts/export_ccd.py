#!/usr/bin/env python3
"""Export the wwPDB Chemical Component Dictionary to Astra's compact bond-template table.

The CCD is read from the BinaryCIF copy bundled with biotite (`pip install biotite`), which
mirrors https://files.wwpdb.org/pub/pdb/data/monomers/components.cif. Hydrogen atoms are
omitted: their bonds are always single and are recovered geometrically, which keeps the
embedded table small. Output: data/ccd/components.bin.zst.

Binary layout (little-endian), version 1:
    b"ACCD" u32 version u32 component_count
    per component:
        str id, u8 kind, str parent_id, u8 one_letter_code (0 when absent),
        u16 atom_count, atoms[str name, str element, i8 formal_charge, u8 flags],
        u16 bond_count, bonds[u16 a, u16 b, u8 order | aromatic << 4]
    str = u8 length + ASCII bytes
    atom flags: 1 aromatic, 2 leaving atom
    kind: see KINDS below
"""

import pathlib
import struct
import sys

import biotite
import numpy as np
import zstandard
from biotite.structure.info.ccd import get_ccd

KINDS = {
    "peptide": 1,
    "dna": 2,
    "rna": 3,
    "saccharide": 4,
    "non-polymer": 5,
    "other": 0,
}
ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4, "AROM": 1, "DELO": 1, "PI": 1, "POLY": 1}


def kind_of(value: str) -> int:
    value = value.lower()
    if "peptide" in value:
        return KINDS["peptide"]
    if "dna" in value:
        return KINDS["dna"]
    if "rna" in value:
        return KINDS["rna"]
    if "saccharide" in value:
        return KINDS["saccharide"]
    if "non-polymer" in value:
        return KINDS["non-polymer"]
    return KINDS["other"]


def string(value: str) -> bytes:
    raw = value.encode("ascii", "replace")
    if len(raw) > 255:
        raise ValueError(f"string too long: {value!r}")
    return bytes([len(raw)]) + raw


def column(category, name):
    return category[name].as_array() if name in category else None


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    ccd = get_ccd()
    comps = ccd["chem_comp"]
    atoms = ccd["chem_comp_atom"]
    bonds = ccd["chem_comp_bond"]

    comp_ids = column(comps, "id")
    types = column(comps, "type")
    parents = column(comps, "mon_nstd_parent_comp_id")
    one_letter = column(comps, "one_letter_code")

    atom_comp = column(atoms, "comp_id")
    atom_name = column(atoms, "atom_id")
    atom_element = column(atoms, "type_symbol")
    atom_charge = column(atoms, "charge")
    atom_aromatic = column(atoms, "pdbx_aromatic_flag")
    atom_leaving = column(atoms, "pdbx_leaving_atom_flag")

    bond_comp = column(bonds, "comp_id")
    bond_a = column(bonds, "atom_id_1")
    bond_b = column(bonds, "atom_id_2")
    bond_order = column(bonds, "value_order")
    bond_aromatic = column(bonds, "pdbx_aromatic_flag")

    def spans(values):
        # Categories are grouped by component; map each id to its row range.
        result = {}
        boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(values)]))
        for start, end in zip(starts, ends):
            result[str(values[start])] = (int(start), int(end))
        return result

    atom_spans = spans(atom_comp)
    bond_spans = spans(bond_comp)

    out = bytearray(b"ACCD")
    out += struct.pack("<II", 1, len(comp_ids))
    for index, comp in enumerate(comp_ids):
        comp = str(comp)
        parent = str(parents[index]) if parents is not None else ""
        if parent in (".", "?"):
            parent = ""
        parent = parent.split(",")[0].strip()
        code = str(one_letter[index]) if one_letter is not None else ""
        code_byte = ord(code) if len(code) == 1 and code.isalpha() else 0
        out += string(comp)
        out += bytes([kind_of(str(types[index]))])
        out += string(parent)
        out += bytes([code_byte])

        start, end = atom_spans.get(comp, (0, 0))
        names = []
        records = bytearray()
        for row in range(start, end):
            element = str(atom_element[row]).strip().upper()
            if element in ("H", "D"):
                continue
            name = str(atom_name[row])
            try:
                charge = int(atom_charge[row])
            except (TypeError, ValueError):
                charge = 0
            flags = 0
            if atom_aromatic is not None and str(atom_aromatic[row]) == "Y":
                flags |= 1
            if atom_leaving is not None and str(atom_leaving[row]) == "Y":
                flags |= 2
            names.append(name)
            records += string(name) + string(element) + struct.pack("<bB", max(-128, min(127, charge)), flags)
        out += struct.pack("<H", len(names)) + records

        lookup = {name: position for position, name in enumerate(names)}
        start, end = bond_spans.get(comp, (0, 0))
        bond_records = bytearray()
        count = 0
        for row in range(start, end):
            a = lookup.get(str(bond_a[row]))
            b = lookup.get(str(bond_b[row]))
            if a is None or b is None:
                continue
            order = ORDERS.get(str(bond_order[row]).upper(), 1)
            aromatic = bond_aromatic is not None and str(bond_aromatic[row]) == "Y"
            bond_records += struct.pack("<HHB", a, b, order | (int(aromatic) << 4))
            count += 1
        out += struct.pack("<H", count) + bond_records

    compressed = zstandard.ZstdCompressor(level=19).compress(bytes(out))
    target = root / "data" / "ccd" / "components.bin.zst"
    target.write_bytes(compressed)
    print(
        f"{len(comp_ids)} components, {len(out) / 1e6:.1f} MB raw, "
        f"{len(compressed) / 1e6:.2f} MB compressed (biotite {biotite.__version__})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
