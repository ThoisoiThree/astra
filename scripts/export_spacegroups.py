#!/usr/bin/env python3
"""Export crystallographic space-group settings to data/symmetry/spacegroups.tsv.

Uses gemmi's tables (International Tables for Crystallography, Vol. A; `pip install gemmi`).
Columns: number, Hermann-Mauguin symbol, extended symbol (with setting), short symbol,
Hall symbol and the general-position operators as coordinate triplets separated by ';'.
"""

import pathlib

import gemmi


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    lines = [f"# gemmi {gemmi.__version__}: number\thm\txhm\tshort\thall\toperators"]
    for sg in gemmi.spacegroup_table():
        operators = ";".join(op.triplet() for op in sg.operations())
        lines.append("\t".join([str(sg.number), sg.hm, sg.xhm(), sg.short_name(), sg.hall, operators]))
    (root / "data" / "symmetry" / "spacegroups.tsv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
