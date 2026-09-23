# Space groups

`spacegroups.tsv` lists the crystallographic space-group settings of the International
Tables for Crystallography, Vol. A (number, Hermann–Mauguin and Hall symbols and the
general-position operators as coordinate triplets). Astra uses it to build unit cells and
symmetry mates from CRYST1 or `_cell`/`_symmetry` records.

It is generated from the tables in [gemmi](https://gemmi.readthedocs.io/) with:

```sh
pip install gemmi
python3 scripts/export_spacegroups.py
```
