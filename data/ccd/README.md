# Chemical Component Dictionary

`components.bin.zst` is a compact export of the wwPDB Chemical Component Dictionary (CCD):
for every component its polymer class, standard parent, one-letter code, heavy atoms (name,
element, formal charge, aromatic and leaving-atom flags) and bonds with their orders.
Hydrogen atoms are omitted; their bonds are always single and are recovered geometrically.

Astra uses it to connect residues by their chemical templates, including double, triple and
aromatic bonds, instead of guessing connectivity from distances.

The data come from the copy of `components.cif` bundled with
[biotite](https://www.biotite-python.org/) 1.6.0 and are regenerated with:

```sh
pip install biotite zstandard
python3 scripts/export_ccd.py
```

The CCD is maintained by the wwPDB and distributed without restriction under the
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) dedication
(see https://www.wwpdb.org/about/usage-policies).
