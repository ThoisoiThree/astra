# Structure preparation at a specified pH

**Actions → Prepare structure · pH** restores missing protein heavy atoms and prepares standard protein residues
and water in residues touched by a named selection. The calculation is native Rust and
runs on the cancellable application background worker. The default is pH 7.0;
the supported input range is 0–14.

## Workflow

1. Choose a **Named selection**, pH and the neutral histidine tautomer: HIE (NE2–H, default) or HID (ND1–H).
2. Leave **Restore missing heavy atoms first** enabled to repair incomplete protein residues, or disable it to prepare H only.
3. Click **Prepare structure at this pH**.
4. Inspect the heavy-atom and protonation reports for prepared and skipped residues and ambiguous sites.
5. Run **Hydrogen bonds · AMOEBA 2018** with the desired named selection.

At least one selected atom selects its entire residue for preparation, including
when only a hydrogen is selected. Other residues retain their atom records and
coordinates. The full structure supplies covalent connectivity and steric
surroundings, so a selection boundary does not create an artificial terminus.
An empty or missing selection cannot start preparation. To prepare the whole
structure, create a named selection with the expression `all`.

Preparation replaces existing H in prepared residues and leaves all existing heavy-atom
coordinates fixed. Heavy repair and protonation have separate reports: a repaired
residue can still be skipped by protonation if its requested state is unsupported. A named
selection `prepared_h_ph_…` identifies the newly constructed hydrogens and is
saved in `.mol` scenes; `restored_heavy` identifies reconstructed heavy atoms. The detailed last-preparation report is shown in Actions
for the active document; it is not persisted in scene files.

Undo/redo restores the molecular topology, original atoms, selections and
formatting. Static selections retain surviving members and include new H when
their parent was selected. Reconstructed heavy atoms inherit membership from
their residue’s CA anchor; expression-based selections are reevaluated on the
new structure. Existing AMOEBA result objects are removed because they describe
a different protonation state; Undo restores them. Ordinary fixed-endpoint
measurements remain. Results are discarded if the scene changes during preparation.

## Heavy-atom reconstruction

The default repair step uses embedded ideal heavy-atom coordinates and bonds for
all 20 standard amino acids from the [wwPDB Chemical Component Dictionary](https://west.wwpdb.org/data/ccd).
The [data provenance](../data/amoeba/CCD.md) records the source and export procedure.
No download or Python interpreter is required at runtime.

Each residue requires existing N, CA and C anchors. The ideal template is aligned
to those anchors; every existing atom remains fixed. Missing side-chain atoms and
carbonyl O can be reconstructed. OXT is added at the last observed protein residue
of a chain only when its C has no external covalent bond. An internal chain break
is not interpreted as a free terminus.

A deterministic search samples rotations about eligible single bonds and refines
new coordinates against template bond lengths, angles, ring geometry,
stereochemistry and nearby steric contacts. Carbonyl oxygen placement uses an
available peptide partner to preserve the peptide plane. Lengths are in ångströms.
Residues with incompatible anchors, unknown atoms, incorrect stereochemistry,
new bond errors exceeding 0.15 Å or unresolved nonbonded contacts below 1.2 Å are
left unchanged and reported with a reason. Cancellation discards the result.

These coordinates are structural models, not experimentally determined positions
or AMOEBA-minimized conformers. The search does not use a statistical rotamer
library or guarantee the correct side-chain conformation. It does not reconstruct
missing residues, backbone gaps, ligands or nucleic-acid heavy atoms. Preparation
then assigns the pH-dependent AMOEBA template and builds explicit H; energetic
H-bond scoring remains a separate action.

## Tabulated pKa model

The independent-site values are from the
[EMBOSS `Epk.dat` table](https://emboss.bioinformatics.nl/cgi-bin/emboss/help/iep#data-files):

| Group | pKa |
| --- | ---: |
| Asp | 3.9 |
| Glu | 4.1 |
| His | 6.5 |
| Cys | 8.5 |
| Tyr | 10.1 |
| Lys | 10.8 |
| Arg | 12.5 |
| N-terminus | 8.6 |
| C-terminus | 3.6 |

The reported protonated fraction is `1 / (1 + 10^(pH − pKa))`. One discrete
structure is built: protonated at `pH <= pKa`, deprotonated otherwise. The tie at
pH = pKa is deterministic, not evidence that the protonated state is preferred.
Sites within one pH unit of pKa are flagged as ambiguous. Fractions are estimates
from independent sites; neither coupled titration nor local-environment pKa
shifts are predicted. Histidine tautomer choice is explicit because pH alone
cannot distinguish its two neutral tautomers. Disulfide cysteines use CYX and
are not treated as titratable thiols.

## Templates and unsupported structures

Heavy atoms, internal connectivity and external covalent bonds must match a
complete AMOEBA 2018 template. Protein protonation variants include ASH/ASP,
GLH/GLU, HIS/HID/HIE, CYS/CYD, TYR/TYD and LYS/LYD. The chosen template name is
stored as the residue name, making the prepared state reproducible on reload.

Missing atoms that could not be restored are listed. The tool does not add
capping groups at internal chain breaks. AMOEBA 2018 does not provide neutral arginine or
the neutral terminal alternatives used by this pH model. A residue requesting
one of those states is skipped, not silently assigned the charged template.
Single amino acids with both free termini may also lack a matching template.
Nucleic acids, ligands and monatomic species remain unchanged: this preparation
model covers protein titration and neutral water only. Existing prepared DNA/RNA
can still be analyzed by the AMOEBA energy tool.

Preparation of a complete residue does not guarantee its covalent component
will pass subsequent AMOEBA parameterization: a connected incomplete residue
can still exclude the whole component, as described in [AMOEBA analysis](amoeba.md).

## Hydrogen geometry

Template bonds determine the number, names and parent atoms of hydrogens.
Construction uses planar or tetrahedral valence geometry, a 104.52° water angle,
and ideal X–H lengths: C 1.09 Å, N 1.01 Å, O 0.96 Å, S 1.34 Å. Deterministic
sampling of rotatable H groups and water orientations reduces steric clashes
against surrounding heavy atoms and already constructed H. Degenerate geometry
causes the residue to be skipped.

This local steric search is not AMOEBA energy minimization or a prediction of
hydrogen-bond existence. It does not globally optimize interacting water/H
orientations, and steric clashes can remain in crowded structures. All added
coordinates are modelled. Energetic H-bond scoring is performed separately by
the native AMOEBA kernel; its energy definition and many-body decomposition are
unchanged.

## Validation

The Rust tests cover pH-dependent state changes, neutral histidine tautomers,
all 20 amino-acid side chains including rings, conflicting stereochemistry,
missing anchors, unsupported states, repeated preparation, cancellation, original
index remapping, preservation of heavy coordinates, and AMOEBA template/SCF
compatibility of a reconstructed 582-atom protein fixture and water cluster.
