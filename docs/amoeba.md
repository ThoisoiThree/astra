# AMOEBA 2018 hydrogen-bond analysis

**Info → Enable experimental features** reveals the **Experimental features**
section in Actions. This switch is off by default on each application launch.

Actions → **Hydrogen bonds · AMOEBA 2018** takes a named selection. The calculation
runs on the existing background worker and can be canceled. Its result is a
separate measurement object in the molecule manager, with shared color,
visibility, thickness, label size, rename and delete controls. Undo/redo and `.mol`
files retain the complete report, including candidates hidden by the energy filter.

## Implementation and dependencies

The complete nonbonded calculation runs in native Rust inside the application.
AMOEBA 2018 templates and parameters are embedded in the executable from
`data/amoeba/amoeba2018.json`; analysis requires no Python interpreter, OpenMM
installation, external worker process, or network access.

The data is a reproducible export of OpenMM 8.4.0's `amoeba2018.xml`, with
standard bond definitions and PDB naming aliases. Only parameter data is imported;
Rust performs topology matching, local-frame rotation, electrostatics, vdW,
mutual polarization and group decoupling. See `data/amoeba/LICENSE` for attribution.

## Preparation and scope

Coordinates are fixed, in Å in the application and nm inside the numerical kernel. **Explicit
hydrogens are required and are never added or moved.** Exact force-field template
matching identifies protonation/tautomer states from the hydrogen inventory and
chemical connectivity; this is not a pKa predictor. Use [Actions → Prepare structure · pH](protonation.md) for native protein/water
preparation, or prepare the structure externally before analysis. Incomplete, unmatched or unsupported
residues are reported as `unparameterized` and have no energetic H-bond results.

The embedded standard residue bond definitions and atom-name aliases construct the
chemical topology. Disulfides and unsupported-residue connectivity use the input
bond graph. A 2.5 Å check prevents a covalent link across a chain break; it is not
a hydrogen-bond criterion. Viewer-inferred metal coordination bonds are not
interpreted as covalent AMOEBA ion bonds.

If an unparameterized residue is covalently attached to other residues, the
entire connected component is excluded and every excluded residue is listed.
This conservative boundary avoids introducing artificial termini, capping atoms
or changing the surviving atoms' templates/local frames. Thus an incomplete
residue can exclude a protein chain. The full energy below always means the
**entire remaining parameterized structure**, including atoms outside the named
selection. Excluded components do not polarize that environment. An all-excluded
input creates an empty result object with the parameterization report.

Parameters come from OpenMM's `amoeba2018.xml` templates for proteins, DNA, RNA,
water and ions. No generic charges, element-radius vdW estimates, or geometric
H-bond substitution are used. Rendering radii are unrelated to energy parameters.
`molecule::amoeba::parameters::ForceField` is the provider boundary for future
Poltype 2 / prepared parameter import. `ForceField::from_json` loads the same
validated template format as the built-in data. A ligand adapter must supply
complete topology, multipole-frame rules, polarization domains, vdW parameters
and donor/acceptor chemistry. A ligand parameter generator and import UI are not
part of this implementation.

## Candidates and groups

D, H and A must all be in the named selection. D is a nitrogen, oxygen or sulfur
with an explicit bonded hydrogen. Acceptor chemistry handles oxygen/sulfur sites,
carboxyl/phosphate protonation, template-identified histidine nitrogens, neutral
lysine, the standard nucleobase acceptor nitrogens and monatomic halide anions. Amide nitrogens,
guanidinium and protonated amines are excluded. Covalently adjacent sites and
sites in the same group are excluded.

The operational **functional groups are AMOEBA polarization domains**, the
transitive closure of each template-assigned `PolarizationCovalent11` map. The
whole domain is used even if some of its atoms lie outside the named selection.
These domains are reproducible template-derived partitions, not just D/H and A
point sites; a domain can be larger than a conventional named chemical group.
Both groups' exact contiguous atom indices are stored in each candidate.

A spatial grid finds candidate D–A pairs within 6 Å by default (configurable from
5 to 12 Å). This only bounds the search. Neither D–H–A angle nor H–A distance
classifies a bond. Those quantities are reported as diagnostics. All candidate
scores are retained, including unfavorable ones.

## Energy definition

The nonperiodic, vacuum, fixed-coordinate Hamiltonian is:

```
E_nonbonded = E_permanent_multipoles + E_mutual_polarization + E_vdW
ΔE_pair(A,B) = E_full − E_AB_decoupled
```

In the decoupled evaluation, **only A↔B cross terms** are removed:

- Permanent electrostatic interactions, with multipoles through quadrupoles.
- Permanent-field contributions in both AMOEBA d- and p-scaled polarization fields.
- Both directions of the induced-dipole coupling operator between A and B.
- Buffered 14-7 vdW interactions, preserving template combining rules, reduction
  sites, exclusions and all remaining interactions.

A/environment, B/environment and environment/environment interactions are
unchanged. Local frames and all positions are held fixed. Induced dipoles on
**all** atoms reconverge in each decoupled system. Ordinary OpenMM covalent maps
alone cannot implement this experiment, because induced/induced coupling remains
active. The Rust kernel implements the Reference Thole-damped field operator and SCF
solve explicitly, along with permanent multipole and buffered 14-7 vdW energies.
No bonded, implicit-solvent, PME or periodic-image terms are evaluated.

The mutual equations are solved by conjugate gradients after symmetric
`sqrt(alpha)` scaling, for both the AMOEBA d and p fields. The polarization energy
is `−(1/2) k_e mu_d · E_p`, equivalently `−(1/2) k_e mu_p · E_d` at convergence.
The RMS dipole-equation residual is checked in Debye (default `1e-6 D`, maximum
500 iterations). A nonpositive operator, nonconvergence, failed reciprocity or
nonfinite result fails the analysis. Independent OpenMM Reference results are committed as regression fixtures;
runtime convergence and reciprocity checks do not depend on OpenMM.

Stored contributions, all in kJ/mol, are:

```
E_perm contribution = E_perm(full) − E_perm(decoupled)
E_vdW contribution  = E_vdW(full)  − E_vdW(decoupled)
polarization response = E_pol(full) − E_pol(decoupled)
ΔE_pair = sum of these three contributions
```

**Negative ΔE is stabilizing**: allowing A↔B coupling lowers the total energy.
This is a group-decoupling score in the specified environment, not an isolated
pair Coulomb energy, a binding free energy or an additive decomposition over
all candidate H-bonds. Candidates sharing the same group pair share a cached
score. Summing these scores double-counts many-body response and group terms.

The display default is `ΔE < 0 kJ/mol`. The editable energy threshold is only a
visual filter on the continuous scores. Each unique unordered D–A atom pair is
rendered once, with its D–A distance in Å, avoiding overlapping copies for
multiple donor hydrogens. Candidate details retain D, H, A, both group atom lists,
all energy components, D–A and H–A distances, angle, status and SCF diagnostics.
Hover a candidate row for these details.

## Cost and validation

This is a CPU reference analysis, with O(N²) work per SCF iteration and linear
persistent solver storage. There is no electrostatic distance cutoff. A separate
SCF solve is required for every distinct candidate group pair. Large solvated
structures can be slow; using a smaller named selection reduces candidate
solves but does not remove their environment. Cancellation is checked during topology preparation, pair loops and SCF
iterations on the application background thread. Results from scenes edited while a calculation runs are discarded.

The normal Rust suite requires no additional dependencies:

```sh
cargo test molecule::amoeba
cargo test --test amoeba
```

Committed OpenMM Reference fixtures exercise every one of the 134 AMOEBA 2018
residue templates (with real caps where necessary), a 582-atom protein, and a
water trimer. Tests compare lab-frame moments and each nonbonded energy component,
including the many-body environment response to group decoupling. Synthetic
coordinates in the template cases test the equations and parameter assignment;
they are not proposed molecular conformations. Further tests cover interleaved
atom records, original index mapping, missing atoms, nonconvergence, selection
scope, display filtering and scene/style round trips.

Python is used only by developers who regenerate parameter/oracle data or run
the independent reference checks:

```sh
python3 -m venv .venv-amoeba
.venv-amoeba/bin/python -m pip install -r tests/reference/amoeba/requirements.txt
.venv-amoeba/bin/python tests/reference/amoeba/export_parameters.py
.venv-amoeba/bin/python tests/reference/amoeba/export_oracles.py
.venv-amoeba/bin/python -m unittest discover -s tests/reference/amoeba -v
```

On Windows use `.venv-amoeba\Scripts\python.exe`. These scripts are neither
embedded nor invoked by the application or Cargo build.

Implementation references:

- [OpenMM AMOEBA force-field documentation](https://docs.openmm.org/8.0.0/userguide/application/02_running_sims.html#amoeba)
- [AmoebaMultipoleForce API](https://docs.openmm.org/development/api-python/generated/openmm.openmm.AmoebaMultipoleForce.html)
- Repository reference `reference/AmoebaReferenceMultipoleForce.cpp`, especially
  `setupScaleMaps`, `applyRotationMatrixToParticle`, `getAndScaleInverseRs`,
  `calculateFixedMultipoleFieldPairIxn` and `calculateInducedDipolePairIxns`.
  The adapted equations and exported parameter data retain the source's MIT notice
  in `data/amoeba/LICENSE`.
