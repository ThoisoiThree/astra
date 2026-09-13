# Portions copyright (c) 2006-2025 Stanford University and Simbios.
# Contributors: Pande Group
#  *
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#  *
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#  *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#  */

"""Developer-only OpenMM reference oracle. Never invoked by the Rust application.

Fixed-coordinate, nonperiodic AMOEBA 2018 group decoupling. See docs/amoeba.md.

OpenMM owns template assignment, permanent multipoles and buffered 14-7 vdW.
The mutual solver below implements the Reference multipole field equations so
A--B induced/induced coupling can actually be removed (ordinary OpenMM covalent
exclusions do NOT remove that coupling). No bonded energy is evaluated.
"""
import json
from itertools import product
import math
import sys

import numpy as np
import openmm as mm
from openmm import app, unit

COULOMB = 138.93545764438198  # kJ mol^-1 nm e^-2, OpenMM ONE_4PI_EPS0
DEBYE = 48.033324


def scalar(value):
    return value.value_in_unit_system(unit.md_unit_system) if unit.is_quantity(value) else value


def normalize(v):
    length = np.linalg.norm(v)
    if length < 1e-12:
        raise ValueError("Degenerate AMOEBA local frame")
    return v / length


def lab_moments(force, xyz):
    """Reference applyRotationMatrixToParticle, including ZThenX chirality."""
    charges, dipoles, quadrupoles, alphas, damps, tholes = [], [], [], [], [], []
    for i in range(len(xyz)):
        charge, dip, quad, axis, z, x, y, thole, damp, alpha = force.getMultipoleParameters(i)
        dip = np.array(scalar(dip), dtype=float)
        quad = np.array(scalar(quad), dtype=float).reshape(3, 3)
        if axis == force.ZThenX and y >= 0:
            volume = np.dot(np.cross(xyz[z]-xyz[y], xyz[x]-xyz[y]), xyz[i]-xyz[y])
            if volume < 0:
                reflection = np.diag([1., -1., 1.])
                dip = reflection @ dip
                quad = reflection @ quad @ reflection
        rotation = np.eye(3)
        if z >= 0:
            vz = normalize(xyz[z]-xyz[i])
            if axis == force.ZOnly:
                vx = np.array([1., 0., 0.]) if abs(vz[0]) < .866 else np.array([0., 1., 0.])
            else:
                vx = xyz[x]-xyz[i]
                if axis == force.Bisector:
                    vz = normalize(vz + normalize(vx))
                elif axis == force.ZBisect:
                    vx = normalize(normalize(vx) + normalize(xyz[y]-xyz[i]))
                elif axis == force.ThreeFold:
                    vx = normalize(vx)
                    vz = normalize(vz + vx + normalize(xyz[y]-xyz[i]))
            vx = normalize(vx-vz*np.dot(vz, vx))
            rotation = np.column_stack((vx, np.cross(vz, vx), vz))
        charges.append(scalar(charge))
        dipoles.append(rotation @ dip)
        quadrupoles.append(rotation @ quad @ rotation.T)
        alphas.append(scalar(alpha))
        damps.append(scalar(damp))
        tholes.append(scalar(thole))
    return tuple(np.asarray(v) for v in (charges, dipoles, quadrupoles, alphas, damps, tholes))


class MutualSolver:
    """Matrix-free SCF; O(N^2) work, O(N) persistent storage.

    Uses both AMOEBA d and p fields. Energy is -1/2 mu_d . E_p; it is
    equivalently -1/2 mu_p . E_d at convergence. The operator is symmetric
    after scaling by sqrt(alpha). All environment atoms participate.
    """
    def __init__(self, force, xyz, tolerance=1e-6, max_iterations=500):
        self.xyz = np.asarray(xyz)
        self.q, self.mu, self.quad, self.alpha, self.damp, self.thole = lab_moments(force, xyz)
        self.sqrt_alpha = np.sqrt(self.alpha)[:, None]
        self.maps = [[set(force.getCovalentMap(i, k)) for k in range(8)] for i in range(len(xyz))]
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def row(self, i, groups):
        r = self.xyz-self.xyz[i]
        distance = np.linalg.norm(r, axis=1)
        if np.any((distance < 1e-8) & (np.arange(len(r)) != i)):
            raise ValueError("Overlapping AMOEBA atoms")
        distance[i] = np.inf
        rr3, rr5, rr7 = distance**-3, 3*distance**-5, 15*distance**-7
        damping = self.damp[i]*self.damp
        valid = (damping != 0) & np.isfinite(distance)
        exponent = np.zeros(len(r))
        exponent[valid] = -np.minimum(self.thole[i], self.thole[valid])*(distance[valid]/damping[valid])**3
        active = valid & (exponent > -50)
        e = exponent[active]
        exp = np.exp(e)
        rr3[active] *= 1-exp
        rr5[active] *= 1-(1-e)*exp
        rr7[active] *= 1-(1-e+.6*e*e)*exp
        # Mask every cross term, without touching A/environment or B/environment.
        if groups is not None:
            a, b = groups
            excluded = b if i in a else a if i in b else ()
            idx = list(excluded)
            rr3[idx] = rr5[idx] = rr7[idx] = 0
        return r, rr3, rr5, rr7

    def fields(self, groups):
        ed = np.zeros_like(self.xyz)
        ep = np.zeros_like(self.xyz)
        for i in range(len(self.xyz)):
            r, rr3, rr5, rr7 = self.row(i, groups)
            qr = np.einsum('nij,nj->ni', self.quad, r)
            factor = rr3*self.q - rr5*np.sum(self.mu*r, axis=1) + rr7*np.sum(qr*r, axis=1)
            field = -r*factor[:, None] - self.mu*rr3[:, None] + qr*(2*rr5[:, None])
            ds = np.ones(len(r)); ps = np.ones(len(r))
            # Match setupScaleMaps: p12=p13=0, p14=1 (0.5 within p11).
            ps[list(self.maps[i][0] | self.maps[i][1])] = 0
            ps[list(self.maps[i][2] & self.maps[i][4])] = .5
            ds[list(self.maps[i][4])] = 0
            ed[i] = np.sum(field*ds[:, None], axis=0)
            ep[i] = np.sum(field*ps[:, None], axis=0)
        return ed, ep

    def apply(self, x, groups):
        dip = self.sqrt_alpha*x
        field = np.zeros_like(dip)
        for i in range(len(dip)):
            r, rr3, rr5, _ = self.row(i, groups)
            field[i] = np.sum(-dip*rr3[:, None] + r*(rr5*np.sum(r*dip, axis=1))[:, None], axis=0)
        return x-self.sqrt_alpha*field

    def converge(self, field, groups):
        rhs = self.sqrt_alpha*field
        x = np.zeros_like(rhs)
        residual = rhs.copy()
        direction = residual.copy()
        norm2 = np.sum(residual*residual)
        for iteration in range(self.max_iterations+1):
            error = DEBYE*np.sqrt(np.sum((self.sqrt_alpha*residual)**2)/len(x))
            if error <= self.tolerance:
                # Recompute the true residual rather than relying on CG recurrence.
                true = rhs-self.apply(x, groups)
                error = DEBYE*np.sqrt(np.sum((self.sqrt_alpha*true)**2)/len(x))
                if error <= self.tolerance:
                    return self.sqrt_alpha*x, iteration, float(error)
            if iteration == self.max_iterations:
                break
            product = self.apply(direction, groups)
            denom = np.sum(direction*product)
            if not np.isfinite(denom) or denom <= 0:
                raise ValueError("Mutual polarization is unstable (non-positive SCF operator)")
            step = norm2/denom
            x += step*direction
            residual -= step*product
            next_norm2 = np.sum(residual*residual)
            direction = residual+(next_norm2/norm2)*direction
            norm2 = next_norm2
        raise ValueError(f"Mutual polarization did not converge in {self.max_iterations} iterations")

    def energy(self, groups=None):
        ed, ep = self.fields(groups)
        mud, nd, rd = self.converge(ed, groups)
        mup, np_, rp = self.converge(ep, groups)
        energy = -.5*COULOMB*np.sum(mud*ep)
        reciprocal = -.5*COULOMB*np.sum(mup*ed)
        if abs(energy-reciprocal) > max(1e-5, abs(energy)*1e-6):
            raise ValueError("Polarization reciprocity check failed; tighten SCF tolerance")
        return float(energy), max(nd, np_), max(rd, rp), mud


def clone(force):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force))


def evaluate_forces(forces, xyz):
    system = mm.System()
    for _ in xyz:
        system.addParticle(1.)
    for k, force in enumerate(forces):
        force.setForceGroup(k)
        system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions(xyz*unit.nanometer)
    values = [context.getState(getEnergy=True, groups={k}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for k in range(len(forces))]
    del context, integrator
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite AMOEBA energy")
    return values


def permanent_force(force, groups=None):
    result = clone(force)
    for i in range(result.getNumMultipoles()):
        params = list(result.getMultipoleParameters(i))
        params[-1] = 0.0
        result.setMultipoleParameters(i, *params)
    if groups is not None:
        a, b = groups
        for side, other in ((a, b), (b, a)):
            for i in side:
                # Remove from later maps so a 1-4/1-5 scale cannot overwrite zero.
                for k in range(1, 4):
                    result.setCovalentMap(i, k, sorted(set(result.getCovalentMap(i, k))-other))
                result.setCovalentMap(i, 0, sorted(set(result.getCovalentMap(i, 0)) | other))
    return result


def vdw_force(force, groups=None):
    result = clone(force)
    if groups is not None:
        a, b = groups
        for side, other in ((a, b), (b, a)):
            for i in side:
                result.setParticleExclusions(i, sorted(set(result.getParticleExclusions(i)) | other))
    return result


def decompose(multipole, vdw, xyz, pairs, tolerance=1e-6, max_iterations=500):
    solver = MutualSolver(multipole, xyz, tolerance, max_iterations)
    full_pol, iterations, residual, dipoles = solver.energy()
    full_perm, full_vdw, reference = evaluate_forces([permanent_force(multipole), clone(vdw), clone(multipole)], xyz)
    # Independent oracle on every structure, not just the test fixtures.
    if abs(full_perm+full_pol-reference) > max(.002, abs(reference)*1e-7):
        raise ValueError(f"AMOEBA/OpenMM polarization validation failed: {full_perm+full_pol-reference:.6g} kJ/mol")
    cache = {}
    for a, b in pairs:
        key = tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))
        if key in cache:
            continue
        groups = (set(a), set(b))
        if groups[0] & groups[1] or not all(groups):
            raise ValueError("Functional groups must be nonempty and disjoint")
        dec_pol, n, r, _ = solver.energy(groups)
        dec_perm, dec_vdw = evaluate_forces([permanent_force(multipole, groups), vdw_force(vdw, groups)], xyz)
        perm, vd, pol = full_perm-dec_perm, full_vdw-dec_vdw, full_pol-dec_pol
        cache[key] = dict(delta_energy=perm+vd+pol, permanent=perm, vdw=vd, polarization=pol,
                          scf_iterations=n, scf_residual_debye=r)
    return cache, dict(full_permanent=full_perm, full_vdw=full_vdw, full_polarization=full_pol,
                       scf_iterations=iterations, scf_residual_debye=residual)


def parameterize(request, forcefield=None):
    """Template provider boundary: a future Poltype/OpenMM XML provider goes here.

    Existing H coordinates determine protonation; no H are invented or moved.
    A component with an unmatched residue is excluded intact, avoiding artificial
    termini or changed local frames at the edge of a parameterized fragment.
    """
    ff = forcefield or app.ForceField('amoeba2018.xml')
    app.PDBFile._loadNameReplacementTables()
    top = app.Topology()
    # OpenMM requires contiguous chains/residues even when the input file
    # interleaves ATOM/HETATM records or appends hydrogens at the end. Preserve
    # first-appearance order within the hierarchy, and map all indices explicitly.
    grouped = {}
    for original_index, item in enumerate(request['atoms']):
        key = (item['residue_number'], item['insertion_code'], item['residue_name'])
        grouped.setdefault(item['chain'], {}).setdefault(key, []).append((original_index, item))
    atoms, top_to_input = [], []
    input_to_top = [None]*len(request['atoms'])
    for chain_id, residues in grouped.items():
        chain = top.addChain(chain_id)
        for (number, insertion_code, residue_name), entries in residues.items():
            canonical = app.PDBFile._residueNameReplacements.get(residue_name, residue_name)
            variants = {'CYD': 'CYS', 'LYD': 'LYS', 'TYD': 'TYR'}
            canonical = variants.get(canonical, canonical)
            if canonical[:2] in ('RA', 'RC', 'RG', 'RU') and canonical[2:] in ('', 'N', '3', '5'):
                canonical = canonical[1]
            elif canonical[:2] in ('DA', 'DC', 'DG', 'DT') and canonical[2:] in ('N', '3', '5'):
                canonical = canonical[:2]
            residue = top.addResidue(canonical, chain, str(number), insertion_code)
            for original_index, item in entries:
                element = app.Element.getBySymbol(item['element']) if item['element'] != '?' else None
                name = app.PDBFile._atomNameReplacements.get(canonical, {}).get(item['name'], item['name'])
                atom = top.addAtom(name, element, residue)
                input_to_top[original_index] = atom.index
                top_to_input.append(original_index)
                atoms.append(atom)
    # Standard bond definitions establish chemical topology independently of
    # the viewer's distance-based covalent bond guesses.
    top.createStandardBonds()
    bonded = {tuple(sorted((a.index, b.index))) for a, b in top.bonds()}
    for original_a, original_b in request['bonds']:
        a, b = input_to_top[original_a], input_to_top[original_b]
        # Viewer bond inference may connect a metal to coordinating water. Those
        # are not covalent topology edges in the AMOEBA ion templates.
        if any(atom.element is not None and atom.element.symbol not in ('H', 'C', 'N', 'O', 'P', 'S')
               and len(list(atom.residue.atoms())) == 1 for atom in (atoms[a], atoms[b])):
            continue
        if atoms[a].name == atoms[b].name == 'SG' and atoms[a].residue != atoms[b].residue:
            bonded.add(tuple(sorted((a, b))))
        elif atoms[a].residue.name not in top._standardBonds or atoms[b].residue.name not in top._standardBonds:
            bonded.add(tuple(sorted((a, b))))
    # Rebuild to suppress links across actual chain breaks; this affects only
    # covalent topology, never the H-bond existence criterion.
    xyz = np.array([request['atoms'][i]['position'] for i in top_to_input], dtype=float)*.1
    top._bonds = []
    for a, b in sorted(bonded):
        if atoms[a].residue != atoms[b].residue and np.linalg.norm(xyz[a]-xyz[b]) > .25:
            continue
        top.addBond(atoms[a], atoms[b])
    unmatched = set(ff.getUnmatchedResidues(top))
    excluded = set(unmatched)
    changed = True
    while changed:
        changed = False
        for a, b in top.bonds():
            if (a.residue in excluded) != (b.residue in excluded):
                excluded.update((a.residue, b.residue)); changed = True
    statuses = []
    for residue in top.residues():
        if residue in excluded:
            reason = ('No exact AMOEBA 2018 template / missing explicit H or unresolved protonation'
                      if residue in unmatched else 'Covalently connected to an unparameterized residue; boundary not capped')
            statuses.append(dict(residue=f'{residue.chain.id}:{residue.id}{residue.insertionCode}:{residue.name}',
                                 atoms=[top_to_input[a.index] for a in residue.atoms()], status='unparameterized', reason=reason))
    included_top = [a.index for a in atoms if a.residue not in excluded]
    included = [top_to_input[i] for i in included_top]
    if not included:
        return None, statuses
    modeller = app.Modeller(top, xyz*unit.nanometer)
    modeller.delete(list(excluded))
    templates = ff.getMatchingTemplates(modeller.topology)
    for residue, template in zip(modeller.topology.residues(), templates):
        statuses.append(dict(residue=f'{residue.chain.id}:{residue.id}{residue.insertionCode}:{residue.name}',
                             atoms=[included[a.index] for a in residue.atoms()], status='parameterized', reason=template.name))
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff,
                             polarization='mutual', mutualInducedTargetEpsilon=request['tolerance'],
                             constraints=None, rigidWater=False, removeCMMotion=False)
    multipole = next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    vdw = next(f for f in system.getForces() if isinstance(f, mm.AmoebaVdwForce))
    multipole.setMutualInducedMaxIterations(request['max_iterations'])
    return (modeller.topology, xyz[included_top], included, clone(multipole), clone(vdw), templates), statuses


def chemical_candidates(top, xyz, included, multipole, selected, cutoff, templates):
    atoms = list(top.atoms())
    neighbors = [set() for _ in atoms]
    for a, b in top.bonds():
        neighbors[a.index].add(b.index); neighbors[b.index].add(a.index)
    # AMOEBA's template-derived polarization domains define the operational
    # functional groups. Take transitive closure, retaining their full atoms
    # even when the named selection contains only the D/H/A atoms.
    parent = list(range(len(atoms)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(atoms)):
        for j in multipole.getCovalentMap(i, multipole.PolarizationCovalent11):
            parent[root(j)] = root(i)
    groups = {}
    for i in range(len(atoms)):
        groups.setdefault(root(i), set()).add(i)
    def symbol(i):
        return atoms[i].element.symbol if atoms[i].element else '?'
    def accepts(i):
        atom = atoms[i]
        element = symbol(i)
        hydrogens = any(symbol(j) == 'H' for j in neighbors[i])
        if element in ('F', 'Cl', 'Br', 'I'):
            return not neighbors[i] and scalar(multipole.getMultipoleParameters(i)[0]) < 0
        if element == 'O':
            # A protonated carboxyl/phosphate oxygen is not an acceptor.
            acidic = any(symbol(j) == 'P' or (symbol(j) == 'C' and sum(symbol(k) == 'O' for k in neighbors[j]) >= 2) for j in neighbors[i])
            return not (hydrogens and acidic)
        if element == 'S':
            return len(neighbors[i]) <= 2
        if element != 'N':
            return False
        template = templates[atom.residue.index].name
        residue = template[1:] if len(template) == 4 and template[:1] in ('N', 'C') else template
        # Standard nucleic acid acceptor nitrogens; the template/H pattern
        # resolves protonated tautomers rather than an angular heuristic.
        base = atom.residue.name
        allowed = {'A': {'N1', 'N3', 'N7'}, 'DA': {'N1', 'N3', 'N7'},
                   'G': {'N3', 'N7'}, 'DG': {'N3', 'N7'},
                   'C': {'N3'}, 'DC': {'N3'}, 'U': set(), 'DT': set(), 'T': set()}
        if base in allowed:
            return atom.name in allowed[base] and not hydrogens
        if residue in ('HID', 'HIE', 'HIS', 'HIP') and atom.name in ('ND1', 'NE2'):
            return not hydrogens
        # Backbone/sidechain amides, guanidinium and protonated amines do not accept.
        if atom.name == 'N' or residue in ('ARG', 'ASN', 'GLN', 'TRP') or len(neighbors[i]) >= 4:
            return False
        return residue in ('LYD',) and atom.name == 'NZ'
    acceptors = [i for i in range(len(atoms)) if included[i] in selected and accepts(i)]
    cells = {}
    def cell(i):
        return tuple(np.floor(xyz[i]*10/cutoff).astype(int))
    for a in acceptors:
        cells.setdefault(cell(a), []).append(a)
    result = []
    for d in range(len(atoms)):
        if included[d] not in selected or symbol(d) not in ('N', 'O', 'S'):
            continue
        center = cell(d)
        nearby = sorted(a for offset in product((-1, 0, 1), repeat=3)
                        for a in cells.get(tuple(center[k]+offset[k] for k in range(3)), []))
        for h in sorted(neighbors[d]):
            if symbol(h) != 'H' or included[h] not in selected:
                continue
            for a in nearby:
                if a == d or a in neighbors[d] or a in neighbors[h] or root(a) == root(d):
                    continue
                da = float(np.linalg.norm(xyz[d]-xyz[a])*10)
                if da > cutoff:
                    continue
                ha = float(np.linalg.norm(xyz[h]-xyz[a])*10)
                u, v = normalize(xyz[d]-xyz[h]), normalize(xyz[a]-xyz[h])
                angle = math.degrees(math.acos(float(np.clip(np.dot(u, v), -1, 1))))
                result.append(dict(donor=included[d], hydrogen=included[h], acceptor=included[a],
                    donor_group=sorted(included[j] for j in groups[root(d)]),
                    acceptor_group=sorted(included[j] for j in groups[root(a)]),
                    donor_position=(xyz[d]*10).tolist(), acceptor_position=(xyz[a]*10).tolist(),
                    da_distance=da, ha_distance=ha, dha_angle=angle, parameterization_status='parameterized',
                    _groups=(groups[root(d)], groups[root(a)])))
    return result


def analyze(request):
    if not (0 < request['tolerance'] <= .001) or not (1 <= request['max_iterations'] <= 10000):
        raise ValueError('Invalid SCF settings')
    if not (5 <= request['cutoff'] <= 12):
        raise ValueError('Neighbor cutoff must be 5–12 Å')
    prepared, statuses = parameterize(request)
    report = dict(version=1, selection=request['selection'], force_field='AMOEBA 2018 / OpenMM 8.4.0',
                  cutoff=request['cutoff'], tolerance=request['tolerance'], max_iterations=request['max_iterations'],
                  energy_threshold=0., candidates=[], residues=statuses, full_permanent=0., full_vdw=0.,
                  full_polarization=0., scf_iterations=0, scf_residual_debye=0.)
    if prepared is None:
        return report
    top, xyz, included, multipole, vdw, templates = prepared
    candidates = chemical_candidates(top, xyz, included, multipole, set(request['selected']), request['cutoff'], templates)
    scores, diagnostics = decompose(multipole, vdw, xyz, [c['_groups'] for c in candidates], request['tolerance'], request['max_iterations'])
    for candidate in candidates:
        a, b = candidate.pop('_groups')
        key = tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))
        candidate.update(scores[key])
    report.update(diagnostics)
    report['candidates'] = candidates
    return report


if __name__ == '__main__':
    try:
        if mm.__version__ != '8.4':
            raise ValueError(f'Validated backend requires OpenMM 8.4.0, found {mm.__version__}')
        result = analyze(json.load(sys.stdin))
        json.dump(result, sys.stdout, allow_nan=False)
    except Exception as error:
        print(f'AMOEBA analysis failed: {error}', file=sys.stderr)
        sys.exit(1)
