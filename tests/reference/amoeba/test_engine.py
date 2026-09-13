import unittest
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app, unit
from reference_engine import MutualSolver, decompose, evaluate_forces, permanent_force, analyze, clone, parameterize, lab_moments


def waters(count=3):
    top = app.Topology(); chain = top.addChain('A'); xyz=[]
    for i in range(count):
        residue = top.addResidue('HOH', chain, str(i+1))
        o = top.addAtom('O', app.element.oxygen, residue)
        h1 = top.addAtom('H1', app.element.hydrogen, residue)
        h2 = top.addAtom('H2', app.element.hydrogen, residue)
        top.addBond(o, h1); top.addBond(o, h2)
        xyz.extend(np.array([[0,0,0],[.09572,0,0],[-.0239987,.0926627,0]])+[i*.29, i*.015, 0])
    xyz = np.asarray(xyz)
    system=app.ForceField('amoeba2018.xml').createSystem(top, nonbondedMethod=app.NoCutoff, polarization='mutual', mutualInducedTargetEpsilon=1e-8)
    mp=next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
    vd=next(f for f in system.getForces() if isinstance(f, mm.AmoebaVdwForce))
    return top, xyz, clone(mp), clone(vd)


class EnergyTests(unittest.TestCase):
    def test_interleaved_chains_and_atoms_preserve_original_indices_and_energy(self):
        top, xyz, _, _ = waters(3)
        atoms = [dict(name=a.name, element=a.element.symbol,
                      chain='A' if a.index < 6 else 'B', residue_number=int(a.residue.id),
                      insertion_code='', residue_name='HOH', position=(xyz[a.index]*10).tolist())
                 for a in top.atoms()]
        atoms.append(dict(name='C1', element='C', chain='C', residue_number=4,
                          insertion_code='', residue_name='LIG', position=[20, 0, 0]))
        bonds = [[a.index, b.index] for a, b in top.bonds()]
        base = dict(atoms=atoms, bonds=bonds, selection='waters', selected=list(range(9)),
                    cutoff=6., tolerance=1e-8, max_iterations=500)
        expected = analyze(base)
        expected_candidates = {(c['donor'], c['hydrogen'], c['acceptor']): c
                               for c in expected['candidates']}
        for order in ([0,1,2,6,7,8,9,3,4,5], [0,3,6,9,1,4,7,2,5,8]):
            with self.subTest(order=order):
                old_to_new = {old: new for new, old in enumerate(order)}
                actual = analyze(dict(base, atoms=[atoms[i] for i in order],
                                      bonds=[[old_to_new[a], old_to_new[b]] for a,b in bonds],
                                      selected=[old_to_new[i] for i in range(9)]))
                self.assertEqual(len(actual['candidates']), len(expected_candidates))
                for c in actual['candidates']:
                    key = tuple(order[c[k]] for k in ('donor', 'hydrogen', 'acceptor'))
                    ref = expected_candidates[key]
                    for field in ('delta_energy', 'permanent', 'vdw', 'polarization',
                                  'da_distance', 'ha_distance', 'dha_angle'):
                        self.assertAlmostEqual(c[field], ref[field], places=6)
                    for field in ('donor_group', 'acceptor_group'):
                        self.assertEqual(sorted(order[i] for i in c[field]), ref[field])
                    for field in ('donor_position', 'acceptor_position'):
                        np.testing.assert_allclose(c[field], ref[field], atol=1e-12)
                actual_status = {(r['residue'], r['status']): sorted(order[i] for i in r['atoms'])
                                 for r in actual['residues']}
                expected_status = {(r['residue'], r['status']): r['atoms'] for r in expected['residues']}
                self.assertEqual(actual_status, expected_status)
                # Selecting only one water still produces no inter-water candidates.
                selected = [old_to_new[i] for i in range(3)]
                request = dict(base, atoms=[atoms[i] for i in order],
                               bonds=[[old_to_new[a], old_to_new[b]] for a,b in bonds], selected=selected)
                self.assertEqual(analyze(request)['candidates'], [])

    def test_reordered_input_bonds_preserve_excluded_components(self):
        top, xyz, _, _ = waters(2)
        atoms = [dict(name=a.name, element=a.element.symbol, chain='A',
                      residue_number=int(a.residue.id), insertion_code='', residue_name='HOH',
                      position=(xyz[a.index]*10).tolist()) for a in top.atoms()]
        atoms.append(dict(name='C1', element='C', chain='B', residue_number=1,
                          insertion_code='', residue_name='LIG', position=[3.0, 1.5, 0]))
        bonds = [[a.index, b.index] for a,b in top.bonds()] + [[3, 6]]
        request = dict(atoms=atoms, bonds=bonds, tolerance=1e-8, max_iterations=500)
        expected, statuses = parameterize(request)
        self.assertEqual(expected[2], [0, 1, 2])
        order = [6, 0, 3, 1, 4, 2, 5]
        old_to_new = {old: new for new, old in enumerate(order)}
        actual, reordered_statuses = parameterize(dict(request, atoms=[atoms[i] for i in order],
            bonds=[[old_to_new[a], old_to_new[b]] for a,b in bonds]))
        self.assertEqual([order[i] for i in actual[2]], expected[2])
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-12)
        self.assertEqual(
            {(r['residue'], r['status'], r['reason']): sorted(order[i] for i in r['atoms']) for r in reordered_statuses},
            {(r['residue'], r['status'], r['reason']): r['atoms'] for r in statuses})

    def test_full_scf_matches_openmm(self):
        _, xyz, mp, vd = waters()
        solver=MutualSolver(mp,xyz,1e-8)
        pol, _, _, dipoles=solver.energy()
        perm, ref=evaluate_forces([permanent_force(mp), mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(mp))],xyz)
        self.assertAlmostEqual(perm+pol,ref,places=6)
        system=mm.System()
        for _ in xyz: system.addParticle(1)
        system.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(mp)))
        integrator=mm.VerletIntegrator(.001)
        context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
        context.setPositions(xyz*unit.nanometer)
        expected=np.array(system.getForce(0).getInducedDipoles(context))
        np.testing.assert_allclose(dipoles,expected,atol=2e-10)

    def test_protein_polarization_matches_reference(self):
        # The pinned OpenMM distribution supplies this complete, explicitly
        # protonated villin fixture. Keep the entire protein, remove solvent.
        pdb = app.PDBFile(str(Path(app.__file__).parent/'data/test.pdb'))
        model = app.Modeller(pdb.topology, pdb.positions)
        model.delete([r for r in model.topology.residues() if r.name in ('HOH', 'Cl', 'NA')])
        system = app.ForceField('amoeba2018.xml').createSystem(model.topology, nonbondedMethod=app.NoCutoff,
                                                             polarization='mutual', mutualInducedTargetEpsilon=1e-8)
        mp = next(f for f in system.getForces() if isinstance(f, mm.AmoebaMultipoleForce))
        xyz = np.array(model.positions.value_in_unit(unit.nanometer))
        pol, _, _, _ = MutualSolver(mp, xyz, 1e-8).energy()
        perm, ref = evaluate_forces([permanent_force(mp), clone(mp)], xyz)
        self.assertLess(abs(perm+pol-ref), 1e-5)

    def test_all_local_frames_and_chirality_match_openmm(self):
        xyz = np.array([[0,0,0],[.5,.1,0],[.1,.7,.1],[0,.2,.8],[.8,.5,.1],[.6,0,.7]])
        mp = mm.AmoebaMultipoleForce()
        mp.setPolarizationType(mp.Mutual)
        mp.setMutualInducedTargetEpsilon(1e-9)
        for i in range(6):
            mp.addMultipole(.05, [.001,.002,.003], [.0001,.00002,0,.00002,-.00004,.00001,0,.00001,-.00006],
                            i, -1 if i == mp.NoAxisType else (i+1)%6, (i+2)%6, (i+3)%6, .39, .001**(1/6), .001)
        system=mm.System()
        for _ in xyz: system.addParticle(1)
        system.addForce(clone(mp)); integrator=mm.VerletIntegrator(.001)
        context=mm.Context(system,integrator,mm.Platform.getPlatformByName('Reference'))
        context.setPositions(xyz*unit.nanometer)
        expected=np.array(system.getForce(0).getLabFramePermanentDipoles(context))
        np.testing.assert_allclose(lab_moments(mp,xyz)[1],expected,atol=1e-12)
        pol,_,_,_=MutualSolver(mp,xyz,1e-9).energy()
        perm,ref=evaluate_forces([permanent_force(mp),clone(mp)],xyz)
        self.assertAlmostEqual(perm+pol,ref,places=7)

    def test_nucleic_acid_and_ion_templates(self):
        ff=app.ForceField('amoeba2018.xml')
        for template, name in [('RAN','A'),('RCN','C'),('RGN','G'),('RUN','U'),
                               ('DAN','DA'),('DCN','DC'),('DGN','DG'),('DTN','DT'),
                               *[(n,n) for n in ('LI','NA','K','RB','CS','BE','MG','CA','SR','BA','ZN','F','Cl','Br','I')]]:
            with self.subTest(template=template):
                t=ff._templates[template]
                # Parameter assignment depends on topology, not arbitrary coordinates.
                atoms=[dict(name=a.name,element=a.element.symbol,chain='A',residue_number=1,insertion_code='',
                            residue_name=name,position=[i%5,i//5,0]) for i,a in enumerate(t.atoms)]
                prepared,status=parameterize(dict(atoms=atoms,bonds=t.bonds,tolerance=1e-6,max_iterations=500))
                self.assertIsNotNone(prepared)
                self.assertEqual(status[0]['status'],'parameterized')

    def test_halide_acceptors_and_ignored_ion_coordination_bonds(self):
        top, xyz, _, _ = waters(1)
        atoms=[dict(name=a.name,element=a.element.symbol,chain='A',residue_number=1,insertion_code='',
                    residue_name='HOH',position=(xyz[a.index]*10).tolist()) for a in top.atoms()]
        atoms.append(dict(name='CL',element='Cl',chain='B',residue_number=1,insertion_code='',residue_name='Cl',position=[3.2,0,0]))
        request=dict(atoms=atoms,bonds=[[0,1],[0,2],[0,3]],selection='water_chloride',selected=list(range(4)),cutoff=6.,tolerance=1e-8,max_iterations=500)
        report=analyze(request)
        self.assertTrue(all(r['status']=='parameterized' for r in report['residues']))
        self.assertEqual(len(report['candidates']),2)
        self.assertTrue(all(c['acceptor']==3 for c in report['candidates']))

    def test_missing_h_is_unparameterized_not_geometry_fallback(self):
        atom=dict(name='O',element='O',chain='A',residue_number=1,insertion_code='',residue_name='HOH',position=[0,0,0])
        report=analyze(dict(atoms=[atom],bonds=[],selection='oxygen',selected=[0],cutoff=6.,tolerance=1e-6,max_iterations=500))
        self.assertEqual(report['candidates'],[])
        self.assertEqual(report['residues'][0]['status'],'unparameterized')

    def test_decoupling_dimer_gives_isolated_monomers(self):
        _, xyz, mp, vd=waters(2)
        scores, full=decompose(mp,vd,xyz,[(set(range(3)),set(range(3,6)))],1e-8)
        score=next(iter(scores.values()))
        _, monoxyz, monomp, monovd=waters(1)
        mono=evaluate_forces([monomp, monovd],monoxyz)
        expected=sum(full[k] for k in ('full_permanent','full_vdw','full_polarization'))-2*sum(mono)
        self.assertAlmostEqual(score['delta_energy'],expected,places=6)
        self.assertAlmostEqual(score['delta_energy'],score['permanent']+score['vdw']+score['polarization'])

    def test_environment_retained_and_responds(self):
        _, xyz, mp, _=waters(3)
        solver=MutualSolver(mp,xyz,1e-8)
        _,_,_,full=solver.energy()
        _,_,_,dec=solver.energy((set(range(3)),set(range(3,6))))
        self.assertGreater(np.linalg.norm(full[6:]-dec[6:]),1e-7)
        r,rr3,_,_=solver.row(0,(set(range(3)),set(range(3,6))))
        self.assertTrue(np.all(rr3[3:6]==0)); self.assertTrue(np.all(rr3[6:]>0))

    def test_fail_closed_on_nonconvergence(self):
        _,xyz,mp,_=waters()
        with self.assertRaisesRegex(ValueError,'did not converge'):
            MutualSolver(mp,xyz,1e-14,0).energy()

    def test_report_retains_unfavorable_candidates_and_unmatched_residues(self):
        top, xyz, _, _=waters(2)
        atoms=[dict(name=a.name, element=a.element.symbol, chain='A', residue_number=int(a.residue.id),
                    insertion_code='', residue_name='HOH', position=(xyz[a.index]*10).tolist()) for a in top.atoms()]
        atoms.append(dict(name='C1',element='C',chain='B',residue_number=1,insertion_code='',residue_name='LIG',position=[20,0,0]))
        request=dict(atoms=atoms,bonds=[[a.index,b.index] for a,b in top.bonds()],selection='water',selected=list(range(6)),cutoff=6.,tolerance=1e-8,max_iterations=500)
        report=analyze(request)
        self.assertEqual(len(report['candidates']),4)
        self.assertTrue(any(r['status']=='unparameterized' for r in report['residues']))
        request['selected']=[0,1,2]
        self.assertEqual(analyze(request)['candidates'],[])


if __name__=='__main__': unittest.main()
