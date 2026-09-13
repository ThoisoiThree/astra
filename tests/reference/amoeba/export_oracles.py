"""Regenerate native regression fixtures with independent OpenMM Reference energies."""
import json
from pathlib import Path
import numpy as np
import openmm as mm
from openmm import app,unit
from reference_engine import clone,permanent_force,evaluate_forces,lab_moments,decompose
from test_engine import waters
ff=app.ForceField('amoeba2018.xml')
def fixture(top,xyz,name,groups=None):
    xyz=(np.asarray(xyz)*10).astype(np.float32).astype(float)*.1
    system=ff.createSystem(top,nonbondedMethod=app.NoCutoff,polarization='mutual',mutualInducedTargetEpsilon=1e-8,rigidWater=False)
    mp=next(f for f in system.getForces() if isinstance(f,mm.AmoebaMultipoleForce))
    vd=next(f for f in system.getForces() if isinstance(f,mm.AmoebaVdwForce))
    perm,v,combined=evaluate_forces([permanent_force(mp),clone(vd),clone(mp)],xyz)
    q,mu,quad,alpha,damp,thole=lab_moments(mp,xyz)
    atoms=[dict(name=a.name,element=a.element.symbol,chain=a.residue.chain.id,residue_number=int(a.residue.id),residue_name=a.residue.name,position=(xyz[a.index]*10).tolist()) for a in top.atoms()]
    result=dict(name=name,atoms=atoms,bonds=[[a.index,b.index] for a,b in top.bonds()],permanent=perm,vdw=v,polarization=combined-perm,q=q.tolist(),mu=mu.tolist(),quad=quad.tolist(),alpha=alpha.tolist())
    if groups:
        scores,full=decompose(mp,vd,xyz,groups,1e-9)
        result['decoupling']=[dict(groups=[list(a),list(b)],**s) for (a,b),s in scores.items()]
    return result
out=[]
top,xyz,_,_=waters(3)
out.append(fixture(top,xyz,'water_trimer',[(set(range(3)),set(range(3,6)))]))
pdb=app.PDBFile(str(Path(app.__file__).parent/'data/test.pdb'))
model=app.Modeller(pdb.topology,pdb.positions)
model.delete([r for r in model.topology.residues() if r.name in ('HOH','Cl','NA')])
out.append(fixture(model.topology,np.array(model.positions.value_in_unit(unit.nanometer)),'villin'))
for name,t in ff._templates.items():
    top=app.Topology();chain=top.addChain('A');blocks=[]
    def add(name):
        t=ff._templates[name];r=top.addResidue(name,chain,str(len(blocks)+1));atoms=[top.addAtom(a.name,a.element,r) for a in t.atoms]
        for a,b in t.bonds:top.addBond(atoms[a],atoms[b])
        block={a.name:a for a in atoms};blocks.append((name,block));return block
    center=add(name);links=[]
    # Fill the root template's external valences with real template caps.
    for idx in t.externalBonds:
        atom=t.atoms[idx].name
        if atom=='N': cap=add('ACE');links.append((cap['C'],center['N']))
        elif atom=='C':cap=add('NME');links.append((center['C'],cap['N']))
        elif atom=='SG':
            cap=add('CYX');ace=add('ACE');nme=add('NME');links.extend([(center['SG'],cap['SG']),(ace['C'],cap['N']),(cap['C'],nme['N'])])
        elif atom=='P':cap=add('DA5' if name.startswith('D') else 'RA5');links.append((cap["O3'"],center['P']))
        elif atom=="O3'":cap=add('DA3' if name.startswith('D') else 'RA3');links.append((center["O3'"],cap['P']))
        else:raise ValueError((name,atom))
    # Build residue paths from directed peptide/phosphodiester external links.
    atomblock={a.index:k for k,(_,b) in enumerate(blocks) for a in b.values()}
    nxt={};prev={}
    for a,b in links:
        if a.name=='SG':continue
        i,j=atomblock[a.index],atomblock[b.index];nxt[i]=j;prev[j]=i
    new=app.Topology();mapping={}
    for start in range(len(blocks)):
        if start in prev:continue
        ch=new.addChain(chr(65+len(list(new.chains()))));k=start
        while True:
            bn,block=blocks[k];res=new.addResidue(bn,ch,str(k+1))
            for a in block.values():mapping[a.index]=new.addAtom(a.name,a.element,res)
            if k not in nxt:break
            k=nxt[k]
    for a,b in list(top.bonds())+links:new.addBond(mapping[a.index],mapping[b.index])
    top=new
    n=top.getNumAtoms();i=np.arange(n);xyz=np.column_stack((i*.7,np.sin(i*1.7)*.4,np.cos(i*2.1)*.4))
    for k,(a,b) in enumerate(links):
        ai,bi=mapping[a.index].index,mapping[b.index].index
        xyz[bi]=xyz[ai]+[.17,.08,.03]
    try:out.append(fixture(top,xyz,name))
    except Exception as e:raise RuntimeError(name) from e
top,xyz,_,_=waters(1)
xyz=list(xyz)
for k,name in enumerate(('LI','NA','K','RB','CS','BE','MG','CA','SR','BA','ZN','F','Cl','Br','I')):
    t=ff._templates[name];r=top.addResidue(name,top.addChain(name),'1')
    top.addAtom(t.atoms[0].name,t.atoms[0].element,r)
    xyz.append([.32,0,0] if name=='Cl' else [1.2*(k+1),.4,.3])
out.append(fixture(top,xyz,'water_ions'))
Path('tests/fixtures/amoeba_oracles.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print('Exported',len(out),'systems')

# All six local-frame conventions, including reflected chirality.
frames=[]
for reflected in (False,True):
    xyz=np.array([[0,0,0],[.5,.1,0],[.1,.7,.1],[0,.2,.8],[.8,.5,.1],[.6,0,.7]])
    if reflected:xyz[:,1]*=-1
    mp=mm.AmoebaMultipoleForce();mp.setPolarizationType(mp.Mutual);mp.setMutualInducedTargetEpsilon(1e-10)
    for i in range(6):
        mp.addMultipole(.05,[.001,.002,.003],[.0001,.00002,0,.00002,-.00004,.00001,0,.00001,-.00006],i,-1 if i==5 else (i+1)%6,(i+2)%6,(i+3)%6,.39,.001**(1/6),.001)
    sys=mm.System()
    for _ in xyz:sys.addParticle(1.)
    sys.addForce(clone(mp));integrator=mm.VerletIntegrator(.001);ctx=mm.Context(sys,integrator,mm.Platform.getPlatformByName('Reference'));ctx.setPositions(xyz)
    induced=np.array(sys.getForce(0).getInducedDipoles(ctx)).tolist()
    mu=np.array(sys.getForce(0).getLabFramePermanentDipoles(ctx)).tolist()
    perm,total=evaluate_forces([permanent_force(mp),clone(mp)],xyz)
    frames.append(dict(xyz=xyz.tolist(),mu=mu,induced=induced,permanent=perm,polarization=total-perm))
    del ctx,integrator
Path('tests/fixtures/amoeba_frames.json').write_text(json.dumps(frames,separators=(',',':'))+'\n')
