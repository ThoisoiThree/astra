"""Developer-only, reproducible export of OpenMM 8.4 AMOEBA 2018 data.
The application reads the committed JSON directly; Python is not a build dependency.
"""
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from openmm import app
root = Path(app.__file__).parent / 'data'
ff = ET.parse(root/'amoeba2018.xml').getroot()
app.PDBFile._loadNameReplacementTables()
types = {int(t.attrib['name']): {'class': int(t.attrib['class']), 'element': t.attrib['element']} for t in ff.find('AtomTypes')}
residues = []
for r in ff.find('Residues'):
    residues.append(dict(name=r.attrib['name'], atoms=[dict(name=a.attrib['name'], type=int(a.attrib['type'])) for a in r.findall('Atom')], bonds=[[int(b.attrib['from']), int(b.attrib['to'])] for b in r.findall('Bond')], external=[int(b.attrib['from']) for b in r.findall('ExternalBond')]))
mp = ff.find('AmoebaMultipoleForce')
multipoles = []
for m in mp.findall('Multipole'):
    a=m.attrib
    q11,q21,q22,q31,q32,q33=[float(a[k]) for k in ['q11','q21','q22','q31','q32','q33']]
    multipoles.append(dict(type=int(a['type']), axes=[int(a.get(k,0)) for k in ['kz','kx','ky']], charge=float(a['c0']), dipole=[float(a[k]) for k in ['d1','d2','d3']], quadrupole=[q11,q21,q31,q21,q22,q32,q31,q32,q33]))
polar={int(p.attrib['type']):dict(alpha=float(p.attrib['polarizability']),thole=float(p.attrib['thole']),groups=[int(v) for k,v in p.attrib.items() if k.startswith('pgrp')]) for p in mp.findall('Polarize')}
vdw={int(v.attrib['class']):dict(radius=float(v.attrib['sigma'])*.5,epsilon=float(v.attrib['epsilon']),reduction=float(v.attrib['reduction'])) for v in ff.find('AmoebaVdwForce').findall('Vdw')}
pairs=[dict(classes=[int(p.attrib['class1']),int(p.attrib['class2'])],radius=float(p.attrib['sigma']),epsilon=float(p.attrib['epsilon'])) for p in ff.find('AmoebaVdwForce').findall('Pair')]
bonds={r.attrib['name']:[[b.attrib['from'],b.attrib['to']] for b in r] for r in ET.parse(root/'residues.xml').getroot()}
data=dict(version=1,types=types,residues=residues,multipoles=multipoles,polar=polar,vdw=vdw,pairs=pairs,standard_bonds=bonds,residue_aliases=app.PDBFile._residueNameReplacements,atom_aliases=app.PDBFile._atomNameReplacements)
Path('data/amoeba/amoeba2018.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
