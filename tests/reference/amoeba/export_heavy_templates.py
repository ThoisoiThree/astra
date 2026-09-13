"""Developer-only download/export of wwPDB CCD ideal heavy-atom coordinates.
No network or Python is needed by the application or its build.
"""
import concurrent.futures
import hashlib
import json
import shlex
import urllib.request
from pathlib import Path

NAMES='ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split()
def rows(text,category):
    lines=text.splitlines();i=0
    while i<len(lines):
        if lines[i].strip()!='loop_':i+=1;continue
        i+=1;columns=[]
        while i<len(lines) and not lines[i].strip():i+=1
        while i<len(lines) and lines[i].startswith('_'):
            columns.append(lines[i].strip());i+=1
        if not columns or not columns[0].startswith(category+'.'):continue
        values=[]
        while i<len(lines) and lines[i].strip() not in ('#','loop_') and not lines[i].startswith('_'):
            values.extend(shlex.split(lines[i],posix=True));i+=1
        assert len(values)%len(columns)==0
        return [dict(zip([c.split('.',1)[1] for c in columns],values[k:k+len(columns)])) for k in range(0,len(values),len(columns))]
    raise ValueError(category)
def export(name):
    url=f'https://files.rcsb.org/ligands/download/{name}.cif'
    raw=urllib.request.urlopen(url,timeout=40).read();text=raw.decode()
    atoms=[dict(name=a['atom_id'],element=a['type_symbol'],position=[float(a[f'pdbx_model_Cartn_{k}_ideal']) for k in 'xyz']) for a in rows(text,'_chem_comp_atom') if a['type_symbol']!='H']
    indices={a['name']:i for i,a in enumerate(atoms)}
    bonds=[dict(atoms=[indices[b['atom_id_1']],indices[b['atom_id_2']]],order=b['value_order']) for b in rows(text,'_chem_comp_bond') if b['atom_id_1'] in indices and b['atom_id_2'] in indices]
    return name,dict(atoms=atoms,bonds=bonds,source=url,sha256=hashlib.sha256(raw).hexdigest())
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:data=dict(pool.map(export,NAMES))
    Path('data/amoeba/heavy_templates.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
    print('Exported',len(data),'CCD templates')
