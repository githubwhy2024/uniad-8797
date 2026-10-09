"""Audit the ordinary deployment closure and separate storage from cut traffic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from resources import terminal
from tool_run import sha,save


def inspect(run):
    result=terminal(run);report=json.loads((run/'planning_package.json').read_text())
    if result['artifacts']['planning_package.json']['sha256']!=sha(run/'planning_package.json'):
        raise ValueError('package report changed')
    root=Path(report['bundle']);manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'manifest.json')!=report['manifest_sha256']:
        raise ValueError('package manifest differs')
    declared={'manifest.json'}|{r['path'] for r in manifest['files'].values()}
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=declared or any(p.resolve()!=p for p in root.rglob('*')):
        raise ValueError('package has unbound files or developer links')
    groups={};rows=[]
    for role,row in manifest['files'].items():
        path=root/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('package file binding differs')
        groups.setdefault(row['sha256'],[]).append(role)
        rows.append(dict(role=role,**row))
    return root,manifest,dict(manifest_sha256=report['manifest_sha256'],files=len(rows),total_bytes=sum(r['bytes'] for r in rows)+(root/'manifest.json').stat().st_size,role_bytes={r['role']:r['bytes'] for r in rows},duplicate_content_groups=[r for r in groups.values() if len(r)>1],ordinary_closure=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-run',type=Path,required=True);p.add_argument('--reference-package',type=Path);a=p.parse_args()
    root,manifest,report=inspect(a.package_run)
    if a.reference_package:
        _,_,ref=inspect(a.reference_package);report['reference']=ref;report['saved_bytes']=ref['total_bytes']-report['total_bytes']
    pool=manifest['profile'].get('shared_constants')
    if pool:
        path=root/manifest['files']['shared_constant_pool']['path']
        with np.load(path,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        if set(arrays)!={r['key'] for r in pool['entries']} or pool['unique_data_bytes']!=sum(v.nbytes for v in arrays.values()):
            raise ValueError('shared constant storage closure differs')
        hashes={k:hashlib.sha256(memoryview(v).cast('B')).hexdigest() for k,v in arrays.items()}
        for row in pool['entries']:
            v=arrays[row['key']]
            if hashes[row['key']]!=row['data_sha256'] or v.size!=int(np.prod(row['shape'])) or str(v.dtype)!=row['dtype']:
                raise ValueError('shared view binding differs')
        report['shared_constants']=dict(physical_buffers=len(arrays),logical_views=len(pool['entries']),loaded_unique_bytes=pool['unique_data_bytes'],compressed_resource_bytes=path.stat().st_size,lossless=True)
    report.update(status='pass',public_inputs=len(manifest['profile']['abi']['inputs']),public_outputs=len(manifest['profile']['abi']['outputs']),runtime_only=True,scope='Actual CPU deployment closure: libraries/runtime/learned state/solver and shared constants only. Build cpp/bin/ONNX/datasets/raw frame archives are excluded. Storage is independent of cumulative cut-input extents; target ARM/HTP packaging remains Q5.')
    save(Path.cwd()/'deployment_resources.json',report)


if __name__=='__main__':main()
