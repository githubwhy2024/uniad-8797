"""Check real accepted-prefix reuse from an unchanged failed build provider."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
import build_partitions as build
from tool_run import sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--build-run',type=Path,required=True);a=p.parse_args()
    target=build.terminal(a.build_run);record=json.loads((a.build_run/'partition_build.json').read_text());binding=record['reuse_binding'];prior=Path(binding['ancestor'])
    before={name:sha(prior/name) for name in ('status.json','result.json','partition_build_progress.json')}
    plan=json.loads(Path(record['plan']).read_text());hints=json.loads(Path(record['layout_hints']).read_text())['layouts'];recipe=record['recipe'];backend=record['backend_sha256'];checks={}
    rows,actual=build.reusable_prefix(prior,plan,recipe,backend,hints)
    checks['actual_prefix_exact']=len(rows)==binding['reused_prefix']==213 and actual['partial_provider']['provider_full_build_accepted'] is False
    checks['provider_stays_failed']=json.loads((prior/'status.json').read_text())['status']=='failed' and not (prior/'partition_build.json').exists()
    for label in ('recipe','backend','layout','running'):
        rp=copy.deepcopy(recipe);bp=backend;hp=copy.deepcopy(hints);original=build.process_identity
        if label=='recipe':rp['onnx/qnn/partition_model.py']='0'*64
        if label=='backend':bp='0'*64
        if label=='layout':
            row=next(v for part in plan['parts'][:213] for v in part['inputs']+part['outputs'] if v['source_name'] in hints)
            hp[row['source_name']]='INVALID'
        if label=='running':
            identity=json.loads((prior/'status.json').read_text())['controller']
            build.process_identity=lambda pid:identity if pid==identity['pid'] else original(pid)
        try:build.reusable_prefix(prior,plan,rp,bp,hp)
        except ValueError:checks[label+'_rejected']=True
        else:checks[label+'_rejected']=False
        finally:build.process_identity=original
    changed=copy.deepcopy(plan);changed['parts'][0]['native_model_sha256']='0'*64
    rows,_=build.reusable_prefix(prior,changed,recipe,backend,hints);checks['changed_cut_not_reused']=len(rows)==0
    parent_plan=json.loads(Path(json.loads((prior/'launch.json').read_text())['argv'][json.loads((prior/'launch.json').read_text())['argv'].index('--plan')+1]).read_text())
    forged=copy.deepcopy(json.loads((prior/'partition_build_progress.json').read_text())['completed_parts'][0]);forged['library_sha256']='0'*64
    try:build.verify_record(forged,parent_plan['parts'][0])
    except ValueError:checks['changed_library_rejected']=True
    else:checks['changed_library_rejected']=False
    checks['provider_evidence_unmodified']=before=={name:sha(prior/name) for name in before}
    save(Path.cwd()/'partial_build_reuse_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,provider=str(prior),provider_status='failed',reused_prefix=213,new_build_result_sha256=sha(a.build_run/'result.json'),orchestrator_sha256=sha(build.__file__),scope='Real independently passing compiled prefix reused under byte/ABI/helper/backend/layout gates. Running-policy branch constructed without touching provider. Whole failed provider remains unaccepted; no neural claim.'))
    if not all(checks.values()):raise ValueError('partial prefix reuse control failed')


if __name__=='__main__':main()
