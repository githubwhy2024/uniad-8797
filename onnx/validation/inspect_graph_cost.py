"""Compare hash-bound exported and CPU graphs without executing either."""
import argparse
import gc
import gzip
import json
import sys
from pathlib import Path
import onnx
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'onnx/qnn'))
from graph_cost import cost_ledger
from tool_run import sha,save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True)
    p.add_argument('--logical-proof',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--cpu-proof',type=Path,required=True)
    a=p.parse_args()
    if sha(a.logical_model)!=a.logical_model_sha256 or sha(a.plan)!=a.plan_sha256:raise ValueError('ledger input identity differs')
    plan=json.loads(a.plan.read_text());cpu=Path(plan['source_model'])
    if sha(cpu)!=plan['source_model_sha256']:raise ValueError('CPU source identity differs')
    logical_proof=json.loads(a.logical_proof.read_text());cpu_proof=json.loads(a.cpu_proof.read_text())
    if logical_proof['derived_model_sha256']!=a.logical_model_sha256 or not logical_proof['all_node_output_extents_proven']:raise ValueError('logical shape certificate differs')
    if cpu_proof['derived_model_sha256']!=plan['source_model_sha256'] or not cpu_proof['all_node_output_extents_proven']:raise ValueError('CPU shape certificate differs')
    # Bind both metadata certificates; their respective root proofs precede this ledger.
    ledgers={}
    for role,path,digest in [('logical',a.logical_model,a.logical_model_sha256),('cpu',cpu,plan['source_model_sha256'])]:
        model=onnx.load(str(path),load_external_data=False)
        report=cost_ledger(model,digest,plan if role=='cpu' else None)
        if report['summary']['missing_shape_names'] or report['summary']['dense_coverage_gaps']:raise ValueError('cost ledger coverage incomplete: '+role)
        artifact=Path.cwd()/(role+'_graph_cost.json.gz')
        with gzip.open(artifact,'wt',encoding='utf8',compresslevel=6) as stream:json.dump(report,stream,separators=(',',':'))
        ledgers[role]=dict(model=str(path),model_sha256=digest,ledger=str(artifact),ledger_sha256=sha(artifact),summary=report['summary'])
        del model,report;gc.collect()
        save(Path.cwd()/'graph_cost_progress.json',dict(completed=role,scope='Static no-NN work/dependency ledger'))
    assert ledgers['cpu']['summary']['dense_macs_per_frame']==2341215237155
    total=ledgers['cpu']['summary']['cut_consumer_logical_bytes']
    from_plan=sum(r['bytes'] for part in plan['parts'] for r in part['inputs'] if any(r['source_name']==v['source_name'] for producer in plan['parts'] for v in producer['outputs']))
    if total!=from_plan:raise ValueError('independent cross-cut consumer ledger differs')
    summary=dict(status='pass',checks=dict(both_root_certified_sources=True,complete_static_shape_coverage=True,independent_cpu_executable_plan_coverage=True,independent_cross_cut_consumption=True),
                 ledgers=ledgers,plan=str(a.plan),plan_sha256=a.plan_sha256,
                 certificates={str(path):sha(path) for path in [a.logical_proof,a.cpu_proof]},
                 helper_sha256=sha(ROOT/'onnx/qnn/graph_cost.py'),script_sha256=sha(__file__),
                 dense_macs_delta_cpu_minus_logical=ledgers['cpu']['summary']['dense_macs_per_frame']-ledgers['logical']['summary']['dense_macs_per_frame'],
                 scope='Original planning-export and current CPU-lowered executable graph costs. No tensor payloads copied, no inference/FP16/quantization, no board/FPS prediction. Explicit SSA/cut extents are not physical DDR/copy/peak RAM/disk.')
    save(Path.cwd()/'graph_cost_summary.json',summary)
    print(json.dumps(dict(status='pass',graphs={k:v['summary'] for k,v in ledgers.items()})))


if __name__=='__main__':main()
