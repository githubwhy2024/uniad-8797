"""One bounded monolithic probe after portable coverage/attention controls."""
import argparse,json,sys
from pathlib import Path
import onnx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from graph_cost import graph_dependencies
from partition_model import make_parts
import shape_dataflow as engine
from tool_run import save,sha
from resources import terminal
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-plan',type=Path,required=True);p.add_argument('--reference-plan-sha256',required=True);p.add_argument('--probe-benchmark',type=Path,required=True);p.add_argument('--probe-benchmark-sha256',required=True);a=p.parse_args()
    if sha(a.reference_plan)!=a.reference_plan_sha256 or sha(a.probe_benchmark)!=a.probe_benchmark_sha256:raise ValueError('monolithic prerequisite changed')
    terminal(a.probe_benchmark.parent);benchmark=json.loads(a.probe_benchmark.read_text())
    if benchmark['status']!='pass' or benchmark['fastest_under_local_probe_constraints']['policy']!='whole_attention':raise ValueError('monolithic probe lacks representative evidence')
    reference=json.loads(a.reference_plan.read_text());source=Path(reference['source_model'])
    if sha(source)!=reference['source_model_sha256']:raise ValueError('current streaming source changed')
    model=onnx.load(str(source));types,constants,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('monolithic root proof failed')
    _,_,live,roots,_=graph_dependencies(model);executed=sorted(live-roots)
    if executed!=[i for r in reference['parts'] for i in r['source_node_indices']]:raise ValueError('monolithic independent coverage differs')
    pool=reference['shared_constants']
    if sha(pool['path'])!=pool['sha256']:raise ValueError('immutable pool changed')
    # Extent sum is no longer used as a physical-memory bound. A single
    # source-preserving graph is only a probe, constrained at converter/load.
    plan=make_parts(model,types,512*1024**3,Path.cwd(),shared=[r['source_name'] for r in pool['entries']],segments_override=[(0,len(executed))])
    plan.update(status='pass',source_model=str(source),source_model_sha256=reference['source_model_sha256'],shared_constants=pool,shared_constants_helper_sha256=reference['shared_constants_helper_sha256'],script_sha256=sha(ROOT/'onnx/qnn/partition_model.py'),engine_sha256=sha(engine.__file__),policy='monolithic_source_preserving_probe',reference_plan_sha256=a.reference_plan_sha256,representative_benchmark_sha256=a.probe_benchmark_sha256,resource_probe=dict(converter_address_space_gib=18,native_address_space_gib=18,whole_host_memory_kib=24606364,extent_sum_is_not_a_memory_bound=True),scope='Single exact current streaming source graph plus reused immutable pool. No precision/operator/Host change. Independent cut audit required;18GiB converter/native controls must pass before any integrated real frame. This is not monolithic/task/target acceptance.')
    save(Path.cwd()/'partition_plan.json',plan);print(json.dumps(dict(status='pass',parts=1,executable_nodes=len(executed),source_model_sha256=plan['source_model_sha256'],cut_models_unique=len({r[k] for r in plan['parts'] for k in ['logical_model','native_model']}))))
if __name__=='__main__':main()
