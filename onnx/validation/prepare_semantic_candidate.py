"""Materialize one selected semantic blueprint using existing immutable data."""
import argparse,json,sys
from pathlib import Path
import onnx
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from graph_cost import graph_dependencies
from partition_model import make_parts
import shape_dataflow as engine
from tool_run import save,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--blueprint',type=Path,required=True);p.add_argument('--blueprint-sha256',required=True);a=p.parse_args()
    if sha(a.blueprint)!=a.blueprint_sha256:raise ValueError('selected blueprint changed')
    candidate=json.loads(a.blueprint.read_text());reference_path=Path(candidate['reference_plan'])
    if candidate['status']!='pass' or sha(reference_path)!=candidate['reference_plan_sha256']:raise ValueError('semantic source reference changed')
    reference=json.loads(reference_path.read_text());model_path=Path(candidate['source_model'])
    if sha(model_path)!=candidate['source_model_sha256'] or candidate['source_model_sha256']!=reference['source_model_sha256']:raise ValueError('semantic source model changed')
    model=onnx.load(str(model_path));types,constants,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('selected semantic source roots failed')
    _,_,live,roots,_=graph_dependencies(model);executed=sorted(live-roots)
    if [i for row in candidate['parts'] for i in row['source_node_indices']]!=executed:raise ValueError('independent semantic candidate coverage differs')
    segments=[];start=0
    for row in candidate['parts']:
        end=start+len(row['source_node_indices']);segments.append((start,end));start=end
    pool=reference['shared_constants']
    if sha(pool['path'])!=pool['sha256']:raise ValueError('shared pool changed')
    plan=make_parts(model,types,candidate['hard_produced_bytes'],Path.cwd(),shared=[r['source_name'] for r in pool['entries']],segments_override=segments)
    plan.update(status='pass',source_model=str(model_path),source_model_sha256=candidate['source_model_sha256'],shared_constants=pool,shared_constants_helper_sha256=reference['shared_constants_helper_sha256'],script_sha256=sha(ROOT/'onnx/qnn/partition_model.py'),engine_sha256=sha(engine.__file__),semantic_blueprint=str(a.blueprint),semantic_blueprint_sha256=a.blueprint_sha256,semantic_regions=candidate['regions'],scope='One selected full-coverage portable semantic blueprint materialized with exact original operations/weights and reused immutable pool; cumulative extent is a proxy. Actual converter/native resources/execute/own-state/task acceptance separate.')
    save(Path.cwd()/'partition_plan.json',plan);print(json.dumps(dict(status='pass',parts=len(plan['parts']),source_model_sha256=plan['source_model_sha256'])))
if __name__=='__main__':main()
