#!/usr/bin/env python3
"""Replay checked QNN graph transformations in fresh, durable tool attempts."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
from tool_run import ROOT,sha,save,process_identity,stamp,CONVERTER_ENV,UNIAD_PYTHON

STEPS=[
 ('static','prepare_model.py','preparation.json',['model.static-controls.onnx'],['--bind-shape-controls','--fold-reduceprod','--prune-empty-concat']),
 ('dcn','lower_transpose.py','routing.json',['model.rank4-dcn.onnx'],[]),
 ('singleton','lower_singleton.py','singleton.json',['model.rank5-singleton.onnx'],[]),
 ('packed','lower_rank.py','packing.json',['model.packed-rank.onnx'],[]),
 ('protected','protect_reshape.py','view_protection.json',['model.protected-view.onnx'],[]),
 ('ports','adapt_boundary.py','boundary.json',['model.native-ports.onnx'],[]),
 ('tile','prepare_model.py','preparation.json',['model.tiled-expands.onnx'],['--tile-expands']),
 ('broadcast','lower_broadcast.py','broadcast.json',['model.grouped-broadcast.onnx'],[]),
 ('boolean','lower_boolean.py','boolean.json',['model.boolean-concat.onnx'],[]),
 ('scatter','normalize_scatter.py','scatter.json',['model.nonnegative-scatter.onnx'],[]),
 ('reduction','lower_reduction.py','reduction.json',['model.integer-select.onnx'],[]),
 ('root-extents','prepare_model.py','preparation.json',['model.root-extents.onnx'],['--fold-root-extents']),
 ('integer-clip','lower_integer_clip.py','integer_clip.json',['model.integer-clip.onnx'],[]),
 ('integer-scatter','lower_integer_scatter.py','integer_scatter.json',['model.integer-scatter.onnx'],[]),
 ('gather-elements','lower_gather_elements.py','gather_elements.json',['model.rank4-gather-elements.onnx'],[]),
]
GENERIC_STEPS=[
 ('query-streaming','stream_attention.py','query_streaming.json',['model.query-streamed.onnx'],[]),
 ('identity-views','eliminate_identity_views.py','identity_views.json',['model.identity-eliminated.onnx'],[]),
]

def recipe(chunk_queries, share_immutable_mib):
 return dict(schema='q4-graph-recipe-v1', cpu_sdk_compatibility=[dict(name=n,script=s,script_sha256=sha(Path(__file__).parent/s),report=r,models=m,flags=f) for n,s,r,m,f in STEPS],
             generic_graph=[dict(name=n,script=s,script_sha256=sha(Path(__file__).parent/s),report=r,models=m,flags=(['--chunk-queries',str(chunk_queries)] if n=='query-streaming' else f)) for n,s,r,m,f in GENERIC_STEPS],
             partition=dict(script='partition_model.py',script_sha256=sha(Path(__file__).parent/'partition_model.py'),shared_pool_helper_sha256=sha(Path(__file__).parent/'shared_constants.py'),flags=['--share-immutable-mib',str(share_immutable_mib)]),
             portable_helpers=dict(attention_regions_sha256=sha(Path(__file__).parent/'attention_regions.py'),shape_engine_sha256=sha(ROOT/'onnx/validation/shape_dataflow.py'),backend_contract_sha256=sha(Path(__file__).parent/'backend_contract.py')),
             classification='CPU SDK compatibility sequence is a host reference implementation. Query streaming, identity-view elimination, shared immutable operands and Host lifetime/ownership are generic candidates; target still needs its own SDK/backend validation.',
             scope='Graph transformation recipe only; proof/audit, compiled native ABI, own state/task, portable release and target are separate.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--convert',action='store_true');p.add_argument('--generic-profile',action='store_true');p.add_argument('--chunk-queries',type=int,default=2048);p.add_argument('--share-immutable-mib',type=int,default=16);p.add_argument('--partition',action='store_true');p.add_argument('--describe',action='store_true');args=p.parse_args()
 root=Path.cwd();model=args.model.absolute();digest=args.model_sha256
 if root.resolve()!=root or not root.is_relative_to(ROOT/'onnx/runs'):raise ValueError('fresh ordinary Q4 run required')
 if sha(model)!=digest:raise ValueError('source model differs')
 if args.chunk_queries<1 or args.share_immutable_mib<1:raise ValueError('positive block/resource threshold required')
 description=recipe(args.chunk_queries,args.share_immutable_mib);save(root/'graph_recipe.json',dict(description,source_model=str(model),source_model_sha256=digest,script_sha256=sha(__file__)))
 if args.describe:return
 steps=list(STEPS)
 if args.generic_profile:steps += [(n,s,r,m,(['--chunk-queries',str(args.chunk_queries)] if n=='query-streaming' else f)) for n,s,r,m,f in GENERIC_STEPS]
 host=str(CONVERTER_ENV/'bin/python');uniad=UNIAD_PYTHON;completed=[];child=None
 def terminate(signum,frame):
  if child is not None and child.poll() is None:
   child.send_signal(signum);child.wait()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 def step(name,stage,source,digest,expects,command):
  nonlocal child
  run=root/name;argv=[host,str(ROOT/'onnx/qnn/tool_run.py'),'--run-dir',str(run),'--stage',stage,'--model',str(source),'--model-sha256',digest]
  if stage=='convert':argv+=['--diagnose-converter','--address-space-gib','18']
  for expected in expects:argv+=['--expect',expected]
  if command:argv+=['--']+command
  child=subprocess.Popen(argv,cwd=ROOT)
  save(root/'profile_progress.json',dict(stage=name,active_run=str(run),child=process_identity(child.pid),completed=completed,command=argv,started_utc=stamp(),scope='Preparation/control/converter only; no neural acceptance.'))
  code=child.wait();child=None
  if code:raise RuntimeError('profile step failed: '+name)
  status=json.loads((run/'status.json').read_text());result=json.loads((run/'result.json').read_text())
  if status['status']!='pass' or result['status']!='pass' or status['result_sha256']!=sha(run/'result.json'):raise ValueError('step terminal evidence differs')
  completed.append(dict(stage=name,run_dir=str(run),result_sha256=sha(run/'result.json')))
  return run
 for index,(name,script,report,models,flags) in enumerate(steps):
  run=step(f'{index:02d}-{name}','control',model,digest,[report]+models,[uniad,str(ROOT/'onnx/qnn'/script),'--model',str(model),'--model-sha256',digest]+flags)
  record=json.loads((run/report).read_text())
  if record['status']!='pass' or record['source_model_sha256']!=digest or sha(record['model'])!=record['model_sha256']:raise ValueError('prepared candidate source differs')
  model=Path(record['model']);digest=record['model_sha256']
 run=step(f'{len(steps):02d}-proof','proof',model,digest,['shape_proof.json','extent_certificate.json','model.shape-certified.onnx'],[uniad,str(ROOT/'onnx/validation/prove_graph_extents.py'),'--model',str(model),'--model-sha256',digest])
 step(f'{len(steps)+1:02d}-audit','proof',model,digest,['audit.json'],[uniad,str(ROOT/'onnx/validation/audit_graph_extents.py'),'--proof-dir',str(run)])
 proof=json.loads((run/'shape_proof.json').read_text());model=Path(proof['derived_model']);digest=proof['derived_model_sha256']
 if args.partition:step(f'{len(steps)+2:02d}-partition','control',model,digest,['partition_plan.json'],[uniad,str(ROOT/'onnx/qnn/partition_model.py'),'--model',str(model),'--model-sha256',digest,'--share-immutable-mib',str(args.share_immutable_mib)])
 if args.convert:step(f'{len(steps)+2:02d}-convert','convert',model,digest,[],[])
 save(root/'profile_preparation.json',dict(status='pass',source_model=str(args.model),source_model_sha256=args.model_sha256,model=str(model),model_sha256=digest,steps=completed,recipe_sha256=sha(root/'graph_recipe.json'),generic_profile=args.generic_profile,script_sha256=sha(__file__),scope='Graph controls/root/audit and optionally converter only; actual model load/frames/tasks separate.'))
if __name__=='__main__':main()
