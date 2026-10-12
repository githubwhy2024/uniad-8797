#!/usr/bin/env python3
"""Load actual QNN CPU graph and advance selected real frames with own state."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from resources import bind,session_for
from tool_run import sha,save,ROOT,process_identity
sys.path.insert(0,str(ROOT/'onnx/fixed'))
from host import FixedStateTransaction,pad_v1_initial_state,state_digest

class ObservedSession:
 def __init__(self,session):self.session=session;self.outputs=None;self.observation=None
 def __getattr__(self,name):return getattr(self.session,name)
 def run(self,names,feed):
  def identity(values):return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()) for n,v in values.items()}
  before=identity(feed);self.outputs=None;self.observation=dict(inputs=before,outputs_returned=False);started=time.monotonic();values=self.session.run(names,feed);ordered=names if names is not None else [v.name for v in self.session.get_outputs()];self.outputs=dict(zip(ordered,values));unchanged=before==identity(feed);self.observation.update(outputs_returned=True,outputs=identity(self.outputs),inputs_unmodified=unchanged,inference_seconds=time.monotonic()-started)
  if not unchanged:raise ValueError('actual QNN partition/session mutated logical inputs')
  return values

def memory_snapshot():
 fields={}
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.split(':',1)[0] in ('VmPeak','VmSize','VmHWM','VmRSS'):fields[line.split(':',1)[0]]=line.split(':',1)[1].strip()
 fields['address_space_limit_bytes']=list(resource.getrlimit(resource.RLIMIT_AS));return fields

def main():
 p=argparse.ArgumentParser();backend=p.add_mutually_exclusive_group(required=True);backend.add_argument('--compile-run',type=Path);backend.add_argument('--partition-build',type=Path);p.add_argument('--bridge-build',type=Path,required=True);p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True);p.add_argument('--load-only',action='store_true');p.add_argument('--inputs',type=Path);p.add_argument('--inputs-sha256');p.add_argument('--resume-from',type=Path);p.add_argument('--stop-after',type=int,default=6);args=p.parse_args()
 if not 1<=args.stop_after<=6:raise ValueError('stop-after must be within six-frame scope')
 run=Path.cwd()
 if args.partition_build:
  from partition_session import bind_partitions
  manifest=bind_partitions(args.partition_build,args.bridge_build,args.logical_model,args.logical_model_sha256)
 else:manifest=bind(args.compile_run,args.bridge_build,args.logical_model,args.logical_model_sha256)
 save(run/'profile.json',manifest);pin=sha(run/'profile.json')
 rows=[];prior_completed=[]
 if args.resume_from:
  prior=args.resume_from.absolute()
  if prior.resolve()!=prior or not prior.is_relative_to(ROOT/'onnx/runs') or prior==run:raise ValueError('resume ancestor must be another ordinary Q4 run')
  status=json.loads((prior/'status.json').read_text())
  for role in ('controller','child'):
   identity=status.get(role)
   if identity and identity.get('startticks') is not None and process_identity(identity['pid']).get('startticks')==identity['startticks']:raise ValueError('ancestor tool process is still active')
  if sha(prior/'profile.json')!=pin:raise ValueError('resume profile/runtime/backend differs')
  progress=json.loads((prior/'neural_progress.json').read_text());prior_completed=progress.get('committed_frames',progress.get('completed_frames',[]))
  if not isinstance(prior_completed,list) or not prior_completed:raise ValueError('no committed frame journal to resume')
  if progress['profile_sha256']!=pin or len(prior_completed)>=args.stop_after:raise ValueError('resume pin or unfinished scope differs')
  for row in prior_completed:
   for artifact in ('final_plan','checkpoint','outputs'):
    path=Path(row[artifact+'_path'])
    if path.resolve()!=path or not path.is_relative_to(ROOT/'onnx/runs') or sha(path)!=row[artifact+'_sha256']:raise ValueError('ancestor committed artifact differs')
 if args.resume_from:
  save(run/'resume_binding.json',dict(ancestor=str(prior),ancestor_status_sha256=sha(prior/'status.json'),ancestor_progress_sha256=sha(prior/'neural_progress.json'),profile_sha256=pin,committed_tokens=[r['token'] for r in prior_completed]))
 if not args.load_only:
  if not args.inputs or sha(args.inputs)!=args.inputs_sha256:raise ValueError('real input manifest hash differs')
  inputs=json.loads(args.inputs.read_text())
  if inputs['status']!='pass' or len(inputs['frames'])!=6:raise ValueError('six accepted real inputs required')
  rows=inputs['frames']
  save(run/'input_binding.json',dict(path=str(args.inputs.absolute()),sha256=args.inputs_sha256))
  if args.resume_from:
   previous=json.loads((args.resume_from/'input_binding.json').read_text())
   if previous['sha256']!=args.inputs_sha256:raise ValueError('resume input sequence differs')
   if [r['token'] for r in prior_completed]!=[r['token'] for r in rows[:len(prior_completed)]]:raise ValueError('resume committed token prefix differs')
  for row in rows:
   if sha(row['inputs'])!=row['inputs_sha256']:raise ValueError('real input file hash differs')
 save(run/'neural_progress.json',dict(stage='loading',completed_frames=prior_completed,profile_sha256=pin,memory=memory_snapshot()))
 started=time.monotonic()
 with session_for(manifest) as session:
  if args.load_only and hasattr(session,'load_all'):session.load_all()
  load_seconds=time.monotonic()-started
  save(run/'native_abi.json',session.native_abi)
  if args.load_only:
   save(run/'neural_result.json',dict(status='pass',stage='loaded',profile_sha256=pin,load_seconds=load_seconds,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Actual QNN CPU graph composition/finalization and ABI only; no real frames or task acceptance.'));return
  save(run/'neural_progress.json',dict(stage='initializing_host',committed_frames=prior_completed,profile_sha256=pin,memory=memory_snapshot()))
  with np.load(manifest['assets']['initial_state']['path'],allow_pickle=False) as archive:initial={n:archive[n].copy() for n in archive.files}
  if initial['query'].shape[0]==901:initial=pad_v1_initial_state(initial)
  observed=ObservedSession(session)
  runtime=FixedStateTransaction(observed,initial,model_sha256=manifest['assets']['native_model']['sha256'],can_bus_mode='official_test_legacy',id_scope='session');runtime.bundle_manifest_sha256=pin
  completed=copy.deepcopy(prior_completed)
  if completed:
   runtime.load_checkpoint(completed[-1]['checkpoint_path'])
   if state_digest(runtime.state)!=completed[-1]['state_sha256'] or runtime.metadata!=rows[len(completed)-1]['metadata']:raise ValueError('resume committed state/frame context differs')
  for row in rows[len(completed):args.stop_after]:
   before=state_digest(runtime.state);metadata_before=copy.deepcopy(runtime.metadata)
   save(run/'neural_progress.json',dict(stage='frame',active_token=row['token'],completed_frames=len(completed),profile_sha256=pin,previous_state_sha256=before,committed_frames=completed,memory=memory_snapshot()))
   with np.load(row['inputs'],allow_pickle=False) as archive:feed={n:archive[n].copy() for n in archive.files}
   started=time.monotonic()
   try:
    final=runtime.advance_planning(feed,metadata=row['metadata'],new_scene=row['new_scene'],optimizer_source=manifest['assets']['collision_optimizer']['path'],optimizer_sha256=manifest['assets']['collision_optimizer']['sha256'],checkpoint_path=run/('frame'+str(len(completed))+'.checkpoint.npz'))
   except BaseException as error:
    save(run/'native_abi.json',session.native_abi)
    save(run/'neural_result.json',dict(status='failed',stage='frame',failed_token=row['token'],completed_frames=completed,error_type=type(error).__name__,error=str(error),state_unchanged=state_digest(runtime.state)==before,metadata_unchanged=runtime.metadata==metadata_before,profile_sha256=pin,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Real QNN failure retained; task metrics not evaluated.'));raise
   save(run/'native_abi.json',session.native_abi)
   output_path=run/('frame'+str(len(completed))+'.outputs.npz');np.savez(output_path,**observed.outputs)
   elapsed=time.monotonic()-started;path=run/('frame'+str(len(completed))+'.final.npz');np.savez(path,planning_final=final)
   completed.append(dict(token=row['token'],new_scene=row['new_scene'],elapsed_seconds=elapsed,final_plan_path=str(path),final_plan_sha256=sha(path),state_sha256=state_digest(runtime.state),checkpoint_path=str(run/('frame'+str(len(completed))+'.checkpoint.npz')),checkpoint_sha256=sha(run/('frame'+str(len(completed))+'.checkpoint.npz')),planning_info=runtime.last_planning_info,outputs_path=str(output_path),outputs_sha256=sha(output_path),observation=observed.observation))
   # Reload after a committed frame before the next call; same native session,
   # no ORT/PT recurrent tensors are injected.
   restored=FixedStateTransaction(observed,initial,model_sha256=runtime.model_sha256,can_bus_mode=runtime.can_bus_mode,id_scope=runtime.id_scope);restored.bundle_manifest_sha256=pin;restored.load_checkpoint(completed[-1]['checkpoint_path'])
   if state_digest(restored.state)!=state_digest(runtime.state) or restored.metadata!=runtime.metadata:raise ValueError('committed checkpoint restore differs')
   runtime=restored;save(run/'neural_progress.json',dict(stage='between_frames',completed_frames=completed,profile_sha256=pin))
  save(run/'neural_result.json',dict(status='pass',stage='six_real_frames' if len(completed)==6 else 'partial_real_frames',completed_frames=completed,profile_sha256=pin,load_seconds=load_seconds,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Own-state real QNN Host/solver/checkpoint short sequence only; mini404 task metrics not evaluated.'))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  path=Path.cwd()/'neural_result.json';progress_path=Path.cwd()/'neural_progress.json'
  if not path.exists():
   progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
   save(path,dict(status='failed',stage=progress.get('stage','profile_binding'),profile_sha256=progress.get('profile_sha256'),committed_frames=progress.get('committed_frames',progress.get('completed_frames',[])),error_type=type(error).__name__,error=str(error),memory=memory_snapshot(),scope='Actual QNN/Host stage failure; task metrics not evaluated.'))
  raise
