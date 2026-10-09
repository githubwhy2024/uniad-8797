#!/usr/bin/env python3
"""Execute frozen mini tokens on actual QNN with per-frame task records and resume."""
import cProfile  # Load stdlib profile before adding the QNN tools directory.
import argparse,ast,copy,hashlib,json,os,pickle,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save,process_identity
from resources import session_for,terminal
from partition_session import bind_partitions
from run_qnn_real_frames import ObservedSession,memory_snapshot
sys.path[:0]=[str(ROOT/'onnx/fixed'),str(ROOT)]
import host
import qnn_planning
from qnn_mini_scope import dataset_scope

SPLITS=('mini_val','mini_train')
ADAPTER_ROOT=ROOT/'onnx/reference/adapters'
ADAPTERS={
 'frame_feed':('frame_feed.py','a5d737e69e428b2fe0453a7692fae5f258882f3f87e1094713e559f8b0d9827f'),
 'logical_task_view':('logical_task_view.py','8a32721e7df59bdcf620f33f843279ae3950c675fe51d3fbe159fd744468c7a0'),
}

def array_identity(values):
 return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()) for n,v in values.items()}

def frozen_functions():
 functions={};bindings={}
 for name,(file,digest) in ADAPTERS.items():
  path=ADAPTER_ROOT/file
  if sha(path)!=digest:raise ValueError('frozen adapter source differs: '+name)
  fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name);namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace);functions[name]=namespace[name];bindings[name]=dict(path=str(path),sha256=digest,ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest())
 return functions,bindings

def resume_journal(prior,pin,reference,limit,sequence):
 prior=prior.absolute()
 if prior.resolve()!=prior or not prior.is_relative_to(ROOT/'onnx/runs') or prior==Path.cwd():raise ValueError('different ordinary Q4 ancestor required')
 status=json.loads((prior/'status.json').read_text())
 for role in ('controller','child'):
  p=status.get(role)
  if p and p.get('startticks') is not None and process_identity(p['pid']).get('startticks')==p['startticks']:raise ValueError('mini ancestor process still active')
 progress=json.loads((prior/'mini_progress.json').read_text());completed=progress['committed_frames']
 if sha(prior/'profile.json')!=pin or progress['profile_sha256']!=pin or progress['reference_sha256']!=reference:raise ValueError('mini resume candidate/source/runtime/inputs differ')
 if not completed or len(completed)>=limit:raise ValueError('no unfinished committed mini scope')
 previous=None
 for i,row in enumerate(completed):
  if (row['split'],row['token'])!=sequence[i] or row['sequence_index']!=i:raise ValueError('committed mini token prefix differs')
  if previous is not None and row['split']==previous['split'] and row['before_state_sha256']!=previous['state_sha256']:raise ValueError('committed mini state chain is broken')
  for key in ('record','checkpoint','frame'):
   path=Path(row[key+'_path'])
   if path.resolve()!=path or not path.is_relative_to(ROOT/'onnx/runs') or sha(path)!=row[key+'_sha256']:raise ValueError('committed mini artifact differs')
  with open(row['record_path'],'rb') as stream:record=pickle.load(stream)
  if record['token']!=row['token'] or record['scene_token']!=row['scene_token']:raise ValueError('committed task record identity differs')
  previous=row
 save(Path.cwd()/'resume_binding.json',dict(ancestor=str(prior),status_sha256=sha(prior/'status.json'),progress_sha256=sha(prior/'mini_progress.json'),profile_sha256=pin,committed_prefix=len(completed)));return copy.deepcopy(completed)

def main():
 p=argparse.ArgumentParser();p.add_argument('--planning-only',action='store_true');p.add_argument('--partition-build',type=Path,required=True);p.add_argument('--bridge-build',type=Path,required=True);p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True);p.add_argument('--short-run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-sha256',required=True);p.add_argument('--resume-from',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--validation-split',choices=('all','mini_val'),default='all');args=p.parse_args();run=Path.cwd()
 selected=dataset_scope(args.validation_split);selected_splits=tuple(selected['splits']);expected_frames=selected['frames']
 if args.stop_after is None:args.stop_after=expected_frames
 if not 1<=args.stop_after<=expected_frames:raise ValueError('mini stop-after outside declared frozen scope')
 if sha(args.reference)!=args.reference_sha256:raise ValueError('frozen mini reference differs')
 reference=json.loads(args.reference.read_text());functions,adapters=frozen_functions()
 base=bind_partitions(args.partition_build,args.bridge_build,args.logical_model,args.logical_model_sha256)
 task_scope='planning' if args.planning_only else 'full'
 if len(base['abi']['outputs'])!=(25 if args.planning_only else 47):raise ValueError('mini task scope and actual 25/47 profile differ')
 short_tool=terminal(args.short_run)
 for name in ('profile.json','neural_result.json'):
  if sha(args.short_run/name)!=short_tool['artifacts'][name]['sha256']:raise ValueError('accepted short-sequence artifact differs')
 short=json.loads((args.short_run/'neural_result.json').read_text())
 if short['status']!='pass' or short['stage']!='six_real_frames' or len(short['completed_frames'])!=6:raise ValueError('full profile own-state short sequence not accepted')
 for row in short['completed_frames']:
  for name in ('checkpoint','outputs','final_plan'):
   if sha(row[name+'_path'])!=row[name+'_sha256']:raise ValueError('accepted short-sequence frame artifact differs')
 if json.loads((args.short_run/'profile.json').read_text())!=base:raise ValueError('mini profile differs from accepted actual short sequence')
 manifest=copy.deepcopy(base)
 for name,path in [('mini_driver',Path(__file__)),('mini_reference',args.reference),('task_adapter',ROOT/'onnx/fixed/validate.py')]:manifest['assets'][name]=dict(path=str(path.absolute()),sha256=sha(path),bytes=path.stat().st_size)
 if args.planning_only:
  path=Path(qnn_planning.__file__);manifest['assets']['planning_adapter']=dict(path=str(path.absolute()),sha256=sha(path),bytes=path.stat().st_size)
 manifest['task_scope']=task_scope;manifest['task_adapters']=adapters;manifest['dataset_scope']=selected;scope_helper=Path(__file__).with_name('qnn_mini_scope.py');manifest['assets']['dataset_scope_helper']=dict(path=str(scope_helper),sha256=sha(scope_helper),bytes=scope_helper.stat().st_size);save(run/'profile.json',manifest);pin=sha(run/'profile.json')
 for row in reference['assets'].values():
  if row.get('exists') and row.get('sha256') and sha(row['path'])!=row['sha256']:raise ValueError('frozen mini/evaluator asset differs: '+row['path'])
 for split in selected_splits:
  row=reference['splits'][split]
  if row['frames']!=(81 if split=='mini_val' else 323) or sha(row['info_path'])!=row['info_sha256']:raise ValueError('frozen split identity/count differs')
 sequence=[(s,t) for s in selected_splits for t in reference['splits'][s]['tokens']]
 if len(sequence)!=expected_frames or len({t for s,t in sequence})!=expected_frames:raise ValueError('mini manifest is not complete declared unique tokens')
 completed=resume_journal(args.resume_from,pin,args.reference_sha256,args.stop_after,sequence) if args.resume_from else []
 producer=json.loads((args.resume_from/'manifest.json').read_text())['repository_head'] if args.resume_from else json.loads((run/'source_identity.json').read_text())['source_commit']
 mini_manifest=copy.deepcopy(reference);mini_manifest['repository_head']=producer;mini_manifest['q4_qnn']=dict(profile_sha256=pin,reference=str(args.reference.absolute()),reference_sha256=args.reference_sha256,driver_sha256=sha(__file__),backend=manifest['backend'],task_adapters=adapters,task_scope=task_scope,dataset_scope=selected);save(run/'manifest.json',mini_manifest)
 def progress(stage,committed=None,**extra):save(run/'mini_progress.json',dict(stage=stage,committed_frames=completed if committed is None else committed,profile_sha256=pin,reference_sha256=args.reference_sha256,dataset_scope=selected,memory=memory_snapshot(),**extra))
 progress('building_frozen_datasets')
 import export,validate as ev
 from prepare_scene import frame_metadata
 import torch
 from mmcv.parallel import collate,scatter
 from mmdet3d.datasets import build_dataset
 torch.set_num_threads(4);torch.manual_seed(0);cwd=Path.cwd();os.chdir(ROOT)
 try:
  cfg=export.build_cfg(reference['assets']['config']['path'],'legacy_cuda');datasets={}
  for split in selected_splits:
   test=copy.deepcopy(cfg.data.test);test.ann_file=reference['splits'][split]['info_path'];test.data_root=reference['data_root'];test.test_mode=True;test.file_client_args=dict(backend='disk');dataset=build_dataset(test)
   if [i['token'] for i in dataset.data_infos]!=reference['splits'][split]['tokens']:raise ValueError('dataset token order differs')
   datasets[split]=dataset
 finally:os.chdir(cwd)
 coder=None if args.planning_only else ev._make_evaluator_bbox_coder(pc_range=cfg.point_cloud_range)
 with np.load(manifest['assets']['initial_state']['path'],allow_pickle=False) as archive:initial={n:archive[n].copy() for n in archive.files}
 if initial['query'].shape[0]==901:initial=host.pad_v1_initial_state(initial)
 summaries={};last_split=None;runtime=None
 with session_for(manifest) as session:
  observed=ObservedSession(session)
  def fresh():
   result=host.FixedStateTransaction(observed,initial,model_sha256=manifest['assets']['native_model']['sha256'],can_bus_mode='official_test_legacy',id_scope='session');result.bundle_manifest_sha256=pin;return result
  for split in selected_splits:
   dataset=datasets[split];rows=[r for r in completed if r['split']==split];runtime=fresh()
   if rows:
    runtime.load_checkpoint(rows[-1]['checkpoint_path'])
    if host.state_digest(runtime.state)!=rows[-1]['state_sha256'] or runtime.metadata!=rows[-1]['metadata']:raise ValueError('mini restored state/frame metadata differs')
   for index in range(len(rows),len(dataset)):
    if len(completed)>=args.stop_after:break
    info=dataset.data_infos[index];before=host.state_digest(runtime.state);before_meta=copy.deepcopy(runtime.metadata);progress('frame',split=split,index=index,active_token=info['token'],previous_state_sha256=before);started=time.monotonic()
    if list(info['cams'])!=reference['splits'][split]['camera_order']:raise ValueError('mini camera order differs')
    # Existing pipeline uses relative dataset resources under the frozen checkout.
    os.chdir(ROOT)
    try:data=scatter(collate([dataset[index]],samples_per_gpu=1),[-1])[0]
    finally:os.chdir(cwd)
    image=np.ascontiguousarray(ev._unwrap_single_tensor(data['img'],'img').numpy());command=int(ev._unwrap_single_tensor(data['command'],'command').item());feed,metadata,reset=functions['frame_feed'](info,image,frame_metadata(info),command,runtime.metadata)
    cameras=[dict(camera=n,path=cam['data_path'],sha256=sha(ROOT/cam['data_path'] if not Path(cam['data_path']).is_absolute() else cam['data_path'])) for n,cam in info['cams'].items()]
    input_seconds=time.monotonic()-started;candidate=fresh()
    if rows:candidate.load_checkpoint(rows[-1]['checkpoint_path'])
    incoming=candidate._incoming(reset);input_tracks=int(incoming['track_count']);directory=run/split;directory.mkdir(exist_ok=True);stem=directory/f'frame{index:03d}';observed.outputs=None
    try:
     graph_started=time.monotonic();final=candidate.advance_planning(feed,metadata=metadata,new_scene=reset,optimizer_source=manifest['assets']['collision_optimizer']['path'],optimizer_sha256=manifest['assets']['collision_optimizer']['sha256'])
     if observed.observation['inputs']!=array_identity({**feed,**incoming}):raise ValueError('QNN did not consume its own incoming state')
     outputs=observed.outputs
     if args.planning_only:record=qnn_planning.record(outputs,info,final,candidate.last_planning_info)
     else:
      view=functions['logical_task_view'](outputs,True)
      for name in ('cls_scores','bbox_preds','past_trajectories'):view[name]=outputs[name][:,:,:input_tracks]
      motion=host.motion_evaluator_view(view);map_result=host.merge_map_masks(view,reject_ambiguous_ties=True);labels,masks=ev._native_map_gt_from_scattered(data);counts=host.map_iou_counts(map_result,labels,masks);occ_ids=host.consecutive_occupancy_ids(view['occ_instances']);valid=not bool(ev._unwrap_single_tensor(data['gt_occ_has_invalid_frame'],'gt_occ_has_invalid_frame').item());detection=ev._decode_detection_for_evaluator(view,coder)
      record=ev._stateful_evaluator_record(view,info,detection,motion,counts,occ_ids,final,candidate.last_planning_info,valid)
     graph_host_seconds=time.monotonic()-graph_started;record_path=stem.with_suffix('.record.pkl');ev.atomic_pickle(record_path,record);checkpoint=stem.with_suffix('.checkpoint.npz');candidate.save_checkpoint(checkpoint)
     restored=fresh();restored.load_checkpoint(checkpoint)
     if host.state_digest(restored.state)!=host.state_digest(candidate.state) or restored.metadata!=metadata:raise ValueError('new mini checkpoint restore differs')
     frame_path=stem.with_suffix('.json');frame=dict(split=split,index=index,sequence_index=len(completed),token=info['token'],scene_token=info['scene_token'],metadata=metadata,new_scene=reset,before_state_sha256=before,state_sha256=host.state_digest(candidate.state),input_tracks=input_tracks,next_tracks=int(outputs['next_track_count']),decoded=int(outputs['decoded_count']),vehicles=int(outputs['vehicle_count']),overflow_flags=outputs['overflow_flags'].tolist(),observation=observed.observation,camera_sources=cameras,record_path=str(record_path),record_sha256=sha(record_path),checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),input_seconds=input_seconds,graph_and_host_seconds=graph_host_seconds,total_seconds=time.monotonic()-started)
     save(frame_path,frame);entry={**frame,'frame_path':str(frame_path),'frame_sha256':sha(frame_path)};progress('between_frames',committed=completed+[entry]);completed.append(entry);rows.append(entry);runtime=restored
    except BaseException as error:
     if observed.outputs is not None:np.savez(stem.with_suffix('.rejected.outputs.npz'),**observed.outputs)
     save(run/'native_abi.json',session.native_abi);save(run/'mini_result.json',dict(status='failed',stage='frame',error_type=type(error).__name__,error=str(error),failed_token=info['token'],committed_frames=len(completed),published_state_unchanged=host.state_digest(runtime.state)==before,published_metadata_unchanged=runtime.metadata==before_meta,profile_sha256=pin,task_acceptance='not_evaluated'));raise
    print(json.dumps(dict(split=split,index=index,token=info['token'],committed=len(completed),seconds=entry['total_seconds'])),flush=True)
   if len(rows)==len(dataset):
    records=[]
    for row in rows:
     with open(row['record_path'],'rb') as stream:records.append(pickle.load(stream))
    tokens=[x['token'] for x in records]
    if tokens!=reference['splits'][split]['tokens']:raise ValueError('mini saved records not frozen complete split')
    payload=dict(format=qnn_planning.FORMAT if args.planning_only else 'uniad-stateful-qnn-cpu-results-v2',repository_head=producer,backend='q4_qnn_partition_cpu_float32',split=split,scope='full',task_scope=task_scope,dataset_scope=selected,scene_names=reference['splits'][split]['scene_names_in_execution_order'],tokens=tokens,token_order_sha256=ev.sequence_sha256(tokens),evaluator_record_schema=qnn_planning.SCHEMA if args.planning_only else ev.EVALUATOR_RECORD_SCHEMA,records=records,collision_solver_invocations=sum(int(x['planning']['collision_solver_ran']) for x in records),model_sha256=manifest['assets']['native_model']['sha256'],initial_state_sha256=manifest['assets']['initial_state']['sha256'],profile_sha256=pin)
    path=run/split/'full/results.pkl';ev.atomic_pickle(path,payload);summaries[split]=dict(status='pass',frames=len(records),results=str(path),results_sha256=sha(path),token_order_sha256=payload['token_order_sha256']);save(path.parent/'summary.json',summaries[split]);del records,payload
   if len(completed)>=args.stop_after:break
  save(run/'native_abi.json',session.native_abi)
 save(run/'mini_result.json',dict(status='pass',execution_status='complete' if len(completed)==expected_frames else 'partial',stage=selected['complete_stage'] if len(completed)==expected_frames else 'partial_mini_records',frames=len(completed),task_scope=task_scope,dataset_scope=selected,split_summaries=summaries,profile_sha256=pin,reference_sha256=args.reference_sha256,journal_sha256=sha(run/'mini_progress.json'),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,task_acceptance='not_evaluated',scope='Actual QNN own-state and saved scope-specific records only; task metrics, portable package and board acceptance separate.'))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  path=Path.cwd()/'mini_result.json'
  if not path.exists():save(path,dict(status='failed',stage='binding_or_inputs',error_type=type(error).__name__,error=str(error),task_acceptance='not_evaluated',memory=memory_snapshot()))
  raise
