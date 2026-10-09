#!/usr/bin/env python3
"""Independently audit QNN mini token, artifact, own-state and saved-record closure."""
import cProfile
import argparse,copy,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save
from resources import terminal
sys.path.insert(0,str(ROOT/'onnx/fixed'))
import host,validate as ev
import qnn_planning
from qnn_mini_scope import require_complete,scope_of

def identities(values):
 return {n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()) for n,v in values.items()}

def same_record(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same_record(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same_record(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--mini-run',type=Path,required=True);a=p.parse_args();run=a.mini_run.absolute();tool=terminal(run);result=json.loads((run/'mini_result.json').read_text());profile=json.loads((run/'profile.json').read_text());manifest=json.loads((run/'manifest.json').read_text());journal=json.loads((run/'mini_progress.json').read_text());checks={};scope=result.get('task_scope','full');planning_only=scope=='planning'
 if scope not in ('planning','full'):raise ValueError('unknown QNN mini task scope')
 def check(name,value):
  checks[name]=bool(value)
  if not value:raise ValueError('mini audit rejected: '+name)
 selected=require_complete(result,profile,manifest['q4_qnn'],journal);expected_frames=selected['frames'];selected_splits=selected['splits']
 check('complete_actual_qnn_record_run',result['task_acceptance']=='not_evaluated')
 for file in ('mini_result.json','profile.json'):
  check(file+'_terminal_hash',sha(run/file)==tool['artifacts'][file]['sha256'])
 check('profile_journal_result_pin',sha(run/'profile.json')==result['profile_sha256']==journal['profile_sha256']==manifest['q4_qnn']['profile_sha256'])
 check('journal_terminal_hash',sha(run/'mini_progress.json')==result['journal_sha256'])
 check('actual_backend_identity',profile['backend']['type']=='QNN_CPU' and profile['backend']['precision']=='float32' and profile['backend']['execution']=='partitioned' and len(profile['abi']['outputs'])==(25 if planning_only else 47) and profile.get('task_scope','full')==scope and manifest['q4_qnn'].get('task_scope','full')==scope)
 for name,row in profile['assets'].items():check('asset_'+name,sha(row['path'])==row['sha256'])
 frames=journal['committed_frames'];sequence=[(s,t) for s in selected_splits for t in manifest['splits'][s]['tokens']]
 check('declared_unique_frozen_tokens',len(frames)==expected_frames and [(r['split'],r['token']) for r in frames]==sequence and len({r['token'] for r in frames})==expected_frames)
 with np.load(profile['assets']['initial_state']['path'],allow_pickle=False) as archive:initial={n:archive[n].copy() for n in archive.files}
 if initial['query'].shape[0]==901:initial=host.pad_v1_initial_state(initial)
 runtime=None;last_split=None;last_scene=None;frame_checks=[]
 for number,row in enumerate(frames):
  label='frame'+str(number);split=row['split'];fresh_split=split!=last_split
  if fresh_split:
   runtime=host.FixedStateTransaction(None,initial,model_sha256=profile['assets']['native_model']['sha256'],can_bus_mode='official_test_legacy',id_scope='session');runtime.bundle_manifest_sha256=result['profile_sha256'];last_scene=None
  check(label+'_sequence_and_scene',row['sequence_index']==number and row['new_scene']==(row['scene_token']!=last_scene))
  check(label+'_before_state',host.state_digest(runtime.state)==row['before_state_sha256'])
  incoming=runtime._incoming(row['new_scene']);expected_input=identities(incoming)
  check(label+'_actual_own_state_inputs',all(row['observation']['inputs'].get(n)==v for n,v in expected_input.items()))
  check(label+'_inputs_unchanged',row['observation']['inputs_unmodified'] is True and row['observation']['outputs_returned'] is True)
  for kind in ('frame','record','checkpoint'):
   path=Path(row[kind+'_path']);check(label+'_'+kind+'_hash',path.resolve()==path and path.is_relative_to(ROOT/'onnx/runs') and sha(path)==row[kind+'_sha256'])
  frame=json.loads(Path(row['frame_path']).read_text());check(label+'_frame_journal_identical',all(frame.get(k)==v for k,v in row.items() if k not in ('frame_path','frame_sha256')))
  runtime.load_checkpoint(row['checkpoint_path']);check(label+'_checkpoint_state_metadata',host.state_digest(runtime.state)==row['state_sha256'] and runtime.metadata==row['metadata'])
  state_outputs={('bev_embed' if n=='prev_bev' else 'next_'+n):v for n,v in runtime.state.items()};observations=identities(state_outputs)
  check(label+'_actual_state_outputs',all(row['observation']['outputs'].get(n)==v for n,v in observations.items()))
  with open(row['record_path'],'rb') as stream:record=pickle.load(stream)
  check(label+'_task_identity',record['schema']==(qnn_planning.SCHEMA if planning_only else ev.EVALUATOR_RECORD_SCHEMA) and record['token']==row['token'] and record['scene_token']==row['scene_token'])
  if planning_only:qnn_planning.validate_record(record)
  raw=record['planning']['raw'];optimized=record['planning']['optimized'];check(label+'_raw_planning_actual_output',identities({'planning_raw':raw})['planning_raw']==row['observation']['outputs']['planning_raw'])
  check(label+'_final_planning_abi',optimized.shape==(1,6,2) and optimized.dtype==np.float32 and np.isfinite(optimized).all())
  check(label+'_overflow_rejected_before_commit',row['overflow_flags']==[False,False,False])
  frame_checks.append(dict(sequence_index=number,split=split,token=row['token'],state_sha256=row['state_sha256']));last_split=split;last_scene=row['scene_token']
 for split in selected_splits:
  row=result['split_summaries'][split];check(split+'_result_hash',sha(row['results'])==row['results_sha256'])
  with open(row['results'],'rb') as stream:payload=pickle.load(stream)
  rows=[r for r in frames if r['split']==split]
  check(split+'_declared_dataset_scope',scope_of(payload)==selected)
  check(split+'_complete_saved_payload',payload['repository_head']==manifest['repository_head'] and payload['backend']=='q4_qnn_partition_cpu_float32' and payload['scope']=='full' and payload.get('task_scope','full')==scope and payload['format']==(qnn_planning.FORMAT if planning_only else 'uniad-stateful-qnn-cpu-results-v2') and payload['evaluator_record_schema']==(qnn_planning.SCHEMA if planning_only else ev.EVALUATOR_RECORD_SCHEMA) and payload['tokens']==manifest['splits'][split]['tokens'] and payload['profile_sha256']==result['profile_sha256'] and len(payload['records'])==len(rows))
  record_equal=True
  for record,row in zip(payload['records'],rows):
   with open(row['record_path'],'rb') as stream:per_frame=pickle.load(stream)
   record_equal=record_equal and same_record(record,per_frame)
  check(split+'_record_content_identical',record_equal)
  check(split+'_token_hash',ev.sequence_sha256(payload['tokens'])==payload['token_order_sha256']==manifest['splits'][split]['token_order_sha256'])
  del payload
 save(Path.cwd()/'mini_audit.json',dict(status='pass',frames=expected_frames,task_scope=scope,dataset_scope=selected,checks=checks,frame_checks=frame_checks,mini_result_sha256=sha(run/'mini_result.json'),profile_sha256=result['profile_sha256'],manifest_sha256=sha(run/'manifest.json'),journal_sha256=sha(run/'mini_progress.json'),auditor_sha256=sha(__file__),scope='Independent frozen token/source/artifact, actual own-state input/output and evaluator record closure; task metric acceptance remains separate.'))
if __name__=='__main__':main()
