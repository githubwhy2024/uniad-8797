#!/usr/bin/env python3
"""Run an actual many-part CPU graph with repeated own-state boundary controls."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from partition_session import bind_partitions,PartitionSession
from tool_run import sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--expected-step',type=int,default=64);p.add_argument('--build-run',type=Path,required=True);p.add_argument('--bridge-build',type=Path,required=True);p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True);a=p.parse_args();manifest=bind_partitions(a.build_run,a.bridge_build,a.logical_model,a.logical_model_sha256);save(Path.cwd()/'profile.json',manifest);checks={};statistics=[]
 with PartitionSession(manifest) as session:
  session.load_all();checks['actual_load_all_parts']=len(session.native_abi['parts'])==len(manifest['parts']);feed=dict(state=np.zeros((512,512),np.float32),delta=np.ones((512,512),np.float32))
  for i in range(6):
   before={k:v.copy() for k,v in feed.items()};output=session.run(None,feed)[0];expected=np.full((512,512),a.expected_step*(i+1),np.float32);diff=float(np.abs(output-expected).max());checks['frame'+str(i)+'_shape_dtype']=output.shape==expected.shape and output.dtype==expected.dtype;checks['frame'+str(i)+'_exact_control_arithmetic']=diff==0;checks['frame'+str(i)+'_inputs_unchanged']=all(np.array_equal(v,before[k]) for k,v in feed.items());checks['frame'+str(i)+'_all_parts_executed']=len(session.last_execution)==len(manifest['parts']);statistics.append(dict(frame=i,max_abs_diff=diff,part_executions=session.last_execution));feed['state']=output
  checks['empty_output_selection']=session.run([],feed)==[]
  for label,bad in [('shape',dict(feed,state=np.zeros((512,511),np.float32))),('dtype',dict(feed,delta=feed['delta'].astype(np.float64))),('nonfinite',dict(feed,delta=np.full((512,512),np.nan,np.float32))),('names',dict(state=feed['state']))]:
   try:session.run(None,bad)
   except ValueError:checks[label+'_rejected']=True
   else:checks[label+'_rejected']=False
 try:session.run(None,feed)
 except RuntimeError:checks['closed_rejected']=True
 else:checks['closed_rejected']=False
 save(Path.cwd()/'partition_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,statistics=statistics,profile_sha256=sha(Path.cwd()/'profile.json'),scope='Synthetic many-part actual CPU API only; no UniAD or task acceptance.'));save(Path.cwd()/'native_abi.json',session.native_abi)
 if not all(checks.values()):raise ValueError('partition session control failed')
if __name__=='__main__':main()
