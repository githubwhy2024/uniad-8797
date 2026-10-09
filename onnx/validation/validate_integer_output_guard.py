#!/usr/bin/env python3
"""Actual integer output rejection retains bytes and leaves input unchanged."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save
from session import NativeSession,IntegerOutputRangeError
from resources import terminal

def main():
 p=argparse.ArgumentParser();p.add_argument('--compile-run',type=Path,required=True);p.add_argument('--bridge-build',type=Path,required=True);a=p.parse_args();terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());row=lambda n,s,d:dict(name=n,native_name=n,shape=s,native_shape=s,dtype=d);abi=dict(schema='qnn-native-abi-v1',inputs=[row('scores',[300,10],'float32')],outputs=[dict(row(n,[300] if n in ('labels','modded') else [301],'int64'),integer_range=[0,0]) for n in ('labels','modded','joined','scattered')]);backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';scores=np.zeros((300,10),np.float32);scores[:,5]=2;before=scores.copy();checks={}
 with NativeSession(bridge['library'],b['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  try:session.run(None,dict(scores=scores))
  except IntegerOutputRangeError as error:checks.update(integer_guard_rejected=True,actual_rejected_output_exact=error.output_name=='labels' and error.output_value.dtype==np.int64 and np.array_equal(error.output_value,np.full(300,5,np.int64)),declared_range_retained=error.integer_range==[0,0])
  else:checks['integer_guard_rejected']=False
  checks['input_unchanged']=np.array_equal(scores,before);save(Path.cwd()/'native_abi.json',session.native_abi)
 save(Path.cwd()/'integer_guard_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,scope='Actual compiled CPU integer guard only; intentionally narrow control range, no change to model semantics or production bounds/task acceptance.'))
 if not all(checks.values()):raise ValueError('integer output rejection control failed')
if __name__=='__main__':main()
