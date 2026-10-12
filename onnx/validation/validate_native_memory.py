#!/usr/bin/env python3
"""Measure actual CPU graph residency with repeated live-sized operation outputs."""
import argparse,json,resource,sys,time
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import sha,save,SDK
from session import NativeSession
from resources import terminal

def memory():
 result={}
 for line in Path('/proc/self/status').read_text().splitlines():
  key,value=line.split(':',1)
  if key in ('VmSize','VmRSS','VmHWM'):result[key+'_kib']=int(value.strip().split()[0])
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--depth',type=int,required=True);p.add_argument('--make-model',action='store_true');p.add_argument('--alias-control',action='store_true');p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args()
 if a.depth not in (4,64):raise ValueError('fixed small memory-control depth required')
 if a.make_model:
  shape=[512,512];ins=[H.make_tensor_value_info(n,onnx.TensorProto.FLOAT,shape) for n in ('state','delta')];outs=[H.make_tensor_value_info('result',onnx.TensorProto.FLOAT,shape)];nodes=[];prev='state'
  if a.alias_control:nodes.extend([H.make_node('Identity',['bias'],['bias_alias'],name='weight_identity'),H.make_node('Add',['delta','bias_alias'],['effective_delta'],name='runtime_delta')])
  for i in range(a.depth):
   name='result' if i==a.depth-1 else 'intermediate_'+str(i);nodes.append(H.make_node('Add',[prev,'effective_delta' if a.alias_control else 'delta'],[name],name='addition_'+str(i)));prev=name
  model=H.make_model(H.make_graph(nodes,'memory_control',ins,outs,initializer=[N.from_array(np.ones((512,512),np.float32),name='bias')] if a.alias_control else []),opset_imports=[H.make_opsetid('',17)]);model.ir_version=8;onnx.checker.check_model(model,full_check=True);path=Path.cwd()/'model.memory.onnx';onnx.save(model,str(path));save(Path.cwd()/'memory_model.json',dict(status='pass',model=str(path),model_sha256=sha(path),depth=a.depth,intermediate_output_bytes=(a.depth-1)*512*512*4));return
 terminal(a.compile_run);terminal(a.bridge_build);build=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';abi=dict(schema='qnn-native-abi-v1',inputs=[dict(name=n,native_name=n,shape=[512,512],native_shape=[512,512],dtype='float32') for n in ('state','delta')],outputs=[dict(name='result',native_name='result',shape=[512,512],native_shape=[512,512],dtype='float32')]);hashes=dict(bridge=bridge['library_sha256'],model_lib=build['library_sha256'],backend_lib=build['cpu_backend_sha256']);before=memory();started=time.monotonic()
 with NativeSession(bridge['library'],build['library'],backend,abi,hashes) as s:
  loaded=memory();seconds=time.monotonic()-started;feed=dict(state=np.zeros((512,512),np.float32),delta=np.ones((512,512),np.float32));out=s.run(None,feed)[0];executed=memory();actual=float(out.flat[0]);max_abs=float(np.abs(out-a.depth).max());finite=bool(np.isfinite(out).all())
 after=memory();save(Path.cwd()/'memory_control.json',dict(status='pass' if finite and max_abs==0 else 'failed',depth=a.depth,before=before,loaded=loaded,executed=executed,closed=after,load_seconds=seconds,observed_first_value=actual,max_abs_diff=max_abs,intermediate_output_bytes=(a.depth-1)*512*512*4,model_sha256=build['source_model_sha256'],resources=hashes,script_sha256=sha(__file__),scope='Synthetic actual CPU graph residency/API only; no UniAD frames or task acceptance.'))
 if not finite or max_abs:raise ValueError('synthetic API control failed')
if __name__=='__main__':main()
