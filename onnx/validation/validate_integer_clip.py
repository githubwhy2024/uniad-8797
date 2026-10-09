#!/usr/bin/env python3
"""Actual CPU comparison/select integer clipping controls, no neural acceptance."""
import argparse,json,sys
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save
from resources import terminal
from session import NativeSession,DTYPES

def main():
 p=argparse.ArgumentParser();p.add_argument('--make-model',action='store_true');p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args();root=Path.cwd()
 if a.make_model:
  vi=lambda n:H.make_tensor_value_info(n,H.TensorProto.INT64,[1,32]);nodes=[H.make_node('Clip',['x','one',''],['lower'],name='lower'),H.make_node('Clip',['x','zero','max199'],['bounded'],name='bounded'),H.make_node('Clip',['x','','max59'],['upper'],name='upper')];weights=[N.from_array(np.array(v,np.int64),name=n) for n,v in [('one',1),('zero',0),('max199',199),('max59',59)]];m=H.make_model(H.make_graph(nodes,'integer-clip-control',[vi('x')],[vi(n) for n in ('lower','bounded','upper')],weights),opset_imports=[H.make_opsetid('',18)]);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,str(root/'model.onnx'));save(root/'control_model.json',dict(status='pass',model_sha256=sha(root/'model.onnx'),scope='Constructed three scalar-bound integer Clip patterns only.'));return
 terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());net=json.loads((Path(b['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind,names in [('inputs',['x']),('outputs',['lower','bounded','upper'])]:
  for n in names:
   row=net[n]
   if DTYPES[row['data_type']]!='int64' or row['dims']!=[1,32]:raise ValueError('integer control converter boundary differs')
   abi[kind].append(dict(name=n,native_name=n,shape=[1,32],native_shape=row['dims'],dtype='int64',integer_range=[-2**31,2**31-1]))
 backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';checks={};stats=[]
 with NativeSession(bridge['library'],b['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  points=np.array([-2**31,-100,-1,0,1,2,6,58,59,60,198,199,200,2**31-1],np.int64)
  for i in range(len(points)):
   x=np.resize(np.roll(points,i),(1,32));before=x.copy();out=session.run(None,{'x':x});expected=[np.maximum(x,1),np.clip(x,0,199),np.minimum(x,59)]
   checks[f'case{i}_exact_integer_select']=all(np.array_equal(v,e) and v.shape==e.shape and v.dtype==e.dtype for v,e in zip(out,expected));checks[f'case{i}_input_unchanged']=np.array_equal(x,before);stats.append(dict(case=i,all_exact=checks[f'case{i}_exact_integer_select']))
  for name,value in [('above',2**31),('below',-2**31-1)]:
   try:session.run(None,{'x':np.full((1,32),value,np.int64)})
   except ValueError:checks[name+'_rejected_before_api']=True
   else:checks[name+'_rejected_before_api']=False
  save(root/'native_abi.json',session.native_abi)
 save(root/'integer_clip_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,statistics=stats,compile_result_sha256=sha(a.compile_run/'result.json'),scope='Actual typed CPU integer select/clip and signed32-boundary controls only; no UniAD recurrence or task acceptance.'))
 if not all(checks.values()):raise ValueError('actual CPU integer Clip replacement rejected')
if __name__=='__main__':main()
