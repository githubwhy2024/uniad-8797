#!/usr/bin/env python3
"""Actual CPU GatherElements routing controls, with no neural acceptance."""
import argparse,json,sys
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save
from resources import terminal
from session import NativeSession,DTYPES

def main():
 p=argparse.ArgumentParser();p.add_argument('--make-model',action='store_true');p.add_argument('--require-exact',action='store_true');p.add_argument('--rank',type=int,choices=(4,5),default=5);p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args();root=Path.cwd()
 shape=[301,4,6,12,2] if a.rank==5 else [301,4,6,256];idxshape=[301,1]+shape[2:]
 if a.make_model:
  vi=lambda n,d,s:H.make_tensor_value_info(n,d,s)
  m=H.make_model(H.make_graph([H.make_node('GatherElements',['data','indices'],['routed'],axis=1,name='route')],'gather-elements-routing',[vi('data',H.TensorProto.FLOAT,shape),vi('indices',H.TensorProto.INT64,idxshape)],[vi('routed',H.TensorProto.FLOAT,idxshape)]),opset_imports=[H.make_opsetid('',18)]);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,str(root/'model.onnx'));save(root/'control_model.json',dict(status='pass',model_sha256=sha(root/'model.onnx'),rank=a.rank));return
 terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());br=json.loads((a.bridge_build/'bridge_build.json').read_text());net=json.loads((Path(b['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind,names in [('inputs',['data','indices']),('outputs',['routed'])]:
  for n in names:
   row=net[n];s=shape if n=='data' else idxshape;d=DTYPES[row['data_type']];e=dict(name=n,native_name=n,shape=s,native_shape=row['dims'],dtype=d)
   if row['dims']!=s:e['wire_view']='singleton_axes'
   if d=='int64':e['integer_range']=[-2**31,2**31-1]
   abi[kind].append(e)
 backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';cases=[]
 with NativeSession(br['library'],b['library'],backend,abi,dict(bridge=br['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  data=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
  for name,idx in [('zero',np.zeros(idxshape,np.int64)),('two',np.full(idxshape,2,np.int64)),('mixed',np.broadcast_to((np.arange(301)%4).reshape([301,1]+[1]*(a.rank-2)),idxshape).copy())]:
   before=idx.copy();case=dict(case=name)
   try:
    out=session.run(None,dict(data=data,indices=idx))[0];case.update(executed=True,exact=np.array_equal(out,np.take_along_axis(data,idx,axis=1)),input_unchanged=np.array_equal(idx,before))
   except Exception as error:case.update(executed=False,error_type=type(error).__name__,error=str(error),input_unchanged=np.array_equal(idx,before))
   cases.append(case)
  save(root/'native_abi.json',session.native_abi)
 save(root/'gather_elements_control.json',dict(status='failed' if a.require_exact and not all(v.get('exact',False) for v in cases) else 'pass',diagnostic_complete=True,all_exact=all(v.get('exact',False) for v in cases),cases=cases,scope='Constructed integer addressing actual CPU diagnostic only; pass means capture completed, not neural acceptance.'))
 if a.require_exact and not all(v.get('exact',False) for v in cases):raise ValueError('actual packed GatherElements control failed')
if __name__=='__main__':main()
