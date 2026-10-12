#!/usr/bin/env python3
"""Bounded ScatterND addressing controls matching map-head coordinate ranks."""
import argparse,json,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save
from resources import terminal
from session import NativeSession,DTYPES

def main():
 p=argparse.ArgumentParser();p.add_argument('--make-model',action='store_true');p.add_argument('--flatten',action='store_true');p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args();root=Path.cwd();shapes=dict(data=[1,300,4],indices=[1,300,2,3],updates=[1,300,2],routed=[1,300,4])
 if a.make_model:
  vi=lambda n:H.make_tensor_value_info(n,H.TensorProto.INT64 if n=='indices' else H.TensorProto.FLOAT,shapes[n]);nodes=[];weights=[]
  if a.flatten:
   nodes=[H.make_node('Reshape',['indices','index_shape'],['indices2d'],name='flatten_indices'),H.make_node('Reshape',['updates','update_shape'],['updates1d'],name='flatten_updates')];weights=[N.from_array(np.array([600,3],np.int64),name='index_shape'),N.from_array(np.array([600],np.int64),name='update_shape')]
  nodes.append(H.make_node('ScatterND',['data','indices2d' if a.flatten else 'indices','updates1d' if a.flatten else 'updates'],['routed'],name='route'));m=H.make_model(H.make_graph(nodes,'scatter-routing',[vi(n) for n in ('data','indices','updates')],[vi('routed')],weights),opset_imports=[H.make_opsetid('',18)]);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,str(root/'model.onnx'));save(root/'control_model.json',dict(status='pass',model_sha256=sha(root/'model.onnx'),flatten=a.flatten));return
 terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());convert=Path(b['resources']['model.cpp']['path']).parent;net=json.loads((convert/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind,names in [('inputs',['data','indices','updates']),('outputs',['routed'])]:
  for n in names:
   row=net[n];dtype=DTYPES[row['data_type']];entry=dict(name=n,native_name=n,shape=shapes[n],native_shape=row['dims'],dtype=dtype)
   if row['dims']!=shapes[n]:entry['wire_view']='singleton_axes'
   if dtype=='int64':entry['integer_range']=[-2**31,2**31-1]
   abi[kind].append(entry)
 source=json.loads((convert/'source_identity.json').read_text());opts=ort.SessionOptions();opts.intra_op_num_threads=1;reference=ort.InferenceSession(source['model'],opts,providers=['CPUExecutionProvider']);checks={};cases=[];backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so'
 with NativeSession(bridge['library'],b['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  data=np.arange(1200,dtype=np.float32).reshape(shapes['data']);updates=(-np.arange(600,dtype=np.float32)-.125).reshape(shapes['updates']);indices=np.zeros(shapes['indices'],np.int64);indices[0,:,:,1]=np.arange(300)[:,None];indices[0,:,:,2]=[0,2]
  for label,idx in [('normal',indices),('reversed',indices[:,::-1].copy())]:
   feed=dict(data=data,indices=idx,updates=updates);before={k:v.copy() for k,v in feed.items()};expected=reference.run(None,feed)[0];row=dict(case=label)
   try:
    out=session.run(None,feed)[0];row.update(executed=True,expected_exact=np.array_equal(out,expected),inputs_unmodified=all(np.array_equal(v,before[k]) for k,v in feed.items()))
   except Exception as e:row.update(executed=False,error_type=type(e).__name__,error=str(e))
   cases.append(row)
  save(root/'native_abi.json',session.native_abi)
 if a.flatten:checks={r['case']+'_exact_addressing':r.get('expected_exact',False) and r.get('inputs_unmodified',False) for r in cases}
 save(root/'scatter_control.json',dict(status='pass' if all(checks.values()) else 'failed',execution_status='bounded_diagnostic_complete',checks=checks,cases=cases,flatten=a.flatten,scope='Actual CPU ScatterND routing control only; unchanged float payload bytes compared for addressing, no neural tensor parity/task acceptance.'))
 if not all(checks.values()):raise ValueError('flattened CPU ScatterND routing rejected')
if __name__=='__main__':main()
