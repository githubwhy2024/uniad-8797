#!/usr/bin/env python3
"""Bounded CPU dynamic integer-addressing controls; no neural acceptance."""
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
  vi=lambda n,d,s:H.make_tensor_value_info(n,d,s)
  nodes=[H.make_node('Gather',['data','indices'],['routed'],axis=0,name='route')]
  m=H.make_model(H.make_graph(nodes,'dynamic-index-routing',[vi('data',H.TensorProto.FLOAT,[10,2]),vi('indices',H.TensorProto.INT64,[1,8])],[vi('routed',H.TensorProto.FLOAT,[1,8,2])]),opset_imports=[H.make_opsetid('',18)]);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,str(root/'model.onnx'));save(root/'control_model.json',dict(status='pass',model_sha256=sha(root/'model.onnx')));return
 terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());net=json.loads((Path(b['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind,names in [('inputs',['data','indices']),('outputs',['routed'])]:
  for n in names:
   row=net[n];dtype=DTYPES[row['data_type']];shape={'data':[10,2],'indices':[1,8],'routed':[1,8,2]}[n];entry=dict(name=n,native_name=n,shape=shape,native_shape=row['dims'],dtype=dtype)
   if row['dims']!=shape:entry['wire_view']='singleton_axes'
   if dtype=='int64':entry['integer_range']=[-2**31,2**31-1]
   abi[kind].append(entry)
 checks={};cases=[];backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so'
 with NativeSession(bridge['library'],b['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  data=np.arange(20,dtype=np.float32).reshape(10,2)
  for label,points in [('positive',[0,1,9,2,4,8,7,3]),('negative',[-1,-10,0,1,-3,9,-2,4]),('out_of_range',[10,1,2,3,4,5,6,7])]:
   x=np.array(points,np.int64)[None,:];row=dict(case=label,indices=points)
   try:
    out=session.run(None,dict(data=data,indices=x))[0];row.update(executed=True,shape=list(out.shape),dtype=str(out.dtype),expected_exact=np.array_equal(out,data[x]) if label!='out_of_range' else False)
   except Exception as e:row.update(executed=False,error_type=type(e).__name__,error=str(e))
   cases.append(row)
  checks['positive_exact']=cases[0].get('expected_exact',False);checks['invalid_rejected']=not cases[2]['executed'];save(root/'native_abi.json',session.native_abi)
  # One-part constructed manifest exercises the production failure journal against
 # the same actual compiled control library; it does not claim a partition audit.
 from partition_session import PartitionSession
 part=dict(index=0,inputs=[dict(name=v['name'],source_name=v['name']) for v in abi['inputs']],outputs=[dict(name=v['name'],source_name=v['name'],dtype=v['dtype'],shape=v['shape']) for v in abi['outputs']],drop_after=[])
 row=dict(index=0,library=b['library'],library_sha256=b['library_sha256'],abi=abi)
 manifest=dict(assets=dict(bridge=dict(path=bridge['library'],sha256=bridge['library_sha256']),backend_lib=dict(path=str(backend),sha256=sha(backend))),abi={kind:[dict(v,parent_shape=v['shape']) for v in abi[kind]] for kind in ('inputs','outputs')},parts=[row],plan=dict(parts=[part]))
 with PartitionSession(manifest) as session:
  bad=dict(data=data.copy(),indices=np.full((1,8),10,np.int64));before={k:v.copy() for k,v in bad.items()}
  try:session.run(None,bad)
  except RuntimeError as error:checks['native_error_propagated']=str(error)=='graphExecute error 6000'
  else:checks['native_error_propagated']=False
  failure=json.loads((root/'rejected_part.json').read_text());checks['error_input_journal_bound']=failure['part']==0 and failure['error_type']=='RuntimeError' and sha(failure['inputs'])==failure['inputs_sha256']
  checks['uninitialized_output_not_saved']='outputs' not in failure and not (root/'rejected_part.output.npz').exists()
  with np.load(failure['inputs'],allow_pickle=False) as archive:checks['saved_input_exact']=set(archive.files)==set(bad) and all(np.array_equal(archive[k],v) for k,v in bad.items())
  checks['rejected_input_unmodified']=all(np.array_equal(bad[k],v) for k,v in before.items())
  valid=dict(data=data,indices=np.arange(8,dtype=np.int64)[None,:]);checks['later_valid_execution']=np.array_equal(session.run(None,valid)[0],data[valid['indices']])
 save(root/'index_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,scope='Bounded actual CPU addressing diagnostic; negative behavior recorded, no neural/task acceptance.'))
 if not all(checks.values()):raise ValueError('positive integer addressing or invalid-input rejection failed')
if __name__=='__main__':main()
