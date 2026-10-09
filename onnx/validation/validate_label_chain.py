#!/usr/bin/env python3
"""Actual CPU integer label routing controls matching detector/motion branches."""
import argparse,json,sys
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import SDK,sha,save
from resources import terminal
from session import NativeSession,DTYPES

def main():
 p=argparse.ArgumentParser();p.add_argument('--make-model',action='store_true');p.add_argument('--require-exact',action='store_true');p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args();root=Path.cwd();shapes=dict(scores=[300,10],labels=[300],modded=[300],joined=[301],scattered=[301])
 if a.make_model:
  vi=lambda n:H.make_tensor_value_info(n,H.TensorProto.FLOAT if n=='scores' else H.TensorProto.INT64,shapes[n]);nodes=[H.make_node('ArgMax',['scores'],['labels'],axis=1,keepdims=0,name='labels'),H.make_node('Mod',['labels','ten'],['modded'],name='modulo'),H.make_node('Concat',['modded','tail'],['joined'],axis=0,name='join'),H.make_node('ScatterND',['joined','index','ego'],['scattered'],name='scatter')];weights=[N.from_array(np.array(10,np.int64),name='ten'),N.from_array(np.array([6],np.int64),name='tail'),N.from_array(np.array([[300]],np.int64),name='index'),N.from_array(np.array([0],np.int64),name='ego')];m=H.make_model(H.make_graph(nodes,'label-chain',[vi('scores')],[vi(n) for n in ('labels','modded','joined','scattered')],weights),opset_imports=[H.make_opsetid('',18)]);m.ir_version=8;onnx.checker.check_model(m,full_check=True);onnx.save(m,str(root/'model.onnx'));save(root/'control_model.json',dict(status='pass',model_sha256=sha(root/'model.onnx')));return
 terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());net=json.loads((Path(b['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind,names in [('inputs',['scores']),('outputs',['labels','modded','joined','scattered'])]:
  for n in names:
   row=net[n];dtype=DTYPES[row['data_type']];entry=dict(name=n,native_name=n,shape=shapes[n],native_shape=row['dims'],dtype=dtype)
   if row['dims']!=shapes[n]:entry['wire_view']='singleton_axes'
   if dtype=='int64':entry['integer_range']=[-2**31,2**31-1]
   abi[kind].append(entry)
 backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';cases=[]
 with NativeSession(bridge['library'],b['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=b['library_sha256'],backend_lib=sha(backend))) as session:
  for shift in (0,3,7):
   label=(np.arange(300)+shift)%10;scores=np.zeros((300,10),np.float32);scores[np.arange(300),label]=3;before=scores.copy();expected=dict(labels=label.astype(np.int64),modded=label.astype(np.int64),joined=np.concatenate([label,[6]]).astype(np.int64),scattered=np.concatenate([label,[0]]).astype(np.int64));actual=dict(zip(expected,session.run(None,dict(scores=scores))));path=root/('case'+str(shift)+'.outputs.npz');np.savez(path,**actual);cases.append(dict(shift=shift,inputs_unmodified=np.array_equal(before,scores),outputs=str(path),outputs_sha256=sha(path),statistics={n:dict(exact=np.array_equal(actual[n],v),minimum=int(actual[n].min()),maximum=int(actual[n].max()),mismatch_count=int((actual[n]!=v).sum())) for n,v in expected.items()}))
  save(root/'native_abi.json',session.native_abi)
 exact=all(all(v['exact'] for v in r['statistics'].values()) and r['inputs_unmodified'] for r in cases)
 save(root/'label_control.json',dict(status='failed' if a.require_exact and not exact else 'pass',execution_status='bounded_diagnostic_complete',cases=cases,all_exact=exact,scope='Actual typed CPU integer ArgMax/Mod/Concat/Scatter endpoint diagnostic; mismatches retained, no neural/task acceptance.'))
 if a.require_exact and not exact:raise ValueError('integer label chain addressing rejected')
if __name__=='__main__':main()
