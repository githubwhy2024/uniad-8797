#!/usr/bin/env python3
"""Normalize proven constant negative Gather/ScatterND indices for QNN CPU."""
import argparse,copy,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('source model differs')
 m=onnx.load(str(a.model));original=copy.deepcopy(m);types,constants,errors=engine.prove_static_dataflow(m)
 if errors:raise ValueError('source root inference failed')
 changes=[];existing={v for n in m.graph.node for v in list(n.input)+list(n.output)}|{v.name for v in m.graph.initializer};opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
 for n in m.graph.node:
  if n.op_type not in ('ScatterND','Gather') or n.input[1] not in constants or not (constants[n.input[1]]<0).any():continue
  indices=constants[n.input[1]];data_shape=engine.fixed_extents(types[n.input[0]])
  k=indices.shape[-1] if n.op_type=='ScatterND' else 1
  if k<1 or k>len(data_shape) or indices.dtype not in (np.dtype('int32'),np.dtype('int64')):raise ValueError('constant Scatter coordinate contract differs')
  axis=next((at.i for at in n.attribute if at.name=='axis'),0)%len(data_shape)
  extents=np.array(data_shape[:k] if n.op_type=='ScatterND' else data_shape[axis],indices.dtype)
  if np.any(indices < -extents) or np.any(indices >= extents):raise ValueError('original Scatter indices invalid')
  normalized=np.where(indices<0,indices+extents,indices).astype(indices.dtype)
  prefix=n.name+'/qnn_nonnegative';name=prefix+'_indices'
  if name in existing:raise ValueError('normalized index name collides')
  existing.add(name);new=copy.deepcopy(n);new.input[1]=name
  # Integer data witnesses test every element, including int64 boundary values.
  data_info=H.make_tensor_value_info(n.input[0],onnx.TensorProto.INT64,data_shape)
  inputs=[data_info]
  if n.op_type=='ScatterND':
   update_shape=engine.fixed_extents(types[n.input[2]]);inputs.append(H.make_tensor_value_info(n.input[2],onnx.TensorProto.INT64,update_shape))
  output_info=H.make_tensor_value_info(n.output[0],onnx.TensorProto.INT64,engine.fixed_extents(types[n.output[0]]))
  ref=engine.checked_model([copy.deepcopy(n)],inputs,[output_info],[N.from_array(indices,name=n.input[1])]);candidate=engine.checked_model([new],inputs,[output_info],[N.from_array(normalized,name=name)])
  sessions=[ort.InferenceSession(x.SerializeToString(),opts,providers=['CPUExecutionProvider']) for x in (ref,candidate)]
  for value in (0,-2**53-3,np.iinfo(np.int64).min,np.iinfo(np.int64).max):
   data=np.arange(np.prod(data_shape),dtype=np.int64).reshape(data_shape);data.flat[-1]=value
   feed={n.input[0]:data}
   if n.op_type=='ScatterND':feed[n.input[2]]=np.full(update_shape,value,np.int64)
   x,y=[s.run(None,feed)[0] for s in sessions]
   if not np.array_equal(x,y):raise ValueError('normalized Scatter integer addressing differs')
  changes.append(dict(node=n.name,op_type=n.op_type,original_indices=indices.tolist(),normalized_indices=normalized.tolist(),data_shape=data_shape,index_dtype=str(indices.dtype),actual_ort_every_int64_exact=True))
  m.graph.initializer.append(N.from_array(normalized,name=name));n.input[1]=name
 if not changes:raise ValueError('no proven negative routing constant found')
 onnx.checker.check_model(m,full_check=True)
 if any(v.SerializeToString()!=m.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
 for boundary in ('input','output'):
  if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(m.graph,boundary)]:raise ValueError('ordered boundary changed')
 out=Path.cwd()/'model.nonnegative-scatter.onnx';onnx.save(m,str(out));save(Path.cwd()/'scatter.json',dict(status='pass',source_model_sha256=a.model_sha256,model=str(out),model_sha256=sha(out),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),changes=changes,scope='Root-proven per-node constant Gather/Scatter index normalization only; no QNN int64 arithmetic or neural acceptance implied.'))
if __name__=='__main__':main()
