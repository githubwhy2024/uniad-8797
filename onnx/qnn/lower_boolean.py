#!/usr/bin/env python3
"""Carry boolean routing through exact int32 0/1 for this QNN CPU backend."""
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
 if errors:raise ValueError('root inference failed')
 nodes=[];changes=[];controls={};existing={v for n in m.graph.node for v in list(n.input)+list(n.output)}
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
 for n in m.graph.node:
  if n.op_type not in ('Concat','Slice') or types[n.output[0]].tensor_type.elem_type!=onnx.TensorProto.BOOL:nodes.append(copy.deepcopy(n));continue
  data_inputs=list(n.input) if n.op_type=='Concat' else [n.input[0]]
  if any(types[v].tensor_type.elem_type!=onnx.TensorProto.BOOL for v in data_inputs):raise ValueError('mixed boolean data operands')
  prefix=n.name+'/qnn_bool';names=[prefix+f'/input_{i}' for i in range(len(data_inputs))];joined=prefix+'/joined'
  if (set(names)|{joined})&existing:raise ValueError('boolean concat names collide')
  existing.update(names+[joined]);chain=[H.make_node('Cast',[v],[name],name=name,to=onnx.TensorProto.INT32) for v,name in zip(data_inputs,names)]
  inner=copy.deepcopy(n);inner.name=prefix+'/'+n.op_type.lower();del inner.input[:];inner.input.extend(names if n.op_type=='Concat' else names+list(n.input[1:]));inner.output[0]=joined;chain.extend([inner,H.make_node('Cast',[joined],list(n.output),name=prefix+'/restore_bool',to=onnx.TensorProto.BOOL)])
  shapes=[engine.fixed_extents(types[v]) for v in n.input];key=(n.op_type,tuple(shapes),tuple((at.name,str(H.get_attribute_value(at))) for at in n.attribute),tuple((i,str(constants[v].tolist())) for i,v in enumerate(n.input) if v in constants))
  if key not in controls:
   inputs=[H.make_value_info(v,types[v]) for v in dict.fromkeys(n.input) if v not in constants];initializers=[N.from_array(constants[v],name=v) for v in dict.fromkeys(n.input) if v in constants];outputs=[H.make_value_info(v,types[v]) for v in n.output]
   ref=engine.checked_model([copy.deepcopy(n)],inputs,outputs,initializers);new=engine.checked_model(chain,inputs,outputs,initializers)
   sessions=[ort.InferenceSession(x.SerializeToString(),options,providers=['CPUExecutionProvider']) for x in (ref,new)]
   for case in ('false','true','alternating'):
    feed={v:(np.zeros(shape,np.bool_) if case=='false' else np.ones(shape,np.bool_) if case=='true' else (np.arange(np.prod(shape),dtype=np.int32).reshape(shape)%2).astype(np.bool_)) for v,shape in zip(n.input,shapes) if v not in constants}
    x,y=[s.run(None,feed)[0] for s in sessions]
    if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):raise ValueError('boolean 0/1 concat differed')
   controls[key]=dict(id=len(controls),shapes=shapes,output_shape=engine.fixed_extents(types[n.output[0]]),cases=['false','true','alternating'],actual_every_bool_exact=True)
  changes.append(dict(node=n.name,op_type=n.op_type,control_id=controls[key]['id']));nodes.extend(chain)
 if not changes:raise ValueError('no unsupported boolean routing found')
 del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.value_info[:];onnx.checker.check_model(m,full_check=True)
 if [v.SerializeToString() for v in original.graph.initializer]!=[v.SerializeToString() for v in m.graph.initializer]:raise ValueError('original weights changed')
 for boundary in ('input','output'):
  if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(m.graph,boundary)]:raise ValueError('ordered ABI changed')
 out=Path.cwd()/'model.boolean-concat.onnx';onnx.save(m,str(out));save(Path.cwd()/'boolean.json',dict(status='pass',source_model_sha256=a.model_sha256,model=str(out),model_sha256=sha(out),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),changes=changes,controls=list(controls.values()),scope='Exact boolean↔int32 0/1 concatenation/slicing only; neural task acceptance separate.'))
if __name__=='__main__':main()
