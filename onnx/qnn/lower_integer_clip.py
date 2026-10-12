#!/usr/bin/env python3
"""Lower integer Clip to exact integer comparison/select after actual CPU failure."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def lower(model,types,constants):
 original=copy.deepcopy(model);nodes=[];weights=[];changes=[];controls={};existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}|{t.name for t in model.graph.initializer};options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
 for node in model.graph.node:
  dtype=types[node.input[0]].tensor_type.elem_type if node.input else None
  if node.op_type!='Clip' or dtype not in (H.TensorProto.INT64,H.TensorProto.INT32):nodes.append(copy.deepcopy(node));continue
  if len(node.input)>3 or len(node.output)!=1 or node.attribute:raise ValueError('unsupported integer Clip form')
  bounds=[]
  for index in (1,2):
   if index>=len(node.input) or not node.input[index]:bounds.append(None);continue
   value=constants.get(node.input[index])
   if value is None or value.size!=1 or value.dtype!=np.dtype(H.tensor_dtype_to_np_dtype(dtype)):raise ValueError('integer Clip bound lacks scalar integer root')
   bounds.append(int(value.item()))
  if bounds==[None,None] or (bounds[0] is not None and bounds[1] is not None and bounds[0]>bounds[1]):raise ValueError('invalid integer clip bound policy')
  prefix=node.name+'/qnn_integer_clip';chain=[];added=[];value=node.input[0];steps=[(i,b) for i,b in enumerate(bounds) if b is not None]
  def fresh(name):
   result=prefix+'/'+name
   if result in existing:raise ValueError('integer Clip generated name collides')
   existing.add(result);return result
  for pos,(index,bound) in enumerate(steps):
   threshold=fresh('bound'+str(index));condition=fresh('condition'+str(index));output=node.output[0] if pos==len(steps)-1 else fresh('clipped'+str(index));added.append(N.from_array(np.array(bound,dtype=H.tensor_dtype_to_np_dtype(dtype)),name=threshold));chain.append(H.make_node('Less' if index==0 else 'Greater',[value,threshold],[condition],name=condition));chain.append(H.make_node('Where',[condition,threshold,value],[output],name=fresh('select'+str(index))));value=output
  key=(dtype,tuple(bounds))
  if key not in controls:
   # Source rank is irrelevant to elementwise Clip; exercise the scalar integer bounds on rank2.
   source=copy.deepcopy(node);del source.input[:];source.input.extend(['x']+[(f'b{i}' if b is not None else '') for i,b in enumerate(bounds)]);del source.output[:];source.output.append('y')
   reference_weights=[N.from_array(np.array(b,dtype=H.tensor_dtype_to_np_dtype(dtype)),name=f'b{i}') for i,b in enumerate(bounds) if b is not None]
   in_info=H.make_tensor_value_info('x',dtype,[1,13]);out_info=H.make_tensor_value_info('y',dtype,[1,13]);reference=H.make_model(H.make_graph([source],'source-integer-clip',[in_info],[out_info],reference_weights),opset_imports=list(model.opset_import));reference.ir_version=model.ir_version
   control_chain=copy.deepcopy(chain)
   for n in control_chain:
    for field in ('input','output'):
     names=getattr(n,field)
     for i,v in enumerate(names):names[i]='x' if v==node.input[0] else 'y' if v==node.output[0] else v
   candidate=H.make_model(H.make_graph(control_chain,'integer-select-clip',[in_info],[out_info],added),opset_imports=list(model.opset_import));candidate.ir_version=model.ir_version
   sessions=[ort.InferenceSession(m.SerializeToString(),options,providers=['CPUExecutionProvider']) for m in (reference,candidate)];limits=np.iinfo(H.tensor_dtype_to_np_dtype(dtype));points=[limits.min,limits.max,-1,0,1,2,6,59,199]
   for bound in bounds:
    if bound is not None:points.extend([max(limits.min,bound-1),bound,min(limits.max,bound+1)])
   if dtype==H.TensorProto.INT64:points.extend([-2**53-1,2**53+1])
   points=np.array(points,dtype=H.tensor_dtype_to_np_dtype(dtype));values=np.resize(points,(1,13));count=0
   for shift in range(len(points)):
    feed={'x':np.resize(np.roll(points,shift),(1,13))};a,b=[s.run(None,feed)[0] for s in sessions]
    if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):raise ValueError('actual integer Clip/select control differs')
    count+=1
   controls[key]=dict(dtype=str(values.dtype),bounds=bounds,cases=count,actual_ort_exact=True,includes_integer_extremes=True)
  changes.append(dict(node=node.name,dtype=H.TensorProto.DataType.Name(dtype),shape=list(engine.fixed_extents(types[node.input[0]])),bounds=bounds,source_node_sha256=hashlib.sha256(node.SerializeToString()).hexdigest(),replacement_nodes=[n.name for n in chain]));nodes.extend(chain);weights.extend(added)
 if not changes:raise ValueError('no integer Clip matched')
 untouched={n.name:n for n in original.graph.node if n.name not in {r['node'] for r in changes}}
 if any(n.name in untouched and n.SerializeToString()!=untouched[n.name].SerializeToString() for n in nodes):raise ValueError('non-Clip source operation changed')
 del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(weights);del model.graph.value_info[:]
 for kind in ('input','output'):
  if [v.SerializeToString() for v in getattr(model.graph,kind)]!=[v.SerializeToString() for v in getattr(original.graph,kind)]:raise ValueError('integer Clip changed public ordered ABI')
 if any(t.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,t in enumerate(original.graph.initializer)):raise ValueError('integer Clip changed source weights')
 onnx.checker.check_model(model,full_check=True);return dict(changes=changes,controls=list(controls.values()),non_clip_nodes_unchanged=True,source_weight_bytes_unchanged=True,ordered_abi_unchanged=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('source model differs')
 model=onnx.load(str(a.model));types,constants,errors=engine.prove_static_dataflow(model)
 if errors:raise ValueError('fresh source roots fail')
 evidence=lower(model,types,constants);path=Path.cwd()/'model.integer-clip.onnx';onnx.save(model,str(path));save(Path.cwd()/'integer_clip.json',dict(status='pass',source_model=str(a.model.absolute()),source_model_sha256=a.model_sha256,model=str(path),model_sha256=sha(path),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),scope='Integer Clip exact comparison/select only; no float arithmetic, tolerance/threshold/dtype changes or neural/task acceptance.',**evidence))
if __name__=='__main__':main()
