#!/usr/bin/env python3
"""Replace the unsupported integer SCA ReduceMax with integer comparisons/selects."""
import argparse,copy,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('source identity differs')
 m=onnx.load(str(a.model));original=copy.deepcopy(m);types,constants,errors=engine.prove_static_dataflow(m)
 if errors:raise ValueError('root inference failed')
 existing={v for n in m.graph.node for v in list(n.input)+list(n.output)}|{v.name for v in m.graph.initializer};nodes=[];changes=[];opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
 for n in m.graph.node:
  if n.op_type!='ReduceMax' or types[n.input[0]].tensor_type.elem_type!=onnx.TensorProto.INT64:nodes.append(copy.deepcopy(n));continue
  shape=engine.fixed_extents(types[n.input[0]]);outshape=engine.fixed_extents(types[n.output[0]]);attrs={at.name:H.get_attribute_value(at) for at in n.attribute}
  axes=constants.get(n.input[1]) if len(n.input)==2 else np.array(attrs.get('axes',[]),np.int64)
  if n.name!='__q3_sca_max' or shape!=(6,) or outshape!=() or axes is None or axes.tolist()!=[0] or attrs.get('keepdims',1)!=0:raise ValueError('unsupported integer reduction pattern')
  chain=[];weights=[];prefix=n.name+'/qnn_select';winner=None
  def fresh(suffix):
   name=prefix+'/'+suffix
   if name in existing:raise ValueError('integer reduction name collision')
   existing.add(name);return name
  for index in range(6):
   idx=fresh(f'index_{index}');value=fresh(f'value_{index}');weights.append(N.from_array(np.array([index],np.int64),name=idx));chain.append(H.make_node('Gather',[n.input[0],idx],[value],name=value,axis=0))
   if winner is None:winner=value;continue
   greater=fresh(f'greater_{index}');selected=fresh(f'selected_{index}');chain.extend([H.make_node('Greater',[winner,value],[greater],name=greater),H.make_node('Where',[greater,winner,value],[selected],name=selected)]);winner=selected
  vector_shape=fresh('vector_shape');vector=fresh('vector');weights.append(N.from_array(np.array([1],np.int64),name=vector_shape));chain.append(H.make_node('Reshape',[winner,vector_shape],[vector],name=fresh('canonical_vector')))
  scalar_shape=fresh('scalar_shape');weights.append(N.from_array(np.array([],np.int64),name=scalar_shape));chain.append(H.make_node('Reshape',[vector,scalar_shape],list(n.output),name=fresh('restore_scalar')))
  ins=[H.make_value_info(n.input[0],types[n.input[0]])];outs=[H.make_value_info(n.output[0],types[n.output[0]])];ref=H.make_model(H.make_graph([copy.deepcopy(n)],'actual-source-reduction',ins,outs,initializer=[N.from_array(constants[n.input[1]],name=n.input[1])] if len(n.input)==2 else []),opset_imports=list(original.opset_import));ref.ir_version=original.ir_version;onnx.checker.check_model(ref,full_check=True);new=engine.checked_model(chain,ins,outs,weights);sessions=[ort.InferenceSession(x.SerializeToString(),opts,providers=['CPUExecutionProvider']) for x in (ref,new)]
  cases=[]
  for base in (np.array([0,1,10201,40000,7,3],np.int64),np.array([np.iinfo(np.int64).min,-2**53-1,-1,0,2**53+1,np.iinfo(np.int64).max],np.int64)):
   for shift in range(6):cases.append(np.roll(base,shift))
  cases.extend([np.full(6,v,np.int64) for v in (0,40000,-1,np.iinfo(np.int64).min,np.iinfo(np.int64).max)])
  for value in cases:
   x,y=[s.run(None,{n.input[0]:value})[0] for s in sessions]
   if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):raise ValueError('integer max selection differs')
  changes.append(dict(node=n.name,source_shape=shape,output_shape=outshape,actual_integer_cases=len(cases),actual_ort_exact=True,arithmetic='Gather/Greater/Where/pure scalar view; no float conversion or integer arithmetic'))
  nodes.extend(chain);m.graph.initializer.extend(weights)
 if not changes:raise ValueError('observed integer ReduceMax absent')
 del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.value_info[:];onnx.checker.check_model(m,full_check=True)
 if any(v.SerializeToString()!=m.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
 for boundary in ('input','output'):
  if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(m.graph,boundary)]:raise ValueError('ordered ABI differs')
 out=Path.cwd()/'model.integer-select.onnx';onnx.save(m,str(out));save(Path.cwd()/'reduction.json',dict(status='pass',source_model_sha256=a.model_sha256,model=str(out),model_sha256=sha(out),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),changes=changes,scope='Integer selection equivalence only; actual CPU graph load/frames/tasks and signed32 range policy remain separate.'))
if __name__=='__main__':main()
