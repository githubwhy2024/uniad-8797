#!/usr/bin/env python3
"""Replace root-proven one-dimensional single integer ScatterND with exact routing."""
import argparse,copy,hashlib,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def rewrite(node,length,index):
 prefix=node.name+'/qnn_integer_scatter';nodes=[];weights=[];pieces=[]
 for label,start,end in [('prefix',0,index),('suffix',index+1,length)]:
  if start==end:continue
  names=[]
  for name,value in [('start',start),('end',end),('axis',0),('step',1)]:
   v=prefix+'/'+label+'_'+name;weights.append(N.from_array(np.array([value],np.int64),name=v));names.append(v)
  out=prefix+'/'+label;nodes.append(H.make_node('Slice',[node.input[0]]+names,[out],name=out))
  if label=='prefix':pieces.append(out)
  else:pieces.extend([node.input[2],out])
 if index==length-1:pieces.append(node.input[2])
 nodes.append(H.make_node('Concat',pieces,list(node.output),axis=0,name=prefix+'/join') if len(pieces)>1 else H.make_node('Identity',pieces,list(node.output),name=prefix+'/only_update'))
 return nodes,weights

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('integer Scatter source differs')
 model=onnx.load(str(a.model));original=copy.deepcopy(model);types,constants,errors=engine.prove_static_dataflow(model)
 if errors:raise ValueError('integer Scatter source roots fail')
 changes=[];nodes=[];weights=[];controls=[];opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3;existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}|{t.name for t in model.graph.initializer}
 for n in model.graph.node:
  dtype=types[n.input[0]].tensor_type.elem_type if n.input else None
  if n.op_type!='ScatterND' or dtype not in (H.TensorProto.INT64,H.TensorProto.INT32):nodes.append(copy.deepcopy(n));continue
  shape=engine.fixed_extents(types[n.input[0]]);idx=constants.get(n.input[1]);update=engine.fixed_extents(types[n.input[2]]);attrs={v.name:H.get_attribute_value(v) for v in n.attribute}
  if len(shape)!=1 or idx is None or idx.shape!=(1,1) or idx.dtype not in (np.dtype('int64'),np.dtype('int32')) or update!=(1,) or attrs.get('reduction',b'none')!=b'none' or set(attrs)-{'reduction'}:raise ValueError('integer Scatter form is outside verified single-position routing')
  position=int(idx.item());length=shape[0]
  if position<0:position+=length
  if not 0<=position<length:raise ValueError('integer Scatter source coordinate invalid')
  chain,added=rewrite(n,length,position)
  for value in [t.name for t in added]+[v for nd in chain[:-1] for v in nd.output]:
   if value in existing:raise ValueError('integer Scatter generated name collides')
   existing.add(value)
  # Independent source routing witnesses include all int64 bits; no narrowing casts.
  npdtype=H.tensor_dtype_to_np_dtype(dtype);limits=np.iinfo(npdtype)
  for size,pos in [(7,0),(7,3),(7,6),(length,position)]:
   refnode=H.make_node('ScatterND',['x','index','updates'],['y'],name='single');cn,cw=rewrite(refnode,size,pos);vi=lambda name,sz:H.make_tensor_value_info(name,dtype,[sz]);ins=[vi('x',size),vi('updates',1)];outs=[vi('y',size)];ref=engine.checked_model([refnode],ins,outs,[N.from_array(np.array([[pos]],np.int64),name='index')]);candidate=engine.checked_model(cn,ins,outs,cw);sessions=[ort.InferenceSession(m.SerializeToString(),opts,providers=['CPUExecutionProvider']) for m in (ref,candidate)]
   for value in [limits.min,limits.max,-1,0]:
    x=np.arange(size,dtype=npdtype);x[0]=limits.min;x[-1]=limits.max;feed=dict(x=x,updates=np.array([value],npdtype));rv,cv=[s.run(None,feed)[0] for s in sessions]
    if rv.dtype!=cv.dtype or rv.shape!=cv.shape or not np.array_equal(rv,cv):raise ValueError('integer Scatter exact routing control differs')
   controls.append(dict(dtype=str(np.dtype(npdtype)),length=size,position=pos,actual_ort_cases=4,all_exact=True))
  changes.append(dict(node=n.name,dtype=H.TensorProto.DataType.Name(dtype),length=length,position=position,source_node_sha256=hashlib.sha256(n.SerializeToString()).hexdigest(),replacement_nodes=[v.name for v in chain]));nodes.extend(chain);weights.extend(added)
 if not changes:raise ValueError('no supported integer Scatter source found')
 untouched={n.name:n for n in original.graph.node if n.name not in {r['node'] for r in changes}}
 if any(n.name in untouched and n.SerializeToString()!=untouched[n.name].SerializeToString() for n in nodes):raise ValueError('non-Scatter source operation changed')
 del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(weights);del model.graph.value_info[:]
 for kind in ('input','output'):
  if [v.SerializeToString() for v in getattr(model.graph,kind)]!=[v.SerializeToString() for v in getattr(original.graph,kind)]:raise ValueError('integer Scatter changed ordered public ABI')
 if any(t.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,t in enumerate(original.graph.initializer)):raise ValueError('integer Scatter changed source weight bytes')
 onnx.checker.check_model(model,full_check=True);path=Path.cwd()/'model.integer-scatter.onnx';onnx.save(model,str(path));save(Path.cwd()/'integer_scatter.json',dict(status='pass',model=str(path),model_sha256=sha(path),source_model=str(a.model.absolute()),source_model_sha256=a.model_sha256,script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),changes=changes,controls=controls,ordered_abi_unchanged=True,non_scatter_nodes_unchanged=True,source_weight_bytes_unchanged=True,scope='Root-proven single integer ScatterND replaced by pure integer Slice/Concat; all integer bits preserved, no float cast or precision change. Actual CPU/neural/task acceptance remain separate.'))
if __name__=='__main__':main()
