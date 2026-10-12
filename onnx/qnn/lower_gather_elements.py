#!/usr/bin/env python3
"""Pack equal non-gather trailing dimensions for CPU rank-five GatherElements."""
import argparse,copy,hashlib,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def rewrite(node,data_shape,index_shape):
 if len(data_shape)!=5 or len(index_shape)!=5 or any(v<=0 for v in data_shape+index_shape):raise ValueError('positive rank-five GatherElements roots required')
 attrs={v.name:H.get_attribute_value(v) for v in node.attribute};axis=int(attrs.get('axis',0))
 if set(attrs)-{'axis'} or not -5<=axis<5:raise ValueError('unsupported GatherElements attributes')
 axis%=5
 if axis>=3 or data_shape[3:]!=index_shape[3:]:raise ValueError('only equal trailing non-gather dimensions can be merged')
 if any(index_shape[i]>data_shape[i] for i in range(5) if i!=axis):raise ValueError('GatherElements index extent exceeds data extent')
 base=node.name+'/qnn_rank4_gather_elements';weights=[];nodes=[];ports=[]
 for label,source,shape in [('data',node.input[0],data_shape),('indices',node.input[1],index_shape)]:
  packed=list(shape[:3])+[shape[3]*shape[4]];key=base+'/'+label+'_shape';out=base+'/'+label;weights.append(N.from_array(np.array(packed,np.int64),name=key));nodes.append(H.make_node('Reshape',[source,key],[out],name=out));ports.append(out)
 routed=base+'/routed';nodes.append(H.make_node('GatherElements',ports,[routed],axis=axis,name=routed));key=base+'/output_shape';weights.append(N.from_array(np.array(index_shape,np.int64),name=key));nodes.append(H.make_node('Reshape',[routed,key],list(node.output),name=base+'/restore'))
 return nodes,weights

def controls():
 rows=[];opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
 for dtype in [H.TensorProto.FLOAT,H.TensorProto.INT64]:
  for axis in [0,1,2,-4]:
   ds=(3,4,2,3,2);ins=list(ds);ins[axis%5]=1;ins=tuple(ins);node=H.make_node('GatherElements',['data','indices'],['out'],axis=axis,name='control');chain,weights=rewrite(node,ds,ins)
   vi=lambda n,d,s:H.make_tensor_value_info(n,d,list(s));inputs=[vi('data',dtype,ds),vi('indices',H.TensorProto.INT64,ins)];outputs=[vi('out',dtype,ins)]
   models=[engine.checked_model([node],inputs,outputs,[]),engine.checked_model(chain,inputs,outputs,weights)];sessions=[ort.InferenceSession(m.SerializeToString(),opts,providers=['CPUExecutionProvider']) for m in models]
   npdtype=H.tensor_dtype_to_np_dtype(dtype);data=np.arange(np.prod(ds),dtype=npdtype).reshape(ds)
   if dtype==H.TensorProto.INT64:data.flat[0]=np.iinfo(np.int64).min;data.flat[-1]=np.iinfo(np.int64).max
   else:data.flat[0]=-0.;data.flat[1]=np.nan;data.flat[2]=np.inf
   for label,points in [('first',np.zeros(ins,np.int64)),('last',np.full(ins,ds[axis%5]-1,np.int64)),('negative',np.full(ins,-1,np.int64)),('mixed',(np.arange(np.prod(ins)).reshape(ins)%ds[axis%5]).astype(np.int64))]:
    feed=dict(data=data,indices=points);ref,out=[s.run(None,feed)[0] for s in sessions]
    if ref.dtype!=out.dtype or ref.shape!=out.shape or ref.tobytes()!=out.tobytes():raise ValueError('GatherElements packed routing differs')
    rows.append(dict(dtype=str(np.dtype(npdtype)),axis=axis,case=label,exact_bits=True))
 # A varying last extent cannot be packed without changing index coordinates.
 try:rewrite(H.make_node('GatherElements',['x','i'],['o'],axis=1,name='bad'),(3,4,2,3,2),(3,1,2,2,2))
 except ValueError:pass
 else:raise ValueError('unequal trailing extent was silently accepted')
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('GatherElements source identity differs')
 model=onnx.load(str(a.model));original=copy.deepcopy(model);types,constants,errors=engine.prove_static_dataflow(model)
 if errors:raise ValueError('GatherElements source roots fail')
 existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}|{t.name for t in model.graph.initializer};nodes=[];weights=[];changes=[]
 for n in model.graph.node:
  if n.op_type!='GatherElements' or len(engine.fixed_extents(types[n.input[0]]))<=4:nodes.append(copy.deepcopy(n));continue
  ds=engine.fixed_extents(types[n.input[0]]);ins=engine.fixed_extents(types[n.input[1]]);chain,added=rewrite(n,ds,ins)
  for name in [t.name for t in added]+[v for op in chain[:-1] for v in op.output]:
   if name in existing:raise ValueError('GatherElements generated name collision')
   existing.add(name)
  nodes.extend(chain);weights.extend(added);changes.append(dict(node=n.name,data_shape=list(ds),index_shape=list(ins),source_node_sha256=hashlib.sha256(n.SerializeToString()).hexdigest(),replacement_nodes=[v.name for v in chain]))
 if not changes:raise ValueError('no rank-five GatherElements source found')
 replaced={v['node'] for v in changes};untouched={n.name:n.SerializeToString() for n in original.graph.node if n.name not in replaced}
 if any(n.name in untouched and n.SerializeToString()!=untouched[n.name] for n in nodes):raise ValueError('unrelated operation changed')
 del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(weights);del model.graph.value_info[:]
 for kind in ('input','output'):
  if [v.SerializeToString() for v in getattr(model.graph,kind)]!=[v.SerializeToString() for v in getattr(original.graph,kind)]:raise ValueError('public ABI changed')
 if any(t.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,t in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
 witnesses=controls();onnx.checker.check_model(model,full_check=True);path=Path.cwd()/'model.rank4-gather-elements.onnx';onnx.save(model,str(path));save(Path.cwd()/'gather_elements.json',dict(status='pass',model=str(path),model_sha256=sha(path),source_model=str(a.model.absolute()),source_model_sha256=a.model_sha256,script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),changes=changes,controls=witnesses,ordered_abi_unchanged=True,non_gather_elements_nodes_unchanged=True,source_weight_bytes_unchanged=True,scope='Equal root-proven trailing non-gather dimensions packed by Reshape, original dtype/index/data bytes retained; actual CPU and neural/task acceptance separate.'))
if __name__=='__main__':main()
