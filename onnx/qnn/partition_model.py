#!/usr/bin/env python3
"""Cut a proven graph at dependency boundaries without changing its operations."""
import argparse,copy,hashlib,json,math,re,sys
from pathlib import Path
import numpy as np,onnx
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def layout_hints(model,types):
 """Track only rank-four standard spatial axes and exact axes-preserving routes."""
 axes={};spatial={'Conv','ConvTranspose','BatchNormalization','MaxPool','AveragePool','GridSample'};elementwise={'Identity','Relu','LeakyRelu','Sigmoid','Tanh','Clip','Cast','Add','Sub','Mul','Div','Pow','Where','Equal','Greater','Less','GreaterOrEqual','LessOrEqual','And','Or','Not','Slice','Resize'}
 for node in model.graph.node:
  outputs=[v for v in node.output if len(engine.fixed_extents(types[v]) or ())==4]
  if not outputs:continue
  tag=None
  if node.op_type in spatial:tag=('N','C','H','W')
  elif node.op_type=='Concat' and all(v in axes for v in node.input) and len({axes[v] for v in node.input})==1:tag=axes[node.input[0]]
  elif node.op_type=='Transpose' and node.input[0] in axes:
   perm=next((list(at.ints) for at in node.attribute if at.name=='perm'),[3,2,1,0]);tag=tuple(axes[node.input[0]][i] for i in perm)
  elif node.op_type in elementwise:
   known={axes[v] for v in node.input if v in axes}
   if len(known)==1:tag=next(iter(known))
  elif node.op_type=='Reshape' and node.input[0] in axes and engine.fixed_extents(types[node.input[0]])==engine.fixed_extents(types[node.output[0]]):tag=axes[node.input[0]]
  if tag is not None:
   for v in outputs:axes[v]=tag
 return {v:''.join(a) for v,a in axes.items() if ''.join(a) in ('NCHW','NHWC')}

def make_parts(model,types,budget,output,shared=None,segments_override=None):
 shared=set() if shared is None else set(shared)
 initializers={t.name:t for t in model.graph.initializer};constants={};immutable=set(initializers);root_indices=[]
 for i,n in enumerate(model.graph.node):
  if n.op_type=='Constant' or (n.input and all(v in immutable for v in n.input if v) and n.op_type not in ('RandomNormal','RandomUniform','RandomNormalLike','RandomUniformLike','Multinomial','Dropout')):
   immutable.update(n.output);constants.update({v:n for v in n.output});root_indices.append(i)
 original_outputs={v.name for v in model.graph.output};original_inputs={v.name for v in model.graph.input};producer={v:(i,n) for i,n in enumerate(model.graph.node) for v in n.output};needed=set(original_outputs);live=set();pending=list(needed)
 while pending:
  value=pending.pop()
  if value not in producer:continue
  i,n=producer[value]
  if i in live:continue
  live.add(i)
  for name in n.input:
   if name and name not in needed:needed.add(name);pending.append(name)
 indexed=[(i,n) for i,n in enumerate(model.graph.node) if i in live and i not in root_indices];nodes=[n for i,n in indexed];position={v:i for i,n in enumerate(nodes) for v in n.output};last_use={v:i for i,n in enumerate(nodes) for v in n.input if v}
 for value in original_outputs:last_use[value]=len(nodes)
 def size(value):
  t=types[value];dims=engine.fixed_extents(t)
  if dims is None:raise ValueError('unproved cut tensor: '+value)
  return math.prod(dims)*np.dtype(H.tensor_dtype_to_np_dtype(t.tensor_type.elem_type)).itemsize
 segments=[];start=0;total=0
 for i,n in enumerate(nodes):
  amount=sum(size(v) for v in n.output)
  if total+amount>budget and i>start:segments.append((start,i,total));start=i;total=0
  total+=amount
 if nodes:segments.append((start,len(nodes),total))
 if segments_override is not None:
  selected_segments=[];expected_start=0
  for first,last in segments_override:
   if first!=expected_start or not first<last<=len(nodes):raise ValueError('explicit segment coverage/order differs')
   amount=sum(size(v) for n in nodes[first:last] for v in n.output)
   if amount>budget:raise ValueError('explicit segment exceeds declared extent guard')
   selected_segments.append((first,last,amount));expected_start=last
  if expected_start!=len(nodes):raise ValueError('explicit segments omit executable nodes')
  segments=selected_segments
 wire={v:('cut_'+str(i)) for i,v in enumerate(types) if v not in original_inputs|original_outputs}
 for value in original_inputs|original_outputs:
  if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',value):raise ValueError('public port name must be preserved safely')
  wire[value]=value
 if len(set(wire.values()))!=len(wire):raise ValueError('stable cut port name collides')
 parts=[]
 for number,(start,end,total) in enumerate(segments):
  selected=nodes[start:end];produced={v for n in selected for v in n.output};used={v for n in selected for v in n.input if v};incoming=sorted(used-produced-set(initializers)-set(constants),key=lambda v:(position.get(v,-1),v));outgoing=[v for n in selected for v in n.output if last_use.get(v,-1)>=end]
  if not outgoing:raise ValueError('live partition has no outgoing dependencies')
  # Constants are immutable roots and can be repeated in consuming parts.
  root_needed=set();pending_roots=[v for n in selected for v in n.input if v in constants]
  while pending_roots:
   value=pending_roots.pop()
   if value in shared:continue
   i,n=producer[value]
   if i in root_needed:continue
   root_needed.add(i);pending_roots.extend(v for v in n.input if v in constants)
  roots=[copy.deepcopy(model.graph.node[i]) for i in sorted(root_needed)];used.update(v for n in roots for v in n.input if v)
  incoming=sorted(set(incoming)|(used&shared),key=lambda v:(position.get(v,-1),v))
  aliases={v:wire[v] for v in incoming+outgoing};newnodes=[]
  for source in roots+selected:
   n=copy.deepcopy(source)
   for field in ('input','output'):
    values=getattr(n,field)
    for i,v in enumerate(values):values[i]=aliases.get(v,v)
   newnodes.append(n)
  def info(value):return H.make_value_info(aliases[value],types[value])
  graph=H.make_graph(newnodes,'part_'+str(number),[info(v) for v in incoming],[info(v) for v in outgoing],initializer=[copy.deepcopy(t) for v,t in initializers.items() if v in used and v not in shared]);part=H.make_model(graph,opset_imports=list(model.opset_import));part.ir_version=model.ir_version
  onnx.checker.check_model(part,full_check=True)
  directory=output/f'part{number:03d}';directory.mkdir();logical=directory/'model.logical.onnx';onnx.save(part,str(logical));native=copy.deepcopy(part);allnames={v for n in part.graph.node for v in list(n.input)+list(n.output)}|{v.name for v in part.graph.initializer};prefix=[];suffix=[];input_alias={};output_alias={}
  for kind,values in [('inputs',native.graph.input),('outputs',native.graph.output)]:
   for v in values:
    if engine.fixed_extents(v.type)!=():continue
    inner=v.name+'_logical_scalar_'+kind;control=v.name+'_scalar_shape_'+kind
    if inner in allnames or control in allnames:raise ValueError('scalar view name collision')
    shape=[] if kind=='inputs' else [1];native.graph.initializer.append(N.from_array(np.array(shape,np.int64),name=control));v.type.tensor_type.shape.dim.add().dim_value=1
    if kind=='inputs':input_alias[v.name]=inner;prefix.append(H.make_node('Reshape',[v.name,control],[inner],name=inner))
    else:output_alias[v.name]=inner;suffix.append(H.make_node('Reshape',[inner,control],[v.name],name=inner))
  rewritten=[]
  for source in native.graph.node:
   n=copy.deepcopy(source)
   for field in ('input','output'):
    values=getattr(n,field)
    for i,v in enumerate(values):values[i]=input_alias.get(v,output_alias.get(v,v))
   rewritten.append(n)
  del native.graph.node[:];native.graph.node.extend(prefix+rewritten+suffix);onnx.checker.check_model(native,full_check=True);path=logical if native.SerializeToString()==part.SerializeToString() else directory/'model.native.onnx'
  if path!=logical:onnx.save(native,str(path))
  frontier={v for v,p in position.items() if p<end and last_use.get(v,-1)>=end}
  def row(v):return dict(source_name=v,name=wire[v],shape=list(engine.fixed_extents(types[v])),dtype=str(H.tensor_dtype_to_np_dtype(types[v].tensor_type.elem_type)),bytes=size(v))
  record=dict(index=number,source_node_indices=[indexed[i][0] for i in range(start,end)],source_node_sha256=[hashlib.sha256(nodes[i].SerializeToString()).hexdigest() for i in range(start,end)],logical_model=str(logical),logical_model_sha256=sha(logical),native_model=str(path),native_model_sha256=sha(path),inputs=[row(v) for v in incoming],outputs=[row(v) for v in outgoing],estimated_produced_bytes=total,estimated_frontier_bytes=sum(size(v) for v in frontier),drop_after=[v for v in frontier|used if v not in original_outputs|shared and last_use.get(v,-1)<end]);parts.append(record);save(output/'partition_progress.json',dict(completed_parts=len(parts),total_parts=len(segments),last_index=number))
 return dict(schema='qnn-graph-partition-plan-v1',parts=parts,budget_bytes=budget,immutable_root_node_indices=root_indices,kept_source_nodes=len(live),omitted_unreachable_source_nodes=[i for i in range(len(model.graph.node)) if i not in live],original_inputs=[v.name for v in model.graph.input],original_outputs=[v.name for v in model.graph.output],peak_estimated_produced_bytes=max(p['estimated_produced_bytes'] for p in parts),peak_estimated_frontier_bytes=max(p['estimated_frontier_bytes'] for p in parts),scope='Exact source operations/weights and cut dependencies only; estimates are not actual QNN peak memory or neural acceptance.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--budget-mib',type=int,default=2048);p.add_argument('--layouts-only',action='store_true');p.add_argument('--share-immutable-mib',type=int);a=p.parse_args()
 if sha(a.model)!=a.model_sha256:raise ValueError('source model identity differs')
 if not 1<=a.budget_mib<=4096:raise ValueError('partition budget outside prepared range')
 model=onnx.load(str(a.model));types,constants,errors=engine.prove_static_dataflow(model)
 if errors:raise ValueError('source root inference failed')
 if a.layouts_only:
  save(Path.cwd()/'layout_hints.json',dict(status='pass',source_model_sha256=a.model_sha256,layouts=layout_hints(model,types),script_sha256=sha(__file__),engine_sha256=sha(engine.__file__),scope='Source standard spatial axes and exact routing declaration only; converter ABI/execution/tasks separate.'));return
 pool=None
 if a.share_immutable_mib is not None:
  if not 1<=a.share_immutable_mib<=1024:raise ValueError('immutable pool threshold outside prepared range')
  from shared_constants import pool_values,pack_pool
  pool=pack_pool(pool_values(model,types,constants,a.share_immutable_mib*1024**2),Path.cwd()/'immutable_constants.npz',a.share_immutable_mib*1024**2)
 report=make_parts(model,types,a.budget_mib*1024**2,Path.cwd(),[v['source_name'] for v in pool['entries']] if pool else None)
 if pool is not None:report['shared_constants']=pool;report['shared_constants_helper_sha256']=sha(Path(__file__).with_name('shared_constants.py'))
 report.update(status='pass',source_model=str(a.model.absolute()),source_model_sha256=a.model_sha256,script_sha256=sha(__file__),engine_sha256=sha(engine.__file__));save(Path.cwd()/'partition_plan.json',report)
if __name__=='__main__':main()
