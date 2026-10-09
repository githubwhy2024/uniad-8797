#!/usr/bin/env python3
"""Independently verify operation coverage, cut provenance and immutable roots."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import onnx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import sha,save
import partition_model
import shape_dataflow as engine

def audit(plan):
 source=onnx.load(plan['source_model'])
 if sha(plan['source_model'])!=plan['source_model_sha256']:raise ValueError('partition source model differs')
 types,constants,errors=engine.prove_static_dataflow(source)
 if errors:raise ValueError('fresh parent roots fail')
 roots={t.name:t for t in source.graph.initializer};constant_nodes={};immutable=set(roots);immutable_indices=[]
 for i,n in enumerate(source.graph.node):
  if n.op_type=='Constant' or (n.input and all(v in immutable for v in n.input if v) and n.op_type not in ('RandomNormal','RandomUniform','RandomNormalLike','RandomUniformLike','Multinomial','Dropout')):
   immutable.update(n.output);constant_nodes[n.name]=n;immutable_indices.append(i)
 if plan.get('immutable_root_node_indices',[])!=immutable_indices:raise ValueError('immutable root dependency classification differs')
 shared=set()
 if 'shared_constants' in plan:
  from shared_constants import verify_pool
  if sha(Path(partition_model.__file__).with_name('shared_constants.py'))!=plan['shared_constants_helper_sha256']:raise ValueError('immutable pool helper differs')
  shared=set(verify_pool(source,types,constants,plan['shared_constants']))
  if not shared<=immutable:raise ValueError('shared operand is not an immutable root')
 producer={v:i for i,n in enumerate(source.graph.node) for v in n.output};available={v.name for v in source.graph.input}|shared;seen=[];checks=[]
 live=set();pending=[v.name for v in source.graph.output]
 while pending:
  v=pending.pop()
  if v not in producer or producer[v] in live:continue
  i=producer[v];live.add(i);pending.extend(source.graph.node[i].input)
 for record in plan['parts']:
  path=Path(record['logical_model'])
  if sha(path)!=record['logical_model_sha256'] or sha(record['native_model'])!=record['native_model_sha256']:raise ValueError('part model identity differs')
  model=onnx.load(str(path));onnx.checker.check_model(model,full_check=True);cuts=record['inputs']+record['outputs'];rename={v['name']:v['source_name'] for v in cuts}
  for kind in ('inputs','outputs'):
   values=list(getattr(model.graph,'input' if kind=='inputs' else 'output'))
   if [v.name for v in values]!=[r['name'] for r in record[kind]]:raise ValueError('ordered part cut ABI differs')
   for v,row in zip(values,record[kind]):
    expected=types[row['source_name']]
    if v.type.SerializeToString()!=expected.SerializeToString():raise ValueError('cut is not anchored to parent root dtype/extents')
  if not {r['source_name'] for r in record['inputs']}<=available:raise ValueError('part consumes an unproduced frontier')
  for weight in model.graph.initializer:
   if weight.name not in roots or weight.SerializeToString()!=roots[weight.name].SerializeToString():raise ValueError('part initializer differs from parent')
  computational=[]
  for n in model.graph.node:
   restored=copy.deepcopy(n)
   for field in ('input','output'):
    values=getattr(restored,field)
    for i,v in enumerate(values):values[i]=rename.get(v,v)
   if n.name in constant_nodes:
    if restored.name not in constant_nodes or restored.SerializeToString()!=constant_nodes[restored.name].SerializeToString():raise ValueError('part Constant differs from immutable parent root')
   else:computational.append(restored)
  if len(computational)!=len(record['source_node_indices']) or len(record['source_node_sha256'])!=len(record['source_node_indices']):raise ValueError('operation coverage count differs')
  for node,i,digest in zip(computational,record['source_node_indices'],record['source_node_sha256']):
   original=source.graph.node[i]
   if original.SerializeToString()!=node.SerializeToString() or hashlib.sha256(original.SerializeToString()).hexdigest()!=digest:raise ValueError('cut changes a parent operation or order')
  native=onnx.load(record['native_model']);nt,_,native_errors=engine.prove_static_dataflow(native)
  if native_errors:raise ValueError('native part root inference failed')
  logical_names={n.name:n for n in model.graph.node};views={};view_weights=set();native_body=[]
  native_inputs={v.name for v in native.graph.input};native_outputs={v.name for v in native.graph.output};native_weights={t.name:t for t in native.graph.initializer}
  for n in native.graph.node:
   if n.name in logical_names:native_body.append(n);continue
   if n.op_type!='Reshape' or len(n.input)!=2 or len(n.output)!=1 or n.input[1] not in native_weights:raise ValueError('unexpected native part operation')
   target=onnx.numpy_helper.to_array(native_weights[n.input[1]]).tolist()
   if n.input[0] in native_inputs and engine.fixed_extents(nt[n.input[0]])==(1,) and engine.fixed_extents(nt[n.output[0]])==() and target==[]:views[n.output[0]]=n.input[0]
   elif n.output[0] in native_outputs and engine.fixed_extents(nt[n.input[0]])==() and engine.fixed_extents(nt[n.output[0]])==(1,) and target==[1]:views[n.input[0]]=n.output[0]
   else:raise ValueError('native extra operation is not an exact scalar wire view')
   view_weights.add(n.input[1])
  if len(native_body)!=len(model.graph.node):raise ValueError('native operation coverage differs')
  for n,logical in zip(native_body,model.graph.node):
   restored=copy.deepcopy(n)
   for field in ('input','output'):
    values=getattr(restored,field)
    for i,v in enumerate(values):values[i]=views.get(v,v)
   if restored.SerializeToString()!=logical.SerializeToString():raise ValueError('native body changes logical part operations')
  for t in model.graph.initializer:
   if t.name not in native_weights or native_weights[t.name].SerializeToString()!=t.SerializeToString():raise ValueError('native weights change logical roots')
  if set(native_weights)-{t.name for t in model.graph.initializer}!=view_weights:raise ValueError('unexpected native root injection')
  seen.extend(record['source_node_indices']);available.update(r['source_name'] for r in record['outputs']);checks.append(dict(index=record['index'],node_count=len(computational),inputs_from_parent_roots_or_prior_actual_outputs=True,operations_and_initializer_bytes_unchanged=True))
 expected=[i for i in range(len(source.graph.node)) if i in live and i not in immutable_indices]
 if seen!=expected or not {v.name for v in source.graph.output}<=available:raise ValueError('coverage/order/public output closure fails')
 omitted=[i for i in range(len(source.graph.node)) if i not in live]
 if omitted!=plan['omitted_unreachable_source_nodes']:raise ValueError('unreachable-node declaration differs')
 return dict(status='pass',source_model_sha256=plan['source_model_sha256'],parts=len(checks),checks=checks,covered_computational_nodes=len(seen),omitted_unreachable_nodes=len(omitted),scope='Independent parent-root cut provenance and exact operation/weight coverage only; scalar transport/QNN execution/task metrics separate.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);a=p.parse_args()
 if sha(a.plan)!=a.plan_sha256:raise ValueError('plan identity differs')
 result=audit(json.loads(a.plan.read_text()));result.update(plan_sha256=a.plan_sha256,script_sha256=sha(__file__));save(Path.cwd()/'partition_audit.json',result)
if __name__=='__main__':main()
