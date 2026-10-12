#!/usr/bin/env python3
"""Lower the observed DCN rank-six shuffle to bit-preserving rank-four shuffles."""
import argparse,copy,json,math,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def route(shape,perm,source,prefix,target,final_shape):
    order=list(range(len(shape)));nodes=[];initializers=[];steps=[];last=source
    for destination,axis in enumerate(perm):
        position=order.index(axis)
        if position==destination:continue
        dims=[math.prod(shape[v] for v in order[:destination]),math.prod(shape[v] for v in order[destination:position]),shape[axis],math.prod(shape[v] for v in order[position+1:])]
        name=prefix+'/shuffle_'+str(len(steps));control=name+'_shape';initializers.append(N.from_array(np.array(dims,np.int64),name=control))
        nodes.append(H.make_node('Reshape',[last,control],[name+'_view'],name=name+'/view'))
        nodes.append(H.make_node('Transpose',[name+'_view'],[name+'_moved'],name=name+'/move',perm=[0,2,1,3]))
        old=order[:];order.insert(destination,order.pop(position));steps.append(dict(axis=axis,destination=destination,grouped_shape=dims,before_order=old,after_order=order[:]))
        last=name+'_moved'
    if order!=perm:raise ValueError('shuffle axes do not match permutation')
    final=prefix+'/final_shape';initializers.append(N.from_array(np.array(final_shape,np.int64),name=final))
    nodes.append(H.make_node('Reshape',[last,final],[target],name=prefix+'/final_view'))
    return nodes,initializers,steps

def verify(shape,perm,final_shape,run,key):
    # Unique int32 indices exercise every element at the actual extents; only
    # shape/permutation operations are compared, no neural floating arithmetic.
    count=math.prod(shape)
    if count>np.iinfo(np.int32).max:raise ValueError('index witness exceeds int32')
    nodes,weights,steps=route(shape,perm,'x','route','y',final_shape)
    candidate=engine.checked_model(nodes,[H.make_tensor_value_info('x',onnx.TensorProto.INT32,[count])],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,final_shape)],weights)
    original=engine.checked_model([H.make_node('Reshape',['x','start'],['v']),H.make_node('Transpose',['v'],['t'],perm=perm),H.make_node('Reshape',['t','end'],['y'])],[H.make_tensor_value_info('x',onnx.TensorProto.INT32,[count])],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,final_shape)],[N.from_array(np.array(shape,np.int64),name='start'),N.from_array(np.array(final_shape,np.int64),name='end')])
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    x=np.arange(count,dtype=np.int32)
    a=ort.InferenceSession(original.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
    b=ort.InferenceSession(candidate.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
    exact=a.shape==b.shape and np.array_equal(a,b)
    if not exact:raise ValueError('actual index permutation differs')
    left=run/(key+'.before.onnx');right=run/(key+'.after.onnx');onnx.save(original,str(left));onnx.save(candidate,str(right))
    return dict(shape=list(shape),permutation=perm,final_shape=list(final_shape),elements=count,actual_ort_every_index_exact=exact,index_dtype='int32',output_sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest(),before_model_sha256=sha(left),after_model_sha256=sha(right),steps=steps)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);args=p.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model);types,_,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('root inference errors')
    producers={v:n for n in model.graph.node for v in n.output};consumers={}
    for n in model.graph.node:
        for v in n.input:consumers.setdefault(v,[]).append(n)
    outputs={v.name for v in model.graph.output};removed=set();replacements={};changes=[];witnesses={};existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}
    for n in model.graph.node:
        if n.op_type!='Transpose' or '/deform_conv/' not in n.name:continue
        shape=engine.fixed_extents(types[n.input[0]]);perm=list(next(a.ints for a in n.attribute if a.name=='perm'))
        if len(shape)!=6:continue
        before=producers[n.input[0]];after_list=consumers[n.output[0]]
        if before.op_type!='Reshape' or len(consumers[n.input[0]])!=1 or len(after_list)!=1 or after_list[0].op_type!='Reshape' or n.input[0] in outputs or n.output[0] in outputs:raise ValueError('rank-six shuffle has unsupported consumer structure')
        after=after_list[0];final_shape=engine.fixed_extents(types[after.output[0]])
        if len(final_shape)>4:raise ValueError('DCN final reshape rank exceeds four')
        prefix=n.name+'/qnn';new_nodes,new_weights,steps=route(shape,perm,before.input[0],prefix,after.output[0],final_shape)
        new_names={v for x in new_nodes for v in x.output}|{v.name for v in new_weights}
        if (new_names-{after.output[0]})&existing:raise ValueError('rank-four replacement names collide')
        existing.update(new_names);model.graph.initializer.extend(new_weights)
        removed.update((before.name,n.name));replacements[after.name]=new_nodes
        key=json.dumps([shape,perm,final_shape]);witness_id='routing_'+str(len(witnesses))
        if key not in witnesses:witnesses[key]=dict(id=witness_id,**verify(shape,perm,final_shape,Path.cwd(),witness_id))
        changes.append(dict(transpose=n.name,before_reshape=before.name,after_reshape=after.name,witness_id=witnesses[key]['id'],steps=steps))
    if not changes:raise ValueError('no observed DCN shuffle matched')
    selected=[]
    for n in model.graph.node:
        if n.name in removed:continue
        selected.extend(replacements.get(n.name,[copy.deepcopy(n)]))
    del model.graph.node[:];model.graph.node.extend(selected)
    produced={v for n in selected for v in n.output};infos=[copy.deepcopy(v) for v in model.graph.value_info if v.name in produced]
    del model.graph.value_info[:];model.graph.value_info.extend(infos)
    for boundary in ('input','output'):
        if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('ordered ABI changed')
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source initializer bytes changed')
    onnx.checker.check_model(model,full_check=True);path=Path.cwd()/'model.rank4-dcn.onnx';onnx.save(model,str(path))
    save(Path.cwd()/'routing.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(path),model_sha256=sha(path),changes=changes,witnesses=list(witnesses.values()),source_initializer_bytes_unchanged=True,ordered_abi_unchanged=True,scope='Data routing only; neural and QNN task acceptance remain separate.',script_sha256=sha(__file__),engine_sha256=sha(engine.__file__)))
    print(json.dumps(dict(status='pass',changed_shuffles=len(changes),actual_extent_witnesses=len(witnesses),model_sha256=sha(path))))
if __name__=='__main__':main()
