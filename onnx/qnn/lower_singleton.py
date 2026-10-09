#!/usr/bin/env python3
"""Remove a proven leading singleton within segmented rank-six arithmetic."""
import argparse,copy,math,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);p.add_argument('--scope',choices=('seg-head','motion-root'),default='seg-head');args=p.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source model identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model);types,constants,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('root inference has errors')
    def in_scope(name):
        return '/seg_head/' in name if args.scope=='seg-head' else name.startswith('/step/motion_head/') and '/' not in name[len('/step/motion_head/'):]
    selected=[n for n in model.graph.node if in_scope(n.name) and any(len(engine.fixed_extents(types[v]) or ())>5 for v in list(n.input)+list(n.output) if v in types)]
    high={v for n in selected for v in list(n.input)+list(n.output) if v in types and len(engine.fixed_extents(types[v]) or ())>5}
    if any(len(engine.fixed_extents(types[v]))!=6 or engine.fixed_extents(types[v])[0]!=1 for v in high):raise ValueError('segmentation tensor is not rank six with singleton batch')
    if high.intersection(v.name for v in model.graph.input) or high.intersection(v.name for v in model.graph.output):raise ValueError('rank-six boundary must remain unchanged')
    produced={v:n for n in model.graph.node for v in n.output};aliases={v:v+'/qnn_rank5' for v in high};existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}
    if existing.intersection(aliases.values()):raise ValueError('rank-five alias collides')
    if any(v not in produced or produced[v] not in selected for v in high):raise ValueError('rank-six ingress is unsupported')
    weights=[];replacements={};checks=[];options=ort.SessionOptions();options.intra_op_num_threads=1
    def shape_weight(name,shape):
        if name in existing:raise ValueError('new shape name collides')
        existing.add(name);weights.append(N.from_array(np.array(shape,np.int64),name=name));return name
    for node in selected:
        if node.op_type not in ('Reshape','Unsqueeze','Slice','Gather','Add','Sub','Mul','Div','Constant','Concat'):raise ValueError('unsupported singleton operation '+node.op_type)
        new=copy.deepcopy(node);before_inputs=list(node.input);new_inputs=[aliases.get(v,v) for v in node.input]
        del new.input[:];new.input.extend(new_inputs)
        del new.output[:];new.output.extend(aliases.get(v,v) for v in node.output)
        chain=[];prefix=node.name+'/qnn_rank5'
        if node.op_type in ('Reshape','Unsqueeze'):
            target=engine.fixed_extents(types[node.output[0]]);target=target[1:] if node.output[0] in high else target
            new=H.make_node('Reshape',[new_inputs[0],shape_weight(prefix+'_shape',target)],list(new.output),name=prefix)
        elif node.op_type=='Constant':
            if node.output[0] not in constants:raise ValueError('rank-six Constant has no root value')
            value=constants[node.output[0]].reshape(engine.fixed_extents(types[node.output[0]])[1:]);new=H.make_node('Constant',[],list(new.output),name=prefix,value=N.from_array(value))
        elif node.op_type=='Slice':
            if node.input[0] not in high or node.output[0] not in high:raise ValueError('slice rank boundary is unsupported')
            old=np.array(constants[node.input[3]],copy=True);axes=np.where(old<0,old+6,old)
            if (axes==0).any():raise ValueError('slice modifies leading batch axis')
            new.input[3]=shape_weight(prefix+'_axes',(axes-1).tolist())
        elif node.op_type=='Concat':
            axis=next(a for a in new.attribute if a.name=='axis');old=axis.i%6
            if old==0 and len(node.input)>1:raise ValueError('Concat modifies leading batch axis')
            if any(v not in high for v in node.input) or any(v not in high for v in node.output):raise ValueError('Concat singleton ranks differ')
            if old==0:
                new=H.make_node('Identity',new_inputs,list(new.output),name=prefix)
            else:axis.i=old-1
        elif node.op_type=='Gather':
            axis=next(a for a in new.attribute if a.name=='axis');old=axis.i%6
            if old==0:raise ValueError('Gather modifies leading batch axis')
            axis.i=old-1
            if node.output[0] not in high:
                temp=prefix+'_reduced';new.output[0]=temp;chain.append(H.make_node('Reshape',[temp,shape_weight(prefix+'_restore',engine.fixed_extents(types[node.output[0]]))],list(node.output),name=prefix+'/restore'))
        else:
            if any(v not in high for v in node.output):raise ValueError('pointwise rank boundary is unsupported')
        chain.insert(0,new);replacements[node.name]=chain
        # Each changed arithmetic operation executes in actual ORT at its real
        # extents. Aliasing only changes singleton views and preserves math.
        runtime=[v for v in node.input if v and v not in constants];init=[N.from_array(constants[v],name=v) for v in node.input if v in constants]
        unique=list(dict.fromkeys(runtime));in_info=[H.make_value_info(v,types[v]) for v in unique];out_info=[H.make_value_info(v,types[v]) for v in node.output]
        ref=engine.checked_model([copy.deepcopy(node)],in_info,out_info,init)
        cand_in=[];feed={};cand_feed={};rng=np.random.default_rng(17)
        for name in unique:
            shape=engine.fixed_extents(types[name]);dtype=H.tensor_dtype_to_np_dtype(types[name].tensor_type.elem_type)
            if np.issubdtype(dtype,np.floating):value=rng.uniform(.25,1.25,size=shape).astype(dtype)
            else:raise ValueError('singleton arithmetic runtime must be floating')
            feed[name]=value;target=aliases.get(name,name);newshape=shape[1:] if name in high else shape
            cand_feed[target]=value.reshape(newshape);cand_in.append(H.make_tensor_value_info(target,types[name].tensor_type.elem_type,newshape))
        candidate_initializers=[N.from_array(constants[name].reshape(engine.fixed_extents(types[name])[1:]) if name in high else constants[name],name=aliases.get(name,name)) for name in node.input if name in constants]
        names={v.name for v in candidate_initializers}
        for value in weights:
            if value.name not in names and any(value.name in x.input for x in chain):candidate_initializers.append(copy.deepcopy(value));names.add(value.name)
        cand_outputs=[H.make_tensor_value_info(aliases.get(v,v),types[v].tensor_type.elem_type,engine.fixed_extents(types[v])[1:] if v in high else engine.fixed_extents(types[v])) for v in node.output]
        candidate=engine.checked_model(chain,cand_in,cand_outputs,candidate_initializers)
        a=ort.InferenceSession(ref.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feed)
        b=ort.InferenceSession(candidate.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,cand_feed)
        exact=all(x.dtype==y.dtype and np.array_equal(x.reshape(-1),y.reshape(-1),equal_nan=True) for x,y in zip(a,b))
        # Floating differences are recorded; task metrics decide neural acceptance.
        if not exact and any(not np.issubdtype(x.dtype,np.floating) for x in a):raise ValueError('singleton integer operation local check differs '+node.name)
        checks.append(dict(node=node.name,op_type=node.op_type,source_shapes={v:list(engine.fixed_extents(types[v])) for v in list(node.input)+list(node.output) if v in types},actual_ort_local_exact=exact))
    nodes=[]
    for node in model.graph.node:nodes.extend(replacements.get(node.name,[copy.deepcopy(node)]))
    del model.graph.node[:];model.graph.node.extend(nodes);model.graph.initializer.extend(weights);del model.graph.value_info[:]
    onnx.checker.check_model(model,full_check=True)
    for boundary in ('input','output'):
        if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('boundary ABI changed')
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weight changed')
    out=Path.cwd()/'model.rank5-singleton.onnx';onnx.save(model,str(out));save(Path.cwd()/'singleton.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(out),model_sha256=sha(out),changed_nodes=len(selected),scope_name=args.scope,aliases=aliases,checks=checks,scope='Local singleton and arithmetic controls; neural and QNN acceptance separate.',script_sha256=sha(__file__),engine_sha256=sha(engine.__file__)))
if __name__=='__main__':main()
