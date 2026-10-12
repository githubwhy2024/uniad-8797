#!/usr/bin/env python3
"""Pack internal high-rank tensors and lower their operations to rank at most five."""
import argparse,copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save
from lower_transpose import route

class Lowering:
    def __init__(self,types,constants,high):
        self.types,self.constants,self.high=types,constants,high;self.weights=[];self.nodes=[];self.sequence=0
    def dims(self,name):return engine.fixed_extents(self.types[name])
    def physical(self,name):return name+'/qnn_packed' if name in self.high else name
    def reset(self,prefix):self.nodes=[];self.weights=[];self.prefix=prefix+'/qnn_packed';self.sequence=0
    def fresh(self):self.sequence+=1;return self.prefix+'/'+str(self.sequence)
    def weight(self,value,dtype=np.int64):
        name=self.fresh();self.weights.append(N.from_array(np.array(value,dtype),name=name));return name
    def op(self,kind,inputs,**attrs):
        out=self.fresh();self.nodes.append(H.make_node(kind,inputs,[out],name=out,**attrs));return out
    def view(self,value,shape):return self.op('Reshape',[value,self.weight(shape)])
    def finish(self,value,node):
        out=node.output[0];shape=[math.prod(self.dims(out))] if out in self.high else self.dims(out)
        self.nodes.append(H.make_node('Reshape',[value,self.weight(shape)],[self.physical(out)],name=self.fresh()))
    def grouped(self,names,outshape):
        shapes=[(1,)*(len(outshape)-len(self.dims(name)))+self.dims(name) for name in names]
        if any(len(s)!=len(outshape) or any(a not in (1,b) for a,b in zip(s,outshape)) for s in shapes):raise ValueError('broadcast extent is inconsistent')
        groups=[];last=None
        for i,size in enumerate(outshape):
            key=tuple(s[i]==1 and size!=1 for s in shapes)
            if not groups or (size!=1 and last is not None and key!=last):groups.append([])
            groups[-1].append(i)
            if size!=1:last=key
        if len(groups)>5:raise ValueError('broadcast needs more than five distinct groups')
        inputs=[self.view(self.physical(name),[math.prod(shape[i] for i in group) for group in groups]) for name,shape in zip(names,shapes)]
        return inputs,[math.prod(outshape[i] for i in group) for group in groups]
    def lower(self,node):
        self.reset(node.name);kind=node.op_type
        if len(node.output)!=1:raise ValueError('high-rank multiple output unsupported')
        if kind in ('Reshape','Unsqueeze','Squeeze'):
            self.finish(self.physical(node.input[0]),node)
        elif kind=='Constant':
            value=N.to_array(next(a.t for a in node.attribute if a.name=='value'))
            self.nodes.append(H.make_node('Constant',[],[self.physical(node.output[0])],name=self.fresh(),value=N.from_array(value.reshape(-1))))
        elif kind in ('Add','Sub','Mul','Div'):
            inputs,_=self.grouped(list(node.input),self.dims(node.output[0]));self.finish(self.op(kind,inputs),node)
        elif kind=='Expand':
            inputs,shape=self.grouped([node.input[0]],self.dims(node.output[0]));self.finish(self.op(kind,[inputs[0],self.weight(shape)]),node)
        elif kind=='Concat':
            axis=next(a.i for a in node.attribute if a.name=='axis')%len(self.dims(node.input[0]));inputs=[]
            for name in node.input:
                shape=self.dims(name);inputs.append(self.view(self.physical(name),[math.prod(shape[:axis]),shape[axis],math.prod(shape[axis+1:])]))
            self.finish(self.op(kind,inputs,axis=1),node)
        elif kind=='Gather':
            shape=self.dims(node.input[0]);axis=next((a.i for a in node.attribute if a.name=='axis'),0)%len(shape)
            data=self.view(self.physical(node.input[0]),[math.prod(shape[:axis]),shape[axis],math.prod(shape[axis+1:])]);index=self.view(self.physical(node.input[1]),[math.prod(self.dims(node.input[1]))])
            self.finish(self.op(kind,[data,index],axis=1),node)
        elif kind=='Transpose':
            shape=self.dims(node.input[0]);perm=list(next(a.ints for a in node.attribute if a.name=='perm'))
            out=node.output[0];target=[math.prod(self.dims(out))] if out in self.high else self.dims(out)
            nodes,weights,_=route(shape,perm,self.physical(node.input[0]),self.prefix,self.physical(out),target)
            self.nodes.extend(nodes);self.weights.extend(weights)
        elif kind=='ScatterND':
            if any(a.name=='reduction' and a.s not in (b'none',b'') for a in node.attribute):raise ValueError('ScatterND reduction unsupported')
            shape=self.dims(node.input[0]);indexshape=self.dims(node.input[1]);k=indexshape[-1];rows=math.prod(indexshape[:-1]);tail=math.prod(shape[k:])
            if k<=0 or k>len(shape) or math.prod(shape)>2**31-1:raise ValueError('ScatterND linear addressing exceeds exact signed32 range')
            data=self.view(self.physical(node.input[0]),[math.prod(shape[:k]),tail]);index=self.view(self.physical(node.input[1]),[rows,k]);linear=None
            for axis in range(k):
                coord=self.op('Gather',[index,self.weight([axis])],axis=1)
                negative=self.op('Less',[coord,self.weight([0])]);shifted=self.op('Add',[coord,self.weight([shape[axis]])]);coord=self.op('Where',[negative,shifted,coord])
                part=self.op('Mul',[coord,self.weight([math.prod(shape[axis+1:k])])])
                linear=part if linear is None else self.op('Add',[linear,part])
            updates=self.view(self.physical(node.input[2]),[rows,tail]);self.finish(self.op(kind,[data,linear,updates]),node)
        else:raise ValueError('unsupported high-rank operation '+kind+' '+node.name)
        return self.nodes,self.weights

def witness(node,chain,weights,lowering,run,number):
    types,constants=lowering.types,lowering.constants;dims=lowering.dims
    runtime=list(dict.fromkeys(name for name in node.input if name and name not in constants));initializers=[N.from_array(constants[name],name=name) for name in dict.fromkeys(node.input) if name in constants]
    ref=engine.checked_model([copy.deepcopy(node)],[H.make_value_info(name,types[name]) for name in runtime],[H.make_value_info(name,types[name]) for name in node.output],initializers)
    candidate_init=[N.from_array(constants[name].reshape(-1) if name in lowering.high else constants[name],name=lowering.physical(name)) for name in dict.fromkeys(node.input) if name in constants]+weights
    candidate_inputs=[H.make_tensor_value_info(lowering.physical(name),types[name].tensor_type.elem_type,[math.prod(dims(name))] if name in lowering.high else dims(name)) for name in runtime]
    candidate_outputs=[H.make_tensor_value_info(lowering.physical(name),types[name].tensor_type.elem_type,[math.prod(dims(name))] if name in lowering.high else dims(name)) for name in node.output]
    candidate=engine.checked_model(chain,candidate_inputs,candidate_outputs,candidate_init)
    rng=np.random.default_rng(123);feed={}
    for name in runtime:
        shape=dims(name);dtype=H.tensor_dtype_to_np_dtype(types[name].tensor_type.elem_type)
        if node.op_type=='ScatterND' and name==node.input[1]:
            value=np.empty(shape,dtype=dtype)
            for axis in range(shape[-1]):value[...,axis]=rng.integers(0,dims(node.input[0])[axis],size=shape[:-1])
        elif node.op_type=='Gather' and name==node.input[1]:
            axis=next((a.i for a in node.attribute if a.name=='axis'),0)%len(dims(node.input[0]));value=rng.integers(0,dims(node.input[0])[axis],size=shape,dtype=dtype)
        elif np.issubdtype(dtype,np.floating):
            value=np.empty(shape,dtype=dtype);flat=value.reshape(-1)
            for start in range(0,flat.size,1048576):
                flat[start:start+1048576]=rng.uniform(.25,1.25,size=min(1048576,flat.size-start))
        elif dtype==np.bool_:value=rng.integers(0,2,size=shape).astype(dtype)
        else:value=rng.integers(0,10,size=shape,dtype=dtype)
        feed[name]=value
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.log_severity_level=3
    a=ort.InferenceSession(ref.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,feed)
    b=ort.InferenceSession(candidate.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{lowering.physical(name):value.reshape(-1) if name in lowering.high else value for name,value in feed.items()})
    stats=[]
    for x,y in zip(a,b):
        if x.dtype!=y.dtype or x.size!=y.size:raise ValueError('packed dtype/element count differs')
        exact=np.array_equal(x.reshape(-1),y.reshape(-1),equal_nan=True)
        if not np.issubdtype(x.dtype,np.floating) and not exact:raise ValueError('packed integer or bool output differs')
        maximum=0.0
        xf,yf=x.reshape(-1),y.reshape(-1)
        for start in range(0,xf.size,1048576):
            maximum=max(maximum,float(np.max(np.abs(xf[start:start+1048576].astype(np.float64)-yf[start:start+1048576].astype(np.float64)))))
        stats.append(dict(dtype=str(x.dtype),elements=x.size,exact=exact,max_abs=maximum))
    left=run/f'control_{number}.before.onnx';right=run/f'control_{number}.after.onnx';onnx.save(ref,str(left));onnx.save(candidate,str(right))
    return dict(id=number,op_type=node.op_type,source_shapes={name:list(dims(name)) for name in list(node.input)+list(node.output) if name in types},stats=stats,before_sha256=sha(left),after_sha256=sha(right))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);args=p.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source model identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model);types,constants,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('source root inference failed')
    high={name for name,t in types.items() if engine.fixed_extents(t) is not None and len(engine.fixed_extents(t))>5}
    if high.intersection(v.name for v in list(model.graph.input)+list(model.graph.output)):raise ValueError('external high-rank ABI requires separate mapping')
    for value in list(original.graph.initializer):
        if value.name in high:model.graph.initializer.append(N.from_array(N.to_array(value).reshape(-1),name=value.name+'/qnn_packed'))
    lowering=Lowering(types,constants,high);selected=[];changes=[];controls={};existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}
    for n in model.graph.node:
        if not high.intersection(list(n.input)+list(n.output)) and not (n.op_type=='Transpose' and len(lowering.dims(n.input[0]))==5):selected.append(copy.deepcopy(n));continue
        chain,weights=lowering.lower(n)
        new={v for x in chain for v in x.output}|{v.name for v in weights}
        if (new-{lowering.physical(v) for v in n.output})&existing:raise ValueError('packed node name collides')
        existing.update(new);selected.extend(chain);model.graph.initializer.extend(weights)
        key=json.dumps([n.op_type,[(types[v].tensor_type.elem_type,lowering.dims(v)) for v in n.input if v in types],[(i,constants[v].tolist()) for i,v in enumerate(n.input) if v in constants],[(a.name,H.get_attribute_value(a).decode() if isinstance(H.get_attribute_value(a),bytes) else H.get_attribute_value(a)) for a in n.attribute if a.type!=onnx.AttributeProto.TENSOR],[hashlib.sha256(a.SerializeToString()).hexdigest() for a in n.attribute if a.type==onnx.AttributeProto.TENSOR]],default=str)
        if key not in controls:
            number=len(controls);controls[key]=witness(n,chain,weights,lowering,Path.cwd(),number);save(Path.cwd()/'control_progress.json',dict(completed_unique_controls=len(controls),last_node=n.name))
        changes.append(dict(node=n.name,op_type=n.op_type,control_id=controls[key]['id']))
    del model.graph.node[:];model.graph.node.extend(selected);del model.graph.value_info[:]
    onnx.checker.check_model(model,full_check=True)
    for boundary in ('input','output'):
        if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('ordered ABI changed')
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source initializer changed')
    out=Path.cwd()/'model.packed-rank.onnx';onnx.save(model,str(out));save(Path.cwd()/'packing.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(out),model_sha256=sha(out),changes=changes,controls=list(controls.values()),high_rank_values=len(high),scope='Internal views, broadcast grouping, shuffles and exact signed32 scatter addressing; neural task acceptance separate.',script_sha256=sha(__file__),engine_sha256=sha(engine.__file__)))
if __name__=='__main__':main()
