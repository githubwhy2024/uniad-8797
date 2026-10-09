#!/usr/bin/env python3
"""Expose scalar ONNX ports as explicit one-element native QNN ports."""
import argparse,copy,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
import shape_dataflow as engine
from prove_graph_extents import sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);args=p.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model);existing={v for n in model.graph.node for v in list(n.input)+list(n.output)}|{v.name for v in model.graph.initializer};mapping={'schema':'qnn-scalar-port-v1','inputs':[],'outputs':[]};prefix=[];suffix=[];input_alias={};output_alias={};weights=[];checks={}
    for kind,values in [('inputs',model.graph.input),('outputs',model.graph.output)]:
        for v in values:
            shape=engine.fixed_extents(v.type)
            if shape is None:raise ValueError('dynamic external port')
            row=dict(name=v.name,dtype=str(H.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type)),logical_shape=list(shape),native_shape=[1] if shape==() else list(shape))
            mapping[kind].append(row)
            if shape!=():continue
            alias=v.name+'/qnn_scalar_internal';control=v.name+'/qnn_scalar_'+kind+'_shape'
            if {alias,control}&existing:raise ValueError('scalar port alias collides')
            if kind=='inputs' and v.name in {t.name for t in model.graph.initializer}:raise ValueError('scalar initializer input unsupported')
            if kind=='inputs':
                input_alias[v.name]=alias;weights.append(N.from_array(np.array([],np.int64),name=control));prefix.append(H.make_node('Reshape',[v.name,control],[alias],name=alias+'/input_view'))
            else:
                output_alias[v.name]=alias;weights.append(N.from_array(np.array([1],np.int64),name=control));suffix.append(H.make_node('Reshape',[alias,control],[v.name],name=alias+'/output_view'))
            v.type.tensor_type.shape.dim.add().dim_value=1
    if set(input_alias)&set(output_alias):raise ValueError('scalar pass-through boundary needs explicit Identity')
    nodes=[]
    for n in model.graph.node:
        copied=copy.deepcopy(n)
        for i,name in enumerate(copied.input):copied.input[i]=input_alias.get(name,output_alias.get(name,name))
        for i,name in enumerate(copied.output):copied.output[i]=output_alias.get(name,name)
        nodes.append(copied)
    del model.graph.node[:];model.graph.node.extend(prefix+nodes+suffix);model.graph.initializer.extend(weights);del model.graph.value_info[:]
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for dtype in (np.float32,np.int64,np.bool_):
        tensor_type=H.np_dtype_to_tensor_dtype(np.dtype(dtype));constant=[N.from_array(np.array([],np.int64),name='scalar'),N.from_array(np.array([1],np.int64),name='vector')]
        toy=engine.checked_model([H.make_node('Reshape',['x','scalar'],['v']),H.make_node('Reshape',['v','vector'],['y'])],[H.make_tensor_value_info('x',tensor_type,[1])],[H.make_tensor_value_info('y',tensor_type,[1])],constant)
        session=ort.InferenceSession(toy.SerializeToString(),options,providers=['CPUExecutionProvider'])
        values=[False,True] if dtype==np.bool_ else [np.iinfo(np.int64).min,0,np.iinfo(np.int64).max] if dtype==np.int64 else [-3.25,0.,7.5]
        for i,value in enumerate(values):
            x=np.array([value],dtype=dtype);y=session.run(None,{'x':x})[0];checks[str(np.dtype(dtype))+'_'+str(i)]=y.dtype==x.dtype and np.array_equal(x,y)
    if not all(checks.values()):raise ValueError('scalar port view controls failed')
    onnx.checker.check_model(model,full_check=True)
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
    if not prefix and not suffix:raise ValueError('no scalar ports to adapt')
    out=Path.cwd()/'model.native-ports.onnx';onnx.save(model,str(out));save(Path.cwd()/'boundary.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(out),model_sha256=sha(out),mapping=mapping,checks=checks,scalar_inputs=len(prefix),scalar_outputs=len(suffix),scope='One-element scalar wire views only; integer compute ranges and QNN/neural acceptance separate.',script_sha256=sha(__file__)))
if __name__=='__main__':main()
