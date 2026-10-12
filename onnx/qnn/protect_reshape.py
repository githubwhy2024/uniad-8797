#!/usr/bin/env python3
"""Give the installed SDK pattern validator a rank-two view before rank-five reshape."""
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
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-sha256',required=True);args=p.parse_args()
    if sha(args.model)!=args.model_sha256:raise ValueError('source identity differs')
    model=onnx.load(str(args.model));original=copy.deepcopy(model);types,_,errors=engine.prove_static_dataflow(model)
    if errors:raise ValueError('root inference errors')
    existing={v for n in model.graph.node for v in list(n.input)+list(n.output)};nodes=[];changes=[];controls={};options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
    for node in model.graph.node:
        new=copy.deepcopy(node)
        if node.op_type=='Reshape' and len(engine.fixed_extents(types[node.input[0]]))<2 and len(engine.fixed_extents(types[node.output[0]]))==5:
            shape=engine.fixed_extents(types[node.input[0]]);output=engine.fixed_extents(types[node.output[0]]);count=math.prod(shape);prefix=node.name+'/sdk_rank_view';name=prefix+'_shape'
            if {prefix,name}&existing:raise ValueError('rank view name collides')
            existing.update((prefix,name));model.graph.initializer.append(N.from_array(np.array([1,count],np.int64),name=name));nodes.append(H.make_node('Reshape',[node.input[0],name],[prefix],name=prefix));new.input[0]=prefix
            key=(shape,output)
            if key not in controls:
                # The extra view performs no arithmetic; unique integer indices
                # verify all elements at the actual size in the original and new paths.
                if count>2**31-1:raise ValueError('index witness too large')
                initials=[N.from_array(np.array(output,np.int64),name='target'),N.from_array(np.array([1,count],np.int64),name='view')]
                a=engine.checked_model([H.make_node('Reshape',['x','target'],['y'])],[H.make_tensor_value_info('x',onnx.TensorProto.INT32,shape)],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,output)],initials)
                b=engine.checked_model([H.make_node('Reshape',['x','view'],['v']),H.make_node('Reshape',['v','target'],['y'])],[H.make_tensor_value_info('x',onnx.TensorProto.INT32,shape)],[H.make_tensor_value_info('y',onnx.TensorProto.INT32,output)],initials)
                feed={'x':np.arange(count,dtype=np.int32).reshape(shape)};x=ort.InferenceSession(a.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feed)[0];y=ort.InferenceSession(b.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feed)[0]
                if not np.array_equal(x,y):raise ValueError('rank view changed element order')
                controls[key]=dict(source_shape=list(shape),output_shape=list(output),elements=count,actual_ort_every_index_exact=True)
            changes.append(dict(node=node.name,source_shape=list(shape),inserted_shape=[1,count],output_shape=list(output)))
        nodes.append(new)
    if not changes:raise ValueError('no observed SDK rank-one-to-five reshape matched')
    del model.graph.node[:];model.graph.node.extend(nodes);del model.graph.value_info[:];onnx.checker.check_model(model,full_check=True)
    for boundary in ('input','output'):
        if [v.SerializeToString() for v in getattr(original.graph,boundary)]!=[v.SerializeToString() for v in getattr(model.graph,boundary)]:raise ValueError('ordered boundary changed')
    if any(v.SerializeToString()!=model.graph.initializer[i].SerializeToString() for i,v in enumerate(original.graph.initializer)):raise ValueError('source weights changed')
    out=Path.cwd()/'model.protected-view.onnx';onnx.save(model,str(out));save(Path.cwd()/'view_protection.json',dict(status='pass',source_model_sha256=args.model_sha256,model=str(out),model_sha256=sha(out),changes=changes,controls=list(controls.values()),scope='No arithmetic: rank-two intermediate views only; actual QNN and task acceptance separate.',script_sha256=sha(__file__),engine_sha256=sha(engine.__file__)))
if __name__=='__main__':main()
