#!/usr/bin/env python3
"""Root-only extent engine reused from frozen Q3, with Size derived from extents."""
import copy
import math
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N

UPSTREAM_SOURCE = "/home/wangheyao/projects/uniad_onnx_qualcomm_q3/onnx/validation/prove_fixed_shape_dataflow.py"
UPSTREAM_SHA256 = "742a5710776e3915797af4acad8ed79bd0165f44979aaa41616ba2576c557cd6"


CONTROLLERS = {'Reshape': [1], 'Expand': [1], 'ConstantOfShape': [0], 'Tile': [1], 'Resize': [2,3], 'Pad': [1], 'Slice': [1,2,3,4], 'TopK': [1], 'Split': [1], 'Squeeze': [1], 'Unsqueeze': [1], 'ReduceSum': [1], 'Range': [0,1,2]}

def fixed_extents(t):
    if not t.tensor_type.HasField('shape') or any(not d.HasField('dim_value') for d in t.tensor_type.shape.dim):return None
    return tuple(d.dim_value for d in t.tensor_type.shape.dim)


def evaluate_constant_node(node, c, types):
    a={x.name:H.get_attribute_value(x) for x in node.attribute};op=node.op_type
    vals=[c.get(x) if x else None for x in node.input]
    if op=='Constant':return [N.to_array(a['value'])] if 'value' in a else None
    if op=='Shape':
        shape=fixed_extents(types[node.input[0]]) if node.input[0] in types else None
        if shape is not None:return [np.array(shape,dtype=np.int64)[a.get('start',0):a.get('end',len(shape))]]
    if op=='Size':
        shape=fixed_extents(types[node.input[0]]) if node.input[0] in types else None
        if shape is not None:
            count=math.prod(shape)
            if count > np.iinfo(np.int64).max:raise ValueError('Size exceeds int64 extent range')
            return [np.array(count,dtype=np.int64)]
    if any(v is None for v in vals):return None
    if not vals:return None
    if op=='Identity':v=vals[0]
    elif op=='Gather':v=np.take(vals[0],vals[1],axis=a.get('axis',0))
    elif op=='Cast':v=vals[0].astype(H.tensor_dtype_to_np_dtype(a['to']))
    elif op=='Add':v=vals[0]+vals[1]
    elif op=='Sub':v=vals[0]-vals[1]
    elif op=='Mul':v=vals[0]*vals[1]
    elif op=='Div':
        if np.any(vals[1]==0):return None
        v=np.floor_divide(*vals) if vals[0].dtype.kind in 'iu' else vals[0]/vals[1]
        if vals[0].dtype.kind=='i':v=v+(((vals[0]<0)!=(vals[1]<0)) & (np.remainder(*vals)!=0))
    elif op=='Equal':v=np.equal(*vals)
    elif op=='Where':v=np.where(*vals)
    elif op=='Concat':v=np.concatenate(vals,axis=a['axis'])
    elif op=='Unsqueeze':v=np.expand_dims(vals[0],tuple(int(x) for x in (vals[1] if len(vals)>1 else a['axes'])))
    elif op=='Squeeze':v=np.squeeze(vals[0],axis=tuple(int(x) for x in vals[1]) if len(vals)>1 else tuple(a['axes']) if 'axes' in a else None)
    elif op=='Split':
        axis=a.get('axis',0);lengths=vals[1] if len(vals)>1 else a.get('split')
        if lengths is None:
            if vals[0].shape[axis]%len(node.output):return None
            lengths=[vals[0].shape[axis]//len(node.output)]*len(node.output)
        if sum(lengths)!=vals[0].shape[axis]:raise ValueError('Split lengths inconsistent')
        return np.split(vals[0],np.cumsum(lengths)[:-1],axis=axis)
    elif op=='Slice':
        starts,ends=vals[1:3];axes=vals[3] if len(vals)>3 else np.arange(len(starts));steps=vals[4] if len(vals)>4 else np.ones(len(starts),np.int64)
        slices=[slice(None)]*vals[0].ndim
        for ax,st,en,step in zip(axes,starts,ends,steps):slices[int(ax)]=slice(int(st),int(en),int(step))
        v=vals[0][tuple(slices)]
    elif op=='Reshape':
        target=[int(x) for x in vals[1]]
        if not a.get('allowzero',0):target=[vals[0].shape[i] if x==0 else x for i,x in enumerate(target)]
        v=np.reshape(vals[0],target)
    elif op=='Transpose':v=np.transpose(vals[0],a.get('perm'))
    elif op=='ConstantOfShape':
        if np.prod(vals[0])>4096:return None
        scalar=N.to_array(a['value']) if 'value' in a else np.array([0],np.float32)
        v=np.full(tuple(int(x) for x in vals[0]),scalar.item(),scalar.dtype)
    elif op=='Expand':
        target=np.broadcast_shapes(vals[0].shape,tuple(int(x) for x in vals[1]))
        if np.prod(target)>4096:return None
        v=np.broadcast_to(vals[0],target)
    elif op=='Range':
        start,end,step=[x.item() for x in vals]
        if step==0 or abs((end-start)/step)>4096:return None
        v=np.arange(start,end,step,dtype=vals[0].dtype)
    elif op=='ReduceProd':v=np.prod(vals[0],axis=tuple(int(x) for x in vals[1]) if len(vals)>1 else tuple(a['axes']) if 'axes' in a else None,keepdims=bool(a.get('keepdims',1)),dtype=vals[0].dtype)
    elif op=='Neg':v=-vals[0]
    elif op=='Mod':v=np.fmod(*vals) if a.get('fmod',0) else np.mod(*vals)
    else:return None
    return [v]


def prove_static_dataflow(model, checkpoint=lambda *a, **k: None):
    types={v.name:v.type for v in model.graph.input}
    c={};data={};errs=[];values_count=0
    for v in model.graph.initializer:
        types[v.name]=H.make_tensor_type_proto(v.data_type,list(v.dims))
        if np.prod(v.dims)<=4096:c[v.name]=N.to_array(v);data[v.name]=v
    versions={x.domain:x.version for x in model.opset_import};schemas={}
    for number,n in enumerate(model.graph.node):
        for x in n.input:
            if x not in types:types[x]=onnx.TypeProto()
        key=(n.op_type,n.domain)
        if key not in schemas:schemas[key]=onnx.defs.get_schema(n.op_type,versions[n.domain],n.domain)
        schema=schemas[key]
        try:
            serialized=schema._infer_node_outputs(n.SerializeToString(), {x:types[x].SerializeToString() for x in n.input if x}, {x:data[x].SerializeToString() for x in n.input if x and x in data}, {}, versions, model.ir_version)
            outputs={key:onnx.TypeProto.FromString(value) for key,value in serialized.items()}
            if n.op_type in ['GreaterOrEqual','LessOrEqual'] and all(fixed_extents(types[x]) is not None for x in n.input):
                outshape=np.broadcast_shapes(*(fixed_extents(types[x]) for x in n.input))
                outputs={n.output[0]:H.make_tensor_type_proto(onnx.TensorProto.BOOL,list(outshape))}
            types.update(outputs)
        except Exception as e:errs.append(dict(node=n.name,kind='schema',error=str(e)))
        try:values=evaluate_constant_node(n,c,types)
        except Exception as e:values=None;errs.append(dict(node=n.name,kind='value',error=str(e)))
        if values is not None:
            for out,v in zip(n.output,values):
                v=np.asarray(v)
                if v.size<=4096:
                    c[out]=v;data[out]=N.from_array(v,name=out);types[out]=H.make_tensor_type_proto(data[out].data_type,list(v.shape));values_count+=1
        if number%5000==0:checkpoint('static_shape_propagation', processed_nodes=number, total_nodes=len(model.graph.node))
    return types,c,errs


def checked_model(nodes, inputs, outputs, initializers=(), name='static-shape-control-check'):
    model=H.make_model(H.make_graph(nodes,name,inputs,outputs,initializer=list(initializers)),opset_imports=[H.make_opsetid('',17)])
    model.ir_version=8
    onnx.checker.check_model(model)
    return model


def verify_rejection_and_schema_rules():
    checks={}
    # Original internal/output annotations must never become proof axioms.
    x=H.make_tensor_value_info('x',onnx.TensorProto.FLOAT,[16])
    shape=H.make_tensor_value_info('runtime_shape',onnx.TensorProto.INT64,[2])
    y=H.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[4,4])
    model=checked_model([H.make_node('Reshape',['x','runtime_shape'],['y'],name='runtime-dependent-reshape')],[x,shape],[y])
    types,c,errors=prove_static_dataflow(model)
    checks['reject_runtime_shape_despite_declared_static_output']=fixed_extents(types['y']) is None and 'runtime_shape' not in c and not errors
    z=N.from_array(np.array(0,np.int64),name='zero');one=N.from_array(np.array(1,np.int64),name='one')
    limit=H.make_tensor_value_info('runtime_limit',onnx.TensorProto.INT64,[])
    model=checked_model([H.make_node('Range',['zero','runtime_limit','one'],['range'],name='runtime-dependent-range')],[limit],[H.make_tensor_value_info('range',onnx.TensorProto.INT64,[10])],[z,one])
    types,c,errors=prove_static_dataflow(model)
    checks['reject_runtime_range_despite_declared_static_output']=fixed_extents(types['range']) is None and 'range' not in c and not errors
    model=checked_model([H.make_node('GreaterOrEqual',['a','b'],['ge'],name='broadcast-comparison')],[H.make_tensor_value_info('a',onnx.TensorProto.FLOAT,[2,1,4]),H.make_tensor_value_info('b',onnx.TensorProto.FLOAT,[3,4])],[H.make_tensor_value_info('ge',onnx.TensorProto.BOOL,[2,3,4])])
    types,c,errors=prove_static_dataflow(model)
    checks['function_comparison_shape_without_runtime_values']=fixed_extents(types['ge']) == (2,3,4) and 'ge' not in c and not errors
    a=N.from_array(np.array([-7,7,-6,6],np.int64),name='a');b=N.from_array(np.array([3,-3,3,-3],np.int64),name='b')
    model=checked_model([H.make_node('Div',['a','b'],['q'],name='signed-division')],[],[H.make_tensor_value_info('q',onnx.TensorProto.INT64,[4])],[a,b])
    _,c,errors=prove_static_dataflow(model)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=1
    actual=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{})[0]
    checks['integer_division_truncates_toward_zero_against_actual_ort']=np.array_equal(c['q'],actual) and np.array_equal(actual,np.array([-2,-2,-2,-2],np.int64)) and not errors
    if not all(checks.values()):raise ValueError('shape proof negative/schema controls rejected: '+str(checks))
    return checks


def verify_control_arithmetic(model, controls, types, constants, run, d):
    names=sorted({v['tensor'] for v in controls})
    if any(name not in constants for name in names):raise ValueError('unknown shape control value')
    needed=set(names);selected=[]
    for node in reversed(model.graph.node):
        if not needed.intersection(node.output):continue
        if node.op_type in ('Shape', 'Size'):
            # Cut neural-data paths only at independently proven fixed extents.
            value=constants[node.output[0]]
            selected.append(H.make_node('Constant',[],list(node.output),name=node.name+'/proven_extent',value=N.from_array(value)))
        else:
            selected.append(copy.deepcopy(node));needed.update(v for v in node.input if v)
    selected.reverse()
    initializers=[copy.deepcopy(v) for v in model.graph.initializer if v.name in needed]
    graph_inputs={v.name for v in model.graph.input}
    if needed & graph_inputs:raise ValueError('shape control arithmetic still requires runtime input values')
    outputs=[H.make_tensor_value_info(name,H.np_dtype_to_tensor_dtype(constants[name].dtype),list(constants[name].shape)) for name in names]
    subgraph=checked_model(selected,[],outputs,initializers)
    path=run/'shape_control_arithmetic.onnx';onnx.save(subgraph,str(path))
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    actual=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']).run(None,{})
    checks={name:bool(value.dtype==constants[name].dtype and value.shape==constants[name].shape and np.array_equal(value,constants[name])) for name,value in zip(names,actual)}
    if not all(checks.values()):raise ValueError('independent ORT control arithmetic differs')
    d.save(run/'control_arithmetic_comparison.json',dict(status='pass',unique_controls=len(names),operand_slots=len(controls),all_exact=True,checks=checks,model_sha256=d.sha(path),scope='Actual ORT pure control arithmetic; Shape/Size cut only at independently proven fixed extents. Size semantics additionally checked against real ORT cases. No neural/frame execution or runtime input value assumptions.'))
    return dict(unique_controls=len(names),operand_slots=len(controls),all_exact=True,comparison_sha256=d.sha(run/'control_arithmetic_comparison.json'),model_sha256=d.sha(path))
