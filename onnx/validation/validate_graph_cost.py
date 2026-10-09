"""Independent small arithmetic/lifetime controls for the static cost ledger."""
import json,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from graph_cost import cost_ledger
from tool_run import save,sha

def model(nodes,inputs,outputs,weights=(),values=()):
    m=H.make_model(H.make_graph(nodes,'control',inputs,outputs,list(weights),value_info=values),opset_imports=[H.make_opsetid('',16)])
    onnx.checker.check_model(m);return onnx.shape_inference.infer_shapes(m)
def f(name,dims):return H.make_tensor_value_info(name,onnx.TensorProto.FLOAT,dims)
def main():
    rows=[]
    examples=[
        ('grouped_conv',model([H.make_node('Conv',['x','w'],['y'],group=2)],[f('x',[2,4,5,5])],[f('y',[2,6,3,3])],[N.from_array(np.ones((6,2,3,3),np.float32),'w')]),1944),
        ('vector_dot',model([H.make_node('MatMul',['x','z'],['y'])],[f('x',[7]),f('z',[7])],[f('y',[])]),7),
        ('broadcast_matmul',model([H.make_node('MatMul',['x','z'],['y'])],[f('x',[2,1,3,4]),f('z',[1,5,4,6])],[f('y',[2,5,3,6])]),720),
        ('transposed_gemm',model([H.make_node('Gemm',['x','z'],['y'],transA=1,transB=1)],[f('x',[4,3]),f('z',[6,4])],[f('y',[3,6])]),72),
    ]
    for name,m,expected in examples:
        r=cost_ledger(m,'synthetic')['summary']
        if r['dense_macs_per_frame']!=expected or r['dense_coverage_gaps'] or r['missing_shape_names']:raise ValueError(name+' arithmetic contract differs')
        rows.append(dict(name=name,macs=expected))
    m=model([H.make_node('Identity',['x'],['a']),H.make_node('Add',['a','a'],['b']),H.make_node('Add',['a','b'],['c']),H.make_node('Identity',['x'],['dead'])],[f('x',[2])],[f('a',[2]),f('c',[2])],values=[f('dead',[2])])
    plan=dict(source_model_sha256='synthetic',parts=[dict(index=0,source_node_indices=[0,1]),dict(index=1,source_node_indices=[2])])
    r=cost_ledger(m,'synthetic',plan)
    if r['summary']['ssa_dynamic_frontier_bytes']!=24 or r['summary']['cut_consumer_logical_bytes']!=16 or r['summary']['omitted_unreachable_nodes']!=1:raise ValueError('fork/public/lifetime contract differs')
    rows.append(dict(name='fork_public_retention_dead_node',peak_frontier=24,cut_consumer_bytes=16))
    m=model([H.make_node('Constant',[],['w'],value=N.from_array(np.ones((2,),np.float32))),H.make_node('Add',['w','w'],['constant_root']),H.make_node('RandomUniform',[],['rng'],shape=[2]),H.make_node('Add',['rng','constant_root'],['y'])],[],[f('y',[2])])
    r=cost_ledger(m,'synthetic')
    if r['executable_node_indices']!=[2,3] or r['immutable_root_node_indices']!=[0,1]:raise ValueError('nondeterministic roots contract differs')
    rows.append(dict(name='root_and_rng_classification',executable=[2,3],immutable=[0,1]))
    rejected=False
    try:cost_ledger(examples[0][1],'synthetic',dict(source_model_sha256='synthetic',parts=[dict(index=0,source_node_indices=[])]))
    except ValueError:rejected=True
    if not rejected:raise ValueError('incomplete plan not rejected')
    report=dict(status='pass',controls=rows,negative_controls=['missing_executable_node'],helper_sha256=sha(ROOT/'onnx/qnn/graph_cost.py'),scope='Static formula and dynamic fork/public/root/RNG controls; not neural acceptance or physical memory measurement.')
    save(Path.cwd()/'graph_cost_control.json',report);print(json.dumps(dict(status='pass',controls=len(rows)+1)))
if __name__=='__main__':main()
