"""Actual source-derived integer max scalar guard, including singleton wire ABI."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from eliminate_identity_views import rewrite
from session import NativeSession,DTYPES
from resources import terminal
from tool_run import SDK,sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-model',type=Path);p.add_argument('--source-sha256');p.add_argument('--compile-run',type=Path);p.add_argument('--bridge-build',type=Path);a=p.parse_args()
    if a.compile_run is None:
        if sha(a.source_model)!=a.source_sha256:raise ValueError('scalar guard source differs')
        m=onnx.load(str(a.source_model));nodes=[copy.deepcopy(n) for n in m.graph.node if n.name.startswith('__q3_sca_max/qnn_select/')]
        source=next(n.input[0] for n in nodes if n.op_type=='Gather');used={v for n in nodes for v in n.input};output=next(n.output[0] for n in nodes if n.name.endswith('/restore_scalar'))
        for n in nodes:
            for field in ('input','output'):
                values=getattr(n,field)
                for i,name in enumerate(values):values[i]={'maximum': 'maximum',source:'counts',output:'maximum'}.get(name,name)
        model=H.make_model(H.make_graph(nodes,'source-integer-max-scalar',[H.make_tensor_value_info('counts',onnx.TensorProto.INT64,[6])],[H.make_tensor_value_info('maximum',onnx.TensorProto.INT64,[])],[copy.deepcopy(v) for v in m.graph.initializer if v.name in used]),opset_imports=list(m.opset_import));model.ir_version=m.ir_version
        rewrite(model)
        if not any(n.op_type=='Reshape' and any(child.name.endswith('/restore_scalar') and child.input[0] in n.output for child in model.graph.node) for n in model.graph.node):raise ValueError('SDK vector-rank guard lost')
        onnx.save(model,str(Path.cwd()/'model.onnx'))
        save(Path.cwd()/'scalar_guard_model.json',dict(status='pass',model_sha256=sha(Path.cwd()/'model.onnx'),source_sha256=a.source_sha256,scope='Source-derived integer comparison/select circuit only; necessary scalar rank guard retained.'))
        return
    terminal(a.compile_run);terminal(a.bridge_build)
    build=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());net=json.loads((Path(build['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors']
    abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
    for kind,name,shape in [('inputs','counts',[6]),('outputs','maximum',[])]:
        row=net[name];entry=dict(name=name,native_name=name,shape=shape,native_shape=row['dims'],dtype=DTYPES[row['data_type']],integer_range=[0,40000])
        if row['dims']!=shape:entry['wire_view']='singleton_axes'
        abi[kind].append(entry)
    backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so';checks={}
    with NativeSession(bridge['library'],build['library'],backend,abi,dict(bridge=bridge['library_sha256'],model_lib=build['library_sha256'],backend_lib=sha(backend))) as session:
        base=np.array([0,1,10201,40000,7,3],np.int64)
        for i in range(6):
            value=np.roll(base,i);snapshot=value.copy();out=session.run(None,{'counts':value})[0]
            checks['permutation'+str(i)]=bool(out.shape==() and out.dtype==np.int64 and int(out)==int(value.max()) and np.array_equal(snapshot,value))
        for maximum in (0,892,10201,40000):
            value=np.full(6,maximum,np.int64);out=session.run(None,{'counts':value})[0];checks['tie'+str(maximum)]=bool(int(out)==maximum)
        save(Path.cwd()/'native_abi.json',session.native_abi)
    save(Path.cwd()/'scalar_guard_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,scope='Actual QNN CPU integer max and rank-one/scalar wire guard over SCA count domain; no neural acceptance.'))
    if not all(checks.values()):raise ValueError('actual scalar guard failed')


if __name__=='__main__':main()
