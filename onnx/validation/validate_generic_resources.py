"""Independent layout witnesses and actual shared-resource ownership controls."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from eliminate_identity_views import rewrite
from partition_session import bind_partitions, PartitionSession
import partition_session
from borrowed_session import BorrowedNativeSession
from tool_run import sha, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build-run',type=Path)
    p.add_argument('--bridge-build',type=Path)
    p.add_argument('--logical-model',type=Path)
    p.add_argument('--logical-model-sha256')
    a=p.parse_args()
    if a.build_run is None:
        checks={}
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
        for dtype in (onnx.TensorProto.FLOAT,onnx.TensorProto.INT64,onnx.TensorProto.BOOL):
            nodes=[H.make_node('Cast',['x'],['cast'],name='same-type-cast',to=dtype),H.make_node('Identity',['cast'],['a'],name='alias'),H.make_node('Reshape',['a','square'],['b'],name='same-shape'),H.make_node('Transpose',['b'],['t'],name='nonidentity-square',perm=[1,0]),H.make_node('Transpose',['b'],['id'],name='identity-order',perm=[0,1]),H.make_node('Reshape',['t','flat'],['f'],name='rank-change'),H.make_node('Identity',['id'],['public'],name='named-output'),H.make_node('Cast',['t'],['named_cast'],name='public-cast',to=dtype)]
            model=H.make_model(H.make_graph(nodes,'independent-layout-witness',[H.make_tensor_value_info('x',dtype,[2,2])],[H.make_tensor_value_info('f',dtype,[4]),H.make_tensor_value_info('public',dtype,[2,2]),H.make_tensor_value_info('named_cast',dtype,[2,2])],[N.from_array(np.array([2,2],np.int64),'square'),N.from_array(np.array([4],np.int64),'flat')]),opset_imports=[H.make_opsetid('',18)]);model.ir_version=8
            candidate=copy.deepcopy(model);removed=rewrite(candidate)
            dt=H.tensor_dtype_to_np_dtype(dtype)
            x=np.array([[False,True],[True,False]],dtype=dt) if dtype==onnx.TensorProto.BOOL else np.array([[0,2],[11,19]],dtype=dt)
            ref=ort.InferenceSession(model.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{'x':x})
            out=ort.InferenceSession(candidate.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{'x':x})
            checks[str(dtype)+'_all_elements_bit_exact']=all(np.array_equal(v,w) for v,w in zip(ref,out))
            checks[str(dtype)+'_rank_and_real_shuffle_kept']={n.name for n in candidate.graph.node}=={'nonidentity-square','rank-change','named-output'}
            checks[str(dtype)+'_public_abi_kept']=[v.SerializeToString() for v in candidate.graph.output]==[v.SerializeToString() for v in model.graph.output]
        base=(np.arange(512*512,dtype=np.float32)%17).reshape(512,512)*.25
        nodes=[H.make_node('Identity',['base'],['view'],name='base-alias'),H.make_node('Reshape',['view','shape'],['other'],name='base-view'),H.make_node('Add',['state','base'],['v1'],name='add0'),H.make_node('Add',['v1','delta'],['v2'],name='add1'),H.make_node('Add',['v2','other'],['v3'],name='add2'),H.make_node('Add',['v3','delta'],['next'],name='add3')]
        vi=lambda name:H.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[512,512])
        model=H.make_model(H.make_graph(nodes,'shared-immutable-recurrence',[vi('state'),vi('delta')],[vi('next')],[N.from_array(base,'base'),N.from_array(np.array([512,512],np.int64),'shape')]),opset_imports=[H.make_opsetid('',18)]);model.ir_version=8
        onnx.save(model,str(Path.cwd()/'logical.onnx'))
        rewrite(model);onnx.save(model,str(Path.cwd()/'model.onnx'))
        save(Path.cwd()/'generic_resource_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,model_sha256=sha(Path.cwd()/'model.onnx'),scope='Independent element-routing witnesses; real rank changes and nonidentity square transposes must survive. Actual shared-resource backend and neural metrics separate.'))
        if not all(checks.values()):raise ValueError('generic resource control failed')
        return
    manifest=bind_partitions(a.build_run,a.bridge_build,a.logical_model,a.logical_model_sha256)
    partition_session.NativeSession=BorrowedNativeSession
    save(Path.cwd()/'profile.json',manifest)
    checks={}
    base=(np.arange(512*512,dtype=np.float32)%17).reshape(512,512)*.25
    pool=manifest['plan']['shared_constants']
    checks['one_physical_constant_buffer']=pool['unique_data_bytes']==base.nbytes
    checks['only_internal_cut_operand']=len(manifest['abi']['inputs'])==2 and all(r['source_name'] not in {'state','delta'} for r in pool['entries'])
    feed=dict(state=np.zeros_like(base),delta=np.ones_like(base));saved_first=None
    with PartitionSession(manifest) as session:
        initial_ids={n:v.ctypes.data for n,v in session._shared_values.items()}
        for i in range(6):
            before={n:v.copy() for n,v in feed.items()}
            result=session.run(None,feed)[0]
            expected=before['state']+base+before['delta']+base+before['delta']
            checks['frame'+str(i)+'_exact_recurrence']=np.array_equal(result,expected)
            checks['frame'+str(i)+'_inputs_unchanged']=all(np.array_equal(v,before[n]) for n,v in feed.items())
            checks['frame'+str(i)+'_pool_storage_reused']=initial_ids=={n:v.ctypes.data for n,v in session._shared_values.items()} and all(np.array_equal(v,base) for v in session._shared_values.values())
            if saved_first is None:saved_first=(result,result.copy())
            checks['frame'+str(i)+'_retained_output_unchanged']=np.array_equal(*saved_first)
            feed['state']=result
        checks['actual_all_parts_executed']=len(session.last_execution)==len(manifest['parts'])
        save(Path.cwd()/'native_abi.json',session.native_abi)
    resource=manifest['assets']['shared_constant_pool']
    for name in ('hash','dtype','shape','duplicate','key','collision'):
        bad=copy.deepcopy(manifest);row=bad['plan']['shared_constants']['entries'][0]
        if name=='hash':bad['assets']['shared_constant_pool']['sha256']='0'*64
        if name=='dtype':row['dtype']='float64'
        if name=='shape':row['shape']=[511,512]
        if name=='duplicate':bad['plan']['shared_constants']['entries'].append(copy.deepcopy(row))
        if name=='key':row['key']='missing'
        if name=='collision':row['source_name']='state'
        try:
            s=PartitionSession(bad);s.close()
        except (ValueError,KeyError):checks[name+'_rejected_before_execute']=True
        else:checks[name+'_rejected_before_execute']=False
    save(Path.cwd()/'generic_resource_backend_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,profile_sha256=sha(Path.cwd()/'profile.json'),shared_data_bytes=pool['unique_data_bytes'],resource_bytes=resource['bytes'],scope='Actual many-part repeated-state borrowing, shared storage, retained-output ownership and fail-closed resource controls. No UniAD task acceptance.'))
    if not all(checks.values()):raise ValueError('shared constant backend control failed')


if __name__=='__main__':main()
