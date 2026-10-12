#!/usr/bin/env python3
"""Actual mixed-dtype QNN boundary and repeated session controls, separate from UniAD."""
import argparse,ctypes as C,json,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as H,numpy_helper as N
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from session import NativeSession,sha,DTYPES
from sdk_environment import cpu_backend

def main():
    p=argparse.ArgumentParser();p.add_argument('--make-model',action='store_true');p.add_argument('--vector-count',action='store_true');p.add_argument('--column-ids',action='store_true');p.add_argument('--guard-signed32',action='store_true');p.add_argument('--model',type=Path);p.add_argument('--logical-model',type=Path);p.add_argument('--logical-model-sha256');p.add_argument('--bridge',type=Path);p.add_argument('--model-lib',type=Path);p.add_argument('--backend',type=Path,default=None);args=p.parse_args()
    if args.make_model:
        ins=[H.make_tensor_value_info(n,t,s) for n,t,s in [('state',onnx.TensorProto.FLOAT,[2,3]),('delta',onnx.TensorProto.FLOAT,[2,3]),('ids',onnx.TensorProto.INT64,[4]),('mask',onnx.TensorProto.BOOL,[4]),('count',onnx.TensorProto.INT64,[1] if args.vector_count else [])]]
        outs=[H.make_tensor_value_info(n,t,s) for n,t,s in [('next_state',onnx.TensorProto.FLOAT,[2,3]),('next_ids',onnx.TensorProto.INT64,[4]),('next_mask',onnx.TensorProto.BOOL,[4]),('next_count',onnx.TensorProto.INT64,[1] if args.vector_count else [])]]
        graph=H.make_graph([H.make_node('Add',['state','delta'],['next_state']),H.make_node('Gather',['ids','reverse'],['next_ids'],axis=0),H.make_node('Not',['mask'],['next_mask']),H.make_node('Identity',['count'],['next_count'])],'mixed_native_control',ins,outs,[N.from_array(np.array([3,2,1,0],np.int64),name='reverse')])
        if args.column_ids:
            graph.node[1].output[0]='gathered_ids'
            graph.node.append(H.make_node('Reshape',['gathered_ids','column_shape'],['next_ids']))
            graph.initializer.append(N.from_array(np.array([4,1],np.int64),name='column_shape'))
            del graph.output[1].type.tensor_type.shape.dim[:]
            graph.output[1].type.tensor_type.shape.dim.add().dim_value=4
            graph.output[1].type.tensor_type.shape.dim.add().dim_value=1
        model=H.make_model(graph,opset_imports=[H.make_opsetid('',16)]);model.ir_version=8;onnx.checker.check_model(model,full_check=True);out=Path.cwd()/'model.native-control.onnx';onnx.save(model,str(out))
        (Path.cwd()/'control_model.json').write_text(json.dumps(dict(status='pass',model=str(out),model_sha256=sha(out),script_sha256=sha(__file__),scope='Synthetic boundary control; not UniAD neural acceptance.'),indent=2)+'\n');return
    args.backend = args.backend or cpu_backend()
    logical=args.logical_model or args.model
    if args.logical_model and sha(logical)!=args.logical_model_sha256:raise ValueError('logical ABI model identity differs')
    proto=onnx.load(str(logical));lib=C.CDLL(str(args.bridge));lib.q4_qnn_create.argtypes=[C.c_char_p,C.c_char_p];lib.q4_qnn_create.restype=C.c_void_p;lib.q4_qnn_describe.argtypes=[C.c_void_p];lib.q4_qnn_describe.restype=C.c_char_p;lib.q4_qnn_error.restype=C.c_char_p;lib.q4_qnn_destroy.argtypes=[C.c_void_p]
    handle=lib.q4_qnn_create(str(args.backend).encode(),str(args.model_lib).encode())
    if not handle:raise RuntimeError(lib.q4_qnn_error().decode())
    try:native=json.loads(lib.q4_qnn_describe(handle))
    finally:lib.q4_qnn_destroy(handle)
    abi=dict(schema='qnn-native-abi-v1');checks={}
    for kind,values in [('inputs',proto.graph.input),('outputs',proto.graph.output)]:
        actual={r['name']:r for r in native[kind]};abi[kind]=[]
        for v in values:
            if v.name not in actual:raise ValueError('synthetic ABI needs unchanged name: '+v.name)
            row=actual[v.name];dtype=str(H.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type));shape=[d.dim_value for d in v.type.tensor_type.shape.dim]
            if DTYPES[row['dtype_code']]!=dtype:raise ValueError('native dtype changed: '+v.name)
            mapping=dict(name=v.name,native_name=v.name,dtype=dtype,shape=shape,native_shape=row['shape'])
            if shape!=row['shape'] and not (shape==[] and row['shape']==[1]):mapping['wire_view']='singleton_axes'
            abi[kind].append(mapping)
    if args.guard_signed32:
        for rows in (abi['inputs'],abi['outputs']):
            for row in rows:
                if row['dtype']=='int64':row['integer_range']=[-2**31,2**31-1]
    resources={n:sha(v) for n,v in [('bridge',args.bridge),('model_lib',args.model_lib),('backend_lib',args.backend)]}
    (Path.cwd()/'abi.json').write_text(json.dumps(dict(source_model_sha256=sha(args.model),**abi),indent=2)+'\n')
    if args.guard_signed32:
        bad_view=json.loads(json.dumps(abi))
        changed=False
        for rows in (bad_view['inputs'],bad_view['outputs']):
            for row in rows:
                if 'wire_view' in row:row.pop('wire_view');changed=True
        if changed:
            try:
                with NativeSession(args.bridge,args.model_lib,args.backend,bad_view,resources):pass
            except ValueError:checks['undeclared_singleton_wire_rejected']=True
            else:checks['undeclared_singleton_wire_rejected']=False
        unrestricted=json.loads(json.dumps(abi))
        for rows in (unrestricted['inputs'],unrestricted['outputs']):
            for row in rows:row.pop('integer_range',None)
        try:
            with NativeSession(args.bridge,args.model_lib,args.backend,unrestricted,resources):pass
        except ValueError:checks['unbounded_int64_abi_rejected']=True
        else:checks['unbounded_int64_abi_rejected']=False
    with NativeSession(args.bridge,args.model_lib,args.backend,abi,resources) as session:
        feed=dict(state=np.zeros((2,3),np.float32),delta=np.ones((2,3),np.float32),ids=np.array([2**53+1,-2**53-3,np.iinfo(np.int64).max,np.iinfo(np.int64).min],np.int64),mask=np.array([False,True,False,True]),count=np.full([1] if args.vector_count else [],2**53+7,np.int64))
        if args.guard_signed32:
            for label,bad in [('large_ids',dict(ids=feed['ids'])),('large_count',dict(count=feed['count']))]:
                restricted=dict(feed,ids=np.array([-2**31,-7,42,2**31-1],np.int64),count=np.full([1] if args.vector_count else [],2**31-1,np.int64));restricted.update(bad)
                try:session.run(None,restricted)
                except ValueError:checks[label+'_rejected_before_execute']=True
                else:checks[label+'_rejected_before_execute']=False
            feed.update(ids=np.array([-2**31,-7,42,2**31-1],np.int64),count=np.full([1] if args.vector_count else [],2**31-1,np.int64))
        baseline={k:v.copy() for k,v in feed.items()}
        diagnostic=[]
        for frame in range(6):
            before_call={k:v.copy() for k,v in feed.items()}
            outs=session.run(None,feed);names=[v.name for v in session.get_outputs()];values=dict(zip(names,outs))
            for name,expected in [('next_state',feed['state']+feed['delta']),('next_ids',feed['ids'][::-1]),('next_mask',~feed['mask']),('next_count',feed['count'])]:
                checks['frame_'+str(frame)+'_'+name]=values[name].dtype==expected.dtype and values[name].shape==expected.shape and np.array_equal(values[name],expected)
            if frame==0:diagnostic=[dict(output=n,actual=values[n].tolist(),expected=v.tolist()) for n,v in [('next_ids',feed['ids'][::-1]),('next_count',feed['count'])]]
            checks['frame_'+str(frame)+'_frozen_ids']=np.array_equal(values['next_ids'],baseline['ids'][::-1] if frame%2==0 else baseline['ids'])
            checks['frame_'+str(frame)+'_frozen_count']=np.array_equal(values['next_count'],baseline['count'])
            checks['frame_'+str(frame)+'_inputs_not_mutated']=all(np.array_equal(feed[k],v) for k,v in before_call.items())
            feed.update(state=values['next_state'],ids=values['next_ids'],mask=values['next_mask'],count=values['next_count'])
        checks['inputs_not_mutated']=all(np.array_equal(v,baseline[k]) for k,v in baseline.items() if k=='delta')
        for label,change in [('wrong_dtype',dict(ids=feed['ids'].astype(np.float64))),('wrong_extent',dict(state=np.zeros((3,2),np.float32))),('nonfinite',dict(delta=np.full((2,3),np.nan,np.float32))),('missing_input',None)]:
            bad=dict(feed);bad.update(change or {});
            if change is None:bad.pop('count')
            try:session.run(None,bad)
            except ValueError:checks[label+'_rejected']=True
            else:checks[label+'_rejected']=False
        fresh=session.run(['next_count'],feed)[0];checks['selected_output_order_and_scalar']=fresh.shape==((1,) if args.vector_count else ()) and fresh.dtype==np.int64 and int(fresh.item())==(2**31-1 if args.guard_signed32 else 2**53+7)
    try:session.run(None,feed)
    except RuntimeError:checks['closed_session_rejected']=True
    else:checks['closed_session_rejected']=False
    result=dict(status='pass' if all(checks.values()) else 'failed',checks=checks,resource_hashes=resources,source_model_sha256=sha(args.model),native_abi=native,logical_model_sha256=sha(logical),first_frame_integer_diagnostic=diagnostic,integer_policy='signed32_guarded' if args.guard_signed32 else 'unrestricted_int64_test',scope='Synthetic actual CPU QNN API controls only; no UniAD or task acceptance.')
    (Path.cwd()/'native_controls.json').write_text(json.dumps(result,indent=2)+'\n')
    if not all(checks.values()):raise ValueError('native session boundary controls failed')
if __name__=='__main__':main()
