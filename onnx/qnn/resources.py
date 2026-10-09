"""Bind a compiled QNN profile to explicit logical/native ABI and host resources."""
import json,os,platform,sys
from pathlib import Path
import casadi,numpy as np,onnx
from session import NativeSession,sha,DTYPES
from tool_run import ROOT,SDK

RANGES={
 'img_shape':[1,2**31-1], 'command':[0,2],
 'max_obj_id':[0,2**31-1-1285], 'obj_idxes':[-3,2**31-1],
 'disappear_time':[0,2**31-2], 'track_count':[901,1285],
 'next_max_obj_id':[0,2**31-1], 'next_obj_idxes':[-3,2**31-1],
 'next_disappear_time':[0,2**31-1], 'next_track_count':[901,1285],
 'survivor_count_raw':[0,1285], 'decoded_count':[0,300],
 'vehicle_count':[0,96], 'vehicle_count_raw':[0,300],
 'sca_visible_count_raw':[0,40000], 'occ_segmentation':[0,1],
 # Full-validation outputs: class counts, fixed slots and explicit padding
 # are taken from the frozen decoder/occupancy/map source, not observations.
 'occ_instances':[0,96], 'track_labels':[0,9],
 'track_ids':[-3,2**31-1], 'track_query_indices':[-1,1284],
 'map_selected_labels':[0,2], 'map_selected_query_indices':[0,299],
}
def planning_spec():
 spec_path=Path(os.environ.get('UNIAD_BUNDLE_SPEC',str(ROOT/'onnx/qnn/planning_bundle_spec.template.json'))).expanduser()
 if not spec_path.is_absolute():spec_path=ROOT/spec_path
 spec=json.loads(spec_path.read_text())
 for row in [*spec.values(),*spec.get('evidence',{}).values()]:
  if isinstance(row,dict) and 'path' in row:
   path=Path(row['path']).expanduser();row['path']=str(path if path.is_absolute() else ROOT/path)
 spec['project_root']=str(ROOT)
 return spec


def terminal(run):
 run=Path(run).absolute();status=json.loads((run/'status.json').read_text());result=json.loads((run/'result.json').read_text())
 if status['status']!='pass' or result['status']!='pass' or status['result_sha256']!=sha(run/'result.json'):raise ValueError('tool evidence is not hash-bound terminal pass')
 return result

def bind(compile_run,bridge_build,logical_model,logical_sha256):
 compile_run=Path(compile_run).absolute();terminal(compile_run)
 build=json.loads((compile_run/'model_build.json').read_text())
 source=json.loads((compile_run/'source_identity.json').read_text())
 if source['model_sha256']!=build['source_model_sha256']:raise ValueError('compiled model/source mismatch')
 convert_run=Path(build['resources']['model.cpp']['path']).parent;result=terminal(convert_run)
 if build['converter_result_sha256']!=sha(convert_run/'result.json'):raise ValueError('converter/build mismatch')
 if result['source_identity_sha256']!=sha(convert_run/'source_identity.json'):raise ValueError('converter source identity mismatch')
 for name,row in build['resources'].items():
  if sha(row['path'])!=row['sha256'] or Path(row['path']).stat().st_size!=row['bytes']:raise ValueError('converter artifact differs: '+name)
 bridge_build=Path(bridge_build).absolute();terminal(bridge_build)
 bridge=json.loads((bridge_build/'bridge_build.json').read_text())
 if sha(bridge['source'])!=bridge['source_sha256']:raise ValueError('bridge source differs from build')
 logical_model=Path(logical_model).absolute()
 if sha(logical_model)!=logical_sha256:raise ValueError('logical model hash differs')
 logical=onnx.load(str(logical_model));native=onnx.load(source['model'])
 if sha(source['model'])!=source['model_sha256']:raise ValueError('native candidate model differs')
 net_path=convert_run/'model_net.json'
 if 'model_net.json' in result['artifacts'] and result['artifacts']['model_net.json']['sha256']!=sha(net_path):raise ValueError('converter network artifact hash differs')
 net=json.loads(net_path.read_text());tensors=net['graph']['tensors']
 if 'float_bw=32' not in net['converter_command']:raise ValueError('converter FP32 policy absent')
 abi=dict(schema='qnn-native-abi-v1')
 for kind,vs,code in [('inputs',logical.graph.input,0),('outputs',logical.graph.output,1)]:
  native_values=list(getattr(native.graph,'input' if kind=='inputs' else 'output'))
  if [v.name for v in vs]!=[v.name for v in native_values]:raise ValueError('logical/native ordered names differ')
  actual={n:v for n,v in tensors.items() if v['type']==code}
  if set(actual)!={v.name for v in vs}:raise ValueError('converter boundary names differ')
  abi[kind]=[]
  for v,nv in zip(vs,native_values):
   shape=[d.dim_value for d in v.type.tensor_type.shape.dim];dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type));row=actual[v.name]
   if nv.type.tensor_type.elem_type!=v.type.tensor_type.elem_type or DTYPES[row['data_type']]!=dtype:raise ValueError('boundary dtype differs: '+v.name)
   physical=row['dims'];candidate_shape=[d.dim_value for d in nv.type.tensor_type.shape.dim]
   if [d for d in shape if d!=1]!=[d for d in candidate_shape if d!=1] or [d for d in shape if d!=1]!=[d for d in physical if d!=1]:raise ValueError('boundary permutation or extent change: '+v.name)
   mapping=dict(name=v.name,native_name=v.name,shape=shape,native_shape=physical,dtype=dtype)
   if shape!=physical:mapping['wire_view']='singleton_axes'
   if dtype=='int64':
    if v.name not in RANGES:raise ValueError('model integer arithmetic range undefined: '+v.name)
    mapping['integer_range']=RANGES[v.name]
   abi[kind].append(mapping)
 spec=planning_spec()
 def asset(path,expected=None):
  path=Path(path).absolute();digest=sha(path)
  if expected is not None and digest!=expected:raise ValueError('resource hash differs: '+str(path))
  return dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
 assets={
  'logical_model':asset(logical_model,logical_sha256),'native_model':asset(source['model'],source['model_sha256']),
  'model_lib':asset(build['library'],build['library_sha256']),'bridge':asset(bridge['library'],bridge['library_sha256']),
  'backend_lib':asset(SDK/'lib/x86_64-linux-clang/libQnnCpu.so',build['cpu_backend_sha256']),
  'converter_net':asset(net_path),'initial_state':asset(ROOT/'onnx/resources/initial_state.npz',spec['initial_state']['sha256']),
  'collision_optimizer':asset(spec['collision_optimizer']['path'],spec['collision_optimizer']['sha256']),
  'host':asset(ROOT/'onnx/fixed/host.py'),'state_contract':asset(ROOT/'onnx/fixed/state_contract.py'),
  'adapter':asset(ROOT/'onnx/qnn/session.py'),'profile':asset(__file__),'frame_driver':asset(ROOT/'onnx/validation/run_qnn_real_frames.py'),
 }
 manifest=dict(schema='qnn-host-profile-candidate-v1',assets=assets,abi=abi,
  host_policy=dict(can_bus_mode='official_test_legacy',id_scope='session',coordinate_mode='legacy_int'),
  runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,casadi=casadi.__version__),
  backend=dict(type='QNN_CPU',sdk=str(SDK),precision='float32'),
  converter_result_sha256=sha(convert_run/'result.json'),compile_result_sha256=sha(compile_run/'result.json'),
  scope='Hash-bound development profile only; not a portable release or neural task acceptance.')
 return manifest

def session_for(manifest):
 if manifest['backend'].get('execution')=='partitioned':
  from partition_session import PartitionSession
  return PartitionSession(manifest)
 assets=manifest['assets'];names=('bridge','model_lib','backend_lib')
 return NativeSession(*(assets[n]['path'] for n in names),manifest['abi'],{n:assets[n]['sha256'] for n in names})
