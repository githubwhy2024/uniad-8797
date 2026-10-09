"""Execute source-bound QNN graph parts with one native context alive at a time."""
import copy,json,math,platform,resource,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import casadi,numpy as np,onnx
from session import NativeSession,NonfiniteOutputError,IntegerOutputRangeError,sha,all_finite
from resources import RANGES,terminal,planning_spec
from tool_run import ROOT,SDK,save
from sdk_environment import sdk_identity


def bind_partitions(build_run,bridge_build,logical_model,logical_sha256):
 build_run=Path(build_run).absolute();terminal(build_run);build_path=build_run/'partition_build.json';build=json.loads(build_path.read_text());plan_path=Path(build['plan']);plan=json.loads(plan_path.read_text())
 if sha(plan_path)!=build['plan_sha256'] or sha(build['audit'])!=build['audit_sha256'] or json.loads(Path(build['audit']).read_text())['status']!='pass':raise ValueError('partition build/coverage identities differ')
 if len(build['parts'])!=len(plan['parts']):raise ValueError('partition build coverage is incomplete')
 layout_report=None
 if 'layout_hints' in build:
  if sha(build['layout_hints'])!=build['layout_hints_sha256']:raise ValueError('partition source layout identity differs')
  layout_report=json.loads(Path(build['layout_hints']).read_text())
  if layout_report['status']!='pass' or layout_report['source_model_sha256']!=plan['source_model_sha256']:raise ValueError('partition source layout provenance differs')
 bridge_build=Path(bridge_build).absolute();terminal(bridge_build);bridge=json.loads((bridge_build/'bridge_build.json').read_text());backend=SDK/'lib/x86_64-linux-clang/libQnnCpu.so'
 if sha(bridge['source'])!=bridge['source_sha256'] or sha(backend)!=build['backend_sha256']:raise ValueError('partition bridge/backend differs')
 logical_model=Path(logical_model).absolute()
 if sha(logical_model)!=logical_sha256 or sha(plan['source_model'])!=plan['source_model_sha256']:raise ValueError('partition logical/parent model differs')
 logical=onnx.load(str(logical_model));parent=onnx.load(plan['source_model']);abi=dict(schema='qnn-logical-partition-abi-v1')
 for kind in ('inputs','outputs'):
  values=list(getattr(logical.graph,'input' if kind=='inputs' else 'output'));pv=list(getattr(parent.graph,'input' if kind=='inputs' else 'output'))
  if [v.name for v in values]!=[v.name for v in pv]:raise ValueError('parent/logical ordered boundary names differ')
  abi[kind]=[]
  for v,pv in zip(values,pv):
   shape=[d.dim_value for d in v.type.tensor_type.shape.dim];parent_shape=[d.dim_value for d in pv.type.tensor_type.shape.dim];dtype=str(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))
   if v.type.tensor_type.elem_type!=pv.type.tensor_type.elem_type or [d for d in shape if d!=1]!=[d for d in parent_shape if d!=1]:raise ValueError('parent/logical boundary dtype/axis differs')
   row=dict(name=v.name,shape=shape,parent_shape=parent_shape,dtype=dtype)
   if dtype=='int64':
    if v.name not in RANGES:raise ValueError('undefined public integer range: '+v.name)
    row['integer_range']=RANGES[v.name]
   abi[kind].append(row)
 def asset(path,expected=None):
  p=Path(path).absolute();h=sha(p)
  if expected is not None and h!=expected:raise ValueError('partition resource differs: '+str(p))
  return dict(path=str(p),sha256=h,bytes=p.stat().st_size)
 spec=planning_spec();assets=dict(logical_model=asset(logical_model,logical_sha256),native_model=asset(plan['source_model'],plan['source_model_sha256']),initial_state=asset(ROOT/'onnx/resources/initial_state.npz',spec['initial_state']['sha256']),collision_optimizer=asset(spec['collision_optimizer']['path'],spec['collision_optimizer']['sha256']),bridge=asset(bridge['library'],bridge['library_sha256']),backend_lib=asset(backend,build['backend_sha256']),partition_plan=asset(plan_path,build['plan_sha256']),partition_build=asset(build_path),partition_audit=asset(build['audit'],build['audit_sha256']),adapter=asset(ROOT/'onnx/qnn/session.py'),atomic_save_source=asset(ROOT/'onnx/qnn/tool_run.py'),partition_adapter=asset(__file__),profile=asset(ROOT/'onnx/qnn/resources.py'),frame_driver=asset(ROOT/'onnx/validation/run_qnn_real_frames.py'),host=asset(ROOT/'onnx/fixed/host.py'),state_contract=asset(ROOT/'onnx/fixed/state_contract.py'))
 if 'shared_constants' in plan:
  pool=plan['shared_constants'];assets['shared_constant_pool']=asset(pool['path'],pool['sha256'])
 if layout_report is not None:assets['layout_hints']=asset(build['layout_hints'],build['layout_hints_sha256'])
 for row,part in zip(build['parts'],plan['parts']):
  convert=terminal(row['converter_run']);compile=terminal(row['compile_run']);actual=json.loads((Path(row['compile_run'])/'model_build.json').read_text())
  if row['index']!=part['index'] or row['native_model_sha256']!=part['native_model_sha256'] or actual['source_model_sha256']!=part['native_model_sha256'] or actual['library']!=row['library'] or actual['library_sha256']!=row['library_sha256']:raise ValueError('compiled part provenance differs')
  if sha(Path(row['converter_run'])/'result.json')!=row['converter_result_sha256'] or sha(Path(row['compile_run'])/'result.json')!=row['compile_result_sha256']:raise ValueError('part terminal result differs')
  if layout_report is not None:
   for kind in ('inputs','outputs'):
    expected={r['name']:layout_report['layouts'][r['source_name']] for r in part[kind] if r['source_name'] in layout_report['layouts']}
    key=kind[:-1]+'_layouts.json';artifact=convert['artifacts'].get(key)
    if expected:
     if artifact is None or sha(artifact['path'])!=artifact['sha256'] or json.loads(Path(artifact['path']).read_text())!=expected:raise ValueError('compiled cut layout policy differs from source')
    elif artifact is not None:raise ValueError('unexpected cut layout declaration')
  assets['part_'+str(row['index'])]=asset(row['library'],row['library_sha256']);assets['part_model_'+str(row['index'])]=asset(part['native_model'],part['native_model_sha256'])
 return dict(schema='qnn-host-profile-candidate-v1',assets=assets,abi=abi,parts=copy.deepcopy(build['parts']),plan=plan,host_policy=dict(can_bus_mode='official_test_legacy',id_scope='session',coordinate_mode='legacy_int'),runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,casadi=casadi.__version__),backend=dict(type='QNN_CPU',sdk=str(SDK),sdk_identity=sdk_identity(SDK),precision='float32',execution='partitioned'),scope='Source-audited development partition profile; actual per-part loads/own-state frames/tasks/portable release remain separate.')

class PartitionSession:
 def __init__(self,manifest):
  self.manifest=manifest;self._closed=False;self._lock=threading.Lock();self.last_execution=[];self.native_abi=dict(schema='qnn-partition-descriptors-v1',parts=[]);self.resource_hashes={n:v['sha256'] for n,v in manifest['assets'].items()};self._maps=manifest['abi']
  self._shared_values={}
  pool=manifest.get('shared_constants',manifest.get('plan',{}).get('shared_constants'))
  if pool is not None:
   resource_row=manifest['assets'].get('shared_constant_pool')
   if resource_row is None or sha(resource_row['path'])!=resource_row['sha256'] or resource_row['sha256']!=pool['sha256']:raise ValueError('shared immutable resource differs')
   with np.load(resource_row['path'],allow_pickle=False) as archive:
    if set(archive.files)!={r['key'] for r in pool['entries']}:raise ValueError('shared immutable key closure differs')
    buffers={k:archive[k] for k in archive.files}
   cut_inputs={v['source_name'] for part in manifest['plan']['parts'] for v in part['inputs']}
   cut_outputs={v['source_name'] for part in manifest['plan']['parts'] for v in part['outputs']}
   if not {v['source_name'] for v in pool['entries']}<=cut_inputs-cut_outputs-{r['name'] for r in self._maps['inputs']}:raise ValueError('shared root cut closure differs')
   hashes={}
   for key,data in buffers.items():
    if not data.flags.c_contiguous or not data.flags.aligned or not data.flags.writeable or not all_finite(data):raise ValueError('shared immutable buffer contract differs')
    hashes[key]=__import__('hashlib').sha256(memoryview(data).cast('B')).hexdigest()
   for row in pool['entries']:
    data=buffers[row['key']]
    if data.dtype!=np.dtype(row['dtype']) or data.nbytes!=row['bytes'] or hashes[row['key']]!=row['data_sha256'] or row['key']!='sha_'+hashes[row['key']]:raise ValueError('shared immutable data differs')
    if row['source_name'] in self._shared_values or row['source_name'] in {r['name'] for r in self._maps['inputs']}:raise ValueError('shared root name collision')
    self._shared_values[row['source_name']]=data.reshape(row['shape'])
  for v in manifest['assets'].values():
   if sha(v['path'])!=v['sha256']:raise ValueError('partition resource identity differs')
 def get_inputs(self):return self._info('inputs')
 def get_outputs(self):return self._info('outputs')
 def _info(self,kind):return [SimpleNamespace(name=r['name'],shape=r['shape'],type='tensor('+('float' if r['dtype']=='float32' else r['dtype'])+')') for r in self._maps[kind]]
 def _native(self,row):
  a=self.manifest['assets'];abi=copy.deepcopy(row['abi']);part=self.manifest['plan']['parts'][row['index']]
  for kind in ('inputs','outputs'):
   sources={v['name']:v['source_name'] for v in part[kind]}
   for value in abi[kind]:
    source=sources[value['name']]
    public_names={v['name'] for v in self._maps[kind]}
    if source.endswith('/qnn_scalar_internal') and source[:-len('/qnn_scalar_internal')] in public_names:source=source[:-len('/qnn_scalar_internal')]
    if value['dtype']=='int64' and source in RANGES:value['integer_range']=RANGES[source]
  return NativeSession(a['bridge']['path'],row['library'],a['backend_lib']['path'],abi,dict(bridge=a['bridge']['sha256'],model_lib=row['library_sha256'],backend_lib=a['backend_lib']['sha256']))
 def load_all(self):
  with self._lock:
   if self._closed:raise RuntimeError('partition session closed')
   descriptions=[]
   for row in self.manifest['parts']:
    with self._native(row) as session:descriptions.append(dict(index=row['index'],actual=session.native_abi))
   self.native_abi['parts']=descriptions
 def run(self,names,feed):
  with self._lock:
   if self._closed:raise RuntimeError('partition session closed')
   if set(feed)!={r['name'] for r in self._maps['inputs']}:raise ValueError('partition public input names differ')
   values=dict(self._shared_values)
   for row in self._maps['inputs']:
    v=feed[row['name']]
    if not isinstance(v,np.ndarray) or list(v.shape)!=row['shape'] or v.dtype!=np.dtype(row['dtype']):raise ValueError('partition logical input ABI differs: '+row['name'])
    if np.issubdtype(v.dtype,np.floating) and not all_finite(v):raise ValueError('nonfinite partition public input')
    if 'integer_range' in row and ((v<row['integer_range'][0]).any() or (v>row['integer_range'][1]).any()):raise ValueError('partition public integer input exceeds exact range')
    values[row['name']]=v.reshape(row['parent_shape'])
   observations=[];descriptions=[];self.native_abi['parts']=[]
   for row,part in zip(self.manifest['parts'],self.manifest['plan']['parts']):
    started=time.monotonic();save(Path.cwd()/'partition_execution.json',dict(stage='part',active_part=part['index'],total_parts=len(self.manifest['parts']),completed_parts=observations));inputs={r['name']:values[r['source_name']] for r in part['inputs']}
    with self._native(row) as session:
     loaded=time.monotonic()-started;descriptions.append(dict(index=row['index'],actual=session.native_abi));self.native_abi['parts']=descriptions
     try:out=session.run(None,inputs)
     except Exception as error:
      failure=dict(status='failed',part=part['index'],error_type=type(error).__name__,error=str(error),scope='Actual rejected native execution and incoming cut retained; strict execution/finite guards and public Host state remain unchanged.')
      if isinstance(error,(NonfiniteOutputError,IntegerOutputRangeError)):
       failure.update(output=error.output_name,shape=list(error.output_value.shape),dtype=str(error.output_value.dtype))
       if isinstance(error,NonfiniteOutputError):failure['nonfinite_count']=int((~np.isfinite(error.output_value)).sum())
       else:failure.update(integer_range=error.integer_range,invalid_count=int(((error.output_value<error.integer_range[0])|(error.output_value>error.integer_range[1])).sum()))
      try:
       input_path=Path.cwd()/'rejected_part.inputs.npz';np.savez(input_path,**inputs);failure.update(inputs=str(input_path),inputs_sha256=sha(input_path))
       if isinstance(error,(NonfiniteOutputError,IntegerOutputRangeError)):
        output_path=Path.cwd()/'rejected_part.output.npz';np.savez(output_path,**{error.output_name:error.output_value});failure.update(outputs=str(output_path),outputs_sha256=sha(output_path))
      except BaseException as diagnostic_error:failure['capture_error']=str(diagnostic_error)
      save(Path.cwd()/'rejected_part.json',failure);raise
    for r,v in zip(part['outputs'],out):
     if v.dtype!=np.dtype(r['dtype']) or list(v.shape)!=r['shape']:raise ValueError('partition actual cut output differs')
     values[r['source_name']]=v
    for value in part['drop_after']:values.pop(value,None)
    observations.append(dict(index=part['index'],load_seconds=loaded,total_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
   result={}
   for row in self._maps['outputs']:
    v=values[row['name']].reshape(row['shape'])
    if v.dtype!=np.dtype(row['dtype']) or (np.issubdtype(v.dtype,np.floating) and not all_finite(v)):raise ValueError('partition public output ABI/nonfinite differs')
    if 'integer_range' in row and ((v<row['integer_range'][0]).any() or (v>row['integer_range'][1]).any()):raise ValueError('partition public integer output exceeds exact range')
    result[row['name']]=v
   self.last_execution=observations;self.native_abi['parts']=descriptions;save(Path.cwd()/'partition_execution.json',dict(stage='complete',completed_parts=observations));return [result[n] for n in (names if names is not None else [r['name'] for r in self._maps['outputs']])]
 def close(self):
  with self._lock:self._closed=True;self._shared_values.clear()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
