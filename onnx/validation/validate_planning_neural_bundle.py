#!/usr/bin/env python3
"""Execute relocated planning25 own-state prefix and new-process resume."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qnn'))
from resources import terminal
from tool_run import ROOT, save, sha, process_identity
sys.path.insert(0, str(ROOT / 'onnx/fixed'))
from host import prepare_fixed_frame_feed
from planning_identity import verify_scored_bundle


OBSERVER = '''import copy,json,resource,runpy,sys,time
from pathlib import Path
sys.argv=sys.argv[1:]
sys.path.insert(0,str(Path(sys.argv[0]).parent))
import planning_bundle as loaded
totals={};frames=[]
def timed(name,function):
 def invoke(*args,**kwargs):
  start=time.monotonic()
  try:return function(*args,**kwargs)
  finally:
   row=totals.setdefault(name,dict(calls=0,seconds=0.));row['calls']+=1;row['seconds']+=time.monotonic()-start
 return invoke
native_class=loaded.partition_runtime.NativeSession
original_init=native_class.__init__
def native_init(self,*args,**kwargs):
 timed('native_graph_load_and_abi',original_init)(self,*args,**kwargs)
 self._lib.q4_qnn_execute=timed('bridge_execute_including_native_boundary',self._lib.q4_qnn_execute)
native_class.__init__=native_init
native_class.run=timed('native_call_with_python_boundary',native_class.run)
partition_class=getattr(loaded.partition_runtime,'ResourcePartitionSession',loaded.partition_runtime.PartitionSession)
partition_class.run=timed('partition_inference_with_lazy_load',partition_class.run)
loaded.host.FixedStateTransaction.advance=timed('state_advance_including_partition',loaded.host.FixedStateTransaction.advance)
loaded.host.FixedStateTransaction.save_checkpoint=timed('checkpoint_atomic_write',loaded.host.FixedStateTransaction.save_checkpoint)
loaded.host.optimize_planning=timed('planning_postprocess_including_solver',loaded.host.optimize_planning)
loaded.np.lib.npyio.NpzFile.__getitem__=timed('npz_array_read_and_decompress',loaded.np.lib.npyio.NpzFile.__getitem__)
loaded.np.savez=timed('npz_uncompressed_write',loaded.np.savez)
loaded.np.savez_compressed=timed('npz_compressed_write',loaded.np.savez_compressed)
original_process=loaded.host.PlanningStream.process
def process(self,*args,**kwargs):
 before=copy.deepcopy(totals);start=time.monotonic()
 result=original_process(self,*args,**kwargs)
 delta={key:dict(calls=value['calls']-before.get(key,{}).get('calls',0),seconds=value['seconds']-before.get(key,{}).get('seconds',0.)) for key,value in totals.items()}
 def seconds(key):return delta.get(key,{}).get('seconds',0.)
 frames.append(dict(frame_id=kwargs.get('frame_id'),status=result['status'],process_seconds=time.monotonic()-start,
  measured_calls=delta,state_contract_excluding_qnn_seconds=max(0.,seconds('state_advance_including_partition')-seconds('partition_inference_with_lazy_load')),
  python_boundary_transport_validation_seconds=max(0.,seconds('native_call_with_python_boundary')-seconds('bridge_execute_including_native_boundary')),
  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 return result
loaded.host.PlanningStream.process=process
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 mapped={Path(line.split()[-1]).name:line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if len(line.split())>5 and line.split()[-1].startswith('/')}
 Path('runtime_dependencies.json').write_text(json.dumps(dict(mapped={name:mapped.get(name) for name in ['libc++.so.1','libc++abi.so.1','libunwind.so.1']},ml_modules=[name for name in ['onnx','onnxruntime','torch','mmcv','mmdet'] if name in sys.modules]),indent=2)+'\\n')
 Path('host_measurements.json').write_text(json.dumps(dict(totals=totals,frames=frames,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Observed x86 CPU candidate calls; inclusive categories overlap. Bridge execute includes its native boundary handling, not a pure device kernel profiler. NPZ array reads include decompression; checkpoint time includes its compressed write. Instrumentation does not alter arguments/results.'),indent=2)+'\\n')
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-run', type=Path, required=True)
    parser.add_argument('--relocation-run', type=Path, required=True)
    parser.add_argument('--short-run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    package_tool, relocation_tool, short_tool = [terminal(path) for path in (args.package_run, args.relocation_run, args.short_run)]
    record = json.loads((args.package_run / 'planning_package.json').read_text())
    control = json.loads((args.relocation_run / 'planning_relocation_control.json').read_text())
    short = json.loads((args.short_run / 'neural_result.json').read_text())
    for tool, path, file in [(package_tool,args.package_run,'planning_package.json'),(relocation_tool,args.relocation_run,'planning_relocation_control.json'),(short_tool,args.short_run,'neural_result.json')]:
        if sha(path/file) != tool['artifacts'][file]['sha256']:
            raise ValueError('planning portable evidence binding differs')
    if control['status'] != 'pass' or control['manifest_sha256'] != record['manifest_sha256'] or short['status'] != 'pass' or short['stage'] != 'six_real_frames' or len(short['completed_frames']) != 6:
        raise ValueError('planning resource/short sequence prerequisite incomplete')
    source, pin = Path(record['bundle']), record['manifest_sha256']
    if sha(source/'manifest.json') != pin or sha(args.inputs) != args.inputs_sha256:
        raise ValueError('planning candidate/input manifest binding differs')
    manifest = json.loads((source/'manifest.json').read_text())
    profile = json.loads((args.short_run/'profile.json').read_text())
    if manifest['lineage']['native_model_sha256'] != profile['assets']['native_model']['sha256'] or manifest['profile']['abi'] != profile['abi'] or manifest['host_policy'] != profile['host_policy']:
        raise ValueError('portable candidate differs from accepted source profile')
    if [row['library_sha256'] for row in manifest['profile']['parts']] != [row['library_sha256'] for row in profile['parts']]:
        raise ValueError('portable actual cut library bytes differ')
    for role in ('bridge','backend_lib','host','state_contract','initial_state','collision_optimizer','session_adapter'):
        source_role = 'adapter' if role == 'session_adapter' else role
        if manifest['files'][role]['sha256'] != profile['assets'][source_role]['sha256']:
            raise ValueError('portable runtime source differs: '+role)
    if manifest.get('transport_policy') != profile.get('transport_policy'):
        raise ValueError('portable candidate transport differs from accepted source profile')
    if profile.get('transport_policy') is not None:
        role = 'borrowed_input_adapter'
        if role not in manifest['files'] or manifest['files'][role]['sha256'] != profile['assets'][role]['sha256']:
            raise ValueError('portable borrowed adapter source differs')
    binding = verify_scored_bundle(source, manifest, profile, args.short_run)
    inputs = json.loads(args.inputs.read_text())
    if inputs['status'] != 'pass' or len(inputs['frames']) != 6 or [row['token'] for row in inputs['frames']] != [row['token'] for row in short['completed_frames']]:
        raise ValueError('portable six-frame token sequence differs')
    external = Path(tempfile.mkdtemp(prefix='q4-planning-neural-relocation-'))
    bundle, outside = external/'candidate', external/'outside'
    shutil.copytree(source,bundle,symlinks=False);outside.mkdir()
    requests, previous, checks = [], {}, {}
    for index,row in enumerate(inputs['frames']):
        if sha(row['inputs']) != row['inputs_sha256']:
            raise ValueError('portable original synchronized frame differs')
        with np.load(row['inputs'],allow_pickle=False) as archive:
            feed={name:archive[name].copy() for name in archive.files}
        metadata=row['metadata'];bus=feed['can_bus'][0].copy()
        bus[:3]=np.asarray(metadata['position'],np.float32);bus[-1]=metadata['angle']
        frame=dict(scene_token=row['scene_token'],timestamp=metadata['timestamp'],command=int(feed['command'][0]),
            image=feed['img'],can_bus_absolute=bus,l2g_r=np.asarray(metadata['rotation'],np.float32),
            l2g_t=np.asarray(metadata['translation'],np.float32),lidar2img=feed['lidar2img'],img_shape=feed['img_shape'])
        prepared,context,reset=prepare_fixed_frame_feed(previous,can_bus_mode=manifest['host_policy']['can_bus_mode'],**frame)
        checks['frame'+str(index)+'_prepared_feed_exact']=all(np.array_equal(value,feed[name]) for name,value in prepared.items()) and context==metadata and reset==row['new_scene']
        if not checks['frame'+str(index)+'_prepared_feed_exact']:
            raise ValueError('portable absolute metadata reconstruction changed a prepared input')
        tensor_path=outside/('frame'+str(index)+'.tensors.npz')
        np.savez(tensor_path,**{key:value for key,value in frame.items() if key not in ('scene_token','timestamp','command')})
        requests.append(dict(frame_id=row['token'],scene_token=row['scene_token'],timestamp=metadata['timestamp'],command=frame['command'],tensors=str(tensor_path),tensors_sha256=sha(tensor_path)))
        previous=copy.deepcopy(metadata)
    if args.prepare_only:
        save(Path.cwd()/'planning_neural_preparation.json',dict(status='pass',checks=checks,frames=requests,
            manifest_sha256=pin,external_root=str(external),scope='Copied candidate and exact six-frame caller input preparation only; no neural execution.'))
        return
    env=os.environ.copy()
    for name in ('QNN_SDK_ROOT','QAIRT_SDK_ROOT','QNN_SDK_PATH','QAIRT_SDK','QNN_CONVERTER_ENV'):env.pop(name,None)
    env.update(PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(bundle/'lib'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    reports=[]
    for stage,rows in [('prefix',requests[:2]),('resume',requests[2:])]:
        cwd=outside/stage;cwd.mkdir();request=cwd/'request.json';save(request,dict(schema='qnn-planning-frame-request-v1',frames=rows));output=cwd/'evidence'
        command=[sys.executable,str(bundle/'runtime/planning_bundle.py'),'--bundle',str(bundle),'--manifest-sha256',pin,'--allow-candidate-validation','--request',str(request),'--request-sha256',sha(request),'--output-dir',str(output),'--max-state-gap-seconds','10','--max-consecutive-failures','1']
        if stage=='resume':
            checkpoint=reports[0]['frames'][-1];command+=['--resume-checkpoint',checkpoint['checkpoint'],'--resume-checkpoint-sha256',checkpoint['checkpoint_sha256']]
        command=[sys.executable,'-c',OBSERVER]+command[1:]
        with (cwd/'worker.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
            save(Path.cwd()/'planning_neural_progress.json',dict(stage=stage,child=process_identity(child.pid),command=command,external_root=str(external),manifest_sha256=pin))
            code=child.wait()
        # Preserve ordinary evidence under the owning Q4 run even though actual
        # execution occurred outside the project. Bundle libraries already have
        # their pinned, ordinary source package; only execution evidence copies.
        owned=Path.cwd()/'relocation_evidence'/stage
        shutil.copytree(cwd,owned,symlinks=False)
        if code != 0:
            raise ValueError('portable '+stage+' failed; retained '+str(cwd/'worker.log'))
        dependencies=json.loads((cwd/'runtime_dependencies.json').read_text())
        checks[stage+'_copied_cpp_dependency_closure']=all(dependencies['mapped'][name]==str(bundle/'lib'/name) for name in ('libc++.so.1','libc++abi.so.1','libunwind.so.1'))
        checks[stage+'_no_developer_ml_modules']=dependencies['ml_modules']==[]
        measurements=json.loads((cwd/'host_measurements.json').read_text())
        checks[stage+'_actual_host_measurements']=len(measurements['frames'])==len(rows) and measurements['totals']['bridge_execute_including_native_boundary']['calls']==len(rows)*len(manifest['profile']['parts'])
        report=json.loads((output/'planning_result.json').read_text())
        if report['status']!='pass' or report['manifest_sha256']!=pin or len(report['frames'])!=len(rows):raise ValueError('portable frame report differs')
        reports.append(report)
    frames=reports[0]['frames']+reports[1]['frames'];checks['new_process_checkpoint_restore']=reports[1]['starting_state_sha256']==reports[0]['frames'][-1]['state_sha256']
    differences=[];prior=reports[0]['starting_state_sha256']
    for index,(actual,expected,row) in enumerate(zip(frames,short['completed_frames'],inputs['frames'])):
        checks['frame'+str(index)+'_own_state_chain']=actual['incoming_state_sha256']==prior;prior=actual['state_sha256']
        checks['frame'+str(index)+'_identity_validity']=actual['frame_id']==row['token'] and actual['scene_token']==row['scene_token'] and actual['timestamp']==row['metadata']['timestamp'] and actual['valid'] is True and actual['state_committed'] is True
        for key in ('checkpoint','final_plan'):
            path=actual['checkpoint'] if key=='checkpoint' else actual['final_plan']
            if sha(path)!=actual[key+'_sha256']:raise ValueError('portable committed artifact differs')
        with np.load(actual['final_plan'],allow_pickle=False) as arc:plan=arc['planning_final'].copy()
        with np.load(expected['final_plan_path'],allow_pickle=False) as arc:reference=arc['planning_final'].copy()
        checks['frame'+str(index)+'_final_contract']=bool(plan.shape==(1,6,2) and plan.dtype==np.float32 and np.isfinite(plan).all())
        differences.append(dict(token=row['token'],max_abs=float(np.max(np.abs(plan-reference))),mean_abs=float(np.mean(np.abs(plan-reference)))))
    checks['actual_solver_branch']=any(row['planning_info']['solver_ran'] for row in frames)
    save(Path.cwd()/'planning_neural_relocation.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,frames=frames,final_plan_differences=differences,
        execution_identity=binding,manifest_sha256=pin,source_short_result_sha256=sha(args.short_run/'neural_result.json'),external_root=str(external),
        host_measurements={stage:dict(path=str(Path.cwd()/'relocation_evidence'/stage/'host_measurements.json'),sha256=sha(Path.cwd()/'relocation_evidence'/stage/'host_measurements.json')) for stage in ('prefix','resume')},
        preserved_evidence={str(path.relative_to(Path.cwd())):dict(sha256=sha(path),bytes=path.stat().st_size) for path in (Path.cwd()/'relocation_evidence').rglob('*') if path.is_file()},
        scope='Actual relocated planning25 six-frame own-state/new-process checkpoint/Host/solver only; mini404 task/release/board acceptance separate.'))
    if not all(checks.values()):raise ValueError('portable six-frame recurrence failed')


if __name__=='__main__':main()
