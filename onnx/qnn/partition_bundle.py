"""Pinned constructed partition CPU components, with no developer/SDK imports."""
import argparse,copy,json,platform,sys
from pathlib import Path
import numpy as np
import session as native_session
from session import sha
import native_bundle as file_loader
from native_bundle import ordinary


def load_partition_bundle(root,pin):
 root=Path(root).absolute()
 if root.resolve()!=root or not root.is_dir():raise ValueError('ordinary component bundle root required')
 if not isinstance(pin,str) or len(pin)!=64 or any(v not in '0123456789abcdef' for v in pin):raise ValueError('explicit component manifest SHA256 required')
 path=ordinary(root,'manifest.json')
 if sha(path)!=pin:raise ValueError('partition component manifest pin differs')
 m=json.loads(path.read_text())
 if m['schema']!='qnn-partition-component-bundle-v1' or m['purpose']!='constructed_partition_control' or m['backend']!=dict(type='QNN_CPU',precision='float32',target='x86_64-linux-clang'):raise ValueError('unsupported partition component contract')
 if m['runtime_versions']!=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine()):raise ValueError('partition runtime version differs')
 assets={};seen=set()
 for role,row in m['files'].items():
  p=ordinary(root,row['path'])
  if row['path'] in seen or sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('partition component file binding differs: '+role)
  seen.add(row['path']);assets[role]=dict(path=str(p),sha256=row['sha256'],bytes=row['bytes'])
 import partition_runtime
 modules={'session_adapter':native_session.__file__,'ordinary_loader':file_loader.__file__,'partition_adapter':partition_runtime.__file__,'loader':__file__}
 required=set(modules)|{'bridge','backend_lib','libc++','libc++abi','libunwind'}
 if not required<=set(assets):raise ValueError('partition component runtime closure incomplete')
 for role,path in modules.items():
  if Path(path).resolve()!=ordinary(root,m['files'][role]['path']) or sha(path)!=assets[role]['sha256']:raise ValueError('imported partition component runtime differs: '+role)
 profile=copy.deepcopy(m['profile']);profile['assets']=assets
 if len(profile['parts'])!=len(profile['plan']['parts']) or [v['index'] for v in profile['parts']]!=list(range(len(profile['parts']))):raise ValueError('partition component coverage/order differs')
 for row,part in zip(profile['parts'],profile['plan']['parts']):
  role='part_'+str(row['index'])
  if role not in assets or row['library']!=m['files'][role]['path'] or row['library_sha256']!=assets[role]['sha256'] or row['index']!=part['index']:raise ValueError('partition component library binding differs')
  if any([v['name'] for v in row['abi'][kind]]!=[v['name'] for v in part[kind]] for kind in ['inputs','outputs']):raise ValueError('partition component cut ABI differs')
  row['library']=assets[role]['path']
 return m,profile,partition_runtime


def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--outputs',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--control-frames',type=int,default=1);p.add_argument('--control-step',type=int,default=64);a=p.parse_args();m,profile,runtime=load_partition_bundle(a.bundle,a.manifest_sha256)
 for path in [a.outputs,a.report]:
  if path.exists() or path.absolute().resolve()!=path.absolute() or path.absolute().is_relative_to(a.bundle.absolute()):raise ValueError('new ordinary component evidence path outside bundle required')
 if not 1<=a.control_frames<=6:raise ValueError('constructed recurrent scope is one to six frames')
 with np.load(a.inputs,allow_pickle=False) as arc:feed={k:arc[k].copy() for k in arc.files}
 if a.control_frames>1 and (set(feed)!= {'state','delta'} or feed['state'].shape!=(512,512) or feed['delta'].shape!=(512,512) or not np.array_equal(feed['state'],np.zeros((512,512),np.float32)) or not np.array_equal(feed['delta'],np.ones((512,512),np.float32))):raise ValueError('constructed recurrence input contract differs')
 checks={};rows=[]
 with runtime.PartitionSession(profile) as session:
  session.load_all();checks['all_actual_parts_loaded']=len(session.native_abi['parts'])==len(profile['parts'])
  for index in range(a.control_frames):
   before={k:v.copy() for k,v in feed.items()};actual=dict(zip([v.name for v in session.get_outputs()],session.run(None,feed)));checks['frame'+str(index)+'_inputs_unchanged']=all(np.array_equal(feed[k],v) for k,v in before.items())
   if a.control_frames>1:
    checks['frame'+str(index)+'_actual_control_routing']=len(actual)==1 and np.array_equal(next(iter(actual.values())),np.full((512,512),a.control_step*(index+1),np.float32));feed['state']=next(iter(actual.values())).copy()
   rows.append(dict(index=index,actual_parts=len(session.last_execution)))
  mapped={}
  for line in Path('/proc/self/maps').read_text().splitlines():
   fields=line.split()
   if len(fields)>5 and fields[-1].startswith('/'):mapped[Path(fields[-1]).name]=fields[-1]
  closure={role:mapped.get(Path(profile['assets'][role]['path']).name)==profile['assets'][role]['path'] for role in ['libc++','libc++abi','libunwind']};checks['copied_cpp_dependency_closure']=all(closure.values())
 np.savez(a.outputs,**actual);report=dict(status='pass' if all(checks.values()) else 'failed',execution_status='complete',checks=checks,frames=rows,manifest_sha256=a.manifest_sha256,inputs_sha256=sha(a.inputs),outputs_sha256=sha(a.outputs),cpp_dependency_closure=closure,scope='Actual relocated constructed many-part CPU component only; no UniAD Host/learned state/planning/task or board acceptance.');a.report.write_text(json.dumps(report,indent=2)+'\n')
 if not all(checks.values()):raise ValueError('relocated partition component control failed')
if __name__=='__main__':main()
