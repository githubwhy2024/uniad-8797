"""Pinned, relative-path native CPU component bundles, independent of SDK checkout."""
import argparse,json,platform,sys
from pathlib import Path,PurePosixPath
import numpy as np
import session as native_session
from session import NativeSession,sha

def ordinary(root,name):
 if not isinstance(name,str) or not name:raise ValueError('bundle file path must be a nonempty relative string')
 rel=PurePosixPath(name)
 if rel.is_absolute() or rel.as_posix()!=name or any(v in ('..','.') for v in rel.parts) or '\\' in name:raise ValueError('bundle file path must be canonical and relative')
 path=root/Path(*rel.parts)
 if path.resolve()!=path or not path.is_file() or not path.is_relative_to(root):raise ValueError('bundle file must be ordinary and contained')
 return path

def load_native_bundle(root,expected_manifest_sha256):
 root=Path(root).absolute()
 if root.resolve()!=root or not root.is_dir():raise ValueError('bundle root must be an ordinary directory')
 if not isinstance(expected_manifest_sha256,str) or len(expected_manifest_sha256)!=64 or any(v not in '0123456789abcdef' for v in expected_manifest_sha256):raise ValueError('an explicit SHA256 bundle pin is required')
 path=ordinary(root,'manifest.json')
 if sha(path)!=expected_manifest_sha256:raise ValueError('native bundle pin differs')
 manifest=json.loads(path.read_text())
 if manifest['schema']!='qnn-native-component-bundle-v1' or manifest['backend']!=dict(type='QNN_CPU',precision='float32',target='x86_64-linux-clang') or manifest['purpose']!='constructed_native_control':raise ValueError('unsupported native component bundle contract')
 versions=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine())
 if manifest['runtime_versions']!=versions:raise ValueError('native bundle runtime dependency versions differ')
 seen=set();assets={}
 for role,row in manifest['files'].items():
  file=ordinary(root,row['path'])
  if row['path'] in seen or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:raise ValueError('native bundle file binding differs: '+role)
  seen.add(row['path']);assets[role]=str(file)
 if not {'bridge','model_lib','backend_lib','session_adapter','loader','libc++','libc++abi','libunwind'}<=set(assets):raise ValueError('native bundle runtime closure is incomplete')
 for role,name in [('session_adapter','session.py'),('loader','native_bundle.py')]:
  if sha(native_session.__file__ if role=='session_adapter' else __file__)!=manifest['files'][role]['sha256']:raise ValueError('imported bundle runtime differs')
 return manifest,assets

def session_from_bundle(root,expected_manifest_sha256):
 manifest,assets=load_native_bundle(root,expected_manifest_sha256)
 return NativeSession(assets['bridge'],assets['model_lib'],assets['backend_lib'],manifest['abi'],{role:manifest['files'][role]['sha256'] for role in ('bridge','model_lib','backend_lib')})

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--outputs',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 manifest,assets=load_native_bundle(a.bundle,a.manifest_sha256)
 for output in (a.outputs,a.report):
  output=output.absolute()
  if output.is_relative_to(a.bundle.absolute()) or output.exists() or output.resolve()!=output:raise ValueError('component execution evidence needs a new ordinary path outside the bundle')
 with np.load(a.inputs,allow_pickle=False) as archive:feed={k:archive[k].copy() for k in archive.files}
 before={n:np.ascontiguousarray(v).tobytes() for n,v in feed.items()}
 with session_from_bundle(a.bundle,a.manifest_sha256) as session:
  actual=dict(zip([v.name for v in session.get_outputs()],session.run(None,feed)));abi=session.native_abi
  mapped={}
  for line in Path('/proc/self/maps').read_text().splitlines():
   fields=line.split()
   if len(fields)>5 and fields[-1].startswith('/'):
    name=Path(fields[-1]).name
    if name in ('libc++.so.1','libc++abi.so.1','libunwind.so.1'):mapped[name]=fields[-1]
  closure={role:dict(path=mapped.get(Path(assets[role]).name),from_bundle=mapped.get(Path(assets[role]).name)==assets[role]) for role in ('libc++','libc++abi','libunwind')}
 np.savez(a.outputs,**actual)
 unchanged=all(np.ascontiguousarray(feed[n]).tobytes()==v for n,v in before.items());closed=all(v['from_bundle'] for v in closure.values())
 report=dict(status='pass' if unchanged and closed else 'failed',execution_status='complete',manifest_sha256=a.manifest_sha256,inputs_sha256=sha(a.inputs),outputs_sha256=sha(a.outputs),inputs_unmodified=unchanged,native_abi=abi,cpp_dependency_closure=closure,scope='Actual relocated CPU native component execution only; no UniAD/Host/recurrence/task/planning package or board acceptance.')
 a.report.write_text(json.dumps(report,indent=2)+'\n')
 if not report['inputs_unmodified'] or not all(v['from_bundle'] for v in closure.values()):raise ValueError('component runtime dependency closure or input immutability failed')
if __name__=='__main__':main()
