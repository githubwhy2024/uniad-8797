#!/usr/bin/env python3
"""Package one actual constructed model as a pinned, relocatable CPU component."""
import argparse,json,os,platform,shutil,subprocess,sys
from pathlib import Path
import numpy as np,onnx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,SDK,sha,save
from resources import terminal
from session import DTYPES

def main():
 p=argparse.ArgumentParser();p.add_argument('--compile-run',type=Path,required=True);p.add_argument('--bridge-build',type=Path,required=True);a=p.parse_args();compile_result=terminal(a.compile_run);terminal(a.bridge_build);b=json.loads((a.compile_run/'model_build.json').read_text());bridge=json.loads((a.bridge_build/'bridge_build.json').read_text());convert=Path(b['resources']['model.cpp']['path']).parent;converted=terminal(convert);source=json.loads((convert/'source_identity.json').read_text())
 if sha(convert/'result.json')!=b['converter_result_sha256'] or source['model_sha256']!=b['source_model_sha256'] or sha(source['model'])!=source['model_sha256']:raise ValueError('component compiler/source binding differs')
 # This first component control deliberately accepts only the tiny verified label graph.
 model=onnx.load(source['model']);expected=['labels','modded','joined','scattered']
 if [v.name for v in model.graph.input]!=['scores'] or [v.name for v in model.graph.output]!=expected or len(model.graph.node)>12:raise ValueError('constructed component control graph differs')
 net=json.loads((convert/'model_net.json').read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1',inputs=[],outputs=[])
 for kind in ('inputs','outputs'):
  for vi in getattr(model.graph,'input' if kind=='inputs' else 'output'):
   shape=[v.dim_value for v in vi.type.tensor_type.shape.dim];dtype=str(onnx.helper.tensor_dtype_to_np_dtype(vi.type.tensor_type.elem_type));actual=net[vi.name]
   if DTYPES[actual['data_type']]!=dtype or [v for v in shape if v!=1]!=[v for v in actual['dims'] if v!=1]:raise ValueError('constructed component physical ABI differs')
   row=dict(name=vi.name,native_name=vi.name,shape=shape,native_shape=actual['dims'],dtype=dtype)
   if shape!=actual['dims']:row['wire_view']='singleton_axes'
   if dtype=='int64':row['integer_range']=[-2**31,2**31-1]
   abi[kind].append(row)
 root=Path.cwd();bundle=root/'bundle'
 if root.resolve()!=root:raise ValueError('ordinary package run required')
 bundle.mkdir();(bundle/'lib').mkdir();(bundle/'runtime').mkdir();files={}
 def copy(role,source,dest,expected=None):
  source=Path(source);digest=sha(source)
  if expected is not None and digest!=expected:raise ValueError('component source bytes differ: '+role)
  target=bundle/dest;shutil.copyfile(source,target)
  if target.resolve()!=target or not target.is_file() or sha(target)!=digest:raise ValueError('component destination is not an exact ordinary copy')
  files[role]=dict(path=dest,sha256=digest,bytes=target.stat().st_size)
 copy('model_lib',b['library'],'lib/model.so',b['library_sha256']);copy('bridge',bridge['library'],'lib/bridge.so',bridge['library_sha256']);copy('backend_lib',SDK/'lib/x86_64-linux-clang/libQnnCpu.so','lib/libQnnCpu.so',b['cpu_backend_sha256']);copy('session_adapter',ROOT/'onnx/qnn/session.py','runtime/session.py');copy('loader',ROOT/'onnx/qnn/native_bundle.py','runtime/native_bundle.py')
 dependencies={};ldd=subprocess.run(['/usr/bin/ldd',str(SDK/'lib/x86_64-linux-clang/libQnnCpu.so')],check=True,capture_output=True,text=True).stdout
 for line in ldd.splitlines():
  fields=line.split()
  if len(fields)>=3 and fields[1]=='=>' and fields[2].startswith('/'):dependencies[fields[0]]=fields[2]
 for role,name in [('libc++','libc++.so.1'),('libc++abi','libc++abi.so.1'),('libunwind','libunwind.so.1')]:
  if name not in dependencies:raise ValueError('CPU C++ runtime dependency unresolved')
  copy(role,dependencies[name],'lib/'+name)
 os_dependencies={name:dict(sha256=sha(path),bytes=Path(path).stat().st_size) for name,path in dependencies.items() if name not in ('libc++.so.1','libc++abi.so.1','libunwind.so.1')}
 manifest=dict(schema='qnn-native-component-bundle-v1',purpose='constructed_native_control',backend=dict(type='QNN_CPU',precision='float32',target='x86_64-linux-clang'),runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine()),files=files,abi=abi,os_dependencies=os_dependencies,lineage=dict(source_model_sha256=source['model_sha256'],compiler_result_sha256=sha(a.compile_run/'result.json'),converter_result_sha256=sha(convert/'result.json'),bridge_result_sha256=sha(a.bridge_build/'result.json')),scope='Constructed native CPU component only; production planning/Host/state/task acceptance absent.')
 save(bundle/'manifest.json',manifest);save(root/'package_control.json',dict(status='pass',bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),files=len(files),os_dependencies=os_dependencies,scope='Exact ordinary relative-path package creation only; outside-cwd API and negative pin/path/ABI controls required separately.'))
if __name__=='__main__':main()
