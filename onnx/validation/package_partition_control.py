#!/usr/bin/env python3
"""Export unchanged production partition session AST into a standalone control package."""
import argparse,ast,copy,json,platform,shutil,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,SDK,sha,save
from resources import terminal
from partition_session import bind_partitions


def main():
 p=argparse.ArgumentParser();p.add_argument('--build-run',type=Path,required=True);p.add_argument('--bridge-build',type=Path,required=True);p.add_argument('--logical-model',type=Path,required=True);p.add_argument('--logical-model-sha256',required=True);a=p.parse_args();root=Path.cwd();profile=bind_partitions(a.build_run,a.bridge_build,a.logical_model,a.logical_model_sha256)
 if [v['name'] for v in profile['abi']['inputs']]!=['state','delta'] or len(profile['abi']['outputs'])!=1 or any(v['shape']!=[512,512] or v['dtype']!='float32' for v in profile['abi']['inputs']+profile['abi']['outputs']):raise ValueError('constructed partition control graph differs')
 bundle=root/'bundle';bundle.mkdir();(bundle/'lib').mkdir();(bundle/'runtime').mkdir();files={}
 def copy_file(role,source,dest,expected=None):
  path=Path(source);digest=sha(path)
  if expected and digest!=expected:raise ValueError('partition package source differs: '+role)
  target=bundle/dest;shutil.copyfile(path,target)
  if target.resolve()!=target or sha(target)!=digest:raise ValueError('partition package copy differs')
  files[role]=dict(path=dest,sha256=digest,bytes=target.stat().st_size)
 for role,name in [('session_adapter','session.py'),('ordinary_loader','native_bundle.py'),('loader','partition_bundle.py')]:copy_file(role,ROOT/'onnx/qnn'/name,'runtime/'+name)
 for role,name in [('bridge','bridge.so'),('backend_lib','libQnnCpu.so')]:copy_file(role,profile['assets'][role]['path'],'lib/'+name,profile['assets'][role]['sha256'])
 parts=[]
 for row in profile['parts']:
  value=copy.deepcopy(row);role='part_'+str(row['index']);dest='lib/part'+str(row['index']).zfill(3)+'.so';copy_file(role,row['library'],dest,row['library_sha256']);parts.append(dict(index=row['index'],library=dest,library_sha256=row['library_sha256'],abi=row['abi']))
 # Copy the complete class and atomic journal function unchanged, omitting only
 # development binders/imports. No separate reimplementation of execution math.
 source=ROOT/'onnx/qnn/partition_session.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PartitionSession');tools=ast.parse((ROOT/'onnx/qnn/tool_run.py').read_text());savefn=next(n for n in tools.body if isinstance(n,ast.FunctionDef) and n.name=='save');rs=ast.parse((ROOT/'onnx/qnn/resources.py').read_text());ranges=next(n for n in rs.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RANGES' for t in n.targets));imports=ast.parse('import copy,json,math,os,resource,tempfile,threading,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport numpy as np\nfrom session import NativeSession,NonfiniteOutputError,IntegerOutputRangeError,sha\n').body;module=ast.fix_missing_locations(ast.Module(body=imports+[copy.deepcopy(ranges),copy.deepcopy(savefn),copy.deepcopy(cls)],type_ignores=[]));generated=bundle/'runtime/partition_runtime.py';generated.write_text(ast.unparse(module)+'\n');parsed=ast.parse(generated.read_text());restored=next(n for n in parsed.body if isinstance(n,ast.ClassDef));
 if ast.dump(restored)!=ast.dump(cls):raise ValueError('standalone partition class AST changed')
 files['partition_adapter']=dict(path='runtime/partition_runtime.py',sha256=sha(generated),bytes=generated.stat().st_size)
 ldd=subprocess.check_output(['/usr/bin/ldd',profile['assets']['backend_lib']['path']],text=True);deps={}
 for line in ldd.splitlines():
  v=line.split()
  if len(v)>=3 and v[1]=='=>' and v[2].startswith('/'):deps[v[0]]=v[2]
 for role,name in [('libc++','libc++.so.1'),('libc++abi','libc++abi.so.1'),('libunwind','libunwind.so.1')]:copy_file(role,deps[name],'lib/'+name)
 cut=[{k:copy.deepcopy(v[k]) for k in ['index','inputs','outputs','drop_after']} for v in profile['plan']['parts']]
 runtime_profile=dict(abi=profile['abi'],parts=parts,plan=dict(parts=cut))
 import hashlib
 lineage=dict(build_result_sha256=sha(a.build_run/'result.json'),build_manifest_sha256=sha(a.build_run/'partition_build.json'),bridge_result_sha256=sha(a.bridge_build/'result.json'),logical_model_sha256=a.logical_model_sha256,partition_source_sha256=sha(source),partition_class_ast_sha256=hashlib.sha256(ast.dump(cls).encode()).hexdigest(),atomic_save_ast_sha256=hashlib.sha256(ast.dump(savefn).encode()).hexdigest(),ranges_ast_sha256=hashlib.sha256(ast.dump(ranges).encode()).hexdigest())
 m=dict(schema='qnn-partition-component-bundle-v1',purpose='constructed_partition_control',backend=dict(type='QNN_CPU',precision='float32',target='x86_64-linux-clang'),runtime_versions=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine()),files=files,profile=runtime_profile,lineage=lineage,os_dependencies={n:dict(sha256=sha(p),bytes=Path(p).stat().st_size) for n,p in deps.items() if n not in ['libc++.so.1','libc++abi.so.1','libunwind.so.1']},scope='Constructed CPU partition component only; production Host/UniAD task/package acceptance absent.');save(bundle/'manifest.json',m);save(root/'partition_package_control.json',dict(status='pass',bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),parts=len(parts),files=len(files),lineage=lineage,scope='Package creation and unchanged class AST only; actual relocation and rejection controls required.'))
if __name__=='__main__':main()
