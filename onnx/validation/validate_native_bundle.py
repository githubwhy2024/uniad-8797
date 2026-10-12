#!/usr/bin/env python3
"""Execute a native component outside the checkout and reject changed bundle contracts."""
import argparse,copy,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import sha,save
from resources import terminal

def main():
 p=argparse.ArgumentParser();p.add_argument('--package-run',type=Path,required=True);a=p.parse_args();terminal(a.package_run);record_path=a.package_run/'package_control.json';record=json.loads(record_path.read_text());source=Path(record['bundle'])
 if sha(source/'manifest.json')!=record['manifest_sha256']:raise ValueError('component source pin differs')
 root=Path.cwd();work=Path(tempfile.mkdtemp(prefix='q4-native-relocation-'));bundle=work/'bundle';shutil.copytree(source,bundle);manifest=json.loads((bundle/'manifest.json').read_text());env=os.environ.copy();env.update(PYTHONPATH='',LD_LIBRARY_PATH=str(bundle/'lib'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1');checks={};cases=[];scores=np.zeros((300,10),np.float32);labels=(np.arange(300)+7)%10;scores[np.arange(300),labels]=3;inputs=work/'inputs.npz';np.savez(inputs,scores=scores);out=work/'outputs.npz';report=work/'execution.json'
 command=[sys.executable,str(bundle/'runtime/native_bundle.py'),'--bundle',str(bundle),'--manifest-sha256',record['manifest_sha256'],'--inputs',str(inputs),'--outputs',str(out),'--report',str(report)];child=subprocess.run(command,cwd=work,env=env,text=True,capture_output=True);(root/'relocated.log').write_text(child.stdout+child.stderr)
 if child.returncode:raise RuntimeError('actual relocated native component failed; see relocated.log')
 data=json.loads(report.read_text());checks['actual_relocated_execution']=data['status']=='pass';checks['bundled_cpp_dependencies']=all(v['from_bundle'] for v in data['cpp_dependency_closure'].values());checks['inputs_unmodified']=data['inputs_unmodified']
 expected=dict(labels=labels.astype(np.int64),modded=labels.astype(np.int64),joined=np.concatenate([labels,[6]]).astype(np.int64),scattered=np.concatenate([labels,[0]]).astype(np.int64))
 with np.load(out,allow_pickle=False) as archive:
  for name,value in expected.items():checks[name+'_exact_integer_routing']=archive[name].dtype==value.dtype and np.array_equal(archive[name],value)
 # Each negative case starts from an untouched ordinary copy. The child imports
 # only bundle runtime files, with no checkout or SDK path in Python/LD search.
 worker=work/'inspect.py';worker.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(sys.argv[1])/'runtime'))\nfrom native_bundle import load_native_bundle,session_from_bundle\nif sys.argv[3]=='native':\n with session_from_bundle(sys.argv[1],sys.argv[2]):pass\nelse:load_native_bundle(sys.argv[1],sys.argv[2])\n")
 labels_to_test=['pin','missing','bytes','absolute','escape','alias_path','symlink','root_symlink','schema','purpose','backend','version','duplicate','actual_abi']
 for label in labels_to_test:
  case=work/('case_'+label);shutil.copytree(bundle,case);candidate=copy.deepcopy(manifest);pin=record['manifest_sha256'];mode='inspect';changed=False;target=case
  if label=='pin':pin='0'*64
  elif label=='missing':(case/candidate['files']['model_lib']['path']).unlink()
  elif label=='bytes':
   f=case/candidate['files']['model_lib']['path'];f.write_bytes(f.read_bytes()+b'changed')
  elif label=='absolute':candidate['files']['model_lib']['path']=str(bundle/'lib/model.so');changed=True
  elif label=='escape':candidate['files']['model_lib']['path']='../bundle/lib/model.so';changed=True
  elif label=='alias_path':candidate['files']['model_lib']['path']='lib/./model.so';changed=True
  elif label=='symlink':
   f=case/candidate['files']['model_lib']['path'];f.unlink();f.symlink_to(bundle/'lib/model.so')
  elif label=='root_symlink':target=work/'aliased-root';target.symlink_to(case,target_is_directory=True)
  elif label=='schema':candidate['schema']='different';changed=True
  elif label=='purpose':candidate['purpose']='planning';changed=True
  elif label=='backend':candidate['backend']['type']='HTP';changed=True
  elif label=='version':candidate['runtime_versions']['numpy']='different';changed=True
  elif label=='duplicate':candidate['files']['duplicate']=copy.deepcopy(candidate['files']['model_lib']);changed=True
  elif label=='actual_abi':candidate['abi']['inputs'][0]['shape']=[299,10];changed=True;mode='native'
  if changed:(case/'manifest.json').write_text(json.dumps(candidate,indent=2)+'\n');pin=sha(case/'manifest.json')
  case_env=dict(env,LD_LIBRARY_PATH=str(case/'lib'));argv=[sys.executable,str(worker),str(target),pin,mode];done=subprocess.run(argv,cwd=work,env=case_env,text=True,capture_output=True);checks[label+'_rejected']=done.returncode!=0;cases.append(dict(case=label,rejected=done.returncode!=0,returncode=done.returncode,error_tail=(done.stderr or done.stdout)[-700:]));(root/(label+'.log')).write_text(done.stdout+done.stderr)
 shutil.copyfile(report,root/'relocated_execution.json');shutil.copyfile(out,root/'relocated_outputs.npz');save(root/'native_bundle_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,manifest_sha256=record['manifest_sha256'],package_result_sha256=sha(a.package_run/'result.json'),outside_checkout_directory=str(work),command=command,environment={n:env[n] for n in ('PYTHONPATH','LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},scope='Constructed relocated native CPU component only; no UniAD Host/state/task/planning bundle or board acceptance. Temporary ordinary copies retained for diagnosis.'))
 if not all(checks.values()):raise ValueError('native component relocation controls rejected')
if __name__=='__main__':main()
