#!/usr/bin/env python3
"""Actual outside-cwd many-part recurrence and pinned bundle rejection controls."""
import argparse,copy,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save
from resources import terminal


def main():
 p=argparse.ArgumentParser();p.add_argument('--package-run',type=Path,required=True);a=p.parse_args();tool=terminal(a.package_run);record=json.loads((a.package_run/'partition_package_control.json').read_text())
 if sha(a.package_run/'partition_package_control.json')!=tool['artifacts']['partition_package_control.json']['sha256'] or record['status']!='pass':raise ValueError('partition package source differs')
 source=Path(record['bundle']);pin=record['manifest_sha256']
 if sha(source/'manifest.json')!=pin:raise ValueError('source package manifest changed')
 external=Path(tempfile.mkdtemp(prefix='q4-partition-relocation-'));bundle=external/'component';shutil.copytree(source,bundle,symlinks=False);cwd=external/'outside';cwd.mkdir();inputs=cwd/'inputs.npz';np.savez(inputs,state=np.zeros((512,512),np.float32),delta=np.ones((512,512),np.float32));env=os.environ.copy();env.update(PYTHONPATH='',LD_LIBRARY_PATH=str(bundle/'lib'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1');checks={};cases=[]
 def execute(root,digest,output,extra=None,load_only=False):
  loader=root/'runtime/partition_bundle.py';run_cwd=cwd/output;run_cwd.mkdir();out=run_cwd/'outputs.npz';report=run_cwd/'report.json'
  if load_only:
   code="import sys;sys.path.insert(0,sys.argv[1]);from partition_bundle import load_partition_bundle;load_partition_bundle(sys.argv[2],sys.argv[3])"
   argv=[sys.executable,'-c',code,str(root/'runtime'),str(root),digest]
  else:argv=[sys.executable,str(loader),'--bundle',str(root),'--manifest-sha256',digest,'--inputs',str(inputs),'--outputs',str(out),'--report',str(report),'--control-frames','6']
  with (run_cwd/'worker.log').open('wb') as stream:result=subprocess.run(argv,cwd=run_cwd,env={**env,'LD_LIBRARY_PATH':str(root/'lib')},stdout=stream,stderr=subprocess.STDOUT)
  return result.returncode,report,run_cwd/'worker.log'
 code,path,log=execute(bundle,pin,'valid');checks['actual_outside_cwd_execution']=code==0
 if path.exists():
  report=json.loads(path.read_text());checks['six_actual_own_state_control_frames']=report['status']=='pass' and len(report['frames'])==6 and all(report['checks'].values());checks['copied_cpp_dependency_closure']=all(report['cpp_dependency_closure'].values())
  with np.load(path.parent/'outputs.npz',allow_pickle=False) as archive:checks['final_constructed_state_exact']=len(archive.files)==1 and np.array_equal(archive[archive.files[0]],np.full((512,512),384,np.float32))
 checks['bundle_files_are_ordinary']=all(p.resolve()==p for p in bundle.rglob('*'));checks['no_sdk_or_q3_runtime_paths']=not any(s in (bundle/'manifest.json').read_text() for s in [str(ROOT),str(Path.home()),'/opt/qcom'])
 for label in ['wrong_pin','missing_library','changed_library','absolute_path','escape_path','alias_path','duplicate_file','schema','purpose','backend','numpy_version','file_symlink','root_symlink','actual_abi']:
  clone=external/label;shutil.copytree(source,clone,symlinks=False);m=json.loads((clone/'manifest.json').read_text());digest=pin;modified=False;load_only=True
  if label=='wrong_pin':digest='0'*64
  elif label=='missing_library':(clone/m['files']['part_0']['path']).unlink()
  elif label=='changed_library':
   file=clone/m['files']['part_0']['path'];data=bytearray(file.read_bytes());data[-1]^=1;file.write_bytes(data)
  elif label in ['absolute_path','escape_path','alias_path']:
   m['files']['part_0']['path']={'absolute_path':'/tmp/library.so','escape_path':'../library.so','alias_path':'lib/../lib/part000.so'}[label];modified=True
  elif label=='duplicate_file':m['files']['duplicate']=m['files']['part_0'].copy();modified=True
  elif label=='schema':m['schema']='unknown';modified=True
  elif label=='purpose':m['purpose']='production';modified=True
  elif label=='backend':m['backend']['type']='HTP';modified=True
  elif label=='numpy_version':m['runtime_versions']['numpy']='wrong';modified=True
  elif label=='file_symlink':
   file=clone/m['files']['part_0']['path'];file.unlink();file.symlink_to(source/m['files']['part_0']['path'])
  elif label=='root_symlink':
   link=external/'root-alias';link.symlink_to(clone,target_is_directory=True);clone=link
  elif label=='actual_abi':m['profile']['parts'][0]['abi']['inputs'][0]['shape']=[511,512];modified=True;load_only=False
  if modified:save(clone/'manifest.json',m);digest=sha(clone/'manifest.json')
  code,path,log=execute(clone,digest,label,load_only=load_only);checks[label+'_rejected']=code!=0
  if label=='actual_abi':checks['actual_abi_descriptor_rejection_reason']='unsupported logical/native extent mapping' in log.read_text()
  cases.append(dict(case=label,exit_code=code,log=str(log),log_sha256=sha(log)))
 save(Path.cwd()/'partition_relocation_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,package_manifest_sha256=pin,actual_report=str(cwd/'valid/report.json'),actual_report_sha256=sha(cwd/'valid/report.json') if (cwd/'valid/report.json').exists() else None,external_root=str(external),scope='Actual relocated constructed 16-part CPU recurrence and boundary rejection only; no UniAD Host/state/planning/mini or board acceptance.'))
 if not all(checks.values()):raise ValueError('partition relocation component control failed')
if __name__=='__main__':main()
