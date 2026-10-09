#!/usr/bin/env python3
"""Exercise actual prior-build bindings and bounded reuse rejection controls."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from build_partitions import reusable_prefix,verify_record
from tool_run import ROOT,sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--build-run',type=Path,required=True);a=p.parse_args();b=json.loads((a.build_run/'partition_build.json').read_text());plan=json.loads(Path(b['plan']).read_text());hints=json.loads(Path(b['layout_hints']).read_text())['layouts'];recipe={n:sha(ROOT/n) for n in b['recipe']};checks={}
 rows,binding=reusable_prefix(a.build_run,plan,recipe,b['backend_sha256'],hints);checks['actual_source_library_abi_full_prefix_verified']=len(rows)==len(plan['parts'])
 altered=copy.deepcopy(plan);altered['parts'][0]['native_model_sha256']='0'*64;partial,record=reusable_prefix(a.build_run,altered,recipe,b['backend_sha256'],hints);checks['changed_cut_stops_reuse']=len(partial)==0
 def rejects(name,fn):
  try:fn()
  except ValueError:checks[name]=True
  else:checks[name]=False
 changed=recipe.copy();changed['onnx/qnn/build_model.py']='0'*64;rejects('changed_compiler_rejected',lambda:reusable_prefix(a.build_run,plan,changed,b['backend_sha256'],hints))
 rejects('changed_backend_rejected',lambda:reusable_prefix(a.build_run,plan,recipe,'0'*64,hints))
 changed_hints=hints.copy();changed_hints[plan['parts'][0]['inputs'][0]['source_name']]='NCHW';rejects('changed_layout_rejected',lambda:reusable_prefix(a.build_run,plan,recipe,b['backend_sha256'],changed_hints))
 row=copy.deepcopy(rows[0]);row['library_sha256']='0'*64;rejects('changed_library_digest_rejected',lambda:verify_record(row,plan['parts'][0]))
 row=copy.deepcopy(rows[0]);row['abi']['inputs'][0]['shape']=[1];rejects('changed_declared_abi_rejected',lambda:verify_record(row,plan['parts'][0]))
 save(Path.cwd()/'build_reuse_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,binding=binding,scope='Actual source/compiled library prefix bindings and rejection rules; no neural or task acceptance.'))
 if not all(checks.values()):raise ValueError('build reuse control failed')
if __name__=='__main__':main()
