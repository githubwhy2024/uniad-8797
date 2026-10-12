#!/usr/bin/env python3
"""Convert and compile a cut plan with a resumable completed-part journal."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
from tool_run import ROOT,SDK,CONVERTER_ENV,sha,save,process_identity,stamp
from session import DTYPES


def terminal(run):
 run=Path(run);status=json.loads((run/'status.json').read_text());result=json.loads((run/'result.json').read_text())
 if status['status']!='pass' or result['status']!='pass' or status['result_sha256']!=sha(run/'result.json'):raise ValueError('part tool result is not terminal/hash-bound pass')
 return result

def interface(part,convert):
 result=terminal(convert);net_path=Path(convert)/'model_net.json'
 if result['artifacts']['model_net.json']['sha256']!=sha(net_path):raise ValueError('actual part network identity differs')
 actual=json.loads(net_path.read_text())['graph']['tensors'];abi=dict(schema='qnn-native-abi-v1')
 for kind in ('inputs','outputs'):
  abi[kind]=[]
  for row in part[kind]:
   tensor=actual[row['name']]
   if DTYPES[tensor['data_type']]!=row['dtype']:raise ValueError('part native dtype differs')
   native_shape=tensor['dims']
   if [d for d in row['shape'] if d!=1]!=[d for d in native_shape if d!=1]:raise ValueError('part native non-singleton extent differs')
   entry=dict(name=row['name'],native_name=row['name'],shape=row['shape'],native_shape=native_shape,dtype=row['dtype'])
   if row['shape']!=native_shape:entry['wire_view']='singleton_axes'
   if row['dtype']=='int64':entry['integer_range']=[-2**31,2**31-1]
   abi[kind].append(entry)
 return abi

def verify_record(row,part):
 terminal(row['converter_run']);terminal(row['compile_run'])
 actual=json.loads((Path(row['compile_run'])/'model_build.json').read_text())
 if actual['library']!=row['library'] or actual['library_sha256']!=row['library_sha256'] or actual['source_model_sha256']!=part['native_model_sha256'] or interface(part,row['converter_run'])!=row['abi']:raise ValueError('completed part build/ABI differs')
 if row['index']!=part['index'] or row['native_model_sha256']!=part['native_model_sha256'] or sha(part['native_model'])!=part['native_model_sha256']:raise ValueError('completed part candidate differs')
 for key in ('library','converter_audit'):
  if sha(row[key])!=row[key+'_sha256']:raise ValueError('completed part artifact differs')
 if sha(Path(row['converter_run'])/'result.json')!=row['converter_result_sha256'] or sha(Path(row['compile_run'])/'result.json')!=row['compile_result_sha256']:raise ValueError('completed tool provenance differs')

def reusable_prefix(prior,plan,recipe,backend_sha,hints):
 prior=Path(prior).absolute()
 if prior.resolve()!=prior or not prior.is_relative_to(ROOT/'onnx/runs'):raise ValueError('ordinary Q4 prior build required')
 path=prior/'partition_build.json';partial_binding=None
 if path.exists():
  result=terminal(prior)
  if result['artifacts']['partition_build.json']['sha256']!=sha(path):raise ValueError('prior build report differs')
  old=json.loads(path.read_text())
 else:
  status=json.loads((prior/'status.json').read_text());result=json.loads((prior/'result.json').read_text())
  if status['status'] not in ('failed','interrupted') or status['result_sha256']!=sha(prior/'result.json') or result['status']!=status['status']:raise ValueError('partial provider has no hash-bound failure/interruption')
  for role in ('controller','child'):
   identity=status.get(role)
   if identity and identity.get('startticks') is not None and process_identity(identity['pid']).get('startticks')==identity['startticks']:raise ValueError('partial provider is still active')
  progress=json.loads((prior/'partition_build_progress.json').read_text());argv=json.loads((prior/'launch.json').read_text())['argv'];source_plan=Path(argv[argv.index('--plan')+1]);plan_digest=argv[argv.index('--plan-sha256')+1]
  if sha(source_plan)!=plan_digest or progress['plan_sha256']!=plan_digest:raise ValueError('partial plan identity differs')
  terminal(prior/'audit');terminal(prior/'layouts')
  source=json.loads(source_plan.read_text());audit=prior/'audit/partition_audit.json';layouts=prior/'layouts/layout_hints.json'
  if json.loads(audit.read_text())['plan_sha256']!=plan_digest or json.loads(layouts.read_text())['source_model_sha256']!=source['source_model_sha256'] or len(progress['completed_parts'])>=len(source['parts']):raise ValueError('partial source/audit/layout closure differs')
  old=dict(status='pass',schema='qnn-partition-build-v1',backend_sha256=progress['backend_sha256'],recipe=progress['recipe'],plan=str(source_plan),plan_sha256=plan_digest,audit=str(audit),audit_sha256=sha(audit),layout_hints=str(layouts),layout_hints_sha256=sha(layouts),parts=progress['completed_parts'])
  # Local eligible-prefix descriptor only; no passing ancestor report is saved.
  path=prior/'partition_build_progress.json'
  partial_binding=dict(provider_status=status['status'],provider_result_sha256=sha(prior/'result.json'),provider_progress_sha256=sha(path),completed_parts=len(old['parts']),provider_full_build_accepted=False)
 # The orchestration script may gain reuse support; every converter/compile/audit
 # helper and the actual backend must still match the accepted prior build.
 if old['backend_sha256']!=backend_sha or set(old['recipe'])!=set(recipe) or any(old['recipe'][n]!=h for n,h in recipe.items() if n!='onnx/qnn/build_partitions.py'):raise ValueError('prior conversion/compilation recipe or backend differs')
 if old['status']!='pass' or old['schema']!='qnn-partition-build-v1' or sha(old['plan'])!=old['plan_sha256'] or sha(old['audit'])!=old['audit_sha256'] or sha(old['layout_hints'])!=old['layout_hints_sha256']:raise ValueError('prior build source/audit/layout lineage differs')
 old_plan=json.loads(Path(old['plan']).read_text());old_hints=json.loads(Path(old['layout_hints']).read_text())['layouts']
 if len(old['parts'])!=len(old_plan['parts']) and partial_binding is None:raise ValueError('prior actual build is incomplete')
 completed=[]
 for row,old_part,new_part in zip(old['parts'],old_plan['parts'],plan['parts']):
  if new_part['native_model_sha256']!=old_part['native_model_sha256']:break
  if any(old_hints.get(v['source_name'])!=hints.get(v['source_name']) for kind in ('inputs','outputs') for v in new_part[kind]):raise ValueError('byte-identical part has a changed spatial layout policy')
  verify_record(row,old_part);verify_record(row,new_part);completed.append(row.copy())
 binding=dict(ancestor=str(prior),ancestor_result_sha256=sha(prior/'result.json'),ancestor_build_sha256=sha(path),ancestor_plan_sha256=old['plan_sha256'],reused_prefix=len(completed),native_source_byte_identical=True,converter_compile_helpers_unchanged=True,spatial_layout_unchanged=True,scope='Only independently verified byte-identical compiled cut prefix reused; every changed cut is actually converted and compiled.')
 if partial_binding is not None:binding['partial_provider']=partial_binding
 return completed,binding

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--resume-from',type=Path);p.add_argument('--reuse-build',type=Path);a=p.parse_args();root=Path.cwd()
 if root.resolve()!=root or not root.is_relative_to(ROOT/'onnx/runs'):raise ValueError('ordinary Q4 build path required')
 if sha(a.plan)!=a.plan_sha256:raise ValueError('plan identity differs')
 plan=json.loads(a.plan.read_text());hints={};reuse_binding=None
 if a.resume_from and a.reuse_build:raise ValueError('resume and explicit prior-build reuse are mutually exclusive')
 recipe={n:sha(ROOT/n) for n in ('onnx/qnn/build_partitions.py','onnx/qnn/partition_model.py','onnx/qnn/build_model.py','onnx/qnn/tool_run.py','onnx/qnn/audit_converter.py','onnx/validation/audit_partitions.py')};backend_sha=sha(SDK/'lib/x86_64-linux-clang/libQnnCpu.so');completed=[];child=None
 if 'shared_constants' in plan:
  if sha(ROOT/'onnx/qnn/shared_constants.py')!=plan['shared_constants_helper_sha256']:raise ValueError('immutable pool helper differs')
  recipe['onnx/qnn/shared_constants.py']=plan['shared_constants_helper_sha256']
 if a.resume_from:
  prior=a.resume_from.absolute()
  if prior.resolve()!=prior or not prior.is_relative_to(ROOT/'onnx/runs') or prior==root:raise ValueError('ordinary different Q4 ancestor required')
  status=json.loads((prior/'status.json').read_text())
  for role in ('controller','child'):
   identity=status.get(role)
   if identity and identity.get('startticks') is not None and process_identity(identity['pid']).get('startticks')==identity['startticks']:raise ValueError('ancestor is still active')
  progress=json.loads((prior/'partition_build_progress.json').read_text())
  if progress['plan_sha256']!=a.plan_sha256 or progress['recipe']!=recipe or progress['backend_sha256']!=backend_sha:raise ValueError('ancestor build recipe/runtime/candidate differs')
  completed=progress['completed_parts']
  if len(completed)>=len(plan['parts']):raise ValueError('ancestor already completed the requested scope')
  for row,part in zip(completed,plan['parts']):verify_record(row,part)
  save(root/'build_resume_binding.json',dict(ancestor=str(prior),progress_sha256=sha(prior/'partition_build_progress.json'),plan_sha256=a.plan_sha256,completed_prefix=len(completed)))
 def terminate(signum,frame):
  if child is not None and child.poll() is None:child.send_signal(signum);child.wait()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 def step(name,stage,model,digest,expects,command):
  nonlocal child
  if any(sha(ROOT/n)!=h for n,h in recipe.items()):raise ValueError('build recipe changed during execution')
  run=root/name;args=[str(CONVERTER_ENV/'bin/python'),str(ROOT/'onnx/qnn/tool_run.py'),'--run-dir',str(run),'--stage',stage,'--model',str(model),'--model-sha256',digest]
  if stage=='convert':
   args+=['--diagnose-converter','--address-space-gib','18']
   index=int(name[4:7]);part=plan['parts'][index];layouts={r['name']:hints[r['source_name']] for r in part['outputs'] if r['source_name'] in hints}
   input_layouts={r['name']:hints[r['source_name']] for r in part['inputs'] if r['source_name'] in hints}
   if input_layouts:
    policy=root/(name+'-input-layouts.json');save(policy,input_layouts);args+=['--input-layouts',str(policy),'--input-layouts-sha256',sha(policy)]
   if layouts:
    policy=root/(name+'-output-layouts.json');save(policy,layouts);args+=['--output-layouts',str(policy),'--output-layouts-sha256',sha(policy)]
  for e in expects:args+=['--expect',e]
  if command:args+=['--']+command
  child=subprocess.Popen(args,cwd=ROOT);save(root/'partition_build_progress.json',dict(active_run=str(run),stage=stage,child=process_identity(child.pid),command=args,completed_parts=completed,plan_sha256=a.plan_sha256,recipe=recipe,backend_sha256=backend_sha,updated_utc=stamp()));code=child.wait();child=None
  if code:raise RuntimeError('partition build step failed: '+name)
  terminal(run);return run
 audit=step('audit','proof',plan['source_model'],plan['source_model_sha256'],['partition_audit.json'],[__import__('os').environ.get('UNIAD_PYTHON',__import__('sys').executable),str(ROOT/'onnx/validation/audit_partitions.py'),'--plan',str(a.plan.absolute()),'--plan-sha256',a.plan_sha256])
 layouts_run=step('layouts','control',plan['source_model'],plan['source_model_sha256'],['layout_hints.json'],[__import__('os').environ.get('UNIAD_PYTHON',__import__('sys').executable),str(ROOT/'onnx/qnn/partition_model.py'),'--model',plan['source_model'],'--model-sha256',plan['source_model_sha256'],'--layouts-only'])
 layout_report=json.loads((layouts_run/'layout_hints.json').read_text())
 if layout_report['status']!='pass' or layout_report['source_model_sha256']!=plan['source_model_sha256']:raise ValueError('layout declaration source differs')
 hints=layout_report['layouts']
 if a.reuse_build:
  completed,reuse_binding=reusable_prefix(a.reuse_build,plan,recipe,backend_sha,hints);save(root/'build_reuse_binding.json',reuse_binding)
 for part in plan['parts'][len(completed):]:
  index=part['index'];m=part['native_model'];h=part['native_model_sha256'];convert=step(f'part{index:03d}-convert','convert',m,h,[],[]);compile=step(f'part{index:03d}-compile','compile',m,h,['model_build.json','lib/x86_64-linux-clang/libq4_part.so'],[str(CONVERTER_ENV/'bin/python'),str(ROOT/'onnx/qnn/build_model.py'),'--convert-run',str(convert),'--name','q4_part']);build=json.loads((compile/'model_build.json').read_text());net_path=convert/'model_net.json';net=json.loads(net_path.read_text());audit_path=convert/'converter_audit.json';actual=net['graph']['tensors'];abi=interface(part,convert)
  completed.append(dict(index=index,native_model_sha256=h,converter_run=str(convert),converter_result_sha256=sha(convert/'result.json'),compile_run=str(compile),compile_result_sha256=sha(compile/'result.json'),library=build['library'],library_sha256=build['library_sha256'],converter_audit=str(audit_path),converter_audit_sha256=sha(audit_path),abi=abi))
  save(root/'partition_build_progress.json',dict(stage='between_parts',completed_parts=completed,plan_sha256=a.plan_sha256,recipe=recipe,backend_sha256=backend_sha,updated_utc=stamp()))
 save(root/'partition_build.json',dict(status='pass',schema='qnn-partition-build-v1',plan=str(a.plan.absolute()),plan_sha256=a.plan_sha256,audit=str(audit/'partition_audit.json'),audit_sha256=sha(audit/'partition_audit.json'),layout_hints=str(layouts_run/'layout_hints.json'),layout_hints_sha256=sha(layouts_run/'layout_hints.json'),parts=completed,recipe=recipe,backend_sha256=backend_sha,reuse_binding=reuse_binding,scope='Actual part converters/CPU builds and source coverage only; lazy actual loads, neural frames and task metrics separate.'))
if __name__=='__main__':main()
