#!/usr/bin/env python3
"""Use the frozen six-task evaluator in a fresh process, adding only QNN comparisons."""
import cProfile
import argparse,ast,hashlib,importlib,json,os,signal,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save,process_identity
from resources import terminal
from reference_paths import resolve_reference
BASE=ROOT/'onnx/reference/metrics/result.json'
BASE_SHA='c77c3fd9d462eeb00aa677343467e6dc4e8fb4d2622e582ebd1eb9991140704d'
SPLITS=('mini_val','mini_train')
from qnn_mini_scope import require_complete,check_counts

def bind_reference():
 if sha(BASE)!=BASE_SHA:raise ValueError('frozen three-way task acceptance differs')
 base=json.loads(BASE.read_text())
 if base['status']!='pass' or base['execution_status']!='complete' or base['overall_pass'] is not True:raise ValueError('frozen reference tasks are not accepted')
 for relative,digest in base['identity']['original_sources_sha256'].items():
  if sha(ROOT/relative)!=digest:raise ValueError('frozen task/model source differs: '+relative)
 references={}
 for split in SPLITS:
  for backend in ('native','pt','ort'):
   row=base['identity']['native_reuse'][split] if backend=='native' else base['evaluations'][backend+'/'+split]
   path=row['evaluation'] if backend=='native' else row['path'];digest=row['evaluation_sha256'] if backend=='native' else row['sha256']
   path=resolve_reference(path)
   if sha(path)!=digest:raise ValueError('frozen evaluation report differs')
   report=json.loads(Path(path).read_text());references[backend+'/'+split]=dict(path=str(path),sha256=digest,planning_raw_available=report['metrics'].get('planning_raw') is not None)
 return dict(base_result_sha256=BASE_SHA,references=references,original_sources_sha256=base['identity']['original_sources_sha256'])

def evaluator():
 sys.path[:0]=[str(ROOT/'onnx/fixed'),str(ROOT)];ev=importlib.import_module('validate')
 for name in ('validate','assets'):
  if Path(importlib.import_module(name).__file__).resolve()!=ROOT/'onnx/fixed'/(name+'.py'):raise ValueError('foreign frozen evaluator import')
 if 'export' in sys.modules:raise ValueError('inference-only export stubs present in evaluation worker')
 return ev

def metric_control(ev):
 path=ROOT/'onnx/reference/adapters/metric_control.py'
 if sha(path)!='330ce3eaceb4b5d32e4a7e3673d7d571cdad66dde7ed75a5ba30f3fb60d8850c':raise ValueError('frozen metric preflight source differs')
 fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='metric_control');namespace=dict(Path=Path,sha=sha);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace);result=namespace['metric_control'](ev);result['source_sha256']=sha(path);result['ast_sha256']=hashlib.sha256(ast.dump(fn).encode()).hexdigest();return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--mini-run',type=Path);p.add_argument('--audit-run',type=Path);p.add_argument('--worker-split',choices=SPLITS);args=p.parse_args();root=Path.cwd();reference=bind_reference();ev=evaluator();control=metric_control(ev);save(root/'metric_preflight.json',dict(status='pass',reference=reference,metric_control=control,validator_sha256=sha(ROOT/'onnx/fixed/validate.py'),policy_sha256=hashlib.sha256(json.dumps(ev.FINAL_ACCEPTANCE_POLICY,sort_keys=True).encode()).hexdigest(),scope='Frozen reference/report/policy and actual real metric classes only; QNN task metrics not evaluated.'))
 if args.preflight_only:return
 if args.mini_run is None or args.audit_run is None:raise ValueError('complete QNN mini run and independent completion audit required')
 mini_tool=terminal(args.mini_run);terminal(args.audit_run);mini=json.loads((args.mini_run/'mini_result.json').read_text());audit=json.loads((args.audit_run/'mini_audit.json').read_text())
 if mini.get('task_scope','full')!='full' or audit.get('task_scope','full')!='full':raise ValueError('six-task evaluator requires full47 records and audit')
 selected=require_complete(mini,audit);expected_standard,expected_raw=check_counts(selected,'full')
 if sha(args.mini_run/'mini_result.json')!=mini_tool['artifacts']['mini_result.json']['sha256'] or audit['status']!='pass' or audit['mini_result_sha256']!=sha(args.mini_run/'mini_result.json'):raise ValueError('QNN declared mini record/audit acceptance incomplete')
 profile=json.loads((args.mini_run/'profile.json').read_text());require_complete(mini,profile)
 if sha(args.mini_run/'profile.json')!=mini_tool['artifacts']['profile.json']['sha256'] or sha(args.mini_run/'profile.json')!=audit['profile_sha256'] or sha(args.mini_run/'manifest.json')!=audit['manifest_sha256'] or sha(args.mini_run/'mini_progress.json')!=audit['journal_sha256']:raise ValueError('audited six-task mini artifact differs')
 if args.worker_split:
  if args.worker_split not in selected['splits']:raise ValueError('worker split outside declared dataset')
  row=mini['split_summaries'][args.worker_split]
  if sha(row['results'])!=row['results_sha256']:raise ValueError('QNN saved task records differ')
  os.chdir(ROOT);ev.evaluate_results(SimpleNamespace(results=row['results'],output_dir=str(root/'evaluation'/args.worker_split),threads=4));return
 child=None
 def terminate(signum,frame):
  if child is not None and child.poll() is None:child.send_signal(signum);child.wait()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 reports={};comparisons={};standard_count=raw_count=0;overall=True
 for split in selected['splits']:
  # Fresh interpreter per split, no export.py and no inherited inference stubs.
  worker=root/('worker-'+split);worker.mkdir();argv=[sys.executable,str(Path(__file__).resolve()),'--mini-run',str(args.mini_run.absolute()),'--audit-run',str(args.audit_run.absolute()),'--worker-split',split]
  with (worker/'worker.log').open('wb') as stream:
   child=subprocess.Popen(argv,cwd=worker,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);save(root/'evaluation_progress.json',dict(stage='evaluate',split=split,child=process_identity(child.pid),argv=argv,log=str(worker/'worker.log'),reports=reports,comparisons=comparisons));code=child.wait();child=None
  if code:raise RuntimeError('QNN evaluator failed: '+split)
  report_path=worker/'evaluation'/split/'evaluation.json';report=json.loads(report_path.read_text());saved=mini['split_summaries'][split]
  if report['results_sha256']!=saved['results_sha256'] or report['token_order_sha256']!=saved['token_order_sha256'] or report['backend']!='q4_qnn_partition_cpu_float32' or report['frames']!=saved['frames'] or not report.get('planning_raw_available'):raise ValueError('QNN formal evaluation identity/backend/raw differs')
  reports[split]=dict(path=str(report_path),sha256=sha(report_path))
  for backend in ('native','pt','ort'):
   key=backend+'/'+split;path=root/'comparison'/backend/(split+'.json');path.parent.mkdir(parents=True,exist_ok=True)
   try:ev.compare_evaluations(SimpleNamespace(native_evaluation=reference['references'][key]['path'],candidate_evaluation=str(report_path),output=str(path)))
   except SystemExit as error:
    if error.code not in (0,2) or not path.is_file():raise
   comparison=json.loads(path.read_text());standard=comparison['checks'];raw=comparison['planning_raw_supplement']['checks'];standard_count+=len(standard);raw_count+=len(raw);overall=overall and comparison['overall_pass'] and all(x['pass'] for x in raw)
   comparisons[key]=dict(path=str(path),sha256=sha(path),overall_pass=comparison['overall_pass'],standard_checks=len(standard),raw_checks=len(raw),raw_available=comparison['planning_raw_supplement']['available'])
 if standard_count!=expected_standard or raw_count!=expected_raw:raise ValueError('QNN comparison coverage differs from declared frozen split policy')
 save(root/'task_result.json',dict(status='pass' if overall else 'failed_acceptance',execution_status='complete',acceptance_status='accepted' if overall else 'rejected',overall_pass=bool(overall),task_scope='full',dataset_scope=selected,frames=selected['frames'],standard_checks=standard_count,raw_planning_checks=raw_count,mini_result_sha256=sha(args.mini_run/'mini_result.json'),mini_audit_sha256=sha(args.audit_run/'mini_audit.json'),reference=reference,reports=reports,comparisons=comparisons,scope='QNN full47 own-state declared mini dataset six tasks with frozen metric tolerances; separate from planning25, portable release, HTP/8797 acceptance.'))
 # Execution completion remains distinguishable from task rejection in task_result.json.
if __name__=='__main__':main()
