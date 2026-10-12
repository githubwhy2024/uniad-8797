#!/usr/bin/env python3
"""Evaluate planning25 own-state declared mini dataset records using frozen real planning metrics."""
import cProfile
import argparse,copy,hashlib,json,os,pickle,signal,subprocess,sys
from pathlib import Path
import numpy as np
from evaluate_qnn_saved import bind_reference,evaluator,metric_control,SPLITS
from qnn_planning import FORMAT,SCHEMA,validate_record,frozen_metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save,process_identity
from resources import terminal
from qnn_mini_scope import require_complete,scope_of,check_counts


def bound_mini(run,audit_run):
 tool=terminal(run);audit_tool=terminal(audit_run);mini=json.loads((run/'mini_result.json').read_text());audit=json.loads((audit_run/'mini_audit.json').read_text());profile=json.loads((run/'profile.json').read_text())
 if sha(run/'mini_result.json')!=tool['artifacts']['mini_result.json']['sha256'] or sha(run/'profile.json')!=tool['artifacts']['profile.json']['sha256'] or sha(audit_run/'mini_audit.json')!=audit_tool['artifacts']['mini_audit.json']['sha256']:raise ValueError('planning mini/audit artifact pin differs')
 require_complete(mini,audit,profile)
 if mini.get('task_scope')!='planning' or audit['status']!='pass' or audit.get('task_scope')!='planning' or audit['mini_result_sha256']!=sha(run/'mini_result.json') or len(profile['abi']['outputs'])!=25:raise ValueError('planning25 actual own-state mini404/audit gate incomplete')
 if sha(run/'manifest.json')!=audit['manifest_sha256'] or sha(run/'profile.json')!=audit['profile_sha256'] or sha(run/'mini_progress.json')!=audit['journal_sha256']:raise ValueError('audited planning mini source/journal differs')
 for asset in profile['assets'].values():
  if sha(asset['path'])!=asset['sha256']:raise ValueError('audited planning runtime asset differs')
 return mini,json.loads((run/'manifest.json').read_text())


def worker(split,mini,manifest,ev,metrics,binding,root):
 import torch
 from mmcv import Config
 from mmcv.parallel import collate,scatter
 ev.install_tensorboard_import_stub()
 import projects.mmdet3d_plugin
 from mmdet3d.datasets import build_dataset
 torch.set_num_threads(4);os.environ['CUDA_VISIBLE_DEVICES']='';saved=mini['split_summaries'][split];path=Path(saved['results'])
 if sha(path)!=saved['results_sha256']:raise ValueError('planning saved records changed')
 with path.open('rb') as stream:payload=pickle.load(stream)
 sm=manifest['splits'][split];records=payload['records']
 if split not in scope_of(mini)['splits'] or scope_of(payload)!=scope_of(mini):raise ValueError('planning worker outside declared dataset')
 if payload['format']!=FORMAT or payload['evaluator_record_schema']!=SCHEMA or payload['scope']!='full' or payload['task_scope']!='planning' or payload['backend']!='q4_qnn_partition_cpu_float32' or payload['profile_sha256']!=mini['profile_sha256'] or payload['repository_head']!=manifest['repository_head'] or len(records)!=(81 if split=='mini_val' else 323):raise ValueError('planning-only payload gate differs')
 if [r['token'] for r in records]!=payload['tokens'] or payload['tokens']!=sm['tokens'] or ev.sequence_sha256(payload['tokens'])!=sm['token_order_sha256']:raise ValueError('planning frozen token order differs')
 for r in records:validate_record(r)
 for name in ['config','checkpoint','motion_anchor','mini_scene_table','mini_sample_table']:
  a=manifest['assets'][name]
  if sha(a['path'])!=a['sha256']:raise ValueError('frozen planning metric asset differs: '+name)
 if sha(sm['info_path'])!=sm['info_sha256']:raise ValueError('frozen split info differs')
 os.chdir(ROOT);config=Config.fromfile(manifest['assets']['config']['path']);cfg=copy.deepcopy(config.data.test);cfg.ann_file=sm['info_path'];cfg.data_root=manifest['data_root'];cfg.test_mode=True;cfg.file_client_args=dict(backend='disk');dataset=build_dataset(cfg)
 if [i['token'] for i in dataset.data_infos]!=sm['tokens']:raise ValueError('planning dataset token order differs')
 planning,raw=metrics(dataset,records,repair_native_planning_x=False)
 report=dict(format='qnn-planning-evaluation-v1',task_scope='planning',scope='full',backend=payload['backend'],split=split,frames=len(records),results_sha256=saved['results_sha256'],profile_sha256=mini['profile_sha256'],token_order_sha256=sm['token_order_sha256'],checkpoint_sha256=ev._asset_identity(manifest,'checkpoint'),config_sha256=ev._asset_identity(manifest,'config'),motion_anchor_sha256=ev._asset_identity(manifest,'motion_anchor'),metrics=dict(planning=planning,planning_raw=raw),planning_raw_available=raw is not None,metric_binding=binding)
 save(root/'planning_evaluation.json',ev.finite_jsonable(report))


def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--mini-run',type=Path);p.add_argument('--audit-run',type=Path);p.add_argument('--worker-split',choices=SPLITS);args=p.parse_args();root=Path.cwd();reference=bind_reference();ev=evaluator();control=metric_control(ev);metrics,binding=frozen_metrics(ev);policy={k:ev.FINAL_ACCEPTANCE_POLICY['planning'][k] for k in ['L2','obj_col','obj_box_col']}
 save(root/'planning_metric_preflight.json',dict(status='pass',reference=reference,metric_control=control,metric_binding=binding,policy_version=ev.FINAL_ACCEPTANCE_POLICY_VERSION,policy=policy,scope='Real metric classes and literal frozen planning statements/policy only; no planning25 task acceptance.'))
 if args.preflight_only:return
 if args.mini_run is None or args.audit_run is None:raise ValueError('planning mini404 and independent audit required')
 mini,manifest=bound_mini(args.mini_run,args.audit_run);selected=scope_of(mini);expected_standard,expected_raw=check_counts(selected,'planning')
 if args.worker_split:worker(args.worker_split,mini,manifest,ev,metrics,binding,root);return
 child=None
 def terminate(signum,frame):
  if child is not None and child.poll() is None:child.send_signal(signum);child.wait()
  raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
 reports={};comparisons={};standard=raw_count=0;overall=True
 for split in selected['splits']:
  directory=root/('worker-'+split);directory.mkdir();argv=[sys.executable,str(Path(__file__).resolve()),'--mini-run',str(args.mini_run.absolute()),'--audit-run',str(args.audit_run.absolute()),'--worker-split',split]
  with (directory/'worker.log').open('wb') as stream:
   child=subprocess.Popen(argv,cwd=directory,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);save(root/'planning_evaluation_progress.json',dict(stage='evaluate',split=split,child=process_identity(child.pid),argv=argv,log=str(directory/'worker.log'),reports=reports,comparisons=comparisons));code=child.wait();child=None
  if code:raise RuntimeError('planning metric worker failed: '+split)
  path=directory/'planning_evaluation.json';report=json.loads(path.read_text());saved=mini['split_summaries'][split]
  if report['results_sha256']!=saved['results_sha256'] or report['token_order_sha256']!=saved['token_order_sha256'] or report['frames']!=saved['frames'] or report['profile_sha256']!=mini['profile_sha256'] or report['metric_binding']!=binding:raise ValueError('planning evaluation report identity differs')
  reports[split]=dict(path=str(path),sha256=sha(path))
  for backend in ['native','pt','ort']:
   key=backend+'/'+split;ref=reference['references'][key]
   if sha(ref['path'])!=ref['sha256']:raise ValueError('frozen planning reference changed')
   baseline=json.loads(Path(ref['path']).read_text())
   identity={k:baseline.get(k)==report.get(k) for k in ['split','frames','token_order_sha256','checkpoint_sha256','config_sha256','motion_anchor_sha256']}
   if not all(identity.values()):raise ValueError('planning comparison identity gate failed: '+str(identity))
   checks=[ev._accept_numeric('planning.'+k,baseline['metrics']['planning'][k],report['metrics']['planning'][k],v) for k,v in policy.items()];raw_checks=[]
   if ref['planning_raw_available']:raw_checks=[ev._accept_numeric('planning_raw.'+k,baseline['metrics']['planning_raw'][k],report['metrics']['planning_raw'][k],v) for k,v in policy.items()]
   accepted=all(x['pass'] for x in checks+raw_checks);standard+=len(checks);raw_count+=len(raw_checks);overall=overall and accepted;comparison=root/'comparison'/backend/(split+'.json');comparison.parent.mkdir(parents=True,exist_ok=True)
   save(comparison,dict(status='pass' if accepted else 'failed_acceptance',overall_pass=accepted,identity=identity,policy_version=ev.FINAL_ACCEPTANCE_POLICY_VERSION,policy=policy,checks=checks,raw_checks=raw_checks,reference_sha256=ref['sha256'],candidate_sha256=sha(path)))
   comparisons[key]=dict(path=str(comparison),sha256=sha(comparison),overall_pass=accepted,standard_checks=len(checks),raw_checks=len(raw_checks))
 if standard!=expected_standard or raw_count!=expected_raw:raise ValueError('planning comparison coverage differs from declared frozen split policy')
 save(root/'task_result.json',dict(status='pass' if overall else 'failed_acceptance',execution_status='complete',acceptance_status='accepted' if overall else 'rejected',overall_pass=overall,task_scope='planning',dataset_scope=selected,frames=selected['frames'],standard_checks=standard,raw_planning_checks=raw_count,mini_result_sha256=sha(args.mini_run/'mini_result.json'),mini_audit_sha256=sha(args.audit_run/'mini_audit.json'),reference=reference,metric_binding=binding,policy=policy,reports=reports,comparisons=comparisons,scope='Planning25 own-state declared mini dataset planning task only, frozen formulas/tolerances; separate from full47 six tasks, portable release and 8797.'))
if __name__=='__main__':main()
