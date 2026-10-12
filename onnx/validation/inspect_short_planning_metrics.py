#!/usr/bin/env python3
"""Frozen planning-metric diagnostics for a committed short-frame snapshot only."""
import argparse,copy,hashlib,json,os,pickle,sys
from pathlib import Path
import numpy as np
from evaluate_qnn_saved import bind_reference,evaluator,metric_control
from qnn_planning import record,frozen_metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save


def main():
 p=argparse.ArgumentParser();p.add_argument('--short-run',type=Path,required=True);a=p.parse_args();root=Path.cwd();reference=bind_reference();ev=evaluator();control=metric_control(ev);metric,binding=frozen_metrics(ev);run=a.short_run.absolute();data=(run/'neural_progress.json').read_bytes();progress=json.loads(data);profile=json.loads((run/'profile.json').read_text());rows=progress.get('committed_frames',progress.get('completed_frames'))
 if not isinstance(rows,list):raise ValueError('short progress has no durable committed frame list')
 if not rows or sha(run/'profile.json')!=progress['profile_sha256'] or profile['backend']['type']!='QNN_CPU' or len(profile['abi']['outputs']) not in [25,47]:raise ValueError('no committed QNN short frames bound to actual profile')
 for asset in profile['assets'].values():
  if sha(asset['path'])!=asset['sha256']:raise ValueError('short candidate runtime resource differs')
 save(root/'committed_snapshot.json',dict(short_run=str(run),progress_snapshot_sha256=hashlib.sha256(data).hexdigest(),profile_sha256=progress['profile_sha256'],frames=rows,scope='Immutable committed prefix snapshot; live outer run is not declared accepted.'))
 import torch
 from mmcv import Config
 ev.install_tensorboard_import_stub()
 import projects.mmdet3d_plugin
 from mmdet3d.datasets import build_dataset
 torch.set_num_threads(4);os.environ['CUDA_VISIBLE_DEVICES']='';groups={};identities={};policy={k:ev.FINAL_ACCEPTANCE_POLICY['planning'][k] for k in ['L2','obj_col','obj_box_col']}
 for split in ['mini_val','mini_train']:
  paths={};payloads={};manifest=None
  for backend in ['native','pt','ort']:
   ref=reference['references'][backend+'/'+split];rep=json.loads(Path(ref['path']).read_text());path,_,m=ev._evidence_manifest_for_results(rep['results_path'])
   if sha(path)!=rep['results_sha256']:raise ValueError('frozen saved reference record differs')
   with open(path,'rb') as stream:payloads[backend]=pickle.load(stream)
   paths[backend]=dict(results_sha256=rep['results_sha256'],report_sha256=ref['sha256'])
   if backend=='ort':manifest=m
  sm=manifest['splits'][split];bytoken={t:i for i,t in enumerate(sm['tokens'])};selected=[r for r in rows if r['token'] in bytoken]
  if not selected:continue
  identities[split]=paths;indices=[bytoken[r['token']] for r in selected];candidate=[]
  for row,index in zip(selected,indices):
   for key in ['outputs','final_plan','checkpoint']:
    if sha(row[key+'_path'])!=row[key+'_sha256']:raise ValueError('committed short frame artifact differs')
   with np.load(row['outputs_path'],allow_pickle=False) as arc:raw=arc['planning_raw'].copy()
   with np.load(row['final_plan_path'],allow_pickle=False) as arc:
    if len(arc.files)!=1:raise ValueError('short final plan artifact differs')
    final=arc[arc.files[0]].copy()
   info=payloads['ort']['records'][index]
   if info['token']!=row['token']:raise ValueError('reference row token differs')
   candidate.append(record(dict(planning_raw=raw),info,final,row['planning_info']))
  os.chdir(ROOT);cfg=Config.fromfile(manifest['assets']['config']['path']);test=copy.deepcopy(cfg.data.test);test.ann_file=sm['info_path'];test.data_root=manifest['data_root'];test.test_mode=True;test.file_client_args=dict(backend='disk');dataset=build_dataset(test)
  if [i['token'] for i in dataset.data_infos]!=sm['tokens']:raise ValueError('frozen diagnostic GT token order differs')
  class SelectedDataset:
   def __getitem__(self,index):return dataset[indices[index]]
  final_metrics,raw_metrics=metric(SelectedDataset(),candidate,repair_native_planning_x=False);comparisons={}
  for backend,payload in payloads.items():
   records=[copy.deepcopy(payload['records'][i]) for i in indices]
   if [v['token'] for v in records]!=[v['token'] for v in candidate]:raise ValueError('matched diagnostic reference token order differs')
   repaired=payload.get('backend')=='native_pt_cpu_cuda_contract' and payload.get('format')=='uniad-native-cpu-results-v1';fp,rp=metric(SelectedDataset(),records,repair_native_planning_x=repaired);checks=[ev._accept_numeric('planning.'+k,fp[k],final_metrics[k],v) for k,v in policy.items()];raw_checks=[]
   if rp is not None:raw_checks=[ev._accept_numeric('planning_raw.'+k,rp[k],raw_metrics[k],v) for k,v in policy.items()]
   comparisons[backend]=dict(frozen_tolerance_checks=checks,raw_checks=raw_checks,diagnostic_within_tolerance=all(x['pass'] for x in checks+raw_checks),legacy_native_record_repair=repaired)
  groups[split]=dict(frames=len(selected),tokens=[r['token'] for r in selected],metrics=dict(planning=final_metrics,planning_raw=raw_metrics),comparisons=comparisons)
 if sum(v['frames'] for v in groups.values())!=len(rows):raise ValueError('some short frame tokens lack frozen GT/reference')
 save(root/'short_planning_diagnostic.json',ev.finite_jsonable(dict(status='complete',frames=len(rows),task_acceptance='not_evaluated',groups=groups,reference_records=identities,metric_binding=binding,metric_control=control,policy=policy,snapshot_sha256=sha(root/'committed_snapshot.json'),scope='Committed short-frame task-metric diagnostic only; subset/state-history differs from complete mini404, no task/profile promotion. Floating tensor values are not compared.')))
if __name__=='__main__':main()
