#!/usr/bin/env python3
"""Replay frozen accepted planning records with the literal metric extraction."""
import argparse,copy,json,os,pickle,sys
from pathlib import Path
import numpy as np
from evaluate_qnn_saved import bind_reference,evaluator,metric_control
from qnn_planning import record,validate_record,frozen_metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['mini_val','mini_train'],default='mini_val');args=p.parse_args();root=Path.cwd();reference=bind_reference();ev=evaluator();preflight=metric_control(ev);metrics,binding=frozen_metrics(ev);ref=reference['references']['ort/'+args.split];baseline=json.loads(Path(ref['path']).read_text());path,manifest_path,manifest=ev._evidence_manifest_for_results(baseline['results_path'])
 if sha(path)!=baseline['results_sha256']:raise ValueError('frozen accepted records differ')
 with path.open('rb') as stream:payload=pickle.load(stream)
 records=[];checks={};source_hash=[]
 for row in payload['records']:
  planning=row['planning'];source_hash.append([sha_array(planning[n]) for n in ['raw','optimized']]);out=record(dict(planning_raw=planning['raw']),row,planning['optimized'],dict(solver_ran=planning['collision_solver_ran'],selected_cells=planning['collision_selected_cells']));validate_record(out);records.append(out)
 checks['frozen_record_copy_and_identity']=len(records)==baseline['frames'] and [r['token'] for r in records]==manifest['splits'][args.split]['tokens']
 checks['raw_and_optimized_are_owned_copies']=all(not np.shares_memory(a['planning'][n],b['planning'][n]) for a,b in zip(records,payload['records']) for n in ['raw','optimized'])
 def rejects(label,alter):
  r=copy.deepcopy(records[0]);alter(r)
  try:validate_record(r)
  except ValueError:checks[label]=True
  else:checks[label]=False
 rejects('shape_rejected',lambda r:r['planning'].__setitem__('raw',np.zeros((6,2),np.float32)))
 rejects('dtype_rejected',lambda r:r['planning'].__setitem__('optimized',r['planning']['optimized'].astype(np.float64)))
 rejects('nonfinite_rejected',lambda r:r['planning']['raw'].__setitem__((0,0,0),np.nan))
 rejects('foreign_task_field_rejected',lambda r:r.__setitem__('detection',{}))
 rejects('invalid_solver_flag_rejected',lambda r:r['planning'].__setitem__('collision_solver_ran',1))
 rejects('negative_selected_cells_rejected',lambda r:r['planning'].__setitem__('collision_selected_cells',-1))
 rejects('missing_identity_rejected',lambda r:r.__setitem__('token',None))
 import torch
 from mmcv import Config
 ev.install_tensorboard_import_stub()
 import projects.mmdet3d_plugin
 from mmdet3d.datasets import build_dataset
 torch.set_num_threads(4);os.environ['CUDA_VISIBLE_DEVICES']='';os.chdir(ROOT);cfg=Config.fromfile(manifest['assets']['config']['path']);test=copy.deepcopy(cfg.data.test);sm=manifest['splits'][args.split];test.ann_file=sm['info_path'];test.data_root=manifest['data_root'];test.test_mode=True;test.file_client_args=dict(backend='disk');dataset=build_dataset(test)
 checks['frozen_dataset_token_order']=[i['token'] for i in dataset.data_infos]==sm['tokens']
 before=[[sha_array(r['planning'][n]) for n in ['raw','optimized']] for r in records];planning,raw=metrics(dataset,records,repair_native_planning_x=False);diff={}
 for scope,values in [('planning',planning),('planning_raw',raw)]:
  for key in ['L2','obj_col','obj_box_col']:
   actual=np.array(values[key],np.float64);expected=np.array(baseline['metrics'][scope][key],np.float64);finite=np.isfinite(actual)&np.isfinite(expected);d=float(np.abs(actual[finite]-expected[finite]).max()) if finite.any() else 0.;diff[scope+'.'+key]=d;checks['frozen_'+scope+'_'+key+'_reproduced']=actual.shape==expected.shape and np.array_equal(np.isnan(actual),np.isnan(expected)) and np.allclose(actual,expected,atol=1e-6,rtol=1e-6,equal_nan=True)
 checks['planning_metric_cannot_mutate_saved_records']=before==[[sha_array(r['planning'][n]) for n in ['raw','optimized']] for r in records]
 checks['frozen_reference_records_unmodified']=source_hash==[[sha_array(r['planning'][n]) for n in ['raw','optimized']] for r in payload['records']]
 save(root/'planning_adapter_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,metric_difference=diff,metric_binding=binding,metric_preflight=preflight,reference=ref,split=args.split,frames=len(records),scope='Replay of accepted Q3 ORT records/real GT and schema rejection only; no QNN neural, planning25 mini404 or task acceptance.'))
 if not all(checks.values()):raise ValueError('planning adapter/frozen metric replay failed')

def sha_array(v):
 import hashlib
 return dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest())
if __name__=='__main__':main()
