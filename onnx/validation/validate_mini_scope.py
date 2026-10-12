"""Declared dataset closure controls; constructed fixtures never claim inference."""
import copy,json
from pathlib import Path
from qnn_mini_scope import dataset_scope,scope_of,require_complete,check_counts


def main():
 checks={}
 def mini(selection):
  s=dataset_scope(selection);return dict(status='pass',execution_status='complete',stage=s['complete_stage'],frames=s['frames'],dataset_scope=s,split_summaries={name:dict(status='pass',frames=81 if name=='mini_val' else 323) for name in s['splits']})
 val=mini('mini_val');allmini=mini('all')
 checks['valid_val81']=require_complete(val)==dataset_scope('mini_val')
 checks['valid_all404']=require_complete(allmini)==dataset_scope()
 legacy=copy.deepcopy(allmini);legacy.pop('dataset_scope');checks['historical_404_defaults_only_to_all']=require_complete(legacy)==dataset_scope()
 checks['full81_96_plus_6']=check_counts(dataset_scope('mini_val'),'full')==(96,6)
 checks['planning81_9_plus_6']=check_counts(dataset_scope('mini_val'),'planning')==(9,6)
 checks['full404_192_plus_12']=check_counts(dataset_scope(),'full')==(192,12)
 checks['planning404_18_plus_12']=check_counts(dataset_scope(),'planning')==(18,12)
 def reject(label,action):
  try:action()
  except (ValueError,KeyError):checks[label]=True
  else:checks[label]=False
 for field,value in [('frames',80),('stage','partial_mini_records'),('stage','mini404_records'),('execution_status','partial')]:
  bad=copy.deepcopy(val);bad[field]=value;reject(field+'_'+str(value),lambda bad=bad:require_complete(bad))
 bad=copy.deepcopy(val);bad.pop('dataset_scope');reject('81_without_scope_marker',lambda:require_complete(bad))
 for label,change in [('wrong_declared_frames',lambda x:x['dataset_scope'].update(frames=80)),('extra_declared_train',lambda x:x['dataset_scope']['splits'].append('mini_train')),('missing_saved_val',lambda x:x.update(split_summaries={})),('extra_saved_train',lambda x:x['split_summaries'].update(mini_train=dict(status='pass',frames=323))),('short_saved_val',lambda x:x['split_summaries']['mini_val'].update(frames=80))]:
  bad=copy.deepcopy(val);change(bad);reject(label,lambda bad=bad:require_complete(bad))
 reject('audit_scope_mismatch',lambda:require_complete(val,dict(dataset_scope=dataset_scope(),frames=404)))
 reject('profile_scope_missing',lambda:require_complete(val,{}))
 reject('unknown_selection',lambda:dataset_scope('mini_train'))
 reject('unknown_task_scope',lambda:check_counts(dataset_scope('mini_val'),'other'))
 Path('mini_scope_control.json').write_text(json.dumps(dict(status='pass' if all(checks.values()) else 'failed',checks=checks,scope='Constructed scope completion and coverage controls only; no actual neural/metric acceptance.'),indent=2)+'\n')
 print(json.dumps(dict(status='pass' if all(checks.values()) else 'failed',checks=len(checks))))
 if not all(checks.values()):raise ValueError('dataset scope controls failed')


if __name__=='__main__':main()
