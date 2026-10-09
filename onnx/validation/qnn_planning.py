"""Planning-only records and exact frozen planning-metric body extraction."""
import ast,copy,hashlib
from pathlib import Path
import numpy as np

SCHEMA='qnn-planning-record-v1'
FORMAT='uniad-qnn-planning-results-v1'

def record(outputs,info,final,collision_info):
 values={}
 for key,value in [('raw',outputs['planning_raw']),('optimized',final)]:
  if not isinstance(value,np.ndarray) or value.shape!=(1,6,2) or value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('planning record ABI/nonfinite violation: '+key)
  values[key]=value.copy()
 if type(collision_info['solver_ran']) is not bool or type(collision_info['selected_cells']) is not int or collision_info['selected_cells']<0:raise ValueError('planning solver record contract differs')
 values.update(collision_solver_ran=collision_info['solver_ran'],collision_selected_cells=collision_info['selected_cells'])
 return dict(schema=SCHEMA,token=info['token'],scene_token=info['scene_token'],timestamp=int(info['timestamp']),planning=values)

def validate_record(value):
 if set(value)!={'schema','token','scene_token','timestamp','planning'} or value['schema']!=SCHEMA:raise ValueError('planning-only record schema differs')
 if not isinstance(value['token'],str) or not value['token'] or not isinstance(value['scene_token'],str) or not value['scene_token'] or type(value['timestamp']) is not int:raise ValueError('planning-only record identity differs')
 v=value['planning']
 if set(v)!={'raw','optimized','collision_solver_ran','collision_selected_cells'}:raise ValueError('planning record fields differ')
 record({'planning_raw':v['raw']},value,v['optimized'],dict(solver_ran=v['collision_solver_ran'],selected_cells=v['collision_selected_cells']))

def frozen_metrics(ev):
 """Select unchanged planning statements, including stock clone/mutation handling."""
 path=Path(ev.__file__);fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_formal_occ_planning_metrics');body=[];kept=[]
 names={'planning_metric','raw_flags','raw_available','planning_raw_metric','progress','planning','planning_raw'}
 for statement in fn.body:
  include=False
  if isinstance(statement,ast.Import):include=True
  elif isinstance(statement,ast.ImportFrom):include=not any(n.name in ('IntersectionOverUnion','PanopticMetric') for n in statement.names)
  elif isinstance(statement,ast.Assign):include=any(isinstance(t,ast.Name) and t.id in names for t in statement.targets)
  elif isinstance(statement,ast.If):include='raw_flags' in ast.unparse(statement.test) or ast.unparse(statement.test)=='planning_raw_metric is not None'
  elif isinstance(statement,ast.For) and isinstance(statement.target,ast.Tuple) and any(isinstance(v,ast.Name) and v.id=='record' for v in statement.target.elts):
   selected=copy.deepcopy(statement);start=next(i for i,v in enumerate(selected.body) if isinstance(v,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(n,ast.Name) and n.id=='gt_plan' for n in t.elts) for t in v.targets));prefix=[v for v in selected.body[:start] if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('sample','data') for t in v.targets)];selected.body=prefix+selected.body[start:];body.append(selected);kept.append(ast.dump(selected));continue
  if include:body.append(copy.deepcopy(statement));kept.append(ast.dump(statement))
 # No occupancy record or occupancy evaluator is needed by this extracted body.
 body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id='planning',ctx=ast.Load()),ast.Name(id='planning_raw',ctx=ast.Load())],ctx=ast.Load())))
 derived=copy.deepcopy(fn);derived.name='planning_metrics';derived.body=body;module=ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[]));text=ast.unparse(module)
 if 'record[\'occupancy\']' in text or 'IntersectionOverUnion' in text or 'PanopticMetric' in text:raise ValueError('planning extraction retained occupancy evaluator')
 namespace=dict(ev.__dict__);exec(compile(module,str(path),'exec'),namespace)
 binding=dict(source=str(path),source_sha256=ev.sha256(path),source_function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),selected_statements_sha256=hashlib.sha256('\n'.join(kept).encode()).hexdigest(),derived_function_ast_sha256=hashlib.sha256(ast.dump(derived).encode()).hexdigest(),scope='Unmodified frozen planning statements and real PlanningMetric; occupancy statements excluded, no new metric formulas.')
 return namespace['planning_metrics'],binding
