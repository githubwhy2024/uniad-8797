#!/usr/bin/env python3
"""Real micro-child controls for preflight and completed task declarations."""
import argparse,json,secrets,subprocess,sys
from pathlib import Path
import onnx
from onnx import helper as H
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import ROOT,save,sha


def main():
 model=H.make_model(H.make_graph([H.make_node('Identity',['x'],['y'])],'micro',[H.make_tensor_value_info('x',onnx.TensorProto.FLOAT,[1])],[H.make_tensor_value_info('y',onnx.TensorProto.FLOAT,[1])]),opset_imports=[H.make_opsetid('',13)]);path=Path.cwd()/'model.onnx';onnx.save(model,str(path));checks={};cases=[]
 rows=[('illegal_control','control','decision.json',dict(status='pass',execution_status='complete',overall_pass=True),False),('illegal_escape','execute','../decision.json',{},False),('task_pass','execute','decision.json',dict(status='pass',execution_status='complete',overall_pass=True),True),('task_rejected','execute','decision.json',dict(status='failed_acceptance',execution_status='complete',overall_pass=False),True),('contradictory_task','execute','decision.json',dict(status='pass',execution_status='complete',overall_pass=False),True),('control_only','control',None,{},True)]
 for label,stage,name,claim,launch in rows:
  run=Path.cwd()/label;code="from pathlib import Path;import json;Path('started').write_text('started');Path('decision.json').write_text("+repr(json.dumps(claim))+')'
  command=[sys.executable,str(ROOT/'onnx/qnn/tool_run.py'),'--run-dir',str(run),'--stage',stage,'--model',str(path),'--model-sha256',sha(path)]
  if name:command+=['--acceptance-result',name]
  command+=['--',sys.executable,'-c',code]
  done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);result=json.loads((run/'status.json').read_text());expected='pass' if label in ('task_pass','control_only') else 'failed_acceptance' if label=='task_rejected' else 'failed'
  checks[label]=result['status']==expected and (run/'started').exists()==launch and (result['child'] is not None)==launch
  if label=='task_rejected':checks[label]=checks[label] and result['execution_status']=='complete' and result['acceptance_status']=='task_rejected'
  if label=='control_only':checks[label]=checks[label] and result['overall_pass'] is None
  cases.append(dict(case=label,returncode=done.returncode,status=result['status'],child_launched=(run/'started').exists(),result_sha256=sha(run/'result.json')))
 save(Path.cwd()/'tool_acceptance_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,tool_sha256=sha(ROOT/'onnx/qnn/tool_run.py'),scope='Actual tiny child processes and malformed-declaration preflight; no model inference or real task decisions.'))
 if not all(checks.values()):raise ValueError('tool acceptance declaration control failed')
if __name__=='__main__':main()
