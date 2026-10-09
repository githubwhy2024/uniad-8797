"""Malformed native-layout policies must fail before converter launch."""
import json,subprocess,sys
from pathlib import Path
import onnx
from onnx import helper as H
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from tool_run import sha,save


def main():
 checks={};cases=[];base=dict(model='NCHW',native='NHWC')
 rows=[('hash','convert',[1,3,2,4],onnx.TensorProto.FLOAT,{'x':base},None,None,True),('stage','control',[1,3,2,4],onnx.TensorProto.FLOAT,{'x':base},None,None,False),('unknown_port','convert',[1,3,2,4],onnx.TensorProto.FLOAT,{'unknown':base},None,None,False),('rank','convert',[2,3],onnx.TensorProto.FLOAT,{'x':base},None,None,False),('integer','convert',[1,3,2,4],onnx.TensorProto.INT32,{'x':base},None,None,False),('unsupported_layout','convert',[1,3,2,4],onnx.TensorProto.FLOAT,{'x':dict(model='NCHW',native='NCHW')},None,None,False),('input_conflict','convert',[1,3,2,4],onnx.TensorProto.FLOAT,{'x':base},{'x':'NHWC'},None,False),('output_conflict','convert',[1,3,2,4],onnx.TensorProto.FLOAT,{'y':base},None,{'y':'NHWC'},False)]
 for label,stage,shape,dtype,policy,inputs,outputs,bad_hash in rows:
  model=H.make_model(H.make_graph([H.make_node('Identity',['x'],['y'])],'layout_control',[H.make_tensor_value_info('x',dtype,shape)],[H.make_tensor_value_info('y',dtype,shape)]),opset_imports=[H.make_opsetid('',13)]);path=Path.cwd()/(label+'.onnx');onnx.save(model,str(path));policy_path=Path.cwd()/(label+'.json');save(policy_path,policy);run=Path.cwd()/label
  command=[sys.executable,str(ROOT/'onnx/qnn/tool_run.py'),'--run-dir',str(run),'--stage',stage,'--model',str(path),'--model-sha256',sha(path),'--native-layouts',str(policy_path),'--native-layouts-sha256','0'*64 if bad_hash else sha(policy_path)]
  for role,value in [('input',inputs),('output',outputs)]:
   if value:
    source=Path.cwd()/(label+'.'+role+'.json');save(source,value);command+=['--'+role+'-layouts',str(source),'--'+role+'-layouts-sha256',sha(source)]
  completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);record=json.loads((run/'result.json').read_text());status=json.loads((run/'status.json').read_text());checks[label]=completed.returncode!=0 and status['status']=='failed' and status['child'] is None and not (run/'launch.json').exists();cases.append(dict(case=label,result_sha256=sha(run/'result.json'),preflight_rejected=checks[label]))
 save(Path.cwd()/'native_layout_policy_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,tool_sha256=sha(ROOT/'onnx/qnn/tool_run.py'),scope='Eight invalid policy controls, no converter child/inference/environment change; valid compiled pair controls separate.'))
 if not all(checks.values()):raise ValueError('native-layout policy preflight failed')


if __name__=='__main__':main()
