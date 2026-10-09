#!/usr/bin/env python3
"""Exercise the mini journal's durable-prefix rejection controls; no neural run."""
import argparse,copy,json,os,pickle,sys
from pathlib import Path
import numpy as np
from run_qnn_mini import resume_journal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from tool_run import sha,save,process_identity

def main():
 root=Path.cwd();checks={};sequence=[('mini_val','token0'),('mini_val','token1'),('mini_val','token2')]
 def fixture(label):
  prior=root/label;prior.mkdir();save(prior/'profile.json',dict(schema='constructed-resume-control'));pin=sha(prior/'profile.json');save(prior/'status.json',dict(status='constructed_fixture_no_neural_acceptance'));rows=[]
  for i in range(2):
   record=prior/f'frame{i}.record.pkl'
   with record.open('wb') as stream:pickle.dump(dict(token='token'+str(i),scene_token='scene'),stream)
   frame=prior/f'frame{i}.json';save(frame,dict(token='token'+str(i)));checkpoint=prior/f'frame{i}.npz';np.savez(checkpoint,fixture=np.array([i],np.int64))
   row=dict(sequence_index=i,split='mini_val',token='token'+str(i),scene_token='scene',before_state_sha256='initial' if i==0 else 'state0',state_sha256='state'+str(i))
   for name,path in [('record',record),('frame',frame),('checkpoint',checkpoint)]:row.update({name+'_path':str(path),name+'_sha256':sha(path)})
   rows.append(row)
  progress=dict(profile_sha256=pin,reference_sha256='reference',committed_frames=rows);save(prior/'mini_progress.json',progress);return prior,pin,progress
 for label in ('valid','wrong_pin','wrong_reference','empty','already_finished','wrong_token','wrong_index','broken_state_chain','record_tampered','checkpoint_tampered','frame_tampered','record_identity','active_process'):
  prior,pin,progress=fixture(label);requested=3;ref='reference'
  if label=='wrong_pin':pin='0'*64
  if label=='wrong_reference':ref='different'
  if label=='empty':progress['committed_frames']=[]
  if label=='already_finished':requested=2
  if label=='wrong_token':progress['committed_frames'][1]['token']='other'
  if label=='wrong_index':progress['committed_frames'][1]['sequence_index']=99
  if label=='broken_state_chain':progress['committed_frames'][1]['before_state_sha256']='other'
  if label.endswith('_tampered'):
   kind=label.split('_')[0];path=Path(progress['committed_frames'][1][kind+'_path']);path.write_bytes(b'changed')
  if label=='record_identity':
   path=Path(progress['committed_frames'][1]['record_path'])
   with path.open('wb') as stream:pickle.dump(dict(token='other',scene_token='scene'),stream)
   progress['committed_frames'][1]['record_sha256']=sha(path)
  if label=='active_process':save(prior/'status.json',dict(status='constructed_fixture_no_neural_acceptance',child=process_identity(os.getpid())))
  save(prior/'mini_progress.json',progress)
  try:actual=resume_journal(prior,pin,ref,requested,sequence)
  except (ValueError,FileNotFoundError):checks[label]=label!='valid'
  else:checks[label]=label=='valid' and actual==progress['committed_frames']
 save(root/'mini_resume_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,validator_sha256=sha(__file__),mini_driver_sha256=sha(Path(__file__).with_name('run_qnn_mini.py')),scope='Constructed durable token/state/artifact/source prefix rejection only; no valid neural checkpoint, graph execution, mini404 or task acceptance.'))
 if not all(checks.values()):raise ValueError('mini resume controls failed')
if __name__=='__main__':main()
