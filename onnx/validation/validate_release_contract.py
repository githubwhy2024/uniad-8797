#!/usr/bin/env python3
"""Constructed release-loader envelopes and real missing-task promotion rejection."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from resources import terminal
from tool_run import ROOT,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-run',type=Path,required=True);a=p.parse_args();tool=terminal(a.package_run);record=json.loads((a.package_run/'planning_package.json').read_text())
    if sha(a.package_run/'planning_package.json')!=tool['artifacts']['planning_package.json']['sha256']:raise ValueError('release contract fixture source differs')
    source=Path(record['bundle']);m=json.loads((source/'manifest.json').read_text())
    if sha(source/'manifest.json')!=record['manifest_sha256']:raise ValueError('release contract fixture pin differs')
    external=Path(tempfile.mkdtemp(prefix='q4-release-contract-'));checks={};cases=[]
    def clone(label,manifest):
        root=external/label;shutil.copytree(source,root,copy_function=os.link,symlinks=False);value=copy.deepcopy(manifest);path=root/value['files']['loader']['path'];path.unlink();shutil.copyfile(ROOT/'onnx/qnn/planning_bundle.py',path);value['files']['loader'].update(sha256=sha(path),bytes=path.stat().st_size);save(root/'manifest.json',value);return root,sha(root/'manifest.json')
    def load(label,manifest,candidate,expect_success):
        root,pin=clone(label,manifest);cwd=external/(label+'-cwd');cwd.mkdir();code='import sys;sys.path.insert(0,sys.argv[1]);from planning_bundle import load_planning_bundle;load_planning_bundle(sys.argv[2],sys.argv[3],allow_candidate='+str(candidate)+')'
        env=os.environ.copy();env.update(PYTHONPATH='',LD_LIBRARY_PATH=str(root/'lib'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
        with (cwd/'worker.log').open('wb') as log:result=subprocess.run([sys.executable,'-c',code,str(root/'runtime'),str(root),pin],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        checks[label]=(result.returncode==0)==expect_success;cases.append(dict(case=label,expected_success=expect_success,exit_code=result.returncode,manifest_sha256=pin,log=str(cwd/'worker.log'),log_sha256=sha(cwd/'worker.log')))
    load('current_candidate_explicit',m,True,True);load('current_candidate_no_opt_in',m,False,False)
    # These summaries are deliberately constructed contract fixtures. No actual
    # mini/task release is claimed or created by this validation.
    release=copy.deepcopy(m);digest=hashlib.sha256(b'constructed loader evidence only').hexdigest()
    def lane(standard):return dict(frames=404,standard_checks=standard,raw_planning_checks=12,mini_result_sha256=digest,mini_audit_sha256=digest,task_result_sha256=digest,profile_sha256=digest,native_model_sha256=m['lineage']['native_model_sha256'])
    release.update(schema='qnn-planning-host-release-v1',purpose='planning_host_release',task_acceptance='accepted',acceptance=dict(planning=lane(18),full=lane(192),package=dict(frames=6,candidate_manifest_sha256=record['manifest_sha256'],resource_control_sha256=digest,neural_control_sha256=digest)))
    load('constructed_release_envelope',release,False,True)
    for label in ['release_decision','planning_count','full_count','raw_count','frames','portable_count','evidence_pin','native_model']:
        bad=copy.deepcopy(release)
        if label=='release_decision':bad['task_acceptance']='not_evaluated'
        elif label=='planning_count':bad['acceptance']['planning']['standard_checks']=17
        elif label=='full_count':bad['acceptance']['full']['standard_checks']=191
        elif label=='raw_count':bad['acceptance']['planning']['raw_planning_checks']=11
        elif label=='frames':bad['acceptance']['planning']['frames']=6
        elif label=='portable_count':bad['acceptance']['package']['frames']=2
        elif label=='evidence_pin':bad['acceptance']['planning']['task_result_sha256']='missing'
        elif label=='native_model':bad['acceptance']['planning']['native_model_sha256']='0'*64
        load(label,bad,False,False)
    from qnn_mini_scope import dataset_scope
    val=copy.deepcopy(release);declared=dataset_scope('mini_val');val['acceptance']['dataset_scope']=declared
    for role,standard in [('planning',9),('full',96)]:val['acceptance'][role].update(dataset_scope=declared,frames=81,standard_checks=standard,raw_planning_checks=6)
    load('constructed_val81_release',val,False,True)
    for label in ['val81_frames80','val81_scope_missing','val81_scope_malformed','val81_role_scope_mixed','val81_planning_wrong_count','val81_full_wrong_count','val81_raw_wrong_count']:
        bad=copy.deepcopy(val)
        if label=='val81_frames80':bad['acceptance']['planning']['frames']=80
        elif label=='val81_scope_missing':bad['acceptance'].pop('dataset_scope')
        elif label=='val81_scope_malformed':bad['acceptance']['dataset_scope']['frames']=404
        elif label=='val81_role_scope_mixed':bad['acceptance']['full']['dataset_scope']=dataset_scope()
        elif label=='val81_planning_wrong_count':bad['acceptance']['planning']['standard_checks']=18
        elif label=='val81_full_wrong_count':bad['acceptance']['full']['standard_checks']=192
        elif label=='val81_raw_wrong_count':bad['acceptance']['planning']['raw_planning_checks']=12
        load(label,bad,False,False)
    cwd=external/'real_promotion_without_tasks';cwd.mkdir();command=[sys.executable,str(ROOT/'onnx/validation/release_planning.py')]
    for name,path in [('package-run',a.package_run),('resource-run',ROOT/'onnx/runs/q4-planning-host-relocation-ed31aeee'),('neural-run',external/'not-yet-neural'),('planning-mini',external/'not-yet-planning-mini'),('planning-audit',external/'not-yet-planning-audit'),('planning-evaluate',external/'not-yet-planning-evaluate'),('full-mini',external/'not-yet-full-mini'),('full-audit',external/'not-yet-full-audit'),('full-evaluate',external/'not-yet-full-evaluate')]:command+=['--'+name,str(path)]
    with (cwd/'worker.log').open('wb') as log:result=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    checks['actual_pending_task_promotion_rejected']=result.returncode!=0 and not (cwd/'bundle').exists() and not (cwd/'planning_package.json').exists()
    checks['frozen_candidate_manifest_unchanged']=sha(source/'manifest.json')==record['manifest_sha256']
    save(Path.cwd()/'release_contract_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,cases=cases,external_root=str(external),loader_sha256=sha(ROOT/'onnx/qnn/planning_bundle.py'),promoter_sha256=sha(ROOT/'onnx/validation/release_planning.py'),scope='Constructed loader envelope/count/model/pin controls and actual pending-task promotion rejection only; no QNN task/release acceptance.'))
    if not all(checks.values()):raise ValueError('release contract control failed')


if __name__=='__main__':main()
