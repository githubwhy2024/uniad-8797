#!/usr/bin/env python3
"""Promote portable CPU planning only after both real declared-dataset task gates pass."""
import argparse
import ast
import copy
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from resources import terminal
from tool_run import ROOT,save,sha
from evaluate_qnn_saved import bind_reference
from qnn_mini_scope import require_complete,scope_of,check_counts
from planning_identity import verify_scored_bundle


def artifact(run,name):
    run=Path(run).absolute();tool=terminal(run);path=run/name
    if name not in tool['artifacts'] or sha(path)!=tool['artifacts'][name]['sha256']:
        raise ValueError('release prerequisite artifact binding differs: '+name)
    return json.loads(path.read_text())


def task_lane(mini_run,audit_run,evaluate_run,scope):
    mini=artifact(mini_run,'mini_result.json');profile=artifact(mini_run,'profile.json');audit=artifact(audit_run,'mini_audit.json');task=artifact(evaluate_run,'task_result.json')
    selected=require_complete(mini,audit,profile);standard,raw_expected=check_counts(selected,scope)
    if scope_of(task)!=selected or mini['task_scope']!=scope:
        raise ValueError('release requires corresponding actual declared mini dataset')
    if profile['backend']['type']!='QNN_CPU' or profile['backend']['precision']!='float32' or len(profile['abi']['outputs'])!=(25 if scope=='planning' else 47):
        raise ValueError('release corresponding actual backend/ABI differs')
    if sha(mini_run/'profile.json')!=mini['profile_sha256'] or audit['status']!='pass' or audit['frames']!=selected['frames'] or not all(audit['checks'].values()):
        raise ValueError('release independent mini audit incomplete')
    if audit['mini_result_sha256']!=sha(mini_run/'mini_result.json') or audit['profile_sha256']!=mini['profile_sha256'] or audit['manifest_sha256']!=sha(mini_run/'manifest.json') or audit['journal_sha256']!=sha(mini_run/'mini_progress.json'):
        raise ValueError('release mini audit source binding differs')
    if task['status']!='pass' or task['execution_status']!='complete' or task['overall_pass'] is not True or task['standard_checks']!=standard or task['raw_planning_checks']!=raw_expected:
        raise ValueError('release corresponding frozen task acceptance incomplete')
    if task['mini_result_sha256']!=sha(mini_run/'mini_result.json') or task['mini_audit_sha256']!=sha(audit_run/'mini_audit.json') or (scope=='planning' and task['task_scope']!='planning'):
        raise ValueError('release task/mini lineage differs')
    reference=bind_reference()
    if task['reference']!=reference:raise ValueError('release frozen reference binding differs')
    tree=ast.parse((ROOT/'onnx/fixed/validate.py').read_text());constants={target.id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) for target in node.targets if isinstance(target,ast.Name) and target.id in ('FINAL_ACCEPTANCE_POLICY','FINAL_ACCEPTANCE_POLICY_VERSION')}
    policy=constants['FINAL_ACCEPTANCE_POLICY'];expected_policy={key:policy['planning'][key] for key in ('L2','obj_col','obj_box_col')} if scope=='planning' else policy
    if set(task['comparisons'])!={backend+'/'+split for backend in ('native','pt','ort') for split in selected['splits']}:
        raise ValueError('release frozen comparison coverage differs')
    standard_count=raw_count=0
    for key,row in task['comparisons'].items():
        if sha(row['path'])!=row['sha256'] or row['overall_pass'] is not True:raise ValueError('release comparison artifact differs')
        report=json.loads(Path(row['path']).read_text())
        raw_checks=report['raw_checks'] if scope=='planning' else report['planning_raw_supplement']['checks']
        accepted=report['status']=='pass' if scope=='planning' else report['pass'] is True
        if not accepted or report['overall_pass'] is not True or not all(check['pass'] for check in report['checks']+raw_checks):raise ValueError('release individual frozen task checks differ')
        if report['policy']!=expected_policy or report['policy_version']!=constants['FINAL_ACCEPTANCE_POLICY_VERSION']:raise ValueError('release frozen metric policy differs')
        split=key.split('/')[1];evaluation=task['reports'][split]
        if sha(evaluation['path'])!=evaluation['sha256']:raise ValueError('release scored report binding differs')
        if scope=='planning':
            if report['reference_sha256']!=reference['references'][key]['sha256'] or report['candidate_sha256']!=evaluation['sha256']:raise ValueError('release planning comparison provenance differs')
        elif report['native_evaluation']!=reference['references'][key]['path'] or report['candidate_evaluation']!=evaluation['path'] or report['candidate_backend']!='q4_qnn_partition_cpu_float32':raise ValueError('release six-task comparison provenance differs')
        standard_count+=len(report['checks']);raw_count+=len(raw_checks)
    if standard_count!=standard or raw_count!=raw_expected:raise ValueError('release check count differs')
    return dict(frames=selected['frames'],dataset_scope=selected,standard_checks=standard,raw_planning_checks=raw_expected,
        mini_result_sha256=sha(mini_run/'mini_result.json'),mini_audit_sha256=sha(audit_run/'mini_audit.json'),
        task_result_sha256=sha(evaluate_run/'task_result.json'),profile_sha256=mini['profile_sha256'],native_model_sha256=profile['assets']['native_model']['sha256']),profile


def main():
    p=argparse.ArgumentParser()
    for name in ('package-run','resource-run','neural-run','planning-mini','planning-audit','planning-evaluate','full-mini','full-audit','full-evaluate'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    # Gate before creating/copying a release directory. Never promote Q3 or a subset.
    planning,profile=task_lane(a.planning_mini,a.planning_audit,a.planning_evaluate,'planning')
    full,_=task_lane(a.full_mini,a.full_audit,a.full_evaluate,'full')
    if full['dataset_scope']!=planning['dataset_scope']:raise ValueError('release baseline and final task dataset scope differs')
    package=artifact(a.package_run,'planning_package.json');resources=artifact(a.resource_run,'planning_relocation_control.json');neural=artifact(a.neural_run,'planning_neural_relocation.json')
    source=Path(package['bundle']);pin=package['manifest_sha256']
    if sha(source/'manifest.json')!=pin or resources['status']!='pass' or neural['status']!='pass' or resources['manifest_sha256']!=pin or neural['manifest_sha256']!=pin or not all(resources['checks'].values()) or not all(neural['checks'].values()) or len(neural['frames'])!=6:
        raise ValueError('release corresponding actual portable resource/neural gates incomplete')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['schema']!='qnn-planning-host-candidate-v1' or manifest['task_acceptance']!='not_evaluated' or manifest['lineage']['native_model_sha256']!=planning['native_model_sha256'] or manifest['profile']['abi']!=profile['abi']:
        raise ValueError('release candidate/source planning model differs')
    if [v['library_sha256'] for v in manifest['profile']['parts']]!=[v['library_sha256'] for v in profile['parts']]:raise ValueError('release actual model libraries differ from scored planning25')
    for role,source_role in [('bridge','bridge'),('backend_lib','backend_lib'),('host','host'),('state_contract','state_contract'),('initial_state','initial_state'),('collision_optimizer','collision_optimizer'),('session_adapter','adapter')]:
        if manifest['files'][role]['sha256']!=profile['assets'][source_role]['sha256']:raise ValueError('release scored runtime resource differs: '+role)
    if manifest.get('transport_policy')!=profile.get('transport_policy'):raise ValueError('release candidate/scored input transport policy differs')
    if profile.get('transport_policy') is not None:
        role='borrowed_input_adapter'
        if role not in manifest['files'] or manifest['files'][role]['sha256']!=profile['assets'][role]['sha256']:raise ValueError('release scored borrowed input adapter differs')
    binding=verify_scored_bundle(source,manifest,profile,a.planning_mini)
    destination=Path.cwd()/'bundle';shutil.copytree(source,destination,symlinks=False)
    current_loader=ROOT/'onnx/qnn/planning_bundle.py';target=destination/manifest['files']['loader']['path'];shutil.copyfile(current_loader,target);manifest['files']['loader'].update(sha256=sha(target),bytes=target.stat().st_size)
    manifest.update(schema='qnn-planning-host-release-v1',purpose='planning_host_release',task_acceptance='accepted',
        acceptance=dict(dataset_scope=planning['dataset_scope'],execution_identity=binding,planning=planning,full=full,package=dict(frames=6,candidate_manifest_sha256=pin,resource_control_sha256=sha(a.resource_run/'planning_relocation_control.json'),neural_control_sha256=sha(a.neural_run/'planning_neural_relocation.json'))),
        scope='Current-host x86 FP32 QNN CPU planning release backed by both actual declared-dataset task gates and candidate portable six frames. Release-pin relocation still requires its own validation; board work is Q5.')
    manifest['lineage'].update(candidate_manifest_sha256=pin,release_builder_sha256=sha(__file__))
    save(destination/'manifest.json',manifest)
    save(Path.cwd()/'planning_package.json',dict(status='pass',bundle=str(destination),manifest_sha256=sha(destination/'manifest.json'),parts=len(manifest['profile']['parts']),files=len(manifest['files']),lineage=manifest['lineage'],task_acceptance='accepted',acceptance=manifest['acceptance'],scope='Release assembly after actual task/candidate gates only; final release-pin outside-cwd resource/neural validation required.'))


if __name__=='__main__':main()
