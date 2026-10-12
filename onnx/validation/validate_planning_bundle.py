#!/usr/bin/env python3
"""Relocate real planning candidate resources; check Host/solver without neural work."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qnn'))
from resources import terminal
from tool_run import ROOT, sha, save


WORKER = '''import copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,sys.argv[1])
import planning_bundle as loader
root,pin,fixture,reference,report=map(str,sys.argv[2:7])
stream=loader.create_planning_stream(root,pin,max_state_gap_seconds=1.,max_consecutive_failures=2,allow_candidate=True)
runtime=stream.runtime
before=loader.host.state_digest(runtime.state);context=copy.deepcopy(runtime.metadata)
checks={}
try:
 checkpoint=Path.cwd()/'initial.checkpoint.npz';runtime.save_checkpoint(checkpoint);runtime.load_checkpoint(checkpoint)
 checks['initial_checkpoint_roundtrip']=loader.host.state_digest(runtime.state)==before and runtime.metadata==context
 with np.load(checkpoint,allow_pickle=False) as arc:state={n:arc[n].copy() for n in arc.files}
 meta=json.loads(str(state['metadata'].item()));meta['planning_bundle_sha256']='0'*64;state['metadata']=np.array(json.dumps(meta));bad=Path.cwd()/'foreign.checkpoint.npz';np.savez_compressed(bad,**state)
 try:runtime.load_checkpoint(bad);checks['foreign_manifest_checkpoint_rejected']=False
 except ValueError:checks['foreign_manifest_checkpoint_rejected']=loader.host.state_digest(runtime.state)==before and runtime.metadata==context
 bad_frame=dict(scene_token='control',timestamp=0.,image=np.zeros((1,),np.float32),can_bus_absolute=np.zeros((18,),np.float32),l2g_r=np.eye(3,dtype=np.float32),l2g_t=np.zeros((3,),np.float32),lidar2img=np.zeros((1,6,4,4),np.float32),img_shape=np.full((1,6,2),[928,1600],np.int64),command=0)
 result=stream.process(frame_id='rejected-control',**bad_frame)
 checks['invalid_frame_rejected_before_native']=result['status']=='rejected' and not result['valid'] and result['plan'] is None and loader.host.state_digest(runtime.state)==before and runtime.metadata==context and runtime.session.native_abi['parts']==[]
 assets=runtime.session.manifest['assets']
 with np.load(fixture,allow_pickle=False) as arc:raw=arc['planning_raw'].copy();occ=arc['occ_segmentation'].copy()
 final,info=loader.host.optimize_planning(raw,occ,coordinate_mode='legacy_int',optimizer_source=assets['collision_optimizer']['path'])
 checks['actual_copied_casadi_solver']=info['solver_ran'] is True and info['selected_cells']>0
 checks['final_plan_contract']=final.dtype==np.float32 and final.shape==(1,6,2) and np.isfinite(final).all()
 with np.load(reference,allow_pickle=False) as arc:expected=arc['planning_final'].copy()
 difference=dict(max_abs=float(np.max(np.abs(final-expected))),mean_abs=float(np.mean(np.abs(final-expected))))
 checks['no_developer_ml_imports']=not any(n in sys.modules for n in ('onnx','onnxruntime','torch','mmcv','mmdet'))
 checks['no_neural_session_loaded']=runtime.session.native_abi['parts']==[]
 checks['copied_runtime_origins']=all(Path(p.__file__).resolve().is_relative_to(Path(root)) for p in (loader,loader.host,loader.session,loader.partition_runtime,loader.state_contract,loader.native_bundle))
 checks={key:bool(value) for key,value in checks.items()}
 Path(report).write_text(json.dumps(dict(status='pass' if all(checks.values()) else 'failed',checks=checks,planning_info=info,final_plan_difference=difference,initial_state_digest=before,manifest_sha256=pin,scope='Relocated learned initialization/checkpoint/Host guards and actual copied CasADi only; no neural/task/package release acceptance.'),indent=2)+'\\n')
 if not all(checks.values()):raise ValueError('planning resource/Host control failed')
finally:runtime.session.close()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-run', type=Path, required=True)
    parser.add_argument('--short-run', type=Path, required=True)
    args = parser.parse_args()
    tool = terminal(args.package_run)
    record = json.loads((args.package_run / 'planning_package.json').read_text())
    if sha(args.package_run / 'planning_package.json') != tool['artifacts']['planning_package.json']['sha256'] or record['status'] != 'pass':
        raise ValueError('planning candidate assembly evidence differs')
    short_tool = terminal(args.short_run)
    short = json.loads((args.short_run / 'neural_result.json').read_text())
    if sha(args.short_run / 'neural_result.json') != short_tool['artifacts']['neural_result.json']['sha256'] or short['status'] != 'pass':
        raise ValueError('real planning prefix evidence differs')
    frame = next(row for row in short['completed_frames'] if row['planning_info']['solver_ran'])
    for key in ('outputs', 'final_plan'):
        if sha(frame[key + '_path']) != frame[key + '_sha256']:
            raise ValueError('actual solver fixture binding differs')
    source = Path(record['bundle'])
    pin = record['manifest_sha256']
    if sha(source / 'manifest.json') != pin:
        raise ValueError('planning package manifest changed')
    external = Path(tempfile.mkdtemp(prefix='q4-planning-host-relocation-'))
    bundle = external / 'candidate'
    shutil.copytree(source, bundle, symlinks=False)
    cwd = external / 'outside'
    cwd.mkdir()
    fixture, reference = cwd / 'solver_inputs.npz', cwd / 'expected_final.npz'
    with np.load(frame['outputs_path'], allow_pickle=False) as arc:
        np.savez(fixture, planning_raw=arc['planning_raw'], occ_segmentation=arc['occ_segmentation'])
    shutil.copyfile(frame['final_plan_path'], reference)
    worker = cwd / 'worker.py'
    worker.write_text(WORKER)
    env = os.environ.copy()
    for name in ('QNN_SDK_ROOT', 'QAIRT_SDK_ROOT', 'QNN_SDK_PATH', 'QAIRT_SDK', 'QNN_CONVERTER_ENV'):
        env.pop(name, None)
    env.update(PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    checks, cases = {}, []

    def execute(root, digest, name, preflight=False, candidate=True):
        run = cwd / name
        run.mkdir()
        report = run / 'report.json'
        if preflight:
            code = 'import sys;sys.path.insert(0,sys.argv[1]);from planning_bundle import load_planning_bundle;load_planning_bundle(sys.argv[2],sys.argv[3],allow_candidate=' + str(candidate) + ')'
            command = [sys.executable, '-c', code, str(root / 'runtime'), str(root), digest]
        else:
            command = [sys.executable, str(worker), str(root / 'runtime'), str(root), digest, str(fixture), str(reference), str(report)]
        with (run / 'worker.log').open('wb') as log:
            result = subprocess.run(command, cwd=run, env={**env, 'LD_LIBRARY_PATH': str(root / 'lib')}, stdout=log, stderr=subprocess.STDOUT)
        return result.returncode, report, run / 'worker.log'

    code, report, log = execute(bundle, pin, 'valid')
    checks['actual_outside_cwd_host_solver']=code == 0
    if report.exists():
        actual = json.loads(report.read_text())
        checks.update(actual['checks'])
    checks['bundle_files_ordinary'] = all(path.resolve() == path for path in bundle.rglob('*'))
    checks['no_developer_runtime_paths'] = not any(value in (bundle / 'manifest.json').read_text() for value in (str(ROOT), str(Path.home()), '/opt/qcom'))
    is_release = json.loads((source / 'manifest.json').read_text())['purpose'] == 'planning_host_release'
    labels = ['wrong_pin', 'purpose', 'backend', 'casadi_version', 'policy', 'capacity',
              'missing_solver', 'changed_solver', 'absolute_path', 'escape_path', 'duplicate_file', 'file_symlink', 'root_symlink', 'initial_digest']
    if is_release:
        code, _, _ = execute(bundle, pin, 'release_without_candidate_opt_in', preflight=True, candidate=False)
        checks['accepted_release_without_candidate_opt_in'] = code == 0
        labels += ['release_task_count', 'release_model', 'release_decision']
    else:
        labels += ['candidate_not_authorized']
    for label in labels:
        clone = external / label
        # Ordinary hardlinks keep large immutable libraries local; every mutation
        # unlinks/replaces its destination first, preserving source/candidate bytes.
        shutil.copytree(source, clone, copy_function=os.link, symlinks=False)
        manifest = json.loads((clone / 'manifest.json').read_text())
        digest, modified = pin, False
        solver = clone / manifest['files']['collision_optimizer']['path']
        if label == 'wrong_pin': digest = '0' * 64
        elif label == 'purpose': manifest['purpose'] = 'unsupported_planning_purpose'; modified = True
        elif label == 'backend': manifest['backend']['type'] = 'HTP'; modified = True
        elif label == 'casadi_version': manifest['runtime_versions']['casadi'] = 'wrong'; modified = True
        elif label == 'policy': manifest['host_policy']['can_bus_mode'] = 'scene_reset'; modified = True
        elif label == 'capacity': manifest['capacities']['track'] -= 1; modified = True
        elif label == 'missing_solver': solver.unlink()
        elif label == 'changed_solver': data=solver.read_bytes(); solver.unlink(); solver.write_bytes(data + b'\n# changed\n')
        elif label == 'absolute_path': manifest['files']['collision_optimizer']['path'] = '/tmp/optimizer.py'; modified = True
        elif label == 'escape_path': manifest['files']['collision_optimizer']['path'] = '../optimizer.py'; modified = True
        elif label == 'duplicate_file': manifest['files']['duplicate'] = manifest['files']['host'].copy(); modified = True
        elif label == 'file_symlink': solver.unlink(); solver.symlink_to(source / manifest['files']['collision_optimizer']['path'])
        elif label == 'root_symlink': link=external/'root-alias'; link.symlink_to(clone,target_is_directory=True); clone=link
        elif label == 'initial_digest': manifest['initial_state_digest'] = '0' * 64; modified = True
        elif label == 'release_task_count': manifest['acceptance']['planning']['standard_checks'] = 17; modified = True
        elif label == 'release_model': manifest['acceptance']['planning']['native_model_sha256'] = '0' * 64; modified = True
        elif label == 'release_decision': manifest['task_acceptance'] = 'not_evaluated'; modified = True
        if modified:
            save(clone / 'manifest.json', manifest)
            digest = sha(clone / 'manifest.json')
        code, path, case_log = execute(clone, digest, label, preflight=True, candidate=label != 'candidate_not_authorized')
        checks[label + '_rejected'] = code != 0
        cases.append(dict(case=label, exit_code=code, log=str(case_log), log_sha256=sha(case_log)))
    checks['source_manifest_unchanged'] = sha(source / 'manifest.json') == pin
    save(Path.cwd() / 'planning_relocation_control.json', dict(status='pass' if all(checks.values()) else 'failed',
        checks=checks, cases=cases, manifest_sha256=pin, actual_report=str(report),
        actual_report_sha256=sha(report) if report.exists() else None, external_root=str(external),
        solver_fixture=dict(source_outputs_sha256=frame['outputs_sha256'], source_final_sha256=frame['final_plan_sha256'],
            fixture_sha256=sha(fixture), reference_sha256=sha(reference)),
        scope='Actual relocated Host/state/checkpoint/copy-of-CasADi and candidate boundary controls only; neural/task/release/board acceptance absent.'))
    if not all(checks.values()):
        raise ValueError('planning candidate relocation resource control failed')


if __name__ == '__main__':
    main()
