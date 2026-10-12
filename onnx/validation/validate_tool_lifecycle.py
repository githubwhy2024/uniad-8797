#!/usr/bin/env python3
"""Validate real process failure/SIGTERM journaling; no QNN neural acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import onnx
from onnx import helper


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = args.run_dir.absolute()
    if not run.is_relative_to(root / 'onnx/runs') or run.resolve() != run:
        raise ValueError('control evidence must be in ordinary Q4 runs path')
    run.mkdir(parents=True, exist_ok=False)
    runner = root / 'onnx/qnn/tool_run.py'
    x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2])
    model = helper.make_model(helper.make_graph([helper.make_node('Identity', ['x'], ['y'])],
                                               'lifecycle-only', [x], [y]),
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    path = run / 'control.onnx'
    onnx.save(model, str(path))
    checks = {}
    attempts = {}
    for case in ('success', 'nonzero', 'bad_identity', 'sigterm', 'task_pass', 'task_rejected', 'task_inconsistent'):
        target = run / case
        digest = '0' * 64 if case == 'bad_identity' else sha(path)
        code = 'from pathlib import Path; Path("marker.json").write_text("{}")'
        if case == 'nonzero':
            code = 'raise SystemExit(7)'
        elif case == 'sigterm':
            code = 'import time; time.sleep(180)'
        command = [sys.executable, str(runner), '--run-dir', str(target), '--stage', 'control',
                   '--model', str(path), '--model-sha256', digest]
        if case == 'success':
            command += ['--expect', 'marker.json']
        if case.startswith('task_'):
            claim=dict(status='pass' if case!='task_rejected' else 'failed_acceptance',execution_status='complete',overall_pass=case=='task_pass',scope='Constructed process acceptance-decision control only; no QNN metrics.')
            code='from pathlib import Path; Path("task.json").write_text('+repr(json.dumps(claim))+')'
            command[command.index('control')]='execute'
            command += ['--acceptance-result','task.json']
        command += ['--', sys.executable, '-c', code]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        launched = None
        if case == 'sigterm':
            launched = json.loads(proc.stdout.readline())
            os.kill(proc.pid, signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)
        result = json.loads((target / 'result.json').read_text())
        status = json.loads((target / 'status.json').read_text())
        expected = 'pass' if case in ('success','task_pass') else 'failed_acceptance' if case=='task_rejected' else 'interrupted' if case == 'sigterm' else 'failed'
        checks[case + '_status'] = result['status'] == status['status'] == expected
        checks[case + '_exit'] = (proc.returncode == 0) == (case in ('success','task_pass'))
        checks[case + '_terminal_identity'] = status['stage'] == 'complete' and status['result_sha256'] == sha(target / 'result.json')
        if case in ('task_pass','task_rejected'):
            checks[case+'_completed_decision_retained']=result['execution_status']=='complete' and result['tool_exit_code']==0 and result['overall_pass']==(case=='task_pass') and result['artifacts']['task.json']['sha256']==sha(target/'task.json')
        else:checks[case + '_not_task_acceptance'] = result['overall_pass'] is None
        if case == 'sigterm':
            try:
                stat = Path('/proc', str(launched['child']['pid']), 'stat').read_text().split(') ', 1)[1].split()
                alive = int(stat[19]) == launched['child']['startticks'] and stat[0] != 'Z'
            except FileNotFoundError:
                alive = False
            checks['sigterm_child_not_running'] = not alive
        if case == 'nonzero':
            checks['nonzero_original_exit_retained'] = result['tool_exit_code'] == 7
        if case == 'bad_identity':
            checks['bad_identity_no_child'] = result['child'] is None
        if case == 'success':
            previous = sha(target / 'status.json')
            repeated = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            checks['refuse_existing_attempt_without_overwrite'] = repeated.returncode != 0 and sha(target / 'status.json') == previous
        attempts[case] = dict(path=str(target), result_sha256=sha(target / 'result.json'),
                              status=expected, exit_code=proc.returncode)
    report = dict(status='pass' if all(checks.values()) else 'failed', execution_status='complete',
                  overall_pass=all(checks.values()), checks=checks, attempts=attempts,
                  runner_sha256=sha(runner), validator_sha256=sha(__file__),
                  scope='Process journal, original failure, SIGTERM cleanup and attempt isolation only; no neural/backend/metric acceptance.')
    (run / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], checks=len(checks), result_sha256=sha(run / 'result.json'))))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
