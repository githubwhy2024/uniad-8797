"""Actual mixed-dtype and repeated-call controls for input borrowing."""
import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onnx/qnn'))
from tool_run import sha, save
from sdk_environment import cpu_backend
import borrowed_session as borrowed
import session as native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-control-run', type=Path, required=True)
    args = parser.parse_args()
    fixture = args.native_control_run.absolute()
    status = json.loads((fixture / 'status.json').read_text())
    terminal = json.loads((fixture / 'result.json').read_text())
    if status['status'] != 'pass' or terminal['status'] != 'pass' or status['result_sha256'] != sha(fixture / 'result.json'):
        raise ValueError('native mixed fixture not terminal pass')
    command = json.loads((fixture / 'launch.json').read_text())['argv']
    source = ROOT / 'onnx/validation/validate_native_session.py'
    spec = importlib.util.spec_from_file_location('frozen_native_control', source)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    control.NativeSession = borrowed.BorrowedNativeSession
    sys.argv = [str(source)] + command[command.index(str(source)) + 1:]
    control.main()
    report = json.loads((Path.cwd() / 'native_controls.json').read_text())
    checks = dict(report['checks'])
    shim = object.__new__(borrowed.BorrowedNativeSession)
    contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)
    arrays = dict(contiguous=contiguous, strided=contiguous[:, ::2],
                  readonly=contiguous.copy(),
                  unaligned=np.ndarray((2,), dtype=np.float32, buffer=bytearray(9), offset=1))
    arrays['readonly'].flags.writeable = False
    for label, value in arrays.items():
        packed = shim._prepare_input(value)
        checks[label + '_routing'] = bool(np.array_equal(value, packed) and packed.flags.c_contiguous and packed.flags.aligned and packed.flags.writeable)
        checks[label + '_ownership'] = bool(np.shares_memory(value, packed) == (label == 'contiguous'))
    abi = json.loads((Path.cwd() / 'abi.json').read_text())
    # Extract named file options without relying on flags without values.
    def option(name):
        return sys.argv[sys.argv.index(name) + 1]
    resources = {role: option(flag) for role, flag in [('bridge', '--bridge'), ('model_lib', '--model-lib')]}
    resources['backend_lib'] = option('--backend') if '--backend' in sys.argv else str(cpu_backend())
    hashes = {role: sha(path) for role, path in resources.items()}
    feed = dict(state=np.zeros((2, 3), np.float32), delta=np.ones((2, 3), np.float32),
                ids=np.array([-2**31, -7, 42, 2**31 - 1], np.int64),
                mask=np.array([False, True, False, True]), count=np.array([2**31 - 1], np.int64))
    timing = {}
    for name, cls in [('baseline', native.NativeSession), ('borrowed', borrowed.BorrowedNativeSession)]:
        with cls(resources['bridge'], resources['model_lib'], resources['backend_lib'], abi, hashes) as sess:
            first = sess.run(None, feed)
            snapshots = [array.copy() for array in first]
            started = time.monotonic()
            for _ in range(100):
                result = sess.run(None, feed)
            timing[name] = (time.monotonic() - started) / 100
            checks[name + '_retained_outputs_not_overwritten'] = all(np.array_equal(a, b) for a, b in zip(first, snapshots))
            if name == 'baseline':
                expected = [array.copy() for array in result]
            else:
                checks['actual_baseline_borrowed_outputs_exact'] = all(np.array_equal(a, b) for a, b in zip(result, expected))
            noncontiguous = dict(feed, state=np.zeros((2, 6), np.float32)[:, ::2])
            checks[name + '_actual_strided_fallback'] = all(np.array_equal(a, b) for a, b in zip(sess.run(None, noncontiguous), expected))
            readonly = {key: value.copy() for key, value in feed.items()}
            for value in readonly.values():
                value.flags.writeable = False
            checks[name + '_actual_readonly_inputs_unchanged'] = all(np.array_equal(a, b) for a, b in zip(sess.run(None, readonly), expected)) and all(np.array_equal(readonly[key], value) for key, value in feed.items())
    save(Path.cwd() / 'borrowed_session_control.json', dict(status='pass' if all(checks.values()) else 'failed',
        checks=checks, transport=borrowed.TRANSPORT, provenance=borrowed.PROVENANCE,
        control_source_sha256=sha(source), fixture_result_sha256=sha(fixture / 'result.json'),
        tiny_call_mean_seconds=timing, scope='Actual constructed mixed dtype CPU control only. Tiny timings are not UniAD speed claims; own-state real sequence/mini metrics remain required.'))
    if not all(checks.values()):
        raise ValueError('borrowed-input control failed')


if __name__ == '__main__':
    main()
