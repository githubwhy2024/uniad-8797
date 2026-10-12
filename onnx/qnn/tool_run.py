#!/usr/bin/env python3
"""Run one Q4 tool attempt with durable identity, logs and terminal status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
from sdk_environment import sdk_root, converter_environment, sdk_identity

SDK = sdk_root()
CONVERTER_ENV = converter_environment()
UNIAD_PYTHON = os.environ.get('UNIAD_PYTHON', sys.executable)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def save(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1].split()
        return dict(pid=pid, pgid=os.getpgid(pid), startticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, pgid=None, startticks=None)


def environment():
    env = os.environ.copy()
    for name, values in {
        'PATH': [str(CONVERTER_ENV / 'bin'), str(SDK / 'bin/x86_64-linux-clang')],
        'PYTHONPATH': [str(SDK / 'lib/python')],
        'LD_LIBRARY_PATH': [str(SDK / 'lib/x86_64-linux-clang'), str(CONVERTER_ENV / 'lib')],
    }.items():
        env[name] = ':'.join(dict.fromkeys(values + [entry for entry in env.get(name, '').split(':') if entry]))
    env.update(QAIRT_SDK=str(SDK), QAIRT_SDK_ROOT=str(SDK), QNN_SDK_ROOT=str(SDK),
               SNPE_ROOT=str(SDK), QNN_CONVERTER_ENV=str(CONVERTER_ENV))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['OMP_NUM_THREADS'] = '4'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '4'
    return env


def source_identity(model, expected):
    import numpy
    import onnx
    actual = sha(model)
    if actual != expected:
        raise ValueError('model SHA256 differs from requested identity')
    proto = onnx.load(str(model), load_external_data=False)
    if any(v.data_location == onnx.TensorProto.EXTERNAL for v in proto.graph.initializer):
        raise ValueError('external weights require an explicitly bound resource manifest')
    def boundary(value):
        tensor = value.type.tensor_type
        if not tensor.HasField('shape') or any(not d.HasField('dim_value') for d in tensor.shape.dim):
            raise ValueError('source boundary extent is not fixed: ' + value.name)
        return dict(name=value.name, shape=[d.dim_value for d in tensor.shape.dim],
                    dtype=str(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)))
    sources = [p for directory in ('fixed', 'qnn', 'validation') for p in sorted((ROOT / 'onnx' / directory).iterdir()) if p.is_file() and p.suffix in ('.py', '.cpp', '.h', '.hpp')]
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_files={str(p.relative_to(ROOT)): sha(p) for p in sources},
                fixed_manifest_sha256=sha(ROOT / 'onnx/fixed/SOURCE_MANIFEST.json'),
                dynamic_manifest_sha256=sha(ROOT / 'onnx/dynamic/SOURCE_MANIFEST.json'),
                model=str(model), resolved_model=str(model.resolve()), model_sha256=actual,
                initial_state_sha256=sha(ROOT / 'onnx/resources/initial_state.npz'),
                inputs=[boundary(v) for v in proto.graph.input], outputs=[boundary(v) for v in proto.graph.output],
                node_count=len(proto.graph.node), python=sys.executable, python_version=sys.version,
                numpy_version=numpy.__version__, onnx_version=onnx.__version__, sdk=str(SDK),
                sdk_identity=sdk_identity(SDK),
                qnn_cpu_library_sha256=sha(SDK / 'lib/x86_64-linux-clang/libQnnCpu.so'))


def convert_command(model, run, identity, diagnose=False, output_layouts=None, input_layouts=None):
    command = [str(SDK / 'bin/x86_64-linux-clang/qnn-onnx-converter'),
               '--input_network', str(model), '--output_path', str(run / 'model.cpp'),
               '--float_bitwidth', '32', '--no_simplification', '--preserve_io']
    # SDK OTHER overrides declared spatial axes; equal bgr/bgr has no color transformation.
    for row in identity['inputs']:
        command += ['--input_dtype', row['name'], row['dtype'],
                    '--input_encoding', row['name'], *(['bgr', 'bgr'] if row['name'] in (input_layouts or {}) else ['other']),
                    '--input_layout', row['name'], (input_layouts or {}).get(row['name'], 'NONTRIVIAL')]
    if output_layouts:
        command += ['--custom_io', str(run / 'custom_io.yaml')]
    if diagnose:
        command = [str(CONVERTER_ENV / 'bin/python'), str(ROOT / 'onnx/qnn/diagnose_converter.py')] + command[1:]
    return command


def run(args):
    run_dir = args.run_dir.absolute()
    allowed = ROOT / 'onnx/runs'
    if not run_dir.is_relative_to(allowed) or run_dir.resolve() != run_dir:
        raise ValueError('run must be a Q4 ordinary path below onnx/runs')
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='running', stage='preflight', execution_status='running',
                 acceptance_status='not_evaluated', overall_pass=None, started_utc=stamp(),
                 controller=process_identity(os.getpid()), child=None,
                 log=str(run_dir / 'tool.log'), result=str(run_dir / 'result.json'),
                 next_check='Read ' + str(run_dir / 'status.json') + ' and tail ' + str(run_dir / 'tool.log'),
                 scope='One host tool attempt only; neural recurrence/task/target acceptance separate.')
    child = None
    interrupted = None
    def terminate(signum, frame):
        nonlocal interrupted
        interrupted = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, terminate)
    save(run_dir / 'status.json', state)
    try:
        # Reject declaration errors before model loading or any child side effect.
        if args.acceptance_result:
            if args.stage != 'execute':
                raise ValueError('task acceptance declaration requires execute stage')
            declared = Path(args.acceptance_result)
            if declared.is_absolute() or '..' in declared.parts or not declared.parts:
                raise ValueError('task acceptance artifact must be below the run directory')
        if not (ROOT / '.git').is_dir() or Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip()) != ROOT:
            raise ValueError('independent project Git root required')
        identity = source_identity(args.model.absolute(), args.model_sha256)
        save(run_dir / 'source_identity.json', identity)
        env = environment()
        input_layouts = None
        if args.input_layouts:
            if args.stage != 'convert' or sha(args.input_layouts) != args.input_layouts_sha256:raise ValueError('input layout policy identity/stage differs')
            input_layouts = json.loads(args.input_layouts.read_text())
            inputs = {r['name']:r for r in identity['inputs']}
            if not input_layouts or any(n not in inputs or len(inputs[n]['shape']) != 4 or v not in ('NCHW','NHWC') for n,v in input_layouts.items()):raise ValueError('unsupported input layout policy')
            save(run_dir / 'input_layouts.json', input_layouts)
        layouts = None
        if args.output_layouts:
            if args.stage != 'convert' or sha(args.output_layouts) != args.output_layouts_sha256:raise ValueError('output layout policy identity/stage differs')
            layouts = json.loads(args.output_layouts.read_text())
            outputs = {r['name']:r for r in identity['outputs']}
            if not layouts or any(n not in outputs or len(outputs[n]['shape']) != 4 or v not in ('NCHW','NHWC') for n,v in layouts.items()):raise ValueError('unsupported output layout policy')
            save(run_dir / 'output_layouts.json', layouts)
            (run_dir / 'custom_io.yaml').write_text(''.join('- IOName: ' + n + '\n  Layout:\n    Model: ' + v + '\n    Custom: ' + v + '\n' for n,v in layouts.items()))
        native_layouts = None
        if args.native_layouts:
            if args.stage != 'convert' or sha(args.native_layouts) != args.native_layouts_sha256:
                raise ValueError('native layout policy identity/stage differs')
            native_layouts = json.loads(args.native_layouts.read_text())
            ports = {r['name']:r for r in identity['inputs'] + identity['outputs']}
            allowed = [dict(model='NCHW',native='NHWC'), dict(model='NHWC',native='NCHW')]
            if not native_layouts or any(n not in ports or len(ports[n]['shape']) != 4 or ports[n]['dtype'] != 'float32' or v not in allowed for n,v in native_layouts.items()):
                raise ValueError('unsupported explicit FP32 native layout policy')
            if input_layouts and any(n in input_layouts and input_layouts[n] != v['model'] for n,v in native_layouts.items()):
                raise ValueError('source input/native layout policy conflicts')
            if layouts and any(n in layouts and layouts[n] != v['model'] for n,v in native_layouts.items()):
                raise ValueError('source output/native layout policy conflicts')
            save(run_dir / 'native_layouts.json', native_layouts)
            merged = {n:dict(model=v,native=v) for n,v in (layouts or {}).items()}
            merged.update(native_layouts)
            (run_dir / 'custom_io.yaml').write_text(''.join('- IOName: ' + n + '\n  Layout:\n    Model: ' + v['model'] + '\n    Custom: ' + v['native'] + '\n' for n,v in merged.items()))
        command = convert_command(args.model.absolute(), run_dir, identity, args.diagnose_converter, layouts or native_layouts, input_layouts) if args.stage == 'convert' else args.command
        if not command:
            raise ValueError('a command is required')
        if command[0] == '--':
            command = command[1:]
        if args.address_space_gib is not None:
            if not 1 <= args.address_space_gib <= 64:
                raise ValueError('address space limit must be between 1 and 64 GiB')
            command = ['/usr/bin/prlimit', '--as=' + str(args.address_space_gib * 1024**3), '--'] + command
        executable = Path(command[0])
        launch = dict(argv=command, command=shlex.join(command), cwd=str(run_dir),
                      controller=state['controller'], env={name: env[name] for name in
                      ('PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                      source_identity_sha256=sha(run_dir / 'source_identity.json'),
                      tool_sha256=sha(executable) if executable.is_file() else None,
                      started_utc=state['started_utc'], stage=args.stage,
                      address_space_limit_gib=args.address_space_gib,
                      input_frame_scope='No recurrent frames for conversion/compile/proof tool stages.')
        save(run_dir / 'launch.json', launch)
        if interrupted is not None:
            raise InterruptedError('interrupted before launching tool')
        with (run_dir / 'tool.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(stage=args.stage, child=process_identity(child.pid), command=launch['command'], cwd=str(run_dir))
            launch['child'] = state['child']
            save(run_dir / 'launch.json', launch)
            save(run_dir / 'status.json', state)
            print(json.dumps(dict(status='running', run_dir=str(run_dir), controller=state['controller'], child=state['child'])), flush=True)
            code = child.wait()
        state['tool_exit_code'] = code
        state['log_sha256'] = sha(run_dir / 'tool.log')
        if interrupted is not None:
            raise InterruptedError('interrupted by signal ' + str(interrupted))
        if code != 0:
            raise RuntimeError('tool exited ' + str(code))
        if args.stage == 'convert':
            from audit_converter import describe
            audit = describe(args.model.absolute(), run_dir / 'model_net.json', native_layouts=native_layouts)
            save(run_dir / 'converter_audit.json', audit)
            bin_present = (run_dir / 'model.bin').is_file()
            if not bin_present and audit['static_tensor_count']:
                raise ValueError('converter static tensors require model.bin')
            state['weight_artifact'] = 'model.bin' if bin_present else 'none_no_static_tensors'
        expected = args.expect or ((['model.cpp', 'model_net.json', 'converter_audit.json'] + (['model.bin'] if bin_present else [])) if args.stage == 'convert' else [])
        if layouts:
            expected = list(expected) + ['output_layouts.json', 'custom_io.yaml']
        if native_layouts:
            expected = list(expected) + ['native_layouts.json'] + ([] if layouts else ['custom_io.yaml'])
        if input_layouts:
            expected = list(expected) + ['input_layouts.json']
        if args.acceptance_result:
            expected = list(expected) + [args.acceptance_result]
        artifacts = {}
        for name in expected:
            path = run_dir / name
            if not path.is_relative_to(run_dir) or path.resolve() != path or not path.is_file() or not path.stat().st_size:
                raise ValueError('expected nonempty ordinary artifact absent: ' + name)
            artifacts[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
        state['artifacts'] = artifacts
        state.update(status='pass', acceptance_status=args.stage + '_tool_pass_task_not_evaluated')
        if args.acceptance_result:
            claim=json.loads((run_dir / args.acceptance_result).read_text())
            if claim.get('execution_status')!='complete' or type(claim.get('overall_pass')) is not bool or claim.get('status')!=('pass' if claim['overall_pass'] else 'failed_acceptance'):raise ValueError('task acceptance result status/decision is inconsistent')
            state.update(status=claim['status'],overall_pass=claim['overall_pass'],acceptance_status='task_accepted' if claim['overall_pass'] else 'task_rejected',acceptance_result=args.acceptance_result)
    except BaseException as error:
        state.update(status='interrupted' if interrupted is not None else 'failed',
                     acceptance_status='not_accepted', error_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc())
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if (run_dir / 'tool.log').is_file():
            state['log_sha256'] = sha(run_dir / 'tool.log')
            with (run_dir / 'tool.log').open('rb') as stream:
                stream.seek(max(0, stream.seek(0, 2) - 8192))
                state['log_tail'] = stream.read().decode(errors='replace')
    finally:
        state.update(stage='complete', execution_status='complete' if state['status'] in ('pass','failed_acceptance') else 'failed',
                     finished_utc=stamp(), elapsed_seconds=time.monotonic() - started,
                     source_identity_sha256=sha(run_dir / 'source_identity.json') if (run_dir / 'source_identity.json').is_file() else None)
        save(run_dir / 'result.json', state)
        state['result_sha256'] = sha(run_dir / 'result.json')
        save(run_dir / 'status.json', state)
    print(json.dumps(dict(status=state['status'], run_dir=str(run_dir), result_sha256=state['result_sha256'])), flush=True)
    return 0 if state['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=('convert', 'compile', 'proof', 'control', 'execute'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--expect', action='append')
    parser.add_argument('--acceptance-result', help='Explicit completed task decision artifact; preserves failed_acceptance on tool exit zero')
    parser.add_argument('--input-layouts', type=Path)
    parser.add_argument('--input-layouts-sha256')
    parser.add_argument('--output-layouts', type=Path)
    parser.add_argument('--output-layouts-sha256')
    parser.add_argument('--native-layouts', type=Path)
    parser.add_argument('--native-layouts-sha256')
    parser.add_argument('--diagnose-converter', action='store_true')
    parser.add_argument('--address-space-gib', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    sys.exit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
