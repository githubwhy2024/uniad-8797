"""Copy a verified CPU candidate and bind its tested borrowed-input transport."""
import argparse
import ast
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onnx/qnn'))
from tool_run import save, sha
from resources import terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-run', type=Path, required=True)
    parser.add_argument('--short-run', type=Path, required=True)
    parser.add_argument('--control-run', type=Path, required=True)
    args = parser.parse_args()
    for path, name in [(args.package_run, 'planning_package.json'),
                       (args.short_run, 'neural_result.json'),
                       (args.control_run, 'borrowed_session_control.json')]:
        record = terminal(path)
        if record['artifacts'][name]['sha256'] != sha(path / name):
            raise ValueError('borrowed package prerequisite changed')
    original = json.loads((args.package_run / 'planning_package.json').read_text())
    short = json.loads((args.short_run / 'neural_result.json').read_text())
    profile = json.loads((args.short_run / 'profile.json').read_text())
    if short['status'] != 'pass' or short['stage'] != 'six_real_frames' or len(short['completed_frames']) != 6 or short['profile_sha256'] != sha(args.short_run / 'profile.json'):
        raise ValueError('borrowed package requires its own actual six-frame recurrence')
    source = Path(original['bundle'])
    if sha(source / 'manifest.json') != original['manifest_sha256']:
        raise ValueError('source candidate manifest changed')
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['purpose'] != 'planning_host_candidate' or manifest['task_acceptance'] != 'not_evaluated':
        raise ValueError('start from an immutable unpromoted CPU candidate')
    if manifest['lineage']['native_model_sha256'] != profile['assets']['native_model']['sha256'] or manifest['profile']['abi'] != profile['abi'] or [p['library_sha256'] for p in manifest['profile']['parts']] != [p['library_sha256'] for p in profile['parts']]:
        raise ValueError('borrowed runtime changed actual model libraries or ABI')
    for role, key in [('bridge', 'bridge'), ('backend_lib', 'backend_lib'), ('host', 'host'),
                      ('state_contract', 'state_contract'), ('initial_state', 'initial_state'),
                      ('collision_optimizer', 'collision_optimizer'), ('session_adapter', 'adapter')]:
        if manifest['files'][role]['sha256'] != profile['assets'][key]['sha256']:
            raise ValueError('borrowed package source runtime differs: ' + role)
    adapter = profile['assets']['borrowed_input_adapter']
    if sha(adapter['path']) != adapter['sha256'] or profile['transport_policy']['type'] != 'synchronous_borrowed_numpy_inputs':
        raise ValueError('borrowed adapter source differs')
    destination = Path.cwd() / 'bundle'
    shutil.copytree(source, destination, symlinks=False)
    for role, path, relative in [('borrowed_input_adapter', Path(adapter['path']), 'runtime/borrowed_session.py'),
                                 ('loader', ROOT / 'onnx/qnn/planning_bundle.py', manifest['files']['loader']['path'])]:
        target = destination / relative
        shutil.copyfile(path, target)
        manifest['files'][role] = dict(path=relative, bytes=target.stat().st_size, sha256=sha(target))
    runtime = destination / manifest['files']['partition_adapter']['path']
    original_text = runtime.read_text()
    tree = ast.parse(original_text)
    original_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PartitionSession')
    injection = '\nimport borrowed_session\nif Path(borrowed_session.__file__).resolve() != Path(__file__).parent / "borrowed_session.py":\n    raise ValueError("borrowed module must come from the copied runtime")\nNativeSession = borrowed_session.BorrowedNativeSession\n'
    runtime.write_text(original_text + injection)
    final_class = next(node for node in ast.parse(runtime.read_text()).body if isinstance(node, ast.ClassDef) and node.name == 'PartitionSession')
    if ast.dump(original_class) != ast.dump(final_class):
        raise ValueError('borrowed candidate must preserve partition liveness and strict guards')
    manifest['files']['partition_adapter'].update(sha256=sha(runtime), bytes=runtime.stat().st_size)
    manifest['transport_policy'] = copy.deepcopy(profile['transport_policy'])
    manifest['lineage'].update(source_candidate_manifest_sha256=original['manifest_sha256'],
        borrowed_short_result_sha256=sha(args.short_run / 'result.json'),
        borrowed_control_result_sha256=sha(args.control_run / 'result.json'),
        borrowed_builder_sha256=sha(__file__))
    manifest['scope'] = 'Portable borrowed-input FP32 CPU candidate; own mini404 task and outside-cwd six-frame gates required before release.'
    save(destination / 'manifest.json', manifest)
    save(Path.cwd() / 'planning_package.json', dict(status='pass', bundle=str(destination),
         manifest_sha256=sha(destination / 'manifest.json'), files=len(manifest['files']),
         parts=len(manifest['profile']['parts']), lineage=manifest['lineage'], task_acceptance='not_evaluated',
         scope='Borrowed candidate assembly only; no task/release/board acceptance.'))


if __name__ == '__main__':
    main()
