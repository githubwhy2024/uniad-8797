#!/usr/bin/env python3
"""Assemble one verified planning CPU candidate, including optional borrowed I/O."""
import argparse
import ast
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import casadi
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qnn'))
from planning_identity import generated_runtime,generated_resource_runtime,portable_resource_policy
from tool_run import ROOT, sha, save
from partition_session import bind_partitions
sys.path.insert(0, str(ROOT / 'onnx/fixed'))
import host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-run', type=Path, required=True)
    parser.add_argument('--bridge-build', type=Path, required=True)
    parser.add_argument('--logical-model', type=Path, required=True)
    parser.add_argument('--logical-model-sha256', required=True)
    parser.add_argument('--borrowed-short-run',type=Path)
    parser.add_argument('--borrowed-control-run',type=Path)
    parser.add_argument('--resource-short-run',type=Path)
    parser.add_argument('--resource-control-run',type=Path)
    args = parser.parse_args()
    profile = bind_partitions(args.build_run, args.bridge_build, args.logical_model, args.logical_model_sha256)
    if len(profile['abi']['inputs']) != 23 or len(profile['abi']['outputs']) != 25:
        raise ValueError('only independently bound planning25 candidate may be packaged')
    borrowed = None;accepted=None;resource_adapter=None
    if (args.resource_short_run is None)!=(args.resource_control_run is None):raise ValueError('resource package needs both actual prerequisites')
    if args.resource_short_run is not None:
        if args.borrowed_short_run is not None or args.borrowed_control_run is not None:raise ValueError('select one scored execution profile')
        from resources import terminal
        from resource_partition_session import PROVENANCE
        for run,file in [(args.resource_short_run,'neural_result.json'),(args.resource_control_run,'resource_partition_control.json')]:
            tool=terminal(run)
            if tool['artifacts'][file]['sha256']!=sha(run/file):raise ValueError('resource package prerequisite differs')
        short=json.loads((args.resource_short_run/'neural_result.json').read_text());accepted=json.loads((args.resource_short_run/'profile.json').read_text());control=json.loads((args.resource_control_run/'resource_partition_control.json').read_text())
        if short['status']!='pass' or short['stage']!='six_real_frames' or len(short['completed_frames'])!=6 or short['profile_sha256']!=sha(args.resource_short_run/'profile.json') or control['status']!='pass' or not all(control['guards'].values()) or control['provenance']!=PROVENANCE:raise ValueError('resource actual gates differ')
        if short['peak_rss_kib']*1024>accepted['graph_resource_policy']['source']['physical_rss_limit_bytes']:raise ValueError('resource short sequence exceeded whole-process physical peak budget')
        if control['adapter_sha256']!=accepted['assets']['resource_adapter']['sha256'] or json.loads((args.resource_control_run/'source_identity.json').read_text())['source_files']['onnx/qnn/graph_resources.py']!=accepted['assets']['graph_resource_helper']['sha256']:raise ValueError('resource control implementation differs from actual short execution')
        stripped=copy.deepcopy(accepted)
        for role in ('resource_adapter','graph_resource_helper','borrowed_input_adapter','resource_frame_driver','resource_profile_helper','resource_policy'):
            asset=stripped['assets'].pop(role)
            if sha(asset['path'])!=asset['sha256']:raise ValueError('scored resource source changed: '+role)
        for field in ('transport_policy','graph_resource_policy','resource_provenance'):stripped.pop(field)
        if stripped!=profile:raise ValueError('resource source/model/runtime profile differs')
        profile=accepted;borrowed=profile['assets']['borrowed_input_adapter'];resource_adapter=profile['assets']['resource_adapter']
    if (args.borrowed_short_run is None)!=(args.borrowed_control_run is None):
        raise ValueError('direct borrowed package needs both actual prerequisites')
    if args.borrowed_short_run is not None:
        from resources import terminal
        from borrowed_session import PROVENANCE
        for run,file in [(args.borrowed_short_run,'neural_result.json'),(args.borrowed_control_run,'borrowed_session_control.json')]:
            tool=terminal(run)
            if tool['artifacts'][file]['sha256']!=sha(run/file):raise ValueError('borrowed package prerequisite differs')
        short=json.loads((args.borrowed_short_run/'neural_result.json').read_text())
        accepted=json.loads((args.borrowed_short_run/'profile.json').read_text())
        control=json.loads((args.borrowed_control_run/'borrowed_session_control.json').read_text())
        if short['status']!='pass' or short['stage']!='six_real_frames' or len(short['completed_frames'])!=6 or short['profile_sha256']!=sha(args.borrowed_short_run/'profile.json') or control['status']!='pass' or not all(control['checks'].values()) or control['provenance']!=PROVENANCE:raise ValueError('borrowed actual gates differ')
        borrowed=accepted['assets']['borrowed_input_adapter']
        expected=copy.deepcopy(profile)
        expected['assets']['borrowed_input_adapter']=copy.deepcopy(borrowed)
        expected['transport_policy']=dict(type='synchronous_borrowed_numpy_inputs',source=PROVENANCE)
        if accepted!=expected or sha(borrowed['path'])!=borrowed['sha256']:raise ValueError('direct borrowed source/model/runtime profile differs')
    bundle = Path.cwd() / 'bundle'
    bundle.mkdir()
    for name in ('runtime', 'lib', 'resources'):
        (bundle / name).mkdir()
    files = {}

    def copy_file(role, source, dest, expected=None):
        path = Path(source)
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('planning package source differs: ' + role)
        target = bundle / dest
        shutil.copyfile(path, target)
        if target.resolve() != target or sha(target) != digest:
            raise ValueError('planning package copy differs: ' + role)
        files[role] = dict(path=dest, sha256=digest, bytes=target.stat().st_size)

    for role, name in [('session_adapter', 'session.py'), ('ordinary_loader', 'native_bundle.py'), ('loader', 'planning_bundle.py')]:
        copy_file(role, ROOT / 'onnx/qnn' / name, 'runtime/' + name)
    if borrowed is not None:
        copy_file('borrowed_input_adapter',borrowed['path'],'runtime/borrowed_session.py',borrowed['sha256'])
    if resource_adapter is not None:
        row=profile['assets']['graph_resource_helper'];copy_file('graph_resource_helper',row['path'],'runtime/graph_resources.py',row['sha256'])
    for role, dest in [('host', 'runtime/host.py'), ('state_contract', 'runtime/state_contract.py'),
                       ('initial_state', 'resources/initial_state.npz'), ('collision_optimizer', 'resources/collision_optimization.py'),
                       ('bridge', 'lib/bridge.so'), ('backend_lib', 'lib/libQnnCpu.so')]:
        row = profile['assets'][role]
        copy_file(role, row['path'], dest, row['sha256'])
    if 'shared_constant_pool' in profile['assets']:
        row=profile['assets']['shared_constant_pool']
        copy_file('shared_constant_pool',row['path'],'resources/immutable_constants.npz',row['sha256'])
    parts = []
    for row in profile['parts']:
        dest = 'lib/part' + str(row['index']).zfill(3) + '.so'
        copy_file('part_' + str(row['index']), row['library'], dest, row['library_sha256'])
        parts.append(dict(index=row['index'], library=dest, library_sha256=row['library_sha256'], abi=copy.deepcopy(row['abi'])))
        if resource_adapter is not None:parts[-1]['native_model_sha256']=row['native_model_sha256']

    # Export the tested class and journal/range constants without developer imports.
    source = ROOT / 'onnx/qnn/partition_session.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PartitionSession')
    tools = ast.parse((ROOT / 'onnx/qnn/tool_run.py').read_text())
    savefn = next(n for n in tools.body if isinstance(n, ast.FunctionDef) and n.name == 'save')
    resources = ast.parse((ROOT / 'onnx/qnn/resources.py').read_text())
    ranges = next(n for n in resources.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RANGES' for t in n.targets))
    generated = bundle / 'runtime/partition_runtime.py'
    generated.write_text(generated_runtime(cls, ranges, savefn, borrowed is not None))
    if resource_adapter is not None:generated.write_text(generated_resource_runtime(cls,ranges,savefn,Path(resource_adapter['path']).read_text()))
    restored = next(n for n in ast.parse(generated.read_text()).body if isinstance(n, ast.ClassDef))
    if ast.dump(restored) != ast.dump(cls):
        raise ValueError('standalone planning partition class AST changed')
    files['partition_adapter'] = dict(path='runtime/partition_runtime.py', sha256=sha(generated), bytes=generated.stat().st_size)

    ldd = subprocess.check_output(['/usr/bin/ldd', profile['assets']['backend_lib']['path']], text=True)
    deps = {}
    for line in ldd.splitlines():
        values = line.split()
        if len(values) >= 3 and values[1] == '=>' and values[2].startswith('/'):
            deps[values[0]] = values[2]
    for role, name in [('libc++', 'libc++.so.1'), ('libc++abi', 'libc++abi.so.1'), ('libunwind', 'libunwind.so.1')]:
        copy_file(role, deps[name], 'lib/' + name)
    cuts = [{key: copy.deepcopy(row[key]) for key in ('index', 'inputs', 'outputs', 'drop_after')} for row in profile['plan']['parts']]
    with np.load(bundle / files['initial_state']['path'], allow_pickle=False) as archive:
        initial = {name: archive[name].copy() for name in archive.files}
    if initial['query'].shape[0] == 901:
        initial = host.pad_v1_initial_state(initial)
    if host.validate_fixed_state(initial) != host.FRESH or int(initial['max_obj_id']) != 0:
        raise ValueError('planning package learned initial contract differs')
    lineage = dict(build_result_sha256=sha(args.build_run / 'result.json'), build_manifest_sha256=sha(args.build_run / 'partition_build.json'),
        bridge_result_sha256=sha(args.bridge_build / 'result.json'), logical_model_sha256=args.logical_model_sha256,
        native_model_sha256=profile['assets']['native_model']['sha256'], partition_source_sha256=sha(source),
        partition_class_ast_sha256=hashlib.sha256(ast.dump(cls).encode()).hexdigest(),
        atomic_save_ast_sha256=hashlib.sha256(ast.dump(savefn).encode()).hexdigest(),
        ranges_ast_sha256=hashlib.sha256(ast.dump(ranges).encode()).hexdigest(),
        sdk_release=profile['backend']['sdk_identity']['release'],
        sdk_identity=copy.deepcopy(profile['backend']['sdk_identity']))
    manifest = dict(schema='qnn-planning-host-candidate-v1', purpose='planning_host_candidate', task_acceptance='not_evaluated',
        backend=dict(type='QNN_CPU', precision='float32', target='x86_64-linux-clang'),
        runtime_versions=dict(python=platform.python_version(), numpy=np.__version__, casadi=casadi.__version__, machine=platform.machine()),
        host_policy=profile['host_policy'], capacities=dict(fresh=host.FRESH, track=host.TRACK_SLOTS, survivor=host.SURVIVOR_CAPACITY,
            decoded=host.DECODED_SLOTS, vehicle=host.VEHICLE_SLOTS, sca=host.SCA_CAPACITY),
        initial_state_digest=host.state_digest(initial), files=files,
        profile=dict(abi=profile['abi'], parts=parts, plan=dict(parts=cuts)), lineage=lineage,
        os_dependencies={name: dict(sha256=sha(path), bytes=Path(path).stat().st_size) for name, path in deps.items()
            if name not in ('libc++.so.1', 'libc++abi.so.1', 'libunwind.so.1')},
        scope='Portable CPU planning candidate only; mini404/task/relocated neural/board release acceptance separate.')
    if borrowed is not None:
        manifest['transport_policy']=copy.deepcopy(accepted['transport_policy'])
        if resource_adapter is None:manifest['lineage'].update(borrowed_short_result_sha256=sha(args.borrowed_short_run/'result.json'),borrowed_control_result_sha256=sha(args.borrowed_control_run/'result.json'),direct_borrowed_builder_sha256=sha(__file__))
    if resource_adapter is not None:
        manifest['graph_resource_policy']=portable_resource_policy(profile['graph_resource_policy'])
        manifest['lineage'].update(resource_short_result_sha256=sha(args.resource_short_run/'result.json'),resource_control_result_sha256=sha(args.resource_control_run/'result.json'),resource_source_sha256=resource_adapter['sha256'],resource_builder_sha256=sha(__file__))
    if 'shared_constants' in profile['plan']:
        pool=copy.deepcopy(profile['plan']['shared_constants']);pool.pop('path')
        manifest['profile']['shared_constants']=pool
    save(bundle / 'manifest.json', manifest)
    save(Path.cwd() / 'planning_package.json', dict(status='pass', bundle=str(bundle), manifest_sha256=sha(bundle / 'manifest.json'),
        files=len(files), parts=len(parts), lineage=lineage, task_acceptance='not_evaluated',
        scope='Candidate assembly and unchanged Host/partition implementation only; relocation controls required.'))


if __name__ == '__main__':
    main()
