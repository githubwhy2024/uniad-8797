"""Standalone pinned QNN planning candidate; production results use frozen Host."""
import argparse
import copy
import json
import platform
import os
import threading
import time
from pathlib import Path

import casadi
import numpy as np
import host
import native_bundle
import partition_runtime
import session
import state_contract
from native_bundle import ordinary
from session import sha
_memory_guard_started=False

def start_resource_memory_guard(limit_bytes):
    global _memory_guard_started
    if type(limit_bytes)!=int or limit_bytes!=18*1024**3:raise ValueError('host physical resource budget differs')
    if _memory_guard_started:return
    _memory_guard_started=True
    def enforce():
        while True:
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.split(':',1)[0] in ('VmRSS','VmHWM','VmSize','VmPeak')}
            if int(fields['VmHWM'].split()[0])*1024>limit_bytes:
                partition_runtime.save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_process_peak_rss_budget_exceeded',limit_bytes=limit_bytes,memory=fields));os._exit(86)
            time.sleep(2)
    threading.Thread(target=enforce,daemon=True,name='resource-rss-guard').start()

class ResourceStateTransaction(host.FixedStateTransaction):
    def advance_planning(self,*args,**kwargs):
        if kwargs.get('new_scene',False):self.session.reset_state()
        return super().advance_planning(*args,**kwargs)


def load_planning_bundle(root, pin, *, allow_candidate=False):
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError('ordinary planning bundle root required')
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('explicit planning manifest SHA256 required')
    path = ordinary(root, 'manifest.json')
    if sha(path) != pin:
        raise ValueError('planning bundle pin differs')
    manifest = json.loads(path.read_text())
    candidate = manifest['schema'] == 'qnn-planning-host-candidate-v1' and manifest['purpose'] == 'planning_host_candidate'
    release = manifest['schema'] == 'qnn-planning-host-release-v1' and manifest['purpose'] == 'planning_host_release'
    if not candidate and not release:
        raise ValueError('unsupported planning bundle contract')
    if candidate and (allow_candidate is not True or manifest['task_acceptance'] != 'not_evaluated'):
        raise ValueError('explicit candidate validation required; task release not accepted')
    if release:
        acceptance = manifest['acceptance']
        if manifest['task_acceptance'] != 'accepted':
            raise ValueError('planning release task acceptance missing')
        declared=acceptance.get('dataset_scope')
        legacy=dict(schema='qnn-mini-dataset-scope-v1',selection='all',splits=['mini_val','mini_train'],frames=404,complete_stage='mini404_records')
        val=dict(schema='qnn-mini-dataset-scope-v1',selection='mini_val',splits=['mini_val'],frames=81,complete_stage='mini_val81_records')
        if declared is None:declared=legacy
        if declared not in (legacy,val):raise ValueError('planning release validation dataset scope differs')
        for role, per_split in [('planning', 9), ('full', 96)]:
            standard=per_split*len(declared['splits'])
            row = acceptance[role]
            if row.get('dataset_scope',legacy)!=declared or row['frames'] != declared['frames'] or row['standard_checks'] != standard or row['raw_planning_checks'] != 6*len(declared['splits']):
                raise ValueError('planning release task coverage differs')
            for key in ('mini_result_sha256', 'mini_audit_sha256', 'task_result_sha256', 'profile_sha256'):
                value = row[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('planning release evidence binding differs')
        if acceptance['planning']['native_model_sha256'] != manifest['lineage']['native_model_sha256'] or acceptance['package']['frames'] != 6:
            raise ValueError('planning release model/portable coverage differs')
        for key in ('candidate_manifest_sha256', 'resource_control_sha256', 'neural_control_sha256'):
            value = acceptance['package'][key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('planning release package binding differs')
    if manifest['backend'] != dict(type='QNN_CPU', precision='float32', target='x86_64-linux-clang'):
        raise ValueError('planning backend differs')
    if manifest['runtime_versions'] != dict(python=platform.python_version(), numpy=np.__version__, casadi=casadi.__version__, machine=platform.machine()):
        raise ValueError('planning runtime versions differ')
    if manifest['host_policy'] != dict(can_bus_mode='official_test_legacy', id_scope='session', coordinate_mode='legacy_int'):
        raise ValueError('planning Host policy differs')
    capacities = dict(fresh=host.FRESH, track=host.TRACK_SLOTS, survivor=host.SURVIVOR_CAPACITY,
                      decoded=host.DECODED_SLOTS, vehicle=host.VEHICLE_SLOTS, sca=host.SCA_CAPACITY)
    if manifest['capacities'] != capacities:
        raise ValueError('planning capacities differ')
    assets, seen = {}, set()
    for role, row in manifest['files'].items():
        file = ordinary(root, row['path'])
        if row['path'] in seen or sha(file) != row['sha256'] or file.stat().st_size != row['bytes']:
            raise ValueError('planning file binding differs: ' + role)
        seen.add(row['path'])
        assets[role] = dict(path=str(file), sha256=row['sha256'], bytes=row['bytes'])
    modules = dict(session_adapter=session.__file__, ordinary_loader=native_bundle.__file__,
                   partition_adapter=partition_runtime.__file__, host=host.__file__,
                   state_contract=state_contract.__file__, loader=__file__)
    if manifest.get('graph_resource_policy') is not None:
        import graph_resources
        modules['graph_resource_helper']=graph_resources.__file__
    required = set(modules) | {'bridge', 'backend_lib', 'libc++', 'libc++abi', 'libunwind',
                               'initial_state', 'collision_optimizer'}
    if not required <= set(assets):
        raise ValueError('planning runtime closure incomplete')
    for role, path in modules.items():
        if Path(path).resolve() != ordinary(root, manifest['files'][role]['path']) or sha(path) != assets[role]['sha256']:
            raise ValueError('imported planning runtime differs: ' + role)
    profile = copy.deepcopy(manifest['profile'])
    profile['assets'] = assets
    if manifest.get('graph_resource_policy') is not None:profile['graph_resource_policy']=copy.deepcopy(manifest['graph_resource_policy'])
    if len(profile['abi']['inputs']) != 23 or len(profile['abi']['outputs']) != 25:
        raise ValueError('planning public ABI differs')
    rows, cuts = profile['parts'], profile['plan']['parts']
    if not rows or len(rows) != len(cuts) or [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('planning cut coverage/order differs')
    for row, cut in zip(rows, cuts):
        role = 'part_' + str(row['index'])
        if role not in assets or row['index'] != cut['index'] or row['library'] != manifest['files'][role]['path'] or row['library_sha256'] != assets[role]['sha256']:
            raise ValueError('planning model library binding differs')
        if any([v['name'] for v in row['abi'][kind]] != [v['name'] for v in cut[kind]] for kind in ('inputs', 'outputs')):
            raise ValueError('planning cut ABI differs')
        row['library'] = assets[role]['path']
    with np.load(assets['initial_state']['path'], allow_pickle=False) as archive:
        initial = {name: archive[name].copy() for name in archive.files}
    if initial['query'].shape[0] == 901:
        initial = host.pad_v1_initial_state(initial)
    if host.validate_fixed_state(initial) != host.FRESH or int(initial['max_obj_id']) != 0 or host.state_digest(initial) != manifest['initial_state_digest']:
        raise ValueError('planning learned initialization differs')
    host.verified_optimizer_source(assets['collision_optimizer']['path'], assets['collision_optimizer']['sha256'])
    return manifest, profile, initial


def create_planning_stream(root, pin, *, max_state_gap_seconds, max_consecutive_failures,
                           allow_candidate=False):
    manifest, profile, initial = load_planning_bundle(root, pin, allow_candidate=allow_candidate)
    if 'graph_resource_policy' in profile:start_resource_memory_guard(profile['graph_resource_policy']['source']['physical_rss_limit_bytes'])
    native = partition_runtime.ResourcePartitionSession(profile,profile['graph_resource_policy']) if 'graph_resource_policy' in profile else partition_runtime.PartitionSession(profile)
    try:
        transaction=ResourceStateTransaction if 'graph_resource_policy' in profile else host.FixedStateTransaction
        runtime = transaction(native, initial,
            model_sha256=manifest['lineage']['native_model_sha256'],
            can_bus_mode=manifest['host_policy']['can_bus_mode'], id_scope=manifest['host_policy']['id_scope'])
        runtime.bundle_manifest_sha256 = pin
        stream = host.PlanningStream(runtime, optimizer_source=profile['assets']['collision_optimizer']['path'],
            optimizer_sha256=profile['assets']['collision_optimizer']['sha256'],
            max_state_gap_seconds=max_state_gap_seconds, max_consecutive_failures=max_consecutive_failures)
    except BaseException:
        native.close()
        raise
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--allow-candidate-validation', action='store_true')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-state-gap-seconds', type=float, required=True)
    parser.add_argument('--max-consecutive-failures', type=int, required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resume-checkpoint-sha256')
    args = parser.parse_args()
    if sha(args.request) != args.request_sha256:
        raise ValueError('planning request binding differs')
    request = json.loads(args.request.read_text())
    if request['schema'] != 'qnn-planning-frame-request-v1' or not isinstance(request['frames'], list) or not request['frames']:
        raise ValueError('nonempty synchronized frame request required')
    output = args.output_dir.absolute()
    if output.exists() or output.resolve() != output or output.is_relative_to(args.bundle.absolute()):
        raise ValueError('new ordinary planning evidence directory outside bundle required')
    if bool(args.resume_checkpoint) != bool(args.resume_checkpoint_sha256):
        raise ValueError('checkpoint and its SHA256 required together')
    stream = create_planning_stream(args.bundle, args.manifest_sha256,
        max_state_gap_seconds=args.max_state_gap_seconds, max_consecutive_failures=args.max_consecutive_failures,
        allow_candidate=args.allow_candidate_validation)
    output.mkdir(parents=True)
    completed = []
    try:
        if args.resume_checkpoint:
            if sha(args.resume_checkpoint) != args.resume_checkpoint_sha256:
                raise ValueError('planning checkpoint binding differs')
            stream.runtime.load_checkpoint(args.resume_checkpoint)
        starting_state = host.state_digest(stream.runtime.state)
        for index, row in enumerate(request['frames']):
            if sha(row['tensors']) != row['tensors_sha256']:
                raise ValueError('planning frame tensor binding differs')
            with np.load(row['tensors'], allow_pickle=False) as archive:
                tensors = {name: archive[name].copy() for name in archive.files}
            if set(tensors) != {'image', 'can_bus_absolute', 'l2g_r', 'l2g_t', 'lidar2img', 'img_shape'}:
                raise ValueError('planning synchronized frame fields differ')
            checkpoint = output / ('frame' + str(index) + '.checkpoint.npz')
            incoming_state = host.state_digest(stream.runtime.state)
            result = stream.process(frame_id=row['frame_id'], scene_token=row['scene_token'],
                timestamp=row['timestamp'], command=row['command'], checkpoint_path=checkpoint, **tensors)
            record = {key: value for key, value in result.items() if key != 'plan'}
            record['incoming_state_sha256'] = incoming_state
            if result['status'] == 'accepted':
                final = output / ('frame' + str(index) + '.final.npz')
                np.savez(final, planning_final=result['plan'])
                record.update(final_plan=str(final), final_plan_sha256=sha(final),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    state_sha256=host.state_digest(stream.runtime.state),
                    planning_info=copy.deepcopy(stream.runtime.last_planning_info))
            completed.append(record)
            partition_runtime.save(output / 'planning_progress.json', dict(frames=completed, manifest_sha256=args.manifest_sha256))
            if result['status'] != 'accepted':
                raise ValueError('planning frame rejected: ' + str(result['rejection_code']))
        partition_runtime.save(output / 'planning_result.json', dict(status='pass', frames=completed,
            manifest_sha256=args.manifest_sha256, starting_state_sha256=starting_state, task_acceptance='not_evaluated',
            scope='Relocated planning candidate own-state inference only; task and board acceptance separate.'))
    except BaseException as error:
        partition_runtime.save(output / 'planning_result.json', dict(status='failed', frames=completed,
            manifest_sha256=args.manifest_sha256, error_type=type(error).__name__, error=str(error),
            task_acceptance='not_evaluated', scope='Failed relocated candidate execution retained.'))
        raise
    finally:
        stream.runtime.session.close()


if __name__ == '__main__':
    main()
