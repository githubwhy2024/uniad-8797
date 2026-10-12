"""Executable audit for the retained stateful host contract.

This is intentionally a lightweight contract test: it does not load UniAD,
ONNX Runtime, nuScenes, or the checkpoint. It verifies the host-side behavior
that must match native UniAD before task-level metric comparison is meaningful.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from host import (
    consecutive_occupancy_ids,
    map_iou_counts,
    merge_map_masks,
    motion_evaluator_view,
    occupancy_positions,
    optimize_planning,
)
from host import STATE_NAMES, StatefulRuntime, preprocess_base_images


BASE_INPUT_NAMES = [
    "img", "can_bus", "l2g_r_mat", "lidar2img", "img_shape", "command",
    "has_prev_bev", "prev_l2g_r", "prev_l2g_t", "l2g_t", "time_delta",
] + list(STATE_NAMES) + ["prev_bev", "max_obj_id"]


class _Name:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Record host feeds and return a structurally valid deterministic state."""

    def __init__(self, id_increment=3):
        self.feeds = []
        self.id_increment = int(id_increment)
        self._inputs = [_Name(name) for name in BASE_INPUT_NAMES]
        self._outputs = [_Name("bev_embed"), _Name("next_max_obj_id")]
        self._outputs += [_Name("next_" + name) for name in STATE_NAMES]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _requested, feed):
        self.feeds.append({name: value.copy() for name, value in feed.items()})
        outputs = [feed["prev_bev"].copy(),
                   np.asarray(feed["max_obj_id"] + self.id_increment, dtype=np.int64)]
        outputs.extend(feed[name].copy() for name in STATE_NAMES)
        return outputs


def make_initial_state():
    n = 901
    return {
        "query": np.zeros((n, 512), np.float32),
        "ref_pts": np.zeros((n, 3), np.float32),
        "pred_boxes": np.zeros((n, 10), np.float32),
        "obj_idxes": np.full((n,), -1, np.int64),
        "disappear_time": np.zeros((n,), np.int64),
        "mem_bank": np.zeros((n, 4, 256), np.float32),
        "mem_padding_mask": np.ones((n, 4), np.bool_),
        "save_period": np.zeros((n,), np.float32),
        "prev_bev": np.zeros((40000, 1, 256), np.float32),
        "max_obj_id": np.asarray(0, dtype=np.int64),
    }


def frame(position, angle, translation, timestamp):
    bus = np.zeros((18,), np.float32)
    bus[3] = 1.0
    bus[:3] = np.asarray(position, np.float32)
    bus[-1] = np.float32(angle)
    return dict(
        timestamp=float(timestamp),
        image=np.zeros((1, 6, 3, 928, 1600), np.float32),
        can_bus_absolute=bus,
        l2g_r=np.eye(3, dtype=np.float32),
        l2g_t=np.asarray(translation, np.float32),
        lidar2img=np.repeat(np.eye(4, dtype=np.float32)[None, None], 6, axis=1),
        img_shape=np.asarray([[[928, 1600]] * 6], np.int64),
        command=0,
    )


def assert_allclose(actual, expected, label, atol=1e-6):
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def check_preprocess():
    images = [np.zeros((900, 1600, 3), np.uint8) for _ in range(6)]
    tensor, shape = preprocess_base_images(images)
    if tensor.shape != (1, 6, 3, 928, 1600) or tensor.dtype != np.float32:
        raise AssertionError("preprocess tensor shape/dtype mismatch")
    if shape.dtype != np.int64 or shape.tolist() != [[[928, 1600]] * 6]:
        raise AssertionError("preprocess img_shape mismatch")
    expected = -np.asarray([103.530, 116.280, 123.675], np.float32)
    assert_allclose(tensor[0, 0, :, 0, 0], expected, "BGR normalization")
    if np.any(tensor[:, :, :, 900:, :] != 0):
        raise AssertionError("padding must be zero after normalization")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "padding": "bottom_to_928_zero", "color_order": "BGR"}


def check_legacy_runtime():
    session = FakeSession()
    runtime = StatefulRuntime(session, make_initial_state(),
                              can_bus_mode="official_test_legacy", id_scope="session")

    runtime.step(scene_token="scene-A", **frame([10, 20, 0], 30, [1, 2, 3], 1.0))
    first = session.feeds[-1]
    assert_allclose(first["can_bus"][0, :3], [10, 20, 0], "legacy first-frame position")
    assert_allclose(first["can_bus"][0, -1], 30, "legacy first-frame angle")
    if bool(first["has_prev_bev"]) or float(first["time_delta"]) != 0.0:
        raise AssertionError("first frame must have no temporal BEV")
    if int(first["max_obj_id"]) != 0:
        raise AssertionError("initial ID counter mismatch")

    runtime.step(scene_token="scene-A", **frame([11, 23, 0], 35, [2, 4, 6], 1.5))
    second = session.feeds[-1]
    assert_allclose(second["can_bus"][0, :3], [1, 3, 0], "within-scene position delta")
    assert_allclose(second["can_bus"][0, -1], 5, "within-scene angle delta")
    if not bool(second["has_prev_bev"]) or not np.isclose(float(second["time_delta"]), 0.5):
        raise AssertionError("second frame temporal metadata mismatch")
    if int(second["max_obj_id"]) != 3:
        raise AssertionError("ID counter did not advance through session")
    assert_allclose(second["prev_l2g_t"], [1, 2, 3], "previous l2g translation")

    runtime.step(scene_token="scene-B", **frame([100, 200, 0], 100, [9, 8, 7], 2.0))
    scene_change = session.feeds[-1]
    assert_allclose(scene_change["can_bus"][0, :3], [89, 177, 0],
                    "legacy cross-scene position subtraction")
    assert_allclose(scene_change["can_bus"][0, -1], 65,
                    "legacy cross-scene angle subtraction")
    if bool(scene_change["has_prev_bev"]) or float(scene_change["time_delta"]) != 0.0:
        raise AssertionError("scene change must reset BEV/time feedback")
    if int(scene_change["max_obj_id"]) != 6:
        raise AssertionError("session-scoped ID counter must survive scene reset")
    if not np.array_equal(scene_change["query"], runtime.initial["query"]):
        raise AssertionError("scene change did not restore learned track initialization")
    if not np.array_equal(scene_change["mem_bank"], runtime.initial["mem_bank"]):
        raise AssertionError("scene change did not reset memory bank")

    return {
        "first_frame_bus": scene_vector(first["can_bus"][0]),
        "within_scene_bus": scene_vector(second["can_bus"][0]),
        "cross_scene_bus": scene_vector(scene_change["can_bus"][0]),
        "scene_change_has_prev_bev": bool(scene_change["has_prev_bev"]),
        "scene_change_input_max_obj_id": int(scene_change["max_obj_id"]),
        "id_scope": "session",
    }


def scene_vector(bus):
    return {"xyz": [float(x) for x in bus[:3]], "angle_deg": float(bus[-1])}


def check_scene_reset_alternative():
    session = FakeSession()
    runtime = StatefulRuntime(session, make_initial_state(),
                              can_bus_mode="scene_reset", id_scope="session")
    runtime.step(scene_token="scene-A", **frame([10, 20, 0], 30, [1, 2, 3], 1.0))
    first = session.feeds[-1]
    runtime.step(scene_token="scene-B", **frame([100, 200, 0], 100, [9, 8, 7], 2.0))
    second = session.feeds[-1]
    assert_allclose(first["can_bus"][0, :3], [0, 0, 0], "scene-reset first xyz")
    assert_allclose(first["can_bus"][0, -1], 0, "scene-reset first angle")
    assert_allclose(second["can_bus"][0, :3], [0, 0, 0], "scene-reset new-scene xyz")
    assert_allclose(second["can_bus"][0, -1], 0, "scene-reset new-scene angle")
    return {"new_scene_bus_zeroed": True, "reference_baseline": False}


def check_map_contract():
    masks = np.zeros((4, 4, 4), np.float32)
    masks[0, 0:2, 0:2] = .9
    masks[1, 2:4, 0:2] = .8
    masks[2, 0:2, 2:4] = .7
    masks[3, 1:4, 1:4] = .6
    outputs = {
        "map_mask_scores": masks,
        "map_selected_boxes": np.asarray([
            [0, 0, 1, 1, .8],
            [0, 0, 1, 1, .7],
            [0, 0, 1, 1, .6],
        ], np.float32),
        "map_selected_labels": np.asarray([0, 1, 2], np.int64),
    }
    merged = merge_map_masks(outputs, reject_ambiguous_ties=True)
    expected_lane = masks[:3] > .5
    if not np.array_equal(merged["lane"] > 0, expected_lane):
        raise AssertionError("map thing-mask order/merge mismatch")
    if not np.array_equal(merged["drivable"], masks[-1] > .5):
        raise AssertionError("map drivable mask mismatch")

    gt_labels = np.asarray([0, 1, 2, 3], np.int64)
    gt_masks = np.concatenate((expected_lane.astype(np.uint8), (masks[-1:] > .5).astype(np.uint8)), axis=0)
    counts = map_iou_counts(merged, gt_labels, gt_masks)
    for name in ("drivable_iou", "lanes_iou", "divider_iou", "crossing_iou", "contour_iou"):
        if not np.isclose(counts[name], 1.0):
            raise AssertionError(f"map IoU contract mismatch for {name}")

    tied = {
        "map_mask_scores": np.asarray([
            [[.8, 0], [0, 0]],
            [[0, .8], [0, 0]],
            [[.6, .6], [.6, .6]],
        ], np.float32),
        "map_selected_boxes": np.asarray([
            [0, 0, 1, 1, .8],
            [0, 0, 1, 1, .8],
        ], np.float32),
        "map_selected_labels": np.asarray([0, 1], np.int64),
    }
    try:
        merge_map_masks(tied, reject_ambiguous_ties=True)
    except ValueError as error:
        if "tie order" not in str(error):
            raise
    else:
        raise AssertionError("ambiguous native map sort tie was not rejected")

    return {
        "thing_rows": 3,
        "stuff_role": "drivable_raw_mask_only_in_native_merge",
        "iou_fixture": "all_five_native_map_ious_equal_1",
        "ambiguous_equal_score_ties": "detected_and_rejected_in_validation_mode",
    }


def check_motion_contract():
    count, modes, steps = 2, 6, 12
    traj = np.zeros((count, modes, steps, 2), np.float32)
    traj[0, ..., 0] = 1.0
    traj[1, ..., 0] = 2.0
    log_scores = np.linspace(-3.0, -0.1, count * modes, dtype=np.float32).reshape(count, modes)
    outputs = {
        "track_boxes": np.zeros((count, 9), np.float32),
        "track_scores": np.asarray([.9, .8], np.float32),
        "track_labels": np.asarray([0, 8], np.int64),
        "track_ids": np.asarray([10, 11], np.int64),
        "track_query_indices": np.asarray([4, 9], np.int64),
        "motion_xy": traj,
        "motion_log_scores": log_scores,
    }
    view = motion_evaluator_view(outputs)
    if not np.array_equal(view["traj"], traj) or not np.array_equal(view["traj_scores"], log_scores):
        raise AssertionError("motion evaluator view reordered or transformed outputs")
    if not np.all(view["traj"][0, ..., 0] == 1.0) or not np.all(view["traj"][1, ..., 0] == 2.0):
        raise AssertionError("motion rows no longer align with decoded track order")
    return {
        "track_rows": count,
        "modes": modes,
        "steps": steps,
        "trajectory_values": "model_native_cumulative_local_xy",
        "mode_scores": "native_log_softmax_preserved_without_probability_conversion",
        "row_binding": "identical_to_decoded_track_order",
    }


def check_occupancy_contract():
    """Validate native prediction relabel semantics without running OccHead."""
    seg = np.zeros((1, 5, 1, 200, 200), np.int64)
    raw = np.zeros((1, 5, 200, 200), np.int64)

    # Same raw query ID in two frames must remain one temporally consistent ID.
    seg[0, 0, 0, 10, 10] = 1
    raw[0, 0, 10, 10] = 4
    seg[0, 1, 0, 10, 11] = 1
    raw[0, 1, 10, 11] = 4
    # A second raw query ID becomes the next consecutive ID even if there is a gap.
    seg[0, 2, 0, 20, 20] = 1
    raw[0, 2, 20, 20] = 9

    relabeled = consecutive_occupancy_ids(raw)
    if not np.array_equal(np.unique(relabeled), np.asarray([0, 1, 2], np.int64)):
        raise AssertionError("occupancy IDs were not relabeled consecutively over the full sequence")
    if relabeled[0, 0, 10, 10] != 1 or relabeled[0, 1, 10, 11] != 1:
        raise AssertionError("temporal occupancy instance identity was not preserved")
    if relabeled[0, 2, 20, 20] != 2:
        raise AssertionError("second occupancy instance did not receive the next consecutive ID")
    if not np.array_equal(relabeled > 0, seg[:, :, 0] > 0):
        raise AssertionError("occupancy semantic foreground and instance foreground disagree")

    empty = consecutive_occupancy_ids(np.zeros_like(raw))
    if np.any(empty):
        raise AssertionError("native no-query occupancy contract must remain all background")

    return {
        "segmentation_shape": [1, 5, 1, 200, 200],
        "instance_shape": [1, 5, 200, 200],
        "instance_relabel": "sorted_unique_over_entire_5_frame_tensor",
        "metric_ranges": {"30x30": [70, 130], "100x100": [0, 200]},
        "metric_frame_filter": "gt_occ_has_invalid_frame_must_be_false",
        "no_query": "all_background",
    }


def check_planning_contract():
    """Validate the native collision-postprocess boundary without requiring CasADi."""
    raw_plan = np.asarray([[
        [0.0, 0.0], [0.2, 0.0], [0.4, 0.0],
        [0.6, 0.0], [0.8, 0.0], [1.0, 0.0],
    ]], np.float32)
    empty_occ = np.zeros((1, 5, 1, 200, 200), np.int64)
    final_plan, info = optimize_planning(raw_plan, empty_occ, coordinate_mode="legacy_int")
    if not np.array_equal(final_plan, raw_plan):
        raise AssertionError("collision postprocess changed a plan with no selected occupancy")
    if info["solver_ran"] or info["selected_cells"] != 0:
        raise AssertionError("collision solver must not run with zero selected occupancy")

    # Reproduce the native long-tensor coordinate assignment. Cell (row=101,
    # col=102) maps to float centres (1.25, .75) but is truncated in-place to
    # integer metres (1, 0). Future planning steps 4/5/6 all reuse occ frame 4.
    occ = np.zeros((1, 5, 200, 200), np.int64)
    occ[0, 4, 101, 102] = 1
    probe_plan = np.zeros((1, 6, 2), np.float32)
    probe_plan[0, 3:, :] = np.asarray([1.0, 0.0], np.float32)
    positions = occupancy_positions(probe_plan, occ, coordinate_mode="legacy_int")
    if any(len(positions[t]) != 0 for t in range(3)):
        raise AssertionError("planning occupancy horizon selected the wrong early frames")
    for t in range(3, 6):
        if positions[t].shape != (1, 2) or not np.array_equal(positions[t][0], [1, 0]):
            raise AssertionError("legacy planning occupancy coordinate/horizon contract mismatch")

    return {
        "graph_output": "planning_raw_pre_collision_optimization",
        "native_reference": "host_optimize_planning_with_legacy_int_coordinates",
        "zero_occupancy": "raw_plan_preserved_without_solver",
        "occupancy_horizon": "t_uses_min(t_plus_1,4)",
        "legacy_cell_conversion": "axis_swap_then_float_expression_assigned_into_int64",
        "metric_prediction": "optimized_sdc_traj",
        "metric_collision_grid": "ground_truth_future_segmentation_steps_1_through_6",
    }






"""Compare Host schema/lifecycle against frozen pre-refactor NumPy runtime.

Uses deterministic session outputs, not neural inference or task acceptance.
"""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import types
import traceback

import host as candidate

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = '632fc91e0f963d59e5f3b3e63b7aafc84a053052'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_checks(run):
    source = subprocess.check_output(['git', 'show', REFERENCE + ':onnx/stateful_runtime.py'], cwd=ROOT)
    reference = types.ModuleType('frozen_stateful_runtime')
    exec(compile(source, REFERENCE + ':onnx/stateful_runtime.py', 'exec'), reference.__dict__)
    checks = []
    initial = make_initial_state()
    assert candidate.STATE_NAMES == reference.STATE_NAMES

    def verdict(module, state):
        try:
            module.validate_state(state)
            return 'pass'
        except Exception as error:
            return type(error).__name__ + ': ' + str(error)

    def compare_state(name, state, valid):
        a, b = verdict(reference, state), verdict(candidate, state)
        assert a == b, (name, a, b)
        assert (a == 'pass') == valid, (name, a)
        checks.append(name)

    compare_state('initial_901', initial, True)
    dynamic = dict(initial)
    for name in reference.STATE_NAMES:
        dynamic[name] = np.concatenate((initial[name], initial[name][:2]))
    dynamic['obj_idxes'][901:] = [0, 1]
    dynamic['max_obj_id'] = np.array(2, np.int64)
    compare_state('survivors_903', dynamic, True)
    for name, value in initial.items():
        altered = dict(initial)
        altered[name] = value.astype(np.float64)
        compare_state('dtype_' + name, altered, False)
        altered[name] = value[..., None]
        compare_state('shape_' + name, altered, False)
    missing = dict(initial); del missing['mem_bank']
    compare_state('missing_field', missing, False)
    duplicate = dict(dynamic); duplicate['obj_idxes'] = dynamic['obj_idxes'].copy(); duplicate['obj_idxes'][-1] = 0
    compare_state('duplicate_id', duplicate, False)
    bad = dict(initial); bad['query'] = initial['query'].copy(); bad['query'][0, 0] = np.nan
    compare_state('nonfinite_state', bad, False)

    class Session:
        failure = None
        def get_inputs(self):
            names = ['img','can_bus','l2g_r_mat','lidar2img','img_shape','command',
                     'has_prev_bev','prev_l2g_r','prev_l2g_t','l2g_t','time_delta'] + list(reference.STATE_NAMES) + ['prev_bev','max_obj_id']
            return [types.SimpleNamespace(name=n) for n in names]
        def get_outputs(self):
            return [types.SimpleNamespace(name=n) for n in ['bev_embed','next_max_obj_id'] + ['next_'+n for n in reference.STATE_NAMES]]
        def run(self, requested, feed):
            if self.failure == 'exception': raise RuntimeError('injected session failure')
            self.last = {n: feed[n].tolist() for n in ['can_bus','has_prev_bev','time_delta','max_obj_id']}
            values = [feed['prev_bev'].copy(), np.asarray(feed['max_obj_id'] + 3, dtype=np.int64)]
            values.extend(feed[n].copy() for n in reference.STATE_NAMES)
            if self.failure == 'nonfinite': values[0].flat[0] = np.nan
            if self.failure == 'invalid_id': values[1] = np.array(-1, np.int64)
            return values

    def state_view(runtime):
        metadata = {}
        for name in ['scene','timestamp','position','angle','rotation','translation']:
            value = getattr(runtime, name)
            metadata[name] = value.tolist() if isinstance(value, np.ndarray) else value
        return (candidate.state_digest(runtime.state), metadata)

    for policy in ['official_test_legacy','scene_reset']:
        for scope in ['session','scene']:
            sessions = [Session(), Session()]
            runtimes = [m.StatefulRuntime(s, initial, can_bus_mode=policy, id_scope=scope, model_sha256='test-model')
                        for m,s in zip([reference,candidate],sessions)]
            def step(scene, index):
                params = frame([index,index*2,0],index*5,[index,0,0],float(index))
                for rt in runtimes: rt.step(scene_token=scene,**params)
                assert state_view(runtimes[0]) == state_view(runtimes[1])
                assert sessions[0].last == sessions[1].last
            step('A',1); step('A',2)
            for failure in ['exception','nonfinite','invalid_id']:
                before = [state_view(rt) for rt in runtimes]
                for session,rt,old in zip(sessions,runtimes,before):
                    session.failure = failure
                    try: rt.step(scene_token='A',**frame([3,6,0],15,[3,0,0],3.))
                    except (RuntimeError,ValueError): pass
                    else: raise AssertionError('failure was accepted: '+failure)
                    assert state_view(rt) == old
                    session.failure = None
            saved = run / (policy + '_' + scope + '.npz')
            runtimes[0].save_checkpoint(saved)
            resumed = candidate.StatefulRuntime(Session(),initial,can_bus_mode=policy,id_scope=scope,model_sha256='test-model')
            resumed.load_checkpoint(saved)
            assert state_view(resumed) == state_view(runtimes[0])
            # Bidirectional checkpoint compatibility, followed by continued step.
            candidate_file = run / (policy + '_' + scope + '_candidate.npz')
            runtimes[1].save_checkpoint(candidate_file)
            runtimes[0].load_checkpoint(candidate_file)
            step('B',3)
            rejected = candidate.StatefulRuntime(Session(),initial,can_bus_mode=policy,id_scope=scope,model_sha256='other-model')
            before = state_view(rejected)
            try: rejected.load_checkpoint(saved)
            except ValueError: pass
            else: raise AssertionError('model hash mismatch accepted')
            assert state_view(rejected) == before
            checks.append(policy + '/' + scope + '/reset_rollback_restart_hash')
    return dict(status='pass',checks=checks,reference=dict(commit=REFERENCE,sha256=hashlib.sha256(source).hexdigest()),
                scope='NumPy schema/session lifecycle only; not neural/state-transition parity',
                source_hashes={n:digest(ROOT/'onnx'/n) for n in ['state_contract.py','host.py','check_host.py']})






def main():
    parser=argparse.ArgumentParser(description="Host contract and frozen lifecycle checks")
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=False)
    try:
        result=run_checks(run)
        result['contract']={name:globals()['check_'+name]() for name in ['preprocess','legacy_runtime','scene_reset_alternative','map_contract','motion_contract','occupancy_contract','planning_contract']}
    except Exception:
        (run/'result.json').write_text(json.dumps(dict(status='fail',traceback=traceback.format_exc()),indent=2)+'\n');raise
    (run/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(result['checks']),'lifecycle and',len(result['contract']),'contract groups')

if __name__ == '__main__':main()
