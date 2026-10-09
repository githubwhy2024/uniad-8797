"""NumPy/ORT host state management; no PyTorch or MMDetection at inference.

The caller supplies synchronized/validated camera/pose metadata. This is an
offline inference adapter, not an autonomous vehicle control interface.
"""
import numpy as np
import copy
import hashlib
import json
import os
import tempfile
import time
import threading
from pathlib import Path


from state_contract import TRACK_STATE_NAMES as STATE_NAMES, STATE_DTYPES, state_shapes


def state_digest(state):
    """Canonical binding to the exact learned initialization, not just shapes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = np.ascontiguousarray(state[name])
        digest.update(json.dumps([name, str(state[name].dtype), list(state[name].shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def preprocess_base_images(images):
    """Base test pipeline: six native 900x1600 BGR images, normalize then pad.

    Camera ordering must already match training calibration order. No resize,
    RGB swap, augmentation, or calibration modification is performed here.
    """
    if len(images) != 6:
        raise ValueError("six camera images required")
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    # Write directly to the final NCHW allocation; padding stays zero.
    output = np.zeros((1, 6, 3, 928, 1600), dtype=np.float32)
    for camera, image in enumerate(images):
        if image.shape != (900, 1600, 3):
            raise ValueError("expected native 900x1600 BGR; resize requires matching calibration changes")
        for channel in range(3):
            source = image[:, :, channel]
            if source.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
                source = source.astype(np.float32, copy=False)
            np.subtract(source, mean[channel], out=output[0, camera, channel, :900], dtype=np.float32)
    return output, np.array([[[928, 1600]] * 6], dtype=np.int64)


def validate_state(state):
    if set(state) != set(STATE_NAMES) | {"prev_bev", "max_obj_id"}:
        raise ValueError("initial/runtime state fields do not match v1 contract")
    n = state["query"].shape[0]
    shapes = state_shapes(n)
    if n < 901:
        raise ValueError("missing fixed 901 initialized queries")
    for name, shape in shapes.items():
        expected_type = np.dtype(STATE_DTYPES[name]).type
        value = state[name]
        if value.shape != shape or value.dtype != expected_type:
            raise ValueError(f"invalid state {name}: {value.shape} {value.dtype}; expected {shape} {expected_type}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite state {name}")
    if not np.all(state["obj_idxes"][:901] == -1) or np.any(state["obj_idxes"][901:] < 0):
        raise ValueError("state must contain fresh 901 rows followed by surviving objects")
    active_ids = state["obj_idxes"][901:]
    if state["max_obj_id"] < 0 or np.any(state["disappear_time"] < 0):
        raise ValueError("negative ID counter/disappearance counter")
    if len(np.unique(active_ids)) != len(active_ids) or np.any(active_ids >= state["max_obj_id"]):
        raise ValueError("duplicate/out-of-range persistent IDs")


class StatefulRuntime:
    """Atomic state advancement: failed inference/validation does not commit.

    can_bus_mode is required: scene_reset is the intended scene-reset policy;
    official_test_legacy reproduces local forward_test's subtraction across
    scenes (it writes scene_token before testing whether it was None).
    id_scope=session matches local tracker counter persistence across scenes;
    id_scope=scene is an explicit alternative, never silently mixed.
    """

    def __init__(self, session, initial_state, *, can_bus_mode, id_scope="session", model_sha256=None):
        if can_bus_mode not in ("scene_reset", "official_test_legacy"):
            raise ValueError("choose an explicit can_bus mode")
        if id_scope not in ("session", "scene"):
            raise ValueError("id_scope must be session or scene")
        validate_state(initial_state)
        if initial_state["query"].shape[0] != 901 or int(initial_state["max_obj_id"]) != 0:
            raise ValueError("initial state must have exactly 901 rows and next ID zero")
        self.session = session
        self.initial = {k: v.copy() for k, v in initial_state.items()}
        self.state = {k: v.copy() for k, v in initial_state.items()}
        self.can_bus_mode, self.id_scope = can_bus_mode, id_scope
        self.model_sha256 = model_sha256
        self.initial_sha256 = state_digest(initial_state)
        self.scene = self.timestamp = self.position = self.angle = self.rotation = self.translation = None

    @classmethod
    def from_files(cls, onnx_path, initial_state_path, *, can_bus_mode, id_scope="session", threads=4):
        import onnxruntime as ort
        manifest = json.loads(Path(initial_state_path).with_suffix(".json").read_text())
        def digest(path):
            value = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    value.update(block)
            return value.hexdigest()
        if manifest["onnx_sha256"] != digest(onnx_path) or manifest["state_sha256"] != digest(initial_state_path):
            raise ValueError("ONNX/initial-state bundle hashes do not match; do not mix model versions")
        with np.load(initial_state_path, allow_pickle=False) as archive:
            state = {name: archive[name].copy() for name in archive.files}
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # Dynamic query/agent counts can retain large buffers between frames.
        # Match the low-memory acceptance allocator policy; graph unchanged.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
        return cls(session, state, can_bus_mode=can_bus_mode, id_scope=id_scope,
                   model_sha256=manifest["onnx_sha256"])

    def save_checkpoint(self, path):
        """Atomically replace a caller-selected NPZ; never serialize pickle.

        Includes all host metadata, because BEV/query alone cannot resume a
        temporal sequence. from_files binds the checkpoint to model bytes.
        Directly constructed sessions must supply a verified model_sha256.
        """
        if not self.model_sha256:
            raise ValueError("checkpoint requires a model hash; construct via from_files")
        validate_state(self.state)
        metadata = dict(format="uniad-host-state-v1", model_sha256=self.model_sha256,
                        initial_sha256=self.initial_sha256, can_bus_mode=self.can_bus_mode,
                        id_scope=self.id_scope, scene=self.scene, timestamp=self.timestamp,
                        position=None if self.position is None else self.position.tolist(),
                        angle=self.angle, rotation=None if self.rotation is None else self.rotation.tolist(),
                        translation=None if self.translation is None else self.translation.tolist(),
                        state_sha256=state_digest(self.state))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=np.array(json.dumps(metadata, allow_nan=False)), **self.state)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, path):
        """Validate everything before replacing live state; failure is atomic."""
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(self.initial) | {"metadata"}:
                raise ValueError("checkpoint field mismatch")
            metadata = json.loads(str(archive["metadata"].item()))
            state = {name: archive[name].copy() for name in self.initial}
        required = dict(format="uniad-host-state-v1", model_sha256=self.model_sha256,
                        initial_sha256=self.initial_sha256, can_bus_mode=self.can_bus_mode, id_scope=self.id_scope)
        if not self.model_sha256 or any(metadata.get(k) != v for k, v in required.items()):
            raise ValueError("checkpoint model/initial-state/policy mismatch")
        validate_state(state)
        if metadata.get("state_sha256") != state_digest(state):
            raise ValueError("checkpoint state checksum mismatch")
        scene, timestamp = metadata["scene"], metadata["timestamp"]
        position, angle = metadata["position"], metadata["angle"]
        rotation, translation = metadata["rotation"], metadata["translation"]
        if scene is None:
            if any(v is not None for v in (timestamp, position, angle, rotation, translation)) or state_digest(state) != self.initial_sha256:
                raise ValueError("invalid pre-first-frame checkpoint")
        else:
            if (not isinstance(scene, str) or not scene or not isinstance(timestamp, (float, int))
                    or isinstance(timestamp, bool) or not np.isfinite(timestamp)
                    or not isinstance(angle, (float, int)) or isinstance(angle, bool) or not np.isfinite(angle)):
                raise ValueError("invalid checkpoint scene/time/angle")
            position, rotation, translation = [np.asarray(v, dtype=np.float32) for v in (position, rotation, translation)]
            for value, shape in [(position, (3,)), (rotation, (3, 3)), (translation, (3,))]:
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError("invalid checkpoint pose metadata")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
                raise ValueError("checkpoint rotation is not proper rigid rotation")
        self.state = state
        self.scene, self.timestamp, self.position, self.angle = scene, timestamp, position, angle
        self.rotation, self.translation = rotation, translation

    def step(self, *, scene_token, timestamp, image, can_bus_absolute, l2g_r,
             l2g_t, lidar2img, img_shape, command):
        if not isinstance(scene_token, str) or not scene_token:
            raise ValueError("nonempty scene_token required")
        if not np.isfinite(timestamp):
            raise ValueError("finite timestamp in seconds required")
        if isinstance(command, (bool, np.bool_)) or int(command) != command or not 0 <= command < 3:
            raise ValueError("command must be integer 0,1,2 (three learned navigation embeddings)")
        requirements = [(image, (1, 6, 3, 928, 1600), np.float32),
                        (can_bus_absolute, (18,), np.float32), (l2g_r, (3, 3), np.float32),
                        (l2g_t, (3,), np.float32), (lidar2img, (1, 6, 4, 4), np.float32),
                        (img_shape, (1, 6, 2), np.int64)]
        for value, shape, dtype in requirements:
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"invalid frame tensor: expected {shape} {dtype}")
        if not np.allclose(l2g_r.T @ l2g_r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(l2g_r), 1., atol=1e-4):
            raise ValueError("pose rotation must be finite, nonsingular, proper rigid rotation")
        if not np.all(img_shape == [928, 1600]):
            raise ValueError("img_shape must describe the padded base image")
        new_scene = scene_token != self.scene
        if not new_scene and timestamp <= self.timestamp:
            raise ValueError("timestamps must increase strictly within each scene")
        incoming = {k: v.copy() for k, v in (self.initial if new_scene else self.state).items()}
        if new_scene and self.id_scope == "session":
            incoming["max_obj_id"] = self.state["max_obj_id"].copy()
        bus = can_bus_absolute.copy()
        if new_scene and self.can_bus_mode == "scene_reset":
            bus[:3], bus[-1] = 0, 0
        else:
            bus[:3] -= 0 if self.position is None else self.position
            bus[-1] -= 0 if self.angle is None else self.angle
        feed = dict(img=image, can_bus=bus[None], l2g_r_mat=l2g_r[None], lidar2img=lidar2img,
                    img_shape=img_shape, command=np.array([command], dtype=np.int64),
                    has_prev_bev=np.array(not new_scene, dtype=np.bool_),
                    prev_l2g_r=l2g_r if new_scene else self.rotation,
                    prev_l2g_t=l2g_t if new_scene else self.translation, l2g_t=l2g_t,
                    time_delta=np.array(0. if new_scene else timestamp-self.timestamp, dtype=np.float32), **incoming)
        input_names = [item.name for item in self.session.get_inputs()]
        if set(input_names) != set(feed):
            raise ValueError("runtime/model input contract mismatch")
        values = self.session.run(None, {name: feed[name] for name in input_names})
        result = dict(zip([item.name for item in self.session.get_outputs()], values))
        if any(value.dtype.kind == "f" and not np.isfinite(value).all() for value in values):
            raise ValueError("model produced nonfinite outputs; state not committed")
        next_state = {name: result["next_" + name].copy() for name in STATE_NAMES}
        next_state.update(prev_bev=result["bev_embed"].copy(), max_obj_id=result["next_max_obj_id"].copy())
        validate_state(next_state)
        self.state = next_state
        self.scene, self.timestamp = scene_token, timestamp
        self.position, self.angle = can_bus_absolute[:3].copy(), float(can_bus_absolute[-1])
        self.rotation, self.translation = l2g_r.copy(), l2g_t.copy()
        return result


"""Host postprocessing, keeping neural computations inside the exported graph."""
import importlib.util



def consecutive_occupancy_ids(instances):
    """Official sorted-unique relabeling (usually includes background zero)."""
    if instances.dtype.kind not in "iu" or np.any(instances < 0):
        raise ValueError("nonnegative integer occupancy IDs required")
    _, inverse = np.unique(instances, return_inverse=True)
    return inverse.reshape(instances.shape).astype(np.int64)


def merge_map_masks(outputs, *, num_things=3, num_stuff=1,
                    quality_things=.1, quality_stuff=.25,
                    overlap_things=.4, overlap_stuff=.2,
                    reject_ambiguous_ties=False):
    """Pansegformer greedy merge after learned mask decoding.

    Equal-quality ties use stable incoming order by default. Native
    ``torch.sort(..., descending=True)`` does not promise stable equal-key
    ordering, so validation can set ``reject_ambiguous_ties=True`` to fail
    rather than silently choose a potentially different merge order.
    """
    all_masks = outputs["map_mask_scores"]
    boxes = outputs["map_selected_boxes"].copy()
    labels = outputs["map_selected_labels"]
    if all_masks.ndim != 3 or boxes.shape != (len(labels), 5) or all_masks.shape[0] != len(labels)+num_stuff:
        raise ValueError("map output shapes do not match")
    if not np.isfinite(all_masks).all() or not np.isfinite(boxes).all():
        raise ValueError("nonfinite map output")
    masks = all_masks[:-num_stuff]
    segmented = masks > .5
    quality = (masks * segmented.astype(np.float32)).sum((1, 2)) / (segmented.sum((1, 2)).astype(np.float32) + 1)
    scores = boxes[:, -1] * (quality ** 2)
    if reject_ambiguous_ties:
        thresholds = np.where(labels < num_things, quality_things, quality_stuff)
        eligible_scores = scores[scores >= thresholds]
        if len(eligible_scores) > 1:
            _, counts = np.unique(eligible_scores, return_counts=True)
            if np.any(counts > 1):
                raise ValueError("equal eligible map merge scores: native torch.sort tie order is unspecified")
    order = np.argsort(-scores, kind="stable")
    masks, labels, boxes, segmented, scores = masks[order], labels[order], boxes[order], segmented[order], scores[order]
    boxes[:, -1] = scores
    height, width = all_masks.shape[-2:]
    panoptic = np.zeros((2, height, width), np.int64)
    lane = np.zeros((num_things, height, width), np.int64)
    lane_score = np.zeros((num_things, height, width), all_masks.dtype)
    unique_id = 1
    for mask_scores, label, score in zip(masks, labels, scores):
        thing = label < num_things
        if score < (quality_things if thing else quality_stuff):
            continue
        mask = mask_scores > .5
        area = int(mask.sum())
        overlap = int((mask & (panoptic[0] > 0)).sum())
        if area == 0 or overlap / area > (overlap_things if thing else overlap_stuff):
            continue
        if overlap:
            mask = mask & (panoptic[0] == 0)
        panoptic[0, mask] = label
        if thing:
            lane[label, mask] = 1
            lane_score[label, mask] = mask_scores[mask]
            panoptic[1, mask] = unique_id
            unique_id += 1
    things = labels < num_things
    return dict(bbox=boxes[things][:100], labels=labels[things][:100], segm=segmented[things][:100],
                panoptic=panoptic.transpose(1, 2, 0), lane=lane, lane_score=lane_score,
                drivable=all_masks[-1] > .5, score_list=all_masks,
                stuff_score_list=scores[labels >= num_things])


def map_iou_counts(map_result, gt_labels, gt_masks, *, num_things=3):
    """Reproduce PansegformerHead.forward_test map IoU numerators/unions.

    ``gt_masks`` follows NuScenesE2EDataset.get_data_info(): thing-instance
    masks first and the drivable-area mask last. The returned keys mirror the
    native ``ret_iou`` dictionary, but values are plain Python numbers.
    """
    labels = np.asarray(gt_labels)
    masks = np.asarray(gt_masks)
    lane = np.asarray(map_result["lane"])
    drivable = np.asarray(map_result["drivable"])
    if labels.ndim != 1 or masks.ndim != 3 or len(labels) != len(masks) or len(masks) < 1:
        raise ValueError("invalid map ground-truth layout")
    if lane.shape != (num_things, *masks.shape[-2:]) or drivable.shape != masks.shape[-2:]:
        raise ValueError("map prediction/ground-truth shape mismatch")
    if labels[-1] != num_things:
        raise ValueError("expected final ground-truth mask to be drivable-area label")

    def counts(pred, target):
        pred = np.asarray(pred) > 0
        target = np.asarray(target) > 0
        intersection = int(np.logical_and(pred, target).sum())
        union = int(pred.sum() + target.sum() - intersection)
        return intersection, union, intersection / (union + 1e-13)

    drivable_i, drivable_u, drivable_iou = counts(drivable, masks[-1])
    lanes_i, lanes_u, lanes_iou = counts(lane.sum(0) > 0, masks[:-1].sum(0) > 0)
    per_class = []
    for class_id in range(num_things):
        gt = masks[labels == class_id].sum(0) > 0
        per_class.append(counts(lane[class_id], gt))
    names = ("divider", "crossing", "contour")
    result = {
        "drivable_intersection": drivable_i, "drivable_union": drivable_u,
        "lanes_intersection": lanes_i, "lanes_union": lanes_u,
        "drivable_iou": float(drivable_iou), "lanes_iou": float(lanes_iou),
    }
    for name, (intersection, union, iou) in zip(names, per_class):
        result[f"{name}_intersection"] = intersection
        result[f"{name}_union"] = union
        result[f"{name}_iou"] = float(iou)
    return result


def motion_evaluator_view(outputs, *, modes=6, steps=12):
    """Return the exact motion subset consumed by native dataset formatting.

    Native ``MotionHead.get_trajs`` returns five parameters per future step,
    but ``NuScenesE2EDataset._format_bbox`` immediately slices ``[..., :2]``.
    The exported graph therefore needs only XY plus the unchanged log-softmax
    mode scores. Row order must stay identical to decoded track-box order.
    """
    required = ("track_boxes", "track_scores", "track_labels", "track_ids",
                "track_query_indices", "motion_xy", "motion_log_scores")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise ValueError(f"missing motion/track outputs: {missing}")
    boxes = np.asarray(outputs["track_boxes"])
    count = boxes.shape[0] if boxes.ndim == 2 else -1
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("track_boxes must have shape [N,9]")
    for name in ("track_scores", "track_labels", "track_ids", "track_query_indices"):
        if np.asarray(outputs[name]).shape != (count,):
            raise ValueError(f"{name} row count does not match decoded tracks")
    traj = np.asarray(outputs["motion_xy"])
    score = np.asarray(outputs["motion_log_scores"])
    if traj.shape != (count, modes, steps, 2) or score.shape != (count, modes):
        raise ValueError("motion output shape does not match decoded-track order")
    if not np.isfinite(traj).all() or not np.isfinite(score).all():
        raise ValueError("nonfinite motion output")
    return {"traj": traj.copy(), "traj_scores": score.copy()}


def occupancy_positions(planning, occupancy, *, coordinate_mode="legacy_int", filter_range=5.):
    """Select near-trajectory cells in the official horizon/axis convention.

    The local official method assigns metre coordinates into an int64 nonzero
    tensor, truncating cell centres. legacy_int reproduces it; cell_centers is
    an explicit alternate algorithm and must not be called official-equivalent.
    """
    if coordinate_mode not in ("legacy_int", "cell_centers"):
        raise ValueError("invalid occupancy coordinate mode")
    if planning.shape != (1, 6, 2) or not np.isfinite(planning).all():
        raise ValueError("finite 1x6x2 planning trajectory required")
    if occupancy.shape == (1, 5, 1, 200, 200):
        occupancy = occupancy[:, :, 0]
    if occupancy.shape != (1, 5, 200, 200):
        raise ValueError("expected 5-frame 200x200 occupancy")
    result = []
    coordinates = {}
    for t in range(6):
        horizon = min(t+1, 4)
        if horizon not in coordinates:
            cells = np.argwhere(occupancy[0, horizon] != 0)[:, [1, 0]]
            if coordinate_mode == "cell_centers":
                cells = cells.astype(np.float32)
            # Assignment intentionally retains int64 in legacy mode.
            cells[:, 0] = (cells[:, 0] - 100) * .5 + .25
            cells[:, 1] = (cells[:, 1] - 100) * .5 + .25
            coordinates[horizon] = cells
        cells = coordinates[horizon]
        distance = ((planning[0, t] - cells) ** 2).sum(-1)
        result.append(cells[distance < filter_range**2])
    return result


_planning_solver_local = threading.local()


def planning_optimizer(source, positions, sigma, alpha):
    """Reuse one bounded, thread-owned parameterized frozen solver problem.

    Source bytes are checked on every call. Obstacle order, objective weights,
    reference initial guess and IPOPT options are retained. Padding contributes
    zero cost. More than 512 selected cells at any horizon uses the original
    problem rather than truncating obstacles. No solution is used as a warm start.
    """
    data = source.read_bytes()
    key = (str(source), hashlib.sha256(data).hexdigest(), float(sigma), float(alpha))
    cached = getattr(_planning_solver_local, "entry", None)
    if cached is None or cached[0] != key:
        # Compile the bytes we hashed; do not reopen a changing source for exec.
        from types import ModuleType
        module = ModuleType("uniad_host_collision_optimizer")
        module.__file__ = str(source)
        exec(compile(data, str(source), "exec"), module.__dict__)

        class ParameterizedOptimizer(module.CollisionNonlinearOptimizer):
            def _create_parameters(self):
                super()._create_parameters()
                self.obstacles = [self._optimizer.parameter(2, 512) for _ in range(6)]
                self.enabled = [self._optimizer.parameter(1, 512) for _ in range(6)]

            def _set_objective(self):
                from casadi import sum2
                cost_stage = module.sumsqr(self.ref_traj[:2, :] - module.vertcat(self.position_x, self.position_y))
                cost_collision = 0
                normalizer = 1 / (2.507 * self.sigma)
                for t in range(6):
                    dx = self.position_x[t] - self.obstacles[t][0, :]
                    dy = self.position_y[t] - self.obstacles[t][1, :]
                    cost_collision += sum2(self.enabled[t] * self.alpha_collision * normalizer * module.exp(-(dx**2 + dy**2)/2/self.sigma**2))
                self._optimizer.minimize(cost_stage + cost_collision)

            def set_positions(self, points):
                for t, values in enumerate(points):
                    coordinates = np.zeros((2, 512), dtype=np.float64)
                    enabled = np.zeros((1, 512), dtype=np.float64)
                    coordinates[:, :len(values)] = values.T
                    enabled[:, :len(values)] = 1
                    self._optimizer.set_value(self.obstacles[t], coordinates)
                    self._optimizer.set_value(self.enabled[t], enabled)

        cached = (key, module, ParameterizedOptimizer(6, .5, sigma, alpha, positions))
        _planning_solver_local.entry = cached
    if any(len(v) > 512 for v in positions):
        return cached[1].CollisionNonlinearOptimizer(6, .5, sigma, alpha, positions), False
    optimizer = cached[2]
    optimizer.set_positions(positions)
    return optimizer, True


def optimize_planning(planning, occupancy, *, coordinate_mode="legacy_int", filter_range=5., sigma=1., alpha=5., optimizer_source=None, timings=None):
    """Call the frozen collision objective; never report raw as optimized."""
    start = time.perf_counter()
    positions = occupancy_positions(planning, occupancy, coordinate_mode=coordinate_mode, filter_range=filter_range)
    selected_at = time.perf_counter()
    count = sum(len(p) for p in positions)
    info = dict(coordinate_mode=coordinate_mode, selected_cells=count, solver_ran=False)
    if not count:
        if timings is not None:
            timings.update(selection_seconds=selected_at-start, total_seconds=time.perf_counter()-start)
        return planning.copy(), info
    source = (Path(optimizer_source).resolve() if optimizer_source is not None else
              Path(__file__).resolve().parents[2] / "projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/collision_optimization.py")
    optimizer, reused_problem = planning_optimizer(source, positions, sigma, alpha)
    built_at = time.perf_counter()
    try:
        optimizer.set_reference_trajectory(planning[0])
        reference_at = time.perf_counter()
        solution = optimizer.solve()
        solved_at = time.perf_counter()
        result = np.stack((solution.value(optimizer.position_x), solution.value(optimizer.position_y)), axis=-1)[None]
        if not np.isfinite(result).all():
            raise ValueError("collision optimizer returned nonfinite result")
        final = result.astype(planning.dtype)
    except BaseException:
        # A failed solve must not leave cached problem state available for retry.
        _planning_solver_local.entry = None
        raise
    info["solver_ran"] = True
    if timings is not None:
        timings.update(selection_seconds=selected_at-start, resource_problem_seconds=built_at-selected_at,
                       reference_seconds=reference_at-built_at, solve_seconds=solved_at-reference_at,
                       conversion_seconds=time.perf_counter()-solved_at,
                       total_seconds=time.perf_counter()-start, parameterized_problem=reused_problem)
    return final, info


# Q3 fixed transaction core; real session/policy integration is a separate gate.
from state_contract import (TRACK_STATE_FIELDS, TRACK_STATE_NAMES, FRESH, TRACK_SLOTS,
    SURVIVOR_CAPACITY, DECODED_SLOTS, VEHICLE_SLOTS, FIXED_STATE_NAMES,
    SCA_CAPACITY, SCA_CAMERAS, SCA_BEV_QUERIES)

class CapacityOverflow(ValueError):
    """A fixed graph reported a true preselection overflow; no state committed."""


def _scalar(value, name, *, nonnegative=True):
    if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != np.int64:
        raise ValueError(f"{name} must be scalar int64")
    number = int(value)
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _prefix(mask, count, capacity, name):
    if not isinstance(mask, np.ndarray) or mask.shape != (capacity,) or mask.dtype != np.bool_:
        raise ValueError(f"{name} must be bool[{capacity}]")
    if not np.array_equal(mask, np.arange(capacity) < count):
        raise ValueError(f"{name} differs from packed count {count}")


def validate_fixed_state(state):
    if set(state) != FIXED_STATE_NAMES:
        raise ValueError("v2 fixed state field names differ")
    count = _scalar(state["track_count"], "track_count")
    if not FRESH <= count <= TRACK_SLOTS:
        raise ValueError("track_count outside [901,1285]")
    _prefix(state["track_valid_mask"], count, TRACK_SLOTS, "track_valid_mask")
    shapes = state_shapes(TRACK_SLOTS)
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(STATE_DTYPES[name]):
            raise ValueError(f"invalid fixed state {name} shape/dtype")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"nonfinite fixed state {name}")
    ids = state["obj_idxes"]
    if not np.all(ids[:FRESH] == -1) or not np.all(ids[FRESH:count] >= 0):
        raise ValueError("fresh/survivor ID order changed")
    if not np.all(ids[count:] == -3):
        raise ValueError("padding ID must be -3")
    live = ids[FRESH:count]
    next_id = _scalar(state["max_obj_id"], "max_obj_id")
    if len(np.unique(live)) != len(live) or np.any(live >= next_id):
        raise ValueError("duplicate/out-of-range survivor ID")
    if np.any(state["disappear_time"] < 0) or np.any(state["disappear_time"][count:] != 0):
        raise ValueError("invalid disappearance counter")
    if not np.all(state["mem_padding_mask"][count:]):
        raise ValueError("padding history slots must be masked")
    for name in ("query", "ref_pts", "pred_boxes", "mem_bank", "save_period"):
        if np.any(state[name][count:] != 0):
            raise ValueError(f"padding {name} must be zero")
    return count


def pad_v1_initial_state(state):
    """Convert only the exact 901-row learned initial state, never live v1 state."""
    validate_state(state)
    if state["query"].shape[0] != FRESH or int(state["max_obj_id"]) != 0:
        raise ValueError("only exact pre-first-frame v1 initialization can be padded")
    out = {}
    for name, tail, dtype in TRACK_STATE_FIELDS:
        fill = True if name == "mem_padding_mask" else -3 if name == "obj_idxes" else 0
        value = np.full((TRACK_SLOTS,) + tail, fill, dtype=np.dtype(dtype))
        value[:FRESH] = state[name]
        out[name] = value
    out.update(prev_bev=state["prev_bev"].copy(), max_obj_id=state["max_obj_id"].copy(),
               track_count=np.array(FRESH, dtype=np.int64),
               track_valid_mask=(np.arange(TRACK_SLOTS) < FRESH))
    validate_fixed_state(out)
    return out



def validate_sca_outputs(outputs, flags):
    """Reject missing, stale or contradictory SCA diagnostics before state commit."""
    raw = outputs.get("sca_visible_count_raw")
    if (not isinstance(raw, np.ndarray) or raw.shape != (SCA_CAMERAS,)
            or raw.dtype != np.int64 or np.any(raw < 0) or np.any(raw > SCA_BEV_QUERIES)):
        raise ValueError("sca_visible_count_raw must be int64[6] within the configured grid")
    for index, name in enumerate(("survivor_overflow", "vehicle_overflow", "sca_overflow")):
        value = outputs.get(name)
        if not isinstance(value, np.ndarray) or value.shape != () or value.dtype != np.bool_:
            raise ValueError(f"{name} must be scalar bool")
        if bool(value) != bool(flags[index]):
            raise ValueError(f"{name} differs from aggregate overflow_flags")
    if bool(flags[2]) != bool(np.any(raw > SCA_CAPACITY)):
        raise ValueError("SCA raw count and overflow flag inconsistent")
    return raw



def validate_frame_metadata(metadata):
    """Validate the pose/time needed to resume a committed planning frame."""
    names = {"scene", "timestamp", "position", "angle", "rotation", "translation"}
    if not isinstance(metadata, dict) or set(metadata) != names:
        raise ValueError("planning metadata must contain scene/time/absolute pose")
    if not isinstance(metadata["scene"], str) or not metadata["scene"]:
        raise ValueError("nonempty scene required")
    for name in ("timestamp", "angle"):
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("finite numeric " + name + " required")
    for name, shape in (("position", (3,)), ("rotation", (3, 3)), ("translation", (3,))):
        value = np.asarray(metadata[name], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid planning metadata " + name)
    rotation = np.asarray(metadata["rotation"], dtype=np.float32)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4):
        raise ValueError("planning metadata rotation must be proper rigid rotation")
    json.dumps(metadata, allow_nan=False)


def prepare_fixed_frame_feed(previous, *, can_bus_mode, scene_token, timestamp,
                             image, can_bus_absolute, l2g_r, l2g_t, lidar2img,
                             img_shape, command):
    """Prepare eleven non-state inputs using the last successful frame only."""
    if can_bus_mode not in ("scene_reset", "official_test_legacy"):
        raise ValueError("explicit can_bus policy required")
    if previous:
        validate_frame_metadata(previous)
    if not isinstance(scene_token, str) or not scene_token:
        raise ValueError("nonempty scene_token required")
    if isinstance(timestamp, (bool, np.bool_)) or not np.isscalar(timestamp) or not np.isfinite(timestamp):
        raise ValueError("finite timestamp in seconds required")
    if isinstance(command, (bool, np.bool_)) or not np.isscalar(command) or int(command) != command or not 0 <= command < 3:
        raise ValueError("command must be integer 0,1,2")
    requirements = [(image, (1, 6, 3, 928, 1600), np.float32),
                    (can_bus_absolute, (18,), np.float32), (l2g_r, (3, 3), np.float32),
                    (l2g_t, (3,), np.float32), (lidar2img, (1, 6, 4, 4), np.float32),
                    (img_shape, (1, 6, 2), np.int64)]
    for value, shape, dtype in requirements:
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"invalid frame tensor: expected {shape} {dtype}")
    if not np.allclose(l2g_r.T @ l2g_r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(l2g_r), 1., atol=1e-4):
        raise ValueError("pose rotation must be proper rigid rotation")
    if not np.all(img_shape == [928, 1600]):
        raise ValueError("img_shape must describe padded base image")
    reset = scene_token != previous.get("scene")
    if not reset and timestamp <= previous["timestamp"]:
        raise ValueError("timestamps must increase strictly within each scene")
    bus = can_bus_absolute.copy()
    if reset and can_bus_mode == "scene_reset":
        bus[:3], bus[-1] = 0, 0
    else:
        bus[:3] -= 0 if previous.get("position") is None else np.asarray(previous["position"], np.float32)
        bus[-1] -= 0 if previous.get("angle") is None else previous["angle"]
    feed = dict(img=image, can_bus=bus[None], l2g_r_mat=l2g_r[None], lidar2img=lidar2img,
                img_shape=img_shape, command=np.array([command], np.int64),
                has_prev_bev=np.array(not reset, np.bool_),
                prev_l2g_r=l2g_r.copy() if reset else np.asarray(previous["rotation"], np.float32),
                prev_l2g_t=l2g_t.copy() if reset else np.asarray(previous["translation"], np.float32),
                l2g_t=l2g_t, time_delta=np.array(0. if reset else timestamp-previous["timestamp"], np.float32))
    metadata = dict(scene=scene_token, timestamp=float(timestamp), position=can_bus_absolute[:3].tolist(),
                    angle=float(can_bus_absolute[-1]), rotation=l2g_r.tolist(), translation=l2g_t.tolist())
    validate_frame_metadata(metadata)
    return feed, metadata, reset


class FrameDeadlineExceeded(ValueError):
    """The frame expired before the state/checkpoint commit boundary."""


class TransientInferenceError(RuntimeError):
    """Backend explicitly classified a temporary, retryable inference failure."""


class PlanningResourceError(ValueError):
    """A deployment resource is missing or differs from its pinned identity."""


class PlanningPostprocessError(ValueError):
    """A frame's collision optimizer failed; no final planning result exists."""


def check_frame_deadline(deadline):
    if deadline is None:
        return
    if isinstance(deadline, (bool, np.bool_)) or not np.isscalar(deadline) or not np.isfinite(deadline):
        raise ValueError("deadline must be finite monotonic time")
    if time.monotonic() >= deadline:
        raise FrameDeadlineExceeded("planning frame deadline expired")


def verified_optimizer_source(path, expected_sha256):
    try:
        source = Path(path).resolve(strict=True)
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise PlanningResourceError(str(error)) from error
    if actual != expected_sha256:
        raise PlanningResourceError("collision optimizer resource SHA256 differs")
    return source


class FixedStateTransaction:
    """Candidate v2 Host transaction around a session-like tensor runtime."""

    def __init__(self, session, initial_state, *, model_sha256, can_bus_mode, id_scope="session"):
        if (not isinstance(model_sha256, str) or len(model_sha256) != 64
                or any(char not in "0123456789abcdef" for char in model_sha256.lower())):
            raise ValueError("verified hexadecimal model SHA256 required")
        if can_bus_mode not in ("scene_reset", "official_test_legacy") or id_scope not in ("session", "scene"):
            raise ValueError("invalid Host policy")
        if validate_fixed_state(initial_state) != FRESH or int(initial_state["max_obj_id"]) != 0:
            raise ValueError("initial v2 state must contain only 901 fresh rows")
        self.session = session
        self.initial = {k: v.copy() for k, v in initial_state.items()}
        self.state = {k: v.copy() for k, v in initial_state.items()}
        self.model_sha256 = model_sha256
        self.initial_sha256 = state_digest(initial_state)
        self.can_bus_mode, self.id_scope = can_bus_mode, id_scope
        self.metadata = {}

    @classmethod
    def from_planning_bundle(cls, root, *, expected_manifest_sha256,
                             can_bus_mode, id_scope="session", threads=4):
        """Load only pinned bundle resources; no project directory or ML builder."""
        from assets import load_planning_bundle_manifest, sha256
        import state_contract
        import onnxruntime as ort
        import casadi
        manifest, paths = load_planning_bundle_manifest(root, expected_manifest_sha256=expected_manifest_sha256)
        if sha256(__file__) != manifest["artifacts"]["host"]["sha256"] or sha256(state_contract.__file__) != manifest["artifacts"]["state_contract"]["sha256"]:
            raise ValueError("loaded Host/state contract differs from bundle")
        import assets
        if sha256(assets.__file__) != manifest["artifacts"]["assets"]["sha256"]:
            raise ValueError("loaded asset verifier differs from bundle")
        expected_capacities = dict(fresh=FRESH, track=TRACK_SLOTS, survivor=SURVIVOR_CAPACITY,
                                  decoded=DECODED_SLOTS, vehicle=VEHICLE_SLOTS, sca=SCA_CAPACITY)
        if manifest["capacities"] != expected_capacities:
            raise ValueError("bundle capacities differ from Host state contract")
        if manifest["host_policy"] != dict(can_bus_mode=can_bus_mode, id_scope=id_scope, coordinate_mode="legacy_int"):
            raise ValueError("bundle Host policy differs")
        tested = manifest["runtime_versions"]
        if tested != dict(numpy=np.__version__, onnxruntime=ort.__version__, casadi=casadi.__version__):
            raise ValueError("runtime versions differ from the tested bundle")
        if type(threads) is not int or threads < 1:
            raise ValueError("positive integer ORT thread count required")
        with np.load(paths["initial_state"], allow_pickle=False) as archive:
            initial = {name: archive[name].copy() for name in archive.files}
        if state_digest(initial) != manifest["initial_state_digest"]:
            raise ValueError("bundle learned initialization differs")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(paths["model"]), sess_options=options, providers=["CPUExecutionProvider"])
        kinds = {"float32": "tensor(float)", "int64": "tensor(int64)", "bool": "tensor(bool)"}
        for actual, expected in [(session.get_inputs(), manifest["inputs"]), (session.get_outputs(), manifest["outputs"])]:
            if [dict(name=v.name, shape=list(v.shape), dtype=v.type) for v in actual] != [dict(name=v["name"], shape=v["shape"], dtype=kinds[v["dtype"]]) for v in expected]:
                raise ValueError("actual ORT ordered shape/dtype ABI differs from bundle")
        runtime = cls(session, initial, model_sha256=manifest["artifacts"]["model"]["sha256"], can_bus_mode=can_bus_mode, id_scope=id_scope)
        runtime.bundle_manifest_sha256 = expected_manifest_sha256
        runtime.planning_optimizer_source = paths["collision_optimizer"]
        runtime.planning_optimizer_sha256 = manifest["artifacts"]["collision_optimizer"]["sha256"]
        runtime.bundle_root = Path(root).resolve()
        return runtime

    def _incoming(self, new_scene):
        state = {k: v.copy() for k, v in (self.initial if new_scene else self.state).items()}
        if new_scene and self.id_scope == "session":
            state["max_obj_id"] = self.state["max_obj_id"].copy()
        validate_fixed_state(state)
        return state

    def advance(self, frame_feed, *, metadata, new_scene=False):
        """Run once and atomically commit only a complete, overflow-free result."""
        if not isinstance(new_scene, bool) or not isinstance(metadata, dict):
            raise ValueError("explicit scene flag and metadata dictionary required")
        json.dumps(metadata, allow_nan=False)  # checkpoint-safe before inference
        incoming = self._incoming(new_scene)
        if set(frame_feed) & set(incoming):
            raise ValueError("frame feed cannot override state tensors")
        feed = {**frame_feed, **incoming}
        if hasattr(self.session, "get_inputs"):
            expected = {item.name for item in self.session.get_inputs()}
            if expected != set(feed):
                raise ValueError("v2 runtime input contract differs")
        values = self.session.run(None, feed)
        if isinstance(values, dict):
            outputs = values
        else:
            names = [item.name for item in self.session.get_outputs()]
            if len(names) != len(values) or len(set(names)) != len(names):
                raise ValueError("v2 runtime output contract differs")
            outputs = dict(zip(names, values))
        for name, value in outputs.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"non-tensor output {name}")
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"nonfinite output {name}")
        flags = outputs["overflow_flags"]
        if flags.shape != (3,) or flags.dtype != np.bool_:
            raise ValueError("overflow_flags must be bool[3]")
        validate_sca_outputs(outputs, flags)
        raw_survivor = _scalar(outputs["survivor_count_raw"], "survivor_count_raw")
        raw_vehicle = _scalar(outputs["vehicle_count_raw"], "vehicle_count_raw")
        packed_track = _scalar(outputs["next_track_count"], "next_track_count")
        packed_vehicle = _scalar(outputs["vehicle_count"], "vehicle_count")
        decoded = _scalar(outputs["decoded_count"], "decoded_count")
        if (raw_survivor > TRACK_SLOTS or raw_vehicle > DECODED_SLOTS
                or not FRESH <= packed_track <= TRACK_SLOTS
                or not 0 <= packed_vehicle <= VEHICLE_SLOTS
                or not 0 <= decoded <= DECODED_SLOTS
                or raw_vehicle > decoded):
            raise ValueError("raw/packed output count outside ABI limits")
        _prefix(outputs["next_track_valid_mask"], packed_track, TRACK_SLOTS, "next_track_valid_mask")
        _prefix(outputs["decoded_valid_mask"], decoded, DECODED_SLOTS, "decoded_valid_mask")
        _prefix(outputs["vehicle_valid_mask"], packed_vehicle, VEHICLE_SLOTS, "vehicle_valid_mask")
        if (packed_track != FRESH + min(raw_survivor, SURVIVOR_CAPACITY)
                or packed_vehicle != min(raw_vehicle, VEHICLE_SLOTS)
                or bool(flags[0]) != (raw_survivor > SURVIVOR_CAPACITY)
                or bool(flags[1]) != (raw_vehicle > VEHICLE_SLOTS)):
            raise ValueError("raw/packed count, mask or overflow flag inconsistent")
        if bool(flags.any()):
            raise CapacityOverflow(f"v2 overflow flags={flags.tolist()} raw_survivor={raw_survivor} raw_vehicle={raw_vehicle}")
        next_state = {name: outputs["next_" + name].copy() for name in TRACK_STATE_NAMES}
        next_state.update(prev_bev=outputs["bev_embed"].copy(),
                          max_obj_id=outputs["next_max_obj_id"].copy(),
                          track_count=outputs["next_track_count"].copy(),
                          track_valid_mask=outputs["next_track_valid_mask"].copy())
        validate_fixed_state(next_state)
        self.state = next_state
        self.metadata = copy.deepcopy(metadata)
        return outputs

    def advance_planning(self, frame_feed, *, metadata, new_scene=False,
                         optimizer_source, optimizer_sha256, checkpoint_path=None, deadline_monotonic=None):
        """Commit state and pose only after planning and optional checkpoint succeed.

        A single caller owns this temporal stream. Inference, postprocessing and
        checkpoint errors leave the last successful frame available for retry.
        The application receives only the final float32 planning trajectory.
        """
        check_frame_deadline(deadline_monotonic)
        validate_frame_metadata(metadata)
        source = verified_optimizer_source(optimizer_source, optimizer_sha256)
        staged = copy.copy(self)
        # advance only assigns staged fields; its input state is copied before
        # session.run, so an inference failure cannot mutate live state arrays.
        outputs = staged.advance(frame_feed, metadata=metadata, new_scene=new_scene)
        raw, occupancy = outputs["planning_raw"], outputs["occ_segmentation"]
        if raw.shape != (1, 6, 2) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise ValueError("planning_raw must be finite float32[1,6,2]")
        if occupancy.shape != (1, 5, 1, 200, 200) or occupancy.dtype != np.int64:
            raise ValueError("occ_segmentation must be int64[1,5,1,200,200]")
        if np.any((occupancy != 0) & (occupancy != 1)):
            raise ValueError("occ_segmentation must be binary")
        try:
            final, information = optimize_planning(raw, occupancy, coordinate_mode="legacy_int", optimizer_source=source)
        except (RuntimeError, ValueError) as error:
            raise PlanningPostprocessError(str(error)) from error
        if (not isinstance(final, np.ndarray) or final.shape != (1, 6, 2)
                or final.dtype != np.float32 or not np.isfinite(final).all()):
            raise ValueError("collision optimizer returned invalid planning ABI")
        if (not isinstance(information, dict) or information.get("coordinate_mode") != "legacy_int"
                or type(information.get("solver_ran")) is not bool
                or type(information.get("selected_cells")) is not int
                or information["selected_cells"] < 0
                or information["solver_ran"] != (information["selected_cells"] > 0)):
            raise ValueError("invalid collision optimization information")
        # Complete copies/validation before the atomic checkpoint replacement.
        final = final.copy()
        information = copy.deepcopy(information)
        check_frame_deadline(deadline_monotonic)
        if checkpoint_path is not None:
            staged.save_checkpoint(checkpoint_path, deadline_monotonic=deadline_monotonic)
        self.state, self.metadata = staged.state, staged.metadata
        self.last_planning_info = information
        return final

    def step_planning(self, *, optimizer_source, optimizer_sha256, checkpoint_path=None, deadline_monotonic=None, **frame):
        """Prepare pose/time inputs and transact one complete planning frame."""
        check_frame_deadline(deadline_monotonic)
        feed, metadata, reset = prepare_fixed_frame_feed(self.metadata, can_bus_mode=self.can_bus_mode, **frame)
        return self.advance_planning(feed, metadata=metadata, new_scene=reset,
                                     optimizer_source=optimizer_source, optimizer_sha256=optimizer_sha256,
                                     checkpoint_path=checkpoint_path, deadline_monotonic=deadline_monotonic)

    def _validate_checkpoint_context(self, state, metadata):
        """Bind resume metadata to committed state or the exact learned reset."""
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint Host metadata must be a dictionary")
        if metadata:
            validate_frame_metadata(metadata)
            return
        reset = dict(state)
        if self.id_scope == "session":
            # Explicit temporal reset preserves session IDs while resetting BEV,
            # learned fresh queries, references and all history/count/mask fields.
            reset["max_obj_id"] = self.initial["max_obj_id"]
        if state_digest(reset) != self.initial_sha256:
            raise ValueError("empty checkpoint metadata requires exact initialization or temporal reset")

    def save_checkpoint(self, path, *, deadline_monotonic=None):
        validate_fixed_state(self.state)
        self._validate_checkpoint_context(self.state, self.metadata)
        meta = {"format": "uniad-host-state-v2", "model_sha256": self.model_sha256,
                "initial_sha256": self.initial_sha256,
                "can_bus_mode": self.can_bus_mode, "id_scope": self.id_scope,
                "capacities": {"track": TRACK_SLOTS, "decoded": DECODED_SLOTS, "vehicle": VEHICLE_SLOTS},
                "state_sha256": state_digest(self.state), "host_metadata": copy.deepcopy(self.metadata)}
        if hasattr(self, "bundle_manifest_sha256"):
            meta["planning_bundle_sha256"] = self.bundle_manifest_sha256
        encoded = json.dumps(meta, allow_nan=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, metadata=np.array(encoded), **self.state)
                stream.flush()
                os.fsync(stream.fileno())
            check_frame_deadline(deadline_monotonic)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, path):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FIXED_STATE_NAMES | {"metadata"}:
                raise ValueError("v2 checkpoint field mismatch")
            meta = json.loads(str(archive["metadata"].item()))
            candidate = {name: archive[name].copy() for name in FIXED_STATE_NAMES}
        if not isinstance(meta, dict):
            raise ValueError("v2 checkpoint envelope must be a dictionary")
        expected = {"format": "uniad-host-state-v2", "model_sha256": self.model_sha256,
                    "initial_sha256": self.initial_sha256,
                    "can_bus_mode": self.can_bus_mode, "id_scope": self.id_scope,
                    "capacities": {"track": TRACK_SLOTS, "decoded": DECODED_SLOTS, "vehicle": VEHICLE_SLOTS}}
        if hasattr(self, "bundle_manifest_sha256"):
            expected["planning_bundle_sha256"] = self.bundle_manifest_sha256
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("v2 checkpoint identity/policy/capacity mismatch")
        validate_fixed_state(candidate)
        if meta.get("state_sha256") != state_digest(candidate) or not isinstance(meta.get("host_metadata"), dict):
            raise ValueError("v2 checkpoint checksum/metadata mismatch")
        self._validate_checkpoint_context(candidate, meta["host_metadata"])
        self.state = candidate
        self.metadata = copy.deepcopy(meta["host_metadata"])


class PlanningStream:
    """One synchronized frame stream with explicit rejection and recovery results.

    Gap/failure limits are caller configuration, not validated vehicle limits.
    Default retries are zero. One retry requires explicit backend classification,
    a monotonic deadline and a configured estimate of the full attempt budget.
    A single caller owns the stream; application control fallback is separate.
    """
    def __init__(self, runtime, *, optimizer_source, optimizer_sha256,
                 max_state_gap_seconds, max_consecutive_failures,
                 max_retries=0, retry_budget_seconds=None):
        if not isinstance(runtime, FixedStateTransaction):
            raise ValueError("a fixed planning Host runtime is required")
        if isinstance(max_state_gap_seconds, (bool, np.bool_)) or not np.isscalar(max_state_gap_seconds) or not np.isfinite(max_state_gap_seconds) or max_state_gap_seconds <= 0:
            raise ValueError("positive explicit state gap limit required")
        if type(max_consecutive_failures) is not int or max_consecutive_failures < 1:
            raise ValueError("positive explicit failure count limit required")
        if type(max_retries) is not int or max_retries not in (0, 1):
            raise ValueError("max_retries must be zero or one")
        if max_retries and (isinstance(retry_budget_seconds, (bool, np.bool_)) or not np.isscalar(retry_budget_seconds) or not np.isfinite(retry_budget_seconds) or retry_budget_seconds <= 0):
            raise ValueError("a positive retry attempt budget is required")
        self.optimizer_source = verified_optimizer_source(optimizer_source, optimizer_sha256)
        self.optimizer_sha256 = optimizer_sha256
        self.runtime = runtime
        self.max_state_gap_seconds = float(max_state_gap_seconds)
        self.max_consecutive_failures = max_consecutive_failures
        self.max_retries = max_retries
        self.retry_budget_seconds = retry_budget_seconds
        self.consecutive_failures = 0
        self.blocked = False
        self.block_reason = None

    @classmethod
    def from_bundle(cls, root, *, expected_manifest_sha256, can_bus_mode,
                    max_state_gap_seconds, max_consecutive_failures, id_scope="session",
                    threads=4, max_retries=0, retry_budget_seconds=None):
        runtime = FixedStateTransaction.from_planning_bundle(root,
            expected_manifest_sha256=expected_manifest_sha256, can_bus_mode=can_bus_mode,
            id_scope=id_scope, threads=threads)
        return cls(runtime, optimizer_source=runtime.planning_optimizer_source,
                   optimizer_sha256=runtime.planning_optimizer_sha256,
                   max_state_gap_seconds=max_state_gap_seconds,
                   max_consecutive_failures=max_consecutive_failures,
                   max_retries=max_retries, retry_budget_seconds=retry_budget_seconds)

    def _result(self, frame_id, frame, *, status, plan=None, code=None, message=None, attempts=0):
        timestamp = frame.get("timestamp")
        if not isinstance(timestamp, (int, float, np.integer, np.floating)) or isinstance(timestamp, (bool, np.bool_)) or not np.isfinite(timestamp):
            timestamp = None
        return dict(frame_id=frame_id, scene_token=frame.get("scene_token"),
                    timestamp=None if timestamp is None else float(timestamp), status=status,
                    valid=status == "accepted", plan=plan, state_committed=status == "accepted",
                    rejection_code=code, rejection_message=message, attempts=attempts,
                    consecutive_failures=self.consecutive_failures,
                    recovery_action="recover" if self.blocked else "continue_latest",
                    last_successful_timestamp=self.runtime.metadata.get("timestamp"))

    def _reject(self, frame_id, frame, code, message, attempts, *, fatal=False):
        self.consecutive_failures += 1
        if fatal or self.consecutive_failures >= self.max_consecutive_failures:
            self.blocked = True
            self.block_reason = code if fatal else "consecutive_failures"
        return self._result(frame_id, frame, status="blocked" if self.blocked else "rejected",
                            code=code, message=message, attempts=attempts)

    def process(self, *, frame_id, deadline_monotonic=None, checkpoint_path=None, **frame):
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("nonempty frame_id required")
        if self.blocked:
            return self._result(frame_id, frame, status="blocked", code=self.block_reason,
                                message="explicit recovery acknowledgement required")
        previous = self.runtime.metadata
        timestamp = frame.get("timestamp")
        if (previous and previous.get("scene") == frame.get("scene_token")
                and isinstance(timestamp, (float, int, np.floating, np.integer))
                and not isinstance(timestamp, (bool, np.bool_)) and np.isfinite(timestamp)
                and timestamp - previous["timestamp"] > self.max_state_gap_seconds):
            return self._reject(frame_id, frame, "state_gap", "last successful state is outside the configured gap limit", 0, fatal=True)
        attempts = 0
        while True:
            try:
                check_frame_deadline(deadline_monotonic)
                attempts += 1
                plan = self.runtime.step_planning(**frame, optimizer_source=self.optimizer_source,
                    optimizer_sha256=self.optimizer_sha256, checkpoint_path=checkpoint_path,
                    deadline_monotonic=deadline_monotonic)
            except TransientInferenceError as error:
                can_retry = (attempts <= self.max_retries and deadline_monotonic is not None
                    and time.monotonic() + self.retry_budget_seconds < deadline_monotonic)
                if can_retry:
                    continue
                return self._reject(frame_id, frame, "transient_inference", str(error), attempts)
            except CapacityOverflow as error:
                return self._reject(frame_id, frame, "capacity_overflow", str(error), attempts)
            except FrameDeadlineExceeded as error:
                return self._reject(frame_id, frame, "deadline", str(error), attempts)
            except PlanningResourceError as error:
                return self._reject(frame_id, frame, "resource_identity", str(error), attempts, fatal=True)
            except PlanningPostprocessError as error:
                return self._reject(frame_id, frame, "postprocess", str(error), attempts)
            except (ValueError, KeyError) as error:
                return self._reject(frame_id, frame, "invalid_frame_or_output", str(error), attempts)
            except OSError as error:
                return self._reject(frame_id, frame, "checkpoint_or_io", str(error), attempts, fatal=True)
            except Exception as error:
                return self._reject(frame_id, frame, "unclassified_runtime", str(error), attempts, fatal=True)
            self.consecutive_failures = 0
            return self._result(frame_id, frame, status="accepted", plan=plan, attempts=attempts)

    def process_latest(self, pending, *, deadline_monotonic=None, checkpoint_path=None):
        """Take the last received complete frame; superseded frames never infer."""
        if not isinstance(pending, (list, tuple)) or not pending:
            raise ValueError("nonempty bounded sequence of (frame_id, frame) required")
        for item in pending:
            if (not isinstance(item, (tuple, list)) or len(item) != 2
                    or not isinstance(item[0], str) or not item[0] or not isinstance(item[1], dict)):
                raise ValueError("invalid complete frame queue")
        frame_id, frame = pending[-1]
        result = self.process(frame_id=frame_id, deadline_monotonic=deadline_monotonic,
                              checkpoint_path=checkpoint_path, **frame)
        result["superseded_frame_ids"] = [item[0] for item in pending[:-1]]
        return result

    def recover(self, *, reason, reset_temporal_state, checkpoint_path=None):
        """Explicitly acknowledge recovery; temporal reset preserves session IDs.

        The caller resolves the underlying backend/resource problem first.
        Reset behavior is an explicit recovery mode, not silent normal inference.
        """
        if not isinstance(reason, str) or not reason or type(reset_temporal_state) is not bool:
            raise ValueError("explicit recovery reason and reset decision required")
        verified_optimizer_source(self.optimizer_source, self.optimizer_sha256)
        staged = copy.copy(self.runtime)
        if reset_temporal_state:
            staged.state = staged._incoming(True)
            staged.metadata = {}
        if checkpoint_path is not None:
            staged.save_checkpoint(checkpoint_path)
        self.runtime.state, self.runtime.metadata = staged.state, staged.metadata
        self.consecutive_failures = 0
        self.blocked = False
        self.block_reason = None
        if hasattr(self.runtime, "last_planning_info"):
            del self.runtime.last_planning_info
        return dict(recovery_reason=reason, temporal_state_reset=reset_temporal_state,
                    max_obj_id=int(self.runtime.state["max_obj_id"]), id_scope=self.runtime.id_scope)
