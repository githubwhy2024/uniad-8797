"""NumPy/ORT host state management; no PyTorch or MMDetection at inference.

The caller supplies synchronized/validated camera/pose metadata. This is an
offline inference adapter, not an autonomous vehicle control interface.
"""
import numpy as np
import hashlib
import json
import os
import tempfile
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
    outputs = []
    for image in images:
        if image.shape != (900, 1600, 3):
            raise ValueError("expected native 900x1600 BGR; resize requires matching calibration changes")
        normalized = image.astype(np.float32) - mean
        padded = np.zeros((928, 1600, 3), dtype=np.float32)
        padded[:900] = normalized
        outputs.append(padded.transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(outputs)[None]), np.array([[[928, 1600]] * 6], dtype=np.int64)


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
    for t in range(6):
        cells = np.argwhere(occupancy[0, min(t+1, 4)] != 0)[:, [1, 0]]
        if coordinate_mode == "cell_centers":
            cells = cells.astype(np.float32)
        # Assignment intentionally retains int64 in legacy mode.
        cells[:, 0] = (cells[:, 0] - 100) * .5 + .25
        cells[:, 1] = (cells[:, 1] - 100) * .5 + .25
        distance = ((planning[0, t] - cells) ** 2).sum(-1)
        result.append(cells[distance < filter_range**2])
    return result


def optimize_planning(planning, occupancy, *, coordinate_mode="legacy_int", filter_range=5., sigma=1., alpha=5.):
    """Call the existing CasADi solver; never silently report raw as optimized."""
    positions = occupancy_positions(planning, occupancy, coordinate_mode=coordinate_mode, filter_range=filter_range)
    count = sum(len(p) for p in positions)
    info = dict(coordinate_mode=coordinate_mode, selected_cells=count, solver_ran=False)
    if not count:
        return planning.copy(), info
    source = Path(__file__).resolve().parents[2] / "projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/collision_optimization.py"
    spec = importlib.util.spec_from_file_location("uniad_host_collision_optimizer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    optimizer = module.CollisionNonlinearOptimizer(6, .5, sigma, alpha, positions)
    optimizer.set_reference_trajectory(planning[0])
    solution = optimizer.solve()
    result = np.stack((solution.value(optimizer.position_x), solution.value(optimizer.position_y)), axis=-1)[None]
    if not np.isfinite(result).all():
        raise ValueError("collision optimizer returned nonfinite result")
    info["solver_ran"] = True
    return result.astype(planning.dtype), info
