#!/usr/bin/env python3
"""One evidence entry point for native PT ↔ Stateful PT ↔ ORT mini validation.

Stage 2B-4a prepares/verifies the immutable 404-frame execution manifest.
Stage 2B-4b1 adds a CPU-only execution preflight. Later substages extend this
SAME script with backend execution and evaluation; no additional runner is
introduced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import pickle
import numpy as np
import subprocess
import sys
import time
import types
from pathlib import Path

from assets import EXPECTED, OFFICIAL_MINI_SPLITS


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORMAT = "uniad-mini-equivalence-v1"
SPLITS = tuple(OFFICIAL_MINI_SPLITS)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_pickle(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def atomic_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def asset(path, *, required=True):
    path = resolve(path)
    exists = path.is_file()
    if required and not exists:
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256(path) if exists else None,
    }


def load_info_source(path):
    payload = load_pickle(path)
    if not isinstance(payload, dict) or {"infos", "metadata"} - set(payload):
        raise ValueError(f"{path}: expected dict with infos + metadata")
    if payload["metadata"].get("version") != "v1.0-trainval":
        raise ValueError(f"{path}: expected stock v1.0-trainval metadata")
    return payload


def load_mini_tables(data_root):
    root = data_root / "v1.0-mini"
    scenes = json.loads((root / "scene.json").read_text(encoding="utf-8"))
    samples = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    scene_name_by_token = {row["token"]: row["name"] for row in scenes}
    sample_by_token = {row["token"]: row for row in samples}
    official_names = {
        scene_name
        for split_names in OFFICIAL_MINI_SPLITS.values()
        for scene_name in split_names
    }
    if len(scenes) != 10 or {row["name"] for row in scenes} != official_names:
        raise ValueError("v1.0-mini scene table is not the expected official 10-scene split")
    if len(samples) != 404 or len(sample_by_token) != 404:
        raise ValueError("v1.0-mini sample table must contain exactly 404 unique samples")
    return scene_name_by_token, sample_by_token


def validate_ordered_split(split, infos, scene_name_by_token, sample_by_token):
    expected_scene_names = set(OFFICIAL_MINI_SPLITS[split])
    expected = EXPECTED[split]
    tokens = [item["token"] for item in infos]
    if len(tokens) != expected["frames"] or len(tokens) != len(set(tokens)):
        raise ValueError(f"{split}: wrong frame count or duplicate token")

    camera_orders = set()
    scene_blocks = []
    for info in infos:
        token = info["token"]
        sample = sample_by_token.get(token)
        if sample is None:
            raise ValueError(f"{split}: info token missing from mini sample table: {token}")
        if sample["scene_token"] != info["scene_token"]:
            raise ValueError(f"{token}: scene token differs between info and mini table")
        if int(sample["timestamp"]) != int(info["timestamp"]):
            raise ValueError(f"{token}: timestamp differs between info and mini table")
        if sample["prev"] != info.get("prev", "") or sample["next"] != info.get("next", ""):
            raise ValueError(f"{token}: prev/next chain differs between info and mini table")
        order = tuple(info.get("cams", {}).keys())
        if len(order) != 6:
            raise ValueError(f"{token}: expected six serialized cameras")
        camera_orders.add(order)
        if not scene_blocks or scene_blocks[-1] != info["scene_token"]:
            scene_blocks.append(info["scene_token"])

    if len(camera_orders) != 1:
        raise ValueError(f"{split}: camera serialization order changes across frames")
    if len(scene_blocks) != expected["scenes"] or len(scene_blocks) != len(set(scene_blocks)):
        raise ValueError(f"{split}: timestamp order does not form complete contiguous scenes")
    actual_scene_names = {scene_name_by_token[token] for token in scene_blocks}
    if actual_scene_names != expected_scene_names:
        raise ValueError(f"{split}: wrong official scene membership")

    for scene_token in scene_blocks:
        frames = [item for item in infos if item["scene_token"] == scene_token]
        if frames[0].get("prev", "") or frames[-1].get("next", ""):
            raise ValueError(f"{scene_name_by_token[scene_token]}: truncated scene")
        for previous, current in zip(frames, frames[1:]):
            if (
                previous["next"] != current["token"]
                or current["prev"] != previous["token"]
                or int(current["timestamp"]) <= int(previous["timestamp"])
            ):
                raise ValueError(f"{scene_name_by_token[scene_token]}: broken temporal chain")

    return {
        "frames": len(infos),
        "scenes": len(scene_blocks),
        "scene_names_in_execution_order": [scene_name_by_token[token] for token in scene_blocks],
        "camera_order": list(next(iter(camera_orders))),
        "tokens": tokens,
        "token_order_sha256": sequence_sha256(tokens),
    }


def select_split(split, records, metadata, scene_name_by_token, sample_by_token):
    wanted = set(OFFICIAL_MINI_SPLITS[split])
    infos = sorted(
        [item for item in records if scene_name_by_token.get(item["scene_token"]) in wanted],
        key=lambda item: int(item["timestamp"]),
    )
    summary = validate_ordered_split(split, infos, scene_name_by_token, sample_by_token)
    mini_metadata = copy.deepcopy(metadata)
    mini_metadata["version"] = "v1.0-mini"
    return {"infos": infos, "metadata": mini_metadata}, summary


def validate_onnx_bundle(
    onnx_path, state_path, manifest_path, *,
    checkpoint_path=None, config_path=None,
):
    paths = [resolve(onnx_path), resolve(state_path), resolve(manifest_path)]
    present = [path.is_file() for path in paths]
    if any(present) and not all(present):
        raise ValueError("ONNX, initial-state NPZ and its JSON manifest must be all present or all absent")
    if not all(present):
        return None
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    actual = {
        "onnx_sha256": sha256(paths[0]),
        "state_sha256": sha256(paths[1]),
    }
    for key, value in actual.items():
        if manifest.get(key) != value:
            raise ValueError(f"{key} does not match initial-state manifest")
    if checkpoint_path is not None:
        expected = sha256(resolve(checkpoint_path))
        if manifest.get("checkpoint_sha256") != expected:
            raise ValueError(
                "ONNX bundle checkpoint hash does not match validation checkpoint"
            )
        actual["checkpoint_sha256"] = expected
    if config_path is not None:
        expected = sha256(resolve(config_path))
        if manifest.get("config_sha256") != expected:
            raise ValueError(
                "ONNX bundle config hash does not match validation config"
            )
        actual["config_sha256"] = expected
    if manifest.get("contract") != "stateful-v1":
        raise ValueError("unexpected ONNX initial-state contract")
    actual["contract"] = manifest["contract"]
    return actual


def prepare(args):
    run_dir = resolve(args.run_dir)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {run_dir}")

    data_root = resolve(args.data_root)
    train_path = resolve(args.train_info)
    val_path = resolve(args.val_info)
    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)
    anchor_path = resolve(args.motion_anchor)
    for path in (train_path, val_path, config_path, checkpoint_path, anchor_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    bundle_binding = validate_onnx_bundle(
        args.onnx,
        args.initial_state,
        args.initial_state_manifest,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    train = load_info_source(train_path)
    val = load_info_source(val_path)
    train_tokens = {item["token"] for item in train["infos"]}
    val_tokens = {item["token"] for item in val["infos"]}
    if not train_tokens.isdisjoint(val_tokens):
        raise ValueError("stock temporal train/val info sources overlap in sample tokens")

    scene_name_by_token, sample_by_token = load_mini_tables(data_root)
    records = list(train["infos"]) + list(val["infos"])
    split_records = {}
    all_tokens = set()
    for split in SPLITS:
        payload, summary = select_split(
            split, records, train["metadata"], scene_name_by_token, sample_by_token
        )
        info_path = run_dir / "inputs" / f"{split}.pkl"
        atomic_pickle(info_path, payload)
        summary["info_path"] = str(info_path)
        summary["info_sha256"] = sha256(info_path)
        split_records[split] = summary
        all_tokens.update(summary["tokens"])

    if all_tokens != set(sample_by_token) or len(all_tokens) != 404:
        raise ValueError("prepared mini_train + mini_val do not cover mini sample.json exactly")
    if not set(split_records["mini_train"]["tokens"]).isdisjoint(split_records["mini_val"]["tokens"]):
        raise ValueError("prepared mini_train and mini_val overlap")

    manifest = {
        "format": FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository_head": git_head(),
        "data_root": str(data_root),
        "assets": {
            "config": asset(config_path),
            "checkpoint": asset(checkpoint_path),
            "motion_anchor": asset(anchor_path),
            "train_info_source": asset(train_path),
            "val_info_source": asset(val_path),
            "mini_scene_table": asset(data_root / "v1.0-mini" / "scene.json"),
            "mini_sample_table": asset(data_root / "v1.0-mini" / "sample.json"),
            "onnx": asset(args.onnx, required=False),
            "initial_state": asset(args.initial_state, required=False),
            "initial_state_manifest": asset(args.initial_state_manifest, required=False),
        },
        "splits": split_records,
        "onnx_bundle_binding": bundle_binding,
        "coverage_contract": {
            "inference_frames": 404,
            "inference_scenes": 10,
            "sample_tokens_equal_v1.0-mini_sample_table": True,
            "metric_coverage": "use each native task's own valid-frame/filtering rule",
        },
        "execution_contract": {
            "device": "cpu_only",
            "native_world_size": 1,
            "stock_formal_split": "mini_val",
            "mini_train": "validation extension using the same evaluator formulas with explicit mini_train split",
            "stateful_ort_can_bus": "official_test_legacy",
            "stateful_ort_tracking_id_scope": "session",
            "planning": "collision optimization is mandatory host postprocess before final planning metrics",
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    print(json.dumps({
        "ok": True,
        "run_dir": str(run_dir),
        "repository_head": manifest["repository_head"],
        "mini_train": {k: split_records["mini_train"][k] for k in ("frames", "scenes", "info_sha256")},
        "mini_val": {k: split_records["mini_val"][k] for k in ("frames", "scenes", "info_sha256")},
        "total_frames": 404,
        "total_scenes": 10,
        "device": "cpu_only",
    }, indent=2))


def verify(args):
    run_dir = resolve(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported/corrupt equivalence manifest")
    checks = {"repository_head": manifest["repository_head"] == git_head()}
    for name, record in manifest["assets"].items():
        path = Path(record["path"])
        if record["exists"]:
            checks[f"asset:{name}"] = path.is_file() and sha256(path) == record["sha256"]
        else:
            checks[f"asset:{name}:still_absent"] = not path.exists()
    all_tokens = set()
    for split in SPLITS:
        record = manifest["splits"][split]
        path = Path(record["info_path"])
        payload = load_pickle(path)
        tokens = [item["token"] for item in payload["infos"]]
        checks[f"{split}:info_hash"] = path.is_file() and sha256(path) == record["info_sha256"]
        checks[f"{split}:version"] = payload["metadata"].get("version") == "v1.0-mini"
        checks[f"{split}:frame_count"] = len(tokens) == EXPECTED[split]["frames"]
        checks[f"{split}:token_order"] = tokens == record["tokens"] and sequence_sha256(tokens) == record["token_order_sha256"]
        all_tokens.update(tokens)
    checks["total_unique_tokens_404"] = len(all_tokens) == 404
    checks["cpu_only_contract"] = manifest.get("execution_contract", {}).get("device") == "cpu_only"
    bundle = manifest.get("onnx_bundle_binding")
    if bundle is None:
        checks["onnx_bundle_binding"] = not any(
            manifest["assets"][name]["exists"]
            for name in ("onnx", "initial_state", "initial_state_manifest")
        )
    else:
        try:
            rebound = validate_onnx_bundle(
                manifest["assets"]["onnx"]["path"],
                manifest["assets"]["initial_state"]["path"],
                manifest["assets"]["initial_state_manifest"]["path"],
                checkpoint_path=manifest["assets"]["checkpoint"]["path"],
                config_path=manifest["assets"]["config"]["path"],
            )
            checks["onnx_bundle_binding"] = rebound == bundle
        except Exception:
            checks["onnx_bundle_binding"] = False
    report = {"ok": all(checks.values()), "run_dir": str(run_dir), "checks": checks}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


def error_text(error):
    text = str(error).replace("\n", " ").strip()
    if len(text) > 600:
        text = text[:600] + "..."
    return f"{type(error).__name__}: {text}"


def install_tensorboard_import_stub():
    """Bypass TensorBoard-only imports without changing UniAD metric math."""
    if "torch.utils.tensorboard" in sys.modules:
        return False

    tensorboard_module = types.ModuleType("torch.utils.tensorboard")
    writer_module = types.ModuleType("torch.utils.tensorboard.writer")
    summary_module = types.ModuleType("torch.utils.tensorboard.summary")
    tensorboard_module.__path__ = []

    class DisabledSummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            if name.startswith("add_") or name in {"flush", "close"}:
                return lambda *args, **kwargs: None
            raise AttributeError(name)

        def flush(self):
            return None

        def close(self):
            return None

    def disabled_hparams(*args, **kwargs):
        return None, None, None

    tensorboard_module.SummaryWriter = DisabledSummaryWriter
    tensorboard_module.FileWriter = DisabledSummaryWriter
    tensorboard_module.writer = writer_module
    tensorboard_module.summary = summary_module
    writer_module.SummaryWriter = DisabledSummaryWriter
    writer_module.FileWriter = DisabledSummaryWriter
    summary_module.hparams = disabled_hparams
    sys.modules["torch.utils.tensorboard"] = tensorboard_module
    sys.modules["torch.utils.tensorboard.writer"] = writer_module
    sys.modules["torch.utils.tensorboard.summary"] = summary_module
    return True


def install_mmcv_msda_cuda_contract_cpu_shim():
    """Emulate MMCV 1.6.1 CUDA MSDA level indexing on the CPU fallback.

    MMCV's CUDA wrapper derives num_levels from spatial_shapes.size(0), while
    the Python fallback derives it from sampling_locations.shape[3]. UniAD's
    panseg head supplies one actual BEV level to an attention module configured
    for four levels. The CUDA kernel therefore reinterprets only the contiguous
    prefix corresponding to the actual level count. Reproduce that pointer
    contract exactly; do not renormalize attention weights.
    """
    import mmcv.ops.multi_scale_deform_attn as msda_module

    current = msda_module.multi_scale_deformable_attn_pytorch
    if getattr(current, "_uniad_cuda_contract_shim", False):
        return False

    original = current

    def cuda_contract_cpu(value, value_spatial_shapes, sampling_locations,
                          attention_weights):
        actual_levels = int(value_spatial_shapes.shape[0])
        configured_levels = int(sampling_locations.shape[3])
        if actual_levels == configured_levels:
            return original(
                value, value_spatial_shapes, sampling_locations, attention_weights
            )
        if actual_levels < 1 or actual_levels > configured_levels:
            raise ValueError(
                f"unsupported MSDA level contract: actual={actual_levels}, "
                f"configured={configured_levels}"
            )
        bs, num_queries, num_heads, _, num_points, two = sampling_locations.shape
        if two != 2 or tuple(attention_weights.shape[:3]) != (
            bs, num_queries, num_heads
        ):
            raise ValueError("unexpected MSDA sampling/weight shape")
        interpreted = bs * num_queries * num_heads * actual_levels * num_points
        locations = (
            sampling_locations.contiguous().reshape(-1)[: interpreted * 2]
            .reshape(bs, num_queries, num_heads, actual_levels, num_points, 2)
        )
        weights = (
            attention_weights.contiguous().reshape(-1)[:interpreted]
            .reshape(bs, num_queries, num_heads, actual_levels, num_points)
        )
        return original(value, value_spatial_shapes, locations, weights)

    cuda_contract_cpu._uniad_cuda_contract_shim = True
    cuda_contract_cpu._uniad_original = original
    msda_module.multi_scale_deformable_attn_pytorch = cuda_contract_cpu
    return True


def cpu_preflight(args):
    """Probe CPU-only native/ORT feasibility without running a UniAD frame."""
    run_dir = resolve(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported/corrupt equivalence manifest")
    if manifest.get("repository_head") != git_head():
        raise ValueError("cpu-preflight requires a manifest prepared from the current repository HEAD")
    if manifest.get("execution_contract", {}).get("device") != "cpu_only":
        raise ValueError("manifest is not bound to the cpu_only execution contract")

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    report = {
        "ok": False,
        "repository_head": git_head(),
        "run_dir": str(run_dir),
        "policy": {
            "device": "cpu_only",
            "cuda_visible_devices_before": previous_cuda_visible,
            "cuda_visible_devices_for_probe": "",
            "stock_tools_test_runner_allowed": False,
            "tensorboard_import_bypassed": True,
            "reason": "tools/test.py and custom_multi_gpu_test hard-code CUDA; validation uses a single-process CPU runner",
        },
        "environment": {},
        "checks": {"cuda_hidden_by_environment": os.environ.get("CUDA_VISIBLE_DEVICES") == ""},
        "stock_dcnv2_cpu": False,
        "requires_dcnv2_cpu_compat": True,
    }

    try:
        import torch
        import mmcv
        report["environment"]["torch"] = torch.__version__
        report["environment"]["torch_cuda_build"] = torch.version.cuda
        report["environment"]["mmcv"] = mmcv.__version__
    except Exception as error:
        report["checks"]["torch_mmcv_import"] = False
        report["environment"]["torch_mmcv_error"] = error_text(error)
        torch = None
    else:
        report["checks"]["torch_mmcv_import"] = True
        report["environment"]["tensorboard_stub_installed"] = install_tensorboard_import_stub()
        report["environment"]["msda_cuda_contract_cpu_shim_installed"] = (
            install_mmcv_msda_cuda_contract_cpu_shim()
        )

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        report["environment"]["onnxruntime"] = ort.__version__
        report["environment"]["ort_available_providers"] = providers
        report["checks"]["ort_cpu_provider"] = "CPUExecutionProvider" in providers
    except Exception as error:
        report["checks"]["ort_cpu_provider"] = False
        report["environment"]["onnxruntime_error"] = error_text(error)

    try:
        import casadi
        report["environment"]["casadi"] = getattr(casadi, "__version__", "unknown")
        report["checks"]["casadi_import"] = True
    except Exception as error:
        report["checks"]["casadi_import"] = False
        report["environment"]["casadi_error"] = error_text(error)

    if torch is not None:
        try:
            from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
            value = torch.arange(8, dtype=torch.float32).reshape(1, 4, 1, 2)
            spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)
            sampling_locations = torch.full((1, 1, 1, 1, 1, 2), 0.5, dtype=torch.float32)
            attention_weights = torch.ones((1, 1, 1, 1, 1), dtype=torch.float32)
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
            report["checks"]["msda_pytorch_cpu"] = (
                tuple(output.shape) == (1, 1, 2) and output.device.type == "cpu"
            )

            import mmcv.ops.multi_scale_deform_attn as msda_module
            patched = msda_module.multi_scale_deformable_attn_pytorch
            original = patched._uniad_original
            mismatch_locations = torch.linspace(
                0.1, 0.9, steps=1 * 2 * 2 * 4 * 4 * 2, dtype=torch.float32
            ).reshape(1, 2, 2, 4, 4, 2)
            mismatch_weights = torch.arange(
                1 * 2 * 2 * 4 * 4, dtype=torch.float32
            ).reshape(1, 2, 2, 16).softmax(-1).reshape(1, 2, 2, 4, 4)
            mismatch_value = torch.arange(
                1 * 4 * 2 * 2, dtype=torch.float32
            ).reshape(1, 4, 2, 2)
            actual = patched(
                mismatch_value, spatial_shapes,
                mismatch_locations, mismatch_weights
            )
            interpreted_locations = mismatch_locations.contiguous().reshape(-1)[:32]
            interpreted_locations = interpreted_locations.reshape(1, 2, 2, 1, 4, 2)
            interpreted_weights = mismatch_weights.contiguous().reshape(-1)[:16]
            interpreted_weights = interpreted_weights.reshape(1, 2, 2, 1, 4)
            expected = original(
                mismatch_value, spatial_shapes,
                interpreted_locations, interpreted_weights
            )
            report["checks"]["msda_cuda_contract_cpu_compat"] = bool(
                torch.equal(actual, expected)
            )
        except Exception as error:
            report["checks"]["msda_pytorch_cpu"] = False
            report["checks"]["msda_cuda_contract_cpu_compat"] = False
            report["environment"]["msda_error"] = error_text(error)

        try:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import rotate
            image = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
            rotated = rotate(
                image,
                angle=3.0,
                interpolation=InterpolationMode.NEAREST,
                center=[4, 4],
            )
            report["checks"]["torchvision_rotate_cpu"] = (
                tuple(rotated.shape) == (1, 8, 8) and rotated.device.type == "cpu"
            )
        except Exception as error:
            report["checks"]["torchvision_rotate_cpu"] = False
            report["environment"]["torchvision_rotate_error"] = error_text(error)

        try:
            from mmcv.ops import ModulatedDeformConv2dPack
            from dcnv2 import ExportableModulatedDeformConv2dPack

            module = ModulatedDeformConv2dPack(
                4, 4, kernel_size=3, stride=1, padding=1, deform_groups=1
            ).eval().cpu()
            probe = torch.linspace(
                -1.0, 1.0, steps=1 * 4 * 8 * 8, dtype=torch.float32
            ).reshape(1, 4, 8, 8)
            with torch.no_grad():
                module.weight.copy_(
                    torch.linspace(
                        -0.15, 0.15, steps=module.weight.numel(),
                        dtype=torch.float32,
                    ).reshape_as(module.weight)
                )
                if module.bias is not None:
                    module.bias.copy_(
                        torch.linspace(
                            -0.03, 0.03, steps=module.bias.numel(),
                            dtype=torch.float32,
                        )
                    )
                module.conv_offset.weight.copy_(
                    torch.linspace(
                        -0.02, 0.02, steps=module.conv_offset.weight.numel(),
                        dtype=torch.float32,
                    ).reshape_as(module.conv_offset.weight)
                )
                module.conv_offset.bias.copy_(
                    torch.linspace(
                        -0.25, 0.25, steps=module.conv_offset.bias.numel(),
                        dtype=torch.float32,
                    )
                )
                dcn_output = module(probe)
                exportable = ExportableModulatedDeformConv2dPack(module).eval().cpu()
                exportable_output = exportable(probe)

            report["stock_dcnv2_cpu"] = (
                tuple(dcn_output.shape) == (1, 4, 8, 8)
                and dcn_output.device.type == "cpu"
                and bool(torch.isfinite(dcn_output).all())
            )
            dcn_diff = (dcn_output - exportable_output).abs()
            report["environment"]["dcnv2_exportable_max_abs"] = float(
                dcn_diff.max().item()
            )
            report["environment"]["dcnv2_exportable_mean_abs"] = float(
                dcn_diff.mean().item()
            )
            report["checks"]["dcnv2_exportable_matches_stock"] = bool(
                torch.allclose(
                    dcn_output, exportable_output, atol=1e-4, rtol=1e-4
                )
            )
        except Exception as error:
            report["dcnv2_error"] = error_text(error)
            report["checks"]["dcnv2_exportable_matches_stock"] = False
        report["requires_dcnv2_cpu_compat"] = not report["stock_dcnv2_cpu"]

        try:
            from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
                IntersectionOverUnion,
                PanopticMetric,
            )
            from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric
            metrics = [
                IntersectionOverUnion(2).cpu(),
                PanopticMetric(n_classes=2, temporally_consistent=True).cpu(),
                PlanningMetric().cpu(),
            ]
            cpu_only = True
            for metric in metrics:
                tensors = list(metric.parameters()) + list(metric.buffers())
                cpu_only = cpu_only and all(tensor.device.type == "cpu" for tensor in tensors)
            report["checks"]["native_metrics_cpu_constructible"] = cpu_only
        except Exception as error:
            report["checks"]["native_metrics_cpu_constructible"] = False
            report["environment"]["native_metrics_error"] = error_text(error)

        try:
            import projects.mmdet3d_plugin.datasets.pipelines  # noqa: F401
            from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset  # noqa: F401
            from mmcv import Config
            from mmdet3d.datasets import build_dataset
            config = Config.fromfile(manifest["assets"]["config"]["path"])
            test_cfg = copy.deepcopy(config.data.test)
            test_cfg.ann_file = str(run_dir / "inputs" / "mini_val.pkl")
            test_cfg.data_root = manifest["data_root"]
            test_cfg.test_mode = True
            test_cfg.file_client_args = dict(backend="disk")
            dataset = build_dataset(test_cfg)
            first = dataset[0]
            image_container = first.get("img")
            image_tensor = image_container.data if hasattr(image_container, "data") else image_container
            report["checks"]["mini_val_dataset_cpu_pipeline"] = (
                len(dataset) == EXPECTED["mini_val"]["frames"]
                and image_tensor is not None
                and getattr(image_tensor, "device", torch.device("cpu")).type == "cpu"
            )
            report["environment"]["mini_val_dataset_len"] = len(dataset)
            report["environment"]["first_sample_token"] = dataset.data_infos[0]["token"]
        except Exception as error:
            report["checks"]["mini_val_dataset_cpu_pipeline"] = False
            report["environment"]["mini_val_dataset_error"] = error_text(error)
    else:
        report["checks"]["msda_pytorch_cpu"] = False
        report["checks"]["torchvision_rotate_cpu"] = False
        report["checks"]["native_metrics_cpu_constructible"] = False
        report["checks"]["mini_val_dataset_cpu_pipeline"] = False

    mandatory = (
        "cuda_hidden_by_environment",
        "torch_mmcv_import",
        "ort_cpu_provider",
        "casadi_import",
        "msda_pytorch_cpu",
        "msda_cuda_contract_cpu_compat",
        "torchvision_rotate_cpu",
        "dcnv2_exportable_matches_stock",
        "native_metrics_cpu_constructible",
        "mini_val_dataset_cpu_pipeline",
    )
    report["ok"] = all(report["checks"].get(name, False) for name in mandatory)
    report["native_stock_cpu_ready"] = report["ok"] and report["stock_dcnv2_cpu"]
    if not report["ok"]:
        report["next_step"] = "resolve failed mandatory CPU preflight checks before native PT inference"
    elif report["requires_dcnv2_cpu_compat"]:
        report["next_step"] = (
            "validate and use one explicit DCNv2-only CPU compatibility shim before native PT inference; "
            "never enable the incompatible GPU and never label that run untouched stock PT"
        )
    else:
        report["next_step"] = "stock native operators are CPU-runnable; proceed to single-process native PT runner"

    output = resolve(args.json_out) if args.json_out else run_dir / "cpu_preflight.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)




def temporal_component_audit(args):
    """Compare native and tensorized temporal primitives without model inference."""
    import types

    run_dir = resolve(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported/corrupt equivalence manifest")
    if manifest.get("repository_head") != git_head():
        raise ValueError(
            "temporal-component-audit requires a manifest prepared from current HEAD"
        )

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from mmcv import Config
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import rotate

    install_tensorboard_import_stub()
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin import (
        Instances, MemoryBank, QueryInteractionModule, RuntimeTrackerBase,
    )
    from projects.mmdet3d_plugin.uniad.detectors.uniad_track import UniADTrack
    from frame_core import StatefulNearestRotation
    from temporal import (
        TensorMemoryBank, TensorQueryInteraction, TensorRuntimeTracker,
        TensorVelocityUpdate,
    )
    from prepare_scene import frame_metadata

    torch.manual_seed(0)
    config = Config.fromfile(manifest["assets"]["config"]["path"])
    infos = load_pickle(manifest["splits"]["mini_val"]["info_path"])["infos"]
    if not infos:
        raise ValueError("mini_val contains no frames")
    scene_token = infos[0]["scene_token"]
    scene_infos = [info for info in infos if info["scene_token"] == scene_token]
    if len(scene_infos) < 2:
        raise ValueError("first mini_val scene is not a complete temporal sequence")
    if scene_infos[0].get("prev", "") or scene_infos[-1].get("next", ""):
        raise ValueError("first mini_val scene is truncated")

    tolerance = {"atol": 1e-5, "rtol": 1e-5}
    report = {
        "format": "uniad-temporal-component-audit-v1",
        "diagnostic_only": True,
        "repository_head": git_head(),
        "run_dir": str(run_dir),
        "scene_name": manifest["splits"]["mini_val"][
            "scene_names_in_execution_order"
        ][0],
        "scene_frames": len(scene_infos),
        "tolerance": tolerance,
        "cuda_visible_devices_before": previous_cuda_visible,
        "cuda_visible_devices_for_audit": "",
        "checks": {},
        "details": {},
    }

    # 1) Exact nearest-neighbour source-pixel mapping on every real temporal
    # rotation angle in the first mini_val scene.
    absolute_angles = [
        float(frame_metadata(info)["can_bus_absolute"][-1])
        for info in scene_infos
    ]
    rotation = StatefulNearestRotation().eval()
    source = torch.arange(
        1, 200 * 200 + 1, dtype=torch.float32
    ).reshape(1, 200, 200)
    rotation_rows = []
    for frame_index in range(1, len(scene_infos)):
        angle = np.float32(absolute_angles[frame_index] - absolute_angles[frame_index - 1])
        native = rotate(
            source,
            angle=float(angle),
            interpolation=InterpolationMode.NEAREST,
            center=[100, 100],
        )
        candidate = rotation(source, torch.tensor(angle, dtype=torch.float32))
        mismatch = int(torch.count_nonzero(native != candidate).item())
        rotation_rows.append({
            "frame_index": frame_index,
            "token": scene_infos[frame_index]["token"],
            "angle_degrees": float(angle),
            "mismatch_pixels": mismatch,
        })
    report["details"]["bev_rotation"] = {
        "frames_compared": len(rotation_rows),
        "mismatch_frames": sum(row["mismatch_pixels"] > 0 for row in rotation_rows),
        "mismatch_pixels_total": sum(row["mismatch_pixels"] for row in rotation_rows),
        "first_mismatch": next(
            (row for row in rotation_rows if row["mismatch_pixels"] > 0), None
        ),
        "per_frame": rotation_rows,
    }
    report["checks"]["bev_rotation_exact_source_pixels"] = all(
        row["mismatch_pixels"] == 0 for row in rotation_rows
    )

    # 2) Native torch.linalg.inv velocity/ref-point update versus the
    # exportable explicit 3x3 inverse on the real pose sequence.
    pc_range = list(config.model.pc_range)
    velocity_adapter = TensorVelocityUpdate(pc_range).eval()
    native_owner = types.SimpleNamespace(pc_range=pc_range)
    refs = torch.linspace(-1.5, 1.5, steps=15, dtype=torch.float32).reshape(5, 3)
    velocities = torch.linspace(-4.0, 4.0, steps=10, dtype=torch.float32).reshape(5, 2)
    velocity_rows = []
    for frame_index in range(1, len(scene_infos)):
        previous = frame_metadata(scene_infos[frame_index - 1])
        current = frame_metadata(scene_infos[frame_index])
        delta = np.float32(
            (float(scene_infos[frame_index]["timestamp"])
             - float(scene_infos[frame_index - 1]["timestamp"])) / 1e6
        )
        previous_r = torch.from_numpy(previous["l2g_r"])
        previous_t = torch.from_numpy(previous["l2g_t"])
        current_r = torch.from_numpy(current["l2g_r"])
        current_t = torch.from_numpy(current["l2g_t"])
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        native = UniADTrack.velo_update(
            native_owner,
            refs.clone(),
            velocities.clone(),
            previous_r,
            previous_t,
            current_r,
            current_t,
            delta_tensor,
        )
        candidate = velocity_adapter(
            refs.clone(),
            velocities.clone(),
            previous_r,
            previous_t,
            current_r,
            current_t,
            delta_tensor,
        )
        diff = (native - candidate).abs()
        velocity_rows.append({
            "frame_index": frame_index,
            "token": scene_infos[frame_index]["token"],
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "allclose": bool(torch.allclose(native, candidate, **tolerance)),
        })
    report["details"]["velocity_update"] = {
        "frames_compared": len(velocity_rows),
        "max_abs": max(row["max_abs"] for row in velocity_rows),
        "mean_abs_mean_over_frames": float(
            np.mean([row["mean_abs"] for row in velocity_rows])
        ),
        "first_not_allclose": next(
            (row for row in velocity_rows if not row["allclose"]), None
        ),
        "per_frame": velocity_rows,
    }
    report["checks"]["velocity_update_allclose"] = all(
        row["allclose"] for row in velocity_rows
    )

    # 3) Runtime tracker: exact threshold, disappearance and sequential-ID
    # semantics, including values immediately around 0.35/0.40.
    score_thresh = float(config.model.score_thresh)
    filter_thresh = float(config.model.filter_score_thresh)
    miss_tolerance = int(config.model.get("miss_tolerance", 5))
    scores = torch.tensor([
        score_thresh - 1e-4,
        score_thresh,
        score_thresh + 1e-4,
        filter_thresh - 1e-4,
        filter_thresh,
        filter_thresh + 1e-4,
        0.1,
        0.9,
    ], dtype=torch.float32)
    ids = torch.tensor([-1, -1, -1, 5, 6, 7, 8, 9], dtype=torch.int64)
    counts = torch.tensor(
        [0, 0, 0, miss_tolerance - 1, 2, 3, miss_tolerance - 1, 4],
        dtype=torch.int64,
    )
    initial_next_id = 10

    native_tracker = RuntimeTrackerBase(
        score_thresh=score_thresh,
        filter_score_thresh=filter_thresh,
        miss_tolerance=miss_tolerance,
    )
    native_tracker.max_obj_id = initial_next_id
    native_tracks = Instances((1, 1))
    native_tracks.scores = scores.clone()
    native_tracks.obj_idxes = ids.clone()
    native_tracks.disappear_time = counts.clone()
    native_tracker.update(native_tracks, None)
    native_visible = torch.logical_and(
        native_tracks.obj_idxes >= 0,
        native_tracks.scores >= filter_thresh,
    )

    tensor_tracker = TensorRuntimeTracker(
        score_thresh=score_thresh,
        filter_score_thresh=filter_thresh,
        miss_tolerance=miss_tolerance,
    )
    tensor_ids, tensor_counts, tensor_next_id, tensor_visible = tensor_tracker(
        scores.clone(), ids.clone(), counts.clone(),
        torch.tensor(initial_next_id, dtype=torch.int64),
    )
    tracker_exact = (
        torch.equal(native_tracks.obj_idxes, tensor_ids)
        and torch.equal(native_tracks.disappear_time, tensor_counts)
        and native_tracker.max_obj_id == int(tensor_next_id.item())
        and torch.equal(native_visible, tensor_visible)
    )
    report["details"]["runtime_tracker"] = {
        "native_ids": native_tracks.obj_idxes.tolist(),
        "tensor_ids": tensor_ids.tolist(),
        "native_disappear_time": native_tracks.disappear_time.tolist(),
        "tensor_disappear_time": tensor_counts.tolist(),
        "native_next_id": int(native_tracker.max_obj_id),
        "tensor_next_id": int(tensor_next_id.item()),
        "native_visible": native_visible.tolist(),
        "tensor_visible": tensor_visible.tolist(),
    }
    report["checks"]["runtime_tracker_exact"] = bool(tracker_exact)

    # 4) MemoryBank: same module parameters, synthetic mixed-history state.
    mem_args = dict(config.model.mem_args)
    memory_module = MemoryBank(mem_args, 256, 256, 256).eval()
    tensor_memory = TensorMemoryBank(memory_module).eval()
    n = 6
    memory_tracks = Instances((1, 1))
    memory_tracks.output_embedding = torch.randn(n, 256) * 0.1
    memory_tracks.scores = torch.tensor(
        [0.7, 0.5, 0.2, 0.8, 0.4, 0.6], dtype=torch.float32
    )
    memory_tracks.mem_bank = torch.randn(n, 4, 256) * 0.05
    memory_tracks.mem_padding_mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ], dtype=torch.bool)
    memory_tracks.save_period = torch.tensor(
        [0, 0, 1, 2, 0, 3], dtype=torch.float32
    )
    native_memory_tracks = copy.deepcopy(memory_tracks)
    with torch.no_grad():
        native_memory_tracks = memory_module(native_memory_tracks)
        tensor_embedding, tensor_bank, tensor_mask, tensor_period = tensor_memory(
            memory_tracks.output_embedding.clone(),
            memory_tracks.scores.clone(),
            memory_tracks.mem_bank.clone(),
            memory_tracks.mem_padding_mask.clone(),
            memory_tracks.save_period.clone(),
        )
    memory_fields = {
        "output_embedding": (
            native_memory_tracks.output_embedding, tensor_embedding, False
        ),
        "mem_bank": (native_memory_tracks.mem_bank, tensor_bank, False),
        "mem_padding_mask": (
            native_memory_tracks.mem_padding_mask, tensor_mask, True
        ),
        "save_period": (
            native_memory_tracks.save_period, tensor_period, False
        ),
    }
    memory_detail = {}
    memory_ok = True
    for name, (native_value, tensor_value, exact) in memory_fields.items():
        if exact:
            same = bool(torch.equal(native_value, tensor_value))
            memory_detail[name] = {"exact": same}
        else:
            diff = (native_value - tensor_value).abs()
            same = bool(torch.allclose(native_value, tensor_value, **tolerance))
            memory_detail[name] = {
                "allclose": same,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
            }
        memory_ok = memory_ok and same
    report["details"]["memory_bank"] = memory_detail
    report["checks"]["memory_bank_allclose"] = bool(memory_ok)

    # 5) QIM: compare active selection and updated query values. Eval mode
    # disables training-only random drop / false-positive injection.
    qim_args = dict(config.model.qim_args)
    qim_module = QueryInteractionModule(qim_args, 256, 256, 256).eval()
    tensor_qim = TensorQueryInteraction(qim_module).eval()
    qim_tracks = Instances((1, 1))
    qim_tracks.query = torch.randn(7, 512) * 0.1
    qim_tracks.output_embedding = torch.randn(7, 256) * 0.1
    qim_tracks.obj_idxes = torch.tensor(
        [-1, 5, 6, -1, 7, -2, 8], dtype=torch.int64
    )
    native_qim_tracks = copy.deepcopy(qim_tracks)
    with torch.no_grad():
        native_active = qim_module._select_active_tracks({
            "track_instances": native_qim_tracks
        })
        native_active = qim_module._update_track_embedding(native_active)
        tensor_query, tensor_active_index = tensor_qim(
            qim_tracks.query.clone(),
            qim_tracks.output_embedding.clone(),
            qim_tracks.obj_idxes.clone(),
        )
    expected_active_index = torch.nonzero(
        qim_tracks.obj_idxes >= 0
    ).squeeze(1)
    qim_diff = (native_active.query - tensor_query).abs()
    qim_index_exact = bool(
        torch.equal(expected_active_index, tensor_active_index)
    )
    qim_query_allclose = bool(
        torch.allclose(native_active.query, tensor_query, **tolerance)
    )
    report["details"]["qim"] = {
        "active_index_exact": qim_index_exact,
        "native_active_count": len(native_active),
        "tensor_active_count": int(tensor_active_index.numel()),
        "query_allclose": qim_query_allclose,
        "query_max_abs": float(qim_diff.max().item()),
        "query_mean_abs": float(qim_diff.mean().item()),
    }
    report["checks"]["qim_allclose"] = (
        qim_index_exact and qim_query_allclose
    )

    report["all_checks_pass"] = all(report["checks"].values())
    report["interpretation"] = (
        "component diagnostics only; a pass does not prove full temporal-model "
        "equivalence and a failure must be localized before changing model semantics"
    )
    output = (
        resolve(args.output)
        if args.output
        else run_dir / "temporal_component_audit.json"
    )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite temporal component audit: {output}"
        )
    atomic_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_execution_manifest(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported/corrupt equivalence manifest")
    if manifest.get("repository_head") != git_head():
        raise ValueError("run manifest was prepared from a different repository HEAD")
    if manifest.get("execution_contract", {}).get("device") != "cpu_only":
        raise ValueError("run manifest is not bound to cpu_only execution")
    for record in manifest["assets"].values():
        path = Path(record["path"])
        if record["exists"] and (not path.is_file() or sha256(path) != record["sha256"]):
            raise ValueError(f"asset changed since prepare: {path}")
    return manifest


def require_native_cpu_preflight(run_dir):
    path = run_dir / "cpu_preflight.json"
    if not path.is_file():
        raise FileNotFoundError(f"run cpu-preflight first: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("repository_head") != git_head():
        raise ValueError("cpu_preflight.json belongs to a different repository HEAD")
    if not report.get("ok") or not report.get("native_stock_cpu_ready"):
        raise ValueError("current run has not passed stock native CPU preflight")
    if report.get("requires_dcnv2_cpu_compat"):
        raise ValueError("native-run refuses a DCNv2 compatibility path")
    return report


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value

def finite_jsonable(value):
    value = jsonable(value)
    if isinstance(value, dict):
        return {str(key): finite_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value



EVALUATOR_RECORD_SCHEMA_V1 = "uniad-evaluator-facing-v1"
EVALUATOR_RECORD_SCHEMA = "uniad-evaluator-facing-v2"
EVALUATOR_RECORD_SCHEMAS = {
    EVALUATOR_RECORD_SCHEMA_V1,
    EVALUATOR_RECORD_SCHEMA,
}


def _numpy_copy(value, name):
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value.copy()


def _coder_contract(coder):
    post = coder.post_center_range
    if hasattr(post, "detach"):
        post = post.detach().cpu().tolist()
    else:
        post = list(post)
    return {
        "class": type(coder).__name__,
        "post_center_range": [float(value) for value in post],
        "pc_range": [float(value) for value in coder.pc_range],
        "max_num": int(coder.max_num),
        "score_threshold": None if coder.score_threshold is None else float(coder.score_threshold),
        "num_classes": int(coder.num_classes),
        "with_nms": bool(coder.with_nms),
        "nms_iou_thres": float(coder.nms_iou_thres),
    }


def _make_evaluator_bbox_coder(reference=None, pc_range=None):
    from projects.mmdet3d_plugin.core.bbox.coders.detr3d_track_coder import (
        DETRTrack3DCoder,
    )
    if reference is not None:
        contract = _coder_contract(reference)
        coder = DETRTrack3DCoder(
            pc_range=contract["pc_range"],
            post_center_range=contract["post_center_range"],
            max_num=contract["max_num"],
            score_threshold=contract["score_threshold"],
            num_classes=contract["num_classes"],
            with_nms=contract["with_nms"],
            iou_thres=contract["nms_iou_thres"],
        )
        if _coder_contract(reference) != _coder_contract(coder):
            raise ValueError(
                "validation detection decoder contract differs from model.bbox_coder: "
                f"model={_coder_contract(reference)} validation={_coder_contract(coder)}"
            )
        return coder

    from projects.mmdet3d_plugin.uniad.detectors.uniad_track import UniADTrack
    default_cfg = copy.deepcopy(
        inspect.signature(UniADTrack.__init__).parameters["bbox_coder"].default
    )
    if pc_range is not None and list(default_cfg["pc_range"]) != list(pc_range):
        raise ValueError(
            "ORT validation config pc_range differs from UniADTrack default "
            f"bbox coder: config={list(pc_range)} coder={list(default_cfg['pc_range'])}"
        )
    return DETRTrack3DCoder(
        pc_range=default_cfg["pc_range"],
        post_center_range=default_cfg["post_center_range"],
        max_num=default_cfg["max_num"],
        score_threshold=default_cfg["score_threshold"],
        num_classes=default_cfg["num_classes"],
        with_nms=default_cfg["with_nms"],
        iou_thres=default_cfg["iou_thres"],
    )


def _decode_detection_for_evaluator(outputs, coder):
    import torch
    cls = torch.from_numpy(
        np.ascontiguousarray(np.asarray(outputs["cls_scores"])[-1, 0])
    )
    boxes = torch.from_numpy(
        np.ascontiguousarray(np.asarray(outputs["bbox_preds"])[-1, 0])
    )
    if cls.ndim != 2 or boxes.ndim != 2 or cls.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"invalid detection tensors: cls={tuple(cls.shape)} boxes={tuple(boxes.shape)}"
        )
    scores = cls.sigmoid().max(dim=-1).values
    obj_idxes = torch.full(
        (cls.shape[0],), -1, dtype=torch.int64, device=cls.device
    )
    decoded = coder.decode_single(
        cls, boxes, scores, obj_idxes, with_mask=True, img_metas=None
    )
    return {
        "boxes": _numpy_copy(decoded["bboxes"], "detection.boxes"),
        "scores": _numpy_copy(decoded["scores"], "detection.scores"),
        "labels": _numpy_copy(decoded["labels"], "detection.labels"),
    }


def _native_evaluator_record(row, info, occ_valid):
    required = (
        "boxes_3d", "scores_3d", "labels_3d", "track_ids",
        "boxes_3d_det", "scores_3d_det", "labels_3d_det",
        "traj", "traj_scores", "ret_iou",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"native evaluator row missing fields: {missing}")

    tracking_boxes = _numpy_copy(row["boxes_3d"], "tracking.boxes")
    native_traj = _numpy_copy(row["traj"], "motion.traj_with_sdc")
    native_traj_scores = _numpy_copy(
        row["traj_scores"], "motion.traj_scores_with_sdc"
    )
    track_count = tracking_boxes.shape[0]
    if native_traj.shape[0] != track_count + 1:
        raise ValueError(
            "native motion contract expected tracking rows + final SDC row: "
            f"tracking={track_count}, traj={native_traj.shape[0]}"
        )
    if native_traj_scores.shape[0] != track_count + 1:
        raise ValueError(
            "native motion-score contract expected tracking rows + final SDC row: "
            f"tracking={track_count}, scores={native_traj_scores.shape[0]}"
        )
    # NuScenesE2EDataset._format_bbox indexes trajectories only by tracking-box
    # keep_idx and consumes [..., :2]. The final SDC row is never serialized.
    evaluator_traj = native_traj[:track_count, ..., :2].copy()
    evaluator_traj_scores = native_traj_scores[:track_count].copy()

    record = {
        "schema": EVALUATOR_RECORD_SCHEMA,
        "token": info["token"],
        "scene_token": info["scene_token"],
        "timestamp": int(info["timestamp"]),
        "detection": {
            "boxes": _numpy_copy(row["boxes_3d_det"], "detection.boxes"),
            "scores": _numpy_copy(row["scores_3d_det"], "detection.scores"),
            "labels": _numpy_copy(row["labels_3d_det"], "detection.labels"),
        },
        "tracking": {
            "boxes": tracking_boxes,
            "scores": _numpy_copy(row["scores_3d"], "tracking.scores"),
            "labels": _numpy_copy(row["labels_3d"], "tracking.labels"),
            "ids": _numpy_copy(row["track_ids"], "tracking.ids"),
        },
        "motion": {
            "traj": evaluator_traj,
            "traj_scores": evaluator_traj_scores,
        },
        "map": {"ret_iou": jsonable(row["ret_iou"])},
        "occupancy": {
            "valid": bool(occ_valid),
            "seg_out": None,
            "ins_seg_out": None,
        },
        "planning": {
            "optimized": _numpy_copy(row["planning_traj"], "planning.optimized"),
        },
    }
    if "planning_raw_traj" in row:
        record["planning"]["raw"] = _numpy_copy(
            row["planning_raw_traj"], "planning.raw"
        )
    if "occ" in row:
        record["occupancy"]["seg_out"] = _numpy_copy(
            row["occ"]["seg_out"], "occupancy.seg_out"
        )
        record["occupancy"]["ins_seg_out"] = _numpy_copy(
            row["occ"]["ins_seg_out"], "occupancy.ins_seg_out"
        )
    if record["motion"]["traj"].shape[0] != track_count:
        raise ValueError("normalized native motion rows do not align with tracking rows")
    return record


def _stateful_evaluator_record(
    outputs, info, detection, motion, map_counts, occ_instances,
    planning_optimized, collision_info, occ_valid,
):
    record = {
        "schema": EVALUATOR_RECORD_SCHEMA,
        "token": info["token"],
        "scene_token": info["scene_token"],
        "timestamp": int(info["timestamp"]),
        "detection": detection,
        "tracking": {
            "boxes": _numpy_copy(outputs["track_boxes"], "tracking.boxes"),
            "scores": _numpy_copy(outputs["track_scores"], "tracking.scores"),
            "labels": _numpy_copy(outputs["track_labels"], "tracking.labels"),
            "ids": _numpy_copy(outputs["track_ids"], "tracking.ids"),
        },
        "motion": {
            "traj": _numpy_copy(motion["traj"], "motion.traj"),
            "traj_scores": _numpy_copy(motion["traj_scores"], "motion.traj_scores"),
        },
        "map": {"ret_iou": jsonable(map_counts)},
        "occupancy": {
            "valid": bool(occ_valid),
            "seg_out": _numpy_copy(outputs["occ_segmentation"], "occupancy.seg_out"),
            "ins_seg_out": _numpy_copy(occ_instances, "occupancy.ins_seg_out"),
        },
        "planning": {
            "raw": _numpy_copy(outputs["planning_raw"], "planning.raw"),
            "optimized": _numpy_copy(planning_optimized, "planning.optimized"),
            "collision_solver_ran": bool(collision_info["solver_ran"]),
            "collision_selected_cells": int(collision_info["selected_cells"]),
        },
    }
    track_count = record["tracking"]["boxes"].shape[0]
    if record["motion"]["traj"].shape[0] != track_count:
        raise ValueError("stateful motion rows do not align with tracking rows")
    return record


def native_run(args):
    """Run stock native UniAD operators in one CPU process.

    Default scope is the first complete scene of the selected split. Full-split
    execution requires --all-scenes explicitly.
    """
    run_dir = resolve(args.run_dir)
    manifest = load_execution_manifest(run_dir)
    preflight = require_native_cpu_preflight(run_dir)
    split = args.split
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if args.all_scenes and args.scene_name:
        raise ValueError("--all-scenes and --scene-name are mutually exclusive")

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import mmcv
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint

    install_tensorboard_import_stub()
    install_mmcv_msda_cuda_contract_cpu_shim()
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
        IntersectionOverUnion,
        PanopticMetric,
    )
    from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import PlanningMetric

    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be >= 1")
        torch.set_num_threads(args.threads)

    config = Config.fromfile(manifest["assets"]["config"]["path"])
    test_cfg = copy.deepcopy(config.data.test)
    test_cfg.ann_file = manifest["splits"][split]["info_path"]
    test_cfg.data_root = manifest["data_root"]
    test_cfg.test_mode = True
    test_cfg.file_client_args = dict(backend="disk")
    dataset = build_dataset(test_cfg)

    dataset_tokens = [item["token"] for item in dataset.data_infos]
    split_record = manifest["splits"][split]
    if dataset_tokens != split_record["tokens"]:
        raise ValueError("native dataset token order differs from immutable manifest")

    scene_name_by_token, _ = load_mini_tables(Path(manifest["data_root"]))
    scene_names = [
        scene_name_by_token[item["scene_token"]]
        for item in dataset.data_infos
    ]
    execution_order = split_record["scene_names_in_execution_order"]
    if args.all_scenes:
        selected_scene_names = list(execution_order)
        scope_name = "full"
    else:
        scene_name = args.scene_name or execution_order[0]
        if scene_name not in execution_order:
            raise ValueError(f"{scene_name} is not in {split}")
        selected_scene_names = [scene_name]
        scope_name = scene_name

    indices = [
        index for index, scene_name in enumerate(scene_names)
        if scene_name in set(selected_scene_names)
    ]
    if not indices:
        raise ValueError("native-run selected zero frames")
    selected_tokens = [dataset_tokens[index] for index in indices]
    if not args.all_scenes:
        chosen = selected_scene_names[0]
        chosen_infos = [dataset.data_infos[index] for index in indices]
        if chosen_infos[0].get("prev", "") or chosen_infos[-1].get("next", ""):
            raise ValueError(f"{chosen}: smoke scope is not a complete scene")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"{chosen}: scene is not contiguous in dataset order")

    output_dir = run_dir / "native" / split / scope_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite native evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    checkpoint = load_checkpoint(
        model, manifest["assets"]["checkpoint"]["path"], map_location="cpu"
    )
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE
    model.cpu()
    model.eval()
    _make_evaluator_bbox_coder(reference=model.bbox_coder)

    eval_occ = bool(getattr(model, "with_occ_head", False))
    eval_planning = bool(getattr(model, "with_planning_head", False))
    ranges = {"30x30": (70, 130), "100x100": (0, 200)}
    iou_metrics = {
        key: IntersectionOverUnion(2).cpu() for key in ranges
    } if eval_occ else {}
    panoptic_metrics = {
        key: PanopticMetric(n_classes=2, temporally_consistent=True).cpu()
        for key in ranges
    } if eval_occ else {}
    planning_metric = PlanningMetric().cpu() if eval_planning else None

    # Native PlanningHead overwrites the pre-collision trajectory when
    # use_col_optim=True. Capture the optimizer input without changing the
    # model's return value so future evidence can compare both sides of the
    # export boundary: neural planning_raw and host/Native optimized planning.
    native_planning_capture = {}
    if eval_planning:
        original_collision_optimization = (
            model.planning_head.collision_optimization
        )

        def capture_native_planning_raw(self, sdc_traj_all, occ_mask):
            if "raw" in native_planning_capture:
                raise RuntimeError(
                    "native planning collision optimizer invoked more than once "
                    "for one frame"
                )
            native_planning_capture["raw"] = sdc_traj_all.detach().clone()
            return original_collision_optimization(sdc_traj_all, occ_mask)

        model.planning_head.collision_optimization = types.MethodType(
            capture_native_planning_raw, model.planning_head
        )

    bbox_results = []
    evaluator_records = []
    num_occ = 0
    started = time.time()
    progress = mmcv.ProgressBar(len(indices))

    for index in indices:
        sample = dataset[index]
        data = collate([sample], samples_per_gpu=1)
        data = scatter(data, [-1])[0]
        if eval_planning:
            native_planning_capture.clear()
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if not isinstance(result, list) or len(result) != 1:
            raise ValueError(f"native UniAD expected one result row, got {type(result)}")
        row = result[0]

        if eval_planning:
            planning = row["planning"]
            segmentation = planning["planning_gt"]["segmentation"]
            sdc_planning = planning["planning_gt"]["sdc_planning"]
            sdc_planning_mask = planning["planning_gt"]["sdc_planning_mask"]
            pred_sdc_traj = planning["result_planning"]["sdc_traj"]
            if bool(getattr(model.planning_head, "use_col_optim", False)):
                if "raw" not in native_planning_capture:
                    raise RuntimeError(
                        "native planning raw trajectory was not captured"
                    )
                row["planning_raw_traj"] = native_planning_capture["raw"]
            else:
                # Without collision optimization, raw and returned planning are
                # the same model output by definition.
                row["planning_raw_traj"] = pred_sdc_traj.detach().clone()
            # PlanningMetric.update() mutates trajs[..., 0] in place. Preserve
            # an evaluator-facing copy before invoking the stock metric and feed
            # clones to the metric so validation bookkeeping cannot alter model
            # outputs or GT tensors.
            row["planning_traj"] = pred_sdc_traj.detach().clone()
            row["planning_traj_gt"] = sdc_planning
            row["command"] = planning["planning_gt"]["command"]
            planning_metric(
                pred_sdc_traj[:, :6, :2].clone(),
                sdc_planning[0][0, :, :6, :2].clone(),
                sdc_planning_mask[0][0, :, :6, :2].clone(),
                segmentation[0][:, [1, 2, 3, 4, 5, 6]].clone(),
            )

        occ_valid = False
        if eval_occ:
            occ_has_invalid_frame = data["gt_occ_has_invalid_frame"][0]
            occ_valid = not bool(occ_has_invalid_frame.item())
            if occ_valid and "occ" in row:
                num_occ += 1
                for key, grid in ranges.items():
                    limits = slice(grid[0], grid[1])
                    iou_metrics[key](
                        row["occ"]["seg_out"][..., limits, limits].contiguous(),
                        row["occ"]["seg_gt"][..., limits, limits].contiguous(),
                    )
                    panoptic_metrics[key](
                        row["occ"]["ins_seg_out"][..., limits, limits].contiguous().detach(),
                        row["occ"]["ins_seg_gt"][..., limits, limits].contiguous(),
                    )

        evaluator_records.append(
            _native_evaluator_record(
                row, dataset.data_infos[index], occ_valid
            )
        )
        row.pop("occ", None)
        row.pop("planning", None)
        bbox_results.append(row)
        progress.update()

    occ_results = None
    if eval_occ:
        occ_results = {}
        for key in ranges:
            panoptic_scores = panoptic_metrics[key].compute()
            for metric_name, value in panoptic_scores.items():
                occ_results[metric_name] = occ_results.get(metric_name, []) + [
                    100 * value[1].item()
                ]
            iou_scores = iou_metrics[key].compute()
            occ_results["iou"] = occ_results.get("iou", []) + [
                100 * iou_scores[1].item()
            ]
        occ_results["num_occ"] = num_occ
        occ_results["ratio_occ"] = num_occ / len(indices)

    planning_results = planning_metric.compute() if eval_planning else None
    elapsed = time.time() - started
    payload = {
        "format": "uniad-native-cpu-results-v3",
        "repository_head": git_head(),
        "backend": "native_pt_cpu_cuda_contract",
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "tokens": selected_tokens,
        "token_order_sha256": sequence_sha256(selected_tokens),
        "evaluator_record_schema": EVALUATOR_RECORD_SCHEMA,
        "records": evaluator_records,
        "bbox_results": bbox_results,
        "occ_results_computed": occ_results,
        "planning_results_computed": planning_results,
    }
    atomic_pickle(output_dir / "results.pkl", payload)
    summary = {
        "ok": True,
        "backend": payload["backend"],
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "frames": len(indices),
        "token_order_sha256": payload["token_order_sha256"],
        "checkpoint_sha256": manifest["assets"]["checkpoint"]["sha256"],
        "cpu_preflight_stock_ready": preflight["native_stock_cpu_ready"],
        "msda_cuda_contract_cpu_compat": preflight["checks"]["msda_cuda_contract_cpu_compat"],
        "cuda_visible_devices_before": previous_cuda_visible,
        "cuda_visible_devices_for_run": "",
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": elapsed,
        "occ_results_computed": jsonable(occ_results),
        "planning_results_computed": jsonable(planning_results),
        "results_path": str(output_dir / "results.pkl"),
        "results_sha256": sha256(output_dir / "results.pkl"),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


class _SessionIO:
    def __init__(self, name):
        self.name = name


class _TorchStatefulSession:
    """ORT-like session facade backed by the actual export-adapted PT wrapper."""

    def __init__(self, wrapper, input_names, output_names):
        self.wrapper = wrapper
        self._inputs = [_SessionIO(name) for name in input_names]
        self._outputs = [_SessionIO(name) for name in output_names]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        import torch
        if output_names is not None:
            raise ValueError("validation PT session only supports all outputs")
        tensors = []
        for item in self._inputs:
            value = feed[item.name]
            if not isinstance(value, np.ndarray):
                value = np.asarray(value)
            array = value.copy() if value.ndim == 0 else np.ascontiguousarray(value)
            tensor = torch.from_numpy(array)
            if tuple(tensor.shape) != tuple(value.shape):
                raise ValueError(
                    f"stateful PT adapter changed ABI shape for {item.name}: "
                    f"{value.shape} -> {tuple(tensor.shape)}"
                )
            tensors.append(tensor)
        with torch.no_grad():
            outputs = self.wrapper(*tensors)
        if len(outputs) != len(self._outputs):
            raise ValueError("stateful PT wrapper output count mismatch")
        arrays = [value.detach().cpu().numpy() for value in outputs]
        output_by_name = {
            item.name: value for item, value in zip(self._outputs, arrays)
        }
        if output_by_name["next_max_obj_id"].shape != ():
            raise ValueError(
                "stateful PT wrapper violated scalar next_max_obj_id ABI: "
                f"{output_by_name['next_max_obj_id'].shape}"
            )
        return arrays


def _unwrap_single_tensor(value, name):
    import torch
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name}: expected one augmentation, got {len(value)}")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected tensor after CPU scatter, got {type(value)}")
    return value


def _native_map_gt_from_scattered(data):
    """Mirror PansegformerHead.forward_test's gt_lane_*[0][0] indexing."""
    labels = _unwrap_single_tensor(data["gt_lane_labels"], "gt_lane_labels")
    masks = _unwrap_single_tensor(data["gt_lane_masks"], "gt_lane_masks")
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError(
            "gt_lane_labels must be batch-1 after augmentation unwrap; "
            f"got {tuple(labels.shape)}"
        )
    if masks.ndim != 4 or masks.shape[0] != 1:
        raise ValueError(
            "gt_lane_masks must be batch-1 after augmentation unwrap; "
            f"got {tuple(masks.shape)}"
        )
    labels = labels[0]
    masks = masks[0]
    if labels.ndim != 1 or masks.ndim != 3 or labels.shape[0] != masks.shape[0]:
        raise ValueError(
            "native map GT layout mismatch after batch indexing: "
            f"labels={tuple(labels.shape)} masks={tuple(masks.shape)}"
        )
    return labels.detach().cpu().numpy(), masks.detach().cpu().numpy()


def stateful_run(args):
    """Run the exact export-adapted Stateful PT wrapper on one/full mini split."""
    run_dir = resolve(args.run_dir)
    manifest = load_execution_manifest(run_dir)
    require_native_cpu_preflight(run_dir)
    split = args.split
    if args.all_scenes and args.scene_name:
        raise ValueError("--all-scenes and --scene-name are mutually exclusive")

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import mmcv
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    from export import build_model as build_stateful_model
    from temporal import (
        StatefulStep, INPUT_NAMES, OUTPUT_NAMES,
    )
    from temporal import TRACK_STATE_NAMES, clone_state_for_input
    from host import StatefulRuntime
    from prepare_scene import frame_metadata
    from host import (
        consecutive_occupancy_ids,
        map_iou_counts,
        merge_map_masks,
        motion_evaluator_view,
        optimize_planning,
    )

    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be >= 1")
        torch.set_num_threads(args.threads)

    cfg, model = build_stateful_model(
        manifest["assets"]["config"]["path"],
        manifest["assets"]["checkpoint"]["path"],
    )
    wrapper = StatefulStep(model, cfg.occflow_grid_conf).eval().cpu()
    detection_coder = _make_evaluator_bbox_coder(reference=model.bbox_coder)

    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.ann_file = manifest["splits"][split]["info_path"]
    test_cfg.data_root = manifest["data_root"]
    test_cfg.test_mode = True
    test_cfg.file_client_args = dict(backend="disk")
    dataset = build_dataset(test_cfg)

    dataset_tokens = [item["token"] for item in dataset.data_infos]
    split_record = manifest["splits"][split]
    if dataset_tokens != split_record["tokens"]:
        raise ValueError("stateful dataset token order differs from immutable manifest")

    scene_name_by_token, _ = load_mini_tables(Path(manifest["data_root"]))
    scene_names = [
        scene_name_by_token[item["scene_token"]]
        for item in dataset.data_infos
    ]
    execution_order = split_record["scene_names_in_execution_order"]
    if args.all_scenes:
        selected_scene_names = list(execution_order)
        scope_name = "full"
    else:
        scene_name = args.scene_name or execution_order[0]
        if scene_name not in execution_order:
            raise ValueError(f"{scene_name} is not in {split}")
        selected_scene_names = [scene_name]
        scope_name = scene_name

    selected_set = set(selected_scene_names)
    indices = [
        index for index, scene_name in enumerate(scene_names)
        if scene_name in selected_set
    ]
    if not indices:
        raise ValueError("stateful-run selected zero frames")
    if not args.all_scenes:
        chosen_infos = [dataset.data_infos[index] for index in indices]
        if chosen_infos[0].get("prev", "") or chosen_infos[-1].get("next", ""):
            raise ValueError("stateful smoke scope is not a complete scene")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("stateful smoke scene is not contiguous")

    output_dir = run_dir / "stateful_pt" / split / scope_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite stateful PT evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    initial_tensors = clone_state_for_input(wrapper.cycle.initial_state())
    initial_state = {
        name: value.detach().cpu().numpy().copy()
        for name, value in zip(TRACK_STATE_NAMES, initial_tensors)
    }
    initial_state.update(
        prev_bev=np.zeros(
            (model.bev_h * model.bev_w, 1, 256), dtype=np.float32
        ),
        max_obj_id=np.array(0, dtype=np.int64),
    )
    session = _TorchStatefulSession(wrapper, INPUT_NAMES, OUTPUT_NAMES)
    runtime = StatefulRuntime(
        session,
        initial_state,
        can_bus_mode="official_test_legacy",
        id_scope="session",
        model_sha256=manifest["assets"]["checkpoint"]["sha256"],
    )

    records = []
    solver_invocations = 0
    started = time.time()
    progress = mmcv.ProgressBar(len(indices))
    for index in indices:
        info = dataset.data_infos[index]
        sample = dataset[index]
        data = scatter(collate([sample], samples_per_gpu=1), [-1])[0]
        image = _unwrap_single_tensor(data["img"], "img")
        if tuple(image.shape) != (1, 6, 3, 928, 1600):
            raise ValueError(f"unexpected stateful image shape: {tuple(image.shape)}")
        if image.dtype != torch.float32 or image.device.type != "cpu":
            raise ValueError("stateful image must be CPU float32")

        command_tensor = _unwrap_single_tensor(data["command"], "command")
        command_values = command_tensor.reshape(-1)
        if command_values.numel() != 1:
            raise ValueError(f"unexpected command shape: {tuple(command_tensor.shape)}")
        command = int(command_values.item())

        metadata = frame_metadata(info)
        outputs = runtime.step(
            scene_token=info["scene_token"],
            timestamp=float(info["timestamp"]) / 1e6,
            image=np.ascontiguousarray(image.numpy()),
            can_bus_absolute=metadata["can_bus_absolute"],
            l2g_r=metadata["l2g_r"],
            l2g_t=metadata["l2g_t"],
            lidar2img=metadata["lidar2img"],
            img_shape=np.asarray([[[928, 1600]] * 6], dtype=np.int64),
            command=command,
        )

        motion = motion_evaluator_view(outputs)
        map_result = merge_map_masks(
            outputs, reject_ambiguous_ties=True
        )
        gt_lane_labels, gt_lane_masks = _native_map_gt_from_scattered(data)
        map_counts = map_iou_counts(
            map_result, gt_lane_labels, gt_lane_masks
        )
        occ_instances = consecutive_occupancy_ids(outputs["occ_instances"])
        occ_valid = not bool(
            _unwrap_single_tensor(
                data["gt_occ_has_invalid_frame"], "gt_occ_has_invalid_frame"
            ).item()
        )
        planning_optimized, collision_info = optimize_planning(
            outputs["planning_raw"],
            outputs["occ_segmentation"],
            coordinate_mode="legacy_int",
        )
        solver_invocations += int(collision_info["solver_ran"])
        detection = _decode_detection_for_evaluator(
            outputs, detection_coder
        )
        records.append(
            _stateful_evaluator_record(
                outputs, info, detection, motion, map_counts, occ_instances,
                planning_optimized, collision_info, occ_valid,
            )
        )
        progress.update()

    tokens = [record["token"] for record in records]
    payload = {
        "format": "uniad-stateful-pt-cpu-results-v2",
        "repository_head": git_head(),
        "backend": "stateful_export_adapted_pt_cpu",
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "tokens": tokens,
        "token_order_sha256": sequence_sha256(tokens),
        "evaluator_record_schema": EVALUATOR_RECORD_SCHEMA,
        "records": records,
        "collision_solver_invocations": solver_invocations,
    }
    atomic_pickle(output_dir / "results.pkl", payload)
    summary = {
        "ok": True,
        "backend": payload["backend"],
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "frames": len(records),
        "token_order_sha256": payload["token_order_sha256"],
        "checkpoint_sha256": manifest["assets"]["checkpoint"]["sha256"],
        "cuda_visible_devices_before": previous_cuda_visible,
        "cuda_visible_devices_for_run": "",
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.time() - started,
        "collision_solver_invocations": solver_invocations,
        "results_path": str(output_dir / "results.pkl"),
        "results_sha256": sha256(output_dir / "results.pkl"),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def ort_run(args):
    """Run the exported ONNX bundle with StatefulRuntime on CPU."""
    run_dir = resolve(args.run_dir)
    manifest = load_execution_manifest(run_dir)
    require_native_cpu_preflight(run_dir)
    split = args.split
    if args.all_scenes and args.scene_name:
        raise ValueError("--all-scenes and --scene-name are mutually exclusive")

    required_assets = ("onnx", "initial_state", "initial_state_manifest")
    missing = [
        name for name in required_assets
        if not manifest["assets"][name]["exists"]
    ]
    if missing:
        raise FileNotFoundError(
            f"ORT run requires prepared ONNX bundle assets: {missing}"
        )
    initial_state_path = Path(manifest["assets"]["initial_state"]["path"])
    expected_state_manifest = initial_state_path.with_suffix(".json").resolve()
    actual_state_manifest = Path(
        manifest["assets"]["initial_state_manifest"]["path"]
    ).resolve()
    if expected_state_manifest != actual_state_manifest:
        raise ValueError(
            "StatefulRuntime.from_files expects the initial-state JSON beside "
            "the NPZ with the same stem"
        )

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    import mmcv
    install_tensorboard_import_stub()
    import projects.mmdet3d_plugin  # noqa: F401
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    from host import StatefulRuntime
    from prepare_scene import frame_metadata
    from host import (
        consecutive_occupancy_ids,
        map_iou_counts,
        merge_map_masks,
        motion_evaluator_view,
        optimize_planning,
    )

    threads = 4 if args.threads is None else args.threads
    if threads < 1:
        raise ValueError("--threads must be >= 1")

    config = Config.fromfile(manifest["assets"]["config"]["path"])
    test_cfg = copy.deepcopy(config.data.test)
    test_cfg.ann_file = manifest["splits"][split]["info_path"]
    test_cfg.data_root = manifest["data_root"]
    test_cfg.test_mode = True
    test_cfg.file_client_args = dict(backend="disk")
    dataset = build_dataset(test_cfg)

    dataset_tokens = [item["token"] for item in dataset.data_infos]
    split_record = manifest["splits"][split]
    if dataset_tokens != split_record["tokens"]:
        raise ValueError("ORT dataset token order differs from immutable manifest")

    scene_name_by_token, _ = load_mini_tables(Path(manifest["data_root"]))
    scene_names = [
        scene_name_by_token[item["scene_token"]]
        for item in dataset.data_infos
    ]
    execution_order = split_record["scene_names_in_execution_order"]
    if args.all_scenes:
        selected_scene_names = list(execution_order)
        scope_name = "full"
    else:
        scene_name = args.scene_name or execution_order[0]
        if scene_name not in execution_order:
            raise ValueError(f"{scene_name} is not in {split}")
        selected_scene_names = [scene_name]
        scope_name = scene_name

    selected_set = set(selected_scene_names)
    indices = [
        index for index, scene_name in enumerate(scene_names)
        if scene_name in selected_set
    ]
    if not indices:
        raise ValueError("ort-run selected zero frames")
    if not args.all_scenes:
        chosen_infos = [dataset.data_infos[index] for index in indices]
        if chosen_infos[0].get("prev", "") or chosen_infos[-1].get("next", ""):
            raise ValueError("ORT smoke scope is not a complete scene")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("ORT smoke scene is not contiguous")

    output_dir = run_dir / "ort" / split / scope_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite ORT evidence: {output_dir}")
    output_dir.mkdir(parents=True)

    runtime = StatefulRuntime.from_files(
        manifest["assets"]["onnx"]["path"],
        manifest["assets"]["initial_state"]["path"],
        can_bus_mode="official_test_legacy",
        id_scope="session",
        threads=threads,
    )
    if runtime.model_sha256 != manifest["assets"]["onnx"]["sha256"]:
        raise ValueError("ORT runtime model hash differs from immutable manifest")
    detection_coder = _make_evaluator_bbox_coder(
        pc_range=config.model.pc_range
    )

    records = []
    solver_invocations = 0
    started = time.time()
    progress = mmcv.ProgressBar(len(indices))
    for index in indices:
        info = dataset.data_infos[index]
        sample = dataset[index]
        data = scatter(collate([sample], samples_per_gpu=1), [-1])[0]
        image = _unwrap_single_tensor(data["img"], "img")
        if tuple(image.shape) != (1, 6, 3, 928, 1600):
            raise ValueError(f"unexpected ORT image shape: {tuple(image.shape)}")
        if image.dtype != torch.float32 or image.device.type != "cpu":
            raise ValueError("ORT image pipeline must produce CPU float32")

        command_tensor = _unwrap_single_tensor(data["command"], "command")
        command_values = command_tensor.reshape(-1)
        if command_values.numel() != 1:
            raise ValueError(f"unexpected command shape: {tuple(command_tensor.shape)}")
        command = int(command_values.item())

        metadata = frame_metadata(info)
        outputs = runtime.step(
            scene_token=info["scene_token"],
            timestamp=float(info["timestamp"]) / 1e6,
            image=np.ascontiguousarray(image.numpy()),
            can_bus_absolute=metadata["can_bus_absolute"],
            l2g_r=metadata["l2g_r"],
            l2g_t=metadata["l2g_t"],
            lidar2img=metadata["lidar2img"],
            img_shape=np.asarray([[[928, 1600]] * 6], dtype=np.int64),
            command=command,
        )

        motion = motion_evaluator_view(outputs)
        map_result = merge_map_masks(
            outputs, reject_ambiguous_ties=True
        )
        gt_lane_labels, gt_lane_masks = _native_map_gt_from_scattered(data)
        map_counts = map_iou_counts(
            map_result, gt_lane_labels, gt_lane_masks
        )
        occ_instances = consecutive_occupancy_ids(outputs["occ_instances"])
        occ_valid = not bool(
            _unwrap_single_tensor(
                data["gt_occ_has_invalid_frame"], "gt_occ_has_invalid_frame"
            ).item()
        )
        planning_optimized, collision_info = optimize_planning(
            outputs["planning_raw"],
            outputs["occ_segmentation"],
            coordinate_mode="legacy_int",
        )
        solver_invocations += int(collision_info["solver_ran"])
        detection = _decode_detection_for_evaluator(
            outputs, detection_coder
        )
        records.append(
            _stateful_evaluator_record(
                outputs, info, detection, motion, map_counts, occ_instances,
                planning_optimized, collision_info, occ_valid,
            )
        )
        progress.update()

    tokens = [record["token"] for record in records]
    payload = {
        "format": "uniad-ort-cpu-results-v2",
        "repository_head": git_head(),
        "backend": "onnxruntime_cpu",
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "tokens": tokens,
        "token_order_sha256": sequence_sha256(tokens),
        "evaluator_record_schema": EVALUATOR_RECORD_SCHEMA,
        "records": records,
        "collision_solver_invocations": solver_invocations,
    }
    atomic_pickle(output_dir / "results.pkl", payload)
    summary = {
        "ok": True,
        "backend": payload["backend"],
        "split": split,
        "scope": scope_name,
        "scene_names": selected_scene_names,
        "frames": len(records),
        "token_order_sha256": payload["token_order_sha256"],
        "onnx_sha256": manifest["assets"]["onnx"]["sha256"],
        "initial_state_sha256": manifest["assets"]["initial_state"]["sha256"],
        "cuda_visible_devices_before": previous_cuda_visible,
        "cuda_visible_devices_for_run": "",
        "ort_threads": threads,
        "elapsed_seconds": time.time() - started,
        "collision_solver_invocations": solver_invocations,
        "results_path": str(output_dir / "results.pkl"),
        "results_sha256": sha256(output_dir / "results.pkl"),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _nested(record, path):
    value = record
    for part in path.split("."):
        value = value[part]
    return value


def _compare_array(left, right, *, exact, atol, rtol):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "exact": bool(exact),
        }
    result = {
        "shape_equal": True,
        "shape": list(left.shape),
        "exact": bool(exact),
        "elements": int(left.size),
    }
    if exact:
        mismatch = int(np.count_nonzero(left != right))
        result.update(
            mismatch_count=mismatch,
            mismatch_ratio=0.0 if left.size == 0 else mismatch / left.size,
            equal=(mismatch == 0),
        )
        return result
    left64 = left.astype(np.float64, copy=False)
    right64 = right.astype(np.float64, copy=False)
    diff = np.abs(left64 - right64)
    denom = np.maximum(np.abs(right64), 1e-12)
    result.update(
        max_abs=0.0 if diff.size == 0 else float(diff.max()),
        mean_abs=0.0 if diff.size == 0 else float(diff.mean()),
        max_rel=0.0 if diff.size == 0 else float((diff / denom).max()),
        allclose=bool(np.allclose(left64, right64, atol=atol, rtol=rtol)),
        atol=float(atol),
        rtol=float(rtol),
    )
    return result


def _aggregate_map_records(records):
    result = {}
    specs = {
        "drivable": ("drivable_intersection", "drivable_union"),
        "lanes": ("lanes_intersection", "lanes_union"),
        "divider": ("divider_intersection", "divider_union"),
        "crossing": ("crossing_intersection", "crossing_union"),
        "contour": ("contour_intersection", "contour_union"),
    }
    for name, (intersection_key, union_key) in specs.items():
        intersection = sum(
            float(record["map"]["ret_iou"][intersection_key])
            for record in records
        )
        union = sum(
            float(record["map"]["ret_iou"][union_key])
            for record in records
        )
        result[name] = {
            "intersection": intersection,
            "union": union,
            "iou": None if union == 0 else intersection / union,
        }
    return result


def _tracking_id_index(record):
    tracking = record["tracking"]
    ids = np.asarray(tracking["ids"]).reshape(-1).astype(np.int64)
    scores = np.asarray(tracking["scores"]).reshape(-1)
    labels = np.asarray(tracking["labels"]).reshape(-1)
    boxes = np.asarray(tracking["boxes"])
    count = len(ids)
    if len(scores) != count or len(labels) != count or boxes.shape[0] != count:
        raise ValueError(
            "tracking evaluator rows are not aligned: "
            f"ids={count} scores={len(scores)} labels={len(labels)} boxes={boxes.shape}"
        )
    if len(np.unique(ids)) != count:
        raise ValueError("duplicate evaluator-facing tracking IDs in one frame")
    order = [int(value) for value in ids]
    rows = {
        int(obj_id): {
            "score": float(scores[index]),
            "label": int(labels[index]),
            "box": np.asarray(boxes[index]),
        }
        for index, obj_id in enumerate(ids)
    }
    return order, rows


def _tracking_first_seen(records):
    first_seen = {}
    for frame_index, record in enumerate(records):
        order, _ = _tracking_id_index(record)
        for obj_id in order:
            first_seen.setdefault(obj_id, frame_index)
    return first_seen


def _tracking_only_detail(obj_id, row, own_first_seen, other_first_seen):
    box = np.asarray(row["box"]).reshape(-1)
    return {
        "id": int(obj_id),
        "score": float(row["score"]),
        "label": int(row["label"]),
        "xyz": [float(value) for value in box[:3]],
        "own_first_seen_frame": own_first_seen.get(obj_id),
        "other_first_seen_frame": other_first_seen.get(obj_id),
        "ever_seen_other": obj_id in other_first_seen,
    }


def _pairwise_record_compare(left_payload, right_payload, *, atol, rtol):
    left_records = left_payload["records"]
    right_records = right_payload["records"]
    if len(left_records) != len(right_records):
        raise ValueError("record count differs between backends")
    left_raw_flags = [
        "raw" in record.get("planning", {}) for record in left_records
    ]
    right_raw_flags = [
        "raw" in record.get("planning", {}) for record in right_records
    ]
    if any(left_raw_flags) and not all(left_raw_flags):
        raise ValueError("left planning.raw is present for only a subset of frames")
    if any(right_raw_flags) and not all(right_raw_flags):
        raise ValueError("right planning.raw is present for only a subset of frames")
    planning_raw_comparison_available = (
        bool(left_records)
        and all(left_raw_flags)
        and all(right_raw_flags)
    )
    continuous = [
        "detection.boxes",
        "detection.scores",
        "tracking.boxes",
        "tracking.scores",
        "motion.traj",
        "motion.traj_scores",
        "planning.optimized",
    ]
    if planning_raw_comparison_available:
        continuous.append("planning.raw")
    continuous = tuple(continuous)
    exact = (
        "detection.labels",
        "tracking.labels",
        "tracking.ids",
        "occupancy.seg_out",
        "occupancy.ins_seg_out",
    )
    field_stats = {
        path: {
            "frames_compared": 0,
            "shape_mismatch_frames": 0,
            "allclose_frames": 0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "mean_abs_sum": 0.0,
        }
        for path in continuous
    }
    exact_stats = {
        path: {
            "frames_compared": 0,
            "shape_mismatch_frames": 0,
            "mismatch_elements": 0,
            "elements": 0,
        }
        for path in exact
    }
    map_count_mismatches = 0
    occupancy_valid_mismatches = 0
    frame_details = []
    left_tracking_first_seen = _tracking_first_seen(left_records)
    right_tracking_first_seen = _tracking_first_seen(right_records)

    for frame_index, (left, right) in enumerate(zip(left_records, right_records)):
        if left["token"] != right["token"]:
            raise ValueError("pairwise compare token mismatch")
        left_tracking_order, left_tracking_rows = _tracking_id_index(left)
        right_tracking_order, right_tracking_rows = _tracking_id_index(right)
        left_tracking_ids = set(left_tracking_order)
        right_tracking_ids = set(right_tracking_order)
        left_only_ids = sorted(left_tracking_ids - right_tracking_ids)
        right_only_ids = sorted(right_tracking_ids - left_tracking_ids)
        common_ids = sorted(left_tracking_ids & right_tracking_ids)
        common_score_abs = [
            abs(
                left_tracking_rows[obj_id]["score"]
                - right_tracking_rows[obj_id]["score"]
            )
            for obj_id in common_ids
        ]
        common_box_abs = [
            np.abs(
                left_tracking_rows[obj_id]["box"]
                - right_tracking_rows[obj_id]["box"]
            )
            for obj_id in common_ids
        ]
        frame_detail = {
            "frame_index": frame_index,
            "token": left["token"],
            "left_tracking_count": len(left_tracking_order),
            "right_tracking_count": len(right_tracking_order),
            "left_motion_count": int(np.asarray(left["motion"]["traj"]).shape[0]),
            "right_motion_count": int(np.asarray(right["motion"]["traj"]).shape[0]),
            "tracking_id_order_equal": left_tracking_order == right_tracking_order,
            "tracking_id_set_equal": not left_only_ids and not right_only_ids,
            "left_only_tracking": [
                _tracking_only_detail(
                    obj_id,
                    left_tracking_rows[obj_id],
                    left_tracking_first_seen,
                    right_tracking_first_seen,
                )
                for obj_id in left_only_ids
            ],
            "right_only_tracking": [
                _tracking_only_detail(
                    obj_id,
                    right_tracking_rows[obj_id],
                    right_tracking_first_seen,
                    left_tracking_first_seen,
                )
                for obj_id in right_only_ids
            ],
            "tracking_common_id_count": len(common_ids),
            "tracking_common_score_max_abs": (
                None if not common_score_abs else float(max(common_score_abs))
            ),
            "tracking_common_score_mean_abs": (
                None if not common_score_abs else float(np.mean(common_score_abs))
            ),
            "tracking_common_box_max_abs": (
                None if not common_box_abs
                else float(max(np.max(value) for value in common_box_abs))
            ),
            "tracking_common_box_mean_abs": (
                None if not common_box_abs
                else float(np.mean([np.mean(value) for value in common_box_abs]))
            ),
        }
        for path in continuous:
            report = _compare_array(
                _nested(left, path), _nested(right, path),
                exact=False, atol=atol, rtol=rtol,
            )
            stats = field_stats[path]
            stats["frames_compared"] += 1
            if not report["shape_equal"]:
                stats["shape_mismatch_frames"] += 1
                continue
            stats["allclose_frames"] += int(report["allclose"])
            stats["max_abs"] = max(stats["max_abs"], report["max_abs"])
            stats["max_rel"] = max(stats["max_rel"], report["max_rel"])
            stats["mean_abs_sum"] += report["mean_abs"]

        for path in exact:
            report = _compare_array(
                _nested(left, path), _nested(right, path),
                exact=True, atol=atol, rtol=rtol,
            )
            stats = exact_stats[path]
            stats["frames_compared"] += 1
            if not report["shape_equal"]:
                stats["shape_mismatch_frames"] += 1
                continue
            stats["mismatch_elements"] += report["mismatch_count"]
            stats["elements"] += report["elements"]

        left_map = left["map"]["ret_iou"]
        right_map = right["map"]["ret_iou"]
        count_keys = [
            key for key in left_map
            if key.endswith("_intersection") or key.endswith("_union")
        ]
        map_mismatch = (
            set(left_map) != set(right_map)
            or any(int(left_map[key]) != int(right_map[key]) for key in count_keys)
        )
        if map_mismatch:
            map_count_mismatches += 1
        occupancy_valid_mismatch = (
            bool(left["occupancy"]["valid"]) != bool(right["occupancy"]["valid"])
        )
        if occupancy_valid_mismatch:
            occupancy_valid_mismatches += 1

        detection_scores = _compare_array(
            left["detection"]["scores"], right["detection"]["scores"],
            exact=False, atol=atol, rtol=rtol,
        )
        planning = _compare_array(
            left["planning"]["optimized"], right["planning"]["optimized"],
            exact=False, atol=atol, rtol=rtol,
        )
        occ_seg = _compare_array(
            left["occupancy"]["seg_out"], right["occupancy"]["seg_out"],
            exact=True, atol=atol, rtol=rtol,
        )
        frame_detail.update({
            "tracking_count_mismatch": (
                frame_detail["left_tracking_count"]
                != frame_detail["right_tracking_count"]
            ),
            "map_count_mismatch": bool(map_mismatch),
            "occupancy_valid_mismatch": bool(occupancy_valid_mismatch),
            "detection_scores_max_abs": (
                None if not detection_scores["shape_equal"]
                else detection_scores["max_abs"]
            ),
            "detection_scores_mean_abs": (
                None if not detection_scores["shape_equal"]
                else detection_scores["mean_abs"]
            ),
            "planning_max_abs": (
                None if not planning["shape_equal"] else planning["max_abs"]
            ),
            "planning_mean_abs": (
                None if not planning["shape_equal"] else planning["mean_abs"]
            ),
            "occupancy_semantic_mismatch_elements": (
                None if not occ_seg["shape_equal"]
                else occ_seg["mismatch_count"]
            ),
        })
        frame_details.append(frame_detail)

    for stats in field_stats.values():
        compared = stats["frames_compared"] - stats["shape_mismatch_frames"]
        stats["mean_abs_mean_over_frames"] = (
            None if compared == 0 else stats.pop("mean_abs_sum") / compared
        )
    diagnostic_allclose = (
        all(
            stats["shape_mismatch_frames"] == 0
            and stats["allclose_frames"] == stats["frames_compared"]
            for stats in field_stats.values()
        )
        and all(
            stats["shape_mismatch_frames"] == 0
            and stats["mismatch_elements"] == 0
            for stats in exact_stats.values()
        )
        and map_count_mismatches == 0
        and occupancy_valid_mismatches == 0
    )
    first_tracking_divergence = next(
        (
            detail for detail in frame_details
            if detail["tracking_count_mismatch"]
        ),
        None,
    )
    first_tracking_id_order_divergence = next(
        (
            detail for detail in frame_details
            if not detail["tracking_id_order_equal"]
        ),
        None,
    )
    first_tracking_id_set_divergence = next(
        (
            detail for detail in frame_details
            if not detail["tracking_id_set_equal"]
        ),
        None,
    )
    first_new_visible_id_asymmetry = next(
        (
            detail for detail in frame_details
            if any(
                item["own_first_seen_frame"] == detail["frame_index"]
                and item["other_first_seen_frame"] != detail["frame_index"]
                for item in (
                    detail["left_only_tracking"]
                    + detail["right_only_tracking"]
                )
            )
        ),
        None,
    )
    first_map_divergence = next(
        (detail for detail in frame_details if detail["map_count_mismatch"]),
        None,
    )
    return {
        "diagnostic_only": True,
        "planning_raw_comparison_available": planning_raw_comparison_available,
        "left_planning_raw_available": bool(left_records) and all(left_raw_flags),
        "right_planning_raw_available": bool(right_records) and all(right_raw_flags),
        "continuous": field_stats,
        "exact": exact_stats,
        "map_count_mismatch_frames": map_count_mismatches,
        "occupancy_valid_mismatch_frames": occupancy_valid_mismatches,
        "first_tracking_count_divergence": first_tracking_divergence,
        "first_tracking_id_order_divergence": first_tracking_id_order_divergence,
        "first_tracking_id_set_divergence": first_tracking_id_set_divergence,
        "first_new_visible_id_asymmetry": first_new_visible_id_asymmetry,
        "first_map_count_divergence": first_map_divergence,
        "frame_details": frame_details,
        "diagnostic_allclose": diagnostic_allclose,
    }


def _evidence_manifest_for_results(path):
    path = resolve(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    for parent in path.parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            if manifest.get("format") == FORMAT:
                return path, candidate, manifest
    raise FileNotFoundError(
        f"no {FORMAT} manifest found above evidence file: {path}"
    )


def _asset_identity(manifest, name):
    record = manifest["assets"][name]
    if not record.get("exists"):
        return None
    return record.get("sha256")


def compare_smoke(args):
    """Compare three evaluator-facing evidence files across validation commits."""
    evidence_args = {
        "native": args.native_results,
        "stateful_pt": args.stateful_results,
        "ort": args.ort_results,
    }
    payloads = {}
    manifests = {}
    paths = {}
    manifest_paths = {}
    for name, value in evidence_args.items():
        path, manifest_path, manifest = _evidence_manifest_for_results(value)
        paths[name] = path
        manifest_paths[name] = manifest_path
        manifests[name] = manifest
        payloads[name] = load_pickle(path)

    identity = {}
    for name, payload in payloads.items():
        manifest = manifests[name]
        identity[f"{name}:result_head_matches_own_manifest"] = (
            payload.get("repository_head") == manifest.get("repository_head")
        )
        payload_schema = payload.get("evaluator_record_schema")
        identity[f"{name}:schema"] = (
            payload_schema in EVALUATOR_RECORD_SCHEMAS
            and len(payload.get("records", [])) == len(payload.get("tokens", []))
            and all(
                record.get("schema") == payload_schema
                for record in payload.get("records", [])
            )
        )
        tokens = payload.get("tokens")
        identity[f"{name}:token_hash_self_consistent"] = (
            tokens is not None
            and sequence_sha256(tokens) == payload.get("token_order_sha256")
        )

    reference = payloads["native"]
    split = reference.get("split")
    scope_name = reference.get("scope")
    expected_tokens = reference.get("tokens")
    for name, payload in payloads.items():
        identity[f"{name}:same_split"] = payload.get("split") == split
        identity[f"{name}:same_scope"] = payload.get("scope") == scope_name
        identity[f"{name}:same_tokens"] = payload.get("tokens") == expected_tokens

    identity_assets = (
        "checkpoint",
        "config",
        "motion_anchor",
        "mini_scene_table",
        "mini_sample_table",
    )
    for asset_name in identity_assets:
        expected = _asset_identity(manifests["native"], asset_name)
        for name, manifest in manifests.items():
            identity[f"{name}:same_{asset_name}"] = (
                expected is not None
                and _asset_identity(manifest, asset_name) == expected
            )

    # ORT evidence must additionally carry a cryptographic bundle binding to
    # the same checkpoint/config identities used by the PT evidence.
    ort_binding = manifests["ort"].get("onnx_bundle_binding")
    identity["ort:bundle_binding_present"] = ort_binding is not None
    if ort_binding is not None:
        identity["ort:bundle_checkpoint_matches"] = (
            ort_binding.get("checkpoint_sha256")
            == _asset_identity(manifests["native"], "checkpoint")
        )
        identity["ort:bundle_config_matches"] = (
            ort_binding.get("config_sha256")
            == _asset_identity(manifests["native"], "config")
        )

    identity_ok = all(identity.values())
    provenance = {
        name: {
            "results_path": str(paths[name]),
            "results_sha256": sha256(paths[name]),
            "manifest_path": str(manifest_paths[name]),
            "repository_head": manifests[name]["repository_head"],
            "checkpoint_sha256": _asset_identity(manifests[name], "checkpoint"),
            "config_sha256": _asset_identity(manifests[name], "config"),
            "motion_anchor_sha256": _asset_identity(manifests[name], "motion_anchor"),
        }
        for name in payloads
    }

    if not identity_ok:
        report = {
            "identity_ok": False,
            "identity": identity,
            "provenance": provenance,
            "final_acceptance": None,
            "note": "identity/provenance/schema gate failed; numerical comparison is invalid",
        }
    else:
        pairs = {
            "native_vs_stateful": _pairwise_record_compare(
                payloads["native"], payloads["stateful_pt"],
                atol=args.atol, rtol=args.rtol,
            ),
            "stateful_vs_ort": _pairwise_record_compare(
                payloads["stateful_pt"], payloads["ort"],
                atol=args.atol, rtol=args.rtol,
            ),
            "native_vs_ort": _pairwise_record_compare(
                payloads["native"], payloads["ort"],
                atol=args.atol, rtol=args.rtol,
            ),
        }
        report = {
            "identity_ok": True,
            "identity": identity,
            "provenance": provenance,
            "split": split,
            "scope": scope_name,
            "frames": len(expected_tokens),
            "diagnostic_tolerance": {
                "atol": float(args.atol),
                "rtol": float(args.rtol),
            },
            "map_scene_metrics": {
                name: _aggregate_map_records(payload["records"])
                for name, payload in payloads.items()
            },
            "pairs": pairs,
            "final_acceptance": None,
            "note": (
                "cross-commit scene-level task-final diagnostic only; formal "
                "detection/tracking/motion benchmark metrics require a complete "
                "split, and final equivalence acceptance is intentionally not "
                "decided here"
            ),
        }

    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite comparison evidence: {output}")
    atomic_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not identity_ok:
        raise SystemExit(1)



FORMAL_EVALUATION_FORMAT = "uniad-formal-evaluation-v1"


def _normalized_box_tensor(value, name):
    import torch
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 9:
        raise ValueError(f"{name}: expected [N,9], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite boxes")
    return torch.from_numpy(np.ascontiguousarray(array))


def _records_to_native_bbox_results(records):
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    rows = []
    for record in records:
        if record.get("schema") not in EVALUATOR_RECORD_SCHEMAS:
            raise ValueError("formal evaluator received unsupported record schema")
        det = record["detection"]
        track = record["tracking"]
        motion = record["motion"]

        track_boxes = _normalized_box_tensor(track["boxes"], "tracking.boxes")
        det_boxes = _normalized_box_tensor(det["boxes"], "detection.boxes")
        track_count = int(track_boxes.shape[0])
        if np.asarray(track["scores"]).shape != (track_count,):
            raise ValueError("tracking score rows do not match boxes")
        if np.asarray(track["labels"]).shape != (track_count,):
            raise ValueError("tracking label rows do not match boxes")
        if np.asarray(track["ids"]).shape != (track_count,):
            raise ValueError("tracking id rows do not match boxes")
        if np.asarray(motion["traj"]).shape[0] != track_count:
            raise ValueError("motion rows do not match tracking rows")
        if np.asarray(motion["traj_scores"]).shape[0] != track_count:
            raise ValueError("motion score rows do not match tracking rows")

        det_count = int(det_boxes.shape[0])
        if np.asarray(det["scores"]).shape != (det_count,):
            raise ValueError("detection score rows do not match boxes")
        if np.asarray(det["labels"]).shape != (det_count,):
            raise ValueError("detection label rows do not match boxes")

        ret_iou = {}
        for key, value in record["map"]["ret_iou"].items():
            ret_iou[key] = torch.tensor([float(value)], dtype=torch.float64)

        rows.append({
            "boxes_3d": LiDARInstance3DBoxes(track_boxes, box_dim=9),
            "scores_3d": torch.from_numpy(
                np.ascontiguousarray(np.asarray(track["scores"]))
            ),
            "labels_3d": torch.from_numpy(
                np.ascontiguousarray(np.asarray(track["labels"], dtype=np.int64))
            ),
            "track_ids": torch.from_numpy(
                np.ascontiguousarray(np.asarray(track["ids"], dtype=np.int64))
            ),
            "traj": torch.from_numpy(
                np.ascontiguousarray(np.asarray(motion["traj"]))
            ),
            "traj_scores": torch.from_numpy(
                np.ascontiguousarray(np.asarray(motion["traj_scores"]))
            ),
            "boxes_3d_det": LiDARInstance3DBoxes(det_boxes, box_dim=9),
            "scores_3d_det": torch.from_numpy(
                np.ascontiguousarray(np.asarray(det["scores"]))
            ),
            "labels_3d_det": torch.from_numpy(
                np.ascontiguousarray(np.asarray(det["labels"], dtype=np.int64))
            ),
            "ret_iou": ret_iou,
        })
    return rows


def _native_occ_gt_from_scattered(data, *, n_future=4, ignore_index=255):
    import torch
    seg = _unwrap_single_tensor(data["gt_segmentation"], "gt_segmentation")
    ins = _unwrap_single_tensor(data["gt_instance"], "gt_instance")
    if seg.ndim != 4 or seg.shape[0] != 1:
        raise ValueError(
            f"gt_segmentation must be [1,T,H,W], got {tuple(seg.shape)}"
        )
    if ins.ndim != 4 or ins.shape[0] != 1:
        raise ValueError(f"gt_instance must be [1,T,H,W], got {tuple(ins.shape)}")
    frames = n_future + 1
    if seg.shape[1] < frames or ins.shape[1] < frames:
        raise ValueError("occupancy GT does not contain required future frames")
    seg_gt = seg[:, :frames].long().unsqueeze(2)
    ins_old = ins[:, :frames].long()
    ins_new = torch.zeros_like(ins_old)
    new_id = 1
    for unique_id in torch.unique(ins_old):
        value = int(unique_id.item())
        if value in (0, ignore_index):
            continue
        ins_new[ins_old == unique_id] = new_id
        new_id += 1
    return seg_gt, ins_new


def _native_planning_gt_from_scattered(data):
    seg = _unwrap_single_tensor(data["gt_segmentation"], "gt_segmentation")
    planning = _unwrap_single_tensor(data["sdc_planning"], "sdc_planning")
    planning_mask = _unwrap_single_tensor(
        data["sdc_planning_mask"], "sdc_planning_mask"
    )
    if seg.ndim != 4 or seg.shape[0] != 1 or seg.shape[1] < 7:
        raise ValueError(
            f"planning segmentation must be [1,T>=7,H,W], got {tuple(seg.shape)}"
        )
    for name, value in (
        ("sdc_planning", planning),
        ("sdc_planning_mask", planning_mask),
    ):
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] < 6 or value.shape[3] < 2:
            raise ValueError(
                f"{name} must match Native [augmentation][batch,:,6,xy] contract; "
                f"got {tuple(value.shape)}"
            )
    gt = planning[0, :, :6, :2]
    mask = planning_mask[0, :, :6, :2]
    future_seg = seg[:, [1, 2, 3, 4, 5, 6]]
    if gt.shape != (1, 6, 2) or mask.shape != (1, 6, 2):
        raise ValueError(
            f"unexpected planning GT shapes: gt={tuple(gt.shape)} mask={tuple(mask.shape)}"
        )
    return gt, mask, future_seg


def _formal_map_metrics(records):
    aggregated = _aggregate_map_records(records)
    return {
        name + "_iou": values["iou"]
        for name, values in aggregated.items()
    }


def _formal_occ_planning_metrics(
    dataset, records, *, repair_native_planning_x=False
):
    import torch
    import mmcv
    from mmcv.parallel import collate, scatter
    from projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin import (
        IntersectionOverUnion,
        PanopticMetric,
    )
    from projects.mmdet3d_plugin.uniad.dense_heads.planning_head_plugin import (
        PlanningMetric,
    )

    ranges = {"30x30": (70, 130), "100x100": (0, 200)}
    iou_metrics = {
        key: IntersectionOverUnion(2).cpu() for key in ranges
    }
    panoptic_metrics = {
        key: PanopticMetric(n_classes=2, temporally_consistent=True).cpu()
        for key in ranges
    }
    planning_metric = PlanningMetric().cpu()
    raw_flags = [
        isinstance(record.get("planning"), dict)
        and "raw" in record["planning"]
        for record in records
    ]
    if any(raw_flags) and not all(raw_flags):
        raise ValueError(
            "planning.raw evidence is only present for a subset of frames"
        )
    raw_available = bool(records) and all(raw_flags)
    planning_raw_metric = PlanningMetric().cpu() if raw_available else None
    num_occ = 0
    progress = mmcv.ProgressBar(len(records))

    for index, record in enumerate(records):
        sample = dataset[index]
        data = scatter(collate([sample], samples_per_gpu=1), [-1])[0]
        invalid = bool(
            _unwrap_single_tensor(
                data["gt_occ_has_invalid_frame"], "gt_occ_has_invalid_frame"
            ).item()
        )
        record_valid = bool(record["occupancy"]["valid"])
        if record_valid != (not invalid):
            raise ValueError(
                f"{record['token']}: saved occupancy valid flag differs from dataset GT"
            )

        if record_valid:
            seg_gt, ins_gt = _native_occ_gt_from_scattered(data)
            pred_seg = torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(record["occupancy"]["seg_out"], dtype=np.int64)
                )
            )
            pred_ins = torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(record["occupancy"]["ins_seg_out"], dtype=np.int64)
                )
            )
            if pred_seg.shape != seg_gt.shape or pred_ins.shape != ins_gt.shape:
                raise ValueError(
                    f"{record['token']}: occupancy prediction/GT shape mismatch: "
                    f"seg={tuple(pred_seg.shape)}/{tuple(seg_gt.shape)} "
                    f"ins={tuple(pred_ins.shape)}/{tuple(ins_gt.shape)}"
                )
            num_occ += 1
            for key, grid in ranges.items():
                limits = slice(grid[0], grid[1])
                iou_metrics[key](
                    pred_seg[..., limits, limits].contiguous(),
                    seg_gt[..., limits, limits].contiguous(),
                )
                panoptic_metrics[key](
                    pred_ins[..., limits, limits].contiguous(),
                    ins_gt[..., limits, limits].contiguous(),
                )

        gt_plan, gt_plan_mask, planning_seg = _native_planning_gt_from_scattered(data)
        pred_plan = torch.from_numpy(
            np.ascontiguousarray(
                np.asarray(record["planning"]["optimized"], dtype=np.float32)
            )
        )
        if repair_native_planning_x:
            # Native result v1 was serialized after PlanningMetric.update(),
            # which flips trajs[..., 0] in place. Undo that bookkeeping-only
            # mutation before replaying the stock metric. New v2 results are
            # serialized before metric mutation and never take this path.
            pred_plan = pred_plan.clone()
            pred_plan[..., 0] = -pred_plan[..., 0]
        if pred_plan.shape != (1, 6, 2):
            raise ValueError(
                f"{record['token']}: optimized planning must be [1,6,2], "
                f"got {tuple(pred_plan.shape)}"
            )
        planning_metric(
            pred_plan.clone(),
            gt_plan.clone(),
            gt_plan_mask.clone(),
            planning_seg.clone(),
        )
        if planning_raw_metric is not None:
            pred_raw = torch.from_numpy(
                np.ascontiguousarray(
                    np.asarray(
                        record["planning"]["raw"], dtype=np.float32
                    )
                )
            )
            if pred_raw.shape != (1, 6, 2):
                raise ValueError(
                    f"{record['token']}: raw planning must be [1,6,2], "
                    f"got {tuple(pred_raw.shape)}"
                )
            planning_raw_metric(
                pred_raw.clone(),
                gt_plan.clone(),
                gt_plan_mask.clone(),
                planning_seg.clone(),
            )
        progress.update()

    occ = {}
    for key in ranges:
        panoptic_scores = panoptic_metrics[key].compute()
        for metric_name, value in panoptic_scores.items():
            occ[metric_name] = occ.get(metric_name, []) + [
                100 * value[1].item()
            ]
        iou_scores = iou_metrics[key].compute()
        occ["iou"] = occ.get("iou", []) + [100 * iou_scores[1].item()]
    occ["num_occ"] = num_occ
    occ["ratio_occ"] = num_occ / len(records)

    planning = {
        key: value.detach().cpu().tolist()
        for key, value in planning_metric.compute().items()
    }
    planning_raw = None
    if planning_raw_metric is not None:
        planning_raw = {
            key: value.detach().cpu().tolist()
            for key, value in planning_raw_metric.compute().items()
        }
    return occ, planning, planning_raw


def _motion_epa_scores(evaluator):
    """Expose the exact EPA values that the original MotionEval only prints."""
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval_motion import (
        accumulate,
        accumulate_motion,
        traj_fde,
    )
    scores = {}
    for class_name in evaluator.cfg.class_names:
        _, _, det_fp, det_gt = accumulate(
            evaluator.gt_boxes,
            evaluator.pred_boxes,
            class_name,
            evaluator.cfg.dist_fcn_callable,
            2.0,
        )
        _, traj_tp, _, _ = accumulate_motion(
            evaluator.gt_boxes,
            evaluator.pred_boxes,
            class_name,
            evaluator.cfg.dist_fcn_callable,
            traj_fde,
            2.0,
            2.0,
        )
        scores[class_name] = float(
            (traj_tp - 0.5 * det_fp) / (det_gt + 1e-5)
        )
    return scores


def _run_det_track_motion_evaluators(dataset, native_rows, split, output_dir):
    import mmcv
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import (
        NuScenesEval_custom,
    )
    from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval_motion import (
        MotionEval,
    )

    formatted_dir = output_dir / "formatted"
    result_path, _ = dataset.format_results(
        native_rows, jsonfile_prefix=str(formatted_dir)
    )
    det_result_path, _ = dataset.format_results_det(
        native_rows, jsonfile_prefix=str(formatted_dir)
    )

    metrics = {}
    eval_mod = set(dataset.eval_mod or [])

    if "det" in eval_mod:
        det_dir = output_dir / "detection"
        evaluator = NuScenesEval_custom(
            dataset.nusc,
            config=dataset.eval_detection_configs,
            result_path=det_result_path,
            eval_set=split,
            output_dir=str(det_dir),
            verbose=True,
            overlap_test=dataset.overlap_test,
            data_infos=dataset.data_infos,
        )
        evaluator.main(plot_examples=0, render_curves=False)
        metrics["detection"] = mmcv.load(str(det_dir / "metrics_summary.json"))

    if "track" in eval_mod:
        track_dir = output_dir / "tracking"
        evaluator = TrackingEval(
            config=config_factory("tracking_nips_2019"),
            result_path=result_path,
            eval_set=split,
            output_dir=str(track_dir),
            verbose=True,
            nusc_version=dataset.version,
            nusc_dataroot=dataset.data_root,
        )
        evaluator.main()
        metrics["tracking"] = mmcv.load(str(track_dir / "metrics_summary.json"))

    if "motion" in eval_mod:
        motion_metrics = {}
        for category in ("motion_category", "detection_category"):
            category_metrics = {}
            for mode in ("standard", "motion_map", "epa"):
                mode_dir = output_dir / "motion" / category / mode
                evaluator = MotionEval(
                    dataset.nusc,
                    config=dataset.eval_detection_configs,
                    result_path=result_path,
                    eval_set=split,
                    output_dir=str(mode_dir),
                    verbose=True,
                    overlap_test=dataset.overlap_test,
                    data_infos=dataset.data_infos,
                    category_convert_type=category,
                )
                category_metrics[mode] = evaluator.main(
                    plot_examples=0,
                    render_curves=False,
                    eval_mode=mode,
                )
                if mode == "epa":
                    category_metrics["epa_scores"] = _motion_epa_scores(
                        evaluator
                    )
            motion_metrics[category] = category_metrics
        metrics["motion"] = motion_metrics

    return metrics, str(result_path), str(det_result_path)


def evaluate_results(args):
    """Run the final task evaluators from a complete saved inference result."""
    results_path, manifest_path, manifest = _evidence_manifest_for_results(
        args.results
    )
    payload = load_pickle(results_path)
    if payload.get("repository_head") != manifest.get("repository_head"):
        raise ValueError("result producer HEAD does not match its own manifest")
    payload_schema = payload.get("evaluator_record_schema")
    if payload_schema not in EVALUATOR_RECORD_SCHEMAS:
        raise ValueError("formal evaluation requires a supported evaluator-facing schema")

    split = payload.get("split")
    if split not in SPLITS:
        raise ValueError(f"unsupported result split: {split}")
    if payload.get("scope") != "full":
        raise ValueError(
            "formal evaluator requires scope=full; scene smoke is diagnostic only"
        )
    records = payload.get("records", [])
    if any(record.get("schema") != payload_schema for record in records):
        raise ValueError(
            "formal evaluation record schema differs from payload declaration"
        )
    if payload_schema == EVALUATOR_RECORD_SCHEMA:
        if any("raw" not in record.get("planning", {}) for record in records):
            raise ValueError(
                "evaluator-facing-v2 requires planning.raw on every frame"
            )
    expected = EXPECTED[split]
    if len(records) != expected["frames"]:
        raise ValueError(
            f"{split}: formal evaluation requires {expected['frames']} frames, "
            f"got {len(records)}"
        )
    tokens = [record["token"] for record in records]
    split_manifest = manifest["splits"][split]
    if (
        tokens != payload.get("tokens")
        or tokens != split_manifest["tokens"]
        or sequence_sha256(tokens) != payload.get("token_order_sha256")
    ):
        raise ValueError("formal evaluator token order differs from immutable split")

    for name in ("config", "mini_scene_table", "mini_sample_table"):
        record = manifest["assets"][name]
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"formal evaluator asset changed: {name}")
    info_path = Path(split_manifest["info_path"])
    if not info_path.is_file() or sha256(info_path) != split_manifest["info_sha256"]:
        raise ValueError("formal evaluator split PKL changed")

    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite formal evaluator output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch
    from mmcv import Config

    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be >= 1")
        torch.set_num_threads(args.threads)

    install_tensorboard_import_stub()
    import projects.mmdet3d_plugin  # noqa: F401
    from mmdet3d.datasets import build_dataset

    config = Config.fromfile(manifest["assets"]["config"]["path"])
    test_cfg = copy.deepcopy(config.data.test)
    test_cfg.ann_file = split_manifest["info_path"]
    test_cfg.data_root = manifest["data_root"]
    test_cfg.test_mode = True
    test_cfg.file_client_args = dict(backend="disk")
    dataset = build_dataset(test_cfg)
    dataset_tokens = [item["token"] for item in dataset.data_infos]
    if dataset_tokens != tokens:
        raise ValueError("formal evaluator dataset token order differs from evidence")

    native_rows = _records_to_native_bbox_results(records)
    started = time.time()
    official_metrics, result_json, det_result_json = (
        _run_det_track_motion_evaluators(
            dataset, native_rows, split, output_dir
        )
    )
    map_metrics = _formal_map_metrics(records)
    legacy_native_planning_repair = (
        payload.get("backend") == "native_pt_cpu_cuda_contract"
        and payload.get("format") == "uniad-native-cpu-results-v1"
    )
    occ_metrics, planning_metrics, planning_raw_metrics = (
        _formal_occ_planning_metrics(
            dataset,
            records,
            repair_native_planning_x=legacy_native_planning_repair,
        )
    )
    if legacy_native_planning_repair:
        expected_planning = payload.get("planning_results_computed")
        if not isinstance(expected_planning, dict):
            raise ValueError(
                "legacy Native planning repair requires producer-computed "
                "planning metrics for verification"
            )
        for key in ("L2", "obj_col", "obj_box_col"):
            expected_metric = np.asarray(
                expected_planning[key], dtype=np.float64
            )
            replayed_metric = np.asarray(
                planning_metrics[key], dtype=np.float64
            )
            if not np.allclose(
                expected_metric, replayed_metric, atol=1e-6, rtol=1e-6
            ):
                raise ValueError(
                    "legacy Native planning repair did not reproduce the "
                    f"producer-computed {key}: "
                    f"expected={expected_metric.tolist()} "
                    f"replayed={replayed_metric.tolist()}"
                )

    report = {
        "format": FORMAL_EVALUATION_FORMAT,
        "results_path": str(results_path),
        "results_sha256": sha256(results_path),
        "manifest_path": str(manifest_path),
        "producer_repository_head": payload["repository_head"],
        "backend": payload.get("backend"),
        "split": split,
        "split_role": (
            "official_nuscenes_mini_validation"
            if split == "mini_val"
            else "validation_extension_same_evaluator_formulas"
        ),
        "scope": "full",
        "frames": len(records),
        "scenes": expected["scenes"],
        "token_order_sha256": payload["token_order_sha256"],
        "checkpoint_sha256": _asset_identity(manifest, "checkpoint"),
        "config_sha256": _asset_identity(manifest, "config"),
        "motion_anchor_sha256": _asset_identity(manifest, "motion_anchor"),
        "formatted_result_json": result_json,
        "formatted_detection_json": det_result_json,
        "metrics": {
            **official_metrics,
            "map": map_metrics,
            "occupancy": occ_metrics,
            "planning": planning_metrics,
            "planning_raw": planning_raw_metrics,
        },
        "planning_raw_available": planning_raw_metrics is not None,
        "legacy_native_planning_record_repair": legacy_native_planning_repair,
        "cuda_visible_devices_before": previous_cuda_visible,
        "cuda_visible_devices_for_evaluation": "",
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.time() - started,
    }
    report = finite_jsonable(report)
    atomic_json(output_dir / "evaluation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))



FINAL_ACCEPTANCE_POLICY_VERSION = "uniad-final-metric-policy-v1"
FINAL_ACCEPTANCE_POLICY = {
    "detection": {
        "metrics": {
            "nd_score": 0.01,
            "mean_ap": 0.01,
        },
        "unit": "absolute metric fraction",
    },
    "tracking": {
        "metrics": {
            "amota": 0.01,
            "amotp": 0.05,
        },
        "unit": "AMOTA absolute fraction; AMOTP native metric units",
    },
    "motion": {
        "standard_tp_errors": {
            "min_ade_err": 0.10,
            "min_fde_err": 0.15,
            "miss_rate_err": 0.01,
        },
        "motion_map_mean_ap": 0.01,
        "epa_per_class": 0.02,
        "unit": "ADE/FDE metres; rate/AP/EPA absolute fraction",
    },
    "map": {
        "all_iou": 0.01,
        "unit": "absolute IoU fraction",
    },
    "occupancy": {
        "iou_pq_sq_rq": 1.0,
        "unit": "percentage points",
    },
    "planning": {
        "L2": 0.10,
        "obj_col": 0.01,
        "obj_box_col": 0.01,
        "unit": "L2 metres; collision absolute rate",
    },
}


def _evaluation_metric(metrics, path):
    value = metrics
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"evaluation metric missing: {path}")
        value = value[key]
    return value


def _accept_numeric(name, native, candidate, tolerance):
    native = np.asarray(native, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if native.shape != candidate.shape:
        return {
            "name": name,
            "pass": False,
            "reason": "shape_mismatch",
            "native_shape": list(native.shape),
            "candidate_shape": list(candidate.shape),
            "tolerance": tolerance,
        }
    finite = np.isfinite(native) & np.isfinite(candidate)
    same_nonfinite = np.array_equal(np.isnan(native), np.isnan(candidate))
    if not same_nonfinite:
        return {
            "name": name,
            "pass": False,
            "reason": "nonfinite_pattern_mismatch",
            "tolerance": tolerance,
        }
    diff = np.abs(native[finite] - candidate[finite])
    max_abs = 0.0 if diff.size == 0 else float(diff.max())
    return {
        "name": name,
        "pass": bool(max_abs <= tolerance),
        "max_abs_diff": max_abs,
        "tolerance": float(tolerance),
        "native": finite_jsonable(native.tolist()),
        "candidate": finite_jsonable(candidate.tolist()),
    }


def compare_evaluations(args):
    native_path = resolve(args.native_evaluation)
    candidate_path = resolve(args.candidate_evaluation)
    for path in (native_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    native = json.loads(native_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for name, report in (("native", native), ("candidate", candidate)):
        if report.get("format") != FORMAL_EVALUATION_FORMAT:
            raise ValueError(f"{name}: not a formal UniAD evaluation")
        if report.get("scope") != "full":
            raise ValueError(f"{name}: formal comparison requires full scope")

    identity = {
        "same_split": native.get("split") == candidate.get("split"),
        "same_frames": native.get("frames") == candidate.get("frames"),
        "same_token_order": (
            native.get("token_order_sha256")
            == candidate.get("token_order_sha256")
        ),
        "same_checkpoint": (
            native.get("checkpoint_sha256")
            == candidate.get("checkpoint_sha256")
        ),
        "same_config": (
            native.get("config_sha256") == candidate.get("config_sha256")
        ),
        "same_motion_anchor": (
            native.get("motion_anchor_sha256")
            == candidate.get("motion_anchor_sha256")
        ),
    }
    if not all(identity.values()):
        raise ValueError(f"formal evaluation identity gate failed: {identity}")

    nm = native["metrics"]
    cm = candidate["metrics"]
    checks = []

    for key, tolerance in FINAL_ACCEPTANCE_POLICY["detection"]["metrics"].items():
        checks.append(_accept_numeric(
            f"detection.{key}",
            _evaluation_metric(nm, f"detection.{key}"),
            _evaluation_metric(cm, f"detection.{key}"),
            tolerance,
        ))

    for key, tolerance in FINAL_ACCEPTANCE_POLICY["tracking"]["metrics"].items():
        checks.append(_accept_numeric(
            f"tracking.{key}",
            _evaluation_metric(nm, f"tracking.{key}"),
            _evaluation_metric(cm, f"tracking.{key}"),
            tolerance,
        ))

    # DetectionMotionMetrics inherits nuScenes DetectionMetrics.serialize().
    # The trajectory TP metrics (minADE/minFDE/miss-rate) are therefore stored
    # per class in label_tp_errors, not in the aggregate tp_errors dictionary
    # (which only contains the five stock nuScenes detection TP errors).
    # Apply the already-frozen tolerance to every motion-category class by
    # comparing the class-aligned vector and taking its maximum absolute diff.
    for key, tolerance in FINAL_ACCEPTANCE_POLICY["motion"]["standard_tp_errors"].items():
        path = "motion.motion_category.standard.label_tp_errors"
        native_rows = _evaluation_metric(nm, path)
        candidate_rows = _evaluation_metric(cm, path)
        native_classes = sorted(native_rows)
        candidate_classes = sorted(candidate_rows)
        check_name = f"{path}.*.{key}"
        if native_classes != candidate_classes:
            checks.append({
                "name": check_name,
                "pass": False,
                "reason": "class_set_mismatch",
                "native_classes": native_classes,
                "candidate_classes": candidate_classes,
                "tolerance": float(tolerance),
            })
            continue
        native_values = []
        candidate_values = []
        for class_name in native_classes:
            if key not in native_rows[class_name] or key not in candidate_rows[class_name]:
                raise KeyError(
                    "evaluation metric missing: "
                    f"{path}.<class>.{key}"
                )
            native_values.append(native_rows[class_name][key])
            candidate_values.append(candidate_rows[class_name][key])
        check = _accept_numeric(
            check_name,
            native_values,
            candidate_values,
            tolerance,
        )
        check["classes"] = native_classes
        checks.append(check)
    path = "motion.motion_category.motion_map.mean_ap"
    checks.append(_accept_numeric(
        path,
        _evaluation_metric(nm, path),
        _evaluation_metric(cm, path),
        FINAL_ACCEPTANCE_POLICY["motion"]["motion_map_mean_ap"],
    ))
    native_epa = _evaluation_metric(
        nm, "motion.motion_category.epa_scores"
    )
    candidate_epa = _evaluation_metric(
        cm, "motion.motion_category.epa_scores"
    )
    if set(native_epa) != set(candidate_epa):
        checks.append({
            "name": "motion.motion_category.epa_scores",
            "pass": False,
            "reason": "class_set_mismatch",
        })
    else:
        for class_name in sorted(native_epa):
            checks.append(_accept_numeric(
                f"motion.motion_category.epa_scores.{class_name}",
                native_epa[class_name],
                candidate_epa[class_name],
                FINAL_ACCEPTANCE_POLICY["motion"]["epa_per_class"],
            ))

    for key in (
        "drivable_iou", "lanes_iou", "divider_iou",
        "crossing_iou", "contour_iou",
    ):
        checks.append(_accept_numeric(
            f"map.{key}",
            _evaluation_metric(nm, f"map.{key}"),
            _evaluation_metric(cm, f"map.{key}"),
            FINAL_ACCEPTANCE_POLICY["map"]["all_iou"],
        ))

    for key in ("iou", "pq", "sq", "rq"):
        checks.append(_accept_numeric(
            f"occupancy.{key}",
            _evaluation_metric(nm, f"occupancy.{key}"),
            _evaluation_metric(cm, f"occupancy.{key}"),
            FINAL_ACCEPTANCE_POLICY["occupancy"]["iou_pq_sq_rq"],
        ))
    for key in ("num_occ", "ratio_occ"):
        checks.append(_accept_numeric(
            f"occupancy.{key}",
            _evaluation_metric(nm, f"occupancy.{key}"),
            _evaluation_metric(cm, f"occupancy.{key}"),
            0.0,
        ))

    for key in ("L2", "obj_col", "obj_box_col"):
        checks.append(_accept_numeric(
            f"planning.{key}",
            _evaluation_metric(nm, f"planning.{key}"),
            _evaluation_metric(cm, f"planning.{key}"),
            FINAL_ACCEPTANCE_POLICY["planning"][key],
        ))

    final_metric_pass = all(check["pass"] for check in checks)

    # planning_raw is a separately frozen export-boundary supplement. It does
    # not retroactively change the already-frozen final task-metric v1 gate.
    # New v2 evaluator evidence carries it; legacy v1 evidence remains valid
    # but reports this supplemental gate as unavailable.
    native_raw = nm.get("planning_raw")
    candidate_raw = cm.get("planning_raw")
    raw_checks = []
    raw_available = (
        isinstance(native_raw, dict)
        and isinstance(candidate_raw, dict)
    )
    if raw_available:
        for key in ("L2", "obj_col", "obj_box_col"):
            raw_checks.append(_accept_numeric(
                f"planning_raw.{key}",
                native_raw[key],
                candidate_raw[key],
                FINAL_ACCEPTANCE_POLICY["planning"][key],
            ))
    raw_pass = (
        all(check["pass"] for check in raw_checks)
        if raw_available else None
    )

    result = {
        "format": "uniad-formal-evaluation-comparison-v1",
        "policy_version": FINAL_ACCEPTANCE_POLICY_VERSION,
        "policy": FINAL_ACCEPTANCE_POLICY,
        "native_evaluation": str(native_path),
        "candidate_evaluation": str(candidate_path),
        "candidate_backend": candidate.get("backend"),
        "split": native["split"],
        "identity": identity,
        "checks": checks,
        "pass": final_metric_pass,
        "planning_raw_supplement": {
            "policy": {
                "version": "uniad-planning-raw-policy-v1",
                "metrics": FINAL_ACCEPTANCE_POLICY["planning"],
                "scope": (
                    "pre-collision-optimization planning trajectory evaluated "
                    "with the stock PlanningMetric formulas"
                ),
            },
            "available": raw_available,
            "checks": raw_checks,
            "pass": raw_pass,
        },
        "pass_with_planning_raw": (
            final_metric_pass and raw_pass
            if raw_available else None
        ),
        "overall_pass": (
            final_metric_pass and raw_pass
            if raw_available else final_metric_pass
        ),
    }
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite acceptance report: {output}")
    atomic_json(output, finite_jsonable(result))
    print(json.dumps(finite_jsonable(result), indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["overall_pass"] else 2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_cmd = sub.add_parser("prepare")
    prepare_cmd.add_argument("--run-dir", required=True)
    prepare_cmd.add_argument("--train-info", default="data/infos/nuscenes_infos_temporal_train.pkl")
    prepare_cmd.add_argument("--val-info", default="data/infos/nuscenes_infos_temporal_val.pkl")
    prepare_cmd.add_argument("--data-root", default="data/nuscenes")
    prepare_cmd.add_argument("--config", default="projects/configs/stage2_e2e/base_e2e.py")
    prepare_cmd.add_argument("--checkpoint", default="ckpts/uniad_base_e2e.pth")
    prepare_cmd.add_argument("--motion-anchor", default="data/others/motion_anchor_infos_mode6.pkl")
    prepare_cmd.add_argument("--onnx", default="onnx/uniad_stage2_stateful_v3.onnx")
    prepare_cmd.add_argument("--initial-state", default="onnx/uniad_stage2_stateful_v3.initial_state.npz")
    prepare_cmd.add_argument("--initial-state-manifest", default="onnx/uniad_stage2_stateful_v3.initial_state.json")
    prepare_cmd.set_defaults(func=prepare)

    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--run-dir", required=True)
    verify_cmd.set_defaults(func=verify)

    cpu_cmd = sub.add_parser("cpu-preflight")
    cpu_cmd.add_argument("--run-dir", required=True)
    cpu_cmd.add_argument("--json-out")
    cpu_cmd.set_defaults(func=cpu_preflight)

    temporal_cmd = sub.add_parser("temporal-component-audit")
    temporal_cmd.add_argument("--run-dir", required=True)
    temporal_cmd.add_argument("--output")
    temporal_cmd.set_defaults(func=temporal_component_audit)

    native_cmd = sub.add_parser("native-run")
    native_cmd.add_argument("--run-dir", required=True)
    native_cmd.add_argument("--split", choices=SPLITS, default="mini_val")
    native_cmd.add_argument("--scene-name")
    native_cmd.add_argument("--all-scenes", action="store_true")
    native_cmd.add_argument("--threads", type=int)
    native_cmd.set_defaults(func=native_run)

    stateful_cmd = sub.add_parser("stateful-run")
    stateful_cmd.add_argument("--run-dir", required=True)
    stateful_cmd.add_argument("--split", choices=SPLITS, default="mini_val")
    stateful_cmd.add_argument("--scene-name")
    stateful_cmd.add_argument("--all-scenes", action="store_true")
    stateful_cmd.add_argument("--threads", type=int)
    stateful_cmd.set_defaults(func=stateful_run)

    ort_cmd = sub.add_parser("ort-run")
    ort_cmd.add_argument("--run-dir", required=True)
    ort_cmd.add_argument("--split", choices=SPLITS, default="mini_val")
    ort_cmd.add_argument("--scene-name")
    ort_cmd.add_argument("--all-scenes", action="store_true")
    ort_cmd.add_argument("--threads", type=int)
    ort_cmd.set_defaults(func=ort_run)

    compare_cmd = sub.add_parser("compare-smoke")
    compare_cmd.add_argument("--native-results", required=True)
    compare_cmd.add_argument("--stateful-results", required=True)
    compare_cmd.add_argument("--ort-results", required=True)
    compare_cmd.add_argument("--output", required=True)
    compare_cmd.add_argument("--atol", type=float, default=1e-4)
    compare_cmd.add_argument("--rtol", type=float, default=1e-4)
    compare_cmd.set_defaults(func=compare_smoke)

    eval_cmd = sub.add_parser("evaluate-results")
    eval_cmd.add_argument("--results", required=True)
    eval_cmd.add_argument("--output-dir", required=True)
    eval_cmd.add_argument("--threads", type=int)
    eval_cmd.set_defaults(func=evaluate_results)

    accept_cmd = sub.add_parser("compare-evaluations")
    accept_cmd.add_argument("--native-evaluation", required=True)
    accept_cmd.add_argument("--candidate-evaluation", required=True)
    accept_cmd.add_argument("--output", required=True)
    accept_cmd.set_defaults(func=compare_evaluations)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)




if __name__ == "__main__":
    main()
