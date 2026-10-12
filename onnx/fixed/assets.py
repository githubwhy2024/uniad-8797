#!/usr/bin/env python3
"""Preflight local assets and nuScenes-mini coverage for equivalence validation.

This script intentionally performs no model inference and uses only the Python
standard library. Stock UniAD train/val info PKLs describe the full trainval
split; official nuScenes mini_train/mini_val membership is defined separately
by the nuScenes devkit scene-name lists.
"""

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "mini_train": {"frames": 323, "scenes": 8},
    "mini_val": {"frames": 81, "scenes": 2},
}
OFFICIAL_MINI_SPLITS = {
    "mini_train": [
        "scene-0061",
        "scene-0553",
        "scene-0655",
        "scene-0757",
        "scene-0796",
        "scene-1077",
        "scene-1094",
        "scene-1100",
    ],
    "mini_val": [
        "scene-0103",
        "scene-0916",
    ],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve_camera_path(raw_path, data_root):
    raw = Path(raw_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((ROOT / raw, data_root / raw))
        parts = raw.parts
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "nuscenes":
            candidates.append(data_root.joinpath(*parts[2:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else raw


def load_info_source(name, path):
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or "infos" not in payload or "metadata" not in payload:
        raise ValueError(f"{path}: expected dict with infos + metadata")
    infos = payload["infos"]
    return {
        "name": name,
        "path": str(path),
        "version": payload["metadata"].get("version"),
        "frames": len(infos),
        "scenes": len({item["scene_token"] for item in infos}),
        "infos": infos,
    }


def load_mini_scenes(data_root):
    scene_path = data_root / "v1.0-mini" / "scene.json"
    rows = json.loads(scene_path.read_text(encoding="utf-8"))
    name_to_token = {row["name"]: row["token"] for row in rows}
    token_to_name = {row["token"]: row["name"] for row in rows}
    if len(name_to_token) != len(rows) or len(token_to_name) != len(rows):
        raise ValueError(f"{scene_path}: duplicate scene name or token")
    return scene_path, rows, name_to_token, token_to_name


def inspect_mini_split(
    name,
    records,
    expected_scene_names,
    token_to_name,
    source_versions,
    data_root,
    check_images=True,
):
    expected_scene_names = list(expected_scene_names)
    expected_scene_name_set = set(expected_scene_names)
    selected_raw = [
        (source_name, info)
        for source_name, info in records
        if token_to_name.get(info["scene_token"]) in expected_scene_name_set
    ]
    infos = sorted((info for _, info in selected_raw), key=lambda item: item["timestamp"])
    tokens = [item["token"] for item in infos]
    scenes = [item["scene_token"] for item in infos]
    scene_counts = Counter(scenes)
    actual_scene_names = {token_to_name[token] for token in scene_counts}
    transitions = [scenes[0]] if scenes else []
    for previous, current in zip(scenes, scenes[1:]):
        if current != previous:
            transitions.append(current)

    origin_frames = Counter(source_name for source_name, _ in selected_raw)
    origin_scenes = {}
    for source_name, info in selected_raw:
        origin_scenes.setdefault(source_name, set()).add(info["scene_token"])

    camera_count_errors = []
    missing_images = []
    for index, info in enumerate(infos):
        cams = info.get("cams", {})
        if len(cams) != 6:
            camera_count_errors.append({
                "index": index,
                "token": info.get("token"),
                "count": len(cams),
            })
            continue
        if check_images:
            for camera_name, camera in cams.items():
                candidate = resolve_camera_path(camera["data_path"], data_root)
                if not candidate.is_file():
                    missing_images.append({
                        "index": index,
                        "token": info["token"],
                        "camera": camera_name,
                        "path": str(candidate),
                    })

    expected = EXPECTED[name]
    checks = {
        "source_versions_are_v1.0-trainval": all(
            version == "v1.0-trainval" for version in source_versions.values()
        ),
        "official_scene_names_exact": actual_scene_names == expected_scene_name_set,
        "frame_count": len(infos) == expected["frames"],
        "scene_count": len(scene_counts) == expected["scenes"],
        "tokens_unique": len(tokens) == len(set(tokens)),
        "scenes_contiguous_after_timestamp_sort": len(transitions) == len(scene_counts),
        "six_cameras_per_frame": not camera_count_errors,
        "camera_files_exist": (not missing_images) if check_images else None,
    }
    return {
        "name": name,
        "official_scene_names": expected_scene_names,
        "actual_scene_names": sorted(actual_scene_names),
        "source_origin_frames": dict(origin_frames),
        "source_origin_scenes": {
            source_name: len(scene_tokens)
            for source_name, scene_tokens in origin_scenes.items()
        },
        "mini_frames": len(infos),
        "mini_scenes": len(scene_counts),
        "scene_counts": dict(scene_counts),
        "scene_order": [token_to_name[token] for token in transitions],
        "first_token": tokens[0] if tokens else None,
        "last_token": tokens[-1] if tokens else None,
        "camera_count_errors": camera_count_errors[:20],
        "missing_camera_files": missing_images[:20],
        "missing_camera_file_count": len(missing_images),
        "checks": checks,
        "tokens": tokens,
        "scene_tokens": sorted(scene_counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-info", default="data/infos/nuscenes_infos_temporal_train.pkl")
    parser.add_argument("--val-info", default="data/infos/nuscenes_infos_temporal_val.pkl")
    parser.add_argument("--data-root", default="data/nuscenes")
    parser.add_argument("--checkpoint", default="ckpts/uniad_base_e2e.pth")
    parser.add_argument("--motion-anchor", default="data/others/motion_anchor_infos_mode6.pkl")
    parser.add_argument("--onnx", default="onnx/uniad_stage2_stateful_v3.onnx")
    parser.add_argument("--initial-state", default="onnx/uniad_stage2_stateful_v3.initial_state.npz")
    parser.add_argument(
        "--initial-state-manifest",
        default="onnx/uniad_stage2_stateful_v3.initial_state.json",
    )
    parser.add_argument(
        "--require-onnx",
        action="store_true",
        help="make the ONNX + initial-state bundle mandatory for this preflight",
    )
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--json-out", help="optional path for the full JSON report")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    mini_scene_table = data_root / "v1.0-mini" / "scene.json"
    native_paths = {
        "checkpoint": resolve_path(args.checkpoint),
        "motion_anchor": resolve_path(args.motion_anchor),
        "train_info": resolve_path(args.train_info),
        "val_info": resolve_path(args.val_info),
        "mini_scene_table": mini_scene_table,
        "nuscenes_samples": data_root / "samples",
        "nuscenes_maps": data_root / "maps",
    }
    onnx_paths = {
        "onnx": resolve_path(args.onnx),
        "initial_state": resolve_path(args.initial_state),
        "initial_state_manifest": resolve_path(args.initial_state_manifest),
    }

    report = {
        "native_assets": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in native_paths.items()
        },
        "onnx_assets": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in onnx_paths.items()
        },
        "info_sources": {},
        "mini_scene_table": {},
        "splits": {},
        "cross_split_checks": {},
        "onnx_bundle_checks": {},
    }

    native_assets_ok = all(item["exists"] for item in report["native_assets"].values())
    split_ok = False
    required_split_files = (
        native_paths["train_info"].is_file()
        and native_paths["val_info"].is_file()
        and mini_scene_table.is_file()
    )
    if required_split_files:
        train_source = load_info_source("train_info", native_paths["train_info"])
        val_source = load_info_source("val_info", native_paths["val_info"])
        sources = [train_source, val_source]
        source_versions = {source["name"]: source["version"] for source in sources}
        report["info_sources"] = {
            source["name"]: {
                "path": source["path"],
                "version": source["version"],
                "frames": source["frames"],
                "scenes": source["scenes"],
            }
            for source in sources
        }

        scene_path, mini_scene_rows, name_to_token, token_to_name = load_mini_scenes(data_root)
        mini_scene_names = set(name_to_token)
        official_scene_names = {
            scene_name
            for split_names in OFFICIAL_MINI_SPLITS.values()
            for scene_name in split_names
        }
        report["mini_scene_table"] = {
            "path": str(scene_path),
            "scenes": len(mini_scene_rows),
            "scene_names": [row["name"] for row in mini_scene_rows],
            "official_mini_train": OFFICIAL_MINI_SPLITS["mini_train"],
            "official_mini_val": OFFICIAL_MINI_SPLITS["mini_val"],
        }

        records = [
            (source["name"], info)
            for source in sources
            for info in source["infos"]
        ]
        source_token_sets = {
            source["name"]: {item["token"] for item in source["infos"]}
            for source in sources
        }
        source_scene_sets = {
            source["name"]: {item["scene_token"] for item in source["infos"]}
            for source in sources
        }

        split_reports = {}
        split_tokens = {}
        split_scenes = {}
        for split_name, split_scene_names in OFFICIAL_MINI_SPLITS.items():
            split_report = inspect_mini_split(
                split_name,
                records,
                split_scene_names,
                token_to_name,
                source_versions,
                data_root,
                check_images=not args.skip_image_check,
            )
            split_tokens[split_name] = set(split_report.pop("tokens"))
            split_scenes[split_name] = set(split_report.pop("scene_tokens"))
            split_reports[split_name] = split_report
        report["splits"] = split_reports

        covered_tokens = split_tokens["mini_train"] | split_tokens["mini_val"]
        covered_scenes = split_scenes["mini_train"] | split_scenes["mini_val"]
        report["cross_split_checks"] = {
            "source_token_sets_disjoint": source_token_sets["train_info"].isdisjoint(
                source_token_sets["val_info"]
            ),
            "source_scene_sets_disjoint": source_scene_sets["train_info"].isdisjoint(
                source_scene_sets["val_info"]
            ),
            "official_split_names_match_mini_scene_table": official_scene_names == mini_scene_names,
            "token_sets_disjoint": split_tokens["mini_train"].isdisjoint(
                split_tokens["mini_val"]
            ),
            "scene_sets_disjoint": split_scenes["mini_train"].isdisjoint(
                split_scenes["mini_val"]
            ),
            "mini_scene_table_has_10_scenes": len(mini_scene_names) == 10,
            "all_mini_scenes_covered_exactly": covered_scenes == set(token_to_name),
            "total_frames_404": len(covered_tokens) == 404,
            "total_scenes_10": len(covered_scenes) == 10,
        }
        split_checks = [
            value
            for split in report["splits"].values()
            for value in split["checks"].values()
            if value is not None
        ]
        split_ok = all(split_checks) and all(report["cross_split_checks"].values())

    onnx_exists = all(path.is_file() for path in onnx_paths.values())
    if onnx_exists:
        manifest = json.loads(
            onnx_paths["initial_state_manifest"].read_text(encoding="utf-8")
        )
        report["onnx_bundle_checks"] = {
            "onnx_hash_matches_manifest": (
                manifest.get("onnx_sha256") == sha256(onnx_paths["onnx"])
            ),
            "initial_state_hash_matches_manifest": (
                manifest.get("state_sha256") == sha256(onnx_paths["initial_state"])
            ),
        }
    elif args.require_onnx:
        report["onnx_bundle_checks"] = {"all_three_bundle_files_exist": False}

    onnx_ok = (not args.require_onnx) or (
        onnx_exists and all(report["onnx_bundle_checks"].values())
    )
    report["ok"] = bool(native_assets_ok and split_ok and onnx_ok)

    if args.json_out:
        output = resolve_path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report["json_out"] = str(output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["ok"] else 1)




def planning_bundle_file(root, relative):
    """Resolve a canonical relative artifact path without external symlinks."""
    from pathlib import PurePosixPath
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("bundle artifact path must be relative POSIX text")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
        raise ValueError("noncanonical or escaping bundle artifact path")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*parsed.parts)
    for item in [path] + list(path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("bundle artifacts cannot depend on symlinks")
    path.resolve(strict=True).relative_to(root)
    if not path.is_file():
        raise ValueError("bundle artifact is not a regular file")
    return path


def load_planning_bundle_manifest(root, *, expected_manifest_sha256):
    """Verify the pinned manifest and every model/state/runtime/solver artifact."""
    root = Path(root).resolve(strict=True)
    path = planning_bundle_file(root, "manifest.json")
    if sha256(path) != expected_manifest_sha256:
        raise ValueError("planning bundle manifest SHA256 differs")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "uniad-planning-bundle-v1":
        raise ValueError("unsupported planning bundle format")
    names = {"model", "initial_state", "host", "state_contract", "assets", "collision_optimizer"}
    if set(manifest.get("artifacts", {})) != names:
        raise ValueError("planning bundle artifact roles differ")
    paths = {}
    for name, entry in manifest["artifacts"].items():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("invalid planning bundle artifact entry")
        artifact = planning_bundle_file(root, entry["path"])
        if sha256(artifact) != entry["sha256"]:
            raise ValueError("planning bundle artifact SHA256 differs: " + name)
        paths[name] = artifact
    if len(set(paths.values())) != len(paths):
        raise ValueError("bundle artifact roles must have distinct files")
    return manifest, paths


if __name__ == "__main__":
    main()
