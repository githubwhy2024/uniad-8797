"""Prepare one complete temporal-info scene for the existing host CLI.

Only trusted local pickle is accepted with explicit opt-in. Commands must be
provided from the project's trajectory-label pipeline, never guessed here.
Default is preflight only; --out writes a new raw-BGR frame package.
This is data preparation, NOT model parity or real-data acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import traceback

import numpy as np
from pyquaternion import Quaternion


CAMERAS = {"CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_npz(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"expected finite shape {shape}")
    return result


def rotation(value):
    q = array(value, (4,))
    if not np.isclose(np.linalg.norm(q), 1., atol=1e-4):
        raise ValueError("expected unit quaternion wxyz")
    return Quaternion(q)


def frame_metadata(info):
    """Match local NuScenesE2EDataset.get_data_info, including row-vector pose."""
    cams = info["cams"]
    if set(cams) != CAMERAS:
        raise ValueError("expected six named nuScenes cameras")
    ego = rotation(info["ego2global_rotation"])
    lidar = rotation(info["lidar2ego_rotation"])
    ego_t = array(info["ego2global_translation"], (3,))
    lidar_t = array(info["lidar2ego_translation"], (3,))
    bus = array(info["can_bus"], (18,)).copy()
    bus[:3], bus[3:7] = ego_t, ego.elements
    forward = ego.rotation_matrix @ np.array([1., 0., 0.])
    yaw = np.degrees(np.arctan2(forward[1], forward[0]))
    if yaw < 0:
        yaw += 360
    bus[-2], bus[-1] = np.radians(yaw), yaw
    projections = []
    for cam in cams.values():  # Preserve serialized training/calibration order.
        r = array(cam["sensor2lidar_rotation"], (3, 3))
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(r), 1., atol=1e-4):
            raise ValueError("invalid camera rotation")
        inverse = np.linalg.inv(r)
        translation = array(cam["sensor2lidar_translation"], (3,)) @ inverse.T
        transform = np.eye(4)
        transform[:3, :3], transform[3, :3] = inverse.T, -translation
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = array(cam["cam_intrinsic"], (3, 3))
        projections.append(intrinsic @ transform.T)
    return dict(can_bus_absolute=bus.astype(np.float32),
                l2g_r=(lidar.rotation_matrix.T @ ego.rotation_matrix.T).astype(np.float32),
                l2g_t=(lidar_t @ ego.rotation_matrix.T + ego_t).astype(np.float32),
                lidar2img=np.asarray(projections, dtype=np.float32)[None])


def select_scene(infos, scene, commands):
    frames = sorted([i for i in infos if i["scene_token"] == scene], key=lambda i: i["timestamp"])
    if len(frames) < 2:
        raise ValueError("need a complete scene with at least two keyframes")
    if frames[0]["prev"] or frames[-1]["next"]:
        raise ValueError("scene is truncated; first prev and last next must be empty")
    tokens = [i["token"] for i in frames]
    if len(tokens) != len(set(tokens)):
        raise ValueError("duplicate sample token")
    order = list(frames[0]["cams"])
    for index, info in enumerate(frames):
        if list(info["cams"]) != order:
            raise ValueError("camera order changed inside scene")
        time = float(info["timestamp"])
        if not np.isfinite(time) or time <= 0:
            raise ValueError("timestamp must be positive finite microseconds")
        command = commands.get(info["token"])
        if type(command) is not int or command not in (0, 1, 2):
            raise ValueError(f"missing or invalid explicit command for {info['token']}")
        if index and (time <= float(frames[index-1]["timestamp"]) or
                      info["prev"] != frames[index-1]["token"] or frames[index-1]["next"] != info["token"]):
            raise ValueError("broken sample chain or non-increasing timestamps")
        frame_metadata(info)
    return frames


def image_path(raw, root, prefix):
    path = PurePosixPath(raw)
    if prefix:
        try:
            path = path.relative_to(PurePosixPath(prefix))
        except ValueError as error:
            raise ValueError("image path does not match explicit stored prefix") from error
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image path must be relative to data-root after explicit prefix removal")
    resolved = (root / str(path)).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"image missing or outside data-root: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infos", required=True)
    parser.add_argument("--trusted-pickle", action="store_true", help="acknowledge pickle can execute code; only your trusted infos")
    parser.add_argument("--scene-token", required=True)
    parser.add_argument("--commands-json", required=True, help="sample token -> integer command from project trajectory labels")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stored-path-prefix", default="", help="explicit prefix to remove from stored image paths")
    parser.add_argument("--out", help="new package directory; omit for read-only metadata/path preflight")
    args = parser.parse_args()
    if not args.trusted_pickle:
        parser.error("refusing pickle without --trusted-pickle")
    output = Path(args.out).resolve() if args.out else None
    if output is not None and output.exists():
        parser.error("refusing to overwrite an existing package")
    with open(args.infos, "rb") as stream:
        document = pickle.load(stream)
    commands = json.loads(Path(args.commands_json).read_text())
    frames = select_scene(document["infos"], args.scene_token, commands)
    paths = [[image_path(cam["data_path"], Path(args.data_root), args.stored_path_prefix)
              for cam in frame["cams"].values()] for frame in frames]
    record = dict(stage="preflight_passed", frame_count=len(frames), camera_order=list(frames[0]["cams"]),
                  infos_sha256=sha256(args.infos), commands_sha256=sha256(args.commands_json),
                  preparer_sha256=sha256(__file__), scene_token=args.scene_token,
                  timestamp_unit="seconds_in_manifest_converted_from_infos_microseconds",
                  command_provenance="user_supplied_project_trajectory_labels_not_computed_or_verified",
                  model_parity_validated=False, real_data_acceptance_complete=False,
                  image_decoding_validated=False)
    if output is None:
        print(json.dumps(record, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest, sources = [], []
    try:
        import cv2
        save_json(output / "status.json", record)
        for index, (info, cameras) in enumerate(zip(frames, paths)):
            images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in cameras]
            if any(im is None or im.shape != (900, 1600, 3) or im.dtype != np.uint8 for im in images):
                raise ValueError("expected six native 900x1600 uint8 BGR images; no resizing allowed")
            filename = f"frame-{index:06d}.npz"
            save_npz(output / filename, dict(images_bgr=np.stack(images), **frame_metadata(info)))
            manifest.append(dict(scene_token=info["scene_token"], timestamp=float(info["timestamp"])/1e6,
                                 command=commands[info["token"]], frame_npz=filename))
            sources.append(dict(token=info["token"], images=[dict(path=str(p), sha256=sha256(p)) for p in cameras],
                                frame_npz=filename, frame_sha256=sha256(output / filename)))
        save_json(output / "frames.json", dict(format="uniad-frames-v1", frames=manifest))
        save_json(output / "sources.json", dict(frames=sources))
        record.update(stage="prepared", image_decoding_validated=True, manifest_sha256=sha256(output / "frames.json"))
        save_json(output / "status.json", record)
    except BaseException:
        record.update(stage="failed_or_interrupted", error=traceback.format_exc())
        save_json(output / "status.json", record)
        raise
    print(json.dumps(record, indent=2))




if __name__ == "__main__":
    main()
