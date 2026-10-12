"""Frozen metric/input adapter; lineage recorded in provenance.json."""
import numpy as np
def frame_feed(info, image, metadata, command, previous):
    timestamp = float(info['timestamp']) / 1e6
    reset = previous.get('scene') != info['scene_token']
    bus = metadata['can_bus_absolute'].copy()
    # Match accepted Native/Stateful official-test legacy behavior, including
    # persistent previous position/yaw when moving to a different scene.
    bus[:3] -= 0 if previous.get('position') is None else np.asarray(previous['position'], np.float32)
    bus[-1] -= 0 if previous.get('angle') is None else previous['angle']
    r, t = metadata['l2g_r'], metadata['l2g_t']
    if image.shape != (1,6,3,928,1600) or image.dtype != np.float32 or not np.isfinite(image).all():
        raise ValueError('real image shape/dtype/finite differs')
    if command not in [0, 1, 2]:
        raise ValueError('invalid navigation command')
    if not reset and timestamp <= previous['timestamp']:
        raise ValueError('non-increasing scene timestamp')
    feed = dict(img=image, can_bus=bus[None], l2g_r_mat=r[None], lidar2img=metadata['lidar2img'],
                img_shape=np.array([[[928,1600]]*6],np.int64), command=np.array([command],np.int64),
                has_prev_bev=np.array(not reset,np.bool_), prev_l2g_r=r if reset else np.asarray(previous['rotation'],np.float32),
                prev_l2g_t=t if reset else np.asarray(previous['translation'],np.float32), l2g_t=t,
                time_delta=np.array(0. if reset else timestamp-previous['timestamp'],np.float32))
    next_meta = dict(scene=info['scene_token'], timestamp=timestamp, position=metadata['can_bus_absolute'][:3].tolist(),
                     angle=float(metadata['can_bus_absolute'][-1]), rotation=r.tolist(), translation=t.tolist())
    return feed, next_meta, reset
