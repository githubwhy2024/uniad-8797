"""Stateful-v1 field order and Host shape/dtype contract, without ML imports.

Track shapes are tails after the dynamic track-count axis. Host-only shapes
are complete. This is a schema, not a state transition or reset policy.
"""
TRACK_STATE_FIELDS = (
    ('query', (512,), 'float32'),
    ('ref_pts', (3,), 'float32'),
    ('pred_boxes', (10,), 'float32'),
    ('obj_idxes', (), 'int64'),
    ('disappear_time', (), 'int64'),
    ('mem_bank', (4, 256), 'float32'),
    ('mem_padding_mask', (4,), 'bool'),
    ('save_period', (), 'float32'),
)
HOST_STATE_FIELDS = (
    ('prev_bev', (40000, 1, 256), 'float32'),
    ('max_obj_id', (), 'int64'),
)
TRACK_STATE_NAMES = tuple(name for name, _, _ in TRACK_STATE_FIELDS)
STATE_DTYPES = {name: dtype for name, _, dtype in TRACK_STATE_FIELDS + HOST_STATE_FIELDS}


def state_shapes(track_count):
    shapes = {name: (track_count,) + tail for name, tail, _ in TRACK_STATE_FIELDS}
    shapes.update({name: shape for name, shape, _ in HOST_STATE_FIELDS})
    return shapes


# Q3 fixed ABI capacities and graph/Host shared recurrent fields.
FRESH = 901
TRACK_SLOTS = 1285
SURVIVOR_CAPACITY = TRACK_SLOTS - FRESH
DECODED_SLOTS = 300
VEHICLE_SLOTS = 96
FIXED_STATE_NAMES = set(TRACK_STATE_NAMES) | {"prev_bev", "max_obj_id", "track_count", "track_valid_mask"}

# Shared SCA diagnostic ABI; query count is the configured 200x200 grid.
SCA_CAPACITY = 10201
SCA_CAMERAS = 6
SCA_BEV_QUERIES = 40000
SCA_OUTPUT_NAMES = ("sca_visible_count_raw", "sca_overflow", "overflow_flags")
