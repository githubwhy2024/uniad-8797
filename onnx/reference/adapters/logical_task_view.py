"""Frozen metric/input adapter; lineage recorded in provenance.json."""
import numpy as np
def logical_task_view(result, fixed):
    if not fixed:
        return result
    out = dict(result)
    current = result["cls_scores"].shape[2]
    if current != 1285:
        raise ValueError("fixed detector axis changed")
    valid = result["decoded_valid_mask"]
    vehicle = result["vehicle_valid_mask"]
    for name in (
        "track_boxes",
        "track_scores",
        "track_labels",
        "track_ids",
        "track_query_indices",
        "motion_xy",
        "motion_log_scores",
    ):
        out[name] = out[name][valid]
    out["motion_features"] = out["motion_features"][:, :, vehicle]
    for name in ("occ_logits", "occ_scores"):
        out[name] = out[name][:, vehicle]
    # Detector valid input count is supplied by caller below.
    return out
