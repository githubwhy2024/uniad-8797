"""Export-only, inference-only tensor state operations for UniAD.

No Instances, mutable global ID counter, or runtime Python loops over tracks.
StatefulStep composes Layer 1 with prepare/transition and the recurrent ABI.
"""
import torch
from torch import nn
from torch.nn import functional as F

from attention import qualcomm_multihead_attention_forward


from state_contract import TRACK_STATE_NAMES


def clone_state_for_input(state):
    """Do not let example inputs alias learned initial-state parameters.

    Otherwise torch.onnx.trace can replace a parameter used to create the
    next 901 fresh queries with the external dynamic query input itself.
    """
    return tuple(value.detach().clone() for value in state)


class TensorRuntimeTracker(nn.Module):
    """Equivalent to RuntimeTrackerBase.update(..., iou_thre=None).

    Caller marks the SDC row -2 before this operation. max_obj_id is the NEXT
    available ID. The returned visible mask is for current-frame decoding;
    QIM keeps all surviving IDs >= 0, including temporarily sleeping tracks.
    """

    def __init__(self, score_thresh=0.4, filter_score_thresh=0.35,
                 miss_tolerance=5):
        super().__init__()
        self.score_thresh = float(score_thresh)
        self.filter_score_thresh = float(filter_score_thresh)
        self.miss_tolerance = int(miss_tolerance)

    def forward(self, scores, obj_idxes, disappear_time, max_obj_id):
        confident = scores >= self.score_thresh
        new = torch.logical_and(obj_idxes == -1, confident)
        sleeping = torch.logical_and(obj_idxes >= 0,
                                     scores < self.filter_score_thresh)
        counts = torch.where(confident, torch.zeros_like(disappear_time),
                             disappear_time) + sleeping.to(torch.int64)
        allocation = torch.cumsum(new.to(torch.int64), dim=0) - 1 + max_obj_id
        ids = torch.where(new, allocation, obj_idxes)
        dead = torch.logical_and(sleeping, counts >= self.miss_tolerance)
        ids = torch.where(dead, torch.full_like(ids, -1), ids)
        visible = torch.logical_and(ids >= 0, scores >= self.filter_score_thresh)
        return ids, counts, max_obj_id + new.to(torch.int64).sum(), visible


def inverse_3x3(matrix):
    """Standard-op inverse, for nonsingular 3x3 pose matrices only.

    Host must validate poses. Do not clamp determinant and silently change a
    singular pose. This also avoids assuming transpose == inverse for inputs
    that have small numerical non-orthogonality.
    """
    a, b, c = matrix[0].unbind()
    d, e, f = matrix[1].unbind()
    g, h, i = matrix[2].unbind()
    cofactors = torch.stack((e*i-f*h, f*g-d*i, d*h-e*g,
                            c*h-b*i, a*i-c*g, b*g-a*h,
                            b*f-c*e, c*d-a*f, a*e-b*d)).reshape(3, 3)
    determinant = a*cofactors[0, 0] + b*cofactors[0, 1] + c*cofactors[0, 2]
    return cofactors.transpose(0, 1) / determinant


class TensorVelocityUpdate(nn.Module):
    """UniADTrack.velo_update, row-vector l2g convention; returns logit xyz.

    Caller applies only xy to active refs, recomputing z from learned query
    position as the official frame preparation does. Does not reorder state.
    """

    def __init__(self, pc_range):
        super().__init__()
        self.register_buffer("lower", torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer("extent", torch.tensor(pc_range[3:], dtype=torch.float32)
                             - self.lower)

    def forward(self, ref_pts, velocity, prev_r, prev_t, curr_r, curr_t, delta):
        xyz = ref_pts.sigmoid() * self.extent + self.lower
        velocity3 = torch.cat((velocity, torch.zeros_like(velocity[:, :1])), dim=-1)
        xyz = xyz + velocity3 * delta.to(torch.float32)
        xyz = (xyz @ prev_r + prev_t - curr_t) @ inverse_3x3(curr_r)
        normalized = ((xyz - self.lower) / self.extent).clamp(0, 1)
        # Same epsilon/clamping as mmdet inverse_sigmoid, including out-of-range.
        return torch.log(normalized.clamp(min=1e-5) /
                         (1 - normalized).clamp(min=1e-5))


def attention_batch_first(module, query, key, value, padding_mask):
    """UniAD B,L,C adapter over the shared Qualcomm explicit MHA core.

    MemoryBank/QIM use batch-first tensors even though their underlying
    torch.nn.MultiheadAttention modules were constructed with batch_first=False.
    Keep that local ABI while sharing exactly one Q/K/V -> MatMul -> Softmax ->
    MatMul -> output-projection implementation with the rest of the model.
    """
    if module.batch_first:
        output, _ = qualcomm_multihead_attention_forward(
            module, query, key, value,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return output

    output, _ = qualcomm_multihead_attention_forward(
        module,
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        key_padding_mask=padding_mask,
        need_weights=False,
    )
    return output.transpose(0, 1)


class TensorMemoryBank(nn.Module):
    """Official eval temporal attention then history save, without empty branches.

    All rows are processed independently. Rows without usable history receive
    one safe zero key; their attention result is discarded with Where. One
    extra batch row makes N=0 portable and is removed from the outputs.
    """

    def __init__(self, memory_bank):
        super().__init__()
        self.bank = memory_bank

    def forward(self, embedding, scores, mem_bank, mem_mask, save_period):
        bank = self.bank
        c, m = bank.save_proj.in_features, bank.max_his_length
        embed = torch.cat((embedding, embedding.new_zeros((1, c))), dim=0)
        memory = torch.cat((mem_bank, mem_bank.new_zeros((1, m, c))), dim=0)
        mask = torch.cat((mem_mask, mem_mask.new_ones((1, m))), dim=0)
        valid = mask[:, -1] == 0
        safe_memory = torch.where(valid[:, None, None], memory, torch.zeros_like(memory))
        fallback_mask = (torch.arange(m, device=mask.device) < m - 1)[None, :].expand_as(mask)
        # ORT 1.19 CPU has no bool-valued Where kernel.
        safe_mask = torch.logical_or(torch.logical_and(valid[:, None], mask),
                                     torch.logical_and((valid == 0)[:, None], fallback_mask))
        attended = attention_batch_first(bank.temporal_attn, embed[:, None],
                                         safe_memory, safe_memory, safe_mask)[:, 0]
        refined = bank.temporal_norm1(embed + attended)
        refined = bank.temporal_norm2(refined + bank.temporal_fc2(F.relu(bank.temporal_fc1(refined))))
        refined = torch.where(valid[:, None], refined, embed)[:-1]
        save = torch.logical_and(save_period == 0, scores > bank.save_thresh)
        period = torch.where(save_period > 0, save_period - 1, save_period)
        period = torch.where(save, torch.full_like(period, bank.save_period), period)
        shifted = torch.cat((mem_bank[:, 1:], bank.save_proj(refined)[:, None]), dim=1)
        shifted_mask = torch.cat((mem_mask[:, 1:], torch.zeros_like(mem_mask[:, :1])), dim=1)
        updated_mask = torch.logical_or(torch.logical_and(save[:, None], shifted_mask),
                                       torch.logical_and((save == 0)[:, None], mem_mask))
        return (refined, torch.where(save[:, None, None], shifted, mem_bank), updated_mask, period)


class TensorQueryInteraction(nn.Module):
    """Select all surviving IDs, update learned query, return active indices.

    Caller concatenates initial tracks with these rows for EVERY state field.
    A dummy key is masked whenever real tracks exist, so it never changes their
    softmax; with zero real tracks it prevents an empty/all-masked attention.
    """

    def __init__(self, query_interaction):
        super().__init__()
        self.qim = query_interaction

    def forward(self, query, embedding, obj_idxes):
        qim = self.qim
        c = qim.self_attn.embed_dim
        active_index = torch.nonzero(obj_idxes >= 0).squeeze(1)
        selected_query = query[active_index]
        selected_embed = embedding[active_index]
        q = torch.cat((selected_query, query.new_zeros((1, 2*c))), dim=0)
        embed = torch.cat((selected_embed, embedding.new_zeros((1, c))), dim=0)
        query_pos, query_feat = q[:, :c], q[:, c:]
        count = torch._shape_as_tensor(active_index)[0]
        index = torch.arange(q.shape[0], device=q.device)
        mask = torch.logical_and(index == count, count > 0)[None, :]
        attention_query = (query_pos + embed)[None]
        tgt2 = attention_batch_first(qim.self_attn, attention_query,
                                    attention_query, embed[None], mask)[0]
        tgt = qim.norm1(embed + tgt2)
        tgt = qim.norm2(tgt + qim.linear2(qim.activation(qim.linear1(tgt))))
        if qim.update_query_pos:  # Configuration, not data-dependent.
            query_pos = qim.norm_pos(query_pos + qim.linear_pos2(qim.activation(qim.linear_pos1(tgt))))
        query_feat = qim.norm_feat(query_feat + qim.linear_feat2(qim.activation(qim.linear_feat1(tgt))))
        return torch.cat((query_pos, query_feat), dim=-1)[:-1], active_index


class TensorTrackStateCycle(nn.Module):
    """Prepare detector inputs and create next-frame state in official order."""

    def __init__(self, model):
        super().__init__()
        self.query_embedding = model.query_embedding
        self.reference_points = model.reference_points
        self.mem_length = model.mem_bank_len
        self.sdc_index = model.num_query
        self.velocity = TensorVelocityUpdate(model.pc_range)
        self.tracker = TensorRuntimeTracker(model.track_base.score_thresh,
                                           model.track_base.filter_score_thresh,
                                           model.track_base.miss_tolerance)
        self.memory = TensorMemoryBank(model.memory_bank)
        self.qim = TensorQueryInteraction(model.query_interact)

    def initial_state(self):
        query = self.query_embedding.weight
        n, double_c = query.shape
        c = double_c // 2
        return (query, self.reference_points(query[:, :c]), query.new_zeros((n, 10)),
                torch.full((n,), -1, device=query.device, dtype=torch.int64),
                torch.zeros(n, device=query.device, dtype=torch.int64),
                query.new_zeros((n, self.mem_length, c)),
                torch.ones((n, self.mem_length), device=query.device, dtype=torch.bool),
                query.new_zeros(n))

    def prepare(self, state, prev_r, prev_t, curr_r, curr_t, delta, has_prev):
        query, refs, boxes, ids, *_ = state
        active = ids >= 0
        active_index = torch.nonzero(active).squeeze(1)
        other_index = torch.nonzero(ids < 0).squeeze(1)
        order = torch.cat((other_index, active_index))
        moved = self.velocity(refs[active_index], boxes[active_index, -2:],
                              prev_r, prev_t, curr_r, curr_t, delta)
        learned = self.reference_points(query[active_index, :query.shape[1] // 2])
        updated = torch.cat((moved[:, :2], learned[:, 2:]), dim=-1)
        updated = torch.where(has_prev.to(torch.bool), updated, refs[active_index])
        ordered = tuple(field[order] for field in state)
        return (ordered[0], torch.cat((refs[other_index], updated)), *ordered[2:])

    def forward(self, query, refs, boxes, ids, counts, memory, mask, period,
                next_id, cls_scores, pred_boxes, last_refs, embedding):
        # refs and prior boxes have already served the detector. New refs/boxes
        # must be persisted, not stale inputs.
        scores = cls_scores.sigmoid().max(dim=-1).values
        ids = torch.where(torch.arange(ids.shape[0], device=ids.device) == self.sdc_index,
                          torch.full_like(ids, -2), ids)
        ids, counts, next_id, visible = self.tracker(scores, ids, counts, next_id)
        refined, memory, mask, period = self.memory(embedding, scores, memory, mask, period)
        active_query, active_index = self.qim(query, refined, ids)
        current = (query, last_refs, pred_boxes, ids, counts, memory, mask, period)
        selected = (active_query,) + tuple(field[active_index] for field in current[1:])
        state = tuple(torch.cat((fresh, old), dim=0)
                      for fresh, old in zip(self.initial_state(), selected))
        return ids, visible, next_id, *state


def decode_visible_tracks(coder, cls_scores, boxes, scores, ids, visible, img_metas=None):
    """Fixed K<=901 ranking, then visibility filtering, for dynamic N>=901.

    Invisible tracks receive score -1 so they cannot displace visible tracks.
    The final mask explicitly removes padding even when coder threshold=0
    disables its own threshold filter. No IoU NMS is supported here.
    """
    ranked_scores = torch.where(visible, scores, torch.full_like(scores, -1))
    decoded = coder.decode_single(cls_scores, boxes, ranked_scores, ids,
                                  with_mask=True, img_metas=img_metas)
    index = decoded["bbox_index"][decoded["mask"]]
    keep = visible[index]
    return (decoded["bboxes"][keep], decoded["scores"][keep],
            decoded["labels"][keep], decoded["obj_idxes"][keep], index[keep])


"""UniAD stateful step v1. Experimental until full-graph/sequence acceptance."""



from frame_core import FrameCore, TrackBBoxesTensor
from frame_core import patch_stateful_encoder
from frame_core import SinOnlyNearestRotation
from frame_core import patch_stateful_spatial
from frame_core import TensorMapDecoder, MAP_OUTPUT_NAMES


INPUT_NAMES = ["img", "can_bus", "l2g_r_mat", "lidar2img", "img_shape", "prev_bev",
               "command", "has_prev_bev", "prev_l2g_r", "prev_l2g_t", "l2g_t", "time_delta",
               "max_obj_id"] + list(TRACK_STATE_NAMES)
OUTPUT_NAMES = [
    "bev_embed", "cls_scores", "bbox_preds", "past_trajectories", "motion_features",
    "occ_logits", "occ_scores", "occ_segmentation", "occ_instances", "planning_raw",
    "track_boxes", "track_scores", "track_labels", "track_ids", "track_query_indices",
    "motion_xy", "motion_log_scores", "sdc_motion_xy",
    "map_class_logits", "map_box_coords", "next_max_obj_id",
] + ["next_" + name for name in TRACK_STATE_NAMES] + MAP_OUTPUT_NAMES


def dynamic_axes():
    axes = {name: {0: "track_count"} for name in TRACK_STATE_NAMES}
    axes.update({"next_" + name: {0: "next_track_count"} for name in TRACK_STATE_NAMES})
    for name in ["cls_scores", "bbox_preds", "past_trajectories"]:
        axes[name] = {2: "track_count"}
    for name in ["track_boxes", "track_scores", "track_labels", "track_ids",
                 "track_query_indices", "motion_xy", "motion_log_scores"]:
        axes[name] = {0: "decoded_track_count"}
    axes["motion_features"] = {2: "vehicle_count"}
    axes["occ_logits"] = axes["occ_scores"] = {1: "vehicle_count"}
    return axes


class StatefulStep(FrameCore):
    def __init__(self, model, occflow_grid_conf=None):
        super().__init__(model, occflow_grid_conf)
        if model.bbox_coder.with_nms or model.bbox_coder.max_num > model.num_query:
            raise ValueError("stateful decode requires no NMS and K <= initialized object count")
        self.cycle = TensorTrackStateCycle(model)
        self.map_decoder = TensorMapDecoder(model.seg_head)
        patch_stateful_encoder(model.pts_bbox_head.transformer.encoder)
        model.pts_bbox_head.transformer._onnx_stateful_rotation = SinOnlyNearestRotation()
        patch_stateful_spatial(model.pts_bbox_head.transformer.encoder,
                               bev_h=model.bev_h, bev_w=model.bev_w)

    def forward(self, img, can_bus, l2g_r_mat, lidar2img, img_shape, prev_bev,
                command, has_prev_bev, prev_l2g_r, prev_l2g_t, l2g_t, time_delta,
                max_obj_id, query, ref_pts, pred_boxes, obj_idxes, disappear_time,
                mem_bank, mem_padding_mask, save_period):
        metadata = [{"can_bus": can_bus[0], "l2g_r_mat": l2g_r_mat[0],
                     "lidar2img": lidar2img[0], "img_shape": img_shape[0],
                     "has_prev_bev": has_prev_bev, "scene_token": "EXPORT",
                     "sample_idx": "EXPORT", "box_type_3d": None, "pc_range": self.model.pc_range}]
        state = self.cycle.prepare((query, ref_pts, pred_boxes, obj_idxes, disappear_time,
                                    mem_bank, mem_padding_mask, save_period),
                                   prev_l2g_r, prev_l2g_t, l2g_r_mat[0], l2g_t, time_delta, has_prev_bev)
        bev, position = self.model.get_bevs(img, metadata, prev_bev=prev_bev)
        detection = self.model.pts_bbox_head.get_detections(
            bev, object_query_embeds=state[0], ref_points=state[1], img_metas=metadata)
        logits = detection["all_cls_scores"][-1, 0]
        boxes = detection["all_bbox_preds"][-1, 0]
        embeddings = detection["query_feats"][-1][0]
        ids, visible, next_id, *next_state = self.cycle(
            *state, max_obj_id, logits, boxes, detection["last_ref_points"][0], embeddings)
        scores = logits.sigmoid().max(dim=-1).values
        decoded = decode_visible_tracks(self.model.bbox_coder, logits, boxes, scores,
                                        ids, visible, metadata)
        track_boxes, track_scores, track_labels, track_ids, indices = decoded
        sdc_index = self.model.num_query
        sdc = self.model.bbox_coder.decode_single(
            logits[sdc_index:sdc_index+1], boxes[sdc_index:sdc_index+1],
            scores[sdc_index:sdc_index+1], ids[sdc_index:sdc_index+1], with_mask=False, img_metas=metadata)
        outs_track = {
            "bev_embed": bev, "bev_pos": position,
            "track_query_embeddings": embeddings[indices],
            "track_bbox_results": [[TrackBBoxesTensor(track_boxes), track_scores, track_labels,
                                    indices, torch.ones_like(track_scores, dtype=torch.bool)]],
            "sdc_embedding": embeddings[sdc_index],
            "sdc_track_bbox_results": [[TrackBBoxesTensor(sdc["bboxes"]), sdc["scores"], sdc["labels"],
                                        sdc["bbox_index"], sdc["mask"]]],
        }
        outputs, trajectories, seg = self._forward_heads(
            img, bev, position, detection, outs_track, command, return_aux=True)
        # get_trajs includes all decoded objects + final SDC (not just vehicles).
        # xy are model-native relative displacements in metres; do not label
        # them absolute global waypoints. Scores are log probabilities.
        trajectory = trajectories[0]["traj"]
        trajectory_score = trajectories[0]["traj_scores"]
        memory, memory_mask, _, map_query, _, map_query_pos, hw_lvl = seg["args_tuple"]
        map_outputs = self.map_decoder(seg["outputs_classes"][-1], seg["outputs_coords"][-1],
                                       memory, memory_mask, map_query, map_query_pos, hw_lvl)
        return (*outputs, *decoded, trajectory[:-1, ..., :2], trajectory_score[:-1],
                trajectory[-1, ..., :2], seg["outputs_classes"][-1],
                seg["outputs_coords"][-1], next_id, *next_state, *map_outputs)
