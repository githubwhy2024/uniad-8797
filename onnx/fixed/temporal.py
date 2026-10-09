"""Export-only, inference-only tensor state operations for UniAD.

No Instances, mutable global ID counter, or runtime Python loops over tracks.
StatefulStep composes Layer 1 with prepare/transition and the recurrent ABI.
"""
import torch
import types
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


class FixedSlotRuntimeTracker(TensorRuntimeTracker):
    """Q3.3 candidate: fixed-row tracker with explicit valid slots.

    This is not installed in StatefulStep yet. The candidate preserves the
    v1 tracker result on valid logical rows and never allocates an ID to
    padding. Survivor overflow is reported from the full untruncated mask;
    Host rejection and QIM packing belong to later Q3 stages.
    """

    def __init__(self, *, survivor_capacity, score_thresh=0.4,
                 filter_score_thresh=0.35, miss_tolerance=5):
        super().__init__(score_thresh, filter_score_thresh, miss_tolerance)
        if survivor_capacity < 0:
            raise ValueError("survivor_capacity must be nonnegative")
        self.survivor_capacity = int(survivor_capacity)

    def forward(self, scores, obj_idxes, disappear_time, max_obj_id, valid_mask):
        valid = valid_mask.to(torch.bool)
        confident = scores >= self.score_thresh
        new = valid & (obj_idxes == -1) & confident
        sleeping = valid & (obj_idxes >= 0) & (scores < self.filter_score_thresh)
        counts = torch.where(confident, torch.zeros_like(disappear_time),
                             disappear_time) + sleeping.to(torch.int64)
        counts = torch.where(valid, counts, torch.zeros_like(counts))
        allocation = torch.cumsum(new.to(torch.int64), dim=0) - 1 + max_obj_id
        ids = torch.where(new, allocation, obj_idxes)
        dead = sleeping & (counts >= self.miss_tolerance)
        ids = torch.where(dead, torch.full_like(ids, -1), ids)
        ids = torch.where(valid, ids, torch.full_like(ids, -3))
        visible = valid & (ids >= 0) & (scores >= self.filter_score_thresh)
        survivor_count = (valid & (ids >= 0)).to(torch.int64).sum()
        overflow = survivor_count > self.survivor_capacity
        return ids, counts, max_obj_id + new.to(torch.int64).sum(), visible, overflow


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


class FixedSlotTrackPrepare(nn.Module):
    """Q3.4 candidate: prepare a valid-prefix fixed state without NonZero.

    Under the v2 Host contract the valid logical order is already 901 fresh
    rows followed by surviving IDs. Only those survivors receive the legacy
    velocity/reference update. Padding is left untouched and must be masked
    from the detector by the later integrated ABI; this class is not installed.
    """

    def __init__(self, reference_points, pc_range):
        super().__init__()
        self.reference_points = reference_points
        self.velocity = TensorVelocityUpdate(pc_range)

    def forward(self, state, valid_mask, prev_r, prev_t, curr_r, curr_t, delta, has_prev):
        query, refs, boxes, ids, *_ = state
        active = valid_mask.to(torch.bool) & (ids >= 0)
        moved = self.velocity(refs, boxes[:, -2:], prev_r, prev_t, curr_r, curr_t, delta)
        learned = self.reference_points(query[:, :query.shape[1] // 2])
        updated = torch.cat((moved[:, :2], learned[:, 2:]), dim=-1)
        updated = torch.where(has_prev.to(torch.bool), updated, refs)
        prepared_refs = torch.where(active[:, None], updated, refs)
        return (query, prepared_refs, *state[2:])


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


class FixedSlotMemoryBank(TensorMemoryBank):
    """Q3.4 candidate: fixed-row MemoryBank with invalid slots isolated.

    The v1 MemoryBank already computes each track row independently and uses
    a fixed four-slot history. Running it over every fixed slot preserves all
    valid rows; padding outputs are restored to their input values. This class
    is not installed in StatefulStep yet.
    """

    def forward(self, embedding, scores, mem_bank, mem_mask, save_period, valid_mask):
        refined, updated_bank, updated_mask, updated_period = super().forward(
            embedding, scores, mem_bank, mem_mask, save_period
        )
        valid = valid_mask.to(torch.bool)
        return (
            torch.where(valid[:, None], refined, embedding),
            torch.where(valid[:, None, None], updated_bank, mem_bank),
            torch.where(valid[:, None], updated_mask, mem_mask),
            torch.where(valid, updated_period, save_period),
        )


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


class FixedSlotQueryInteraction(TensorQueryInteraction):
    """Q3.4 candidate: stable fixed-capacity survivor QIM, not installed.

    Distinct descending scores based on original row index let fixed TopK
    select every valid survivor in v1 order when count <= capacity. Masked
    padding and the safe dummy never enter valid attention. On overflow the
    count/flag are returned before packing; Host must reject that frame.
    """

    def __init__(self, query_interaction, *, survivor_capacity):
        super().__init__(query_interaction)
        if survivor_capacity < 1:
            raise ValueError("survivor_capacity must be positive")
        self.survivor_capacity = int(survivor_capacity)

    def forward(self, query, embedding, obj_idxes, valid_mask):
        qim = self.qim
        c = qim.self_attn.embed_dim
        active = valid_mask.to(torch.bool) & (obj_idxes >= 0)
        survivor_count = active.to(torch.int64).sum()
        overflow = survivor_count > self.survivor_capacity
        row = torch.arange(query.shape[0], device=query.device)
        order_score = torch.where(
            active, (query.shape[0] - row).to(query.dtype),
            torch.full_like(row, -1).to(query.dtype),
        )
        _, chosen = torch.topk(order_score, self.survivor_capacity, largest=True)
        slot = torch.arange(self.survivor_capacity, device=query.device)
        selected_valid = slot < survivor_count
        safe_index = torch.where(selected_valid, chosen, torch.zeros_like(chosen))
        selected_index = torch.where(selected_valid, chosen, torch.full_like(chosen, -1))
        selected_query = torch.where(selected_valid[:, None], query[safe_index],
                                     torch.zeros_like(query[safe_index]))
        selected_embed = torch.where(selected_valid[:, None], embedding[safe_index],
                                     torch.zeros_like(embedding[safe_index]))
        q = torch.cat((selected_query, query.new_zeros((1, 2*c))), dim=0)
        embed = torch.cat((selected_embed, embedding.new_zeros((1, c))), dim=0)
        query_pos, query_feat = q[:, :c], q[:, c:]
        padding_mask = torch.cat((~selected_valid, (survivor_count > 0).reshape(1)))[None, :]
        attention_query = (query_pos + embed)[None]
        tgt2 = attention_batch_first(qim.self_attn, attention_query,
                                    attention_query, embed[None], padding_mask)[0]
        tgt = qim.norm1(embed + tgt2)
        tgt = qim.norm2(tgt + qim.linear2(qim.activation(qim.linear1(tgt))))
        if qim.update_query_pos:
            query_pos = qim.norm_pos(query_pos + qim.linear_pos2(qim.activation(qim.linear_pos1(tgt))))
        query_feat = qim.norm_feat(query_feat + qim.linear_feat2(qim.activation(qim.linear_feat1(tgt))))
        updated = torch.cat((query_pos, query_feat), dim=-1)[:-1]
        updated = torch.where(selected_valid[:, None], updated, torch.zeros_like(updated))
        return updated, selected_index, survivor_count, selected_valid, overflow


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


class FixedSlotTrackStateCycle(TensorTrackStateCycle):
    """Q3.4 uninstalled fixed-slot candidate for the post-detector state cycle.

    Component modules are kept separate from the v1 path. For an accepted
    frame, all S survivors are packed after the 901 learned fresh rows in v1
    order. An overflow frame may contain a prefix for safe fixed computation,
    but its flag requires Host rejection before any state commit.
    """

    def __init__(self, model, *, survivor_capacity):
        super().__init__(model)
        self.survivor_capacity = int(survivor_capacity)
        self.fixed_prepare = FixedSlotTrackPrepare(model.reference_points, model.pc_range)
        self.fixed_tracker = FixedSlotRuntimeTracker(
            survivor_capacity=survivor_capacity,
            score_thresh=model.track_base.score_thresh,
            filter_score_thresh=model.track_base.filter_score_thresh,
            miss_tolerance=model.track_base.miss_tolerance,
        )
        self.fixed_memory = FixedSlotMemoryBank(model.memory_bank)
        self.fixed_qim = FixedSlotQueryInteraction(
            model.query_interact, survivor_capacity=survivor_capacity
        )

    @staticmethod
    def _pack_field(field, safe_index, valid, fill):
        chosen = field[safe_index]
        shape = (valid.shape[0],) + (1,) * (chosen.dim() - 1)
        return torch.where(valid.reshape(shape), chosen, torch.full_like(chosen, fill))

    def prepare_fixed(self, state, valid_mask, prev_r, prev_t, curr_r, curr_t, delta, has_prev):
        return self.fixed_prepare(state, valid_mask, prev_r, prev_t,
                                  curr_r, curr_t, delta, has_prev)

    def forward(self, query, refs, boxes, ids, counts, memory, mask, period,
                valid_mask, next_id, cls_scores, pred_boxes, last_refs, embedding):
        scores = cls_scores.sigmoid().max(dim=-1).values
        ids = torch.where(torch.arange(ids.shape[0], device=ids.device) == self.sdc_index,
                          torch.full_like(ids, -2), ids)
        ids, counts, next_id, visible, tracker_overflow = self.fixed_tracker(
            scores, ids, counts, next_id, valid_mask
        )
        refined, memory, mask, period = self.fixed_memory(
            embedding, scores, memory, mask, period, valid_mask
        )
        active_query, selected_index, survivor_count, selected_valid, qim_overflow = self.fixed_qim(
            query, refined, ids, valid_mask
        )
        safe_index = torch.where(selected_valid, selected_index,
                                 torch.zeros_like(selected_index))
        current = (query, last_refs, pred_boxes, ids, counts, memory, mask, period)
        padding = (0, 0, -3, 0, 0, True, 0)
        selected = (active_query,) + tuple(
            self._pack_field(field, safe_index, selected_valid, fill)
            for field, fill in zip(current[1:], padding)
        )
        next_state = tuple(torch.cat((fresh, old), dim=0)
                           for fresh, old in zip(self.initial_state(), selected))
        next_valid = torch.cat((torch.ones(901, device=query.device, dtype=torch.bool),
                                selected_valid))
        next_count = 901 + torch.minimum(
            survivor_count, torch.tensor(self.survivor_capacity, device=query.device)
        )
        overflow = tracker_overflow | qim_overflow
        return ids, visible, next_id, *next_state, next_count, next_valid, survivor_count, overflow


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


def decode_visible_tracks_fixed(coder, cls_scores, boxes, scores, ids, visible):
    """Keep the coder's 300 TopK rows; pack spatially valid tracks first."""
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    capacity = 300
    if coder.with_nms or coder.max_num != capacity or cls_scores.shape[0] < capacity:
        raise ValueError("fixed decoder requires no NMS and coder max_num=300")
    ranked = torch.where(visible, scores, torch.full_like(scores, -1))
    _, chosen = ranked.topk(capacity)
    labels = cls_scores.sigmoid().max(dim=-1).indices[chosen] % coder.num_classes
    decoded_boxes = denormalize_bbox(boxes[chosen], coder.pc_range)
    chosen_scores = ranked[chosen]
    limits = torch.as_tensor(coder.post_center_range, device=boxes.device,
                             dtype=decoded_boxes.dtype)
    valid = ((decoded_boxes[:, :3] >= limits[:3]).all(dim=1)
             & (decoded_boxes[:, :3] <= limits[3:]).all(dim=1)
             & visible[chosen])
    if coder.score_threshold:
        valid = valid & (chosen_scores > coder.score_threshold)
    raw_count = valid.to(torch.int64).sum()
    row = torch.arange(capacity, device=boxes.device)
    order = torch.where(valid, capacity-row, torch.full_like(row, -1))
    _, packed = torch.topk(order, capacity)
    packed_valid = row < raw_count
    def select(value, fill):
        selected = value[packed]
        shape = (capacity,) + (1,) * (selected.dim()-1)
        return torch.where(packed_valid.reshape(shape), selected,
                           torch.full_like(selected, fill))
    return ((select(decoded_boxes, 0), select(chosen_scores, 0),
             select(labels, 0), select(ids[chosen], -1), select(chosen, -1)),
            raw_count, packed_valid)


def decode_sdc_fixed(coder, cls_scores, boxes, scores, ids):
    """Decode the one fixed SDC row without the coder's boolean selection."""
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

    if cls_scores.shape[0] != 1 or boxes.shape[0] != 1 or scores.shape != (1,):
        raise ValueError("SDC decode requires exactly one fixed row")
    labels = cls_scores.sigmoid().max(dim=-1).indices % coder.num_classes
    index = torch.zeros((1,), dtype=torch.int64, device=boxes.device)
    return {
        "bboxes": denormalize_bbox(boxes, coder.pc_range),
        "scores": scores,
        "labels": labels,
        "bbox_index": index,
        "mask": torch.ones_like(index, dtype=torch.bool),
    }


def _fixed_track_agent_forward(self, query, key, query_pos=None, key_pos=None):
    """Mask invalid decoded keys in motion's cross-agent attention."""
    batch, agents, modes, channels = query.shape
    if query_pos is not None:
        query = query + query_pos
    if key_pos is not None:
        key = key + key_pos
    valid = self._motion_agent_valid_mask
    if valid.shape != (agents,):
        raise ValueError("motion agent mask differs from fixed agent axis")
    memory = key.expand(batch*agents, -1, -1)
    query = query.flatten(start_dim=0, end_dim=1)
    mask = (~valid)[None, :].expand(batch*agents, -1)
    return self.interaction_transformer(
        query, memory, memory_key_padding_mask=mask
    ).view(batch, agents, modes, channels)


def _fixed_motion_forward_test(self, bev_embed, outs_track, outs_seg):
    """Run 300 decoded agents plus SDC, then pack 96 vehicle slots."""
    decoded_valid = outs_track['decoded_valid_mask']
    if decoded_valid.shape != (300,):
        raise ValueError("fixed motion requires 300 decoded slots")
    query = outs_track['track_query_embeddings'][None, None]
    query = torch.cat((query, outs_track['sdc_embedding'][None, None, None]), dim=2)
    boxes = outs_track['track_bbox_results']
    sdc = outs_track['sdc_track_bbox_results']
    for index in range(5):
        if index == 0:
            boxes[0][0].tensor = torch.cat((boxes[0][0].tensor, sdc[0][0].tensor))
        else:
            boxes[0][index] = torch.cat((boxes[0][index], sdc[0][index]))
    labels = boxes[0][2]
    labels[-1] = 0
    agent_valid = torch.cat((decoded_valid, decoded_valid.new_ones(1)))
    for layer in self.motionformer.track_agent_interaction_layers:
        layer._motion_agent_valid_mask = agent_valid
    _, _, _, lane_query, _, lane_query_pos, _ = outs_seg['args_tuple']
    motion = self(bev_embed, query, lane_query, lane_query_pos, boxes)
    trajectories = self.get_trajs(motion, boxes)
    motion['track_scores'] = boxes[0][1][None]
    motion['sdc_traj_query'] = motion['traj_query'][:, :, -1]
    motion['sdc_track_query'] = motion['track_query'][:, -1]
    motion['sdc_track_query_pos'] = motion['track_query_pos'][:, -1]
    vehicle = decoded_valid.clone()
    allowed = torch.zeros_like(vehicle)
    for category in self.vehicle_id_list:
        allowed |= labels[:300] == category
    vehicle &= allowed
    raw_count = vehicle.to(torch.int64).sum()
    capacity = 96
    row = torch.arange(300, device=vehicle.device)
    order = torch.where(vehicle, 300-row, torch.full_like(row, -1))
    _, selected = torch.topk(order, capacity)
    valid = torch.arange(capacity, device=vehicle.device) < raw_count
    def select(value, axis):
        picked = value.index_select(axis, selected)
        shape = [1] * picked.dim()
        shape[axis] = capacity
        return torch.where(valid.reshape(shape), picked, torch.zeros_like(picked))
    motion['traj_query'] = select(motion['traj_query'][:, :, :300], 2)
    motion['track_query'] = select(motion['track_query'][:, :300], 1)
    motion['track_query_pos'] = select(motion['track_query_pos'][:, :300], 1)
    motion['track_scores'] = select(motion['track_scores'][:, :300], 1)
    motion['vehicle_count_raw'] = raw_count
    motion['vehicle_count'] = torch.minimum(raw_count, torch.tensor(capacity,
                                           device=raw_count.device))
    motion['vehicle_valid_mask'] = valid
    motion['vehicle_overflow'] = raw_count > capacity
    self._motion_vehicle_count = motion['vehicle_count']
    self._motion_vehicle_valid_mask = valid
    self._motion_vehicle_count_raw = raw_count
    self._motion_vehicle_overflow = motion['vehicle_overflow']
    return trajectories, motion


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


class FixedRecurrentStatefulStep(StatefulStep):
    """CPU PT candidate with Q3 fixed recurrent and downstream agent axes.

    Track, decoded, motion-agent and vehicle tensors have fixed capacities.
    The Host converts count/mask outputs to logical task records for evaluation.
    This is not yet a fully audited ONNX or 8797 production ABI. The caller
    rejects overflow before committing the returned state.
    """

    track_slots = 1285
    survivor_capacity = 384

    def __init__(self, model, occflow_grid_conf=None):
        super().__init__(model, occflow_grid_conf)
        if model.num_query + 1 != 901:
            raise ValueError("fixed recurrent candidate requires 901 learned fresh rows")
        self.cycle = FixedSlotTrackStateCycle(
            model, survivor_capacity=self.survivor_capacity
        )
        for layer in model.motion_head.motionformer.track_agent_interaction_layers:
            layer.forward = types.MethodType(_fixed_track_agent_forward, layer)
        model.motion_head.forward_test = types.MethodType(
            _fixed_motion_forward_test, model.motion_head
        )

    def initial_fixed_state(self):
        """Return eight padded state tensors, packed count and prefix mask."""
        fresh = self.cycle.initial_state()
        if fresh[0].shape[0] != 901:
            raise ValueError("unexpected learned initial-state length")
        fills = (0, 0, 0, -3, 0, 0, True, 0)
        padded = tuple(
            torch.cat((field, torch.full(
                (self.survivor_capacity, *field.shape[1:]), fill,
                dtype=field.dtype, device=field.device,
            )), dim=0)
            for field, fill in zip(fresh, fills)
        )
        valid = torch.arange(self.track_slots, device=fresh[0].device) < 901
        return (*padded, torch.tensor(901, dtype=torch.int64,
                                      device=fresh[0].device), valid)

    def forward(self, img, can_bus, l2g_r_mat, lidar2img, img_shape, prev_bev,
                command, has_prev_bev, prev_l2g_r, prev_l2g_t, l2g_t, time_delta,
                max_obj_id, query, ref_pts, pred_boxes, obj_idxes, disappear_time,
                mem_bank, mem_padding_mask, save_period, track_count,
                track_valid_mask):
        if query.shape[0] != self.track_slots or track_valid_mask.shape != (self.track_slots,):
            raise ValueError("fixed recurrent input must contain 1285 track slots")
        expected = torch.arange(self.track_slots, device=query.device) < track_count
        if not torch.equal(track_valid_mask.to(torch.bool), expected):
            raise ValueError("track_valid_mask must match packed track_count")
        metadata = [{"can_bus": can_bus[0], "l2g_r_mat": l2g_r_mat[0],
                     "lidar2img": lidar2img[0], "img_shape": img_shape[0],
                     "has_prev_bev": has_prev_bev, "scene_token": "EXPORT",
                     "sample_idx": "EXPORT", "box_type_3d": None,
                     "pc_range": self.model.pc_range}]
        state = self.cycle.prepare_fixed(
            (query, ref_pts, pred_boxes, obj_idxes, disappear_time,
             mem_bank, mem_padding_mask, save_period), track_valid_mask,
            prev_l2g_r, prev_l2g_t, l2g_r_mat[0], l2g_t, time_delta,
            has_prev_bev,
        )
        bev, position = self.model.get_bevs(img, metadata, prev_bev=prev_bev)
        detection = self.model.pts_bbox_head.get_detections(
            bev, object_query_embeds=state[0], ref_points=state[1],
            img_metas=metadata, query_key_padding_mask=(~track_valid_mask)[None],
        )
        logits = detection["all_cls_scores"][-1, 0]
        boxes = detection["all_bbox_preds"][-1, 0]
        embeddings = detection["query_feats"][-1][0]
        (ids, visible, next_id, *cycle_outputs) = self.cycle(
            *state, track_valid_mask, max_obj_id, logits, boxes,
            detection["last_ref_points"][0], embeddings,
        )
        next_state = cycle_outputs[:8]
        next_count, next_valid, raw_survivor, overflow = cycle_outputs[8:]
        scores = logits.sigmoid().max(dim=-1).values
        decoded, decoded_count, decoded_valid = decode_visible_tracks_fixed(
            self.model.bbox_coder, logits, boxes, scores, ids, visible,
        )
        track_boxes, track_scores, track_labels, track_ids, indices = decoded
        sdc_index = self.model.num_query
        sdc = decode_sdc_fixed(
            self.model.bbox_coder,
            logits[sdc_index:sdc_index+1], boxes[sdc_index:sdc_index+1],
            scores[sdc_index:sdc_index+1], ids[sdc_index:sdc_index+1],
        )
        outs_track = {
            "bev_embed": bev, "bev_pos": position,
            "track_query_embeddings": embeddings[indices],
            "decoded_valid_mask": decoded_valid,
            "track_bbox_results": [[TrackBBoxesTensor(track_boxes), track_scores,
                                    track_labels, indices,
                                    decoded_valid]],
            "sdc_embedding": embeddings[sdc_index],
            "sdc_track_bbox_results": [[TrackBBoxesTensor(sdc["bboxes"]),
                                         sdc["scores"], sdc["labels"],
                                         sdc["bbox_index"], sdc["mask"]]],
        }
        outputs, trajectories, seg = self._forward_heads(
            img, bev, position, detection, outs_track, command, return_aux=True,
        )
        trajectory = trajectories[0]["traj"]
        trajectory_score = trajectories[0]["traj_scores"]
        memory, memory_mask, _, map_query, _, map_query_pos, hw_lvl = seg["args_tuple"]
        map_outputs = self.map_decoder(
            seg["outputs_classes"][-1], seg["outputs_coords"][-1], memory,
            memory_mask, map_query, map_query_pos, hw_lvl,
        )
        return (*outputs, *decoded, trajectory[:-1, ..., :2],
                trajectory_score[:-1], trajectory[-1, ..., :2],
                seg["outputs_classes"][-1], seg["outputs_coords"][-1],
                next_id, *next_state, *map_outputs, next_count, next_valid,
                raw_survivor, overflow, decoded_count, decoded_valid,
                self.model.motion_head._motion_vehicle_count,
                self.model.motion_head._motion_vehicle_valid_mask,
                self.model.motion_head._motion_vehicle_count_raw,
                self.model.motion_head._motion_vehicle_overflow)
