"""Layer 1: single-frame neural computation and Qualcomm graph adapters.

Layer 2 composes these tensor operations; this module contains no session or
CLI ownership. Qualcomm-derived sections retain their source/license notices
inline while sharing one coherent import and section structure.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class TrackBBoxesTensor:
    """模拟 LiDARInstance3DBoxes 的最小接口, 仅提供 motion_head 需要的属性。

    motion_head 的 _extract_tracking_centers 和 anchor_coordinate_transform 只访问:
      - bboxes.gravity_center  (N, 3)
      - bboxes.yaw              (N,)
    motion_head.forward_test 会 cat 操作: track_boxes[0][0].tensor = cat(...)
    所以 gravity_center/yaw 必须从 tensor 动态计算, 不能缓存。

    gravity_center = [cx, cy, cz + h/2]  (LiDARInstance3DBoxes 的定义)
    yaw = rot
    """

    def __init__(self, bbox_tensor):
        """bbox_tensor: (N, 9) [cx, cy, cz, l, w, h, rot, vx, vy]"""
        self.tensor = bbox_tensor

    @property
    def gravity_center(self):
        t = self.tensor
        return torch.stack([t[..., 0], t[..., 1], t[..., 2] + t[..., 5] / 2.0], dim=-1)

    @property
    def yaw(self):
        return self.tensor[..., 6]

    def to(self, device):
        return TrackBBoxesTensor(self.tensor.to(device))


class FrameCore(nn.Module):
    """Single-frame neural heads shared with Layer 2 orchestration."""

    def __init__(self, model, occflow_grid_conf=None):
        super().__init__()
        self.model = model
        self.query_embedding = model.query_embedding
        self.reference_points = model.reference_points
        self.occflow_grid_conf = occflow_grid_conf or {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
        }


    def _forward_heads(self, img, bev_embed, bev_pos, det_output, outs_track,
                       command, return_aux=False):
        """Shared downstream chain; legacy default retains the 10-output ABI."""

        # 3. seg_head
        # 只需要 args_tuple 供 motion head 使用; get_bboxes 是评估/可视化后处理,
        # 其结果不被任何下游消费, 不调用以缩小 trace 范围。
        seg_pred = self.model.seg_head(bev_embed)
        outs_seg = {"args_tuple": seg_pred["args_tuple"]}

        # 4. motion_head
        traj_results, outs_motion = self.model.motion_head.forward_test(
            bev_embed, outs_track, outs_seg
        )
        # bev_pos 来自 track 输出, planning_head 需要 (见 uniad_e2e.py:304)
        outs_motion["bev_pos"] = bev_pos

        # 5. occ_head
        # occ_head.forward_test 强制调 get_occ_labels, 导出时传 dummy gt (全零)
        # n_future+1=5 帧, grid 200x200 (从 occflow_grid_conf 算)
        occ_n_future = self.model.occ_head.n_future  # 4
        occ_grid_conf = self.occflow_grid_conf
        occ_grid_h = int((occ_grid_conf['ybound'][1] - occ_grid_conf['ybound'][0]) /
                         occ_grid_conf['ybound'][2])  # 200
        occ_grid_w = int((occ_grid_conf['xbound'][1] - occ_grid_conf['xbound'][0]) /
                         occ_grid_conf['xbound'][2])  # 200
        dummy_gt_seg = [torch.zeros(occ_n_future + 1, occ_grid_h, occ_grid_w, dtype=torch.long, device=img.device)]
        dummy_gt_ins = [torch.zeros(occ_n_future + 1, occ_grid_h, occ_grid_w, dtype=torch.long, device=img.device)]
        # gt_img_is_valid: get_occ_labels 做 [0] 后需 2D (1, rf+nf)
        occ_rf = self.model.occ_head.receptive_field  # 3
        dummy_gt_valid = [torch.ones(1, occ_rf + occ_n_future, dtype=torch.long, device=img.device)]
        # ``OccHead.forward_test`` 的 no_query 是 Python 分支，legacy trace
        # 会按 dummy 的 agent 数将其固化。仅在 N=0 时动态补一个零分 dummy
        # query，使神经网络分支始终可执行；N>0 时补集为空，数值完全不变。
        # dummy 的 track score 为 0，因此 N=0 时 seg_out 仍严格为全零。
        real_query_count = torch._shape_as_tensor(outs_motion["track_query"])[1]
        needs_dummy = (real_query_count == 0).reshape(1)
        occ_motion = dict(outs_motion)
        dummy_track = torch.zeros_like(
            outs_motion["sdc_track_query"][:, None, :]
        )
        dummy_track_pos = torch.zeros_like(
            outs_motion["sdc_track_query_pos"][:, None, :]
        )
        dummy_traj = torch.zeros_like(
            outs_motion["sdc_traj_query"][:, :, None, :, :]
        )
        dummy_score = torch.zeros_like(
            outs_motion["sdc_track_query"][:, :1]
        )
        occ_motion["track_query"] = torch.cat(
            (outs_motion["track_query"], dummy_track[:, needs_dummy]), dim=1
        )
        occ_motion["track_query_pos"] = torch.cat(
            (outs_motion["track_query_pos"], dummy_track_pos[:, needs_dummy]),
            dim=1,
        )
        occ_motion["traj_query"] = torch.cat(
            (outs_motion["traj_query"], dummy_traj[:, :, needs_dummy]), dim=2
        )
        occ_motion["track_scores"] = torch.cat(
            (outs_motion["track_scores"], dummy_score[:, needs_dummy]), dim=1
        )
        outs_occ = self.model.occ_head.forward_test(
            bev_embed, occ_motion,
            no_query=False,
            gt_segmentation=dummy_gt_seg, gt_instance=dummy_gt_ins,
            gt_img_is_valid=dummy_gt_valid,
        )

        # Remove the optional dummy from per-agent outputs.  Keep seg_out from
        # the padded computation: when N=0 the zero score makes it all-zero;
        # when N>0 no dummy was inserted.
        padded_query_count = torch._shape_as_tensor(
            outs_occ["pred_ins_logits"]
        )[1]
        real_query_mask = torch.arange(
            padded_query_count, device=img.device
        ) < real_query_count
        pred_ins_logits = outs_occ["pred_ins_logits"][:, real_query_mask]
        pred_ins_sigmoid = outs_occ["pred_ins_sigmoid"][:, real_query_mask]

        # 6. planning_head
        # planning_head.forward_test → forward(bev, occ, bev_pos, traj_q, track_q, command)
        result_planning = self.model.planning_head.forward_test(
            bev_embed, outs_motion, outs_occ, command
        )

        # occ 实例分割: 不调用官方 make_instance_seg_consecutive
        # (内部 torch.unique + 数据依赖 Python 循环, trace 会把实例数固化进图)。
        # 按导出边界原则, 图内只输出 raw instance seg (argmax + 前景 mask),
        # id 连续化重标注属于部署侧后处理 (occ_head_plugin/utils.py)。
        fg_mask = (outs_occ["seg_out"].squeeze(2) == 1)           # (b, t, h, w) bool
        ins_seg_raw = (
            outs_occ["pred_ins_sigmoid"].argmax(dim=1) + 1
        ) * fg_mask.long()                                        # bg=0, fg 从 1 起

        outputs = (
            bev_embed,
            det_output["all_cls_scores"],
            det_output["all_bbox_preds"],
            det_output["all_past_traj_preds"],
            outs_motion["traj_query"],
            pred_ins_logits,
            pred_ins_sigmoid,
            outs_occ["seg_out"],
            ins_seg_raw,
            result_planning["sdc_traj"],
        )
        if return_aux:
            return outputs, traj_results, seg_pred
        return outputs


# -----------------------------------------------------------------------------
# Qualcomm fixed-capacity SpatialCrossAttention
# -----------------------------------------------------------------------------
# Qualcomm BEVFormer fixed-capacity SCA, adapted to the UniAD PT ABI.
#
# Source: qualcomm/ai-hub-models @ 1bc4be97a9c87f0aaed18451f8a4146130cef243
# src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
# SpatialCrossAttention optimized path and custom_utils.py::ScatterND.
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# See QUALCOMM_THIRD_PARTY_LICENSE.txt (BSD-3-Clause).
#
# The official TopK / IndexSelect / validity masking / ScatterND computation is
# reused. UniAD retains packed multi-level KV, its existing MSDA module and the
# validated camera-count divisor. Capacity is measured for UniAD, not copied
# from BEVFormer Tiny. This module is for eval/PT correctness at this stage.

# 404-frame metadata-only survey: max=9704; ceil(sqrt(max*1.05))**2.
# See frozen 35c8cac:onnx/runs/measure_qualcomm_sca_visibility/result.json. Overflow is an error,
# not a claim that this corpus bound applies to arbitrary new calibrations.
UNIAD_SCA_CAPACITY = 10201


def compute_fixed_sca_indices(bev_mask, capacity, height, width):
    """Compute Qualcomm fixed-SCA TopK/index tensors once per encoder forward.

    Qualcomm's optimized BEVFormer computes the visible-query TopK/index/mask
    on the first encoder layer and reuses the resulting tensors on later
    layers. UniAD has the same invariant: point-sampling geometry and bev_mask
    are fixed across all encoder layers for one frame.
    """
    side = math.isqrt(capacity)
    if side * side != capacity or not 0 < capacity <= height * width:
        raise ValueError(
            'Fixed SCA requires positive square capacity <= BEV queries'
        )
    counts = bev_mask[:, 0].any(-1).sum(-1)
    if bool((counts > capacity).any()):
        raise RuntimeError(
            f'SCA capacity overflow: counts={counts.tolist()}, '
            f'capacity={capacity}'
        )

    indexes, masks = [], []
    for mask_per_img in bev_mask.to(torch.float32):
        scores, index = torch.topk(
            mask_per_img[0].sum(-1), capacity, largest=True
        )
        indexes.append(index)
        masks.append(scores >= 1)
    indexes = torch.stack(indexes)
    masks = torch.stack(masks)
    indices_2d = torch.stack(
        (indexes // width, indexes % width), dim=-1
    ).reshape(indexes.shape[0], side, side, 2)
    return indexes, indices_2d, masks


class ScatterND(Function):
    """Direct Qualcomm helper; per-camera TopK indices must be unique."""
    @staticmethod
    def symbolic(g, canvas, indices, updates):
        return g.op('ScatterND', canvas, indices, updates, reduction_s='add')

    @staticmethod
    def forward(ctx, canvas, indices, updates):
        canvas_ = canvas.clone()
        for i, index in enumerate(indices):
            canvas_[index[:, :, 0], index[:, :, 1]] += updates[i]
        return canvas_


def fixed_spatial_forward(self, query, key, value, residual=None, query_pos=None,
                          key_padding_mask=None, reference_points=None, spatial_shapes=None,
                          reference_points_cam=None, bev_mask=None, level_start_index=None,
                          flag='encoder', computed_indices_1d=None,
                          computed_indices_2d=None,
                          computed_indices_1d_mask=None, **kwargs):
    if self.training:
        raise RuntimeError('Qualcomm fixed SCA is an inference-only adapter')
    height, width = self._qualcomm_bev_hw
    capacity = self._qualcomm_sca_capacity
    side = math.isqrt(capacity)
    if query.shape[0] != 1 or query.shape[1] != height * width:
        raise ValueError('Fixed SCA requires batch=1 and the configured BEV shape')

    supplied = (
        computed_indices_1d is not None,
        computed_indices_2d is not None,
        computed_indices_1d_mask is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError('Fixed SCA computed indices must be supplied together')
    if all(supplied):
        indexes = computed_indices_1d
        indices_2d = computed_indices_2d
        masks = computed_indices_1d_mask
    else:
        indexes, indices_2d, masks = compute_fixed_sca_indices(
            bev_mask, capacity, height, width
        )

    expected_1d = (self.num_cams, capacity)
    expected_2d = (self.num_cams, side, side, 2)
    if tuple(indexes.shape) != expected_1d or tuple(masks.shape) != expected_1d:
        raise ValueError('Fixed SCA computed 1-D index/mask shape mismatch')
    if tuple(indices_2d.shape) != expected_2d:
        raise ValueError('Fixed SCA computed 2-D index shape mismatch')

    inp_residual = query if residual is None else residual
    query = query if query_pos is None else query + query_pos
    value = query if value is None else value
    key = query if key is None else key
    depth = reference_points_cam.shape[3]
    # Direct Qualcomm selection: positive depth count wins over zero padding.
    queries = [
        torch.index_select(query[0], 0, index) for index in indexes
    ]
    references = [
        torch.index_select(reference_points_cam[i, 0], 0, index)
        for i, index in enumerate(indexes)
    ]
    # UniAD uses [cams, tokens, batch, channels], not Tiny's single NCHW map.
    key = key.permute(2, 0, 1, 3).reshape(self.num_cams, -1, self.embed_dims)
    value = value.permute(2, 0, 1, 3).reshape(self.num_cams, -1, self.embed_dims)
    attended = self.deformable_attention(
        query=torch.stack(queries), key=key, value=value,
        reference_points=torch.stack(references).reshape(self.num_cams, capacity, depth, 2),
        spatial_shapes=spatial_shapes, level_start_index=level_start_index)
    # Mask after attention as in the official sca_masking_before_gridsample=False path.
    updates = (
        attended * masks.to(query.dtype)[..., None]
    ).reshape(self.num_cams, side, side, self.embed_dims)
    # Official helper consumes a 2-D canvas; indices_2d is shared across
    # encoder layers when supplied by the stateful encoder.
    slots = ScatterND.apply(
        torch.zeros_like(query[0]).reshape(height, width, self.embed_dims),
        indices_2d,
        updates,
    ).reshape(1, height * width, self.embed_dims)
    count = bev_mask.any(-1).permute(1, 2, 0).sum(-1).clamp(min=1).to(query.dtype)
    return self.dropout(self.output_proj(slots / count[..., None])) + inp_residual


# -----------------------------------------------------------------------------
# Stateful BEV encoder patch
# -----------------------------------------------------------------------------
# Explicit first-frame handling, installed only on the stateful export model.


def _stateful_encoder_forward(
    self, bev_query, key, value, *args, bev_h=None, bev_w=None, bev_pos=None,
    spatial_shapes=None, level_start_index=None, valid_ratios=None,
    prev_bev=None, shift=0., img_metas=None, **kwargs,
):
    if "has_prev_bev" not in img_metas[0]:
        return self._stateful_original_forward(
            bev_query, key, value, *args, bev_h=bev_h, bev_w=bev_w,
            bev_pos=bev_pos, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index, valid_ratios=valid_ratios,
            prev_bev=prev_bev, shift=shift, img_metas=img_metas, **kwargs)

    # Batch=1 and a mandatory prev tensor are the stateful interface contract.
    has_prev = img_metas[0]["has_prev_bev"].to(torch.bool)
    ref_3d = self.get_reference_points(
        bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
        self.num_points_in_pillar, dim="3d", bs=bev_query.size(1),
        device=bev_query.device, dtype=bev_query.dtype)
    ref_2d = self.get_reference_points(
        bev_h, bev_w, dim="2d", bs=bev_query.size(1),
        device=bev_query.device, dtype=bev_query.dtype)
    ref_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)
    shifted = ref_2d + torch.where(has_prev, shift, torch.zeros_like(shift))[:, None, None, :]
    batch, length, levels, _ = ref_2d.shape
    hybrid_refs = torch.stack((shifted, ref_2d), dim=1).reshape(batch*2, length, levels, 2)
    query = bev_query.permute(1, 0, 2)
    position = bev_pos.permute(1, 0, 2)
    # Official history branch uses the initial query in this pair for all
    # layers. Official None branch constructs [current query,current query]
    # inside EACH layer's TemporalSelfAttention, not once before the encoder.
    history_pair = torch.stack((prev_bev.permute(1, 0, 2), query), dim=1).reshape(batch*2, length, query.shape[-1])

    # Qualcomm optimized BEVFormer computes SCA TopK/index/mask once and
    # carries them across encoder layers. The point-sampling geometry and
    # bev_mask are invariant across UniAD encoder layers, so do the same.
    computed_indices_1d = None
    computed_indices_2d = None
    computed_indices_1d_mask = None
    if hasattr(self, "_qualcomm_sca_capacity"):
        height, width = self._qualcomm_bev_hw
        computed_indices_1d, computed_indices_2d, computed_indices_1d_mask = (
            compute_fixed_sca_indices(
                bev_mask, self._qualcomm_sca_capacity, height, width
            )
        )

    intermediate = []
    for layer in self.layers:
        current_pair = torch.stack((query, query), dim=1).reshape(batch*2, length, query.shape[-1])
        temporal_value = torch.where(has_prev, history_pair, current_pair)
        query = layer(
            query, key, value, *args, bev_pos=position, ref_2d=hybrid_refs,
            ref_3d=ref_3d, bev_h=bev_h, bev_w=bev_w,
            spatial_shapes=spatial_shapes, level_start_index=level_start_index,
            reference_points_cam=ref_cam, bev_mask=bev_mask,
            prev_bev=temporal_value,
            computed_indices_1d=computed_indices_1d,
            computed_indices_2d=computed_indices_2d,
            computed_indices_1d_mask=computed_indices_1d_mask,
            **kwargs)
        if self.return_intermediate:
            intermediate.append(query)
    if self.return_intermediate:
        return torch.stack(intermediate)
    return query


def patch_stateful_encoder(encoder):
    """Keep legacy calls unchanged; activate with img_metas has_prev_bev tensor."""
    if hasattr(encoder, "_stateful_original_forward"):
        return
    encoder._stateful_original_forward = encoder.forward
    encoder.forward = types.MethodType(_stateful_encoder_forward, encoder)


# Fixed-SCA installation. The dynamic reference remains in frozen Git history.


def patch_stateful_spatial(encoder, bev_h=200, bev_w=200, max_len=UNIAD_SCA_CAPACITY):
    # The encoder-level attributes let stateful_bev compute Qualcomm's fixed
    # TopK/index/mask tensors once and reuse them across all SCA layers.
    encoder._qualcomm_bev_hw = (bev_h, bev_w)
    encoder._qualcomm_sca_capacity = max_len
    patched = 0
    for module in encoder.modules():
        if module.__class__.__name__ != "SpatialCrossAttention":
            continue
        if hasattr(module, "_stateful_spatial_original_forward"):
            continue
        module._stateful_spatial_original_forward = module.forward
        module._qualcomm_bev_hw = (bev_h, bev_w)
        module._qualcomm_sca_capacity = max_len
        module.forward = types.MethodType(fixed_spatial_forward, module)
        patched += 1
    return patched


# -----------------------------------------------------------------------------
# Learned map-mask decoding
# -----------------------------------------------------------------------------
# Host-side panoptic merging remains outside the neural graph.


MAP_OUTPUT_NAMES = ["map_mask_scores", "map_selected_boxes", "map_selected_labels",
                    "map_selected_query_indices", "map_stuff_class_scores"]


class TensorMapDecoder(nn.Module):
    """Inference prefix of PansegformerHead.get_bboxes, batch=1.

    Masks are raw nonnegative decoder scores (not sigmoid probabilities).
    Rows: selected things (TopK class/query pairs, duplicates allowed), then
    stuff classes. Boxes include the original class score in their last column,
    before mask-quality rescore and greedy panoptic merging on the host.
    """

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, cls_score, bbox_pred, memory, memory_mask, query, query_pos, hw_lvl):
        head = self.head
        height, width = head.canvas_size
        indices, boxes, labels = head._get_bboxes_single(
            cls_score[0], bbox_pred[0], (height, width, 3), 1, False)
        things = query[:, indices]
        stuff = head.stuff_query.weight[None, :, :head.embed_dims]
        stuff_pos = head.stuff_query.weight[None, :, head.embed_dims:]
        # Official things mask decoder does not consume query_pos.
        thing_masks, _, _ = head.things_mask_head(
            memory, memory_mask, None, things, None, None, hw_lvl=hw_lvl)
        stuff_masks, _, stuff_queries = head.stuff_mask_head(
            memory, memory_mask, None, stuff, None, stuff_pos, hw_lvl=hw_lvl)
        masks = torch.cat((thing_masks, stuff_masks), dim=1).squeeze(-1)
        masks = masks.reshape(-1, *hw_lvl[0])
        masks = F.interpolate(masks[None], size=(height, width), mode="bilinear", align_corners=False)[0]
        stuff_scores = head.cls_stuff_branches[-1](stuff_queries[-1]).sigmoid().reshape(-1)
        return masks, boxes, labels, indices, stuff_scores


# -----------------------------------------------------------------------------
# BEV rotation adapters
# -----------------------------------------------------------------------------


def polynomial_sincos(angle):
    """Range-reduced sin/cos approximation, not universal libm bit equality.

    On [-pi/4, pi/4], truncation bounds are <1e-19 for sin through x^17
    and <3e-18 for cos through x^16. Floating-point evaluation/reduction adds
    rounding error: require independent source-pixel regression, not just
    approximate trig comparison. Validated rotation angles span [-720,720].
    """
    quarter = torch.round(angle / (math.pi / 2.))
    x = angle - quarter * (math.pi / 2.)
    x2 = x * x
    sine = torch.full_like(x, 1. / math.factorial(17))
    cosine = torch.full_like(x, 1. / math.factorial(16))
    for degree in range(7, -1, -1):
        sine = sine * x2 + ((-1.) ** degree / math.factorial(2 * degree + 1))
        cosine = cosine * x2 + ((-1.) ** degree / math.factorial(2 * degree))
    sine = sine * x
    quadrant = torch.remainder(quarter.to(torch.int64), 4)
    c = torch.where(quadrant == 0, cosine, torch.where(quadrant == 1, -sine,
                    torch.where(quadrant == 2, -cosine, sine)))
    s = torch.where(quadrant == 0, sine, torch.where(quadrant == 1, cosine,
                    torch.where(quadrant == 2, -sine, -cosine)))
    return c, s


def _nearest_gather(bev, cosine, sine):
    """Shared exact source-index mapping for candidates A/B."""
    h, w = bev.shape[-2:]
    zero = torch.zeros_like(cosine)
    theta = torch.stack(
        (cosine, -sine, zero, sine, cosine, zero)
    ).reshape(1, 2, 3).to(bev.dtype)
    x = torch.arange(w, dtype=bev.dtype, device=bev.device) - w * .5 + .5
    y = torch.arange(h, dtype=bev.dtype, device=bev.device) - h * .5 + .5
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack(
        (gx, gy, torch.ones_like(gx)), -1
    ).reshape(1, h * w, 3)
    scale = torch.tensor(
        [w * .5, h * .5], dtype=bev.dtype, device=bev.device
    )
    normalized = theta.transpose(1, 2) / scale
    # Preserve the already-validated evaluation order.
    xp = base[..., 0:1] * normalized[:, 0:1, :]
    yp = base[..., 1:2] * normalized[:, 1:2, :]
    grid = xp + yp
    pixel = ((grid + 1.) / 2.) * (scale * 2.) - .5
    indices = torch.round(pixel).to(torch.int64)[0]
    ix, iy = indices[:, 0], indices[:, 1]
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    flat = iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)
    output = bev.flatten(1).index_select(1, flat)
    output = output * valid.to(bev.dtype)[None]
    return output.reshape_as(bev)


class StatefulNearestRotation(nn.Module):
    """Candidate A: frozen validated polynomial-trig implementation."""

    def forward(self, bev, angle_degrees):
        # Rank-one explicit DOUBLE constant prevents legacy exporter scalar
        # type analysis from silently demoting radians/trig to float32.
        angle = angle_degrees.reshape(1).double() * torch.tensor(
            [math.pi / 180.], dtype=torch.float64, device=bev.device
        )
        c, s = polynomial_sincos(angle)
        return _nearest_gather(bev, c, s)


class SinOnlyNearestRotation(nn.Module):
    """Candidate B: historical validated DOUBLE Sin-only simplification.

    This matches master::onnx/test_rotation_sin_candidate.py:
        sin(x) = Sin(x)
        cos(x) = Sin(x + pi/2)

    Keeping rank-one DOUBLE trig is part of the historical candidate. It avoids
    the float32 exporter demotion that previously changed nearest source pixels.
    """

    def forward(self, bev, angle_degrees):
        angle = angle_degrees.reshape(1).double() * torch.tensor(
            [math.pi / 180.], dtype=torch.float64, device=bev.device
        )
        offset = torch.tensor(
            [math.pi / 2.], dtype=torch.float64, device=bev.device
        )
        cosine = torch.sin(angle + offset)
        sine = torch.sin(angle)
        return _nearest_gather(bev, cosine, sine)
