# coding: utf-8
"""Export-only patches that keep UniAD's runtime inputs in the ONNX graph."""
import math
import sys
import types

import torch
import torch.nn.functional as F
from einops import rearrange

from attention import qualcomm_multihead_attention_forward


def _multihead_attention_qualcomm_forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    average_attn_weights=True,
    is_causal=False,
):
    """Explicit Qualcomm-style MHA instead of PyTorch's SDPA/legacy switch."""
    return qualcomm_multihead_attention_forward(
        self,
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        need_weights=need_weights,
        attn_mask=attn_mask,
        average_attn_weights=average_attn_weights,
        is_causal=is_causal,
    )


def patch_multihead_attention_for_onnx(module):
    """Install the shared explicit Qualcomm MHA math on torch MHA modules."""
    patched = 0
    for child in module.modules():
        if not isinstance(child, torch.nn.MultiheadAttention):
            continue
        if hasattr(child, "_onnx_original_forward"):
            continue
        child._onnx_original_forward = child.forward
        child.forward = types.MethodType(_multihead_attention_qualcomm_forward, child)
        patched += 1
    return patched


def _group_mode_query_pos_tensorized(self, bbox_results, mode_query_pos):
    """Select each agent's anchor group without a Python agent loop.

    The official implementation iterates over ``mode_query_pos.shape[1]``.
    That count is data-dependent after active-track filtering, so legacy ONNX
    tracing would unroll the loop for the dummy input's agent count.  Gather
    along the group dimension instead, preserving exactly the same values
    while keeping the agent dimension dynamic.
    """
    grouped_batches = []
    cls2group = self.cls2group.to(mode_query_pos.device)
    for batch_index, bbox_result in enumerate(bbox_results):
        labels = bbox_result[2].to(mode_query_pos.device)
        grouped_labels = cls2group[labels]
        batch_values = mode_query_pos[batch_index]
        index_shape = (grouped_labels.shape[0], 1) + (1,) * (
            batch_values.dim() - 2
        )
        expand_shape = (-1, 1) + tuple(batch_values.shape[2:])
        gather_index = grouped_labels.reshape(index_shape).expand(expand_shape)
        grouped_batches.append(
            torch.gather(batch_values, 1, gather_index).squeeze(1)
        )
    return torch.stack(grouped_batches)


def _qualcomm_atan2(y, x):
    """Direct Qualcomm BEVFormer tensor atan2 candidate.

    Source:
      qualcomm/ai-hub-models@1bc4be97a9c87f0aaed18451f8a4146130cef243
      projects/mmdet3d_plugin/custom_utils.py::atan2

    Kept as a candidate helper until Q1.5b establishes whether its axis/
    quadrant behavior and the official can-bus shift convention are strictly
    compatible with UniAD's validated Stateful semantics.
    """
    out = torch.atan(y / (x + 1e-8))
    out = out + ((y > 0) & (x < 0)).to(out.dtype) * math.pi
    out = out - ((y < 0) & (x < 0)).to(out.dtype) * math.pi
    positive_y_axis = ((y > 0) & (x == 0)).to(out.dtype)
    negative_y_axis = ((y < 0) & (x == 0)).to(out.dtype)
    out = out * (1.0 - positive_y_axis)
    out = out + positive_y_axis * (math.pi / 2.0)
    out = out * (1.0 - negative_y_axis)
    out = out + negative_y_axis * (-math.pi / 2.0)
    return out


def _qualcomm_can_bus_shift(
    can_bus,
    *,
    real_h,
    real_w,
    use_shift=True,
):
    """Direct Qualcomm BEVFormer export ego-shift candidate.

    This intentionally mirrors Qualcomm's tensor graph:
      delta_x/delta_y -> sqrt -> custom atan2 -> absolute ego yaw
      -> sin/cos -> normalized [shift_x, shift_y].

    It is not installed in the production Stateful path until focused
    validation proves compatibility with UniAD's l2g-matrix shift semantics.
    """
    if can_bus.dim() != 2 or can_bus.shape[-1] != 18:
        raise ValueError(
            "Qualcomm can-bus shift expects [B,18], got "
            f"{tuple(can_bus.shape)}"
        )
    delta_x = can_bus[:, 0]
    delta_y = can_bus[:, 1]
    ego_angle = can_bus[:, -2] / math.pi * 180.0
    translation_length = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = _qualcomm_atan2(delta_y, delta_x) / math.pi * 180.0
    bev_angle = ego_angle - translation_angle
    radians = bev_angle / 180.0 * math.pi
    shift_y = translation_length * torch.cos(radians) / float(real_h)
    shift_x = translation_length * torch.sin(radians) / float(real_w)
    enabled = float(bool(use_shift))
    return torch.stack((shift_x, shift_y), dim=-1) * enabled


def _get_reference_points_float32(self, *args, **kwargs):
    """Keep reference-point grids in float32 for ONNX Runtime GridSample."""
    return self._onnx_original_get_reference_points(*args, **kwargs).to(
        dtype=torch.float32
    )


def _rotate_bev_nearest(bev, angle_degrees):
    """Tensor-angle equivalent of torchvision.rotate for centered BEV maps."""
    radians_per_degree = angle_degrees.new_tensor(math.pi / 180.0).to(bev.dtype)
    angle = angle_degrees.to(dtype=bev.dtype) * radians_per_degree
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(cosine)
    theta = torch.stack((cosine, -sine, zero, sine, cosine, zero)).reshape(1, 2, 3)
    height, width = bev.shape[-2:]
    xs = (torch.arange(width, device=bev.device, dtype=bev.dtype) * 2.0 + 1.0) / width - 1.0
    ys = (torch.arange(height, device=bev.device, dtype=bev.dtype) * 2.0 + 1.0) / height - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs)
    base_grid = torch.stack(
        (grid_x, grid_y, torch.ones_like(grid_x)), dim=-1
    ).unsqueeze(0)
    grid = torch.matmul(base_grid, theta.transpose(1, 2))
    return F.grid_sample(
        bev.unsqueeze(0), grid, mode="nearest", padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)


def _get_bev_features_tensor_metadata(
    self, mlvl_feats, bev_queries, bev_h, bev_w, real_h, real_w,
    grid_length=(0.512, 0.512), bev_pos=None, prev_bev=None, img_metas=None,
):
    batch_size = mlvl_feats[0].size(0)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, batch_size, 1)
    bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

    can_bus = torch.stack([meta["can_bus"] for meta in img_metas]).to(bev_queries)
    global_to_lidar = torch.stack(
        [meta["l2g_r_mat"] for meta in img_metas]
    ).to(bev_queries)
    delta_lidar = torch.matmul(
        global_to_lidar, can_bus[:, :3].unsqueeze(-1)
    ).squeeze(-1)
    shift = torch.stack(
        (delta_lidar[:, 0] / real_w, delta_lidar[:, 1] / real_h), dim=-1
    ) * float(self.use_shift)

    if prev_bev is not None:
        if prev_bev.shape[1] == bev_h * bev_w:
            prev_bev = prev_bev.permute(1, 0, 2)
        if self.rotate_prev_bev:
            rotated = []
            for batch_index in range(batch_size):
                bev = prev_bev[:, batch_index].reshape(
                    bev_h, bev_w, -1
                ).permute(2, 0, 1)
                # Stateful wrapper opts into the separately validated v2
                # implementation; retain legacy export behavior by default.
                rotate_bev = getattr(self, "_onnx_stateful_rotation", _rotate_bev_nearest)
                bev = rotate_bev(bev, can_bus[batch_index, -1])
                rotated.append(bev.permute(1, 2, 0).reshape(bev_h * bev_w, -1))
            prev_bev = torch.stack(rotated, dim=1)

    can_bus_embedding = self.can_bus_mlp(can_bus)[None, :, :]
    bev_queries = bev_queries + can_bus_embedding * float(self.use_can_bus)

    feat_flatten = []
    spatial_shapes = []
    for level, feat in enumerate(mlvl_feats):
        _, _, _, height, width = feat.shape
        feat = feat.flatten(3).permute(1, 0, 3, 2)
        if self.use_cams_embeds:
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
        feat = feat + self.level_embeds[
            None, None, level:level + 1, :
        ].to(feat.dtype)
        spatial_shapes.append((height, width))
        feat_flatten.append(feat)

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=bev_pos.device
    )
    level_start_index = torch.cat((
        spatial_shapes.new_zeros((1,)),
        spatial_shapes.prod(1).cumsum(0)[:-1],
    ))
    feat_flatten = feat_flatten.permute(0, 2, 1, 3)

    return self.encoder(
        bev_queries, feat_flatten, feat_flatten,
        bev_h=bev_h, bev_w=bev_w, bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        prev_bev=prev_bev, shift=shift, img_metas=img_metas,
    )


def _point_sampling_tensor_metadata(self, reference_points, pc_range, img_metas):
    """Qualcomm BEVFormer export-style point sampling with strict UniAD depth.

    Direct source basis:
      qualcomm/ai-hub-models@1bc4be97a9c87f0aaed18451f8a4146130cef243
      projects/mmdet3d_plugin/bevformer/modules/encoder.py
      ::BEVFormerEncoder.point_sampling_export

    Reused graph structure:
      normalized ref -> metric xyz1
      -> per-depth camera MatMul
      -> explicit x/y/z tensors
      -> reciprocal depth
      -> camera-wise image normalization
      -> camera-first reference/mask output.

    Intentional UniAD correctness adaptations:
      * stateful/export ABI is batch=1;
      * preserve per-camera img_shape instead of assuming all cameras share
        img_metas[0]['img_shape'][0];
      * preserve the frozen strict denominator max(z, 1e-5), NOT Qualcomm's
        deployment/quantization clamp max(z, 1.0).
    """
    if len(img_metas) != 1 or reference_points.shape[0] != 1:
        raise ValueError(
            "Qualcomm point-sampling adapter requires the UniAD export "
            "batch=1 contract"
        )

    device = reference_points.device
    lidar2img = img_metas[0]["lidar2img"].to(
        device=device, dtype=torch.float32
    )
    img_shape = img_metas[0]["img_shape"].to(
        device=device, dtype=torch.float32
    )
    if lidar2img.dim() != 3 or lidar2img.shape[-2:] != (4, 4):
        raise ValueError(
            "lidar2img must have shape [num_cam,4,4], got "
            f"{tuple(lidar2img.shape)}"
        )
    if img_shape.dim() != 2 or img_shape.shape[0] != lidar2img.shape[0]:
        raise ValueError(
            "img_shape must have shape [num_cam,>=2] matching lidar2img"
        )

    scale = reference_points.new_tensor(
        [
            pc_range[3] - pc_range[0],
            pc_range[4] - pc_range[1],
            pc_range[5] - pc_range[2],
        ],
        dtype=torch.float32,
    ).view(1, 1, 1, 3)
    origin = reference_points.new_tensor(
        pc_range[:3], dtype=torch.float32
    ).view(1, 1, 1, 3)
    metric = reference_points.to(torch.float32) * scale + origin
    homogeneous = torch.cat(
        (metric, torch.ones_like(metric[..., :1])), dim=-1
    )

    # [B,D,Q,4] -> [D,4,Q], matching Qualcomm's export-only
    # reference_points_not_repeated representation for B=1.
    points = homogeneous.permute(1, 0, 2, 3)[:, 0].transpose(1, 2)
    num_depth = points.shape[0]

    xs, ys, zs = [], [], []
    for depth_index in range(num_depth):
        projected = torch.matmul(lidar2img, points[depth_index])
        x_value, y_value, z_value, _ = projected.split(1, dim=1)
        xs.append(x_value.squeeze(1).unsqueeze(-1))
        ys.append(y_value.squeeze(1).unsqueeze(-1))
        zs.append(z_value.squeeze(1).unsqueeze(-1))

    x_values = torch.cat(xs, dim=-1)
    y_values = torch.cat(ys, dim=-1)
    z_values = torch.cat(zs, dim=-1)

    epsilon = 1e-5
    positive_depth = z_values > epsilon
    inv_z = torch.reciprocal(
        torch.maximum(z_values, torch.full_like(z_values, epsilon))
    )

    inv_width = torch.reciprocal(img_shape[:, 1]).view(-1, 1, 1)
    inv_height = torch.reciprocal(img_shape[:, 0]).view(-1, 1, 1)
    x_values = x_values * inv_z * inv_width
    y_values = y_values * inv_z * inv_height

    bev_mask = (
        positive_depth
        & (y_values > 0.0)
        & (y_values < 1.0)
        & (x_values < 1.0)
        & (x_values > 0.0)
    )

    reference_points_cam = torch.stack((x_values, y_values), dim=-1)
    return reference_points_cam.unsqueeze(1), bev_mask.unsqueeze(1)


def _get_detections_with_query_padding_mask(
    self, bev_embed, object_query_embeds=None, ref_points=None,
    img_metas=None, query_key_padding_mask=None,
):
    """Pass a mask through the original UniAD head only on a fixed model."""
    original = self._original_get_detections
    if query_key_padding_mask is None:
        return original(bev_embed, object_query_embeds=object_query_embeds,
                        ref_points=ref_points, img_metas=img_metas)
    transformer = self.transformer
    if getattr(transformer, "_pending_query_padding_mask", None) is not None:
        raise RuntimeError("nested fixed detector mask call")
    transformer._pending_query_padding_mask = query_key_padding_mask
    try:
        return original(bev_embed, object_query_embeds=object_query_embeds,
                        ref_points=ref_points, img_metas=img_metas)
    finally:
        transformer._pending_query_padding_mask = None


def _compute_occupancy_mask_logits_matmul(query, state):
    """Contract channels while keeping batch, agent, and spatial order."""
    batch, _, height, width = state.shape
    flattened = state.reshape(batch, state.shape[1], height * width)
    return torch.matmul(query, flattened).reshape(
        batch, query.shape[1], height, width
    )


def _compute_temporal_occupancy_logits_matmul(query, states):
    """Use rank-three MatMul over B*T, then restore B,Q,T,H,W."""
    batch, steps, channels, height, width = states.shape
    queries = query.shape[2]
    left = query.reshape(batch * steps, queries, channels)
    right = states.reshape(batch * steps, channels, height * width)
    logits = torch.matmul(left, right)
    return logits.reshape(batch, steps, queries, height, width).permute(
        0, 2, 1, 3, 4
    )


def _forward_occupancy_with_matmul(self, x, ins_query):
    """Fixed occupancy forward; only the final contraction differs from UniAD."""
    base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])
    base_state = self.bev_sampler(base_state)
    base_state = self.bev_light_proj(base_state)
    base_state = self.base_downscale(base_state)
    base_ins_query = ins_query
    last_state = base_state
    last_ins_query = base_ins_query
    future_states = []
    mask_preds = []
    temporal_query = []
    temporal_embed_for_mask_attn = []
    n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
    assert n_trans_layer_each_block >= 1
    for i in range(self.n_future_blocks):
        cur_state = self.downscale_convs[i](last_state)
        cur_ins_query = self.temporal_mlps[i](last_ins_query)
        temporal_query.append(cur_ins_query)
        (attn_mask, mask_pred, cur_ins_emb_for_mask_attn) = self.get_attn_mask(cur_state, cur_ins_query)
        attn_masks = [None, attn_mask]
        mask_preds.append(mask_pred)
        temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)
        cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
        cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')
        for j in range(n_trans_layer_each_block):
            trans_layer_ind = i * n_trans_layer_each_block + j
            trans_layer = self.transformer_decoder.layers[trans_layer_ind]
            cur_state = trans_layer(query=cur_state, key=cur_ins_query, value=cur_ins_query, query_pos=None, key_pos=None, attn_masks=attn_masks, query_key_padding_mask=None, key_padding_mask=None)
        cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0] // 8)
        cur_state = self.upsample_adds[i](cur_state, last_state)
        future_states.append(cur_state)
        last_state = cur_state
    future_states = torch.stack(future_states, dim=1)
    temporal_query = torch.stack(temporal_query, dim=1)
    mask_preds = torch.stack(mask_preds, dim=2)
    ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)
    future_states = self.dense_decoder(future_states)
    ins_occ_query = self.query_to_occ_feat(ins_query)
    ins_occ_logits = _compute_temporal_occupancy_logits_matmul(ins_occ_query, future_states)
    return (mask_preds, ins_occ_logits)


def _build_occupancy_attention_mask(self, state, ins_query):
    """Keep invalid vehicles masked, including all-background BEV pixels."""
    valid = self._occupancy_vehicle_valid_mask
    if valid is None or valid.shape != (ins_query.shape[1],):
        raise ValueError("fixed occupancy vehicle mask differs from agent axis")
    ins_embed = self.temporal_mlp_for_mask(ins_query)
    mask_pred = _compute_occupancy_mask_logits_matmul(ins_embed, state)
    attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
    attn_mask = rearrange(attn_mask, "b q h w -> b (h w) q").unsqueeze(1).repeat(
        1, self.num_heads, 1, 1).flatten(0, 1).detach()
    valid = valid.to(torch.bool)
    attn_mask = attn_mask | (~valid)[None, None, :]
    all_masked = attn_mask.all(dim=-1)
    # UniAD exposes all real keys when spatial gating masks every real agent.
    # With zero real vehicles, expose only one safe dummy key to avoid NaNs.
    safe_dummy = torch.arange(valid.shape[0], device=valid.device) == 0
    fallback_keys = valid | (~valid.any() & safe_dummy)
    fallback_mask = (~fallback_keys)[None, None, :]
    attn_mask = torch.where(all_masked[..., None], fallback_mask, attn_mask)
    upsampled = F.interpolate(mask_pred, self.bev_size, mode="bilinear",
                               align_corners=False)
    return attn_mask, upsampled, ins_embed


def _forward_occupancy_with_fixed_vehicle_slots(
    self, bev_feat, outs_dict, no_query=False,
    gt_segmentation=None, gt_instance=None, gt_img_is_valid=None,
):
    """Fixed inference path without dummy-GT Unique or variable agent slicing."""
    if no_query:
        raise ValueError("fixed occupancy uses explicit 96-slot agent input")
    valid = outs_dict["vehicle_valid_mask"]
    if valid.shape != (96,) or valid.dtype != torch.bool:
        raise ValueError("fixed occupancy requires bool[96] vehicle_valid_mask")
    if getattr(self, "_occupancy_vehicle_valid_mask", None) is not None:
        raise RuntimeError("nested fixed occupancy call")
    self._occupancy_vehicle_valid_mask = valid
    try:
        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)
        ins_query = torch.where(valid[None, :, None], ins_query,
                                torch.zeros_like(ins_query))
        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)
    finally:
        self._occupancy_vehicle_valid_mask = None
    pred_ins_sigmoid = pred_ins_logits[:, :, :1 + self.n_future].sigmoid()
    if self.test_with_track_score:
        scores = outs_dict["track_scores"].to(pred_ins_sigmoid)
        pred_ins_sigmoid = pred_ins_sigmoid * scores[:, :, None, None, None]
    pred_ins_sigmoid = torch.where(
        valid[None, :, None, None, None], pred_ins_sigmoid,
        torch.zeros_like(pred_ins_sigmoid),
    )
    seg_out = (pred_ins_sigmoid.max(dim=1).values > self.test_seg_thresh)
    return {
        "pred_ins_logits": pred_ins_logits,
        "pred_ins_sigmoid": pred_ins_sigmoid,
        "seg_out": seg_out.long().unsqueeze(2),
    }


def _get_states_and_refs_with_query_padding_mask(
    self, bev_embed, object_query_embed, bev_h, bev_w, reference_points,
    reg_branches=None, cls_branches=None, img_metas=None,
    query_key_padding_mask=None,
):
    """Pass a fixed-slot validity mask only to decoder self-attention.

    The v1 path calls the saved original method exactly. For a Q3 fixed-slot
    candidate, self-attention masks padding keys while BEV cross-attention
    keeps its separate (None) key mask. This does not modify Layer 1 math.
    """
    if query_key_padding_mask is None:
        query_key_padding_mask = getattr(self, "_pending_query_padding_mask", None)
    if query_key_padding_mask is None:
        return self._onnx_original_get_states_and_refs(
            bev_embed, object_query_embed, bev_h, bev_w, reference_points,
            reg_branches=reg_branches, cls_branches=cls_branches, img_metas=img_metas,
        )
    bs = bev_embed.shape[1]
    query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
    query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
    query = query.unsqueeze(0).expand(bs, -1, -1)
    reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1).sigmoid()
    query = query.permute(1, 0, 2)
    query_pos = query_pos.permute(1, 0, 2)
    inter_states, inter_references = self.decoder(
        query=query, key=None, value=bev_embed, query_pos=query_pos,
        reference_points=reference_points, reg_branches=reg_branches,
        cls_branches=cls_branches,
        spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
        level_start_index=torch.tensor([0], device=query.device),
        img_metas=img_metas, query_key_padding_mask=query_key_padding_mask,
    )
    return inter_states, reference_points, inter_references


def patch_model_for_onnx_export(model):
    """Install export-only behavior without editing UniAD's official modules."""
    transformer = model.pts_bbox_head.transformer
    transformer.get_bev_features = types.MethodType(
        _get_bev_features_tensor_metadata, transformer
    )
    transformer._onnx_original_get_states_and_refs = transformer.get_states_and_refs
    transformer.get_states_and_refs = types.MethodType(
        _get_states_and_refs_with_query_padding_mask, transformer
    )
    transformer.encoder.point_sampling = types.MethodType(
        _point_sampling_tensor_metadata, transformer.encoder
    )
    transformer.encoder._onnx_original_get_reference_points = (
        transformer.encoder.get_reference_points
    )
    transformer.encoder.get_reference_points = types.MethodType(
        _get_reference_points_float32, transformer.encoder
    )
    head = model.pts_bbox_head
    head._original_get_detections = head.get_detections
    head.get_detections = types.MethodType(_get_detections_with_query_padding_mask, head)
    occ_head = model.occ_head
    occ_head._original_get_attn_mask = occ_head.get_attn_mask
    occ_head.get_attn_mask = types.MethodType(_build_occupancy_attention_mask, occ_head)
    occ_head._original_occupancy_forward = occ_head.forward
    occ_head.forward = types.MethodType(_forward_occupancy_with_matmul, occ_head)
    occ_head._original_occupancy_forward_test = occ_head.forward_test
    occ_head.forward_test = types.MethodType(_forward_occupancy_with_fixed_vehicle_slots, occ_head)
    patch_multihead_attention_for_onnx(model)
    model.motion_head.group_mode_query_pos = types.MethodType(
        _group_mode_query_pos_tensorized, model.motion_head
    )
    model.planning_head.use_col_optim = False

    motion_module = sys.modules.get(
        "projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.motion_deformable_attn"
    )
    if motion_module is not None:
        motion_module.copy = types.SimpleNamespace(
            deepcopy=lambda value: value.clone()
        )
