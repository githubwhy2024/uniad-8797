"""Qualcomm-style explicit Multi-Head Attention for UniAD inference.

Official source:
  qualcomm/ai-hub-models @ 1bc4be97a9c87f0aaed18451f8a4146130cef243
  src/qai_hub_models/models/bevformer/external_repos/bevformertiny_minimal.diff
  projects/mmdet3d_plugin/bevformer/modules/MultiheadAttention.py
  ::Torch_nn_MultiheadAttention_Optimized

The official implementation makes the deployment-friendly computation explicit:
packed Q/K/V projection -> per-head scaling -> MatMul -> Softmax -> MatMul ->
output projection.  Its BEVFormer-Tiny implementation uses a 4-D spatial ABI
and its optimized path does not implement the key-padding semantics needed by
UniAD's MemoryBank/QIM.  This adapter keeps that official arithmetic while
generalizing only the tensor ABI and masks required by UniAD.

Q4-only performance changes such as split-head execution and Linear->Conv1x1
are intentionally not included here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _batch_first(tensor: torch.Tensor, batch_first: bool, unbatched: bool) -> torch.Tensor:
    if unbatched:
        return tensor.unsqueeze(0)
    return tensor if batch_first else tensor.transpose(0, 1)


def _restore_layout(tensor: torch.Tensor, batch_first: bool, unbatched: bool) -> torch.Tensor:
    if unbatched:
        return tensor[0]
    return tensor if batch_first else tensor.transpose(0, 1)


def _apply_attention_mask(
    logits: torch.Tensor,
    attn_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
    target_len: int,
    source_len: int,
) -> torch.Tensor:
    if attn_mask is None:
        return logits

    if attn_mask.dim() == 2:
        if tuple(attn_mask.shape) != (target_len, source_len):
            raise ValueError(
                f"2-D attn_mask must be {(target_len, source_len)}, "
                f"got {tuple(attn_mask.shape)}"
            )
        mask = attn_mask[None, None]
    elif attn_mask.dim() == 3:
        expected = (batch_size * num_heads, target_len, source_len)
        if tuple(attn_mask.shape) != expected:
            raise ValueError(f"3-D attn_mask must be {expected}, got {tuple(attn_mask.shape)}")
        mask = attn_mask.reshape(batch_size, num_heads, target_len, source_len)
    else:
        raise ValueError("attn_mask must be 2-D or 3-D")

    if mask.dtype == torch.bool or not torch.is_floating_point(mask):
        return logits.masked_fill(mask.to(torch.bool), float("-inf"))
    return logits + mask.to(dtype=logits.dtype)


def _apply_key_padding_mask(
    logits: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
    batch_size: int,
    source_len: int,
    unbatched: bool,
) -> torch.Tensor:
    if key_padding_mask is None:
        return logits

    mask = key_padding_mask
    if unbatched:
        if mask.dim() != 1 or mask.shape[0] != source_len:
            raise ValueError(
                f"unbatched key_padding_mask must be ({source_len},), "
                f"got {tuple(mask.shape)}"
            )
        mask = mask[None]
    elif mask.dim() != 2 or tuple(mask.shape) != (batch_size, source_len):
        raise ValueError(
            f"key_padding_mask must be {(batch_size, source_len)}, "
            f"got {tuple(mask.shape)}"
        )

    mask = mask[:, None, None, :]
    if mask.dtype == torch.bool or not torch.is_floating_point(mask):
        return logits.masked_fill(mask.to(torch.bool), float("-inf"))
    return logits + mask.to(dtype=logits.dtype)


def qualcomm_multihead_attention_forward(
    module: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = True,
    attn_mask: torch.Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
):
    """Explicit eval-compatible MHA using Qualcomm's deployment arithmetic.

    Supports the standard equal-QKV-dimension MultiheadAttention configuration
    used by UniAD, including sequence-first/batch-first and the masks required
    by MemoryBank/QIM.  Bias-K/V, zero-attention tokens and causal inference are
    deliberately rejected because UniAD does not use them and Qualcomm's
    frozen BEVFormer path does not require them.
    """
    if is_causal:
        raise ValueError("Qualcomm UniAD MHA adapter does not support is_causal=True")
    if module.bias_k is not None or module.bias_v is not None:
        raise ValueError("Qualcomm UniAD MHA adapter does not support bias_k/bias_v")
    if module.add_zero_attn:
        raise ValueError("Qualcomm UniAD MHA adapter does not support add_zero_attn")
    if not getattr(module, "_qkv_same_embed_dim", True):
        raise ValueError("Qualcomm UniAD MHA adapter requires equal Q/K/V embed dims")
    if module.in_proj_weight is None:
        raise ValueError("Qualcomm UniAD MHA adapter requires packed in_proj_weight")
    if query.dim() not in (2, 3) or key.dim() != query.dim() or value.dim() != query.dim():
        raise ValueError("query/key/value must all be 2-D or all be 3-D")

    unbatched = query.dim() == 2
    q_input = _batch_first(query, module.batch_first, unbatched)
    k_input = _batch_first(key, module.batch_first, unbatched)
    v_input = _batch_first(value, module.batch_first, unbatched)

    if q_input.shape[0] != k_input.shape[0] or q_input.shape[0] != v_input.shape[0]:
        raise ValueError("query/key/value batch sizes must match")
    if k_input.shape[1] != v_input.shape[1]:
        raise ValueError("key/value sequence lengths must match")

    embed_dim = module.embed_dim
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError("embed_dim must be divisible by num_heads")

    weight_q, weight_k, weight_v = module.in_proj_weight.chunk(3, dim=0)
    if module.in_proj_bias is None:
        bias_q = bias_k = bias_v = None
    else:
        bias_q, bias_k, bias_v = module.in_proj_bias.chunk(3, dim=0)

    # Same deployment-oriented arithmetic as Qualcomm's optimized MHA:
    # explicit projection, explicit head reshape, explicit scaled MatMuls.
    q = F.linear(q_input, weight_q, bias_q)
    k = F.linear(k_input, weight_k, bias_k)
    v = F.linear(v_input, weight_v, bias_v)

    batch_size, target_len, _ = q.shape
    source_len = k.shape[1]
    q = q.reshape(batch_size, target_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, source_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, source_len, num_heads, head_dim).transpose(1, 2)

    logits = torch.matmul(q * (head_dim ** -0.5), k.transpose(-2, -1))
    logits = _apply_attention_mask(
        logits, attn_mask, batch_size, num_heads, target_len, source_len
    )
    logits = _apply_key_padding_mask(
        logits, key_padding_mask, batch_size, source_len, unbatched
    )

    weights = torch.softmax(logits, dim=-1)
    if module.training and module.dropout > 0.0:
        weights = F.dropout(weights, p=module.dropout)

    output = torch.matmul(weights, v)
    output = output.transpose(1, 2).reshape(batch_size, target_len, embed_dim)
    output = F.linear(output, module.out_proj.weight, module.out_proj.bias)
    output = _restore_layout(output, module.batch_first, unbatched)

    if not need_weights:
        return output, None

    reported_weights = weights.mean(dim=1) if average_attn_weights else weights
    if unbatched:
        reported_weights = reported_weights[0]
    return output, reported_weights


# coding: utf-8
"""ONNX export adapters for UniAD deformable attention."""
import sys



def custom_multi_scale_deformable_attn_pytorch_single_grid(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_grid: torch.Tensor,
    attention_weights: torch.Tensor,
    num_heads: int,
    grid_sample_mode: str,
    is_training: bool,
) -> torch.Tensor:
    """Qualcomm BEVFormer single-grid MSDA sampling core.

    Direct source:
      qualcomm/ai-hub-models
      commit 1bc4be97a9c87f0aaed18451f8a4146130cef243
      src/qai_hub_models/models/bevformer/external_repos/
      bevformertiny_minimal.diff
      ::custom_multi_scale_deformable_attn_pytorch_single_grid

    The arithmetic below intentionally follows the Qualcomm implementation:
    reshape value per head -> GridSample -> weighted sum.  UniAD ABI adaptation
    is kept outside this function in
    ``multi_scale_deformable_attn_pytorch_qualcomm_single_level``.
    """
    bs, _, _, _ = value.shape
    _, num_queries, _, _ = sampling_grid.shape

    # Single-level attention (num_levels == 1); index the one spatial shape
    # directly instead of looping and discarding all but level 0.
    H_, W_ = value_spatial_shapes[0]
    value_l_ = value.reshape(bs * num_heads, -1, H_, W_)
    sampling_value = F.grid_sample(
        value_l_,
        sampling_grid,
        mode=grid_sample_mode,
        align_corners=False,
    )

    output = (
        sampling_value.permute(0, 2, 3, 1) * attention_weights
    ).sum(1)

    return output


def multi_scale_deformable_attn_pytorch_qualcomm_single_level(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
):
    """Adapt UniAD/MMCV's packed one-level ABI to Qualcomm's BEVFormer core.

    Qualcomm's public BEVFormer helper consumes NCHW value maps plus a sampling
    grid already arranged as ``[batch*heads, points, queries, 2]`` in [-1, 1].
    UniAD/MMCV supplies packed value tokens and normalized sampling locations:
    ``value=[B, HW, heads, dim]`` and
    ``sampling_locations=[B, Q, heads, 1, points, 2]``.

    Only those layout/range conversions are performed here.  The sampling and
    weighted-sum arithmetic stays in the directly reused Qualcomm helper above.
    """
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, location_heads, parameter_levels, num_points, _ = (
        sampling_locations.shape
    )
    data_levels = value_spatial_shapes.shape[0]

    if data_levels != 1 or parameter_levels != 1:
        raise RuntimeError(
            "Qualcomm BEVFormer single-grid MSDA requires exactly one data "
            f"and parameter level; got data_levels={data_levels}, "
            f"parameter_levels={parameter_levels}"
        )
    if location_heads != num_heads:
        raise RuntimeError(
            "MSDA head mismatch between value and sampling locations: "
            f"{num_heads} vs {location_heads}"
        )

    height, width = value_spatial_shapes[0]

    # UniAD packed tokens -> Qualcomm BEVFormer NCHW channel packing.
    value_nchw = (
        value.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch_size, num_heads * hidden_dim, height, width)
    )

    # MMCV normalized [0, 1] coordinates -> GridSample [-1, 1] coordinates.
    # Qualcomm's BEVFormer optimized path places points before queries so the
    # weighted reduction is a direct sum over dimension 1.
    sampling_grid = (
        (sampling_locations[:, :, :, 0, :, :] * 2.0 - 1.0)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(batch_size * num_heads, num_points, num_queries, 2)
    )

    # Qualcomm's caller expands the trailing weight dimension to dim_per_head.
    weights = (
        attention_weights[:, :, :, 0, :]
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch_size * num_heads, num_points, num_queries, 1)
        .expand(batch_size * num_heads, num_points, num_queries, hidden_dim)
    )

    output = custom_multi_scale_deformable_attn_pytorch_single_grid(
        value_nchw,
        value_spatial_shapes,
        sampling_grid,
        weights,
        num_heads,
        "bilinear",
        False,
    )

    return (
        output.reshape(batch_size, num_heads, num_queries, hidden_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_size, num_queries, num_heads * hidden_dim)
    )



def multi_scale_deformable_attention_qualcomm(
    value, value_spatial_shapes, sampling_locations, attention_weights,
):
    """Qualcomm Mask2Former multi-level core with a batch-grid adaptation.

    Derived from qualcomm/ai-hub-models @
    1bc4be97a9c87f0aaed18451f8a4146130cef243:
    src/qai_hub_models/models/mask2former/model_patches.py
    ::multi_scale_deformable_attention (BSD-3-Clause).
    Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
    See QUALCOMM_THIRD_PARTY_LICENSE.txt.

    Retains the official flattened [B*Q, heads, levels*points, 2] location
    ABI, per-level sampling, concat and single weighted reduction. The one
    arithmetic-path adaptation is restoring B before folding B*heads in each
    grid: the upstream transpose(0, 1) alone assumes B=1.
    """
    batch_size, _, num_heads, hidden_dim = value.shape
    num_queries, num_heads, num_points, _ = sampling_locations.shape
    num_queries //= batch_size
    value_list = value.split(
        [height * width for height, width in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    sampling_grids = sampling_grids.split(num_points // len(value_spatial_shapes), 2)
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # UniAD B=6 camera-folded path: preserve batch/head identity.
        sampling_grid_l_ = (
            sampling_grids[level_id]
            .reshape(batch_size, num_queries, num_heads, -1, 2)
            .transpose(1, 2).flatten(0, 1)
        )
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_points
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1).view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


def multi_scale_deformable_attn_pytorch_qualcomm_multi_level(
    value, value_spatial_shapes, sampling_locations, attention_weights,
):
    """Adapt levels-matched MMCV inputs to the Qualcomm Mask2Former core."""
    batch_size, num_queries, heads, levels, points, _ = sampling_locations.shape
    if levels != value_spatial_shapes.shape[0] or heads != value.shape[2]:
        raise RuntimeError("Qualcomm multi-level MSDA requires matching levels/heads")
    return multi_scale_deformable_attention_qualcomm(
        value, value_spatial_shapes,
        sampling_locations.reshape(batch_size * num_queries, heads, levels * points, 2),
        attention_weights.reshape(batch_size, num_queries, heads, levels * points),
    )


def multi_scale_deformable_attn_pytorch_legacy_cuda(
    value, value_spatial_shapes, sampling_locations, attention_weights,
):
    """Reproduce UniAD's legacy CUDA level indexing, then use Qualcomm sampling.

    UniAD's released checkpoint was trained with batch size 1. Its seg head
    creates four parameter levels but passes one row in ``spatial_shapes``.
    mmcv CUDA uses the latter as its pointer stride, so it consumes the first
    quarter of the flattened location/weight storage.

    For one-data-level paths, sampling is now delegated to Qualcomm's public
    BEVFormer single-grid implementation after the required UniAD ABI adapter.
    Levels-matched multi-level paths use Qualcomm Mask2Former sampling with
    the minimal UniAD batch/layout adapter.
    """
    batch_size, _, num_heads, _ = value.shape
    _, num_queries, _, parameter_levels, num_points, _ = sampling_locations.shape
    data_levels = value_spatial_shapes.shape[0]

    if parameter_levels == data_levels:
        if data_levels == 1:
            return multi_scale_deformable_attn_pytorch_qualcomm_single_level(
                value,
                value_spatial_shapes,
                sampling_locations,
                attention_weights,
            )
        return multi_scale_deformable_attn_pytorch_qualcomm_multi_level(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )

    if batch_size != 1 or data_levels != 1:
        raise RuntimeError(
            "legacy CUDA MSDA export only supports batch_size=1 and one data level; "
            f"got batch_size={batch_size}, data_levels={data_levels}, "
            f"parameter_levels={parameter_levels}"
        )

    flattened_locations = sampling_locations.reshape(
        batch_size, num_queries * num_heads * parameter_levels, num_points, 2
    )
    flattened_weights = attention_weights.reshape(
        batch_size, num_queries * num_heads * parameter_levels, num_points
    )
    used_rows = num_queries * num_heads
    sampling_locations = flattened_locations[:, :used_rows].reshape(
        batch_size, num_queries, num_heads, 1, num_points, 2
    )
    attention_weights = flattened_weights[:, :used_rows].reshape(
        batch_size, num_queries, num_heads, 1, num_points
    )

    return multi_scale_deformable_attn_pytorch_qualcomm_single_level(
        value,
        value_spatial_shapes,
        sampling_locations,
        attention_weights,
    )


def patch_msda_for_legacy_cuda():
    """Patch imported references while preserving the original MMCV fallback."""
    import mmcv.ops.multi_scale_deform_attn as msda_mod

    official_impl = msda_mod.multi_scale_deformable_attn_pytorch
    multi_scale_deformable_attn_pytorch_legacy_cuda._official_mmcv_impl = official_impl
    msda_mod.multi_scale_deformable_attn_pytorch = (
        multi_scale_deformable_attn_pytorch_legacy_cuda
    )

    module_names = [
        "projects.mmdet3d_plugin.uniad.modules.decoder",
        "projects.mmdet3d_plugin.uniad.modules.spatial_cross_attention",
        "mmdet.models.utils.transformer",
        "projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.motion_deformable_attn",
    ]
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(
            module, "multi_scale_deformable_attn_pytorch"
        ):
            module.multi_scale_deformable_attn_pytorch = (
                multi_scale_deformable_attn_pytorch_legacy_cuda
            )
