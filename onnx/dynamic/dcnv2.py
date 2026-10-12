# coding: utf-8
"""ONNX-exportable reimplementation of ModulatedDeformConv2dPack.

原始 `mmcv.ops.ModulatedDeformConv2dPack` forward 调用 C++ 扩展
`modulated_deform_conv2d`，没有可直接复用的标准 ONNX symbolic。

当前 production 路径使用 Qualcomm CenterNet 公开的 DCNv2 改写结构并
泛化到 UniAD camera-folded N=6：
  explicit padding
  -> p0/pk + learned offsets
  -> four-corner Gather
  -> manual bilinear interpolation
  -> modulation mask
  -> sampled-patch reshape
  -> ordinary Conv2d

旧 GridSample reference 已归档到冻结 Git；本文件只保留当前 production DCNv2。

数学语义:
  原始 DCNv2 对每个输出位置 (h, w) 和卷积核位置 (p_h, p_w):
    p_n = (h + p_h*dilation, w + p_w*dilation)        # 基准采样点
    p_n' = p_n + offset(h, w, p_h, p_w)               # 加偏移
    w_pn = weight(p_h, p_w) * mask(h, w, p_h, p_w)    # 调制权重
    out(h,w) = sum_p w_pn * input(p_n')                # 双线性插值采样

当前 UniAD production 支持实际 backbone 配置: 3x3, stride1, pad1,
dilation1, deform_groups=1, groups=1。
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def _to_pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)




def _qualcomm_calculate_p0(h, w, stride_h, stride_w, *, device, dtype):
    """Qualcomm CenterNet calculate_p0, generalized only for device/dtype."""
    p0_y, p0_x = torch.meshgrid(
        torch.arange(0, h * stride_h, stride_h, device=device, dtype=dtype),
        torch.arange(0, w * stride_w, stride_w, device=device, dtype=dtype),
        indexing="ij",
    )
    p0_y = p0_y.view(1, 1, h, w)
    p0_x = p0_x.view(1, 1, h, w)
    return torch.cat([p0_y, p0_x], dim=1)


def _qualcomm_calculate_pk(
    kernel_h, kernel_w, dilation_h, dilation_w, *, device, dtype
):
    """Qualcomm CenterNet calculate_pk, generalized only for device/dtype."""
    pk_y, pk_x = torch.meshgrid(
        torch.arange(
            0,
            kernel_h * dilation_h,
            step=dilation_h,
            device=device,
            dtype=dtype,
        ),
        torch.arange(
            0,
            kernel_w * dilation_w,
            step=dilation_w,
            device=device,
            dtype=dtype,
        ),
        indexing="ij",
    )
    pk_y = pk_y.reshape(-1, 1, 1, 1)
    pk_x = pk_x.reshape(-1, 1, 1, 1)
    return torch.cat([pk_y, pk_x], dim=1)


def _qualcomm_gather_nhwc(input_nhwc, y, x):
    """Batch-preserving equivalent of Qualcomm's B=1 advanced indexing."""
    n, h, w, c = input_nhwc.shape
    flat = input_nhwc.reshape(n, h * w, c)
    index = (y * w + x).reshape(n, -1)
    gathered = torch.gather(
        flat,
        1,
        index.unsqueeze(-1).expand(-1, -1, c),
    )
    return gathered.reshape(*y.shape, c)


def _qualcomm_bilinear_sample_batched(input_tensor, coords):
    """Qualcomm four-corner bilinear sampler generalized from B=1 to B=N.

    Official CenterNet shape:
        input [1,C,H,W], coords [K,2,Hout,Wout].
    UniAD generalization:
        input [N,C,H,W], coords [N,K,2,Hout,Wout].

    The interpolation arithmetic, clamping and four-corner order remain the
    Qualcomm implementation; only the batch dimension is retained and Gather
    replaces B=1 advanced indexing.
    """
    n, _, h, w = input_tensor.shape
    coords = coords.permute(0, 1, 3, 4, 2)
    coords_y_fp = coords[..., 0]
    coords_x_fp = coords[..., 1]
    coords_y_floor = torch.floor(coords_y_fp)
    coords_x_floor = torch.floor(coords_x_fp)

    y0 = coords_y_floor.to(torch.long).clamp(0, h - 1)
    y1 = (coords_y_floor.to(torch.long) + 1).clamp(0, h - 1)
    x0 = coords_x_floor.to(torch.long).clamp(0, w - 1)
    x1 = (coords_x_floor.to(torch.long) + 1).clamp(0, w - 1)

    diff_y = coords_y_fp - coords_y_floor
    diff_x = coords_x_fp - coords_x_floor
    diff_y_inv = 1.0 - diff_y
    diff_x_inv = 1.0 - diff_x

    wa = diff_x_inv * diff_y_inv
    wd = diff_x * diff_y
    wc = diff_x_inv * diff_y
    wb = diff_x * diff_y_inv

    input_nhwc = input_tensor.permute(0, 2, 3, 1)
    ia = _qualcomm_gather_nhwc(input_nhwc, y0, x0)
    ib = _qualcomm_gather_nhwc(input_nhwc, y0, x1)
    ic = _qualcomm_gather_nhwc(input_nhwc, y1, x0)
    id_ = _qualcomm_gather_nhwc(input_nhwc, y1, x1)

    return (
        wa.unsqueeze(-1) * ia
        + wb.unsqueeze(-1) * ib
        + wc.unsqueeze(-1) * ic
        + wd.unsqueeze(-1) * id_
    )


def modulated_deform_conv2d_qualcomm_generalized(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    """Qualcomm CenterNet DCNv2 algorithm generalized to UniAD N=B*num_cam.

    Direct algorithm source:
      qualcomm/ai-hub-models@1bc4be97a9c87f0aaed18451f8a4146130cef243
      src/qai_hub_models/models/templates/centernet/model_patches.py
      calculate_p0 / calculate_pk / bilinear_sample / custom_deformconv2d

    Qualcomm's public implementation assumes B=1 through squeeze(0),
    offset.view(K,2,...) and B-free advanced indexing. UniAD reaches the
    backbone as camera-folded N=6, so this version preserves the official
    arithmetic while carrying N through coordinates, four-corner Gather,
    mask application, sampled-patch layout and final ordinary Conv2d.

    This is the promoted Q2 production implementation. Focused PT validation
    passed for N=1/N=6, border/mask cases and UniAD stage3/stage4 spatial
    shapes; the production wrapper installation gate is also recorded as PASS.
    """
    stride_h, stride_w = _to_pair(stride)
    pad_h, pad_w = _to_pair(padding)
    dil_h, dil_w = _to_pair(dilation)
    kernel_h, kernel_w = weight.shape[-2:]
    n, c_in, h_in, w_in = input.shape
    _, offset_channels, h_out, w_out = offset.shape
    k = kernel_h * kernel_w

    if groups != 1:
        raise ValueError("Q2 UniAD candidate currently requires groups=1")
    if deform_groups != 1:
        raise ValueError(
            "Q2 UniAD candidate currently requires deform_groups=1"
        )
    if offset_channels != deform_groups * 2 * k:
        raise ValueError("unexpected DCNv2 offset channel count")
    if mask.shape != (n, deform_groups * k, h_out, w_out):
        raise ValueError("unexpected DCNv2 mask shape")

    expected_h = (
        h_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    expected_w = (
        w_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if h_out != expected_h or w_out != expected_w:
        raise ValueError("offset spatial shape does not match DCNv2 geometry")

    input_padded = F.pad(
        input,
        (pad_w, pad_w, pad_h, pad_h),
        mode="constant",
        value=0,
    )

    p0 = _qualcomm_calculate_p0(
        h_out,
        w_out,
        stride_h,
        stride_w,
        device=input.device,
        dtype=input.dtype,
    ).unsqueeze(1)
    pk = _qualcomm_calculate_pk(
        kernel_h,
        kernel_w,
        dil_h,
        dil_w,
        device=input.device,
        dtype=input.dtype,
    ).unsqueeze(0)

    # Preserve the existing/MMCV channel ABI: offset is interpreted as
    # [N, K, 2, Hout, Wout], exactly as Qualcomm B=1 offset.view(K,2,...).
    coords = p0 + pk + offset.reshape(n, k, 2, h_out, w_out)
    sampled = _qualcomm_bilinear_sample_batched(input_padded, coords)

    sampled = sampled * mask.reshape(
        n, k, h_out, w_out
    ).unsqueeze(-1)

    # Qualcomm B=1 layout generalized without changing K row-major order:
    # [N,K,Hout,Wout,C]
    # -> [N,Hout*kH,Wout*kW,C]
    # -> ordinary Conv2d(kernel=kH/kW, stride=kH/kW).
    sampled = sampled.reshape(
        n, kernel_h, kernel_w, h_out, w_out, c_in
    )
    sampled = sampled.permute(0, 3, 1, 4, 2, 5).reshape(
        n, h_out * kernel_h, w_out * kernel_w, c_in
    )
    sampled = sampled.permute(0, 3, 1, 2)

    return F.conv2d(
        sampled,
        weight,
        bias,
        stride=(kernel_h, kernel_w),
        groups=groups,
    )


class ExportableModulatedDeformConv2d(nn.Module):
    """可导出的 ModulatedDeformConv2d (不含 conv_offset)。

    包装一个已训练的 mmcv ModulatedDeformConv2d 的 weight/bias，
    forward 使用已通过 Q2 PT gate 的 Qualcomm generalized DCNv2。
    """

    def __init__(self, src_conv):
        super().__init__()
        self.stride = src_conv.stride
        self.padding = src_conv.padding
        self.dilation = src_conv.dilation
        self.groups = src_conv.groups
        self.deform_groups = src_conv.deform_groups
        # 直接复用原 weight/bias (共享引用, 不复制)
        self.weight = src_conv.weight
        self.bias = src_conv.bias

    def forward(self, x, offset, mask):
        return modulated_deform_conv2d_qualcomm_generalized(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups,
        )


class ExportableModulatedDeformConv2dPack(nn.Module):
    """可导出的 ModulatedDeformConv2dPack (含 conv_offset)。

    替换 mmcv.ops.ModulatedDeformConv2dPack，保留 conv_offset，
    deformable conv 本体使用 Qualcomm generalized DCNv2。
    """

    def __init__(self, src_pack):
        super().__init__()
        self.stride = src_pack.stride
        self.padding = src_pack.padding
        self.dilation = src_pack.dilation
        self.groups = src_pack.groups
        self.deform_groups = src_pack.deform_groups
        self.kernel_size = src_pack.kernel_size
        # conv_offset 是普通 Conv2d, 本身可导出, 直接复用
        self.conv_offset = src_pack.conv_offset
        # deformable conv 本体用可导出版本
        self.deform_conv = ExportableModulatedDeformConv2d(src_pack)

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask)


def replace_dcnv2_with_exportable(model):
    """递归把模型里所有 ModulatedDeformConv2dPack 替换为可导出版本。

    在 load_checkpoint 之后调用。替换后 weight/conv_offset 权重共享。
    Q2 focused PT 已与旧 GridSample reference 和 MMCV DCNv2 交叉验证；
    当前 production 支持 UniAD 实际 groups=1、deform_groups=1 配置。
    """
    from mmcv.ops import ModulatedDeformConv2dPack
    replaced = 0
    for name, module in model.named_children():
        if isinstance(module, ModulatedDeformConv2dPack):
            new_mod = ExportableModulatedDeformConv2dPack(module)
            setattr(model, name, new_mod)
            replaced += 1
        else:
            replaced += replace_dcnv2_with_exportable(module)
    return replaced
