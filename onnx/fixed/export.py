"""Production model build/export entrypoint and existing-graph bundle command.

Fixed Q3 export binds the candidate to the accepted mini reference and keeps
source relocation evidence separate from model acceptance. build_model only
constructs/patches the model; export.py and export.py bundle are the two CLIs.
"""
import argparse
import copy
import signal
import ast
import subprocess
import hashlib
import json
import os
import os.path as osp
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import onnx
from onnx import helper as H, numpy_helper as N

warnings.filterwarnings("ignore")
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "..", "..")))


def _mock_unused_imports():
    """Mock 掉导出不需要的重依赖 (pytorch_lightning → tensorboard → protobuf 冲突)。"""
    import types
    pl = types.ModuleType("pytorch_lightning")
    pl_m = types.ModuleType("pytorch_lightning.metrics")
    pl_mm = types.ModuleType("pytorch_lightning.metrics.metric")
    pl_f = types.ModuleType("pytorch_lightning.metrics.functional")
    pl_fc = types.ModuleType("pytorch_lightning.metrics.functional.classification")
    pl_fr = types.ModuleType("pytorch_lightning.metrics.functional.reduction")

    class _Metric:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return None
    pl_mm.Metric = _Metric
    pl_fc.stat_scores_multiple_classes = lambda *a, **kw: None
    pl_fr.reduce = lambda *a, **kw: None
    pl_m.metric = pl_mm
    pl_m.functional = pl_f
    pl_f.classification = pl_fc
    pl_f.reduction = pl_fr
    pl.metrics = pl_m
    for n, m in [
        ("pytorch_lightning", pl),
        ("pytorch_lightning.metrics", pl_m),
        ("pytorch_lightning.metrics.metric", pl_mm),
        ("pytorch_lightning.metrics.functional", pl_f),
        ("pytorch_lightning.metrics.functional.classification", pl_fc),
        ("pytorch_lightning.metrics.functional.reduction", pl_fr),
    ]:
        sys.modules[n] = m


_mock_unused_imports()

# Import the deployment graph only after the unused heavy dependencies above
# have been stubbed; this ordering is part of the existing build environment.
from temporal import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    StatefulStep,
    TensorTrackStateCycle,
    clone_state_for_input,
    dynamic_axes,
    FixedRecurrentStatefulStep,
    TRACK_STATE_NAMES,
)


def build_cfg(config_path, seg_msda_mode):
    from mmcv import Config
    cfg = Config.fromfile(config_path)
    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_path = _module_dir.split("/")[0]
        for m in _module_dir.split("/")[1:]:
            _module_path = _module_path + "." + m
        importlib.import_module(_module_path)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if seg_msda_mode == "single_level":
        seg_transformer = cfg.model.seg_head.transformer
        seg_transformer.num_feature_levels = 1
        seg_transformer.encoder.transformerlayers.attn_cfgs.num_levels = 1
        seg_transformer.decoder.transformerlayers.attn_cfgs[1].num_levels = 1
    return cfg


def register_onnx_symbolics():
    """注册当前 PyTorch 2.0.1 缺失/不足的 ONNX symbolics。

    - atan2: 复刻新版 PyTorch 官方 TorchScript exporter 的象限完整分解;
    - bool __iand__/__ior__: 调用点均为 bool mask, 用标准 And/Or;
    - SDPA: 复刻新版 PyTorch 的 MatMul/Softmax/MatMul 分解, 与 rank 无关;
    - unflatten: Shape/Slice/Concat/Reshape 分解。
    全部为标准 ONNX 算子, 不引入自定义 domain。
    """
    import numpy as np
    from torch.onnx import register_custom_op_symbolic as _register_sym
    from torch.onnx.symbolic_helper import parse_args as _parse_args

    @_parse_args("v", "v")
    def _atan2_symbolic(g, y, x):
        atan = g.op("Atan", g.op("Div", y, x))
        zero = g.op("Constant", value_t=torch.tensor(0.0))
        pi = g.op("Constant", value_t=torch.tensor(np.pi))
        second_or_third = g.op(
            "Where", g.op("Greater", y, zero),
            g.op("Add", atan, pi), g.op("Sub", atan, pi),
        )
        return g.op("Where", g.op("Less", x, zero), second_or_third, atan)

    _register_sym("aten::atan2", _atan2_symbolic, 9)

    # 当前调用点语义均为 bool 逻辑 (0/1 标志), 用 ONNX 标准逻辑算子 And/Or。
    # 注意 motion_head.filter_vehicle_query 的 vehicle_mask |= (labels == id):
    # vehicle_mask 是 zeros_like(labels) 的 int64, PyTorch 按位或提升为 int64。
    # 若直接 Or(int64, bool), ORT 类型检查拒绝 (Or 只接受 bool);
    # 但若把输出直接变成 bool, 下游 vehicle_mask > 0 的 Greater 又会收到 bool。
    # 因此: 两侧 Cast->bool 做逻辑运算, 再 Cast 回左侧原 dtype (保持数据流类型)。
    def _logic_in_place(g, self, other, op):
        self_dtype = self.type().scalarType()
        a = self if self_dtype == "Bool" else g.op("Cast", self, to_i=9)
        b = other if other.type().scalarType() == "Bool" else g.op("Cast", other, to_i=9)
        out = g.op(op, a, b)
        if self_dtype != "Bool":
            out = g.op("Cast", out, to_i=7)  # 回 int64, 与原 in-place 张量 dtype 一致
            out.setType(self.type())
        return out

    @_parse_args("v", "v")
    def _iand_symbolic(g, self, other):
        return _logic_in_place(g, self, other, "And")
    _register_sym("aten::__iand_", _iand_symbolic, 9)

    @_parse_args("v", "v")
    def _ior_symbolic(g, self, other):
        return _logic_in_place(g, self, other, "Or")
    _register_sym("aten::__ior_", _ior_symbolic, 9)

    # SDPA (scaled_dot_product_attention) 在 opset 20 才有, torch 2.0.1 最高 opset 18
    # 需手动注册 SDPA symbolic: softmax(Q@K^T/sqrt(d) + mask) @ V
    # trace 中 K 的 rank 可能不可知 (type().sizes() 为 None), 因此不依赖
    # Transpose 的静态 perm, 统一把前导维 collapse 成 3D 做 batched MatMul,
    # 再恢复 Q 的前导维。不支持 causal/dropout 时显式报错,
    # 避免静默导出语义错误的图。
    def _last_two(g, shape_tensor):
        rank_1d = g.op(
            "Unsqueeze", g.op("Size", shape_tensor),
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.long)),
        )
        start = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.long))
        return g.op("Slice", shape_tensor, start, rank_1d)

    @_parse_args("v", "v", "v", "v", "f", "b")
    def _sdpa_symbolic(g, q, k, v, attn_mask, dropout_p, is_causal):
        assert not is_causal, "SDPA symbolic: is_causal=True 未实现"
        assert dropout_p == 0.0, "SDPA symbolic: 导出仅支持 eval (dropout_p=0)"
        minus_one = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        neg_two = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.long))

        q_shape = g.op("Shape", q)
        k_shape = g.op("Shape", k)
        v_shape = g.op("Shape", v)
        # (..., S, D) -> (-1, S, D): 前导维合并, 与 rank 无关
        q3 = g.op("Reshape", q, g.op("Concat", minus_one, _last_two(g, q_shape), axis_i=0))
        k3 = g.op("Reshape", k, g.op("Concat", minus_one, _last_two(g, k_shape), axis_i=0))
        v3 = g.op("Reshape", v, g.op("Concat", minus_one, _last_two(g, v_shape), axis_i=0))
        k3t = g.op("Transpose", k3, perm_i=[0, 2, 1])
        scores = g.op("MatMul", q3, k3t)
        # 缩放 1/sqrt(d): d = Q 最后一维 (head_dim)
        head_dim = g.op("Gather", q_shape, minus_one)
        scale = g.op("Reciprocal", g.op("Sqrt", g.op("Cast", head_dim, to_i=1)))
        scores = g.op("Mul", scores, scale)
        # NoneType 的 Value 调 scalarType 会触发 JIT 断言, 先判类型;
        # mask 实际是 None (NoneType) 时视同无 mask
        mask_dtype = None
        if attn_mask is not None:
            try:
                mask_dtype = attn_mask.type().scalarType()
            except RuntimeError:
                mask_dtype = None
        if mask_dtype is not None:
            if mask_dtype == "Bool":
                # bool mask: True 表示保留, 转为加性 0/-inf
                neg_inf = g.op("Constant", value_t=torch.tensor(float("-inf")))
                zero_f = g.op("Constant", value_t=torch.tensor(0.0))
                attn_mask = g.op("Where", attn_mask, zero_f, neg_inf)
            # 加性 mask 同样 collapse 前导维, 保证与 scores 的 3D 形状广播一致
            # (本导出 batch=1, 广播安全)
            m_shape = g.op("Shape", attn_mask)
            attn_mask = g.op(
                "Reshape", attn_mask,
                g.op("Concat", minus_one, _last_two(g, m_shape), axis_i=0),
            )
            scores = g.op("Add", scores, attn_mask)
        attn = g.op("Softmax", scores, axis_i=-1)
        out3 = g.op("MatMul", attn, v3)
        # 恢复 q 的前导维 + V 的最后两维
        q_leading = g.op("Slice", q_shape, zero, neg_two)
        out_shape = g.op("Concat", q_leading, _last_two(g, v_shape), axis_i=0)
        return g.op("Reshape", out3, out_shape)
    _register_sym("aten::scaled_dot_product_attention", _sdpa_symbolic, 13)

    # 注册 unflatten 的 symbolic, 用 Reshape 替代
    # unflatten(input, dim, sizes) → reshape(input, shape[:dim]+sizes+shape[dim+1:])
    # sizes 在 MultiheadAttention 里可能是动态 tensor (含 seq_len), 用 "v" 接受
    @_parse_args("v", "i", "v")
    def _unflatten_symbolic(g, self, dim, sizes):
        self_shape = g.op("Shape", self)
        if dim < 0:
            rank = self.type().dim()
            if rank is None:
                return _sym_helper._unimplemented(
                    "aten::unflatten", "negative dim with unknown input rank"
                )
            dim += rank
        if dim == 0:
            before = g.op("Constant", value_t=torch.tensor([], dtype=torch.long))
        else:
            before = g.op("Slice", self_shape,
                          g.op("Constant", value_t=torch.tensor([0], dtype=torch.long)),
                          g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long)))
        # shape[dim+1:]: end 用 shape 的 rank, 需 unsqueeze 成 1D
        self_rank = g.op("Size", self_shape)  # 0D
        self_rank_1d = g.op("Unsqueeze", self_rank,
                            g.op("Constant", value_t=torch.tensor([0], dtype=torch.long)))
        after_start = g.op("Constant", value_t=torch.tensor([dim + 1], dtype=torch.long))
        after = g.op("Slice", self_shape, after_start, self_rank_1d)
        new_shape = g.op("Concat", before, sizes, after, axis_i=0)
        return g.op("Reshape", self, new_shape)
    _register_sym("aten::unflatten", _unflatten_symbolic, 13)

    # torch 2.0.1 的 aten::mean (含 mean.dim) 在 opset 18 有两个问题:
    # 1) opset 18 起 ReduceMean 的 axes 已改为输入, 而官方 symbolic 仍写成属性
    #    (ONNX checker 拒绝);
    # 2) torch 自带的 ReduceMean 形状传播不知道 axes 是输入, 会把输出标成
    #    "全部约减"的形状, 与 ONNX shape inference 冲突, setType 也救不回来。
    # 因此不产出 ReduceMean, 按数学定义分解为 Div(ReduceSum(axes), count):
    # ReduceSum 的 axes-input 形态在 torch 2.0.1/opset 18 已验证正常。
    # count 用 Shape+Gather+ReduceProd 动态构造, 与 rank/静态形状无关。
    from torch.onnx._internal.registration import registry as _sym_registry
    from torch.onnx import symbolic_helper as _sym_helper

    def _mean_decompose(g, self, dim=None, keepdim=0):
        shape = g.op("Shape", self)
        if dim is None:  # 全部约减
            summed = g.op("ReduceSum", self, keepdims_i=keepdim)
            count = g.op("ReduceProd", shape, keepdims_i=0)
        else:
            axes = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
            summed = g.op("ReduceSum", self, axes, keepdims_i=keepdim)
            dim_sizes = g.op("Gather", shape, axes)
            count = g.op("ReduceProd", dim_sizes, keepdims_i=0)
        return g.op("Div", summed, g.op("Cast", count, to_i=1))

    def _mean_symbolic(g, *args):
        if len(args) == 4:  # mean.dim(self, dim, keepdim, dtype)
            @_parse_args("v", "is", "b", "none")
            def _mean_dim(g, self, dim, keepdim, dtype):
                return _mean_decompose(g, self, dim, keepdim)
            return _mean_dim(g, *args)
        if len(args) == 2:  # mean(self, dtype) 全约减
            @_parse_args("v", "none")
            def _mean_all(g, self, dtype):
                return _mean_decompose(g, self)
            return _mean_all(g, *args)
        return _sym_helper._unimplemented("aten::mean", f"{len(args)} args")
    _sym_registry.register("aten::mean", 18, _mean_symbolic)

    # torch 2.0.1 的 aten::max / aten::min (dim 形式) 与 mean 同病:
    # ReduceMax/ReduceMin 的 axes 在 opset 18 已改为输入, 官方 symbolic 仍写属性
    # (checker 拒绝), 且 torch 形状传播对 axes-输入形态的 Reduce* 会标错形状。
    # 因此 dim 形式不产出 ReduceMax/ReduceMin, 分解为 ArgMax/ArgMin + GatherElements:
    #   idx_keep = ArgMax(self, axis=dim, keepdims=1)   (axis 仍是属性, 合法)
    #   values   = GatherElements(self, idx_keep, axis=dim)  (取回被选元素本身)
    # values 数值精确 (就是原张量元素), 且与 indices 严格同源;
    # ONNX ArgMax 默认 select_last_index=0 (取第一个最大值), 与 PyTorch 一致。
    # 全约减 (ReduceMax 无 axes) 和 max(x, y) (Max) 两种形态在 opset 18 合法, 沿用。
    def _max_min_dim_decompose(g, self, dim, keepdim, is_max):
        arg_op = "ArgMax" if is_max else "ArgMin"
        idx_keep = g.op(arg_op, self, axis_i=dim, keepdims_i=1)
        values = g.op("GatherElements", self, idx_keep, axis_i=dim)
        if keepdim:
            return values, idx_keep
        axes = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
        return g.op("Squeeze", values, axes), g.op("Squeeze", idx_keep, axes)

    def _max_symbolic(g, self, dim_or_y=None, keepdim=None):
        if dim_or_y is None and keepdim is None:  # torch.max(input) 全约减
            return g.op("ReduceMax", self, keepdims_i=0)
        if keepdim is None:  # torch.max(input, other) 逐元素
            return g.op("Max", self, dim_or_y)
        dim = _sym_helper._get_const(dim_or_y, "i", "dim")
        keepdim = _sym_helper._get_const(keepdim, "i", "keepdim")
        return _max_min_dim_decompose(g, self, dim, keepdim, True)

    def _min_symbolic(g, self, dim_or_y=None, keepdim=None):
        if dim_or_y is None and keepdim is None:  # torch.min(input) 全约减
            return g.op("ReduceMin", self, keepdims_i=0)
        if keepdim is None:  # torch.min(input, other) 逐元素
            return g.op("Min", self, dim_or_y)
        dim = _sym_helper._get_const(dim_or_y, "i", "dim")
        keepdim = _sym_helper._get_const(keepdim, "i", "keepdim")
        return _max_min_dim_decompose(g, self, dim, keepdim, False)

    _sym_registry.register("aten::max", 18, _max_symbolic)
    _sym_registry.register("aten::min", 18, _min_symbolic)


def build_model(config, checkpoint):
    from mmdet3d.models import build_model as build
    from mmcv.runner import load_checkpoint
    from model_patches import patch_model_for_onnx_export
    from dcnv2 import replace_dcnv2_with_exportable
    from attention import patch_msda_for_legacy_cuda
    cfg = build_cfg(config, "legacy_cuda")
    model = build(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu", strict=True)
    model.eval()
    patch_model_for_onnx_export(model)
    replace_dcnv2_with_exportable(model)
    patch_msda_for_legacy_cuda()
    register_onnx_symbolics()
    return cfg, model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_initial_state_bundle(wrapper, onnx_path, output, config, checkpoint):
    """Write the host recurrent initialization bundle for this exact ONNX."""
    from temporal import TRACK_STATE_NAMES, clone_state_for_input

    model_path = Path(onnx_path).resolve()
    output = Path(output).resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing initial-state bundle: "
            f"{output} / {metadata_path}"
        )
    initial = clone_state_for_input(wrapper.cycle.initial_state())
    state = {
        name: value.detach().cpu().numpy()
        for name, value in zip(TRACK_STATE_NAMES, initial)
    }
    # TRACK_STATE_NAMES is the recurrent state ABI and must stay aligned with
    # clone_state_for_input(). Host-only recurrent fields are added below.
    if len(state) != len(initial):
        raise ValueError("initial track-state ABI length mismatch")
    model = wrapper.model
    state.update(
        prev_bev=np.zeros(
            (model.bev_h * model.bev_w, 1, 256), dtype=np.float32
        ),
        max_obj_id=np.array(0, dtype=np.int64),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **state)
    metadata = {
        "contract": "stateful-v1",
        "onnx_sha256": sha256(model_path),
        "state_sha256": sha256(output),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config),
        "onnx_path": str(model_path),
        "state_path": str(output),
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in state.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata_path, metadata


# -----------------------------------------------------------------------------
# Stateful export CLI
# -----------------------------------------------------------------------------
# Base synthetic inputs remain available to the frozen refactor verifier.
# The fixed CLI below owns source-bound export and fixed initialization bundles.


def make_inputs(wrapper, seed=0):
    model = wrapper.model
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 6, 3, 928, 1600, generator=generator)
    bus = torch.zeros(1, 18)
    bus[0, 3] = 1.
    return (image, bus, torch.eye(3)[None], torch.eye(4)[None, None].repeat(1, 6, 1, 1),
            torch.tensor([[[928, 1600]] * 6]), torch.zeros(model.bev_h*model.bev_w, 1, 256),
            torch.tensor([0]), torch.tensor(False), torch.eye(3), torch.zeros(3), torch.zeros(3),
            torch.tensor(0.), torch.tensor(0), *clone_state_for_input(wrapper.cycle.initial_state()))


def advance(inputs, outputs):
    old = dict(zip(INPUT_NAMES, inputs))
    result = dict(zip(OUTPUT_NAMES, outputs))
    old.update(prev_bev=result["bev_embed"], has_prev_bev=torch.tensor(True),
               prev_l2g_r=old["l2g_r_mat"][0], prev_l2g_t=old["l2g_t"],
               time_delta=torch.tensor(.5), max_obj_id=result["next_max_obj_id"])
    for name in INPUT_NAMES[13:]:
        old[name] = result["next_" + name]
    return tuple(old[name] for name in INPUT_NAMES)



def lower_bool_where(model):
    """Replace BOOL Where with broadcast-equivalent And/Not/Or operators.

    ORT 1.19 CPU has no bool-valued Where kernel. For every boolean triple,
    Where(c, a, b) == (c & a) | (~c & b), including broadcast inputs. Tensor
    names at the graph boundary and original initializers are preserved.
    The input ModelProto is mutated only after inference/type validation.
    """
    import onnx
    from onnx import TensorProto, helper

    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    used = set()

    def collect(graph):
        used.update(
            value.name for value in (*graph.input, *graph.output, *graph.value_info)
        )
        used.update(value.name for value in graph.initializer)
        for node in graph.node:
            used.update((*node.input, *node.output, node.name))
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect(attribute.g)

    collect(model.graph)
    changes = []
    plans = []

    def inspect(original, typed, inherited, depth=0):
        types = dict(inherited)
        types.update(
            (value.name, value.type.tensor_type.elem_type)
            for value in (*typed.input, *typed.output, *typed.value_info)
        )
        types.update((value.name, value.data_type) for value in typed.initializer)
        if len(original.node) != len(typed.node):
            raise ValueError("shape inference changed graph structure")
        for index, (node, typed_node) in enumerate(zip(original.node, typed.node)):
            if node.op_type == "Where" and node.domain in ("", "ai.onnx"):
                output_type = types.get(node.output[0])
                if output_type is None:
                    raise ValueError(f"cannot classify Where output: {node.name}")
                if output_type == TensorProto.BOOL:
                    if len(node.input) != 3 or len(node.output) != 1 or node.attribute:
                        raise ValueError(f"unexpected Where schema: {node.name}")
                    if any(types.get(name) != TensorProto.BOOL for name in node.input):
                        raise ValueError(
                            f"bool Where input type missing/different: {node.name}"
                        )
                    plans.append((original, index, node, depth))
            typed_attributes = {
                attribute.name: attribute for attribute in typed_node.attribute
            }
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    inspect(
                        attribute.g,
                        typed_attributes[attribute.name].g,
                        types,
                        depth + 1,
                    )

    inspect(model.graph, inferred.graph, {})
    del inferred

    def fresh(base):
        name = base
        suffix = 0
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        return name

    replacements = {}
    for serial, (graph, index, node, depth) in enumerate(plans):
        prefix = f"__q3_bool_where_{serial}"
        first = fresh(prefix + "_first")
        inverse = fresh(prefix + "_inverse")
        second = fresh(prefix + "_second")
        condition, yes, no = node.input
        nodes = [
            helper.make_node(
                "And", [condition, yes], [first], name=fresh(prefix + "_and_first")
            ),
            helper.make_node(
                "Not", [condition], [inverse], name=fresh(prefix + "_not")
            ),
            helper.make_node(
                "And", [inverse, no], [second], name=fresh(prefix + "_and_second")
            ),
            helper.make_node("Or", [first, second], list(node.output), name=node.name),
        ]
        key = id(graph)
        if key not in replacements:
            replacements[key] = (graph, {}, depth)
        replacements[key][1][index] = nodes
        changes.append(
            {
                "name": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
    for graph, by_index, _ in sorted(
        replacements.values(), key=lambda item: item[2], reverse=True
    ):
        nodes = []
        for index, node in enumerate(graph.node):
            nodes.extend(by_index.get(index, [node]))
        del graph.node[:]
        graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return {
        "recipe": "bool Where(c,a,b) -> Or(And(c,a),And(Not(c),b))",
        "rewritten_count": len(changes),
        "rewritten_nodes": changes,
        "claim_limit": "boolean representation equivalence; not full PT/ORT/Host acceptance",
    }

def append_sca_diagnostics(model, *, capacity=None):
    """Append true pre-TopK visibility counts and the Host overflow contract.

    This audits the full camera/depth reduction route. Symbolic query extent is
    recorded, not replaced: the sum covers the complete ranking vector at run
    time, while proof of the configured extent remains a separate static gate.
    """
    import onnx
    from onnx import TensorProto as T, helper as H, numpy_helper as N
    from state_contract import SCA_CAPACITY, SCA_CAMERAS, SCA_BEV_QUERIES, SCA_OUTPUT_NAMES

    capacity = SCA_CAPACITY if capacity is None else capacity
    if capacity != SCA_CAPACITY:
        raise ValueError("recipe requires the frozen SCA capacity")
    graph = model.graph
    used = {v.name for v in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
    used.update(x for node in graph.node for x in (*node.input, *node.output, node.name))
    if any(name in used for name in SCA_OUTPUT_NAMES):
        raise ValueError("SCA diagnostics already present or names collide")
    constants = {v.name: N.to_array(v) for v in graph.initializer if np.prod(v.dims) <= 16}
    producers = {value: node for node in graph.node for value in node.output}
    for node in graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR and np.prod(attribute.t.dims) <= 16:
                    constants[node.output[0]] = N.to_array(attribute.t)
    def attr(node, name, default=None):
        return next((H.get_attribute_value(a) for a in node.attribute if a.name == name), default)
    def constant(name, expected):
        return name in constants and np.array_equal(constants[name], np.asarray(expected, dtype=np.int64))
    selected = [node for node in graph.node if node.op_type == "TopK"
                and node.name.startswith("/step/encoder/TopK") and constant(node.input[1], [capacity])]
    expected = ["/step/encoder/TopK"] + [f"/step/encoder/TopK_{i}" for i in range(1, SCA_CAMERAS)]
    if [node.name for node in selected] != expected:
        raise ValueError("reviewed six camera TopK nodes missing/different")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    typed = {v.name: v.type.tensor_type for v in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)}
    shared_split = None
    inferred_extents = []
    for camera, topk in enumerate(selected):
        score = typed[topk.input[0]]
        if score.elem_type != T.FLOAT or len(score.shape.dim) != 1:
            raise ValueError(f"camera ranking type/rank differs: {topk.name}")
        dim = score.shape.dim[0]
        if dim.HasField("dim_value") and dim.dim_value != SCA_BEV_QUERIES:
            raise ValueError("camera ranking query extent differs from configured grid")
        inferred_extents.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
        reduction = producers[topk.input[0]]
        gather = producers[reduction.input[0]]
        squeeze = producers[gather.input[0]]
        split = producers[squeeze.input[0]]
        if (reduction.op_type != "ReduceSum" or attr(reduction, "keepdims") != 0
                or len(reduction.input) != 2 or not constant(reduction.input[1], [-1])
                or gather.op_type != "Gather" or attr(gather, "axis", 0) != 0
                or not constant(gather.input[1], 0)
                or squeeze.op_type != "Squeeze" or not constant(squeeze.input[1], [0])
                or split.op_type != "Split" or attr(split, "axis", 0) != 0
                or not constant(split.input[1], [1] * SCA_CAMERAS)
                or squeeze.input[0] != split.output[camera]):
            raise ValueError(f"camera/depth/full-vector route differs: {topk.name}")
        if shared_split is not None and split.name != shared_split.name:
            raise ValueError("camera rankings do not share the full mask")
        shared_split = split
        cast = producers[split.input[0]]
        mask_type = typed[cast.input[0]]
        mask_dims = mask_type.shape.dim
        if (cast.op_type != "Cast" or attr(cast, "to") != T.FLOAT
                or mask_type.elem_type != T.BOOL or len(mask_dims) != 4
                or [mask_dims[i].dim_value for i in (0, 1, 3)] != [SCA_CAMERAS, 1, 4]):
            raise ValueError("full SCA mask is not six-camera boolean depth geometry")
    for name in ("survivor_overflow", "vehicle_overflow"):
        value = typed[name]
        if value.elem_type != T.BOOL or len(value.shape.dim) != 0:
            raise ValueError("existing overflow flags must be scalar bool")
    del inferred

    def fresh(name):
        if name in used:
            raise ValueError(f"diagnostic tensor/node name collision: {name}")
        used.add(name)
        return name
    zero, limit, axis = [fresh(name) for name in ("__q3_sca_zero", "__q3_sca_capacity", "__q3_sca_axis0")]
    added_constants = [N.from_array(np.array(0, np.float32), name=zero),
                       N.from_array(np.array(capacity, np.int64), name=limit),
                       N.from_array(np.array([0], np.int64), name=axis)]
    nodes, counts = [], []
    for camera, source in enumerate(selected):
        prefix = f"__q3_sca_cam{camera}"
        visible, integers, raw, expanded = [fresh(prefix + suffix) for suffix in ("_visible", "_visible_i64", "_raw", "_raw_1d")]
        nodes.extend([
            H.make_node("Greater", [source.input[0], zero], [visible], name=fresh(prefix+"_greater")),
            H.make_node("Cast", [visible], [integers], to=T.INT64, name=fresh(prefix+"_cast")),
            H.make_node("ReduceSum", [integers, axis], [raw], keepdims=0, name=fresh(prefix+"_sum")),
            H.make_node("Unsqueeze", [raw, axis], [expanded], name=fresh(prefix+"_expand")),
        ])
        counts.append(expanded)
    raw_counts, sca_overflow, aggregate = [fresh(name) for name in SCA_OUTPUT_NAMES]
    maximum = fresh("__q3_sca_maximum")
    nodes.extend([
        H.make_node("Concat", counts, [raw_counts], axis=0, name=fresh("__q3_sca_counts")),
        H.make_node("ReduceMax", [raw_counts, axis], [maximum], keepdims=0, name=fresh("__q3_sca_max")),
        H.make_node("Greater", [maximum, limit], [sca_overflow], name=fresh("__q3_sca_overflow")),
    ])
    flags = []
    for index, source in enumerate(("survivor_overflow", "vehicle_overflow", sca_overflow)):
        name = fresh(f"__q3_overflow_flag_{index}")
        nodes.append(H.make_node("Unsqueeze", [source, axis], [name], name=fresh(f"__q3_overflow_expand_{index}")))
        flags.append(name)
    nodes.append(H.make_node("Concat", flags, [aggregate], axis=0, name=fresh("__q3_overflow_flags")))
    graph.initializer.extend(added_constants)
    graph.node.extend(nodes)
    graph.output.extend([H.make_tensor_value_info(raw_counts, T.INT64, [SCA_CAMERAS]),
                         H.make_tensor_value_info(sca_overflow, T.BOOL, []),
                         H.make_tensor_value_info(aggregate, T.BOOL, [3])])
    onnx.checker.check_model(model)
    return {"recipe": "full pre-TopK positive depth-score count; three Host flags",
            "capacity": capacity, "camera_count": SCA_CAMERAS,
            "configured_bev_queries": SCA_BEV_QUERIES, "inferred_rank_extents": inferred_extents,
            "ranking_input_names": [node.input[0] for node in selected],
            "full_boolean_mask": producers[shared_split.input[0]].input[0],
            "added_node_names": [node.name for node in nodes], "output_names": list(SCA_OUTPUT_NAMES),
            "flag_order": ["survivor", "vehicle", "sca"],
            "claim_limit": "preselection counts cover full runtime vectors; static extent and real Host acceptance remain separate"}


def tensor_dimensions(value):
    return [d.dim_value if d.HasField('dim_value') else d.dim_param or None for d in value.type.tensor_type.shape.dim]

def shape_value_is_complete(a):
    return a.dtype != object or not any((v is None for v in a.flat))

def small_graph_constants(g):
    c = {v.name: N.to_array(v) for v in g.initializer if np.prod(v.dims) <= 4096 and v.data_type in [1, 7, 9, 6]}
    for n in g.node:
        if n.op_type == 'Constant':
            for a in n.attribute:
                if a.name == 'value' and np.prod(a.t.dims) <= 4096:
                    c[n.output[0]] = N.to_array(a.t)
    return c

def specialize_static_shape_controls(model, checkpoint=lambda *a, **k: None):
    m = copy.deepcopy(model)
    report = []
    for iteration in range(30):
        m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=True)
        g = m.graph
        c = small_graph_constants(g)
        producers = {o: n for n in g.node for o in n.output}
        shapes = {v.name: [d.dim_value if d.HasField('dim_value') else None for d in v.type.tensor_type.shape.dim] for v in list(g.input) + list(g.value_info) + list(g.output) if v.type.tensor_type.HasField('shape')}
        changes = []
        nodes = []
        for n in g.node:
            if n.op_type == 'Einsum' and all((i in shapes and all((d is not None for d in shapes[i])) for i in n.input)):
                eq = next((a.s.decode() for a in n.attribute if a.name == 'equation'))
                (left, right) = eq.split('->')
                letters = {}
                assert len(left.split(',')) == len(n.input) and '.' not in eq
                for (term, i) in zip(left.split(','), n.input):
                    local = {}
                    assert len(term) == len(shapes[i])
                    for (char, dim) in zip(term, shapes[i]):
                        assert char not in local or local[char] == dim
                        local[char] = dim
                        assert char not in letters or letters[char] == dim or letters[char] == 1 or (dim == 1)
                        letters[char] = max(letters.get(char, 1), dim)
                tensor_dimensions = [letters[char] for char in right]
                if shapes.get(n.output[0]) != tensor_dimensions:
                    vals = {v.name: v for v in list(g.input) + list(g.value_info) + list(g.output)}
                    dtype = vals[n.input[0]].type.tensor_type.elem_type
                    out = H.make_tensor_value_info(n.output[0], dtype, tensor_dimensions)
                    if n.output[0] in vals:
                        vals[n.output[0]].CopyFrom(out)
                    else:
                        g.value_info.append(out)
                    shapes[n.output[0]] = tensor_dimensions
                    changes.append({'einsum_shape': n.name, 'equation': eq, 'inputs': [shapes[i] for i in n.input], 'output': tensor_dimensions})
            at = {a.name: H.get_attribute_value(a) for a in n.attribute}
            args = [c.get(i) for i in n.input if i]
            v = None
            try:
                if n.op_type == 'Shape' and n.input[0] in shapes:
                    v = np.array(shapes[n.input[0]], dtype=object)
                elif args and all((a is not None for a in args)):
                    if n.op_type == 'Gather':
                        v = np.take(args[0], args[1].astype(np.int64), axis=at.get('axis', 0))
                    elif n.op_type == 'Slice':
                        slices = [slice(None)] * args[0].ndim
                        axes = args[3] if len(args) > 3 else np.arange(len(args[1]))
                        steps = args[4] if len(args) > 4 else np.ones(len(args[1]), np.int64)
                        for (ax, st, en, step) in zip(axes, args[1], args[2], steps):
                            slices[int(ax)] = slice(int(st), int(en), int(step))
                        v = args[0][tuple(slices)]
                    elif n.op_type == 'Concat':
                        v = np.concatenate(args, axis=at['axis'])
                    elif n.op_type == 'Unsqueeze':
                        v = args[0]
                        for ax in sorted(args[1].astype(np.int64)):
                            v = np.expand_dims(v, int(ax))
                    elif n.op_type == 'Squeeze':
                        v = np.squeeze(args[0], axis=tuple((int(x) for x in args[1])) if len(args) > 1 else None)
                    elif n.op_type == 'Reshape' and shape_value_is_complete(args[1]):
                        target = [int(x) for x in args[1]]
                        if not at.get('allowzero', 0):
                            target = [args[0].shape[i] if x == 0 else x for (i, x) in enumerate(target)]
                        v = np.reshape(args[0], tuple(target))
                    elif all((shape_value_is_complete(a) for a in args)):
                        if n.op_type == 'ConstantOfShape' and np.prod(args[0]) <= 4096:
                            scalar = N.to_array(at['value']) if 'value' in at else np.array([0], np.float32)
                            v = np.full(tuple((int(x) for x in args[0])), scalar.item(), scalar.dtype)
                        elif n.op_type == 'Expand' and np.prod(args[1]) <= 4096:
                            v = np.broadcast_to(args[0], tuple((int(x) for x in args[1])))
                        elif n.op_type == 'Transpose':
                            v = np.transpose(args[0], at.get('perm'))
                        elif n.op_type == 'Range' and abs((args[1].item() - args[0].item()) / args[2].item()) <= 4096:
                            v = np.arange(args[0].item(), args[1].item(), args[2].item(), dtype=args[0].dtype)
                        elif n.op_type == 'Mod':
                            v = np.fmod(*args) if at.get('fmod', 0) else np.mod(*args)
                        elif n.op_type == 'Equal':
                            v = np.equal(*args)
                        elif n.op_type == 'Where':
                            v = np.where(*args)
                        elif n.op_type == 'Add':
                            v = args[0] + args[1]
                        elif n.op_type == 'Sub':
                            v = args[0] - args[1]
                        elif n.op_type == 'Mul':
                            v = args[0] * args[1]
                        elif n.op_type == 'Div':
                            (left, right) = [a.astype(np.int64) if a.dtype == object else a for a in args]
                            if left.dtype.kind in 'iu':
                                v = np.floor_divide(left, right)
                                if left.dtype.kind == 'i':
                                    v = v + (((left < 0) != (right < 0)) & (np.remainder(left, right) != 0))
                            else:
                                v = left / right
                        elif n.op_type == 'Cast':
                            v = args[0].astype(H.tensor_dtype_to_np_dtype(at['to']))
                        elif n.op_type == 'Identity':
                            v = args[0]
                        elif n.op_type == 'ReduceProd':
                            v = np.prod(args[0], axis=tuple((int(x) for x in args[1])) if len(args) > 1 else None, keepdims=bool(at.get('keepdims', 1)))
            except (ValueError, TypeError, IndexError):
                v = None
            if v is not None and len(n.output) == 1 and (np.asarray(v).size <= 4096):
                c[n.output[0]] = np.asarray(v)
            if n.op_type == 'If' and n.input[0] in c and shape_value_is_complete(c[n.input[0]]):
                val = c[n.input[0]]
                assert val.size == 1
                branch = at['then_branch' if bool(val.item()) else 'else_branch']
                assert len(branch.output) == len(n.output)
                if any((x.op_type not in ['Constant', 'Squeeze', 'Identity'] for x in branch.node)):
                    raise ValueError('only shape squeeze branches may specialize')
                nodes.extend(copy.deepcopy(branch.node))
                nodes.extend((H.make_node('Identity', [a.name], [b], name=n.name + '/static_output') for (a, b) in zip(branch.output, n.output)))
                changes.append({'if': n.name, 'condition': bool(val.item())})
                continue
            if n.op_type in ['Reshape', 'Expand', 'ConstantOfShape', 'Tile', 'Resize', 'Pad']:
                indices = {'Reshape': [1], 'Expand': [1], 'ConstantOfShape': [0], 'Tile': [1], 'Resize': [2, 3], 'Pad': [1]}[n.op_type]
                for i in indices:
                    if i >= len(n.input) or n.input[i] not in c:
                        continue
                    v = c[n.input[i]]
                    if not shape_value_is_complete(v):
                        continue
                    if v.dtype == object:
                        v = v.astype(np.int64)
                    producer = producers.get(n.input[i])
                    if producer is not None and producer.op_type == 'Constant':
                        continue
                    name = n.name + '/static_shape_' + str(i)
                    nodes.append(H.make_node('Constant', [], [name], name=name, value=N.from_array(v)))
                    changes.append({'node': n.name, 'index': i, 'previous_input': n.input[i], 'value': v.tolist(), 'dtype': str(v.dtype)})
                    n.input[i] = name
            nodes.append(n)
        del g.node[:]
        g.node.extend(nodes)
        report.extend((dict(iteration=iteration, **x) for x in changes))
        checkpoint('shape_propagation', iteration=iteration, changes=len(changes), remaining_ifs=sum((n.op_type == 'If' for n in nodes)))
        if not changes:
            break
    else:
        raise ValueError('shape specialization did not converge')
    m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True, data_prop=True)
    return (m, report)

def verify_shape_specialization_preserves_math(source, derived, report):
    original = {n.name: n for n in source.graph.node}
    altered = {x['node'] for x in report if 'node' in x}
    removed = {x['if'] for x in report if 'if' in x}
    current = {n.name: n for n in derived.graph.node}
    for (name, node) in original.items():
        if name in removed:
            if node.op_type != 'If' or name in current:
                raise ValueError('invalid If specialization')
            continue
        changed = current[name]
        if name in altered:
            if node.op_type not in ['Reshape', 'Expand', 'ConstantOfShape', 'Tile', 'Resize', 'Pad']:
                raise ValueError('non-shape operand changed')
            restored = copy.deepcopy(changed)
            indices = {x['index'] for x in report if x.get('node') == name}
            for i in indices:
                restored.input[i] = node.input[i]
            if restored.SerializeToString() != node.SerializeToString():
                raise ValueError('shape rewrite changed arithmetic')
        elif changed.SerializeToString() != node.SerializeToString():
            raise ValueError('unapproved old-node change: ' + name)
    if len(source.graph.initializer) != len(derived.graph.initializer) or any((a.SerializeToString() != b.SerializeToString() for (a, b) in zip(source.graph.initializer, derived.graph.initializer))):
        raise ValueError('weights changed')
    if [v.SerializeToString() for v in source.graph.input] != [v.SerializeToString() for v in derived.graph.input]:
        raise ValueError('input ABI changed')
    if [(v.name, v.type.tensor_type.elem_type) for v in source.graph.output] != [(v.name, v.type.tensor_type.elem_type) for v in derived.graph.output]:
        raise ValueError('output name/type ABI changed')
    return {'unchanged_weights_inputs_and_output_names_types': True, 'old_node_math_preserved': True, 'shape_operand_nodes': len(altered), 'specialized_ifs': len(removed)}

ROOT = Path(__file__).resolve().parents[2]
EXTRA_IN = ["track_count", "track_valid_mask"]
EXTRA_OUT = [
    "next_track_count",
    "next_track_valid_mask",
    "survivor_count_raw",
    "survivor_overflow",
    "decoded_count",
    "decoded_valid_mask",
    "vehicle_count",
    "vehicle_valid_mask",
    "vehicle_count_raw",
    "vehicle_overflow",
]
FIXED_INPUT_NAMES = INPUT_NAMES + EXTRA_IN
FIXED_OUTPUT_NAMES = OUTPUT_NAMES + EXTRA_OUT
from state_contract import SCA_OUTPUT_NAMES
FIXED_GRAPH_OUTPUT_NAMES = FIXED_OUTPUT_NAMES + list(SCA_OUTPUT_NAMES)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()



def verify_review_hardening_source_identity(fixed_dir, manifest, mini_root):
    """Bind checkpoint/entrypoint guards to focused acceptance and frozen math."""
    import importlib.util
    promotion = manifest["review_hardening_promotion"]
    previous = Path(promotion["previous_fixed_dir"])
    accepted = Path(promotion["accepted_candidate_dir"])
    if sha(previous / "SOURCE_MANIFEST.json") != promotion["previous_manifest_sha256"]:
        raise ValueError("pre-hardening manifest changed")
    if sha(accepted / "SOURCE_MANIFEST.json") != promotion["candidate_manifest_sha256"]:
        raise ValueError("hardening candidate manifest changed")
    if set(promotion["previous_files"]) != set(promotion["candidate_files"]) or len(promotion["previous_files"]) != 13:
        raise ValueError("hardening lineage must retain the corresponding 13 files")
    for name, digest in promotion["previous_files"].items():
        if sha(previous / name) != digest:
            raise ValueError("pre-hardening source changed: " + name)
    entry = promotion["accepted_evidence"]
    if sha(entry["path"]) != entry["sha256"] or sha(entry["script_path"]) != entry["script_sha256"]:
        raise ValueError("checkpoint hardening evidence changed")
    evidence = json.loads(Path(entry["path"]).read_text())
    if (evidence["status"] != "pass" or evidence["execution_status"] != "complete" or not evidence["checks"] or not all(evidence["checks"].values())
            or evidence["candidate_files"] != promotion["candidate_files"] or evidence["candidate_manifest_sha256"] != promotion["candidate_manifest_sha256"]
            or evidence["script_sha256"] != entry["script_sha256"]):
        raise ValueError("checkpoint hardening acceptance is incomplete")
    spec = importlib.util.spec_from_file_location("frozen_pre_hardening_export_lineage", previous / "export.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    inherited = frozen.verify_mini_fixed_sources(mini_root)
    if set(manifest["files"]) != {p.name for p in fixed_dir.glob("*.py")} or len(manifest["files"]) != 13:
        raise ValueError("fixed deployment surface must retain the corresponding 13 files")
    checks = {}
    for name, digest in manifest["files"].items():
        if sha(fixed_dir / name) != digest or sha(accepted / name) != promotion["candidate_files"][name]:
            raise ValueError("hardening source differs from accepted lineage: " + name)
        before, after = ast.parse((accepted / name).read_text()), ast.parse((fixed_dir / name).read_text())
        if name == "export.py":
            helpers = [n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "verify_review_hardening_source_identity"]
            if len(helpers) != 1:
                raise ValueError("unexpected hardening export helpers")
            after.body.remove(helpers[0])
            node = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "verify_mini_fixed_sources")
            expected = ast.parse('if "review_hardening_promotion" in manifest: return verify_review_hardening_source_identity(fixed_dir, manifest, mini_root)').body[0]
            branches = [n for n in node.body if ast.dump(n) == ast.dump(expected)]
            if len(branches) != 1:
                raise ValueError("hardening lineage dispatch differs")
            node.body.remove(branches[0])
        if ast.dump(before) != ast.dump(after):
            raise ValueError("accepted inference/export definitions changed: " + name)
        checks[name] = dict(canonical_sha256=digest, accepted_hardening_sha256=promotion["candidate_files"][name],
                            previous_canonical_sha256=inherited[name]["canonical_sha256"], mode="accepted_context_guards_with_frozen_math_lineage")
    return checks


def verify_production_source_identity(fixed_dir, manifest, mini_root):
    """Follow immutable metric lineage, then bind the accepted production additions."""
    import importlib.util
    promotion = manifest["production_promotion"]
    previous = Path(promotion["previous_fixed_dir"])
    accepted = Path(promotion["accepted_candidate_dir"])
    if sha(previous / "SOURCE_MANIFEST.json") != promotion["previous_manifest_sha256"]:
        raise ValueError("historical canonical manifest changed")
    if sha(accepted / "SOURCE_MANIFEST.json") != promotion["candidate_manifest_sha256"]:
        raise ValueError("accepted production candidate manifest changed")
    for name, digest in promotion["previous_files"].items():
        if sha(previous / name) != digest:
            raise ValueError("historical canonical source changed: " + name)
    for entry in promotion["accepted_evidence"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("accepted production evidence changed")
        evidence = json.loads(Path(entry["path"]).read_text())
        if evidence["status"] != "pass" or not all(evidence["checks"].values()):
            raise ValueError("production acceptance not passed")
    spec = importlib.util.spec_from_file_location("frozen_validation_export_lineage", previous / "export.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    inherited = frozen.verify_mini_fixed_sources(mini_root)
    checks = {}
    if set(manifest["files"]) != {p.name for p in fixed_dir.glob("*.py")} or len(manifest["files"]) != 13:
        raise ValueError("production surface must retain the corresponding 13 files")
    for name, digest in manifest["files"].items():
        if sha(fixed_dir / name) != digest or sha(accepted / name) != promotion["candidate_files"][name]:
            raise ValueError("production source differs from its accepted lineage: " + name)
        before, after = ast.parse((accepted / name).read_text()), ast.parse((fixed_dir / name).read_text())
        if name == "export.py":
            definitions = {n.name: n for n in after.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            original = {n.name: n for n in before.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            if set(definitions) != set(original) | {"verify_production_source_identity"}:
                raise ValueError("unexpected production export definitions")
            node = definitions["verify_mini_fixed_sources"]
            branches = [i for i, n in enumerate(node.body) if isinstance(n, ast.If) and ast.dump(n.test) == ast.dump(ast.parse('"production_promotion" in manifest', mode="eval").body)]
            if len(branches) != 1:
                raise ValueError("production lineage dispatch differs")
            branch = node.body.pop(branches[0])
            expected = ast.parse('if "production_promotion" in manifest: return verify_production_source_identity(fixed_dir, manifest, mini_root)').body[0]
            if ast.dump(branch) != ast.dump(expected):
                raise ValueError("production lineage dispatch changes export behavior")
            for function in original:
                if ast.dump(original[function]) != ast.dump(definitions[function]):
                    raise ValueError("accepted export/math definition changed: " + function)
        elif ast.dump(before) != ast.dump(after):
            raise ValueError("accepted production module changed: " + name)
        checks[name] = dict(canonical_sha256=digest, accepted_production_sha256=promotion["candidate_files"][name], historical_canonical_sha256=inherited[name]["canonical_sha256"], mode="production_additions_with_frozen_math_lineage")
    return checks


def verify_mini_fixed_sources(mini_root):
    """Bind canonical math to the candidate that passed the frozen mini gates."""
    fixed_dir = Path(__file__).resolve().parent
    manifest = json.loads((fixed_dir / "SOURCE_MANIFEST.json").read_text())
    if "review_hardening_promotion" in manifest:
        return verify_review_hardening_source_identity(fixed_dir, manifest, mini_root)
    if "production_promotion" in manifest:
        return verify_production_source_identity(fixed_dir, manifest, mini_root)
    lineage = manifest["canonical_promotion"]
    accepted = Path(lineage["accepted_metrics_result"])
    if Path(mini_root).resolve() != accepted.parent or sha(accepted) != lineage["accepted_metrics_result_sha256"]:
        raise ValueError("accepted mini result binding differs")
    candidate = Path(lineage["accepted_candidate_dir"])
    if sha(candidate / "SOURCE_MANIFEST.json") != lineage["candidate_manifest_sha256"]:
        raise ValueError("accepted source manifest changed")
    if set(manifest["files"]) != {p.name for p in fixed_dir.glob("*.py")} or len(manifest["files"]) != 13:
        raise ValueError("fixed deployment surface must contain 13 Python files")
    import re
    inverse = {v: k for k, v in {**lineage["function_rename_map"], **lineage["attribute_rename_map"]}.items()}
    pattern = re.compile(r"\b(" + "|".join(re.escape(x) for x in inverse) + r")\b")
    checks = {}
    for filename, digest in manifest["files"].items():
        if sha(fixed_dir / filename) != digest or sha(candidate / filename) != lineage["candidate_files"][filename]:
            raise ValueError("canonical or accepted source changed: " + filename)
        old = ast.parse((candidate / filename).read_text())
        actual = ast.parse(pattern.sub(lambda m: inverse[m.group()], (fixed_dir / filename).read_text()))
        if filename == "export.py":
            before = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            after = {n.name: n for n in actual.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            names = manifest["canonical_export_integration"]["preserved_definitions"]
            if any(ast.dump(before[name]) != ast.dump(after[name]) for name in names):
                raise ValueError("accepted model/export math changed")
        elif ast.dump(old) != ast.dump(actual):
            raise ValueError("change beyond functional naming: " + filename)
        checks[filename] = {"accepted_sha256": lineage["candidate_files"][filename], "canonical_sha256": digest,
                            "mode": "preserved_model_export_definitions" if filename == "export.py" else "normalized_ast_identity"}
    proof = manifest["canonical_export_integration"]
    shape_path = Path(proof["shape_algorithm_source"])
    if sha(shape_path) != proof["shape_algorithm_sha256"]:
        raise ValueError("accepted shape proof source changed")
    original = {n.name: n for n in ast.parse(shape_path.read_text()).body if isinstance(n, ast.FunctionDef)}
    inverse_shape = {v: k for k, v in proof["shape_function_rename_map"].items()}
    shape_pattern = re.compile(r"\b(" + "|".join(re.escape(x) for x in inverse_shape) + r")\b")
    normalized = ast.parse(shape_pattern.sub(lambda m: inverse_shape[m.group()], (fixed_dir / "export.py").read_text()))
    current = {n.name: n for n in normalized.body if isinstance(n, ast.FunctionDef)}
    for name in proof["shape_function_rename_map"]:
        if ast.dump(original[name]) != ast.dump(current[name]):
            raise ValueError("integrated shape algorithm changed: " + name)
    return checks


def mini_preflight(mini_root):
    mini_root = mini_root.resolve()
    sources = verify_mini_fixed_sources(mini_root)
    manifest = json.loads((Path(__file__).parent / "SOURCE_MANIFEST.json").read_text())
    lineage = manifest["canonical_promotion"]
    result = json.loads((mini_root / "result.json").read_text())
    pipeline = json.loads((mini_root / "pipeline_status.json").read_text())
    audit = json.loads((mini_root / "completion_audit.json").read_text())
    digest = sha(mini_root / "result.json")
    if not (pipeline.get("status") == "pass" and pipeline.get("execution_status") == "complete"
            and pipeline.get("overall_pass") is True and pipeline.get("result_sha256") == digest
            and audit.get("status") == "pass" and audit.get("result_sha256") == digest
            and all(audit["checks"].values())
            and sha(mini_root / "completion_audit.json") == lineage["accepted_metrics_audit_sha256"]):
        raise ValueError("latest three-way mini acceptance is not complete and hash-bound")
    for backend in ["pt", "ort"]:
        binding = result["identity"]["runs"][backend]
        if binding["fixed_sources"] != lineage["candidate_files"]:
            raise ValueError("accepted backend source differs: " + backend)
        run = Path(binding["path"])
        if sha(run / backend / "result.json") != binding["result_sha256"] or sha(run / "completion_audit.json") != binding["audit_sha256"]:
            raise ValueError("accepted backend result differs: " + backend)
    for filename, digest in result["identity"]["original_sources_sha256"].items():
        if sha(ROOT / filename) != digest:
            raise ValueError("frozen original source changed: " + filename)
    reference = manifest["canonical_export_integration"]["accepted_export"]
    export_run = Path(reference["path"])
    if sha(export_run / "result.json") != reference["result_sha256"]:
        raise ValueError("accepted full export binding differs")
    exported = json.loads((export_run / "result.json").read_text())
    if exported["model_sha256"] != reference["model_sha256"] or sha(exported["derived_model"]) != reference["model_sha256"]:
        raise ValueError("accepted full graph changed")
    assets = {k: exported["identity"][k] for k in ["config", "checkpoint", "motion_anchor"]}
    for key, asset in assets.items():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("accepted asset changed: " + key)
    return {"mini_run": str(mini_root), "mini_result_sha256": sha(mini_root / "result.json"),
            "mini_audit_sha256": sha(mini_root / "completion_audit.json"),
            "fixed_sources": manifest["files"], "mini_reference_source_checks": sources,
            "source_manifest_sha256": sha(Path(__file__).parent / "SOURCE_MANIFEST.json"),
            "mini_acceptance_scope": "Accepted candidate task metrics; new canonical graph requires its own gates",
            "accepted_export": reference, "export_script_sha256": sha(__file__), "source_commit": head(), **assets}


def fixed_inputs(wrapper):
    base = make_inputs(wrapper)[:13]
    state = wrapper.initial_fixed_state()
    result = (*base, *state)
    if len(result) != len(FIXED_INPUT_NAMES):
        raise ValueError("fixed input count differs from ABI")
    return result


def advance_fixed(inputs, outputs):
    old = dict(zip(FIXED_INPUT_NAMES, inputs))
    result = dict(zip(FIXED_OUTPUT_NAMES, outputs))
    old.update(
        prev_bev=result["bev_embed"],
        has_prev_bev=torch.tensor(True),
        prev_l2g_r=old["l2g_r_mat"][0],
        prev_l2g_t=old["l2g_t"],
        time_delta=torch.tensor(0.5),
        max_obj_id=result["next_max_obj_id"],
        track_count=result["next_track_count"],
        track_valid_mask=result["next_track_valid_mask"],
    )
    for name in TRACK_STATE_NAMES:
        old[name] = result["next_" + name]
    return tuple(old[name] for name in FIXED_INPUT_NAMES)


def check_outputs(inputs, outputs):
    if len(outputs) != len(FIXED_OUTPUT_NAMES):
        raise ValueError("fixed output count differs from ABI")
    result = dict(zip(FIXED_OUTPUT_NAMES, outputs))
    for name, value in result.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"nonfinite {name}")
    count = int(result["next_track_count"])
    survivor = int(result["survivor_count_raw"])
    decoded = int(result["decoded_count"])
    vehicle = int(result["vehicle_count"])
    vehicle_raw = int(result["vehicle_count_raw"])
    if not (
        901 <= count <= 1285
        and count == 901 + min(survivor, 384)
        and 0 <= decoded <= 300
        and 0 <= vehicle <= 96
        and vehicle == min(vehicle_raw, 96)
        and not bool(result["survivor_overflow"])
        and not bool(result["vehicle_overflow"])
    ):
        raise ValueError("fixed count/overflow sanity failed")
    for name, length, valid in (
        ("next_track_valid_mask", 1285, count),
        ("decoded_valid_mask", 300, decoded),
        ("vehicle_valid_mask", 96, vehicle),
    ):
        mask = result[name]
        if mask.shape != (length,) or not torch.equal(
            mask, torch.arange(length, device=mask.device) < valid
        ):
            raise ValueError(f"{name} differs from packed count")
    return {
        "input_tracks": int(inputs[-2]),
        "next_tracks": count,
        "raw_survivors": survivor,
        "decoded": decoded,
        "vehicle": vehicle,
        "raw_vehicle": vehicle_raw,
        "output_shapes": {name: list(value.shape) for name, value in result.items()},
    }


def install_trace_safe_bool_padding(wrapper):
    """Keep the PT pack semantics while avoiding Torch 2.0.1 bool full_like tracing."""

    def pack_field(field, safe_index, valid, fill):
        chosen = field[safe_index]
        shape = (valid.shape[0],) + (1,) * (chosen.dim() - 1)
        if type(fill) is bool:
            padding = torch.ones_like(chosen) if fill else torch.zeros_like(chosen)
        else:
            padding = torch.full_like(chosen, fill)
        return torch.where(valid.reshape(shape), chosen, padding)

    wrapper.cycle._pack_field = pack_field


class FixedGraphStep(torch.nn.Module):
    """Bind track_count to the graph mask; Host rejects mismatches before inference."""

    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, *inputs):
        count, valid = inputs[-2:]
        count_prefix = torch.arange(valid.shape[0], device=valid.device) < count
        effective_valid = valid & count_prefix
        return self.step(*inputs[:-1], effective_valid)


def write_bundle(wrapper, model_path, run_root, identity):
    state = {
        name: value.detach().cpu().numpy().copy()
        for name, value in zip(
            (*TRACK_STATE_NAMES, *EXTRA_IN), wrapper.initial_fixed_state()
        )
    }
    state.update(
        prev_bev=np.zeros((40000, 1, 256), np.float32),
        max_obj_id=np.array(0, np.int64),
    )
    path = run_root / "initial_state.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **state)
    metadata = {
        "contract": "q3-fixed-state-candidate-v2",
        "model_sha256": sha(model_path),
        "state_sha256": sha(path),
        "config_sha256": identity["config"]["sha256"],
        "checkpoint_sha256": identity["checkpoint"]["sha256"],
        "fields": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        },
    }
    save(run_root / "initial_state.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.frames < 3:
        parser.error("threads >= 1 and at least three sanity frames required")
    if args.preflight_only:
        identity = mini_preflight(args.mini_run)
        print(json.dumps({"status": "preflight_pass", **identity}))
        return
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "stage": "building",
        "pid": os.getpid(),
        "started_utc": stamp(),
        "identity": None,
        "input_names": FIXED_INPUT_NAMES,
        "output_names": FIXED_GRAPH_OUTPUT_NAMES,
        "pt_output_names": FIXED_OUTPUT_NAMES,
        "output_profile": "validation_full_47",
        "production_output_profile_frozen": False,
        "frames": [],
    }

    def checkpoint(stage, **values):
        record.update(stage=stage, **values)
        record.update(execution_status="running" if record["status"] == "running" else "complete",
                      overall_pass=False)
        save(run_root / "status.json", record)
        if record.get("execution_status") == "complete":
            final = dict(record)
            final.pop("result_sha256", None)
            save(run_root / "result.json", final)
            record["result_sha256"] = sha(run_root / "result.json")
            save(run_root / "status.json", record)
        save(run_root / "pipeline_status.json", record)
        print(json.dumps({"stage": stage, "pid": os.getpid()}), flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"canonical export received signal {signum}")

    previous_signal = signal.signal(signal.SIGTERM, interrupt)
    try:
        checkpoint("preflight")
        identity = mini_preflight(args.mini_run)
        record["identity"] = identity
        save(run_root / "launch.json", {"argv": sys.argv, "cwd": str(Path.cwd()), "python": sys.executable,
             "pid": os.getpid(), "pgid": os.getpgid(0), "start_ticks": Path("/proc/self/stat").read_text().split()[21],
             "identity": identity, "started_utc": record["started_utc"]})
        checkpoint("building")
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        cfg, model = build_model(
            identity["config"]["path"], identity["checkpoint"]["path"]
        )
        wrapper = FixedRecurrentStatefulStep(model, cfg.occflow_grid_conf).eval().cpu()
        with torch.no_grad():
            inputs = fixed_inputs(wrapper)
            trace_inputs = inputs
            reference_outputs = None
            later_inputs = None
            for frame in range(args.frames):
                checkpoint("sanity_forward", current_frame=frame)
                if frame == args.frames - 1:
                    later_inputs = inputs
                outputs = wrapper(*inputs)
                if frame == 0:
                    reference_outputs = tuple(
                        value.detach().clone() for value in outputs
                    )
                record["frames"].append(check_outputs(inputs, outputs))
                checkpoint("sanity_frame_passed")
                inputs = advance_fixed(inputs, outputs)
        checkpoint("sanity_passed")
        checkpoint("export_compat_parity")
        install_trace_safe_bool_padding(wrapper)
        with torch.no_grad():
            compatible_outputs = wrapper(*trace_inputs)
        if len(compatible_outputs) != len(reference_outputs) or any(
            not torch.equal(before, after)
            for before, after in zip(reference_outputs, compatible_outputs)
        ):
            raise ValueError("trace-safe bool padding changed fixed PT outputs")
        graph_wrapper = FixedGraphStep(wrapper).eval()
        with torch.no_grad():
            graph_first = graph_wrapper(*trace_inputs)
            later_step = wrapper(*later_inputs)
            graph_later = graph_wrapper(*later_inputs)
        if any(
            not torch.equal(before, after)
            for before, after in zip(compatible_outputs, graph_first)
        ) or any(
            not torch.equal(before, after)
            for before, after in zip(later_step, graph_later)
        ):
            raise ValueError("graph count-mask binding changed valid PT outputs")
        checkpoint(
            "export_compat_parity_passed",
            export_compat_patch="bool full_like to ones_like/zeros_like in fixed state packing",
            same_input_outputs_exact=True,
            graph_count_mask_binding_exact=True,
            count_mask_input_validation="Host pre-session required",
        )
        del reference_outputs, compatible_outputs, graph_first, later_step, graph_later
        if args.sanity_only:
            checkpoint(
                "complete",
                status="sanity_pass",
                finished_utc=stamp(),
                q3_6_internal_graph_audited=False,
                ort_validated=False,
            )
            return
        model_path = run_root / "model.onnx"
        checkpoint("exporting")
        with torch.enable_grad():
            torch.onnx.export(
                graph_wrapper,
                trace_inputs,
                str(model_path),
                opset_version=18,
                input_names=FIXED_INPUT_NAMES,
                output_names=FIXED_OUTPUT_NAMES,
                dynamic_axes=None,
                do_constant_folding=True,
            )
        checkpoint("checking")
        import onnx

        onnx.checker.check_model(str(model_path))
        model_proto = onnx.load(str(model_path), load_external_data=False)

        checkpoint("lowering_bool_where")
        raw_path = run_root / "model.raw.onnx"
        if raw_path.exists():
            raise FileExistsError(raw_path)
        os.replace(model_path, raw_path)
        lowering = lower_bool_where(model_proto)
        sca_diagnostics = append_sca_diagnostics(model_proto)
        compatible_path = run_root / "model.compat.onnx"
        onnx.save(model_proto, str(compatible_path))
        checkpoint("specializing_static_shape_controls")
        compatible_model = model_proto
        model_proto, proof = specialize_static_shape_controls(compatible_model, checkpoint)
        preservation = verify_shape_specialization_preserves_math(compatible_model, model_proto, proof)
        save(run_root / "shape_proof.json", proof)
        onnx.save(model_proto, str(model_path))
        checkpoint(
            "checking",
            graph_compat=lowering,
            sca_diagnostics=sca_diagnostics,
            graph_compat_sha256=sha(__file__),
            raw_model_sha256=sha(raw_path),
        )
        onnx.checker.check_model(str(model_path))
        graph = model_proto.graph
        actual_inputs = [value.name for value in graph.input]
        actual_outputs = [value.name for value in graph.output]
        if actual_inputs != FIXED_INPUT_NAMES or actual_outputs != FIXED_GRAPH_OUTPUT_NAMES:
            checkpoint(
                "interface_rejected",
                actual_inputs=actual_inputs,
                actual_outputs=actual_outputs,
                missing_inputs=[x for x in FIXED_INPUT_NAMES if x not in actual_inputs],
                missing_outputs=[
                    x for x in FIXED_GRAPH_OUTPUT_NAMES if x not in actual_outputs
                ],
            )
            raise ValueError("exported ONNX interface differs from fixed ABI")
        if any(node.domain not in ("", "ai.onnx") for node in graph.node):
            raise ValueError("unexpected ONNX custom domain")
        shapes = {}
        for value in (*graph.input, *graph.output):
            dims = value.type.tensor_type.shape.dim
            shapes[value.name] = [
                int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
                for dim in dims
            ]
        if not all(isinstance(dim, int) for dims in shapes.values() for dim in dims):
            raise ValueError("canonical boundary retains dynamic extents")
        if any(n.op_type in ("Einsum", "If") for n in graph.node):
            raise ValueError("canonical graph retains Einsum or If")
        residual = [{"name": v.name, "shape": tensor_dimensions(v), "dtype": v.type.tensor_type.elem_type}
                    for v in graph.value_info if not v.type.tensor_type.HasField("shape")
                    or any(not isinstance(x, int) for x in tensor_dimensions(v))]
        save(run_root / "intermediate_shape_residual.json", {"unresolved_count": len(residual), "values": residual,
             "scope": "Unresolved metadata requires separate internal data-flow closure; not automatic dynamic-shape evidence."})
        checkpoint("loading_ort_session")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        runtime = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        for actual, expected in [(runtime.get_inputs(), graph.input), (runtime.get_outputs(), graph.output)]:
            if [(v.name, v.shape) for v in actual] != [(v.name, tensor_dimensions(v)) for v in expected]:
                raise ValueError("ORT session boundary differs from canonical static ABI")
        del runtime
        bundle = write_bundle(wrapper, model_path, run_root, identity)
        checkpoint(
            "complete",
            status="pass",
            acceptance_status="canonical_trace_checker_session_only",
            derived_model=str(model_path),
            compatible_model=str(compatible_path),
            compatible_model_sha256=sha(compatible_path),
            shape_proof_sha256=sha(run_root / "shape_proof.json"),
            shape_proof_entries=len(proof),
            preservation=preservation,
            internal_unresolved_metadata=len(residual),
            ort_session_loaded=True,
            fresh_ort_frame_execution=False,
            model_sha256=sha(model_path),
            model_bytes=model_path.stat().st_size,
            node_count=len(graph.node),
            interface_shapes=shapes,
            all_interface_dims_static=all(
                isinstance(dim, int) for dims in shapes.values() for dim in dims
            ),
            initial_state=bundle,
            finished_utc=stamp(),
            q3_6_internal_graph_audited=False,
            ort_validated=False,
        )
    except BaseException:
        checkpoint(
            "failed_or_interrupted",
            status="failed_or_interrupted",
            error=traceback.format_exc(),
            finished_utc=stamp(),
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


def bundle_main():
    """Create a fixed v2 initialization bundle for an existing reviewed graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-run", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.onnx.is_file() or args.threads < 1:
        parser.error("existing ONNX and threads >= 1 required")
    identity = mini_preflight(args.mini_run)
    import onnx
    graph = onnx.load(str(args.onnx), load_external_data=False).graph
    if ([value.name for value in graph.input] != FIXED_INPUT_NAMES
            or [value.name for value in graph.output] != FIXED_GRAPH_OUTPUT_NAMES):
        parser.error("existing graph differs from the fixed validation ABI")
    args.run_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    with torch.no_grad():
        cfg, model = build_model(identity["config"]["path"], identity["checkpoint"]["path"])
        wrapper = FixedRecurrentStatefulStep(model, cfg.occflow_grid_conf).eval().cpu()
        metadata = write_bundle(wrapper, args.onnx, args.run_root, identity)
    print(json.dumps({"status": "bundle_written", "run_root": str(args.run_root), **metadata}))


class PlanningBundleAcceptanceError(ValueError):
    """The derived graph or initialization failed its accepted identity gate."""


def derive_certified_planning_model(full, accepted, certificate, shape_proof):
    """Prune only dead paths, then attach the frozen root-derived extent proof."""
    keep = [v.name for v in accepted.graph.output]
    if len(full.graph.input) != 23 or len(full.graph.output) != 47 or len(keep) != 25:
        raise PlanningBundleAcceptanceError("expected full23/47 and planning23/25 ABI")
    model = copy.deepcopy(full)
    graph = model.graph
    producers = {out: i for i, node in enumerate(graph.node) for out in node.output if out}
    live, pending = set(), list(keep)
    while pending:
        i = producers.get(pending.pop())
        if i is not None and i not in live:
            live.add(i)
            pending.extend(graph.node[i].input)
    retained = [copy.deepcopy(n) for i, n in enumerate(graph.node) if i in live]
    del graph.node[:]; graph.node.extend(retained)
    outputs = [copy.deepcopy(v) for v in graph.output if v.name in set(keep)]
    del graph.output[:]; graph.output.extend(outputs)
    used = {name for node in graph.node for name in list(node.input) + list(node.output)} | set(keep) | {v.name for v in graph.input}
    weights = [copy.deepcopy(v) for v in graph.initializer if v.name in used]
    del graph.initializer[:]; graph.initializer.extend(weights)
    del graph.value_info[:]
    core = copy.deepcopy(accepted); del core.graph.value_info[:]
    if model.SerializeToString() != core.SerializeToString():
        raise PlanningBundleAcceptanceError("liveness derivative differs from accepted math/weights/ABI")
    names = [out for node in graph.node for out in node.output if out]
    extents = certificate["extents"]
    if names != list(extents) or len(names) != shape_proof["proven_node_outputs"]:
        raise PlanningBundleAcceptanceError("certificate does not cover every produced tensor in order")
    boundaries = {v.name for v in list(graph.input) + list(graph.output)}
    accepted_info = {v.name: v for v in accepted.graph.value_info}
    for name in names:
        item = extents[name]
        if name in boundaries:
            continue
        value = H.make_tensor_value_info(name, onnx.TensorProto.DataType.Value(item["dtype"]), item["shape"])
        if name not in accepted_info or value.SerializeToString() != accepted_info[name].SerializeToString():
            raise PlanningBundleAcceptanceError("extent certificate metadata differs: " + name)
        graph.value_info.append(value)
    if model.SerializeToString() != accepted.SerializeToString():
        raise PlanningBundleAcceptanceError("production model differs from accepted certified model")
    onnx.checker.check_model(model, full_check=True)
    return model


def create_planning_resource_bundle(spec, destination, checkpoint):
    """Create an atomic, relative-path bundle using accepted graph/state resources."""
    import gc
    import shutil
    import casadi
    import onnxruntime as ort
    import host
    import assets
    from state_contract import FRESH, TRACK_SLOTS, SURVIVOR_CAPACITY, DECODED_SLOTS, VEHICLE_SLOTS, SCA_CAPACITY
    source = Path(__file__).resolve().parent
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    source_files = {p.name: sha(p) for p in source.glob("*.py")}
    if source_files != manifest["files"] or len(source_files) != 13:
        raise ValueError("producer sources differ from the corresponding 13-file manifest")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=spec["project_root"], text=True).strip()
    for name, digest in source_files.items():
        relative = (source / name).relative_to(spec["project_root"])
        blob = subprocess.check_output(["git", "show", revision + ":" + str(relative)], cwd=spec["project_root"])
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("producer source differs from Git: " + name)
    def bound(role):
        entry = spec[role]
        path = Path(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError("production source artifact differs: " + role)
        return path
    paths = {name: bound(name) for name in ["full_model", "accepted_model", "initial_state", "shape_proof", "extent_certificate", "collision_optimizer"]}
    evidence = {}
    for role, entry in spec["evidence"].items():
        path = Path(entry["path"])
        if sha(path) != entry["sha256"]:
            raise ValueError("accepted evidence differs: " + role)
        audit = json.loads(path.read_text())
        if audit["status"] != "pass" or not audit.get("checks") or not all(audit["checks"].values()):
            raise ValueError("unaccepted bundle provenance: " + role)
        evidence[role] = entry["sha256"]
    proof = json.loads(paths["shape_proof"].read_text())
    certificate = json.loads(paths["extent_certificate"].read_text())
    if (proof["derived_model_sha256"] != spec["accepted_model"]["sha256"] or proof["extent_certificate_sha256"] != spec["extent_certificate"]["sha256"]
            or not proof["all_node_output_extents_proven"] or not proof["all_control_values_proven_constant"] or not proof["all_actual_control_arithmetic_exact"]):
        raise PlanningBundleAcceptanceError("shape certificate acceptance is incomplete")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("bundle destination already exists")
    stage = destination.parent / ".bundle-staging"
    stage.mkdir(exist_ok=False)
    checkpoint("deriving_production_planning_graph", source_commit=revision, producer_sources=source_files)
    full = onnx.load(str(paths["full_model"]), load_external_data=False)
    accepted = onnx.load(str(paths["accepted_model"]), load_external_data=False)
    if any(t.external_data for m in [full, accepted] for t in m.graph.initializer):
        raise ValueError("production baseline must have embedded weights")
    model = derive_certified_planning_model(full, accepted, certificate, proof)
    onnx.save(model, str(stage / "model.onnx"))
    if sha(stage / "model.onnx") != spec["accepted_model"]["sha256"]:
        raise PlanningBundleAcceptanceError("serialized production model is not byte-exact")
    def abi(values):
        return [dict(name=v.name, dtype=str(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type))),
                     shape=[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in values]
    inputs, outputs = abi(model.graph.input), abi(model.graph.output)
    node_count = len(model.graph.node)
    del model, accepted, full; gc.collect()
    checkpoint("copying_verified_runtime_and_resources")
    (stage / "runtime").mkdir(); (stage / "resources").mkdir()
    entries = {"model": "model.onnx", "initial_state": "initial_state.npz", "host": "runtime/host.py",
               "state_contract": "runtime/state_contract.py", "assets": "runtime/assets.py", "collision_optimizer": "resources/collision_optimization.py"}
    for role, target in entries.items():
        if role == "model": continue
        origin = paths[role] if role in paths else source / (role + ".py")
        shutil.copyfile(origin, stage / target)
    with np.load(stage / "initial_state.npz", allow_pickle=False) as z:
        initial = {k: z[k].copy() for k in z.files}
    if host.validate_fixed_state(initial) != FRESH or int(initial["max_obj_id"]) != 0:
        raise PlanningBundleAcceptanceError("learned initial state is not fresh")
    artifacts = {role: dict(path=target, sha256=sha(stage / target)) for role, target in entries.items()}
    bundled = dict(format="uniad-planning-bundle-v1", artifacts=artifacts, inputs=inputs, outputs=outputs,
        initial_state_digest=host.state_digest(initial), capacities=dict(fresh=FRESH, track=TRACK_SLOTS, survivor=SURVIVOR_CAPACITY, decoded=DECODED_SLOTS, vehicle=VEHICLE_SLOTS, sca=SCA_CAPACITY),
        host_policy=dict(can_bus_mode="official_test_legacy", id_scope="session", coordinate_mode="legacy_int"),
        runtime_versions=dict(numpy=np.__version__, onnxruntime=ort.__version__, casadi=casadi.__version__),
        provenance=dict(full_model_sha256=spec["full_model"]["sha256"], extent_certificate_sha256=spec["extent_certificate"]["sha256"], shape_proof_sha256=spec["shape_proof"]["sha256"], accepted_evidence=evidence, producer_sources=source_files, source_commit=revision),
        application_result="Final optimized plan only; rejected results carry frame identity and no valid plan.",
        caller_limits="Require explicit state-gap/failure/retry configuration; vehicle parameters remain unaccepted.")
    save(stage / "manifest.json", bundled)
    manifest_sha = sha(stage / "manifest.json")
    assets.load_planning_bundle_manifest(stage, expected_manifest_sha256=manifest_sha)
    os.replace(stage, destination)
    return dict(bundle_root=str(destination), bundle_manifest_sha256=manifest_sha, model_sha256=artifacts["model"]["sha256"], initial_state_sha256=artifacts["initial_state"]["sha256"], producer_source_commit=revision, producer_sources=source_files, node_count=node_count, inputs=len(inputs), outputs=len(outputs),
                exact_accepted_model_bytes=True, scope="Planning export from full accepted graph, frozen extent metadata and explicit model/state/runtime/solver bundle. Checker only; runtime startup/recurrence/standalone acceptance is separate.")


def production_bundle_main():
    parser = argparse.ArgumentParser(description="Derive planning graph and package pinned deployment resources without PT tracing.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pipeline-status", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root.resolve(); run.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", stage="preflight", execution_status="running", acceptance_status="production_bundle_pending", overall_pass=False, pid=os.getpid(), spec_sha256=sha(args.spec))
    def checkpoint(stage, **extra):
        record.update(stage=stage, **extra)
        save(run / "status.json", record)
        save(args.pipeline_status, record)
        print(json.dumps(dict(stage=stage, status=record["status"], pid=os.getpid())), flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt("production bundle received signal " + str(signum))
    previous = signal.signal(signal.SIGTERM, interrupt)
    error = None
    try:
        checkpoint("preflight")
        spec = json.loads(args.spec.read_text())
        if spec.get("format") != "uniad-planning-export-spec-v1":
            raise ValueError("unsupported production export spec")
        result = create_planning_resource_bundle(spec, run / "bundle", checkpoint)
        result.update(status="pass", stage="complete", execution_status="complete", acceptance_status="production_bundle_derived_checker_only", spec_sha256=record["spec_sha256"])
    except BaseException as caught:
        error = caught
        result = dict(status="failed_acceptance" if isinstance(caught, PlanningBundleAcceptanceError) else "failed", stage="failed", execution_status="complete", acceptance_status="production_bundle_rejected" if isinstance(caught, PlanningBundleAcceptanceError) else "production_bundle_execution_failed", error=traceback.format_exc(), spec_sha256=record["spec_sha256"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(run / "result.json", result)
    checkpoint(result["stage"], status=result["status"], execution_status="complete", acceptance_status=result["acceptance_status"], result_sha256=sha(run / "result.json"), finished_utc=result["finished_utc"])
    if error is not None:
        raise error


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "production-bundle":
        del sys.argv[1]
        production_bundle_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "bundle":
        del sys.argv[1]
        bundle_main()
    else:
        main()
