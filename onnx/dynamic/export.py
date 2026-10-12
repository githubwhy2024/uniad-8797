"""Production model build/export entrypoint and existing-graph bundle command.

Q1/Q2 production adapters are installed before sanity/export. Recurrent shapes
remain dynamic until Q3 staticization. build_model only constructs/patches the
model; use export.py --help or export.py bundle --help for the two CLIs.
"""
import argparse
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
# Default execution runs three synthetic frames; --export writes a new ONNX
# path after sanity and never overwrites an existing model. This is not a
# QAIRT/QNN/QAM8797P acceptance test.


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="projects/configs/stage2_e2e/base_e2e.py")
    parser.add_argument("--checkpoint", default="ckpts/uniad_base_e2e.pth")
    parser.add_argument("--out", default="onnx/uniad_stage2_stateful_v3.onnx")
    parser.add_argument("--export", action="store_true")
    parser.add_argument(
        "--bundle", action="store_true",
        help="with --export, also write the paired initial-state NPZ/JSON",
    )
    parser.add_argument(
        "--initial-state-out",
        help="bundle NPZ path; default is <onnx stem>.initial_state.npz",
    )
    parser.add_argument(
        "--revision", default="qualcomm-q1q2-pre-refactor",
        help="evidence label only; recurrent contract remains stateful-v1",
    )
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("at least three sanity frames are required to consume generated active queries")
    if args.bundle and not args.export:
        parser.error("--bundle requires --export")
    if args.export and Path(args.out).exists():
        parser.error("refusing to overwrite existing ONNX; select a new candidate path")
    if args.bundle:
        bundle_out = (
            Path(args.initial_state_out)
            if args.initial_state_out
            else Path(args.out).with_suffix(".initial_state.npz")
        )
        if bundle_out.exists() or bundle_out.with_suffix(".json").exists():
            parser.error(
                "refusing to overwrite existing initial-state bundle; "
                "select a new candidate path"
            )
    else:
        bundle_out = None
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    root = Path(__file__).resolve().parents[1] / "runs"
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="stateful-", dir=root))
    record = dict(pid=os.getpid(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  arguments=vars(args), contract="stateful-v1", model_revision=args.revision,
                  spatial_rebatch="fixed_topk_indexselect_scatternd_shared_encoder_indices",
                  rotation="double_sin_only_float32_coordinate_round_gather",
                  dcnv2="qualcomm_centernet_gather_bilinear_conv2d_n6",
                  track_state_shape="dynamic_pre_q3",
                  frames=[], input_names=INPUT_NAMES,
                  output_names=OUTPUT_NAMES, run_dir=str(run_dir))
    def checkpoint(stage, **extra):
        record.update(stage=stage, **extra)
        temporary = run_dir / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(run_dir / "status.json")
        print(f"[{stage}] pid={os.getpid()} evidence={run_dir}", flush=True)
    try:
        checkpoint("building")
        cfg, model = build_model(args.config, args.checkpoint)
        wrapper = StatefulStep(model, cfg.occflow_grid_conf).eval()
        with torch.no_grad():
            inputs = make_inputs(wrapper)
            # Trace on first-frame tensors, but has_prev is a graph input and
            # component tests prove that both encoder branches stay live.
            trace_inputs = inputs
            for frame in range(args.frames):
                checkpoint("sanity_forward", current_frame=frame)
                start = time.monotonic()
                outputs = wrapper(*inputs)
                assert len(outputs) == len(OUTPUT_NAMES)
                for name, tensor in zip(OUTPUT_NAMES, outputs):
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise ValueError(f"nonfinite output {name} in frame {frame}")
                result = dict(zip(OUTPUT_NAMES, outputs))
                n = result["next_query"].shape[0]
                assert torch.all(result["next_obj_idxes"][:901] == -1)
                for name in INPUT_NAMES[13:]:
                    assert result["next_" + name].shape[0] == n
                info = dict(frame=frame, seconds=time.monotonic()-start,
                            input_track_count=inputs[13].shape[0], next_track_count=n,
                            next_id=int(result["next_max_obj_id"]),
                            shapes={name: list(value.shape) for name, value in result.items()})
                record["frames"].append(info)
                checkpoint("sanity_frame_passed")
                print(json.dumps(info), flush=True)
                inputs = advance(inputs, outputs)
            checkpoint("sanity_passed")
            if args.export:
                if not any(frame["input_track_count"] > 901 for frame in record["frames"]):
                    raise ValueError("sanity did not consume dynamic track count >901; add frames or a targeted state fixture")
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                checkpoint("exporting")
                # Same grad-enabled tracing mode as the verified legacy
                # exporter: no_grad enables the unexportable PyTorch 2.0.1
                # fused TransformerEncoderLayer/MHA inference fast path.
                with torch.enable_grad():
                    torch.onnx.export(wrapper, trace_inputs, args.out, opset_version=18,
                                      input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
                                      dynamic_axes=dynamic_axes(), do_constant_folding=True)
                checkpoint("checking")
                import onnx
                onnx.checker.check_model(args.out)
                graph = onnx.load(args.out)
                actual_inputs = [i.name for i in graph.graph.input]
                assert actual_inputs == INPUT_NAMES, actual_inputs
                assert [o.name for o in graph.graph.output] == OUTPUT_NAMES
                assert all(node.domain in ("", "ai.onnx") for node in graph.graph.node)
                digest = hashlib.sha256()
                with open(args.out, "rb") as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(block)
                export_record = {
                    "onnx_path": str(Path(args.out).resolve()),
                    "bytes": Path(args.out).stat().st_size,
                    "sha256": digest.hexdigest(),
                    "ort_validated": False,
                    "official_cuda_validated": False,
                }
                if args.bundle:
                    checkpoint("writing_initial_state_bundle", **export_record)
                    bundle_path, bundle_manifest_path, bundle_metadata = (
                        write_initial_state_bundle(
                            wrapper,
                            args.out,
                            bundle_out,
                            args.config,
                            args.checkpoint,
                        )
                    )
                    export_record.update(
                        initial_state_path=str(bundle_path),
                        initial_state_manifest_path=str(bundle_manifest_path),
                        initial_state_sha256=bundle_metadata["state_sha256"],
                    )
                checkpoint("export_checker_passed", **export_record)
    except Exception:
        checkpoint("failed", error=traceback.format_exc())
        raise


# -----------------------------------------------------------------------------
# Initial-state bundle CLI
# -----------------------------------------------------------------------------
# Only preprocessing/export uses PyTorch; runtime loads this bundle with NumPy.


def bundle_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="onnx/uniad_stage2_stateful_v3.onnx")
    parser.add_argument("--out", default="onnx/uniad_stage2_stateful_v3.initial_state.npz")
    parser.add_argument("--config", default="projects/configs/stage2_e2e/base_e2e.py")
    parser.add_argument("--checkpoint", default="ckpts/uniad_base_e2e.pth")
    args = parser.parse_args()
    model_path, output = Path(args.onnx), Path(args.out)
    metadata_path = output.with_suffix(".json")
    if not model_path.is_file():
        parser.error("ONNX must exist before producing its initialization bundle")
    if output.exists() or metadata_path.exists():
        parser.error("refusing to overwrite an existing initialization bundle")
    torch.set_num_threads(4)
    with torch.no_grad():
        _, model = build_model(args.config, args.checkpoint)
        source = SimpleNamespace(model=model, cycle=TensorTrackStateCycle(model))
        write_initial_state_bundle(source, model_path, output, args.config, args.checkpoint)
    print(f"Saved {output} and {metadata_path}; ONNX/initial-state/checkpoint hashes recorded", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bundle":
        del sys.argv[1]
        bundle_main()
    else:
        main()
