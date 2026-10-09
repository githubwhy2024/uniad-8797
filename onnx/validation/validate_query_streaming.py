"""Query streaming controls including partial final chunks and sampler boundaries."""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as H, numpy_helper as N
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qnn'))
from stream_attention import streamed_nodes
from tool_run import sha, save, SDK
from session import NativeSession, DTYPES
from resources import terminal


def fixture(queries=17, chunk=4, batch=2, channels=3):
    features = ['feature' + str(i) for i in range(4)]
    grids = ['grid' + str(i) for i in range(4)]
    sizes = [(7, 9), (4, 5), (3, 2), (1, 1)]
    inputs = [H.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [batch, channels, *s]) for n, s in zip(features, sizes)]
    inputs += [H.make_tensor_value_info(n, onnx.TensorProto.FLOAT, [batch, queries, 8, 2]) for n in grids]
    inputs += [H.make_tensor_value_info('attention', onnx.TensorProto.FLOAT, [batch, 1, queries, 32])]
    attrs = dict(mode='bilinear', padding_mode='zeros', align_corners=0)
    nodes = [H.make_node('GridSample', [f, g], ['sample' + str(i)], **attrs) for i, (f, g) in enumerate(zip(features, grids))]
    nodes += [H.make_node('Concat', ['sample' + str(i) for i in range(4)], ['sampled'], axis=-1), H.make_node('Mul', ['sampled', 'attention'], ['product']), H.make_node('ReduceSum', ['product', 'axis'], ['result'], keepdims=0)]
    def model(chain, initializers=()):
        result = H.make_model(H.make_graph(chain, 'query-streaming-control', inputs, [H.make_tensor_value_info('result', onnx.TensorProto.FLOAT, [batch, channels, queries])], list(initializers)), opset_imports=[H.make_opsetid('', 18)])
        result.ir_version = 8
        onnx.checker.check_model(result, full_check=True)
        return result
    reference = model(nodes, [N.from_array(np.array([-1], np.int64), 'axis')])
    candidate = model(streamed_nodes(grids, features, 'attention', 'result', (batch, channels, queries, 32), chunk, 'stream', attrs))
    return reference, candidate


def feed(model, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for row in model.graph.input:
        dims = [d.dim_value for d in row.type.tensor_type.shape.dim]
        value = rng.uniform(-1.7, 1.7, size=dims).astype(np.float32)
        if row.name.startswith('grid'):
            value.reshape(-1)[:8] = [-1., 1., 0., 0., -2., 2., -0.5, 0.5]
        result[row.name] = value
    return result


def stats(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(b).all():
        raise ValueError('streamed output contract differs')
    return dict(exact=bool(np.array_equal(a, b)), max_abs=float(np.max(np.abs(a.astype(np.float64)-b))), shape=list(b.shape), dtype=str(b.dtype))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--compile-run', type=Path)
    p.add_argument('--bridge-build', type=Path)
    a = p.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    if a.compile_run is None:
        rows = []
        for queries, chunk, batch, channels in [(1, 4, 2, 3), (17, 4, 2, 3), (17, 17, 2, 3), (10201, 2048, 1, 1)]:
            ref, candidate = fixture(queries, chunk, batch, channels)
            for seed in (9, 21):
                inputs = feed(ref, seed)
                snapshots = {n: v.copy() for n, v in inputs.items()}
                expected = ort.InferenceSession(ref.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
                actual = ort.InferenceSession(candidate.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
                if any(not np.array_equal(v, snapshots[n]) for n, v in inputs.items()):
                    raise ValueError('streaming mutated inputs')
                rows.append(dict(queries=queries, chunk_queries=chunk, seed=seed, **stats(expected, actual)))
        ref, candidate = fixture()
        onnx.save(ref, str(Path.cwd()/'reference.onnx'))
        onnx.save(candidate, str(Path.cwd()/'model.onnx'))
        save(Path.cwd()/'input_layouts.json', {'feature'+str(i): 'NCHW' for i in range(4)})
        save(Path.cwd()/'query_streaming_control.json', dict(status='pass', controls=rows, script_sha256=sha(__file__), streaming_sha256=sha(Path(__file__).resolve().parents[1]/'qnn/stream_attention.py'), candidate_sha256=sha(Path.cwd()/'model.onnx'), scope='Constructed FP32 ORT controls; floating statistics recorded; actual backend and task acceptance separate.'))
        return
    terminal(a.compile_run)
    terminal(a.bridge_build)
    build = json.loads((a.compile_run/'model_build.json').read_text())
    bridge = json.loads((a.bridge_build/'bridge_build.json').read_text())
    net = json.loads((Path(build['resources']['model.cpp']['path']).parent/'model_net.json').read_text())['graph']['tensors']
    ref, candidate = fixture()
    abi = dict(schema='qnn-native-abi-v1', inputs=[], outputs=[])
    for kind in ('inputs', 'outputs'):
        for v in getattr(candidate.graph, kind[:-1]):
            row = net[v.name]
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            entry = dict(name=v.name, native_name=v.name, shape=dims, native_shape=row['dims'], dtype=DTYPES[row['data_type']])
            if row['dims'] != dims:
                entry['wire_view'] = 'singleton_axes'
            abi[kind].append(entry)
    backend = SDK/'lib/x86_64-linux-clang/libQnnCpu.so'
    rows = []
    with NativeSession(bridge['library'], build['library'], backend, abi, dict(bridge=bridge['library_sha256'], model_lib=build['library_sha256'], backend_lib=sha(backend))) as session:
        old_outputs = None
        for seed in (9, 21, 9):
            inputs = feed(ref, seed)
            snapshots = {n: v.copy() for n, v in inputs.items()}
            expected = ort.InferenceSession(ref.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
            actual = session.run(None, inputs)[0]
            if any(not np.array_equal(v, snapshots[n]) for n, v in inputs.items()):
                raise ValueError('backend mutated source inputs')
            if old_outputs and not np.array_equal(*old_outputs):
                raise ValueError('backend overwrote retained output')
            old_outputs = (actual, actual.copy())
            rows.append(dict(seed=seed, **stats(expected, actual)))
        save(Path.cwd()/'native_abi.json', session.native_abi)
    save(Path.cwd()/'query_streaming_backend_control.json', dict(status='pass', cases=rows, compiled_library_sha256=build['library_sha256'], scope='Actual FP32 CPU streaming construction control; no neural/task/board acceptance.'))


if __name__ == '__main__':
    main()
