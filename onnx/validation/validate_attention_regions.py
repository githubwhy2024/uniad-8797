"""Portable attention matching controls, with legacy emitted-graph identity."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onnx/qnn'))
from attention_regions import attention_regions, static_types, small_constants
from stream_attention import rewrite, streamed_nodes
from tool_run import sha, save
from validate_query_streaming import fixture, feed, stats
import shape_dataflow as engine


def match(model):
    types, constants, errors = engine.prove_static_dataflow(model)
    if errors:
        raise ValueError('fixture root inference failed')
    return attention_regions(model, types, constants)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-record', type=Path, required=True)
    parser.add_argument('--logical-model', type=Path, required=True)
    parser.add_argument('--logical-model-sha256', required=True)
    args = parser.parse_args()
    if sha(args.logical_model) != args.logical_model_sha256:
        raise ValueError('logical source identity differs')
    rows = []
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.log_severity_level = 3
    for queries, chunk, batch, channels in [(1,4,2,3),(17,4,2,3),(17,17,2,3),(10201,2048,1,1)]:
        direct, packed = fixture(queries, queries, batch, channels)
        # Anonymous and arbitrary names must work; no observed CPU suffix is used.
        for index, node in enumerate(direct.graph.node):
            node.name = 'semantic_' + str(index)
        for source_form, source in [('standard', direct), ('packed', packed)]:
            for emit_packed in (False, True):
                candidate = copy.deepcopy(source)
                changes = rewrite(candidate, chunk, expected_regions=1, packed_multiply=emit_packed)
                for seed in (9,21):
                    inputs = feed(direct, seed)
                    snapshots = {k: v.copy() for k,v in inputs.items()}
                    expected = ort.InferenceSession(direct.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
                    actual = ort.InferenceSession(candidate.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
                    if any(not np.array_equal(value, snapshots[key]) for key,value in inputs.items()):
                        raise ValueError('attention rewrite mutated input')
                    rows.append(dict(queries=queries, chunk=chunk, source_form=source_form,
                                     emission='cpu_rank3' if emit_packed else 'standard_onnx', seed=seed,
                                     source_adapter=changes[0]['source_adapter'], **stats(expected,actual)))
    # Removing a sampled array with another consumer or a public exposure is forbidden.
    guards = []
    for kind in ('outside_consumer', 'public_intermediate', 'level_policy'):
        reference, _ = fixture()
        if kind == 'outside_consumer':
            reference.graph.node.append(onnx.helper.make_node('Identity',['sample0'],['extra'],name='other_consumer'))
        elif kind == 'public_intermediate':
            reference.graph.output.append(onnx.helper.make_tensor_value_info('sample0',onnx.TensorProto.FLOAT,[2,3,17,8]))
        else:
            sampling = reference.graph.node[0]
            for attr in sampling.attribute:
                if attr.name == 'align_corners':attr.i=1
        try:
            match(reference)
        except ValueError as error:
            guards.append(dict(case=kind,rejected=True,error=str(error)))
        else:
            raise ValueError('unsafe sampling region accepted: '+kind)
    logical = onnx.load(str(args.logical_model),load_external_data=False)
    source_regions = attention_regions(logical,static_types(logical),small_constants(logical))
    if len(source_regions)!=6 or any(r['adapter']!='standard_onnx' for r in source_regions):
        raise ValueError('six original source regions not recognized')
    del logical
    legacy=json.loads(args.legacy_record.read_text()); predecessor=Path(legacy['source_model'])
    if sha(predecessor)!=legacy['source_model_sha256'] or sha(legacy['model'])!=legacy['model_sha256']:
        raise ValueError('legacy source/candidate changed')
    candidate=onnx.load(str(predecessor))
    changes=rewrite(candidate,2048)
    candidate_sha=hashlib.sha256(candidate.SerializeToString()).hexdigest()
    if candidate_sha!=legacy['model_sha256']:
        raise ValueError('portable matcher changed accepted CPU graph bytes')
    report=dict(status='pass',checks=dict(source_original_six_regions=True,legacy_cpu_graph_exact_identity=True,
                private_intermediate_and_policy_guards=True,all_32_focused_output_contracts=True),
                logical_model_sha256=args.logical_model_sha256,legacy_record=str(args.legacy_record),
                legacy_record_sha256=sha(args.legacy_record),legacy_model_sha256=candidate_sha,
                logical_regions=[dict(anchor=r['anchor'],output=r['output'],shape=list(r['shape']),node_indices=r['node_indices'],node_sha256=r['node_sha256']) for r in source_regions],
                controls=rows,guards=guards,legacy_changes=changes,
                helpers={name:sha(ROOT/'onnx/qnn'/name) for name in ('attention_regions.py','stream_attention.py')},
                scope='Constructed standard ONNX/CPU-adapter local ORT controls only. Full logical model matching is read-only. Legacy emitted CPU model bytes unchanged. No full NN, recurrence/task or device acceptance; floating statistics recorded.')
    save(Path.cwd()/'attention_region_control.json',report)
    print(json.dumps(dict(status='pass',controls=len(rows),guards=len(guards),logical_regions=len(source_regions),legacy_model_sha256=candidate_sha)))


if __name__=='__main__':main()
