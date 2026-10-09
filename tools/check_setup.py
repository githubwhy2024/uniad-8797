#!/usr/bin/env python3
"""Check baseline resources, source identity, and optional ONNX interfaces."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from fetch_resources import ROOT, destination, matches, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-sources', action='store_true', help='Require the initial snapshot source hashes')
    parser.add_argument('--onnx', action='store_true', help='Check model structure and initialize learned state; no inference')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'onnx/resources/manifest.json').read_text())
    for name, row in manifest['resources'].items():
        if not matches(destination(row['path']), row):
            raise SystemExit('Missing or changed baseline resource: ' + name)
    if args.baseline_sources:
        snapshot = json.loads((ROOT / 'onnx/reference/source_snapshot.json').read_text())
        for name, digest in snapshot['files'].items():
            if sha(destination(name)) != digest:
                raise SystemExit('Baseline source changed: ' + name)
    common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=ROOT, text=True).strip()
    if (ROOT / common).resolve() != ROOT / '.git' or not (ROOT / '.git').is_dir():
        raise SystemExit('Independent project Git directory required')
    result = dict(status='pass', resources=len(manifest['resources']), independent_git=True,
                  baseline_sources_checked=args.baseline_sources, neural_inference=False,
                  target_acceptance='not_evaluated')
    if args.onnx:
        import numpy as np
        import onnx
        interfaces = {}
        for role, outputs in [('planning_model', 25), ('full_model', 47)]:
            path = destination(manifest['resources'][role]['path'])
            model = onnx.load(str(path), load_external_data=False)
            if any(value.data_location == onnx.TensorProto.EXTERNAL for value in model.graph.initializer):
                raise ValueError('unexpected external ONNX weights')
            onnx.checker.check_model(model)
            if len(model.graph.input) != 23 or len(model.graph.output) != outputs:
                raise ValueError('model public interface differs')
            for value in list(model.graph.input) + list(model.graph.output):
                shape = value.type.tensor_type.shape
                if any(not dim.HasField('dim_value') for dim in shape.dim):
                    raise ValueError('dynamic public boundary: ' + value.name)
            interfaces[role] = dict(inputs=23, outputs=outputs, nodes=len(model.graph.node))
            del model
        host_path = ROOT / 'onnx/fixed/host.py'
        sys.path.insert(0, str(host_path.parent))
        spec = importlib.util.spec_from_file_location('baseline_host', host_path)
        host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(host)
        with np.load(destination(manifest['resources']['initial_state']['path']), allow_pickle=False) as archive:
            initial = {name: archive[name].copy() for name in archive.files}
        if initial['query'].shape[0] == 901:
            initial = host.pad_v1_initial_state(initial)
        if host.validate_fixed_state(initial) != host.FRESH or int(initial['max_obj_id']) != 0:
            raise ValueError('learned initial state differs')
        host.verified_optimizer_source(destination(manifest['resources']['collision_optimizer']['path']),
                                       manifest['resources']['collision_optimizer']['sha256'])
        result.update(onnx_interfaces=interfaces, learned_state_valid=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
