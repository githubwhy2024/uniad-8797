"""Resolve frozen reference reports without depending on a developer checkout."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def resolve_reference(path):
    mapping = json.loads((ROOT/'onnx/reference/path_map.json').read_text())
    relative = mapping[str(path)]
    result = ROOT/relative
    if result.resolve() != result or not result.is_relative_to(ROOT/'onnx/reference'):
        raise ValueError('ordinary frozen reference required')
    return result
