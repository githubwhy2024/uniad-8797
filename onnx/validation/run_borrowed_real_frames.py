"""Run the frozen real-frame driver with the borrowed-input CPU candidate."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onnx/qnn'))
from tool_run import save, sha
import borrowed_session
import partition_session

source = ROOT / 'onnx/validation/run_qnn_real_frames.py'
EXPECTED_DRIVER = '8f3d7bb1b59d26b505b3c3a778df514215ea398d21a500f1c3fb8509fc03dac4'
if sha(source) != EXPECTED_DRIVER:
    raise ValueError('borrowed frame driver source identity differs')
original_bind = partition_session.bind_partitions


def bind_candidate(*args, **kwargs):
    profile = original_bind(*args, **kwargs)
    path = Path(borrowed_session.__file__).absolute()
    profile['assets']['borrowed_input_adapter'] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    profile['transport_policy'] = dict(type='synchronous_borrowed_numpy_inputs', source=borrowed_session.PROVENANCE)
    return profile


# Local to this separate candidate process; active baseline workers are untouched.
partition_session.NativeSession = borrowed_session.BorrowedNativeSession
partition_session.bind_partitions = bind_candidate
spec = importlib.util.spec_from_file_location('frozen_real_frame_driver', source)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
try:
    driver.main()
finally:
    save(Path.cwd() / 'borrowed_input_transport.json', dict(
        status='observed', transport=borrowed_session.TRANSPORT, provenance=borrowed_session.PROVENANCE,
        adapter_sha256=sha(borrowed_session.__file__), driver_sha256=sha(source),
        scope='Actual candidate input borrowing/copy fallback counters; neural/task/performance decisions remain separate.'))
