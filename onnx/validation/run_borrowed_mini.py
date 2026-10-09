"""Run frozen mini404 driver with the source-bound borrowed-input candidate."""
import cProfile
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'onnx/qnn'))
from tool_run import sha, save
import borrowed_session
import partition_session

original_bind = partition_session.bind_partitions


def bind_candidate(*args, **kwargs):
    profile = original_bind(*args, **kwargs)
    path = Path(borrowed_session.__file__).absolute()
    profile['assets']['borrowed_input_adapter'] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    profile['transport_policy'] = dict(type='synchronous_borrowed_numpy_inputs', source=borrowed_session.PROVENANCE)
    return profile


partition_session.NativeSession = borrowed_session.BorrowedNativeSession
partition_session.bind_partitions = bind_candidate
source = ROOT / 'onnx/validation/run_qnn_mini.py'
spec = importlib.util.spec_from_file_location('frozen_mini_driver', source)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
try:
    driver.main()
finally:
    save(Path.cwd() / 'borrowed_input_transport.json', dict(
        status='observed', transport=borrowed_session.TRANSPORT, provenance=borrowed_session.PROVENANCE,
        adapter_sha256=sha(borrowed_session.__file__), driver_sha256=sha(source),
        scope='Actual candidate input borrow/copy counters; mini404 audit/task gates remain independent.'))
