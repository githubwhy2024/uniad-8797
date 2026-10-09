"""Resolve locally installed host tools without selecting a developer's paths."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


def sdk_root():
    value = next((os.environ[name] for name in
                  ('QAIRT_SDK', 'QAIRT_SDK_ROOT', 'QNN_SDK_ROOT')
                  if os.environ.get(name)), None)
    # An unconfigured SDK is not silently replaced with a historical install.
    return Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[2] / '.local/sdk'


def converter_environment():
    return Path(os.environ.get('QNN_CONVERTER_ENV') or sys.prefix).expanduser().resolve()


def cpu_backend():
    return sdk_root() / 'lib/x86_64-linux-clang/libQnnCpu.so'


def sdk_identity(root=None):
    root = Path(root or sdk_root()).resolve()
    executable = root / 'bin/x86_64-linux-clang/qnn-net-run'
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(root / 'lib/x86_64-linux-clang') + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    output = subprocess.check_output([str(executable), '--version'], env=env,
                                     text=True, stderr=subprocess.STDOUT).strip()
    match = re.search(r'v?(\d+\.\d+\.\d+\.\d{6})\d*(?:_\d+)?', output)
    if match is None:
        raise ValueError('Installed QNN tool did not report a recognized SDK build: ' + output)
    return dict(release=match.group(1), build=match.group(0),
                version_output=next(line for line in output.splitlines() if match.group(0) in line),
                version_tool_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
