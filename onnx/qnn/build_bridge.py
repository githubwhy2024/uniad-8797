#!/usr/bin/env python3
"""Build the native CPU session bridge against the installed QNN headers."""
import argparse,json,os,shlex,shutil,subprocess
from pathlib import Path
from session import sha
from tool_run import SDK

def main():
    p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,default=SDK);p.add_argument('--cxx',default=os.environ.get('CXX','clang++'));args=p.parse_args()
    sdk=args.sdk;source=Path(__file__).with_name('session_bridge.cpp');out=Path.cwd()/'libsession_bridge.so';compiler_args=shlex.split(args.cxx)
    if not compiler_args or shutil.which(compiler_args[0]) is None:raise ValueError('C++ compiler not found; set CXX or --cxx')
    compiler=Path(shutil.which(compiler_args[0]))
    command=[str(compiler),*compiler_args[1:],'-std=c++17','-Wall','-Wextra','-Werror','-fPIC','-shared',str(source),'-I'+str(sdk/'include/QNN'),'-I'+str(sdk/'share/QNN/converter/jni'),'-ldl','-o',str(out)]
    subprocess.run(command,check=True)
    headers={str(v.relative_to(sdk)):sha(v) for root in (sdk/'include/QNN',sdk/'share/QNN/converter/jni') for v in sorted(root.rglob('*')) if v.is_file() and v.suffix in ('.h','.hpp')}
    result=dict(status='pass',scope='Native bridge compilation only; model/backend execution is separate.',source=str(source),source_sha256=sha(source),library=str(out),library_sha256=sha(out),command=command,compiler_sha256=sha(compiler.resolve()),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),sdk_header_hashes=headers)
    (Path.cwd()/'bridge_build.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
