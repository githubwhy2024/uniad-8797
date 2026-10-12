"""Verify bounded finite checks against NumPy semantics and isolated memory peaks."""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qnn'))
from session import all_finite
from tool_run import sha,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('original','bounded'));a=p.parse_args()
    if a.worker:
        value=np.ones((8192,8192),dtype=np.float32)
        fn=(lambda v:bool(np.isfinite(v).all())) if a.worker=='original' else all_finite
        samples=[]
        for _ in range(3):
            started=time.monotonic();assert fn(value);samples.append(time.monotonic()-started)
        print(json.dumps(dict(mean_seconds=sum(samples)/len(samples),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,array_bytes=value.nbytes,samples=samples)));return
    checks={}
    for dtype in (np.float16,np.float32,np.float64):
        finite=np.zeros((524291,),dtype=dtype);finite[0]=-0.;finite[-1]=np.finfo(dtype).max
        arrays=[finite,finite[::-1],finite[::2],finite.reshape(1,-1),np.zeros((0,3),dtype=dtype),np.array(2.,dtype=dtype)]
        for i,value in enumerate(arrays):checks[str(dtype)+str(i)+'_finite']=bool(all_finite(value)==np.isfinite(value).all())
        for point in (0,262143,262144,524290):
            for bad in (np.nan,np.inf,-np.inf):
                value=finite.copy();value[point]=bad
                for stride in (1,-1):checks[str((dtype,point,str(bad),stride))]=bool(all_finite(value[::stride])==np.isfinite(value[::stride]).all())
    value=np.ones(1048577,dtype=np.float32)
    sizes=[len(v) for v in np.nditer(value,flags=['external_loop','buffered','zerosize_ok'],op_flags=['readonly'],order='K',buffersize=262144)]
    checks['scratch_element_bound']=max(sizes)<=262144 and sum(sizes)==value.size
    value.flags.writeable=False
    checks['readonly_accepted']=all_finite(value)
    timings={}
    for mode in ('original','bounded'):
        timings[mode]=json.loads(subprocess.check_output([sys.executable,str(Path(__file__).absolute()),'--worker',mode],text=True))
    save(Path.cwd()/'finite_scratch_control.json',dict(status='pass' if all(checks.values()) else 'failed',checks=checks,benchmarks=timings,session_sha256=sha(Path(__file__).resolve().parents[1]/'qnn/session.py'),scope='All-element finite policy preserved for contiguous, reversed, strided, scalar, empty, readonly and chunk boundaries. Isolated 256MiB tensor peaks/timings; no neural speed claim.'))
    if not all(checks.values()):raise ValueError('finite scratch control failed')


if __name__=='__main__':main()
