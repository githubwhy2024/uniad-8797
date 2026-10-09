"""Source-preserving frame guards and resource recovery with a dummy native ABI."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from resource_partition_session import ResourcePartitionSession,PROVENANCE,TimedNativeLibrary
import resource_partition_session as implementation
from tool_run import sha,save
def main():
    class Library:
        marker='forwarded property'
        def q4_qnn_execute(self,*args):
            if args==('failure',):raise RuntimeError('controlled native API failure')
            return args
    timed=TimedNativeLibrary(Library())
    if timed.q4_qnn_execute('pointer',12,23)!=('pointer',12,23) or timed.marker!='forwarded property':raise ValueError('API timer changed arguments, returns or forwarding')
    try:timed.q4_qnn_execute('failure')
    except RuntimeError:pass
    else:raise ValueError('API timer swallowed native exception')
    if timed.execute_count!=2 or timed.execute_seconds<0:raise ValueError('API timer count differs')
    asset=Path.cwd()/'dummy_resource.txt';asset.write_text('dummy resource identity, not a native library');digest=sha(asset)
    abi=dict(schema='qnn-native-abi-v1',inputs=[dict(name='x',native_name='x',shape=[2],native_shape=[2],dtype='float32')],outputs=[dict(name='y',native_name='y',shape=[2],native_shape=[2],dtype='float32')])
    part=dict(index=0,inputs=[dict(source_name='x',name='x',shape=[2],dtype='float32',bytes=8)],outputs=[dict(source_name='y',name='y',shape=[2],dtype='float32',bytes=8)],drop_after=['x'])
    row=dict(index=0,native_model_sha256=digest,library=str(asset),library_sha256=digest,abi=abi)
    manifest=dict(assets={k:dict(path=str(asset),sha256=digest,bytes=asset.stat().st_size) for k in ('bridge','backend_lib')},parts=[row],plan=dict(parts=[part]),abi=dict(inputs=[dict(name='x',shape=[2],parent_shape=[2],dtype='float32')],outputs=[dict(name='y',shape=[2],parent_shape=[2],dtype='float32')]))
    policy=dict(context_estimates_bytes={'0':10},retain_indices=[0],max_retained_contexts=1,max_active_contexts=2,context_budget_bytes=20,source='dummy control only')
    owner=ResourcePartitionSession(manifest,policy);created=[];fault={'mode':None}
    class Native:
        def __init__(self):self.native_abi=abi;self.closed=False;self.closes=0;created.append(self)
        def run(self,names,feed):
            if self.closed:raise RuntimeError('closed dummy context')
            if fault['mode']=='execute':raise RuntimeError('controlled native failure')
            if fault['mode']=='dtype':return [feed['x'].astype(np.int32)]
            if fault['mode']=='shape':return [np.zeros(3,np.float32)]
            if fault['mode']=='nonfinite':return [np.full(2,np.nan,np.float32)]
            return [feed['x']+np.float32(1)]
        def close(self):
            if not self.closed:self.closed=True;self.closes+=1
    owner._native=lambda row:Native();held=[]
    for cycle in range(3):
        x=np.array([cycle,2],np.float32);before=x.copy();out=owner.run(None,{'x':x})[0];held.append((out,out.copy()))
        if not np.array_equal(before,x) or not np.array_equal(out,x+1):raise ValueError('source flow or input ownership changed')
    counts=owner.graph_resources.snapshot()
    if (counts['prepare_count'],counts['cache_hit'],counts['cache_miss'])!=(1,2,1):raise ValueError('source adapter did not reuse context across three calls')
    guards={}
    def rejects(name,fn):
        try:fn()
        except (ValueError,RuntimeError):guards[name]=True
        else:raise ValueError('source guard accepted '+name)
    rejects('wrong_public_names',lambda:owner.run(None,{'bad':np.zeros(2,np.float32)}))
    rejects('wrong_input_shape',lambda:owner.run(None,{'x':np.zeros(3,np.float32)}))
    rejects('wrong_input_dtype',lambda:owner.run(None,{'x':np.zeros(2,np.int32)}))
    rejects('nonfinite_input',lambda:owner.run(None,{'x':np.array([0,np.nan],np.float32)}))
    if owner.graph_resources.snapshot()['prepare_count']!=1 or owner.graph_resources.snapshot()['retained_count']!=1:raise ValueError('pre-execution input rejection destroyed healthy resource')
    owner.reset_state();owner.run(None,{'x':np.zeros(2,np.float32)})
    if owner.graph_resources.snapshot()['prepare_count']!=1:raise ValueError('state reset invalidated graph')
    for mode in ('execute','dtype','shape','nonfinite'):
        fault['mode']=mode;rejects('native_'+mode,lambda:owner.run(None,{'x':np.zeros(2,np.float32)}));fault['mode']=None
        if owner.graph_resources.snapshot()['retained_count']!=0:raise ValueError('bad native resource retained after '+mode)
        owner.run(None,{'x':np.zeros(2,np.float32)})
    owner.close();owner.close();rejects('closed',lambda:owner.run(None,{'x':np.zeros(2,np.float32)}))
    if not all(v.closed and v.closes==1 for v in created) or not all(np.array_equal(a,b) for a,b in held):raise ValueError('resource close/old output ownership differs')
    # A short-lived peak can fall between monitor samples. Both preparation
    # and completion must reject from VmHWM before any frame/state promotion.
    original_memory=implementation.memory_snapshot
    limited=dict(policy,source=dict(physical_rss_limit_bytes=18*1024**3))
    for phase in ('prepare','execute'):
        peak={'exceeded':phase=='prepare'}
        implementation.memory_snapshot=lambda:dict(VmHWM=str((18*1024**2+1) if peak['exceeded'] else 1024)+' kB')
        checked=ResourcePartitionSession(manifest,limited);native_objects=[]
        class PeakNative(Native):
            def run(self,names,feed):
                values=super().run(names,feed);peak['exceeded']=True;return values
        def make_native(row):
            native=PeakNative();native_objects.append(native);return native
        checked._native=make_native
        try:
            rejects('physical_peak_after_'+phase,lambda:checked.run(None,{'x':np.zeros(2,np.float32)}))
            if checked.graph_resources.snapshot()['retained_count'] or not all(v.closed and v.closes==1 for v in native_objects):raise ValueError('over-budget context was retained')
        finally:checked.close();implementation.memory_snapshot=original_memory
    report=dict(status='pass',guards=guards,three_cycle_counts=counts,final_counts=owner.graph_resources.snapshot(),api_timer_forwarding_and_failure=True,all_resources_closed_once=True,old_outputs_owned=True,provenance=PROVENANCE,adapter_sha256=sha(ROOT/'onnx/qnn/resource_partition_session.py'),scope='Original frame public ABI/lifetime/finite checks and exact guarded resource scope with dummy backend. Real source graph load/execute/own-state/task acceptance separate; no float-tolerance relaxation.')
    save(Path.cwd()/'resource_partition_control.json',report);print(json.dumps(dict(status='pass',guards=len(guards),three_cycle_prepares=counts['prepare_count'],three_cycle_hits=counts['cache_hit'])))
if __name__=='__main__':main()
