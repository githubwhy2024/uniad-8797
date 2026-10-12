"""Source-guarded partition execution with portable bounded graph lifetimes."""
import ast
import copy
import hashlib
import inspect
import json
import textwrap
import time
from pathlib import Path
import partition_session as source
from borrowed_session import BorrowedNativeSession
from graph_resources import GraphResources,ResourceKey
from borrowed_session import TRANSPORT

SOURCE_SHA256='b1f514a75d7dad0d0b28f2061b4445e12c47fd9e98c67faac8ad74dcf38ea59b'
if source.sha(source.__file__)!=SOURCE_SHA256:raise ValueError('resource adapter requires exact partition source')

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def memory_snapshot():
    return {line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if line.split(':',1)[0] in ('VmSize','VmRSS','VmPeak','VmHWM')}

class TimedNativeLibrary:
    """Forward identical native calls, recording API duration without profiling."""
    def __init__(self,library):self.library=library;self.execute_seconds=0.;self.execute_count=0
    def __getattr__(self,name):return getattr(self.library,name)
    def q4_qnn_execute(self,*args):
        started=time.perf_counter();self.execute_count+=1
        try:return self.library.q4_qnn_execute(*args)
        finally:self.execute_seconds+=time.perf_counter()-started

class ResourcePartitionSession(source.PartitionSession):
    def __init__(self,manifest,policy):
        manifest=copy.deepcopy(manifest);policy=copy.deepcopy(policy)
        if set(policy)!= {'context_estimates_bytes','retain_indices','max_retained_contexts','max_active_contexts','context_budget_bytes','source'}:raise ValueError('resource policy fields differ')
        super().__init__(manifest)
        self.policy=policy;self._identity=digest(manifest);self._policy_sha=digest(policy)
        self._estimates={int(i):v for i,v in policy['context_estimates_bytes'].items()}
        if set(self._estimates)!={row['index'] for row in manifest['parts']}:raise ValueError('context estimate coverage differs')
        self._retained=list(policy['retain_indices'])
        if len(set(self._retained))!=len(self._retained) or not set(self._retained)<=set(self._estimates):raise ValueError('resource retention scope differs')
        retention=[self._key(manifest['parts'][i],0) for i in self._retained]
        self.graph_resources=GraphResources(policy['max_retained_contexts'],policy['max_active_contexts'],policy['context_budget_bytes'],retention)
        self._pool_hashes={}
        pool=manifest.get('shared_constants',manifest.get('plan',{}).get('shared_constants'))
        if pool:
            for row in pool['entries']:
                if row['key'] not in self._pool_hashes:self._pool_hashes[row['key']]=(self._shared_values[row['source_name']].reshape(-1),row['data_sha256'])
        self._part_pool_keys={row['index']:{entry['key'] for entry in pool['entries'] if entry['source_name'] in {r['source_name'] for r in self.manifest['plan']['parts'][row['index']]['inputs']}} for row in self.manifest['parts']} if pool else {}
        self.resource_observations=[]

    def _key(self,row,epoch):
        a=self.manifest['assets']
        return ResourceKey(row['native_model_sha256'],row['library_sha256'],digest(row['abi']),a['backend_lib']['sha256'],a['bridge']['sha256'],digest(dict(profile=self._identity,policy=self._policy_sha)),epoch)

    def _check_host_memory_budget(self,snapshot):
        policy_source=self.policy['source']
        limit=policy_source.get('physical_rss_limit_bytes') if isinstance(policy_source,dict) else None
        if limit is not None and int(snapshot['VmHWM'].split()[0])*1024>limit:
            source.save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_process_peak_rss_budget_exceeded',limit_bytes=limit,memory=snapshot,scope='Host profile only; reject before publishing outputs or advancing state.'))
            raise ValueError('actual process peak RSS exceeded host profile budget')

    def _execute_part(self,row,execute):
        if digest(self.manifest)!=self._identity or digest(self.policy)!=self._policy_sha:raise ValueError('resource profile/policy changed')
        manager=self.graph_resources;key=self._key(row,manager.epoch)
        # Explicit invalidation changes epoch. The selected profile may be
        # re-established without silently reusing an incompatible resource.
        if not manager.retention:manager.retention={self._key(self.manifest['parts'][i],manager.epoch):p for p,i in enumerate(self._retained)}
        before=manager.snapshot();transport_before=dict(TRANSPORT);api_seconds=0.;api_count=0;root_guard_seconds=0.;root_guard_bytes=0
        memory_before=memory_snapshot()
        def prepare():
            source.save(Path.cwd()/'resource_prepare_progress.json',dict(stage='before_native_prepare',index=row['index'],memory=memory_snapshot()))
            native=self._native(row)
            try:self._check_host_memory_budget(memory_snapshot())
            except BaseException:
                native.close()
                raise
            if hasattr(native,'_lib'):native._lib=TimedNativeLibrary(native._lib)
            source.save(Path.cwd()/'resource_prepare_progress.json',dict(stage='after_native_prepare',index=row['index'],memory=memory_snapshot()))
            return native,1
        def invoke(session):
            nonlocal api_seconds,api_count,root_guard_seconds,root_guard_bytes
            timed=getattr(session,'_lib',None);prior_seconds=getattr(timed,'execute_seconds',0.);prior_count=getattr(timed,'execute_count',0)
            result=execute(session)
            api_seconds=getattr(timed,'execute_seconds',0.)-prior_seconds;api_count=getattr(timed,'execute_count',0)-prior_count
            # Pool arrays already own one writable physical buffer per key.
            # Borrow them directly and check immutability; no second copy/cache.
            # Only these roots were exposed to the current native graph. All
            # used physical keys remain checked once per synchronous call.
            started=time.perf_counter()
            for key in self._part_pool_keys.get(row['index'],()):
                value,expected=self._pool_hashes[key];root_guard_bytes+=value.nbytes
                if hashlib.sha256(memoryview(value).cast('B')).hexdigest()!=expected:raise ValueError('native execution modified immutable shared input')
            root_guard_seconds=time.perf_counter()-started
            return result
        result=manager.execute(key,self._estimates[row['index']],prepare,invoke)
        self._check_host_memory_budget(memory_snapshot())
        after=manager.snapshot();self.resource_observations.append(dict(index=row['index'],prepare_count=after['prepare_count']-before['prepare_count'],finalize_count=after['finalize_count']-before['finalize_count'],cache_hit=after['cache_hit']-before['cache_hit'],cache_miss=after['cache_miss']-before['cache_miss'],eviction=after['eviction']-before['eviction'],prepare_seconds=after['prepare_seconds']-before['prepare_seconds'],execute_seconds=after['execute_seconds']-before['execute_seconds'],close_seconds=after['close_seconds']-before['close_seconds'],native_api_execute_seconds=api_seconds,native_api_execute_count=api_count,shared_root_guard_seconds=root_guard_seconds,shared_root_guard_bytes=root_guard_bytes,input_transport_delta={k:TRANSPORT[k]-transport_before[k] for k in TRANSPORT},memory_before=memory_before,memory_after=memory_snapshot(),retained_count=after['retained_count'],retained_estimated_bytes=after['retained_estimated_bytes']))
        return result

    def reset_state(self):self.graph_resources.reset_state()
    def invalidate_resources(self):
        self.graph_resources.invalidate_resources()
        pool=self.manifest.get('shared_constants',self.manifest.get('plan',{}).get('shared_constants'))
        if pool and any(hashlib.sha256(memoryview(v).cast('B')).hexdigest()!=h for v,h in self._pool_hashes.values()):
            row=self.manifest['assets']['shared_constant_pool']
            if source.sha(row['path'])!=row['sha256']:raise ValueError('shared pool changed during recovery')
            with source.np.load(row['path'],allow_pickle=False) as archive:buffers={k:archive[k] for k in archive.files}
            self._pool_hashes={}
            for entry in pool['entries']:
                value=buffers[entry['key']]
                if hashlib.sha256(memoryview(value).cast('B')).hexdigest()!=entry['data_sha256']:raise ValueError('shared pool recovery differs')
                self._shared_values[entry['source_name']]=value.reshape(entry['shape'])
                self._pool_hashes[entry['key']]=(value,entry['data_sha256'])
    def close(self):
        with self._lock:
            if self._closed:return
            self._closed=True
            try:self.graph_resources.close()
            finally:self._shared_values.clear();self._pool_hashes.clear()


original=ast.parse(textwrap.dedent(inspect.getsource(source.PartitionSession.run)))
candidate=copy.deepcopy(original)
class ResourceExecution(ast.NodeTransformer):
    changed=0
    def visit_With(self,node):
        self.generic_visit(node)
        if len(node.items)!=1 or ast.dump(node.items[0].context_expr)!=ast.dump(ast.parse('self._native(row)',mode='eval').body):return node
        if not isinstance(node.items[0].optional_vars,ast.Name) or node.items[0].optional_vars.id!='session':raise ValueError('partition native binding changed')
        self.changed+=1
        fn=ast.parse('def execute_resource(session):\n    return None').body[0]
        fn.body=node.body+[ast.parse('return out,loaded').body[0]]
        call=ast.parse('out,loaded = self._execute_part(row,execute_resource)').body[0]
        return [ast.copy_location(fn,node),ast.copy_location(call,node)]
transform=ResourceExecution();candidate=ast.fix_missing_locations(transform.visit(candidate))
if transform.changed!=1:raise ValueError('expected exactly one native execution scope')
namespace=dict(vars(source));exec(compile(candidate,__file__,'exec'),namespace)
_resource_run=namespace['run']
def run_with_resources(self,names,feed):
    visits=self.graph_resources.counts['cache_hit']+self.graph_resources.counts['cache_miss']
    self.resource_observations=[]
    try:return _resource_run(self,names,feed)
    except BaseException:
        if self.graph_resources.counts['cache_hit']+self.graph_resources.counts['cache_miss']!=visits:self.invalidate_resources()
        raise
ResourcePartitionSession.run=run_with_resources

# Preserve all original ABI/integer-range mapping; only choose borrowing.
native=ast.parse(textwrap.dedent(inspect.getsource(source.PartitionSession._native)))
class BorrowNative(ast.NodeTransformer):
    changed=0
    def visit_Name(self,node):
        if node.id=='NativeSession':self.changed+=1;return ast.copy_location(ast.Name(id='BorrowedNativeSession',ctx=node.ctx),node)
        return node
replacement=BorrowNative();native=ast.fix_missing_locations(replacement.visit(native))
if replacement.changed!=1:raise ValueError('native constructor replacement differs')
native_namespace=dict(vars(source),BorrowedNativeSession=BorrowedNativeSession);exec(compile(native,__file__,'exec'),native_namespace)
ResourcePartitionSession._native=native_namespace['_native']
PROVENANCE=dict(source_partition_sha256=SOURCE_SHA256,source_run_ast_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),resource_run_ast_sha256=hashlib.sha256(ast.dump(candidate).encode()).hexdigest(),execution_scope_replacements=transform.changed,native_constructor_replacements=replacement.changed,scope='Original run ABI/finite/integer/bool/cut/lifetime/public output/diagnostic logic preserved. One native scope delegates bounded resource manager; one constructor uses exact guarded borrowed-input session. Float/Host/state computations unchanged.')
