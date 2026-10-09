"""Backend-independent bounded immutable graph resources and synchronous calls."""
from dataclasses import dataclass
import re
import threading
import time


@dataclass(frozen=True)
class ResourceKey:
    logical_graph_sha256:str
    compiled_graph_sha256:str
    ordered_native_abi_sha256:str
    backend_library_sha256:str
    bridge_sha256:str
    backend_profile_sha256:str
    resource_epoch:int=0
    def __post_init__(self):
        for name,value in self.__dict__.items():
            if name!='resource_epoch' and not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('invalid graph resource identity: '+name)
        if type(self.resource_epoch)!=int or self.resource_epoch<0:raise ValueError('invalid resource epoch')


class GraphResources:
    """Retain explicitly selected resources, leaving room for transient graphs.

    Estimates bound context/scratch only. The caller reserves frame/state,
    shared arrays and retained returned outputs separately and records RSS.
    No LRU promotion on a sequential scan: hot selection is a measured profile
    decision. Low-priority retained resources can be evicted to make room.
    Factories must clean up a partially failed preparation. Execute callbacks
    must be synchronous and return owned outputs, as the native bridge does.
    """
    def __init__(self,max_retained_contexts,max_active_contexts,max_estimated_bytes,retention=()):
        if type(max_retained_contexts)!=int or type(max_active_contexts)!=int or not 0<=max_retained_contexts<max_active_contexts:raise ValueError('reserve at least one transient context slot')
        if type(max_estimated_bytes)!=int or max_estimated_bytes<1:raise ValueError('positive resource budget required')
        if len(set(retention))!=len(retention):raise ValueError('duplicate resource retention priority')
        self.max_retained_contexts=max_retained_contexts;self.max_active_contexts=max_active_contexts;self.max_estimated_bytes=max_estimated_bytes
        self.retention={k:i for i,k in enumerate(retention)};self.resources={};self.epoch=0;self.closed=False;self.lock=threading.RLock()
        self.counts=dict(prepare_count=0,finalize_count=0,cache_hit=0,cache_miss=0,eviction=0,failed_prepare=0,failed_execute=0,scene_reset=0,invalidation=0,prepare_seconds=0.,execute_seconds=0.,close_seconds=0.,max_active_contexts=0,max_active_estimated_bytes=0)

    def _release(self,key,eviction=False):
        resource,_=self.resources.pop(key)
        started=time.perf_counter()
        try:resource.close()
        finally:
            self.counts['close_seconds']+=time.perf_counter()-started
            if eviction:self.counts['eviction']+=1

    def snapshot(self):
        with self.lock:return dict(self.counts,retained_count=len(self.resources),retained_estimated_bytes=sum(size for _,size in self.resources.values()),resource_epoch=self.epoch,closed=self.closed)

    def execute(self,key,estimated_bytes,prepare,execute):
        with self.lock:
            if self.closed:raise RuntimeError('graph resource manager closed')
            if not isinstance(key,ResourceKey) or key.resource_epoch!=self.epoch:raise ValueError('stale or invalid graph resource key')
            if type(estimated_bytes)!=int or not 0<estimated_bytes<=self.max_estimated_bytes:raise ValueError('resource exceeds context/scratch budget')
            retained=key in self.resources
            if retained:
                resource,prior_size=self.resources[key]
                if prior_size!=estimated_bytes:raise ValueError('resource estimate changed under the same identity')
                self.counts['cache_hit']+=1
            else:
                self.counts['cache_miss']+=1
                while self.resources and (len(self.resources)+1>self.max_active_contexts or sum(s for _,s in self.resources.values())+estimated_bytes>self.max_estimated_bytes):
                    victim=max(self.resources,key=lambda k:self.retention.get(k,10**9));self._release(victim,True)
                start=time.perf_counter();self.counts['prepare_count']+=1
                try:
                    resource,finalized_graph_count=prepare()
                    if type(finalized_graph_count)!=int or finalized_graph_count<0:
                        resource.close();raise ValueError('backend preparation observation invalid')
                    self.counts['finalize_count']+=finalized_graph_count
                except BaseException:
                    self.counts['failed_prepare']+=1;raise
                finally:self.counts['prepare_seconds']+=time.perf_counter()-start
                if key in self.retention and self.max_retained_contexts:
                    if len(self.resources)>=self.max_retained_contexts:
                        victim=max(self.resources,key=lambda k:self.retention.get(k,10**9))
                        if self.retention[key]<self.retention.get(victim,10**9):self._release(victim,True)
                    if len(self.resources)<self.max_retained_contexts:
                        self.resources[key]=(resource,estimated_bytes);retained=True
            active_count=len(self.resources)+(not retained);active_bytes=sum(s for _,s in self.resources.values())+(0 if retained else estimated_bytes)
            self.counts['max_active_contexts']=max(self.counts['max_active_contexts'],active_count)
            self.counts['max_active_estimated_bytes']=max(self.counts['max_active_estimated_bytes'],active_bytes)
            start=time.perf_counter()
            try:return execute(resource)
            except BaseException:
                self.counts['failed_execute']+=1
                if retained:self._release(key,True);retained=False;resource=None
                raise
            finally:
                self.counts['execute_seconds']+=time.perf_counter()-start
                if not retained and resource is not None:
                    started=time.perf_counter()
                    try:resource.close()
                    finally:self.counts['close_seconds']+=time.perf_counter()-started

    def reset_state(self):
        with self.lock:
            if self.closed:raise RuntimeError('graph resource manager closed')
            # Recurrent state belongs to the transaction caller. This operation
            # intentionally records reset without invalidating graph resources.
            self.counts['scene_reset']+=1

    def invalidate_resources(self):
        with self.lock:
            if self.closed:raise RuntimeError('graph resource manager closed')
            errors=[]
            for key in list(self.resources):
                try:self._release(key)
                except BaseException as error:errors.append(error)
            self.epoch+=1;self.retention={};self.counts['invalidation']+=1
            if errors:raise RuntimeError('resource close failed during invalidation') from errors[0]

    def close(self):
        with self.lock:
            if self.closed:return
            try:self.invalidate_resources()
            finally:self.closed=True
