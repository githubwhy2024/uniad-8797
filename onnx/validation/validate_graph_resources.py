"""Cycle, identity, bounded budget and failure controls, independent of NN."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from graph_resources import GraphResources,ResourceKey
from tool_run import save,sha
def key(name,epoch=0):
    digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
    return ResourceKey(digest('logical'+name),digest('compiled'+name),digest('abi'),digest('backend'),digest('bridge'),digest('profile'),epoch)
class Resource:
    def __init__(self):self.closed=False;self.closes=0
    def close(self):
        if not self.closed:self.closes+=1;self.closed=True
    def run(self,x):
        if self.closed:raise RuntimeError('closed dummy backend')
        return np.asarray(x,dtype=np.int32).copy()
def rejected(fn):
    try:fn()
    except (ValueError,RuntimeError):return True
    return False
def main():
    keys=[key(str(i)) for i in range(5)];created=[]
    def prepare():
        resource=Resource();created.append(resource);return resource,1
    manager=GraphResources(2,3,30,keys[:2]);outputs=[]
    for cycle in range(3):
        for index,k in enumerate(keys):
            value=np.array([cycle,index],np.int32);before=value.copy()
            result=manager.execute(k,10,prepare,lambda r:r.run(value));outputs.append((result,result.copy()))
            if not np.array_equal(value,before):raise ValueError('input changed')
    cycles=manager.snapshot()
    if (cycles['prepare_count'],cycles['cache_hit'],cycles['cache_miss'],cycles['eviction'],cycles['retained_count'],cycles['max_active_contexts'],cycles['max_active_estimated_bytes'])!=(11,4,11,0,2,3,30):raise ValueError('three-cycle sticky budget differs')
    # An independently simulated two-entry LRU makes15 prepares/0 hits on this
    # five-key sequential workload. A hot single resource proves neither case.
    order=[];lru_misses=0;lru_hits=0
    for _ in range(3):
        for k in keys:
            if k in order:lru_hits+=1;order.remove(k)
            else:lru_misses+=1
            order.append(k)
            if len(order)>2:order.pop(0)
    if (lru_misses,lru_hits)!=(15,0):raise ValueError('cycle thrashing comparator differs')
    count=manager.snapshot()['prepare_count'];manager.reset_state();manager.execute(keys[0],10,prepare,lambda r:r.run([3]))
    if manager.snapshot()['prepare_count']!=count:raise ValueError('scene reset destroyed compatible resources')
    guards={}
    guards['changed_estimate_same_key']=rejected(lambda:manager.execute(keys[0],11,prepare,lambda r:r.run([1])))
    guards['over_budget_before_prepare']=rejected(lambda:manager.execute(keys[2],31,prepare,lambda r:r.run([1])))
    if manager.snapshot()['prepare_count']!=count:raise ValueError('invalid request prepared a resource')
    def fail(resource):raise RuntimeError('controlled backend execution failure')
    guards['execute_failure']=rejected(lambda:manager.execute(keys[0],10,prepare,fail))
    manager.execute(keys[0],10,prepare,lambda r:r.run([4]))
    if manager.snapshot()['failed_execute']!=1 or manager.snapshot()['prepare_count']!=count+1:raise ValueError('failed context not rebuilt')
    manager.invalidate_resources();guards['stale_epoch']=rejected(lambda:manager.execute(keys[0],10,prepare,lambda r:r.run([5])))
    manager.execute(key('0',1),10,prepare,lambda r:r.run([6]))
    def failed_prepare():raise RuntimeError('controlled failed factory')
    guards['failed_factory']=rejected(lambda:manager.execute(key('1',1),10,failed_prepare,lambda r:r.run([6])))
    if manager.snapshot()['failed_prepare']!=1:raise ValueError('failed factory not recorded')
    manager.close();manager.close();guards['closed_manager']=rejected(lambda:manager.execute(key('0',1),10,prepare,lambda r:r.run([1])))
    if not all(guards.values()) or not all(r.closed and r.closes==1 for r in created) or not all(np.array_equal(a,b) for a,b in outputs):raise ValueError('resource guard/ownership/release failed')
    report=dict(status='pass',cycle_control=cycles,lru_two_slot_comparator=dict(prepares=lru_misses,hits=lru_hits),guards=guards,final_counts=manager.snapshot(),all_resources_closed_exactly_once=True,retained_outputs_owned=True,helper_sha256=sha(ROOT/'onnx/qnn/graph_resources.py'),scope='Portable deterministic five-resource three-cycle identity/budget/failure/state-reset/ownership controls with dummy synchronous backend. Actual SDK counts, memory and integrated short recurrence separate; estimates do not certify device allocations.')
    save(Path.cwd()/'graph_resource_control.json',report);print(json.dumps(dict(status='pass',three_cycle_prepares=cycles['prepare_count'],three_cycle_hits=cycles['cache_hit'],guards=len(guards))))
if __name__=='__main__':main()
