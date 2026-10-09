"""Guarded frozen real-frame driver with a measured bounded resource profile."""
import argparse,faulthandler,importlib.util,json,os,resource,sys,threading,time
faulthandler.enable()
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from tool_run import save,sha
import resources
import partition_session
import borrowed_session
from resource_partition_session import ResourcePartitionSession,PROVENANCE
from resource_profile import bind_resource_partitions,start_host_rss_guard

source=ROOT/'onnx/validation/run_qnn_real_frames.py'
EXPECTED_DRIVER='8f3d7bb1b59d26b505b3c3a778df514215ea398d21a500f1c3fb8509fc03dac4'
if sha(source)!=EXPECTED_DRIVER:raise ValueError('resource driver source identity differs')
sessions=[];observed=[];original_bind=partition_session.bind_partitions
options=argparse.ArgumentParser(add_help=False)
options.add_argument('--resource-policy',type=Path,required=True)
options.add_argument('--resource-policy-sha256',required=True)
resource_args,driver_args=options.parse_known_args()
sys.argv=[sys.argv[0]]+driver_args
def bind_candidate(*args,**kwargs):
    return bind_resource_partitions(*args,policy_path=resource_args.resource_policy,policy_sha256=resource_args.resource_policy_sha256,**kwargs)
def session_for(profile):
    session=ResourcePartitionSession(profile,profile['graph_resource_policy']);sessions.append(session);return session
partition_session.bind_partitions=bind_candidate;resources.session_for=session_for
spec=importlib.util.spec_from_file_location('frozen_resource_real_frames',source);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
class OwnedObservation(driver.ObservedSession):
    def __init__(self,session):super().__init__(session);self.held=None;self.held_identity=None;observed.append(self)
    def check_owned(self):
        if self.held is not None:
            import hashlib,numpy as np
            now={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()) for n,v in self.held.items()}
            if now!=self.held_identity:raise ValueError('native warm/close overwrote retained frame outputs')
    def run(self,names,feed):
        self.check_owned();values=super().run(names,feed);self.check_owned()
        if self.held is None:self.held=dict(self.outputs);self.held_identity=dict(self.observation['outputs'])
        owner=self.session;self.observation['resource_counts']=owner.graph_resources.snapshot();self.observation['resource_parts']=owner.resource_observations
        save(Path.cwd()/'resource_frame_progress.json',dict(stage='outputs_owned',resource_counts=owner.graph_resources.snapshot(),resource_parts=owner.resource_observations))
        return values
driver.ObservedSession=OwnedObservation
base_transaction=driver.FixedStateTransaction
class ResourceStateTransaction(base_transaction):
    def advance_planning(self,*args,**kwargs):
        if kwargs.get('new_scene',False):self.session.reset_state()
        return super().advance_planning(*args,**kwargs)
driver.FixedStateTransaction=ResourceStateTransaction
start_host_rss_guard()
try:
    if sha(source)!=EXPECTED_DRIVER:raise ValueError('resource driver source identity differs')
    driver.main()
except BaseException as error:
    path=Path.cwd()/'neural_result.json'
    if not path.exists():save(path,dict(status='failed',stage='resource_driver',error_type=type(error).__name__,error=str(error),task_acceptance='not_evaluated'))
    raise
finally:
    ownership=True
    for item in observed:
        try:item.check_owned()
        except BaseException:ownership=False
    save(Path.cwd()/'resource_frame_observations.json',dict(status='observed',retained_outputs_owned_after_close=ownership,resources=[s.graph_resources.snapshot() for s in sessions],input_transport=borrowed_session.TRANSPORT,provenance=PROVENANCE,source_driver_sha256=sha(source),wrapper_sha256=sha(__file__),scope='Measured resource candidate wrapper around exact frozen real-frame/Host/checkpoint driver; same-state control, task and release/target acceptance separate. No output tensor copies added for held-output control, only retained references/hashes.'))
    if not ownership:raise ValueError('retained real-frame outputs lost ownership')
