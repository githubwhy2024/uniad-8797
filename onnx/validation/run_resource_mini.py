"""Run the exact frozen mini driver with the accepted bounded resource profile."""
import cProfile
import argparse,faulthandler,importlib.util,json,resource,sys
from pathlib import Path
faulthandler.enable()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'onnx/qnn'))
from tool_run import sha,save
from resource_profile import bind_resource_partitions,start_host_rss_guard
from resource_partition_session import ResourcePartitionSession,PROVENANCE
import partition_session,resources,borrowed_session
source=ROOT/'onnx/validation/run_qnn_mini.py'
EXPECTED_DRIVER='559185ef661b6574203421bd8a76873fb24bc8bb05a50909861f3391b41c7ff7'
if sha(source)!=EXPECTED_DRIVER:raise ValueError('frozen resource mini driver changed')
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--resource-policy',type=Path,required=True);parser.add_argument('--resource-policy-sha256',required=True)
options,remaining=parser.parse_known_args();sys.argv=[sys.argv[0]]+remaining
sessions=[]
def bind_candidate(*args,**kwargs):return bind_resource_partitions(*args,policy_path=options.resource_policy,policy_sha256=options.resource_policy_sha256,**kwargs)
def session_for(profile):
    session=ResourcePartitionSession(profile,profile['graph_resource_policy']);sessions.append(session);return session
partition_session.bind_partitions=bind_candidate;resources.session_for=session_for
spec=importlib.util.spec_from_file_location('frozen_resource_mini_driver',source);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
original_save=driver.save
def save_bound_mini_profile(path,value):
    # The frozen driver compares its base against the six-frame profile first.
    # Only its own saved task profile adds this executed wrapper identity, so a
    # changed wrapper cannot silently resume an earlier committed mini journal.
    if Path(path).name=='profile.json' and value.get('task_scope') in ('planning','full'):
        value['assets']['resource_mini_driver']=dict(path=str(Path(__file__).absolute()),sha256=sha(__file__),bytes=Path(__file__).stat().st_size)
    return original_save(path,value)
driver.save=save_bound_mini_profile
class ResourceObserved(driver.ObservedSession):
    def run(self,names,feed):
        values=super().run(names,feed)
        self.observation['resource_counts']=self.session.graph_resources.snapshot();self.observation['resource_parts']=self.session.resource_observations
        return values
driver.ObservedSession=ResourceObserved
base_transaction=driver.host.FixedStateTransaction
class ResourceStateTransaction(base_transaction):
    def advance_planning(self,*args,**kwargs):
        if kwargs.get('new_scene',False):self.session.reset_state()
        return super().advance_planning(*args,**kwargs)
driver.host.FixedStateTransaction=ResourceStateTransaction
start_host_rss_guard()
try:
    if sha(source)!=EXPECTED_DRIVER:raise ValueError('frozen mini driver changed before execution')
    driver.main()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak>18*1024**2:
        save(Path.cwd()/'resource_memory_guard.json',dict(status='rejected',reason='actual_complete_mini_peak_rss_budget_exceeded',peak_rss_kib=peak,ceiling_kib=18*1024**2,scope='Whole-process peak includes dataset/Host/checkpoint and finalization; no task promotion.'))
        raise ValueError('complete mini peak exceeded physical host budget')
except BaseException as error:
    path=Path.cwd()/'mini_result.json'
    if not path.exists():save(path,dict(status='failed',stage='resource_driver',error_type=type(error).__name__,error=str(error),task_acceptance='not_evaluated'))
    raise
finally:
    save(Path.cwd()/'resource_mini_observations.json',dict(status='observed',resources=[s.graph_resources.snapshot() for s in sessions],input_transport=borrowed_session.TRANSPORT,provenance=PROVENANCE,source_driver_sha256=sha(source),wrapper_sha256=sha(__file__),scope='Frozen mini task records/checkpoint driver with the same bound resource policy as six-frame acceptance. Duration/resource counters only; task evaluation separate.'))
