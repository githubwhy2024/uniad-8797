#!/usr/bin/env python3
"""Observe optimizer validator exceptions without changing converter decisions."""
import json,runpy,sys,resource,functools
import numpy as np
from pathlib import Path
from qti.aisw.converters.common.converter_ir.op_graph import IROpGraph

original=IROpGraph.get_matched_nodes

def observe(self,sequence,validator=None,ignore_constants=False,use_dfs=False):
    def inspected(nodes):
        try:return validator(nodes)
        except Exception as error:
            rows=[dict(name=n.op.name,op_type=n.op.type,inputs=list(n.input_names),outputs=list(n.output_names),input_shapes=[list(s) for s in self.get_input_shapes(n)],output_shapes=[list(s) for s in self.get_output_shapes(n)]) for n in nodes]
            evidence=dict(error_type=type(error).__name__,error=str(error),nodes=rows,scope='Diagnostic only; original exception is re-raised and optimization decisions unchanged.')
            Path('optimizer_exception.json').write_text(json.dumps(evidence,indent=2)+'\n')
            print(json.dumps(dict(optimizer_exception=evidence)),file=sys.stderr,flush=True)
            raise
    return original(self,sequence,inspected if validator is not None else None,ignore_constants,use_dfs)

IROpGraph.get_matched_nodes=observe
from qti.aisw.converters.backend.ir_to_qnn import QnnConverterBackend
active_graph = None
original_get_ir_graph = QnnConverterBackend.get_ir_graph
def observe_backend_graph(self, graph):
    global active_graph
    active_graph = graph
    return original_get_ir_graph(self, graph)
QnnConverterBackend.get_ir_graph = observe_backend_graph

def describe_buffer(name):
    buffer = active_graph.get_buffer(name)
    producer = buffer.producer
    return dict(name=name, shape=list(buffer.shape), axis_format=str(buffer.axis_format),
                producer=producer.op.name, producer_type=producer.op.type,
                producer_inputs=[dict(name=n, shape=list(active_graph.get_buffer(n).shape),
                                      axis_format=str(active_graph.get_buffer(n).axis_format)) for n in producer.input_names])

progress=dict(events=0,last_saved_rss_kib=0)
def tracked(method):
    @functools.wraps(method)
    def call(self,*args,**kwargs):
        progress['events']+=1
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row=dict(event=method.__name__,node=str(args[0]) if args else None,max_rss_kib=rss,events=progress['events'])
        if method.__name__=='add_tensor' and len(args)>3:
            row.update(tensor=str(args[1]),object_type=type(args[3]).__name__)
            if isinstance(args[3],np.ndarray):row.update(shape=list(args[3].shape),bytes=int(args[3].nbytes))
        if progress['events']%256==0 or rss-progress['last_saved_rss_kib']>256*1024:
            Path('backend_progress.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(backend_progress=row)),file=sys.stderr,flush=True)
            progress['last_saved_rss_kib']=rss
        try:return method(self,*args,**kwargs)
        except Exception as error:
            row.update(error_type=type(error).__name__,error=str(error))
            if method.__name__ == 'add_node' and len(args) > 2 and active_graph is not None:
                try:
                    row['optimized_inputs'] = [describe_buffer(name) for name in args[2]]
                    row['optimized_outputs'] = [describe_buffer(name) for name in active_graph.nodes_by_name[str(args[0])].output_names]
                except Exception as diagnostic_error:
                    row['snapshot_error'] = str(diagnostic_error)
            Path('backend_exception.json').write_text(json.dumps(row,indent=2)+'\n')
            raise
    return call
QnnConverterBackend.add_tensor=tracked(QnnConverterBackend.add_tensor)
QnnConverterBackend.add_node=tracked(QnnConverterBackend.add_node)
from sdk_environment import sdk_root
sdk=sdk_root()/'bin/x86_64-linux-clang/qnn-onnx-converter'
sys.argv=[str(sdk)]+sys.argv[1:]
runpy.run_path(str(sdk),run_name='__main__')
