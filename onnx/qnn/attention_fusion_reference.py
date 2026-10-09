"""Backend-independent FP32 attention grouping reference; no target SDK kernel."""
import numpy as np
from onnx import helper as H,numpy_helper as N


def level_reduced_nodes(grids,features,weights,output,shape,chunk_queries,prefix,attributes,packed_multiply=True):
    """Avoid sampled-level concatenation by reducing each level before summing.

    Retains level/point order and sampler attributes. Four short reductions
    change FP32 addition grouping, so task acceptance is still mandatory before
    any full-model promotion. This is an optional graph prototype, not the
    default Q4 model or a claim about SDK scratch allocation.
    """
    batch,channels,queries,points=shape
    if points!=32 or len(grids)!=4 or len(features)!=4 or chunk_queries<1:
        raise ValueError('four levels/eight points and positive query block required')
    nodes=[];ordinal=0
    def op(kind,inputs,**attrs):
        nonlocal ordinal
        name=prefix+'/'+str(ordinal);ordinal+=1;nodes.append(H.make_node(kind,inputs,[name],name=name,**attrs));return name
    def constant(values):return op('Constant',[],value=N.from_array(np.array(values,dtype=np.int64)))
    def view(value,dims):return op('Reshape',[value,constant(dims)])
    reduced=[]
    for start in range(0,queries,chunk_queries):
        end=min(start+chunk_queries,queries);width=end-start;first,last=constant([start]),constant([end]);ga,wa=constant([1]),constant([2]);block_weight=op('Slice',[weights,first,last,wa]);accumulator=None
        for level,(grid,feature) in enumerate(zip(grids,features)):
            local_grid=op('Slice',[grid,first,last,ga]);sample=op('GridSample',[feature,local_grid],**attributes)
            local_weight=op('Slice',[block_weight,constant([level*8]),constant([(level+1)*8]),constant([3])])
            if packed_multiply:
                product=op('Mul',[view(sample,[batch,channels,width*8]),view(local_weight,[batch,1,width*8])]);product=view(product,[batch,channels,width,8])
            else:product=op('Mul',[sample,local_weight])
            partial=op('ReduceSum',[product,constant([-1])],keepdims=0)
            accumulator=partial if accumulator is None else op('Add',[accumulator,partial])
        reduced.append(accumulator)
    nodes.append(H.make_node('Concat',reduced,[output],name=prefix+'/reduced',axis=2));return nodes
