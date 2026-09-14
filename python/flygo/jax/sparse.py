"""Exact sparse multiplication with bounded gather tiles and an explicit transpose VJP.

The layout changes storage only. Edge IDs still index the original parameter
vector. Padding has zero weight and is never an optimizer parameter.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

LAYOUT_VERSION = 'degree-buckets-v1'


def build_layout(source, destination, neurons, *, tile_edges=32768, tile_rows=128):
    source=np.asarray(source,np.int32);destination=np.asarray(destination,np.int32)
    if source.ndim!=1 or source.shape!=destination.shape or not len(source):
        raise ValueError('Expected nonempty aligned edge arrays')
    if tile_edges<1 or tile_rows<1 or neurons<1 or min(source.min(),destination.min())<0 \
            or max(source.max(),destination.max())>=neurons:
        raise ValueError('Invalid graph or tile bound')
    edges=len(source)

    def layout(src,dst):
        order=np.argsort(dst,kind='stable').astype(np.int32)
        counts=np.bincount(dst,minlength=neurons)
        indptr=np.r_[0,np.cumsum(counts)]
        buckets=[];inverse=np.empty(neurons,np.int32);offset=0
        degree=1
        while degree<=max(1,int(counts.max()))*2:
            rows=np.flatnonzero((counts<=degree)&((counts>degree//2) if degree>1 else True))
            if len(rows):
                block_rows=max(1,min(tile_rows,tile_edges//degree))
                padded=((len(rows)+block_rows-1)//block_rows)*block_rows
                positions=indptr[rows,None]+np.arange(degree)
                valid=np.arange(degree)<counts[rows,None]
                canonical=order[np.minimum(positions,edges-1)]
                edge_ids=np.full((padded,degree),edges,np.int32)
                neighbors=np.zeros((padded,degree),np.int32)
                edge_ids[:len(rows)]=np.where(valid,canonical,edges)
                neighbors[:len(rows)]=np.where(valid,src[canonical],0)
                shape=(-1,block_rows,degree)
                buckets.append(dict(edge_ids=edge_ids.reshape(shape),neighbors=neighbors.reshape(shape)))
                inverse[rows]=np.arange(offset,offset+len(rows),dtype=np.int32)
                offset+=padded
            if degree>=int(counts.max()):
                break
            degree*=2
        return dict(buckets=tuple(buckets),inverse=inverse)

    size=((edges+tile_edges-1)//tile_edges)*tile_edges
    def chunks(array):
        padded=np.zeros(size,np.int32);padded[:edges]=array
        return padded.reshape(-1,tile_edges)
    return dict(forward=layout(source,destination),transpose=layout(destination,source),
                edge_source=chunks(source),edge_destination=chunks(destination))


def _multiply(weights,state,layout):
    weights=jnp.concatenate([weights,jnp.zeros(1,weights.dtype)])
    outputs=[]
    for bucket in layout['buckets']:
        def step(_,indices):
            edge_ids,neighbors=indices
            values=weights[edge_ids][...,None]*state[neighbors]
            return None,jnp.sum(values,axis=1)
        _,result=jax.lax.scan(step,None,(bucket['edge_ids'],bucket['neighbors']))
        outputs.append(result.reshape(-1,state.shape[1]))
    return jnp.concatenate(outputs,axis=0)[layout['inverse']]


@jax.custom_vjp
def multiply(weights,state,layout):
    return _multiply(weights,state,layout['forward'])


def _forward(weights,state,layout):
    return _multiply(weights,state,layout['forward']),(weights,state,layout)


def _backward(residual,cotangent):
    weights,state,layout=residual
    def edge_step(_,indices):
        source,destination=indices
        gradient=jnp.sum(state[source]*cotangent[destination],axis=1)
        return None,gradient
    _,gradient=jax.lax.scan(edge_step,None,(layout['edge_source'],layout['edge_destination']))
    gradient=gradient.reshape(-1)[:weights.shape[0]]
    propagated=_multiply(weights,cotangent,layout['transpose'])
    return gradient,propagated,None


multiply.defvjp(_forward,_backward)
