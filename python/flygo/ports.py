"""Audited port artifacts and matched retinotopic/shuffled sensory overlays."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .qualify import sha256


def load_ports(path:Path,*,graph_id,features,groups,seed):
    receipt=json.loads(path.with_suffix('.json').read_text())
    expected=dict(graph_id=graph_id,features=features,groups=groups,seed=seed)
    if any(receipt.get(k)!=v for k,v in expected.items()) or sha256(path)!=receipt['sha256']:
        raise ValueError('Port artifact identity or configuration differs')
    with np.load(path,allow_pickle=False) as data:
        ports={k:data[k].copy() for k in ('input_index','output_group','output_scale')}
    return ports,receipt


def infer_columns(source,destination,counts,coordinates,sides,photoreceptors,root_sides,*,type_ids=None):
    """Synapse-count votes to directly annotated, same-side downstream columns.

    Missing/ambiguous coordinates remain explicit. No graph edges are changed.
    Distances are in the two annotation indices, not physical micrometres.
    """
    n=len(coordinates)
    photo=np.zeros(n,bool);photo[photoreceptors]=True
    mapped=np.isfinite(coordinates).all(axis=1)&np.isin(sides,['L','R'])
    eligible=photo[source]&mapped[destination]
    candidates=np.flatnonzero(eligible)
    valid_side=root_sides[source[candidates]]==sides[destination[candidates]]
    edges=candidates[valid_side]
    order=np.argsort(source[edges],kind='stable');edges=edges[order]
    owners,starts=np.unique(source[edges],return_index=True)
    result=dict(coordinates=np.full((n,2),np.nan),confidence=np.zeros(n),
                synapses=np.zeros(n,np.int64),spread=np.full(n,np.nan),
                target_type_agreement=np.full(n,np.nan),
                side_rejected=np.bincount(source[candidates[~valid_side]],minlength=n).astype(np.int32))
    for owner,begin,end in zip(owners,starts,np.r_[starts[1:],len(edges)]):
        selected=edges[begin:end];target=destination[selected]
        columns,membership=np.unique(coordinates[target],axis=0,return_inverse=True)
        mass=np.bincount(membership,weights=counts[selected])
        winner=int(np.argmax(mass));total=float(mass.sum())
        tied=np.count_nonzero(mass==mass[winner])>1
        result['synapses'][owner]=int(total)
        result['confidence'][owner]=0 if tied else mass[winner]/total
        if not tied:result['coordinates'][owner]=columns[winner]
        result['spread'][owner]=np.sqrt(np.average(np.sum((columns-columns[winner])**2,axis=1),weights=mass))
        if type_ids is not None and not tied:
            votes=[]
            for kind in np.unique(type_ids[target]):
                mask=type_ids[target]==kind
                per_type=np.bincount(membership[mask],weights=counts[selected[mask]],minlength=len(columns))
                votes.append(bool(np.count_nonzero(per_type==per_type.max())==1 and np.argmax(per_type)==winner))
            result['target_type_agreement'][owner]=np.mean(votes)
    return result


def retinal_overlay(base,inference,photoreceptors,sides,type_ids,*,seed,confidence=.5,min_synapses=4):
    """Affine index-to-board attachment; shuffle within side/type/channel controls."""
    selected=photoreceptors[(inference['confidence'][photoreceptors]>=confidence)&
                            (inference['synapses'][photoreceptors]>=min_synapses)&
                            np.isfinite(inference['coordinates'][photoreceptors]).all(axis=1)]
    if not len(selected):raise ValueError('No photoreceptors meet the declared audit threshold')
    spatial={k:v.copy() for k,v in base.items()}
    shuffled={k:v.copy() for k,v in base.items()}
    if np.any(base['input_index'][selected]<0):raise ValueError('Overlay cells must already be sensory ports')
    channels=base['input_index']%12
    bounds={}
    for side in ('L','R'):
        cells=selected[sides[selected]==side]
        if not len(cells):continue
        xy=inference['coordinates'][cells]
        low,high=xy.min(axis=0),xy.max(axis=0)
        if np.any(high<=low):raise ValueError('Coordinate extent is degenerate')
        # Direct affine mapping of annotation indices, nearest grid point.
        # This is an explicit adapter hypothesis, not a retinal calibration.
        ij=np.rint(8*(xy-low)/(high-low)).astype(np.int32)
        spatial['input_index'][cells]=(ij[:,1]*9+ij[:,0])*12+channels[cells]
        bounds[side]=dict(min=low.tolist(),max=high.tolist())
    shuffled['input_index'][selected]=spatial['input_index'][selected]
    rng=np.random.default_rng(seed+510307)
    strata=np.stack([(sides[selected]=='R').astype(np.int32),type_ids[selected],channels[selected]],axis=1)
    _,membership=np.unique(strata,axis=0,return_inverse=True)
    for group in np.unique(membership):
        cells=selected[membership==group]
        shuffled['input_index'][cells]=rng.permutation(spatial['input_index'][cells])
    return spatial,shuffled,dict(selected=selected,bounds=bounds,strata=strata)


def visual_readout(base,coordinates,sides,type_ids,bounds,*,seed,groups=656,channels=8):
    """Spatial or matched-shuffled pools of directly annotated visual cells."""
    if channels<1 or groups<81*channels:raise ValueError('Readout groups must fit all board/channel slots')
    candidate=(base['input_index']<0)&np.isfinite(coordinates).all(axis=1)&np.isin(sides,['L','R'])
    selected=[];points=[];outside=0
    for side in ('L','R'):
        cells=np.flatnonzero(candidate&(sides==side))
        if not len(cells):continue
        low,high=np.asarray(bounds[side]['min']),np.asarray(bounds[side]['max'])
        if np.any(high<=low):raise ValueError('Readout coordinate extent is degenerate')
        xy=coordinates[cells]
        valid=((xy>=low)&(xy<=high)).all(axis=1);outside+=int((~valid).sum())
        cells=cells[valid];ij=np.rint(8*(coordinates[cells]-low)/(high-low)).astype(np.int32)
        selected.extend(cells);points.extend(ij[:,1]*9+ij[:,0])
    selected=np.asarray(selected,np.int32);points=np.asarray(points,np.int32)
    if not len(selected):raise ValueError('No annotated readout cells fall within the sensory bounds')
    types,membership=np.unique(type_ids[selected],return_inverse=True)
    rng=np.random.default_rng(seed+610309)
    type_channel=rng.permutation(len(types))%channels
    assigned=points*channels+type_channel[membership]
    spatial={k:v.copy() for k,v in base.items()}
    spatial['output_group'].fill(-1);spatial['output_group'][selected]=assigned
    shuffled={k:v.copy() for k,v in spatial.items()}
    strata=np.stack([(sides[selected]=='R').astype(np.int32),type_ids[selected]],axis=1)
    _,stratum=np.unique(strata,axis=0,return_inverse=True)
    for group in np.unique(stratum):
        cells=selected[stratum==group]
        shuffled['output_group'][cells]=rng.permutation(spatial['output_group'][cells])
    counts=np.bincount(assigned,minlength=groups)
    for ports in (spatial,shuffled):
        actual=np.bincount(ports['output_group'][selected],minlength=groups)
        np.testing.assert_array_equal(actual,counts)
        ports['output_scale'].fill(0)
        ports['output_scale'][selected]=(1/np.sqrt(counts[ports['output_group'][selected]])).astype(np.float32)
    return spatial,shuffled,dict(selected=selected,strata=strata,candidate_cells=int(candidate.sum()),
        outside_sensory_bounds=outside,board_point_coverage=len(np.unique(points)),empty_groups=int((counts==0).sum()),
        group_counts=counts.tolist(),type_channels={str(kind):int(channel) for kind,channel in zip(types,type_channel)})
