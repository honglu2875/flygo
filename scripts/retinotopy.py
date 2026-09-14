#!/usr/bin/env python3
"""Audit retained optic columns and freeze matched sensory attachment artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,initialize,load_graph
from flygo.ports import infer_columns,retinal_overlay,visual_readout
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seeds',type=lambda x:list(map(int,x.split(','))),default=[1,2,3])
    p.add_argument('--groups',type=int,default=656)
    p.add_argument('--confidence',type=float,default=.5)
    p.add_argument('--min-synapses',type=int,default=4)
    p.add_argument('--visual-readout',action='store_true',help='Also freeze spatial/shuffled visual readouts with the spatial input held fixed')
    p.add_argument('--cpus',type=lambda x:list(map(int,x.split(','))),default=[117,118,119])
    args=p.parse_args();pin(args.cpus)
    if not .5<=args.confidence<=1 or args.min_synapses<1:p.error('Require majority confidence and positive synapse mass')
    out=args.output;out.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=128*(1<<20),heap=2*GIB,purpose='retinotopic sensory audit'):
        out.mkdir(parents=True,exist_ok=False)
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path)
        manifest=graph['manifest']
        def extra(name):
            path=graph_path/(name+'.npy')
            if sha256(path)!=manifest['arrays'][name]['sha256']:raise ValueError('Graph annotation hash mismatch')
            return np.load(path,mmap_mode='r')
        body,count=extra('body_id'),extra('synapse_count')
        annotation=feather.read_table(graph_path/'annotations.feather')
        np.testing.assert_array_equal(annotation['bodyId'].to_numpy(),body)
        coord=np.stack([annotation[name].to_numpy() for name in ('assignedOlHex1','assignedOlHex2')],axis=1)
        root_side=annotation['rootSide'].to_numpy()
        soma_side=annotation['somaSide'].to_numpy()
        side=np.where(np.isin(soma_side,['L','R']),soma_side,root_side)
        types=annotation['type'].to_numpy()
        sensory=graph['sensory']
        photo=sensory[annotation['class'].to_numpy()[sensory]=='visual']
        inferred=infer_columns(graph['src'],graph['dst'],count,coord,side,photo,root_side,type_ids=graph['type_id'])
        np.savez(out/'columns.npz',**inferred,photoreceptors=photo,body_id=body)
        report=dict(schema_version=1,graph_id=manifest['graph_id'],annotation_sha256=sha256(graph_path/'annotations.feather'),
                    created=time.time(),direct_columns=int(np.isfinite(coord).all(axis=1).sum()),
                    direct_sensory_columns=int(np.isfinite(coord[sensory]).all(axis=1).sum()),
                    photoreceptors=len(photo),photo_types=dict(Counter(types[photo])),
                    inferred=int(np.isfinite(inferred['coordinates'][photo]).all(axis=1).sum()),
                    missing_or_tied=int(np.isnan(inferred['coordinates'][photo,0]).sum()),
                    side_rejected_edges=int(inferred['side_rejected'].sum()),
                    confidence_quantiles=np.quantile(inferred['confidence'][photo],[0,.1,.5,.9,1]).tolist(),
                    target_type_agreement_quantiles=np.nanquantile(inferred['target_type_agreement'][photo],[0,.1,.5,.9,1]).tolist(),
                    spread_index_rms_quantiles=np.nanquantile(inferred['spread'][photo],[0,.1,.5,.9,1]).tolist(),
                    columns_sha256=sha256(out/'columns.npz'),
                    rule='Outgoing canonical-edge synapse-count majority to directly annotated same-side columns; ties remain missing',
                    threshold=dict(confidence=args.confidence,min_synapses=args.min_synapses),artifacts=[])
        for seed in args.seeds:
            cfg=FlyConfig(groups=args.groups,seed=seed)
            base,_=initialize(graph,cfg)
            spatial,shuffled,audit=retinal_overlay(base,inferred,photo,root_side,graph['type_id'],
                seed=seed,confidence=args.confidence,min_synapses=args.min_synapses)
            selected=audit.pop('selected');strata=audit.pop('strata')
            np.testing.assert_array_equal(np.sort(spatial['input_index'][selected]),np.sort(shuffled['input_index'][selected]))
            for variant,ports in [('spatial',spatial),('shuffled',shuffled)]:
                path=out/f'{variant}-g{cfg.groups}-s{seed}.npz';np.savez(path,**ports)
                receipt=dict(schema_version=1,graph_id=manifest['graph_id'],features=cfg.features,groups=cfg.groups,seed=seed,
                    variant='retinal-overlay-v1-'+variant,sha256=sha256(path),columns_sha256=report['columns_sha256'],
                    selected=len(selected),threshold=report['threshold'],bounds=audit['bounds'],
                    input_feature_counts=np.bincount(ports['input_index'][sensory],minlength=cfg.features).tolist(),
                    changed=int(np.count_nonzero(ports['input_index']!=base['input_index'])),
                    attachment='Per-side annotation indices affinely rescaled to 0..8; nearest square point; original input channel retained',
                    control='Shuffle assigned board points within side, photoreceptor type and original feature channel',
                    unchanged='All other sensory assignments, output pools, parameters and topology')
                atomic_json(path.with_suffix('.json'),receipt)
                report['artifacts'].append(dict(path=str(path),**{k:receipt[k] for k in ('sha256','seed','variant','selected','changed')}))
            report['selected_by_type']=dict(Counter(types[selected]))
            report['selected_by_side']=dict(Counter(root_side[selected]))
            report['spatial_board_point_coverage']=len(np.unique(spatial['input_index'][selected]//12))
            if args.visual_readout:
                local,scrambled,readout_audit=visual_readout(spatial,coord,side,graph['type_id'],audit['bounds'],
                                                          seed=seed,groups=cfg.groups)
                cells=readout_audit.pop('selected');readout_audit.pop('strata')
                for variant,ports in [('spatial',local),('shuffled',scrambled)]:
                    path=out/f'visual-readout-{variant}-g{cfg.groups}-s{seed}.npz';np.savez(path,**ports)
                    receipt=dict(schema_version=1,graph_id=manifest['graph_id'],features=cfg.features,groups=cfg.groups,seed=seed,
                        variant='spatial-input-visual-readout-v1-'+variant,sha256=sha256(path),
                        input_source=str(out/f'spatial-g{cfg.groups}-s{seed}.npz'),columns_sha256=report['columns_sha256'],
                        readout_neurons=len(cells),**readout_audit,
                        side_rule='somaSide when L/R, otherwise rootSide; sensory input uses rootSide',
                        attachment='Directly annotated columns within the same sensory bounds; 81 board points by 8 seeded type channels; inverse-sqrt actual pool counts',
                        control='Shuffle output groups within side and type; identical cells, input, pool counts, parameter shapes and full recurrent topology')
                    atomic_json(path.with_suffix('.json'),receipt)
                    report['artifacts'].append(dict(path=str(path),**{k:receipt[k] for k in ('sha256','seed','variant','readout_neurons')}))
        atomic_json(out/'result.json',dict(status='passed',**report))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
