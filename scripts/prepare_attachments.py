#!/usr/bin/env python3
"""Freeze the qualified spherical map plus a shared nonvisual context route."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.feather as feather
from flygo.attachments import VERSION, context_ports, load_attachment
from flygo.data.corpus import atomic_json
from flygo.fly import load_graph
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='119')
    args=p.parse_args();pin(list(map(int,args.cpus.split(','))))
    args.output.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=64<<20,heap=4*GIB,purpose='spherical/context input preparation'):
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists() or args.output.with_suffix('.json').exists():
            raise ValueError('Attachment artifacts are immutable')
        root=args.root;bank=json.loads((args.bank/'result.json').read_text())
        source=args.bank/'renderer.npz'
        if bank['status']!='complete' or sha256(source)!=bank['files']['renderer.npz']:
            raise ValueError('Spherical geometry bank identity mismatch')
        path=Path(json.loads((root/'runs/m4/graph.json').read_text())['path']);graph=load_graph(path)
        if graph['manifest']['graph_id']!=bank['graph_id']:raise ValueError('Graph identity mismatch')
        atlas=json.loads((root/'ports/retinal-overlay-v1/result.json').read_text())
        if atlas['annotation_sha256']!=sha256(path/'annotations.feather'):
            raise ValueError('Annotation identity mismatch')
        table=feather.read_table(path/'annotations.feather')
        superclass=table['superclass'].to_numpy().astype(str)
        kinds=table['type'].to_numpy().astype(str)
        outgoing=np.bincount(graph['src'],minlength=len(superclass))
        candidates=np.asarray(graph['sensory'])
        context=candidates[(superclass[candidates]!='ol_sensory') &
            ~np.char.startswith(kinds[candidates],'R') & (outgoing[candidates]>0)]
        with np.load(source,allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in ('index','weight','unit','side','patch','sensors','motors',
                'sensor_body_id','motor_body_id','motor_superclass')}
        np.testing.assert_array_equal(table['bodyId'].to_numpy()[arrays['sensors']],arrays['sensor_body_id'])
        np.testing.assert_array_equal(table['bodyId'].to_numpy()[arrays['motors']],arrays['motor_body_id'])
        seed=918421
        ports=context_ports(len(superclass),arrays['sensors'],arrays['motors'],context,seed=seed)
        np.savez(args.output,**arrays,context_nodes=context,
                 context_body_id=table['bodyId'].to_numpy()[context],**ports)
        receipt=dict(version=VERSION,status='complete',created=time.time(),sha256=sha256(args.output),
            graph_id=bank['graph_id'],dataset_id=bank['dataset_id'],context_seed=seed,
            features=len(arrays['sensors'])+84,groups=len(arrays['motors']),
            visual_bank_sha256=sha256(args.bank/'result.json'),renderer_sha256=sha256(source),
            annotation_sha256=atlas['annotation_sha256'],script_sha256=sha256(Path(__file__)),
            geometry=bank['geometry'],context=dict(nodes=len(context),
                classes=dict(Counter(superclass[context])),
                feature_node_counts=np.bincount(ports['input_index'][context]-len(arrays['sensors']),minlength=84).tolist(),
                selection='Existing sensory inventory, excluding optic sensory and R-prefixed types; outgoing degree > 0. No zero-in-degree criterion.',
                encoding='81 point legalities in board order; black-to-play; current-player signed komi/81; consecutive passes/2.',
                interpretation='Balanced seeded engineering assignment shared across arms, not an inferred biological code for Go context.'),
            modes=dict(history='Unchanged L 0/2, R 1/3 montage.',
                       current='Lag zero in all existing patch locations; geometry and drive gains unchanged.',
                       neutral='Every photoreceptor at .5; identical nonvisual context.'))
        atomic_json(args.output.with_suffix('.json'),receipt)
        load_attachment(args.output,graph_id=bank['graph_id'],dataset_id=bank['dataset_id'])
        print(json.dumps(dict(status='complete',path=str(args.output),features=receipt['features'],
                             groups=receipt['groups'],context=receipt['context'])),flush=True)


if __name__=='__main__':main()
