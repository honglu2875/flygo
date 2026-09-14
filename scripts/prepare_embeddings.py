#!/usr/bin/env python3
"""Freeze coherent retinal geometry and genuine recent-divergence replay pairs."""
import argparse
from collections import Counter,defaultdict
import json,hashlib
from pathlib import Path
import time
import numpy as np
import pyarrow.feather as feather

from flygo import _native
from flygo.data.branches import VERSION as PAIR_VERSION,mine_pairs,probe_families
from flygo.data.corpus import atomic_json,identity,split_for_family
from flygo.fly import load_graph
from flygo.go import GameConfig,_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget
from flygo.vision import VERSION as INPUT_VERSION,SphericalRenderer,hex_chart,individual_motor_ports


def prepare(args):
    started=time.time();out=args.output;out.resolve().relative_to(args.root.resolve())
    with StorageBudget(args.root).reserve(files=256<<20,heap=4*GIB,purpose='spherical embedding pair preparation'):
        out.mkdir(parents=True,exist_ok=False)
        source=args.root/'releases/v0-1m/manifest.json'
        manifest=json.loads(source.read_text())
        if identity({k:v for k,v in manifest.items() if k!='dataset_id'}) != manifest['dataset_id']:
            raise ValueError('Frozen source identity mismatch')
        replica=json.loads((source.parent/'replication.json').read_text())
        if replica['status']!='passed' or replica['dataset_id']!=manifest['dataset_id']:
            raise ValueError('Require verified source replication')
        games=[]
        for record in manifest['records']:
            if record['split']!='train':continue
            if split_for_family(record['opening_family'])!='train':raise ValueError('Family split mismatch')
            path=args.root/record['path']
            if sha256(path)!=record['sha256']:raise ValueError('Source game hash mismatch')
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(data['metadata'].tobytes())
                if any(metadata[k]!=record[k] for k in ('game_id','opening_family','split','teacher_sha256','rows')):
                    raise ValueError('Source metadata mismatch')
                games.append(dict(record,actions=data['actions'].copy(),raw_value=data['raw_value'].copy()))
        pairs,audit=mine_pairs(games,seed=args.seed)
        families=defaultdict(list)
        for p in pairs:families[p['opening_family']].append(p)
        held_out=probe_families(families)
        for p in pairs:p['split']='probe' if p['opening_family'] in held_out else 'train'
        rng=np.random.default_rng(args.seed);selected=[]
        for name in sorted(families):
            items=families[name]
            selected.extend(items[i] for i in rng.choice(len(items),min(len(items),32),replace=False))
        # Bound storage before allocating, independent of the release's growth.
        if len(selected)>12000:raise ValueError('Register a larger bank before expanding beyond 12k pairs')
        mining=dict(stage='mined',**audit,selected=len(selected),
                    selected_by_split=dict(Counter(p['split'] for p in selected)),
                    families_by_split={s:len({p['opening_family'] for p in selected if p['split']==s}) for s in ('train','probe')})
        atomic_json(out/'mining.json',mining)
        print(json.dumps(mining),flush=True)
        needed=defaultdict(set)
        for p in selected:
            for k in ('good','bad'):needed[p[k]].add(p['ply'])
        states={}
        for index,plies in needed.items():
            actions=np.ascontiguousarray(games[index]['actions'],np.int32)
            x=_native.replay_features(_json(GameConfig()),actions,np.array([0,len(actions)],np.int64)).reshape(-1,9,9,12)
            for ply in plies:states[index,ply]=x[ply].copy()
        x=[];kept=[];identical=0
        for p in selected:
            good,bad=states[p['good'],p['ply']],states[p['bad'],p['ply']]
            # Reject targets distinguishable only through omitted context/long history.
            if np.array_equal(good[...,:8],bad[...,:8]):identical+=1;continue
            if not np.array_equal(good[0,0,8:11],bad[0,0,8:11]):continue
            x.append(np.stack([good,bad]));kept.append(p)
        if not kept:raise ValueError('No eligible causal pairs')
        features=np.asarray(x,np.float32)
        def visual_key(x):
            visual=x[...,:8]
            return min(hashlib.sha256(np.rot90(np.flip(visual,axis=1) if flip else visual,
                turns,axes=(0,1)).tobytes()).digest() for flip in (False,True) for turns in range(4))
        train_keys={visual_key(x) for i,p in enumerate(kept) if p['split']=='train' for x in features[i]}
        novel=np.asarray([p['split']=='train' or all(visual_key(x) not in train_keys for x in features[i])
                          for i,p in enumerate(kept)])
        overlap_rejected=int(np.sum(~novel));features=features[novel]
        kept=[p for p,valid in zip(kept,novel) if valid]
        split=np.asarray([p['split'] for p in kept])
        if min(np.sum(split=='train'),np.sum(split=='probe'))<32:
            raise ValueError('Need at least 32 training and 32 family-held-out probe pairs')
        np.savez(out/'pairs.npz',features=features,split=split,
                 quality=np.asarray([[games[p[k]]['raw_value'][p['ply']] for k in ('good','bad')] for p in kept],np.float32))
        records=[]
        for p in kept:
            records.append({**p,**{k:games[p[k]]['game_id'] for k in ('good','bad')}})
        atomic_json(out/'pairs.json',records)
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path);table=feather.read_table(graph_path/'annotations.feather')
        columns=args.root/'ports/retinal-overlay-v1/columns.npz'
        receipt=json.loads((columns.parent/'result.json').read_text())
        if (receipt['graph_id']!=graph['manifest']['graph_id'] or receipt['columns_sha256']!=sha256(columns)
                or receipt['annotation_sha256']!=sha256(graph_path/'annotations.feather')):
            raise ValueError('Atlas provenance mismatch')
        with np.load(columns,allow_pickle=False) as data:
            np.testing.assert_array_equal(data['body_id'],table['bodyId'].to_numpy())
            photo=data['photoreceptors']
            qualified=np.isfinite(data['coordinates'][photo]).all(axis=1)&(data['confidence'][photo]>=.5)&(data['synapses'][photo]>=4)
            sensors=photo[qualified];coordinates=data['coordinates'][sensors].copy()
        side=table['rootSide'].to_numpy()[sensors].astype(str)
        kinds=table['superclass'].to_numpy()
        motors=np.flatnonzero(np.isin(kinds,['descending_neuron','cb_motor','vnc_motor']))
        reference=np.stack([table[k].to_numpy() for k in ('assignedOlHex1','assignedOlHex2')],axis=1)
        reference=np.unique(reference[np.isfinite(reference).all(axis=1)],axis=0)
        origin=(reference.min(axis=0)+reference.max(axis=0))/2
        qr=reference-origin;xy=np.column_stack([qr[:,0]-.5*qr[:,1],np.sqrt(3)/2*qr[:,1]])
        scale=np.deg2rad(80)/np.linalg.norm(xy,axis=1).max()
        unit=hex_chart(coordinates,origin=origin,radians_per_column=scale)
        patch_spec=dict(patch_centers_degrees=[[-25.,-30.],[-25.,30.]],halfwidth_degrees=25.,sigma_degrees=5.,neighbors=4)
        renderer=SphericalRenderer.build(unit,side,**patch_spec)
        geometry_audit=renderer.audit()
        if any(row['covered_board_points']!=81 for row in geometry_audit):
            raise ValueError('Spherical adapter leaves a board point unobserved: '+json.dumps(geometry_audit))
        sampling=np.zeros((len(sensors),324),np.float64)
        np.add.at(sampling,(np.arange(len(sensors))[:,None],renderer.index),renderer.weight)
        singular=np.linalg.svd(sampling,compute_uv=False)
        rank=int(np.sum(singular>1e-6*singular[0]))
        if rank!=324:raise ValueError('Visual sampler does not distinguish all four boards')
        ports=individual_motor_ports(len(kinds),sensors,motors)
        np.savez(out/'renderer.npz',**renderer.arrays(),sensors=sensors,motors=motors,
                 sensor_body_id=table['bodyId'].to_numpy()[sensors],motor_body_id=table['bodyId'].to_numpy()[motors],
                 motor_superclass=kinds[motors].astype(str),**ports)
        input_data=renderer.render(features[:64].reshape(-1,9,9,12))
        contract=dict(schema_version=1,status='complete',created=time.time(),seconds=time.time()-started,
            graph_id=graph['manifest']['graph_id'],dataset_id=manifest['dataset_id'],source_manifest_sha256=sha256(source),
            teacher_sha256=sorted({g['teacher_sha256'] for g in games}),input_version=INPUT_VERSION,pair_version=PAIR_VERSION,
            source_game_sha256={g['game_id']:g['sha256'] for i,g in enumerate(games) if i in needed},
            seed=args.seed,pair_audit=dict(audit,retained=len(kept),identical_visual_pairs_rejected=identical,
                probe_pairs_rejected_for_training_visual_overlap=overlap_rejected,
                probe_visual_D4_overlap=0,
                probe_rule='exact quarter of eligible families by fixed hash order, frozen before training',
                probe_families=sorted(held_out),
                pairs_by_split=dict(Counter(split)),families_by_split={s:len({p['opening_family'] for p in kept if p['split']==s}) for s in ('train','probe')},
                divergence_counts=dict(Counter(str(p['divergence']) for p in kept)),
                ply_quantiles=np.quantile([p['ply'] for p in kept],[0,.1,.5,.9,1]).tolist()),
            geometry=dict(kind='engineered common hex exponential chart, not measured optical axes',
                axial_convention='120 degrees: (1,0), (0,1), (1,1) are neighbors; orientation unregistered',
                origin=origin.tolist(),radians_per_column=float(scale),reference_columns=len(reference),
                reference_cap_degrees=80,per_eye_rescaling=False,reflection=False,
                patch_spec=patch_spec,sampling_rank=rank,sampling_condition=float(singular[0]/singular[-1]),
                minimum_singular=float(singular[-1]),
                sensors=len(sensors),excluded_photoreceptors=len(photo)-len(sensors),motors=len(motors),patches=geometry_audit,
                luminance='own .95, opponent .05, empty/missing/uncovered .5; same current-player perspective in both eyes',
                context='No legality, turn, komi or pass channels in visual drive. Pair members match ply and global context. '
                        'This ordinal representation pilot does not produce calibrated values or a legal Go policy.',
                input_range=[float(input_data.min()),float(input_data.max())]),
            files={name:sha256(out/name) for name in ('pairs.npz','pairs.json','renderer.npz')},
            code_sha256={str(Path(p).resolve()):sha256(Path(p)) for p in (__file__,)},
            scope='Training-release-only discovery/probe split by original D4 opening family. Exact common prefixes >=20 plies, '
                  'endpoints 1–6 plies later at the same turn; raw teacher value gap >=.2. Bounded game-pair sampling is not exhaustive.')
        atomic_json(out/'result.json',contract)
        print(json.dumps(dict(status='complete',pairs=contract['pair_audit'],geometry=contract['geometry'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=918420)
    p.add_argument('--cpus',default='119')
    args=p.parse_args();pin([int(c) for c in args.cpus.split(',')]);prepare(args)


if __name__=='__main__':main()
