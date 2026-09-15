#!/usr/bin/env python3
"""Freeze external head constraints from the audited individual motor inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from flygo.attachments import load_attachment
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.readout import HeadMask, side_policy, shuffle_sides, load_head_mask
import flygo.readout as readout_module
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root;plan_bytes=args.plan.read_bytes();plan=json.loads(plan_bytes);pin([56,57,58,59])
    plan_sha256=hashlib.sha256(plan_bytes).hexdigest()
    args.output.resolve().relative_to(root.resolve())
    with StorageBudget(root).reserve(files=8<<20,heap=2*GIB,purpose='immutable readout-mask preparation'):
        if args.output.exists():raise ValueError('Prepared head-mask directories are immutable')
        attachment=root/plan['input_map']
        if sha256(attachment)!=plan['input_map_sha256']:raise ValueError('Registered input map differs')
        _,ports,receipt=load_attachment(attachment,graph_id=plan['graph_id'])
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        annotation=graph/'annotations.feather'
        if sha256(annotation)!=plan['annotation_sha256']:raise ValueError('Registered annotations differ')
        for evidence in plan['development_evidence']:
            path=root/evidence['path']
            if sha256(path)!=evidence['sha256'] or json.loads(path.read_text())['status']!='complete':
                raise ValueError('Candidate-discovery or input-selection evidence differs')
        with np.load(attachment,allow_pickle=False) as data:motors=data['motors'].copy()
        if (len(motors)!=plan['groups'] or receipt['groups']!=plan['groups']
                or not np.array_equal(ports['output_group'][motors],np.arange(len(motors)))
                or np.count_nonzero(ports['output_group']>=0)!=len(motors)):
            raise ValueError('Every output coordinate must represent one declared motor')
        columns=['bodyId','type','superclass','somaSide','rootSide']
        table=feather.read_table(annotation,columns=columns).take(motors)
        rows=table.to_pylist();sides=np.asarray(table['somaSide'].to_pylist());kinds=np.asarray(table['superclass'].to_pylist())
        counts={key:int(np.count_nonzero(sides==key)) for key in ('L','R','M')}
        if counts!=plan['expected_soma_sides'] or any(row['rootSide'] not in (None,'') for row in rows):
            raise ValueError('Audited soma-side counts or absent root-side fields changed')
        selected=np.asarray([row['type'] in plan['value_types'] for row in rows],np.uint8)
        bodies=sorted(row['bodyId'] for row,use in zip(rows,selected) if use)
        if bodies!=plan['expected_value_body_ids']:raise ValueError('Complete candidate cell types changed')
        shuffled=shuffle_sides(sides,kinds,seed=plan['shuffle_seed'])
        anatomical=side_policy(sides,size=plan['size']);control=side_policy(shuffled,size=plan['size'])
        np.testing.assert_array_equal(anatomical.sum(axis=1),control.sum(axis=1))
        np.testing.assert_array_equal(shuffled[sides=='M'],sides[sides=='M'])
        for kind in np.unique(kinds):
            np.testing.assert_array_equal(np.sort(shuffled[kinds==kind]),np.sort(sides[kinds==kind]))
        matrices={
            'dense':(np.ones((plan['actions'],plan['groups']),np.uint8),np.ones(plan['groups'],np.uint8)),
            'value-group':(np.ones((plan['actions'],plan['groups']),np.uint8),selected),
            'soma-side':(anatomical,np.ones(plan['groups'],np.uint8)),
            'shuffled-side':(control,np.ones(plan['groups'],np.uint8)),
        }
        if set(plan['arms'])!=set(matrices):raise ValueError('Unexpected head-mask arms')
        args.output.mkdir(parents=True)
        (args.output/'plan.json').write_bytes(plan_bytes)
        records=[]
        for arm in plan['arms']:
            provenance=dict(kind=arm,graph_id=plan['graph_id'],attachment_sha256=receipt['sha256'],
                annotation_sha256=plan['annotation_sha256'],preparation_plan_sha256=plan_sha256,
                motor_order_sha256=hashlib.sha256(np.asarray(motors,dtype='<i8').tobytes()).hexdigest())
            mask=HeadMask(*matrices[arm],json.dumps(provenance));mask.validate_shape(actions=plan['actions'],groups=plan['groups'])
            path=args.output/(arm+'.npz');np.savez(path,**mask.checkpoint_arrays())
            record=dict(status='complete',sha256=sha256(path),contract=mask.contract,
                policy_fan_in=mask.policy.sum(axis=1).tolist(),value_body_ids=[row['bodyId'] for row,use in zip(rows,mask.value) if use])
            atomic_json(path.with_suffix('.json'),record)
            recovered=load_head_mask(path,graph_id=plan['graph_id'],attachment_sha256=receipt['sha256'])
            if recovered.contract!=mask.contract:raise AssertionError('Mask artifact did not recover exactly')
            records.append(dict(arm=arm,path=str(path.relative_to(root)),sha256=record['sha256'],contract=mask.contract))
        inventory=[dict(group=i,node=int(node),**row,shuffled_side=str(shuffled[i])) for i,(node,row) in enumerate(zip(motors,rows))]
        atomic_json(args.output/'inventory.json',dict(motors=inventory,soma_side_counts=counts,value_body_ids=bodies))
        atomic_json(args.output/'result.json',dict(status='complete',created=time.time(),plan=plan,plan_sha256=plan_sha256,
            inventory_sha256=sha256(args.output/'inventory.json'),records=records,source_sha256=sha256(Path(__file__)),
            mask_source_sha256=sha256(Path(readout_module.__file__)),
            scope='No network evaluations, label access or optimizer updates; masks constrain only the external decoder.'))
        print(json.dumps(dict(status='complete',output=str(args.output),arms=[dict(arm=r['arm'],policy=r['contract']['policy_coefficients'],value=r['contract']['value_coefficients']) for r in records])),flush=True)


if __name__=='__main__':main()
