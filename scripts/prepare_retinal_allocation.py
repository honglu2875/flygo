#!/usr/bin/env python3
"""Qualify the registered one-board-per-eye map without changing neural ports."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.feather as feather

from flygo.attachments import VisualContext, load_attachment
from flygo.data.corpus import atomic_json
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB, StorageBudget
from flygo.vision import SphericalRenderer
from probe_biological_ports import probes


SPEC = dict(patch_centers_degrees=[[-25.,0.]], halfwidth_degrees=[25.,55.],
            sigma_degrees=5., neighbors=4, patch_lags=[[0,0]])


def sampling(renderer):
    matrix=np.zeros((len(renderer.index),81),np.float64)
    np.add.at(matrix,(np.arange(len(matrix))[:,None],renderer.index%81),renderer.weight)
    return matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();pin([56,57,58,59]);root=args.root
    args.output.resolve().relative_to(root.resolve())
    if args.output.exists():raise ValueError('Geometry studies are immutable')
    with StorageBudget(root).reserve(files=32<<20,heap=4*GIB,purpose='one-board-per-eye geometry qualification'):
        args.output.mkdir(parents=True)
        graph=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        graph_id=json.loads((graph/'manifest.json').read_text())['graph_id']
        reference,ports,receipt=load_attachment(args.source,graph_id=graph_id)
        if receipt['sha256']!='9ce4b1d99e499871a540d06f06dccc563e2e9c77e9ad277a6f2399a04345e087':
            raise ValueError('The registered reference attachment differs')
        with np.load(args.source,allow_pickle=False) as data:arrays={key:data[key].copy() for key in data.files}
        rebuilt=SphericalRenderer.build(arrays['unit'],arrays['side'],**receipt['geometry']['patch_spec'])
        for key,value in rebuilt.arrays().items():
            if value.tobytes()!=arrays[key].tobytes():raise ValueError('Reference renderer changed: '+key)
        candidate=SphericalRenderer.build(arrays['unit'],arrays['side'],**SPEC)
        x,legal,nuisance,selection=probes(root,256,972091)
        if selection['dataset_id']!=receipt['dataset_id']:raise ValueError('Probe release differs')
        atomic_json(args.output/'selection.json',selection)
        annotation=feather.read_table(graph/'annotations.feather',columns=['type','bodyId'])
        if sha256(graph/'annotations.feather')!=receipt['annotation_sha256']:
            raise ValueError('Annotation identity differs')
        kinds=annotation['type'].to_numpy()[arrays['sensors']].astype(str)
        audits=[]
        for name,renderer in [('reference_current',reference.renderer),('large_current',candidate)]:
            matrix=sampling(renderer);eyes=[]
            for eye in ('L','R'):
                use=renderer.side==eye
                singular=np.linalg.svd(matrix[use],compute_uv=False)
                eyes.append(dict(eye=eye,sensors=int(use.sum()),rank=int(np.sum(singular>1e-6*singular[0])),
                    singular_values=singular.tolist(),condition=float(singular[0]/singular[-1]),
                    covered_points=int(np.count_nonzero(matrix[use].sum(axis=0))),
                    illuminated_sensors=int(np.count_nonzero(matrix[use].sum(axis=1)))))
            adapter=VisualContext(renderer);encoded=adapter.encode(x,'current')
            np.testing.assert_array_equal(encoded[:,len(kinds):],reference.encode(x,'current')[:,len(kinds):])
            changed=x.copy();changed[...,2:8]=0
            np.testing.assert_array_equal(encoded,adapter.encode(changed,'current'))
            contrast=encoded[:,:len(kinds)]-.5;drive=[]
            for eye in ('L','R'):
                for kind in sorted(set(kinds)):
                    use=(renderer.side==eye)&(kinds==kind)
                    if not use.any():continue
                    drive.append(dict(eye=eye,type=kind,sensors=int(use.sum()),
                        illuminated_sensors=int(np.count_nonzero(matrix[use].sum(axis=1))),
                        varying_sensors=int(np.sum(contrast[:,use].std(axis=0,ddof=1)>1e-6)),
                        rms_contrast=float(np.sqrt(np.square(contrast[:,use],dtype=np.float64).mean())),
                        initial_gain=.2,
                        rms_initial_drive_contrast=float(.2*np.sqrt(np.square(contrast[:,use],dtype=np.float64).mean()))))
            audits.append(dict(name=name,eyes=eyes,patches=renderer.audit(),drive_by_eye_type=drive))
        arrays.update(candidate.arrays())
        artifact=args.output/'attachment.npz';np.savez(artifact,**arrays)
        for key in arrays:
            if key not in candidate.arrays():
                with np.load(args.source,allow_pickle=False) as data:
                    np.testing.assert_array_equal(arrays[key],data[key])
        result=dict(status='complete',created=time.time(),source_attachment_sha256=receipt['sha256'],
            graph_id=graph_id,dataset_id=receipt['dataset_id'],spec=SPEC,audits=audits,
            attachment_sha256=sha256(artifact),source_sha256=sha256(Path(__file__)),
            renderer_source_sha256=sha256(Path(__import__('flygo.vision',fromlist=['x']).__file__)),
            scope='Geometry and visual drive on 256 fixed training-family positions. No target labels, neural updates, output remapping, final-test inputs or TPU.')
        if any(row['rank']!=81 or row['covered_points']!=81 for row in audits[-1]['eyes']):
            result['status']='failed';result['reason']='Candidate fails per-eye coverage/rank gate; no qualified attachment receipt is published.'
            atomic_json(args.output/'result.json',result)
            raise ValueError(result['reason'])
        geometry={key:value for key,value in receipt['geometry'].items() if key not in
                  ('patch_spec','sampling_rank','sampling_condition','minimum_singular','patches','context','input_range')}
        geometry.update(layout='One current board per eye in the original bounding rectangle.',patch_spec=SPEC,
                        sampling_rank=81,per_eye_sampling=audits[-1]['eyes'],patches=candidate.audit())
        new_receipt={key:value for key,value in receipt.items() if key not in ('renderer_sha256','visual_bank_sha256')}
        new_receipt.update(created=time.time(),sha256=sha256(artifact),geometry=geometry,
            source_attachment_sha256=receipt['sha256'],script_sha256=sha256(Path(__file__)),
            modes=dict(current='One lag-zero board per eye.',history='Same as current: all samples address lag zero.',
                       neutral='Every photoreceptor at .5; identical nonvisual context.'))
        atomic_json(artifact.with_suffix('.json'),new_receipt)
        restored,restored_ports,_=load_attachment(artifact,graph_id=graph_id,dataset_id=receipt['dataset_id'])
        np.testing.assert_array_equal(restored.encode(x,'current'),VisualContext(candidate).encode(x,'current'))
        for key in ports:np.testing.assert_array_equal(ports[key],restored_ports[key])
        atomic_json(args.output/'result.json',result)
        print(json.dumps(dict(status=result['status'],artifact=str(artifact),audits=[dict(name=r['name'],eyes=[
            {k:v for k,v in eye.items() if k!='singular_values'} for eye in r['eyes']]) for r in audits])),flush=True)


if __name__=='__main__':main()
