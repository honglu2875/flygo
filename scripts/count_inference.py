#!/usr/bin/env python3
"""Count CPU recurrence arithmetic on a declared sample of held-out Go inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from flygo import _native
from flygo.cost import fly_cost
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def count_edges(states,out_degree):
    """Match the finite-state CPU's batch-wide zero-row threshold exactly."""
    records=[];neurons=len(out_degree);edges=int(out_degree.sum())
    for state in states[:-1]:
        if state.shape[0]!=neurons or not np.isfinite(state).all():
            raise ValueError('Expected finite node-major recurrent states')
        active=np.any(state>0,axis=1)
        skip=int(active.sum())*5<neurons*4
        traversed=int(out_degree[active].sum()) if skip else edges
        records.append(dict(active_rows=int(active.sum()),skip_zero_rows=skip,
                            edge_multiply_adds_per_position=traversed))
    return records


def distribution(values):
    return dict(mean=float(np.mean(values)),minimum=float(np.min(values)),
                p50=float(np.quantile(values,.5)),p90=float(np.quantile(values,.9)),
                maximum=float(np.max(values)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpus',default='117,118,119')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    plan=json.loads(args.plan.read_text());count=plan['positions'];batches=plan['batches']
    if not 1<=count<=4096 or not batches or any(b<1 or b>128 or count%b for b in batches):
        raise ValueError('Bounded complete batches required')
    args.output.resolve().relative_to(args.root.resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    with StorageBudget(args.root).reserve(files=64*(1<<20),heap=12*GIB,purpose='counted CPU inference work'):
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path)
        manifest,arrays,indexes=load_release(args.root,args.root/'releases'/plan['release']/'manifest.json',cache=True)
        indices=np.random.default_rng(plan['sample_seed']).choice(indexes['validation'],count,replace=False)
        np.save(args.output/'indices.npy',indices)
        degree=np.bincount(graph['src'],minlength=len(graph['type_id']))
        results=[]
        for case in plan['cases']:
            checkpoint=args.root/case['checkpoint'] if case.get('checkpoint') else None
            if checkpoint:
                model,metadata=load_player(checkpoint,graph_path,threads=len(cpus))
                if metadata['dataset_id']!=manifest['dataset_id']:raise ValueError('Checkpoint release mismatch')
            else:model=RustFly(graph,FlyConfig(steps=4,threads=len(cpus)))
            if model.config.rate_softness:
                raise ValueError('This byte-qualified zero-row ledger currently supports the hard-rate model only')
            before=model.checkpoint_arrays();records=[]
            for batch in batches:
                nominal=fly_cost(len(degree),len(graph['src']),len(graph['sensory']),
                    passes=model.config.steps,groups=model.config.groups,batch_size=batch,
                    readout_neurons=int(np.count_nonzero(model.ports['output_group']>=0)),
                    readout_mean_scale=model.config.readout_mean_scale)
                other=nominal['arithmetic_flops']-nominal['parts']['sparse_aggregation']
                # Cold preparation is explicitly outside the measured warm call.
                started=time.perf_counter();model.infer(arrays['features'][indices[:batch]])
                prime_seconds=time.perf_counter()-started
                details=[]
                for begin in range(0,count,batch):
                    started=time.perf_counter()
                    prediction=model.infer(arrays['features'][indices[begin:begin+batch]],trace=True)
                    seconds=time.perf_counter()-started
                    passes=count_edges(prediction['states'],degree)
                    uncached=other+2*sum(row['edge_multiply_adds_per_position'] for row in passes)
                    warm_cached=uncached-2*passes[0]['edge_multiply_adds_per_position']
                    details.append(dict(offset=begin,passes=passes,zero_skipped_flops_per_position=uncached,
                        cached_zero_skipped_flops_per_position=warm_cached,seconds=seconds))
                records.append(dict(batch_size=batch,nominal=nominal,first_call_seconds=prime_seconds,
                    zero_skipped_flops_per_position=distribution([d['zero_skipped_flops_per_position'] for d in details]),
                    cached_zero_skipped_flops_per_position=distribution([d['cached_zero_skipped_flops_per_position'] for d in details]),
                    seconds_per_position=distribution([d['seconds']/batch for d in details]),batches=details))
            for key,value in model.checkpoint_arrays().items():
                if value.tobytes()!=before[key].tobytes():raise AssertionError('Counting mutated '+key)
            results.append(dict(case=case['name'],checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=sha256(checkpoint) if checkpoint else None,records=records))
            atomic_json(args.output/'result.json',dict(status='complete' if len(results)==len(plan['cases']) else 'running',
                dataset_id=manifest['dataset_id'],graph_id=graph['manifest']['graph_id'],plan=plan,
                plan_sha256=sha256(args.plan),indices_sha256=sha256(args.output/'indices.npy'),
                source_sha256=sha256(Path(__file__)),native_sha256=sha256(Path(_native.__file__)),cpus=cpus,
                results=results,constant_message_cache_bytes=4*len(degree),
                constant_message_preparation_flops=2*len(graph['src']),
                scope='Counts algorithmic FP multiply/add work for the finite-state CPU row-skipping rule. Adapter/dynamics/head work uses cost.py convention. Not a hardware instruction counter; excludes masks, integer work, memory traffic, nonlinear functions, weight transforms, Go features and search. Timing includes trace copies. Warm cache preparation is separate; original nominal matched-CNN study is unchanged.',
                final_test='not evaluated'))
            print(json.dumps(dict(case=case['name'],status='complete')),flush=True)
            del model,before


if __name__=='__main__':main()
