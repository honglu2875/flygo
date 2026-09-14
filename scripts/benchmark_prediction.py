#!/usr/bin/env python3
"""Bounded fresh-process CPU prediction latency and peak RSS; no learning."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import resource
import time

from flygo import _native
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release,Sampler
from flygo.fly import FlyConfig,RustFly,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget,GIB


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--cpus',default='117,118,119')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--passes',type=int,help='Profiling-only depth override; never a trained-strength result')
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--reference',type=Path,help='Require identical inputs and prediction bytes')
    p.add_argument('--prune',action='store_true',help='Optional finite-horizon readout dependency execution')
    args=p.parse_args();cpus=list(map(int,args.cpus.split(',')));pin(cpus)
    if (not 1<=args.batch_size<=128 or not 1<=args.repetitions<=10
            or (args.passes is not None and not 1<=args.passes<=64)):
        p.error('Choose B=1..128, 1..10 repetitions and at most 64 profiling passes')
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Benchmark reports are immutable')
    with StorageBudget(args.root).reserve(files=1<<20,heap=12*GIB,purpose='bounded prediction benchmark'):
        graph=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        manifest,arrays,indexes=load_release(args.root,args.root/'releases/v0-1m/manifest.json',cache=True)
        model=(load_player(args.checkpoint,graph,threads=len(cpus))[0] if args.checkpoint else
               RustFly(load_graph(graph),FlyConfig(steps=4,threads=len(cpus))))
        family='cnn' if hasattr(model.config,'blocks') else 'fly'
        if args.prune and family!='fly':raise ValueError('Readout pruning applies only to fly models')
        if args.passes is not None:
            if family!='fly':raise ValueError('A recurrent-pass override applies only to fly models')
            model.config=replace(model.config,steps=args.passes)
        batch=Sampler(arrays,indexes,714091).batch(args.batch_size)[0]
        hashes=None;samples=[]
        loaded_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        for repetition in range(args.repetitions+1):
            start=time.perf_counter()
            prediction=model.infer(batch,prune=True) if args.prune else model.infer(batch)
            elapsed=time.perf_counter()-start
            current={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in prediction.items()}
            if hashes is not None and current!=hashes:raise AssertionError('Repeated prediction changed')
            hashes=current
            if repetition:samples.append(elapsed)
            else:cold=elapsed
        result=dict(status='complete',dataset_id=manifest['dataset_id'],
            graph_id=model.graph['manifest']['graph_id'],source_native_sha256=sha256(Path(_native.__file__)),
            checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
            batch_size=args.batch_size,passes=getattr(model.config,'steps',None),cpus=cpus,
            family=family,model_config=asdict(model.config),
            pruned=args.prune,
            input_sha256=hashlib.sha256(batch.tobytes()).hexdigest(),prediction_sha256=hashes,
            cold_seconds=cold,warm_seconds=samples,loaded_peak_rss_bytes=loaded_peak,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scope='Fresh process high-water RSS includes model/data initialization. Warm prediction excludes loading and first cache preparation. Any depth override is an implementation probe, not a trained architecture or playing-strength result.')
        if args.prune:
            result['dependency_counts']=getattr(model,'core',model).prediction_dependencies()
        if args.reference:
            other=json.loads(args.reference.read_text())
            for key in ('dataset_id','graph_id','checkpoint_sha256','batch_size','passes','cpus','input_sha256','prediction_sha256'):
                if result[key]!=other[key]:raise AssertionError('Prediction reference differs: '+key)
            result['reference_sha256']=sha256(args.reference)
        atomic_json(args.output,result)
        print(json.dumps(result))


if __name__=='__main__':main()
