#!/usr/bin/env python3
"""Measure common readout offsets from retained full-state CPU fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np

from flygo.data.corpus import atomic_json
from flygo.fly import FlyConfig,RustFly,firing_rate,load_graph
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import GIB,StorageBudget


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/dev/shm/flygo'))
    p.add_argument('--profiles',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();pin([117,118,119])
    args.output.resolve().relative_to(args.root.resolve())
    if args.output.exists():raise ValueError('Audit reports are immutable')
    with StorageBudget(args.root).reserve(files=1<<20,heap=8*GIB,purpose='read-only pooled readout audit'):
        graph_path=Path(json.loads((args.root/'runs/m4/graph.json').read_text())['path'])
        graph=load_graph(graph_path);records=[]
        for directory in args.profiles:
            report=json.loads((directory/'result.json').read_text())
            if report['status']!='complete':raise ValueError('Use completed profiles')
            checkpoint=Path(report['checkpoint']) if report['checkpoint'] else None
            model=(load_player(checkpoint,graph_path,threads=3)[0] if checkpoint else
                   RustFly(graph,FlyConfig(steps=4,threads=3)))
            params=model.parameters();ports=model.ports
            with np.load(directory/'batch-32.npz',allow_pickle=False) as saved:
                voltage=saved['states'][-1];logits=saved['logits'];values=saved['value']
            selected=ports['output_group']>=0;groups=model.config.groups
            rate=firing_rate(voltage,model.config.rate_softness)
            pool=np.zeros((groups,voltage.shape[1]),np.float32)
            scale=ports['output_scale'][selected]*params['readout_gain'][selected]
            np.add.at(pool,ports['output_group'][selected],scale[:,None]*rate[selected])
            reconstructed=pool.T@params['policy_weight'].reshape(model.config.actions,groups).T+params['policy_bias']
            linear=pool.T@params['value_weight']+params['value_bias'][0]
            np.testing.assert_allclose(reconstructed,logits,rtol=3e-4,atol=3e-6)
            np.testing.assert_allclose(np.tanh(linear),values,rtol=3e-4,atol=3e-6)
            z=pool.astype(np.float64);mean=z.mean(axis=0)
            centered=z-mean[None,:];energy=float(np.square(z).sum())
            damped=centered+mean[None,:]/np.sqrt(groups)
            records.append(dict(profile=str(directory),profile_sha256=sha256(directory/'result.json'),
                fixture_sha256=sha256(directory/'batch-32.npz'),checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                rate_softness=model.config.rate_softness,groups=groups,batch=voltage.shape[1],
                common_component_energy_fraction=float(groups*np.square(mean).sum()/max(energy,1e-30)),
                pooled_mean=float(z.mean()),pooled_rms=float(np.sqrt(np.square(z).mean())),
                centered_rms=float(np.sqrt(np.square(centered).mean())),
                input_variation_rms=float(np.sqrt(np.var(z,axis=1).mean())),
                mean_damped_rms=float(np.sqrt(np.square(damped).mean())),
                value_linear_range=[float(linear.min()),float(linear.max())],
                value_saturated_fraction=float(np.mean(np.abs(values)>.98))))
            del model,params,voltage,rate,pool
        atomic_json(args.output,dict(status='complete',records=records,
            scope='Read-only reconstruction of retained B32 training-input fixtures, checked against native policy/value outputs. No new labels, optimizer updates or validation selection. Mean damping is a hypothetical fixed external readout transform; these measurements do not establish a training benefit.',
            script_sha256=sha256(Path(__file__))))
        print(json.dumps(records,indent=2))


if __name__=='__main__':main()
