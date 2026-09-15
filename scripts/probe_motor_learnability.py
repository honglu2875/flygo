#!/usr/bin/env python3
"""Cache real motor features and compare policy decoders on an identical bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from flygo.attachments import runtime_hashes
from flygo.data.corpus import atomic_json
from flygo.data.loader import load_release
from flygo.play import load_player
from flygo.qualify import sha256
from flygo.runtime import pin
from flygo.storage import StorageBudget, GIB

# The worker freezes this small helper beside the script, while the scientific
# recurrent package remains at its original immutable source.
try:
    import readout_probe as probe
except ModuleNotFoundError:
    from flygo import readout_probe as probe


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/dev/shm/flygo'))
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpus', default=','.join(map(str, range(92, 116))))
    args = p.parse_args(); root = args.root; out = args.output
    cpus = list(map(int, args.cpus.split(','))); pin(cpus)
    plan = json.loads(args.plan.read_text())
    case = next(x for x in plan['checkpoints'] if x['seed'] == args.seed)
    out.resolve().relative_to(root)
    with StorageBudget(root).reserve(files=512<<20, heap=12*GIB, purpose='motor learnability bank and policy heads'):
        out.mkdir(parents=True, exist_ok=False); started = time.time()
        manifest, arrays, indexes = load_release(root, root/'releases'/plan['release']/'manifest.json', cache=True)
        checkpoint = root/'runs'/case['run_id']/'checkpoints'/f"step-{plan['checkpoint_step']:08d}.npz"
        receipt = json.loads(checkpoint.with_suffix('.json').read_text())
        if receipt['replica_status'] != 'verified':
            raise ValueError('Representation checkpoint needs a verified replica')
        graph = Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
        player, metadata = load_player(checkpoint, graph, threads=len(cpus))
        if metadata['dataset_id'] != manifest['dataset_id'] or player.config.seed != args.seed or player.config.steps != 8:
            raise ValueError('Unexpected representation source')
        before = {k:array_hash(v) for k,v in player.parameters().items()}
        selected = np.flatnonzero(player.ports['output_group'] >= 0)
        motors = selected[np.argsort(player.ports['output_group'][selected])]
        np.testing.assert_array_equal(player.ports['output_group'][motors], np.arange(2129))
        rng = np.random.default_rng(plan['selection_seed'])
        chosen = {split:rng.choice(indexes[split], count, replace=False)
                  for split,count in [('train', plan['training_positions']), ('validation', plan['validation_positions'])]}
        symmetries = rng.integers(0, 8, len(chosen['train']))
        families = {split:sorted({manifest['records'][int(i)]['opening_family'] for i in arrays['game_index'][ids]})
                    for split,ids in chosen.items()}
        if set(families['train']) & set(families['validation']):
            raise ValueError('Training and validation opening families overlap')
        bank = {}; selection = {}
        for split, ids in chosen.items():
            x = np.lib.format.open_memmap(out/(split+'-motors.npy'), mode='w+', dtype=np.float32, shape=(len(ids), 2129))
            legal = arrays['legal'][ids].copy(); q = arrays['raw_policy'][ids].copy()
            base = np.empty((len(ids),82), np.float32); values = np.empty(len(ids), np.float32)
            for start in range(0, len(ids), 32):
                stop = min(start+32, len(ids)); features = arrays['features'][ids[start:stop]].copy()
                if split == 'train':
                    for b, symmetry in enumerate(symmetries[start:stop]):
                        flip, turns = int(symmetry)//4, int(symmetry)%4
                        board = features[b]; policy = q[start+b,:81].reshape(9,9); mask = legal[start+b,:81].reshape(9,9)
                        if flip:
                            board=np.flip(board,axis=1); policy=np.flip(policy,axis=1); mask=np.flip(mask,axis=1)
                        features[b] = np.rot90(board, turns, axes=(0,1))
                        q[start+b,:81] = np.rot90(policy,turns).ravel()
                        legal[start+b,:81] = np.rot90(mask,turns).ravel()
                output = player.infer(features, trace=True)
                x[start:stop] = np.maximum(output['states'][-1][motors].T, 0)
                base[start:stop] = output['logits']; values[start:stop] = output['value']
                if stop % 1024 == 0 or stop == len(ids):
                    x.flush()
                    atomic_json(out/'status.json', dict(state='extracting', split=split, positions=stop, total=len(ids), updated=time.time()))
                del output
            if np.any(q[~legal] != 0) or not np.isfinite(x).all():
                raise ValueError('Invalid frozen probe targets or features')
            x.flush(); bank[split] = dict(features=x, legal=legal, target=q, baseline=base)
            np.savez(out/(split+'-labels.npz'), indices=ids, symmetries=symmetries if split=='train' else np.zeros(len(ids),np.int64),
                     legal=legal, policy=q, baseline=base, baseline_value=values, target_value=arrays['raw_value'][ids],
                     game_index=arrays['game_index'][ids], motors=motors)
            selection[split] = dict(positions=len(ids), families=families[split], indices_sha256=array_hash(ids),
                labels_sha256=sha256(out/(split+'-labels.npz')), motors_sha256=sha256(out/(split+'-motors.npy')))
        if before != {k:array_hash(v) for k,v in player.parameters().items()}:
            raise AssertionError('Feature extraction changed the recurrent checkpoint')
        del player, arrays, indexes
        mean, scale, std = probe.normalizer(bank['train']['features'], plan['normalization_std_floor'])
        np.savez(out/'normalization.npz', mean=mean, scale=scale, std=std)
        atomic_json(out/'bank.json', dict(status='complete',created=time.time(),selection=selection,
            normalization_sha256=sha256(out/'normalization.npz'), checkpoint_sha256=receipt['sha256'],
            dataset_id=manifest['dataset_id'], runtime_sha256=runtime_hashes(), plan_sha256=sha256(args.plan),
            source_sha256=sha256(Path(__file__)), helper_sha256=sha256(Path(probe.__file__)),
            varying_motors_above_floor=int(np.count_nonzero(std>plan['normalization_std_floor'])),
            motor_std_quantiles=np.quantile(std,[0,.25,.5,.75,.9,.99,1]).tolist(),
            source_parameters_unchanged=True, seconds=time.time()-started))
        baseline = {split:{k:float(v.mean()) for k,v in probe.position_metrics(data['baseline'],data['legal'],data['target']).items()}
                    for split,data in bank.items()}
        trials = []
        for variant in plan['variants']:
            transformed = {split:probe.transform(data['features'],variant,mean,scale,top_k=plan['gating_top_k'])
                           for split,data in bank.items()}
            for rate in plan['rates']:
                name = variant+'-lr'+format(rate,'.3g'); target = out/name; target.mkdir()
                start = time.time(); fit_rng = np.random.default_rng(args.seed+910000)
                params = probe.initialize(2129,82,hidden=plan['hidden'] if variant=='scaled-mlp' else 0,seed=args.seed+920000)
                first = {k:np.zeros_like(v) for k,v in params.items()}; second = {k:np.zeros_like(v) for k,v in params.items()}
                curve=[]; status='complete'; error=None
                try:
                    for step in range(plan['updates']+1):
                        if step:
                            ids=fit_rng.integers(0,len(transformed['train']),plan['batch_size'])
                            data=bank['train']; _,gradient=probe.loss_and_grad(params,transformed['train'][ids],data['baseline'][ids],data['legal'][ids],data['target'][ids])
                            norm=probe.adam(params,gradient,first,second,step,rate=rate,epsilon=plan['epsilon'],clip=plan['clip'])
                        if step in plan['evaluations']:
                            metrics={}; position={}
                            for split,data in bank.items():
                                logits=probe.forward(params,transformed[split],data['baseline'])
                                if step==0:np.testing.assert_array_equal(logits,data['baseline'])
                                if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite policy logits')
                                position[split]=probe.position_metrics(logits,data['legal'],data['target'])
                                metrics[split]={k:float(v.mean()) for k,v in position[split].items()}
                            curve.append(dict(step=step,exposures=step*plan['batch_size'],seconds=time.time()-start,metrics=metrics))
                            atomic_json(target/'curve.json',curve)
                            if step==plan['updates']:
                                np.savez(target/'position-metrics.npz',**{split+'/'+k:v for split,m in position.items() for k,v in m.items()})
                        if step % 64 == 0:
                            atomic_json(out/'status.json',dict(state='fitting',variant=variant,rate=rate,step=step,completed_trials=len(trials),updated=time.time()))
                except (FloatingPointError,OverflowError) as exc:
                    status='failed';error=str(exc)
                np.savez(target/'checkpoint.npz',**{'param/'+k:v for k,v in params.items()},
                         **{'first/'+k:v for k,v in first.items()},**{'second/'+k:v for k,v in second.items()},step=step,
                         sampler=np.frombuffer(json.dumps(fit_rng.bit_generator.state).encode(),np.uint8))
                learned=int(sum(v.size for v in params.values()))
                trial=dict(variant=variant,rate=rate,status=status,error=error,step=step,head_exposures=step*plan['batch_size'],
                    checkpoint_exposures=plan['checkpoint_exposures'],curve=curve,seconds=time.time()-start,
                    checkpoint_sha256=sha256(target/'checkpoint.npz'),parameters=learned,
                    added_dense_matmul_flops=2*(2129*plan['hidden']+plan['hidden']*82) if variant=='scaled-mlp' else 2*2129*82,
                    feature_nonzero_fraction={split:float(np.count_nonzero(z)/z.size) for split,z in transformed.items()})
                atomic_json(target/'result.json',trial);trials.append(trial)
                atomic_json(out/'result.json',dict(status='complete' if len(trials)==12 else 'running',created=time.time(),seed=args.seed,
                    baseline=baseline,trials=trials,bank_sha256=sha256(out/'bank.json'),plan_sha256=sha256(args.plan),
                    source_sha256=sha256(Path(__file__)),helper_sha256=sha256(Path(probe.__file__)),seconds=time.time()-started,
                    scope=plan['scope']))
                print(json.dumps(dict(seed=args.seed,variant=variant,rate=rate,status=status,last=curve[-1])),flush=True)
        atomic_json(out/'status.json',dict(state='complete',updated=time.time()))


if __name__ == '__main__': main()
