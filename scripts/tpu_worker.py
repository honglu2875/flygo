#!/usr/bin/env python3
"""Bounded TPU qualification worker. Every host executes the same collective order."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    from flygo.runtime import pin
    from flygo.storage import GIB, StorageBudget
    from flygo.data.corpus import atomic_json
    pin(config['cpus'])
    root = Path(config['root'])
    out = args.config.parent
    report = dict(status='initializing', pid=os.getpid(), config=config, started=time.time())
    atomic_json(out / 'status.json', report)
    distributed = False
    try:
        with StorageBudget(root).reserve(files=2*GIB, heap=32*GIB, purpose='TPU SPMD qualification'):
            import jax
            import jax.numpy as jnp
            import numpy as np
            from jax.sharding import NamedSharding, PartitionSpec as P
            from jax.experimental import multihost_utils as mh
            jax.config.update('jax_default_matmul_precision', 'highest')
            jax.distributed.initialize(initialization_timeout=90, heartbeat_timeout_seconds=90)
            distributed = True
            devices = jax.devices()
            assert jax.process_count() == config['expected_processes']
            assert len(devices) == config['expected_devices']
            assert all(d.platform == 'tpu' for d in devices)
            pin(config['cpus'])
            mesh = jax.make_mesh((len(devices),), ('data',))
            sharding = NamedSharding(mesh, P('data'))
            local = np.full((jax.local_device_count(), 128), jax.process_index()+1, np.float32)
            global_array = jax.make_array_from_process_local_data(sharding, local)
            mean = jax.jit(lambda x: jnp.mean(x))(global_array)
            assert float(mean) == 2.5
            report.update(status='probe_passed', process_index=jax.process_index(),
                          process_count=jax.process_count(), device_count=len(devices),
                          local_devices=[dict(id=d.id, kind=d.device_kind,
                                              memory=d.memory_stats()) for d in jax.local_devices()],
                          collective_mean=float(mean), jax=jax.__version__,
                          affinity=[sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()])
            atomic_json(out / 'status.json', report)
            print(json.dumps({k:v for k,v in report.items() if k not in ('config','affinity')}), flush=True)
            if config['mode'] in ('benchmark','audit'):
                from flygo.jax.benchmark import benchmark
                benchmark(config, report, out, mesh)
            elif config['mode'] in ('qualify','restore'):
                from flygo.jax.qualify import qualify
                qualify(config,report,out,mesh)
            elif config['mode']=='train':
                from flygo.train import main as train
                plan=config['training']
                options=['--backend','tpu','--root',str(root),'--run-id',config['run_id']+'-learner',
                         '--cpus',','.join(map(str,config['cpus'])),'--threads',str(len(config['cpus'])),'--peer','']
                for key in ('release','passes','groups','batch_size','rate','clip','seed','eval_every',
                            'eval_batch_size','eval_positions','checkpoint_every'):
                    options+=['--'+key.replace('_','-'),str(plan[key])]
                options+=['--steps',str(plan['updates'])]
                for key in ('model','channels','blocks'):
                    if key in plan:options+=['--'+key,str(plan[key])]
                if config.get('ports'):options+=['--ports',config['ports']]
                if plan.get('rate_scales'):options+=['--rate-scales',json.dumps(plan['rate_scales'])]
                if config.get('restore_from'):options+=['--resume',config['restore_from']]
                report.update(status='training',learner_run=config['run_id']+'-learner')
                atomic_json(out/'status.json',report)
                train(options)
                report['training']=json.loads((root/'runs'/report['learner_run']/'status.json').read_text())
            elif config['mode']=='control':
                from flygo.jax.control import qualify
                qualify(config,report,out,mesh)
            mh.sync_global_devices('qualification-complete')
            report.update(status='passed', elapsed_seconds=time.time()-report['started'])
    except BaseException as error:
        report.update(status='failed', error=repr(error), elapsed_seconds=time.time()-report['started'])
        raise
    finally:
        atomic_json(out / 'status.json', report)
        if report['status']=='passed':atomic_json(out/'result.json',report)
        if distributed:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
