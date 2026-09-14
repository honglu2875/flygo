"""Full-size CNN CPU reference and independent TPU-lowering qualification."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import jax
import numpy as np

from .cnn import CNNConfig,JaxCNN
from ..checkpoint import save_checkpoint
from ..cost import fly_cost,cnn_cost
from ..data.corpus import atomic_json
from ..data.loader import load_release,Sampler
from ..qualify import sha256


def prepare(root,out,*,threads=3):
    out.mkdir(parents=True,exist_ok=False)
    cfg=CNNConfig(threads=threads)
    manifest,arrays,indexes=load_release(root,root/'releases/v0-1m/manifest.json',cache=True)
    sampler=Sampler(arrays,indexes,1);model=JaxCNN(cfg)
    timings=[]
    for step in range(1,4):
        batch=sampler.batch(32)
        np.savez(out/f'batch-{step}.npz',**dict(zip(('features','legal','policy','value'),batch)))
        if step==1:np.savez(out/'forward.npz',**model.infer(batch[0],trace=True))
        parts,gradient=model.loss_and_grad(*batch)
        np.savez(out/f'gradient-{step}.npz',**gradient,losses=np.asarray([parts['policy_loss'],parts['value_loss']]))
        started=time.perf_counter();metrics=model.train_step(*batch)
        timings.append(dict(**metrics,seconds=time.perf_counter()-started))
        np.savez(out/f'state-{step}.npz',**model.checkpoint_arrays())
    report=dict(status='passed',model_config=asdict(cfg),dataset_id=manifest['dataset_id'],batch_size=32,
        reference='JAX CPU FP32, highest precision, complete model and all parameter groups',updates=timings,
        jax=jax.__version__,devices=jax.device_count(),processes=jax.process_count(),
        source_module=__file__,per_device_batch=32//jax.device_count(),
        arithmetic=dict(fly=fly_cost(165122,15270273,15912),cnn=cnn_cost()),
        files={p.name:sha256(p) for p in sorted(out.glob('*.npz'))})
    atomic_json(out/'reference.json',report)
    return report


def qualify(config,report,out,mesh):
    source=Path(config['control_reference'])
    reference=json.loads((source/'reference.json').read_text())
    if reference['status']!='passed':raise ValueError('CNN CPU reference is incomplete')
    for name,want in reference['files'].items():
        if sha256(source/name)!=want:raise ValueError('CNN CPU reference checksum failed')
    cfg=CNNConfig(**{**reference['model_config'],'threads':len(config['cpus'])})
    model=JaxCNN(cfg,mesh=mesh)
    errors={};failures=[]
    def check(actual,expected,name,*,rtol=3e-3,atol=5e-6):
        actual,expected=np.asarray(actual),np.asarray(expected)
        errors[name]=float(np.max(np.abs(actual-expected)))
        try:np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,err_msg=name)
        except AssertionError as error:failures.append(str(error))
    def gate(stage):
        atomic_json(out/'control-parity.json',dict(stage=stage,errors=errors,failures=failures))
        if model.collective_any(bool(failures)):raise AssertionError('CNN parity failed: '+stage)
    for step in range(1,4):
        if step>1:
            # Compare derivatives of the same function at the same parameters.
            # Tiny Adam differences near cancellation can cross ReLU boundaries
            # in free-running trajectories; those failed attempts are retained.
            with np.load(source/f'state-{step-1}.npz') as previous:model.restore_arrays(previous)
        with np.load(source/f'batch-{step}.npz') as data:
            batch=tuple(data[k] for k in ('features','legal','policy','value'))
        if step==1:
            actual=model.infer(batch[0],trace=True)
            with np.load(source/'forward.npz') as expected:
                for name,want in expected.items():check(actual[name],want,'forward/'+name,rtol=3e-4,atol=3e-6)
            gate('forward')
        losses,gradient=model.loss_and_grad(*batch)
        with np.load(source/f'gradient-{step}.npz') as expected:
            check([losses['policy_loss'],losses['value_loss']],expected['losses'],'loss/'+str(step),rtol=3e-4,atol=3e-6)
            for name,actual in gradient.items():check(actual,expected[name],f'gradient/{step}/'+name)
        gate('gradients '+str(step))
        model.train_step(*batch)
        with np.load(source/f'state-{step}.npz') as expected:
            for name,actual in model.checkpoint_arrays().items():check(actual,expected[name],f'update/{step}/'+name)
        gate('update '+str(step))
    receipt=save_checkpoint(model,Sampler({},{}),out/'checkpoint.npz',dict(dataset_id=reference['dataset_id'],
        checkpoint_role='Numerical fixture and replica check; frozen batches, not a training continuation'),root=Path(config['root']))
    receipt=model.verify_checkpoint_copies(receipt)
    atomic_json(out/'checkpoint.json',receipt)
    report['control_parity']=dict(status='passed',cpu_reference_sha256=sha256(source/'reference.json'),
        model_config=asdict(cfg),batch_size=32,updates=3,max_absolute_error=max(errors.values()),
        checkpoint_sha256=receipt['sha256'],replica_status=receipt['replica_status'],
        transitions='Three checkpoint-aligned CPU/TPU transitions; original pointwise tolerances',
        trajectory='Free-running CPU/TPU trajectories are not claimed to remain identical')
