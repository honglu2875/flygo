"""Real-corpus SPMD ownership, Rust update parity and fresh-process recovery gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import jax
from jax.experimental import multihost_utils as mh
import numpy as np

from .learner import JaxFly
from ..checkpoint import load_checkpoint,save_checkpoint
from ..data.corpus import atomic_json
from ..data.loader import load_release,Sampler
from ..fly import FlyConfig,RustFly,load_graph
from ..runtime import pin


def digest(arrays):
    h=hashlib.sha256()
    for array in arrays:
        a=np.ascontiguousarray(array)
        h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def agree(value):
    raw=np.frombuffer(bytes.fromhex(value),np.uint8)
    copies=np.asarray(mh.process_allgather(raw,tiled=False))
    if not np.all(copies==copies[0]):
        raise AssertionError('Controllers disagree on global data or replicated state')


def qualify(config,report,out,mesh):
    root=Path(config['root']);chief=jax.process_index()==0
    graph_path=Path(json.loads((root/'runs/m4/graph.json').read_text())['path'])
    manifest,arrays,indexes=load_release(root,root/'releases'/config['release']/'manifest.json',cache=True)
    graph=load_graph(graph_path)
    cfg=FlyConfig(steps=config['steps'],groups=config['groups'],threads=len(config['cpus']),seed=config['seed'])
    ports=None
    if config.get('ports'):
        from ..ports import load_ports
        ports,_=load_ports(Path(config['ports']),graph_id=graph['manifest']['graph_id'],
                          features=cfg.features,groups=cfg.groups,seed=cfg.seed)
    model=JaxFly(graph,cfg,mesh=mesh,ports=ports)
    rust=RustFly(graph,cfg,ports=ports) if chief else None
    scales=config.get('rate_scales') or None
    sampler=Sampler(arrays,indexes,cfg.seed)
    batch_size=32
    errors={};failures=[];delta_diagnostics=[];batches=[]
    report.update(status='real_data_qualification',dataset_id=manifest['dataset_id'],global_batch=batch_size)
    atomic_json(out/'status.json',report)

    def check(got,want,name,rtol=3e-3,atol=5e-6):
        if not chief:return
        got,want=np.asarray(got),np.asarray(want)
        errors[name]=float(np.max(np.abs(got-want)))
        try:np.testing.assert_allclose(got,want,rtol=rtol,atol=atol,err_msg=name)
        except AssertionError as error:failures.append(str(error))

    def gate(stage):
        atomic_json(out/'parity.json',dict(status='failed' if failures else 'passed',stage=stage,
                    process_index=jax.process_index(),errors=errors,failures=failures,batches=batches,
                    delta_diagnostics=delta_diagnostics))
        if not bool(mh.broadcast_one_to_all(np.asarray(not failures),is_source=chief)):
            raise AssertionError(stage+' failed; see process-0 parity.json')
        pin(config['cpus'])

    def batch():
        result=sampler.batch(batch_size)
        fingerprint=digest(result);agree(fingerprint);batches.append(fingerprint)
        return result

    def checkpoint(step):
        path=out/f'step-{step:08d}.npz'
        receipt=save_checkpoint(model,sampler,path,dict(dataset_id=manifest['dataset_id']),root=root)
        agree(receipt['sha256'])
        receipt.update(replica_status='verified',peer='all four SPMD controllers; independently saved and SHA-256 compared')
        atomic_json(path.with_suffix('.json'),receipt)
        return path

    if config['mode']=='restore':
        source=Path(config['restore_from'])
        load_checkpoint(source,model,sampler,dataset_id=manifest['dataset_id'])
        if chief:load_checkpoint(source,rust,dataset_id=manifest['dataset_id'])
        reference=json.loads((source.parent/'continuation.json').read_text())
        current=batch()
        if batches[-1]!=reference['batch_sha256']:
            raise AssertionError('Restored sampler chose a different batch')
        metrics=model.train_step(*current,rate_scales=scales)
        state=model.checkpoint_arrays()
        fingerprint={k:digest([v]) for k,v in state.items()}
        if fingerprint!=reference['state_hashes']:
            raise AssertionError('Fresh-process TPU continuation was not bitwise reproducible')
        if chief:
            rust.train_step(*current,rate_scales=scales)
            for name,want in rust.checkpoint_arrays().items():check(state[name],want,'portable/'+name)
        gate('fresh-process and portable continuation')
        report['recovery']=dict(status='passed',source=str(source),next_update=metrics,
                                batch_sha256=batches[-1],bitwise=True)
    else:
        begin=time.perf_counter()
        for step in range(1,4):
            current=batch()
            if step==1:
                result=model.infer(current[0],trace=True)
                if chief:
                    expected=rust.infer(current[0],trace=True)
                    for name,want in expected.items():check(result[name],want,'forward/'+name,3e-4,3e-6)
                gate('full-state forward')
            losses,gradient=model.loss_and_grad(*current)
            if chief:
                expected,rgrad=rust.loss_and_grad(*current)
                for name,want in expected.items():check(losses[name],want,'loss/'+str(step)+'/'+name,3e-4,3e-6)
                for name,want in rgrad.items():check(gradient[name],want,'gradient/'+str(step)+'/'+name)
            gate('gradient '+str(step))
            before=model.parameters() if chief else None
            old=rust.parameters() if chief else None
            metrics=model.train_step(*current,rate_scales=scales)
            state=model.checkpoint_arrays()
            agree(digest([state[k] for k in sorted(state)]))
            if chief:
                rust.train_step(*current,rate_scales=scales)
                for name,want in rust.checkpoint_arrays().items():check(state[name],want,'update/'+str(step)+'/'+name)
                for name,want in rust.parameters().items():
                    delta=state['param/'+name]-before[name];expected=want-old[name]
                    difference=np.abs(delta-expected)
                    threshold=5e-6+3e-3*np.abs(expected)
                    indices=np.flatnonzero(difference>threshold)
                    delta_diagnostics.append(dict(step=step,group=name,max_absolute=float(difference.max()),
                        relative_delta_outliers=len(indices),outliers=[dict(index=int(i),
                            actual=float(delta[i]),reference=float(expected[i]),
                            gradient=float(gradient[name][i]),reference_gradient=float(rgrad[name][i]))
                            for i in indices[:20]]))
                    # Adam amplifies near-cancelled gradients around epsilon.
                    # Audit that sensitivity, while independently checking the
                    # actual optimizer arithmetic from its common moments.
                    first=state['first/'+name]/np.float32(1-.9**step)
                    second=state['second/'+name]/np.float32(1-.999**step)
                    group_rate=np.float32(.003)*np.float32((scales or {}).get(name,1.0))
                    formula=before[name]-group_rate*first/(np.sqrt(second)+np.float32(1e-8))
                    check(state['param/'+name],formula,'adam_formula/'+str(step)+'/'+name,3e-5,3e-6)
            gate('update '+str(step))
            if step==2:checkpoint(step)
        checkpoint(3)
        atomic_json(out/'continuation.json',dict(batch_sha256=batches[-1],
                    state_hashes={k:digest([v]) for k,v in state.items()},metrics=metrics))
        report['real_data']=dict(status='passed',updates=3,batch_sha256=batches,
                                 seconds=time.perf_counter()-begin,checkpoint_copies=4)
    atomic_json(out/'result.json',report)
