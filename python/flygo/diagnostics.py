"""Occasional optimizer/activity measurements; never alter parameters or sampler state."""
from __future__ import annotations
import time
import numpy as np
from .fly import firing_rate


def norm(array):
    return float(np.sqrt(np.square(np.asarray(array,dtype=np.float64)).sum()))


def measure(model,batch,before,training_metrics,*,clip,training_batch_size=None):
    started=time.perf_counter()
    after=model.parameters()
    _,gradient=model.loss_and_grad(*batch)
    groups={}
    for name,value in after.items():
        delta=value-before[name]
        previous_norm=norm(before[name])
        groups[name]=dict(parameters=len(value),parameter_norm=norm(value),
                          update_norm=norm(delta),update_max=float(np.max(np.abs(delta))),
                          relative_update=norm(delta)/max(previous_norm,1e-30),
                          post_update_gradient_norm=norm(gradient[name]),
                          nonzero_gradients=int(np.count_nonzero(gradient[name])))
    old_strength=np.logaddexp(np.float32(0),before['edge'])
    strength=np.logaddexp(np.float32(0),after['edge'])
    ratio=strength/old_strength
    output=model.infer(batch[0],trace=True)
    states=[]
    for state in output['states']:
        rate=firing_rate(state,model.config.rate_softness)
        states.append(dict(active_fraction=float(np.mean(rate!=0)),
                           positive_voltage_fraction=float(np.mean(state>0)),
                           rate_mean=float(rate.mean(dtype=np.float64)),rate_max=float(rate.max()),
                           rms=norm(state)/np.sqrt(state.size),
                           min=float(state.min()),max=float(state.max())))
    return dict(kind='diagnostics',step=training_metrics['step'],
                rate_softness=model.config.rate_softness,
                training_gradient_norm=training_metrics['gradient_norm'],
                clipped=bool(training_metrics['gradient_norm']>clip),clip=clip,groups=groups,
                edge_strength=dict(mean=float(strength.mean(dtype=np.float64)),max=float(strength.max()),
                                   step_ratio_quantiles=np.quantile(ratio,[0,.01,.5,.99,1]).tolist()),
                states=states,value_saturated_fraction=float(np.mean(np.abs(output['value'])>.98)),
                seconds=time.perf_counter()-started,diagnostic_forward_positions=2*len(batch[0]),
                diagnostic_backward_positions=len(batch[0]),
                diagnostic_batch_positions=len(batch[0]),training_batch_positions=training_batch_size or len(batch[0]),
                gradient_scope='Post-update gradient on a deterministic prefix of the training batch; no extra optimizer step')
