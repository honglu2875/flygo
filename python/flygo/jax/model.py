"""Independent differentiable equations; small/full CPU parity before TPU specialization."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .numerics import softplus, sigmoid, log_softmax, firing_rate, HIGHEST


def forward(params, graph, ports, features, *, steps, groups, actions, rate_softness=0.0):
    # Reference uses edge messages. Production Rust avoids this E*B allocation.
    # Keep full-graph JAX parity batches small until a TPU kernel is qualified.
    n = graph['type_id'].shape[0]
    x = jnp.asarray(features,dtype=jnp.float32).reshape(features.shape[0],-1).T
    source, destination = graph['src'],graph['dst']
    weight = graph['sign'][source] * softplus(params['edge'])
    alpha = 0.01 + 0.98 * sigmoid(params['leak'][graph['type_id']])
    bias = params['bias'][graph['type_id']]
    input_index = ports['input_index']
    safe_input = jnp.maximum(input_index,0)
    drive = jnp.where((input_index >= 0)[:,None],params['input_gain'][safe_input,None]*x[safe_input],0)
    state = jnp.full((n,x.shape[1]),0.01,dtype=jnp.float32)
    def step(state,_):
        if 'layout' in graph:
            from .sparse import multiply
            total=multiply(weight,firing_rate(state,rate_softness),graph['layout'])
        else:
            messages = weight[:,None]*firing_rate(state[source],rate_softness)
            total = jax.ops.segment_sum(messages,destination,num_segments=n,indices_are_sorted=True)
        next_state = (1-alpha[:,None])*state+alpha[:,None]*(total+bias[:,None]+drive)
        return next_state,next_state
    state,states = jax.lax.scan(step,state,None,length=steps)
    readout = ports['output_group']
    contributions = jnp.where((readout >= 0)[:,None],
        (ports['output_scale']*params['readout_gain'])[:,None]*firing_rate(state,rate_softness),0)
    pooled = jax.ops.segment_sum(contributions,jnp.maximum(readout,0),num_segments=groups)
    logits = jnp.matmul(pooled.T,params['policy_weight'].reshape(actions,groups).T,
                        precision=jax.lax.Precision.HIGHEST) + params['policy_bias']
    value = jax.lax.tanh(jnp.matmul(pooled.T,params['value_weight'],precision=jax.lax.Precision.HIGHEST)
                         + params['value_bias'][0],accuracy=HIGHEST)
    initial = jnp.full((1,n,x.shape[1]),0.01,dtype=jnp.float32)
    return dict(logits=logits,value=value,states=jnp.concatenate([initial,states],axis=0))


def loss(params,graph,ports,features,legal,policy,value,*,steps,groups,actions,rate_softness=0.0):
    result=forward(params,graph,ports,features,steps=steps,groups=groups,actions=actions,rate_softness=rate_softness)
    return teacher_loss(result,legal,policy,value)


def teacher_loss(result,legal,policy,value):
    """Shared teacher targets and reductions for fly models and neural controls."""
    logits=jnp.where(legal,result['logits'],-1e30)
    policy_loss=-(policy*log_softmax(logits)).sum(axis=-1).mean()
    value_loss=jnp.square(result['value']-value).mean()
    return policy_loss+value_loss,(policy_loss,value_loss)


def adam(params,grad,first,second,step,*,rate=0.003,clip=1.0,
         norm_dtype=jnp.float64,corrections=None):
    # Enable x64 in the parity runner to match the Rust global norm reduction.
    norm=jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(norm_dtype))) for g in grad.values()))
    scale=jnp.minimum(1.0,clip/jnp.maximum(norm,1e-30)).astype(jnp.float32)
    if corrections is None:
        c1=jnp.asarray(1.0-0.9**step,dtype=jnp.float32)
        c2=jnp.asarray(1.0-0.999**step,dtype=jnp.float32)
    else:
        c1,c2=corrections
    first={k:0.9*first[k]+0.1*(grad[k]*scale) for k in params}
    second={k:0.999*second[k]+0.001*jnp.square(grad[k]*scale) for k in params}
    rates=rate if isinstance(rate,dict) else {k:rate for k in params}
    params={k:params[k]-rates[k]*(first[k]/c1)/(jnp.sqrt(second[k]/c2)+1e-8) for k in params}
    return params,first,second,norm
