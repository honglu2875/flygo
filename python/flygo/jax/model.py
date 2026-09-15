"""Independent differentiable equations; small/full CPU parity before TPU specialization."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .numerics import softplus, sigmoid, log_softmax, firing_rate, condition_readout, HIGHEST


def value_score(params, pooled, head_mask):
    weight = params['value_weight']
    if head_mask is not None:
        weight = jnp.where(head_mask['value_weight'], weight, 0)
    return jnp.matmul(pooled.T, weight, precision=jax.lax.Precision.HIGHEST) + params['value_bias'][0]


def forward(params, graph, ports, features, *, steps, groups, actions, rate_softness=0.0,readout_mean_scale=1.0,return_embedding=False,head_mask=None):
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
    if readout_mean_scale!=1:
        pooled=condition_readout(pooled,readout_mean_scale)
    policy_weight = params['policy_weight']
    if head_mask is not None:
        policy_weight = jnp.where(head_mask['policy_weight'], policy_weight, 0)
    logits = jnp.matmul(pooled.T,policy_weight.reshape(actions,groups).T,
                        precision=jax.lax.Precision.HIGHEST) + params['policy_bias']
    score = value_score(params, pooled, head_mask)
    value = jax.lax.tanh(score,accuracy=HIGHEST)
    initial = jnp.full((1,n,x.shape[1]),0.01,dtype=jnp.float32)
    result = dict(logits=logits,value=value,states=jnp.concatenate([initial,states],axis=0))
    if return_embedding: result.update(embedding=pooled.T,score=score)
    return result


def loss(params,graph,ports,features,legal,policy,value,*,steps,groups,actions,rate_softness=0.0,readout_mean_scale=1.0,head_mask=None,value_core_scale=1.0):
    from ..objectives import value_core_scale as canonical_scale
    scale = canonical_scale(value_core_scale)
    result=forward(params,graph,ports,features,steps=steps,groups=groups,actions=actions,
                   rate_softness=rate_softness,readout_mean_scale=readout_mean_scale,head_mask=head_mask,
                   return_embedding=scale!=1)
    if scale != 1:
        pooled = result['embedding'].T
        fixed = jax.lax.stop_gradient(pooled)
        # Identity in the forward pass; only the value-to-pool cotangent is scaled.
        routed = fixed + jnp.float32(scale) * (pooled - fixed)
        result['value'] = jax.lax.tanh(value_score(params, routed, head_mask), accuracy=HIGHEST)
    return teacher_loss(result,legal,policy,value)


def teacher_loss(result,legal,policy,value):
    """Shared teacher targets and reductions for fly models and neural controls."""
    logits=jnp.where(legal,result['logits'],-1e30)
    policy_loss=-(policy*log_softmax(logits)).sum(axis=-1).mean()
    value_loss=jnp.square(result['value']-value).mean()
    return policy_loss+value_loss,(policy_loss,value_loss)


def adam(params,grad,first,second,step,*,rate=0.003,clip=1.0,
         norm_dtype=jnp.float64,corrections=None,epsilon=1e-8,clip_mode='global',return_stats=False):
    from ..optimizer import validate_clip_mode
    validate_clip_mode(clip_mode)
    # Enable x64 in the parity runner to match the Rust global norm reduction.
    squared={k:jnp.sum(jnp.square(g.astype(norm_dtype))) for k,g in grad.items()}
    norm=jnp.sqrt(sum(squared.values()))
    group_norms={k:jnp.sqrt(v) for k,v in squared.items()}
    if clip_mode=='global':
        scale=jnp.minimum(1.0,clip/jnp.maximum(norm,1e-30)).astype(jnp.float32)
        factors={k:scale for k in params}
    else:
        factors={k:jnp.minimum(1.0,clip/jnp.maximum(v,1e-30)).astype(jnp.float32) for k,v in group_norms.items()}
    if corrections is None:
        c1=jnp.asarray(1.0-0.9**step,dtype=jnp.float32)
        c2=jnp.asarray(1.0-0.999**step,dtype=jnp.float32)
    else:
        c1,c2=corrections
    first={k:0.9*first[k]+0.1*(grad[k]*factors[k]) for k in params}
    second={k:0.999*second[k]+0.001*jnp.square(grad[k]*factors[k]) for k in params}
    rates=rate if isinstance(rate,dict) else {k:rate for k in params}
    params={k:params[k]-rates[k]*(first[k]/c1)/(jnp.sqrt(second[k]/c2)+epsilon) for k in params}
    result=(params,first,second,norm)
    return (*result,group_norms,factors) if return_stats else result
