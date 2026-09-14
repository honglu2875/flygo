"""Conventional residual CNN control; never part of the fixed fly circuit."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from .learner import JaxLearner
from .numerics import HIGHEST
from .model import teacher_loss


@dataclass(frozen=True)
class CNNConfig:
    channels:int=64
    blocks:int=10
    features:int=972
    actions:int=82
    threads:int=16
    seed:int=1


def initialize(config):
    if min(config.channels,config.blocks)<1 or (config.features,config.actions)!=(972,82):
        raise ValueError('The CNN control requires positive width/depth and 9x9 features/actions')
    rng=np.random.default_rng(config.seed);params={};c=config.channels
    def conv(name,k,inputs,outputs,scale=None):
        std=np.sqrt(2/(k*k*inputs)) if scale is None else scale
        params[name+'/kernel']=rng.normal(0,std,(k,k,inputs,outputs)).astype(np.float32)
        params[name+'/bias']=np.zeros(outputs,np.float32)
    def dense(name,inputs,outputs,scale):
        params[name+'/weight']=rng.normal(0,scale,(inputs,outputs)).astype(np.float32)
        params[name+'/bias']=np.zeros(outputs,np.float32)
    conv('stem',3,12,c)
    for block in range(config.blocks):
        conv(f'block{block}/first',3,c,c);conv(f'block{block}/second',3,c,c)
    conv('policy_hidden',3,c,c)
    conv('policy',1,c,1,.02)
    dense('pass',c,1,.02)
    dense('value_hidden',c,c,np.sqrt(2/c))
    dense('value',c,1,.02)
    return params


def convolution(x,kernel,bias):
    return jax.lax.conv_general_dilated(x,kernel,(1,1),'SAME',
        dimension_numbers=('NHWC','HWIO','NHWC'),precision=jax.lax.Precision.HIGHEST)+bias


def forward(params,graph,ports,features,*,blocks):
    def conv(name,x):return convolution(x,params[name+'/kernel'],params[name+'/bias'])
    def dense(name,x):
        return jnp.matmul(x,params[name+'/weight'],precision=jax.lax.Precision.HIGHEST)+params[name+'/bias']
    state=jax.nn.relu(conv('stem',features.reshape(-1,9,9,12)))
    scale=np.float32(1/np.sqrt(2*blocks))
    for block in range(blocks):
        branch=jax.nn.relu(conv(f'block{block}/first',state))
        state=jax.nn.relu(state+scale*conv(f'block{block}/second',branch))
    policy_hidden=jax.nn.relu(conv('policy_hidden',state))
    board_logits=conv('policy',policy_hidden).reshape(-1,81)
    pass_logit=dense('pass',policy_hidden.mean(axis=(1,2)))
    value_hidden=jax.nn.relu(dense('value_hidden',state.mean(axis=(1,2))))
    value=jax.lax.tanh(dense('value',value_hidden)[:,0],accuracy=HIGHEST)
    # The trace represents the final convolutional feature field.
    return dict(logits=jnp.concatenate([board_logits,pass_logit],axis=1),value=value,
                states=state.reshape(len(features),-1).T[None,:,:])


def loss(params,graph,ports,features,legal,policy,value,*,blocks):
    result=forward(params,graph,ports,features,blocks=blocks)
    return teacher_loss(result,legal,policy,value)


class JaxCNN(JaxLearner):
    model_version='residual-cnn-v1'
    numerical_runtime='jax-highest-fp32-norm-cnn-v1'

    def __init__(self,config=CNNConfig(),*,params=None,mesh=None):
        super().__init__(dict(manifest=dict(graph_id='dense-control-no-fly-edges-v1')),config,
            ports={},params=initialize(config) if params is None else params,compute_graph={},
            forward_function=forward,loss_function=loss,kwargs=dict(blocks=config.blocks),mesh=mesh)
