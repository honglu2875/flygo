"""Explicit nominal inference arithmetic, separate from backend padding and time."""


def fly_cost(neurons,edges,sensory,*,passes=4,groups=656,actions=82,batch_size=1,readout_neurons=None):
    readout=neurons-sensory if readout_neurons is None else readout_neurons
    if not 0<=readout<=neurons-sensory:raise ValueError('Readout count must fit the nonsensory population')
    parts=dict(sparse_aggregation=2*passes*edges,rate_dynamics=6*passes*neurons,
               input_gain=sensory,readout_pool=2*readout+readout/batch_size,
               heads=2*groups*(actions+1))
    return dict(arithmetic_flops=sum(parts.values()),parts=parts,readout_neurons=readout,
                relu_comparisons=passes*neurons+readout,tanh_evaluations=1,
                weight_transforms='Cached across predictions at fixed parameters; report cold transforms separately',
                convention='2 operations per multiply-add; exact nonzero edges; no loss, masks, Go features or search')


def cnn_cost(*,channels=64,blocks=10):
    c=channels;area=81
    conv3=lambda inputs,outputs:2*area*9*inputs*outputs
    parts=dict(stem=conv3(12,c)+area*c,
        residual_blocks=blocks*(2*conv3(c,c)+4*area*c),
        policy_hidden=conv3(c,c)+area*c,policy_projection=2*area*c+area,
        pooling=2*area*c,pass_head=2*c+1,value_head=2*c*c+c+2*c+1)
    return dict(arithmetic_flops=sum(parts.values()),parts=parts,
                relu_comparisons=(2+2*blocks)*area*c+c,tanh_evaluations=1,
                nominal_3x3_site_pairs=729,nonpadding_3x3_site_pairs=625,
                convention='Standard SAME-convolution count includes boundary zero padding; accelerator channel/tile padding is additional')
