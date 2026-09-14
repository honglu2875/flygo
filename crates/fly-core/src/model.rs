//! Sparse sensory attachment and disjoint readout; all learned computation uses the graph.
use crate::{CoreGrad, CoreParams, Executor, Graph, Rate, recurrent};
use rayon::prelude::*;

#[derive(Clone)]
pub struct Ports {
    pub input_index: Vec<i32>,
    pub output_group: Vec<i32>,
    pub output_scale: Vec<f32>,
    pub features: usize,
    pub groups: usize,
    pub actions: usize,
}

#[derive(Clone)]
pub struct Params {
    pub core: CoreParams,
    pub input_gain: Vec<f32>,
    pub readout_gain: Vec<f32>,
    pub policy_weight: Vec<f32>,
    pub policy_bias: Vec<f32>,
    pub value_weight: Vec<f32>,
    pub value_bias: Vec<f32>,
}
pub type Grad = Params;
pub struct Targets<'a> {
    pub legal: &'a [u8],
    pub policy: &'a [f32],
    pub value: &'a [f32],
}
pub struct Model {
    pub graph: Graph,
    pub ports: Ports,
    pub executor: Executor,
    pub rate: Rate,
    readout_nodes: Vec<Vec<usize>>,
}
pub struct Output {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub pooled: Vec<f32>,
    pub tape: recurrent::Tape,
}

impl Params {
    pub fn zeros(model: &Model) -> Self {
        let p = &model.ports;
        Self {
            core: CoreGrad::zeros(&model.graph),
            input_gain: vec![0.0; p.features],
            readout_gain: vec![0.0; model.graph.neurons()],
            policy_weight: vec![0.0; p.groups * p.actions],
            policy_bias: vec![0.0; p.actions],
            value_weight: vec![0.0; p.groups],
            value_bias: vec![0.0; 1],
        }
    }
    pub fn arrays(&self) -> [&[f32]; 9] {
        [
            &self.core.edge,
            &self.core.leak,
            &self.core.bias,
            &self.input_gain,
            &self.readout_gain,
            &self.policy_weight,
            &self.policy_bias,
            &self.value_weight,
            &self.value_bias,
        ]
    }
    pub fn arrays_mut(&mut self) -> [&mut [f32]; 9] {
        [
            &mut self.core.edge,
            &mut self.core.leak,
            &mut self.core.bias,
            &mut self.input_gain,
            &mut self.readout_gain,
            &mut self.policy_weight,
            &mut self.policy_bias,
            &mut self.value_weight,
            &mut self.value_bias,
        ]
    }
}

impl Model {
    pub fn new(graph: Graph, ports: Ports, threads: usize) -> Result<Self, String> {
        Self::with_rate(graph, ports, threads, Rate::default())
    }
    pub fn with_rate(graph: Graph, ports: Ports, threads: usize, rate: Rate) -> Result<Self, String> {
        let n = graph.neurons();
        if ports.features == 0
            || ports.groups == 0
            || ports.actions == 0
            || ports.input_index.len() != n
            || ports.output_group.len() != n
            || ports.output_scale.len() != n
        {
            return Err("Invalid port dimensions".into());
        }
        for i in 0..n {
            if ports.input_index[i] < -1
                || ports.input_index[i] >= ports.features as i32
                || ports.output_group[i] < -1
                || ports.output_group[i] >= ports.groups as i32
                || (ports.input_index[i] >= 0 && ports.output_group[i] >= 0)
                || !ports.output_scale[i].is_finite()
                || ports.output_scale[i] < 0.0
            {
                return Err("Invalid or overlapping sensory/readout ports".into());
            }
        }
        let mut readout_nodes = vec![Vec::new(); ports.groups];
        for (node, &group) in ports.output_group.iter().enumerate() {
            if group >= 0 { readout_nodes[group as usize].push(node); }
        }
        Ok(Self {
            graph,
            ports,
            executor: Executor::new(threads)?,
            rate,
            readout_nodes,
        })
    }
    pub fn validate(&self, params: &Params) -> Result<(), String> {
        params.core.validate(&self.graph)?;
        let p = &self.ports;
        let lengths = [
            self.graph.edges(),
            self.graph.types,
            self.graph.types,
            p.features,
            self.graph.neurons(),
            p.groups * p.actions,
            p.actions,
            p.groups,
            1,
        ];
        if params
            .arrays()
            .iter()
            .zip(lengths)
            .any(|(v, n)| v.len() != n || v.iter().any(|x| !x.is_finite()))
        {
            return Err("Invalid model parameter array".into());
        }
        Ok(())
    }
    pub fn forward(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
    ) -> Result<Output, String> {
        self.forward_prepared(params, input, batch, steps, None)
    }
    /// Cached transforms must correspond to these unchanged parameters.
    pub fn forward_prepared(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
        prepared: Option<&recurrent::Prepared>,
    ) -> Result<Output, String> {
        self.validate(params)?;
        let p = &self.ports;
        if batch == 0 || input.len() != p.features * batch || input.iter().any(|x| !x.is_finite()) {
            return Err("Input must be finite feature-major [F,B]".into());
        }
        let mut drive = vec![0.0; self.graph.neurons() * batch];
        for (node, &feature) in p.input_index.iter().enumerate() {
            if feature >= 0 {
                let feature = feature as usize;
                for b in 0..batch {
                    drive[node * batch + b] =
                        params.input_gain[feature] * input[feature * batch + b];
                }
            }
        }
        let tape = match prepared {
            Some(prepared) => recurrent::forward_prepared(
                &self.graph,
                &self.executor,
                &params.core,
                &drive,
                batch,
                steps,
                prepared,
            )?,
            None => recurrent::forward(
                &self.graph,
                &self.executor,
                &params.core,
                &drive,
                batch,
                steps,
                self.rate,
            )?,
        };
        let state = tape.states.last().unwrap();
        let mut pooled = vec![0.0; p.groups * batch];
        // Pools are independent; each retains the original ascending neuron
        // order, so parallel execution does not change floating-point sums.
        self.executor.pool.install(|| {
            pooled.par_chunks_mut(batch).enumerate().for_each(|(group, row)| {
                for &node in &self.readout_nodes[group] {
                    let scale = p.output_scale[node] * params.readout_gain[node];
                    for b in 0..batch {
                        row[b] += scale * self.rate.value(state[node * batch + b]);
                    }
                }
            })
        });
        let mut logits = vec![0.0; batch * p.actions];
        let mut values = vec![0.0; batch];
        for b in 0..batch {
            for action in 0..p.actions {
                let mut value = params.policy_bias[action];
                for group in 0..p.groups {
                    value +=
                        params.policy_weight[action * p.groups + group] * pooled[group * batch + b];
                }
                logits[b * p.actions + action] = value;
            }
            let mut value = params.value_bias[0];
            for group in 0..p.groups {
                value += params.value_weight[group] * pooled[group * batch + b];
            }
            values[b] = value.tanh();
        }
        Ok(Output {
            logits,
            values,
            pooled,
            tape,
        })
    }
    pub fn backward(
        &self,
        params: &Params,
        input: &[f32],
        output: &Output,
        dlogits: &[f32],
        dvalues: &[f32],
    ) -> Result<Grad, String> {
        let p = &self.ports;
        let batch = output.tape.batch;
        if dlogits.len() != batch * p.actions
            || dvalues.len() != batch
            || input.len() != p.features * batch
        {
            return Err("Invalid model cotangent shapes".into());
        }
        let mut grad = Grad::zeros(self);
        let mut dpool = vec![0.0; p.groups * batch];
        for b in 0..batch {
            for action in 0..p.actions {
                let g = dlogits[b * p.actions + action];
                grad.policy_bias[action] += g;
                for group in 0..p.groups {
                    grad.policy_weight[action * p.groups + group] +=
                        g * output.pooled[group * batch + b];
                    dpool[group * batch + b] += g * params.policy_weight[action * p.groups + group];
                }
            }
            let g = dvalues[b] * (1.0 - output.values[b] * output.values[b]);
            grad.value_bias[0] += g;
            for group in 0..p.groups {
                grad.value_weight[group] += g * output.pooled[group * batch + b];
                dpool[group * batch + b] += g * params.value_weight[group];
            }
        }
        let state = output.tape.states.last().unwrap();
        let mut dstate = vec![0.0; state.len()];
        self.executor.pool.install(|| {
            grad.readout_gain.par_iter_mut().zip(dstate.par_chunks_mut(batch))
                .enumerate().for_each(|(node, (gain_grad, row))| {
                let group = p.output_group[node];
                if group >= 0 {
                    for b in 0..batch {
                        let g = dpool[group as usize * batch + b] * p.output_scale[node];
                        *gain_grad += g * self.rate.value(state[node * batch + b]);
                        row[b] = self.rate.pullback(state[node * batch + b], g * params.readout_gain[node]);
                    }
                }
            })
        });
        let (core, ddrive) = recurrent::backward(
            &self.graph,
            &self.executor,
            &params.core,
            &output.tape,
            &dstate,
        )?;
        grad.core = core;
        for (node, &feature) in p.input_index.iter().enumerate() {
            if feature >= 0 {
                let feature = feature as usize;
                for b in 0..batch {
                    grad.input_gain[feature] +=
                        ddrive[node * batch + b] * input[feature * batch + b];
                }
            }
        }
        Ok(grad)
    }
    pub fn loss_and_grad(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
        targets: Targets<'_>,
    ) -> Result<(f64, f64, Grad), String> {
        let Targets {
            legal,
            policy,
            value,
        } = targets;
        let actions = self.ports.actions;
        if legal.len() != batch * actions
            || policy.len() != legal.len()
            || value.len() != batch
            || policy.iter().any(|&x| !x.is_finite() || x < 0.0)
            || value.iter().any(|x| !x.is_finite() || x.abs() > 1.00001)
        {
            return Err("Invalid policy/value targets".into());
        }
        let output = self.forward(params, input, batch, steps)?;
        let mut dlogits = vec![0.0; batch * actions];
        let mut dvalues = vec![0.0; batch];
        let mut policy_loss = 0.0_f64;
        let mut value_loss = 0.0_f64;
        for b in 0..batch {
            let offset = b * actions;
            let sum: f32 = policy[offset..offset + actions].iter().sum();
            if (sum - 1.0).abs() > 1e-4
                || !(0..actions).any(|a| legal[offset + a] != 0)
                || (0..actions).any(|a| legal[offset + a] == 0 && policy[offset + a] != 0.0)
            {
                return Err("Targets must normalize over legal moves only".into());
            }
            let max = (0..actions)
                .filter(|&a| legal[offset + a] != 0)
                .map(|a| output.logits[offset + a])
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = (0..actions)
                .filter(|&a| legal[offset + a] != 0)
                .map(|a| (output.logits[offset + a] - max).exp())
                .sum();
            for a in 0..actions {
                if legal[offset + a] != 0 {
                    let logp = output.logits[offset + a] - max - denominator.ln();
                    policy_loss -= f64::from(policy[offset + a] * logp) / batch as f64;
                    dlogits[offset + a] = (logp.exp() - policy[offset + a]) / batch as f32;
                }
            }
            let delta = output.values[b] - value[b];
            value_loss += f64::from(delta * delta) / batch as f64;
            dvalues[b] = 2.0 * delta / batch as f32;
        }
        let grad = self.backward(params, input, &output, &dlogits, &dvalues)?;
        Ok((policy_loss, value_loss, grad))
    }
}
