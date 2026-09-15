//! Sparse sensory attachment and disjoint readout; all learned computation uses the graph.
use crate::{CoreGrad, CoreParams, Executor, Graph, Rate, recurrent};
use rayon::prelude::*;
use crate::dependency::ReadoutPlan;
use std::sync::{Arc, Mutex};

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
    readout_mean_scale: f32,
    readout_plan: Mutex<Option<Arc<ReadoutPlan>>>,
    head_mask: Option<HeadMask>,
}
struct HeadMask {
    policy: Vec<Vec<usize>>,
    value: Vec<usize>,
    policy_enabled: Vec<u8>,
    value_enabled: Vec<u8>,
}
/// Prediction has no backward tape and cannot accidentally enter backward.
pub struct Prediction {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub scores: Vec<f32>,
    pub pooled: Vec<f32>,
}
pub struct Output {
    pub logits: Vec<f32>,
    pub values: Vec<f32>,
    pub scores: Vec<f32>,
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
            readout_mean_scale: 1.0,
            readout_plan: Mutex::new(None),
            head_mask: None,
        })
    }
    /// Constrain external decoder coefficients; recurrent connectivity is unchanged.
    pub fn with_head_mask(mut self, policy: Vec<u8>, value: Vec<u8>) -> Result<Self, String> {
        let p = &self.ports;
        if policy.len() != p.actions * p.groups || value.len() != p.groups
            || policy.iter().chain(&value).any(|&x| x > 1)
            || policy.chunks(p.groups).any(|row| !row.contains(&1)) || !value.contains(&1)
        {
            return Err("Invalid binary head mask".into());
        }
        let selected = |row: &[u8]| row.iter().enumerate()
            .filter_map(|(i, &enabled)| (enabled == 1).then_some(i)).collect();
        // An explicit all-enabled artifact executes the original dense path.
        self.head_mask = if policy.iter().chain(&value).all(|&x| x == 1) { None } else {
            Some(HeadMask {
                policy: policy.chunks(p.groups).map(selected).collect(), value: selected(&value),
                policy_enabled: policy, value_enabled: value,
            })
        };
        Ok(self)
    }
    #[inline]
    fn policy_groups(&self, action: usize, mut visit: impl FnMut(usize)) {
        if let Some(mask) = &self.head_mask {
            for &group in &mask.policy[action] { visit(group); }
        } else {
            for group in 0..self.ports.groups { visit(group); }
        }
    }
    #[inline]
    fn value_groups(&self, mut visit: impl FnMut(usize)) {
        if let Some(mask) = &self.head_mask {
            for &group in &mask.value { visit(group); }
        } else {
            for group in 0..self.ports.groups { visit(group); }
        }
    }
    /// Derivatives and moments on disabled coefficients must never enter Adam.
    pub fn validate_head_zeros(&self, arrays: &Params) -> Result<(), String> {
        if let Some(mask) = &self.head_mask {
            for (array, enabled) in [(&arrays.policy_weight, &mask.policy_enabled),
                                     (&arrays.value_weight, &mask.value_enabled)] {
                if array.len() != enabled.len() || array.iter().zip(enabled).any(|(&v, &e)| e == 0 && v != 0.0) {
                    return Err("Disabled head coefficients need zero gradients and moments".into());
                }
            }
        }
        Ok(())
    }
    /// Fixed, invertible conditioning of pooled features before the heads.
    pub fn with_readout_mean_scale(mut self, scale: f32) -> Result<Self, String> {
        if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
            return Err("Readout mean scale must be finite and in (0, 1]".into());
        }
        self.readout_mean_scale = scale;
        Ok(self)
    }
    // This symmetric linear transform is also its own transpose, so the same
    // operation applies to the pooled cotangent. It is per prediction, not a
    // batch statistic. The baseline bypass preserves every existing bit.
    fn condition_readout(&self, pooled: &mut [f32], batch: usize) {
        if self.readout_mean_scale == 1.0 { return; }
        let groups = self.ports.groups;
        for b in 0..batch {
            // Sum deviations from one pool before restoring the scaled common
            // term. Summing hundreds of nearly equal positive pools directly
            // loses low-order bits that Adam can amplify near a zero feature.
            let origin = pooled[b];
            let mut sum = 0.0;
            for group in 0..groups {
                pooled[group * batch + b] -= origin;
                sum += pooled[group * batch + b];
            }
            let correction = self.readout_mean_scale * origin
                - (1.0 - self.readout_mean_scale) * (sum / groups as f32);
            for group in 0..groups { pooled[group * batch + b] += correction; }
        }
    }
    pub fn validate(&self, params: &Params) -> Result<(), String> {
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
        let arrays = params.arrays();
        // These lengths include the core arrays. Check every value once using
        // this model's pinned executor; the old core check scanned E twice.
        if arrays.iter().zip(lengths).any(|(v, n)| v.len() != n)
            || self.executor.pool.install(|| {
                arrays.par_iter().any(|v| v.par_iter().any(|x| !x.is_finite()))
            }) {
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
        let drive = self.input_drive(params, input, batch)?;
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
        let prediction = self.readout(params, tape.states.last().unwrap(), batch);
        Ok(Output {
            logits: prediction.logits,
            values: prediction.values,
            scores: prediction.scores,
            pooled: prediction.pooled,
            tape,
        })
    }
    pub fn predict(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
    ) -> Result<Prediction, String> {
        self.predict_prepared(params, input, batch, steps, None)
    }
    /// Same equations as forward, without retaining a differentiation tape.
    pub fn predict_prepared(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
        prepared: Option<&recurrent::Prepared>,
    ) -> Result<Prediction, String> {
        let drive = self.input_drive(params, input, batch)?;
        let state = match prepared {
            Some(prepared) => recurrent::predict_prepared(
                &self.graph, &self.executor, &params.core, &drive, batch, steps, prepared,
            )?,
            None => recurrent::predict(
                &self.graph, &self.executor, &params.core, &drive, batch, steps, self.rate,
            )?,
        };
        Ok(self.readout(params, &state, batch))
    }
    fn plan(&self, steps: usize) -> Result<Arc<ReadoutPlan>, String> {
        let mut cached = self.readout_plan.lock().map_err(|error| error.to_string())?;
        if cached.as_ref().is_none_or(|plan| plan.required.len() != steps) {
            *cached = Some(Arc::new(ReadoutPlan::new(&self.graph, &self.ports.output_group, steps)?));
        }
        Ok(cached.as_ref().unwrap().clone())
    }
    /// Static incoming-edge counts, before zero skipping and the first-message cache.
    pub fn prediction_dependencies(&self, steps: usize) -> Result<Vec<(usize, usize)>, String> {
        Ok(self.plan(steps)?.counts(&self.graph))
    }
    /// Optional readout-only execution. Unneeded rows are not evaluated or returned;
    /// callers needing full states or global divergence checks must use forward.
    pub fn predict_pruned_prepared(
        &self, params: &Params, input: &[f32], batch: usize, steps: usize,
        prepared: Option<&recurrent::Prepared>,
    ) -> Result<Prediction, String> {
        let plan = self.plan(steps)?;
        let drive = self.input_drive(params, input, batch)?;
        let owned;
        let prepared = match prepared {
            Some(prepared) => prepared,
            None => {
                owned = recurrent::prepare(&self.graph, &self.executor, &params.core, self.rate)?;
                &owned
            }
        };
        let state = recurrent::predict_required_prepared(&self.graph, &self.executor,
            &params.core, &drive, batch, prepared, &plan.required)?;
        Ok(self.readout(params, &state, batch))
    }
    fn input_drive(&self, params: &Params, input: &[f32], batch: usize) -> Result<Vec<f32>, String> {
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
        Ok(drive)
    }
    fn readout(&self, params: &Params, state: &[f32], batch: usize) -> Prediction {
        let p = &self.ports;
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
        self.condition_readout(&mut pooled, batch);
        let mut logits = vec![0.0; batch * p.actions];
        let mut values = vec![0.0; batch];
        let mut scores = vec![0.0; batch];
        for b in 0..batch {
            for action in 0..p.actions {
                let mut value = f64::from(params.policy_bias[action]);
                self.policy_groups(action, |group| {
                    value +=
                        f64::from(params.policy_weight[action * p.groups + group])
                            * f64::from(pooled[group * batch + b]);
                });
                logits[b * p.actions + action] = value as f32;
            }
            let mut value = f64::from(params.value_bias[0]);
            self.value_groups(|group| {
                value += f64::from(params.value_weight[group]) * f64::from(pooled[group * batch + b]);
            });
            scores[b] = value as f32;
            values[b] = scores[b].tanh();
        }
        Prediction { logits, values, scores, pooled }
    }
    pub fn backward(
        &self,
        params: &Params,
        input: &[f32],
        output: &Output,
        dlogits: &[f32],
        dvalues: &[f32],
    ) -> Result<Grad, String> {
        self.backward_with_embedding(params, input, output, dlogits, dvalues, None, None)
    }
    /// Add a cotangent at the individual/pool readout, before the task heads.
    /// The optional array is group-major [G,B], matching Output::pooled.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_with_embedding(
        &self,
        params: &Params,
        input: &[f32],
        output: &Output,
        dlogits: &[f32],
        dvalues: &[f32],
        dembedding: Option<&[f32]>,
        dscores: Option<&[f32]>,
    ) -> Result<Grad, String> {
        self.backward_routed(params, input, output, dlogits, dvalues, dembedding, dscores, 1.0)
    }
    /// Scale only the value-head contribution entering the shared representation.
    /// Generic embedding VJPs retain their literal cotangents and use scale one.
    #[allow(clippy::too_many_arguments)]
    fn backward_routed(
        &self,
        params: &Params,
        input: &[f32],
        output: &Output,
        dlogits: &[f32],
        dvalues: &[f32],
        dembedding: Option<&[f32]>,
        dscores: Option<&[f32]>,
        value_core_scale: f32,
    ) -> Result<Grad, String> {
        let p = &self.ports;
        let batch = output.tape.batch;
        if dlogits.len() != batch * p.actions
            || dvalues.len() != batch
            || input.len() != p.features * batch
            || dembedding.is_some_and(|x| x.len() != p.groups * batch || x.iter().any(|v| !v.is_finite()))
            || dscores.is_some_and(|x| x.len() != batch || x.iter().any(|v| !v.is_finite()))
            || dlogits.iter().chain(dvalues).any(|v| !v.is_finite())
        {
            return Err("Invalid model cotangent shapes".into());
        }
        let mut grad = Grad::zeros(self);
        // Small task-head reductions can cancel almost completely. Accumulate
        // them in FP64 and round once at the boundary to the FP32 circuit.
        let mut policy_weight = vec![0.0_f64; p.actions * p.groups];
        let mut policy_bias = vec![0.0_f64; p.actions];
        let mut value_weight = vec![0.0_f64; p.groups];
        let mut value_bias = 0.0_f64;
        let mut dpool = vec![0.0_f64; p.groups * batch];
        for b in 0..batch {
            for action in 0..p.actions {
                let g = f64::from(dlogits[b * p.actions + action]);
                policy_bias[action] += g;
                self.policy_groups(action, |group| {
                    policy_weight[action * p.groups + group] +=
                        g * f64::from(output.pooled[group * batch + b]);
                    dpool[group * batch + b] += g * f64::from(params.policy_weight[action * p.groups + group]);
                });
            }
            let g = dvalues[b] * (1.0 - output.values[b] * output.values[b])
                + dscores.map_or(0.0, |g| g[b]);
            let g = f64::from(g);
            let core_g = if value_core_scale == 1.0 { g } else { g * f64::from(value_core_scale) };
            value_bias += g;
            self.value_groups(|group| {
                value_weight[group] += g * f64::from(output.pooled[group * batch + b]);
                dpool[group * batch + b] += core_g * f64::from(params.value_weight[group]);
            });
        }
        grad.policy_weight = policy_weight.into_iter().map(|x| x as f32).collect();
        grad.policy_bias = policy_bias.into_iter().map(|x| x as f32).collect();
        grad.value_weight = value_weight.into_iter().map(|x| x as f32).collect();
        grad.value_bias[0] = value_bias as f32;
        let mut dpool: Vec<f32> = dpool.into_iter().map(|x| x as f32).collect();
        if let Some(extra) = dembedding {
            for (g, &value) in dpool.iter_mut().zip(extra) { *g += value; }
        }
        self.condition_readout(&mut dpool, batch);
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
        self.loss_and_grad_with_value_core_scale(params, input, batch, steps, targets, 1.0)
    }
    /// CE + MSE metrics and full head gradients; shared gradient is g_policy + scale*g_value.
    /// Prediction, recurrence, and both task-head derivatives are unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn loss_and_grad_with_value_core_scale(
        &self,
        params: &Params,
        input: &[f32],
        batch: usize,
        steps: usize,
        targets: Targets<'_>,
        value_core_scale: f32,
    ) -> Result<(f64, f64, Grad), String> {
        if !value_core_scale.is_finite() || value_core_scale < 0.0 {
            return Err("Value core scale must be finite and nonnegative".into());
        }
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
            let sum: f64 = policy[offset..offset + actions].iter().map(|&x| f64::from(x)).sum();
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
            let denominator: f64 = (0..actions)
                .filter(|&a| legal[offset + a] != 0)
                .map(|a| (f64::from(output.logits[offset + a]) - f64::from(max)).exp())
                .sum();
            for a in 0..actions {
                if legal[offset + a] != 0 {
                    let shifted = f64::from(output.logits[offset + a]) - f64::from(max);
                    let logp = shifted - denominator.ln();
                    policy_loss -= f64::from(policy[offset + a]) * logp / batch as f64;
                    // d[-sum(q * log_softmax(logits))]/dlogit = sum(q) * p - q.
                    // Accepted FP32 targets need not sum to exactly one. Reuse
                    // the probability normalization directly, without exp(logp).
                    let probability = shifted.exp() / denominator;
                    dlogits[offset + a] = ((sum * probability - f64::from(policy[offset + a])) / batch as f64) as f32;
                }
            }
            let delta = output.values[b] - value[b];
            value_loss += f64::from(delta * delta) / batch as f64;
            dvalues[b] = 2.0 * delta / batch as f32;
        }
        let grad = self.backward_routed(params, input, &output, &dlogits, &dvalues,
            None, None, value_core_scale)?;
        Ok((policy_loss, value_loss, grad))
    }
}
