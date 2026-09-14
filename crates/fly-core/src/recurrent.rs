//! v[t+1] = (1-a[type])*v[t] + a[type]*(W rate(v[t]) + b[type] + I).
//! W[e] = sign[src[e]]*softplus(theta[e]); a = .01 + .98*sigmoid(leak).
use crate::{Executor, Graph, Rate, sigmoid, softplus};
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct CoreParams {
    pub edge: Vec<f32>,
    pub leak: Vec<f32>,
    pub bias: Vec<f32>,
}
pub type CoreGrad = CoreParams;
pub struct Tape {
    pub states: Vec<Vec<f32>>,
    pub weights: Arc<Vec<f32>>,
    pub alpha: Arc<Vec<f32>>,
    pub batch: usize,
    pub rate: Rate,
}

/// Reusable transforms for unchanged parameters. Rebuild after any parameter update.
#[derive(Clone)]
pub struct Prepared {
    weights: Arc<Vec<f32>>,
    alpha: Arc<Vec<f32>>,
    initial_message: Arc<Vec<f32>>,
    rate: Rate,
}

pub fn prepare(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    rate: Rate,
) -> Result<Prepared, String> {
    params.validate(graph)?;
    let weights: Vec<f32> = executor.pool.install(|| {
        params
            .edge
            .par_iter()
            .enumerate()
            .map(|(e, &x)| graph.sign[graph.src[e]] * softplus(x))
            .collect()
    });
    let alpha = params
        .leak
        .iter()
        .map(|&x| 0.01 + 0.98 * sigmoid(x))
        .collect();
    // Every prediction starts at the same state. Compute its message once at
    // B=1, retaining the canonical edge summation order, then broadcast it.
    // Training rebuilds this cache after each parameter update.
    let initial_message = executor.multiply(graph, &weights, &vec![rate.value(0.01); graph.neurons()], 1);
    Ok(Prepared {
        weights: Arc::new(weights),
        alpha: Arc::new(alpha),
        initial_message: Arc::new(initial_message),
        rate,
    })
}

impl CoreParams {
    pub fn zeros(graph: &Graph) -> Self {
        Self {
            edge: vec![0.0; graph.edges()],
            leak: vec![0.0; graph.types],
            bias: vec![0.0; graph.types],
        }
    }
    pub fn validate(&self, graph: &Graph) -> Result<(), String> {
        if self.edge.len() != graph.edges()
            || self.leak.len() != graph.types
            || self.bias.len() != graph.types
            || self
                .edge
                .iter()
                .chain(&self.leak)
                .chain(&self.bias)
                .any(|x| !x.is_finite())
        {
            return Err("Invalid core parameter shapes or values".into());
        }
        Ok(())
    }
}

pub fn forward(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    rate: Rate,
) -> Result<Tape, String> {
    let prepared = prepare(graph, executor, params, rate)?;
    forward_prepared(graph, executor, params, drive, batch, steps, &prepared)
}

pub fn forward_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
) -> Result<Tape, String> {
    if batch == 0
        || steps == 0
        || steps > 1024
        || drive.len() != graph.neurons() * batch
        || drive.iter().any(|x| !x.is_finite())
        || prepared.weights.len() != graph.edges()
        || prepared.alpha.len() != graph.types
        || prepared.initial_message.len() != graph.neurons()
    {
        return Err("Invalid recurrent input or step count".into());
    }
    let weights = prepared.weights.clone();
    let alpha = prepared.alpha.clone();
    let mut states: Vec<Vec<f32>> = vec![vec![0.01; drive.len()]];
    for step in 0..steps {
        let previous = states.last().unwrap();
        let mut next = if step == 0 {
            let mut message = vec![0.0; drive.len()];
            executor.pool.install(|| {
                message.par_chunks_mut(batch).enumerate().for_each(|(node, row)| {
                    row.fill(prepared.initial_message[node]);
                })
            });
            message
        } else {
            let rate: Vec<_> = executor.pool.install(|| {
                previous.par_iter().map(|&x| prepared.rate.value(x)).collect()
            });
            executor.multiply(graph, &weights, &rate, batch)
        };
        executor.pool.install(|| {
            next.par_chunks_mut(batch)
                .enumerate()
                .for_each(|(node, out)| {
                    let group = graph.type_id[node];
                    let a = alpha[group];
                    for (b, value) in out.iter_mut().enumerate() {
                        let index = node * batch + b;
                        *value = (1.0 - a) * previous[index]
                            + a * (*value + params.bias[group] + drive[index]);
                    }
                })
        });
        if next.iter().any(|x| !x.is_finite()) {
            return Err(
                "Non-finite recurrent state; reduce the update rate or recurrent depth".into(),
            );
        }
        states.push(next);
    }
    Ok(Tape {
        states,
        weights,
        alpha,
        batch,
        rate: prepared.rate,
    })
}

pub fn backward(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    tape: &Tape,
    last_gradient: &[f32],
) -> Result<(CoreGrad, Vec<f32>), String> {
    let batch = tape.batch;
    if last_gradient.len() != graph.neurons() * batch {
        return Err("Invalid recurrent cotangent".into());
    }
    let mut grad = CoreGrad::zeros(graph);
    let mut drive_grad = vec![0.0; last_gradient.len()];
    let mut state_grad = last_gradient.to_vec();
    let transpose_weights = executor.transpose_weights(graph, &tape.weights);
    for step in (0..tape.states.len() - 1).rev() {
        let previous = &tape.states[step];
        let next = &tape.states[step + 1];
        let rate: Vec<_> = executor.pool.install(|| {
            previous.par_iter().map(|&x| tape.rate.value(x)).collect()
        });
        let mut message_grad = vec![0.0; state_grad.len()];
        for node in 0..graph.neurons() {
            let group = graph.type_id[node];
            let a = tape.alpha[group];
            let s = sigmoid(params.leak[group]);
            for b in 0..batch {
                let index = node * batch + b;
                let g = state_grad[index];
                message_grad[index] = a * g;
                drive_grad[index] += a * g;
                grad.bias[group] += a * g;
                // next-prev = a*(message+bias+drive-prev), avoids another sparse multiply.
                grad.leak[group] += g * (next[index] - previous[index]) / a * 0.98 * s * (1.0 - s);
            }
        }
        executor.edge_vjp(graph, &rate, &message_grad, batch, &mut grad.edge);
        let propagated =
            executor.transpose_prepared(graph, &transpose_weights, &message_grad, batch);
        executor.pool.install(|| {
            state_grad
                .par_chunks_mut(batch)
                .enumerate()
                .for_each(|(node, row)| {
                    let a = tape.alpha[graph.type_id[node]];
                    for (b, g) in row.iter_mut().enumerate() {
                        let index = node * batch + b;
                        *g = (1.0 - a) * *g + tape.rate.pullback(previous[index], propagated[index]);
                    }
                })
        });
    }
    executor.pool.install(|| {
        grad.edge.par_iter_mut().enumerate().for_each(|(edge, g)| {
            *g *= graph.sign[graph.src[edge]] * sigmoid(params.edge[edge]);
        })
    });
    Ok((grad, drive_grad))
}
