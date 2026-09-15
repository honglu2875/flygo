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
    // Reset predictions start at the same state. Compute their message once at
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
    forward_start_prepared(graph, executor, params, drive, batch, steps, prepared, None)
}

/// Full tape from an explicit node-major [N,B] state. The reset-message cache
/// is never valid for this path; transformed weights and leaks are reusable.
#[allow(clippy::too_many_arguments)]
pub fn forward_from_state_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<Tape, String> {
    forward_start_prepared(graph, executor, params, drive, batch, steps, prepared, Some(initial_state))
}

#[allow(clippy::too_many_arguments)]
fn forward_start_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
    initial_state: Option<&[f32]>,
) -> Result<Tape, String> {
    validate_input(graph, drive, batch, steps, prepared)?;
    let mut states = vec![starting_state(drive.len(), initial_state)?];
    for step in 0..steps {
        states.push(next_state(graph, executor, params, drive, batch,
            states.last().unwrap(), step == 0 && initial_state.is_none(), prepared, None)?);
    }
    Ok(Tape {
        states,
        weights: prepared.weights.clone(),
        alpha: prepared.alpha.clone(),
        batch,
        rate: prepared.rate,
    })
}

/// Prediction retains only the current state; memory does not grow with depth.
pub fn predict(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    rate: Rate,
) -> Result<Vec<f32>, String> {
    let prepared = prepare(graph, executor, params, rate)?;
    predict_prepared(graph, executor, params, drive, batch, steps, &prepared)
}

/// Cached transforms must correspond to these unchanged parameters.
pub fn predict_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
) -> Result<Vec<f32>, String> {
    predict_start_prepared(graph, executor, params, drive, batch, steps, prepared, None)
}

/// Streaming prediction from an explicit state, with O(NB) state memory.
/// Returns a complete state suitable for a subsequent call, not a pruned state.
#[allow(clippy::too_many_arguments)]
pub fn predict_from_state_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<Vec<f32>, String> {
    predict_start_prepared(graph, executor, params, drive, batch, steps, prepared, Some(initial_state))
}

#[allow(clippy::too_many_arguments)]
fn predict_start_prepared(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
    initial_state: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    validate_input(graph, drive, batch, steps, prepared)?;
    let mut state = starting_state(drive.len(), initial_state)?;
    for step in 0..steps {
        state = next_state(graph, executor, params, drive, batch, &state,
            step == 0 && initial_state.is_none(), prepared, None)?;
    }
    Ok(state)
}

fn starting_state(length: usize, initial_state: Option<&[f32]>) -> Result<Vec<f32>, String> {
    match initial_state {
        Some(state) if state.len() != length || state.iter().any(|x| !x.is_finite()) =>
            Err("Initial state must be finite node-major [N,B]".into()),
        Some(state) => Ok(state.to_vec()),
        None => Ok(vec![0.01; length]),
    }
}

/// Only Model constructs these graph/port-specific dependency masks. This state
/// is valid at the selected outputs, and must not be reused as recurrent memory.
pub(crate) fn predict_required_prepared(
    graph: &Graph, executor: &Executor, params: &CoreParams, drive: &[f32],
    batch: usize, prepared: &Prepared, required: &[Arc<[bool]>],
) -> Result<Vec<f32>, String> {
    validate_input(graph, drive, batch, required.len(), prepared)?;
    let mut state = vec![0.01; drive.len()];
    for (step, rows) in required.iter().enumerate() {
        state = next_state(graph, executor, params, drive, batch, &state,
            step == 0, prepared, Some(rows))?;
    }
    Ok(state)
}

fn validate_input(
    graph: &Graph,
    drive: &[f32],
    batch: usize,
    steps: usize,
    prepared: &Prepared,
) -> Result<(), String> {
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
    Ok(())
}

// Both prediction and differentiation use this exact arithmetic and reduction
// order. Only the caller decides whether previous states remain in a tape.
#[allow(clippy::too_many_arguments)]
fn next_state(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    drive: &[f32],
    batch: usize,
    previous: &[f32],
    initial: bool,
    prepared: &Prepared,
    required: Option<&[bool]>,
) -> Result<Vec<f32>, String> {
    let mut next = if initial {
        let mut message = vec![0.0; drive.len()];
        executor.pool.install(|| {
            message.par_chunks_mut(batch).enumerate().for_each(|(node, row)| {
                if required.is_none_or(|rows| rows[node]) {
                    row.fill(prepared.initial_message[node]);
                }
            })
        });
        message
    } else {
        let rate: Vec<_> = executor.pool.install(|| {
            previous.par_iter().map(|&x| prepared.rate.value(x)).collect()
        });
        match required {
            Some(rows) => executor.multiply_required(graph, &prepared.weights, &rate, batch, rows),
            None => executor.multiply(graph, &prepared.weights, &rate, batch),
        }
    };
    executor.pool.install(|| {
        next.par_chunks_mut(batch)
            .enumerate()
            .for_each(|(node, out)| {
                if required.is_some_and(|rows| !rows[node]) { return; }
                let group = graph.type_id[node];
                let a = prepared.alpha[group];
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
    Ok(next)
}

pub fn backward(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    tape: &Tape,
    last_gradient: &[f32],
) -> Result<(CoreGrad, Vec<f32>), String> {
    let (grad, drive_grad, _) = backward_with_state(graph, executor, params, tape, last_gradient)?;
    Ok((grad, drive_grad))
}

/// Return parameter, sensory-drive and initial-state cotangents. The final
/// cotangent may include a later chunk's initial-state gradient; no detach is
/// implicit here. Parameters must be unchanged since the tape was produced.
pub fn backward_with_state(
    graph: &Graph,
    executor: &Executor,
    params: &CoreParams,
    tape: &Tape,
    last_gradient: &[f32],
) -> Result<(CoreGrad, Vec<f32>, Vec<f32>), String> {
    let batch = tape.batch;
    if last_gradient.len() != graph.neurons() * batch || last_gradient.iter().any(|x| !x.is_finite()) {
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
    Ok((grad, drive_grad, state_grad))
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
