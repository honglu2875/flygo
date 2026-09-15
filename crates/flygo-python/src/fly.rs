//! Thin NumPy boundary for the owned Rust model and optimizer.
use fly_core::{CoreParams, Graph, Model, Params, Ports, Rate, Targets, optim::{Adam, ClipMode, UpdateStats}};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use std::sync::Mutex;

struct State {
    model: Model,
    params: Params,
    adam: Adam,
    prepared: Option<fly_core::recurrent::Prepared>,
    revision: u64,
    last_update: Option<UpdateStats>,
}
#[pyclass]
pub struct FlyModel {
    state: Mutex<State>,
}
type Arrays<'py> = Vec<Bound<'py, PyArray1<f32>>>;
type Infer<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Arrays<'py>,
);

fn parse_params(arrays: Vec<PyReadonlyArray1<'_, f32>>) -> PyResult<Params> {
    if arrays.len() != 9 {
        return Err(PyValueError::new_err(
            "Expected nine named parameter groups",
        ));
    }
    let mut arrays = arrays
        .into_iter()
        .map(|x| x.as_slice().map(|v| v.to_vec()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter();
    Ok(Params {
        core: CoreParams {
            edge: arrays.next().unwrap(),
            leak: arrays.next().unwrap(),
            bias: arrays.next().unwrap(),
        },
        input_gain: arrays.next().unwrap(),
        readout_gain: arrays.next().unwrap(),
        policy_weight: arrays.next().unwrap(),
        policy_bias: arrays.next().unwrap(),
        value_weight: arrays.next().unwrap(),
        value_bias: arrays.next().unwrap(),
    })
}
fn export<'py>(py: Python<'py>, params: &Params) -> Arrays<'py> {
    params
        .arrays()
        .into_iter()
        .map(|x| x.to_vec().into_pyarray(py))
        .collect()
}

#[pymethods]
impl FlyModel {
    #[new]
    #[pyo3(signature = (indptr, src, type_id, sign, input_index, output_group, output_scale,
                       features, groups, actions, threads, params, rate_softness=0.0, readout_mean_scale=1.0,
                       policy_mask=None, value_mask=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i32>,
        src: PyReadonlyArray1<'_, i32>,
        type_id: PyReadonlyArray1<'_, i32>,
        sign: PyReadonlyArray1<'_, f32>,
        input_index: PyReadonlyArray1<'_, i32>,
        output_group: PyReadonlyArray1<'_, i32>,
        output_scale: PyReadonlyArray1<'_, f32>,
        features: usize,
        groups: usize,
        actions: usize,
        threads: usize,
        params: Vec<PyReadonlyArray1<'_, f32>>,
        rate_softness: f32,
        readout_mean_scale: f32,
        policy_mask: Option<PyReadonlyArray1<'_, u8>>,
        value_mask: Option<PyReadonlyArray1<'_, u8>>,
    ) -> PyResult<Self> {
        let (indptr, src, type_id, sign) = (
            indptr.as_slice()?.to_vec(),
            src.as_slice()?.to_vec(),
            type_id.as_slice()?.to_vec(),
            sign.as_slice()?.to_vec(),
        );
        let ports = Ports {
            input_index: input_index.as_slice()?.to_vec(),
            output_group: output_group.as_slice()?.to_vec(),
            output_scale: output_scale.as_slice()?.to_vec(),
            features,
            groups,
            actions,
        };
        let params = parse_params(params)?;
        let policy_mask = policy_mask.map(|a| a.as_slice().map(|v| v.to_vec())).transpose()?;
        let value_mask = value_mask.map(|a| a.as_slice().map(|v| v.to_vec())).transpose()?;
        let state = py
            .detach(|| {
                let graph = Graph::new(&indptr, &src, &type_id, &sign)?;
                let mut model = Model::with_rate(graph, ports, threads, Rate::new(rate_softness)?)?
                    .with_readout_mean_scale(readout_mean_scale)?;
                model = match (policy_mask, value_mask) {
                    (Some(policy), Some(value)) => model.with_head_mask(policy, value)?,
                    (None, None) => model,
                    _ => return Err("Both head mask arrays must be provided".into()),
                };
                model.validate(&params)?;
                let adam = Adam::new(&model);
                Ok::<_, String>(State {
                    model,
                    params,
                    adam,
                    prepared: None,
                    revision: 0,
                    last_update: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            state: Mutex::new(state),
        })
    }
    #[pyo3(signature = (input, steps, trace, prune=false))]
    fn infer<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
        steps: usize,
        trace: bool,
        prune: bool,
    ) -> PyResult<Infer<'py>> {
        if trace && prune {
            return Err(PyValueError::new_err("Readout pruning cannot return a full-state trace"));
        }
        let batch = input.shape()[1];
        let input = input.as_slice()?.to_vec();
        let (logits, values, states) = py
            .detach(|| {
                let mut state = self.state.lock().map_err(|e| e.to_string())?;
                if state.prepared.is_none() {
                    state.prepared = Some(fly_core::recurrent::prepare(
                        &state.model.graph,
                        &state.model.executor,
                        &state.params.core,
                        state.model.rate,
                    )?);
                }
                if trace {
                    let output = state.model.forward_prepared(
                        &state.params, &input, batch, steps, state.prepared.as_ref(),
                    )?;
                    Ok::<_, String>((output.logits, output.values, output.tape.states))
                } else if prune {
                    let output = state.model.predict_pruned_prepared(
                        &state.params, &input, batch, steps, state.prepared.as_ref(),
                    )?;
                    Ok((output.logits, output.values, Vec::new()))
                } else {
                    let output = state.model.predict_prepared(
                        &state.params, &input, batch, steps, state.prepared.as_ref(),
                    )?;
                    Ok((output.logits, output.values, Vec::new()))
                }
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            logits.into_pyarray(py),
            values.into_pyarray(py),
            states.into_iter().map(|v| v.into_pyarray(py)).collect(),
        ))
    }
    fn prediction_dependencies(&self, py: Python<'_>, steps: usize) -> PyResult<Vec<(usize, usize)>> {
        py.detach(|| {
            let state = self.state.lock().map_err(|error| error.to_string())?;
            state.model.prediction_dependencies(steps)
        }).map_err(PyValueError::new_err)
    }
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Arrays<'py>> {
        let state = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(export(py, &state.params))
    }
    /// Streaming readout with a revision token for an external objective.
    fn embedding<'py>(
        &self, py: Python<'py>, input: PyReadonlyArray2<'py, f32>, steps: usize,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, u64)> {
        let batch = input.shape()[1];
        let input = input.as_slice()?.to_vec();
        let (pooled, values, revision) = py.detach(|| {
            let mut state = self.state.lock().map_err(|e| e.to_string())?;
            if state.prepared.is_none() {
                state.prepared = Some(fly_core::recurrent::prepare(
                    &state.model.graph, &state.model.executor, &state.params.core, state.model.rate,
                )?);
            }
            let output = state.model.predict_prepared(
                &state.params, &input, batch, steps, state.prepared.as_ref(),
            )?;
            Ok::<_, String>((output.pooled, output.scores, state.revision))
        }).map_err(PyValueError::new_err)?;
        Ok((pooled.into_pyarray(py), values.into_pyarray(py), revision))
    }
    /// Recompute a tape, apply external readout cotangents, optionally update.
    /// No full-CNS tape crosses Python. Revisions reject intervening updates or restores.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(input, steps, cotangent, value_cotangent, revision, update=None, clip_mode="global"))]
    fn embedding_backward<'py>(
        &self, py: Python<'py>, input: PyReadonlyArray2<'py, f32>, steps: usize,
        cotangent: PyReadonlyArray2<'py, f32>, value_cotangent: PyReadonlyArray1<'py, f32>,
        revision: u64, update: Option<(f32, f32, Vec<f32>, f32)>,
        clip_mode: &str,
    ) -> PyResult<(Arrays<'py>, Option<(f64, u64)>)> {
        let clip_mode = clip_mode.parse::<ClipMode>().map_err(PyValueError::new_err)?;
        let batch = input.shape()[1];
        let cotangent_shape = cotangent.shape().to_vec();
        let (input, cotangent, value_cotangent) = (
            input.as_slice()?.to_vec(), cotangent.as_slice()?.to_vec(),
            value_cotangent.as_slice()?.to_vec(),
        );
        let (grad, result) = py.detach(|| {
            let mut state = self.state.lock().map_err(|e| e.to_string())?;
            if revision != state.revision { return Err("Stale embedding revision".into()); }
            if cotangent_shape != [state.model.ports.groups, batch] {
                return Err("Expected group-major embedding cotangent [G,B]".into());
            }
            let State { model, params, adam, prepared, revision, last_update } = &mut *state;
            let output = model.forward_prepared(params, &input, batch, steps, prepared.as_ref())?;
            let grad = model.backward_with_embedding(
                params, &input, &output, &vec![0.0; batch * model.ports.actions],
                &vec![0.0; batch], Some(&cotangent), Some(&value_cotangent),
            )?;
            if let Some((rate, clip, scales, epsilon)) = update {
                let stats = adam.update_with_clipping(model, params, &grad, rate, clip, &scales, epsilon, clip_mode)?;
                let norm = stats.norm;
                *last_update = Some(stats);
                *prepared = None;
                *revision += 1;
                Ok::<_, String>((None, Some((norm, adam.step))))
            } else {
                Ok((Some(grad), None))
            }
        }).map_err(PyValueError::new_err)?;
        Ok((grad.as_ref().map(|g| export(py, g)).unwrap_or_default(), result))
    }
    /// Read-only kernel timings on this graph and a real recurrent state.
    fn profile_sparse(
        &self,
        py: Python<'_>,
        input: PyReadonlyArray2<'_, f32>,
        steps: usize,
        repetitions: usize,
    ) -> PyResult<Vec<(&'static str, Vec<f64>)>> {
        if !(1..=100).contains(&repetitions) {
            return Err(PyValueError::new_err("Choose 1..100 repetitions"));
        }
        let batch = input.shape()[1];
        let input = input.as_slice()?.to_vec();
        py.detach(|| {
            let state = self.state.lock().map_err(|e| e.to_string())?;
            let model = &state.model;
            let output = model.forward(&state.params, &input, batch, steps)?;
            let rate: Vec<_> = output
                .tape
                .states
                .last()
                .unwrap()
                .iter()
                .map(|&x| model.rate.value(x))
                .collect();
            let cotangent: Vec<_> = rate.iter().map(|x| 0.01 + 0.001 * x).collect();
            let transpose_weights = model
                .executor
                .transpose_weights(&model.graph, &output.tape.weights);
            let mut edge = vec![0.0; model.graph.edges()];
            let mut records = Vec::new();
            for kernel in [
                "multiply",
                "parameter_validation",
                "transpose",
                "transpose_prepare",
                "transpose_prepared",
                "edge_vjp",
            ] {
                let mut seconds = Vec::new();
                for sample in 0..=repetitions {
                    edge.fill(0.0);
                    let start = std::time::Instant::now();
                    match kernel {
                        "parameter_validation" => {
                            model.validate(&state.params)?;
                        }
                        "multiply" => {
                            std::hint::black_box(model.executor.multiply(
                                &model.graph,
                                &output.tape.weights,
                                &rate,
                                batch,
                            ));
                        }
                        "transpose" => {
                            std::hint::black_box(model.executor.transpose(
                                &model.graph,
                                &output.tape.weights,
                                &cotangent,
                                batch,
                            ));
                        }
                        "transpose_prepare" => {
                            std::hint::black_box(
                                model
                                    .executor
                                    .transpose_weights(&model.graph, &output.tape.weights),
                            );
                        }
                        "transpose_prepared" => {
                            std::hint::black_box(model.executor.transpose_prepared(
                                &model.graph,
                                &transpose_weights,
                                &cotangent,
                                batch,
                            ));
                        }
                        _ => {
                            model.executor.edge_vjp(
                                &model.graph,
                                &rate,
                                &cotangent,
                                batch,
                                &mut edge,
                            );
                            std::hint::black_box(&edge);
                        }
                    }
                    if sample > 0 {
                        seconds.push(start.elapsed().as_secs_f64());
                    }
                }
                records.push((kernel, seconds));
            }
            Ok::<_, String>(records)
        })
        .map_err(PyValueError::new_err)
    }
    #[allow(clippy::too_many_arguments)]
    fn loss_and_grad<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
        steps: usize,
        legal: PyReadonlyArray2<'py, u8>,
        policy: PyReadonlyArray2<'py, f32>,
        value: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(f64, f64, Arrays<'py>)> {
        let batch = input.shape()[1];
        let (input, legal, policy, value) = (
            input.as_slice()?.to_vec(),
            legal.as_slice()?.to_vec(),
            policy.as_slice()?.to_vec(),
            value.as_slice()?.to_vec(),
        );
        let (pl, vl, grad) = py
            .detach(|| {
                let state = self.state.lock().map_err(|e| e.to_string())?;
                state.model.loss_and_grad(
                    &state.params,
                    &input,
                    batch,
                    steps,
                    Targets {
                        legal: &legal,
                        policy: &policy,
                        value: &value,
                    },
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((pl, vl, export(py, &grad)))
    }
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(input, steps, legal, policy, value, rate, clip, rate_scales=None, epsilon=1e-8, clip_mode="global"))]
    fn train_step(
        &self,
        py: Python<'_>,
        input: PyReadonlyArray2<'_, f32>,
        steps: usize,
        legal: PyReadonlyArray2<'_, u8>,
        policy: PyReadonlyArray2<'_, f32>,
        value: PyReadonlyArray1<'_, f32>,
        rate: f32,
        clip: f32,
        rate_scales: Option<Vec<f32>>,
        epsilon: f32,
        clip_mode: &str,
    ) -> PyResult<(f64, f64, f64, u64)> {
        let clip_mode = clip_mode.parse::<ClipMode>().map_err(PyValueError::new_err)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PyValueError::new_err("Adam epsilon must be finite and positive"));
        }
        let batch = input.shape()[1];
        let (input, legal, policy, value) = (
            input.as_slice()?.to_vec(),
            legal.as_slice()?.to_vec(),
            policy.as_slice()?.to_vec(),
            value.as_slice()?.to_vec(),
        );
        py.detach(|| {
            let mut state = self.state.lock().map_err(|e| e.to_string())?;
            let State {
                model,
                params,
                adam,
                prepared,
                revision,
                last_update,
            } = &mut *state;
            let (pl, vl, grad) = model.loss_and_grad(
                params,
                &input,
                batch,
                steps,
                Targets {
                    legal: &legal,
                    policy: &policy,
                    value: &value,
                },
            )?;
            let stats = adam.update_with_clipping(model, params, &grad, rate, clip,
                rate_scales.as_deref().unwrap_or(&[1.0; 9]), epsilon, clip_mode)?;
            let norm = stats.norm;
            *last_update = Some(stats);
            *prepared = None;
            *revision += 1;
            Ok::<_, String>((pl, vl, norm, adam.step))
        })
        .map_err(PyValueError::new_err)
    }
    fn update_statistics(&self) -> PyResult<Option<(Vec<f64>, Vec<f32>)>> {
        let state = self.state.lock().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.last_update.as_ref().map(|stats| (stats.group_norms.to_vec(), stats.clip_factors.to_vec())))
    }
    fn checkpoint<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Arrays<'py>, Arrays<'py>, Arrays<'py>, u64)> {
        let state = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            export(py, &state.params),
            export(py, &state.adam.first),
            export(py, &state.adam.second),
            state.adam.step,
        ))
    }
    fn restore(
        &self,
        py: Python<'_>,
        params: Vec<PyReadonlyArray1<'_, f32>>,
        first: Vec<PyReadonlyArray1<'_, f32>>,
        second: Vec<PyReadonlyArray1<'_, f32>>,
        step: u64,
    ) -> PyResult<()> {
        let (params, first, second) = (
            parse_params(params)?,
            parse_params(first)?,
            parse_params(second)?,
        );
        py.detach(|| {
            let mut state = self.state.lock().map_err(|e| e.to_string())?;
            for value in [&params, &first, &second] {
                state.model.validate(value)?;
            }
            if second.arrays().iter().any(|a| a.iter().any(|&x| x < 0.0)) {
                return Err("Negative second moment".into());
            }
            state.model.validate_head_zeros(&first)?;
            state.model.validate_head_zeros(&second)?;
            state.params = params;
            state.prepared = None;
            state.revision += 1;
            state.last_update = None;
            state.adam = Adam {
                first,
                second,
                step,
            };
            Ok::<_, String>(())
        })
        .map_err(PyValueError::new_err)
    }
}
