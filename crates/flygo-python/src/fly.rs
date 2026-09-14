//! Thin NumPy boundary for the owned Rust model and optimizer.
use fly_core::{CoreParams, Graph, Model, Params, Ports, Rate, Targets, optim::Adam};
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
                       features, groups, actions, threads, params, rate_softness=0.0))]
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
        let state = py
            .detach(|| {
                let graph = Graph::new(&indptr, &src, &type_id, &sign)?;
                let model = Model::with_rate(graph, ports, threads, Rate::new(rate_softness)?)?;
                model.validate(&params)?;
                let adam = Adam::new(&model);
                Ok::<_, String>(State {
                    model,
                    params,
                    adam,
                    prepared: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            state: Mutex::new(state),
        })
    }
    fn infer<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, f32>,
        steps: usize,
        trace: bool,
    ) -> PyResult<Infer<'py>> {
        let batch = input.shape()[1];
        let input = input.as_slice()?.to_vec();
        let output = py
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
                state.model.forward_prepared(
                    &state.params,
                    &input,
                    batch,
                    steps,
                    state.prepared.as_ref(),
                )
            })
            .map_err(PyValueError::new_err)?;
        let states = if trace {
            output
                .tape
                .states
                .into_iter()
                .map(|v| v.into_pyarray(py))
                .collect()
        } else {
            Vec::new()
        };
        Ok((
            output.logits.into_pyarray(py),
            output.values.into_pyarray(py),
            states,
        ))
    }
    fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Arrays<'py>> {
        let state = self
            .state
            .lock()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(export(py, &state.params))
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
    #[pyo3(signature=(input, steps, legal, policy, value, rate, clip, rate_scales=None))]
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
    ) -> PyResult<(f64, f64, f64, u64)> {
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
            let norm = match rate_scales {
                Some(ref scales) => adam.update_scaled(model, params, &grad, rate, clip, scales)?,
                None => adam.update(model, params, &grad, rate, clip)?,
            };
            *prepared = None;
            Ok::<_, String>((pl, vl, norm, adam.step))
        })
        .map_err(PyValueError::new_err)
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
            state.params = params;
            state.prepared = None;
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
