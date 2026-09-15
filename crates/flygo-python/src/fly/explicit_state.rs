//! Explicit chunk boundaries; the model never owns a game's recurrent state.
use super::{export, parse_params, Arrays, FlyModel, State};
use fly_core::optim::ClipMode;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

pub(super) type InferState<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Arrays<'py>,
    u64,
);

fn prepare(state: &mut State) -> Result<(), String> {
    if state.prepared.is_none() {
        state.prepared = Some(fly_core::recurrent::prepare(
            &state.model.graph, &state.model.executor, &state.params.core, state.model.rate,
        )?);
    }
    Ok(())
}

pub(super) fn infer<'py>(
    owner: &FlyModel, py: Python<'py>, input: PyReadonlyArray2<'py, f32>,
    initial_state: PyReadonlyArray2<'py, f32>, steps: usize, trace: bool,
) -> PyResult<InferState<'py>> {
    let batch = input.shape()[1];
    let shape = initial_state.shape().to_vec();
    let (input, initial) = (input.as_slice()?.to_vec(), initial_state.as_slice()?.to_vec());
    let (logits, values, next, states, revision) = py.detach(|| {
        let mut state = owner.state.lock().map_err(|e| e.to_string())?;
        if shape != [state.model.graph.neurons(), batch] {
            return Err("Expected node-major initial state [N,B]".into());
        }
        prepare(&mut state)?;
        let model = &state.model;
        let prepared = state.prepared.as_ref().unwrap();
        if trace {
            let output = model.forward_from_state_prepared(
                &state.params, &input, batch, steps, prepared, &initial,
            )?;
            let next = output.tape.states.last().unwrap().clone();
            Ok::<_, String>((output.logits, output.values, next, output.tape.states, state.revision))
        } else {
            let (output, next) = model.predict_from_state_prepared(
                &state.params, &input, batch, steps, prepared, &initial,
            )?;
            Ok((output.logits, output.values, next, Vec::new(), state.revision))
        }
    }).map_err(PyValueError::new_err)?;
    Ok((logits.into_pyarray(py), values.into_pyarray(py), next.into_pyarray(py),
        states.into_iter().map(|s| s.into_pyarray(py)).collect(), revision))
}

/// Recompute with unchanged parameters; no full differentiation tape crosses Python.
#[allow(clippy::too_many_arguments)]
pub(super) fn vjp<'py>(
    owner: &FlyModel, py: Python<'py>, input: PyReadonlyArray2<'py, f32>,
    initial_state: PyReadonlyArray2<'py, f32>, steps: usize,
    dlogits: PyReadonlyArray2<'py, f32>, dvalue: PyReadonlyArray1<'py, f32>,
    revision: u64, dnext_state: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<(Arrays<'py>, Bound<'py, PyArray1<f32>>)> {
    let batch = input.shape()[1];
    let initial_shape = initial_state.shape().to_vec();
    let logits_shape = dlogits.shape().to_vec();
    let next_shape = dnext_state.as_ref().map(|s| s.shape().to_vec());
    let next = dnext_state.map(|s| s.as_slice().map(|s| s.to_vec())).transpose()?;
    let (input, initial, logits, values) = (input.as_slice()?.to_vec(),
        initial_state.as_slice()?.to_vec(), dlogits.as_slice()?.to_vec(), dvalue.as_slice()?.to_vec());
    let (grad, dinitial) = py.detach(|| {
        let mut state = owner.state.lock().map_err(|e| e.to_string())?;
        if revision != state.revision { return Err("Stale recurrent-state revision".into()); }
        let expected = [state.model.graph.neurons(), batch];
        if initial_shape != expected || next_shape.as_ref().is_some_and(|s| *s != expected)
            || logits_shape != [batch, state.model.ports.actions]
        {
            return Err("Invalid recurrent state or cotangent shape".into());
        }
        prepare(&mut state)?;
        let output = state.model.forward_from_state_prepared(&state.params, &input, batch,
            steps, state.prepared.as_ref().unwrap(), &initial)?;
        state.model.backward_with_state(&state.params, &input, &output,
            &logits, &values, next.as_deref())
    }).map_err(PyValueError::new_err)?;
    Ok((export(py, &grad), dinitial.into_pyarray(py)))
}

/// Apply an accumulated window gradient only after all its backwards finish.
#[allow(clippy::too_many_arguments)]
pub(super) fn update(
    owner: &FlyModel, py: Python<'_>, gradients: Vec<PyReadonlyArray1<'_, f32>>, revision: u64,
    rate: f32, clip: f32, rate_scales: Vec<f32>, epsilon: f32, clip_mode: &str,
) -> PyResult<(f64, u64)> {
    let grad = parse_params(gradients)?;
    let mode = clip_mode.parse::<ClipMode>().map_err(PyValueError::new_err)?;
    py.detach(|| {
        let mut state = owner.state.lock().map_err(|e| e.to_string())?;
        if revision != state.revision { return Err("Stale recurrent-state revision".into()); }
        let State { model, params, adam, prepared, revision, last_update } = &mut *state;
        let stats = adam.update_with_clipping(model, params, &grad,
            rate, clip, &rate_scales, epsilon, mode)?;
        let norm = stats.norm;
        *last_update = Some(stats);
        *prepared = None;
        *revision += 1;
        Ok::<_, String>((norm, adam.step))
    }).map_err(PyValueError::new_err)
}
