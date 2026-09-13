use go_actors::observations::{self, Config};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};

type PyObservations<'py> = (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<bool>>, String);

/// Full-history replay into the exact actor feature contract, before each action.
#[pyfunction]
pub fn replay_features<'py>(
    py: Python<'py>,
    config_json: &str,
    actions: PyReadonlyArray1<'py, i32>,
    offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use go_actors::game::{Game, GameConfig};
    let config: GameConfig =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let actions = actions.as_slice()?.to_vec();
    let offsets = offsets.as_slice()?.to_vec();
    let result = py
        .detach(|| {
            if offsets.len() < 2
                || offsets[0] != 0
                || offsets.last() != Some(&(actions.len() as i64))
                || offsets.windows(2).any(|x| x[0] > x[1])
                || actions.len() > 1_000_000
            {
                return Err("Invalid packed feature replay offsets or row bound".to_string());
            }
            let probe = Game::new(config.clone())?;
            let row_width = probe.position.features(&probe.legal()).len();
            let elements = actions
                .len()
                .checked_mul(row_width)
                .filter(|&n| n <= 134_217_728)
                .ok_or("Feature replay exceeds the 512 MiB output bound")?;
            let mut features = Vec::with_capacity(elements);
            for pair in offsets.windows(2) {
                let mut game = Game::new(config.clone())?;
                for (ply, &action) in actions[pair[0] as usize..pair[1] as usize]
                    .iter()
                    .enumerate()
                {
                    if action < 0 {
                        return Err("Negative replay action".into());
                    }
                    features.extend(game.position.features(&game.legal()));
                    game.play(1 + (ply % 2) as u8, action as usize)?;
                }
            }
            Ok::<_, String>(features)
        })
        .map_err(PyValueError::new_err)?;
    Ok(result.into_pyarray(py))
}

/// One GIL-free call replays all packed episodes and returns pre-action rows.
#[pyfunction]
pub fn replay_observations<'py>(
    py: Python<'py>,
    config_json: &str,
    actions: PyReadonlyArray1<'py, i32>,
    offsets: PyReadonlyArray1<'py, i64>,
) -> PyResult<PyObservations<'py>> {
    let config: Config =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let actions = actions.as_slice()?.to_vec();
    let offsets = offsets.as_slice()?.to_vec();
    let result = py
        .detach(|| observations::replay(config, &actions, &offsets))
        .map_err(PyValueError::new_err)?;
    let outcomes = serde_json::to_string(&result.outcomes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        result.stones.into_pyarray(py),
        result.legal.into_pyarray(py),
        outcomes,
    ))
}

/// All actions are validated; only suffix observations cross the language boundary.
#[pyfunction]
pub fn replay_suffix_observations<'py>(
    py: Python<'py>,
    config_json: &str,
    actions: PyReadonlyArray1<'py, i32>,
    offsets: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
) -> PyResult<PyObservations<'py>> {
    let config: Config =
        serde_json::from_str(config_json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let actions = actions.as_slice()?.to_vec();
    let offsets = offsets.as_slice()?.to_vec();
    let starts = starts.as_slice()?.to_vec();
    let result = py
        .detach(|| observations::replay_suffix(config, &actions, &offsets, &starts))
        .map_err(PyValueError::new_err)?;
    let outcomes = serde_json::to_string(&result.outcomes)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        result.stones.into_pyarray(py),
        result.legal.into_pyarray(py),
        outcomes,
    ))
}
