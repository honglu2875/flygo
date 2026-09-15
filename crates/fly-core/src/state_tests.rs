//! Explicit-state recurrence must compose in both directions across chunk boundaries.
use super::*;

fn fixture(softness: f32) -> (Graph, Executor, CoreParams, Prepared, Vec<f32>, Vec<f32>) {
    let graph = Graph::new(
        &[0, 2, 4, 5, 7], &[1, 3, 0, 2, 1, 0, 2], &[0, 1, 0, 1], &[1., -1., 1., -1.],
    ).unwrap();
    let executor = Executor::new(2).unwrap();
    let params = CoreParams { edge: vec![-1.2; 7], leak: vec![-0.3, 0.4], bias: vec![0.1, -0.2] };
    let prepared = prepare(&graph, &executor, &params, Rate::new(softness).unwrap()).unwrap();
    let drive = vec![0.2, -0.05, 0.15, 0.3, -0.1, 0.2, 0.05, -0.05];
    let initial = vec![0.4, -0.3, -0.2, 0.6, 0.15, 0.25, -0.4, -0.2];
    (graph, executor, params, prepared, drive, initial)
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 3e-6 + 3e-5 * b.abs(), "{a} != {b}");
    }
}

#[test]
fn explicit_state_matches_reset_and_streaming_chunks() {
    for softness in [0.0, 0.07] {
        let (g, e, p, ready, drive, initial) = fixture(softness);
        let reset = forward_prepared(&g, &e, &p, &drive, 2, 7, &ready).unwrap();
        let explicit = forward_from_state_prepared(&g, &e, &p, &drive, 2, 7, &ready, &[0.01; 8]).unwrap();
        assert_eq!(reset.states, explicit.states);
        let whole = forward_from_state_prepared(&g, &e, &p, &drive, 2, 7, &ready, &initial).unwrap();
        assert_ne!(whole.states.last(), reset.states.last()); // Cannot reuse the fixed reset message.
        let first = predict_from_state_prepared(&g, &e, &p, &drive, 2, 3, &ready, &initial).unwrap();
        let second = predict_from_state_prepared(&g, &e, &p, &drive, 2, 4, &ready, &first).unwrap();
        assert_eq!(first, whole.states[3]);
        assert_eq!(second, *whole.states.last().unwrap());
        // An unrelated call has no hidden carry state and does not mutate inputs.
        assert_eq!(initial, whole.states[0]);
        assert_eq!(reset.states, forward_prepared(&g, &e, &p, &drive, 2, 7, &ready).unwrap().states);
    }
}

#[test]
fn initial_state_cotangent_matches_independent_finite_differences() {
    for softness in [0.0, 0.07] {
        let (g, e, p, ready, drive, initial) = fixture(softness);
        let terminal = [0.1, -0.3, 0.2, 0., -0.7, 0.4, 0.1, 0.2];
        let tape = forward_from_state_prepared(&g, &e, &p, &drive, 2, 4, &ready, &initial).unwrap();
        let (_, _, gradient) = backward_with_state(&g, &e, &p, &tape, &terminal).unwrap();
        let objective = |state: &[f32]| -> f64 {
            let final_state = predict_from_state_prepared(&g, &e, &p, &drive, 2, 4, &ready, state).unwrap();
            final_state.iter().zip(terminal).map(|(&a, b)| f64::from(a) * f64::from(b)).sum()
        };
        for i in 0..initial.len() {
            let mut plus = initial.clone(); let mut minus = initial.clone();
            plus[i] += 0.001; minus[i] -= 0.001;
            let finite = (objective(&plus) - objective(&minus)) / f64::from(plus[i] - minus[i]);
            assert!((f64::from(gradient[i]) - finite).abs() < 2e-5 + 3e-3 * finite.abs(),
                    "softness={softness}, state[{i}]: {} != {finite}", gradient[i]);
        }
    }
}

#[test]
fn chunk_cotangents_compose_without_detaching_or_changing_legacy_gradients() {
    for softness in [0.0, 0.07] {
        let (g, e, p, ready, drive, initial) = fixture(softness);
        let terminal = [0.1, -0.3, 0.2, 0., -0.7, 0.4, 0.1, 0.2];
        let whole = forward_from_state_prepared(&g, &e, &p, &drive, 2, 7, &ready, &initial).unwrap();
        let first = forward_from_state_prepared(&g, &e, &p, &drive, 2, 3, &ready, &initial).unwrap();
        let second = forward_from_state_prepared(&g, &e, &p, &drive, 2, 4, &ready, first.states.last().unwrap()).unwrap();
        let (all, all_drive, all_initial) = backward_with_state(&g, &e, &p, &whole, &terminal).unwrap();
        let (tail, tail_drive, boundary) = backward_with_state(&g, &e, &p, &second, &terminal).unwrap();
        let (head, head_drive, start) = backward_with_state(&g, &e, &p, &first, &boundary).unwrap();
        close(&start, &all_initial);
        for (a, b, expected) in [(&head.edge, &tail.edge, &all.edge), (&head.leak, &tail.leak, &all.leak),
                                  (&head.bias, &tail.bias, &all.bias), (&head_drive, &tail_drive, &all_drive)] {
            let sum: Vec<f32> = a.iter().zip(b).map(|(&a, &b)| a + b).collect();
            close(&sum, expected);
        }
        let (legacy, legacy_drive) = backward(&g, &e, &p, &whole, &terminal).unwrap();
        assert_eq!(legacy.edge, all.edge); assert_eq!(legacy.leak, all.leak);
        assert_eq!(legacy.bias, all.bias); assert_eq!(legacy_drive, all_drive);
    }
}

#[test]
fn malformed_initial_state_or_cotangent_is_rejected() {
    let (g, e, p, ready, drive, initial) = fixture(0.0);
    for invalid in [vec![0.; 7], vec![f32::NAN; 8], vec![f32::INFINITY; 8]] {
        assert!(forward_from_state_prepared(&g, &e, &p, &drive, 2, 3, &ready, &invalid).is_err());
        assert!(predict_from_state_prepared(&g, &e, &p, &drive, 2, 3, &ready, &invalid).is_err());
    }
    let tape = forward_from_state_prepared(&g, &e, &p, &drive, 2, 3, &ready, &initial).unwrap();
    assert!(backward_with_state(&g, &e, &p, &tape, &[f32::NAN; 8]).is_err());
    assert!(backward_with_state(&g, &e, &p, &tape, &[0.; 7]).is_err());
}
