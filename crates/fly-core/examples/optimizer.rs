//! Optimizer-only CPU timing with synthetic gradients and caller-specified shapes.
//! No Go labels, recurrence, checkpoint I/O, or biological graph claim.
use fly_core::optim::{Adam, ClipMode};
use fly_core::{Graph, Model, Params, Ports};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1)
        .map(|s| s.parse().expect("Use positive integer dimensions")).collect();
    assert_eq!(args.len(), 8, "threads neurons edges types features groups actions repetitions");
    let [threads, n, edges, types, features, groups, actions, repetitions] = args[..] else { unreachable!() };
    assert!(args.iter().all(|&x| x > 0) && types <= n && edges / n < n && repetitions >= 4);
    let indptr: Vec<i32> = (0..=n).map(|i| (i * edges / n).try_into().unwrap()).collect();
    let src: Vec<i32> = indptr.windows(2).flat_map(|p| 0..p[1] - p[0]).collect();
    let type_id: Vec<i32> = (0..n).map(|i| (i % types) as i32).collect();
    let graph = Graph::new(&indptr, &src, &type_id, &vec![1.; n]).unwrap();
    let ports = Ports { input_index: vec![-1; n], output_group: vec![-1; n],
        output_scale: vec![0.; n], features, groups, actions };
    let model = Model::new(graph, ports, threads).unwrap();
    let shapes: Vec<_> = Params::zeros(&model).arrays().map(|a| a.len()).to_vec();
    println!("{{\"kind\":\"configuration\",\"threads\":{threads},\"neurons\":{n},\"edges\":{edges},\"shapes\":{shapes:?},\"repetitions\":{repetitions},\"warmup\":4}}");
    for (profile, amplitudes) in [
        ("bias_dominated", [1e-7, 1e-5, 1., 1e-5, 1e-7, 1e-5, 1e-3, 1e-5, 1e-3]),
        ("below_threshold", [1e-7; 9]),
    ] {
        let mut gradient = Params::zeros(&model);
        for (array, amplitude) in gradient.arrays_mut().into_iter().zip(amplitudes) {
            for (i, value) in array.iter_mut().enumerate() {
                *value = ((i % 23) as f32 - 11.) * amplitude / 11.;
            }
        }
        let mut parameters = [Params::zeros(&model), Params::zeros(&model)];
        let mut optimizers = [Adam::new(&model), Adam::new(&model)];
        let modes = [ClipMode::Global, ClipMode::ParameterGroup];
        let mut rates = [1.; 9]; rates[2] = 0.01;
        let mut update = |index: usize| {
            optimizers[index].update_with_clipping(&model, &mut parameters[index],
                &gradient, 0.03, 1., &rates, 1e-6, modes[index]).unwrap()
        };
        for _ in 0..4 { for index in 0..2 { black_box(update(index)); } }
        for repetition in 0..repetitions {
            // Alternate AB and BA, retaining each pair for latency uncertainty.
            let order = if repetition % 2 == 0 { [0, 1] } else { [1, 0] };
            for index in order {
                let start = Instant::now();
                let stats = black_box(update(index));
                let seconds = start.elapsed().as_secs_f64();
                let mode = if index == 0 { "global" } else { "parameter-group" };
                println!("{{\"kind\":\"timing\",\"profile\":\"{profile}\",\"mode\":\"{mode}\",\"pair\":{repetition},\"seconds\":{seconds},\"norm\":{},\"factors\":{:?}}}", stats.norm, stats.clip_factors);
            }
        }
        drop(update);
        for (params, optimizer) in parameters.iter().zip(&optimizers) {
            assert_eq!(optimizer.step, (repetitions + 4) as u64);
            model.validate(params).unwrap();
            model.validate(&optimizer.first).unwrap();
            model.validate(&optimizer.second).unwrap();
        }
    }
}
