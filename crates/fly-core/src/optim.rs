//! Adam with FP32 moments, deterministic FP64 norms and explicit clipping scope.
use crate::{Grad, Model, Params};
use rayon::prelude::*;

// Indexed chunks retain a fixed summation tree across thread counts and work
// stealing. A parallel floating sum over individual elements did not: identical
// models could report different last bits in the norm during exact recovery.
fn squared_norm(array: &[f32]) -> f64 {
    array.par_chunks(16_384)
        .map(|chunk| chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
        .collect::<Vec<_>>().into_iter().sum()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClipMode {
    #[default]
    Global,
    ParameterGroup,
}

impl std::str::FromStr for ClipMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "global" => Ok(Self::Global),
            "parameter-group" => Ok(Self::ParameterGroup),
            _ => Err("Unknown gradient clipping mode".into()),
        }
    }
}

/// Pre-update measurements in Params::arrays order. Factors include FP32 rounding.
#[derive(Clone, Debug)]
pub struct UpdateStats {
    pub norm: f64,
    pub group_norms: [f64; 9],
    pub clip_factors: [f32; 9],
}

pub struct Adam {
    pub first: Params,
    pub second: Params,
    pub step: u64,
}
impl Adam {
    pub fn new(model: &Model) -> Self {
        Self {
            first: Params::zeros(model),
            second: Params::zeros(model),
            step: 0,
        }
    }
    pub fn update(
        &mut self,
        model: &Model,
        params: &mut Params,
        grad: &Grad,
        rate: f32,
        clip: f32,
    ) -> Result<f64, String> {
        self.update_scaled(model, params, grad, rate, clip, &[1.0; 9])
    }
    /// Apply per-group learning-rate multipliers after common global clipping.
    /// A zero multiplier freezes parameters while continuing moment estimation.
    #[allow(clippy::too_many_arguments)]
    pub fn update_scaled(
        &mut self,
        model: &Model,
        params: &mut Params,
        grad: &Grad,
        rate: f32,
        clip: f32,
        rate_scales: &[f32],
    ) -> Result<f64, String> {
        self.update_with_epsilon(model, params, grad, rate, clip, rate_scales, 1e-8)
    }
    /// Epsilon is outside the square root, after second-moment correction.
    #[allow(clippy::too_many_arguments)]
    pub fn update_with_epsilon(
        &mut self,
        model: &Model,
        params: &mut Params,
        grad: &Grad,
        rate: f32,
        clip: f32,
        rate_scales: &[f32],
        epsilon: f32,
    ) -> Result<f64, String> {
        self.update_with_clipping(model, params, grad, rate, clip, rate_scales, epsilon, ClipMode::Global)
            .map(|stats| stats.norm)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_with_clipping(
        &mut self,
        model: &Model,
        params: &mut Params,
        grad: &Grad,
        rate: f32,
        clip: f32,
        rate_scales: &[f32],
        epsilon: f32,
        clip_mode: ClipMode,
    ) -> Result<UpdateStats, String> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Adam epsilon must be finite and positive".into());
        }
        if !rate.is_finite() || rate <= 0.0 || !clip.is_finite() || clip <= 0.0 {
            return Err("Invalid optimizer settings".into());
        }
        if rate_scales.len() != 9
            || rate_scales
                .iter()
                .any(|&x| !x.is_finite() || x < 0.0 || !(rate * x).is_finite())
        {
            return Err("Invalid parameter-group learning-rate multipliers".into());
        }
        model.validate(params)?;
        model.validate(grad)?;
        model.validate_head_zeros(grad)?;
        model.validate_head_zeros(&self.first)?;
        model.validate_head_zeros(&self.second)?;
        let squared = model.executor.pool.install(|| grad.arrays().map(squared_norm));
        // Preserve the legacy group order and reduction tree for global mode.
        let norm = squared.iter().sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("Non-finite gradient norm".into());
        }
        let group_norms = squared.map(f64::sqrt);
        let factor = |norm: f64| (f64::from(clip) / norm.max(1e-30)).min(1.0) as f32;
        let clip_factors = match clip_mode {
            ClipMode::Global => [factor(norm); 9],
            ClipMode::ParameterGroup => group_norms.map(factor),
        };
        self.step += 1;
        let c1 = (1.0 - 0.9_f64.powf(self.step as f64)) as f32;
        let c2 = (1.0 - 0.999_f64.powf(self.step as f64)) as f32;
        for (((((parameter, g), first), second), &multiplier), &scale) in params
            .arrays_mut()
            .into_iter()
            .zip(grad.arrays())
            .zip(self.first.arrays_mut())
            .zip(self.second.arrays_mut())
            .zip(rate_scales)
            .zip(&clip_factors)
        {
            let group_rate = rate * multiplier;
            model.executor.pool.install(|| {
                parameter
                    .par_iter_mut()
                    .zip(g)
                    .zip(first.par_iter_mut())
                    .zip(second.par_iter_mut())
                    .for_each(|(((p, &g), m), v)| {
                        let g = g * scale;
                        *m = 0.9 * *m + 0.1 * g;
                        *v = 0.999 * *v + 0.001 * g * g;
                        if group_rate > 0.0 {
                            *p -= group_rate * (*m / c1) / ((*v / c2).sqrt() + epsilon);
                        }
                    })
            });
        }
        Ok(UpdateStats { norm, group_norms, clip_factors })
    }
}

#[cfg(test)]
mod tests {
    use super::{squared_norm, Adam, ClipMode};
    use crate::{Executor, Graph, Model, Params, Ports};

    fn model() -> Model {
        let graph = Graph::new(&[0, 0, 1], &[0], &[0, 0], &[1., 1.]).unwrap();
        let ports = Ports { input_index: vec![0, -1], output_group: vec![-1, 0],
            output_scale: vec![0., 1.], features: 1, groups: 1, actions: 2 };
        Model::new(graph, ports, 2).unwrap()
    }

    #[test]
    fn large_bias_cannot_clip_another_group_in_parameter_group_mode() {
        let model = model();
        let mut grad = Params::zeros(&model);
        grad.core.bias[0] = 1000.;
        grad.core.edge[0] = 1e-6;
        for mode in [ClipMode::Global, ClipMode::ParameterGroup] {
            let mut params = Params::zeros(&model);
            let mut adam = Adam::new(&model);
            let stats = adam.update_with_clipping(&model, &mut params, &grad, 0.03, 1., &[1.; 9], 1e-6, mode).unwrap();
            let factor = if mode == ClipMode::Global { 0.001_f32 } else { 1. };
            assert_eq!(stats.clip_factors[0], factor);
            assert_eq!(stats.clip_factors[2], 0.001);
            let clipped = f64::from(grad.core.edge[0]) * f64::from(factor);
            let expected = -0.03 * clipped / (clipped.abs() + 1e-6);
            assert!((f64::from(params.core.edge[0]) - expected).abs() < 2e-8);
            assert!((f64::from(adam.first.core.edge[0]) - 0.1 * clipped).abs() < 1e-13);
            assert!((f64::from(adam.second.core.edge[0]) - 0.001 * clipped * clipped).abs() < 1e-21);
            assert_eq!(params.input_gain[0], 0.);
            assert_eq!(adam.first.input_gain[0], 0.);
            assert_eq!(adam.second.input_gain[0], 0.);
        }
    }

    #[test]
    fn group_threshold_is_not_inferred_from_global_norm() {
        let model = model();
        let mut params = Params::zeros(&model);
        let mut grad = Params::zeros(&model);
        grad.core.bias[0] = 0.8;
        grad.input_gain[0] = 0.8;
        let mut adam = Adam::new(&model);
        let stats = adam.update_with_clipping(&model, &mut params, &grad, 0.03, 1., &[1.; 9], 1e-6, ClipMode::ParameterGroup).unwrap();
        assert!(stats.norm > 1.);
        assert!(stats.group_norms.iter().all(|&x| x < 1.));
        assert_eq!(stats.clip_factors, [1.; 9]);
        let zero = Params::zeros(&model);
        let stats = Adam::new(&model).update_with_clipping(&model, &mut params, &zero, 0.03, 1., &[1.; 9], 1e-6, ClipMode::ParameterGroup).unwrap();
        assert_eq!(stats.norm, 0.);
        assert_eq!(stats.clip_factors, [1.; 9]);
    }

    #[test]
    fn norm_is_reproducible_across_worker_counts() {
        let values: Vec<f32> = (0..200_003).map(|i| {
            ((i % 113) as f32 - 56.0) / 127.0 * if i % 23 == 0 { 1e4 } else { 1e-4 }
        }).collect();
        let reference: f64 = values.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let mut bits = None;
        for threads in [1, 2, 7] {
            let executor = Executor::new(threads).unwrap();
            for _ in 0..4 {
                let actual = executor.pool.install(|| squared_norm(&values));
                assert!((actual - reference).abs() <= 1e-12 * reference);
                assert_eq!(*bits.get_or_insert(actual.to_bits()), actual.to_bits());
            }
        }
    }
}
