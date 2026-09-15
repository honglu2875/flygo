//! Adam with FP32 moments, FP64 global norm, clipping before moment updates.
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
        let norm = model.executor.pool.install(|| {
            grad.arrays()
                .iter()
                .map(|array| squared_norm(array))
                .sum::<f64>()
                .sqrt()
        });
        if !norm.is_finite() {
            return Err("Non-finite gradient norm".into());
        }
        let scale = (f64::from(clip) / norm.max(1e-30)).min(1.0) as f32;
        self.step += 1;
        let c1 = (1.0 - 0.9_f64.powf(self.step as f64)) as f32;
        let c2 = (1.0 - 0.999_f64.powf(self.step as f64)) as f32;
        for ((((parameter, g), first), second), &multiplier) in params
            .arrays_mut()
            .into_iter()
            .zip(grad.arrays())
            .zip(self.first.arrays_mut())
            .zip(self.second.arrays_mut())
            .zip(rate_scales)
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
        Ok(norm)
    }
}

#[cfg(test)]
mod tests {
    use super::squared_norm;
    use crate::Executor;

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
