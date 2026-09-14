//! Adam with FP32 moments, FP64 global norm, clipping before moment updates.
use crate::{Grad, Model, Params};
use rayon::prelude::*;

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
        let norm = model.executor.pool.install(|| {
            grad.arrays()
                .iter()
                .map(|array| {
                    array
                        .par_iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                })
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
                            *p -= group_rate * (*m / c1) / ((*v / c2).sqrt() + 1e-8);
                        }
                    })
            });
        }
        Ok(norm)
    }
}
