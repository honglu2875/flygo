//! Nonnegative firing rates. Softness is fixed model configuration, not a parameter.

#[derive(Clone, Copy, Default)]
pub struct Rate {
    softness: f32,
}

impl Rate {
    pub fn new(softness: f32) -> Result<Self, String> {
        if !softness.is_finite() || softness < 0.0 {
            return Err("Rate softness must be finite and nonnegative".into());
        }
        Ok(Self { softness })
    }

    #[inline]
    pub fn value(self, voltage: f32) -> f32 {
        if self.softness == 0.0 {
            voltage.max(0.0)
        } else {
            // Unlike s*softplus(v/s), this remains finite when v/s overflows.
            voltage.max(0.0)
                + self.softness * (-voltage.abs() / self.softness).exp().ln_1p()
        }
    }

    #[inline]
    pub fn pullback(self, voltage: f32, cotangent: f32) -> f32 {
        if self.softness == 0.0 {
            // Preserve the baseline's exact zero and inactive-branch behavior.
            if voltage > 0.0 { cotangent } else { 0.0 }
        } else {
            crate::sigmoid(voltage / self.softness) * cotangent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_rate_has_the_declared_derivative_and_finite_extremes() {
        let rate = Rate::new(0.05).unwrap();
        for voltage in [-0.3, -0.05, 0.0, 0.05, 0.3] {
            let difference = (rate.value(voltage + 0.0001) - rate.value(voltage - 0.0001)) / 0.0002;
            assert!((difference - rate.pullback(voltage, 1.0)).abs() < 0.0002);
            assert!(rate.value(voltage) >= 0.0);
        }
        assert_eq!(rate.pullback(0.0, 1.0), 0.5);
        assert_eq!(rate.value(f32::MAX), f32::MAX);
        assert_eq!(rate.value(-f32::MAX), 0.0);
        assert_eq!(rate.pullback(f32::MAX, 1.0), 1.0);
        assert_eq!(rate.pullback(-f32::MAX, 1.0), 0.0);
        for bad in [-0.1, f32::NAN, f32::INFINITY] {
            assert!(Rate::new(bad).is_err());
        }
    }
}
