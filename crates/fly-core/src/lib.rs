//! Fixed directed graph, rate recurrence and explicit CPU derivatives.
//! Topology is owned by Graph and never enters an optimizer parameter vector.
pub mod graph;
pub mod model;
pub mod optim;
pub mod rate;
pub mod recurrent;
pub mod sparse;

pub use graph::Graph;
pub use model::{Grad, Model, Output, Params, Ports, Prediction, Targets};
pub use rate::Rate;
pub use recurrent::{CoreGrad, CoreParams, Tape};
pub use sparse::Executor;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
