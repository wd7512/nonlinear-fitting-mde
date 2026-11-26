//! Least-squares solvers over a generic residual function ([`Lsq`]).
//!
//! * [`core::solve`] — one loop for Levenberg–Marquardt, Gauss–Newton and the GN→LM hybrid
//! * [`full::FullLsq`] — the model's own residual and Jacobian
//! * [`varpro::VarPro`] — variable projection (Golub–Pereyra, Kaufman Jacobian): linear
//!   parameters are eliminated by a rank-revealing solve inside every residual evaluation

pub mod core;
pub mod full;
pub mod varpro;

use nalgebra::{DMatrix, DVector};

use crate::tracker::{Res, Tracker};

/// A residual vector r(x) ∈ ℝᵐ over n free variables, with its Jacobian.
pub trait Lsq {
    fn n(&self) -> usize;
    fn m(&self) -> usize;
    fn tracker(&mut self) -> &mut Tracker;
    /// Evaluate r(x) into `r` and record the trial; returns Σ r² (non-finite on overflow).
    fn residual(&mut self, x: &DVector<f64>, r: &mut DVector<f64>) -> Res<f64>;
    /// The last evaluated trial becomes the current iterate (VarPro keeps its projection).
    fn accept(&mut self) {}
    /// Jacobian at the current (last accepted) iterate; Ok(false) if non-finite.
    fn jacobian(&mut self, x: &DVector<f64>, j: &mut DMatrix<f64>) -> Res<bool>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Converged,
    Stalled,
    Deadline,
    Failure(String),
}

impl Status {
    pub fn as_str(&self) -> String {
        match self {
            Status::Converged => "converged".into(),
            Status::Stalled => "stalled".into(),
            Status::Deadline => "deadline".into(),
            Status::Failure(e) => format!("failure: {e}"),
        }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Status::Converged => "converged",
            Status::Stalled => "stalled",
            Status::Deadline => "deadline",
            Status::Failure(_) => "failure",
        }
    }
}

/// Inner iteration scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Levenberg–Marquardt (Nielsen damping update, MINPACK column scaling)
    Lm,
    /// Gauss–Newton (rank-truncated step + step halving)
    Gn,
    /// Gauss–Newton while full steps succeed; permanent switch to LM at the first failure
    GnLm,
}
