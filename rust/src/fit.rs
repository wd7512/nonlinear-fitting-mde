//! The fit: variable projection + Levenberg–Marquardt.
//!
//! Structural degenerate cases: no eligible linear parameters → ordinary LM; all parameters
//! linear → one linear solve.

use nalgebra::DVector;

use crate::problem::Problem;
use crate::solvers::core::{solve, Opts};
use crate::solvers::full::FullLsq;
use crate::solvers::varpro::VarPro;
use crate::solvers::{Lsq, Status};
use crate::tracker::{Stop, Tracker};

/// Fit `prob` from `theta0` under `tr`'s deadline. Returns (status, label describing the path).
pub fn fit(prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, &'static str) {
    let opts = Opts::default();
    if !prob.lin.is_empty() && prob.sep.is_some() {
        let mut vp = VarPro::new(prob, tr);
        let beta0 = vp.beta_of(theta0);
        if beta0.is_empty() {
            // fully linear: the single projected solve is the exact answer
            let mut r = DVector::zeros(vp.m());
            return match vp.residual(&beta0, &mut r) {
                Ok(sse) if sse.is_finite() => (Status::Converged, "linear solve"),
                Ok(_) => (Status::Failure("non-finite linear solve".into()), "linear solve"),
                Err(Stop::Deadline) => (Status::Deadline, "linear solve"),
            };
        }
        (solve(&mut vp, &beta0, &opts), "varpro")
    } else {
        let mut full = FullLsq { prob, tr };
        let x0 = DVector::from_column_slice(theta0);
        (solve(&mut full, &x0, &opts), "no linear params")
    }
}
