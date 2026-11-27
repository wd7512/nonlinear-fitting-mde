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

/// Which route `fit` took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// Variable projection + LM: the normal route.
    VarPro,
    /// No eligible linear parameters: LM on all parameters.
    Full,
    /// Every parameter linear: one linear solve.
    Linear,
}

impl Path {
    /// Suffix shown after the fitter's name in reports; `None` for the normal route.
    pub fn label(self) -> Option<&'static str> {
        match self {
            Path::VarPro => None,
            Path::Full => Some("LM fallback"),
            Path::Linear => Some("linear solve"),
        }
    }
}

/// Fit `prob` from `theta0` under `tr`'s deadline. Returns the status and the route taken.
pub fn fit(prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, Path) {
    let opts = Opts::default();
    if !prob.lin.is_empty() && prob.sep.is_some() {
        let mut vp = VarPro::new(prob, tr);
        let beta0 = vp.beta_of(theta0);
        if beta0.is_empty() {
            // fully linear: the single projected solve is the exact answer
            let mut r = DVector::zeros(vp.m());
            return match vp.residual(&beta0, &mut r) {
                Ok(sse) if sse.is_finite() => {
                    vp.tracker().accept_step(true);
                    (Status::Converged, Path::Linear)
                }
                Ok(_) => (Status::Failure("non-finite linear solve".into()), Path::Linear),
                Err(Stop::Deadline) => (Status::Deadline, Path::Linear),
            };
        }
        (solve(&mut vp, &beta0, &opts), Path::VarPro)
    } else {
        let mut full = FullLsq { prob, tr };
        let x0 = DVector::from_column_slice(theta0);
        (solve(&mut full, &x0, &opts), Path::Full)
    }
}
