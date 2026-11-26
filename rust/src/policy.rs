//! Named fitting policies.
//!
//! `auto` is the accepted incumbent. **V0 = VarPro+LM** with its structural degenerate cases:
//! no eligible linear parameters → ordinary LM; all parameters linear → one linear solve.
//! Challengers are tested against it before any routing rule is adopted.

use nalgebra::DVector;

use crate::problem::Problem;
use crate::solvers::core::{solve, Opts};
use crate::solvers::full::FullLsq;
use crate::solvers::varpro::VarPro;
use crate::solvers::{Lsq, Mode, Status};
use crate::tracker::{Stop, Tracker};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub projected: bool,
    pub mode: Mode,
    pub auto: bool,
}

pub const NAMES: &[&str] = &["auto", "lm", "gn", "gn_lm", "varpro_lm", "varpro_gn", "varpro_gn_lm"];

impl Policy {
    pub fn parse(name: &str) -> Result<Policy, String> {
        let (projected, mode, auto) = match name {
            "auto" => (true, Mode::Lm, true),
            "lm" => (false, Mode::Lm, false),
            "gn" => (false, Mode::Gn, false),
            "gn_lm" => (false, Mode::GnLm, false),
            "varpro_lm" => (true, Mode::Lm, false),
            "varpro_gn" => (true, Mode::Gn, false),
            "varpro_gn_lm" => (true, Mode::GnLm, false),
            _ => return Err(format!("unknown policy {name:?}; choose from {NAMES:?}")),
        };
        Ok(Policy { projected, mode, auto })
    }
}

/// Fit `prob` from `theta0` under `tr`'s deadline. Returns (status, label describing the path).
pub fn run(policy: Policy, prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, &'static str) {
    let opts = Opts { mode: policy.mode, ..Opts::default() };
    let k = prob.lin.len();
    if policy.projected && k > 0 && prob.sep.is_some() {
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
        let label = if policy.projected { "no linear params" } else { "full" };
        (solve(&mut full, &x0, &opts), label)
    }
}
