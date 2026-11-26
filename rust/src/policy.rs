//! Named fitting policies.
//!
//! `auto_v1` is an alias for `auto`, the accepted VarPro+LM incumbent:
//! no eligible linear parameters → ordinary LM; all parameters linear → one linear solve.
//! Challengers are tested against it before any routing rule is adopted.

use nalgebra::DVector;

use crate::problem::Problem;
use crate::solvers::core::{solve, solve_with_scratch, Opts};
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

pub const NAMES: &[&str] = &["auto", "auto_v1", "lm", "gn", "gn_lm", "varpro_lm", "varpro_gn", "varpro_gn_lm"];

impl Policy {
    pub fn parse(name: &str) -> Result<Policy, String> {
        let (projected, mode, auto) = match name {
            "auto" | "auto_v1" => (true, Mode::Lm, true),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_is_an_exact_alias_for_auto() {
        assert_eq!(Policy::parse("auto_v1").unwrap(), Policy::parse("auto").unwrap());
    }
}

/// Opt-in benchmark revisions, separate from the unchanged public `Policy` settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkPolicy {
    Existing(Policy),
    AutoV1_1,
    AutoV1_2,
    AutoV1_3,
}

impl BenchmarkPolicy {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "auto_v1.1" => Ok(Self::AutoV1_1),
            "auto_v1.2" => Ok(Self::AutoV1_2),
            "auto_v1.3" => Ok(Self::AutoV1_3),
            _ => Policy::parse(name).map(Self::Existing)
                .map_err(|e| format!("{e}; opt-in revisions: auto_v1.1 through auto_v1.3")),
        }
    }

    pub fn run(self, prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, &'static str) {
        match self {
            Self::Existing(policy) => run(policy, prob, theta0, tr),
            Self::AutoV1_1 => run_revision::<1>(AUTO, prob, theta0, tr),
            Self::AutoV1_2 => run_revision::<2>(AUTO, prob, theta0, tr),
            Self::AutoV1_3 => run_revision::<3>(AUTO, prob, theta0, tr),
        }
    }
}

const AUTO: Policy = Policy { projected: true, mode: Mode::Lm, auto: true };

/// Fit `prob` from `theta0` under `tr`'s deadline. Returns (status, label describing the path).
pub fn run(policy: Policy, prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, &'static str) {
    run_revision::<0>(policy, prob, theta0, tr)
}

fn run_revision<const REV: u8>(
    policy: Policy, prob: &mut Problem, theta0: &[f64], tr: &mut Tracker,
) -> (Status, &'static str) {
    let opts = Opts { mode: policy.mode, ..Opts::default() };
    let k = prob.lin.len();
    if policy.projected && k > 0 && prob.sep.is_some() {
        let mut vp = VarPro::new(prob, tr);
        if REV >= 1 {
            vp.reuse_basis();
        }
        if REV >= 2 {
            vp.reuse_theta();
        }
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
        let status = if REV >= 3 {
            solve_with_scratch::<_, true>(&mut vp, &beta0, &opts)
        } else {
            solve(&mut vp, &beta0, &opts)
        };
        (status, "varpro")
    } else {
        let mut full = FullLsq { prob, tr };
        let x0 = DVector::from_column_slice(theta0);
        let label = if policy.projected { "no linear params" } else { "full" };
        (solve(&mut full, &x0, &opts), label)
    }
}
