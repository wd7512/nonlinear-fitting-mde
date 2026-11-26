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
    use crate::tape::Program;
    use serde_json::json;

    #[test]
    fn v1_is_an_exact_alias_for_auto() {
        assert_eq!(Policy::parse("auto_v1").unwrap(), Policy::parse("auto").unwrap());
    }

    #[test]
    fn revisions_keep_the_same_loss_history_and_parameters() {
        // a * exp(b * x) + c, with a and c projected. Constant x makes the
        // two basis columns rank deficient and exercises the fallback path.
        let full = json!({
            "ops": [
                ["param", 0], ["param", 1], ["param", 2], ["var", 0], ["mul", 2, 3],
                ["exp", 4], ["mul", 0, 5], ["add", 6, 1], ["mul", 6, 3], ["const", 1.0]
            ],
            "outputs": [[7], [5], [9], [8]]
        });
        let mut residual = full.clone();
        residual["outputs"] = json!([[7]]);
        let sep = json!({
            "ops": [
                ["var", 0], ["const", 0.0], ["param", 2], ["mul", 0, 2], ["exp", 3], ["const", 1.0]
            ],
            "outputs": [[1], [4], [5]]
        });
        let programs = [residual, full, sep].map(|v| Program::from_json(&v).unwrap());
        for x in [[0.0, 0.5, 1.0, 2.0, 3.0], [0.5; 5]] {
            let case = json!({
                "p": 3, "m": 5, "x": [x], "y": [4.0, 3.6, 3.3, 2.9, 2.6],
                "denom": 1.0, "linear_idx": [0, 1], "prog_f": 0, "prog_fj": 1, "prog_sep": 2,
                "starts": [["initial", [1.0, 1.0, -0.1]]]
            });
            let fit = |version| {
                let mut prob = Problem::from_json(&case, &programs).unwrap();
                let mut tracker = Tracker::new(1000.0, &[], prob.inv_scale);
                let (status, path) = BenchmarkPolicy::parse(version)
                    .unwrap()
                    .run(&mut prob, &[1.0, 1.0, -0.1], &mut tracker);
                assert_eq!(path, "varpro");
                assert_ne!(status, Status::Deadline);
                (status, tracker.best_theta, tracker.hist_loss, tracker.nfev, tracker.njev)
            };
            let expected = fit("auto_v1");
            assert!(expected.4 > 0);
            for version in ["auto_v1.1", "auto_v1.2", "auto_v1.3", "auto_v1.4", "auto_v1.5"] {
                assert_eq!(fit(version), expected, "{version}");
            }
        }
    }
}

/// Opt-in benchmark revisions, separate from the unchanged public `Policy` settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkPolicy {
    Existing(Policy),
    AutoV1_1,
    AutoV1_2,
    AutoV1_3,
    AutoV1_4,
    AutoV1_5,
}

impl BenchmarkPolicy {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "auto_v1.1" => Ok(Self::AutoV1_1),
            "auto_v1.2" => Ok(Self::AutoV1_2),
            "auto_v1.3" => Ok(Self::AutoV1_3),
            "auto_v1.4" => Ok(Self::AutoV1_4),
            "auto_v1.5" => Ok(Self::AutoV1_5),
            _ => Policy::parse(name).map(Self::Existing)
                .map_err(|e| format!("{e}; opt-in revisions: auto_v1.1 through auto_v1.5")),
        }
    }

    pub fn run(self, prob: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, &'static str) {
        match self {
            Self::Existing(policy) => run(policy, prob, theta0, tr),
            Self::AutoV1_1 => run_revision::<1>(AUTO, prob, theta0, tr),
            Self::AutoV1_2 => run_revision::<2>(AUTO, prob, theta0, tr),
            Self::AutoV1_3 => run_revision::<3>(AUTO, prob, theta0, tr),
            Self::AutoV1_4 => run_revision::<4>(AUTO, prob, theta0, tr),
            Self::AutoV1_5 => run_revision::<5>(AUTO, prob, theta0, tr),
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
        let status = if REV >= 5 {
            solve_with_scratch::<_, true, true, true>(&mut vp, &beta0, &opts)
        } else if REV >= 4 {
            solve_with_scratch::<_, true, true, false>(&mut vp, &beta0, &opts)
        } else if REV >= 3 {
            solve_with_scratch::<_, true, false, false>(&mut vp, &beta0, &opts)
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
