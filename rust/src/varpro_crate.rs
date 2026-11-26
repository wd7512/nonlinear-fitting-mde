//! Baseline: the `varpro` crate (variable projection + Levenberg–Marquardt, Kaufman Jacobian)
//! on the same exported tapes, at its defaults.
//!
//! The crate fits y ≈ Φ(β) α, so a problem qualifies only with both linear and nonlinear
//! parameters and an offset c(β) that does not depend on β (checked at every evaluation).
//! The basis Φ comes from the separable tape; ∂Φ_k/∂β from the model Jacobian, which is affine
//! in α: ∂f/∂β at α = e_k minus at α = 0 (k + 1 Jacobian evaluations per iterate).
//! Like the library itself, it is credited only with the parameters it returns.
//!
//! The crate retries a step whose evaluation fails, so a stop (deadline, unsupported offset)
//! unwinds out of the solve instead (`resume_unwind`, which skips the panic hook).

use std::cell::RefCell;
use std::panic::AssertUnwindSafe;

use nalgebra::DMatrix as Mat35;
use nalgebra034::{DMatrix, DVector};
use varpro::model::SeparableNonlinearModel;
use varpro::problem::SeparableProblemBuilder;
use varpro::solvers::levmar::LevMarSolver;

use symbfitrs::problem::Problem;
use symbfitrs::solvers::Status;
use symbfitrs::tracker::Tracker;

pub const NAME: &str = "varpro_rs";

#[derive(Debug)]
struct Halt(String);

impl std::fmt::Display for Halt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Halt {}

const DEADLINE: &str = "deadline";
/// nalgebra's SVD, which the crate solves with, never returns on NaN or infinite input.
const NONFINITE: &str = "non-finite basis (the crate's SVD does not terminate on it)";

struct State<'a> {
    pb: &'a mut Problem,
    tr: &'a mut Tracker,
    theta: Vec<f64>,
    derivs: Option<Vec<DMatrix<f64>>>, // ∂Φ/∂β_j for the current β, built on first request
}

struct Model<'s, 'a> {
    s: &'s RefCell<State<'a>>,
    c0: Vec<f64>,
}

fn stop(why: &str) -> ! {
    std::panic::resume_unwind(Box::new(Halt(why.into())))
}

impl State<'_> {
    fn set_beta(&mut self, beta: &[f64]) {
        for (&i, &b) in self.pb.nonlin.iter().zip(beta) {
            self.theta[i] = b;
        }
        self.derivs = None;
    }
}

impl SeparableNonlinearModel for Model<'_, '_> {
    type ScalarType = f64;
    type Error = Halt;

    fn parameter_count(&self) -> usize {
        self.s.borrow().pb.nonlin.len()
    }

    fn base_function_count(&self) -> usize {
        self.s.borrow().pb.lin.len()
    }

    fn output_len(&self) -> usize {
        self.s.borrow().pb.m
    }

    fn set_params(&mut self, beta: DVector<f64>) -> Result<(), Halt> {
        self.s.borrow_mut().set_beta(beta.as_slice());
        Ok(())
    }

    fn params(&self) -> DVector<f64> {
        let s = self.s.borrow();
        DVector::from_iterator(s.pb.nonlin.len(), s.pb.nonlin.iter().map(|&i| s.theta[i]))
    }

    fn eval(&self) -> Result<DMatrix<f64>, Halt> {
        let mut s = self.s.borrow_mut();
        let State { pb, tr, theta, .. } = &mut *s;
        let (m, k) = (pb.m, pb.lin.len());
        let mut c = vec![0.0; m];
        let mut phi = Mat35::zeros(m, k);
        if tr.timed(|| pb.separable(theta, &mut c, &mut phi)).is_err() {
            stop(DEADLINE);
        }
        if c != self.c0 {
            stop("the offset depends on the nonlinear parameters");
        }
        if !phi.iter().all(|v| v.is_finite()) {
            stop(NONFINITE);
        }
        Ok(DMatrix::from_column_slice(m, k, phi.as_slice()))
    }

    fn eval_partial_deriv(&self, j: usize) -> Result<DMatrix<f64>, Halt> {
        let mut s = self.s.borrow_mut();
        if s.derivs.is_none() {
            let State { pb, tr, theta, .. } = &mut *s;
            let (m, k, q) = (pb.m, pb.lin.len(), pb.nonlin.len());
            let lin = pb.lin.clone();
            let mut th = theta.clone();
            let mut jac = |th: &[f64], out: &mut Mat35<f64>| tr.timed(|| pb.jacobian_nonlinear(th, out));
            for &i in &lin {
                th[i] = 0.0;
            }
            let mut base = Mat35::zeros(m, q);
            let mut cols = vec![Mat35::zeros(m, q); k];
            let mut ok = jac(&th, &mut base).is_ok();
            for (kk, &i) in lin.iter().enumerate() {
                th[i] = 1.0;
                ok = ok && jac(&th, &mut cols[kk]).is_ok();
                th[i] = 0.0;
            }
            if !ok {
                stop(DEADLINE);
            }
            if !base.iter().chain(cols.iter().flatten()).all(|v| v.is_finite()) {
                stop(NONFINITE);
            }
            // derivs[j][:, kk] = ∂Φ_kk/∂β_j
            let derivs = (0..q)
                .map(|jj| DMatrix::from_fn(m, k, |r, kk| cols[kk][(r, jj)] - base[(r, jj)]))
                .collect();
            s.derivs = Some(derivs);
        }
        Ok(s.derivs.as_ref().unwrap()[j].clone())
    }
}

/// Fit `pb` from `theta0`; records only the returned parameters, if they arrive in time.
pub fn fit(pb: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> Status {
    if pb.lin.is_empty() || pb.nonlin.is_empty() || pb.sep.is_none() {
        return Status::Failure("needs both linear and nonlinear parameters".into());
    }
    let (m, k) = (pb.m, pb.lin.len());
    let mut c0 = vec![0.0; m];
    let mut phi = Mat35::zeros(m, k);
    if tr.timed(|| pb.separable(theta0, &mut c0, &mut phi)).is_err() {
        return Status::Deadline;
    }
    let y = DVector::from_iterator(m, pb.y_eff.iter().zip(&c0).map(|(y, c)| y - c));
    let beta0: Vec<f64> = pb.nonlin.iter().map(|&i| theta0[i]).collect();
    let state = RefCell::new(State { pb, tr, theta: theta0.to_vec(), derivs: None });
    state.borrow_mut().set_beta(&beta0);
    let model = Model { s: &state, c0 };
    let problem = match SeparableProblemBuilder::new(model).observations(y).build() {
        Ok(p) => p,
        Err(e) => return Status::Failure(e.to_string()),
    };
    let solved = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let (res, converged) = match LevMarSolver::default().solve(problem) {
            Ok(r) => (r, true),
            Err(r) => (r, false),
        };
        let alpha = res.linear_coefficients().map(|a| a.iter().copied().collect::<Vec<_>>());
        (alpha, res.nonlinear_parameters(), converged)
    }));
    let State { pb, tr, .. } = state.into_inner();
    match solved {
        Err(e) => match e.downcast::<Halt>() {
            Ok(h) if h.0 == DEADLINE => Status::Deadline,
            Ok(h) => Status::Failure(h.0),
            // the crate's own panic (nalgebra: "Singular value was NaN"): its failure, not ours
            Err(e) => Status::Failure(match (e.downcast_ref::<&str>(), e.downcast_ref::<String>()) {
                (Some(m), _) => format!("panicked: {m}"),
                (_, Some(m)) => format!("panicked: {m}"),
                _ => "panicked".into(),
            }),
        },
        Ok((None, _, _)) => Status::Failure("no linear coefficients".into()),
        Ok((Some(alpha), beta, converged)) => {
            let mut theta = theta0.to_vec();
            for (&i, &a) in pb.lin.iter().zip(&alpha) {
                theta[i] = a;
            }
            for (&i, &b) in pb.nonlin.iter().zip(beta.iter()) {
                theta[i] = b;
            }
            if tr.check().is_err() {
                return Status::Deadline; // returned too late to count
            }
            let mut r = vec![0.0; m];
            let sse = pb.residual(&theta, &mut r);
            tr.record(&theta, sse);
            if !sse.is_finite() {
                Status::Failure("non-finite result".into())
            } else if converged {
                Status::Converged
            } else {
                Status::Stalled
            }
        }
    }
}
