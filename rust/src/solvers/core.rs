//! One loop for LM, GN and the GN→LM hybrid.
//!
//! Every iteration factorises the column-scaled Jacobian once (QR→SVD, see [`crate::linalg`]);
//! LM retries with a larger damping reuse that factorisation. A trial is accepted iff its
//! residual sum of squares is finite and strictly smaller.

use nalgebra::{DMatrix, DVector};

use super::{Lsq, Mode, Status};
use crate::linalg::factor;
use crate::tracker::Stop;

#[derive(Debug, Clone, Copy)]
pub struct Opts {
    pub mode: Mode,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    pub tau: f64,
    pub max_halvings: usize,
    pub loss_floor: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts { mode: Mode::Lm, ftol: 1e-15, xtol: 1e-15, gtol: 1e-15, tau: 1e-3, max_halvings: 30, loss_floor: 1e-32 }
    }
}

pub fn solve<F: Lsq>(f: &mut F, x0: &DVector<f64>, o: &Opts) -> Status {
    solve_with_scratch::<F, false>(f, x0, o)
}

pub(crate) fn solve_with_scratch<F: Lsq, const SCALE: bool>(
    f: &mut F, x0: &DVector<f64>, o: &Opts,
) -> Status {
    match run::<F, SCALE>(f, x0, o) {
        Ok(s) => s,
        Err(Stop::Deadline) => Status::Deadline,
    }
}

fn run<F: Lsq, const SCALE: bool>(f: &mut F, x0: &DVector<f64>, o: &Opts) -> Result<Status, Stop> {
    let (m, n) = (f.m(), f.n());
    let mut x = x0.clone();
    let mut r = DVector::zeros(m);
    let mut sse = f.residual(&x, &mut r)?;
    if !sse.is_finite() {
        return Ok(Status::Failure("non-finite initial residual".into()));
    }
    f.accept();
    let floor = o.loss_floor / f.tracker().inv_scale();
    let mut jac = DMatrix::zeros(m, n);
    let mut js = DMatrix::zeros(m, n);
    let mut d: DVector<f64> = DVector::from_element(n, 0.0);
    let mut delta = DVector::zeros(n);
    let mut step = DVector::zeros(n);
    let mut xt = DVector::zeros(n);
    let mut rt = DVector::zeros(m);
    let mut gn = o.mode != Mode::Lm;
    let mut lam = -1.0; // initialised from the first factorisation that needs it
    let mut nu = 2.0;
    f.tracker().start_iterations();
    loop {
        if sse <= floor {
            return Ok(Status::Converged);
        }
        if !f.jacobian(&x, &mut jac)? {
            return Ok(Status::Failure("non-finite Jacobian".into()));
        }
        // MINPACK scaling: d_j = max over iterations of ‖J_j‖ (1 for zero columns)
        for j in 0..n {
            let cn = jac.column(j).norm();
            d[j] = d[j].max(if cn > 0.0 { cn } else { 1.0 });
            if SCALE {
                let mut col = js.column_mut(j);
                col.copy_from(&jac.column(j));
                col.unscale_mut(d[j]);
            } else {
                js.column_mut(j).copy_from(&(jac.column(j) / d[j]));
            }
        }
        // gradient (cosine) test on the scaled problem
        let rn = r.norm();
        let grad = js.tr_mul(&r);
        let gmax: f64 = (0..n)
            .map(|j| {
                let cn = js.column(j).norm();
                if cn > 0.0 { grad[j].abs() / (cn * rn) } else { 0.0_f64 }
            })
            .fold(0.0, f64::max);
        if gmax <= o.gtol {
            return Ok(Status::Converged);
        }
        let Some(fac) = factor(&js, &r, false) else {
            return Ok(Status::Failure("factorisation failed".into()));
        };
        let xs_norm = x.component_mul(&d).norm();
        let mut pred;
        let mut sse_t;
        if gn {
            pred = fac.step(0.0, &mut step);
            delta.copy_from(&step.component_div(&d));
            xt.copy_from(&(&x + &delta));
            sse_t = f.residual(&xt, &mut rt)?;
            let ok = sse_t.is_finite() && sse_t < sse;
            if !ok && o.mode == Mode::Gn {
                let mut alpha = 1.0;
                let mut found = false;
                for _ in 0..o.max_halvings {
                    alpha *= 0.5;
                    xt.copy_from(&(&x + alpha * &delta));
                    sse_t = f.residual(&xt, &mut rt)?;
                    if sse_t.is_finite() && sse_t < sse {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Ok(Status::Stalled);
                }
                pred *= alpha * (2.0 - alpha); // quadratic model along the GN direction
            } else if !ok {
                gn = false; // GnLm: hand over to LM on this same factorisation
                f.tracker().event("gn->lm");
            }
        } else {
            pred = 0.0;
            sse_t = f64::INFINITY;
        }
        if !gn {
            if lam < 0.0 {
                lam = o.tau * fac.smax * fac.smax;
            }
            loop {
                pred = fac.step(lam, &mut step);
                if step.norm() <= o.xtol * (o.xtol + xs_norm) {
                    return Ok(Status::Converged);
                }
                delta.copy_from(&step.component_div(&d));
                xt.copy_from(&(&x + &delta));
                sse_t = f.residual(&xt, &mut rt)?;
                if sse_t.is_finite() && sse_t < sse {
                    let rho = (sse - sse_t) / pred;
                    lam *= f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3));
                    nu = 2.0;
                    break;
                }
                lam *= nu;
                nu *= 2.0;
                if !lam.is_finite() || lam > 1e32 * fac.smax * fac.smax {
                    return Ok(Status::Stalled);
                }
            }
        }
        // accepted
        let actual = sse - sse_t;
        let small_f = actual <= o.ftol * sse && pred <= o.ftol * sse;
        let small_x = step.norm() <= o.xtol * (o.xtol + xs_norm);
        std::mem::swap(&mut x, &mut xt);
        std::mem::swap(&mut r, &mut rt);
        sse = sse_t;
        f.accept();
        f.tracker().mark_iteration();
        if small_f || small_x {
            return Ok(Status::Converged);
        }
    }
}
