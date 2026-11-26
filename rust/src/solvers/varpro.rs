//! Variable projection for f = Φ(β) α + c(β).
//!
//! Each residual evaluation at β solves the linear least-squares problem for α with a
//! rank-revealing factorisation of Φ and records the full θ = [α, β], so the outer solver is
//! scored on the original residual. Jacobian (Kaufman): J = (I − U Uᵀ) ∂f/∂β at (α(β), β),
//! with U an orthonormal basis of range(Φ).

use nalgebra::{DMatrix, DVector};

use super::Lsq;
use crate::linalg::factor;
use crate::problem::Problem;
use crate::tracker::{Res, Tracker};

struct Proj {
    alpha: DVector<f64>,
    u: DMatrix<f64>,
}

pub struct VarPro<'a> {
    pub prob: &'a mut Problem,
    pub tr: &'a mut Tracker,
    theta: Vec<f64>,
    c: DVector<f64>,
    phi: DMatrix<f64>,
    target: DVector<f64>,
    dnl: DMatrix<f64>,
    trial: Option<Proj>,
    current: Option<Proj>,
}

impl<'a> VarPro<'a> {
    pub fn new(prob: &'a mut Problem, tr: &'a mut Tracker) -> Self {
        let (m, p, k) = (prob.m, prob.p, prob.lin.len());
        VarPro {
            theta: vec![0.0; p],
            c: DVector::zeros(m),
            phi: DMatrix::zeros(m, k),
            target: DVector::zeros(m),
            dnl: DMatrix::zeros(m, p - k),
            trial: None,
            current: None,
            prob,
            tr,
        }
    }

    pub fn beta_of(&self, theta: &[f64]) -> DVector<f64> {
        DVector::from_iterator(self.prob.nonlin.len(), self.prob.nonlin.iter().map(|&i| theta[i]))
    }

    /// Solve for α at β; residual into r; returns Σ r² (∞ if Φ or c overflowed).
    fn project(&mut self, beta: &DVector<f64>, r: &mut DVector<f64>) -> f64 {
        let prob = &mut *self.prob;
        for (&i, &b) in prob.nonlin.iter().zip(beta.iter()) {
            self.theta[i] = b;
        }
        for &i in &prob.lin {
            self.theta[i] = 0.0;
        }
        prob.separable(&self.theta, self.c.as_mut_slice(), &mut self.phi);
        if !(self.c.iter().all(|v| v.is_finite()) && self.phi.iter().all(|v| v.is_finite())) {
            self.trial = None;
            r.fill(f64::INFINITY);
            return f64::INFINITY;
        }
        for i in 0..prob.m {
            self.target[i] = prob.y_eff[i] - self.c[i];
        }
        let Some(fac) = factor(&self.phi, &self.target, true) else {
            self.trial = None;
            r.fill(f64::INFINITY);
            return f64::INFINITY;
        };
        let mut alpha = DVector::zeros(prob.lin.len());
        fac.solve_with_g(&mut alpha);
        // r = Φ α − target
        r.copy_from(&self.target);
        r.gemv(1.0, &self.phi, &alpha, -1.0);
        for (&i, &a) in prob.lin.iter().zip(alpha.iter()) {
            self.theta[i] = a;
        }
        let rank = fac.rank();
        let u = fac.u.expect("u requested").columns(0, fac.s.len()).into_owned();
        // keep only the numerically independent directions of range(Φ)
        let keep: Vec<usize> = (0..fac.s.len()).filter(|&i| fac.s[i] > fac.tol).collect();
        let u = if rank == fac.s.len() { u } else { u.select_columns(&keep) };
        self.trial = Some(Proj { alpha, u });
        r.norm_squared()
    }
}

impl Lsq for VarPro<'_> {
    fn n(&self) -> usize {
        self.prob.nonlin.len()
    }
    fn m(&self) -> usize {
        self.prob.m
    }
    fn tracker(&mut self) -> &mut Tracker {
        self.tr
    }
    fn residual(&mut self, beta: &DVector<f64>, r: &mut DVector<f64>) -> Res<f64> {
        self.tr.check()?;
        let t = std::time::Instant::now();
        let sse = self.project(beta, r);
        self.tr.nfev += 1;
        self.tr.fev_ns += t.elapsed().as_nanos() as u64;
        if self.trial.is_some() {
            let theta = std::mem::take(&mut self.theta);
            self.tr.record(&theta, sse);
            self.theta = theta;
        } else {
            self.tr.n_nonfinite += 1;
        }
        Ok(sse)
    }
    fn accept(&mut self) {
        if self.trial.is_some() {
            self.current = self.trial.take();
        }
    }
    fn jacobian(&mut self, beta: &DVector<f64>, j: &mut DMatrix<f64>) -> Res<bool> {
        self.tr.check()?;
        let t = std::time::Instant::now();
        let cur = self.current.as_ref().expect("jacobian before an accepted residual");
        let prob = &mut *self.prob;
        let mut theta = vec![0.0; prob.p];
        for (&i, &b) in prob.nonlin.iter().zip(beta.iter()) {
            theta[i] = b;
        }
        for (&i, &a) in prob.lin.iter().zip(cur.alpha.iter()) {
            theta[i] = a;
        }
        prob.jacobian_nonlinear(&theta, &mut self.dnl);
        // J = D − U (Uᵀ D)
        let utd = cur.u.tr_mul(&self.dnl);
        j.copy_from(&self.dnl);
        j.gemm(-1.0, &cur.u, &utd, 1.0);
        self.tr.njev += 1;
        self.tr.jev_ns += t.elapsed().as_nanos() as u64;
        Ok(j.iter().all(|v| v.is_finite()))
    }
}
