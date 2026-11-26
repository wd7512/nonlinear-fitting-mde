//! Rank-revealing least-squares factorisation: A = U Σ Vᵀ via Householder QR then an
//! SVD of the small triangular factor (tall A), or a direct SVD (wide A).
//!
//! One factorisation serves every damping value λ of a Levenberg–Marquardt iteration
//! (δ(λ) = −V diag(σ/(σ²+λ)) Uᵀb), so rejected steps never refactorise.

use nalgebra::{DMatrix, DVector};

pub struct Factor {
    /// singular values (any order)
    pub s: DVector<f64>,
    /// right singular vectors as columns, n × r
    pub v: DMatrix<f64>,
    /// Uᵀ b
    pub g: DVector<f64>,
    /// left singular vectors, m × r (only if requested)
    pub u: Option<DMatrix<f64>>,
    pub smax: f64,
    /// singular values ≤ tol are treated as zero (rank truncation)
    pub tol: f64,
}

pub fn factor(a: &DMatrix<f64>, b: &DVector<f64>, want_u: bool) -> Option<Factor> {
    let (m, n) = a.shape();
    if n == 0 || !a.iter().all(|v| v.is_finite()) || !b.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (s, v, g, u) = if m > n {
        let qr = a.clone().qr();
        let r = qr.r();
        let mut qtb = b.clone();
        qr.q_tr_mul(&mut qtb);
        let svd = r.try_svd_unordered(true, true, f64::EPSILON, 10_000)?;
        let ur = svd.u?;
        let vt = svd.v_t?;
        let g = ur.tr_mul(&qtb.rows(0, n));
        let u = if want_u { Some(qr.q() * &ur) } else { None };
        (svd.singular_values, vt.transpose(), g, u)
    } else {
        let svd = a.clone().try_svd_unordered(true, true, f64::EPSILON, 10_000)?;
        let u = svd.u?;
        let vt = svd.v_t?;
        let g = u.tr_mul(b);
        (svd.singular_values, vt.transpose(), g, if want_u { Some(u) } else { None })
    };
    if !s.iter().all(|x| x.is_finite()) {
        return None;
    }
    let smax = s.iter().cloned().fold(0.0, f64::max);
    let tol = smax * (m.max(n) as f64) * f64::EPSILON;
    Some(Factor { s, v, g, u, smax, tol })
}

impl Factor {
    /// Damped step for min ‖A δ + b‖² + λ‖δ‖²: δ = −V diag(σ/(σ²+λ)) g. λ = 0 gives the
    /// rank-truncated Gauss–Newton step. Returns (δ, predicted reduction of ‖·‖²).
    pub fn step(&self, lam: f64, out: &mut DVector<f64>) -> f64 {
        out.fill(0.0);
        let mut pred = 0.0;
        for i in 0..self.s.len() {
            let (s, g) = (self.s[i], self.g[i]);
            if s <= self.tol {
                continue;
            }
            let coef = s / (s * s + lam);
            out.axpy(-coef * g, &self.v.column(i), 1.0);
            let resid = g * lam / (s * s + lam);
            pred += g * g - resid * resid;
        }
        pred
    }

    /// Least-squares solution of A x = c for the same A when U was kept: x = V Σ⁺ Uᵀ c.
    pub fn solve_with_g(&self, out: &mut DVector<f64>) {
        out.fill(0.0);
        for i in 0..self.s.len() {
            if self.s[i] > self.tol {
                out.axpy(self.g[i] / self.s[i], &self.v.column(i), 1.0);
            }
        }
    }

    pub fn rank(&self) -> usize {
        self.s.iter().filter(|&&s| s > self.tol).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gn_step_solves_linear_lsq() {
        let a = DMatrix::from_row_slice(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let x_true = DVector::from_vec(vec![0.5, -2.0]);
        let r = &a * &x_true; // residual at 0 for target r: min ‖A δ − r‖ ⇒ δ = x_true
        let f = factor(&a, &(-&r), false).unwrap();
        let mut d = DVector::zeros(2);
        let pred = f.step(0.0, &mut d);
        assert!((&d - &x_true).norm() < 1e-12);
        assert!((pred - r.norm_squared()).abs() < 1e-10);
    }
}
