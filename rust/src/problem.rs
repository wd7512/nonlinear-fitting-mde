//! A fitting problem: a compiled model bound to its data.
//!
//! residual r(θ) = w ⊙ (f(x; θ) − y); loss = Σ r² · inv_scale (1/m unless the caller rescales).
//! For separable models f = Φ(β) α + c(β), the `sep` program outputs [c, Φ_0, …, Φ_{k-1}]
//! (α set to zero), with α = θ[linear_idx], β = θ[nonlinear_idx].

use nalgebra::DMatrix;

use crate::expr::Compiled;
use crate::tape::Bound;

pub struct Problem {
    pub p: usize,
    pub m: usize,
    pub y_eff: Vec<f64>,
    pub w: Option<Vec<f64>>,
    pub inv_scale: f64,
    pub lin: Vec<usize>,
    pub nonlin: Vec<usize>,
    pub f: Bound,
    pub fj: Bound,
    pub sep: Option<Bound>,
}

impl Problem {
    /// Bind a compiled model to data: `x` holds one column per variable; loss = mean of r².
    pub fn new(c: &Compiled, x: &[Vec<f64>], y: Vec<f64>, w: Option<Vec<f64>>) -> Result<Problem, String> {
        let (m, p) = (y.len(), c.params.len());
        if x.len() != c.variables.len() {
            return Err(format!("expected {} variable columns, got {}", c.variables.len(), x.len()));
        }
        if let Some(col) = x.iter().find(|col| col.len() != m) {
            return Err(format!("x has a column of length {}, y has length {m}", col.len()));
        }
        if w.as_ref().is_some_and(|w| w.len() != m) {
            return Err(format!("weights must have length {m}"));
        }
        if m == 0 {
            return Err("no data points".into());
        }
        let y_eff = match &w {
            Some(w) => y.iter().zip(w).map(|(a, b)| a * b).collect(),
            None => y,
        };
        let lin = c.linear.clone();
        Ok(Problem {
            p,
            m,
            y_eff,
            w,
            inv_scale: 1.0 / m as f64,
            nonlin: (0..p).filter(|i| !lin.contains(i)).collect(),
            lin,
            f: Bound::new(&c.f, x, m, p)?,
            fj: Bound::new(&c.fj, x, m, p)?,
            sep: c.sep.as_ref().map(|s| Bound::new(s, x, m, p)).transpose()?,
        })
    }

    #[inline]
    fn weight(&self, out: &mut [f64]) {
        if let Some(w) = &self.w {
            for (o, wi) in out.iter_mut().zip(w) {
                *o *= wi;
            }
        }
    }

    /// r = w (f − y); returns Σ r² (non-finite if the model overflowed).
    pub fn residual(&mut self, theta: &[f64], out: &mut [f64]) -> f64 {
        self.f.eval(theta);
        self.f.output_into(0, out);
        self.weight(out);
        let mut sse = 0.0;
        for (o, y) in out.iter_mut().zip(&self.y_eff) {
            *o -= y;
            sse += *o * *o;
        }
        sse
    }

    /// Weighted Jacobian of the full model, m × p (column-major).
    pub fn jacobian(&mut self, theta: &[f64], jac: &mut DMatrix<f64>) {
        self.fj.eval(theta);
        let m = self.m;
        let s = jac.as_mut_slice();
        for j in 0..self.p {
            let col = &mut s[j * m..(j + 1) * m];
            self.fj.output_into(1 + j, col);
        }
        if let Some(w) = &self.w {
            for j in 0..self.p {
                for (v, wi) in s[j * m..(j + 1) * m].iter_mut().zip(w) {
                    *v *= wi;
                }
            }
        }
    }

    /// Weighted nonlinear-parameter columns of the Jacobian at θ, m × (p − k).
    pub fn jacobian_nonlinear(&mut self, theta: &[f64], out: &mut DMatrix<f64>) {
        self.fj.eval(theta);
        let m = self.m;
        let s = out.as_mut_slice();
        for (c, &j) in self.nonlin.iter().enumerate() {
            self.fj.output_into(1 + j, &mut s[c * m..(c + 1) * m]);
        }
        if let Some(w) = &self.w {
            for c in 0..self.nonlin.len() {
                for (v, wi) in s[c * m..(c + 1) * m].iter_mut().zip(w) {
                    *v *= wi;
                }
            }
        }
    }

    /// Weighted separable parts at θ (α ignored): c → `c`, Φ → `phi` (m × k).
    pub fn separable(&mut self, theta: &[f64], c: &mut [f64], phi: &mut DMatrix<f64>) {
        let m = self.m;
        let sep = self.sep.as_mut().expect("separable() on a model without linear params");
        sep.eval(theta);
        sep.output_into(0, c);
        let s = phi.as_mut_slice();
        for k in 0..self.lin.len() {
            sep.output_into(1 + k, &mut s[k * m..(k + 1) * m]);
        }
        if let Some(w) = &self.w {
            for (v, wi) in c.iter_mut().zip(w) {
                *v *= wi;
            }
            for k in 0..self.lin.len() {
                for (v, wi) in s[k * m..(k + 1) * m].iter_mut().zip(w) {
                    *v *= wi;
                }
            }
        }
    }

    pub fn loss(&self, sse: f64) -> f64 {
        sse * self.inv_scale
    }
}
