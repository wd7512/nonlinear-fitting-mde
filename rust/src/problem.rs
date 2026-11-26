//! A fitting problem bound to its data, loaded from the JSON exported by `symreg-rs run CONFIG --rust`.
//!
//! residual r(θ) = w ⊙ (f(x; θ) − y); loss = Σ r² / (m · denom).
//! For separable models f = Φ(β) α + c(β), the `sep` program outputs [c, Φ_0, …, Φ_{k-1}]
//! (α set to zero), with α = θ[linear_idx], β = θ[nonlinear_idx].

use nalgebra::DMatrix;
use serde_json::Value;

use crate::tape::{Bound, Program};

pub struct Problem {
    pub pid: String,
    pub p: usize,
    pub m: usize,
    pub y_eff: Vec<f64>,
    pub w: Option<Vec<f64>>,
    pub inv_scale: f64, // 1 / (m · denom)
    pub lin: Vec<usize>,
    pub nonlin: Vec<usize>,
    pub f: Bound,
    pub fj: Bound,
    pub sep: Option<Bound>,
    pub starts: Vec<(String, Vec<f64>)>,
    /// fields copied verbatim into every record (dataset, model, curve, reference, tags, ...)
    pub record: Value,
}

fn f64s(v: &Value, what: &str) -> Result<Vec<f64>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} must be a list"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{what}: non-numeric entry")))
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
        .unwrap_or_default()
}

impl Problem {
    pub fn from_json(v: &Value, programs: &[Program]) -> Result<Problem, String> {
        let pid = v["pid"].as_str().unwrap_or("?").to_string();
        let err = |e: String| format!("{pid}: {e}");
        let p = v["p"].as_u64().ok_or("p")? as usize;
        let m = v["m"].as_u64().ok_or("m")? as usize;
        let x: Vec<Vec<f64>> = v["x"]
            .as_array()
            .ok_or("x")?
            .iter()
            .map(|c| f64s(c, "x"))
            .collect::<Result<_, _>>()
            .map_err(err)?;
        let y = f64s(&v["y"], "y").map_err(err)?;
        let w = if v["w"].is_null() { None } else { Some(f64s(&v["w"], "w").map_err(err)?) };
        let y_eff = match &w {
            Some(w) => y.iter().zip(w).map(|(a, b)| a * b).collect(),
            None => y.clone(),
        };
        let denom = v["denom"].as_f64().ok_or("denom")?;
        let lin = usizes(&v["linear_idx"]);
        let nonlin = (0..p).filter(|i| !lin.contains(i)).collect();
        let prog = |key: &str| -> Result<Option<Bound>, String> {
            match v[key].as_u64() {
                None => Ok(None),
                Some(i) => {
                    let pr = programs.get(i as usize).ok_or("program id out of range")?;
                    Bound::new(pr, &x, m, p).map(Some).map_err(|e| format!("{pid} {key}: {e}"))
                }
            }
        };
        let f = prog("prog_f")?.ok_or("prog_f missing")?;
        let fj = prog("prog_fj")?.ok_or("prog_fj missing")?;
        let sep = if lin.is_empty() { None } else { prog("prog_sep")? };
        if fj.n_outputs() != p + 1 {
            return Err(err(format!("fj has {} outputs, expected {}", fj.n_outputs(), p + 1)));
        }
        if let Some(s) = &sep {
            if s.n_outputs() != lin.len() + 1 {
                return Err(err("sep output count mismatch".into()));
            }
        }
        let starts = v["starts"]
            .as_array()
            .ok_or("starts")?
            .iter()
            .map(|s| Ok((s[0].as_str().unwrap_or("?").to_string(), f64s(&s[1], "start")?)))
            .collect::<Result<Vec<_>, String>>()
            .map_err(err)?;
        Ok(Problem {
            pid,
            p,
            m,
            y_eff,
            w,
            inv_scale: 1.0 / (m as f64 * denom),
            lin,
            nonlin,
            f,
            fj,
            sep,
            starts,
            record: v["record"].clone(),
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
