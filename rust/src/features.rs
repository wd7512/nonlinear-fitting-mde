//! Inexpensive problem features for learning routing rules.
//!
//! *Static* features are known before any evaluation (structure only). *Probe* features need
//! one residual + Jacobian + factorisation at the start; they are computed untimed for
//! analysis and their cost is reported (`probe_ms`) so a rule using them can be charged for it.

use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use serde_json::{json, Map, Value};

use crate::linalg::factor;
use crate::problem::Problem;

pub fn static_features(prob: &Problem) -> Map<String, Value> {
    let (p, m, k) = (prob.p, prob.m, prob.lin.len());
    let mut f = Map::new();
    f.insert("feat_p".into(), json!(p));
    f.insert("feat_m".into(), json!(m));
    f.insert("feat_n_linear".into(), json!(k));
    f.insert("feat_n_nonlinear".into(), json!(p - k));
    f.insert("feat_sep_frac".into(), json!(k as f64 / p as f64));
    f.insert("feat_m_over_p".into(), json!(m as f64 / p as f64));
    f.insert("feat_weighted".into(), json!(prob.w.is_some()));
    f.insert("feat_work_f".into(), json!(prob.f.work()));
    f.insert("feat_work_fj".into(), json!(prob.fj.work()));
    f.insert("feat_work_sep".into(), json!(prob.sep.as_ref().map(|s| s.work())));
    // work per unknown: ≈ cost of one Jacobian column relative to a residual
    f.insert("feat_jac_cost_ratio".into(), json!(prob.fj.work() as f64 / prob.f.work().max(1) as f64));
    f
}

pub fn probe_features(prob: &mut Problem, theta0: &[f64]) -> Map<String, Value> {
    let t = Instant::now();
    let (m, p) = (prob.m, prob.p);
    let mut r = vec![0.0; m];
    let sse0 = prob.residual(theta0, &mut r);
    let mut out = Map::new();
    let mut put = |k: &str, v: Value| {
        out.insert(format!("probe_{k}"), v);
    };
    if !sse0.is_finite() || sse0 == 0.0 {
        put("ok", json!(false));
        return out;
    }
    let mut jac = DMatrix::zeros(m, p);
    prob.jacobian(theta0, &mut jac);
    if !jac.iter().all(|v| v.is_finite()) {
        put("ok", json!(false));
        return out;
    }
    let d: Vec<f64> = (0..p).map(|j| { let n = jac.column(j).norm(); if n > 0.0 { n } else { 1.0 } }).collect();
    let mut js = jac.clone();
    for j in 0..p {
        js.column_mut(j).unscale_mut(d[j]);
    }
    let rv = DVector::from_vec(r);
    let Some(fac) = factor(&js, &rv, false) else {
        put("ok", json!(false));
        return out;
    };
    let mut step = DVector::zeros(p);
    let pred = fac.step(0.0, &mut step);
    let x1: Vec<f64> = (0..p).map(|j| theta0[j] + step[j] / d[j]).collect();
    let mut r1 = vec![0.0; m];
    let sse1 = prob.residual(&x1, &mut r1);
    put("ok", json!(true));
    put("log10_cond", json!(fac.cond().log10()));
    put("rank_deficiency", json!(p - fac.rank()));
    put("gn_pred_frac", json!(pred / sse0)); // fraction of SSE the linear model can remove
    put("gn_ratio", json!(if sse1.is_finite() { (sse0 - sse1) / pred } else { f64::NEG_INFINITY }));
    put("gn_gain", json!(if sse1.is_finite() { 1.0 - sse1 / sse0 } else { f64::NEG_INFINITY }));
    put("log10_loss0", json!((sse0 * prob.inv_scale).log10()));
    put("ms", json!(t.elapsed().as_secs_f64() * 1e3));
    out
}
