//! The installable fitter's entry points: model evaluation, one fit of a compiled model, and a
//! threaded batch of fits.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::expr::Compiled;
use crate::fit::fit;
use crate::problem::Problem;
use crate::solvers::Status;
use crate::tape::Bound;
use crate::tracker::Tracker;

/// Data for one fit: one column per variable, targets, optional weights and start.
#[derive(Clone, Debug, Default)]
pub struct Data {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub w: Option<Vec<f64>>,
    pub p0: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    /// Parameter names, in the order of `params`.
    pub names: Vec<String>,
    /// Best parameters found (the start if no evaluation was finite).
    pub params: Vec<f64>,
    /// Mean of the squared (weighted) residuals at `params`.
    pub loss: f64,
    /// converged | stalled | deadline | failure
    pub status: &'static str,
    pub message: String,
    pub nfev: usize,
    /// Evaluations that gave a non-finite loss (skipped by the solver).
    pub n_nonfinite: usize,
    pub time_ms: f64,
    /// The best loss each time it improved, and when (ms since the call).
    pub history_ms: Vec<f64>,
    pub history_loss: Vec<f64>,
}

/// f(x; θ) at every data point.
pub fn predict(c: &Compiled, x: &[Vec<f64>], theta: &[f64]) -> Result<Vec<f64>, String> {
    let p = c.params.len();
    if theta.len() != p {
        return Err(format!("params has {} values, the model has {p} parameters", theta.len()));
    }
    if x.len() != c.variables.len() {
        return Err(format!("expected {} variable columns, got {}", c.variables.len(), x.len()));
    }
    let m = x.first().map_or(1, Vec::len);
    if x.iter().any(|col| col.len() != m) {
        return Err("x columns have different lengths".into());
    }
    let mut b = Bound::new(&c.f, x, m, p)?;
    b.eval(theta);
    let mut out = vec![0.0; m];
    b.output_into(0, &mut out);
    Ok(out)
}

/// Fit `c` to `data` within `budget_ms`, timed from the call.
pub fn fit_one(c: &Compiled, data: Data, budget_ms: f64) -> Result<Outcome, String> {
    let t0 = Instant::now();
    let p = c.params.len();
    let theta0 = match data.p0 {
        Some(p0) if p0.len() != p => return Err(format!("p0 has {} values, the model has {p} parameters", p0.len())),
        Some(p0) => p0,
        None => vec![1.0; p],
    };
    let mut prob = Problem::new(c, &data.x, data.y, data.w)?;
    let setup_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut tr = Tracker::new(budget_ms - setup_ms, &[], prob.inv_scale);
    let status = match catch_unwind(AssertUnwindSafe(|| fit(&mut prob, &theta0, &mut tr).0)) {
        Ok(s) => s,
        Err(e) => {
            let msg = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned());
            Status::Failure(format!("panicked: {}", msg.unwrap_or_default()))
        }
    };
    let message = match &status {
        Status::Failure(m) => m.clone(),
        _ => String::new(),
    };
    Ok(Outcome {
        names: c.params.clone(),
        params: tr.best_theta.clone().unwrap_or(theta0),
        loss: if tr.best_theta.is_some() { tr.best_loss } else { f64::NAN },
        status: status.kind(),
        message,
        nfev: tr.nfev,
        n_nonfinite: tr.n_nonfinite,
        time_ms: t0.elapsed().as_secs_f64() * 1e3,
        history_ms: tr.hist_t_ms.iter().map(|t| t + setup_ms).collect(),
        history_loss: tr.hist_loss,
    })
}

/// One batch job: a compiled model (or why it could not be compiled) and its data.
pub type Job = (Result<Arc<Compiled>, String>, Data);

/// Run `jobs` on `threads` threads, results in job order. Each job's budget starts when a thread
/// picks it up. `done` counts finished jobs; setting `cancel` stops picking up new ones (their
/// results are `Err("cancelled")`).
pub fn fit_many(
    jobs: Vec<Job>,
    budget_ms: f64,
    threads: usize,
    done: &AtomicUsize,
    cancel: &AtomicBool,
) -> Vec<Result<Outcome, String>> {
    let n = jobs.len();
    let slots: Vec<Mutex<Option<Job>>> = jobs.into_iter().map(|j| Mutex::new(Some(j))).collect();
    let results: Vec<Mutex<Result<Outcome, String>>> = (0..n).map(|_| Mutex::new(Err("cancelled".into()))).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            s.spawn(|| loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    return;
                }
                let (model, data) = slots[i].lock().unwrap().take().expect("each job is taken once");
                let r = model.and_then(|c| fit_one(&c, data, budget_ms));
                *results[i].lock().unwrap() = r;
                done.fetch_add(1, Ordering::Relaxed);
            });
        }
    });
    results.into_iter().map(|r| r.into_inner().unwrap()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::compile;

    fn exp_data(a: f64, b: f64, c: f64) -> Data {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y = x.iter().map(|t| a * (-b * t).exp() + c).collect();
        Data { x: vec![x], y, ..Default::default() }
    }

    #[test]
    fn fits_an_exponential() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let o = fit_one(&c, exp_data(2.0, 0.7, 0.3), 100.0).unwrap();
        assert_eq!(o.status, "converged", "{o:?}");
        for (got, want) in o.params.iter().zip([2.0, 0.7, 0.3]) {
            assert!((got - want).abs() < 1e-8, "{o:?}");
        }
        assert!(o.loss < 1e-20);
        assert_eq!(o.history_loss.last(), Some(&o.loss));
        assert!(o.history_ms.windows(2).all(|w| w[0] <= w[1]));
        assert!(o.history_loss.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn predict_matches_the_expression() {
        let c = compile("a*exp(-b*x) + c", &["x".into()], None).unwrap();
        let f = predict(&c, &[vec![0.0, 1.0]], &[2.0, 0.5, 1.0]).unwrap();
        assert!((f[0] - 3.0).abs() < 1e-15 && (f[1] - (2.0 * (-0.5f64).exp() + 1.0)).abs() < 1e-15);
        assert!(predict(&c, &[vec![0.0]], &[1.0]).is_err());
    }

    #[test]
    fn batch_keeps_order_and_reports_bad_jobs() {
        let vars = vec!["x".to_string()];
        let model = |e: &str| compile(e, &vars, None).map(Arc::new);
        let jobs: Vec<Job> = vec![
            (model("a*exp(-b*x) + c"), exp_data(2.0, 0.7, 0.3)),
            (model("a*exp(-b*x) +"), exp_data(1.0, 1.0, 1.0)),
            (model("a*exp(-b*x) + c"), exp_data(-1.0, 0.2, 5.0)),
            (model("a*x + c"), Data { x: vec![vec![1.0, 2.0]], y: vec![1.0], ..Default::default() }),
        ];
        let (done, cancel) = (AtomicUsize::new(0), AtomicBool::new(false));
        let r = fit_many(jobs, 100.0, 3, &done, &cancel);
        assert_eq!(done.load(Ordering::Relaxed), 4);
        assert!((r[0].as_ref().unwrap().params[0] - 2.0).abs() < 1e-8);
        assert!(r[1].is_err() && r[3].is_err());
        assert!((r[2].as_ref().unwrap().params[2] - 5.0).abs() < 1e-8);
    }
}
