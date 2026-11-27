//! The installable fitter's entry points: one fit of a compiled model, and a threaded batch of
//! (expression, data) jobs that compiles each distinct expression once.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::expr::{compile, Compiled};
use crate::fit::fit;
use crate::problem::Problem;
use crate::solvers::Status;
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
    pub time_ms: f64,
    /// Loss history: time (ms from the start of the fit) and loss at the first evaluation and at
    /// every strict improvement.
    pub hist_t_ms: Vec<f64>,
    pub hist_loss: Vec<f64>,
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
    let mut tr = Tracker::new(budget_ms - t0.elapsed().as_secs_f64() * 1e3, &[], prob.inv_scale);
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
        time_ms: t0.elapsed().as_secs_f64() * 1e3,
        hist_t_ms: std::mem::take(&mut tr.hist_t_ms),
        hist_loss: std::mem::take(&mut tr.hist_loss),
    })
}

/// One batch job: an expression over `variables`, and its data.
#[derive(Clone, Debug)]
pub struct Job {
    pub expr: String,
    pub variables: Vec<String>,
    pub data: Data,
}

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
    let mut models: HashMap<(&str, &[String]), Result<Compiled, String>> = HashMap::new();
    for j in &jobs {
        models
            .entry((&j.expr, &j.variables))
            .or_insert_with(|| compile(&j.expr, &j.variables, None));
    }
    let slots: Vec<Mutex<Option<Data>>> = jobs.iter().map(|j| Mutex::new(Some(j.data.clone()))).collect();
    let results: Vec<Mutex<Result<Outcome, String>>> =
        jobs.iter().map(|_| Mutex::new(Err("cancelled".into()))).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, jobs.len().max(1)) {
            s.spawn(|| loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(i) else { return };
                let data = slots[i].lock().unwrap().take().unwrap_or_default();
                let r = match &models[&(job.expr.as_str(), job.variables.as_slice())] {
                    Ok(c) => fit_one(c, data, budget_ms),
                    Err(e) => Err(e.clone()),
                };
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
    }

    #[test]
    fn batch_keeps_order_and_reports_bad_jobs() {
        let vars = vec!["x".to_string()];
        let job = |expr: &str, d: Data| Job { expr: expr.into(), variables: vars.clone(), data: d };
        let jobs = vec![
            job("a*exp(-b*x) + c", exp_data(2.0, 0.7, 0.3)),
            job("a*exp(-b*x) +", exp_data(1.0, 1.0, 1.0)),
            job("a*exp(-b*x) + c", exp_data(-1.0, 0.2, 5.0)),
            job("a*x + c", Data { x: vec![vec![1.0, 2.0]], y: vec![1.0], ..Default::default() }),
        ];
        let (done, cancel) = (AtomicUsize::new(0), AtomicBool::new(false));
        let r = fit_many(jobs, 100.0, 3, &done, &cancel);
        assert_eq!(done.load(Ordering::Relaxed), 4);
        assert!((r[0].as_ref().unwrap().params[0] - 2.0).abs() < 1e-8);
        assert!(r[1].is_err() && r[3].is_err());
        assert!((r[2].as_ref().unwrap().params[2] - 5.0).abs() < 1e-8);
    }
}
