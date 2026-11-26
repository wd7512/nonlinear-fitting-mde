//! Python bindings (`symbfitrs._core`). Arrays cross as native-endian float64 bytes, so the
//! module needs only the stable ABI; `python/symbfitrs` wraps it in the public API.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::driver::{self, Data, Job, Outcome};
use crate::expr::{self, Compiled};

fn f64s(b: &[u8]) -> PyResult<Vec<f64>> {
    match b.as_chunks::<8>() {
        (words, []) => Ok(words.iter().map(|w| f64::from_ne_bytes(*w)).collect()),
        _ => Err(PyValueError::new_err("array buffer is not float64")),
    }
}

/// x columns, y, weights (float64 bytes) and p0.
type RawData = (Vec<Vec<u8>>, Vec<u8>, Option<Vec<u8>>, Option<Vec<f64>>);
/// (expr, variables, x columns, y, weights, p0)
type RawJob = (String, Vec<String>, Vec<Vec<u8>>, Vec<u8>, Option<Vec<u8>>, Option<Vec<f64>>);

fn data((x, y, w, p0): RawData) -> PyResult<Data> {
    Ok(Data {
        x: x.iter().map(|c| f64s(c)).collect::<PyResult<_>>()?,
        y: f64s(&y)?,
        w: w.map(|w| f64s(&w)).transpose()?,
        p0,
    })
}

type PyOutcome = (Vec<String>, Vec<f64>, f64, &'static str, String, usize, f64);

fn out(o: Outcome) -> PyOutcome {
    (o.names, o.params, o.loss, o.status, o.message, o.nfev, o.time_ms)
}

#[pyclass(frozen, module = "symbfitrs._core")]
struct Model(Arc<Compiled>);

#[pymethods]
impl Model {
    #[getter]
    fn params(&self) -> Vec<String> {
        self.0.params.clone()
    }

    #[getter]
    fn variables(&self) -> Vec<String> {
        self.0.variables.clone()
    }

    #[getter]
    fn linear(&self) -> Vec<usize> {
        self.0.linear.clone()
    }

    /// (names, params, loss, status, message, nfev, time_ms)
    #[pyo3(signature = (x, y, w, p0, budget_ms))]
    fn fit(
        &self,
        py: Python<'_>,
        x: Vec<Vec<u8>>,
        y: Vec<u8>,
        w: Option<Vec<u8>>,
        p0: Option<Vec<f64>>,
        budget_ms: f64,
    ) -> PyResult<PyOutcome> {
        let d = data((x, y, w, p0))?;
        let c = self.0.clone();
        py.detach(move || driver::fit_one(&c, d, budget_ms)).map(out).map_err(PyValueError::new_err)
    }
}

#[pyfunction]
#[pyo3(signature = (expr, variables, params=None))]
fn compile(expr: &str, variables: Vec<String>, params: Option<Vec<String>>) -> PyResult<Model> {
    expr::compile(expr, &variables, params.as_deref()).map(|c| Model(Arc::new(c))).map_err(PyValueError::new_err)
}

/// Jobs are (expr, variables, x, y, w, p0); results in job order, each (outcome tuple, "") or
/// (None, error). `progress(n)` is called with the number of newly finished jobs.
#[pyfunction]
#[pyo3(signature = (jobs, budget_ms, threads, progress=None))]
fn fit_many(
    py: Python<'_>,
    jobs: Vec<RawJob>,
    budget_ms: f64,
    threads: usize,
    progress: Option<Py<PyAny>>,
) -> PyResult<Vec<(Option<PyOutcome>, String)>> {
    let n = jobs.len();
    let jobs = jobs
        .into_iter()
        .map(|(expr, variables, x, y, w, p0)| Ok(Job { expr, variables, data: data((x, y, w, p0))? }))
        .collect::<PyResult<Vec<_>>>()?;
    let (done, cancel) = (AtomicUsize::new(0), AtomicBool::new(false));
    let mut interrupt: Option<PyErr> = None;
    let results = py.detach(|| {
        std::thread::scope(|s| {
            let worker = s.spawn(|| driver::fit_many(jobs, budget_ms, threads, &done, &cancel));
            let mut reported = 0;
            loop {
                let finished = worker.is_finished();
                let d = done.load(Ordering::Relaxed);
                let err = Python::attach(|py| -> PyResult<()> {
                    py.check_signals()?;
                    if let (Some(cb), true) = (&progress, d > reported) {
                        cb.call1(py, (d - reported,))?;
                    }
                    Ok(())
                });
                reported = d;
                if let Err(e) = err {
                    cancel.store(true, Ordering::Relaxed);
                    interrupt.get_or_insert(e);
                }
                if finished || reported == n {
                    return worker.join().expect("fit_many worker panicked");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        })
    });
    if let Some(e) = interrupt {
        return Err(e);
    }
    Ok(results
        .into_iter()
        .map(|r| match r {
            Ok(o) => (Some(out(o)), String::new()),
            Err(e) => (None, e),
        })
        .collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(fit_many, m)?)?;
    Ok(())
}
