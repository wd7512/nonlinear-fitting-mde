//! Per-fit bookkeeping shared by every solver: the cooperative deadline, best-so-far
//! parameters (timestamped, no late credit at checkpoints), evaluation counters and the
//! compressed loss history (first value + every strict improvement).

use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    Deadline,
}

pub type Res<T> = Result<T, Stop>;

pub struct Tracker {
    t0: Instant,
    deadline_ns: u64,
    inv_scale: f64,
    snap_ns: Vec<u64>,
    pub snapshots: Vec<Option<Vec<f64>>>,
    pub best_loss: f64,
    pub best_theta: Option<Vec<f64>>,
    pub hist_t_ms: Vec<f64>,
    pub hist_loss: Vec<f64>,
    pub n_trials: usize,
    pub nfev: usize,
    pub fev_ns: u64,
    pub n_nonfinite: usize,
    /// Opt-in: θ after each accepted update (the start is entry 0), and the accepted-update cap.
    pub trace: Option<Vec<Vec<f64>>>,
    pub max_steps: Option<usize>,
    pub steps: usize,
    pending: Vec<f64>,
}

impl Tracker {
    pub fn new(deadline_ms: f64, budgets_ms: &[f64], inv_scale: f64) -> Tracker {
        let mut snap_ns: Vec<u64> = budgets_ms.iter().map(|b| (b * 1e6) as u64).collect();
        snap_ns.sort_unstable();
        Tracker {
            t0: Instant::now(),
            deadline_ns: (deadline_ms * 1e6) as u64,
            inv_scale,
            snap_ns,
            snapshots: Vec::new(),
            best_loss: f64::INFINITY,
            best_theta: None,
            hist_t_ms: Vec::new(),
            hist_loss: Vec::new(),
            n_trials: 0,
            nfev: 0,
            fev_ns: 0,
            n_nonfinite: 0,
            trace: None,
            max_steps: None,
            steps: 0,
            pending: Vec::new(),
        }
    }

    /// Record θ at the start and after every accepted update; stop after `max_steps` updates.
    pub fn enable_trace(&mut self, trace: bool, max_steps: Option<usize>) {
        self.trace = trace.then(Vec::new);
        self.max_steps = max_steps;
    }

    /// The last finite trial became the iterate. `initial` marks the start (step 0). Returns
    /// true once `max_steps` accepted updates have been taken.
    pub fn accept_step(&mut self, initial: bool) -> bool {
        if let Some(t) = &mut self.trace {
            t.push(self.pending.clone());
        }
        if !initial {
            self.steps += 1;
        }
        self.max_steps.is_some_and(|m| self.steps >= m)
    }

    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.t0.elapsed().as_nanos() as u64
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns() as f64 / 1e6
    }

    /// Cooperative deadline check, done before starting any evaluation.
    #[inline]
    pub fn check(&self) -> Res<()> {
        if self.elapsed_ns() >= self.deadline_ns {
            Err(Stop::Deadline)
        } else {
            Ok(())
        }
    }

    /// Time a residual evaluation, after the deadline check.
    #[inline]
    pub fn timed<T>(&mut self, f: impl FnOnce() -> T) -> Res<T> {
        self.check()?;
        let t = Instant::now();
        let out = f();
        self.nfev += 1;
        self.fev_ns += t.elapsed().as_nanos() as u64;
        Ok(out)
    }

    /// Record a trial with full parameters `theta` and residual sum of squares `sse`.
    pub fn record(&mut self, theta: &[f64], sse: f64) -> f64 {
        let loss = sse * self.inv_scale;
        if !loss.is_finite() {
            self.n_nonfinite += 1;
            return loss;
        }
        self.n_trials += 1;
        if self.trace.is_some() {
            self.pending.clear();
            self.pending.extend_from_slice(theta);
        }
        let t = self.elapsed_ns();
        while self.snapshots.len() < self.snap_ns.len() && t > self.snap_ns[self.snapshots.len()] {
            self.snapshots.push(self.best_theta.clone());
        }
        if self.hist_loss.is_empty() || loss < self.best_loss {
            self.hist_t_ms.push(t as f64 / 1e6);
            self.hist_loss.push(loss);
        }
        if loss < self.best_loss {
            self.best_loss = loss;
            match &mut self.best_theta {
                Some(b) if b.len() == theta.len() => b.copy_from_slice(theta),
                _ => self.best_theta = Some(theta.to_vec()),
            }
        }
        loss
    }

    /// Best θ available at each checkpoint (no later record crossed ⇒ final best).
    pub fn snapshot_thetas(&self) -> Vec<Option<Vec<f64>>> {
        let mut out = self.snapshots.clone();
        while out.len() < self.snap_ns.len() {
            out.push(self.best_theta.clone());
        }
        out
    }
}

impl Tracker {
    /// 1 / (m · denom): converts a residual sum of squares to the normalised loss.
    pub fn inv_scale(&self) -> f64 {
        self.inv_scale
    }
}
