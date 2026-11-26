//! symbfitrs-bench: benchmark driver for the fitter, until the Python bindings replace it.
//!
//!   symbfitrs-bench bench --problems P.json --out records.jsonl
//!                   [--budgets 1,5,10,25] [--deadline-ms 25] [--max-wall-s 600] [--seed 42]
//!                   [--solvers symbfitrs,varpro_rs]
//!
//! `P.json` is written by `python -m bench run CONFIG --fitter`: SymPy-compiled tapes + data + starts.
//! Records follow the Python harness schema, so its scoring and reports apply.
//! Single-threaded, FP64; jobs run in a seeded random order with a cooperative per-fit
//! deadline and a global wall-time cap (unfinished jobs are reported as truncated).
//! `--solvers` adds the `varpro` crate as a native baseline (see `varpro_crate`).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

mod varpro_crate;

use symbfitrs::fit::fit;
use symbfitrs::problem::Problem;
use symbfitrs::solvers::Status;
use symbfitrs::tape::Program;
use symbfitrs::tracker::Tracker;
use serde_json::{json, Map, Value};

const USAGE: &str = "usage: symbfitrs-bench bench --problems P.json --out records.jsonl
                [--budgets 1,5,10,25] [--deadline-ms 25] [--max-wall-s 600] [--seed 42]
                [--solvers symbfitrs,varpro_rs]";

/// Label of the fitter in records and reports.
const NAME: &str = "symbfitrs";

struct Args {
    problems: String,
    out: String,
    budgets: Vec<f64>,
    deadline_ms: Option<f64>,
    max_wall_s: f64,
    seed: u64,
    solvers: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    match it.next().as_deref() {
        Some("bench") => {}
        _ => return Err(USAGE.into()),
    }
    let mut a = Args {
        problems: String::new(),
        out: String::new(),
        budgets: vec![1.0, 5.0, 10.0, 25.0],
        deadline_ms: None,
        max_wall_s: 600.0,
        seed: 42,
        solvers: vec![NAME.into()],
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--problems" => a.problems = v()?,
            "--out" => a.out = v()?,
            "--budgets" => {
                a.budgets = v()?
                    .split(',')
                    .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?
            }
            "--deadline-ms" => a.deadline_ms = Some(v()?.parse().map_err(|_| "--deadline-ms")?),
            "--max-wall-s" => a.max_wall_s = v()?.parse().map_err(|_| "--max-wall-s")?,
            "--seed" => a.seed = v()?.parse().map_err(|_| "--seed")?,
            "--solvers" => a.solvers = v()?.split(',').map(String::from).collect(),
            _ => return Err(format!("unknown argument {k}")),
        }
    }
    if let Some(s) = a.solvers.iter().find(|s| ![NAME, varpro_crate::NAME].contains(&s.as_str())) {
        return Err(format!("unknown solver {s}"));
    }
    if a.problems.is_empty() || a.out.is_empty() {
        return Err("--problems and --out are required".into());
    }
    Ok(a)
}

/// SplitMix64: deterministic job shuffling without extra dependencies.
fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

fn opt_vec(v: &Option<Vec<f64>>) -> Value {
    v.as_ref().map_or(Value::Null, |t| json!(t))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = bench(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// One fit by the named solver; returns its status and report label.
fn run_solver(solver: &str, pb: &mut Problem, theta0: &[f64], tr: &mut Tracker) -> (Status, String) {
    if solver == varpro_crate::NAME {
        return (varpro_crate::fit(pb, theta0, tr), solver.into());
    }
    let (status, path) = fit(pb, theta0, tr);
    let label = match path.label() {
        None => NAME.to_string(),
        Some(l) => format!("{NAME} ({l})"),
    };
    (status, label)
}

fn bench(args: Args) -> Result<(), String> {
    let t_start = Instant::now();
    // ---- load + bind (setup, reported separately) ----
    let t0 = Instant::now();
    let text = std::fs::read_to_string(&args.problems).map_err(|e| format!("{}: {e}", args.problems))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let t_parse = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let programs: Vec<Program> = doc["programs"]
        .as_array()
        .ok_or("programs missing")?
        .iter()
        .map(Program::from_json)
        .collect::<Result<_, _>>()?;
    let mut problems: Vec<Problem> = doc["problems"]
        .as_array()
        .ok_or("problems missing")?
        .iter()
        .map(|v| Problem::from_json(v, &programs))
        .collect::<Result<_, _>>()?;
    let t_bind = t0.elapsed().as_secs_f64();
    let mut meta = doc["meta"].clone();
    let budgets = args.budgets.clone();
    let deadline_ms = args.deadline_ms.unwrap_or_else(|| budgets.iter().cloned().fold(0.0, f64::max));

    // ---- warm-up: one fit of the first problem per solver (caches, page faults) ----
    let t0 = Instant::now();
    if let Some(pb) = problems.first_mut() {
        let theta0 = pb.starts[0].1.clone();
        for solver in &args.solvers {
            let mut tr = Tracker::new(deadline_ms.min(5.0), &budgets, pb.inv_scale);
            run_solver(solver, pb, &theta0, &mut tr);
        }
    }
    let t_warm = t0.elapsed().as_secs_f64();

    let mut jobs: Vec<(usize, usize, &str)> = Vec::new();
    for (i, pb) in problems.iter().enumerate() {
        for s in 0..pb.starts.len() {
            for solver in &args.solvers {
                jobs.push((i, s, solver.as_str()));
            }
        }
    }
    shuffle(&mut jobs, args.seed);

    let out = File::create(&args.out).map_err(|e| format!("{}: {e}", args.out))?;
    let mut out = BufWriter::new(out);
    let emit = |v: &Value, out: &mut BufWriter<File>| {
        writeln!(out, "{v}").ok();
    };
    if let Some(cfg) = meta.get_mut("config").and_then(Value::as_object_mut) {
        cfg.insert("solvers".into(), json!(args.solvers));
        cfg.insert("budgets_ms".into(), json!(budgets));
    }
    if let Some(m) = meta.as_object_mut() {
        m.insert("type".into(), json!("meta"));
        m.insert("engine".into(), json!("rust"));
        m.insert("n_jobs".into(), json!(jobs.len()));
        m.insert("n_problems".into(), json!(problems.len()));
        m.insert("import_s".into(), json!(t_parse));
        m.insert("problem_setup_s".into(), json!(t_bind));
        m.insert("warmup_s".into(), json!(t_warm));
        m.insert("deadline_ms".into(), json!(deadline_ms));
        m.insert("threadpools".into(), json!([]));
        m.insert("thread_env".into(), json!({"rust": "single thread"}));
        m.insert("machine".into(), json!(std::env::consts::ARCH));
    }
    emit(&meta, &mut out);

    // ---- timed loop ----
    let t_loop = Instant::now();
    let mut done = 0usize;
    for &(i, s, solver) in &jobs {
        if t_start.elapsed().as_secs_f64() >= args.max_wall_s {
            emit(&json!({"type": "truncated", "completed": done, "n_jobs": jobs.len()}), &mut out);
            break;
        }
        let pb = &mut problems[i];
        let (start_name, theta0) = pb.starts[s].clone();
        // untimed initial loss (reduction success test)
        let mut r0 = vec![0.0; pb.m];
        let sse0 = pb.residual(&theta0, &mut r0);
        let loss0 = pb.loss(sse0);

        let t_job = Instant::now();
        let mut tr = Tracker::new(deadline_ms, &budgets, pb.inv_scale);
        let (status, label) = run_solver(solver, pb, &theta0, &mut tr);
        let elapsed_ms = tr.elapsed_ms();
        let job_wall_ms = t_job.elapsed().as_secs_f64() * 1e3;

        let small = pb.p <= 128; // bench.config.MAX_THETA_P
        let mut rec = Map::new();
        rec.insert("type".into(), json!("fit"));
        if let Some(fields) = pb.record.as_object() {
            for (k, v) in fields {
                rec.insert(k.clone(), v.clone());
            }
        }
        let fields = [
            ("problem", json!(pb.pid)),
            ("p", json!(pb.p)),
            ("m", json!(pb.m)),
            ("n_linear", json!(pb.lin.len())),
            ("loss0", if loss0.is_finite() { json!(loss0) } else { Value::Null }),
            ("start", json!(start_name)),
            ("solver", json!(solver)),
            ("solver_label", json!(label)),
            ("status", json!(status.kind())),
            ("error", if let Status::Failure(e) = &status { json!(e) } else { Value::Null }),
            ("elapsed_ms", json!(elapsed_ms)),
            ("job_wall_ms", json!(job_wall_ms)),
            ("best_loss", if tr.best_loss.is_finite() { json!(tr.best_loss) } else { json!(f64::MAX) }),
            ("best_theta", if small { opt_vec(&tr.best_theta) } else { Value::Null }),
            (
                "theta_at",
                if small { json!(tr.snapshot_thetas().iter().map(opt_vec).collect::<Vec<_>>()) } else { Value::Null },
            ),
            ("nfev", json!(tr.nfev)),
            ("fev_ms", json!(tr.fev_ns as f64 / 1e6)),
            ("n_nonfinite", json!(tr.n_nonfinite)),
            ("n_trials", json!(tr.n_trials)),
            ("hist_t_ms", json!(tr.hist_t_ms)),
            ("hist_loss", json!(tr.hist_loss)),
        ];
        for (k, v) in fields {
            rec.insert(k.into(), v);
        }
        emit(&Value::Object(rec), &mut out);
        done += 1;
    }
    emit(
        &json!({
            "type": "end",
            "completed": done,
            "loop_s": t_loop.elapsed().as_secs_f64(),
            "worker_total_s": t_start.elapsed().as_secs_f64(),
        }),
        &mut out,
    );
    out.flush().map_err(|e| e.to_string())?;
    eprintln!(
        "symbfitrs-bench: {done}/{} fits in {:.2} s (setup {:.2} s) -> {}",
        jobs.len(),
        t_loop.elapsed().as_secs_f64(),
        t_parse + t_bind,
        args.out
    );
    Ok(())
}
