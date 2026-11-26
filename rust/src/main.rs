//! nlfit-rs: benchmark driver for the Rust inner optimiser.
//!
//!   nlfit-rs bench --problems P.json --out records.jsonl [--policies auto,lm,gn_lm]
//!                  [--budgets 1,5,10,25] [--deadline-ms 25] [--max-wall-s 600]
//!                  [--seed 42] [--no-probe] [--prefix rs_]
//!
//! `P.json` is written by `symreg-rs run CONFIG --rust` (Python): SymPy-compiled tapes + data + starts.
//! Records follow the Python harness schema, so `symreg-rs` scoring and reports apply.
//! Single-threaded, FP64; jobs run in a seeded random order with a cooperative per-fit
//! deadline and a global wall-time cap (unfinished jobs are reported as truncated).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use nlfit_core::features::{probe_features, static_features};
use nlfit_core::policy::{self, Policy};
use nlfit_core::problem::Problem;
use nlfit_core::tape::Program;
use nlfit_core::tracker::Tracker;
use serde_json::{json, Map, Value};

struct Args {
    problems: String,
    out: String,
    policies: Vec<String>,
    budgets: Vec<f64>,
    deadline_ms: Option<f64>,
    max_wall_s: f64,
    seed: u64,
    probe: bool,
    prefix: String,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    match it.next().as_deref() {
        Some("bench") => {}
        _ => return Err(include_str!("main.rs").lines().take(7).collect::<Vec<_>>().join("\n")),
    }
    let mut a = Args {
        problems: String::new(),
        out: String::new(),
        policies: vec!["auto".into()],
        budgets: vec![1.0, 5.0, 10.0, 25.0],
        deadline_ms: None,
        max_wall_s: 600.0,
        seed: 42,
        probe: true,
        prefix: "rs_".into(),
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--problems" => a.problems = v()?,
            "--out" => a.out = v()?,
            "--policies" => a.policies = v()?.split(',').map(str::to_string).collect(),
            "--budgets" => {
                a.budgets = v()?
                    .split(',')
                    .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
                    .collect::<Result<_, _>>()?
            }
            "--deadline-ms" => a.deadline_ms = Some(v()?.parse().map_err(|_| "--deadline-ms")?),
            "--max-wall-s" => a.max_wall_s = v()?.parse().map_err(|_| "--max-wall-s")?,
            "--seed" => a.seed = v()?.parse().map_err(|_| "--seed")?,
            "--prefix" => a.prefix = v()?,
            "--no-probe" => a.probe = false,
            _ => return Err(format!("unknown argument {k}")),
        }
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

fn subsample(ns: &[u64], k: usize) -> Vec<f64> {
    if ns.len() <= k {
        return ns.iter().map(|&x| x as f64 / 1e6).collect();
    }
    (0..k).map(|i| ns[i * (ns.len() - 1) / (k - 1)] as f64 / 1e6).collect()
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

fn bench(args: Args) -> Result<(), String> {
    let t_start = Instant::now();
    let policies: Vec<(String, Policy)> = args
        .policies
        .iter()
        .map(|n| Policy::parse(n).map(|p| (n.clone(), p)))
        .collect::<Result<_, _>>()?;

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

    // ---- warm-up: every policy once on the first problem (caches, page faults) ----
    let t0 = Instant::now();
    if let Some(pb) = problems.first_mut() {
        let theta0 = pb.starts[0].1.clone();
        for (_, pol) in &policies {
            let mut tr = Tracker::new(deadline_ms.min(5.0), &budgets, pb.inv_scale);
            policy::run(*pol, pb, &theta0, &mut tr);
        }
    }
    let t_warm = t0.elapsed().as_secs_f64();

    let mut jobs: Vec<(usize, usize, usize)> = Vec::new();
    for (i, pb) in problems.iter().enumerate() {
        for s in 0..pb.starts.len() {
            for a in 0..policies.len() {
                jobs.push((i, s, a));
            }
        }
    }
    shuffle(&mut jobs, args.seed);

    let out = File::create(&args.out).map_err(|e| format!("{}: {e}", args.out))?;
    let mut out = BufWriter::new(out);
    let emit = |v: &Value, out: &mut BufWriter<File>| {
        writeln!(out, "{v}").ok();
    };
    let names: Vec<String> = policies.iter().map(|(n, _)| format!("{}{n}", args.prefix)).collect();
    if let Some(cfg) = meta.get_mut("config").and_then(Value::as_object_mut) {
        cfg.insert("solvers".into(), json!(names));
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

    // static + probe features once per (problem, start), outside the timed loop
    let mut probe_cache: HashMap<(usize, usize), Map<String, Value>> = HashMap::new();
    let statics: Vec<Map<String, Value>> = problems.iter().map(static_features).collect();

    // ---- timed loop ----
    let t_loop = Instant::now();
    let mut done = 0usize;
    for &(i, s, a) in &jobs {
        if t_start.elapsed().as_secs_f64() >= args.max_wall_s {
            emit(&json!({"type": "truncated", "completed": done, "n_jobs": jobs.len()}), &mut out);
            break;
        }
        let pb = &mut problems[i];
        let (start_name, theta0) = pb.starts[s].clone();
        if args.probe && !probe_cache.contains_key(&(i, s)) {
            probe_cache.insert((i, s), probe_features(pb, &theta0));
        }
        // untimed initial loss (reduction success test)
        let mut r0 = vec![0.0; pb.m];
        let sse0 = pb.residual(&theta0, &mut r0);
        let loss0 = pb.loss(sse0);

        let t_job = Instant::now();
        let mut tr = Tracker::new(deadline_ms, &budgets, pb.inv_scale);
        let (status, path) = policy::run(policies[a].1, pb, &theta0, &mut tr);
        let elapsed_ms = tr.elapsed_ms();
        let job_wall_ms = t_job.elapsed().as_secs_f64() * 1e3;

        let name = &names[a];
        let label = match path {
            "varpro" | "full" => name.clone(),
            "no linear params" => format!("{name} (LM fallback)"),
            other => format!("{name} ({other})"),
        };
        let small = pb.p <= 128;
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
            ("solver", json!(name)),
            ("solver_label", json!(label)),
            ("status", json!(status.kind())),
            ("error", if let nlfit_core::solvers::Status::Failure(e) = &status { json!(e) } else { Value::Null }),
            ("elapsed_ms", json!(elapsed_ms)),
            ("job_wall_ms", json!(job_wall_ms)),
            ("best_loss", if tr.best_loss.is_finite() { json!(tr.best_loss) } else { json!(f64::MAX) }),
            ("best_theta", if small { opt_vec(&tr.best_theta) } else { Value::Null }),
            (
                "theta_at",
                if small { json!(tr.snapshot_thetas().iter().map(opt_vec).collect::<Vec<_>>()) } else { Value::Null },
            ),
            ("nfev", json!(tr.nfev)),
            ("njev", json!(tr.njev)),
            ("fev_ms", json!(tr.fev_ns as f64 / 1e6)),
            ("jev_ms", json!(tr.jev_ns as f64 / 1e6)),
            ("n_nonfinite", json!(tr.n_nonfinite)),
            ("n_trials", json!(tr.n_trials)),
            ("n_iter", json!(tr.iter_ns.len())),
            ("iter_ms", json!(subsample(&tr.iter_ns, 256))),
            ("hist_t_ms", json!(tr.hist_t_ms)),
            ("hist_loss", json!(tr.hist_loss)),
            ("events", json!(tr.events)),
        ];
        for (k, v) in fields {
            rec.insert(k.into(), v);
        }
        for (k, v) in &statics[i] {
            rec.insert(k.clone(), v.clone());
        }
        if let Some(pf) = probe_cache.get(&(i, s)) {
            for (k, v) in pf {
                rec.insert(k.clone(), v.clone());
            }
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
        "nlfit-rs: {done}/{} fits in {:.2} s (setup {:.2} s) -> {}",
        jobs.len(),
        t_loop.elapsed().as_secs_f64(),
        t_parse + t_bind,
        args.out
    );
    Ok(())
}
