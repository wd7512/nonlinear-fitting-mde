//! symbfitrs-bench: runs the native `varpro` crate baseline (`varpro_rs`) on the benchmark's
//! exported problems. symbfitrs itself is benchmarked through the Python package.
//!
//!   symbfitrs-bench bench --problems P.json --out records.jsonl
//!                   [--budgets 1,5,10,25] [--deadline-ms 25] [--max-wall-s 600] [--seed 42]
//!
//! `P.json` is written by `python -m bench run CONFIG --fitter`: expression strings + data +
//! starts, compiled here by the same parser as the package. Records follow the Python harness
//! schema, so its scoring and reports apply. Single-threaded, FP64; jobs run in a seeded random
//! order with a cooperative per-fit deadline and a global wall-time cap (unfinished jobs are
//! reported as truncated).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

mod varpro_crate;

use serde_json::{json, Map, Value};
use symbfitrs::expr::{compile, Compiled};
use symbfitrs::problem::Problem;
use symbfitrs::solvers::Status;
use symbfitrs::tracker::Tracker;

const USAGE: &str = "usage: symbfitrs-bench bench --problems P.json --out records.jsonl
                [--budgets 1,5,10,25] [--deadline-ms 25] [--max-wall-s 600] [--seed 42]";

struct Args {
    problems: String,
    out: String,
    budgets: Vec<f64>,
    deadline_ms: Option<f64>,
    max_wall_s: f64,
    seed: u64,
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

fn opt_vec(v: &Option<Vec<f64>>) -> Value {
    v.as_ref().map_or(Value::Null, |t| json!(t))
}

fn f64s(v: &Value, what: &str) -> Result<Vec<f64>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} must be a list"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{what}: non-numeric entry")))
        .collect()
}

fn strings(v: &Value, what: &str) -> Result<Vec<String>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} must be a list"))?
        .iter()
        .map(|s| s.as_str().map(String::from).ok_or_else(|| format!("{what}: non-string entry")))
        .collect()
}

/// An exported problem: the bound model plus what every record repeats.
struct BenchProblem {
    pb: Problem,
    pid: String,
    starts: Vec<(String, Vec<f64>)>,
    /// fields copied verbatim into every record (dataset, model, curve, reference, tags, ...)
    record: Value,
}

type Key = (String, Vec<String>, Vec<String>);

fn load(v: &Value, models: &mut HashMap<Key, Compiled>) -> Result<BenchProblem, String> {
    let pid = v["pid"].as_str().unwrap_or("?").to_string();
    let err = |e: String| format!("{pid}: {e}");
    let key: Key = (
        v["expr"].as_str().ok_or_else(|| err("expr missing".into()))?.to_string(),
        strings(&v["variables"], "variables").map_err(err)?,
        strings(&v["params"], "params").map_err(err)?,
    );
    if !models.contains_key(&key) {
        let c = compile(&key.0, &key.1, Some(&key.2)).map_err(err)?;
        models.insert(key.clone(), c);
    }
    let x: Vec<Vec<f64>> = v["x"]
        .as_array()
        .ok_or_else(|| err("x missing".into()))?
        .iter()
        .map(|c| f64s(c, "x"))
        .collect::<Result<_, _>>()
        .map_err(err)?;
    let y = f64s(&v["y"], "y").map_err(err)?;
    let w = if v["w"].is_null() { None } else { Some(f64s(&v["w"], "w").map_err(err)?) };
    let denom = v["denom"].as_f64().ok_or_else(|| err("denom missing".into()))?;
    let mut pb = Problem::new(&models[&key], &x, y, w).map_err(err)?;
    pb.inv_scale = 1.0 / (pb.m as f64 * denom);
    let starts = v["starts"]
        .as_array()
        .ok_or_else(|| err("starts missing".into()))?
        .iter()
        .map(|s| Ok((s[0].as_str().unwrap_or("?").to_string(), f64s(&s[1], "start")?)))
        .collect::<Result<Vec<_>, String>>()
        .map_err(err)?;
    Ok(BenchProblem { pb, pid, starts, record: v["record"].clone() })
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
    // ---- load + compile + bind (setup, reported separately) ----
    let t0 = Instant::now();
    let text = std::fs::read_to_string(&args.problems).map_err(|e| format!("{}: {e}", args.problems))?;
    let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let t_parse = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    let mut models = HashMap::new();
    let mut problems: Vec<BenchProblem> = Vec::new();
    let mut skipped = Map::new();
    for v in doc["problems"].as_array().ok_or("problems missing")? {
        match load(v, &mut models) {
            Ok(bp) => problems.push(bp),
            Err(e) => {
                skipped.insert(v["pid"].as_str().unwrap_or("?").to_string(), json!(e));
            }
        }
    }
    let t_bind = t0.elapsed().as_secs_f64();
    let mut meta = doc["meta"].clone();
    let budgets = args.budgets.clone();
    let deadline_ms = args.deadline_ms.unwrap_or_else(|| budgets.iter().cloned().fold(0.0, f64::max));

    // ---- warm-up: one fit of the first problem (caches, page faults) ----
    let t0 = Instant::now();
    if let Some(bp) = problems.first_mut() {
        let theta0 = bp.starts[0].1.clone();
        let mut tr = Tracker::new(deadline_ms.min(5.0), &budgets, bp.pb.inv_scale);
        varpro_crate::fit(&mut bp.pb, &theta0, &mut tr);
    }
    let t_warm = t0.elapsed().as_secs_f64();

    let mut jobs: Vec<(usize, usize)> = Vec::new();
    for (i, bp) in problems.iter().enumerate() {
        jobs.extend((0..bp.starts.len()).map(|s| (i, s)));
    }
    shuffle(&mut jobs, args.seed);

    let out = File::create(&args.out).map_err(|e| format!("{}: {e}", args.out))?;
    let mut out = BufWriter::new(out);
    let emit = |v: &Value, out: &mut BufWriter<File>| {
        writeln!(out, "{v}").ok();
    };
    if let Some(cfg) = meta.get_mut("config").and_then(Value::as_object_mut) {
        cfg.insert("solvers".into(), json!([varpro_crate::NAME]));
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
        if !skipped.is_empty() {
            let all = m.entry("skipped").or_insert_with(|| json!({}));
            if let Some(all) = all.as_object_mut() {
                all.extend(skipped);
            }
        }
    }
    emit(&meta, &mut out);

    // ---- timed loop ----
    let t_loop = Instant::now();
    let mut done = 0usize;
    for &(i, s) in &jobs {
        if t_start.elapsed().as_secs_f64() >= args.max_wall_s {
            emit(&json!({"type": "truncated", "completed": done, "n_jobs": jobs.len()}), &mut out);
            break;
        }
        let bp = &mut problems[i];
        let (start_name, theta0) = bp.starts[s].clone();
        let pb = &mut bp.pb;
        // untimed initial loss (reduction success test)
        let mut r0 = vec![0.0; pb.m];
        let sse0 = pb.residual(&theta0, &mut r0);
        let loss0 = pb.loss(sse0);

        let t_job = Instant::now();
        let mut tr = Tracker::new(deadline_ms, &budgets, pb.inv_scale);
        let status = varpro_crate::fit(pb, &theta0, &mut tr);
        let elapsed_ms = tr.elapsed_ms();
        let job_wall_ms = t_job.elapsed().as_secs_f64() * 1e3;

        let small = pb.p <= 128; // bench.config.MAX_THETA_P
        let mut rec = Map::new();
        rec.insert("type".into(), json!("fit"));
        if let Some(fields) = bp.record.as_object() {
            for (k, v) in fields {
                rec.insert(k.clone(), v.clone());
            }
        }
        let fields = [
            ("problem", json!(bp.pid)),
            ("p", json!(pb.p)),
            ("m", json!(pb.m)),
            ("n_linear", json!(pb.lin.len())),
            ("loss0", if loss0.is_finite() { json!(loss0) } else { Value::Null }),
            ("start", json!(start_name)),
            ("solver", json!(varpro_crate::NAME)),
            ("solver_label", json!(varpro_crate::NAME)),
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
