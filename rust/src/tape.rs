//! Vectorised expression tapes.
//!
//! A [`Program`] is a straight-line list of operations in SSA form (register `i` holds the
//! result of op `i`; operands always refer to earlier registers), exported from SymPy after
//! common-subexpression elimination. Registers are either scalars (no data dependence) or
//! vectors of length `m` (one value per observation).
//!
//! Binding a program to data ([`Bound`]) fixes widths and buffer offsets and evaluates every
//! parameter-independent register once (data columns, `cos(k*pi*x)`, constants), so each
//! evaluation only executes the parameter-dependent ops, over one contiguous buffer.
//!
//! Outputs are lists of registers: a single register (broadcast to `m` if scalar), or `m`
//! scalar registers, one per row (residual-vector problems such as MGH).

use serde_json::Value;

use crate::special::expwerfc;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Const(f64),
    Var(u32),
    Param(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Div(u32, u32),
    PowF(u32, u32),
    Max(u32, u32),
    Min(u32, u32),
    ExpWErfc(u32, u32),
    PowI(u32, i32),
    Neg(u32),
    Sqrt(u32),
    Exp(u32),
    Log(u32),
    Sin(u32),
    Cos(u32),
    Tan(u32),
    Tanh(u32),
    Sinh(u32),
    Cosh(u32),
    Atan(u32),
    Asin(u32),
    Acos(u32),
    Abs(u32),
    Sign(u32),
    Erf(u32),
    Erfc(u32),
    Wrap(u32),
    Step(u32),
}

impl Op {
    fn args(&self) -> (Option<u32>, Option<u32>) {
        use Op::*;
        match *self {
            Const(_) | Var(_) | Param(_) => (None, None),
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) | PowF(a, b) | Max(a, b) | Min(a, b)
            | ExpWErfc(a, b) => (Some(a), Some(b)),
            PowI(a, _) | Neg(a) | Sqrt(a) | Exp(a) | Log(a) | Sin(a) | Cos(a) | Tan(a)
            | Tanh(a) | Sinh(a) | Cosh(a) | Atan(a) | Asin(a) | Acos(a) | Abs(a) | Sign(a)
            | Erf(a) | Erfc(a) | Wrap(a) | Step(a) => (Some(a), None),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Program {
    pub ops: Vec<Op>,
    pub outputs: Vec<Vec<u32>>,
}

fn idx(v: &Value) -> Result<u32, String> {
    v.as_u64().map(|x| x as u32).ok_or_else(|| format!("bad register {v}"))
}

impl Program {
    pub fn from_json(v: &Value) -> Result<Program, String> {
        let ops_v = v["ops"].as_array().ok_or("program.ops missing")?;
        let mut ops = Vec::with_capacity(ops_v.len());
        for (i, o) in ops_v.iter().enumerate() {
            let a = o.as_array().ok_or("op must be an array")?;
            let name = a[0].as_str().ok_or("op name")?;
            let u = |k: usize| idx(&a[k]);
            use Op::*;
            let op = match name {
                "const" => Const(a[1].as_f64().ok_or("const value")?),
                "var" => Var(u(1)?),
                "param" => Param(u(1)?),
                "add" => Add(u(1)?, u(2)?),
                "sub" => Sub(u(1)?, u(2)?),
                "mul" => Mul(u(1)?, u(2)?),
                "div" => Div(u(1)?, u(2)?),
                "powf" => PowF(u(1)?, u(2)?),
                "max" => Max(u(1)?, u(2)?),
                "min" => Min(u(1)?, u(2)?),
                "expwerfc" => ExpWErfc(u(1)?, u(2)?),
                "powi" => PowI(u(1)?, a[2].as_i64().ok_or("powi exponent")? as i32),
                "neg" => Neg(u(1)?),
                "sqrt" => Sqrt(u(1)?),
                "exp" => Exp(u(1)?),
                "log" => Log(u(1)?),
                "sin" => Sin(u(1)?),
                "cos" => Cos(u(1)?),
                "tan" => Tan(u(1)?),
                "tanh" => Tanh(u(1)?),
                "sinh" => Sinh(u(1)?),
                "cosh" => Cosh(u(1)?),
                "atan" => Atan(u(1)?),
                "asin" => Asin(u(1)?),
                "acos" => Acos(u(1)?),
                "abs" => Abs(u(1)?),
                "sign" => Sign(u(1)?),
                "erf" => Erf(u(1)?),
                "erfc" => Erfc(u(1)?),
                "wrap" => Wrap(u(1)?),
                "step" => Step(u(1)?),
                other => return Err(format!("unknown op {other:?}")),
            };
            let (x, y) = op.args();
            if x.is_some_and(|r| r as usize >= i) || y.is_some_and(|r| r as usize >= i) {
                return Err(format!("op {i} refers to a later register"));
            }
            ops.push(op);
        }
        let outputs = v["outputs"]
            .as_array()
            .ok_or("program.outputs missing")?
            .iter()
            .map(|o| {
                o.as_array()
                    .ok_or_else(|| "output must be a list".to_string())?
                    .iter()
                    .map(idx)
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for o in outputs.iter().flatten() {
            if *o as usize >= ops.len() {
                return Err("output register out of range".into());
            }
        }
        Ok(Program { ops, outputs })
    }
}

/// A program bound to data: widths, offsets and pre-evaluated static registers.
#[derive(Clone, Debug)]
pub struct Bound {
    ops: Vec<Op>,
    outputs: Vec<Vec<u32>>,
    off: Vec<usize>,
    width: Vec<usize>,
    dynamic: Vec<u32>, // parameter-dependent ops, in order
    buf: Vec<f64>,
    m: usize,
}

#[inline(always)]
fn unary(buf: &mut [f64], out: usize, a: usize, w: usize, f: impl Fn(f64) -> f64) {
    let (src, dst) = buf.split_at_mut(out);
    for (d, &s) in dst[..w].iter_mut().zip(&src[a..a + w]) {
        *d = f(s);
    }
}

#[inline(always)]
fn binary(
    buf: &mut [f64],
    out: usize,
    (a, wa): (usize, usize),
    (b, wb): (usize, usize),
    w: usize,
    f: impl Fn(f64, f64) -> f64,
) {
    let (src, dst) = buf.split_at_mut(out);
    let dst = &mut dst[..w];
    match (wa == 1, wb == 1) {
        (true, true) => dst[0] = f(src[a], src[b]),
        (true, false) => {
            let x = src[a];
            for (d, &y) in dst.iter_mut().zip(&src[b..b + w]) {
                *d = f(x, y);
            }
        }
        (false, true) => {
            let y = src[b];
            for (d, &x) in dst.iter_mut().zip(&src[a..a + w]) {
                *d = f(x, y);
            }
        }
        (false, false) => {
            for ((d, &x), &y) in dst.iter_mut().zip(&src[a..a + w]).zip(&src[b..b + w]) {
                *d = f(x, y);
            }
        }
    }
}

#[inline(always)]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

impl Bound {
    /// Bind `prog` to data columns (each of length `m`); `n_params` checks parameter indices.
    pub fn new(prog: &Program, vars: &[Vec<f64>], m: usize, n_params: usize) -> Result<Bound, String> {
        let n = prog.ops.len();
        let mut width = vec![1usize; n];
        let mut is_dyn = vec![false; n];
        for (i, op) in prog.ops.iter().enumerate() {
            match *op {
                Op::Var(j) => {
                    if j as usize >= vars.len() {
                        return Err(format!("var {j} but only {} data columns", vars.len()));
                    }
                    width[i] = m;
                }
                Op::Param(j) => {
                    if j as usize >= n_params {
                        return Err(format!("param {j} out of range (p = {n_params})"));
                    }
                    is_dyn[i] = true;
                }
                _ => {
                    let (a, b) = op.args();
                    for r in [a, b].into_iter().flatten() {
                        width[i] = width[i].max(width[r as usize]);
                        is_dyn[i] |= is_dyn[r as usize];
                    }
                }
            }
        }
        let mut off = Vec::with_capacity(n);
        let mut total = 0usize;
        for &w in &width {
            off.push(total);
            total += w;
        }
        let mut bound = Bound {
            ops: prog.ops.clone(),
            outputs: prog.outputs.clone(),
            off,
            width,
            dynamic: (0..n as u32).filter(|&i| is_dyn[i as usize]).collect(),
            buf: vec![0.0; total],
            m,
        };
        for i in 0..n {
            if !is_dyn[i] {
                match prog.ops[i] {
                    Op::Var(j) => {
                        let o = bound.off[i];
                        bound.buf[o..o + m].copy_from_slice(&vars[j as usize][..m]);
                    }
                    _ => bound.exec(i, &[]),
                }
            }
        }
        for o in &bound.outputs {
            if o.len() != 1 && o.len() != m {
                return Err(format!("output has {} registers, expected 1 or {m}", o.len()));
            }
            if o.len() > 1 && o.iter().any(|&r| bound.width[r as usize] != 1) {
                return Err("row-wise outputs must be scalar registers".into());
            }
        }
        Ok(bound)
    }

    pub fn n_dynamic_ops(&self) -> usize {
        self.dynamic.len()
    }

    /// Scalar-equivalent work of one evaluation (sum of dynamic register widths).
    pub fn work(&self) -> usize {
        self.dynamic.iter().map(|&i| self.width[i as usize]).sum()
    }

    #[inline]
    fn exec(&mut self, i: usize, theta: &[f64]) {
        use Op::*;
        let o = self.off[i];
        let w = self.width[i];
        let at = |r: u32| (self.off[r as usize], self.width[r as usize]);
        let buf = &mut self.buf;
        match self.ops[i] {
            Const(c) => buf[o] = c,
            Var(_) => {}
            Param(j) => buf[o] = theta[j as usize],
            Add(a, b) => binary(buf, o, at(a), at(b), w, |x, y| x + y),
            Sub(a, b) => binary(buf, o, at(a), at(b), w, |x, y| x - y),
            Mul(a, b) => binary(buf, o, at(a), at(b), w, |x, y| x * y),
            Div(a, b) => binary(buf, o, at(a), at(b), w, |x, y| x / y),
            PowF(a, b) => binary(buf, o, at(a), at(b), w, f64::powf),
            Max(a, b) => binary(buf, o, at(a), at(b), w, f64::max),
            Min(a, b) => binary(buf, o, at(a), at(b), w, f64::min),
            ExpWErfc(a, b) => binary(buf, o, at(a), at(b), w, expwerfc),
            PowI(a, n) => {
                let a = at(a).0;
                match n {
                    2 => unary(buf, o, a, w, |x| x * x),
                    3 => unary(buf, o, a, w, |x| x * x * x),
                    -1 => unary(buf, o, a, w, |x| 1.0 / x),
                    -2 => unary(buf, o, a, w, |x| 1.0 / (x * x)),
                    _ => unary(buf, o, a, w, |x| x.powi(n)),
                }
            }
            Neg(a) => unary(buf, o, at(a).0, w, |x| -x),
            Sqrt(a) => unary(buf, o, at(a).0, w, f64::sqrt),
            Exp(a) => unary(buf, o, at(a).0, w, f64::exp),
            Log(a) => unary(buf, o, at(a).0, w, f64::ln),
            Sin(a) => unary(buf, o, at(a).0, w, f64::sin),
            Cos(a) => unary(buf, o, at(a).0, w, f64::cos),
            Tan(a) => unary(buf, o, at(a).0, w, f64::tan),
            Tanh(a) => unary(buf, o, at(a).0, w, f64::tanh),
            Sinh(a) => unary(buf, o, at(a).0, w, f64::sinh),
            Cosh(a) => unary(buf, o, at(a).0, w, f64::cosh),
            Atan(a) => unary(buf, o, at(a).0, w, f64::atan),
            Asin(a) => unary(buf, o, at(a).0, w, f64::asin),
            Acos(a) => unary(buf, o, at(a).0, w, f64::acos),
            Abs(a) => unary(buf, o, at(a).0, w, f64::abs),
            Sign(a) => unary(buf, o, at(a).0, w, sign),
            Erf(a) => unary(buf, o, at(a).0, w, libm::erf),
            Erfc(a) => unary(buf, o, at(a).0, w, libm::erfc),
            Wrap(a) => unary(buf, o, at(a).0, w, |x| x.sin().atan2(x.cos())),
            Step(a) => unary(buf, o, at(a).0, w, |x| if x >= 0.0 { 1.0 } else if x < 0.0 { 0.0 } else { f64::NAN }),
        }
    }

    /// Run the parameter-dependent ops for `theta`.
    pub fn eval(&mut self, theta: &[f64]) {
        for k in 0..self.dynamic.len() {
            let i = self.dynamic[k] as usize;
            self.exec(i, theta);
        }
    }

    /// Copy output `k` (after [`eval`](Self::eval)) into `dst` (length `m`), broadcasting scalars.
    pub fn output_into(&self, k: usize, dst: &mut [f64]) {
        let regs = &self.outputs[k];
        if regs.len() == 1 {
            let r = regs[0] as usize;
            let o = self.off[r];
            if self.width[r] == 1 {
                dst[..self.m].fill(self.buf[o]);
            } else {
                dst[..self.m].copy_from_slice(&self.buf[o..o + self.m]);
            }
        } else {
            for (d, &r) in dst.iter_mut().zip(regs) {
                *d = self.buf[self.off[r as usize]];
            }
        }
    }

    pub fn n_outputs(&self) -> usize {
        self.outputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn exp_decay_and_static_hoisting() {
        // f = b1 * exp(-b2 * x)
        let p = Program::from_json(&json!({
            "ops": [["param",0],["param",1],["var",0],["mul",1,2],["neg",3],["exp",4],["mul",0,5],
                    ["const",2.0],["var",0],["mul",7,8]],
            "outputs": [[6],[9],[0]]
        }))
        .unwrap();
        let x = vec![0.0, 1.0, 2.0];
        let mut b = Bound::new(&p, &[x.clone()], 3, 2).unwrap();
        b.eval(&[2.0, 0.5]);
        let mut out = vec![0.0; 3];
        b.output_into(0, &mut out);
        for (i, xi) in x.iter().enumerate() {
            assert!((out[i] - 2.0 * (-0.5 * xi).exp()).abs() < 1e-15);
        }
        b.output_into(1, &mut out); // static 2*x
        assert_eq!(out, vec![0.0, 2.0, 4.0]);
        b.output_into(2, &mut out); // scalar broadcast
        assert_eq!(out, vec![2.0; 3]);
        assert_eq!(b.n_dynamic_ops(), 6);
    }
}
