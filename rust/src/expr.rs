//! Expression strings → tapes: parsing, symbolic differentiation and linear-parameter detection.
//!
//! Expressions live in a hash-consed DAG ([`Graph`]): every node is built through a constructor
//! that folds constants and drops identities (`x + 0`, `1 * x`, …), and identical nodes are
//! stored once, so the DAG is already common-subexpression-eliminated and in topological order.
//! A [`Program`] is the DAG's nodes reachable from the requested outputs.
//!
//! Grammar: numbers, names, `+ - * / ^ **`, parentheses, unary minus, and the functions
//! `exp log ln sqrt sin cos tan tanh sinh cosh atan arctan asin arcsin acos arccos abs sign
//! erf erfc`, `wrap` (angle wrapped to (−π, π], derivative 1), `step` (1 where the argument is
//! ≥ 0, else 0), and two-argument `max min pow` and `expwerfc` (exp(a)·erfc(b) without
//! overflow). `pi` is π. A name is a variable when listed as one and a parameter otherwise, in
//! order of first appearance unless the order is given.

use std::collections::HashMap;

use crate::special::expwerfc;
use crate::tape::{Op, Program};

type Id = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Un {
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Tanh,
    Sinh,
    Cosh,
    Atan,
    Asin,
    Acos,
    Abs,
    Sign,
    Erf,
    Erfc,
    Step,
    Wrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    ExpWErfc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node {
    Const(u64), // f64 bits
    Var(u32),
    Param(u32),
    Un(Un, Id),
    Bin(Bin, Id, Id),
    PowI(Id, i32),
}

fn un_f(op: Un, x: f64) -> f64 {
    match op {
        Un::Neg => -x,
        Un::Sqrt => x.sqrt(),
        Un::Exp => x.exp(),
        Un::Log => x.ln(),
        Un::Sin => x.sin(),
        Un::Cos => x.cos(),
        Un::Tan => x.tan(),
        Un::Tanh => x.tanh(),
        Un::Sinh => x.sinh(),
        Un::Cosh => x.cosh(),
        Un::Atan => x.atan(),
        Un::Asin => x.asin(),
        Un::Acos => x.acos(),
        Un::Abs => x.abs(),
        Un::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                x * 0.0
            }
        }
        Un::Erf => libm::erf(x),
        Un::Erfc => libm::erfc(x),
        Un::Step => {
            if x >= 0.0 {
                1.0
            } else if x < 0.0 {
                0.0
            } else {
                f64::NAN
            }
        }
        Un::Wrap => x.sin().atan2(x.cos()),
    }
}

fn bin_f(op: Bin, a: f64, b: f64) -> f64 {
    match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mul => a * b,
        Bin::Div => a / b,
        Bin::Pow => a.powf(b),
        Bin::Max => a.max(b),
        Bin::Min => a.min(b),
        Bin::ExpWErfc => expwerfc(a, b),
    }
}

/// A hash-consed expression DAG over `n_vars` data variables and `n_params` parameters.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    memo: HashMap<Node, Id>,
}

impl Graph {
    fn add(&mut self, n: Node) -> Id {
        if let Some(&id) = self.memo.get(&n) {
            return id;
        }
        let id = self.nodes.len() as Id;
        self.nodes.push(n);
        self.memo.insert(n, id);
        id
    }

    fn konst(&self, id: Id) -> Option<f64> {
        match self.nodes[id as usize] {
            Node::Const(b) => Some(f64::from_bits(b)),
            _ => None,
        }
    }

    fn c(&mut self, v: f64) -> Id {
        self.add(Node::Const((v + 0.0).to_bits())) // + 0.0 merges -0.0 into 0.0
    }

    fn var(&mut self, j: u32) -> Id {
        self.add(Node::Var(j))
    }

    fn param(&mut self, j: u32) -> Id {
        self.add(Node::Param(j))
    }

    fn un(&mut self, op: Un, a: Id) -> Id {
        if let Some(x) = self.konst(a) {
            return self.c(un_f(op, x));
        }
        if op == Un::Neg {
            if let Node::Un(Un::Neg, inner) = self.nodes[a as usize] {
                return inner;
            }
        }
        self.add(Node::Un(op, a))
    }

    fn neg(&mut self, a: Id) -> Id {
        self.un(Un::Neg, a)
    }

    fn bin(&mut self, op: Bin, a: Id, b: Id) -> Id {
        let (ka, kb) = (self.konst(a), self.konst(b));
        if let (Some(x), Some(y)) = (ka, kb) {
            return self.c(bin_f(op, x, y));
        }
        match op {
            Bin::Add if ka == Some(0.0) => return b,
            Bin::Add | Bin::Sub if kb == Some(0.0) => return a,
            Bin::Sub if ka == Some(0.0) => return self.neg(b),
            Bin::Sub if a == b => return self.c(0.0),
            Bin::Mul if ka == Some(0.0) || kb == Some(0.0) => return self.c(0.0),
            Bin::Mul if ka == Some(1.0) => return b,
            Bin::Mul if kb == Some(1.0) => return a,
            Bin::Mul if ka == Some(-1.0) => return self.neg(b),
            Bin::Mul if kb == Some(-1.0) => return self.neg(a),
            Bin::Div if ka == Some(0.0) => return self.c(0.0),
            Bin::Div if kb == Some(1.0) => return a,
            Bin::Div if kb == Some(-1.0) => return self.neg(a),
            Bin::Pow => {
                if let Some(e) = kb {
                    if e == e.trunc() && e.abs() <= 64.0 {
                        return self.powi(a, e as i32);
                    }
                    if e == 0.5 {
                        return self.un(Un::Sqrt, a);
                    }
                }
            }
            _ => {}
        }
        // canonical operand order for commutative ops, so a*b and b*a share a node
        let (a, b) = if matches!(op, Bin::Add | Bin::Mul | Bin::Max | Bin::Min) && b < a {
            (b, a)
        } else {
            (a, b)
        };
        self.add(Node::Bin(op, a, b))
    }

    fn powi(&mut self, a: Id, n: i32) -> Id {
        match n {
            0 => self.c(1.0),
            1 => a,
            _ => match self.konst(a) {
                Some(x) => self.c(x.powi(n)),
                None => self.add(Node::PowI(a, n)),
            },
        }
    }

    /// ∂(node)/∂(param j), memoised per call.
    fn diff(&mut self, id: Id, j: u32, memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&d) = memo.get(&id) {
            return d;
        }
        let d = match self.nodes[id as usize] {
            Node::Const(_) | Node::Var(_) => self.c(0.0),
            Node::Param(k) => self.c(if k == j { 1.0 } else { 0.0 }),
            Node::PowI(a, n) => {
                let da = self.diff(a, j, memo);
                let p = self.powi(a, n - 1);
                let k = self.c(n as f64);
                let kp = self.bin(Bin::Mul, k, p);
                self.bin(Bin::Mul, kp, da)
            }
            Node::Un(op, a) => {
                let da = self.diff(a, j, memo);
                if self.konst(da) == Some(0.0) {
                    da
                } else {
                    let g = self.un_deriv(op, a, id);
                    self.bin(Bin::Mul, g, da)
                }
            }
            Node::Bin(op, a, b) => {
                let da = self.diff(a, j, memo);
                let db = self.diff(b, j, memo);
                self.bin_deriv(op, a, b, da, db, id)
            }
        };
        memo.insert(id, d);
        d
    }

    /// d op(a) / da, given the node `f` = op(a).
    fn un_deriv(&mut self, op: Un, a: Id, f: Id) -> Id {
        let one = self.c(1.0);
        match op {
            Un::Neg => self.c(-1.0),
            Un::Sqrt => {
                let two = self.c(2.0);
                let t = self.bin(Bin::Mul, two, f);
                self.bin(Bin::Div, one, t)
            }
            Un::Exp => f,
            Un::Log => self.bin(Bin::Div, one, a),
            Un::Sin => self.un(Un::Cos, a),
            Un::Cos => {
                let s = self.un(Un::Sin, a);
                self.neg(s)
            }
            Un::Tan => {
                let c = self.un(Un::Cos, a);
                let c2 = self.powi(c, 2);
                self.bin(Bin::Div, one, c2)
            }
            Un::Tanh => {
                let f2 = self.powi(f, 2);
                self.bin(Bin::Sub, one, f2)
            }
            Un::Sinh => self.un(Un::Cosh, a),
            Un::Cosh => self.un(Un::Sinh, a),
            Un::Atan => {
                let a2 = self.powi(a, 2);
                let d = self.bin(Bin::Add, one, a2);
                self.bin(Bin::Div, one, d)
            }
            Un::Asin | Un::Acos => {
                let a2 = self.powi(a, 2);
                let d = self.bin(Bin::Sub, one, a2);
                let s = self.un(Un::Sqrt, d);
                let r = self.bin(Bin::Div, one, s);
                if op == Un::Asin {
                    r
                } else {
                    self.neg(r)
                }
            }
            Un::Abs => self.un(Un::Sign, a),
            Un::Sign | Un::Step => self.c(0.0),
            Un::Wrap => self.c(1.0),
            Un::Erf | Un::Erfc => {
                let k = self.c(if op == Un::Erf { 1.0 } else { -1.0 } * 2.0 / std::f64::consts::PI.sqrt());
                let a2 = self.powi(a, 2);
                let na2 = self.neg(a2);
                let e = self.un(Un::Exp, na2);
                self.bin(Bin::Mul, k, e)
            }
        }
    }

    fn bin_deriv(&mut self, op: Bin, a: Id, b: Id, da: Id, db: Id, f: Id) -> Id {
        match op {
            Bin::Add => self.bin(Bin::Add, da, db),
            Bin::Sub => self.bin(Bin::Sub, da, db),
            Bin::Mul => {
                let x = self.bin(Bin::Mul, da, b);
                let y = self.bin(Bin::Mul, a, db);
                self.bin(Bin::Add, x, y)
            }
            Bin::Div => {
                // (da - f db) / b
                let fdb = self.bin(Bin::Mul, f, db);
                let n = self.bin(Bin::Sub, da, fdb);
                self.bin(Bin::Div, n, b)
            }
            Bin::Pow => {
                // f (db ln a + b da / a)
                let x = if self.konst(db) == Some(0.0) {
                    self.c(0.0)
                } else {
                    let l = self.un(Un::Log, a);
                    self.bin(Bin::Mul, db, l)
                };
                let y = if self.konst(da) == Some(0.0) {
                    self.c(0.0)
                } else {
                    let t = self.bin(Bin::Mul, b, da);
                    self.bin(Bin::Div, t, a)
                };
                let s = self.bin(Bin::Add, x, y);
                self.bin(Bin::Mul, f, s)
            }
            Bin::ExpWErfc => {
                // ∂/∂a = f; ∂/∂b = −2/√π · exp(a − b²)
                let x = self.bin(Bin::Mul, f, da);
                let y = if self.konst(db) == Some(0.0) {
                    self.c(0.0)
                } else {
                    let k = self.c(-2.0 / std::f64::consts::PI.sqrt());
                    let b2 = self.powi(b, 2);
                    let e = self.bin(Bin::Sub, a, b2);
                    let e = self.un(Un::Exp, e);
                    let ke = self.bin(Bin::Mul, k, e);
                    self.bin(Bin::Mul, ke, db)
                };
                self.bin(Bin::Add, x, y)
            }
            Bin::Max | Bin::Min => {
                // step(a - b) picks a for max; step(b - a) for min
                let d = if op == Bin::Max { self.bin(Bin::Sub, a, b) } else { self.bin(Bin::Sub, b, a) };
                let s = self.un(Un::Step, d);
                let one = self.c(1.0);
                let t = self.bin(Bin::Sub, one, s);
                let x = self.bin(Bin::Mul, s, da);
                let y = self.bin(Bin::Mul, t, db);
                self.bin(Bin::Add, x, y)
            }
        }
    }

    /// Rebuild `id` with parameters in `zero` replaced by 0.
    fn zeroed(&mut self, id: Id, zero: &[bool], memo: &mut HashMap<Id, Id>) -> Id {
        if let Some(&r) = memo.get(&id) {
            return r;
        }
        let r = match self.nodes[id as usize] {
            Node::Param(k) if zero[k as usize] => self.c(0.0),
            Node::Const(_) | Node::Var(_) | Node::Param(_) => id,
            Node::Un(op, a) => {
                let a = self.zeroed(a, zero, memo);
                self.un(op, a)
            }
            Node::Bin(op, a, b) => {
                let a = self.zeroed(a, zero, memo);
                let b = self.zeroed(b, zero, memo);
                self.bin(op, a, b)
            }
            Node::PowI(a, n) => {
                let a = self.zeroed(a, zero, memo);
                self.powi(a, n)
            }
        };
        memo.insert(id, r);
        r
    }

    /// Parameters each node depends on (bitsets over parameter indices).
    fn depends(&self, n_params: usize) -> Vec<Vec<u64>> {
        let words = n_params.div_ceil(64).max(1);
        let mut deps: Vec<Vec<u64>> = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let mut d = vec![0u64; words];
            match *n {
                Node::Param(k) => d[k as usize / 64] |= 1 << (k % 64),
                Node::Un(_, a) | Node::PowI(a, _) => d.copy_from_slice(&deps[a as usize]),
                Node::Bin(_, a, b) => {
                    for (w, (x, y)) in d.iter_mut().zip(deps[a as usize].iter().zip(&deps[b as usize])) {
                        *w = x | y;
                    }
                }
                Node::Const(_) | Node::Var(_) => {}
            }
            deps.push(d);
        }
        deps
    }

    /// A tape computing `outputs` (one register each).
    fn program(&self, outputs: &[Id]) -> Program {
        let mut needed = vec![false; self.nodes.len()];
        let mut stack: Vec<Id> = outputs.to_vec();
        while let Some(i) = stack.pop() {
            if std::mem::replace(&mut needed[i as usize], true) {
                continue;
            }
            match self.nodes[i as usize] {
                Node::Un(_, a) | Node::PowI(a, _) => stack.push(a),
                Node::Bin(_, a, b) => stack.extend([a, b]),
                _ => {}
            }
        }
        let mut reg = vec![u32::MAX; self.nodes.len()];
        let mut ops = Vec::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if !needed[i] {
                continue;
            }
            let r = |id: Id| reg[id as usize];
            let op = match *n {
                Node::Const(b) => Op::Const(f64::from_bits(b)),
                Node::Var(j) => Op::Var(j),
                Node::Param(j) => Op::Param(j),
                Node::PowI(a, k) => Op::PowI(r(a), k),
                Node::Un(u, a) => {
                    let a = r(a);
                    match u {
                        Un::Neg => Op::Neg(a),
                        Un::Sqrt => Op::Sqrt(a),
                        Un::Exp => Op::Exp(a),
                        Un::Log => Op::Log(a),
                        Un::Sin => Op::Sin(a),
                        Un::Cos => Op::Cos(a),
                        Un::Tan => Op::Tan(a),
                        Un::Tanh => Op::Tanh(a),
                        Un::Sinh => Op::Sinh(a),
                        Un::Cosh => Op::Cosh(a),
                        Un::Atan => Op::Atan(a),
                        Un::Asin => Op::Asin(a),
                        Un::Acos => Op::Acos(a),
                        Un::Abs => Op::Abs(a),
                        Un::Sign => Op::Sign(a),
                        Un::Erf => Op::Erf(a),
                        Un::Erfc => Op::Erfc(a),
                        Un::Step => Op::Step(a),
                        Un::Wrap => Op::Wrap(a),
                    }
                }
                Node::Bin(o, a, b) => {
                    let (a, b) = (r(a), r(b));
                    match o {
                        Bin::Add => Op::Add(a, b),
                        Bin::Sub => Op::Sub(a, b),
                        Bin::Mul => Op::Mul(a, b),
                        Bin::Div => Op::Div(a, b),
                        Bin::Pow => Op::PowF(a, b),
                        Bin::Max => Op::Max(a, b),
                        Bin::Min => Op::Min(a, b),
                        Bin::ExpWErfc => Op::ExpWErfc(a, b),
                    }
                }
            };
            reg[i] = ops.len() as u32;
            ops.push(op);
        }
        Program { ops, outputs: outputs.iter().map(|&o| vec![reg[o as usize]]).collect() }
    }
}

// ---- parsing -----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Op(char), // + - * / ^ ( ) ,
}

fn lex(s: &str) -> Result<Vec<Tok>, String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit() || (ch == '.' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.') {
                i += 1;
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let mut k = i + 1;
                if k < c.len() && (c[k] == '+' || c[k] == '-') {
                    k += 1;
                }
                if k < c.len() && c[k].is_ascii_digit() {
                    i = k;
                    while i < c.len() && c[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let t: String = c[start..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number {t:?}"))?));
        } else if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(c[start..i].iter().collect()));
        } else if ch == '*' && c.get(i + 1) == Some(&'*') {
            out.push(Tok::Op('^'));
            i += 2;
        } else if "+-*/^(),".contains(ch) {
            out.push(Tok::Op(ch));
            i += 1;
        } else {
            return Err(format!("unexpected character {ch:?} at position {i}"));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<Tok>,
    pos: usize,
    g: &'a mut Graph,
    vars: &'a [String],
    params: Vec<String>,
    fixed_params: bool,
}

const FUNCS: &[(&str, Un)] = &[
    ("exp", Un::Exp),
    ("log", Un::Log),
    ("ln", Un::Log),
    ("sqrt", Un::Sqrt),
    ("sin", Un::Sin),
    ("cos", Un::Cos),
    ("tan", Un::Tan),
    ("tanh", Un::Tanh),
    ("sinh", Un::Sinh),
    ("cosh", Un::Cosh),
    ("atan", Un::Atan),
    ("arctan", Un::Atan),
    ("asin", Un::Asin),
    ("arcsin", Un::Asin),
    ("acos", Un::Acos),
    ("arccos", Un::Acos),
    ("abs", Un::Abs),
    ("sign", Un::Sign),
    ("erf", Un::Erf),
    ("erfc", Un::Erfc),
    ("wrap", Un::Wrap),
    ("step", Un::Step),
];

const BIN_FUNCS: &[(&str, Bin)] =
    &[("max", Bin::Max), ("min", Bin::Min), ("pow", Bin::Pow), ("expwerfc", Bin::ExpWErfc)];

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, ch: char) -> bool {
        if self.peek() == Some(&Tok::Op(ch)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        if self.eat(ch) {
            Ok(())
        } else {
            Err(format!("expected {ch:?}, found {}", self.found()))
        }
    }

    fn found(&self) -> String {
        match self.peek() {
            None => "end of expression".into(),
            Some(Tok::Num(v)) => format!("{v}"),
            Some(Tok::Name(n)) => format!("{n:?}"),
            Some(Tok::Op(c)) => format!("{c:?}"),
        }
    }

    // sum := term (('+'|'-') term)*
    fn sum(&mut self) -> Result<Id, String> {
        let mut acc = self.term()?;
        loop {
            if self.eat('+') {
                let t = self.term()?;
                acc = self.g.bin(Bin::Add, acc, t);
            } else if self.eat('-') {
                let t = self.term()?;
                acc = self.g.bin(Bin::Sub, acc, t);
            } else {
                return Ok(acc);
            }
        }
    }

    // term := unary (('*'|'/') unary)*
    fn term(&mut self) -> Result<Id, String> {
        let mut acc = self.unary()?;
        loop {
            if self.eat('*') {
                let t = self.unary()?;
                acc = self.g.bin(Bin::Mul, acc, t);
            } else if self.eat('/') {
                let t = self.unary()?;
                acc = self.g.bin(Bin::Div, acc, t);
            } else {
                return Ok(acc);
            }
        }
    }

    // unary := ('-'|'+') unary | power      (so -x^2 = -(x^2))
    fn unary(&mut self) -> Result<Id, String> {
        if self.eat('-') {
            let a = self.unary()?;
            return Ok(self.g.neg(a));
        }
        if self.eat('+') {
            return self.unary();
        }
        self.power()
    }

    // power := atom ('^' unary)?          (right-associative)
    fn power(&mut self) -> Result<Id, String> {
        let base = self.atom()?;
        if self.eat('^') {
            let e = self.unary()?;
            return Ok(self.g.bin(Bin::Pow, base, e));
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Id, String> {
        match self.peek().cloned() {
            Some(Tok::Num(v)) => {
                self.pos += 1;
                Ok(self.g.c(v))
            }
            Some(Tok::Op('(')) => {
                self.pos += 1;
                let e = self.sum()?;
                self.expect(')')?;
                Ok(e)
            }
            Some(Tok::Name(name)) => {
                self.pos += 1;
                if self.peek() == Some(&Tok::Op('(')) {
                    return self.call(&name);
                }
                if let Some(j) = self.vars.iter().position(|v| *v == name) {
                    return Ok(self.g.var(j as u32));
                }
                // a declared or already-seen parameter wins over `pi` and function names
                if let Some(j) = self.params.iter().position(|p| *p == name) {
                    return Ok(self.g.param(j as u32));
                }
                if name == "pi" {
                    return Ok(self.g.c(std::f64::consts::PI));
                }
                if FUNCS.iter().any(|(f, _)| *f == name) || BIN_FUNCS.iter().any(|(f, _)| *f == name) {
                    return Err(format!("{name} is a function: write {name}(...)"));
                }
                if self.fixed_params {
                    return Err(format!("unknown name {name:?}"));
                }
                self.params.push(name);
                Ok(self.g.param((self.params.len() - 1) as u32))
            }
            _ => Err(format!("expected a value, found {}", self.found())),
        }
    }

    fn call(&mut self, name: &str) -> Result<Id, String> {
        self.expect('(')?;
        let mut args = vec![self.sum()?];
        while self.eat(',') {
            args.push(self.sum()?);
        }
        self.expect(')')?;
        if let Some(&(_, op)) = BIN_FUNCS.iter().find(|(f, _)| *f == name) {
            return match args[..] {
                [a, b] => Ok(self.g.bin(op, a, b)),
                _ => Err(format!("{name} takes 2 arguments, got {}", args.len())),
            };
        }
        let op = FUNCS
            .iter()
            .find(|(f, _)| *f == name)
            .map(|&(_, u)| u)
            .ok_or_else(|| format!("unknown function {name:?}"))?;
        match args[..] {
            [a] => Ok(self.g.un(op, a)),
            _ => Err(format!("{name} takes 1 argument, got {}", args.len())),
        }
    }
}

/// A compiled model: tapes for the model, its Jacobian and its separable parts.
#[derive(Clone, Debug)]
pub struct Compiled {
    pub params: Vec<String>,
    pub variables: Vec<String>,
    /// Parameters the model is jointly linear in (eliminated by variable projection).
    pub linear: Vec<usize>,
    /// outputs: [f]
    pub f: Program,
    /// outputs: [f, ∂f/∂θ_0, …]
    pub fj: Program,
    /// outputs: [c, Φ_0, …] with f = c(β) + Σ Φ_k(β) θ_linear[k]; None without linear params
    pub sep: Option<Program>,
}

/// Parse `expr` and build its tapes. `params` fixes the parameter order (and set); by default
/// every name that is not a variable is a parameter, in order of first appearance.
pub fn compile(expr: &str, variables: &[String], params: Option<&[String]>) -> Result<Compiled, String> {
    let mut g = Graph::default();
    let toks = lex(expr)?;
    if let Some(p) = params {
        if let Some(v) = p.iter().find(|p| variables.contains(p)) {
            return Err(format!("{v:?} is both a variable and a parameter"));
        }
    }
    let mut ps = Parser {
        toks,
        pos: 0,
        g: &mut g,
        vars: variables,
        params: params.map(<[String]>::to_vec).unwrap_or_default(),
        fixed_params: params.is_some(),
    };
    let f = ps.sum()?;
    if ps.pos < ps.toks.len() {
        return Err(format!("unexpected {} after the expression", ps.found()));
    }
    let params = ps.params;
    let n = params.len();
    if n == 0 {
        return Err("the expression has no parameters".into());
    }
    let grads: Vec<Id> = (0..n as u32).map(|j| g.diff(f, j, &mut HashMap::new())).collect();

    // greedy jointly-linear set: θ_j joins if ∂f/∂θ_j depends on no parameter in the set or θ_j
    let deps = g.depends(n);
    let mut linear: Vec<usize> = Vec::new();
    for j in 0..n {
        let d = &deps[grads[j] as usize];
        let hits = |k: usize| d[k / 64] >> (k % 64) & 1 == 1;
        if !hits(j) && !linear.iter().any(|&k| hits(k)) {
            linear.push(j);
        }
    }
    let sep = if linear.is_empty() {
        None
    } else {
        let mut zero = vec![false; n];
        for &k in &linear {
            zero[k] = true;
        }
        let c = g.zeroed(f, &zero, &mut HashMap::new());
        let mut outs = vec![c];
        outs.extend(linear.iter().map(|&k| grads[k]));
        Some(g.program(&outs))
    };
    let mut fj_out = vec![f];
    fj_out.extend(&grads);
    Ok(Compiled {
        f: g.program(&[f]),
        fj: g.program(&fj_out),
        sep,
        params,
        variables: variables.to_vec(),
        linear,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Bound;

    fn eval(p: &Program, x: &[f64], theta: &[f64]) -> Vec<Vec<f64>> {
        let mut b = Bound::new(p, &[x.to_vec()], x.len(), theta.len()).unwrap();
        b.eval(theta);
        (0..b.n_outputs())
            .map(|k| {
                let mut o = vec![0.0; x.len()];
                b.output_into(k, &mut o);
                o
            })
            .collect()
    }

    fn vars() -> Vec<String> {
        vec!["x".into()]
    }

    #[test]
    fn parses_and_detects_linear_params() {
        let c = compile("a*exp(-b*x) + c", &vars(), None).unwrap();
        assert_eq!(c.params, ["a", "b", "c"]);
        assert_eq!(c.linear, [0, 2]);
        let x = [0.0, 1.0, 2.0];
        let th = [2.0, 0.5, 0.25];
        let out = eval(&c.fj, &x, &th);
        for (i, &xi) in x.iter().enumerate() {
            let e = (-0.5 * xi).exp();
            assert!((out[0][i] - (2.0 * e + 0.25)).abs() < 1e-15);
            assert!((out[1][i] - e).abs() < 1e-15);
            assert!((out[2][i] - (-2.0 * xi * e)).abs() < 1e-15);
            assert!((out[3][i] - 1.0).abs() < 1e-15);
        }
        let sep = eval(c.sep.as_ref().unwrap(), &x, &th);
        assert_eq!(sep[0], vec![0.0; 3]); // c(β) = 0
    }

    #[test]
    fn precedence_and_functions() {
        let c = compile("-a^2 + b**2*x/2 - max(x, a) + pow(x, 0.5)", &vars(), None).unwrap();
        let (x, th) = ([4.0], [3.0, 2.0]);
        let v = eval(&c.f, &x, &th)[0][0];
        assert!((v - (-9.0 + 4.0 * 4.0 / 2.0 - 4.0 + 2.0)).abs() < 1e-12);
        assert!(compile("a*", &vars(), None).is_err());
        assert!(compile("foo(x)*a", &vars(), None).is_err());
        assert!(compile("2*x", &vars(), None).is_err());
        assert!(compile("wrap*x", &vars(), None).is_err());
        let named = compile("pi*x", &vars(), Some(&["pi".to_string()])).unwrap();
        assert_eq!(named.params, ["pi"]);
        let v = eval(&named.f, &[2.0], &[3.0])[0][0];
        assert_eq!(v, 6.0); // the declared parameter, not π
    }

    #[test]
    fn gradients_match_finite_differences() {
        let exprs = [
            "a*sin(b*x + c)/(1 + d*x^2)",
            "a*tanh(b*x) + log(1 + c*x) + sqrt(d + x)",
            "erf(a*x) * erfc(b - x) + atan(c*x) * abs(d - x) + x^c",
            "exp(-((x - a)/b)^2) * cosh(c) + acos(d/(1 + d^2))",
            "a*wrap(x - b) + step(x - c)*d + expwerfc(a*x, b - x)",
        ];
        let x = [0.3, 0.7, 1.9];
        let th = [1.3, 0.7, 0.4, 0.9];
        for e in exprs {
            let c = compile(e, &vars(), None).unwrap();
            let base = eval(&c.fj, &x, &th);
            for j in 0..th.len() {
                let h = 1e-6;
                let (mut tp, mut tm) = (th, th);
                tp[j] += h;
                tm[j] -= h;
                let (fp, fm) = (eval(&c.f, &x, &tp), eval(&c.f, &x, &tm));
                for i in 0..x.len() {
                    let fd = (fp[0][i] - fm[0][i]) / (2.0 * h);
                    assert!((base[1 + j][i] - fd).abs() < 1e-6 * (1.0 + fd.abs()), "{e} ∂{j} at {i}");
                }
            }
        }
    }
}
