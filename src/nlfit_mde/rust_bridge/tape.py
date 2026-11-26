"""Compile SymPy expressions to the straight-line tapes evaluated by ``rust/src/tape.rs``.

The expressions are CSE'd jointly (model, Jacobian columns, separable parts share work);
each op is ``[name, args...]`` and register ``i`` holds the result of op ``i``. Identical
ops are emitted once. Unsupported SymPy constructs raise ``NotExportable``.
"""

from __future__ import annotations

import sys

import sympy as sp

from ..models.symbolic import expwerfc, step, wrap

UNARY = {
    sp.exp: "exp",
    sp.log: "log",
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.tanh: "tanh",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.atan: "atan",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.erf: "erf",
    sp.erfc: "erfc",
}


class NotExportable(ValueError):
    """The expression uses something the Rust tape cannot evaluate."""


class TapeBuilder:
    def __init__(self, psyms, vsyms):
        self.ops: list[list] = []
        self._ops_memo: dict[tuple, int] = {}
        self._memo: dict[sp.Basic, int] = {}
        for i, s in enumerate(psyms):
            self._memo[s] = self._emit("param", i)
        for j, v in enumerate(vsyms):
            self._memo[v] = self._emit("var", j)

    def _emit(self, *op) -> int:
        if op not in self._ops_memo:
            self._ops_memo[op] = len(self.ops)
            self.ops.append(list(op))
        return self._ops_memo[op]

    def const(self, v: float) -> int:
        return self._emit("const", float(v))

    def build(self, e) -> int:
        e = sp.sympify(e)
        if e in self._memo:
            return self._memo[e]
        reg = self._build(e)
        self._memo[e] = reg
        return reg

    def _build(self, e) -> int:
        if e.is_Number or isinstance(e, sp.NumberSymbol):
            v = float(e)
            if v != v or v in (float("inf"), float("-inf")):
                raise NotExportable(f"non-finite constant {e}")
            return self.const(v)
        if e.is_Symbol:
            raise NotExportable(f"undeclared symbol {e}")
        if e.is_Add:
            terms = list(e.args)
            acc = self.build(terms[0])
            for t in terms[1:]:
                if t.could_extract_minus_sign():
                    acc = self._emit("sub", acc, self.build(-t))
                else:
                    acc = self._emit("add", acc, self.build(t))
            return acc
        if e.is_Mul:
            coeff, factors = e.as_coeff_mul()
            num, den = [], []
            for f in factors:
                if f.is_Pow and f.exp.is_Number and f.exp.is_negative:
                    den.append(sp.Pow(f.base, -f.exp))
                else:
                    num.append(f)
            acc = None
            for f in num:
                r = self.build(f)
                acc = r if acc is None else self._emit("mul", acc, r)
            if coeff == -1 and acc is not None:
                acc = self._emit("neg", acc)
            elif coeff != 1 or acc is None:
                c = self.const(coeff)
                acc = c if acc is None else self._emit("mul", c, acc)
            if den:
                d = None
                for f in den:
                    r = self.build(f)
                    d = r if d is None else self._emit("mul", d, r)
                acc = self._emit("div", acc, d)
            return acc
        if e.is_Pow:
            b, x = e.args
            if x.is_Integer:
                return self._emit("powi", self.build(b), int(x))
            if x == sp.Rational(1, 2):
                return self._emit("sqrt", self.build(b))
            if x == sp.Rational(-1, 2):
                return self._emit("div", self.const(1.0), self._emit("sqrt", self.build(b)))
            if b == sp.E:
                return self._emit("exp", self.build(x))
            return self._emit("powf", self.build(b), self.build(x))
        if isinstance(e, wrap):
            return self._emit("wrap", self.build(e.args[0]))
        if isinstance(e, step):
            return self._emit("step", self.build(e.args[0]))
        if isinstance(e, expwerfc):
            return self._emit("expwerfc", self.build(e.args[0]), self.build(e.args[1]))
        if isinstance(e, sp.log) and len(e.args) == 2:
            return self._emit("div", self.build(sp.log(e.args[0])), self.build(sp.log(e.args[1])))
        if isinstance(e, sp.Max | sp.Min):
            name = "max" if isinstance(e, sp.Max) else "min"
            acc = self.build(e.args[0])
            for a in e.args[1:]:
                acc = self._emit(name, acc, self.build(a))
            return acc
        if e.func in UNARY:
            return self._emit(UNARY[e.func], self.build(e.args[0]))
        raise NotExportable(f"unsupported {type(e).__name__}: {str(e)[:80]}")


def compile_outputs(outputs: list[list[sp.Expr]], psyms, vsyms) -> dict:
    """Tape with one output per entry of ``outputs`` (each a list: 1 expression = a
    vector/scalar over the data, m expressions = one scalar per residual row)."""
    flat = [sp.sympify(e) for out in outputs for e in out]
    repl, reduced = sp.cse(flat, order="none")
    tb = TapeBuilder(psyms, vsyms)
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        for sym, sub in repl:
            tb._memo[sym] = tb.build(sub)
        regs = [tb.build(e) for e in reduced]
    finally:
        sys.setrecursionlimit(old)
    out, k = [], 0
    for o in outputs:
        out.append(regs[k : k + len(o)])
        k += len(o)
    return {"ops": tb.ops, "outputs": out}


def model_programs(rows: list[sp.Expr], psyms, vsyms, linear_idx) -> tuple[dict, dict, dict | None]:
    """(f, fj, sep) programs for a model given as residual rows (1 row = data expression).

    sep outputs [c, Φ_0..Φ_{k-1}] with c = f(α=0) and Φ_j = ∂f/∂α_j; returns None for sep if
    any Φ_j still depends on α (not actually separable).
    """
    f = compile_outputs([rows], psyms, vsyms)
    fj = compile_outputs([rows] + [[sp.diff(r, s) for r in rows] for s in psyms], psyms, vsyms)
    sep = None
    if len(linear_idx):
        alpha = [psyms[i] for i in linear_idx]
        zero = dict.fromkeys(alpha, 0)
        phis = [[sp.diff(r, a) for r in rows] for a in alpha]
        if not any(set(alpha) & e.free_symbols for col in phis for e in col):
            c = [r.subs(zero) for r in rows]
            sep = compile_outputs([c, *phis], psyms, vsyms)
    return f, fj, sep
