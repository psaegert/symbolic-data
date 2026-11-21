"""Readable infix for simplified prefix expressions: no parentheses inside product chains, signs pulled to the front of
products and quotients (-(x1*x2) and x1*(-x2) print as -x1 * x2), parentheses only where they change the meaning."""
from fractions import Fraction
import struct_scores as S

def _exp(tok):
    try:
        f = Fraction(float(tok)).limit_denominator(12)
        if abs(float(f) - float(tok)) < 1e-9: return str(f)
    except Exception: pass
    return tok

def pretty(tokens):
    def par(x, p): return f"({x[0]})" if x[1] < p else x[0]
    def mat(x):
        """Show a pulled-out sign: -x for products and atoms, -(...) for sums."""
        return (("-" + (f"({x[0]})" if x[1] < 2 else x[0])), 2.5, 1) if x[2] < 0 else x
    def T(n):                                                   # -> (string without its sign, precedence, sign)
        t, kids = n
        if not kids:
            if t.startswith("-") and t[1:2].isdigit(): return (t[1:], 9, -1)
            return (t, 9, 1)
        if t == "neg": s, p, g = T(kids[0]); return (s, p, -g)
        if t in ("*", "/", "inv"):
            # the whole * / inv chain as numerator / denominator, signs pulled to the front
            num, den, sign = [], [], 1
            def walk(m, up):
                nonlocal sign
                u, ks = m
                if u == "*": walk(ks[0], up); walk(ks[1], up)
                elif u == "/": walk(ks[0], up); walk(ks[1], not up)
                elif u == "inv": walk(ks[0], not up)
                else:
                    x = T(m); sign *= x[2]
                    if not (x[0] == "1" and x[1] == 9): (num if up else den).append(x)
            walk(n, True)
            top = " * ".join(par(x, 2) for x in num) or "1"
            if not den: return (top, 2 if len(num) > 1 else (num[0][1] if num else 9), sign)
            bottom = par(den[0], 2.1) if len(den) == 1 else "(" + " * ".join(par(x, 2) for x in den) + ")"
            return (f"{par((top, 2 if len(num) > 1 else (num[0][1] if num else 9)), 2)} / {bottom}", 2, sign)
        if t in ("+", "-"):
            a, b = T(kids[0]), T(kids[1]); sb = -b[2] if t == "-" else b[2]
            return (f"{mat(a)[0]} {'-' if sb < 0 else '+'} {par(b, 1.5 if sb < 0 else 1)}", 1, 1)
        if t in ("pow", "rootn"):
            e = kids[1]
            ex = _exp(e[0]) if not e[1] else mat(T(e))[0].replace(" / ", "/")
            return (f"{t}({mat(T(kids[0]))[0]}, {ex})", 9, 1)
        return (f"{t}({mat(T(kids[0]))[0]})", 9, 1)
    return mat(T(S.parse(tokens)))[0]

if __name__ == "__main__":
    # numeric check against the prefix expressions on random positive inputs
    import json, gzip, math, random
    import numpy as np
    F = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
         "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
         "exp": math.exp, "log": math.log, "abs": abs, "pow": lambda a, b: a ** b, "rootn": lambda a, k: a ** (1 / k)}
    def ev(n, env):
        t, k = n
        if not k: return env[t] if S.is_var(t) else float(t)
        if t in ("pow", "rootn"): return F[t](ev(k[0], env), ev(k[1], env))
        if t == "neg": return -ev(k[0], env)
        if t == "inv": return 1 / ev(k[0], env)
        if len(k) == 2:
            a, b = ev(k[0], env), ev(k[1], env)
            return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[t]
        return F[t](ev(k[0], env))
    rows = json.load(gzip.open("rows_r10c.json.gz", "rt"))["r10c"] + json.load(gzip.open("rows_ladder.json.gz", "rt"))["t8"]
    rnd = random.Random(0); checked = bad = 0
    for r in rows:
        env = {f"x{i}": rnd.uniform(0.2, 0.9) for i in range(1, 18)}
        try: want = ev(S.parse(r["simp"]), env)
        except Exception: continue
        if not (isinstance(want, float) and math.isfinite(want)): continue
        s = pretty(r["simp"])
        got = eval(s.replace("^", "**"), dict(F, **env))
        checked += 1
        if not (isinstance(got, (int, float)) and math.isclose(got, want, rel_tol=1e-7, abs_tol=1e-9)): bad += 1; print("MISMATCH", s, got, want)
    print(f"checked {checked}, mismatches {bad}")
