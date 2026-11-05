"""Piecewise ground truths as exact formulas over floor and Abs (srbf's catalog survey, piecewise_check.py).

step(t) = 1 + floor(t / (1 + |t|)) = [t >= 0] for every real t, so
  [a <= b] = step(b - a),  [a < b] = 1 - step(a - b),  [a = b] = step(a - b) * step(b - a),
  And = product, Or = 1 - (1 - p)(1 - q), Not = 1 - p,
and Piecewise((e1, c1), (e2, c2), ..., (en, True)) = e1*[c1] + (1 - [c1]) * Piecewise((e2, c2), ...).
The rewrite multiplies the unused branch by 0, so a branch that is NaN off its own region still breaks it."""
import sympy
from sympy import Abs, And, Eq, Not, Or, Piecewise, floor


def step(t):
    return 1 + floor(t / (1 + Abs(t)))


def ind(c):
    if c is sympy.true:
        return sympy.Integer(1)
    if c is sympy.false:
        return sympy.Integer(0)
    if isinstance(c, Piecewise):   # a boolean-valued Piecewise (sympy folds 'Piecewise(...) < 2' into one)
        out = sympy.Integer(0)
        for b, cc in reversed(c.args):
            k = ind(cc)
            out = ind(b) * k + (1 - k) * out
        return out
    if isinstance(c, sympy.logic.boolalg.ITE):   # if a then b else c, as sympy writes a nested boolean switch
        k = ind(c.args[0])
        return k * ind(c.args[1]) + (1 - k) * ind(c.args[2])
    if isinstance(c, sympy.core.relational.Relational):
        c = c.func(rewrite(c.lhs), rewrite(c.rhs), evaluate=False)
    if isinstance(c, And):
        return sympy.Mul(*[ind(a) for a in c.args])
    if isinstance(c, Or):
        return 1 - sympy.Mul(*[1 - ind(a) for a in c.args])
    if isinstance(c, Not):
        return 1 - ind(c.args[0])
    if isinstance(c, Eq):
        return step(c.lhs - c.rhs) * step(c.rhs - c.lhs)
    if isinstance(c, (sympy.LessThan,)):
        return step(c.rhs - c.lhs)
    if isinstance(c, (sympy.GreaterThan,)):
        return step(c.lhs - c.rhs)
    if isinstance(c, (sympy.StrictLessThan,)):
        return 1 - step(c.lhs - c.rhs)
    if isinstance(c, (sympy.StrictGreaterThan,)):
        return 1 - step(c.rhs - c.lhs)
    raise ValueError(f"condition {c}")


def rewrite(e):
    e = e.replace(lambda n: isinstance(n, Piecewise), lambda n: _pw(n))
    return e


def _pw(n):
    pairs = [(rewrite(a), c) for a, c in n.args]
    out = pairs[-1][0] if pairs[-1][1] is sympy.true else sympy.nan
    for a, c in reversed(pairs[:-1] if pairs[-1][1] is sympy.true else pairs):
        k = ind(c)
        out = a * k + (1 - k) * out if out is not sympy.nan else a * k
    return out
