"""Counts-first term grammar (arm B of the 2026-09-03 expression-prior comparison).

A law is a sum of terms; a term is a product of factors: a monomial body (variables, some
raised to a typed power) and zero, one or rarely two responses, each one function of a small
linear-form argument. Every COUNT is a stated coin, drawn in order and truncated only by the
operator budget that is left, so the coins are probabilities rather than derivation counts and
the drawn operator count is realized exactly:

  roster      m response classes at 1 : 1 : 1/2 : 1/4 for m = 0..3, drawn without replacement
              by catalog class mass; functions are peers within a class
  terms       drawn one after another until the budget is spent (a '+'/'-' join costs one
              operator), so the number of terms -- and with it the number of responses --
              rises with length
  responses   r per term at 1 : 1/4 : 1/8 for r = 0, 1, 2; never the same function twice in
              one term
  amplitude   a term with responses also carries a monomial body with probability 1/2
  atoms       a monomial has a atoms at 2^-a (each extra factor one bit); an atom carries a
              typed power with probability `power_coin` (1/4)
  argument    a response argument has m operators at 2^-(m+1) (a bare variable half the time);
              it is a linear form (Mono | Mono +- Poly), or with probability `nest` (1/8) a
              whole sub-law (composition)
  division    '/' is available to the law with probability `division_coin` (else '*' only)

Leaves are returned as ``None`` and filled by the caller's leaf draw (the legacy variables +
constant rule, untouched by owner ruling); typed-slot literals are written at placement through
`slot_literal`. Signs live in literals: `neg` and `inv` are never drawn here and appear only
where canonicalization spells them.
"""
from typing import Any, Callable, Sequence
import math

import numpy as np

HALF = 0.5


class TermGrammar:
    def __init__(
        self,
        simplipy_engine: Any,
        operator_weights: dict[str, float],
        typed_slots: dict[str, Any],
        classes: dict[str, Sequence[str]],
        *,
        slot_literal: Callable[[str, np.random.Generator], str],
        roster_weights: Sequence[float] = (1.0, 1.0, 0.5, 0.25),
        response_weights: Sequence[float] = (1.0, 0.25, 0.125),
        power_coin: float = 0.25,
        nest: float = 0.125,
        division_coin: float | None = 0.5,
        term_count: str = 'until_spent',
        per_operators: int = 4,
        nmax: int = 64,
    ) -> None:
        self.arity = dict(simplipy_engine.operator_arity)
        self.classes: dict[str, list[str]] = {}
        for name, ops in classes.items():
            ops = list(ops)
            bad = [op for op in ops if op not in self.arity or self.arity[op] != 1]
            if not ops or bad:
                raise ValueError(f'term_grammar.classes[{name!r}]: needs unary operators, got {bad or ops}')
            self.classes[name] = ops
        self.class_names = list(self.classes)
        self.class_mass = [float(sum(float(operator_weights.get(op, 0)) for op in self.classes[c])) for c in self.class_names]
        if any(m <= 0 for m in self.class_mass):
            raise ValueError('term_grammar: every class needs positive catalog mass')
        self.typed_slots = typed_slots
        self.pow_ops = [op for op in ('pow', 'rootn') if op in typed_slots and op in self.arity]
        self.pow_w = [float(operator_weights.get(op, 1.0)) for op in self.pow_ops]
        self.slot_literal = slot_literal
        self.roster_weights = [float(w) for w in roster_weights]
        self.response_weights = [float(w) for w in response_weights]
        self.power_coin = float(power_coin) if self.pow_ops else 0.0
        self.nest = float(nest)
        self.division_coin = None if division_coin is None else float(division_coin)
        # 'until_spent': terms are drawn one after another until the budget is spent (the term
        # count is emergent). 'per_operators': K ~ U{1 .. 1 + n // per_operators} terms are drawn
        # FIRST, the budget is split by a uniform composition, and each term realizes its share
        # exactly by rejection on its count vector (the coins are kept, no derivation counts).
        if term_count not in ('until_spent', 'per_operators'):
            raise ValueError("term_grammar.term_count must be 'until_spent' or 'per_operators'")
        self.term_count = term_count
        self.per_operators = max(1, int(per_operators))
        self._build_tables(int(nmax))

    # ---- exact-size tables for the small pieces (linear-form arguments) ---------------------
    def _build_tables(self, nmax: int) -> None:
        # ZM[b]: monomials with b operators under the atom coins (a atoms at 2^-a, j powers at
        # p^j (1-p)^(a-j) C(a, j)); ZP[m]: linear forms Mono | Mono +- Poly at 1/2 per extra monomial.
        p = self.power_coin
        ZM = np.zeros(nmax + 1)
        for a in range(1, nmax + 2):
            for j in range(0, a + 1):
                b = a - 1 + j
                if b <= nmax:
                    ZM[b] += HALF ** a * (p ** j) * ((1 - p) ** (a - j)) * math.comb(a, j)
        ZP = np.zeros(nmax + 1)
        for m in range(nmax + 1):
            ZP[m] = ZM[m] + HALF * sum(ZM[a] * ZP[m - 1 - a] for a in range(m))
        self.ZM, self.ZP, self.nmax = ZM, ZP, nmax

    @staticmethod
    def _pick(rng: np.random.Generator, w: Sequence[float]) -> int:
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        if not s > 0:
            raise ValueError('term_grammar: empty choice')
        return int(rng.choice(len(w), p=w / s))

    def _halving(self, lo: int, hi: int, rng: np.random.Generator) -> int:
        """k on {lo..hi} at 2^-k, truncated (hi < lo returns lo)."""
        if hi <= lo:
            return lo
        return lo + self._pick(rng, [HALF ** k for k in range(hi - lo + 1)])

    def _join(self, factors: list[list], rng: np.random.Generator, ops: Sequence[str]) -> list:
        out = factors[-1]
        for f in reversed(factors[:-1]):
            out = [ops[self._pick(rng, [1.0] * len(ops))]] + f + out
        return out

    # ---- pieces --------------------------------------------------------------------------
    def _atom(self, decorated: bool, rng: np.random.Generator) -> list:
        if not decorated:
            return [None]
        op = self.pow_ops[self._pick(rng, self.pow_w)]
        children: list = [None] * self.arity[op]
        children[self.typed_slots[op]['argument']] = self.slot_literal(op, rng)
        return [op] + children

    def _mono_counts(self, budget: int, rng: np.random.Generator) -> tuple[int, int]:
        """(atoms, powers) under the coins, truncated to `budget` operators."""
        a = self._halving(1, budget + 1, rng)
        room = budget - (a - 1)
        j = int(min(room, rng.binomial(a, self.power_coin))) if self.power_coin > 0 else 0
        return a, j

    def _mono_build(self, a: int, j: int, rng: np.random.Generator, mul_ops: Sequence[str]) -> list:
        deco = np.zeros(a, dtype=bool)
        if j:
            deco[rng.choice(a, j, replace=False)] = True
        return self._join([self._atom(bool(d), rng) for d in deco], rng, mul_ops)

    def _mono_exact(self, b: int, rng: np.random.Generator, mul_ops: Sequence[str]) -> list:
        """A monomial with exactly b operators (used inside size-conditioned arguments)."""
        p = self.power_coin
        opts, lab = [], []
        for a in range(1, b + 2):
            j = b - (a - 1)
            if 0 <= j <= a and (j == 0 or p > 0):
                opts.append(HALF ** a * (p ** j) * ((1 - p) ** (a - j)) * math.comb(a, j))
                lab.append((a, j))
        a, j = lab[self._pick(rng, opts)]
        return self._mono_build(a, j, rng, mul_ops)

    def _poly(self, m: int, rng: np.random.Generator, mul_ops: Sequence[str]) -> list:
        opts, lab = [self.ZM[m]], [('one', m)]
        for a in range(m):
            opts.append(HALF * self.ZM[a] * self.ZP[m - 1 - a])
            lab.append(('more', a))
        kind, a = lab[self._pick(rng, opts)]
        if kind == 'one':
            return self._mono_exact(m, rng, mul_ops)
        sign = ['+', '-'][self._pick(rng, [1.0, 1.0])]
        return [sign] + self._mono_exact(a, rng, mul_ops) + self._poly(m - 1 - a, rng, mul_ops)

    def _argument(self, m: int, rng: np.random.Generator, st: dict[str, Any]) -> list:
        if m >= 1 and rng.random() < self.nest:
            return self._expression(m, rng, st)
        return self._poly(m, rng, st['mul_ops'])

    def _functions(self, r: int, rng: np.random.Generator, st: dict[str, Any]) -> list[str]:
        roster: list[str] = st['roster']
        chosen: list[str] = []
        for _ in range(r):
            avail = [c for c in roster if any(f not in chosen for f in self.classes[c])]
            c = avail[self._pick(rng, [1.0] * len(avail))]
            members = [f for f in self.classes[c] if f not in chosen]
            chosen.append(members[self._pick(rng, [1.0] * len(members))])
        return chosen

    def _term(self, budget: int, rng: np.random.Generator, st: dict[str, Any], exact: bool = False) -> tuple[list, int]:
        """One term costing at most `budget` (>= 1) operators; returns (tokens, cost). With `exact`
        the leftover budget is absorbed by the last response's argument (or by extra body atoms),
        so the term spends exactly `budget`."""
        roster_size = sum(len(self.classes[c]) for c in st['roster'])
        w, rs = [], []
        for r, wr in enumerate(self.response_weights):
            if r > roster_size:
                break
            if r == 0 or 2 * r - 1 <= budget:        # r functions + (r - 1) joins, no body
                w.append(wr)
                rs.append(r)
        r = rs[self._pick(rng, w)]
        body = True if r == 0 else (2 * r <= budget and rng.random() < HALF)
        n_factors = r + int(body)
        left = budget - r - (n_factors - 1)
        a = j = 0
        if body:
            a, j = self._mono_counts(left, rng)
            left -= (a - 1) + j
        funcs = self._functions(r, rng, st)
        ms: list[int] = []
        for _ in funcs:
            m = self._halving(0, left, rng)
            left -= m
            ms.append(m)
        if exact and left > 0:
            if ms:
                ms[-1] += left
            else:
                a += left
            left = 0
        factors: list[list] = []
        if body:
            factors.append(self._mono_build(a, j, rng, st['mul_ops']))
        for f, m in zip(funcs, ms):
            factors.append([f] + self._argument(m, rng, st))
        order = rng.permutation(len(factors))
        return self._join([factors[i] for i in order], rng, st['mul_ops']), budget - left

    @staticmethod
    def _composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
        if parts == 1:
            return [total]
        bars = np.sort(rng.choice(total + parts - 1, parts - 1, replace=False))
        edges = np.concatenate([[-1], bars, [total + parts - 1]])
        return [int(edges[i + 1] - edges[i] - 1) for i in range(parts)]

    def _term_exact(self, share: int, rng: np.random.Generator, st: dict[str, Any]) -> list:
        """A term costing exactly `share` operators (the coins, leftover absorbed)."""
        if share == 0:
            return [None]
        tokens, cost = self._term(share, rng, st, exact=True)
        assert cost == share, (share, cost)
        return tokens

    def _expression(self, n: int, rng: np.random.Generator, st: dict[str, Any]) -> list:
        if n == 0:
            return [None]
        if self.term_count == 'per_operators':
            K = int(rng.integers(1, 1 + n // self.per_operators + 1))
            budget = n - (K - 1)
            shares = [1 + p for p in self._composition(budget - K, K, rng)] if budget >= K else self._composition(budget, K, rng)
            terms = [self._term_exact(s, rng, st) for s in shares]
            order = rng.permutation(K)
            return self._join([terms[i] for i in order], rng, ['+', '-'])
        left = n
        terms: list[list] = []
        while True:
            tokens, cost = self._term(left, rng, st)
            terms.append(tokens)
            left -= cost
            if left == 0:
                break
            left -= 1                                  # the join to the next term
            if left == 0:
                terms.append([None])                   # a bare-leaf term (an offset)
                break
        order = rng.permutation(len(terms))
        return self._join([terms[i] for i in order], rng, ['+', '-'])

    def sample(self, n_operators: int, rng: np.random.Generator) -> list:
        mmax = min(len(self.class_names), len(self.roster_weights) - 1)
        m = self._pick(rng, self.roster_weights[:mmax + 1])
        names, mass, roster = list(self.class_names), list(self.class_mass), []
        for _ in range(m):
            i = self._pick(rng, mass)
            roster.append(names.pop(i))
            mass.pop(i)
        div = self.division_coin is not None and rng.random() < self.division_coin
        st = {'roster': roster, 'mul_ops': ['*', '/'] if div else ['*']}
        tokens = self._expression(int(n_operators), rng, st)
        assert sum(1 for t in tokens if t in self.arity) == n_operators, (n_operators, tokens)
        return tokens
