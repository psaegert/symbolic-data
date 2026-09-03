"""Utilities for sampling operator skeletons for generative catalogs (operator-tree skeletons).

All randomness flows through a passed ``numpy.random.Generator`` (entropy by default when not
supplied); no global ``np.random`` state. Reproducibility comes from threading one Generator
through a sampling session, not from a fixed seed.
"""
from typing import Any, Callable, Sequence

import numpy as np
from simplipy import SimpliPyEngine

from symbolic_data._generate.structure import generate_ubi_dist
from symbolic_data.prior_factory import build_iid_prior_callable, build_prior_callable

# Selection-time placeholder for "this leaf is a number". It never reaches the output:
# every occurrence is materialized independently by ``_constant_literal``. Repeating the
# symbol is how the leaf ALPHABET repeats, not a claim that two constants are equal.
_CONSTANT_SLOT = "<constant>"

#: Literals are drawn a block at a time and handed out one per site. The prior is i.i.d.
#: per literal, so a block is distributionally identical to that many single draws while
#: paying the prior's per-call overhead once instead of once per literal.
_LITERAL_BLOCK = 512


class _Slot:
    """An empty slot that remembers the behaviour classes on its root path (class-budget mode).
    Legacy mode keeps ``None`` for empties, so its RNG stream and output are byte-identical."""
    __slots__ = ("classes",)

    def __init__(self, classes: frozenset) -> None:
        self.classes = classes


def _is_empty(value: Any) -> bool:
    return value is None or isinstance(value, _Slot)


class _BlockDraw:
    """Hands out values from a prior one at a time, refilling in blocks.

    The block is drawn from the caller's own generator, and a generator it has not seen
    starts a fresh block, so the values a session sees always come from that session's
    stream. Draws left over from a previous generator are dropped, not reused.
    """

    __slots__ = ("_draw", "_size", "_values", "_cursor", "_rng")

    def __init__(self, draw: Callable[..., Any], size: int = _LITERAL_BLOCK) -> None:
        self._draw = draw
        self._size = size
        self._values: np.ndarray | None = None
        self._cursor = 0
        self._rng: np.random.Generator | None = None

    def next(self, rng: np.random.Generator) -> float:
        if self._values is None or self._cursor >= self._values.size or rng is not self._rng:
            self._values = np.asarray(self._draw(size=self._size, rng=rng),
                                      dtype=np.float64).reshape(-1)
            self._cursor = 0
            self._rng = rng
        value = float(self._values[self._cursor])
        self._cursor += 1
        return value


class SkeletonSampler:
    """Sample prefix skeletons using the configured operator priors.

    The sampler yields EXPRESSIONS, not parametric templates: a constant leaf is a
    concrete numeric literal drawn from ``literal_prior``, never a ``<constant>``
    placeholder. Abstracting literals into fittable parameters is a downstream,
    vocabulary-dependent concern (a consumer masks the literals its model cannot emit)
    and deliberately does not live here.

    ``typed_slots`` constrains a chosen argument of a chosen operator to a sampled
    LITERAL instead of a general subtree. It exists because the retired hyper-operator
    vocabulary encoded that constraint in the vocabulary itself: ``pow2``..``pow5`` and
    ``pow1_2``..``pow1_5`` were UNARY, so the exponent was always an integer in 2..5.
    With binary ``pow``/``rootn`` the exponent slot would otherwise be filled by the
    general tree sampler, which yields meaningless indices like ``rootn(x, tanh(y))``
    (``simplipy``'s ``c_rootn`` contract: a non-integer or zero index is an invalid
    operation).

    Both ``literal_prior`` and each slot's ``prior`` are ordinary prior configs, so a
    weighted datatype choice is just a mixture like any other::

        pow:
          argument: 1                     # 0-based index of the constrained argument
          prior:
            - weight: 0.5                 # integer branch
              name: choice
              kwargs: {values: [2, -2, 3, -3], weights: [8, 8, 6, 6]}
            - weight: 0.5                 # float branch, precision sampled per draw
              name: rounded
              kwargs: {base: {name: normal, kwargs: {loc: 0, scale: 2}}}
    """

    def __init__(
        self,
        simplipy_engine: SimpliPyEngine,
        sample_strategy: dict[str, Any],
        variables: list[str],
        operator_weights: dict[str, float],
        literal_prior: Callable[..., Any] | dict[str, Any] | list[dict[str, Any]] | None = None,
        typed_slots: dict[str, Any] | None = None,
        operator_profiles: list[dict[str, Any]] | None = None,
        n_unique_variables_prior: Callable[..., Any] | dict[str, Any] | list[dict[str, Any]] | None = None,
        operator_families: list[dict[str, Any]] | None = None,
        operators_per_coin: int | None = None,
        operator_subset: bool | dict[str, Any] = False,
        unary_mass: float = 1.0,
        division_coin: float | None = None,
        class_budget: dict[str, Any] | None = None,
        term_grammar: dict[str, Any] | None = None,
        nesting_decay: float | None = None,
        nesting_transparent: Sequence[str] | None = None,
        alternation_decay: float | None = None,
        alternation_mode: str = 'label',
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.sample_strategy = sample_strategy
        self.variables = variables
        self.n_variables = len(variables)
        self.operator_weights = operator_weights
        self.literal_prior = self._resolve_prior(literal_prior) if literal_prior is not None else None
        # A prior handed in as a bare callable has unknown per-call semantics, so only a
        # config-built one is blocked.
        self._literal_block = (_BlockDraw(build_iid_prior_callable(literal_prior))
                               if isinstance(literal_prior, (dict, list)) else None)
        self.typed_slots = self._validate_typed_slots(typed_slots or {})
        # Per-expression prior on the number of distinct leaf symbols (variables + the constant
        # slot). The legacy draw is UNIFORM up to the leaf count, so long expressions use most
        # of the available variables: measured on the T4 prior, 49% of delivered skeletons use
        # >= 6 distinct variables (31% use 8+) against 12% (1%) of the benchmark laws. The prior
        # is drawn per expression and truncated to [1, min(leaves, n_variables)]; `None` keeps
        # the legacy uniform draw (byte-identical).
        self.n_unique_variables_prior = (self._resolve_prior(n_unique_variables_prior)
                                         if n_unique_variables_prior is not None else None)

        self._n_leaves = 1
        # Shape dial (2026-09-03 prior comparison, arm A): the unary multiplicity the tree
        # recursion sees, against binary = leaf = 1. 1.0 is the legacy uniform shape; 0.25 is
        # "a node is one of five kinds -- four arithmetic combinations or apply-a-function".
        # Applies to the legacy per-node draw and to class-budget mode; profiles, families and
        # subsets derive their own arity masses.
        if not float(unary_mass) > 0:
            raise ValueError(f'unary_mass must be positive, got {unary_mass!r}')
        self._n_unary_operators = float(unary_mass)
        self._n_binary_operators = 1

        # STRUCTURAL slots: a slot-bearing operator exposes only its GROWING seats to the
        # tree walk (effective arity = arity - number of slots), exactly as the retired
        # hyper-operator vocabulary did -- `pow3` was unary because the exponent lived in
        # the name. The slot literal is written in at placement time, so the drawn
        # n_operators is realized EXACTLY (a grow-then-collapse approach instead deletes
        # every operator inside torn-out slot subtrees, smearing the drawn operator-count
        # distribution downward), and pow/rootn compete in the class their effective
        # arity puts them in, matching the pool dynamics the hyper-ops had.
        def _effective_arity(name: str) -> int:
            return simplipy_engine.operator_arity[name] - (1 if name in self.typed_slots else 0)

        self.unary_operators = [name for name in simplipy_engine.operator_arity if _effective_arity(name) == 1]
        self.binary_operators = [name for name in simplipy_engine.operator_arity if _effective_arity(name) == 2]
        # The full per-arity lists: `_activate_profile` narrows the working lists to the active
        # subset, so a profile built AFTER an activation (family subsets are built lazily) must
        # filter against these, not the working ones.
        self._all_unary_operators = list(self.unary_operators)
        self._all_binary_operators = list(self.binary_operators)

        self.unary_operator_probs = self._build_probability_vector(self.unary_operators)
        self.binary_operator_probs = self._build_probability_vector(self.binary_operators)

        max_operators = self.sample_strategy.get("max_operators", 10)
        self.unary_binary_distribution = generate_ubi_dist(
            max_operators,
            self._n_leaves,
            self._n_unary_operators,
            self._n_binary_operators,
        )
        # Per-EXPRESSION operator profiles (2026-09-02, measured on the T4 prior): the per-node
        # draw dilutes every expression -- with p_trans ~ 0.23 per draw and unary draws 82%
        # transcendental, 89% of delivered skeletons contain a transcendental and 0.9% are
        # single-class, against 45% / 19% for the benchmark laws. A profile is drawn ONCE per
        # expression; nodes then sample from the profile's operator subset, and the
        # unary/binary tree recursion runs on the profile's WEIGHT MASS per arity class
        # instead of the legacy fixed (1, 1) multiplicities, so a profile without unary
        # operators grows binary-only trees. `None` is the legacy sampler, byte-identical
        # (no profile draw consumes RNG).
        self._profiles = self._build_profiles(operator_profiles, max_operators)
        self._profile_probs = (np.array([p['weight'] for p in self._profiles], dtype=np.float64)
                               / sum(p['weight'] for p in self._profiles)) if self._profiles else None
        self._active_profile: dict[str, Any] | None = None
        # Per-EXPRESSION operator FAMILIES (owner ruling 2026-09-02): one independent coin per
        # family decides whether the family is available in this expression; a family with
        # p = 1 is always present (the arithmetic base). Every present family carries the same
        # total weight as the base, uniform within the family, so the arity balance is derived,
        # not tuned. The number of families is fixed per expression regardless of length --
        # the per-node draw let every long expression collect every class. Mutually exclusive
        # with `operator_profiles`; `None` is the legacy sampler (no RNG consumed).
        if operator_families and operator_profiles:
            raise ValueError("operator_families and operator_profiles are mutually exclusive")
        self._families = self._validate_families(operator_families)
        self._family_profiles: dict[tuple[int, ...], dict[str, Any]] = {}
        self._max_operators = max_operators
        # Optional: a fresh coin per block of `operators_per_coin` operators, so a longer
        # expression gets more chances to mix families (P(present | n) = 1 - (1-p)^ceil(n/m)).
        self._operators_per_coin = int(operators_per_coin) if operators_per_coin else 0
        # Per-expression operator SUBSET (owner design 2026-09-02), the operator-side twin of
        # the variable draw: how many distinct operators the expression uses is uniform up to
        # its operator count (as the distinct-variable count is uniform up to the leaf count),
        # which ones is a weighted draw without replacement from the catalog, and every node
        # then draws from that subset with the catalog weights. Short expressions come out
        # pure, long ones mix, and no expression collects every class. Exclusive with the
        # other two modes.
        if operator_subset and (operator_families or operator_profiles):
            raise ValueError("operator_subset is exclusive with operator_families / operator_profiles")
        self._operator_subset = bool(operator_subset)
        if self._operator_subset:
            # `operator_subset: {always: [...]}` keeps the listed operators in every subset (the
            # arithmetic base, say) and draws k >= 0 EXTRA operators from the rest.
            always = list(operator_subset.get('always', [])) if isinstance(operator_subset, dict) else []
            unknown = [op for op in always if op not in simplipy_engine.operator_arity]
            if unknown:
                raise ValueError(f'operator_subset.always: unknown operators {unknown}')
            self._subset_always = always
            # `usage: uniform` -- once chosen, the operators of a subset are peers at every node
            # (the variable draw's rule: identities weighted, usage uniform); default keeps the
            # catalog weights at the node draw as well.
            usage = str(operator_subset.get('usage', 'weighted')) if isinstance(operator_subset, dict) else 'weighted'
            if usage not in ('weighted', 'uniform'):
                raise ValueError("operator_subset.usage must be 'weighted' or 'uniform'")
            self._subset_usage = usage
            self._subset_ops = [op for op in self.operator_weights if op in simplipy_engine.operator_arity
                                and float(self.operator_weights[op]) > 0 and op not in always]
            w = np.array([float(self.operator_weights[op]) for op in self._subset_ops])
            self._subset_probs = w / w.sum()
            self._subset_profiles: dict[frozenset, dict[str, Any]] = {}
        # Per-expression division coin: '/' is available to an expression with this probability
        # (the binary shape mass is untouched; only the vocabulary of the binary draw changes).
        self._division_coin = None if division_coin is None else float(division_coin)
        if self._division_coin is not None and not 0.0 <= self._division_coin <= 1.0:
            raise ValueError(f'division_coin must lie in [0, 1], got {division_coin!r}')
        # Class budget (arm C): per expression m ~ U{0..min(max_classes, 1 + n // per_operators)}
        # behaviour classes are opened; at a unary node the base and each open class are PEERS
        # (peer uniform, then member: base by catalog weight, class uniform); a class never
        # appears twice on a root-to-leaf path; a class's binary members ('/') join the binary
        # pool while it is open.
        if class_budget and (operator_families or operator_profiles or operator_subset or term_grammar):
            raise ValueError('class_budget is exclusive with the other operator modes')
        self._class_budget = self._validate_class_budget(class_budget)
        # Depth-decaying unary prior (owner design 2026-09-03): a unary node in F (every unary-
        # effective operator except the transparent ones, default neg/inv) at CHAIN depth c
        # (consecutive F-ancestors directly above it, transparent operators looked through, any
        # binary node resets to 0) carries weight unary_mass * nesting_decay^c; transparent
        # unaries carry unary_mass and leave the depth unchanged. Sampled by the recursive
        # exact-size draw on the depth-indexed count table (the prefix loop cannot carry a
        # per-slot depth), so the operator count is exact and p(skeleton) stays a product.
        self._nesting: dict[str, Any] | None = None
        if nesting_decay is not None:
            if operator_families or operator_profiles or operator_subset or class_budget or term_grammar:
                raise ValueError('nesting_decay is exclusive with the other operator modes')
            gamma = float(nesting_decay)
            if not 0.0 < gamma <= 1.0:
                raise ValueError(f'nesting_decay must lie in (0, 1], got {nesting_decay!r}')
            transparent = list(nesting_transparent) if nesting_transparent is not None else ['neg', 'inv']
            unknown = [op for op in transparent if op not in self.unary_operators]
            if unknown:
                raise ValueError(f'nesting_transparent: not unary-effective operators of this engine: {unknown}')
            w = {op: float(self.operator_weights.get(op, 0)) for op in self.unary_operators}
            F = [op for op in self.unary_operators if op not in transparent and w[op] > 0]
            S = [op for op in self.unary_operators if op in transparent and w[op] > 0]
            total = sum(w.values())
            self._nesting = {
                'gamma': gamma,
                'F': F, 'pF': self._probs_from(F, w), 'mF': sum(w[op] for op in F) / total,
                'S': S, 'pS': self._probs_from(S, w), 'mS': sum(w[op] for op in S) / total,
                'table': None, 'nmax': -1, 'delta': 1.0, 'mode': 'label', 'ctable': None, 'cnmax': -1}
            # Width prior (owner design 2026-09-03, WIDTH_PRIOR.md): a binary node whose class
            # (additive {+, -} or multiplicative {*, /}) differs from its PARENT BACKBONE class
            # (nearest binary ancestor, looked through transparent unaries; a counted unary or the
            # root gives no parent class) costs alternation_decay, once per switch (the Markov
            # rule). mode 'label': the tree shape is the depth prior's exact-size draw, unchanged,
            # and delta only reweights the binary-operator draw given the parent class (delta = 1
            # is byte-identical to the plain nesting draw). mode 'shape': delta enters the
            # exact-size table as a weight, so the shape itself changes with delta.
            if alternation_decay is not None:
                delta = float(alternation_decay)
                if not 0.0 < delta <= 1.0:
                    raise ValueError(f'alternation_decay must lie in (0, 1], got {alternation_decay!r}')
                if alternation_mode not in ('label', 'shape'):
                    raise ValueError(f"alternation_mode must be 'label' or 'shape', got {alternation_mode!r}")
                cls_of = {'+': 'A', '-': 'A', '*': 'M', '/': 'M'}
                bin_class = [cls_of.get(op, op) for op in self.binary_operators]
                classes = sorted(set(bin_class))
                pb = np.asarray(self.binary_operator_probs, dtype=np.float64)
                self._nesting.update({
                    'delta': delta, 'mode': alternation_mode, 'bin_class': bin_class, 'classes': classes,
                    'mK': {k: float(pb[[i for i, kk in enumerate(bin_class) if kk == k]].sum()) for k in classes},
                    'ops_of': {k: [op for op, kk in zip(self.binary_operators, bin_class) if kk == k] for k in classes},
                    'p_of': {k: self._probs_from([op for op, kk in zip(self.binary_operators, bin_class) if kk == k],
                                                 dict(zip(self.binary_operators, pb.tolist()))) for k in classes}})
            self._nesting_table(max_operators)
            if self._nesting['mode'] == 'shape' and self._nesting['delta'] < 1.0:
                self._nesting_table_classes(max_operators)
        # Term grammar (arm B): replaces the tree walk and the per-node draw entirely.
        self._term_grammar = None
        if term_grammar:
            if operator_families or operator_profiles or operator_subset:
                raise ValueError('term_grammar is exclusive with the other operator modes')
            from symbolic_data._generate.term_grammar import TermGrammar
            spec = dict(term_grammar)
            self._term_grammar = TermGrammar(
                simplipy_engine, self.operator_weights, self.typed_slots, spec['classes'],
                slot_literal=self._slot_literal,
                roster_weights=spec.get('roster_weights', (1.0, 1.0, 0.5, 0.25)),
                response_weights=spec.get('response_weights', (1.0, 0.25, 0.125)),
                power_coin=spec.get('power_coin', 0.25),
                nest=spec.get('nest', 0.125),
                term_count=spec.get('term_count', 'until_spent'),
                per_operators=spec.get('per_operators', 4),
                division_coin=spec.get('division_coin', self._division_coin))

    def _nesting_table(self, nmax: int) -> list[list[float]]:
        """C[c][m]: total weight of trees with m operators grown under a slot of chain depth c.
        C[c][0] = 1;  C[c][m] = w1 (mF g^c C[c+1][m-1] + mS C[c][m-1]) + w2 sum_{i+j=m-1} C[0][i] C[0][j]."""
        ns = self._nesting
        assert ns is not None
        if ns['table'] is not None and ns['nmax'] >= nmax:
            return ns['table']
        w1, w2, g, mF, mS = self._n_unary_operators, self._n_binary_operators, ns['gamma'], ns['mF'], ns['mS']
        D = nmax + 2
        C = [[1.0] + [0.0] * nmax for _ in range(D + 1)]
        for m in range(1, nmax + 1):
            binary = w2 * sum(C[0][i] * C[0][m - 1 - i] for i in range(m))
            for c in range(D, -1, -1):
                C[c][m] = w1 * (mF * g ** c * C[min(c + 1, D)][m - 1] + mS * C[c][m - 1]) + binary
        ns['table'], ns['nmax'] = C, nmax
        return C

    def _nesting_table_classes(self, nmax: int) -> tuple[list[list[float]], dict[Any, int]]:
        """Shape-weight width prior: C[s][m] over states s = (c, None) for chain depth c and
        (0, k) for a parent binary class k.  C[s][0] = 1;
        C[(c,k)][m] = w1 (mF g^c C[(c+1,None)][m-1] + mS C[(c,k)][m-1])
                    + w2 sum_k' m_k' delta^[k != None and k' != k] sum_{i+j=m-1} C[(0,k')][i] C[(0,k')][j]."""
        ns = self._nesting
        assert ns is not None
        if ns['ctable'] is not None and ns['cnmax'] >= nmax:
            return ns['ctable']
        w1, w2, g, mF, mS, delta = self._n_unary_operators, self._n_binary_operators, ns['gamma'], ns['mF'], ns['mS'], ns['delta']
        D = nmax + 2
        states = [(c, None) for c in range(D + 1)] + [(0, k) for k in ns['classes']]
        idx = {st: i for i, st in enumerate(states)}
        C = [[1.0] + [0.0] * nmax for _ in states]
        for m in range(1, nmax + 1):
            pair = {k: sum(C[idx[(0, k)]][i] * C[idx[(0, k)]][m - 1 - i] for i in range(m)) for k in ns['classes']}
            for st in reversed(states):
                c, k = st
                up = w1 * (mF * g ** c * C[idx[(min(c + 1, D), None)]][m - 1] + mS * C[idx[st]][m - 1])
                binary = w2 * sum(ns['mK'][k2] * (delta if (k is not None and k2 != k) else 1.0) * pair[k2] for k2 in ns['classes'])
                C[idx[st]][m] = up + binary
        ns['ctable'], ns['cnmax'] = (C, idx), nmax
        return ns['ctable']

    def _sample_nesting(self, n_operators: int, rng: np.random.Generator) -> list[Any]:
        """Recursive exact-size draw from the depth-decaying prior (and the width prior when
        alternation_decay is set); leaves are ``None``."""
        ns = self._nesting
        assert ns is not None
        w1, w2, g, delta = self._n_unary_operators, self._n_binary_operators, ns['gamma'], ns['delta']
        shape_mode = ns['mode'] == 'shape' and delta < 1.0
        label_mode = ns['mode'] == 'label' and delta < 1.0
        if shape_mode:
            CT, idx = self._nesting_table_classes(max(n_operators, ns['cnmax']))
            D = max(c for c, k in idx if k is None)
        else:
            C = self._nesting_table(max(n_operators, ns['nmax']))
            D = len(C) - 1
        arity = self.simplipy_engine.operator_arity
        out: list[Any] = []

        def emit(operator: str, children: list[tuple[int, int, Any]]) -> None:
            slot_spec = self.typed_slots.get(operator)
            out.append(operator)
            child_iter = iter(children)
            for argument in range(arity[operator]):
                if slot_spec is not None and argument == slot_spec['argument']:
                    out.append(self._slot_literal(operator, rng))
                else:
                    grow(*next(child_iter))

        def grow(c: int, m: int, k: Any) -> None:
            # k: the parent backbone class (None under the root or a counted unary)
            if m == 0:
                out.append(None)
                return
            if shape_mode:
                st = idx[(c, None) if k is None else (0, k)]
                pF = w1 * ns['mF'] * g ** c * CT[idx[(min(c + 1, D), None)]][m - 1] if ns['F'] else 0.0
                pS = w1 * ns['mS'] * CT[st][m - 1] if ns['S'] else 0.0
                pairs = {k2: np.array([CT[idx[(0, k2)]][i] * CT[idx[(0, k2)]][m - 1 - i] for i in range(m)], dtype=np.float64)
                         for k2 in ns['classes']}
                pB = {k2: w2 * ns['mK'][k2] * (delta if (k is not None and k2 != k) else 1.0) * float(pairs[k2].sum())
                      for k2 in ns['classes']}
                r = rng.random() * (pF + pS + sum(pB.values()))
                if r < pF:
                    emit(str(rng.choice(ns['F'], p=ns['pF'])), [(min(c + 1, D), m - 1, None)])
                    return
                if r < pF + pS:
                    emit(str(rng.choice(ns['S'], p=ns['pS'])), [(c, m - 1, k)])
                    return
                r -= pF + pS
                k2 = ns['classes'][-1]
                for cand in ns['classes']:
                    if r < pB[cand]:
                        k2 = cand
                        break
                    r -= pB[cand]
                i = int(rng.choice(m, p=pairs[k2] / float(pairs[k2].sum())))
                emit(str(rng.choice(ns['ops_of'][k2], p=ns['p_of'][k2])), [(0, i, k2), (0, m - 1 - i, k2)])
                return
            pF = w1 * ns['mF'] * g ** c * C[min(c + 1, D)][m - 1] if ns['F'] else 0.0
            pS = w1 * ns['mS'] * C[c][m - 1] if ns['S'] else 0.0
            splits = np.array([w2 * C[0][i] * C[0][m - 1 - i] for i in range(m)], dtype=np.float64)
            pB = float(splits.sum())
            r = rng.random() * (pF + pS + pB)
            if r < pF:
                emit(str(rng.choice(ns['F'], p=ns['pF'])), [(min(c + 1, D), m - 1, None)])
            elif r < pF + pS:
                emit(str(rng.choice(ns['S'], p=ns['pS'])), [(c, m - 1, k)])
            else:
                i = int(rng.choice(m, p=splits / pB))
                if label_mode and k is not None:
                    pw = np.array([pb * (1.0 if kk == k else delta) for pb, kk in zip(self.binary_operator_probs, ns['bin_class'])],
                                  dtype=np.float64)
                    op = str(rng.choice(self.binary_operators, p=pw / pw.sum()))
                else:
                    op = str(rng.choice(self.binary_operators, p=self.binary_operator_probs))
                k2 = ns['bin_class'][self.binary_operators.index(op)] if (label_mode or shape_mode) else None
                emit(op, [(0, i, k2), (0, m - 1 - i, k2)])

        grow(0, int(n_operators), None)
        assert sum(1 for t in out if isinstance(t, str) and t in arity) == n_operators
        return out

    def _validate_class_budget(self, spec: dict[str, Any] | None) -> dict[str, Any] | None:
        if not spec:
            return None
        arity = self.simplipy_engine.operator_arity
        classes: dict[str, dict[str, list[str]]] = {}
        for name, ops in dict(spec.get('classes') or {}).items():
            ops = list(ops)
            unknown = [op for op in ops if op not in arity]
            if not ops or unknown:
                raise ValueError(f'class_budget.classes[{name!r}]: unknown operators {unknown or ops}')
            classes[name] = {'unary': [op for op in ops if op in self._all_unary_operators],
                             'binary': [op for op in ops if op in self._all_binary_operators]}
        base_unary = [op for op in spec.get('base_unary', []) if op in self._all_unary_operators]
        base_binary = [op for op in spec.get('base_binary', []) if op in self._all_binary_operators]
        if not base_binary:
            raise ValueError('class_budget.base_binary must not be empty')
        if not base_unary:
            raise ValueError('class_budget.base_unary must not be empty (the base is a peer at every unary node)')
        covered = set(base_unary) | set(base_binary) | {op for c in classes.values() for k in c.values() for op in k}
        uncovered = sorted(op for op, w in self.operator_weights.items() if w > 0 and op in arity and op not in covered)
        if uncovered:
            raise ValueError(f'class_budget: operators with positive operator_weight belong to no class: {uncovered}')
        return {'names': list(classes), 'classes': classes,
                'base_unary': base_unary, 'base_unary_probs': self._build_probability_vector(base_unary),
                'base_binary': base_binary,
                'per_operators': max(1, int(spec.get('per_operators', 4))),
                'max_classes': int(spec.get('max_classes', 3))}

    @staticmethod
    def _resolve_prior(prior: Any) -> Callable[..., Any]:
        """A prior config (dict, weighted list) or an already-built callable -> callable."""
        return prior if callable(prior) else build_prior_callable(prior)

    def _validate_typed_slots(self, typed_slots: dict[str, Any]) -> dict[str, Any]:
        """Fail closed on a slot spec the sampler cannot honor; resolve each slot's prior."""
        arity = self.simplipy_engine.operator_arity
        resolved: dict[str, Any] = {}
        for operator, spec in typed_slots.items():
            if operator not in arity:
                raise ValueError(f"typed_slots: unknown operator {operator!r} for this engine")
            # A UNARY slot would collapse the operator's only child, leaving the tree with
            # zero leaves -- `_get_leaves` then calls rng.integers(1, 1) and raises. A slot
            # only makes sense where a sibling subtree survives to carry the variables.
            if arity[operator] < 2:
                raise ValueError(
                    f"typed_slots[{operator!r}]: only operators of arity >= 2 can carry a "
                    f"constrained slot (got arity {arity[operator]})")
            argument = spec.get("argument")
            if not isinstance(argument, int) or not 0 <= argument < arity[operator]:
                raise ValueError(
                    f"typed_slots[{operator!r}]: 'argument' must be an int in "
                    f"[0, {arity[operator]}), got {argument!r}")
            if spec.get("prior") is None:
                raise ValueError(f"typed_slots[{operator!r}]: missing 'prior'")
            slot_prior = spec["prior"]
            resolved[operator] = {
                "argument": argument,
                "prior": self._resolve_prior(slot_prior),
                "block": (_BlockDraw(build_iid_prior_callable(slot_prior))
                          if isinstance(slot_prior, (dict, list)) else None)}
        return resolved

    def _build_profiles(self, specs: list[dict[str, Any]] | None, max_operators: int) -> list[dict[str, Any]]:
        if not specs:
            return []
        return [self._build_profile(spec, max_operators, i) for i, spec in enumerate(specs)]

    def _validate_families(self, specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not specs:
            return []
        known = set(self.simplipy_engine.operator_arity)
        out: list[dict[str, Any]] = []
        for i, spec in enumerate(specs):
            ops = list(spec.get('operators') or [])
            p = float(spec.get('p', 1.0))
            unknown = [op for op in ops if op not in known]
            if unknown:
                raise ValueError(f'operator_families[{i}]: unknown operators {unknown}')
            if not ops or not 0.0 < p <= 1.0:
                raise ValueError(f'operator_families[{i}]: needs a non-empty operators list and 0 < p <= 1')
            out.append({'name': spec.get('name', '+'.join(ops)), 'p': p, 'operators': ops})
        base = [f for f in out if f['p'] >= 1.0]
        if not base:
            raise ValueError('operator_families: at least one family needs p = 1 (the always-present base)')
        base_mass = float(sum(self.operator_weights.get(op, 0) for f in base for op in f['operators']))
        if base_mass <= 0:
            raise ValueError('operator_families: the base families carry zero operator_weight')
        # Families choose the VOCABULARY; the catalog keeps the TREE SHAPE. The optional
        # families together carry exactly the catalog mass of the operators they cover, split
        # equally among the families present and uniformly within each, so the per-arity
        # unary/binary balance the ubi recursion sees is the catalog's whatever subset is drawn.
        base_ops = {op for f in base for op in f['operators']}
        self._optional_mass = float(sum(self.operator_weights.get(op, 0) for f in out if f['p'] < 1.0
                                        for op in f['operators'] if op not in base_ops))
        # No operator is dropped by omission (owner ruling 2026-09-02): every operator the
        # catalog weights above zero must belong to some family, or it would never be drawn.
        covered = {op for f in out for op in f['operators']}
        uncovered = sorted(op for op, w in self.operator_weights.items() if w > 0 and op in known and op not in covered)
        if uncovered:
            raise ValueError(f'operator_families: operators with positive operator_weight belong to no family: {uncovered}')
        for f in out:
            f['mass'] = base_mass
        return out

    def _family_profile(self, subset: tuple[int, ...]) -> dict[str, Any]:
        """The profile for a drawn family subset (built once per subset, then cached)."""
        profile = self._family_profiles.get(subset)
        if profile is None:
            wt: dict[str, float] = {}
            names: list[str] = []
            optional = [self._families[i] for i in subset if self._families[i]['p'] < 1.0]
            for i in subset:
                f = self._families[i]
                names.append(f['name'])
                if f['p'] >= 1.0:
                    for op in f['operators']:
                        wt[op] = float(self.operator_weights.get(op, 0))
                else:
                    share = self._optional_mass / len(optional) / len(f['operators'])
                    for op in f['operators']:
                        wt[op] = share
            profile = self._build_profile({'name': '+'.join(names), 'weight': 1.0, 'operators': wt},
                                          self._max_operators, -1)
            self._family_profiles[subset] = profile
        return profile

    def _build_profile(self, spec: dict[str, Any], max_operators: int, i: int) -> dict[str, Any]:
        """One profile: operator subset, per-arity draw probabilities and the ubi table built on
        the subset's weight mass. `operators` is a list (catalog weights apply) or a
        {op: weight} mapping (profile-local weights -- a 'trig' profile with sin/cos at the
        catalog's unary weight 1 next to binaries at 10 would otherwise almost never place a
        unary node)."""
        known = set(self.simplipy_engine.operator_arity)
        raw_ops = spec.get('operators') or []
        local = dict(raw_ops) if isinstance(raw_ops, dict) else {}
        ops = list(local) if local else list(raw_ops)
        weight = float(spec.get('weight', 1.0))
        unknown = [op for op in ops if op not in known]
        if unknown:
            raise ValueError(f'operator_profiles[{i}]: unknown operators {unknown}')
        if not ops or weight <= 0:
            raise ValueError(f'operator_profiles[{i}]: needs a non-empty operators list and weight > 0')
        unary = [op for op in self._all_unary_operators if op in ops]
        binary = [op for op in self._all_binary_operators if op in ops]
        wt = {op: float(local.get(op, self.operator_weights.get(op, 0))) for op in ops}
        u = float(sum(wt[op] for op in unary))
        b = float(sum(wt[op] for op in binary))
        if u + b <= 0:
            raise ValueError(f'operator_profiles[{i}]: every listed operator has zero operator_weight')
        # Mass-weighted arity multiplicities, scaled to the legacy total (1 + 1 = 2).
        n_unary, n_binary = 2.0 * u / (u + b), 2.0 * b / (u + b)
        return {
            'name': spec.get('name', '+'.join(ops)), 'weight': weight,
            'unary_operators': unary, 'binary_operators': binary,
            'unary_probs': self._probs_from(unary, wt), 'binary_probs': self._probs_from(binary, wt),
            'n_unary': n_unary, 'n_binary': n_binary,
            'ubi': generate_ubi_dist(max_operators, self._n_leaves, n_unary, n_binary),
        }

    @staticmethod
    def _probs_from(operators: Sequence[str], weights: dict[str, float]) -> np.ndarray:
        if not operators:
            return np.zeros(0)
        p = np.array([weights[op] for op in operators], dtype=np.float64)
        return p / p.sum()

    def _activate_profile(self, profile: dict[str, Any]) -> None:
        """Point the node-level draws at ``profile`` (legacy mode never calls this)."""
        self._active_profile = profile
        self.unary_operators = profile['unary_operators']
        self.binary_operators = profile['binary_operators']
        self.unary_operator_probs = profile['unary_probs']
        self.binary_operator_probs = profile['binary_probs']
        self._n_unary_operators = profile['n_unary']
        self._n_binary_operators = profile['n_binary']
        self.unary_binary_distribution = profile['ubi']

    def _build_probability_vector(self, operators: Sequence[str]) -> np.ndarray:
        probs = np.array([self.operator_weights.get(op, 0) for op in operators], dtype=np.float64)
        return probs / probs.sum()

    @staticmethod
    def _format_literal(value: Any) -> str:
        """Render a sampled number as a token.

        An integral value always takes the INTEGER spelling (``2``, not ``2.0``): the
        two denote the same exact value, ``simplipy``'s description-length measure is
        spelling-free, and the integer spelling is the one a downstream numeric
        vocabulary is likely to contain.
        """
        number = float(value)
        return str(int(number)) if number.is_integer() else repr(number)

    def _slot_literal(self, operator: str, rng: np.random.Generator) -> str:
        """Draw one literal for ``operator``'s constrained slot."""
        slot = self.typed_slots[operator]
        if slot["block"] is not None:
            return self._format_literal(slot["block"].next(rng))
        return self._format_literal(np.atleast_1d(slot["prior"](size=1, rng=rng))[0])

    def _constant_literal(self, rng: np.random.Generator) -> str:
        """Draw one literal for a constant LEAF (each occurrence is independent)."""
        if self.literal_prior is None:
            raise ValueError("a constant leaf was sampled but no 'literal_prior' is configured")
        if self._literal_block is not None:
            return self._format_literal(self._literal_block.next(rng))
        return self._format_literal(np.atleast_1d(self.literal_prior(size=1, rng=rng))[0])

    def _sample_next_pos_ubi(self, n_empty_nodes: int, n_operators: int, rng: np.random.Generator) -> tuple[int, int]:
        if n_empty_nodes >= len(self.unary_binary_distribution):
            self.unary_binary_distribution = generate_ubi_dist(
                n_empty_nodes + 1,
                self._n_leaves,
                self._n_unary_operators,
                self._n_binary_operators,
            )

        probs: list[float] = []
        for index in range(n_empty_nodes):
            probs.append(
                (self._n_leaves ** index)
                * self._n_unary_operators
                * self.unary_binary_distribution[n_empty_nodes - index][n_operators - 1]
            )
        for index in range(n_empty_nodes):
            probs.append(
                (self._n_leaves ** index)
                * self._n_binary_operators
                * self.unary_binary_distribution[n_empty_nodes - index + 1][n_operators - 1]
            )

        probabilities_list = [value / self.unary_binary_distribution[n_empty_nodes][n_operators] for value in probs]
        probabilities = np.array(probabilities_list, dtype=np.float64)

        event = rng.choice(2 * n_empty_nodes, p=probabilities)

        arity = 1 if event < n_empty_nodes else 2
        position = event % n_empty_nodes

        return position, arity

    def _get_leaves(self, t_leaves: int, rng: np.random.Generator) -> list[str]:
        cap = min(t_leaves, self.n_variables)
        if self.n_unique_variables_prior is None:
            n_unique_variables = rng.integers(1, cap + 1)
        else:
            drawn = float(np.atleast_1d(self.n_unique_variables_prior(size=1, rng=rng))[0])
            n_unique_variables = int(min(cap, max(1, round(drawn))))
        unique_variables = rng.choice(self.variables + [_CONSTANT_SLOT], n_unique_variables, replace=False)

        guaranteed_part = unique_variables.copy()
        remaining_part = rng.choice(unique_variables, t_leaves - n_unique_variables, replace=True)
        all_allowed_variables = np.concatenate([guaranteed_part, remaining_part])
        rng.shuffle(all_allowed_variables)

        return [self._constant_literal(rng) if leaf == _CONSTANT_SLOT else str(leaf)
                for leaf in all_allowed_variables.tolist()]

    def sample(self, n_operators: int, rng: np.random.Generator | None = None) -> list[str]:
        rng = rng if rng is not None else np.random.default_rng()
        if self._nesting is not None:
            stack = self._sample_nesting(n_operators, rng)
            leaves = self._get_leaves(t_leaves=sum(1 for value in stack if value is None), rng=rng)
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] is None:
                    stack[index] = leaves.pop()
            assert len(leaves) == 0
            return stack  # type: ignore[return-value]
        if self._term_grammar is not None:
            stack = self._term_grammar.sample(n_operators, rng)
            leaves = self._get_leaves(t_leaves=sum(1 for value in stack if value is None), rng=rng)
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] is None:
                    stack[index] = leaves.pop()
            assert len(leaves) == 0
            return stack  # type: ignore[return-value]
        if self._operator_subset:
            lo = 0 if self._subset_always else 1
            k = int(rng.integers(lo, min(n_operators, len(self._subset_ops)) + 1)) if n_operators > 0 else lo
            extra = rng.choice(self._subset_ops, size=k, replace=False, p=self._subset_probs).tolist() if k else []
            chosen = frozenset(self._subset_always + extra)
            profile = self._subset_profiles.get(chosen)
            if profile is None:
                node_w = {op: (1.0 if self._subset_usage == 'uniform' else float(self.operator_weights[op])) for op in chosen}
                profile = self._build_profile({'name': '+'.join(sorted(chosen)), 'weight': 1.0, 'operators': node_w},
                                              self._max_operators, -1)
                if len(self._subset_profiles) < 4096:
                    self._subset_profiles[chosen] = profile
            self._activate_profile(profile)
        elif self._families:
            # One coin per optional family, in config order (the base families need none).
            flips = max(1, -(-n_operators // self._operators_per_coin)) if self._operators_per_coin else 1
            subset = tuple(i for i, f in enumerate(self._families)
                           if f['p'] >= 1.0 or any(rng.random() < f['p'] for _ in range(flips)))
            self._activate_profile(self._family_profile(subset))
        elif self._profiles:
            self._activate_profile(self._profiles[int(rng.choice(len(self._profiles), p=self._profile_probs))])
        binary_operators, binary_probs = self.binary_operators, self.binary_operator_probs
        budget = self._class_budget
        if budget is not None:
            cap = min(budget['max_classes'], 1 + n_operators // budget['per_operators'])
            m = int(rng.integers(0, cap + 1))
            names = budget['names']
            open_classes = [names[int(i)] for i in rng.choice(len(names), m, replace=False)] if m else []
            binary_operators = list(budget['base_binary']) + [op for c in open_classes for op in budget['classes'][c]['binary']]
            binary_probs = self._build_probability_vector(binary_operators)
            unary_peers = [c for c in open_classes if budget['classes'][c]['unary']]
        if self._division_coin is not None and '/' in binary_operators and not rng.random() < self._division_coin:
            keep = [op != '/' for op in binary_operators]
            binary_probs = binary_probs[np.array(keep)]
            binary_probs = binary_probs / binary_probs.sum()
            binary_operators = [op for op, k in zip(binary_operators, keep) if k]
        stack: list[Any] = [_Slot(frozenset())] if budget is not None else [None]
        n_empty_nodes = 1
        left_leaves = 0
        total_leaves = 1

        for remaining in range(n_operators, 0, -1):
            position, arity = self._sample_next_pos_ubi(n_empty_nodes, remaining, rng)
            insert_index = [index for index, value in enumerate(stack) if _is_empty(value)][left_leaves + position]
            child_classes: frozenset | None = None
            if arity == 1:
                if budget is not None:
                    ctx = stack[insert_index]
                    peers = ['<base>'] + [c for c in unary_peers if c not in ctx.classes]
                    peer = peers[int(rng.integers(len(peers)))]
                    if peer == '<base>':
                        operator = rng.choice(budget['base_unary'], p=budget['base_unary_probs'])
                        child_classes = ctx.classes
                    else:
                        members = budget['classes'][peer]['unary']
                        operator = members[int(rng.integers(len(members)))]
                        child_classes = ctx.classes | {peer}
                else:
                    operator = rng.choice(self.unary_operators, p=self.unary_operator_probs)
            else:
                operator = rng.choice(binary_operators, p=binary_probs)
                if budget is not None:
                    child_classes = stack[insert_index].classes

            operator = str(operator)
            slot_spec = self.typed_slots.get(operator)
            true_arity = self.simplipy_engine.operator_arity[operator]
            # Growing seats only: the ubi bookkeeping runs on the EFFECTIVE arity, and the
            # slot literal is written in right here, never grown as a subtree.
            growing_arity = true_arity - (1 if slot_spec is not None else 0)
            n_empty_nodes += growing_arity - 1 - position
            total_leaves += growing_arity - 1
            left_leaves += position

            empty: Any = _Slot(child_classes) if budget is not None else None
            children: list[Any] = [empty] * true_arity
            if slot_spec is not None:
                children[slot_spec["argument"]] = self._slot_literal(operator, rng)

            stack = (
                stack[:insert_index]
                + [operator]
                + children
                + stack[insert_index + 1:]
            )

        assert len([1 for value in stack if isinstance(value, str) and value in self.simplipy_engine.operator_arity]) == n_operators
        assert len([1 for value in stack if _is_empty(value)]) == total_leaves

        leaves = self._get_leaves(t_leaves=total_leaves, rng=rng)
        assert len(leaves) == total_leaves, f"Expected {total_leaves} leaves, got {len(leaves)}"

        for index in range(len(stack) - 1, -1, -1):
            if _is_empty(stack[index]):
                stack = stack[:index] + [leaves.pop()] + stack[index + 1:]
        assert len(leaves) == 0

        if self._active_profile is not None:
            self._active_profile['ubi'] = self.unary_binary_distribution  # keep any table growth

        return stack  # type: ignore[return-value]
