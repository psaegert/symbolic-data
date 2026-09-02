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
        self._n_unary_operators = 1
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
            for i in subset:
                f = self._families[i]
                names.append(f['name'])
                if f['p'] >= 1.0:
                    for op in f['operators']:
                        wt[op] = float(self.operator_weights.get(op, 0))
                else:
                    share = f['mass'] / len(f['operators'])
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
        if self._families:
            # One coin per optional family, in config order (the base families need none).
            subset = tuple(i for i, f in enumerate(self._families)
                           if f['p'] >= 1.0 or rng.random() < f['p'])
            self._activate_profile(self._family_profile(subset))
        elif self._profiles:
            self._activate_profile(self._profiles[int(rng.choice(len(self._profiles), p=self._profile_probs))])
        stack: list[str | None] = [None]
        n_empty_nodes = 1
        left_leaves = 0
        total_leaves = 1

        for remaining in range(n_operators, 0, -1):
            position, arity = self._sample_next_pos_ubi(n_empty_nodes, remaining, rng)
            if arity == 1:
                operator = rng.choice(self.unary_operators, p=self.unary_operator_probs)
            else:
                operator = rng.choice(self.binary_operators, p=self.binary_operator_probs)

            operator = str(operator)
            slot_spec = self.typed_slots.get(operator)
            true_arity = self.simplipy_engine.operator_arity[operator]
            # Growing seats only: the ubi bookkeeping runs on the EFFECTIVE arity, and the
            # slot literal is written in right here, never grown as a subtree.
            growing_arity = true_arity - (1 if slot_spec is not None else 0)
            n_empty_nodes += growing_arity - 1 - position
            total_leaves += growing_arity - 1
            left_leaves += position

            children: list[str | None] = [None] * true_arity
            if slot_spec is not None:
                children[slot_spec["argument"]] = self._slot_literal(operator, rng)

            insert_index = [index for index, value in enumerate(stack) if value is None][left_leaves]
            stack = (
                stack[:insert_index]
                + [operator]
                + children
                + stack[insert_index + 1:]
            )

        assert len([1 for value in stack if value in self.simplipy_engine.operator_arity]) == n_operators
        assert len([1 for value in stack if value is None]) == total_leaves

        leaves = self._get_leaves(t_leaves=total_leaves, rng=rng)
        assert len(leaves) == total_leaves, f"Expected {total_leaves} leaves, got {len(leaves)}"

        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is None:
                stack = stack[:index] + [leaves.pop()] + stack[index + 1:]
        assert len(leaves) == 0

        if self._active_profile is not None:
            self._active_profile['ubi'] = self.unary_binary_distribution  # keep any table growth

        return stack  # type: ignore[return-value]
