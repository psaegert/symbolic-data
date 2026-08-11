"""Utilities for sampling operator skeletons for generative catalogs (operator-tree skeletons).

All randomness flows through a passed ``numpy.random.Generator`` (entropy by default when not
supplied); no global ``np.random`` state. Reproducibility comes from threading one Generator
through a sampling session, not from a fixed seed.
"""
from typing import Any, Callable, Sequence

import numpy as np
from simplipy import SimpliPyEngine

from symbolic_data._generate.structure import generate_ubi_dist
from symbolic_data.prior_factory import build_prior_callable

# Selection-time placeholder for "this leaf is a number". It never reaches the output:
# every occurrence is materialized independently by ``_constant_literal``. Repeating the
# symbol is how the leaf ALPHABET repeats, not a claim that two constants are equal.
_CONSTANT_SLOT = "<constant>"


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
    ) -> None:
        self.simplipy_engine = simplipy_engine
        self.sample_strategy = sample_strategy
        self.variables = variables
        self.n_variables = len(variables)
        self.operator_weights = operator_weights
        self.literal_prior = self._resolve_prior(literal_prior) if literal_prior is not None else None
        self.typed_slots = self._validate_typed_slots(typed_slots or {})

        self._n_leaves = 1
        self._n_unary_operators = 1
        self._n_binary_operators = 1

        self.unary_operators = [name for name, arity in simplipy_engine.operator_arity.items() if arity == 1]
        self.binary_operators = [name for name, arity in simplipy_engine.operator_arity.items() if arity == 2]

        self.unary_operator_probs = self._build_probability_vector(self.unary_operators)
        self.binary_operator_probs = self._build_probability_vector(self.binary_operators)

        max_operators = self.sample_strategy.get("max_operators", 10)
        self.unary_binary_distribution = generate_ubi_dist(
            max_operators,
            self._n_leaves,
            self._n_unary_operators,
            self._n_binary_operators,
        )

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
            resolved[operator] = {"argument": argument,
                                  "prior": self._resolve_prior(spec["prior"])}
        return resolved

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
        draw = self.typed_slots[operator]["prior"](size=1, rng=rng)
        return self._format_literal(np.atleast_1d(draw)[0])

    def _constant_literal(self, rng: np.random.Generator) -> str:
        """Draw one literal for a constant LEAF (each occurrence is independent)."""
        if self.literal_prior is None:
            raise ValueError("a constant leaf was sampled but no 'literal_prior' is configured")
        return self._format_literal(np.atleast_1d(self.literal_prior(size=1, rng=rng))[0])

    def _subtree_end(self, stack: Sequence[Any], index: int) -> int:
        """Index just past the subtree rooted at ``index`` (``None`` counts as a leaf)."""
        token = stack[index]
        if token is None:
            return index + 1
        end = index + 1
        for _ in range(self.simplipy_engine.operator_arity.get(token, 0)):
            end = self._subtree_end(stack, end)
        return end

    def _collapse_typed_slots(self, stack: list[Any], rng: np.random.Generator) -> list[Any]:
        """Replace every constrained argument subtree with a sampled literal.

        Applied AFTER the structure walk so the Lample-Charton tree-counting
        invariants (``unary_binary_distribution``, the empty-node bookkeeping) are
        untouched; the collapse only shortens what the sampler already produced.
        """
        out: list[Any] = []

        def rebuild(index: int) -> int:
            token = stack[index]
            if token is None:
                out.append(None)
                return index + 1
            out.append(token)
            spec = self.typed_slots.get(token)
            end = index + 1
            for argument in range(self.simplipy_engine.operator_arity.get(token, 0)):
                if spec is not None and argument == spec["argument"]:
                    end = self._subtree_end(stack, end)      # discard the sampled subtree
                    out.append(self._slot_literal(token, rng))
                else:
                    end = rebuild(end)
            return end

        consumed = rebuild(0)
        # Not an `assert`: `python -O` strips those, and a malformed walk would then
        # silently emit a TRUNCATED expression rather than failing.
        if consumed != len(stack):
            raise RuntimeError(
                f"typed-slot collapse consumed {consumed} of {len(stack)} tokens")
        return out

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
        n_unique_variables = rng.integers(1, min(t_leaves, self.n_variables) + 1)
        unique_variables = rng.choice(self.variables + [_CONSTANT_SLOT], n_unique_variables, replace=False)

        guaranteed_part = unique_variables.copy()
        remaining_part = rng.choice(unique_variables, t_leaves - n_unique_variables, replace=True)
        all_allowed_variables = np.concatenate([guaranteed_part, remaining_part])
        rng.shuffle(all_allowed_variables)

        return [self._constant_literal(rng) if leaf == _CONSTANT_SLOT else str(leaf)
                for leaf in all_allowed_variables.tolist()]

    def sample(self, n_operators: int, rng: np.random.Generator | None = None) -> list[str]:
        rng = rng if rng is not None else np.random.default_rng()
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

            arity_value = self.simplipy_engine.operator_arity[operator]
            n_empty_nodes += arity_value - 1 - position
            total_leaves += arity_value - 1
            left_leaves += position

            insert_index = [index for index, value in enumerate(stack) if value is None][left_leaves]
            stack = (
                stack[:insert_index]
                + [str(operator)]
                + [None for _ in range(arity_value)]
                + stack[insert_index + 1:]
            )

        assert len([1 for value in stack if value in self.simplipy_engine.operator_arity]) == n_operators
        assert len([1 for value in stack if value is None]) == total_leaves

        if self.typed_slots:
            stack = self._collapse_typed_slots(stack, rng)
            total_leaves = sum(1 for value in stack if value is None)

        leaves = self._get_leaves(t_leaves=total_leaves, rng=rng)
        assert len(leaves) == total_leaves, f"Expected {total_leaves} leaves, got {len(leaves)}"

        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is None:
                stack = stack[:index] + [leaves.pop()] + stack[index + 1:]
        assert len(leaves) == 0

        return stack  # type: ignore[return-value]
