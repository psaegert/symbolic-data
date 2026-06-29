"""The one central data unit: a :class:`Problem`.

A :class:`Problem` is a single symbolic-regression problem -- a ground-truth skeleton plus
its realized constants and sampled support/validation data (optionally noised). It is produced
by *every* source (curated sets, on-the-fly generation, inline/materialized data); there is no
dict-vs-dataclass split. It supersedes the old ``Sample`` dataclass and srbf's
``EvaluationSample``.

Conventions baked in (settled across the 0.4.0 design review):

* Noise is on the *target* ``y`` only (measurement noise); ``X`` is never noised. We keep both
  the clean ``y_*`` (for ground-truth evaluation, e.g. FVU) and ``y_*_noisy`` (what a model
  actually fits). When the source's noise spec is null/zero, ``y_*_noisy`` is the same array.
* ``meta`` is the catch-all for source-specific provenance (units, moniker, source, a draw
  index, ...). Stable keys may be promoted to real fields later without breaking consumers.
* The placeholder protocol lets a source yield a marked placeholder (instead of silently
  skipping) when it cannot produce a valid problem for a slot, so row-accounting / resume /
  indexing stay aligned and the failure is recorded honestly downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Problem"]


@dataclass
class Problem:
    """One model-agnostic symbolic-regression problem (skeleton + sampled data)."""

    # realized data (float32). Noise is on the target y only; X is never noised.
    x_support: np.ndarray
    y_support: np.ndarray
    y_support_noisy: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    y_validation_noisy: np.ndarray
    # ground truth
    skeleton: tuple[str, ...] | None
    expression: list[str] | None        # GT tokens, constants substituted + normalized
    constants: list[float]              # realized constant literals
    variables: list[str]                # pool variable names; column order of the X arrays
    complexity: int | None              # token length of the substituted expression
    # realized noise applied (provenance). Today a relative-Gaussian scale (std as a fraction
    # of std(y)); the source configures noise via the distribution vocabulary, so this can
    # generalize to a full noise spec (distribution + outliers) without a schema break.
    noise: float | dict | None = 0.0
    # provenance / id
    eq_id: str | None = None            # catalog id, when set-sourced
    meta: dict[str, Any] = field(default_factory=dict)
    # placeholder protocol
    is_placeholder: bool = False
    placeholder_reason: str | None = None

    @property
    def n_variables_used(self) -> int:
        """Distinct pool variables appearing in the skeleton (derived)."""
        if not self.skeleton:
            return 0
        variable_set = set(self.variables)
        return len({token for token in self.skeleton if token in variable_set})

    def is_finite(self) -> bool:
        """True iff every (non-empty) clean X/y array is all-finite -- the validity gate."""
        for array in (self.x_support, self.y_support, self.x_validation, self.y_validation):
            if array.size and not np.all(np.isfinite(array)):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Shallow field mapping (arrays kept as arrays); inverse of :meth:`from_dict`."""
        return {
            "x_support": self.x_support,
            "y_support": self.y_support,
            "y_support_noisy": self.y_support_noisy,
            "x_validation": self.x_validation,
            "y_validation": self.y_validation,
            "y_validation_noisy": self.y_validation_noisy,
            "skeleton": self.skeleton,
            "expression": self.expression,
            "constants": self.constants,
            "variables": self.variables,
            "complexity": self.complexity,
            "noise": self.noise,
            "eq_id": self.eq_id,
            "meta": dict(self.meta),
            "is_placeholder": self.is_placeholder,
            "placeholder_reason": self.placeholder_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Problem":
        """Reconstruct a :class:`Problem` from :meth:`to_dict` output."""
        return cls(**data)

    @classmethod
    def placeholder(
        cls,
        variables: list[str],
        reason: str,
        *,
        skeleton: tuple[str, ...] | None = None,
        eq_id: str | None = None,
    ) -> "Problem":
        """An empty, marked placeholder for a slot the source could not fill."""
        n_variables = len(variables)
        empty_x = np.empty((0, n_variables), dtype=np.float32)
        empty_y = np.empty((0, 1), dtype=np.float32)
        return cls(
            x_support=empty_x,
            y_support=empty_y,
            y_support_noisy=empty_y.copy(),
            x_validation=empty_x.copy(),
            y_validation=empty_y.copy(),
            y_validation_noisy=empty_y.copy(),
            skeleton=tuple(skeleton) if skeleton is not None else None,
            expression=None,
            constants=[],
            variables=list(variables),
            complexity=None,
            noise=0.0,
            eq_id=eq_id,
            is_placeholder=True,
            placeholder_reason=reason,
        )
