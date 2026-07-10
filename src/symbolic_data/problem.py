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

__all__ = ["Problem", "GT_KINDS"]

# Ground-truth kinds: "exact" = synthetic GT that generated y; "reference" = the historically
# accepted law accompanying real measurements (stored in the SAME expression/skeleton fields);
# "none" = black-box (no expression at all). Decides holdout-ability and metric regimes.
GT_KINDS = ("exact", "reference", "none")


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
    # ground-truth kind (see GT_KINDS). None at construction => inferred in __post_init__ from
    # skeleton/expression presence, so 0.10-era dicts and call sites keep working unchanged.
    gt_kind: str | None = None
    # reference-law predictions on the SAME sampled/measured points (real-data catalogs, WP7):
    # the catalog owns the reference expression and precomputes its predictions; downstream
    # derives reference_fvu without ever re-evaluating expressions. None <=> synthetic problem.
    y_reference_support: np.ndarray | None = None
    y_reference_validation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.gt_kind is None:
            has_structure = self.expression is not None or self.skeleton is not None
            self.gt_kind = "exact" if has_structure else "none"
        if self.gt_kind not in GT_KINDS:
            raise ValueError(f"gt_kind must be one of {GT_KINDS} (got {self.gt_kind!r})")
        if not self.is_placeholder:
            if self.gt_kind == "none" and (self.expression is not None or self.skeleton is not None
                                           or self.constants):
                raise ValueError("gt_kind='none' requires expression, skeleton and constants to be empty")
            if self.gt_kind in ("exact", "reference") and self.expression is None and self.skeleton is None:
                raise ValueError(f"gt_kind={self.gt_kind!r} requires a skeleton or expression")

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
            "gt_kind": self.gt_kind,
            "y_reference_support": self.y_reference_support,
            "y_reference_validation": self.y_reference_validation,
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

    @classmethod
    def from_data(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_validation: np.ndarray | None = None,
        y_validation: np.ndarray | None = None,
        expression: list[str] | None = None,
        skeleton: tuple[str, ...] | None = None,
        constants: list[float] | None = None,
        variables: list[str] | None = None,
        gt_kind: str | None = None,
        y_reference_support: np.ndarray | None = None,
        y_reference_validation: np.ndarray | None = None,
        noise: float | dict | None = None,
        eq_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Problem":
        """A :class:`Problem` from MEASURED data (real-world / black-box catalogs).

        Convention: the measured ``y`` IS the fitted target, so ``y_*_noisy`` are copies of
        the clean arrays and ``noise`` defaults to ``None`` (= unknown measurement noise).
        There is no separate clean target for measured data. ``gt_kind`` defaults to
        ``"reference"`` when an expression/skeleton is given (the accepted law accompanying
        the measurements) and ``"none"`` (black-box) otherwise; pass ``gt_kind="exact"``
        explicitly for frozen synthetic data. When an ``expression`` is given without a
        ``skeleton``, the skeleton is best-effort derived via simplipy so decontamination
        and recovery metrics keep working for reference problems."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if x_validation is None or y_validation is None:
            x_validation = np.empty((0, x.shape[1]), dtype=np.float32)
            y_validation = np.empty((0, 1), dtype=np.float32)
        else:
            x_validation = np.asarray(x_validation, dtype=np.float32)
            if x_validation.ndim == 1:
                x_validation = x_validation.reshape(-1, 1)
            y_validation = np.asarray(y_validation, dtype=np.float32).reshape(-1, 1)
        if variables is None:
            variables = [f"x{i}" for i in range(1, x.shape[1] + 1)]
        if skeleton is None and expression is not None:
            try:
                from simplipy import normalize_skeleton
                normalized = normalize_skeleton(list(expression))
                skeleton = tuple(normalized) if normalized is not None else None
            except Exception:
                skeleton = None
        if gt_kind is None:
            gt_kind = "reference" if (expression is not None or skeleton is not None) else "none"
        if gt_kind == "none" and (y_reference_support is not None or y_reference_validation is not None):
            raise ValueError("y_reference_* requires a reference/exact structure; a black-box "
                             "(gt_kind='none') problem has no reference law to predict with")
        # normalize the reference predictions like their y counterparts (float32, column vectors);
        # reject non-finite baselines -- a reference law must be finite on its own support, and the
        # float32 cast silently maps out-of-range float64 values (e.g. a non-log-space rendering of
        # a wide-dynamic-range law) to inf, which would poison every reference_fvu downstream.
        if y_reference_support is not None:
            y_reference_support = np.asarray(y_reference_support, dtype=np.float32).reshape(-1, 1)
            if y_reference_support.shape != y.shape:
                raise ValueError(f"y_reference_support shape {y_reference_support.shape} != y shape {y.shape}")
            if not np.all(np.isfinite(y_reference_support)):
                raise ValueError("y_reference_support contains non-finite values (possibly a "
                                 "float32-range overflow); fix the reference rendering or exclude the points")
        if y_reference_validation is not None:
            y_reference_validation = np.asarray(y_reference_validation, dtype=np.float32).reshape(-1, 1)
            if y_reference_validation.shape != y_validation.shape:
                raise ValueError(f"y_reference_validation shape {y_reference_validation.shape} != "
                                 f"y_validation shape {y_validation.shape}")
            if not np.all(np.isfinite(y_reference_validation)):
                raise ValueError("y_reference_validation contains non-finite values (possibly a "
                                 "float32-range overflow); fix the reference rendering or exclude the points")
        return cls(
            x_support=x,
            y_support=y,
            y_support_noisy=y.copy(),
            x_validation=x_validation,
            y_validation=y_validation,
            y_validation_noisy=y_validation.copy(),
            skeleton=skeleton,
            expression=expression,
            constants=list(constants) if constants else [],
            variables=list(variables),
            complexity=len(expression) if expression is not None else None,
            noise=noise,
            eq_id=eq_id,
            meta=dict(meta) if meta else {},
            gt_kind=gt_kind,
            y_reference_support=y_reference_support,
            y_reference_validation=y_reference_validation,
        )
