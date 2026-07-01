"""The level-2 sampler: one concrete :class:`ProblemSource`.

A ``ProblemSource`` turns a level-1 :class:`~symbolic_data.catalog.ProblemCatalog` (or a generator,
or inline problems) into :class:`~symbolic_data.problem.Problem`s, applying the *usage policy* the
catalog deliberately omits: draw ``method``, ``n_support`` / ``n_validation``, ``noise``,
``problems_per_expression``, holdouts/filters, and (for sets lacking intrinsic ranges) a global
sampling fallback. It is ONE concrete class -- no ABC, no subclasses; the mode is INFERRED from the
config (a ``catalog:`` ref -> SET or GENERATE (a generative ``catalog: {type: ...}`` mapping streams on-the-fly), inline ``problems:`` ->
FIXED).

Reproducibility never comes from a seed: ``rng`` is entropy by default; exact reproduction is
obtained from a fixed/materialized source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

from symbolic_data._evaluation import compile_expression, load_engine
from symbolic_data.catalog import Catalog, ProblemCatalog
from symbolic_data.errors import CatalogEntryError, NoValidSampleFoundError
from symbolic_data.generative import GenerativeCatalog, build_catalog, is_open_generative_ref
from symbolic_data.problem import Problem
from simplipy import normalize_skeleton

# Sampling-policy defaults used when the config / catalog metadata does not specify a value.
_DEFAULT_MAX_TRIALS = 100          # per-slot generation retries before yielding a placeholder Problem
_DEFAULT_GENERATE_N_SUPPORT = 32   # generative catalogs: support size when `sampling.n_support` is unset
_DEFAULT_SET_N_POINTS = 100        # declarative catalogs: support size when catalog meta omits `n_points`


def _entry_variables(entry: Any) -> list[str]:
    """Best-effort variable-name list for a placeholder, across entry shapes (declarative dict /
    generative list)."""
    variables = getattr(entry, "variables", None)
    if isinstance(variables, dict):
        return list(variables.keys())
    if isinstance(variables, (list, tuple)):
        return list(variables)
    return []


def _split_support_validation(x_all: np.ndarray, y_all: np.ndarray, n_support: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """First ``n_support`` points -> support, the rest -> validation (float32)."""
    n_support = max(1, min(n_support, x_all.shape[0]))
    xs = x_all[:n_support].astype(np.float32, copy=True)
    ys = y_all[:n_support].astype(np.float32, copy=True)
    xv = x_all[n_support:].astype(np.float32, copy=True)
    yv = y_all[n_support:].astype(np.float32, copy=True)
    return xs, xv, ys, yv


def _inject_noise(array: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise scaled by ``noise_level * std(array)`` (no-op on empty/constant)."""
    if array.size == 0 or noise_level <= 0.0:
        return array.copy()
    std = float(np.std(array))
    if not np.isfinite(std) or std <= 0:
        return array.copy()
    noise = rng.standard_normal(size=array.shape).astype(np.float32)
    return (array + noise_level * std * noise).astype(np.float32)


class ProblemSource:
    """Produce :class:`Problem`s from a catalog / generator / inline problems (mode inferred)."""

    def __init__(self, config: Mapping[str, Any] | str | Path, *, simplipy_engine: Any = None, rng: np.random.Generator | None = None) -> None:
        self.config = _load_config(config)
        self._rng = rng
        self._engine_spec = simplipy_engine if simplipy_engine is not None else self.config.get("engine", "dev_7-3")
        self._engine = simplipy_engine if not isinstance(simplipy_engine, (str, type(None))) else None
        self._catalog: Catalog | None = None

        # mode inference -- exactly one of catalog / problems. A `catalog:` that is a mapping (with a
        # `type:`) is a GENERATIVE catalog (on-the-fly); a string/path is a DECLARATIVE one.
        present = [k for k in ("catalog", "problems") if k in self.config]
        if len(present) != 1:
            raise ValueError("ProblemSource config must specify exactly one of: catalog | problems")
        if "problems" in self.config:
            self.mode = "fixed"
        else:
            # `catalog:` may be a mapping (`{type: ...}`) OR a pre-built Catalog instance OR a
            # string/path / name. A generative mapping or instance -> generate. A STRING ref is
            # resolved + peeked: an OPEN generative spec streams (generate); a FROZEN generative spec
            # (inline skeletons, e.g. a validation set) or a declarative one iterates a fixed set (set).
            catalog_spec = self.config["catalog"]
            if isinstance(catalog_spec, Mapping) or isinstance(catalog_spec, GenerativeCatalog):
                self.mode = "generate"
            elif isinstance(catalog_spec, (str, Path)):
                self.mode = "generate" if is_open_generative_ref(catalog_spec) else "set"
            else:
                self.mode = "set"

        s = dict(self.config.get("sampling", {}))
        self.method = s.get("method", "procedural" if self.mode == "generate" else "iterate")
        # `n_support: prior` (generative catalogs only): draw the per-sample support size from the
        # catalog's OWN support prior (the training-time pattern -- variable support sizes), rather
        # than a fixed count. It implies no validation split (all realized rows are support).
        self._n_support_from_prior = (s.get("n_support") == "prior")
        self.n_support = None if self._n_support_from_prior else s.get("n_support")
        self.n_validation = s.get("n_validation")
        if self._n_support_from_prior and self.n_validation not in (None, 0):
            raise ValueError("sampling.n_support: 'prior' requires n_validation: 0 (the support size is drawn per sample; there is no validation split)")
        self.noise = float(s.get("noise", 0.0))
        self.problems_per_expression = int(s.get("problems_per_expression", 1))
        self.layout = s.get("layout", "random")          # X-point layout passed to the distribution
        self.max_trials = int(s.get("max_trials", _DEFAULT_MAX_TRIALS))
        # number of expressions to draw from a generative catalog (usage policy); None = unbounded stream
        self._size = int(s["size"]) if s.get("size") is not None else None
        self.holdouts = list(self.config.get("holdouts", []))
        self._exclude_cache: dict[str, set] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path) -> "ProblemSource":
        """Build a source from a config mapping or a path to a yaml config (alias for the constructor
        with default engine and entropy ``rng``)."""
        return cls(config)

    # --- lazy resources ----------------------------------------------------------------------
    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()       # entropy; never a fixed default seed
        return self._rng

    def _get_engine(self):
        if self._engine is None:
            self._engine = load_engine(self._engine_spec)
        return self._engine

    def prepare(self, *, simplipy_engine: Any = None) -> None:
        """Optional: inject a shared simplipy engine (so a model adapter and the source agree)."""
        if simplipy_engine is not None:
            self._engine = load_engine(simplipy_engine)

    def _get_catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = build_catalog(self.config["catalog"])
        return self._catalog

    @property
    def catalog(self) -> Catalog:
        """The :class:`~symbolic_data.catalog.Catalog` this source samples from (built lazily).

        Lets a consumer that ALSO needs the catalog directly -- e.g. a trainer harvesting raw
        skeletons for prompt features -- share the source's single catalog instance (one engine)
        instead of building a second one.
        """
        return self._get_catalog()

    # --- iteration ---------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Problem]:
        if self.mode == "fixed":
            yield from self._iter_fixed()
        else:
            yield from self._iter_catalog()

    def size_hint(self) -> int | None:
        """Best-effort number of Problems this source will yield, or ``None`` when unbounded.

        ``fixed`` -> inline problem count times ``problems_per_expression``; ``generate`` -> the
        configured ``size`` times ``problems_per_expression`` (``None`` for an unbounded stream);
        ``set`` -> the frozen catalog's problem count, or the entry count times
        ``problems_per_expression``.
        """
        if self.mode == "fixed":
            return len(self.config["problems"]) * self.problems_per_expression
        if self.mode == "generate":
            # an unbounded streaming generative source has no finite size
            return self._size * self.problems_per_expression if self._size is not None else None
        catalog = self._get_catalog()
        if getattr(catalog, "frozen", False):
            return len(catalog.problems or [])
        return len(catalog) * self.problems_per_expression

    @property
    def max_n_support(self) -> int | None:
        """Upper bound on a sampled support size (for buffer pre-allocation): a generative catalog's
        configured support maximum when available, else the fixed ``n_support`` (None if neither)."""
        catalog = self._get_catalog()
        support_sampler = getattr(catalog, "support_sampler", None)
        configured = getattr(support_sampler, "configured_max_n_support", None) if support_sampler is not None else None
        if configured is not None:
            return int(configured)
        return int(self.n_support) if self.n_support is not None else None

    # --- catalog modes (declarative SET + generative GENERATE) -------------------------------
    def _resolve_counts(self, catalog: Catalog) -> tuple[int | None, int]:
        # prior mode: the catalog draws the support size per sample; no fixed count, no validation split
        if self._n_support_from_prior:
            return None, 0
        meta = getattr(catalog, "meta", None) or {}
        defaults = meta.get("sampling_defaults", {})
        if self.n_support is not None:
            n_support = int(self.n_support)
        elif isinstance(catalog, GenerativeCatalog):
            n_support = _DEFAULT_GENERATE_N_SUPPORT
        else:
            n_support = int(defaults.get("n_points", _DEFAULT_SET_N_POINTS))
        n_validation = int(self.n_validation) if self.n_validation is not None else n_support
        return n_support, n_validation

    def _iter_catalog(self) -> Iterator[Problem]:
        catalog = self._get_catalog()
        if getattr(catalog, "frozen", False):
            # A frozen/materialized catalog already holds realized Problems -- iterate them
            # directly (no sampling), like a fixed source.
            for problem in (catalog.problems or []):
                if problem.is_placeholder or self._passes_filters(problem):
                    yield problem
            return
        if self._n_support_from_prior and not isinstance(catalog, GenerativeCatalog):
            raise ValueError("sampling.n_support: 'prior' requires a generative catalog (it draws the support size from the catalog's support prior)")
        rng = self._get_rng()
        # generative catalogs carry their own engine; declarative ones borrow the source's
        engine = None if isinstance(catalog, GenerativeCatalog) else self._get_engine()
        n_support, n_validation = self._resolve_counts(catalog)
        for entry in catalog.iter_entries(rng, method=self.method, size=self._size):
            for _ in range(self.problems_per_expression):
                problem = self._realize_problem(catalog, entry, n_support, n_validation, engine, rng)
                if problem.is_placeholder or self._passes_filters(problem):
                    yield problem

    def _realize_problem(self, catalog: Catalog, entry, n_support: int | None, n_validation: int, engine, rng) -> Problem:
        """Realize one catalog entry into a Problem, applying usage policy (split + noise).

        ``catalog.realize`` owns the intrinsic (X, y) sampling; this method owns the usage policy and
        the placeholder protocol: a transient ``NoValidSampleFoundError`` is retried up to
        ``max_trials``, a permanent ``CatalogEntryError`` becomes a placeholder immediately. When
        ``n_support`` is ``None`` (prior mode) the catalog draws the support size and ALL realized
        rows become support (no validation split).
        """
        n_points = None if n_support is None else n_support + n_validation
        eq_id = getattr(entry, "id", None)
        try:
            for _ in range(self.max_trials):
                try:
                    realized = catalog.realize(entry, n_points, rng, engine=engine, layout=self.layout)
                except NoValidSampleFoundError:
                    continue
                split_at = realized.x.shape[0] if n_support is None else n_support
                xs, xv, ys, yv = _split_support_validation(realized.x, realized.y, split_at)
                if xs.size == 0:
                    continue
                return Problem(
                    x_support=xs, y_support=ys, y_support_noisy=_inject_noise(ys, self.noise, rng),
                    x_validation=xv, y_validation=yv, y_validation_noisy=_inject_noise(yv, self.noise, rng),
                    skeleton=realized.skeleton, expression=realized.expression,
                    constants=realized.constants, variables=realized.variables, complexity=realized.complexity,
                    noise=self.noise, eq_id=realized.eq_id, meta=realized.meta,
                )
        except CatalogEntryError as exc:
            return Problem.placeholder(variables=_entry_variables(entry), reason=str(exc), eq_id=eq_id)
        return Problem.placeholder(variables=_entry_variables(entry), reason="max_trials_exhausted", eq_id=eq_id)

    # --- FIXED mode (inline pre-supplied problems) -------------------------------------------
    def _iter_fixed(self) -> Iterator[Problem]:
        for body in self.config["problems"]:
            problem = body if isinstance(body, Problem) else Problem.from_dict(body)
            if problem.is_placeholder or self._passes_filters(problem):
                yield problem

    # --- GENERATE mode (on-the-fly procedural skeletons) -------------------------------------
    # --- filters / holdouts ------------------------------------------------------------------
    def _passes_filters(self, problem: Problem) -> bool:
        for rule in self.holdouts:
            if "filter" in rule and not _passes_filter(problem, rule["filter"]):
                return False
            if "exclude" in rule and self._is_excluded(problem, rule["exclude"]):
                return False
        return True

    def _is_excluded(self, problem: Problem, ref: str) -> bool:
        """Decontamination: drop a problem whose SKELETON matches one in the excluded catalog.

        Matching is at the skeleton level via :func:`simplipy.normalize_skeleton`, which collapses
        constants to ``<constant>`` AND canonicalizes variable names (``v3 -> x3``). So decontamination
        is structural (a generated ``x1..`` skeleton is dropped if it matches a held-out FastSRB
        ``v1..`` expression's structure) and constant-agnostic -- the leak-safe behaviour a held-out
        evaluation set needs. The excluded ``ref`` resolves via :func:`build_catalog`, so it may be a
        declarative catalog (skeletons from its expressions) OR a generative one (its skeleton set);
        never a "catalog" file. Functional-equivalence decontamination is a later refinement.
        """
        # Key off the problem's SKELETON (the structural form), not its `expression`: the exclusion
        # keys are built from skeletons, and a realized `expression` may be the CONCRETE formula (its
        # parsed structure differs from the simplified skeleton). normalize_skeleton canonicalizes
        # variables so the comparison stays cross-namespace (v3 <-> x3).
        if problem.skeleton is None:
            return False
        if ref not in self._exclude_cache:
            self._exclude_cache[ref] = self._exclusion_keys(ref)
        return tuple(normalize_skeleton(list(problem.skeleton))) in self._exclude_cache[ref]

    def _exclusion_keys(self, ref: str) -> set:
        """Skeleton keys of the excluded catalog (normalized: constants collapsed, variables canonical)."""
        catalog = build_catalog(ref)
        keys: set[tuple[str, ...]] = set()
        if isinstance(catalog, GenerativeCatalog):
            for skeleton in catalog.skeletons:
                keys.add(tuple(normalize_skeleton(list(skeleton))))
            return keys
        # A FROZEN ProblemCatalog (a materialized .npz) holds realized Problems in `.problems`, not
        # declarative `.entries` -- `iter_expressions()` would yield nothing, so excluding one would be
        # a SILENT no-op. Key off each problem's normalized expression directly.
        if isinstance(catalog, ProblemCatalog) and getattr(catalog, "frozen", False):
            for problem in (catalog.problems or []):
                if problem.skeleton is not None:
                    keys.add(tuple(normalize_skeleton(list(problem.skeleton))))
            return keys
        engine = self._get_engine()
        for entry in catalog.iter_expressions():
            try:
                compiled = compile_expression(engine, entry.id, entry.prepared, entry.variables, name=catalog.name)
            except Exception:  # noqa: BLE001 - skip an entry that fails to compile
                continue
            keys.add(tuple(normalize_skeleton(list(compiled["prefix"]))))
        return keys

    # --- materialize / freeze ----------------------------------------------------------------
    def materialize(self, n: int | None = None) -> "ProblemSource":
        """Sample once and FREEZE -> a FIXED-mode source that re-iterates the identical Problems.

        This is the reproducibility mechanism (no seeds): the returned source holds the realized
        Problems, so iterating it any number of times yields byte-identical data, and it is
        identical across models/runs. ``n`` caps the number of problems (required for an unbounded
        generate source without a ``size``).
        """
        frozen: list[dict] = []
        for problem in self:
            frozen.append(problem.to_dict())
            if n is not None and len(frozen) >= n:
                break
        return ProblemSource({"problems": frozen})

    def to_catalog(self, name: str | None = None, n: int | None = None) -> ProblemCatalog:
        """Materialize once and return a FROZEN ProblemCatalog (persist via ``.save(path)``).

        The frozen catalog holds the realized Problems; loading it back (``load_catalog``) and
        iterating yields byte-identical data -- the shareable form of ``materialize()``.
        """
        problems: list[Problem] = []
        for problem in self:
            problems.append(problem)
            if n is not None and len(problems) >= n:
                break
        cat_name = name or (str(self.config["catalog"]) if self.mode == "set" else "materialized")
        return ProblemCatalog.from_problems(problems, name=cat_name)


def _passes_filter(problem: Problem, spec: Mapping[str, Any]) -> bool:
    if spec.get("finite") and not problem.is_finite():
        return False
    if "max_complexity" in spec and problem.complexity is not None and problem.complexity > spec["max_complexity"]:
        return False
    if "n_variables" in spec and problem.n_variables_used != spec["n_variables"]:
        return False
    if "max_variables" in spec and problem.n_variables_used > spec["max_variables"]:
        return False
    return True


def _load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, (str, Path)):
        return yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("ProblemSource config must be a mapping or a path to a yaml file")
