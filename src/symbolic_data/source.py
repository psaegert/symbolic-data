"""The level-2 sampler: one concrete :class:`ProblemSource`.

A ``ProblemSource`` turns a level-1 :class:`~symbolic_data.catalog.ProblemCatalog` (or a generator,
or inline problems) into :class:`~symbolic_data.problem.Problem`s, applying the *usage policy* the
catalog deliberately omits: draw ``method``, ``n_support`` / ``n_validation``, ``noise``,
``problems_per_expression``, holdouts/filters, and (for sets lacking intrinsic ranges) a global
sampling fallback. It is ONE concrete class -- no ABC, no subclasses; the mode is INFERRED from the
config (a ``catalog:`` ref -> SET, a ``generator:`` block -> ON-THE-FLY, inline ``problems:`` ->
FIXED).

Reproducibility never comes from a seed: ``rng`` is entropy by default; exact reproduction is
obtained from a fixed/materialized source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml
from simplipy import normalize_expression, normalize_skeleton
from simplipy.utils import substitude_constants

from symbolic_data._evaluation import broadcast_target, compile_expression, evaluate, load_engine
from symbolic_data.catalog import ProblemCatalog
from symbolic_data.distributions import fastsrb_dist
from symbolic_data.problem import Problem

_DRAW_METHODS = {"iterate", "random_without_replacement", "random_with_replacement"}


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
        self._catalog: ProblemCatalog | None = None

        # mode inference -- exactly one of catalog / generator / problems
        present = [k for k in ("catalog", "generator", "problems") if k in self.config]
        if len(present) != 1:
            raise ValueError("ProblemSource config must specify exactly one of: catalog | generator | problems")
        self.mode = {"catalog": "set", "generator": "generate", "problems": "fixed"}[present[0]]

        s = dict(self.config.get("sampling", {}))
        self.method = s.get("method", "procedural" if self.mode == "generate" else "iterate")
        self.n_support = s.get("n_support")
        self.n_validation = s.get("n_validation")
        self.noise = float(s.get("noise", 0.0))
        self.problems_per_expression = int(s.get("problems_per_expression", 1))
        self.layout = s.get("layout", "random")          # X-point layout passed to the distribution
        self.max_trials = int(s.get("max_trials", 100))
        self.holdouts = list(self.config.get("holdouts", []))
        self._exclude_cache: dict[str, set] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path) -> "ProblemSource":
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

    def _get_catalog(self) -> ProblemCatalog:
        if self._catalog is None:
            self._catalog = ProblemCatalog.load(self.config["catalog"])
        return self._catalog

    # --- iteration ---------------------------------------------------------------------------
    def __iter__(self) -> Iterator[Problem]:
        if self.mode == "set":
            yield from self._iter_set()
        elif self.mode == "fixed":
            yield from self._iter_fixed()
        else:
            yield from self._iter_generate()

    def size_hint(self) -> int | None:
        if self.mode == "set":
            catalog = self._get_catalog()
            if catalog.frozen:
                return len(catalog.problems or [])
            return len(catalog) * self.problems_per_expression
        if self.mode == "fixed":
            return len(self.config["problems"]) * self.problems_per_expression
        size = self.config["generator"].get("size")
        return int(size) * self.problems_per_expression if size is not None else None

    # --- SET mode ----------------------------------------------------------------------------
    def _resolve_counts(self, catalog: ProblemCatalog) -> tuple[int, int]:
        defaults = catalog.meta.get("sampling_defaults", {}) if catalog.meta else {}
        n_support = int(self.n_support) if self.n_support is not None else int(defaults.get("n_points", 100))
        n_validation = int(self.n_validation) if self.n_validation is not None else n_support
        return n_support, n_validation

    def _draw_eq_ids(self, catalog: ProblemCatalog, rng: np.random.Generator) -> list[str]:
        eq_ids = list(catalog.entries.keys())
        if self.method == "iterate":
            return eq_ids
        if self.method == "random_without_replacement":
            return [eq_ids[i] for i in rng.permutation(len(eq_ids))]
        if self.method == "random_with_replacement":
            return [eq_ids[i] for i in rng.integers(0, len(eq_ids), size=len(eq_ids))]
        raise ValueError(f"Unknown set draw method {self.method!r}; expected one of {sorted(_DRAW_METHODS)}")

    def _iter_set(self) -> Iterator[Problem]:
        catalog = self._get_catalog()
        if catalog.frozen:
            # A frozen/materialized catalog already holds realized Problems -- iterate them
            # directly (no sampling), like a fixed source.
            for problem in (catalog.problems or []):
                if problem.is_placeholder or self._passes_filters(problem):
                    yield problem
            return
        engine = self._get_engine()
        rng = self._get_rng()
        n_support, n_validation = self._resolve_counts(catalog)
        for eq_id in self._draw_eq_ids(catalog, rng):
            entry = catalog[eq_id]
            for _ in range(self.problems_per_expression):
                problem = self._sample_set_problem(engine, catalog, entry, eq_id, n_support, n_validation, rng)
                if problem.is_placeholder or self._passes_filters(problem):
                    yield problem

    def _sample_set_problem(self, engine, catalog, entry, eq_id, n_support, n_validation, rng) -> Problem:
        try:
            compiled = compile_expression(engine, eq_id, entry.prepared, entry.variables, name=catalog.name)
        except Exception as exc:  # malformed entry -> placeholder, keep accounting aligned
            return Problem.placeholder(variables=list(entry.variables.keys()), reason=f"compile_failed: {exc}", eq_id=eq_id)

        variable_order = compiled["variable_order"]
        n_total = n_support + n_validation
        for _ in range(self.max_trials):
            columns = []
            for key in variable_order:
                spec = entry.variables[key]
                base, sign = spec["sample_type"]
                low, high = spec["sample_range"]
                columns.append(fastsrb_dist(low, high, base=base, sign=sign, layout=self.layout, size=n_total, rng=rng))
            x_all = np.column_stack(columns).astype(float)
            value_map = {var: x_all[:, i] for i, var in enumerate(variable_order)}
            try:
                y_all = broadcast_target(evaluate(compiled, value_map), n_total, eq_id).reshape(-1, 1)
            except Exception:
                continue
            if not (np.all(np.isfinite(x_all)) and np.all(np.isfinite(y_all))):
                continue
            xs, xv, ys, yv = _split_support_validation(x_all, y_all, n_support)
            return Problem(
                x_support=xs, y_support=ys, y_support_noisy=_inject_noise(ys, self.noise, rng),
                x_validation=xv, y_validation=yv, y_validation_noisy=_inject_noise(yv, self.noise, rng),
                skeleton=tuple(compiled["prefix"]), expression=list(compiled["prefix"]),
                constants=[], variables=list(variable_order), complexity=len(compiled["prefix"]),
                noise=self.noise, eq_id=eq_id,
                meta={"prepared_normalized": compiled["normalized_infix"], **{k: v for k, v in entry.meta.items() if k != "sources"}},
            )
        return Problem.placeholder(variables=list(variable_order), reason="max_trials_exhausted", eq_id=eq_id)

    # --- FIXED mode (inline pre-supplied problems) -------------------------------------------
    def _iter_fixed(self) -> Iterator[Problem]:
        for body in self.config["problems"]:
            problem = body if isinstance(body, Problem) else Problem.from_dict(body)
            if problem.is_placeholder or self._passes_filters(problem):
                yield problem

    # --- GENERATE mode (on-the-fly procedural skeletons) -------------------------------------
    def _iter_generate(self) -> Iterator[Problem]:
        # Drives the (private, Generator-threaded) skeleton-generation engine and builds Problems
        # natively (no Sample intermediary). Reproducibility is inferential (multi-draw), not seeded;
        # the engine's randomness is fully controlled by self._get_rng().
        from symbolic_data._generate.skeleton_pool import SkeletonPool, NoValidSampleFoundError

        gen_cfg = self.config["generator"]
        size = gen_cfg.get("size")
        if size is None:
            raise ValueError("generate-mode config['generator'] must set `size` (number of skeletons to generate)")
        rng = self._get_rng()
        n_support = int(self.n_support) if self.n_support is not None else 32
        n_validation = int(self.n_validation) if self.n_validation is not None else n_support
        pool = SkeletonPool.from_config({k: v for k, v in gen_cfg.items() if k != "size"})
        pool.create(int(size), rng=rng)
        if not pool.skeleton_codes:
            pool.skeleton_codes = pool.compile_codes(verbose=False)
        for skeleton in sorted(pool.skeletons):
            code, constants_tokens = pool.skeleton_codes[skeleton]
            for _ in range(self.problems_per_expression):
                problem = self._generate_one(pool, skeleton, code, len(constants_tokens), n_support, n_validation, rng, NoValidSampleFoundError)
                if problem.is_placeholder or self._passes_filters(problem):
                    yield problem

    def _generate_one(self, pool, skeleton, code, n_constants, n_support, n_validation, rng, no_valid_exc) -> Problem:
        variables = list(pool.variables)
        n_total = n_support + n_validation
        for _ in range(self.max_trials):
            try:
                x_all, y_all, literals = pool.sample_data(code, n_constants, n_support=n_total, rng=rng)
            except no_valid_exc:
                continue
            if x_all.size == 0 or y_all.size == 0:
                continue
            xs, xv, ys, yv = _split_support_validation(x_all, y_all, n_support)
            if xs.size == 0:
                continue
            expression, complexity = _gt_metadata(skeleton, literals)
            return Problem(
                x_support=xs, y_support=ys, y_support_noisy=_inject_noise(ys, self.noise, rng),
                x_validation=xv, y_validation=yv, y_validation_noisy=_inject_noise(yv, self.noise, rng),
                skeleton=tuple(skeleton), expression=expression,
                constants=list(np.asarray(literals, dtype=np.float64).ravel().tolist()),
                variables=variables, complexity=complexity, noise=self.noise, eq_id=None,
            )
        return Problem.placeholder(variables=variables, reason="max_trials_exhausted", skeleton=tuple(skeleton))

    # --- filters / holdouts ------------------------------------------------------------------
    def _passes_filters(self, problem: Problem) -> bool:
        for rule in self.holdouts:
            if "filter" in rule and not _passes_filter(problem, rule["filter"]):
                return False
            if "exclude" in rule and self._is_excluded(problem, rule["exclude"]):
                return False
        return True

    def _is_excluded(self, problem: Problem, ref: str) -> bool:
        """Decontamination: drop a problem whose normalized expression matches the excluded catalog.

        Exact normalized-prefix match (leak-safe for shared templates). Functional-equivalence
        decontamination (a frozen evaluation grid) is a 0.4.x refinement.
        """
        if problem.expression is None:
            return False
        if ref not in self._exclude_cache:
            engine = self._get_engine()
            other = ProblemCatalog.load(ref)
            keys: set[tuple[str, ...]] = set()
            for entry in other.iter_expressions():
                try:
                    compiled = compile_expression(engine, entry.id, entry.prepared, entry.variables, name=other.name)
                except Exception:
                    continue
                keys.add(tuple(compiled["prefix"]))
            self._exclude_cache[ref] = keys
        return tuple(problem.expression) in self._exclude_cache[ref]

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


def _gt_metadata(skeleton, literals) -> tuple[list[str] | None, int | None]:
    """Normalized GT expression (constants substituted) + its token-length complexity."""
    skeleton_list = normalize_skeleton(skeleton)
    if skeleton_list is None:
        return None, None
    expression_tokens = substitude_constants(list(skeleton_list), values=literals, inplace=False)
    expression = normalize_expression(expression_tokens)
    complexity = len(expression_tokens) if expression_tokens else None
    return expression, complexity


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
