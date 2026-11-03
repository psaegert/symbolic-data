"""How a declarative catalog entry draws its input rows.

A variable's spec in an entry's ``vars`` block takes one of three forms:

* the ``fastsrb`` shorthand, ``sample_range: [low, high]`` and ``sample_type: [base, sign]`` (every
  release catalog; its draws are unchanged);
* ``dist``: a distribution config as :func:`~symbolic_data.distributions.get_distribution` reads it,
  ``{name: log_uniform, kwargs: {low: 0.1, high: 10}}``, or a list of them, a mixture drawn per value
  (each component may carry a ``weight``; equal weights otherwise);
* ``expr``: a value derived from the entry's ``latent`` variables, ``x2 / x1`` -- a push-forward of the
  distributions the benchmark's authors give for quantities that are not inputs themselves.

An entry may instead draw all its inputs jointly, ``inputs: {name: ..., kwargs: {...}}``, with a sampler
from :data:`JOINT_INPUTS` (for distributions that are not a product over variables, such as clusters
rotated in the whole input space), and may restrict its rows with ``row_constraints``: expressions over the
variables (``v1 > v2``) that every accepted row satisfies (not ``constraints``: FastSRB's entries use that key for
their own notes). A constraint acts like the expression's
domain: rows that violate it are redrawn, so the accepted rows follow the declared distribution
conditioned on the constraint.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from symbolic_data.distributions import fastsrb_dist, get_distribution
from symbolic_data.registry import Registry

#: Samplers that draw an entry's input rows jointly: ``fn(size, n_variables, rng, **kwargs) -> (size, n_variables)``.
JOINT_INPUTS = Registry("joint input sampler", entry_point_group="symbolic_data.joint_inputs")

# The names a derived variable or a constraint may use besides the variables themselves.
_MATH: dict[str, Any] = {
    "pi": np.pi, "e": np.e, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs, "minimum": np.minimum, "maximum": np.maximum,
}


def _evaluate(expr: str, names: Mapping[str, np.ndarray]) -> np.ndarray:
    """An arithmetic expression over named columns (a catalog's own spec, not user input)."""
    with np.errstate(all="ignore"):
        return np.asarray(eval(compile(expr, "<catalog expression>", "eval"), {"__builtins__": {}}, {**_MATH, **names}))


def _mixture(components: Sequence[Mapping[str, Any]], size: int, rng: np.random.Generator) -> np.ndarray:
    """One value per row from a component picked per row (``weight`` per component, equal by default)."""
    weights = np.array([float(c.get("weight", 1.0)) for c in components])
    pick = rng.choice(len(components), size=size, p=weights / weights.sum())
    out = np.empty(size)
    for k, component in enumerate(components):
        rows = np.flatnonzero(pick == k)
        if rows.size:
            config = {key: v for key, v in component.items() if key != "weight"}
            out[rows] = np.asarray(get_distribution(config)(size=rows.size, rng=rng), dtype=float).reshape(-1)
    return out


def draw_variable(spec: Mapping[str, Any], size: int, rng: np.random.Generator, layout: str = "random") -> np.ndarray:
    """``size`` values of one variable from its ``dist`` (a config or a mixture of them) or its fastsrb shorthand."""
    dist = spec.get("dist")
    if dist is None:
        base, sign = spec["sample_type"]
        low, high = spec["sample_range"]
        return fastsrb_dist(low, high, base=base, sign=sign, layout=layout, size=size, rng=rng)
    if isinstance(dist, Mapping):
        return np.asarray(get_distribution(dict(dist))(size=size, rng=rng), dtype=float).reshape(-1)
    return _mixture(dist, size, rng)


def draw_rows(variables: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Any], variable_order: Sequence[str],
              size: int, rng: np.random.Generator, layout: str = "random") -> np.ndarray:
    """``(size, len(variable_order))`` input rows of one entry: its joint sampler, else its latent variables
    then each variable in order (drawn or derived)."""
    joint = meta.get("inputs")
    if joint is not None:
        sampler: Callable[..., np.ndarray] = JOINT_INPUTS.get(joint["name"])
        x = np.asarray(sampler(size=size, n_variables=len(variable_order), rng=rng, **dict(joint.get("kwargs", {}))), dtype=float)
        if x.shape != (size, len(variable_order)):
            raise ValueError(f"joint sampler {joint['name']!r} returned {x.shape}, expected {(size, len(variable_order))}")
        return x
    latent = {name: draw_variable(cfg, size, rng, layout) for name, cfg in (meta.get("latent") or {}).items()}
    columns = []
    for key in variable_order:
        spec = variables[key]
        if "expr" in spec:
            columns.append(np.broadcast_to(_evaluate(spec["expr"], latent), (size,)).astype(float))
        else:
            columns.append(draw_variable(spec, size, rng, layout))
    return np.column_stack(columns).astype(float)


def constraint_mask(meta: Mapping[str, Any], x: np.ndarray, variable_order: Sequence[str]) -> np.ndarray:
    """The rows that satisfy every one of the entry's ``row_constraints`` (all rows when it has none)."""
    constraints = meta.get("row_constraints") or []
    ok = np.ones(x.shape[0], dtype=bool)
    if constraints:
        names = {key: x[:, i] for i, key in enumerate(variable_order)}
        for expr in constraints:
            ok &= np.broadcast_to(np.asarray(_evaluate(expr, names), dtype=bool), ok.shape)
    return ok


def e2e_mixture(size: int, n_variables: int, rng: np.random.Generator, shape: str = "gaussian", k: int = 1,
                scale: float = 1.0) -> np.ndarray:
    """E2E's input distribution (Kamienny et al. 2022, Sec. 1.2; symbolicregression envs/generators.py): a mixture of
    ``k`` clusters, each with a centroid mu ~ N(0, 1)^D, variances s ~ U(0, 1)^D and a Haar rotation R on SO(D)
    (identity for D = 1), weights ~ U(0, 1) normalized and counts ~ Multinomial(size, weights). A Gaussian cluster
    draws N(mu, diag s) @ R, a uniform one (mu + U(-1, 1)^D sqrt(s)) @ R. The rows are whitened per dimension (sample
    mean and population std) and multiplied by ``scale``. The shape and k are fixed per function; everything else is
    drawn afresh for every point set, i.e. every call."""
    if shape not in ("gaussian", "uniform"):
        raise ValueError(f"e2e_mixture shape must be 'gaussian' or 'uniform', got {shape!r}")
    d = n_variables
    mu = rng.normal(0.0, 1.0, (k, d))
    var = rng.uniform(0.0, 1.0, (k, d))
    if d > 1:
        from scipy.stats import special_ortho_group
        rot = [special_ortho_group.rvs(d, random_state=rng) for _ in range(k)]
    else:
        rot = [np.eye(1) for _ in range(k)]
    w = rng.uniform(0.0, 1.0, k)
    counts = rng.multinomial(size, w / w.sum())
    parts = []
    for i, n in enumerate(counts):
        if shape == "gaussian":
            z = rng.normal(0.0, 1.0, (n, d)) * np.sqrt(var[i]) + mu[i]
        else:
            z = mu[i] + rng.uniform(-1.0, 1.0, (n, d)) * np.sqrt(var[i])
        parts.append(z @ rot[i])
    x = np.concatenate(parts)
    std = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.where(std > 0, std, 1.0)
    return rng.permutation(x * scale)


JOINT_INPUTS.register_builtin("e2e_mixture", e2e_mixture)
