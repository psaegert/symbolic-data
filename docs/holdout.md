# Holdout & leak-safety

A `ProblemSource` applies holdout/filter rules as part of its usage policy, via a `holdouts:` block.
Each rule is one of:

- **`{exclude: <ref>}`** — *decontamination*: drop any problem whose **skeleton** matches one in the
  excluded catalog.
- **`{filter: {...}}`** — drop any problem that fails a structural predicate.

```python
from symbolic_data import ProblemSource

source = ProblemSource({
    "catalog": "lample-charton-v24",        # an open generative training recipe
    "sampling": {"n_support": "prior", "n_validation": 0, "size": 1000},
    "holdouts": [
        {"exclude": "fastsrb"},             # decontaminate against the held-out benchmark set
        {"filter": {"finite": True, "max_complexity": 30, "max_variables": 5}},
    ],
})
```

## Decontamination: `{exclude: <ref>}`

Matching is at the **skeleton level** via `simplipy.normalize_skeleton`, which collapses constants to
`<constant>` *and* canonicalizes variable names (`v3 -> x3`). So decontamination is structural and
constant-agnostic: a generated `x1..` skeleton is dropped if it matches a held-out FastSRB `v1..`
expression's structure, regardless of the constant literals. This is the leak-safe behaviour a
held-out evaluation set needs.

The excluded `<ref>` is resolved by the same mechanism as any catalog ref (a name / path / mapping),
so it may be a **declarative** catalog (skeletons taken from its expressions) **or** a **generative**
one (its skeleton set) — enabling cross-namespace decontamination (e.g. training generation excluding
either the declarative FastSRB benchmark or a frozen generative validation set). The exclusion keys are
cached on first use. Functional-equivalence decontamination (beyond structural skeleton match) is a
later refinement.

This replaces the former internal holdout machinery (a separate `HoldoutManager`) that earlier
versions used for training generation; there is no longer a separate `HoldoutManager` in the public
API — holdout is a `ProblemSource` policy.

## Generative catalogs: `holdout_pools` and the family key

A generative catalog (`LampleChartonCatalog`) excludes benchmark laws at the source: every pool listed under
`holdout_pools` registers its laws, and a drawn skeleton whose FAMILY matches a registered one is resampled. A family
is matched in two layers -- the exact prototype (a hash) and its functional image on fixed probe grids, standardized
so the output-affine family `a*f + b` shares one key. The prototype is chosen by the catalog key `holdout_family`:

| key | literals masked | fold | images |
|---|---|---|---|
| `precise` (default since 2026-11-28) | fittable literals only (by role); exponents and root indices are structure | affine reparametrizations only (`X+c`, `c*X`, `X/c` -> `X`; `c/X` -> `inv X`; `c^X` -> `exp X`); additive term signs are spelling; one pass over the constant-free prototype | standardized over the finite entries |
| `legacy` | every literal, exponents and root indices included | every operator with a constant operand collapses to its other operand (`x^2`, `c/(x+a)` and `a*x+b` all become `x`) | non-finite entries zeroed, then standardized |

Under `precise` each law additionally registers the prototypes of its sign variants and of two generic-coefficient
members (an affine wrap and a seeded redraw of its constants).

```yaml
type: lample_charton
holdout_pools: [fastsrb, feynman]
holdout_family: precise   # the default; `legacy` reproduces a catalog built before 2026-11-28
```

Any other value is an error.

### Why `precise` is the default

The legacy key holds out too little where it matters and too much everywhere else. Its fold strips exponents and
root indices and turns `c/X` into `X`, so `x^2`, `x^3`, `sqrt(x)`, `c/(x+a)` and `a*x+b` all reduce to the core `x`.
Whole structure classes (low-order polynomials, sums of rationals, weighted sums) were rejected whenever some benchmark
law reduced to the same core. Meanwhile, members of a held-out law's own family escaped: the canonical form decides by
value whether a coefficient factors out, and zeroing the non-finite part of an image breaks invariance under `a*f + b`.

Leak audit on the 29 production holdout pools (6,660 law renderings; for each law three members with redrawn
constants and one random affine wrap), counting family members that are NOT held out, and the share of ordinary
training draws each key removes:

| | legacy | precise |
|---|---|---|
| laws in canonical form, not held out | 3 | 0 |
| members with redrawn constants, not held out | 53 | 0 |
| affine wraps `a*f + b`, not held out | 1,660 | 7 |
| ordinary training draws held out | 6.42% | 2.80% |

Cost: about +4.7 ms per draw at generation time (11.7 vs 7.0 ms per probed draw), and registration of the 29 pools
takes 199 s instead of 145 s.

### Reproducing a catalog built before the change

The key is baked into a training corpus: which draws were resampled depends on it. A catalog config that names no
key and was used before 2026-11-28 was built with `legacy`. To regenerate such a model's data or audit its holdout,
set `holdout_family: legacy` explicitly in its catalog config; the same config without the key now gets `precise`.
A config that already names its key is unaffected. A catalog pickled before the key existed unpickles as `legacy`.

## Filters: `{filter: {...}}`

A `filter` rule drops problems failing any of its predicates:

| key | predicate |
|---|---|
| `finite: true` | keep only problems whose `(X, y)` arrays are all-finite (`Problem.is_finite()`); applies to realized problems only — placeholders pass through (see below) |
| `max_complexity: N` | keep only problems whose substituted-expression token length is `<= N` |
| `n_variables: N` | keep only problems using exactly `N` distinct pool variables |
| `max_variables: N` | keep only problems using at most `N` distinct pool variables |

Variable counts use `Problem.n_variables_used` (distinct pool variables actually appearing in the
skeleton).

## Placeholders pass through

Holdout/filter rules apply only to *realized* problems: a **placeholder** `Problem`
(`is_placeholder=True`, emitted when the source could not fill a slot) is yielded regardless, so
row-accounting stays aligned. Filter placeholders out downstream with `if problem.is_placeholder:
continue`.

## Reproducibility scope: leak-safety + materialized byte-identity

Two guarantees:

- **Leak-safety** is provided by the structural, variable-canonical `{exclude: <ref>}` matcher above:
  any two consumers excluding the same `<ref>` make the same held-out decisions, so a held-out
  problem stays held out across machines and runs.
- **Byte-identical** regeneration is not a property of live sampling (which uses process entropy);
  it is obtained by *materializing* a source once and freezing it
  (`source.materialize()` / `source.to_catalog(...).save(...)` — see
  [Sampling data](sampling.md#reproducibility-materialize-freeze)). Re-iterating the frozen artifact
  yields byte-identical `Problem`s.

Resolved HF catalogs pin a git revision and per-file sha256, so the *expression set and ranges* a
holdout decision is computed against are themselves versioned and auditable.
