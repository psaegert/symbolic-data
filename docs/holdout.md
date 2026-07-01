# Holdout & leak-safety

A `ProblemSource` applies holdout/filter rules as part of its usage policy, via a `holdouts:` block.
Each rule is one of:

- **`{exclude: <ref>}`** — *decontamination*: drop any problem whose **skeleton** matches one in the
  excluded catalog.
- **`{filter: {...}}`** — drop any problem that fails a structural predicate.

```python
from symbolic_data import ProblemSource

source = ProblemSource({
    "catalog": "lample-charton-v23",        # an open generative training recipe
    "sampling": {"n_support": "prior", "n_validation": 0, "size": 1000},
    "holdouts": [
        {"exclude": "v23-val"},             # decontaminate against the held-out validation set
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
either the declarative FastSRB benchmark or the generative `v23-val` set). The exclusion keys are
cached on first use. Functional-equivalence decontamination (beyond structural skeleton match) is a
later refinement.

This replaces the former internal holdout machinery (a separate `HoldoutManager`) that earlier
versions used for training generation; there is no longer a separate `HoldoutManager` in the public
API — holdout is a `ProblemSource` policy.

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
