# Term prior: higher-level structure for the training prior (research prototype)

Status: research prototype, 2026-11-20 to 2026-11-21. Not wired into the library: the samplers here wrap
`LampleChartonCatalog` internals and run outside the package. Nothing has been trained on this prior yet.

Branch `research/term-prior`, based on `fix/holdout-precise-family` (the precise holdout key the analysis uses).
The full experiment log, round by round, is [LOG.md](LOG.md). An interactive explorer of the draws exists as a
private claude.ai artifact (`explorer/index.html` is its source; its sample banks are regenerable, see below).

## The problem

The T8 prior samples an operator count from p(n) ∝ exp(n^0.7), n ≤ 17, and grows one exact-size Lample–Charton tree.
Its diversity is per node: every operator and leaf is a fresh draw, so expressions with structure above the node
level (sums of similar terms, polynomials, power laws, sums of rationals) are rare. Measured on 3,000 T8.1 draws
against 10,801 formulas from Wikipedia and Wikidata ("real formulas", `results/`):

| | T8 | real formulas |
|---|---|---|
| polynomial with ≥ 3 terms | 0.37% | 1.80% |
| all summands share one template (masked) | 0.1% | 1.7% |
| all factors share one template (masked) | 0.1% | 1.0% |
| root is a product of copies of one template (a power law) | 0.4% | 43.4% |
| operators in chains of ≥ 3 different templates | 24% | 1.7% |

The owner's goal: higher-level diversity, structured but not always structured, without overfitting to a list of
example classes (the "superposition of rationals" was one example of structure, not a target).

## The current prior: round 10c

One draw:

1. **Budget.** B = round(1.3 · n), n ~ p(n) ∝ exp(n^0.7), n = 0…17 (T8's length prior; the owner keeps it).
   Per expression: inheritance strength λ ~ Beta(3, 2) and structure level s ~ U(0, 1).
2. **Term(b, θ).** Every term draws its values θ: flat (probability 0.2), own-tree share f ~ Beta(2, 3), reuse
   ρ ~ Beta(2, 1), spread α ~ log-uniform[0.1, 10], additive share ~ Beta(0.5, 0.5), copy gate (probability 0.5).
   A nested term copies each value from its parent with probability λ^depth and otherwise redraws it from these root
   priors; s is the same for every term of the expression.
3. **Relation or LC tree.** A term with b ≥ 1 operators is, with probability s, one of three relations, chosen from
   a per-expression urn (concentration 1); otherwise it is an LC tree (step 4).
   - *Repeat and combine*: k copies of one nested term joined by one operator. Products: k = 2–6 with probabilities
     51, 26.5, 13, 7, 2.5%, each operator ÷ with probability 0.26, each copy wrapped in its own power or root with
     probability 0.33. Sums: k = 2–5 with 66, 20.5, 8, 5.5%, each operator − with probability 0.24, each copy with its
     own coefficient with probability 0.42. Copies swap variables (probability 0.7, to any of x1…x17) and redraw
     exponents (0.44); constants are always redrawn. A copy that would merge with the others on simplification is
     varied again (up to five times).
   - *Pairing*: two independent nested terms joined by × or ÷ (probability 0.6; ÷ 0.4 of those) or + or −
     (− 0.25 of those); the budget is split uniformly between them.
   - *Composition*: a function (family urn; damped by the nesting decay 0.25 per enclosing function), a power or
     root (T8's exponent prior), or affine c·□ + c, around one nested term that may not itself be a composition.
   Urn base shares: repeat product 31.5%, pairing 21.4%, affine 13.9%, power 12.3%, repeat sum 12.0%, function 8.9%.
4. **LC tree (round 8).** The term grows its own tree with n = b (flat) or max(1, round(f·b)) operators from the T8 tree
   sampler (binary labels follow the additive share; one family urn, concentration 0.36, per expression; function
   nesting counted through binary nodes and across terms). Its L slots form k = 1 + Binomial(L − 1, 1 − ρ) groups
   (at most 17); each group takes one slot and the remaining slots follow Dirichlet(α) weights. The remaining budget
   R = b − n is split by v ~ Dirichlet(1): each of a group's |t| copies gets R·v_t/|t| operators, rounded up with
   probability equal to the fraction. A group with 0 operators is an atom: a variable or the constant, distinct within
   the term, from the parent's variables with probability λ^depth, otherwise fresh. A group with operators holds
   copies of a nested term (first slot the template; further copies swap one operator with probability 0.1, redraw
   exponents with 0.5 and, if the gate is open, swap each variable with 0.5). A nested term must contain a variable.
5. **Weights and simplification.** Operator weights as T8.1, except the six hyperbolic functions (sinh, cosh, tanh and
   inverses) at 0.5 instead of 1. Simplify (permissive canon); resample if every variable cancelled. No length cap.

Special cases: s = 0 gives round 8; round 8 with every addition off gives T8.1 exactly (step 8a below).

### Where each value comes from

| Value | Source |
|---|---|
| Grouping rule (step 4) | Derived from T8: with ρ ~ U(0, 1) and α = ∞ it is T8's leaf rule (k uniform) |
| The three relations | Proposed, then tested: they explain 98.5% of the real formulas (coverage test below) |
| Relation shares | Halfway between equal shares and the real formulas' root shares (owner: halfway) |
| Copies, exponents, ÷, coefficients, exponent changes | Measured on real formulas, set halfway (round 10b) |
| s ~ U(0, 1) | Owner's choice ("structured but not always"; mean 0.5 tried first) |
| Hyperbolic weight 0.5 | Owner's choice |
| Length prior, budget ×1.3 | T8's; ×1.3 matches T8's simplified operator count in round 7 |
| λ ~ Beta(3, 2), variable swap 0.7, urn concentration 1, f, ρ, α priors, flat 0.2 | Guesses, not derived |
| No composition directly in a composition; re-varying merged copies | Fixes for observed failures (round 9) |

### Results (3,000 draws per prior, simplified)

| | T8 | round 8 | round 10b (s mean 0.25) | round 10c (s mean 0.5) | real formulas |
|---|---|---|---|---|---|
| ≥ 3 terms of one form | 0.17% | 1.13% | 2.13% | 4.10% | 1.12% |
| sum of ≥ 2 rational terms | 1.27% | 1.67% | 2.13% | 2.33% | 1.02% |
| polynomial, ≥ 3 terms | 0.37% | 1.37% | 2.63% | 4.37% | 1.80% |
| held out (precise key) | 2.9% | 7.3% | 6.1% | 4.8% | – |
| fully explained by the relations | 37% | 55% | 58% | 62% | 98.5% |
| operators in mixed chains | 47% | 41% | 37% | 34% | 13% |
| nodes in repeated subtrees ≥ half | 0.5% | 5.9% | 11.7% | 18.2% | 7.2% |
| + − / × ÷ alternation ≤ 1 | 21% | 56% | 52% | 51% | 92% |
| functions, share of operators | 12.7% | 12.4% | 10.9% | 8.1% | 3.7% |
| median operators / > 35 tokens | 14 / 0% | 12 / 13% | 13 / 19% | 15 / 27% | 3 / 1% |
| median variables / one variable | 6 / 9.0% | 4 / 13.2% | 5 / 9.1% | 6 / 5.5% | 3 / 15.9% |

Reading: round 10c raises the structured classes and lowers the held-out rate, but the overall profile (mixed
chains, alternation) moves little. By structure level within 10c, expressions with s ≥ 0.8 still put 13% of their
operators into mixed chains (real formulas: 2%): an LC fallback inside a structured expression spends its whole budget
as one random tree. Real formulas are also much shorter (median 3 operators); the owner keeps T8's length prior.

## How the prior got here

- **Rounds 1–7** (`code/rounds-1-7/`, `code/gen_bank.py`, `pool7.py`): a term pool with reuse, per-expression
  hyperparameters, recursion up to three levels, a nesting rule across levels, corpus-tempered function weights.
- **Round 8, the unified term prior** (`pool8.py`): round 7's ~25 rules as one budgeted recursive term, one grouping
  rule and one copy-or-redraw kernel, built up in six steps from T8. 8a reproduces T8.1 (two-sample KS ≤ 0.027 against a
  5% critical value of 0.035 on operators, tokens, variables and additive terms; every rate within |z| < 1). 8f (all
  steps, budget ×1.3) matches round 7. Coherence measures from the traces: nested variables reusing an ancestor's
  5% → 31% and the parent's +/× class 50% → 67% once inheritance is on. The base adds the hyperbolic weights.
- **Round 9, operator-level forms (negative).** A per-expression structure level with forms (sum of k, product of k,
  ratio, power, function, affine) changed nothing measurable: a random composition of one-to-four-operator forms is
  still a random tree, one-slot forms chain (`pow(rootn(rootn(...)))` fourteen deep), and copies that differ only in
  linear constants merge on simplification.
- **Coverage test** (`coverage.py`): a deterministic parse that flattens + − and × ÷ chains and labels each chain by
  the relation that explains it (repeat: one template after masking variables, constants and exponents; pairing: two
  templates, or k copies of one plus one other part; composition: functions and powers). Real formulas: 98.5% fully
  explained. At equal size (3–8 operators) the priors are already 85–92% explained, so much of the gap is length.
- **Round 10** (`relation_term` in `pool8.py`): the three relations with shares halfway to the real formulas.
  **10b**: the guessed parameters measured on real formulas (`measure_params.py`; round 10's output was already close)
  and set halfway; output unchanged within noise. **10c**: s ~ U(0, 1) instead of Beta(0.5, 1.5).
- **Parked** (`parked/`): a layered term-pool shape sampler and a length-conditioned variant (exact length marginal by
  rejection), set aside by the owner; the size analysis is in LOG.md.

## Measures

`classify.py` (structure classes on simplified prefix), `struct_scores.py` (repetition, alternation, root kind),
`coverage.py` (relations), `measure_params.py` (relation parameters), `gen8.py` (held out under the precise key,
coherence from traces), `pretty.py` (readable infix: flat product chains, signs in front; checked numerically on
4,978 draws).

## Reproduce

Paths in the scripts are absolute on the development machine (valkyrie-light): the T8.1 catalog config of the
flash-ansr repository (`configs/v25.0-T8.1-120M/catalog_train.yaml`), the real-formula corpus in the research
repository (`archive/symbolic-data/2026-11-03-family-coherence-corpus/t83/corpus/`), and the symbolic-data source of
the precise holdout key. Run from `design/term-prior/code/` with the flash-ansr environment:

```
export POOL7_FN_WEIGHTS=hyper=0.5                     # round 8 base and later
python gen8.py 3000 400 r10c r10c                     # rows_r10c.json.gz (statistics), bank_r10c.json (400 traces)
python corpus_prefix.py                               # real formulas as simplified prefix (corpus_prefix.json.gz)
python r10_analyze.py && python measure_params.py     # tables; r10_plot.py needs matplotlib
```

`gen8.py N_STATS N_BANK KEYS TAG` runs any ladder key (`t8`, `r7`, `r8a`–`r8f`, `r8`, `r9`, `r9b`, `r10`, `r10b`,
`r10c`) with fixed seeds; T8 and round 7 run without the weight variable. Startup takes about three minutes (the
holdout catalog), then 10–15 ms per draw.

## Open

- Large terms falling back to an LC tree (step 3) bring big random chunks into structured expressions. Proposed next:
  P(LC tree) = (1 − s)^(b/4), so only small terms fall back ("LC at the bottom"); s = 0 still gives round 8.
- Variables: copies swap to any of 17 variables, so 10c's median is 6 (real formulas: 3).
- Whether the model fails on these classes at all (the premise test on T8 models) is not measured.
- Nothing is trained; held-out rates and class rates are prior properties, not model results.
