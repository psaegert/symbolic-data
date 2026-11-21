# Structure classes in the T8 training prior, and what the benchmark holdout removes (2026-11-20)

Question (owner): the prior's diversity is per-node, so "clean" high-level forms (pure polynomials, sums of
rationals, superpositions of similar terms) should be almost never sampled. Is that so, and is it the prior?

Setup: 20,000 raw skeletons from `flash-ansr/configs/v25.0-T8.1-120M/catalog_train.yaml` (symbolic-data 0.18.0 main,
seed 0), sampled with `decontaminate=False`, then `is_held_out` applied as the generate path does
(`measure.py`, classes in `classify.py`, table from `analyze.py`). Delivered = raw conditioned on not held out
(training resamples held-out draws, max_tries 4).

| class | raw % | held out % of class | delivered % |
|---|---|---|---|
| all draws | 100 | 6.4 | 100 |
| polynomial | 2.15 | 60.0 | 0.92 |
| polynomial, >= 3 variable terms | 0.47 | 29.5 | 0.36 |
| univariate polynomial, >= 3 terms | 0.03 (6/20k) | 100 | 0.00 |
| affine in >= 2 variables (weighted sum) | 0.23 | 78.7 | 0.05 |
| >= 2 proper rational terms | 1.26 | 4.7 | 1.29 |
| >= 3 terms, all one motif (superposition) | 0.17 | 54.5 | 0.08 |
| >= 2 terms sharing a non-trivial motif | 0.78 | | 0.70 |

Delivered members of these classes are mostly multi-variable sums of bare variables with unit coefficients
(see `analyze.py` output); clean univariate forms are the ones the holdout removes.

Real-formula corpus (11,007 of the 11,668 Wikidata/Wikipedia formulas of the family-affinity study parsed;
`corpus_classes.py`; coefficients are symbols there, so classes are only roughly comparable): polynomial >= 3 terms
2.3%, sum of >= 2 rational terms 0.9%, superposition >= 3 terms 1.2%, >= 2 additive terms 17%, >= 3 terms 3.4%.
Definitional formulas are mostly single-term; empirical models (Sellmeier, Shomate, multi-exponential, Drude-Lorentz)
are not in this corpus.

## The holdout family key collapses exactly these classes
`holdout_family_prototype` ends in `get_structural_prototype`, which folds every operator with a `<constant>` operand
into its other operand. Since pow/rootn exponents are literals (typed slots), `pow x 2 -> x`, `rootn x 2 -> x`,
`c/(x+a) -> x`. So (`measure.log` PROBE lines, `attribute.log`):
- `x1`, `a*x1+b`, `a*x1^2+b`, `exp(c*x1)`, `log(a*x1+b)`, `a/(x1^2+b)+c` -> prototype `x1`, held out (66 laws
  register `x1`, e.g. fastsrb `8.854e-12*v1**2/2`).
- any 2-term univariate polynomial and `a/(x+b)+c/(x+d)` -> `+ x1 x1`, held out (30 laws, e.g. physo-class
  `-4.905*v1**2 + 3*v1 + 1.5`).
- any 3-term univariate polynomial and 3-term partial-fraction sum -> `+ x1 + x1 x1` (nguyen `x^3+x^2+x`, koza,
  constant catalog).
- `(x+a)/(x^2+b)` -> `/ x1 x1`; `a*x1^2+b*x2^2+c` and `8/(2+x1^2+x2^2)` -> `+ x1 x2` (keijzer); 3-variable linear
  model -> `+ x1 + x2 x3`.
- sums of exponentials / sines with different rates: held out through the image layer (prototype `+ exp x1 exp x1`
  is affine-equivalent to `exp x1`).
The docstring's claim that x^1 and x^3 separate is undone by the final fold. Any structured prior component
(polynomials, partial fractions, mode sums) would be deleted by this key in large part.

Untested: whether the deployed models actually fail on these classes (direct test with T8-20M + GT-skeleton refit
oracle not run); how much a corrected key changes the held-out set.

## 2026-11-20 (later): the precise holdout key, and max_length 40

### Holdout fix (symbolic-data branch `fix/holdout-precise-family`, worktree `.claude/worktrees/holdout-precise`, base origin/main 13b2367, UNCOMMITTED)
Opt-in catalog key `holdout_family: precise` (absent = legacy, byte-identical). Criterion: a draw is held out iff setting
its free (fittable) constants to identity values (shift 0, scale 1) yields a law's family up to output affine;
exponents and root indices are structure (simplipy `mask_fittable` roles). Pieces: role-aware masking; affine-only fold
(`c/X -> inv X`, `c^X -> exp X`, additive signs are spelling); one canonicalize-relabel-mask-fold pass over the
constant-free prototype (variable-loss guarded); finite-only image standardization (legacy zeroes NaNs first, which
breaks a*f+b invariance); registration closure with two generic-coefficient members per law.
Leak audit (`audit.py`; 6,660 law renderings of the 29 T8 pools; members = canonical, 3 redraws of fittable constants,
1 random affine wrap):

| escaped | legacy | precise |
|---|---|---|
| canonical form | 3 | 0 |
| redrawn members (3 passes) | 18/16/19 | 0/0/0 |
| affine wraps | 1,660 | 7 |
| law members escaping precise but not legacy | | 3 |

20k raw T8.1 draws held out 6.42% -> 2.80%; poly >= 3 terms 30.5 -> 6.3%; superposition >= 3 54.5 -> 3.0%; affine >= 2
vars 78.7 -> 53.2%. Cost 11.7 vs 7.0 ms/draw (`timing.py`), registration 199 vs 145 s. Tests: 378 pass (new
tests/test_holdout_precise.py). Still held by the doctrine: K-term sums of one core whose single core is a benchmark law
(a/(x+b)+c/(x+d) -> inv x; sums of exponentials/sines; (x+a)/(x^2+b) -> inv x), since identity settings merge the terms.

### max_length 35 -> 40 (`length_study.py`, `length_study.log`)
Decoder: RoPE, decoder_max_seq_len 256, longest T8 instance 193 -> no architectural limit. T8.1 prior, max_ops 17: 6.6%
of simplified draws exceed 35 (5.8% in 36-40, 0.8% > 40), 17% of 17-op draws; the excluded draws carry 4.4 literals vs
2.4 (the cap is a bias against coefficient-rich forms). Clean classes never exceed 35 in the current prior (rare and
short). max_ops 20 would put the median at 32 and 33% above 35 (length_exponential prior rises with n). Benchmarks:
0.5% of 6,633 law renderings exceed 35 (livermore2 9.4%, erbench-phybench 19.3%, feynman-bonus 5%); FastSRB, Feynman,
SRSD all <= 34. Structured forms (canonical tokens): poly deg 5 / 6 with all coefficients 29 / 35; Sellmeier 3 terms
37; partial fractions 3 / 4 terms 19 / 25; 3 exponentials 22; Pade [2/2] 23; linear model 6 vars 25.

## 2026-11-20: prototype of the owner's term-reuse prior (`compose_sim*.py`, `holdout_composed.py`)
tau0 (LC, 1-4 ops) whose growing leaves take, w.p. q = 0.5, a modified copy of tau1 (LC, 1-4 ops): variables
re-drawn 50%, exponents 50%, constants 90% fresh / 10% shared, 10% arity-preserving operator relabel. T8.1 sampler,
simplified, no holdout unless stated; 6,000 draws per row (% of draws):

| | plain LC | owner's numbers | + coefficient per copy, 1-4 var alphabet, sum combinator always | corpus |
|---|---|---|---|---|
| poly >= 3 terms | 0.57 | 3.1-3.6 | 2.6-3.2 | 2.3 |
| univariate poly >= 3 terms | 0.03 | 0.00 | 0.3-0.6 | 0.05 |
| sum of >= 2 rational terms | 1.4 | 1.6-2.0 | 3.9-4.4 | 0.9 |
| weighted sum >= 2 vars | 0.23 | 3.3-3.5 | 3.9 | |
| superposition >= 3 one motif | 0.17 | 1.6-2.2 | 1.3-1.7 | 1.2 |
| median length | 26 | 10 | 9 | |
| transcendental | 74 | 45 | 28 | 9 |
| held out, precise key | 2.8 | 17 | 49 | |

Diagnosis: copies that differ only in linear coefficients MERGE on canonicalization (c1 f + c2 f): with 2-4 copies per
draw only 46-48% keep >= 2 terms, 18-21% >= 3. Copies must differ in a nonlinear parameter (exponent, inner
shift/rate, variable); tau1 rarely carries a constant or exponent; tau0/tau1 draw variables independently (many
variables); a random LC tau0 rarely puts copies in a sum. Holdout: composed draws are short and benchmark-like
(univariate polys 93% held, weighted sums 89% under the sum combinator).

## 2026-11-20: rounds 2-3 of the term-reuse prior (`compose_v2.py`, `compose_v3.py`, `compose_v3.log`)
Round 2 (owner: n0 larger, q per expression, copies swap to any variable; plus the general term pool C): n0 from the
T8 prior makes tau0 the mess (median 47 tokens, 10 vars, clean classes below plain LC); swaps to any of 17 variables
raise the variable count; the pool with LC's pool-size rule behaves like a bigger LC tree. Round 3 (owner: n0 1-8,
new-variable gate 50% then 50% per variable, hierarchical pool with per-expression hyperparameters rho, alpha, r,
kappa): median 11-13 tokens, median 2-3 vars (19-28% univariate vs 10% plain), poly >= 3 terms 1.2-1.8%, univariate
poly >= 3 terms 0.03-0.07%, sum of rationals 1.4-1.7%, superposition 0.65-1.1%, transcendental 50-54%, terms >= 2
33-36% (plain 45%), held out under the precise key 18-20%. Rows with hyperparameters in compose_v3_rows.json.gz.

## 2026-11-20: round 4 (`compose_v4.py`, `analyze_v4.py`): budget, shared family urn, copy-merge accounting
Budget B ~ U{5..10}, composite n1 = clip(B - n0 +- 1, 1, 10); shared Polya family urn (alpha 0.36, corpus families) over
tau0, all pool terms and copy relabels. Copy redundancy (a copy that simplifies together with an earlier copy of the
same term in the same additive / multiplicative bag): 11-12% of copies, 16-17% of re-uses, 16-17% of draws; by what the
copy changed: constants only 16-17%, variable 6%, exponent 8-11%, operator 9-10%. Shared urn: draws mixing >= 2
families (of draws with a function) 34-37% -> 12% (plain LC 47%); class rates and transcendental share unchanged.
Budget changed little (median length 13 -> 15; trivial <= 5 tokens 19.8 -> 18.8%, plain LC 1.4%): 91% of trivial
draws have NO composite in the pool (31% of draws), P(trivial | no composite) 54% (73% at pool size 1), P(trivial |
>= 1 composite) 3%; 23-24% of trivial draws are constants (pool = {c}).

## 2026-11-20: round 5 (`compose_v5.py`, `analyze_v5.py`, `size_ecdf_round5.png`): budget from the T8 prior
B ~ T8 operator-count prior (0..17, length_exponential); n0 ~ U{1..min(8, B)}; R = B - n0; composite share
r_eff = r + (1-r) R/B; composite sizes n1_j = clip(round(R / (n_T u_j)) +- 1, 1, cap) so the leaves fill R on average;
constants only once a variable-bearing term is in the pool; constant laws at T8's rate (0.3%); shared urn.
vs T8 (4,000 draws): tokens median 21 (cap 4) / 24 (cap 8) vs 26; operators 11 / 13 vs 14; trivial <= 5 tokens
4.2 / 3.7% (round 4 18%, T8 1.4%); univariate 9% (T8 9.5%); leaves' operators / R median 0.88 / 1.00; sum of rationals
2.1 / 1.8% (T8 1.3); shared form in >= 2 terms 3.3% (T8 0.95); poly >= 3 terms 1.1 / 0.7 (T8 0.6); transcendental
66-68% (T8 73); mixed families 12-14% (T8 47); held out 5-6% (T8 2.7).

## 2026-11-20: round 6 (`compose_v6.py`, `analyze_v6.py`, `size_ecdf_round6.png`): alphabet coin flips
tau0 alphabet A0 drawn with LC's size rule; tau1 variables from A0 (50%: full A0 or a random subset per tau1, 50/50),
one shared level-1 alphabet (25%), or each tau1 its own LC draw (25%, = round 5). Fix applied: a composite term always
carries >= 1 variable (small alphabets otherwise made constant-only terms: constant laws 3.2%, before the fix).
vs round 5 (cap 4 / cap 8): median variables 5/6 -> 4/4, univariate 9/8.5% -> 19.5/18.4% (T8 9.5%); by branch
univariate tau0-full 21-24%, tau0-subset 28%, shared-L1 16-17%, own 8-11%; structure classes unchanged within noise
(shared form 3.4-3.9%, sum of rationals 1.6-1.75%, poly >= 3 terms 0.8-0.9%, univariate poly >= 3 terms 0-0.12%);
held out 10.0 / 8.2% (T8 2.7%); trivial 5.1 / 4.1%; constant laws 0.3 / 0.5%.

## 2026-11-20: round 7 (`explorer/pool7.py`, `r7_sim.py`, `r7_analyze.py`, `size_ecdf_round7.png`)
Recursive term pool (<= 3 levels) with per-level hyperparameters (additive share s ~ Beta(.5,.5), own-tree share f ~
Beta(2,3), rho ~ Beta(2,1), alpha, r, kappa, gate, branch) copied from the parent with prob 0.6/0.35/0.2 by depth, else
redrawn from the root prior; terms > 4 ops recurse (20% stay one LC tree <= 8 ops); nesting counted through binary nodes
and across levels (a term starts at the deepest function depth of its slots). Budget scale x1.3 matches T8's operator
count best (KS 0.21; median 13 vs 14; p90 22 vs 17). Nesting at x1.6: function inside a function 29% (old rule) -> 10%
(T8 21%), 3-deep 5.6 -> 0.4% (T8 3%). Operator marginals near T8 (+ - * / ~21% each vs 20.5; functions 7.9 vs 9.1% of
operator tokens). Reuse: uses per composite 2.2 (round 6: 1.63), all-once 16% (34%). Corpus function weights (sqrt
tempered, T8 total kept): inverse trig/hyperbolic 38% -> 12% of function nodes (corpus 1.7%). Corpus check: in 11,668
real formulas, exp 38%, log 18%, sin+cos 31%, abs 9.5%, hyperbolics < 1%, inverse functions 1.7% of function nodes.

## Length conditioning (2026-11-20, `length_conditioning/`)

Owner asked to stop the blow-up and target a smooth length distribution on 1..~50 with an exponential tail to 100.
Proposal: p(E) = t(|E|) * p0(E | |E|), i.e. p0(E) * t(|E|) / q0(|E|) (LC's "size first, then structure", one level up).
Sampler: draw from p0 aborting once > 100 operators (exact: those are rejected anyway), accept with t(N) / (M q0(N)).
1.5M p0 draws per variant (node, reuses the sampler's T8 port), 200k T8 draws.

- Spec (A): 34.7% of draws exceed 100 ops. Acceptance 1.6% (T8 form to 50), 2.2% (T8 shape stretched 17->50), 0.6% (exp(-N/15)).
- Given the length, A is single-tree for N <= 17 (composite 0.1%) and composite for N >= 18 (100%); reuse 10% at 18-25,
  81% at 36-50, ~100% above 50; superposition (>= 2 top-level summands copying one term) only 0.3-5%.
- B (deeper caps 17, 8, 4, 2, 1, T8 form): 13.3% above 100, acceptance 6-8%; same jump at 17/18.
- C (B + root size uniform 0..17): 6.6% above 100; composite 15-16% at 6-17 ops; acceptance 5-7% (rising targets), 33% (decay).
- Tokens ~1.9 x operators: rising targets give median ~96-101 tokens, p99 ~165-169; decaying target median 21, p99 129.
- Draws (draws.js): length-50 expressions are big random root patterns with embedded copies; superposition stays rare.

**Parked 2026-11-20 (owner):** the term-pool spec doc, the shape sampler and length conditioning. Work resumes from round 7.

## 2026-11-20: round 8, the unified term prior (`explorer/pool8.py`, `gen8.py`, `r8_analyze.py`, `r8_plot.py`, `r8_ecdf.png`)
Round 7's ~25 rules rewritten as three primitives: (1) one budgeted recursive Term(b) (atoms = groups with budget 0;
no term kinds, thresholds, random component or level cap); (2) one grouping rule for slots, k = 1 + Bin(L-1, 1-rho)
groups, each one slot, the rest by Dirichlet(alpha) -- with rho ~ U(0,1), alpha = inf this IS T8's leaf rule; urns for
the additive share and families; (3) one copy-or-redraw kernel (hyperparameters and atom variables inherited with
prob lambda^depth, lambda ~ Beta(3,2) per expression; copies redraw parts). Budget split v ~ Dirichlet(1) with
unbiased stochastic rounding (an exact split often does not exist when every group has several slots): raw ops - B
mean ~0, exact 34-41%. T8.1 as trained has no family urn ("legacy leaf draw, no families"); the urn enters at 8c.
Ladder, 3,000 draws each (+ T8 and round 7 regenerated), simplified:
- 8a (T8 special case) vs T8: every KS < 0.027 (5% critical 0.035), every rate |z| < 1 -> T8 reproduced.
- 8b +recursion: ops median 11, nested 74%, fn-in-fn 22.9 -> 6.7% (round-7 nesting rule), held 8.2%.
- 8c +reuse: identical copies merge: ops median 7, trivial 11.5%, one variable 25%, held 16.8%, sums of rationals 0.77%.
- 8d +inheritance: no measurable change on these statistics.
- 8e +copy changes: ops 9, trivial 7.4%, held 9.9%, shared form 4.8%, poly >= 3 terms 2.4% (highest), >= 3 one form 1.6%.
- 8f budget x1.3: ops 12/20 (T8 14/18, r7 13/23), KS 0.19 (r7 0.21), > 35 tokens 13% (r7 22%), trivial 4.5% (r7 5.0),
  one variable 12% (r7 17), shared form 6.9 (r7 6.6), >= 3 one form 1.1 (1.07), sums of rationals 1.0 (r7 1.4, T8 1.3),
  poly >= 3 1.5 (1.5), held 7.6 (8.9). => matches round 7 with far fewer rules; target classes still ~T8 level.
Explorer: priors r8a-r8f (400 traced draws each, lazy-loaded samples_r8*.json; #r8f etc. selects a prior).

### Round 8 adopted as the base (owner 2026-11-20); coherence measures; hyperbolic weights
Owner: round 8 replaces round 7 as the base; root shape (structure classes) deferred; inheritance kept pending a
coherence measure. Base = 8f + the six hyperbolic functions (sinh cosh tanh asinh acosh atanh) at weight 0.5 (T8: 1)
via POOL7_FN_WEIGHTS=hyper=0.5: a function label is hyperbolic 4.0% -> 2.3% each, every other function 4.0 -> 4.5%,
pow/rootn 20 -> 22.7%, family-urn base shares follow; mF 0.926 -> 0.917. Runs: gen8.py ... base / ladder (seeds pinned;
the ladder reproduces the first run draw for draw), rows_{base,ladder}.json.gz, bank_{base,ladder}.json.
Coherence (from traces): nested variable atoms whose variable an ancestor term uses: 8b 8%, 8c 5% -> 8d 31%
(8e 27, 8f 30, base 32); nested trees with the parent's majority +/x class: 8b 51, 8c 50 (chance) -> 8d 67 (base 68).
So inheritance acts, just not on the earlier output statistics. Family urn: >= 2 families among draws with >= 2
functions 88% (T8, 8b) -> 23% (8c onward; round 7 24%).
Base vs 8f: hyperbolic share of function nodes 41.9 -> 25.3%; exp 6.0 -> 7.3, log 5.8 -> 7.5, sin+cos 13.2 -> 16.6;
functions per draw 1.71 -> 1.51 (mass moved to pow/rootn); transcendental 68 -> 62%; lengths unchanged (12/20 ops);
sums of rationals 1.03 -> 1.67 and shared form 6.9 -> 4.8 (about 3 SE each, unexplained). 100 draws each of T8 and
the base: explorer/draws_t8_r8.md (make_md.py).

## 2026-11-21: round 9 -- structure level s and operator-level forms (NEGATIVE) (`pool8.py` r9/r9b, `struct_scores.py`, `corpus_prefix.py`, `r9_analyze.py`, `r9_plot.py`)
Owner: structured but not always; superpositions of rationals possible, not dominant. Design: s ~ Beta(0.5, 1.5) per
expression; a term picks a form (sum of 2-5, product of 2-4, ratio, power/root, function, affine; per-expression form
urn) with probability s, else an LC tree. 9: s inherited through the lambda^depth kernel; 9b: one s for every term.
Structure scores (struct_scores.py): repetition = share of nodes in a repeated subtree (vars/consts masked, + = -),
alternation = most +-/*÷ switches on a root-leaf path, root kind. Real formulas: 10,801 of 11,668 corpus formulas
converted to simplified prefix (corpus_prefix.json.gz).
| | T8 | r8 base | r9 | r9b | real formulas |
| >= 3 terms of one form (%) [target 3-5] | 0.17 | 1.13 | 1.27 | 1.23 | 1.12 |
| sum of >= 2 rationals (%) [2-4] | 1.27 | 1.67 | 1.90 | 1.60 | 1.02 |
| polynomial >= 3 terms (%) [3-5] | 0.37 | 1.37 | 1.93 | 1.87 | 1.80 |
| repetition >= 0.5 (%) | 0.5 | 5.9 | 7.8 | 8.1 | 7.2 |
| alternation <= 1 (%) | 21 | 56 | 54 | 55 | 92 |
| median tokens / <= 5 tokens (%) | 26 / 1.4 | 23 / 4.5 | 23 / 4.7 | 23 / 4.7 | 7 / 47 |
| held out (%) | 2.9 | 7.3 | 7.8 | 9.5 | - |
Diagnosis: (1) r9: 57% of nested terms redraw s (mean 0.25) -> forms only 28% of operators even at s > 0.6; 9b fixes
that (50%) but classes and scores still do not move with s. (2) A random composition of operator-level forms is still
a random tree; one-slot forms (power, root, function, affine) pass their whole budget down -> long chains
(pow(rootn(rootn(pow(...)))) 14 deep at s = 0.92); superposed copies that differ only in linear constants merge on
simplification. Structure is a relation between parts (shared template with nonlinear parameter variation, products of
powers, ratios of polynomials), not a property of single nodes. Also: real formulas are short (median 3 ops).

## 2026-11-21: coverage test of three relations on real formulas (`explorer/coverage.py`, `coverage.json`)
Parse: flatten + - / * ÷ chains; non-constant operands; templates mask vars, consts, exponents (+ = -), summand
coefficients stripped, product factors as powers of their base. Repeat-and-combine = all operands one template;
affine = one non-constant operand; pairing = two templates; mixed = otherwise. Functions/powers = composition.
- Fully explained (no mixed chain): real formulas 88.9%, T8 12.1, r8 28.9, r9b 32.0.
- Most leftovers are k copies + one other part ((2,1) 56%, (3,1) 15%, (4,1) 6%): counting "repeat paired with one
  part" as pairing, real formulas 98.5% explained (T8 37.2, r8 55.3, r9b 59.9); at equal size (3-8 ops) real 98.8,
  T8 84.7, r8 89.5, r9b 91.6 -> much of the gap is length (real median 3 ops vs 12-14).
- Real formulas by operator share: repeat products (monomials) 54%, repeat sums 7.6% (3+ copies 1.9%), affine 9.6,
  pairing 9.2, functions 3.7 (priors ~12), powers 2.9, mixed 13 (priors 40-47). Root: repeat product 43.5%, pairing
  15.5, affine 10.4, power 7.5, repeat sum 7.0, atom 6.0, mixed 9.1, function 1.1.

## 2026-11-21: round 10 -- three relations (`pool8.py` relation_term, `r10_analyze.py`, `r10_plot.py`, `draws_r10.md`)
Round 8 base + one s ~ Beta(0.5, 1.5) per expression; a term with operators is, with probability s, a relation from a
per-expression urn (shares halfway between equal and the real-formula root shares: repeat product 31.5, pairing 21.4,
affine 13.9, power 12.3, repeat sum 12.0, function 8.9 %); otherwise an LC tree. Repeats: k = 2-5 copies of one
nested term, varied (variable swap 0.7, exponent 0.5) and re-varied if simplification would merge them; composition
may not directly contain composition. 3,000 draws (vs T8 / r8 / r9b / real formulas):
- >= 3 terms of one form 2.10% (0.17 / 1.13 / 1.23 / 1.12; target 3-5); sums of rationals 2.07 (1.27 / 1.67 / 1.60 /
  1.02; target 2-4); polynomials >= 3 terms 2.57 (0.37 / 1.37 / 1.87 / 1.80; target 3-5).
- held out 5.6% (2.9 / 7.3 / 9.5); trivial 2.9; > 35 tokens 18.9; median variables 5 (copies swap to any x1..x17).
- repetition >= 0.5 10.8% (real 7.2); coverage (fully explained) 59% (r9b 60, real 98.5); mixed-chain operators 38%
  (r8 41, real 13); alternation <= 1 52% (real 92); root repeat product 5.7% (real 43). 74% of roots are LC trees.
Parameters derived vs guessed: relation set checked by coverage; shares from corpus (halfway); s / halfway / length
owner's choices; k distribution, copy variation, wrappers, urn concentration, lambda are guesses (measurable on corpus).

## 2026-11-21: parameters measured on real formulas; round 10b (`measure_params.py`, `measured_params.json`)
Owner: the superposition of rationals was one example of structure, not a target class. Measured with the coverage
parse (real formulas / round 10 output): product repeats 2/3/4/5/6+ copies 57/23/11/4/5 vs 71/14/7/3/4; sum repeats
87/11/1/0/0 vs 76/13/6/2/3; copies differ by variable 99/89% (product/sum) in both; constant-only variation of sum
copies (e.g. partial fractions) 1%; distinct variables per repeat ~2.4-2.9 both; product factors negative exponent
23 vs 31%, exponent other than +-1 12 vs 19%; coefficients 9 vs 13%; pairing operators equal within 3 points.
Round 10b sets the relation parameters halfway (k distributions, pow wrapper 0.33, ÷ 0.26, coefficient 0.42, minus
0.24, exponent change 0.44): output barely moves (product 3+ copies 29 -> 30%, exponent != +-1 19 -> 17%), all class
rates and scores within noise of round 10. Reason: relations shape only the structured minority (74% of roots LC).
By structure level s (10b traced bank, n per bin 162/109/71/42/16): fully explained 51/55/62/64/88%, mixed-chain
operators 25/21/16/10/4%, alternation <= 1 52/48/48/43/75% (real formulas 98 / 2 / 92) -> s is the lever.

## 2026-11-21: round 10c -- structure level s ~ U(0, 1) (mean 0.5) (owner: "0.5 first")
10b with s ~ Beta(1, 1). 3,000 draws (vs 10b / real formulas): >= 3 terms of one form 4.10% (2.13 / 1.12), sums of
rationals 2.33 (2.13 / 1.02), polynomials >= 3 terms 4.37 (2.63 / 1.80) -- all three inside the earlier ranges; held
out 4.8% (6.1); root is a relation 53%; fully explained 62.4% (58.1 / 98.5); mixed-chain operators 34% (37 / 13);
repetition >= 0.5 18.2% (11.7 / 7.2); alternation <= 1 51% (52 / 92); functions 8.1% of operators (10.9 / 3.7);
transcendental 46%. Longer after simplification (less collapse): median 15 ops / 29 tokens, > 35 tokens 27.4% (19.2);
median variables 6, one variable 5.5% (9.1 / 15.9). By s bin (traced bank, n ~70-90 each): fully explained
56/53/50/69/64%, mixed operators 22/21/23/14/13% -- the 10b high-s bin (88%, n = 16) was noise; even at s >= 0.8
mixed chains stay at 13% (real 2): LC fallbacks inside structured expressions get large budgets.

Rendering (2026-11-21, owner): draws are printed with pretty.py (flat product chains, numerator / denominator, signs pulled
to the front; checked numerically on 4,978 draws); the explorer's TeX does the same and keeps a differently coloured term
grouped inside a product.
