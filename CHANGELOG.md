# Changelog

All notable changes to `symbolic-data` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to semantic versioning.

## [Unreleased]

### Added

- **`operator_profiles`** (optional catalog key): per-EXPRESSION operator-class profiles for the
  skeleton sampler. A profile `{name?, weight, operators}` is drawn once per expression; nodes then
  sample from its operator subset (`operators` is a list under the catalog's `operator_weights`, or a
  `{op: weight}` mapping with profile-local weights), and the unary/binary tree recursion runs on
  the profile's weight mass per arity class instead of the fixed `(1, 1)` multiplicities, so a
  profile without unary operators grows binary-only trees. Motivation, measured on the v25.0-T4
  prior: the per-node draw dilutes every expression -- 89% of delivered skeletons contain a
  transcendental and 0.9% are single-class, against 45% and 19% for the 656 benchmark laws.
  Without the key the sampler is byte-identical (regression fixture
  `tests/fixtures/skeleton_sampler_legacy.json`).

## [0.16.2] - 2026-08-31

### Changed
- **The holdout fold skips its opening canonicalization when it is provably the identity.**
  On the generate path the target canonicalization at the sampling site is byte-for-byte the
  call the fold opens with (same mode, effort, max_passes), applied moments earlier -- and
  `simplify` is idempotent, so the fold's first pass re-derived its own input.
  `is_held_out`/`holdout_family_prototype` gain an opt-in `assume_canonical` flag that only
  the generate call site sets, and only when `simplify_mode` equals the holdout canon;
  registration and foreign probes are untouched, so the family key cannot drift by
  construction (golden test: same seed -> identical skeletons, constants and prototypes).
  Measured on 300 real v25.0-T3 draws: **18.52 -> 25.82 problems/s (+39%)**.
- **`tagged_canonical` takes the consumer's target canon** (`mode=`, owner ruling
  2026-08-31): a corpus whose prefix targets are permissive-canonical is
  permissive-canonical in the tagged dialect too. `None` (default) keeps the engine-default
  call byte-identical, so nothing changes without the consumer asking.

## [0.16.1] - 2026-08-30

### Added
- **`simplipy_engine_modes`** (optional catalog key; needs simplipy >= 0.14.2): the lazy-loading
  profile handed to `SimpliPyEngine.load(modes=...)`. The corpus worker profile is
  `['f64', 'permissive']` -- f64 is the engine's always-eager construction substrate, permissive
  carries both the target canon and the holdout family key -- and a spawned worker inherits the
  lean profile through pickle. A deferred mode still loads additively on first use, so the key
  can never change an output, only memory and startup time. Absent = the full artifact, and the
  loader is called without the kwarg, so older simplipy keeps working without the key.

## [0.16.0] - 2026-08-30

### Added
- **The target canon is a config choice: `simplify_mode`** on the generative catalog. Data is
  generated FROM the simplified skeleton -- target == data by construction -- so every licensed
  rewrite, including the permissive corpus tier, is sound at this site; the mode is a
  corpus-design decision, never a soundness one. The simplipy 0.14 migration had silently
  inherited `Mode.f64` here when `mode` became a per-call argument; the key restores the choice
  (default `'f64'` = current behavior, no silent drift in either direction), and a regression
  test spies the configured mode into the actual `simplify` call.
- **Noise as a mixture, with an outlier channel** (`sampling.noise` accepts a mapping): a
  `p_clean` point mass, additive/multiplicative components with LogUniform level, and an
  orthogonal outlier channel. One definition for training AND evaluation, applied strictly
  post-accept so the expression prior stays rejection-shaped over clean values. `Problem` gains
  `outlier_mask_support`/`outlier_mask_validation` -- the generative contamination labels -- and
  carries the realized per-instance draw; the scalar `noise:` form keeps its legacy semantics
  untouched.
- **The outlier prior is rebuilt on literature-ruled numbers.** `rate` and `magnitude` become
  named distributions (uniform, loguniform, beta, lognormal); the `[lo, hi]` pair forms still
  parse and mean what they always meant -- verified byte-identical against the pre-change module
  over 600 instances. `scale: neighbour` measures magnitude against what the nearest neighbour
  in x fails to predict (the previous residual-sigma ruler put outliers at a median of ~3,000
  sigma; robust statistics places them at 1-25). `sign` becomes a per-problem property drawn
  once, and `min_count` conditions the realized count on being at least one by redrawing, not
  clamping.
- **Domain-aware support oversampling**, opt-in via
  `sample_strategy['support_oversampling_max']` (default 1 = the previous loop, pinned
  byte-identical by golden rng-consumption hashes): a failed try doubles the per-try draw in one
  sampler call and keeps the first n in-domain rows, instead of requiring every row of a draw
  in-domain at once.
- **`token_ops.tagged_canonical(engine, expression)`** -- simplify IN the tagged dialect. 796 of
  3,628 curated-catalog expressions canonicalize differently between dialects, so a tagged
  canonical target cannot be obtained by converting the prefix canonical; the counterexample is
  pinned in the tests.

### Changed
- **simplipy 0.14 migration.** The removed `engine.parse` becomes `read_infix` -- the same raw
  reader renamed, tolerant of unknown vocabulary and spelling-preserving, reproduced
  byte-identically over all 6,780 curated prepared expressions and every frozen SymPy output.
  The `normalize_*` walks move HERE, consumer-side, into `token_ops`: recorded expressions and
  constants are concrete ground truth, and the decontamination key must be stable across engine
  artifacts, so neither may canonicalize through the engine's AC state.
  `masking.mask_values_keep_structure` becomes `mask_fittable`. Engine defaults, v24 catalogs
  and test fixtures pin `acj-4`. simplipy pin: `>=0.14.1,<0.15`.
- **The holdout is ONE family key, used identically by both sides.** The adversarial audit
  reproduced seven leak classes -- worst, 64 of fastsrb's 120 laws never registered at all, and
  the literal-affine family of every registered law trained. `holdout_family_prototype`
  (literals masked -> engine AC-canonicalization -> variable relabeling to a fixpoint ->
  constants folded out) is now THE key for registration and probing, so fold-order asymmetry
  cannot exist by construction. Registration desugars `sqrt` before deriving prototypes; a
  declarative pool that drops ANY law raises with the law list instead of warning; holdout
  images are standardized before rounding, with a positive log-scale grid for half-domain
  respellings. fastsrb registers 120/120 laws as 79 canonical families (was 56 laws as 33
  spellings); over-rejection 0.30% at 2.9 ms/probe, all 13 audit leak probes rejected.
- **The holdout fold simplifies BEFORE masking, at permissive/effort=4.** Mask-first replaced
  every literal with a symbolic `<constant>`, so the AC core could not do exact rational
  arithmetic: `x + x == 2x`, `x * x == x^2` and `2(x+1) == 2x+2` each split one family into two
  prototypes, and a draw of one spelling escaped a holdout registered under the other. The
  fixpoint loop re-masks on every pass, so literals the simplifier introduces are masked too.
  The `'__unevaluable__'` sentinel is retired: it fingerprinted the grid, not the law -- 2 keys
  caused 137 of 252 rejections (4.57% of ALL draws) for a reason unrelated to any benchmark
  overlap. 0.94 ms per prototype, ~5% of the per-instance generation budget, O(1) in pool size.
- **Generation is 17x faster single-core** (2.2 -> 37.5 delivered instances/s on the v24
  catalog; the 8-worker flash-ansr pipeline reaches 182): the independent-dimensions support
  draw is vectorized, a 32-row probe block decides a draw before the full box is paid for, and
  all max_tries boxes are drawn and probed in one expression pass with first-valid-in-order
  selection. The accepted-sample distribution is unchanged (golden pins; 500-instance
  distribution spot-check within noise) -- only the rng call schedule moves, where stated.
- **Every literal of an expression is drawn independently.** A mixture `literal_prior`
  resolved its component once per CALL, so `literal_prior(size=n)` handed an n-constant
  expression n values from a single component -- an expression could never mix an integer
  from a `choice` component with a float from a `rounded` one, although the skeleton
  sampler's per-leaf contract says each literal is its own draw. `build_iid_prior_callable`
  (also exported) resolves the component per VALUE. `build_prior_callable` keeps the
  per-call semantics for priors that genuinely share one regime across a draw.

  One consequence is speed: with a per-value mixture, one `size=n` call is identical to n
  `size=1` calls, so `_first_valid_box` draws every box's constants in a single call.
  Measured on the v25 prior over 300 problems x 5 reps: **32.24 -> 18.33 ms/problem
  (1.76x)**, with non-overlapping rep ranges. Literal marginals are unchanged (KS p=0.59
  against the previous builder; a same-builder control gives the same p-value spread).

  The rng call schedule changes, so a fixed seed yields a different sequence and the
  default-path golden pin in `tests/test_support_oversampling.py` is re-captured.

- **Realized data is stored and judged at float64** (BREAKING for distribution
  reproducibility). Every realized array -- support X, targets y, noisy targets, sampled
  literals -- now carries `symbolic_data.numeric.STORAGE_DTYPE`, which is `float64`. The
  f32 grid the generator used to snap to existed for one reason: the consumer's boundary,
  flash-ansr's support tensors and its 32-bit `<ieee754>` constants format. That boundary
  is gone -- flash-ansr serializes constants as 8 IEEE-754 bytes at binary64 -- so the snap,
  the f32 validity bar and the f32 overflow rejection go with it.

  Two consequences worth naming, both measured:

  - **The `rounded` literal prior works again.** `rounded_dist` exists for
    description-length control -- an unrounded float64 draw denotes a rational with a
    ~17-digit numerator. The f32 snap silently undid it: a draw of `1.01139` was stored as
    `1.011389970779419`, and `substitute_constants` recorded THAT into the expression.
    Measured on the v24 float branch, **98.31% of literals changed spelling** and the mean
    literal `repr` ran **10.79 -> 18.09 characters**; over the full 50/50 float/integer
    prior that is ~49% of all literals (integers survive the snap exactly).
  - **`tools/audit_finite_fraction.py` and the sampler now agree.** The tool measured
    `finite_fraction` in float64 while `catalog.py` rejected points in float32, so the
    shipped metadata over-estimated the realizable fraction and undersized the oversampling
    budget for exactly the extreme-magnitude entries. Closed for free -- the tool needed no
    change, the bar moved to meet it.

  Acceptance widens: points a partial-domain expression could not realize at f32 are
  realizable now (`exp(60x)` is finite across all of [0, 10] at binary64, where the f32 bar
  kept only the ~15% low-x slice), so the accepted sequence for a given seed diverges. RNG
  CONSUMPTION does not change -- verified against the pre-change tree, byte-identical next
  draw on all three pinned seeds -- so this is a change of which draws are accepted, not of
  the stream. The rejection guard itself stays, at the f64 boundary: a multiplicative noise
  draw and the outlier shove multiply, and can still carry a finite target out of range.

- **Frozen `.npz` catalogs are not rebuilt** and stay float32. Rebuilding
  would move every stored value by up to one f32 ulp and invalidate every number ever
  published against them. Catalogs written from now on record `storage_dtype` in their
  `_meta` blob and expose it as `ProblemCatalog.storage_dtype`; an absent marker reads back
  as `"float32"`, so a mixed corpus is detectable rather than silent.

### Fixed
- **A lambdified bigint no longer kills a worker.** `lambdify` folds pure-integer subtrees into
  arbitrary-precision Python ints, and numpy ufuncs refuse them (three separate streaming-worker
  deaths in one training run: `exp`, `sinh`, `cosh`). `safe_f` catches the refusal and returns
  all-NaN -- the universal reject signal -- the four direct call sites route through the same
  contract, and the holdout key maps an all-NaN image to a sentinel BEFORE its nan->0 fill,
  which would otherwise collide with a genuine zero image.
- **The `dtype != float32` gate made rootn extinct.** Any tree containing an f64-upcasting
  realization was silently rejected on every draw: rootn ran at a 1/149 skeleton realize rate
  and 0.8% of delivered instances for that reason alone, not because of its domain (now 35.4%,
  against a 37.2% canonical-skeleton rate). Retired with the f32 boundary.
- **Priors and catalogs survive a process boundary.** A spawned data worker receives its source
  by pickle, and both compiled `skeleton_codes` and the mixture-prior closures refused. Mixtures
  are now a picklable `_MixturePrior` object (per-call vs per-value is data on the object);
  `skeleton_codes` drops on `__getstate__` and rebuilds on arrival. Measured: pickle 25 ms /
  1.6 MB, unpickle + recompile 48 ms.

## [0.15.0] - 2026-08-11

### Changed
- **Typed slots are now STRUCTURAL** (BREAKING for distribution reproducibility). A
  slot-bearing operator exposes only its growing seats to the tree walk (effective arity =
  arity - 1 -- exactly the retired hyper-operators' semantics: `pow3` was unary because the
  exponent lived in the name), and the slot literal is written in at placement time.
  Previously the walk grew a full subtree in the slot seat and a post-pass tore it out,
  deleting every operator inside: the shipped operator-count distribution was the drawn one
  smeared downward (at the v24 prior: mean 3.2 operators deleted per tree, drawn-17 trees
  kept 17 operators only 37% of the time). The drawn `n_operators` is now realized exactly
  (probe: 27.5% shipped at k=17 vs 27.6% configured), no work is grown to be discarded, and
  pow/rootn density is essentially unchanged (~0.100 vs ~0.099 of operator tokens).
  Skeleton corpora generated with 0.14.0 are not distributionally comparable.

## [0.14.0] - 2026-08-11

Generation-2 release: the generative pipeline moves to the current simplipy engine family
(`acj-*`, binary `pow`/`rootn`) and yields concrete expressions.

### Added
- **Typed argument slots** (`typed_slots` catalog key): a chosen argument of a chosen operator
  (the `pow` exponent, the `rootn` index) is drawn from a configurable literal prior instead of
  the general subtree sampler — which would otherwise fill it with arbitrary subtrees like
  `rootn(x, tanh(y))` that the `rootn` contract rejects. The retired hyper-operator vocabulary
  encoded this constraint in operator names (`pow2`..`pow5` were unary); on the binary
  vocabulary it lives in the generator. Slot specs validate fail-closed (unknown operator,
  arity < 2, bad argument index, missing prior).
- **`choice` and `rounded` literal distributions.** `choice` samples an explicit weighted value
  set (weights validated: finite, non-negative, matching length; float64 out regardless of the
  input dtype). `rounded` draws from a base distribution and rounds each draw to a sampled
  number of decimals — by default uniform over 1..the draw's own shortest-round-trip precision —
  so literal description length spans coarse to full precision instead of pinning at float64's
  ~17 digits. Non-finite draws pass through; `-0.0` normalizes to `0.0`.
- **`lample-charton-v24` / `lample-charton-v24-bench` catalogs**: the generation-2 training
  prior on the 23-operator vocabulary (`acj-4-3`), with typed exponent slots (integers
  ±2..±10 with linearly decreasing weights; `pow` also draws rounded float exponents), a mixed
  integer/rounded-float literal prior over the ±10 numeric vocabulary, and per-class operator
  rates carried over from v23 (`*` / `/` absorb the retired `mult_k`/`div_k` mass). The
  `-bench` twin differs only in `simplify: false`: a benchmark corpus must not sit at the
  simplifier's own fixed point.
- `token_ops.desugar_sqrt`: curated formulas and SymPy's printer spell the square root as
  `sqrt(...)`, which the current vocabulary has no operator for (the engine's parser passes
  unknown names through as bare tokens); both parse paths now rewrite it to `rootn(u, 2)`.

### Changed
- **Generative catalogs yield EXPRESSIONS, not templates** (BREAKING). Constant leaves
  materialize as concrete numeric literals drawn from `literal_prior`; there is no
  `<constant>` placeholder and no masking step in generation. Which literals a model must
  abstract into a fittable parameter depends on that model's numeric vocabulary, so masking is
  the consumer's decision (`simplipy.masking` provides the policies). The retired `mask`
  catalog key is rejected loudly (`ValueError`) rather than silently ignored — silently
  ignoring it would generate an unmasked corpus for a config that declares masking. Holdout
  decontamination is unaffected: skeleton hashing normalizes numerals to `<constant>` on both
  sides, so a concrete corpus still matches placeholder-form held-out structures.
- **Only explicit `<constant>` tokens are fittable.** All `explicit_constant_placeholders`
  call sites now pass `convert_numbers_to_constant=False`. Previously digit-only literals —
  including small integer exponents — were silently re-templated into fittable constants and
  resampled at data time, and the count mismatch against skeleton normalization made
  ground-truth substitution fail intermittently (`IndexError`).
- **The default engine is `acj-4-3`** (`ProblemSource`, `compile_expression`); the `dev_7-3`
  defaults were dead on simplipy >= 0.12. The undocumented `engine:` ProblemSource config key
  is renamed to `simplipy_engine`, matching the catalog schema.
- Engine `simplify` calls request `form='explicit'` (simplipy >= 0.12 defaults to its tagged
  n-ary dialect, which the prefix consumers reject) and drop the removed `inplace` keyword.
- The curated evaluation path masks via
  `simplipy.masking.mask(tokens, engine, mask_values_keep_structure)`: value-position literals
  become `<constant>` while exponent / root-index literals stay visible (structure, not
  fittable constants). `engine.mask()` no longer exists in simplipy >= 0.12.
- Dependency floor: `simplipy>=0.12` (the shipped catalogs only load there; the declared floor
  and the shipped assets are coherent again).

### Fixed
- **A call-signature `TypeError` inside skeleton simplification now propagates** instead of
  being wrapped as a retryable no-valid-sample failure. Twice in this project's history a
  removed simplify keyword produced an infinite full-CPU rejection loop (the 0.13.0
  `max_pattern_length` floor; the simplipy 0.12 `inplace` removal) because a programming
  error was treated as a rejectable sample. That class is now structurally closed, in both
  the SimpliPy and SymPy branches.
- Ground-truth metadata for concrete skeletons: with zero sampled literals the tokens ARE the
  ground truth; the placeholder-substitution path runs only for frozen placeholder-form specs.
- SymPy-mode simplification (`simplify: 'sympy'`) keeps concrete literals, matching the
  SimpliPy branch (it previously re-templated them into placeholders).

### Removed
- The generation-1 catalogs `lample-charton-v23` and `v23-val` (hyper-operator vocabulary,
  which simplipy >= 0.12 no longer serves) leave the repo, together with the test that
  resolved `v23-val`. Their pinned, revision-locked manifest entries keep resolving for the
  legacy stack (`symbolic-data < 0.14` with `simplipy < 0.12`), so no installed pair breaks;
  the publish tool preserves hosted entries it no longer finds locally.

## [0.13.0] - 2026-07-26

### Added
- **Masking is now an explicit, optional step — ON by default.** `LampleChartonCatalog`
  (constructor / `from_dict` / `from_config` key `mask`) masks sampled skeletons after
  simplification (`engine.mask`: numeric literals relabelled to `<constant>` + operand
  sorting), and `compile_expression` gained the same `mask` keyword for the catalog
  evaluation path. This restores the catalog contract (masked, normalized skeletons) that
  silently drifted when simplipy 0.9 carved masking out of `simplify`: under simplipy >= 0.9
  the sampler had been emitting UNMASKED skeletons (folded literals survived as literal
  tokens instead of fittable `<constant>` placeholders). Masked skeletons are terminal and
  never re-fed to `simplify`. Set `mask: false` to reproduce the unmasked 0.12.x-on-simplipy>=0.9
  behavior.

### Changed
- **Skeleton simplification is now unrestricted** (BREAKING for distribution reproducibility,
  in lockstep with simplipy 0.10.0's removal of the `max_pattern_length` knob). The generative
  sampler and the catalog evaluation path previously capped rule application at pattern length
  4; rule application now always considers every pattern in the loaded ruleset, so freshly
  generated skeletons can simplify further (e.g. dev_7-3's length-5..7 rules now fire during
  sampling). To reproduce the historical capped distribution byte-for-byte, install
  `symbolic-data<0.13` with the matching-era simplipy. Calling the removed keyword against
  simplipy >= 0.10 raised `TypeError`, which the sampler's rejection loop retried forever —
  0.13.0 is therefore also the compatibility floor for simplipy >= 0.10.

## [0.12.4] - 2026-07-10

### Fixed
- **Per-point rejection is float32-storage-aware**: validity is now judged after casting to the
  STORAGE dtype (float32), so points whose y is finite in float64 but overflows float32 (e.g.
  fast-growing integer-sequence formulas near their support edge, found via two ERBench OEIS
  entries) are rejected like any invalid point instead of shipping as `inf` in the frozen
  Problem arrays. The accepted sample remains the declared distribution conditioned on the
  (float32-)valid domain; `meta.finite_fraction` discloses the fraction as before.

## [0.12.3] - 2026-07-10

### Fixed
- **Holdout registration hardening** (adversarial review of 0.12.2, three verified findings):
  (1) frozen problems may declare `meta["alternate_renderings"]` (v-infix) and every rendering
  joins the structure layer -- previously the textbook Planck form evaded BOTH holdout layers
  because only the log-stabilized stored rendering was registered (executed probe: 12/13
  standard-form laws held out, naive Planck NOT; now 13/13 + 4 alternates). (2) The
  `register_holdout_pool` tail now binds each prototype's OWN variable width (mirror of the
  `is_held_out` binding), so laws wider than the registering catalog keep their functional-image
  layer instead of a swallowed NameError silently dropping it. (3) Image-registration failures
  and reference problems contributing nothing now WARN instead of passing silently.
- **`Problem.from_data` rejects invalid reference baselines**: non-finite `y_reference_*` values
  (e.g. a float32-range overflow from a non-log-space law rendering) and `y_reference_*` on a
  black-box problem (`gt_kind="none"`) now raise instead of shipping inconsistent records.

### Changed
- **`first-principles` catalog REPUBLISHED (same-day correction of the initial v1 publish, no
  downstream consumers existed):** the initial artifact stored the masked skeleton
  (`<constant>` placeholders) in `Problem.expression` with a misaligned constants list; the
  corrected artifact stores the concrete literal-token expression + parse-order constants (the
  realize-path convention; verified: 13/13 expressions evaluable and reproducing
  `y_reference_support` to <1e-5 rel), plus `meta.prepared_infix` (registry identity in the same
  parse space as yaml catalogs) and `meta.alternate_renderings` (planck textbook, rydberg
  log(1/.), schechter product form, bode base-2 pow form; each numerically verified equivalent
  on its finite domain). An independent BLIND re-derivation from the raw PMLB data confirmed all
  13 law forms, constants, and FVUs before republication.

## [0.12.2] - 2026-07-10

### Fixed
- **Frozen catalogs now register as holdout pools**: `GenerativeCatalog.register_holdout_pool`
  derived prototypes by iterating catalog `entries`, which a FROZEN (materialized `.npz`)
  `ProblemCatalog` does not have -- registering one silently held out NOTHING. Prototypes now come
  from each stored problem's skeleton (falling back to normalizing its expression tokens);
  black-box (`gt_kind="none"`) problems contribute nothing, by definition. Closes the
  decontamination hole for measured-data catalogs ahead of the first frozen import
  (`first-principles`).
- **`Problem.from_data` normalizes reference predictions**: `y_reference_support` /
  `y_reference_validation` are now coerced to float32 column vectors (like their `y`
  counterparts) and shape-checked, instead of surviving to the npz as raw float64 1-d arrays.

### Added
- `first-principles` catalog (published to HF): the SRBench-2.0 phenomenological track -- 13 PMLB
  `first_principles_*` measured/frozen datasets (MIT) with refit reference laws from
  EmpiricalBench (Cranmer 2023), MvSR (Russeil et al. 2024), and Bazin et al. 2009; all entries
  `gt_kind="reference"`, all points support (no validation split), reference predictions in
  `y_reference_support`. Builder: `tools/build_first_principles.py` (deterministic; fitted
  constants recover h/k, 2h/c^2, R_inf, G, R). Vendored inputs under
  `assets/upstream/pmlb_first_principles/` (NOTICE inside).

## [0.12.1] - 2026-07-10

### Changed
- **Adaptive oversampling for per-point rejection**: the first draw oversizes by the entry's
  disclosed `meta.finite_fraction` (when present) and top-up batches oversize by the observed
  valid fraction (25% margin, budget-capped) — partial-domain entries typically collect all
  points in one or two evaluation rounds instead of ~1/f iterations. Distribution unchanged
  (i.i.d. points; only batch sizing differs).

## [0.12.0] - 2026-07-10

### Changed
- **Per-point rejection sampling** in `ProblemCatalog.realize`: invalid (non-finite) points are
  rejected individually and topped up, instead of rejecting the whole n-point draw. The accepted
  sample is DISTRIBUTIONALLY IDENTICAL (points are i.i.d. and validity is per-point, so the
  all-valid conditioning factorizes): the declared per-variable distribution conditioned on the
  expression's valid domain, under both modes. Removes the exponential cost (f^-n vs 1/f) that
  exhausted `max_trials` for partial-domain entries; a 200x draws cap keeps degenerate entries
  (f ~ 0) on the honest placeholder path.

### Added
- `tools/audit_finite_fraction.py`: per-entry MC estimate of the valid-domain fraction f, written
  into entry meta (`finite_fraction`, plus `low_validity: true` below f = 0.05) as the standing
  disclosure that sampling follows the conditional law; report-only for published catalogs.

## [0.11.0] - 2026-07-09

Optional-ground-truth schema (the benchmark-import program's P0): real-world and black-box
problems become first-class citizens alongside synthetic ones.

### Added
- **`Problem.gt_kind`**: `"exact"` (synthetic GT that generated y) | `"reference"` (the
  historically accepted law accompanying real measurements — stored in the SAME
  expression/skeleton fields, so every existing consumer works unchanged) | `"none"`
  (black-box). Inferred from skeleton/expression presence when omitted (0.10-era dicts and
  call sites keep working); validated in `__post_init__`; placeholders exempt. Round-trips
  through `to_dict`/`from_dict` and frozen `.npz` catalogs (legacy blobs load with inference).
- **`Problem.from_data(x, y, ...)`**: measured-data constructor. Convention: the measured `y`
  IS the fitted target (`y_*_noisy` are copies; `noise=None` = unknown); variables default
  `x1..xd`; a given `expression` best-effort derives the skeleton via simplipy so
  decontamination and recovery metrics keep working for reference problems.
- **Reference-law predictions**: optional `y_reference_support` / `y_reference_validation`
  arrays on `Problem` (the catalog owns the reference expression and precomputes its
  predictions; downstream derives reference-relative metrics without re-evaluating
  expressions). Persisted in frozen `.npz` catalogs when present.

### Fixed
- **Holdout mirror fixes** (same two defects as flash-ansr): `is_held_out` now accepts and
  forwards `n_variables` (foreign skeletons no longer NameError into a false "held out");
  holdout hash keys are canonicalized via `simplipy.normalize_skeleton` (variable renames and
  numeric literals no longer defeat — or leak through — the exact-match layer).
- Structural source filters (`n_variables`/`max_variables`) are vacuous for problems without a
  skeleton instead of comparing against a meaningless 0.
- `mask_unused_variable_columns` is a no-op without skeleton tokens (previously it zeroed
  EVERY column — destroying black-box inputs).

### Compatibility
- Reading old artifacts is unchanged. `.npz` catalogs written by 0.11.0 require
  symbolic-data ≥ 0.11.0 to read (the scalar blob gains `gt_kind`). Never re-save a published
  frozen catalog in place (forward-only policy: its sha256 would change).
- Declarative yaml catalogs remain synthetic-only by design; measured-data catalogs are
  frozen artifacts (`Problem.from_data` → `ProblemCatalog.from_problems` → `save(.npz)`).

## [0.10.0] - 2026-07-01

Post-release audit round (deferred tiers C + D): one breaking API harmonization plus internal
performance/clarity work. No behavior change to sampling, decontamination, or ground-truth values.

### Changed
- **BREAKING: `LampleChartonCatalog.load(directory)` returns the catalog object only** (was
  `(config_dict, catalog)`), consistent with `ProblemCatalog.load`. Read the config separately via
  `load_config(<dir>/catalog.yaml)` if you need it. Internal callers are updated; there is no
  deprecation alias.

### Performance
- **HF manifest is memoized** per `(repo, filename)` in `resolver.fetch_manifest`: a declarative
  `ProblemSource` resolves 2-3x per build, and the manifest was re-downloaded + re-parsed each time.
  Only successful (non-empty) fetches are cached, so a transient network failure can still recover.
- **`LampleChartonCatalog.split` is O(n)** (was O(n^2)): membership tests use `set`s instead of
  scanning the train/test key lists.
- **`sample_skeleton(new=False)` caches the indexable skeleton tuple** and rebuilds it only when the
  skeleton set size changes, instead of materializing `tuple(self.skeletons)` on every draw (the
  streaming resample path calls this once per sample).

### Internal
- Named the previously-inline sampling defaults (`_DEFAULT_MAX_TRIALS`, `_DEFAULT_GENERATE_N_SUPPORT`,
  `_DEFAULT_SET_N_POINTS`) in `source.py`.

## [0.9.5] - 2026-07-01

### Changed
- **Curated (declarative) `realize` now surfaces the CONCRETE ground truth.** `RealizedExpression`
  for a curated entry carries `expression` = the actual formula (its literal constant values intact)
  and `constants` = those values, matching the generative catalog, instead of leaving `expression` as
  the masked skeleton and `constants` empty. `skeleton` remains the masked structural / recovery form.
  So `Problem.expression` / `constants` (and downstream `ground_truth_infix` / `ground_truth_prefix`)
  are now the exact curated formula, not a `<constant>`-masked skeleton.

### Fixed
- **Decontamination keys off `problem.skeleton`, not `problem.expression`.** Matching the exclusion
  keys (which are built from skeletons) and the method's own docstring; this keeps holdout matching
  structural now that a realized `expression` can be the concrete formula (whose parsed structure
  differs from the simplified skeleton). Behaviour for generative sources is unchanged
  (`normalize_skeleton` of the concrete expression and of the skeleton agree there).

## [0.9.4] - 2026-07-01

### Fixed
- **Excluding a frozen (materialized) `ProblemCatalog` now actually decontaminates.** A frozen catalog
  holds realized Problems in `.problems`, so `iter_expressions()` yields nothing -- `ProblemSource`
  holdout key extraction now keys off `.problems` for a frozen excluded catalog instead of silently
  producing zero exclusion keys (a silent decontamination no-op). `iter_expressions()` on a frozen
  catalog now raises a clear `TypeError` pointing at `.problems` rather than returning empty. Adds the
  first direct test of the declarative realize path.

## [0.9.3] - 2026-07-01

Post-release audit cleanup (mechanical fixes; no public API change).

### Fixed
- `sample_data` accepts a scalar (Python-`float`) evaluation result (float32-coerced) instead of
  silently discarding the sample; `safe_f` guards a 0-d / empty result before indexing.
- `get_distribution` validates its `name` / `base_dist_name` / `param_samplers` keys with a clear
  `ValueError`; `apply_on_nested` (config IO) recurses into dicts/paths nested inside list values.
- Removed a dead, drift-prone duplicate `codify`, two stray debug `print`s (holdout / sample_skeleton
  paths), and reconciled an abstract-vs-concrete `sample_skeleton(new=...)` default; docstring fixes.

## [0.9.2] - 2026-07-01

### Fixed
- **`LampleChartonCatalog.iter_entries` no longer defaults to an unbounded stream.** The default
  `method` is now `"iterate"` (matching `Catalog.iter_entries`), so `list(catalog.iter_entries(rng))`
  is BOUNDED on a catalog with a fixed skeleton set instead of hanging forever, and an OPEN catalog
  with neither a fixed set nor a `size` raises a clear `ValueError` (pass `size=N`, or
  `method="procedural"` for an explicit unbounded training stream). `ProblemSource` is unaffected (it
  passes `method` explicitly per mode).

## [0.9.1] - 2026-07-01

Post-release audit fixes (no API change).

### Fixed
- `ProblemCatalog.realize` now raises `CatalogEntryError` (which the source turns into a placeholder)
  instead of an uncaught `KeyError`/`ValueError` when an entry's per-variable spec is missing or
  malformed, so a single bad entry no longer aborts catalog iteration.
- README quickstart corrected: the curated catalogs are Hugging Face artifacts (network on first use,
  then cached), not bundled package data.

## [0.9.0] - 2026-06-30

Completes the family's by-name catalog transition + a terminology cleanup.

### Added
- **`register_holdout_pool` accepts declarative catalogs + by-name/HF refs.** A training catalog can
  now hold out the canonical (declarative) `fastsrb` benchmark by name, not just a saved skeleton-pool
  directory: a string ref resolves via `build_catalog` (name[@version] → HF, config path, inline), a
  directory still loads as before, and a declarative `ProblemCatalog`'s structural prototypes are
  derived from its expressions in the training catalog's space (variables canonicalized).

### Changed
- **Saved-catalog default filename `skeleton_pool.yaml` → `catalog.yaml`** (`LampleChartonCatalog.save`/
  `.load`). Breaking for loaders of pre-0.9 saved directories by the old name; the family now resolves
  catalogs by name, so nothing depends on the legacy filename.
- Added a package `__version__`; purged the term "skeleton pool" from source + docs (the public API has
  been `LampleChartonCatalog`/`ProblemSource`/`load_catalog` since 0.6; this finishes the prose/CLI/docs
  and drops a legacy `skeleton_pool:` config-key unwrap).

## [0.8.0] - 2026-06-30

Catalogs become **pure Hugging Face artifacts** (not bundled in the wheel), the curated **v23**
catalogs are published, and any catalog — declarative or generative — resolves **by name**.

### Added
- **Resolve generative catalogs by name/path.** `build_catalog(ref)` / `ProblemSource({"catalog": ref})`
  now resolve a string ref (local path or HF `name[@version]`) and dispatch on content: a `type:` spec
  builds a `GenerativeCatalog` — **open** (on-the-fly), or **frozen** if it carries inline `skeletons:`
  — while anything else is a declarative `ProblemCatalog`. `from_config` makes `holdout_pools` optional
  and loads optional inline `skeletons:`.
- **Two v23 catalogs published to the HF assets repo:** `v23-val` (the 1000-skeleton frozen validation
  set, a single self-contained generative spec) and `lample-charton-v23` (the open v23 training recipe).
  Resolve with `ProblemSource({"catalog": "v23-val"})` / `"lample-charton-v23"`.
- **Skeleton-level, variable-canonical decontamination.** `ProblemSource` `holdouts: [{exclude: <ref>}]`
  now drops a problem whose *skeleton* (constants collapsed and variables canonicalized via
  `normalize_skeleton`, e.g. `v1.. -> x1..`) matches the excluded catalog — which may be declarative
  (FastSRB) or generative (v23-val), so cross-namespace decontamination is leak-safe. This replaces the
  internal "skeleton pool" holdout for training generation.

### Changed (breaking)
- **Catalogs are HF-only (pure-HF).** The curated catalogs no longer ship in the wheel; a bare `name`
  resolves only via the HF manifest (network on first use, then cached). The vendored package-data
  offline fallback (`resolver._vendored_path` / `vendored_fallback`) is removed — pass an explicit
  local path for offline use.

### Fixed
- A **frozen** generative catalog in `set` mode now iterates its fixed skeleton set **once** (bounded);
  previously it streamed unbounded, so `list(ProblemSource("v23-val"))` would never terminate despite a
  finite `size_hint`, and a fully-excluded source looped forever.

## [0.7.2] - 2026-06-30

Lets a downstream trainer consume a *saved fixed* generative catalog (a held-out validation pool
loaded from disk) through `ProblemSource`, not just an on-the-fly generator.

### Added
- **`ProblemSource` accepts a pre-built `Catalog` instance** as `config["catalog"]` (a
  `GenerativeCatalog` instance -> generate mode), so a consumer can hand it an already-loaded
  catalog (e.g. `LampleChartonCatalog.load(dir)`) instead of only a config dict / ref.

### Changed
- **`GenerativeCatalog.iter_entries(size=None)`** now streams via `sample_skeleton(new=False)`: an
  EMPTY catalog generates a fresh skeleton each draw (training-time streaming), while a PRE-LOADED
  catalog samples from its existing fixed skeletons (a saved validation pool) -- restoring the old
  worker's `sample_skeleton()` default. (It previously forced `new=True`, which would wrongly
  generate fresh skeletons for a loaded pool.)

## [0.7.1] - 2026-06-30

### Added
- **`ProblemSource.catalog`** -- public accessor for the `Catalog` the source samples from (built
  lazily, cached). Lets a consumer that also needs the catalog directly (e.g. a trainer harvesting
  raw skeletons for prompt features) share the source's single catalog instance -- one simplipy
  engine -- instead of constructing a second one.

## [0.7.0] - 2026-06-30

Adds the training-time generation knobs a downstream trainer needs so it can consume a
`ProblemSource` directly (yielding `Problem`s) instead of reaching past it into the catalog's
low-level samplers. Additive; no breaking change.

### Added
- **`sampling.n_support: prior`** (generative catalogs only) -- draw the per-sample support size
  from the catalog's own `n_support_prior` (variable support sizes, the training pattern) instead of
  a fixed count. Requires `n_validation: 0`: every realized row is support, no validation split. The
  distribution is the catalog's existing `sample_data(n_support=None)` path, unchanged; it errors on
  a declarative catalog (no support prior).
- **`ProblemSource.max_n_support`** -- upper bound on a sampled support size (a generative catalog's
  configured support maximum, else the fixed `n_support`); lets a consumer pre-size buffers.

## [0.6.0] - 2026-06-30

Generalizes the catalog abstraction: a `ProblemSource` now samples from a **`Catalog`**, which is
either a declarative `ProblemCatalog` or an on-the-fly **`GenerativeCatalog`**. The procedural
skeleton engine is no longer a private `SkeletonPool` hidden behind a special `generator:` mode;
it is a first-class, public generative catalog (`LampleChartonCatalog`) that produces fresh
expressions and that flash-ansr (training + prompt features) and srbf (sampling baselines) can
consume directly. (0.5.0 hid the engine entirely; two first-party consumers genuinely need a public
generation API, so 0.6.0 exposes it cleanly as a catalog rather than re-exposing the pool.)

### Added
- **`Catalog` (abstract base)** -- the level-1 thing a `ProblemSource` samples from: supplies
  expressions and realizes each into raw `(X, y)` via its intrinsic sampling (`iter_entries` +
  `realize`). `ProblemCatalog` (declarative) and `GenerativeCatalog` (on-the-fly) both implement it.
- **`GenerativeCatalog` + `LampleChartonCatalog`** -- a public generative catalog that grows random
  unary-binary operator trees (the Lample-Charton recipe). Streams fresh skeletons unbounded
  (`iter_entries(size=None)`) or yields a finite reproducible set (`size=N`); exposes raw
  `sample_skeleton(...)` for structure-only consumers (prompt-term harvesting, sampling baselines).
- **`build_catalog(spec)` + `register_generative_catalog(name, cls)`** -- a string/path resolves to a
  declarative `ProblemCatalog`; a mapping with a `type:` key resolves to the registered generative
  catalog. Third parties can register their own generators.
- **`RealizedExpression`** -- the catalog's intrinsic output (`n_points` of `(X, y)` + ground truth),
  which `ProblemSource` splits/noises into a `Problem`.
- **Unbounded streaming generation.** A generative source without `size` streams `Problem`s forever
  (the training-time mode); `size_hint()` is `None`.

### Changed
- **`ProblemSource` config: `catalog:` replaces `generator:`.** A string/path `catalog:` is a
  declarative set; a mapping `catalog: {type: lample_charton, ...}` is generative. The number of
  expressions to draw moves to `sampling: {size: N}` (usage policy); `generator:` is gone.
- **Shared exceptions** live in `symbolic_data.errors` (`NoValidSampleFoundError` still public; new
  `CatalogEntryError` distinguishes a permanently-unrealizable entry from a transient retry).

### Migration
- `{"generator": {<skeleton-pool cfg>, "size": N}, "sampling": {...}}`
  -> `{"catalog": {<skeleton-pool cfg>, "type": "lample_charton"}, "sampling": {"size": N, ...}}`.
- `from symbolic_data._generate.skeleton_pool import SkeletonPool`
  -> `from symbolic_data import LampleChartonCatalog` (same `from_config`/`load`/`sample_skeleton`/
  `sample_data`/`create`/`clear_holdouts` API; it is now a public `GenerativeCatalog`).

## [0.5.0] - 2026-06-30

Completes the data-layer redesign: `SkeletonPool` (and the whole skeleton machinery) is removed
from the public surface, generate-mode is fully `Generator`-driven, and materialization is
shippable. (0.4.0 was a GitHub milestone; 0.5.0 is the first PyPI release of the new data layer.)

### Added
- **`ProblemSource.materialize()` + `to_catalog()` + frozen catalogs.** `materialize()` returns a
  fixed source that re-iterates byte-identical Problems; `to_catalog()` returns a FROZEN
  `ProblemCatalog` (realized `(X, y)`), persisted as a self-contained `.npz` via `.save()` and
  reloaded with `load_catalog` -- the shareable, exactly-reproducible form. This is the no-seed
  reproducibility mechanism.
- **`materialize` CLI command** -- `symbolic-data materialize -c <source-config> -o <out.npz>`
  samples a ProblemSource once and freezes it to a catalog.

### Changed
- **Generate-mode is fully `numpy.random.Generator`-threaded** -- the skeleton/support/holdout
  sampling no longer touches global `np.random`; the source's Generator controls everything
  (verified by a completeness test: same injected Generator + different global seed -> byte-identical
  output). Generate-mode builds `Problem`s natively.
- **The skeleton engine is now private** (`symbolic_data._generate`): `SkeletonPool`,
  `SkeletonSampler`, `SupportSampler`, `HoldoutManager`, and `structure` are ProblemSource's
  internal generate engine, not public modules/classes.

### Removed (breaking)
- **`Sample` / `sample_from_skeleton` / `iter_samples`** (`samples.py`) -- generate-mode emits
  `Problem` directly.
- **`ParserFactory` / `TestSetParser` (`convert_data.py`)** -- the legacy skeleton-ingest of raw
  benchmark files. Superseded by vendored curated catalogs + decontamination via
  `ProblemSource(holdouts=[{exclude: <catalog>}])`.
- **The `generate-skeleton-pool` / `import-data` / `split-skeleton-pool` CLI commands** -- replaced
  by the single `materialize` command.
- The public `symbolic_data.skeleton_pool` / `.skeleton_sampling` / `.support_sampling` /
  `.holdout` / `.structure` import paths (engine is private under `_generate`).
  `NoValidSampleFoundError` and `token_ops.apply_variable_mapping` remain available.

### Deferred (tracked for a later release)
- Publishing the Hugging Face asset manifest + a frozen `holdout_grid` asset; upgrading holdout
  `exclude` from exact normalized-expression match to functional-equivalence.

## [0.4.0] - 2026-06-29

A ground-up redesign of the data layer around one central unit and a clean, versioned, three-level
stack. **Breaking:** the `load_benchmark` / `SpecBenchmark` / `BENCHMARKS` API and the public
skeleton-sampling classes are removed (see Migration).

### Added
- **`Problem`** -- the one central data unit produced by every source (expression, skeleton,
  constants, X, clean + noisy y for support and validation, complexity, provenance, placeholder
  protocol). Noise is on the target y only; `y_*_noisy is y_*` when noise is zero.
- **`ProblemCatalog` + `load_catalog`** -- the level-1 declarative artifact (`{metadata,
  expressions}`): expressions + their intrinsic per-variable sampling. Curated catalogs `fastsrb`
  (120), `feynman` (100), `nguyen` (12) ship vendored as package data.
- **Versioned, repo-agnostic resolver** (`symbolic_data.resolver`): `load_catalog("name@version")`
  resolves from a Hugging Face dataset manifest with a pinned git revision **and a sha256 integrity
  check**, cached locally; `load_catalog("user/repo:name@version")` loads third-party catalogs;
  vendored package data is the offline fallback. Integrity failures never silently fall back.
- **`ProblemSource`** -- one concrete level-2 class (no ABC/subclasses), mode inferred from config:
  a catalog ref (SET), a `generator` block (on-the-fly GENERATE), or inline `problems` (FIXED). Owns
  the usage policy: draw `method`, `n_support`/`n_validation`, `noise`, `problems_per_expression`,
  `layout`, holdouts/filters, and `materialize()`.
  - Holdouts: a list of `{filter: {finite, max_complexity, n_variables, ...}}` and
    `{exclude: <catalog>}` (decontamination by exact normalized-expression match).
  - `materialize()` -> a FIXED source that re-iterates byte-identical Problems: the no-seed
    reproducibility mechanism (sample once, freeze).
- **Unified distribution framework**: the `fastsrb` distribution interprets the FastSRB
  `sample_range`/`sample_type` recipe as one nestable distribution within the existing
  named/nested/mixture vocabulary. All distributions thread a `numpy.random.Generator`. (Finding:
  log-uniform is base-invariant, so FastSRB's base-10 `log` is value-equivalent to the native
  natural-log `log_uniform`.)

### Changed / Removed (breaking)
- Removed `load_benchmark`, `load_spec`, `BENCHMARKS`, `SpecBenchmark`, `FastSRBBenchmark`, and
  `datasets.py` (replaced by `load_catalog` / `ProblemCatalog` / the resolver).
- The skeleton-sampling machinery (`SkeletonPool`, `SkeletonSampler`, `SupportSampler`,
  `HoldoutManager`, `Sample`, `sample_from_skeleton`, `iter_samples`) is no longer public -- it is an
  internal detail of generate-mode `ProblemSource`. `NoValidSampleFoundError` remains exported.
- Reproducibility is no longer seed-based: sampling threads a `Generator` (entropy by default) and
  exact reproduction comes from `materialize()`.

### Migration
- `load_benchmark("feynman")` -> `load_catalog("feynman")` (returns a `ProblemCatalog`; inspect
  `cat["I.6.2a"].prepared` / `.variables`).
- To get `(X, y)` problems: `ProblemSource({"catalog": "feynman", "sampling": {...}})` then iterate.

### Deferred (tracked for 0.4.x)
- Generate-mode's internal skeleton sampler still uses global `np.random`; threading it onto the
  source's `Generator` and fully folding it into `ProblemSource` internals is a 0.4.1 refinement (it
  is distribution-correct today, behind the clean `ProblemSource` API).
- Publishing the Hugging Face asset manifest + a frozen `holdout_grid` asset; `to_catalog()`
  (persistent frozen catalogs).

## [0.3.0] - 2026-06-28

### Added
- **Curated benchmark loaders `feynman` and `nguyen`** for `load_benchmark`, alongside `fastsrb` --
  all three now vendored as package data from their canonical upstreams (no download) and stamping
  `benchmark.provenance`:
  - `fastsrb` -- the 120-equation FastSRB spec, vendored verbatim from upstream `viktmar/FastSRB`
    `src/expressions.yaml` (MIT).
  - `feynman` -- the 100-equation Feynman Symbolic Regression Database (Udrescu & Tegmark 2020),
    formulas + uniform FSReD ranges (via the `psaegert/ansr-data` `FeynmanEquations.csv` mirror).
  - `nguyen` -- the 12-equation Nguyen suite (Uy et al. 2011), formulas + ranges from the
    `deep-symbolic-optimization` `benchmarks.csv` (Petersen et al. 2021, BSD-3).
- `SpecBenchmark` -- the general spec-driven sampler extracted from `FastSRBBenchmark` (now a thin
  subclass), accepting either a YAML path or an already-parsed mapping. Exported from `symbolic_data`.
- `tools/build_benchmark_specs.py` -- reproducible, self-verifying generator that fetches each
  benchmark from its canonical upstream, converts it, and gates the converted specs on a numerical
  oracle (`simplipy(prepared)` vs `sympy(raw)`, `allclose` on shared inputs) before writing.

### Fixed
- `load_benchmark("fastsrb")` now works out of the box. Previously the default resolved an
  `expressions.yaml` from the `psaegert/ansr-data` HF dataset that was never uploaded there, so the
  default 404'd. The spec is now vendored as package data (HF remains available via `revision=...`).

### Verified
- All 100 Feynman + 12 Nguyen equations pass the numerical oracle at `rtol=1e-9` (the converted
  specs). The same oracle runs offline over the shipped specs in the test suite (sympy-gated), plus a
  finite-sampling integrity guard. `fastsrb` is vendored verbatim, so it is gated on parse +
  finite-sampling integrity instead (118/120 sample finite; `II.24.17` and `B4` are mostly-non-finite
  by construction upstream and are skipped gracefully by `iter_samples`).
- Six known `# variables` count typos in the upstream `FeynmanEquations.csv` are corrected from the
  populated columns (reported by the build script, not silently dropped).

### Licenses
- `THIRD_PARTY_LICENSES` now reproduces the MIT (FastSRB / viktmar) and BSD-3-Clause (DSO) license
  texts and attributes the FSReD source; the curated specs reproduce only mathematical facts
  (formulas, ranges, variable names).

## [0.2.0] - 2026-06-28

### Added
- Data-prep CLI (`symbolic-data generate | import | split-skeleton-pool`) and benchmark ingest
  (`ParserFactory`), with the `[ingest]` extra.

## [0.1.0] - 2026-06-28

### Added
- Initial release: the model-agnostic symbolic-regression data layer carved from flash-ansr --
  skeleton/expression sampling, priors, holdout, `iter_samples`, registries, and `load_benchmark`.
