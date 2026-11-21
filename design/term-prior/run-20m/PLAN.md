# Does the term prior (round 10c) train a better 20M model than T8? Plan, fixed 2026-11-21 before any result

Owner, 2026-11-21: "autonomously think about whether this prior would improve things over T8, do any measurements or
pilots you'd like, and even start a run on this new prior with the T8-20M as a baseline".

## Why it might help, and why it might not

For: T8 draws structured formulas rarely (polynomials with >= 3 terms 0.37%, >= 3 terms of one form 0.17% of draws);
round 10c draws them at 4.4% and 4.1%, uses fewer functions (8.1% of operators against 12.7%; real formulas 3.7%), and
the precise holdout key keeps simple structured training formulas the deployed key removes (H13). Real benchmark laws
are mostly products of powers; round 10c roots are repeat products in 8.1% of draws against 0.4% for T8 (real: 43%).

Against: (1) the premise is untested (H16): T8-20M may already recover structured laws, and its failures on real
problems come mainly from train/real mismatch in data scales and variable use (front study, LEARNINGS section 1);
(2) less low-level diversity (operators in chains of >= 3 different parts 24% -> 16%) may cost coverage on black-box
data; (3) round 10c's simplified draws are longer (27% exceed 35 tokens; training keeps T8's cap of 35, so the long
ones are resampled and the accepted distribution shifts); (4) one training seed per arm; (5) fewer transcendental
functions: 49% of accepted round-10c draws contain one, against 77% of T8's, which may cost recovery on laws with
nested or several transcendental functions.

Accepted training distribution (2026-11-21, from the 3,000-draw banks, draws of at most 35 tokens that the precise key
keeps; T8's draws are already capped): polynomial with >= 3 terms 5.1% (T8 0.34%), >= 3 terms of one form 4.6%
(0.17%), sum of >= 2 rational terms 2.2% (1.3%), a repeated term 4.8% (0.9%), any transcendental 49% (77%), median 25
tokens (27), median 6 variables (6). The cap does not remove the structure: structured draws are shorter.

## Step 0: premise test (local, inference only, before the run's result)

Problems: 200 drawn from the round-10c training catalog (`catalog_train.yaml` here; seed 20261121), exported like the
front study's training check (`front_study/export_training_problems.py`: 512 support + 512 validation points, noise 0,
unused columns zeroed, no selection). Pool: T8-20M, 512 draws, T = 1, the same runner and settings as
`front_study/probe/t8-20m-training` (100 problems from T8's own catalog, pool recovery 27%).

Read-outs: pool recovery (validation numeric recovery of any draw) overall and by structure (root relation of the
draw's trace is not available from the export, so by class: polynomial, sum of one form, rational sum, and by the
coverage parse: fully explained or not), against the T8-catalog pool; by number of used variables and size.

Reading: T8-20M recovers round-10c problems clearly worse than T8-catalog problems at matched size and variable count
-> the structured part is hard for a T8-trained model, so training on it can help (supports H16's premise). About the
same -> structure is not what T8-20M lacks; the run tests the prior's effect on benchmarks only.

## Step 1: the run

- Name `v26.0-TP10c-20M`. Model, tokenizer, optimizer, schedule (1.5M steps, warm-up 15k, constant 1e-4 to 1.2M, decay
  to 0 at 1.5M), batch 128, dataset blocks (noise mixture, complexity / mask / predict-y / residual blocks) and
  validation size: identical to `flash-ansr/configs/v25.0-T8-20M`.
- Catalog: T8-20M's `catalog_train.yaml` plus `type: term_prior` (symbolic-data `research/term-prior` at `7d616e8`,
  defaults = round 10c), the family urn (concentration 0.36), the six hyperbolic functions at weight 0.5,
  `holdout_family: precise`. Unchanged: T8's 29 holdout pools, length prior (budget x1.3), 35-token cap, literal and
  support priors.
- Confounds, stated: the prior, the holdout key and the function weights change together (a package); T8-20M was
  trained earlier with other package versions (flash-ansr 0.14.0 on Helix is the kit's version); one seed each.
- Compute: Helix, one H200, chained 5-day jobs resuming from checkpoints (every 25k steps), `helix_train_full.sbatch`
  adapted. Throughput: T8's pipeline ran at 4.9 it/s with 16 workers (data-bound); the term prior costs roughly 10-15 ms
  more per expression, so 32 workers are requested. A devel smoke test (gate, 500 steps, it/s) comes first.
- Validation loss is not comparable between the arms: each run validates on a fresh stream from its own catalog.
- Stop rules: a non-finite loss, or a throughput below 2 it/s (more than 9 days), stops the run for a decision.

## Step 2: evaluation (pre-registered)

At 1.5M steps (interim looks at 500k and 1M are descriptive only), both models with the same runner, 512 draws, T = 1,
common random numbers per problem:
1. FastSRB, 120 problems: pool recovery (T8-20M: 57.5%).
2. SRBench black-box, 12 datasets: share of bad draws and pool-best test R2 (front study's measures).
3. T8-catalog problems (the existing 100) and fresh round-10c problems (200, a seed not used in step 0).

Success: FastSRB pool recovery not lower than T8-20M's by more than the paired bootstrap noise (95% interval of the
difference includes 0 or is positive) AND round-10c problems recovered better (interval above 0) AND black-box not
worse (pool-best test R2 median, interval includes 0 or positive). Anything else is reported as measured; a gain on
round-10c problems alone shows only that the model learned its own prior.
