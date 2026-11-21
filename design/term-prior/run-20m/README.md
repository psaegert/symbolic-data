# v26.0-TP10c-20M: the round-10c term prior against T8-20M (started 2026-11-21)

Copy of the run record `~/Projects/flash-ansr-rl/runs/term_prior_20m/` on valkyrie-light (that tree is gitignored;
bulk pools and problem pickles stay there). Here: the plan, the configs, the Helix scripts, the premise test's scripts,
per-problem table and logs, `yaml_equiv.py` (YAML vs prototype draws) and `linear_share.py` / `linear_share.log`
(the share of weighted sums of single variables among the structure classes, run from `prior_structure/explorer/`).

Plan, fixed before any result: [PLAN.md](PLAN.md). Prior design: symbolic-data `research/term-prior`
(`design/term-prior/README.md`); research pages `research:projects/symbolic-data/results/2026-11-21-term-prior-rounds-8-to-10.md`
and `2026-11-20-precise-holdout-key.md`.

## Layout

- `configs/v26.0-TP10c-20M/`: T8-20M's six config files; only `catalog_train.yaml` differs (`type: term_prior`,
  `term_prior: {}`, `holdout_family: precise`, the family urn, hyperbolic weights 0.5). Checked 2026-11-21: the YAML
  draws are identical to the reviewed prototype's round 10c, 500 of 500 (scratch `yaml_equiv.py`, holdout pools
  emptied since draws do not depend on them).
- `helix/tp10c_train.sbatch`: the training job (start-up gate, resume from the newest checkpoint, checkpoint pruner
  keeping every 100k plus the two newest); `helix/submit_tp10c_chain.sh`: chained 5-day jobs.
- `premise/`: step 0 (T8-20M on round-10c problems): `export_tp_problems.py`, `run_pools.sh`, `premise_analyze.py`.

## Helix

Workspace `/gpfs/bwfor/work/ws/hd_bc226-srbf/tp10c/` (expires about 2026-12-13; extend before then):
`symbolic-data/` (clone of `research/term-prior` at `7d616e8`, put first on `PYTHONPATH`; the shared `venv_train` is
not modified), `configs/`, `tp10c_train.sbatch`, `runs/v26.0-TP10c-20M/` (status, train.log, checkpoints), `logs/`.
Software: flash-ansr 0.14.0 (kit wheel f6d918e, the same commit as T8-20M), simplipy 0.14.7 (T8-20M: 0.14.6),
torch 2.14.0, symbolic-data branch @ 7d616e8. wandb run id `v26.0-TP10c-20M-helix`.

## Log

- 2026-11-21 02:14 devel pilot 16003894: gate red, `git` is not executable on compute nodes (the stamp now reads
  `symbolic-data.COMMIT`, written on the login node).
- 2026-11-21 02:25 devel pilot 16003920 (A40, 32 CPUs, 1,500 steps): gate ok (catalog build 171 s, 15.5 ms per
  skeleton with the holdout check), TRAIN_RC 0, no NaN. Steady rate 3.1 it/s, GPU 97% busy, workers about 46% of
  32 CPUs: GPU-bound on the A40, so data supply is not the limit there.
- 2026-11-21 02:45 chain submitted on H200: jobs 16004034, 16004035 (afterany).
- 2026-11-21 04:30 premise test done (PLAN.md step 0; `premise/logs/premise_analyze.log`, `premise_more.log`,
  `premise_counts.log`). T8-20M pool recovery: T8-catalog problems 27.0% [19.3, 36.4] (n 100), random round-10c
  problems 41.6% [35.0, 48.6] (n 197; stratified by transcendental x used variables +9.2 points [-2.7, +21.1], the
  gap sits in non-transcendental strata). Of the round-10c polynomials with >= 3 terms, 41 of 48 in the first two
  sets are weighted sums of single variables (recovered 73%), so a third set of 60 non-linear structured problems
  was added. Against unstructured round-10c controls in the same strata (used variables x size x transcendental):
  non-linear structured 34.7% vs 43.2%, -8.6 [-21.9, +4.6] (n 98); rational sums 31.9% vs 38.9%, -7.0 [-24.2, +9.2]
  (n 47); linear sums 73.2% vs 54.6%, +18.5 [-2.0, +39.7] (n 41). Reading (PLAN.md): no detectable deficit on
  structured problems, so the run tests the prior's other effects on benchmarks; a deficit of up to about 20 points
  on non-linear structure is not excluded. The run was kept, as the owner asked for it.
- 2026-11-21 bank check (`linear_share.log`): 96 of round 10c's 131 polynomials with >= 3 terms are linear sums;
  non-linear ones 0.33% (T8) -> 1.17% (round 10c), real formulas 1.12%.
