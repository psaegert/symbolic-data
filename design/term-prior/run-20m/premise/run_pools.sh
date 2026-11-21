#!/usr/bin/env bash
# Premise pools (PLAN.md step 0): T8-20M, 512 draws, T = 1, the front study's runner and settings. 2026-11-21.
set -u
R=/home/psaegert/Projects/flash-ansr-rl/runs/pareto_diag/temp_probe; D=/home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/premise
M=/home/psaegert/models/flash-ansr-v25.0-T8-20M
for b in tp10c tp10cstruct; do
  systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 --unit="tp-pool-$b" \
    nice -n 5 $R/run_env.sh $R/run_probe_temp.py --name t8-20m-$b --model-dir $M --laws $D/data/${b}_laws.csv \
      --problems $D/data/${b}_problems.pkl --out-root $D --workers 4 > $D/logs/pool_$b.log 2>&1
  echo "$b exit $? $(date -Is)" >> $D/logs/pools.done
done
