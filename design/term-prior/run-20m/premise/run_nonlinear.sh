#!/usr/bin/env bash
cd /home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/premise
PYTHONPATH=/home/psaegert/Projects/symbolic-data/.claude/worktrees/term-prior/src /home/psaegert/Projects/flash-ansr/.venv/bin/python \
  export_tp_problems.py data 60 20261123 nonlinear tp10cnonlin > logs/export_nonlin.log 2>&1 || { echo "export failed" >> logs/nonlin.done; exit 1; }
R=/home/psaegert/Projects/flash-ansr-rl/runs/pareto_diag/temp_probe; D=$PWD
$R/run_env.sh $R/run_probe_temp.py --name t8-20m-tp10cnonlin --model-dir /home/psaegert/models/flash-ansr-v25.0-T8-20M \
  --laws $D/data/tp10cnonlin_laws.csv --problems $D/data/tp10cnonlin_problems.pkl --out-root $D --workers 4 > logs/pool_tp10cnonlin.log 2>&1
echo "pool exit $? $(date -Is)" >> logs/nonlin.done
