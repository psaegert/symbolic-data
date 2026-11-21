#!/usr/bin/env bash
cd /home/psaegert/Projects/flash-ansr-rl/runs/term_prior_20m/premise
export PYTHONPATH=/home/psaegert/Projects/symbolic-data/.claude/worktrees/term-prior/src
PY=/home/psaegert/Projects/flash-ansr/.venv/bin/python
$PY export_tp_problems.py data 200 20261121 random tp10c > logs/export_random.log 2>&1; echo "random rc $?" >> logs/export.done
$PY export_tp_problems.py data 60 20261122 structured tp10cstruct > logs/export_struct.log 2>&1; echo "struct rc $?" >> logs/export.done
