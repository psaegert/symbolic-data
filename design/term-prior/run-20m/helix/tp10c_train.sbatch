#!/usr/bin/env bash
# v26.0-TP10c-20M on one Helix GPU: T8-20M's recipe with the round-10c term prior (symbolic-data research/term-prior
# @ 7d616e8, on PYTHONPATH ahead of the kit's installed 0.18.0; the shared venv is not modified). 1.5M steps across
# chained 5-day jobs; every job resumes from the newest checkpoint and exits at once when the run is complete.
# Pilot: PILOT_STEPS=<n> sbatch ... runs n steps into runs/pilot-<name>-w<cpus> (no validation, wandb offline).
#SBATCH -p gpu-single
#SBATCH --qos=pqos-gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH -o logs/%x_%j.out
set -uo pipefail
module load devel/python/3.13.1
W=/gpfs/bwfor/work/ws/hd_bc226-srbf; T=$W/tp10c
source $W/venv_train/bin/activate
export PYTHONPATH=$T/symbolic-data/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $W/flash-ansr-src
NAME=v26.0-TP10c-20M; CONFIG=$T/configs/$NAME/train.yaml; OUT=$T/runs/$NAME
CI=${CHECKPOINT_INTERVAL:-25000}; VI=${VALIDATE_INTERVAL:-5000}; WANDB_MODE_ARG=${WANDB_MODE_ARG:-online}
if [ -n "${PILOT_STEPS:-}" ]; then
  P=$T/configs/pilot-$NAME; rm -rf "$P"; cp -r $T/configs/$NAME "$P"; sed -i -E "s/^steps:.*/steps: $PILOT_STEPS/" "$P/train.yaml"
  CONFIG=$P/train.yaml; OUT=$T/runs/pilot-$NAME-w$SLURM_CPUS_PER_TASK; rm -rf "$OUT"; VI=100000000; CI=100000000; WANDB_MODE_ARG=offline
fi
mkdir -p "$OUT"
STEPS=$(grep -E "^steps:" "$CONFIG" | tr -d "steps:_ ")
echo "=== $NAME job $SLURM_JOB_ID on $(hostname) $(date -Is): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader) | cpus $SLURM_CPUS_PER_TASK | steps $STEPS | ci $CI | vi $VI | wandb $WANDB_MODE_ARG ===" | tee -a "$OUT/status"
if [ -e "$OUT/DONE" ]; then echo "run already complete ($OUT/DONE); nothing to do" | tee -a "$OUT/status"; exit 0; fi
latest=$(ls -d "$OUT"/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | grep -E '^[0-9]+$' | sort -n | tail -1)
RESUME=(); if [ -n "$latest" ]; then RESUME=(--resume-from "$OUT/checkpoint_$latest"); echo "resuming from checkpoint_$latest" | tee -a "$OUT/status"; else
  # Gate on a fresh start: the branch's symbolic_data is the one imported, the catalog is the term prior under the
  # precise key, and it streams skeletons; prints the stamp and the per-skeleton cost with the holdout check.
  python - "$T" "$CONFIG" <<'EOF' 2>&1 | grep -v Warning | tee -a "$OUT/status" || { echo "GATE RED - refusing to train" | tee -a "$OUT/status"; exit 1; }
import sys, os, time, numpy as np, flash_ansr, simplipy, symbolic_data, torch
from flash_ansr.utils.config_io import load_config
from symbolic_data import build_catalog
from symbolic_data.term_prior import TermPriorCatalog
T, config = sys.argv[1], sys.argv[2]
assert symbolic_data.__file__.startswith(T + "/symbolic-data/src"), symbolic_data.__file__
sha = open(T + "/symbolic-data.COMMIT").read().strip()   # written on the login node (git is not executable on compute nodes)
print(f"stamp: flash-ansr {flash_ansr.__version__} | simplipy {simplipy.__version__} | symbolic-data {symbolic_data.__version__} @ {sha} (branch) | torch {torch.__version__}")
cat_path = os.path.join(os.path.dirname(config), "catalog_train.yaml")
t0 = time.time(); cat = build_catalog(load_config(cat_path)); t1 = time.time()
assert isinstance(cat, TermPriorCatalog) and cat.holdout_family == "precise", (type(cat), cat.holdout_family)
rng = np.random.default_rng(0); n = 0; t2 = time.time()
for _ in range(300):
    for _try in range(100):
        try:
            sk = cat.sample_skeleton(new=False, decontaminate=True, rng=rng); n += 1; break
        except Exception as e:
            if type(e).__name__ != "NoValidSampleFoundError": raise
dt = time.time() - t2
print(f"gate ok: TermPriorCatalog, holdout_family precise | build {t1 - t0:.0f} s | {n} skeletons, {1000 * dt / n:.1f} ms each | e.g. {' '.join(sk[0])}")
EOF
fi
( while sleep 1800; do for d in $(ls -d "$OUT"/checkpoint_* 2>/dev/null | sed 's/.*checkpoint_//' | grep -E '^[0-9]+$' | sort -n | head -n -2); do [ $((d % 100000)) -ne 0 ] && rm -rf "$OUT/checkpoint_$d"; done; done ) &
PRUNER=$!
export WANDB_RUN_ID=${NAME}-helix WANDB_RESUME=allow
[ -n "${PILOT_STEPS:-}" ] && export WANDB_RUN_ID=pilot-${NAME}-$SLURM_JOB_ID
python -m flash_ansr train -c "$CONFIG" -v -o "$OUT" -ci "$CI" -vi "$VI" -w "$SLURM_CPUS_PER_TASK" --mode "$WANDB_MODE_ARG" --name "$NAME" "${RESUME[@]}" >> "$OUT/train.log" 2>&1
RC=$?; kill $PRUNER 2>/dev/null
echo "TRAIN_RC=$RC ($(date -Is)) job $SLURM_JOB_ID" | tee -a "$OUT/status"
[ "$RC" -eq 0 ] && [ -z "${PILOT_STEPS:-}" ] && touch "$OUT/DONE"
exit $RC
