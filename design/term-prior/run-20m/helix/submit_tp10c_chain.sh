#!/usr/bin/env bash
# Submit v26.0-TP10c-20M as a chain of 5-day jobs on one GPU type. Usage: ./submit_tp10c_chain.sh <GPU type> [n jobs]
set -euo pipefail
GPU=${1:?gpu type, e.g. H200}; N=${2:-2}
cd /gpfs/bwfor/work/ws/hd_bc226-srbf/tp10c
prev=""; ids=()
for i in $(seq 1 "$N"); do
  dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
  jid=$(sbatch --parsable --gres=gpu:$GPU:1 -J tp10c-20m-$i "${dep[@]}" tp10c_train.sbatch)
  ids+=("$jid"); prev=$jid
done
echo "submitted chain on $GPU: ${ids[*]}"
