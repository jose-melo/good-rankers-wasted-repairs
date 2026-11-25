#!/usr/bin/env bash
# Every run behind the paper's results, then the scoring. Usage: scripts/run_experiments.sh [n_parallel]
# About 30 CPU-hours in total (single-threaded jobs), plus 21 L-Town simulations of a few minutes each.
# Outputs: $GRWR_OUT (default ./outputs): runs/, pool/, cells/<testbed>.csv, and results/<testbed>.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
NP=${1:-4}
OUT="${GRWR_OUT:-$PWD/outputs}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore FSLNET_THREADS=1
PY=${PYTHON:-python}                       # environment of requirements.txt
LTOWN_PY=${LTOWN_PYTHON:-$PY}              # environment of requirements-ltown.txt

$PY scripts/prepare_cats.py
$LTOWN_PY scripts/simulate_ltown.py --published
seq 3000 3019 | xargs -P "$NP" -I{} $LTOWN_PY scripts/simulate_ltown.py --seed {}

run_jobs() {  # $1 job file, $2 output directory
  awk -v out="$2" '{printf "--testbed %s --dataset %s --seed %s --out %s", $1, $2, $3, out;
                    for (i = 4; i <= NF; i++) printf " %s", $i; print ""}' "$1" |
    xargs -P "$NP" -L 1 $PY -m wasted_repairs.run
}
for tb in graphs cats light_tunnel ltown; do run_jobs "jobs/pool_$tb.txt" "$OUT/pool"; done
for tb in graphs cats light_tunnel ltown wind_tunnel; do run_jobs "jobs/eval_$tb.txt" "$OUT/runs"; done

for tb in graphs cats light_tunnel ltown wind_tunnel; do
  pool=(); [ -d "$OUT/pool/$tb" ] && pool=(--pool "$OUT/pool/$tb")
  $PY -m wasted_repairs.score --runs "$OUT/runs/$tb" ${pool[@]+"${pool[@]}"} --out "$OUT/cells/$tb.csv"
done
$PY scripts/collect_results.py "$OUT/cells"
