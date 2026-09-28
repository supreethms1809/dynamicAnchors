#!/usr/bin/env bash
# Rollouts for the positioning analysis, seeds 44-46, run on spark (the RL policies live there).
#   pool    rollouts started at D_train rows (the local-global continuum)
#   robust  rollouts at x* and a nearby x' (instance robustness); needs the
#           containment_eval output for the same seeds in $CFIX
# Seeds 42-43 were run on the Mac. Copy $OUT/pool and $OUT/robust back to
# ../results/local_global/ there, then run
#   python -m revision.local_global_continuum && python -m revision.local_global_continuum --report
#   python -m revision.weakness_battery --report
#
#   bash revision/run_positioning_spark.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python}
SEEDS=${SEEDS:-"44 45 46"}
INST=${INST:-runs/paper_final/emp_tc0p10/results}
CFIX=${CFIX:-runs/paper_final/containment_fix}
OUT=${OUT:-runs/paper_final/local_global}
mkdir -p "$OUT/logs"

for d in "$INST/ddpg" "$INST/maddpg"; do
    [ -d "$d" ] || { echo "missing $d (set INST=...)" >&2; exit 1; }
done
n_cfix=$(ls "$CFIX"/*__seed4[456].json 2>/dev/null | wc -l)
[ "$n_cfix" -ge 72 ] || { echo "expected 72 containment_eval cells in $CFIX, found $n_cfix (set CFIX=...)" >&2; exit 1; }

# Four workers; each skips cells already written, so re-running resumes.
for arm in rlda mada; do
    $PY -m revision.rl_rollout_experiments --job pool --n_start 300 --arms $arm --seeds $SEEDS \
        --inst_dir "$INST" --out "$OUT" > "$OUT/logs/pool_${arm}.log" 2>&1 &
    $PY -m revision.rl_rollout_experiments --job robust --max_points 200 --arms $arm --seeds $SEEDS \
        --inst_dir "$INST" --cfix_dir "$CFIX" --out "$OUT" > "$OUT/logs/robust_${arm}.log" 2>&1 &
done
wait

if grep -h "^missing\|Traceback" "$OUT"/logs/*.log; then
    echo "some rollouts failed; see $OUT/logs" >&2
    exit 1
fi
echo "pool cells: $(ls "$OUT"/pool/*__seed4[456].json | wc -l) (expect 72)   robust cells: $(ls "$OUT"/robust/*__seed4[456].json | wc -l) (expect 72)"
