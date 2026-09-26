#!/usr/bin/env bash
# Instance-level evaluation for seeds 44-46, run on spark (the RL policies live there).
#   1. containment_eval: RL boxes stored (π) and with the containment fix (π+), plus
#      Anchors on the same points, all scored with Anchors' D(z|A).
#   2. tree_instance_eval: the plain global surrogate's leaf for the same points.
# Seeds 42-43 were run on the Mac. Copy $OUT back to ../results/containment_fix/ there,
# then run revision.containment_report and revision.instance_surrogate_report.
#
#   bash revision/run_instance_spark.sh [python]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python}
SEEDS=${SEEDS:-"44 45 46"}
INST=${INST:-runs/paper_final/emp_tc0p10/results}
OUT=${OUT:-runs/paper_final/containment_fix}
mkdir -p "$OUT/logs"

for d in "$INST/ddpg" "$INST/maddpg"; do
    [ -d "$d" ] || { echo "missing $d (set INST=...)" >&2; exit 1; }
done

# RLDA and MADA in parallel; each skips cells already written.
$PY -m revision.containment_eval --arms rlda --seeds $SEEDS --inst_dir "$INST" --out "$OUT" \
    > "$OUT/logs/rlda_s44-46.log" 2>&1 &
P1=$!
$PY -m revision.containment_eval --arms mada --seeds $SEEDS --inst_dir "$INST" --out "$OUT" \
    > "$OUT/logs/mada_s44-46.log" 2>&1 &
P2=$!
wait $P1 $P2

grep -h "^missing\|Traceback" "$OUT"/logs/*_s44-46.log && { echo "containment_eval had failures" >&2; exit 1; } || true

$PY -m revision.tree_instance_eval --seeds $SEEDS --inst_dir "$INST" --cfix_dir "$OUT" \
    > "$OUT/logs/tree_s44-46.log" 2>&1

n_rl=$(ls "$OUT"/*__seed4[456].json 2>/dev/null | wc -l)
n_tree=$(ls "$OUT"/tree/*__seed4[456].json 2>/dev/null | wc -l)
echo "RL/Anchors cells: $n_rl (expect 72)   tree cells: $n_tree (expect 36)"
