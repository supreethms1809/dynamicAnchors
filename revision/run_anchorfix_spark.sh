#!/usr/bin/env bash
# Spark part of the 2026-09-28 fixes, for seeds 44-46 (their classifiers, RL policies and
# rules files live on spark). Nothing is retrained. Run from a checkout of origin/main:
#
#   cd /home/sureshm/dynAnc_instance && git fetch origin && git checkout --detach origin/main
#   bash revision/run_anchorfix_spark.sh <python-of-the-training-env>
#
#   1. SP-/greedy-Anchors baselines from exact bins, empty anchors dropped: main grid
#      (k=1, tau_C 0.10 and 0.20), k-sweep and pool-20, on spark's own classifiers.
#   2. Per-policy selection with the policy floor (0.60): re-score every seed 44-46 RLDA
#      and MADA cell (main grids, k-sweep, ablations), and the same without the floor.
#   3. Instance level, emp and pert grids: containment_eval with exact Anchors bins,
#      cond_fid_all, stored boxes and the policy-OR explanation (pi_or).
#   Every rule and explanation also gets Len under the new criterion (a condition
#   excludes at least one D_train row) and its rule re-printed with exactly those conditions.
# Everything goes under $OUT (default /home/sureshm/dynamicAnchors/runs/paper_final_anchorfix).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:?usage: bash revision/run_anchorfix_spark.sh <python>}
MAIN=${MAIN:-/home/sureshm/dynamicAnchors}
PF=$MAIN/runs/paper_final
OUT=${OUT:-$MAIN/runs/paper_final_anchorfix}
mkdir -p "$OUT"

git merge-base --is-ancestor 327d4de HEAD || { echo "check out origin/main first (needs 327d4de)" >&2; exit 1; }
echo "code: $(git log --oneline -1)"
$PY -m pytest -q tests/test_anchor_predicate_parse.py tests/test_mada_per_policy_selection.py tests/test_rule_length.py

# 1. Anchors-family baselines, seeds 44-46, on this machine's classifiers
$PY revision/rerun_anchor_family_exactbins.py --seeds 44 45 46 --on_spark --runs "$MAIN/runs" \
    --out "$OUT" --python "$PY" --concurrency 8 --apply > "$OUT/anchors_s44-46.log" 2>&1
tail -1 "$OUT/anchors_s44-46.log"                 # expect: done in ...s; 0 failed: []

# 2. per-policy re-score. The pooled, no-floor check must end "0 failed or differed".
$PY -m revision.rescore_per_policy --check 10 --apply --out "$OUT/paper_final_perpolicy"
$PY -m revision.rescore_per_policy --apply --par 8 --out "$OUT/paper_final_perpolicy" \
    > "$OUT/perpolicy_floor_spark.log" 2>&1
tail -1 "$OUT/perpolicy_floor_spark.log"          # expect: done; 0 failed or differed
$PY -m revision.rescore_per_policy --apply --par 8 --policy_floor none \
    --out "$OUT/paper_final_perpolicy_nofloor" > "$OUT/perpolicy_nofloor_spark.log" 2>&1
tail -1 "$OUT/perpolicy_nofloor_spark.log"        # expect: done; 0 failed or differed

# 3. instance level, both grids; containment_eval reads <grid>/conf itself
for g in emp pert; do
    G=$PF/${g}_tc0p10
    O=$OUT/containment_fix_exact_$g
    mkdir -p "$O/logs"
    grep -H precision_estimator: "$G/conf/anchor.yaml" "$G/conf/anchor_single.yaml"
    for arm in rlda mada; do
        $PY -m revision.containment_eval --arms $arm --seeds 44 45 46 --inst_dir "$G/results" \
            --out "$O" > "$O/logs/${arm}_s44-46.log" 2>&1 &
    done
done
wait
if grep -h "^missing\|Traceback" "$OUT"/containment_fix_exact_*/logs/*_s44-46.log; then
    echo "containment_eval had failures" >&2; exit 1
fi
for g in emp pert; do
    echo "$g: $(grep -oh 'precision_estimator=[a-z]*' "$OUT/containment_fix_exact_$g"/logs/*_s44-46.log | sort | uniq -c | tr '\n' ' ')"
done   # expect emp: empirical only, pert: conditional only

echo "anchor-family cells: $(find "$OUT"/baselines_emp "$OUT"/k_sweep "$OUT"/pool20 -name '*_anchors__seed4[456]__*.json' | wc -l) (expect 936)"
echo "per-policy cells: $(find "$OUT/paper_final_perpolicy" -name '*__seed4[456]__*.json' | wc -l) floor, $(find "$OUT/paper_final_perpolicy_nofloor" -name '*__seed4[456]__*.json' | wc -l) no floor (expect 947 each)"
echo "instance cells: $(ls "$OUT"/containment_fix_exact_*/*__seed4[456].json | wc -l) (expect 144)"
# Len (a condition excludes at least one D_train row) and the re-printed rules are
# written by the same code; every new cell must carry them.
echo "rule-set cells without Len: $( { find "$OUT"/baselines_emp "$OUT"/k_sweep "$OUT"/pool20 -name '*_anchors__seed4[456]__*.json'; \
    find "$OUT"/paper_final_perpolicy "$OUT"/paper_final_perpolicy_nofloor -name '*__seed4[456]__*.json'; } \
    | xargs grep -L '"len_criterion"' | wc -l) (expect 0)"
echo "instance cells without n_cond: $(grep -L '"n_cond"' "$OUT"/containment_fix_exact_*/*__seed4[456].json | wc -l) (expect 0)"
