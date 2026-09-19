#!/bin/bash
# Seed-major 2x2 headline into runs/paper_final/. C_train = C_eval = predicted.
#
# One seed of EVERYTHING before the next seed:
#   1. pert τ_C=0.2   2. emp τ_C=0.2   3. emp τ_C=0.1   4. pert τ_C=0.1
#   5. baselines (CART/RandS emp+pert, SP-Anch, Greedy)
#   6. P3–P5 ablations vs emp τ_C=0.1 (11 datasets, no wyodot)
#
# Does not reuse train_fhat_* or train_pert_fid_covtarget02.
# Old paper_final is runs/archived_ctrain_y_tc0p10_20260918/.
#
#   bash revision/run_seed_major_grid.sh --go
#   bash revision/run_seed_major_grid.sh --go 42
cd /Users/ssuresh/dynAnc_codeCleanup/dynamicAnchors/.claude/worktrees/perturb-fid || exit 1
PY=/opt/anaconda3/envs/marl/bin/python
L=revision/run_perturb_fid_seed42.py
MAIN_RUNS="${PAPER_RUNS_DIR:-/Users/ssuresh/dynAnc_codeCleanup/dynamicAnchors/runs}"
PF=runs/paper_final
LOG=$PF/seed_major.out
mkdir -p "$PF"
stamp() { date "+%m-%d %H:%M:%S"; }
say() { echo "[$(stamp)] $*"; }

if [[ "${1:-}" != "--go" ]]; then
  echo "refusing to start: pass --go"
  echo "writes under $PF ; archive is runs/archived_ctrain_y_tc0p10_20260918/"
  exit 2
fi
shift
if [[ $# -eq 0 ]]; then
  SEEDS=(42 43 44 45 46)
else
  SEEDS=("$@")
fi
DS_ABL="iris synthetic wine sick breast_cancer uci_credit mammography housing heloc uci_adult folktables_income_CA_2018"
DS_ALL="$DS_ABL wyodot_kvdw_labeled"
NOCROSS="--override shared_terminal_bonus=0.0 --override inter_class_overlap_weight=0.0 --override shared_reward_weight=0.0"

cell_root() {
  local est=$1 tau=$2
  case "${est}_${tau}" in
    empirical_0.1)   echo "$PF/emp_tc0p10" ;;
    empirical_0.2)   echo "$PF/emp_tc0p20" ;;
    conditional_0.2) echo "$PF/pert_tc0p20" ;;
    conditional_0.1) echo "$PF/pert_tc0p10" ;;
    *) say "bad cell ${est} ${tau}"; exit 1 ;;
  esac
}

run_cell() {
  local est=$1 tau=$2 seed=$3
  local root; root=$(cell_root "$est" "$tau")
  say "CELL start seed=$seed estimator=$est tau_c=$tau root=$root"
  $PY $L --root "$root" --seed "$seed" --estimator "$est" \
      --coverage_basis predicted \
      --override "coverage_target=$tau" --tau_c "$tau" \
      >> "$LOG" 2>&1
  say "CELL done  seed=$seed estimator=$est tau_c=$tau rc=$?"
}

clf_path() {
  local ds=$1 seed=$2
  if [[ "$ds" == wyodot* ]]; then
    echo "$MAIN_RUNS/wyodot_fiveseed_overlap075/dnn/classifiers/${ds}_seed${seed}.pth"
  else
    echo "$MAIN_RUNS/paper_fiveseed_overlap075/classifiers/${ds}_seed${seed}.pth"
  fi
}

run_baselines() {
  local seed=$1 ds clf out
  say "BASELINES start seed=$seed"
  for ds in $DS_ALL; do
    clf=$(clf_path "$ds" "$seed")
    if [[ ! -f "$clf" ]]; then
      say "BASELINES missing classifier $clf (skip $ds)"
      continue
    fi
    out=$PF/baselines_emp
    mkdir -p "$out"
    $PY -m revision.baselines --dataset "$ds" --seed "$seed" --k 1 \
        --tau_p 0.90 --tau_c 0.10 --coverage_basis predicted \
        --fid_estimator empirical \
        --methods cart random_search sp_anchors greedy_anchors \
        --budget_per_class 5 --n_candidates 256 \
        --classifier_path "$clf" --out_dir "$out" \
        >> "$PF/baselines_emp_seed${seed}.log" 2>&1
    out=$PF/baselines_pert
    mkdir -p "$out"
    $PY -m revision.baselines --dataset "$ds" --seed "$seed" --k 1 \
        --tau_p 0.90 --tau_c 0.10 --coverage_basis predicted \
        --fid_estimator perturbed \
        --methods cart random_search \
        --budget_per_class 5 --n_candidates 256 \
        --classifier_path "$clf" --out_dir "$out" \
        >> "$PF/baselines_pert_seed${seed}.log" 2>&1
    say "BASELINES done $ds seed=$seed"
  done
  say "BASELINES complete seed=$seed"
}

run_ablation() {
  local fam=$1 arm=$2 seed=$3; shift 3
  say "ABLATION start $fam/$arm seed=$seed"
  $PY $L --root "$PF/ablations/$fam/$arm" --seed "$seed" \
      --estimator empirical --coverage_basis predicted \
      --override coverage_target=0.1 --tau_c 0.1 \
      --datasets $DS_ABL --arms mada "$@" \
      >> "$LOG" 2>&1
  say "ABLATION done  $fam/$arm seed=$seed rc=$?"
}

run_ablations_seed() {
  local seed=$1
  say "ABLATIONS start seed=$seed (P3–P5, emp τ_C=0.1, predicted)"
  run_ablation coord no_same_class  "$seed" --override agents_per_class=1 --mada_frames_mult 3
  run_ablation coord no_cross_class "$seed" $NOCROSS
  run_ablation coord no_coord       "$seed" $NOCROSS --override same_class_diversity_weight=0.0 \
      --override agents_per_class=1 --mada_frames_mult 3
  run_ablation reward no_width        "$seed" --override gamma=0.0
  run_ablation reward no_drift        "$seed" --override drift_penalty_weight=0.0
  run_ablation reward no_anchor_drift "$seed" --override anchor_drift_penalty_weight=0.0
  run_ablation reward no_local        "$seed" --override gamma=0.0 \
      --override drift_penalty_weight=0.0 --override anchor_drift_penalty_weight=0.0
  $PY $L --root "$PF/ablations/algo/sac_masac" --seed "$seed" \
      --estimator empirical --coverage_basis predicted \
      --override coverage_target=0.1 --tau_c 0.1 \
      --datasets $DS_ABL --arms rlda mada --mada_algo masac --rlda_algo sac \
      >> "$LOG" 2>&1
  say "ABLATION done  algo/sac_masac seed=$seed rc=$?"
  $PY $L --root "$PF/ablations/classifier/random_forest" --seed "$seed" \
      --estimator empirical --coverage_basis predicted \
      --override coverage_target=0.1 --tau_c 0.1 \
      --datasets $DS_ABL --arms rlda mada --classifier_type random_forest --with_baselines \
      >> "$LOG" 2>&1
  say "ABLATION done  classifier/random_forest seed=$seed rc=$?"
  run_ablation overlap w050 "$seed" --override inter_class_overlap_weight=0.5
  run_ablation overlap w100 "$seed" --override inter_class_overlap_weight=1.0
  say "ABLATIONS complete seed=$seed"
}

run_seed() {
  local seed=$1
  say "======== SEED $seed START (2x2 + baselines + ablations) ========"
  run_cell conditional 0.2 "$seed"
  run_cell empirical    0.2 "$seed"
  run_cell empirical    0.1 "$seed"
  run_cell conditional  0.1 "$seed"
  run_baselines "$seed"
  run_ablations_seed "$seed"
  say "======== SEED $seed COMPLETE ========"
}

say "SEED-MAJOR GRID START seeds=${SEEDS[*]}  C_train=predicted  root=$PF"
for s in "${SEEDS[@]}"; do
  run_seed "$s"
done
say "SEED-MAJOR GRID COMPLETE"
