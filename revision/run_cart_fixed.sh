#!/bin/bash
# Re-run the CART baseline with the fixed run_cart (true partition, tree size on
# D_val) for the paper_final grid, kept apart from the paper numbers:
#   ../results/paper_final_cart_fixed/<criterion>/baselines_{emp,pert}/
# Same flags as run_seed_major_grid.sh's baselines; classifiers as in
# revision/abstention_paper_final.py (42-43 local lock, 44-46 from spark).
# Skips cells that already exist. Usage: bash revision/run_cart_fixed.sh [jobs]
cd /Users/ssuresh/dynAnc_codeCleanup/dynamicAnchors || exit 1
PY=/opt/anaconda3/envs/marl/bin/python
OUT=../results/paper_final_cart_fixed
JOBS=${1:-6}
DS="iris synthetic wine sick breast_cancer mammography housing heloc uci_credit uci_adult folktables_income_CA_2018 wyodot_kvdw_labeled"
mkdir -p "$OUT/logs"

jobs_list() {
  for crit in precision_constrained effectiveness; do
    for est in emp pert; do
      for seed in 42 43 44 45 46; do
        for ds in $DS; do
          for tau in 0.10 0.20; do
            echo "$crit $est $seed $ds $tau"
          done
        done
      done
    done
  done
}

run_one() {
  crit=$1 est=$2 seed=$3 ds=$4 tau=$5
  if [[ $seed == 42 || $seed == 43 ]]; then clf=runs/paper_final/emp_tc0p10/classifiers/${ds}_seed${seed}.pth
  else clf=../results/paper_final_updated/emp_tc0p10/classifiers/${ds}_seed${seed}.pth; fi
  fid=empirical; [[ $est == pert ]] && fid=perturbed
  dir=$OUT/$crit/baselines_$est
  out=$dir/${ds}__cart__seed${seed}__tp0p90__tc0p${tau#0.}.json
  [[ -s $out ]] && { echo "skip $crit $est $ds $seed $tau"; return 0; }
  mkdir -p "$dir"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $PY -W ignore -m revision.baselines \
    --dataset "$ds" --seed "$seed" --k 1 --tau_p 0.90 --tau_c "$tau" \
    --coverage_basis predicted --fid_estimator $fid --methods cart \
    --cart_size_criterion "$crit" --classifier_path "$clf" --out_dir "$dir" \
    >> "$OUT/logs/${crit}_${est}_seed${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] rc=$? $crit $est $ds $seed $tau"
}
export -f run_one; export PY OUT
jobs_list | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
echo ALL DONE
