#!/bin/bash
# Resumable continuation of the final run.
#
# The first driver was killed silently mid-run (its wine child survived, reparented
# to PID 1). This version is (a) resumable — any dataset that already has
# consolidated_metrics_all_methods.json is skipped, so re-running after a kill costs
# nothing — and (b) able to adopt an in-flight pipeline via WAIT_PID instead of
# starting a duplicate of it.
#
# Launch detached with daemonize.py so it becomes its own session leader; macOS has
# no setsid(1), and a plain `nohup ... &` stays in the caller's process group, which
# is what let the first driver be killed.

set -u
cd /Users/ssuresh/dynamicAnchors || exit 1

PY=/opt/anaconda3/envs/marl/bin/python
OUT=/Users/ssuresh/dynamicAnchors/comparison_results/final_p90_c20
DRIVER_LOG="$OUT/final_driver.log"

# Adopt an already-running pipeline (passed as WAIT_PID) rather than duplicating it.
if [ -n "${WAIT_PID:-}" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "=== resuming: waiting on in-flight pipeline pid $WAIT_PID ($(date)) ===" | tee -a "$DRIVER_LOG"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "=== in-flight pipeline pid $WAIT_PID exited $(date) ===" | tee -a "$DRIVER_LOG"
fi

RUNS=(
  "iris:90000:360000:20"
  "wine:120000:360000:20"
  "breast_cancer:90000:360000:20"
  "uci_credit:360000:720000:25"
  "uci_default-credit-card-clients:360000:720000:25"
  "uci_adult:360000:720000:25"
  "folktables_income_CA_2018:720000:1080000:25"
)

for spec in "${RUNS[@]}"; do
  IFS=: read -r ds sa ma ninst <<< "$spec"

  if [ -f "$OUT/$ds/consolidated_metrics_all_methods.json" ]; then
    echo "=== [$ds] SKIP (already complete) ===" | tee -a "$DRIVER_LOG"
    continue
  fi

  echo "" | tee -a "$DRIVER_LOG"
  echo "=== [$ds] START $(date) | SA=$sa MA=$ma n_inst=$ninst ===" | tee -a "$DRIVER_LOG"

  "$PY" run_comparison_pipeline.py \
    --dataset "$ds" \
    --algorithm maddpg \
    --seed 42 \
    --device cpu \
    --total_timesteps "$sa" \
    --max_n_frames "$ma" \
    --steps_per_episode 200 \
    --n_instances_per_class "$ninst" \
    --parallel_classes 5 \
    --force_retrain \
    --output_dir "$OUT/$ds" \
    > "$OUT/${ds}_pipeline.log" 2>&1

  rc=$?
  echo "=== [$ds] DONE rc=$rc $(date) ===" | tee -a "$DRIVER_LOG"
done

echo "" | tee -a "$DRIVER_LOG"
echo "=== FINAL RUN finished $(date) ===" | tee -a "$DRIVER_LOG"
