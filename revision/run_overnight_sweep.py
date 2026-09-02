#!/usr/bin/env python3
"""Overnight sweep: 4 arms x N datasets x 5 seeds, seed-major.

Ordering is deliberate and matches the request: for each seed, run ALL datasets
for RLDA(dnn), then MADA(dnn), then RLDA(rf), then MADA(rf), before starting the
next seed. So whatever has finished by morning is a COHERENT PREFIX -- a full
cross-arm comparison at seed 42 -- rather than a ragged mix.

Budgets are the ones used all of 2026-09-01 (MADA 144k frames/agent, RLDA 270k
total steps) rather than the DATASET_CONFIGS defaults (360k-720k / up to 1.08M),
which would put a single seed far beyond one night.

Both arms of a given (dataset, seed, classifier_type) load the SAME classifier
file, so MADA and RLDA always explain an identical black box.
"""
from __future__ import annotations
import os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/opt/anaconda3/envs/marl/bin/python"
SEEDS = [42, 43, 44, 45, 46]
# cheap first, so a truncated stage still yields the small datasets
DATASETS = ["iris", "wine", "breast_cancer", "synthetic", "housing", "uci_credit", "uci_adult"]
MADA_FRAMES = 144_000
RLDA_STEPS = 270_000
# Arm-aware. MADA is ONE process per dataset, so a flat 3 left 11 of 14 cores
# idle during the MADA stages -- which are ~70% of the wall clock. RLDA spawns
# one process per CLASS (up to 4), so it needs a lower dataset-level cap.
CONCURRENCY = {"mada": 7, "rlda": 4}

ROOTS = {"dnn": REPO / "runs" / "sweep_dnn", "rf": REPO / "runs" / "sweep_rf"}
LOGS = REPO / "runs" / "sweep_logs"
ENV = os.environ.copy()
ENV.update({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "WANDB_MODE": "offline", "WANDB_SILENT": "true", "DISABLE_WANDB": "1",
            "PYTHONUNBUFFERED": "1"})


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGS / "sweep_progress.log", "a") as f:
        f.write(line + "\n")


def fit_rf_classifier(dataset: str, seed: int) -> bool:
    """Pre-fit the RandomForest black box. The pipelines only fit DNNs."""
    dest = ROOTS["rf"] / "classifiers" / f"{dataset}_seed{seed}.pth"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'revision');"
        "from fit_shared_classifier import fit_classifier_to_path;"
        f"fit_classifier_to_path('{dataset}', r'{dest}', seed={seed},"
        " device='cpu', classifier_type='random_forest')"
    )
    r = subprocess.run([PY, "-c", code], cwd=str(REPO), env=ENV,
                       stdout=open(LOGS / f"rffit_{dataset}_seed{seed}.log", "w"),
                       stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and dest.exists()
    log(f"    RF classifier {dataset} seed{seed}: {'ok' if ok else 'FAILED'}")
    return ok


def already_done(arm: str, dataset: str, seed: int, ctype: str) -> bool:
    """Resume: skip a (dataset, seed, arm, classifier) whose Track-A result exists."""
    sub = "maddpg" if arm == "mada" else "ddpg"
    f = (ROOTS[ctype] / "results" / sub /
         f"{dataset}__{arm}__seed{seed}__tp0p90__tc0p10.json")
    return f.exists() and f.stat().st_size > 0


def cmd_for(arm: str, dataset: str, seed: int, ctype: str):
    root = ROOTS[ctype]
    if arm == "mada":
        return [PY, str(REPO / "revision" / "run_mada_pipeline.py"),
                "--algo", "maddpg", "--datasets", dataset, "--seed", str(seed),
                "--device", "cpu", "--root", str(root), "--force", "--force-train",
                "--max_n_frames", str(MADA_FRAMES), "--k", "5"]
    return [PY, str(REPO / "revision" / "run_rlda_pipeline.py"),
            "--algo", "ddpg", "--datasets", dataset, "--seed", str(seed),
            "--device", "cpu", "--root", str(root), "--force", "--force-train",
            "--sa_timesteps", str(RLDA_STEPS)]


def run_stage(arm: str, ctype: str, seed: int) -> None:
    log(f"  STAGE seed={seed} arm={arm} classifier={ctype}  ({len(DATASETS)} datasets)")
    if ctype == "rf":
        for ds in DATASETS:
            fit_rf_classifier(ds, seed)
    todo = [d for d in DATASETS if not already_done(arm, d, seed, ctype)]
    skipped = [d for d in DATASETS if d not in todo]
    if skipped:
        log(f"    resume: already done, skipping {skipped}")
    cap = CONCURRENCY[arm]
    pending, running = todo, {}
    while pending or running:
        while pending and len(running) < cap:
            ds = pending.pop(0)
            lp = LOGS / f"seed{seed}_{arm}_{ctype}_{ds}.log"
            p = subprocess.Popen(cmd_for(arm, ds, seed, ctype), cwd=str(REPO), env=ENV,
                                 stdout=open(lp, "w"), stderr=subprocess.STDOUT,
                                 start_new_session=True)
            running[p] = (ds, time.time())
        time.sleep(20)
        for p in list(running):
            if p.poll() is not None:
                ds, t0 = running.pop(p)
                mins = (time.time() - t0) / 60
                log(f"    {ds:>12} {arm}/{ctype} seed{seed}: rc={p.returncode} ({mins:.0f} min)")
    log(f"  STAGE DONE seed={seed} arm={arm} classifier={ctype}")


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    for r in ROOTS.values():
        (r / "classifiers").mkdir(parents=True, exist_ok=True)
    log(f"SWEEP START  seeds={SEEDS} datasets={DATASETS}")
    log(f"  MADA {MADA_FRAMES} frames/agent | RLDA {RLDA_STEPS} total steps | concurrency {CONCURRENCY}")
    for seed in SEEDS:
        log(f"=== SEED {seed} ===")
        for arm, ctype in (("rlda", "dnn"), ("mada", "dnn"), ("rlda", "rf"), ("mada", "rf")):
            run_stage(arm, ctype, seed)
        log(f"=== SEED {seed} COMPLETE ===")
    log("SWEEP COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
