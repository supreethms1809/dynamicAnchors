#!/usr/bin/env python3
"""Re-run SP-/greedy-Anchors on paper_final after the exact-bins fix.

The anchor-family boxes were built from `anchor-exp`'s printed rule, which rounds
every bin edge to 2 decimals, and a strict `x > v` became `x >= v` once the box
was stored as float32. On binary/integer features that made rules near-universal.
The pool also admitted the empty anchor (the class prior), which the RL arms may
not submit. `revision.baselines` now builds each box from the discretizer's exact
bins and drops empty anchors; this re-runs every anchor-family cell of paper_final.
CART and random search never touch that code and are not re-run.

  main    k=1, budget 5, τ_C 0.10 and 0.20      -> <out>/baselines_emp/
  k_sweep k in 2,3,5,10,20, budget max(k, 5)    -> <out>/k_sweep/k<k>/
  pool20  one budget-20 pool reduced at k 1..20 -> <out>/pool20/k<k>/

Classifiers: seeds 42-43 were fit on the Mac, seeds 44-46 on spark. Off spark,
seeds 44-46 use the spark copies in ../results/paper_final{,_updated}/emp_tc0p10/
classifiers; each reproduces spark's train/val/test accuracy exactly on the Mac,
except housing seed 44, which predicts one test row differently off spark.

  python revision/rerun_anchor_family_exactbins.py --seeds 42 43 44 45        # dry run
  python revision/rerun_anchor_family_exactbins.py --seeds 42 43 44 45 --apply
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
DATASETS = [
    "iris", "wine", "synthetic", "breast_cancer", "uci_credit", "sick",
    "mammography", "heloc", "housing", "uci_adult",
    "folktables_income_CA_2018", "wyodot_kvdw_labeled",
]
METHODS = ["sp_anchors", "greedy_anchors"]
TAU_P = 0.90
K_SWEEP = [2, 3, 5, 10, 20]
POOL20_K = [1, 2, 3, 5, 10, 20]
SPARK_SEEDS = {44, 45, 46}
SPARK_COPIES = [REPO.parent / "results" / d / "emp_tc0p10" / "classifiers"
                for d in ("paper_final", "paper_final_updated")]

ENV = os.environ.copy()
ENV.update({
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "WANDB_MODE": "offline", "WANDB_SILENT": "true", "DISABLE_WANDB": "1",
    "PYTHONUNBUFFERED": "1", "DYNANC_COVERAGE_BASIS": "predicted",
})


def classifier_for(ds: str, seed: int, on_spark: bool, runs: Path) -> Optional[Path]:
    sub = "wyodot_fiveseed_overlap075/dnn" if ds.startswith("wyodot") else "paper_fiveseed_overlap075"
    local = runs / sub / "classifiers" / f"{ds}_seed{seed}.pth"
    if seed in SPARK_SEEDS and not on_spark:
        return next((c / f"{ds}_seed{seed}.pth" for c in SPARK_COPIES
                     if (c / f"{ds}_seed{seed}.pth").is_file()), None)
    return local if local.is_file() else None


def jobs(out: Path, seeds: List[int], datasets: List[str], parts: List[str],
         on_spark: bool, py: str, runs: Path) -> Tuple[List[Tuple[str, list, Path]], List[str]]:
    todo, skipped = [], []
    for seed in seeds:
        for ds in datasets:
            clf = classifier_for(ds, seed, on_spark, runs)
            if clf is None:
                skipped.append(f"{ds} s{seed}: no classifier for this machine")
                continue
            base = [py, "-m", "revision.baselines", "--dataset", ds, "--seed", str(seed),
                    "--tau_p", str(TAU_P), "--coverage_basis", "predicted",
                    "--fid_estimator", "empirical", "--classifier_path", str(clf),
                    "--methods", *METHODS, "--n_candidates", "256"]
            if "main" in parts:
                for tc in ("0.10", "0.20"):
                    d = out / "baselines_emp"
                    todo.append((f"main {ds} s{seed} tc{tc}", base + [
                        "--k", "1", "--tau_c", tc, "--budget_per_class", "5",
                        "--out_dir", str(d)], d))
            if "k_sweep" in parts:
                for k in K_SWEEP:
                    d = out / "k_sweep" / f"k{k}"
                    todo.append((f"k_sweep {ds} s{seed} k{k}", base + [
                        "--k", str(k), "--tau_c", "0.10", "--budget_per_class", str(max(k, 5)),
                        "--out_dir", str(d)], d))
            if "pool20" in parts:
                d = out / "pool20"
                todo.append((f"pool20 {ds} s{seed}", base + [
                    "--k", "1", "--tau_c", "0.10", "--budget_per_class", "20",
                    "--k_values", *map(str, POOL20_K), "--out_dir", str(d)], d))
    return todo, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--parts", nargs="+", default=["main", "k_sweep", "pool20"],
                    choices=["main", "k_sweep", "pool20"])
    ap.add_argument("--out", type=Path, default=REPO.parent / "results" / "paper_final_anchorfix")
    ap.add_argument("--on_spark", action="store_true",
                    help="seeds 44-46 use this checkout's own runs/*/classifiers")
    ap.add_argument("--runs", type=Path, default=REPO / "runs",
                    help="runs/ folder holding paper_fiveseed_overlap075/ and wyodot_fiveseed_overlap075/")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    todo, skipped = jobs(a.out, a.seeds, a.datasets, a.parts, a.on_spark, a.python, a.runs)
    for s in skipped:
        print("SKIP", s)
    print(f"{len(todo)} jobs -> {a.out}")
    if not a.apply:
        for name, cmd, _ in todo[:5]:
            print(name, " ".join(cmd))
        print("dry run; pass --apply")
        return 0

    logs = a.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(todo), {}, []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < a.concurrency:
            name, cmd, d = pending.pop(0)
            d.mkdir(parents=True, exist_ok=True)
            lf = open(logs / (name.replace(" ", "_") + ".log"), "w")
            running[name] = (subprocess.Popen(cmd, cwd=REPO, env=ENV, stdout=lf,
                                              stderr=subprocess.STDOUT), lf)
        for name in list(running):
            p, lf = running[name]
            if p.poll() is None:
                continue
            lf.close()
            del running[name]
            if p.returncode:
                failed.append(name)
            n_done = len(todo) - len(pending) - len(running)
            print(f"[{datetime.now():%H:%M:%S}] {'FAIL' if p.returncode else 'ok  '} {name} "
                  f"({n_done}/{len(todo)}, {time.time() - t0:.0f}s)", flush=True)
        time.sleep(1)
    print(f"done in {time.time() - t0:.0f}s; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
