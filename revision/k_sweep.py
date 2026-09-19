#!/usr/bin/env python3
"""Fidelity–coverage curves by sweeping the union size k.

Why
---
Eff and Cov_τ are two *points* on a fidelity–coverage trade-off, and the methods
sit at different points on it by construction: the RL arms are trained to the
precision floor τ_P=0.90 and buy coverage up to it, while the anchor family
overshoots precision on narrow rules. Comparing them at one operating point
answers "who is better at this k", not "who is better".

k — the number of rules allowed in each class union — moves each method along
its own curve. Sweeping it turns the comparison into curve-vs-curve, where
"does RLDA dominate greedy_anchors?" has an answer that does not depend on
where a threshold was put.

The paper cells are k=1 and are NOT touched: k=1 is read from the existing
result trees, and k∈{2,3,5} is evaluated into a scratch tree.

    python -m revision.k_sweep --out <dir>            # run the sweep
    python -m revision.k_sweep --out <dir> --collect  # aggregate only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS, METHODS, SEEDS, PAPER, WYODOT  # noqa: E402

PY = "/opt/anaconda3/envs/marl/bin/python"
K_VALUES = [1, 2, 3, 5, 10, 20]
K_EXTRA = [2, 3, 5, 10, 20]          # k=1 comes from the paper cells
TAU_P, TAU_C = 0.90, 0.10
CONCURRENCY = 4
RL_ARMS = ("rlda", "mada")
BASELINES = ("cart", "random_search")

ENV = os.environ.copy()
ENV.update({
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "WANDB_MODE": "offline", "WANDB_SILENT": "true", "DISABLE_WANDB": "1",
    "PYTHONUNBUFFERED": "1",
})


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def paper_cell(dataset: str, method: str, seed: int) -> Optional[Path]:
    sub = {"rlda": "ddpg", "mada": "maddpg"}.get(method, "baselines")
    name = f"{dataset}__{method}__seed{seed}__tp0p90__tc0p10.json"
    for tree in (PAPER, WYODOT):
        p = tree / sub / name
        if p.is_file():
            return p
    return None


def rules_file_for(dataset: str, method: str, seed: int) -> Optional[str]:
    p = paper_cell(dataset, method, seed)
    if p is None:
        return None
    rf = (json.loads(p.read_text()).get("extra") or {}).get("rules_file")
    return rf if rf and os.path.isfile(rf) else None


def classifier_for(dataset: str, seed: int) -> Optional[Path]:
    for tree in (PAPER, WYODOT):
        p = tree.parent / "classifiers" / f"{dataset}_seed{seed}.pth"
        if p.is_file():
            return p
    return None


def jobs_for(out: Path) -> List[Tuple[str, List[str], Path]]:
    """(label, argv, expected-output-dir) for every k we still need."""
    out_jobs: List[Tuple[str, List[str], Path]] = []
    for k in K_EXTRA:
        kdir = out / f"k{k}"
        for dataset in DATASETS:
            for seed in SEEDS:
                for arm in RL_ARMS:
                    rf = rules_file_for(dataset, arm, seed)
                    if rf is None:
                        continue
                    tgt = kdir / f"{dataset}__{arm}__seed{seed}__tp0p90__tc0p10.json"
                    if tgt.is_file():
                        continue
                    out_jobs.append((
                        f"{dataset} {arm} s{seed} k{k}",
                        [PY, "-m", "revision.evaluate",
                         "--rules_file", rf, "--dataset", dataset, "--method", arm,
                         "--seed", str(seed), "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
                         "--k", str(k), "--out_dir", str(kdir)],
                        kdir,
                    ))
                clf = classifier_for(dataset, seed)
                if clf is None:
                    continue
                done = all(
                    (kdir / f"{dataset}__{m}__seed{seed}__tp0p90__tc0p10.json").is_file()
                    for m in BASELINES
                )
                if done:
                    continue
                out_jobs.append((
                    f"{dataset} baselines s{seed} k{k}",
                    [PY, "-m", "revision.baselines",
                     "--dataset", dataset, "--seed", str(seed), "--k", str(k),
                     "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
                     "--classifier_path", str(clf), "--methods", *BASELINES,
                     "--budget_per_class", "5", "--n_candidates", "256",
                     "--out_dir", str(kdir)],
                    kdir,
                ))
    return out_jobs


def run_all(out: Path) -> int:
    jobs = jobs_for(out)
    log(f"{len(jobs)} jobs (k={K_EXTRA}; k=1 reused from the paper cells)")
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    pending, running, failed = list(jobs), {}, []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < CONCURRENCY:
            label, argv, kdir = pending.pop(0)
            kdir.mkdir(parents=True, exist_ok=True)
            lf = open(logdir / (label.replace(" ", "_") + ".log"), "w")
            proc = subprocess.Popen(argv, cwd=str(REPO), env=ENV,
                                    stdout=lf, stderr=subprocess.STDOUT)
            running[proc.pid] = (proc, label, lf)
        time.sleep(4)
        for pid, (proc, label, lf) in list(running.items()):
            if proc.poll() is None:
                continue
            lf.close()
            del running[pid]
            if proc.returncode != 0:
                failed.append(label)
                log(f"  FAIL {label} rc={proc.returncode}")
        if len(pending) % 40 == 0 and pending:
            log(f"  {len(pending)} queued, {len(running)} running")
    log(f"done in {(time.time()-t0)/60:.1f} min; {len(failed)} failures")
    for f in failed:
        log(f"  failed: {f}")
    return len(failed)


def collect(out: Path) -> Dict[str, Any]:
    """method -> k -> per-dataset means of Fid / Cov / Eff."""
    import statistics as st

    def read(path: Path) -> Optional[Dict[str, float]]:
        if not path.is_file():
            return None
        g = json.loads(path.read_text()).get("global_ruleset") or {}
        fid, cov = g.get("global_fidelity"), g.get("coverage")
        if cov is None:
            return None
        fid_f = float(fid) if fid is not None and fid == fid else None
        return {
            "fid": fid_f,
            "cov": float(cov),
            "eff": (fid_f * float(cov)) if fid_f is not None else 0.0,
        }

    curves: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in METHODS:
        curves[method] = {}
        for k in K_VALUES:
            per_ds = []
            for dataset in DATASETS:
                vals = []
                for seed in SEEDS:
                    if k == 1:
                        p = paper_cell(dataset, method, seed)
                    else:
                        p = out / f"k{k}" / (
                            f"{dataset}__{method}__seed{seed}__tp0p90__tc0p10.json")
                    r = read(p) if p is not None else None
                    if r:
                        vals.append(r)
                if vals:
                    per_ds.append({
                        "dataset": dataset,
                        "fid": st.mean([v["fid"] for v in vals if v["fid"] is not None])
                        if any(v["fid"] is not None for v in vals) else None,
                        "cov": st.mean([v["cov"] for v in vals]),
                        "eff": st.mean([v["eff"] for v in vals]),
                        "n_seeds": len(vals),
                    })
            if not per_ds:
                continue
            fids = [d["fid"] for d in per_ds if d["fid"] is not None]
            curves[method][k] = {
                "n_datasets": len(per_ds),
                "fid": st.mean(fids) if fids else None,
                "cov": st.mean([d["cov"] for d in per_ds]),
                "eff": st.mean([d["eff"] for d in per_ds]),
                "per_dataset": per_ds,
            }
    return curves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--collect", action="store_true", help="skip running, aggregate only")
    args = ap.parse_args()
    out = Path(args.out)
    rc = 0
    if not args.collect:
        rc = run_all(out)
    curves = collect(out)
    (out / "curves.json").write_text(json.dumps(curves, indent=2, default=str))
    print(f"\n{'method':16s} " + "".join(f"{'k='+str(k):>22s}" for k in K_VALUES))
    print(f"{'':16s} " + "".join(f"{'Fid / Cov / Eff':>22s}" for _ in K_VALUES))
    for method in METHODS:
        row = f"{method:16s} "
        for k in K_VALUES:
            c = curves.get(method, {}).get(k)
            row += (f"{c['fid']:.3f} /{c['cov']:.3f} /{c['eff']:.3f}".rjust(22)
                    if c else "—".rjust(22))
        print(row)
    print(f"\nwrote {out / 'curves.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
