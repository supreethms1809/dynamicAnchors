#!/usr/bin/env python3
"""Union-size (k) sweep for the final-comparison variants, seed 42.

k is how many rules a class union may hold. Every method sits somewhere on its
own fidelity-coverage curve, and k moves it along that curve, so a one-k table
answers "who wins at this k", not "who wins".

No retraining: the RL arms re-select from the rule pools already on disk, and the
baselines re-run their (cheap) search at each k. The Anchors family reduces ONE
pool at every k (its explainer sampling is unseeded, so regenerating per k would
confound the curve with explainer noise).

  python revision/k_sweep_variants.py --run           # fill runs/k_sweep_matched_seed42
  python revision/k_sweep_variants.py --collect       # table only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
sys.path.insert(0, str(REPO))

PY = "/opt/anaconda3/envs/marl/bin/python"
SEED = 42
K_VALUES = [1, 2, 3, 5]
TAU_P, TAU_C = 0.90, 0.10
DATASETS = ["iris", "synthetic", "wine", "sick", "breast_cancer", "uci_credit", "mammography",
            "housing", "heloc", "uci_adult", "folktables_income_CA_2018", "wyodot_kvdw_labeled"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def paper_root(ds: str) -> Path:
    sub = "wyodot_fiveseed_overlap075/dnn" if ds.startswith("wyodot") else "paper_fiveseed_overlap075"
    return MAIN / "runs" / sub


# RL variants: label -> (k=1 result dir, method name in the artifact)
# Matched pools: top-K-capped RLDA, per-class-checkpoint MADA (same as recompute_seed42).
RL_VARIANTS = {
    "RLDA-emp": (lambda ds: REPO / "runs/rlda_emp_capped_seed42/results/ddpg", "rlda"),
    "RLDA-pert": (lambda ds: REPO / "runs/perturb_fid_seed42/results/ddpg", "rlda"),
    "MADA-emp": (lambda ds: REPO / "runs/paper_mada_perclass_seed42/results/maddpg", "mada"),
    "MADA-pert": (lambda ds: REPO / "runs/perturb_fid_perclass_seed42/results/maddpg", "mada"),
}
COVERAGE_BASIS = "predicted"
BASE_VARIANTS = {          # label -> (methods, fid_estimator)
    "emp": (["cart", "random_search", "sp_anchors", "greedy_anchors"], "empirical"),
    "pert": (["cart", "random_search"], "perturbed"),
}
BASE_LABEL = {("cart", "emp"): "CART-emp", ("cart", "pert"): "CART-pert",
              ("random_search", "emp"): "RandS-emp", ("random_search", "pert"): "RandS-pert",
              ("sp_anchors", "emp"): "SP-Anch*", ("greedy_anchors", "emp"): "GreedyAnch*"}


def rules_file(ds: str, label: str) -> str | None:
    root, method = RL_VARIANTS[label]
    f = Path(root(ds)) / f"{ds}__{method}__seed{SEED}__tp0p90__tc0p10.json"
    if not f.exists():
        return None
    rf = (json.loads(f.read_text()).get("extra") or {}).get("rules_file")
    return rf if rf and os.path.isfile(rf) else None


def classifier(ds: str) -> Path:
    return paper_root(ds) / "classifiers" / f"{ds}_seed{SEED}.pth"


def result_path(out: Path, k: int, label: str, ds: str, method: str) -> Path:
    return out / f"k{k}" / label / f"{ds}__{method}__seed{SEED}__tp0p90__tc0p10.json"


def jobs(out: Path) -> list[tuple[str, list[str], Path]]:
    js = []
    for k in K_VALUES:
        for label, (_, method) in RL_VARIANTS.items():
            for ds in DATASETS:
                tgt = result_path(out, k, label, ds, method)
                rf = rules_file(ds, label)
                if tgt.exists() or rf is None:
                    continue
                js.append((f"k{k} {label} {ds}",
                           [PY, "-m", "revision.evaluate", "--rules_file", rf, "--dataset", ds,
                            "--method", method, "--seed", str(SEED), "--tau_p", str(TAU_P),
                            "--tau_c", str(TAU_C), "--k", str(k), "--coverage_basis", COVERAGE_BASIS,
                            "--out_dir", str(tgt.parent)],
                           tgt))
        for blabel, (methods, est) in BASE_VARIANTS.items():
            for ds in DATASETS:
                tgts = [result_path(out, k, blabel, ds, m) for m in methods]
                if all(t.exists() for t in tgts):
                    continue
                js.append((f"k{k} baselines-{blabel} {ds}",
                           [PY, "-m", "revision.baselines", "--dataset", ds, "--seed", str(SEED),
                            "--k", str(k), "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
                            "--classifier_path", str(classifier(ds)), "--methods", *methods,
                            "--budget_per_class", "5", "--n_candidates", "256",
                            "--fid_estimator", est, "--coverage_basis", COVERAGE_BASIS,
                            "--out_dir", str(tgts[0].parent)],
                           tgts[0]))
    return js


def run_all(out: Path, workers: int) -> int:
    js = jobs(out)
    print(f"{len(js)} jobs", flush=True)
    logdir = out / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    failures = 0

    def one(job):
        label, argv, tgt = job
        t0 = time.time()
        lp = logdir / (label.replace(" ", "_") + ".log")
        with open(lp, "w") as fh:
            rc = subprocess.run(argv, cwd=str(REPO), env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and tgt.exists()
        print(f"  {'ok  ' if ok else 'FAIL'} {label} ({(time.time()-t0)/60:.1f} min)", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(one, js):
            failures += 0 if ok else 1
    print(f"done, {failures} failures")
    return failures


def rescore(out: Path) -> None:
    """Anchors-sampler Fid for the rules each k selected."""
    for k in K_VALUES:
        for sub in list(RL_VARIANTS) + list(BASE_VARIANTS):
            d = out / f"k{k}" / sub
            if not d.is_dir() or (d / "_dual" / "rescore_rules.csv").exists():
                continue
            subprocess.run([PY, "revision/dual_estimator_rescore.py", "--results_dir", str(d),
                            "--out_dir", str(d / "_dual")], cwd=str(REPO), env=ENV,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def anc_of(d: Path) -> dict:
    p = d / "_dual" / "rescore_rules.csv"
    acc = defaultdict(list)
    if not p.exists():
        return {}
    for r in csv.DictReader(open(p)):
        try:
            acc[(r["dataset"], r["method"])].append(float(r["fid_anchors"]))
        except ValueError:
            pass
    return {k: sum(v) / len(v) for k, v in acc.items()}


def collect(out: Path) -> None:
    rows = defaultdict(dict)      # label -> k -> metric means
    for k in K_VALUES:
        for label, (_, method) in RL_VARIANTS.items():
            rows[label][k] = summarize(out, k, label, method)
        for blabel, (methods, _) in BASE_VARIANTS.items():
            for m in methods:
                rows[BASE_LABEL[(m, blabel)]][k] = summarize(out, k, blabel, m)
    lines = [f"# Union size k sweep (seed {SEED}, {len(DATASETS)} datasets, D_test)", "",
             "Means over datasets. n_rules = mean rules actually selected per class union",
             "(marginal gain can stop below k). feats = mean active features per rule.", ""]
    for metric in ("Fid", "Cov", "Eff", "Anc", "n_rules", "feats"):
        lines += [f"## {metric}", "", "| method | " + " | ".join(f"k={k}" for k in K_VALUES) + " |",
                  "|---" * (len(K_VALUES) + 1) + "|"]
        for label in rows:
            cells = []
            for k in K_VALUES:
                v = rows[label].get(k, {}).get(metric)
                cells.append("—" if v is None or not math.isfinite(v) else f"{v:.3f}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "k_sweep_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out / 'k_sweep_summary.md'}")


def summarize(out: Path, k: int, sub: str, method: str) -> dict:
    anc = anc_of(out / f"k{k}" / sub)
    acc = defaultdict(list)
    for ds in DATASETS:
        f = result_path(out, k, sub, ds, method)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        g = r.get("global_ruleset") or {}
        fid, cov = g.get("global_fidelity"), g.get("coverage")
        if fid is not None and cov is not None:
            acc["Fid"].append(float(fid))
            acc["Cov"].append(float(cov))
            eff = g.get("effectiveness")
            acc["Eff"].append(float(eff) if eff is not None else float(fid) * float(cov))
        n_sel, feats = [], []
        for blk in (r.get("per_class") or {}).values():
            n_sel.append(int(blk.get("n_selected") or blk.get("k") or 0))
            mf = (blk.get("compactness") or {}).get("mean_active_features")
            if mf is not None:
                feats.append(float(mf))
        if n_sel:
            acc["n_rules"].append(sum(n_sel) / len(n_sel))
        if feats:
            acc["feats"].append(sum(feats) / len(feats))
        a = anc.get((ds, method))
        if a is not None:
            acc["Anc"].append(a)
    return {m: (sum(v) / len(v) if v else float("nan")) for m, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "runs" / "k_sweep_matched_seed42"))
    ap.add_argument("--coverage_basis", choices=["true_label", "predicted"], default="predicted")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    global COVERAGE_BASIS
    COVERAGE_BASIS = args.coverage_basis
    out = Path(args.out)
    rc = 0
    if args.run or not args.collect:
        out.mkdir(parents=True, exist_ok=True)
        rc = run_all(out, args.workers)
        rescore(out)
    if args.collect or not args.run:
        rescore(out)
        collect(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
