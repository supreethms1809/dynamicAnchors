#!/usr/bin/env python3
"""Re-select and re-report every seed-42 method under both coverage bases.

  true_label : Cov = P(x in B | y = c)        (the convention so far)
  predicted  : Cov = P(x in B | f_hat(x) = c) (the model being explained)

The basis is not only a reporting change: rule SELECTION on D_val ranks by
LCB(Fid) x (1 + Cov), so a different Cov can pick a different rule. Both bases
are therefore run end-to-end (select on D_val, report on D_test) from the SAME
pools, and every pool is the matched final configuration:

  RLDA-emp / RLDA-pert : top-K-capped RLDA pools (same cap as MADA)
  MADA-emp / MADA-pert : per-class checkpointing
  CART / RandS -emp/-pert, SP-Anch*, GreedyAnch* : revision.baselines

  python revision/recompute_seed42.py            # run + tables
  python revision/recompute_seed42.py --collect  # tables only
  python revision/recompute_seed42.py --seed 43 --bases predicted --labels RLDA-emp MADA-emp CART-emp
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon  # noqa: E402

PY = "/opt/anaconda3/envs/marl/bin/python"
SEED = 42
BASES = ["true_label", "predicted"]
DATASETS = ["iris", "synthetic", "wine", "sick", "breast_cancer", "uci_credit", "mammography",
            "housing", "heloc", "uci_adult", "folktables_income_CA_2018", "wyodot_kvdw_labeled"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
OUT = REPO / "runs" / "recompute_seed42"

def rl_sources(seed):
    """label -> (results dir holding the k=1 artifact whose extra.rules_file is the pool, method)."""
    mada_pert = "perturb_fid_perclass_seed42" if seed == 42 else f"perturb_fid_seed{seed}"
    return {
        "RLDA-emp": (REPO / f"runs/rlda_emp_capped_seed{seed}/results/ddpg", "rlda"),
        "RLDA-pert": (REPO / f"runs/perturb_fid_seed{seed}/results/ddpg", "rlda"),
        "MADA-emp": (REPO / f"runs/paper_mada_perclass_seed{seed}/results/maddpg", "mada"),
        "MADA-pert": (REPO / f"runs/{mada_pert}/results/maddpg", "mada"),
    }


RL = rl_sources(SEED)
LABELS = None  # None = every label; else only these are (re)run
BASE = {  # sub-tree -> (methods, fid_estimator)
    "base_emp": (["cart", "random_search", "sp_anchors", "greedy_anchors"], "empirical"),
    "base_pert": (["cart", "random_search"], "perturbed"),
}
BASE_LABEL = {("base_emp", "cart"): "CART-emp", ("base_pert", "cart"): "CART-pert",
              ("base_emp", "random_search"): "RandS-emp", ("base_pert", "random_search"): "RandS-pert",
              ("base_emp", "sp_anchors"): "SP-Anch*", ("base_emp", "greedy_anchors"): "GreedyAnch*"}
ORDER = ["CART-emp", "CART-pert", "RandS-emp", "RandS-pert", "SP-Anch*", "GreedyAnch*",
         "RLDA-emp", "RLDA-pert", "MADA-emp", "MADA-pert"]


def classifier(ds):
    sub = "wyodot_fiveseed_overlap075/dnn" if ds.startswith("wyodot") else "paper_fiveseed_overlap075"
    return MAIN / "runs" / sub / "classifiers" / f"{ds}_seed{SEED}.pth"


def name(ds, method):
    return f"{ds}__{method}__seed{SEED}__tp0p90__tc0p10.json"


def jobs():
    js = []
    for basis in BASES:
        for label, (src, method) in RL.items():
            if LABELS and label not in LABELS:
                continue
            for ds in DATASETS:
                f = src / name(ds, method)
                tgt = OUT / basis / label / name(ds, method)
                if tgt.exists():
                    continue
                rf = (json.loads(f.read_text()).get("extra") or {}).get("rules_file") if f.exists() else None
                if not rf or not Path(rf).exists():
                    print(f"  MISSING pool: {basis} {label} {ds} ({f})", flush=True)
                    continue
                js.append((f"{basis} {label} {ds}",
                           [PY, "-m", "revision.evaluate", "--rules_file", rf, "--dataset", ds,
                            "--method", method, "--seed", str(SEED), "--tau_p", "0.9", "--tau_c", "0.1",
                            "--k", "1", "--coverage_basis", basis, "--out_dir", str(tgt.parent)], tgt))
        for sub, (methods, est) in BASE.items():
            methods = [m for m in methods if not LABELS or BASE_LABEL[(sub, m)] in LABELS]
            if not methods:
                continue
            for ds in DATASETS:
                tgt = OUT / basis / sub / name(ds, methods[0])
                if all((OUT / basis / sub / name(ds, m)).exists() for m in methods):
                    continue
                js.append((f"{basis} {sub} {ds}",
                           [PY, "-m", "revision.baselines", "--dataset", ds, "--seed", str(SEED),
                            "--k", "1", "--tau_p", "0.9", "--tau_c", "0.1",
                            "--classifier_path", str(classifier(ds)), "--methods", *methods,
                            "--budget_per_class", "5", "--n_candidates", "256",
                            "--fid_estimator", est, "--coverage_basis", basis,
                            "--out_dir", str(tgt.parent)], tgt))
    return js


def run(workers):
    js = jobs()
    print(f"{len(js)} jobs", flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    def one(job):
        label, argv, tgt = job
        with open(OUT / "logs" / (label.replace(" ", "_") + ".log"), "w") as fh:
            rc = subprocess.run(argv, cwd=str(REPO), env=ENV, stdout=fh, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and tgt.exists()
        print(f"  {'ok  ' if ok else 'FAIL'} {label}", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fails = sum(0 if ok else 1 for ok in ex.map(one, js))
    for basis in BASES:
        for d in sorted((OUT / basis).glob("*")):
            if d.is_dir() and not (d / "_dual" / "rescore_rules.csv").exists():
                subprocess.run([PY, "revision/dual_estimator_rescore.py", "--results_dir", str(d),
                                "--out_dir", str(d / "_dual")], cwd=str(REPO), env=ENV,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"done, {fails} failures")
    return fails


def anc_of(d):
    p = d / "_dual" / "rescore_rules.csv"
    acc = defaultdict(list)
    if p.exists():
        for r in csv.DictReader(open(p)):
            try:
                acc[(r["dataset"], r["method"])].append(float(r["fid_anchors"]))
            except ValueError:
                pass
    return {k: sum(v) / len(v) for k, v in acc.items()}


def cell(basis, label):
    """dataset -> metrics for one method under one basis."""
    if label in RL:
        sub, method = label, RL[label][1]
    else:
        sub, method = next((s, m) for (s, m), l in BASE_LABEL.items() if l == label)
    d = OUT / basis / sub
    anc = anc_of(d)
    out = {}
    for ds in DATASETS:
        f = d / name(ds, method)
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        g = r.get("global_ruleset") or {}
        fid, cov = g.get("global_fidelity"), g.get("coverage")
        fid = float("nan") if fid is None else float(fid)
        cov = float(cov) if cov is not None else float("nan")
        cls_cov = [blk["union"]["coverage"] for blk in (r.get("per_class") or {}).values()
                   if blk.get("union", {}).get("coverage") is not None]
        cls_fid = [blk["union"]["fidelity"] for blk in (r.get("per_class") or {}).values()
                   if blk.get("union", {}).get("fidelity") is not None]
        out[ds] = {"Fid": fid, "Cov": cov,
                   "Eff": fid * cov if math.isfinite(fid) and math.isfinite(cov) else float("nan"),
                   "clsCov": sum(cls_cov) / len(cls_cov) if cls_cov else float("nan"),
                   "clsFid": sum(cls_fid) / len(cls_fid) if cls_fid else float("nan"),
                   "Anc": anc.get((ds, method), float("nan")),
                   "basis": (r.get("extra") or {}).get("coverage_basis")}
    return out


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.3f}"


def collect():
    if any(not (OUT / b).is_dir() for b in BASES):
        print(f"collect: {OUT} lacks one of {BASES}; skipping comparison table")
        return
    data = {b: {l: cell(b, l) for l in ORDER} for b in BASES}
    lines = [f"# Coverage basis recompute (seed {SEED}, D_test, k=1)", "",
             "true_label: Cov = P(x in B | y=c)  |  predicted: Cov = P(x in B | f_hat(x)=c)",
             "clsCov / clsFid = mean over classes of the class union. Cov = 1 - abstention",
             "(all test rows). Eff = Fid x Cov. Anc = perturbed Fid. Selection on D_val uses",
             "the same basis as reporting, so the selected rule can differ between bases.", ""]
    for metric in ("clsCov", "clsFid", "Fid", "Cov", "Eff", "Anc"):
        lines += [f"## {metric}  (true_label → predicted)", "",
                  "| dataset | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
        for ds in DATASETS:
            row = []
            for l in ORDER:
                a = data["true_label"][l].get(ds, {}).get(metric)
                b = data["predicted"][l].get(ds, {}).get(metric)
                row.append(f"{fmt(a)} → {fmt(b)}")
            lines.append(f"| `{ds}` | " + " | ".join(row) + " |")
        row = []
        for l in ORDER:
            ms = []
            for b in BASES:
                vs = [data[b][l][ds][metric] for ds in DATASETS
                      if ds in data[b][l] and math.isfinite(data[b][l][ds][metric])]
                ms.append(sum(vs) / len(vs) if vs else float("nan"))
            row.append(f"**{fmt(ms[0])} → {fmt(ms[1])}**")
        lines.append("| **mean** | " + " | ".join(row) + " |")
        lines.append("")

    lines += ["## Does the basis change conclusions? (paired Wilcoxon, n=12, predicted basis)", "",
              "| contrast | metric | mean A | mean B | wins B | p |", "|---|---|---:|---:|---:|---:|"]
    P = data["predicted"]
    for a, b in (("RLDA-emp", "RLDA-pert"), ("MADA-emp", "MADA-pert"), ("RLDA-emp", "MADA-emp"),
                 ("RLDA-pert", "MADA-pert"), ("GreedyAnch*", "RLDA-pert"), ("CART-emp", "RLDA-emp"),
                 ("CART-pert", "RLDA-pert")):
        for metric in ("clsCov", "Eff", "Anc"):
            pr = [(P[a][ds][metric], P[b][ds][metric]) for ds in DATASETS
                  if ds in P[a] and ds in P[b]
                  and math.isfinite(P[a][ds][metric]) and math.isfinite(P[b][ds][metric])]
            if len(pr) < 2:
                continue
            xs, ys = [x for x, _ in pr], [y for _, y in pr]
            w = paired_wilcoxon(ys, xs)
            pv = w.get("pvalue")
            lines.append(f"| {b} vs {a} | {metric} | {fmt(sum(xs)/len(xs))} | {fmt(sum(ys)/len(ys))} "
                         f"| {sum(1 for x, y in pr if y > x + 1e-9)}/{len(pr)} | {'—' if pv is None else f'{pv:.4f}'} |")
    bad = [(b, l, ds) for b in BASES for l in ORDER for ds, v in data[b][l].items() if v.get("basis") not in (b, None)]
    if bad:
        lines += ["", f"WARNING: {len(bad)} artifacts record a different coverage_basis than their tree"]
    (OUT / "coverage_basis_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'coverage_basis_comparison.md'}")


def main():
    global SEED, OUT, RL, BASES, LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bases", nargs="+", choices=BASES, default=list(BASES))
    ap.add_argument("--labels", nargs="+", choices=ORDER, default=None)
    args = ap.parse_args()
    SEED, BASES, LABELS = args.seed, args.bases, args.labels
    OUT = REPO / "runs" / f"recompute_seed{SEED}"
    RL = rl_sources(SEED)
    rc = 0
    if not args.collect:
        rc = run(args.workers)
    BASES = ["true_label", "predicted"]
    collect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
