#!/usr/bin/env python3
"""Compare the perturbed-Fid pilot against the paper-lock seed-42 runs.

Two views, both on D_test:
  * Track A class-level ruleset (revision.evaluate): global Fid, Cov, Eff.
  * Selected rules re-scored by revision.dual_estimator_rescore: empirical Fid,
    Anchors-sampler Fid (the estimand the pilot trains on), gap, active features.

  python revision/dual_estimator_rescore.py \\
      --results_dir runs/perturb_fid_seed42/results \\
      --out_dir runs/perturb_fid_seed42/diagnostics/dual_estimator
  python revision/compare_perturb_fid.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.metrics import paired_wilcoxon  # noqa: E402

MAIN = REPO.parents[2] if REPO.parent.name == "worktrees" else REPO
SEED = 42
DATASETS = [
    "iris", "synthetic", "wine", "sick", "breast_cancer", "uci_credit",
    "mammography", "housing", "heloc", "uci_adult", "folktables_income_CA_2018",
    "wyodot_kvdw_labeled",
]
ARMS = [("rlda", "ddpg"), ("mada", "maddpg")]


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def rule_means(csv_path: Path) -> dict:
    """(dataset, method) -> mean over selected seed-42 rules."""
    acc = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if int(r["seed"]) != SEED:
                continue
            key = (r["dataset"], r["method"])
            for col in ("fid_emp", "fid_anchors", "n_active", "n_covered"):
                v = _f(r[col])
                if math.isfinite(v):
                    acc[key][col].append(v)
    out = {}
    for key, cols in acc.items():
        out[key] = {c: sum(v) / len(v) for c, v in cols.items() if v}
        out[key]["n_rules"] = max(len(v) for v in cols.values())
    return out


def track_a(results_root: Path, dataset: str, arm: str, sub: str) -> dict:
    d = results_root / sub
    files = sorted(d.glob(f"{dataset}__{arm}__seed{SEED}__tp0p90__tc*.json"))
    if not files:
        return {}
    f = files[-1]
    g = json.loads(f.read_text()).get("global_ruleset") or {}
    fid, cov = _f(g.get("global_fidelity")), _f(g.get("coverage"))
    return {"fid": fid, "cov": cov, "eff": fid * cov if math.isfinite(fid) and math.isfinite(cov) else 0.0}


def paper_results_root(dataset: str) -> Path:
    if dataset.startswith("wyodot"):
        return MAIN / "runs" / "wyodot_fiveseed_overlap075" / "dnn" / "results"
    return MAIN / "runs" / "paper_fiveseed_overlap075" / "results"


def fmt(x, nd=3):
    return "—" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "runs" / "perturb_fid_seed42"))
    ap.add_argument("--base_csv", default=str(MAIN / "revision" / "diagnostics" / "dual_estimator" / "rescore_rules.csv"))
    args = ap.parse_args()
    root = Path(args.root)
    new_csv = root / "diagnostics" / "dual_estimator" / "rescore_rules.csv"
    new_r = rule_means(new_csv) if new_csv.exists() else {}
    base_r = rule_means(Path(args.base_csv))

    lines = [f"# Perturbed-Fid pilot vs paper lock (seed {SEED}, D_test)", ""]
    paired = defaultdict(lambda: defaultdict(list))
    for arm, sub in ARMS:
        lines += [f"## {arm.upper()}", "",
                  "| dataset | Fid emp old→new | Fid anchors old→new | greedy_anchors Fid anchors "
                  "| Track A Cov old→new | Track A Eff old→new | active feats old→new |",
                  "|---|---|---|---:|---|---|---|"]
        for ds in DATASETS:
            o, n = base_r.get((ds, arm), {}), new_r.get((ds, arm), {})
            ga = base_r.get((ds, "greedy_anchors"), {})
            ta_o = track_a(paper_results_root(ds), ds, arm, sub)
            ta_n = track_a(root / "results", ds, arm, sub)
            lines.append(
                f"| `{ds}` | {fmt(o.get('fid_emp'))} → {fmt(n.get('fid_emp'))} "
                f"| {fmt(o.get('fid_anchors'))} → **{fmt(n.get('fid_anchors'))}** "
                f"| {fmt(ga.get('fid_anchors'))} "
                f"| {fmt(ta_o.get('cov'))} → {fmt(ta_n.get('cov'))} "
                f"| {fmt(ta_o.get('eff'))} → {fmt(ta_n.get('eff'))} "
                f"| {fmt(o.get('n_active'), 1)} → {fmt(n.get('n_active'), 1)} |"
            )
            if n and o and ta_n and ta_o:
                for name, a, b in (
                    ("fid_anchors", o.get("fid_anchors"), n.get("fid_anchors")),
                    ("fid_emp", o.get("fid_emp"), n.get("fid_emp")),
                    ("cov", ta_o.get("cov"), ta_n.get("cov")),
                    ("eff", ta_o.get("eff"), ta_n.get("eff")),
                    ("new_minus_greedy_fid_anchors", ga.get("fid_anchors"), n.get("fid_anchors")),
                ):
                    if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                        paired[arm][name].append((a, b))
        lines.append("")
        lines += ["| metric | n | mean old/ref | mean new | Δ | Wilcoxon p |", "|---|---:|---:|---:|---:|---:|"]
        for name, pairs in paired[arm].items():
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            w = paired_wilcoxon(b, a) if len(pairs) >= 5 else {}
            pval = w.get("pvalue") if isinstance(w, dict) else math.nan
            lines.append(f"| {name} | {len(pairs)} | {fmt(sum(a) / len(a))} | {fmt(sum(b) / len(b))} "
                         f"| {fmt(sum(b) / len(b) - sum(a) / len(a))} | {fmt(_f(pval), 4)} |")
        lines.append("")
    lines.append("Single seed: treat Δ as direction, not effect size.")
    out = root / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
