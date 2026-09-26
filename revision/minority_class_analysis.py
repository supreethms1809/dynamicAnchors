#!/usr/bin/env python
"""Does a per-class rule set represent minority classes a leaf-budgeted tree drops?

The claim under test
--------------------
The k-sweep showed CART is not beaten on fidelity or coverage, so the remaining
argument against it is the *rule form*: RLDA/MADA build one union per class by
construction, while CART spends a global leaf budget and can allocate none of it
to a rare class. If that is true and systematic, "the rule form is better" stops
being an assertion.

The decisive quantity is therefore **not** conditional Fid or Cov_c on the classes
a method happened to model — it is how often a class gets **no rule at all**, with
those cells scored Cov_c = 0 rather than dropped. Averaging over "classes that got
a rule" is exactly the selection effect that would manufacture a positive result.

Definitions
-----------
minority class   the rarest class of a dataset, by share of the full labelled table
                 (`docs/dataset_eda.json`); `--all-below-uniform` widens this to
                 every class with share < 1/K.
no rule          the class has no entry in `per_class`, or its entry has no
                 selected rule. Scored Cov_c = 0, Fid undefined.
Cov_c            P(x in B_c | y = c) on the test split, from the class union.
Fid              P(y_hat = c | x in B_c) on the test split.

Both k=1 (the paper lock) and k=3 (where CART Pareto-dominates on Fid/Cov) are
reported, because a larger leaf budget is exactly what would let CART cover the
rare class.

    python -m revision.minority_class_analysis
    python -m revision.minority_class_analysis --markdown > section.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS, SEEDS, PAPER, WYODOT  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402

METHODS = ["cart", "rlda", "mada", "sp_anchors", "greedy_anchors"]
K_SWEEP = REPO / "runs" / "k_sweep"
EDA = REPO / "docs" / "dataset_eda.json"


def cell_path(dataset: str, method: str, seed: int, k: int) -> Optional[Path]:
    name = f"{dataset}__{method}__seed{seed}__tp0p90__tc0p10.json"
    if k == 1:
        sub = {"rlda": "ddpg", "mada": "maddpg"}.get(method, "baselines")
        for tree in (PAPER, WYODOT):
            p = tree / sub / name
            if p.is_file():
                return p
        return None
    p = K_SWEEP / f"k{k}" / name
    return p if p.is_file() else None


def class_shares() -> Dict[str, List[float]]:
    rows = json.loads(EDA.read_text())
    return {r["dataset"]: list(r["class_fracs"]) for r in rows}


def per_class_scores(path: Path, n_classes: int) -> Dict[int, Dict[str, Any]]:
    """class -> {has_rule, cov_c, fid}. Absent classes are Cov_c = 0, not missing."""
    cell = json.loads(path.read_text())
    blocks = cell.get("per_class") or {}
    out: Dict[int, Dict[str, Any]] = {}
    for c in range(n_classes):
        block = blocks.get(f"class_{c}")
        rules = (block or {}).get("selected_rules") or []
        union = (block or {}).get("union") or {}
        if not block or not rules or union.get("coverage") is None:
            out[c] = {"has_rule": False, "cov_c": 0.0, "fid": None}
            continue
        out[c] = {
            "has_rule": True,
            "cov_c": float(union["coverage"]),
            "fid": (float(union["fidelity"])
                    if union.get("fidelity") is not None
                    and union["fidelity"] == union["fidelity"] else None),
        }
    return out


def collect(k: int) -> Dict[str, Any]:
    shares = class_shares()
    per_ds: Dict[str, Any] = {}
    for dataset in DATASETS:
        fr = shares.get(dataset)
        if not fr:
            continue
        n_classes = len(fr)
        minority = min(range(n_classes), key=lambda c: fr[c])
        below_uniform = [c for c in range(n_classes) if fr[c] < 1.0 / n_classes]
        rec: Dict[str, Any] = {
            "n_classes": n_classes, "fracs": fr,
            "minority": minority, "below_uniform": below_uniform,
            "methods": {},
        }
        for method in METHODS:
            seeds_data = []
            for seed in SEEDS:
                p = cell_path(dataset, method, seed, k)
                if p is None:
                    continue
                seeds_data.append(per_class_scores(p, n_classes))
            if not seeds_data:
                continue
            by_class = {}
            for c in range(n_classes):
                covs = [d[c]["cov_c"] for d in seeds_data]
                fids = [d[c]["fid"] for d in seeds_data if d[c]["fid"] is not None]
                n_rule = sum(1 for d in seeds_data if d[c]["has_rule"])
                by_class[c] = {
                    "cov_c_mean": st.mean(covs),
                    "cov_c_sd": st.stdev(covs) if len(covs) > 1 else 0.0,
                    "fid_mean": st.mean(fids) if fids else None,
                    "n_seeds": len(seeds_data),
                    "n_seeds_with_rule": n_rule,
                    "n_seeds_no_rule": len(seeds_data) - n_rule,
                }
            rec["methods"][method] = by_class
        per_ds[dataset] = rec
    return per_ds


def summarise(per_ds: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"no_rule_cells": {}, "minority_vectors": {}}
    for method in METHODS:
        total = missing = 0
        min_missing = 0
        for ds, rec in per_ds.items():
            by_class = rec["methods"].get(method)
            if not by_class:
                continue
            for c, v in by_class.items():
                total += v["n_seeds"]
                missing += v["n_seeds_no_rule"]
                if c == rec["minority"]:
                    min_missing += v["n_seeds_no_rule"]
        out["no_rule_cells"][method] = {
            "class_seed_cells": total, "no_rule": missing,
            "minority_no_rule": min_missing,
        }
        cov, fid, dss = [], [], []
        for ds in DATASETS:
            rec = per_ds.get(ds)
            if not rec or method not in rec["methods"]:
                continue
            v = rec["methods"][method][rec["minority"]]
            cov.append(v["cov_c_mean"])
            fid.append(v["fid_mean"])
            dss.append(ds)
        out["minority_vectors"][method] = {"datasets": dss, "cov_c": cov, "fid": fid}
    return out


def _fmt(x: Optional[float], n: int = 3) -> str:
    return "—" if x is None else f"{x:.{n}f}"


def report(k: int, markdown: bool = False) -> List[str]:
    per_ds = collect(k)
    summ = summarise(per_ds)
    L: List[str] = []
    add = L.append

    add(f"### k={k}: minority-class coverage and fidelity")
    add("")
    add("Minority = rarest class of each dataset. **A class with no rule scores "
        "Cov_c = 0**, it is not dropped — that is the whole comparison.")
    add("")
    add("| dataset | minority | share | CART Cov_c | CART Fid | RLDA Cov_c | RLDA Fid "
        "| MADA Cov_c | MADA Fid | no-rule seeds (C/R/M) |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for ds in DATASETS:
        rec = per_ds.get(ds)
        if not rec:
            continue
        c = rec["minority"]
        cells = []
        miss = []
        for m in ("cart", "rlda", "mada"):
            v = (rec["methods"].get(m) or {}).get(c)
            if v is None:
                cells += ["—", "—"]
                miss.append("—")
                continue
            cells += [f"{v['cov_c_mean']:.3f}", _fmt(v["fid_mean"])]
            miss.append(f"{v['n_seeds_no_rule']}/{v['n_seeds']}")
        add(f"| `{ds}` | class_{c} | {rec['fracs'][c]*100:.1f}% | "
            + " | ".join(cells) + " | " + "/".join(miss) + " |")

    add("")
    add(f"### k={k}: classes receiving no rule at all")
    add("")
    add("Every (dataset, class, seed) cell. A cell counts as *no rule* when the "
        "method produced no selected rule for that class.")
    add("")
    add("| method | class×seed cells | no rule | rate | of which minority class |")
    add("|---|---:|---:|---:|---:|")
    for m in METHODS:
        s = summ["no_rule_cells"].get(m)
        if not s:
            continue
        rate = s["no_rule"] / s["class_seed_cells"] if s["class_seed_cells"] else 0.0
        add(f"| `{m}` | {s['class_seed_cells']} | {s['no_rule']} | {rate*100:.1f}% | "
            f"{s['minority_no_rule']} |")

    add("")
    add(f"### k={k}: paired tests on the minority class (n=12 datasets)")
    add("")
    add("| contrast | metric | n | p | Kerby r | mean Δ |")
    add("|---|---|---:|---:|---:|---:|")
    mv = summ["minority_vectors"]
    for a, b in (("rlda", "cart"), ("mada", "cart"), ("mada", "rlda")):
        for key in ("cov_c", "fid"):
            va, vb = mv[a][key], mv[b][key]
            pairs = [(x, y) for x, y in zip(va, vb)
                     if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            xa = [x for x, _ in pairs]
            xb = [y for _, y in pairs]
            r = paired_wilcoxon(xa, xb)
            add(f"| `{a}` vs `{b}` | {'Cov_c' if key=='cov_c' else 'Fid'} | "
                f"{len(pairs)} | {_fmt(r.get('pvalue'), 4)} | "
                f"{_fmt(r.get('effect_size_rank_biserial'))} | "
                f"{_fmt(r.get('mean_diff'))} |")
    add("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3])
    args = ap.parse_args()
    for k in args.k:
        for line in report(k, args.markdown):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
