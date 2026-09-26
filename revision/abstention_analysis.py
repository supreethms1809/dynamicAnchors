#!/usr/bin/env python
"""Is abstention selecting hard regions, or just losing coverage?

The claim under test
--------------------
The RL rule sets abstain on ~23% of test rows; CART, being a partition, must
answer everywhere. If CART's fidelity against f_hat is markedly *worse* on
exactly the rows the RL arms decline to cover, then abstention is picking out
genuinely hard regions and the rule set declines precisely where a partition is
unreliable. That is a measured functional property.

If CART is equally accurate there, abstention is lost coverage and nothing more,
and the claim should be dropped.

Method: rebuild every method's class-union masks from the stored per-rule bounds
(the same reconstruction as `cov_tau_class_gated`, which reproduces the stored
global metrics on 360/360 cells), take A = rows where no RL class union fires,
and score CART's rule-set prediction against f_hat on A versus on its complement.

    python -m revision.abstention_analysis --k 1 3
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.cov_tau_class_gated import (  # noqa: E402
    _classifier_path, _loader_for, _predictions, class_masks,
)
from revision.minority_class_analysis import cell_path  # noqa: E402
from revision.paper_stats import DATASETS, SEEDS  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402


def ruleset_prediction(masks: Dict[int, np.ndarray],
                       fids: Dict[int, float], n: int) -> np.ndarray:
    """Same predictor as `evaluate_ruleset_as_classifier`: fire, tie-break by union Fid."""
    pred = np.full(n, -1, dtype=int)
    if not masks:
        return pred
    classes = sorted(masks)
    stacked = np.stack([masks[c] for c in classes], axis=1)
    n_fired = stacked.sum(axis=1)
    single = n_fired == 1
    if single.any():
        pred[single] = np.array(classes)[stacked[single].argmax(axis=1)]
    conflict = n_fired >= 2
    if conflict.any():
        order = np.argsort([-fids.get(c, -np.inf) for c in classes])
        for row in np.flatnonzero(conflict):
            for j in order:
                if stacked[row, j]:
                    pred[row] = classes[j]
                    break
    return pred


def analyse(k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            clf = _classifier_path(dataset, seed)
            if clf is None:
                continue
            loader = _loader_for(dataset, seed)
            loader.classifier = loader.load_classifier(filepath=str(clf), device="cpu")
            X_unit = np.asarray(loader.X_test_unit, dtype=np.float32)
            X_orig = np.asarray(loader.X_test, dtype=np.float32)
            y_hat = _predictions(loader, loader.X_test_scaled)
            n = len(y_hat)

            cells: Dict[str, Any] = {}
            for m in ("rlda", "mada", "cart"):
                p = cell_path(dataset, m, seed, k)
                if p is None:
                    continue
                cells[m] = class_masks(json.loads(p.read_text()), X_unit, X_orig)
            if not {"rlda", "mada", "cart"} <= set(cells):
                continue

            cart_masks, cart_fids = cells["cart"]
            cart_pred = ruleset_prediction(cart_masks, cart_fids, n)
            cart_answers = cart_pred >= 0

            for arm in ("rlda", "mada"):
                masks, _ = cells[arm]
                fired = np.zeros(n, dtype=bool)
                for m_ in masks.values():
                    fired |= m_
                abstain = ~fired
                # CART fidelity on the two regions, over rows CART answers at all.
                rec: Dict[str, Any] = {
                    "dataset": dataset, "seed": seed, "arm": arm, "k": k,
                    "abstain_rate": float(abstain.mean()),
                    "n_abstain": int(abstain.sum()),
                }
                for name, sel in (("abstained", abstain), ("covered", fired)):
                    m_sel = sel & cart_answers
                    rec[f"cart_fid_{name}"] = (
                        float((cart_pred[m_sel] == y_hat[m_sel]).mean())
                        if m_sel.any() else None
                    )
                    rec[f"n_{name}"] = int(m_sel.sum())
                # Is the black box itself less accurate there? (confound check)
                y_true = np.asarray(loader.y_test)
                for name, sel in (("abstained", abstain), ("covered", fired)):
                    rec[f"clf_acc_{name}"] = (
                        float((y_hat[sel] == y_true[sel]).mean()) if sel.any() else None
                    )
                out.append(rec)
    return out


def report(k: int) -> List[str]:
    rows = analyse(k)
    L = [f"### k={k}: CART fidelity inside vs outside the RL abstention region", ""]
    L.append("`cart_fid_abstained` is CART's agreement with f̂ on exactly the rows the "
             "RL rule set declines to cover; `cart_fid_covered` is the same on the rows "
             "it does cover. `clf acc` is the black box's own accuracy there — the "
             "confound to rule out.")
    L.append("")
    L.append("| dataset | arm | abstain rate | CART Fid on abstained | CART Fid on covered "
             "| Δ | clf acc abstained | clf acc covered |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    per_ds: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        per_ds.setdefault((r["dataset"], r["arm"]), []).append(r)

    def _m(vals):
        v = [x for x in vals if x is not None]
        return st.mean(v) if v else None

    def _f(x, n=3):
        return "—" if x is None else f"{x:.{n}f}"

    deltas: Dict[str, List[float]] = {"rlda": [], "mada": []}
    ds_seen: Dict[str, List[str]] = {"rlda": [], "mada": []}
    for (dataset, arm), rs in sorted(per_ds.items(), key=lambda kv: DATASETS.index(kv[0][0])):
        a = _m([r["cart_fid_abstained"] for r in rs])
        c = _m([r["cart_fid_covered"] for r in rs])
        d = (a - c) if a is not None and c is not None else None
        L.append(
            f"| `{dataset}` | {arm} | {_m([r['abstain_rate'] for r in rs]):.3f} | "
            f"{_f(a)} | {_f(c)} | {_f(d)} | "
            f"{_f(_m([r['clf_acc_abstained'] for r in rs]))} | "
            f"{_f(_m([r['clf_acc_covered'] for r in rs]))} |"
        )
        if d is not None:
            deltas[arm].append(d)
            ds_seen[arm].append(dataset)
    L.append("")
    for arm in ("rlda", "mada"):
        if len(deltas[arm]) < 2:
            continue
        r = paired_wilcoxon(
            [x for x in deltas[arm]], [0.0] * len(deltas[arm])
        )
        neg = sum(1 for x in deltas[arm] if x < 0)
        L.append(
            f"- **{arm.upper()}**: mean Δ (CART on abstained − CART on covered) = "
            f"**{st.mean(deltas[arm]):+.3f}** over {len(deltas[arm])} datasets; "
            f"CART is worse on the abstained region in **{neg}/{len(deltas[arm])}**. "
            f"Wilcoxon vs 0: p={r.get('pvalue'):.4f}."
        )
    L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3])
    args = ap.parse_args()
    for k in args.k:
        for line in report(k):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
