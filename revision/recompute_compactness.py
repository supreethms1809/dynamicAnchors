"""Recompute baseline compactness in place, at the correct feature scale.

`active_feature_mask` defaults to a unit-space range of [0, 1]. CART and the
Anchors family build boxes in ORIGINAL feature units, so every stored
`mean_active_features` for those methods was measured against the wrong scale --
iris CART reported 0 active features for `petal length <= 2.45`, breast_cancer
reported all 16 as active because its features have small numeric magnitudes.

Compactness is a pure function of the stored bounds, so it is recomputed here
rather than by re-running the baselines. Re-running would re-sample the Anchors
explainers and perturb Fid/Cov -- numbers that are already published and are not
what this bug touched. `random_search` clips its boxes into [0, 1] and is left
alone; so are both RL arms, which are unit-space by construction.

    python -m revision.recompute_compactness [--roots ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import compactness_of_box  # noqa: E402

# Methods whose boxes live in original feature units.
ORIGINAL_UNIT_METHODS = {"cart", "greedy_anchors", "sp_anchors"}
FNAME = re.compile(r"^(?P<ds>.+?)__(?P<method>[a-z_]+)__seed(?P<seed>\d+)__tp")
SPARSITY = 0.95

_LOADER_CACHE: dict = {}


def feature_range(dataset: str, seed: int):
    """X_train min/max in original units — the span the baselines seed boxes from."""
    key = (dataset, seed)
    if key not in _LOADER_CACHE:
        from utils.dataset_factory import make_tabular_loader
        ld = make_tabular_loader(dataset, random_state=seed)
        ld.load_dataset()
        ld.preprocess_data()
        _LOADER_CACHE[key] = (
            np.min(ld.X_train, axis=0), np.max(ld.X_train, axis=0)
        )
    return _LOADER_CACHE[key]


def block_compactness(rules, fmin, fmax):
    rows = []
    for r in rules:
        lo, up = r.get("lower_bounds"), r.get("upper_bounds")
        if lo is None or up is None:
            continue
        rows.append(compactness_of_box(
            np.asarray(lo), np.asarray(up),
            sparsity_width_ratio=SPARSITY, feature_min=fmin, feature_max=fmax,
        ))
    if not rows:
        return None
    n_active = [r["n_active_features"] for r in rows]
    return {
        "n_rules": len(rows),
        "mean_active_features": float(np.mean(n_active)),
        "total_active_features": int(np.sum(n_active)),
        "n_features": rows[0]["n_features"],
        "sparsity_width_ratio": SPARSITY,
        "per_rule": rows,
    }


def summary(per_class):
    acts = [
        (b.get("compactness") or {}).get("mean_active_features")
        for b in per_class.values()
    ]
    acts = [a for a in acts if a is not None and np.isfinite(a)]
    return {
        "mean_rules_per_class": (
            float(np.mean([b.get("n_selected", b.get("k", 0)) for b in per_class.values()]))
            if per_class else 0.0
        ),
        "mean_active_features": float(np.mean(acts)) if acts else None,
        "sparsity_width_ratio": SPARSITY,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["runs/sweep_dnn", "runs/sweep_rf"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed = skipped = 0
    for root in args.roots:
        for path in sorted(glob.glob(os.path.join(root, "results", "*", "*.json"))):
            base = os.path.basename(path)
            m = FNAME.match(base)
            if not m or "__instances__" in base:
                continue
            if m["method"] not in ORIGINAL_UNIT_METHODS:
                skipped += 1
                continue
            with open(path) as fh:
                doc = json.load(fh)
            per_class = doc.get("per_class") or {}
            if not per_class:
                continue
            fmin, fmax = feature_range(m["ds"], int(m["seed"]))
            before = (doc.get("compactness") or {}).get("mean_active_features")
            for block in per_class.values():
                comp = block_compactness(block.get("selected_rules") or [], fmin, fmax)
                if comp is not None:
                    block["compactness"] = comp
            doc["compactness"] = summary(per_class)
            after = doc["compactness"]["mean_active_features"]
            doc.setdefault("notes", {})
            if isinstance(doc["notes"], dict):
                doc["notes"]["compactness_rescaled"] = (
                    "mean_active_features recomputed against the X_train feature "
                    "range; the stored value compared original-unit widths against "
                    "a unit-space [0,1] range."
                )
            flag = "" if before == after else "  <-- changed"
            print(f"{base:58s} {before} -> {after}{flag}")
            if not args.dry_run:
                with open(path, "w") as fh:
                    json.dump(doc, fh, indent=1)
            changed += 1

    print(f"\n{'would rewrite' if args.dry_run else 'rewrote'} {changed} artifacts; "
          f"left {skipped} unit-space artifacts untouched")


if __name__ == "__main__":
    main()
