"""
Tier-0 audit #1: reconcile rule-string metrics against box metrics.

The pipeline reports two families of numbers that are NOT measured on the same
region:

  * instP / instC  come from `env._current_metrics()` (inference.py:1156) and
    describe the FULL d-dimensional box: every feature is constrained.
  * clsP / clsC / unionP / unionC come from test_extracted_rules.py, which
    re-parses the PRINTED rule string. build_canonical_rule_key
    (inference.py:109) drops any feature whose interval spans the full range
    (lower <= eps and upper >= 1-eps, eps = max(1e-3, min_width/4)), so the
    printed rule is a RELAXATION of the box.

Dropping constraints can only grow the region, so for every anchor:
      coverage(rule) >= coverage(box)
and precision typically falls. This script measures that gap on stored runs so
we know how much of the "coverage collapse" in Table 1 is an artifact of
comparing a 30-dimensional box against the baseline's 1-2 predicate anchors.

Usage:
    python revision/audit_rule_vs_box.py --run_dir <comparison_results/.../<dataset>_<algo>_<ts>>
    python revision/audit_rule_vs_box.py --all      # sweep every stored run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))

from BenchMARL.test_extracted_rules import parse_rule, check_rule_satisfaction  # noqa: E402


def load_dataset(dataset_name: str, seed: int = 42):
    """Reproduce the pipeline's split/scaling so bounds and data share a space."""
    from BenchMARL.tabular_datasets import TabularDatasetLoader

    loader = TabularDatasetLoader(dataset_name=dataset_name, random_state=seed)
    loader.load_dataset()
    loader.preprocess_data()
    return loader


def _stack_full(loader):
    """Anchors were scored with use_full_dataset=True -> train+test, standardized space."""
    X = np.vstack([loader.X_train_scaled, loader.X_test_scaled]).astype(np.float32)
    y = np.concatenate([loader.y_train, loader.y_test])
    return X, y


def box_mask(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all((X >= lower) & (X <= upper), axis=1)


def audit_run(run_dir: Path, rules_filename: str, method: str) -> Optional[Dict[str, Any]]:
    rules_path = run_dir / rules_filename
    if not rules_path.exists():
        return None

    rules = json.loads(rules_path.read_text())
    dataset = rules.get("metadata", {}).get("dataset")
    if dataset is None:
        # Fall back to the directory naming convention: <dataset>_<algo>_<timestamp>
        dataset = "_".join(run_dir.name.split("_")[:-2])

    loader = load_dataset(dataset)
    X, y = _stack_full(loader)
    feature_names = list(loader.feature_names)
    n = X.shape[0]

    rows: List[Dict[str, Any]] = []
    for class_key, cdata in rules.get("per_class_results", {}).items():
        target_class = cdata.get("class")
        for anchor in cdata.get("all_anchors", []) or []:
            rule_str = anchor.get("rule")
            lo, up = anchor.get("lower_bounds"), anchor.get("upper_bounds")
            if not rule_str or lo is None or up is None:
                continue
            if rule_str == "any values (no tightened features)":
                continue

            lo = np.asarray(lo, dtype=np.float32)
            up = np.asarray(up, dtype=np.float32)

            bmask = box_mask(X, lo, up)
            conds = parse_rule(rule_str)
            rmask = (
                np.ones(n, dtype=bool)
                if len(conds) == 0
                else check_rule_satisfaction(X, feature_names, conds)
            )

            nb, nr = int(bmask.sum()), int(rmask.sum())
            rows.append(
                {
                    "class": target_class,
                    "n_box": nb,
                    "n_rule": nr,
                    "cov_box": nb / n,
                    "cov_rule": nr / n,
                    "n_active_box": int(np.sum(~((lo <= X.min(axis=0)) & (up >= X.max(axis=0))))),
                    "n_active_rule": len(conds),
                    # purity (ground-truth label agreement) on each region
                    "pur_box": float((y[bmask] == target_class).mean()) if nb else float("nan"),
                    "pur_rule": float((y[rmask] == target_class).mean()) if nr else float("nan"),
                    # what the pipeline reported for this anchor
                    "reported_instC": anchor.get("instance_coverage"),
                    "reported_instP": anchor.get("instance_precision"),
                    "rule_str": rule_str,
                }
            )

    if not rows:
        return None

    cb = np.array([r["cov_box"] for r in rows])
    cr = np.array([r["cov_rule"] for r in rows])
    rep = np.array([r["reported_instC"] for r in rows], dtype=float)
    ratio = np.where(cb > 0, cr / np.maximum(cb, 1e-12), np.nan)

    return {
        "run": run_dir.name,
        "dataset": dataset,
        "method": method,
        "n_anchors": len(rows),
        "n_samples": n,
        "mean_cov_box": float(cb.mean()),
        "mean_cov_rule": float(cr.mean()),
        "median_ratio": float(np.nanmedian(ratio)),
        "mean_active_box": float(np.mean([r["n_active_box"] for r in rows])),
        "mean_active_rule": float(np.mean([r["n_active_rule"] for r in rows])),
        "reported_matches_box": bool(np.allclose(rep, cb, atol=2e-3, equal_nan=True)),
        "max_abs_reported_minus_box": float(np.nanmax(np.abs(rep - cb))),
        "n_box_singleton_or_empty": int((np.array([r["n_box"] for r in rows]) <= 1).sum()),
        "rows": rows,
    }


def print_report(res: Dict[str, Any]) -> None:
    print(f"\n{'=' * 78}")
    print(f"{res['dataset']}  [{res['method']}]   {res['run']}")
    print(f"{'=' * 78}")
    print(f"  anchors={res['n_anchors']}  dataset rows={res['n_samples']}")
    print(f"  reported instC reproduces the BOX: {res['reported_matches_box']} "
          f"(max |diff| = {res['max_abs_reported_minus_box']:.5f})")
    print(f"  mean coverage  box  = {res['mean_cov_box']:.5f}")
    print(f"  mean coverage  rule = {res['mean_cov_rule']:.5f}")
    print(f"  median rule/box coverage ratio = {res['median_ratio']:.1f}x")
    print(f"  mean active features:  box = {res['mean_active_box']:.1f}   "
          f"printed rule = {res['mean_active_rule']:.1f}")
    print(f"  anchors whose BOX covers <=1 sample: "
          f"{res['n_box_singleton_or_empty']}/{res['n_anchors']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--all", action="store_true",
                    help="sweep comparison_results/all_datasets_0_{1,2,3}")
    ap.add_argument("--out", type=str, default="revision/audit_rule_vs_box.json")
    args = ap.parse_args()

    targets: List[Path] = []
    if args.all:
        for sweep in ("all_datasets_0_1", "all_datasets_0_2", "all_datasets_0_3"):
            targets += sorted((REPO / "comparison_results" / sweep).glob("*_maddpg_*"))
    elif args.run_dir:
        targets = [Path(args.run_dir).resolve()]
    else:
        ap.error("pass --run_dir or --all")

    results = []
    for run_dir in targets:
        for fname, method in (
            ("extracted_rulesmulti.json", "multi_agent"),
            ("extracted_rules_single_agent.json", "single_agent"),
        ):
            try:
                res = audit_run(run_dir, fname, method)
            except Exception as e:  # keep sweeping if one run is malformed
                print(f"  !! {run_dir.name} [{method}]: {type(e).__name__}: {e}")
                continue
            if res:
                print_report(res)
                results.append(res)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "rows"} for r in results]
    out.write_text(json.dumps({"summary": slim, "detail": results}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
