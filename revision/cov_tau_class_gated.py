#!/usr/bin/env python
"""Per-class-gated precision-constrained coverage, and how it compares to Eff.

The global gate (`track_a_cov_tau`) asks one question of a whole rule set: is
the aggregate Fid ≥ τ_P? If not, ALL of that rule set's coverage scores zero.
That is not the Anchors formulation -- Ribeiro constrains precision *per
anchor* -- and it makes the metric all-or-nothing: a single weak class zeroes
a dataset, and a method whose Fid lands at 0.898 loses everything.

This computes the per-class version instead:

    drop the class unions with union-Fid < τ_P, then score what is left
    with the SAME `evaluate_ruleset_as_classifier` used everywhere else.

So Cov_τ^class is the test-set mass covered by the rules that individually
earned the right to be shipped, and Fid_τ^class is the fidelity of that
surviving set. A class failing the floor costs you that class's coverage, not
the dataset.

Masks are rebuilt from the stored per-rule bounds. RL arms and `random_search`
store unit-space bounds; CART and the anchor family store original units
(`box_space` in `revision/baselines.py`). Every cell is self-checked: the
UNGATED recomputation must reproduce the stored coverage / fidelity /
conflict_rate, otherwise the reconstruction is wrong and the cell is refused.

    python -m revision.cov_tau_class_gated              # comparison table
    python -m revision.cov_tau_class_gated --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS, METHODS, SEEDS, PAPER, WYODOT  # noqa: E402
from utils.eval_harness import evaluate_ruleset_as_classifier  # noqa: E402
from utils.metrics import box_mask  # noqa: E402

TAU_P = 0.90
ORIGINAL_UNIT_METHODS = {"cart", "sp_anchors", "greedy_anchors"}


def result_path(dataset: str, method: str, seed: int) -> Optional[Path]:
    sub = {"rlda": "ddpg", "mada": "maddpg"}.get(method, "baselines")
    name = f"{dataset}__{method}__seed{seed}__tp0p90__tc0p10.json"
    for tree in (PAPER, WYODOT):
        p = tree / sub / name
        if p.is_file():
            return p
    return None


def _loader_for(dataset: str, seed: int):
    from utils.dataset_factory import make_tabular_loader

    loader = make_tabular_loader(dataset, random_state=seed)
    loader.load_dataset()
    loader.preprocess_data()
    return loader


def _classifier_path(dataset: str, seed: int) -> Optional[Path]:
    for tree in (PAPER, WYODOT):
        p = tree.parent / "classifiers" / f"{dataset}_seed{seed}.pth"
        if p.is_file():
            return p
    return None


def _predictions(loader, X_std: np.ndarray) -> np.ndarray:
    import torch
    from utils.networks import predict_proba_torch

    clf = loader.classifier
    if hasattr(clf, "eval"):
        clf.eval()
    with torch.no_grad():
        probs = predict_proba_torch(
            clf, torch.from_numpy(np.asarray(X_std, dtype=np.float32))
        ).cpu().numpy()
    return probs.argmax(axis=1)


def class_masks(
    cell: Dict[str, Any], X_unit: np.ndarray, X_orig: np.ndarray
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Rebuild each class union's test mask and its stored union fidelity."""
    method = cell["method"]
    X = X_orig if method in ORIGINAL_UNIT_METHODS else X_unit
    masks: Dict[int, np.ndarray] = {}
    fids: Dict[int, float] = {}
    for key, block in (cell.get("per_class") or {}).items():
        if not isinstance(block, dict):
            continue
        union = block.get("union") or {}
        cls = union.get("target_class")
        if cls is None:
            try:
                cls = int(str(key).split("_")[-1])
            except ValueError:
                continue
        m = np.zeros(len(X), dtype=bool)
        any_rule = False
        for rule in block.get("selected_rules") or []:
            lo, up = rule.get("lower_bounds"), rule.get("upper_bounds")
            if lo is None or up is None:
                continue
            m |= box_mask(
                X, np.asarray(lo, dtype=np.float32), np.asarray(up, dtype=np.float32)
            )
            any_rule = True
        if not any_rule:
            continue
        masks[int(cls)] = m
        f = union.get("fidelity")
        fids[int(cls)] = float(f) if f is not None and np.isfinite(f) else -np.inf
    return masks, fids


def _res_dict(res) -> Dict[str, float]:
    return {
        "coverage": float(res.coverage),
        "fidelity": float(res.global_fidelity),
        "purity": float(res.global_purity),
        "conflict_rate": float(res.conflict_rate),
        "n_decided": int(res.n_decided),
        "n_fid_agree": int(res.n_fid_agree),
    }


def score_cell(cell, X_unit, X_orig, y, y_hat, tau_p: float = TAU_P):
    """Ungated recomputation (self-check) + per-class-gated scores."""
    masks, fids = class_masks(cell, X_unit, X_orig)
    if not masks:
        return None
    full = _res_dict(evaluate_ruleset_as_classifier(masks, fids, y, y_hat))

    def _gate(t):
        keep = {c: m for c, m in masks.items() if fids.get(c, -np.inf) + 1e-12 >= t}
        if not keep:
            return {"coverage": 0.0, "fidelity": float("nan"), "purity": float("nan"),
                    "conflict_rate": 0.0, "n_decided": 0, "n_fid_agree": 0,
                    "n_classes_kept": 0}
        r = _res_dict(evaluate_ruleset_as_classifier(
            keep, {c: fids[c] for c in keep}, y, y_hat))
        r["n_classes_kept"] = len(keep)
        return r

    # Sensitivity sweep: the headline tau is only defensible if the ordering
    # it produces survives a plausible neighbourhood of tau.
    sweep = {f"{t:.2f}": _gate(t) for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95)}
    keep = {c: m for c, m in masks.items() if fids.get(c, -np.inf) + 1e-12 >= tau_p}
    gated = _gate(tau_p)
    return {
        "recomputed": full,
        "gated": gated,
        "sweep": sweep,
        "n_classes": len(masks),
        "n_classes_kept": len(keep),
        "class_fids": {int(c): float(v) for c, v in fids.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_p", type=float, default=TAU_P)
    ap.add_argument("--json", default=None, help="write raw per-cell results here")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="max |recomputed - stored| allowed on coverage/fidelity")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    n_check_fail = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            loader = _loader_for(dataset, seed)
            clf = _classifier_path(dataset, seed)
            if clf is None:
                print(f"  SKIP {dataset} seed{seed}: no classifier", file=sys.stderr)
                continue
            loader.classifier = loader.load_classifier(filepath=str(clf), device="cpu")
            X_unit = np.asarray(loader.X_test_unit, dtype=np.float32)
            X_orig = np.asarray(loader.X_test, dtype=np.float32)
            y = np.asarray(loader.y_test)
            y_hat = _predictions(loader, loader.X_test_scaled)

            for method in METHODS:
                p = result_path(dataset, method, seed)
                if p is None:
                    continue
                cell = json.loads(p.read_text())
                out = score_cell(cell, X_unit, X_orig, y, y_hat, args.tau_p)
                if out is None:
                    continue
                stored = cell.get("global_ruleset") or {}
                sc, sf = stored.get("coverage"), stored.get("global_fidelity")
                rc, rf = out["recomputed"]["coverage"], out["recomputed"]["fidelity"]
                ok = True
                if sc is not None and abs(float(sc) - rc) > args.tol:
                    ok = False
                if sf is not None and np.isfinite(sf) and np.isfinite(rf) \
                        and abs(float(sf) - rf) > args.tol:
                    ok = False
                if not ok:
                    n_check_fail += 1
                    print(f"  CHECK FAIL {dataset} {method} seed{seed}: "
                          f"stored cov={sc} fid={sf} vs recomputed cov={rc:.4f} "
                          f"fid={rf:.4f}", file=sys.stderr)
                rows.append({
                    "dataset": dataset, "method": method, "seed": seed,
                    "check_ok": ok, **out,
                    "stored_coverage": sc, "stored_fidelity": sf,
                })
    print(f"\nself-check: {len(rows) - n_check_fail}/{len(rows)} cells reproduced "
          f"the stored global metrics within {args.tol}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str))
        print(f"wrote {args.json}", file=sys.stderr)
    return 1 if n_check_fail else 0


if __name__ == "__main__":
    sys.exit(main())
