"""Abstention and same-row fidelity on paper_final (seeds 42-46, emp tau_C = 0.10, k = 1).

Reuses the helpers of revision/abstention_analysis.py and revision/cov_tau_class_gated.py,
but reads the paper_final result tree and the classifiers each seed was trained with:

  seeds 42-43: runs/paper_final/emp_tc0p10/classifiers/{ds}_seed{S}.pth
  seeds 44-46: ../results/paper_final_updated/emp_tc0p10/classifiers/{ds}_seed{S}.pth
               (falls back to ../results/paper_final/emp_tc0p10/classifiers)

For every (dataset, seed, arm in {rlda, mada}) it records, on the test split:
  - the fraction of rows the RL rule set decides,
  - RL fidelity on the rows both RL and CART decide,
  - CART fidelity on those same rows,
  - CART fidelity on rows RL abstains on (and CART decides),
  - classifier accuracy on RL-decided and RL-abstained rows,
  - a reproduction check: RL global fidelity / coverage rebuilt from the masks
    versus the values stored in the result JSON.

Output: ../results/paper_final/abstention_same_rows_5seed.json (one record per cell).
Run from the repo root:  python -m revision.abstention_paper_final
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.abstention_analysis import ruleset_prediction  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
RESULTS = REPO.parent / "results" / "paper_final"
UPDATED = REPO.parent / "results" / "paper_final_updated"
LOCAL = REPO / "runs" / "paper_final"
OUT = RESULTS / "abstention_same_rows_5seed.json"


def classifier_path(ds: str, seed: int) -> Path | None:
    name = f"{ds}_seed{seed}.pth"
    cands = ([LOCAL / "emp_tc0p10" / "classifiers" / name] if seed in (42, 43) else []) + [
        UPDATED / "emp_tc0p10" / "classifiers" / name,
        RESULTS / "emp_tc0p10" / "classifiers" / name,
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def cell(ds: str, method: str, seed: int) -> Path | None:
    name = f"{ds}__{method}__seed{seed}__tp0p90__tc0p10.json"
    sub = {"rlda": "emp_tc0p10/results/ddpg", "mada": "emp_tc0p10/results/maddpg"}.get(method, "baselines_emp")
    p = RESULTS / sub / name
    return p if p.is_file() else None


def frac(a: np.ndarray, b: np.ndarray, sel: np.ndarray):
    return float((a[sel] == b[sel]).mean()) if sel.any() else None


def main() -> int:
    records = []
    for ds in DATASETS:
        for seed in SEEDS:
            clf = classifier_path(ds, seed)
            paths = {m: cell(ds, m, seed) for m in ("rlda", "mada", "cart")}
            if clf is None or any(p is None for p in paths.values()):
                print(f"skip {ds} {seed}: clf={clf} cells={paths}")
                continue
            # Masks rebuilt with the float32 box-face repair and conflicts
            # tie-broken on D_val (revision/rescore_boxes.py); the plain rebuild
            # dropped every POBP = 6 row on folktables seed 45.
            from revision.rescore_boxes import load_seed_data, rebuild, tiebreak_fids
            sd = load_seed_data(ds, seed, clf)
            y_hat = sd.test.y_hat
            y_true = sd.test.y
            n = len(y_hat)
            js = {m: json.loads(p.read_text()) for m, p in paths.items()}

            def predictor(cell):
                rb = rebuild(cell, sd)
                masks = {c: cr.test_union for c, cr in rb.classes.items()}
                return ruleset_prediction(masks, tiebreak_fids(rb, sd, "val"), n)

            cart_pred = predictor(js["cart"])
            cart_dec = cart_pred >= 0
            for arm in ("rlda", "mada"):
                rl_pred = predictor(js[arm])
                dec = rl_pred >= 0
                both = dec & cart_dec
                g = js[arm]["global_ruleset"]
                rec = {
                    "dataset": ds, "seed": seed, "arm": arm, "n_test": n,
                    "decided_frac": float(dec.mean()),
                    "rl_fid_both": frac(rl_pred, y_hat, both),
                    "cart_fid_both": frac(cart_pred, y_hat, both),
                    "cart_fid_rl_covered": frac(cart_pred, y_hat, dec & cart_dec),
                    "cart_fid_rl_abstained": frac(cart_pred, y_hat, (~dec) & cart_dec),
                    "clf_acc_rl_covered": frac(y_hat, y_true, dec),
                    "clf_acc_rl_abstained": frac(y_hat, y_true, ~dec),
                    "n_both": int(both.sum()), "n_abstain": int((~dec).sum()),
                    "check_cov_rebuilt": float(dec.mean()),
                    "check_cov_stored": g.get("coverage"),
                    "check_fid_rebuilt": frac(rl_pred, y_hat, dec),
                    "check_fid_stored": g.get("global_fidelity"),  # D_test tie-break
                    "classifier_path": str(clf),
                }
                records.append(rec)
            print(f"done {ds} {seed}")
    OUT.write_text(json.dumps(records, indent=1))
    bad = [r for r in records if r["check_cov_stored"] is not None and abs(r["check_cov_rebuilt"] - r["check_cov_stored"]) > 1e-6]
    print(f"wrote {len(records)} records to {OUT}; {len(bad)} cells whose rebuilt coverage differs from the stored value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
