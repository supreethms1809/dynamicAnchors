"""One set of trained policies, from local to global by changing selection parameters only.

The policies are the paper's emp_tc0p10 RLDA / MADA (no retraining). An
explanation is always a box from one rollout of the class-c policy started at a
point x (containment on, so x is inside). What changes is how many boxes are
produced and which are kept:

  starts N   rollouts per class, started at N D_train rows f_hat assigns to c
             (`rl_rollout_experiments --job pool`); N = "x*" is the local case:
             one rollout at the input being explained
  k          at most k boxes per class
  tau        a box is admissible only if its D_val Fid >= tau (and it covers at
             least S = max(3, 1% of D_val) D_val rows)

Selection is greedy on D_val: per class, repeatedly add the admissible box that
covers the most not-yet-covered D_val rows of class c, until k boxes or no gain.
The rule set is then read as a classifier on D_test (conflicts -> best D_val
union Fid; no box -> abstain), exactly like the paper's rule sets.

  local   N = x*, k = 1                     one box per input (per-instance explanation)
  class   the paper's rule sets (class-mode pool, k = 1, ranking score)
  global  N = 300, k = inf, tau = 0          boxes until the data is covered

Each point is compared with the plain surrogate of the same size (a tree with as
many leaves as the rule set has boxes, fit to f_hat(D_train), coverage 1) and,
at the local end, with classical Anchors on the same inputs.

    python -m revision.local_global_continuum            # sweep -> results/local_global/continuum.json
    python -m revision.local_global_continuum --report   # tables + figure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import _box, _predictions, load_seed_data  # noqa: E402
from utils.eval_harness import evaluate_ruleset_as_classifier  # noqa: E402
from utils.metrics import active_feature_mask  # noqa: E402

RES = REPO.parent / "results"
LG = RES / "local_global"
CF = RES / "containment_fix"
TAU_P = 0.90
STARTS = (1, 3, 10, 30, 100, 300)
KS = (1, 2, 5, 10, 20, 50, None)
TAUS = (0.90, 0.80, 0.0)
NAME = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot",
        "breast_cancer": "breast cancer", "uci_credit": "uci credit", "uci_adult": "uci adult"}


def select(cands: List[Dict[str, Any]], y_hat_val: np.ndarray, c: int, k, tau: float, S: int) -> List[int]:
    adm = [i for i, b in enumerate(cands) if b["val_n"] >= S and b["val_fid"] >= tau]
    target = y_hat_val == c
    covered = np.zeros_like(target)
    chosen: List[int] = []
    while adm and (k is None or len(chosen) < k):
        gains = [int((cands[i]["val_mask"] & target & ~covered).sum()) for i in adm]
        best = max(range(len(adm)), key=lambda j: (gains[j], cands[adm[j]]["val_fid"], -adm[j]))
        if gains[best] <= 0:
            break
        i = adm.pop(best)
        chosen.append(i)
        covered |= cands[i]["val_mask"]
    return chosen


def score_set(chosen: Dict[int, List[Dict[str, Any]]], sd) -> Dict[str, Any]:
    classes = [c for c, bs in chosen.items() if bs]
    if not classes:
        return {"fid": None, "coverage": 0.0, "eff": 0.0, "n_rules": 0, "cond_per_rule": None}
    tm = {c: np.logical_or.reduce([b["test_mask"] for b in chosen[c]]) for c in classes}
    vm = {c: np.logical_or.reduce([b["val_mask"] for b in chosen[c]]) for c in classes}
    tb = {c: float((sd.val.y_hat[vm[c]] == c).mean()) if vm[c].any() else -np.inf for c in classes}
    g = evaluate_ruleset_as_classifier(tm, tb, sd.test.y, sd.test.y_hat).to_dict()
    boxes = [b for c in classes for b in chosen[c]]
    return {"fid": g["global_fidelity"], "coverage": g["coverage"], "eff": g["effectiveness"],
            "n_rules": len(boxes), "cond_per_rule": float(np.mean([b["n_active"] for b in boxes])),
            "classes_with_rule": len(classes)}


def tree_same_size(sd, n_leaves: int, seed: int) -> float:
    from sklearn.tree import DecisionTreeClassifier
    L = sd.loader
    y_tr = _predictions(L, L.X_train_scaled)
    t = DecisionTreeClassifier(max_leaf_nodes=max(2, n_leaves), random_state=seed).fit(np.asarray(L.X_train, np.float32), y_tr)
    return float((t.predict(sd.test.X_orig) == sd.test.y_hat).mean())


def run_cell(ds: str, arm: str, seed: int) -> Dict[str, Any]:
    pool = json.loads((LG / "pool" / f"{ds}__{arm}__seed{seed}.json").read_text())
    clf = Path(pool["classifier_path"])
    if not clf.is_file():  # pools rolled out on spark name spark paths; use this machine's copy
        ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                          / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
        clf = Path(ref["extra"]["classifier_path"])
    sd = load_seed_data(ds, seed, clf)
    Xv, Xt = sd.val.X_unit, sd.test.X_unit
    S = max(3, math.ceil(0.01 * len(Xv)))
    by_cls: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for b in pool["boxes"]:
        lo, hi = np.asarray(b["lower"], np.float32), np.asarray(b["upper"], np.float32)
        key = (b["cls"], lo.tobytes(), hi.tobytes())
        if key in seen:  # identical boxes from different starts count once
            continue
        seen.add(key)
        c = int(b["cls"])
        vm, tm = _box(Xv, lo, hi), _box(Xt, lo, hi)
        by_cls[c].append({"order": len(by_cls[c]), "val_mask": vm, "test_mask": tm, "val_n": int(vm.sum()),
                          "val_fid": float((sd.val.y_hat[vm] == c).mean()) if vm.any() else 0.0,
                          "n_active": int(active_feature_mask(lo, hi, 0.95).sum())})
    # Pool order is the (random) order the starts were drawn in, so the first N
    # distinct boxes of a class are a random subset of its starts.
    out = {"dataset": ds, "arm": arm, "seed": seed, "S": S, "points": []}
    tree_cache: Dict[int, float] = {}
    for N in STARTS:
        cands = {c: bs[:N] for c, bs in by_cls.items()}
        for tau in TAUS:
            for k in KS:
                chosen = {c: [cands[c][i] for i in select(cands[c], sd.val.y_hat, c, k, tau, S)] for c in cands}
                s = score_set(chosen, sd)
                n = s["n_rules"]
                if n and n not in tree_cache:
                    tree_cache[n] = tree_same_size(sd, n, seed)
                out["points"].append({"starts": N, "tau": tau, "k": k, **s, "tree_same_size_fid": tree_cache.get(n)})
    # local end: one box per input (containment_eval rows, pi+) and Anchors on the same inputs
    rows = json.loads((CF / f"{ds}__{arm}__seed{seed}.json").read_text())["rows"]
    pi = [r["pi_contained"] for r in rows if r.get("pi_contained")]
    an = [r["anchors"] for r in rows if r.get("anchors")]
    tr = json.loads((CF / "tree" / f"{ds}__seed{seed}.json").read_text())["rows"]
    out["local"] = {
        m: {"emp_fid": float(np.nanmean([x["emp_fid"] for x in xs])), "pert_precision": float(np.nanmean([x["cond_fid"] for x in xs])),
            "coverage": float(np.mean([x["coverage"] for x in xs])), "cond": float(np.mean([x["n_active"] for x in xs])), "n": len(xs)}
        for m, xs in (("policy", pi), ("anchors", an), ("tree_d3_leaf", [x["depth 3"] for x in tr]))}
    # the paper's class-level rule set (class-mode pool, k = 1)
    algo = {"rlda": "ddpg", "mada": "maddpg"}[arm]
    cell = json.loads((RES / "paper_final_valtb/emp_tc0p10/results" / algo / f"{ds}__{arm}__seed{seed}__tp0p90__tc0p10.json").read_text())
    g = cell["global_ruleset"]
    n = sum(len(b.get("selected_rules") or []) for b in cell["per_class"].values())
    out["class_paper"] = {"fid": g["global_fidelity"], "coverage": g["coverage"], "eff": g["effectiveness"], "n_rules": n,
                          "cond_per_rule": cell["compactness"]["mean_active_features"],
                          "tree_same_size_fid": tree_cache.get(n) or tree_same_size(sd, n, seed)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        from revision.local_global_report import main as report
        return report()
    res = []
    for arm in ("rlda", "mada"):
        for ds in DATASETS:
            for seed in (42, 43, 44, 45, 46):
                if not (LG / "pool" / f"{ds}__{arm}__seed{seed}.json").is_file():
                    continue
                res.append(run_cell(ds, arm, seed))
                print(f"done {arm} {ds} {seed}", flush=True)
    (LG / "continuum.json").write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
