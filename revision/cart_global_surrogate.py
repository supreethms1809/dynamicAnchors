"""CART as a textbook global surrogate: one tree fit to f_hat, every leaf a rule.

No per-class top-k, no abstention, no leaf ranking: fit
DecisionTreeClassifier(X_train, f_hat(X_train)) and read the whole tree as the
rule set. The tree answers every row (coverage 1), so its fidelity is plain
agreement with f_hat on D_test. Sizes: fixed depths, a fully grown tree
(sklearn defaults), and the depth with the best D_val agreement.

Also scores each tree on exactly the D_test rows RLDA/MADA decide (k = 1 and
k = 5 rule sets from `paper_final_valtb`), so the fidelity comparison is on the
same rows.

    python -m revision.cart_global_surrogate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import (  # noqa: E402
    _predictions, cell_classifier, load_seed_data, rebuild, with_classifier,
)

RES = REPO.parent / "results"
OUT = RES / "paper_final_cart_fixed" / "global_surrogate"
SEEDS = (42, 43, 44, 45, 46)
TAU_P = 0.90
DEPTHS = (2, 3, 4, 5, 8)
TUNE_DEPTHS = tuple(range(1, 16)) + (None,)
RL_CELLS = {
    ("rlda", 1): "paper_final_valtb/emp_tc0p10/results/ddpg",
    ("mada", 1): "paper_final_valtb/emp_tc0p10/results/maddpg",
    ("rlda", 5): "paper_final_valtb/k_sweep/k5",
    ("mada", 5): "paper_final_valtb/k_sweep/k5",
}


def leaf_conditions(tree) -> Dict[int, int]:
    """Distinct features on each root-to-leaf path (= box conditions of the leaf)."""
    t = tree.tree_
    out: Dict[int, int] = {}
    stack = [(0, frozenset())]
    while stack:
        node, feats = stack.pop()
        if t.children_left[node] == -1:
            out[node] = len(feats)
            continue
        f = feats | {int(t.feature[node])}
        stack.append((t.children_left[node], f))
        stack.append((t.children_right[node], f))
    return out


def score_tree(tree, X: np.ndarray, y_hat: np.ndarray, y: np.ndarray, n_classes: int) -> Dict[str, Any]:
    pred = tree.predict(X)
    leaf = tree.apply(X)
    conds = leaf_conditions(tree)
    agree = pred == y_hat
    # Leaf fidelity on the rows that land in it; a row "sits in a weak rule"
    # when its leaf agrees with f_hat on < tau_P of its D_test rows.
    weak = np.zeros(len(X), bool)
    for lf in np.unique(leaf):
        m = leaf == lf
        weak[m] = agree[m].mean() < TAU_P
    per_class = {}
    for c in range(n_classes):
        pc, fc = pred == c, y_hat == c
        per_class[c] = {
            "precision": float(agree[pc].mean()) if pc.any() else None,  # Fid of the class's rules
            "recall": float((pred[fc] == c).mean()) if fc.any() else None,
            "n_leaves": int(sum(1 for lf in conds if tree.tree_.value[lf].argmax() == c)),
        }
    return {
        "fid": float(agree.mean()),
        "acc": float((pred == y).mean()),
        "n_leaves": int(tree.get_n_leaves()),
        "depth": int(tree.get_depth()),
        "cond_per_leaf": float(np.mean(list(conds.values()))),
        "cond_per_row": float(np.mean([conds[lf] for lf in leaf])),
        "max_cond": int(max(conds.values())),
        "rows_in_weak_leaf": float(weak.mean()),
        "classes_with_leaf": int(len({int(tree.tree_.value[lf].argmax()) for lf in conds})),
        "per_class": per_class,
    }


def rl_rows(method: str, k: int, ds: str, seed: int, sd) -> Optional[Dict[str, Any]]:
    path = RES / RL_CELLS[(method, k)] / f"{ds}__{method}__seed{seed}__tp0p90__tc0p10.json"
    if not path.is_file():
        return None
    cell = json.loads(path.read_text())
    sd_c = with_classifier(sd, cell_classifier(cell))
    rb = rebuild(cell, sd_c)
    decided = np.logical_or.reduce([cr.test_union for cr in rb.classes.values()])
    g = cell["global_ruleset"]
    return {"cell": cell, "sd": sd_c, "decided": decided, "fid": g.get("global_fidelity"),
            "cov": g.get("coverage"), "eff": g.get("effectiveness"), "failures": len(rb.failures),
            "cov_rebuilt": float(decided.mean())}


def rules_text(tree, feature_names: List[str], max_leaves: int = 16) -> List[str]:
    t = tree.tree_
    rules = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            rules.append(f"IF {' AND '.join(conds) or 'TRUE'} THEN class {int(v.argmax())} "
                         f"(train n={int(t.n_node_samples[node])}, f_hat purity={v.max() / v.sum():.2f})")
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"])

    walk(0, [])
    return rules[:max_leaves]


def run_cell(ds: str, seed: int) -> Dict[str, Any]:
    from sklearn.tree import DecisionTreeClassifier

    ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                      / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
    sd = load_seed_data(ds, seed, Path(ref["extra"]["classifier_path"]))
    L = sd.loader
    X_tr = np.asarray(L.X_train, np.float32)
    y_hat_tr = _predictions(L, L.X_train_scaled)
    n_classes = int(L.n_classes)
    Xv, Xt = sd.val.X_orig, sd.test.X_orig

    def fit(depth):
        return DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, y_hat_tr)

    trees = {f"depth {d}": fit(d) for d in DEPTHS}
    trees["fully grown"] = fit(None)
    val_fid = {d: float((fit(d).predict(Xv) == sd.val.y_hat).mean()) for d in TUNE_DEPTHS}
    best = max(val_fid.values())
    d_star = next(d for d in TUNE_DEPTHS if val_fid[d] == best)  # shallowest at the best D_val agreement
    trees["val-tuned depth"] = fit(d_star)

    rl = {(m, k): rl_rows(m, k, ds, seed, sd) for (m, k) in RL_CELLS}
    out: Dict[str, Any] = {"dataset": ds, "seed": seed, "n_classes": n_classes,
                           "n_test": int(len(Xt)), "val_tuned_depth": d_star, "val_fid_by_depth":
                           {str(d): v for d, v in val_fid.items()}, "trees": {}, "rl": {}}
    for name, tr in trees.items():
        s = score_tree(tr, Xt, sd.test.y_hat, sd.test.y, n_classes)
        # Agreement on the rows each RL rule set decides, and on the rows it abstains on.
        pred = tr.predict(Xt)
        for (m, k), r in rl.items():
            if r is None:
                continue
            yh = r["sd"].test.y_hat
            d = r["decided"]
            s[f"fid_on_{m}_k{k}_decided"] = float((pred[d] == yh[d]).mean()) if d.any() else None
            s[f"fid_on_{m}_k{k}_abstained"] = float((pred[~d] == yh[~d]).mean()) if (~d).any() else None
        if name in ("depth 3", "val-tuned depth"):
            s["rules_preview"] = rules_text(tr, list(L.feature_names))
        out["trees"][name] = s
    for (m, k), r in rl.items():
        if r is not None:
            out["rl"][f"{m}_k{k}"] = {q: r[q] for q in ("fid", "cov", "eff", "failures", "cov_rebuilt")}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        for seed in SEEDS:
            path = OUT / f"{ds}__seed{seed}.json"
            if path.is_file():
                continue
            res = run_cell(ds, seed)
            path.write_text(json.dumps(res, indent=1, default=str))
            t = res["trees"]
            print(f"{ds} s{seed}: d3 {t['depth 3']['fid']:.3f} full {t['fully grown']['fid']:.3f} "
                  f"({t['fully grown']['n_leaves']} leaves) tuned d={res['val_tuned_depth']} "
                  f"{t['val-tuned depth']['fid']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
