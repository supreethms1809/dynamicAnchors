"""A CART surrogate that may abstain, compared with RLDA / MADA at matched coverage.

The paper's tree is a partition: it decides every row. The learned rule sets
abstain on part of the input space, so part of their Fid advantage could be
nothing more than declining the hard rows. This gives the tree the same option:

  1. Fit the tree exactly as `revision.baselines.run_cart` does (D_train rows,
     labelled by f_hat, `max_leaf_nodes = L`, random_state = seed).
  2. Score every leaf on D_val: Fid = share of its D_val rows where f_hat equals
     the leaf's class. Leaves with no D_val rows cannot be vouched for and go first.
  3. Drop leaves from least to most faithful. Stop at the last drop whose D_val
     coverage is still >= the RL arm's D_val coverage (the tree never covers less
     than the rule set on the selection split).
  4. Report that tree on D_test, next to the RL arm's D_test numbers
     (conflicts tie-broken on D_val, as in revision.rescore_val_tiebreak).

L is swept over {1, 2, 5, 10, 20} x n_classes; `L_val` is the size with the best
D_val Fid at the matched D_val coverage (ties -> smaller tree). Nothing on D_test
is used to choose anything.

Coverage here is GLOBAL coverage: the share of D_test rows the rule set decides
(1 - abstention). Class coverage (predicted basis) = share of rows with f_hat = c
covered by class c's rules; both are stored.

    python -m revision.abstaining_tree
    python -m revision.abstaining_tree --datasets iris --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import (  # noqa: E402
    RESULTS, SeedData, cell_classifier, global_result, load_seed_data, rebuild,
    result_path, with_classifier,
)

LEAF_MULTS = (1, 2, 5, 10, 20)
OUT = RESULTS / "abstaining_tree_5seed.json"


def _predict(loader, X_std) -> np.ndarray:
    from revision.cov_tau_class_gated import _predictions
    return _predictions(loader, X_std)


def leaf_path_features(tree) -> Dict[int, int]:
    """Number of distinct features tested on the path to each leaf."""
    t = tree.tree_
    out: Dict[int, int] = {}

    def walk(node: int, feats: frozenset):
        if t.children_left[node] == t.children_right[node]:
            out[node] = len(feats)
            return
        f = frozenset(feats | {int(t.feature[node])})
        walk(t.children_left[node], f)
        walk(t.children_right[node], f)

    walk(0, frozenset())
    return out


def fit_tree(sd: SeedData, n_leaves: int):
    from sklearn.tree import DecisionTreeClassifier

    loader = sd.loader
    y_hat_train = _predict(loader, loader.X_train_scaled)
    tree = DecisionTreeClassifier(max_leaf_nodes=n_leaves, random_state=sd.seed)
    tree.fit(loader.X_train, y_hat_train)  # original units, as in run_cart
    return tree


class Leaves:
    """A fitted tree's leaves as rules, with D_val / D_test membership (tree.apply,
    so the tree is a true partition: unlike `run_cart`'s boxes, rows outside the
    D_train range still land in a leaf)."""

    def __init__(self, tree, sd: SeedData):
        self.sd = sd
        self.cls = {int(n): int(np.argmax(tree.tree_.value[n][0]))
                    for n in np.flatnonzero(tree.tree_.children_left == -1)}
        self.depth = leaf_path_features(tree)
        self.lv = tree.apply(sd.val.X_orig)
        self.lt = tree.apply(sd.test.X_orig)
        self.pred_val = np.array([self.cls[int(l)] for l in self.lv])
        self.pred_test = np.array([self.cls[int(l)] for l in self.lt])
        stats = []
        for leaf, c in self.cls.items():
            m = self.lv == leaf
            n = int(m.sum())
            stats.append((float((sd.val.y_hat[m] == c).mean()) if n else -1.0, n, leaf))
        stats.sort()  # least faithful first; empty leaves (-1) first; then fewer rows
        self.order = [leaf for _, _, leaf in stats]

    def point(self, kept, n_dropped: int = 0) -> Dict[str, Any]:
        sd = self.sd
        kv = np.isin(self.lv, list(kept))
        kt = np.isin(self.lt, list(kept))
        per_class = {}
        for c in range(int(sd.loader.n_classes)):
            ref = sd.test.y_hat == c
            cm = kt & (self.pred_test == c)
            per_class[str(c)] = {
                "n_leaves": sum(1 for l in kept if self.cls[l] == c),
                "fidelity": float((sd.test.y_hat[cm] == c).mean()) if cm.any() else None,
                "class_coverage": float(cm[ref].mean()) if ref.any() else None,
            }
        return {
            "n_dropped": n_dropped,
            "n_kept": len(kept),
            "kept_leaves": sorted(int(l) for l in kept),
            "n_classes_with_rule": sum(1 for v in per_class.values() if v["n_leaves"]),
            "mean_path_features": float(np.mean([self.depth[l] for l in kept])),
            "max_path_features": int(max(self.depth[l] for l in kept)),
            "val_coverage": float(kv.mean()),
            "val_fidelity": float((self.pred_val[kv] == sd.val.y_hat[kv]).mean()) if kv.any() else None,
            "test_coverage": float(kt.mean()),
            "test_fidelity": float((self.pred_test[kt] == sd.test.y_hat[kt]).mean()) if kt.any() else None,
            "test_purity": float((self.pred_test[kt] == sd.test.y[kt]).mean()) if kt.any() else None,
            "per_class": per_class,
        }


def drop_curve(lv: Leaves, keep_every_class: bool = False) -> List[Dict[str, Any]]:
    """Operating points from dropping leaves in increasing D_val-Fid order.

    keep_every_class: never drop a class's last leaf (the RL arms always give
    every class a rule; a tree that drops a class entirely is not comparable).
    """
    seq = list(lv.order[:-1])
    if keep_every_class:
        left = {c: sum(1 for x in lv.cls.values() if x == c) for c in set(lv.cls.values())}
        seq = []
        for leaf in lv.order:
            if left[lv.cls[leaf]] > 1:
                seq.append(leaf)
                left[lv.cls[leaf]] -= 1
    curve = []
    for n_drop in range(len(seq) + 1):
        dropped = set(seq[:n_drop])
        curve.append(lv.point([l for l in lv.cls if l not in dropped], n_drop))
    return curve


def one_leaf_per_class(lv: Leaves) -> Dict[str, Any]:
    """The paper's CART selection at k = 1 on a tree of any size: per class, the
    leaf with the best D_val ranking score (Wilson LCB(Fid) x (1 + class cov))."""
    from utils.metrics import MIN_SUPPORT_DEFAULT, RANKING_SCORE_LCB_COVERAGE, evaluate_mask, ranking_score

    sd = lv.sd
    best: Dict[int, Any] = {}
    for leaf, c in lv.cls.items():
        m = evaluate_mask(y=sd.val.y, y_hat=sd.val.y_hat, mask=lv.lv == leaf, target_class=c,
                          class_conditional=True, min_support=MIN_SUPPORT_DEFAULT)
        sc = ranking_score(m.fidelity, m.coverage, formula=RANKING_SCORE_LCB_COVERAGE, n_covered=m.n_covered)
        if c not in best or sc > best[c][0]:
            best[c] = (sc, leaf)
    return lv.point([leaf for _, leaf in best.values()])


def matched_point(curve: List[Dict[str, Any]], target_val_cov: float) -> Dict[str, Any]:
    ok = [p for p in curve if p["val_coverage"] + 1e-12 >= target_val_cov]
    return max(ok, key=lambda p: p["n_dropped"]) if ok else curve[0]


def rl_numbers(cell, sd: SeedData) -> Dict[str, Any]:
    sd = with_classifier(sd, cell_classifier(cell))
    rb = rebuild(cell, sd)
    val = global_result(rb, sd, "val", "val")
    test = global_result(rb, sd, "test", "val")
    per_class = {}
    for c, cr in rb.classes.items():
        m, ref = cr.test_union, sd.test.y_hat == c
        per_class[str(c)] = {
            "n_rules": len(cr.test_masks),
            "fidelity": float((sd.test.y_hat[m] == c).mean()) if m.any() else None,
            "class_coverage": float(m[ref].mean()) if ref.any() else None,
        }
    return {
        "val_coverage": float(val.coverage),
        "val_fidelity": float(val.global_fidelity),
        "test_coverage": float(test.coverage),
        "test_fidelity": float(test.global_fidelity),
        "test_purity": float(test.global_purity),
        "n_rules": rb.n_rules,
        "rule_failures": rb.failures,
        "per_class": per_class,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--trees", nargs="+", default=["emp_tc0p10", "emp_tc0p20"])
    args = ap.parse_args()

    records: List[Dict[str, Any]] = []
    for ds in args.datasets:
        for seed in args.seeds:
            sd = load_seed_data(ds, seed)
            n_cls = int(sd.loader.n_classes)
            leaves = {m * n_cls: Leaves(fit_tree(sd, m * n_cls), sd) for m in LEAF_MULTS}
            curves = {L: drop_curve(lv) for L, lv in leaves.items()}
            curves_all = {L: drop_curve(lv, keep_every_class=True) for L, lv in leaves.items()}
            one_per = {L: one_leaf_per_class(lv) for L, lv in leaves.items()}
            for tree_name in args.trees:
                for arm in ("rlda", "mada"):
                    p = result_path(tree_name, arm, ds, seed)
                    if not p.is_file():
                        print(f"  missing {p}")
                        continue
                    cell = json.loads(p.read_text())
                    rl = rl_numbers(cell, sd)
                    rl["mean_active_features"] = (cell.get("compactness") or {}).get("mean_active_features")

                    def matched(cv):
                        return {str(L): {**matched_point(c, rl["val_coverage"]), "full_tree": {
                            k: c[0][k] for k in ("test_coverage", "test_fidelity", "n_kept")}}
                            for L, c in cv.items()}

                    def best_val(by_L, key):
                        L = max(by_L, key=lambda L: (key(by_L[L]), -int(L)))
                        return int(L), by_L[L]

                    by_L = matched(curves)
                    by_L_all = matched(curves_all)
                    one = {str(L): pt for L, pt in one_per.items()}
                    vfid = lambda p: p["val_fidelity"] or -1
                    veff = lambda p: (p["val_fidelity"] or 0) * p["val_coverage"]
                    L_val, at_L_val = best_val(by_L, vfid)
                    L_all, at_L_all = best_val(by_L_all, vfid)
                    L_one, at_L_one = best_val(one, veff)
                    records.append({
                        "dataset": ds, "seed": seed, "tree": tree_name, "arm": arm,
                        "n_classes": n_cls, "rl": rl,
                        # any leaves may go (can drop whole classes)
                        "tree_by_L": by_L, "L_val": L_val, "tree_at_L_val": at_L_val,
                        # every class keeps >= 1 leaf
                        "every_class_by_L": by_L_all, "every_class_L_val": L_all,
                        "every_class_at_L_val": at_L_all,
                        # one leaf per class, paper CART selection; L by D_val Eff
                        "one_per_class_by_L": one, "one_per_class_L_val": L_one,
                        "one_per_class_at_L_val": at_L_one,
                    })
            print(f"done {ds} {seed}", flush=True)
    OUT.write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} records to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
