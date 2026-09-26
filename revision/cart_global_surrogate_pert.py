"""The plain global surrogate (`cart_global_surrogate`) scored on perturbation fidelity.

Every leaf of the tree is a rule for its majority class; its perturbation
fidelity is Anchors' D(z|B) estimate (`dual_estimator_rescore.sample_anchors_conditional`:
draw a train row, patch only the coordinates that violate a predicate with a
train value inside it). The same estimator scores every selected rule of
RLDA/MADA trained on empirical Fid (emp_tc0p10) and on perturbation Fid
(pert_tc0p10), so all numbers come from one estimator.

Trees: fit to f_hat(D_train) ("plain"), or to D_train plus as many
recombined rows labelled by f_hat ("pert fit", what `run_cart` does under the
perturbed estimator). Depths 3 and 5, the depth with the best D_val agreement,
and fully grown.

Rule-set perturbation fidelity = mean over rules weighted by the D_test rows
each covers (for a tree: the leaf of each test row, averaged over rows).

    python -m revision.cart_global_surrogate_pert
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.dual_estimator_rescore import Scorer  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import cell_classifier, load_seed_data, rebuild, with_classifier  # noqa: E402
from utils.metrics import active_feature_mask  # noqa: E402

RES = REPO.parent / "results"
OUT = RES / "paper_final_cart_fixed" / "global_surrogate_pert"
SEEDS = (42, 43, 44, 45, 46)
TAU_P = 0.90
N_SAMPLES = 512
SPARSITY = 0.95
TUNE_DEPTHS = tuple(range(1, 16)) + (None,)
RL_CELLS = {
    "rlda_emp": ("paper_final_valtb/emp_tc0p10/results/ddpg", "rlda"),
    "mada_emp": ("paper_final_valtb/emp_tc0p10/results/maddpg", "mada"),
    "rlda_pert": ("paper_final_valtb/pert_tc0p10/results/ddpg", "rlda"),
    "mada_pert": ("paper_final_valtb/pert_tc0p10/results/maddpg", "mada"),
}
OPEN = np.finfo(np.float32).max


class FastPerturb:
    """`sample_anchors_conditional` with per-feature sorted pools (same distribution:
    a replacement is uniform over the train rows whose value lies in [lo, up])."""

    def __init__(self, sc: Scorer, space: str):
        self.sc, self.space = sc, space
        self.pool = sc.pools[space]
        self.sorted = np.sort(self.pool, axis=0)
        self.f_min, self.f_max = sc.feature_span[space]

    def active(self, lo, up) -> np.ndarray:
        return active_feature_mask(lo, up, sparsity_width_ratio=SPARSITY,
                                   feature_min=self.f_min, feature_max=self.f_max)

    def sample(self, lo, up, active, n, rng) -> np.ndarray:
        z = self.pool[rng.integers(0, len(self.pool), size=n)].copy()
        for j in np.flatnonzero(active):
            viol = (z[:, j] < lo[j]) | (z[:, j] > up[j])
            nv = int(viol.sum())
            if not nv:
                continue
            col = self.sorted[:, j]
            a, b = np.searchsorted(col, lo[j], "left"), np.searchsorted(col, up[j], "right")
            z[viol, j] = col[rng.integers(a, b, size=nv)] if b > a else rng.uniform(lo[j], up[j], size=nv)
        return z

    def fids(self, boxes: List[Tuple[np.ndarray, np.ndarray, int]], rng, n=N_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
        """Perturbation fidelity and number of active features of each (lo, up, cls) box."""
        Z, n_act = [], []
        for lo, up, _ in boxes:
            act = self.active(lo, up)
            n_act.append(int(act.sum()))
            Z.append(self.sample(lo, up, act, n, rng))
        pred = self.sc.predict(np.concatenate(Z), self.space).reshape(len(boxes), n)
        cls = np.array([c for _, _, c in boxes])[:, None]
        return (pred == cls).mean(axis=1), np.array(n_act)


def leaf_boxes(tree, d: int) -> Dict[int, Tuple[np.ndarray, np.ndarray, int]]:
    """Leaf id -> (lower, upper, class) in the tree's input units (float32-exact faces)."""
    t = tree.tree_
    out = {}
    stack = [(0, np.full(d, -OPEN, np.float32), np.full(d, OPEN, np.float32))]
    while stack:
        node, lo, up = stack.pop()
        if t.children_left[node] == -1:
            out[node] = (lo, up, int(t.value[node][0].argmax()))
            continue
        j, thr = t.feature[node], np.float32(t.threshold[node])
        # sklearn routes x <= thr left, comparing float32 x to a float64 threshold.
        left = thr if float(thr) <= t.threshold[node] else np.nextafter(thr, np.float32(-np.inf))
        ul, lr = up.copy(), lo.copy()
        ul[j] = min(ul[j], left)
        lr[j] = max(lr[j], np.nextafter(left, np.float32(np.inf)))
        stack.append((t.children_left[node], lo, ul))
        stack.append((t.children_right[node], lr, up))
    return out


def score_tree(tree, fp: FastPerturb, Xt: np.ndarray, y_hat: np.ndarray, rng) -> Dict[str, Any]:
    boxes = leaf_boxes(tree, Xt.shape[1])
    ids = list(boxes)
    fid, n_act = fp.fids([boxes[i] for i in ids], rng)
    pf = dict(zip(ids, fid))
    leaf = tree.apply(Xt)
    # The boxes must reproduce the tree's routing exactly.
    for i, (lo, up, _) in boxes.items():
        m = np.all((Xt >= lo) & (Xt <= up), axis=1)
        if not np.array_equal(m, leaf == i):
            raise AssertionError(f"leaf {i} box != tree routing")
    row_pf = np.array([pf[i] for i in leaf])
    pred = tree.predict(Xt)
    leaf_cls = {i: c for i, (_, _, c) in boxes.items()}
    per_class = {}
    for c in sorted(set(leaf_cls.values())):
        rows = pred == c
        fc = np.array([pf[i] for i in ids if leaf_cls[i] == c])
        per_class[c] = {
            "n_leaves": int(fc.size),
            "pert_fid_rows": float(row_pf[rows].mean()) if rows.any() else None,
            "pert_fid_rule_mean": float(fc.mean()),
            "rows_pert_ge_tau": float((row_pf[rows] >= TAU_P).mean()) if rows.any() else None,
            "emp_fid": float((y_hat[rows] == c).mean()) if rows.any() else None,
            "test_rows": int(rows.sum()),
        }
    return {
        "per_class": per_class,
        "emp_fid": float((pred == y_hat).mean()),
        "pert_fid_rows": float(row_pf.mean()),
        "pert_fid_rule_mean": float(fid.mean()),
        "rows_pert_ge_tau": float((row_pf >= TAU_P).mean()),
        "n_leaves": len(ids),
        "cond_per_leaf": float(n_act.mean()),
        "depth": int(tree.get_depth()),
    }


def score_rl(cell: Dict[str, Any], sd, fp: FastPerturb, rng) -> Dict[str, Any]:
    rb = rebuild(cell, sd)
    boxes, cover = [], []
    for block in cell["per_class"].values():
        cls = int((block.get("union") or {}).get("target_class"))
        cr = rb.classes[cls]
        rules = [r for r in block["selected_rules"] if r.get("lower_bounds") is not None]
        for r, tm in zip(rules, cr.test_masks):
            lo = np.clip(np.asarray(r["lower_bounds"], np.float32), 0, 1)
            up = np.clip(np.asarray(r["upper_bounds"], np.float32), 0, 1)
            boxes.append((lo, up, cls))
            cover.append(int(tm.sum()))
    fid, n_act = fp.fids(boxes, rng)
    w = np.asarray(cover, float)
    g = cell["global_ruleset"]
    cls_of = np.array([c for _, _, c in boxes])
    per_class = {}
    for c in sorted(set(cls_of.tolist())):
        m = cls_of == c
        wc = w[m]
        per_class[int(c)] = {
            "n_rules": int(m.sum()),
            "pert_fid_rows": float((fid[m] * wc).sum() / wc.sum()) if wc.sum() else None,
            "pert_fid_rule_mean": float(fid[m].mean()),
            "rows_pert_ge_tau": float(wc[fid[m] >= TAU_P].sum() / wc.sum()) if wc.sum() else None,
            "union_emp_fid": (cell["per_class"].get(f"class_{c}", {}).get("union") or {}).get("fidelity"),
        }
    return {
        "per_class": per_class,
        "emp_fid": g["global_fidelity"], "coverage": g["coverage"], "eff": g["effectiveness"],
        "pert_fid_rows": float((fid * w).sum() / w.sum()) if w.sum() else float("nan"),
        "pert_fid_rule_mean": float(fid.mean()),
        "rows_pert_ge_tau": float(w[fid >= TAU_P].sum() / w.sum()) if w.sum() else float("nan"),
        "n_rules": len(boxes), "cond_per_rule": float(n_act.mean()),
        "rebuild_failures": len(rb.failures),
    }


def run_cell(ds: str, seed: int) -> Dict[str, Any]:
    from sklearn.tree import DecisionTreeClassifier

    ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                      / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
    clf = ref["extra"]["classifier_path"]
    scorers: Dict[str, Scorer] = {clf: Scorer(ds, seed, clf)}
    sc = scorers[clf]
    sd = load_seed_data(ds, seed, Path(clf))
    L = sc.loader
    rng = np.random.default_rng(seed + 7)
    X_tr = sc.pools["original"]
    y_tr = sc.predict(X_tr, "original")
    # Recombined rows: every feature drawn independently from its train marginal.
    arng = np.random.default_rng(seed)
    Z = np.column_stack([X_tr[arng.integers(0, len(X_tr), size=len(X_tr)), j] for j in range(X_tr.shape[1])])
    fits = {"plain": (X_tr, y_tr),
            "pert fit": (np.vstack([X_tr, Z]), np.concatenate([y_tr, sc.predict(Z, "original")]))}
    Xv = np.asarray(L.X_val, np.float32)
    Xt = sc.evals["original"]
    fp_o = FastPerturb(sc, "original")
    out: Dict[str, Any] = {"dataset": ds, "seed": seed, "trees": {}, "rl": {}, "val_tuned_depth": {}}
    for fit_name, (X, y) in fits.items():
        def fit(depth):
            return DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X, y)
        val = {dd: float((fit(dd).predict(Xv) == sd.val.y_hat).mean()) for dd in TUNE_DEPTHS}
        d_star = next(dd for dd in TUNE_DEPTHS if val[dd] == max(val.values()))
        out["val_tuned_depth"][fit_name] = d_star
        for name, dd in (("depth 3", 3), ("depth 5", 5), ("val-tuned depth", d_star), ("fully grown", None)):
            out["trees"][f"{fit_name} / {name}"] = score_tree(fit(dd), fp_o, Xt, sc.y_hat_test["original"], rng)
    for key, (sub, method) in RL_CELLS.items():
        path = RES / sub / f"{ds}__{method}__seed{seed}__tp0p90__tc0p10.json"
        if not path.is_file():
            continue
        cell = json.loads(path.read_text())
        cclf = str(cell_classifier(cell) or clf)
        if cclf not in scorers:
            scorers[cclf] = Scorer(ds, seed, cclf)
        out["rl"][key] = score_rl(cell, with_classifier(sd, Path(cclf)), FastPerturb(scorers[cclf], "unit"), rng)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for seed in args.seeds:
            path = OUT / f"{ds}__seed{seed}.json"
            if path.is_file():
                continue
            res = run_cell(ds, seed)
            path.write_text(json.dumps(res, indent=1, default=str))
            t = res["trees"]
            print(f"{ds} s{seed}: plain d3 {t['plain / depth 3']['pert_fid_rows']:.3f} "
                  f"pert-fit d3 {t['pert fit / depth 3']['pert_fid_rows']:.3f} | "
                  + " ".join(f"{k} {v['pert_fid_rows']:.3f}" for k, v in res["rl"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
