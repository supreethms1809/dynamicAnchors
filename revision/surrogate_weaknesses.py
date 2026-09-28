"""Do the known weaknesses of a global surrogate tree show up here, and does RL avoid them?

Re-scores stored rule sets only (emp tau_C = 0.10, k = 1, seeds 42-46). Methods:
RLDA, MADA (D_val tie-break), the paper's CART ("cart legacy"), the fixed CART
(`precision_constrained`), and the two Anchors baselines.

  reliability  Per explained D_test row: is the rule that explains it faithful?
               - share of decided rows whose rule has D_test Fid < tau_P
               - share of decided rows whose rule has Anchors-style conditional Fid
                 < tau_P (per-rule scores from trackb_rules_emp_tc0p10.json)
               - does the rule's D_val Fid (its "stated" precision) hold on D_test?
  size         Leaves a FULL-coverage tree needs before its D_test Fid reaches
               tau_P and RL's Fid (descriptive; reads D_test, selects nothing reported).
  stability    Across the 5 seeds (different split, classifier, training run), how
               much the class-c rule set moves: Jaccard of the rows it covers in one
               common pool (all rows of the dataset, original units) and Jaccard of
               the features it uses. Plus a CART-only bootstrap of D_train at seed 42.
  nominal      Share of rules that cut a nominal feature (>= 3 codes) into an
               arbitrary group of 2 .. n-1 codes by a threshold.

    python -m revision.surrogate_weaknesses
"""
from __future__ import annotations

import argparse
import collections
import itertools
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
    cell_classifier, load_seed_data, rebuild, with_classifier,
)
from utils.eval_harness import unit_to_original  # noqa: E402

TAU_P = 0.90
SEEDS = [42, 43, 44, 45, 46]
ROOT = REPO.parent / "results"
CELLS = {
    "rlda": ("paper_final_valtb/emp_tc0p10/results/ddpg", "rlda", "unit"),
    "mada": ("paper_final_valtb/emp_tc0p10/results/maddpg", "mada", "unit"),
    "cart legacy": ("paper_final_valtb/baselines_emp", "cart", "original"),
    "cart fixed": ("paper_final_cart_fixed/precision_constrained/baselines_emp", "cart", "original"),
    "greedy_anchors": ("paper_final_valtb/baselines_emp", "greedy_anchors", "original"),
    "sp_anchors": ("paper_final_valtb/baselines_emp", "sp_anchors", "original"),
}
TRACKB = ROOT / "paper_final_cart_fixed" / "trackb_rules_emp_tc0p10.json"
TRACKB_NAME = {"cart fixed": "cart precision_constrained"}
# ACS codes that are nominal but stored as integers (not flagged by the loader).
EXTRA_NOMINAL = {"folktables_income_CA_2018": ["COW", "MAR", "RELP", "SEX", "RAC1P", "OCCP", "POBP"]}
OUT = ROOT / "paper_final_cart_fixed" / "surrogate_weaknesses.json"


def cell_path(method: str, ds: str, seed: int) -> Path:
    sub, fm, _ = CELLS[method]
    return ROOT / sub / f"{ds}__{fm}__seed{seed}__tp0p90__tc0p10.json"


def trackb_index() -> Dict:
    """(ds, seed, method, class) -> per-rule conditional Fid, in stored rule order."""
    idx = collections.defaultdict(list)
    if TRACKB.is_file():
        for r in json.loads(TRACKB.read_text()):
            idx[(r["dataset"], r["seed"], r["method"], r["class"])].append(r["fid_anchors"])
    return idx


def original_boxes(cell, sd, space: str) -> Dict[int, List[tuple]]:
    """Per class, the selected boxes in original units, faces at the seed's own
    D_train range (or the unit cube) opened to +-inf so seeds are comparable."""
    L = sd.loader
    tr_min = np.min(L.X_train, axis=0).astype(np.float64)
    tr_max = np.max(L.X_train, axis=0).astype(np.float64)
    out = collections.defaultdict(list)
    for key, block in (cell.get("per_class") or {}).items():
        c = int(key.split("_")[-1])
        for r in block.get("selected_rules") or []:
            lo = np.asarray(r["lower_bounds"], dtype=np.float64)
            hi = np.asarray(r["upper_bounds"], dtype=np.float64)
            if space == "unit":
                open_lo, open_hi = lo <= 1e-6, hi >= 1 - 1e-6
                args = (L.X_min, L.X_range, np.asarray(L.scaler.mean_), np.asarray(L.scaler.scale_))
                lo, hi = unit_to_original(lo, *args), unit_to_original(hi, *args)
            else:
                span = np.maximum(tr_max - tr_min, 1e-12)
                open_lo = lo <= tr_min + 1e-6 * span
                open_hi = hi >= tr_max - 1e-6 * span
            lo = np.where(open_lo, -np.inf, lo)
            hi = np.where(open_hi, np.inf, hi)
            out[c].append((lo, hi))
    return out


def nominal_cuts(cell, sd, space: str, nominal: List[int]) -> List[bool]:
    """Per selected rule: does it group 2..n-1 codes of a nominal feature?"""
    L = sd.loader
    X = np.asarray(L.X_train_unit if space == "unit" else L.X_train, dtype=np.float64)
    flags = []
    for block in (cell.get("per_class") or {}).values():
        for r in block.get("selected_rules") or []:
            lo, hi = np.asarray(r["lower_bounds"]), np.asarray(r["upper_bounds"])
            bad = False
            for j in nominal:
                codes = np.unique(X[:, j])
                if len(codes) < 3:
                    continue
                k = int(((codes >= lo[j] - 1e-9) & (codes <= hi[j] + 1e-9)).sum())
                bad |= 2 <= k < len(codes)
            flags.append(bad)
    return flags


def full_tree_size(sd, targets: Dict[str, float]) -> Dict[str, Any]:
    """Smallest full-coverage tree (leaves, total path conditions) reaching each target Fid."""
    from sklearn.tree import DecisionTreeClassifier
    from revision.abstaining_tree import _predict, leaf_path_features

    L = sd.loader
    y_tr = _predict(L, L.X_train_scaled)
    curve = []
    for n in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        t = DecisionTreeClassifier(max_leaf_nodes=n, random_state=sd.seed).fit(L.X_train, y_tr)
        fid = float((t.predict(sd.test.X_orig) == sd.test.y_hat).mean())
        depth = leaf_path_features(t)
        curve.append({"max_leaf_nodes": n, "leaves": int(t.get_n_leaves()),
                      "conditions": int(sum(depth.values())), "fid": fid})
        if t.get_n_leaves() < n:  # tree stopped growing
            break
    out = {"curve": curve}
    for name, tgt in targets.items():
        hit = [p for p in curve if p["fid"] + 1e-12 >= tgt]
        out[name] = hit[0] if hit else None
    return out


def cart_bootstrap(sd, L_leaves: int, B: int = 10) -> Dict[str, float]:
    """Seed 42 only: refit the fixed-CART tree on bootstrap resamples of D_train
    (same classifier, same labels), pick each class's top D_val leaf as run_cart
    does, and measure how much the class-c rule moves (row Jaccard on D_test)."""
    from sklearn.tree import DecisionTreeClassifier
    from revision.abstaining_tree import Leaves, _predict, one_leaf_per_class

    L = sd.loader
    y_tr = _predict(L, L.X_train_scaled)
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(B):
        i = rng.integers(0, len(y_tr), size=len(y_tr))
        t = DecisionTreeClassifier(max_leaf_nodes=L_leaves, random_state=sd.seed).fit(L.X_train[i], y_tr[i])
        lv = Leaves(t, sd)
        pt = one_leaf_per_class(lv)
        cover.append({lv.cls[l]: lv.lt == l for l in pt["kept_leaves"]})
    js = []
    for a, b in itertools.combinations(cover, 2):
        for c in set(a) | set(b):
            if c in a and c in b:
                u = (a[c] | b[c]).sum()
                js.append((a[c] & b[c]).sum() / u if u else 1.0)
            else:
                js.append(0.0)
    return {"row_jaccard": float(np.mean(js)), "B": B, "max_leaf_nodes": L_leaves}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    tb = trackb_index()
    rec: Dict[str, Any] = {"reliability": [], "nominal": [], "stability": [], "size": [], "bootstrap": []}

    for ds in args.datasets:
        boxes = collections.defaultdict(dict)  # method -> seed -> class -> boxes
        pool = None
        for seed in SEEDS:
            base = load_seed_data(ds, seed)
            L = base.loader
            if pool is None:
                pool = np.vstack([L.X_train, L.X_val, L.X_test]).astype(np.float64)
                pool_min, pool_max = pool.min(0), pool.max(0)
            nominal = sorted(set(L.categorical_indices) | {
                L.feature_names.index(n) for n in EXTRA_NOMINAL.get(ds, []) if n in L.feature_names})
            rl_fid = {}
            for method, (_, _, space) in CELLS.items():
                p = cell_path(method, ds, seed)
                if not p.is_file():
                    continue
                cell = json.loads(p.read_text())
                sd = with_classifier(base, cell_classifier(cell))
                rb = rebuild(cell, sd)
                # -- reliability on D_test rows, per explaining rule
                yv, yt = sd.val.y_hat, sd.test.y_hat
                cond = []
                for c, cr in rb.classes.items():
                    cf = tb.get((ds, seed, TRACKB_NAME.get(method, method), c), [])
                    for i, (vm, tm) in enumerate(zip(cr.val_masks, cr.test_masks)):
                        cond.append((c, tm,
                                     float((yv[vm] == c).mean()) if vm.any() else np.nan,
                                     float((yt[tm] == c).mean()) if tm.any() else np.nan,
                                     cf[i] if i < len(cf) else np.nan))
                decided = np.zeros(len(yt), dtype=bool)
                best_test = np.full(len(yt), -1.0)
                best_cond = np.full(len(yt), -1.0)
                for c, tm, fv, ft, fc in cond:
                    decided |= tm
                    best_test[tm] = np.maximum(best_test[tm], np.nan_to_num(ft, nan=-1))
                    best_cond[tm] = np.maximum(best_cond[tm], np.nan_to_num(fc, nan=-1))
                n_dec = int(decided.sum())
                w = np.array([tm.sum() for _, tm, *_ in cond], dtype=float)
                fv = np.array([x[2] for x in cond]); ft = np.array([x[3] for x in cond])
                ok = np.isfinite(fv) & np.isfinite(ft) & (w > 0)
                rec["reliability"].append({
                    "dataset": ds, "seed": seed, "method": method, "n_decided": n_dec,
                    "coverage": n_dec / len(yt),
                    "rows_rule_below_tau_test": float((best_test[decided] < TAU_P).mean()) if n_dec else None,
                    "rows_rule_below_tau_cond": float((best_cond[decided] < TAU_P).mean()) if n_dec and tb else None,
                    "val_minus_test_fid_wmean": float(np.average(fv[ok] - ft[ok], weights=w[ok])) if ok.any() else None,
                    "abs_val_test_gap_wmean": float(np.average(np.abs(fv[ok] - ft[ok]), weights=w[ok])) if ok.any() else None,
                    "rules_claim_tau_miss_test": float(np.mean((fv[ok] >= TAU_P) & (ft[ok] < TAU_P - 0.05))) if ok.any() else None,
                    "n_rules": len(cond),
                })
                if method in ("rlda", "mada"):
                    g = cell["global_ruleset"]
                    rl_fid[method] = g.get("global_fidelity")
                # -- nominal cuts
                if nominal:
                    fl = nominal_cuts(cell, sd, space, nominal)
                    rec["nominal"].append({"dataset": ds, "seed": seed, "method": method,
                                           "n_rules": len(fl), "rules_with_arbitrary_group": int(sum(fl))})
                # -- boxes for cross-seed stability
                boxes[method][seed] = original_boxes(cell, sd, space)
            # -- full-tree size (descriptive)
            targets = {"tau_p": TAU_P, **{f"fid_{m}": f for m, f in rl_fid.items() if f is not None}}
            sz = full_tree_size(base, targets)
            rec["size"].append({"dataset": ds, "seed": seed, **sz,
                                "rl_fid": rl_fid})
            if seed == 42:
                p = cell_path("cart fixed", ds, seed)
                if p.is_file():
                    Lc = int(json.loads(p.read_text())["extra"]["max_leaf_nodes"])
                    rec["bootstrap"].append({"dataset": ds, **cart_bootstrap(base, Lc)})
            print(f"done {ds} {seed}", flush=True)

        # -- stability across seeds on the common pool
        P = pool
        for method, per_seed in boxes.items():
            cov = {s: {c: np.logical_or.reduce([np.all((P >= lo) & (P <= hi), axis=1) for lo, hi in bx])
                       for c, bx in cls.items()} for s, cls in per_seed.items()}
            feats = {s: {c: set(np.flatnonzero(np.logical_or.reduce(
                [(lo > pool_min) | (hi < pool_max) for lo, hi in bx])).tolist())
                for c, bx in cls.items()} for s, cls in per_seed.items()}
            rj, fj = [], []
            for a, b in itertools.combinations(sorted(cov), 2):
                for c in set(cov[a]) | set(cov[b]):
                    if c in cov[a] and c in cov[b]:
                        u = (cov[a][c] | cov[b][c]).sum()
                        rj.append((cov[a][c] & cov[b][c]).sum() / u if u else 1.0)
                        fu = feats[a][c] | feats[b][c]
                        fj.append(len(feats[a][c] & feats[b][c]) / len(fu) if fu else 1.0)
                    else:
                        rj.append(0.0)
                        fj.append(0.0)
            rec["stability"].append({"dataset": ds, "method": method, "n_seeds": len(cov),
                                     "row_jaccard": float(np.mean(rj)), "feature_jaccard": float(np.mean(fj))})
        OUT.write_text(json.dumps(rec, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
