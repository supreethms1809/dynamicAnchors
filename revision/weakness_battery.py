"""The known weaknesses of global surrogates, run on the plain surrogate AND on RL.

Methods (class level, 12 datasets x seeds 42-46, emp tau_C = 0.10, k = 1):
  tree_d3     DecisionTreeClassifier(max_depth=3) fit to f_hat(D_train); every leaf a rule
  tree_tuned  same, depth with the best D_val agreement (as in `cart_global_surrogate`)
  rlda, mada  the paper's rule sets (`paper_final_valtb`, conflicts tie-broken on D_val)

Every method is reduced to the same object: a list of boxes with a class, and for
each D_test row the label it gives (or abstain) and the rule that explains it
(tree: the leaf; RL: among the fired rules of the decided class, the one with the
best D_val Fid). The tests:

  W1  size             rules, conditions per rule, total conditions
  W2  hidden errors    share of explained rows whose rule has D_test Fid < tau_P;
                       worst class; 10th percentile of the explaining rule's Fid
  W3  minority class   f_hat's rarest D_test class: has a rule? recall, precision
  W4  stability        across seeds (different split, f_hat, fit): per class, Jaccard
                       of the rows its rules cover in one common pool (all rows of the
                       dataset) and of the features they use
  W5  Rashomon         (tree) 20 depth-3 trees with feature subsampling within 0.01
                       D_val agreement of the best: how different are they?
  W6  proxies          share of explained rows whose rule uses a feature that has a
                       |Spearman rho| >= 0.8 partner the rule does not use
  W8  sensitive        (adult, folktables, sick) rows whose f_hat label flips when the
                       sensitive feature changes; does their explanation mention it?
                       and how much surrogate Fid is lost by hiding the feature
  W10 reliability      stated (D_val) vs realised (D_test) Fid of the explaining rule
  W12 hyperparameters  spread of Fid / coverage over each method's own knobs
  W13 nominal cuts     rules that group 2..n-1 codes of a nominal feature
  W14 data             (tree) Fid when fit on 5-100% of D_train

W7 (off-manifold) and W9 (wrong class) come from `cart_global_surrogate_pert` and
`tree_instance_eval`; the instance-level robustness test from
`rl_rollout_experiments --job robust`. All are assembled by `--report`.

    python -m revision.weakness_battery            # compute (writes one JSON per cell)
    python -m revision.weakness_battery --report   # tables
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.cart_global_surrogate_pert import leaf_boxes  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import (  # noqa: E402
    _predictions, cell_classifier, load_seed_data, rebuild, with_classifier,
)
from revision.surrogate_weaknesses import EXTRA_NOMINAL, original_boxes  # noqa: E402
from utils.metrics import active_feature_mask  # noqa: E402

RES = REPO.parent / "results"
OUT = RES / "weakness_battery"
SEEDS = (42, 43, 44, 45, 46)
TAU_P = 0.90
SENSITIVE = {"uci_adult": ["sex", "race"], "folktables_income_CA_2018": ["SEX", "RAC1P"], "sick": ["sex"]}
RL_DIRS = {"rlda": "paper_final_valtb/emp_tc0p10/results/ddpg", "mada": "paper_final_valtb/emp_tc0p10/results/maddpg"}
METHODS = ("tree_d3", "tree_tuned", "rlda", "mada")


@dataclass
class Rule:
    cls: int
    lo: np.ndarray            # original units, +-inf on free faces
    hi: np.ndarray
    active: frozenset         # constrained features
    val_mask: np.ndarray
    test_mask: np.ndarray
    stated: float = float("nan")      # D_val Fid
    realised: float = float("nan")    # D_test Fid
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explainer:
    rules: List[Rule]
    pred: np.ndarray          # D_test label, -1 = abstain
    expl: np.ndarray          # index of the explaining rule, -1 = abstain


def _fid(mask, y_hat, c):
    return float((y_hat[mask] == c).mean()) if mask.any() else float("nan")


def tree_explainer(tree, sd, feat_span) -> Explainer:
    L = sd.loader
    Xv, Xt = sd.val.X_orig, sd.test.X_orig
    boxes = leaf_boxes(tree, Xt.shape[1])
    lv, lt = tree.apply(Xv), tree.apply(Xt)
    ids = list(boxes)
    pos = {leaf: i for i, leaf in enumerate(ids)}
    t = tree.tree_
    rules = []
    for leaf in ids:
        lo, hi, c = boxes[leaf]
        lo = np.where(lo <= -np.finfo(np.float32).max, -np.inf, lo).astype(np.float64)
        hi = np.where(hi >= np.finfo(np.float32).max, np.inf, hi).astype(np.float64)
        act = frozenset(np.flatnonzero((lo > feat_span[0]) | (hi < feat_span[1])).tolist())
        vm, tm = lv == leaf, lt == leaf
        v = t.value[leaf][0]
        rules.append(Rule(c, lo, hi, act, vm, tm, _fid(vm, sd.val.y_hat, c), _fid(tm, sd.test.y_hat, c),
                          {"train_purity": float(v.max() / v.sum()), "train_n": int(t.n_node_samples[leaf])}))
    pred = tree.predict(Xt).astype(int)
    expl = np.array([pos[leaf] for leaf in lt])
    return Explainer(rules, pred, expl)


def rl_explainer(cell, sd) -> Explainer:
    rb = rebuild(cell, sd)
    ob = original_boxes(cell, sd, "unit")
    rules = []
    for c, cr in sorted(rb.classes.items()):
        blk = cell["per_class"][f"class_{c}"]
        units = [r for r in blk["selected_rules"] if r.get("lower_bounds") is not None]
        for i, (vm, tm) in enumerate(zip(cr.val_masks, cr.test_masks)):
            lo_u = np.clip(np.asarray(units[i]["lower_bounds"], float), 0, 1)
            hi_u = np.clip(np.asarray(units[i]["upper_bounds"], float), 0, 1)
            act = frozenset(np.flatnonzero(active_feature_mask(lo_u, hi_u, 0.95)).tolist())
            lo, hi = ob[c][i]
            rules.append(Rule(c, lo, hi, act, vm, tm, _fid(vm, sd.val.y_hat, c), _fid(tm, sd.test.y_hat, c)))
    # Rule set as a classifier: one class fired -> it; several -> best D_val union Fid.
    classes = sorted({r.cls for r in rules})
    union_t = {c: np.logical_or.reduce([r.test_mask for r in rules if r.cls == c]) for c in classes}
    union_v = {c: np.logical_or.reduce([r.val_mask for r in rules if r.cls == c]) for c in classes}
    tb = {c: _fid(union_v[c], sd.val.y_hat, c) for c in classes}
    n = len(sd.test.y_hat)
    pred = np.full(n, -1)
    expl = np.full(n, -1)
    for i in range(n):
        fired = [c for c in classes if union_t[c][i]]
        if not fired:
            continue
        c = max(fired, key=lambda k: (tb[k] if np.isfinite(tb[k]) else -1))
        pred[i] = c
        cand = [j for j, r in enumerate(rules) if r.cls == c and r.test_mask[i]]
        expl[i] = max(cand, key=lambda j: (rules[j].stated if np.isfinite(rules[j].stated) else -1))
    g = cell["global_ruleset"]
    dec = pred >= 0
    agree = int((pred[dec] == sd.test.y_hat[dec]).sum())
    # Coverage must match exactly; the Fid numerator may drift by a row or two
    # where f_hat differs between the training machine and this one (housing s44).
    if abs(dec.mean() - g["coverage"]) > 1e-9 or abs(agree - g["n_fid_agree"]) > 2:
        raise AssertionError(f"RL rebuild mismatch: agree {agree} vs {g['n_fid_agree']}, cov {dec.mean()} vs {g['coverage']}")
    return Explainer(rules, pred, expl)


def nominal_flags(rules: List[Rule], X: np.ndarray, nominal: List[int]) -> List[bool]:
    flags = []
    for r in rules:
        bad = False
        for j in nominal:
            codes = np.unique(X[:, j])
            if len(codes) < 3:
                continue
            k = int(((codes >= r.lo[j] - 1e-9) & (codes <= r.hi[j] + 1e-9)).sum())
            bad |= 2 <= k < len(codes)
        flags.append(bad)
    return flags


def core_metrics(ex: Explainer, sd, proxies: Dict[int, set], nominal: List[int], X_nom: np.ndarray) -> Dict[str, Any]:
    y_hat = sd.test.y_hat
    dec = ex.pred >= 0
    rules = ex.rules
    row_fid = np.array([rules[j].realised if j >= 0 else np.nan for j in ex.expl])
    out: Dict[str, Any] = {
        "fid": float((ex.pred[dec] == y_hat[dec]).mean()) if dec.any() else None,
        "coverage": float(dec.mean()),
        "n_rules": len(rules),
        "cond_per_rule": float(np.mean([len(r.active) for r in rules])),
        "total_conditions": int(sum(len(r.active) for r in rules)),
        # W2
        "rows_rule_below_tau": float((row_fid[dec] < TAU_P).mean()) if dec.any() else None,
        "row_rule_fid_p10": float(np.nanpercentile(row_fid[dec], 10)) if dec.any() else None,
        # W10
        "stated_realised_abs_gap": None, "overclaim_rate": None,
    }
    st = np.array([rules[j].stated if j >= 0 else np.nan for j in ex.expl])
    ok = dec & np.isfinite(st) & np.isfinite(row_fid)
    if ok.any():
        out["stated_realised_abs_gap"] = float(np.abs(st[ok] - row_fid[ok]).mean())
        out["overclaim_rate"] = float(((st[ok] >= TAU_P) & (row_fid[ok] < TAU_P - 0.05)).mean())
    if rules and "train_purity" in rules[0].extra:
        tp = np.array([rules[j].extra["train_purity"] for j in ex.expl])
        out["train_purity_abs_gap"] = float(np.abs(tp - row_fid).mean())
        out["train_purity_overclaim_rate"] = float(((tp >= TAU_P) & (row_fid < TAU_P - 0.05)).mean())
    # W2 / W3 per class
    per_class = {}
    counts = {int(c): int((y_hat == c).sum()) for c in np.unique(y_hat)}
    for c in sorted(counts):
        lab = ex.pred == c
        per_class[c] = {
            "n_rows": counts[c],
            "has_rule": any(r.cls == c for r in rules),
            "precision": float((y_hat[lab] == c).mean()) if lab.any() else None,
            "recall": float((ex.pred[y_hat == c] == c).mean()) if counts[c] else None,
        }
    out["per_class"] = per_class
    prec = [v["precision"] for v in per_class.values() if v["precision"] is not None]
    out["worst_class_precision"] = float(min(prec)) if prec else None
    out["classes_without_rule"] = int(sum(not v["has_rule"] for v in per_class.values()))
    m = min(counts, key=lambda c: counts[c])
    out["minority"] = {"class": m, "share": counts[m] / len(y_hat), **per_class[m]}
    # W6
    prox = [bool(any(proxies.get(j, set()) - set(r.active) for j in r.active)) for r in rules]
    out["rows_rule_has_unused_proxy"] = float(np.mean([prox[j] for j in ex.expl[dec]])) if dec.any() else None
    out["rules_with_unused_proxy"] = float(np.mean(prox)) if prox else None
    # W13
    if nominal:
        nf = nominal_flags(rules, X_nom, nominal)
        out["nominal_rules_grouping_codes"] = float(np.mean(nf)) if nf else None
        out["nominal_rows_grouping_codes"] = float(np.mean([nf[j] for j in ex.expl[dec]])) if dec.any() else None
    return out


def sensitive_metrics(ex: Explainer, sd, feats: List[str]) -> Dict[str, Any]:
    L = sd.loader
    names = list(L.feature_names)
    Xo = np.asarray(L.X_test, np.float64)
    out = {}
    dec = ex.pred >= 0
    for f in feats:
        j = names.index(f)
        codes = np.unique(np.asarray(L.X_train, np.float64)[:, j])
        flips = np.zeros(len(Xo), bool)
        for v in codes:
            Z = Xo.copy()
            Z[:, j] = v
            flips |= _predictions(L, L.scaler.transform(Z)) != sd.test.y_hat
        mention = np.array([j in ex.rules[k].active if k >= 0 else False for k in ex.expl])
        dep = flips & dec
        out[f] = {
            "rows_fhat_depends": float(flips.mean()),
            "dependent_rows_explained": int(dep.sum()),
            "dependent_rows_whose_rule_mentions": float(mention[dep].mean()) if dep.any() else None,
            "explained_rows_whose_rule_mentions": float(mention[dec].mean()) if dec.any() else None,
            "rules_mentioning": float(np.mean([j in r.active for r in ex.rules])),
        }
    return out


def stability_boxes(ex: Explainer) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    d = defaultdict(list)
    for r in ex.rules:
        d[r.cls].append((r.lo, r.hi))
    return d


def run_dataset(ds: str) -> List[Dict[str, Any]]:
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import spearmanr

    recs, boxes = [], defaultdict(dict)
    pool = None
    for seed in SEEDS:
        ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                          / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
        base = load_seed_data(ds, seed, Path(ref["extra"]["classifier_path"]))
        L = base.loader
        X_tr = np.asarray(L.X_train, np.float32)
        y_tr = _predictions(L, L.X_train_scaled)
        span = (X_tr.min(0).astype(np.float64), X_tr.max(0).astype(np.float64))
        if pool is None:
            pool = np.vstack([L.X_train, L.X_val, L.X_test]).astype(np.float64)
        names = list(L.feature_names)
        nominal = sorted(set(getattr(L, "categorical_indices", None) or []) |
                         {names.index(n) for n in EXTRA_NOMINAL.get(ds, []) if n in names})
        rho = np.atleast_2d(spearmanr(X_tr).correlation) if X_tr.shape[1] > 1 else np.ones((1, 1))
        rho = np.nan_to_num(rho)
        proxies = {j: {k for k in range(X_tr.shape[1]) if k != j and abs(rho[j, k]) >= 0.8} for j in range(X_tr.shape[1])}
        gs = json.loads((RES / "paper_final_cart_fixed/global_surrogate" / f"{ds}__seed{seed}.json").read_text())
        d_star = None if gs["val_tuned_depth"] in (None, "None") else int(gs["val_tuned_depth"])

        def fit(depth, X=X_tr, y=y_tr, **kw):
            kw.setdefault("random_state", seed)
            return DecisionTreeClassifier(max_depth=depth, **kw).fit(X, y)

        exps = {"tree_d3": (fit(3), base), "tree_tuned": (fit(d_star), base)}
        for m, sub in RL_DIRS.items():
            cell = json.loads((RES / sub / f"{ds}__{m}__seed{seed}__tp0p90__tc0p10.json").read_text())
            sd = with_classifier(base, cell_classifier(cell))
            exps[m] = (cell, sd)
        for m, (obj, sd) in exps.items():
            ex = tree_explainer(obj, sd, span) if m.startswith("tree") else rl_explainer(obj, sd)
            rec = {"dataset": ds, "seed": seed, "method": m, **core_metrics(ex, sd, proxies, nominal, X_tr)}
            if ds in SENSITIVE:
                rec["sensitive"] = sensitive_metrics(ex, sd, SENSITIVE[ds])
            recs.append(rec)
            boxes[m][seed] = {c: [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in bx]
                              for c, bx in stability_boxes(ex).items()}
        # ---- tree-only tests
        Xv = base.val.X_orig
        tr = {"dataset": ds, "seed": seed, "method": "tree_only"}
        # W5 Rashomon
        cands = [fit(3)] + [fit(3, max_features=0.7, random_state=b) for b in range(20)]
        vf = np.array([(t.predict(Xv) == base.val.y_hat).mean() for t in cands])
        kept = [t for t, v in zip(cands, vf) if v >= vf.max() - 0.01]
        feats = [frozenset(int(f) for f in t.tree_.feature if f >= 0) for t in kept]
        pairs = list(itertools.combinations(range(len(kept)), 2))
        path_feats = []
        for t in kept:
            ex = tree_explainer(t, base, span)
            path_feats.append([ex.rules[k].active for k in ex.expl])
        tr["rashomon"] = {
            "n_candidates": len(cands), "n_within_0p01": len(kept),
            "val_fid_best": float(vf.max()),
            "test_fid_range": float(np.ptp([(t.predict(base.test.X_orig) == base.test.y_hat).mean() for t in kept])),
            "distinct_root_features": len({int(t.tree_.feature[0]) for t in kept}),
            "feature_set_jaccard": float(np.mean([len(feats[a] & feats[b]) / max(len(feats[a] | feats[b]), 1)
                                                  for a, b in pairs])) if pairs else None,
            "rows_explanation_features_differ": float(np.mean([np.mean([pa != pb for pa, pb in zip(path_feats[a], path_feats[b])])
                                                               for a, b in pairs])) if pairs else None,
        }
        # W8 fairwashing: Fid lost by hiding the sensitive features from the surrogate
        if ds in SENSITIVE:
            keep = [k for k in range(X_tr.shape[1]) if names[k] not in SENSITIVE[ds]]
            fw = {}
            for lab, depth in (("depth 3", 3), ("tuned", d_star)):
                full = (fit(depth).predict(base.test.X_orig) == base.test.y_hat).mean()
                hid = (fit(depth, X=X_tr[:, keep]).predict(base.test.X_orig[:, keep]) == base.test.y_hat).mean()
                fw[lab] = {"fid_with": float(full), "fid_hidden": float(hid), "fid_cost": float(full - hid)}
            tr["fairwashing"] = fw
        # W14 data efficiency
        rng = np.random.default_rng(seed)
        de = {}
        for frac in (0.05, 0.10, 0.25, 0.50, 1.00):
            n = max(int(round(frac * len(X_tr))), 10)
            i = rng.choice(len(X_tr), size=n, replace=False) if frac < 1 else np.arange(len(X_tr))
            de[str(frac)] = {lab: float((fit(depth, X=X_tr[i], y=y_tr[i]).predict(base.test.X_orig) == base.test.y_hat).mean())
                             for lab, depth in (("depth 3", 3), ("tuned", d_star))}
        tr["data_efficiency"] = de
        recs.append(tr)
        print(f"  {ds} s{seed}", flush=True)
    # ---- W4 stability across seeds on the common pool
    pmin, pmax = pool.min(0), pool.max(0)
    for m, per_seed in boxes.items():
        cov = {s: {c: np.logical_or.reduce([np.all((pool >= lo) & (pool <= hi), axis=1) for lo, hi in bx])
                   for c, bx in cls.items()} for s, cls in per_seed.items()}
        fts = {s: {c: set(np.flatnonzero(np.logical_or.reduce([(lo > pmin) | (hi < pmax) for lo, hi in bx])).tolist())
                   for c, bx in cls.items()} for s, cls in per_seed.items()}
        rj, fj, per_class = [], [], defaultdict(lambda: ([], []))
        for a, b in itertools.combinations(sorted(per_seed), 2):
            for c in set(cov[a]) | set(cov[b]):
                if c in cov[a] and c in cov[b]:
                    u = (cov[a][c] | cov[b][c]).sum()
                    r_ = (cov[a][c] & cov[b][c]).sum() / u if u else 1.0
                    fu = fts[a][c] | fts[b][c]
                    f_ = len(fts[a][c] & fts[b][c]) / len(fu) if fu else 1.0
                else:
                    r_ = f_ = 0.0
                rj.append(r_); fj.append(f_)
                per_class[c][0].append(r_); per_class[c][1].append(f_)
        recs.append({"dataset": ds, "method": m, "stability": {
            "row_jaccard": float(np.mean(rj)), "feature_jaccard": float(np.mean(fj)),
            "per_class": {int(c): {"row_jaccard": float(np.mean(v[0])), "feature_jaccard": float(np.mean(v[1]))}
                          for c, v in per_class.items()}}})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        from revision.weakness_battery_report import main as report
        return report()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        path = OUT / f"{ds}.json"
        if path.is_file():
            continue
        recs = run_dataset(ds)
        path.write_text(json.dumps(recs, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"done {ds}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
