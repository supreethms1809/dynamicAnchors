"""Tables for `revision.weakness_battery` (plus the pert, instance and robustness inputs).

    python -m revision.weakness_battery --report
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402

RES = REPO.parent / "results"
WB = RES / "weakness_battery"
M = ("tree_d3", "tree_tuned", "rlda", "mada")
LAB = {"tree_d3": "Tree d3", "tree_tuned": "Tree tuned", "rlda": "RLDA", "mada": "MADA"}
SEEDS = (42, 43, 44, 45, 46)
NAME = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot",
        "breast_cancer": "breast cancer", "uci_credit": "uci credit", "uci_adult": "uci adult"}
nm = lambda d: NAME.get(d, d)  # noqa: E731


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def pc(x):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def f1(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.1f}"


def table(h, rows, align=None):
    align = align or "|---" + "|---:" * (len(h) - 1) + "|"
    return "\n".join(["| " + " | ".join(h) + " |", align] + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def holm(ps):
    ps = np.asarray(ps, float)
    o, n, run = np.argsort(ps), len(ps), 0.0
    adj = np.empty(n)
    for r, i in enumerate(o):
        run = max(run, min(1.0, (n - r) * ps[i]))
        adj[i] = run
    return adj


def load() -> Tuple[Dict, Dict, Dict]:
    cell, stab, tree_only = {}, {}, {}
    for ds in DATASETS:
        p = WB / f"{ds}.json"
        if not p.is_file():
            continue
        for r in json.loads(p.read_text()):
            if "stability" in r:
                stab[(ds, r["method"])] = r["stability"]
            elif r["method"] == "tree_only":
                tree_only[(ds, r["seed"])] = r
            else:
                cell[(ds, r["seed"], r["method"])] = r
    return cell, stab, tree_only


def dsmean(cell, m, get: Callable) -> Dict[str, float]:
    out = {}
    for ds in DATASETS:
        v = []
        for s in SEEDS:
            r = cell.get((ds, s, m))
            if r is None:
                continue
            x = get(r)
            if x is not None and np.isfinite(x):
                v.append(float(x))
        out[ds] = float(np.mean(v)) if v else float("nan")
    return out


def grand(d):
    v = [x for x in d.values() if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def metric_block(title: str, per: Dict[str, Dict[str, float]], fmt, better: str, note: str = "",
                 methods=M, datasets=None) -> str:
    """Overall + tests + dataset-wise for one metric; per[m][ds]."""
    datasets = datasets or [d for d in DATASETS if any(np.isfinite(per[m].get(d, np.nan)) for m in methods)]
    out = [f"**{title}** ({'higher' if better == 'high' else 'lower'} is better){(' — ' + note) if note else ''}\n"]
    rows = [[nm(d)] + [fmt(per[m].get(d, np.nan)) for m in methods] for d in datasets]
    rows.append(["**mean**"] + [f"**{fmt(np.nanmean([per[m].get(d, np.nan) for d in datasets]))}**" for m in methods])
    out.append(table(["Dataset"] + [LAB.get(m, m) for m in methods], rows))
    tr = [m for m in methods if m.startswith("tree")]
    rl = [m for m in methods if not m.startswith("tree")]
    pairs = [(a, b) for a in rl for b in tr]
    if pairs:
        ps, st = [], []
        for a, b in pairs:
            x = np.array([per[a].get(d, np.nan) for d in datasets]); y = np.array([per[b].get(d, np.nan) for d in datasets])
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.allclose(x[ok], y[ok]):
                ps.append(1.0)
            else:
                ps.append(paired_wilcoxon(list(x[ok]), list(y[ok]))["pvalue"])
            better_n = int(((x[ok] > y[ok]) if better == "high" else (x[ok] < y[ok])).sum())
            st.append((np.mean(x[ok] - y[ok]) if ok.any() else np.nan, better_n, int(ok.sum())))
        adj = holm(ps)
        out.append("RL minus tree (datasets where RL is better; Holm over the RL × tree contrasts): " + "; ".join(
            f"{LAB[a]} vs {LAB[b]} {st[i][0]:+.3f} ({st[i][1]}/{st[i][2]}; p={adj[i]:.3f})" for i, (a, b) in enumerate(pairs)) + "\n")
    return "\n".join(out)


# ---------------------------------------------------------------- instance robustness
def robustness(max_cells: Optional[int] = None) -> Dict:
    """Jaccard of D_test rows covered by the explanation of x* and of x' (see rl_rollout_experiments)."""
    cache = RES / "local_global" / "robust_scored.json"
    files = sorted((RES / "local_global" / "robust").glob("*.json"))
    if cache.is_file():
        c = json.loads(cache.read_text())
        if c.get("n_files") == len(files):
            return c["data"]
    from sklearn.tree import DecisionTreeClassifier
    from revision.rescore_boxes import _predictions, load_seed_data
    from utils.eval_harness import unit_to_original
    from utils.metrics import active_feature_mask

    data = defaultdict(dict)
    sd_cache = {}
    for f in files[:max_cells]:
        R = json.loads(f.read_text())
        ds, arm, seed = R["dataset"], R["arm"], R["seed"]
        if (ds, seed) not in sd_cache:
            ref = json.loads((RES / "paper_final_cart_fixed/precision_constrained/baselines_emp"
                              / f"{ds}__cart__seed{seed}__tp0p90__tc0p10.json").read_text())
            sd = load_seed_data(ds, seed, Path(ref["extra"]["classifier_path"]))
            L = sd.loader
            y_tr = _predictions(L, L.X_train_scaled)
            gs = json.loads((RES / "paper_final_cart_fixed/global_surrogate" / f"{ds}__seed{seed}.json").read_text())
            dstar = None if gs["val_tuned_depth"] in (None, "None") else int(gs["val_tuned_depth"])
            trees = {m: DecisionTreeClassifier(max_depth=d, random_state=seed).fit(np.asarray(L.X_train, np.float32), y_tr)
                     for m, d in (("tree_d3", 3), ("tree_tuned", dstar))}
            sd_cache[(ds, seed)] = (sd, trees)
        sd, trees = sd_cache[(ds, seed)]
        L = sd.loader
        Xt_u, Xt_o = sd.test.X_unit, sd.test.X_orig
        args = (L.X_min, L.X_range, np.asarray(L.scaler.mean_), np.asarray(L.scaler.scale_))
        jac, same_feats = [], []
        for r in R["rows"]:
            (l0, h0), (l1, h1) = r["box_x"], r["box_x_prime"]
            l0, h0, l1, h1 = (np.asarray(v, np.float32) for v in (l0, h0, l1, h1))
            m0, m1 = np.all((Xt_u >= l0) & (Xt_u <= h0), 1), np.all((Xt_u >= l1) & (Xt_u <= h1), 1)
            u = (m0 | m1).sum()
            jac.append((m0 & m1).sum() / u if u else 1.0)
            same_feats.append(bool(np.array_equal(active_feature_mask(l0, h0, 0.95), active_feature_mask(l1, h1, 0.95))))
        data[f"{ds}|{seed}"][arm] = {"row_jaccard": float(np.mean(jac)), "same_features": float(np.mean(same_feats)), "n": len(jac)}
        if arm == "rlda":  # tree side once per (ds, seed), on the same x / x' pairs
            lt = {m: t.apply(Xt_o) for m, t in trees.items()}
            for m, t in trees.items():
                jac, same = [], []
                for r in R["rows"]:
                    xo = Xt_o[r["index"]][None, :]
                    xp = np.asarray(unit_to_original(np.asarray(r["x_prime_unit"]), *args), np.float32)[None, :]
                    a, b = int(t.apply(xo)[0]), int(t.apply(xp)[0])
                    same.append(a == b)
                    ma, mb = lt[m] == a, lt[m] == b
                    u = (ma | mb).sum()
                    jac.append((ma & mb).sum() / u if u else 1.0)
                data[f"{ds}|{seed}"][m] = {"row_jaccard": float(np.mean(jac)), "same_features": float(np.mean(same)), "n": len(jac)}
        print(f"robust {ds} {arm} {seed}", flush=True)
    cache.write_text(json.dumps({"n_files": len(files), "data": data}))
    return data


def main() -> int:
    cell, stab, tree_only = load()
    parts = []
    get = lambda m, key: dsmean(cell, m, lambda r: r.get(key))  # noqa: E731
    per = lambda key: {m: get(m, key) for m in M}  # noqa: E731

    parts.append("## W1. Readability: size of the explanation\n")
    parts.append(metric_block("Rules (tree: leaves)", per("n_rules"), f1, "low"))
    parts.append(metric_block("Conditions per rule", per("cond_per_rule"), f1, "low"))
    parts.append(metric_block("Total conditions to read", per("total_conditions"), f1, "low"))
    parts.append(metric_block("Fid (tree: all rows; RL: decided rows)", per("fid"), f3, "high"))
    parts.append(metric_block("Coverage", per("coverage"), f3, "high"))

    parts.append("## W2. A high overall fidelity hiding places where the explanation is wrong\n")
    parts.append(metric_block("Share of explained D_test rows whose explaining rule has D_test Fid < 0.90", per("rows_rule_below_tau"), pc, "low"))
    parts.append(metric_block("10th percentile of the explaining rule's D_test Fid, over explained rows", per("row_rule_fid_p10"), f3, "high"))
    parts.append(metric_block("Worst class: lowest class precision", per("worst_class_precision"), f3, "high"))

    parts.append("## W3. Minority classes\n")
    parts.append(metric_block("Classes with no rule at all", per("classes_without_rule"), f1, "low"))
    parts.append(metric_block("Minority class (f̂'s rarest D_test class): recall = share of its rows explained as that class",
                              {m: dsmean(cell, m, lambda r: r["minority"]["recall"]) for m in M}, f3, "high"))
    parts.append(metric_block("Minority class: precision of the rows labelled that class",
                              {m: dsmean(cell, m, lambda r: r["minority"]["precision"]) for m in M}, f3, "high"))
    rows = []
    for ds in DATASETS:
        cls = sorted({int(c) for s in SEEDS for m in M if (ds, s, m) in cell for c in cell[(ds, s, m)]["per_class"]})
        for c in cls:
            def g(m, q):
                v = [cell[(ds, s, m)]["per_class"].get(str(c), {}).get(q) for s in SEEDS if (ds, s, m) in cell]
                v = [x for x in v if x is not None]
                return float(np.mean(v)) if v else float("nan")
            share = np.mean([cell[(ds, s, "tree_d3")]["per_class"].get(str(c), {}).get("n_rows", 0) /
                             sum(v["n_rows"] for v in cell[(ds, s, "tree_d3")]["per_class"].values())
                             for s in SEEDS if (ds, s, "tree_d3") in cell])
            rows.append([nm(ds) if c == cls[0] else "", str(c), pc(share)] + [f"{f3(g(m, 'recall'))} / {f3(g(m, 'precision'))}" for m in M])
    parts.append("**Class-wise recall / precision** (5-seed means)\n")
    parts.append(table(["Dataset", "Class", "Share of D_test"] + [LAB[m] for m in M], rows))

    parts.append("## W4. Stability across seeds (different split, classifier and fit)\n")
    parts.append("Per class, pairwise over the 10 seed pairs: Jaccard of the rows the class's rules cover in one common pool (every row of "
                 "the dataset) and of the features they constrain. A class without a rule in one seed counts 0.\n")
    for key, lab in (("row_jaccard", "Row Jaccard"), ("feature_jaccard", "Feature Jaccard")):
        parts.append(metric_block(lab, {m: {ds: stab.get((ds, m), {}).get(key, np.nan) for ds in DATASETS} for m in M}, f3, "high"))
    rows = []
    for ds in DATASETS:
        cls = sorted({int(c) for m in M for c in (stab.get((ds, m), {}).get("per_class") or {})})
        for c in cls:
            rows.append([nm(ds) if c == cls[0] else "", str(c)] + [
                "{} / {}".format(*(f3((stab.get((ds, m), {}).get("per_class") or {}).get(str(c), {}).get(q, np.nan))
                                   for q in ("row_jaccard", "feature_jaccard"))) for m in M])
    parts.append("**Class-wise row / feature Jaccard**\n")
    parts.append(table(["Dataset", "Class"] + [LAB[m] for m in M], rows))

    parts.append("## W5. Rashomon: many equally faithful surrogates telling different stories (tree only)\n")
    parts.append("21 depth-3 trees (the default and 20 with 70% feature subsampling); kept = within 0.01 of the best D_val agreement.\n")
    rows = []
    for ds in DATASETS:
        rs = [tree_only[(ds, s)]["rashomon"] for s in SEEDS if (ds, s) in tree_only]
        if not rs:
            continue
        mv = lambda q: np.nanmean([r[q] if r[q] is not None else np.nan for r in rs])  # noqa: E731
        rows.append([nm(ds), f1(mv("n_within_0p01")), f1(mv("distinct_root_features")), f3(mv("feature_set_jaccard")),
                     pc(mv("rows_explanation_features_differ")), f3(mv("test_fid_range"))])
    parts.append(table(["Dataset", "Trees kept", "Distinct root features", "Feature-set Jaccard between kept trees",
                        "Rows whose explanation uses different features (pairs of kept trees)", "Test Fid range of kept trees"], rows))
    parts.append("RL returns the rule set of one trained policy per class; the closest analogue of this test is W4 (retraining on another seed).\n")

    parts.append("## W6. Proxies: rules built on one of several highly correlated features\n")
    parts.append(metric_block("Share of explained rows whose rule uses a feature with an unused |Spearman ρ| ≥ 0.8 partner",
                              per("rows_rule_has_unused_proxy"), pc, "low", "descriptive: the explanation could as well name the partner"))

    parts.append("## W7. Behaviour off the data (perturbation fidelity, Anchors' D(z|B))\n")
    GSP = RES / "paper_final_cart_fixed" / "global_surrogate_pert"
    pert = {m: {} for m in M}
    keymap = {"tree_d3": ("trees", "plain / depth 3"), "tree_tuned": ("trees", "plain / val-tuned depth"),
              "rlda": ("rl", "rlda_emp"), "mada": ("rl", "mada_emp")}
    gap = {m: {} for m in M}
    for ds in DATASETS:
        for m, (sec, key) in keymap.items():
            v = [json.loads((GSP / f"{ds}__seed{s}.json").read_text())[sec][key] for s in SEEDS]
            pert[m][ds] = float(np.mean([x["pert_fid_rows"] for x in v]))
            gap[m][ds] = float(np.mean([x["emp_fid"] - x["pert_fid_rows"] for x in v]))
    parts.append(metric_block("Row-weighted perturbation Fid", pert, f3, "high"))
    parts.append(metric_block("Drop from empirical to perturbation Fid", gap, f3, "low"))

    parts.append("## W8. Sensitive features: can the explanation hide what the model uses?\n")
    rows = []
    for ds in ("uci_adult", "folktables_income_CA_2018", "sick"):
        feats = cell.get((ds, 42, "tree_d3"), {}).get("sensitive", {})
        for f in feats:
            def sv(m, q):
                v = [cell[(ds, s, m)]["sensitive"][f][q] for s in SEEDS if (ds, s, m) in cell]
                v = [x for x in v if x is not None]
                return float(np.mean(v)) if v else float("nan")
            fw = [tree_only[(ds, s)]["fairwashing"] for s in SEEDS if (ds, s) in tree_only]
            rows.append([nm(ds), f, pc(sv("tree_d3", "rows_fhat_depends"))] +
                        [pc(sv(m, "dependent_rows_whose_rule_mentions")) for m in M] +
                        [f"{np.mean([x['depth 3']['fid_cost'] for x in fw]):+.3f} / {np.mean([x['tuned']['fid_cost'] for x in fw]):+.3f}"])
    parts.append("Rows whose f̂ label changes when the sensitive feature is set to another observed value, and among those rows (explained "
                 "ones), the share whose explaining rule constrains that feature. Last column: tree Fid lost when the sensitive features "
                 "are removed from the surrogate (depth 3 / tuned) — near zero means a surrogate can hide them at no visible cost.\n")
    parts.append(table(["Dataset", "Feature", "Rows where f̂ depends on it"] + [f"{LAB[m]}: rule mentions it" for m in M] +
                       ["Tree Fid cost of hiding (d3 / tuned)"], rows))
    rows = []
    for ds in ("uci_adult", "folktables_income_CA_2018", "sick"):
        for f in cell.get((ds, 42, "tree_d3"), {}).get("sensitive", {}):
            rows.append([nm(ds), f] + [pc(float(np.mean([cell[(ds, s, m)]["sensitive"][f]["explained_rows_whose_rule_mentions"] or 0
                                                         for s in SEEDS if (ds, s, m) in cell]))) for m in M])
    parts.append("**All explained rows whose rule mentions the feature**\n")
    parts.append(table(["Dataset", "Feature"] + [LAB[m] for m in M], rows))

    parts.append("## W9. Explaining the wrong class (instance level)\n")
    CF = RES / "containment_fix" / "tree"
    wrong = {"tree_d3": {}, "tree_tuned": {}}
    for ds in DATASETS:
        for m, k in (("tree_d3", "depth 3"), ("tree_tuned", "val-tuned depth")):
            wrong[m][ds] = float(np.mean([1 - np.mean([r[k]["tree_label_matches_y_hat"] for r in json.loads((CF / f"{ds}__seed{s}.json").read_text())["rows"]])
                                          for s in SEEDS if (CF / f"{ds}__seed{s}.json").is_file()]))
    wrong["rlda"] = {ds: 0.0 for ds in DATASETS}
    wrong["mada"] = {ds: 0.0 for ds in DATASETS}
    parts.append(metric_block("Share of explained inputs whose explanation is for a class other than ŷ(x*)", wrong, pc, "low",
                              "RL routes each input to the policy of ŷ(x*), so 0 by construction"))

    parts.append("## W10. Always answering: reliability of what the explanation claims\n")
    parts.append(metric_block("|stated − realised| Fid of the explaining rule (stated = D_val Fid)", per("stated_realised_abs_gap"), f3, "low"))
    parts.append(metric_block("Overclaims: stated ≥ 0.90 but realised < 0.85", per("overclaim_rate"), pc, "low"))
    tp = {m: get(m, "train_purity_abs_gap") for m in ("tree_d3", "tree_tuned")}
    parts.append(f"A tree usually states its leaf purity on the data it was fit to; that claim is off by {f3(grand(tp['tree_d3']))} (depth 3) "
                 f"and {f3(grand(tp['tree_tuned']))} (tuned) on D_test, against {f3(grand(get('tree_d3', 'stated_realised_abs_gap')))} and "
                 f"{f3(grand(get('tree_tuned', 'stated_realised_abs_gap')))} for D_val Fid.\n")
    GS = RES / "paper_final_cart_fixed" / "global_surrogate"
    ab = {}
    for m, key in (("tree_d3", "depth 3"), ("tree_tuned", "val-tuned depth")):
        for arm in ("rlda", "mada"):
            ab[(m, arm)] = {ds: float(np.nanmean([(json.loads((GS / f"{ds}__seed{s}.json").read_text())["trees"][key]
                                                  .get(f"fid_on_{arm}_k1_abstained") or np.nan) for s in SEEDS])) for ds in DATASETS}
            ab[(m, arm, "dec")] = {ds: float(np.nanmean([json.loads((GS / f"{ds}__seed{s}.json").read_text())["trees"][key]
                                                         [f"fid_on_{arm}_k1_decided"] for s in SEEDS])) for ds in DATASETS}
    rows = [[nm(ds)] + [f"{f3(ab[(m, arm, 'dec')][ds])} / {f3(ab[(m, arm)][ds])}" for m in ("tree_d3", "tree_tuned") for arm in ("rlda", "mada")]
            for ds in DATASETS]
    rows.append(["**mean**"] + [f"**{f3(grand(ab[(m, arm, 'dec')]))} / {f3(grand(ab[(m, arm)]))}**" for m in ("tree_d3", "tree_tuned") for arm in ("rlda", "mada")])
    parts.append("**Where RL abstains, the tree is least faithful**: tree agreement with f̂ on the rows RL decides / abstains on\n")
    parts.append(table(["Dataset", "Tree d3, RLDA rows", "Tree d3, MADA rows", "Tree tuned, RLDA rows", "Tree tuned, MADA rows"], rows))

    parts.append("## W12. Sensitivity to the method's own knobs\n")
    rows = []
    for ds in DATASETS:
        gsd = [json.loads((GS / f"{ds}__seed{s}.json").read_text())["trees"] for s in SEEDS]
        tf = [np.mean([g[d]["fid"] for g in gsd]) for d in ("depth 2", "depth 3", "depth 4", "depth 5", "depth 8", "val-tuned depth", "fully grown")]
        tl = [np.mean([g[d]["n_leaves"] for g in gsd]) for d in ("depth 2", "fully grown")]
        rl = {}
        for arm, algo in (("rlda", "ddpg"), ("mada", "maddpg")):
            vals = []
            for src in ([f"paper_final_valtb/emp_tc0p10/results/{algo}", f"paper_final_valtb/emp_tc0p20/results/{algo}",
                         f"paper_final_valtb/pert_tc0p10/results/{algo}"] + [f"paper_final_valtb/k_sweep/k{k}" for k in (2, 3, 5, 10, 20)]):
                v = []
                for s in SEEDS:
                    tc = "tc0p20" if "tc0p20" in src else "tc0p10"
                    p = RES / src / f"{ds}__{arm}__seed{s}__tp0p90__{tc}.json"
                    if p.is_file():
                        g = json.loads(p.read_text())["global_ruleset"]
                        v.append((g["global_fidelity"] or np.nan, g["coverage"]))
                if v:
                    vals.append((np.nanmean([a for a, _ in v]), np.mean([b for _, b in v])))
            rl[arm] = vals
        rows.append([nm(ds), f"{min(tf):.3f}–{max(tf):.3f}", f"{tl[0]:.0f}–{tl[1]:.0f}"] +
                    [f"{min(a for a, _ in rl[arm]):.3f}–{max(a for a, _ in rl[arm]):.3f} / {min(b for _, b in rl[arm]):.3f}–{max(b for _, b in rl[arm]):.3f}"
                     for arm in ("rlda", "mada")])
    parts.append("Tree: depth 2 … fully grown (coverage always 1). RL: k ∈ {1, 2, 3, 5, 10, 20}, τ_C ∈ {0.10, 0.20}, trained on empirical or "
                 "perturbation Fid (8 configurations, each a 5-seed mean).\n")
    parts.append(table(["Dataset", "Tree Fid range", "Tree leaves range", "RLDA Fid range / coverage range", "MADA Fid range / coverage range"], rows))

    parts.append("## W13. Nominal features cut into arbitrary groups of codes\n")
    nd = [d for d in DATASETS if np.isfinite(get("tree_d3", "nominal_rows_grouping_codes").get(d, np.nan))]
    parts.append(metric_block("Share of explained rows whose rule groups 2..n−1 codes of a nominal feature by a threshold",
                              per("nominal_rows_grouping_codes"), pc, "low", "datasets with nominal features ≥ 3 codes", datasets=nd))

    parts.append("## W14. Data needed to fit the surrogate (tree only)\n")
    rows = []
    for ds in DATASETS:
        de = [tree_only[(ds, s)]["data_efficiency"] for s in SEEDS if (ds, s) in tree_only]
        if de:
            rows.append([nm(ds)] + [f"{np.mean([d[fr]['depth 3'] for d in de]):.3f} / {np.mean([d[fr]['tuned'] for d in de]):.3f}"
                                    for fr in ("0.05", "0.1", "0.25", "0.5", "1.0")])
    parts.append("Tree Fid on D_test when fit to a random 5–100% of D_train (depth 3 / tuned depth). RL would need retraining on each "
                 "subsample; not run.\n")
    parts.append(table(["Dataset", "5%", "10%", "25%", "50%", "100%"], rows))

    parts.append("## Instance-level robustness: does the explanation move when the input barely moves?\n")
    rob = robustness()
    if rob:
        per_r = {m: {} for m in M}
        same = {m: {} for m in M}
        for ds in DATASETS:
            for m in M:
                v = [rob[k][m] for k in rob if k.split("|")[0] == ds and m in rob[k]]
                if v:
                    per_r[m][ds] = float(np.mean([x["row_jaccard"] for x in v]))
                    same[m][ds] = float(np.mean([x["same_features"] for x in v]))
        seeds = sorted({k.split("|")[1] for k in rob})
        parts.append(f"x' = x* plus Gaussian noise (sd 0.02 of each non-categorical feature's range), same ŷ; RL rolled out at both with the "
                     f"same rollout seed; tree: the leaf of each. Seeds {', '.join(seeds)}, up to 200 inputs per cell.\n")
        parts.append(metric_block("Jaccard of the D_test rows covered by the explanations of x* and x'", per_r, f3, "high"))
        parts.append(metric_block("Same constrained features (tree: same leaf)", same, pc, "high"))
    return write(parts)


def write(parts: List[str]) -> int:
    out = WB / "WEAKNESS_BATTERY.md"
    head = ("# Global-surrogate weaknesses, run on the plain surrogate and on RL\n\n"
            "Generated by `python -m revision.weakness_battery --report`. Class level: 12 datasets × seeds 42–46, emp τ_C = 0.10, k = 1; "
            "Tree d3 / Tree tuned = the unmodified global surrogate (depth 3 / D_val-tuned depth); RLDA / MADA = the paper's rule sets. "
            "Dataset values are 5-seed means; tests are paired Wilcoxon over datasets, Holm-adjusted over the four RL × tree contrasts of "
            "each metric.\n\n")
    out.write_text(head + "\n".join(parts))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
