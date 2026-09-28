"""Full write-up: the unmodified CART global surrogate as the tree baseline vs RLDA / MADA / Anchors.

Assembles every table from the result files; the Summary and Section 5 prose quote them
(check them against the tables after a re-run):
  class level, empirical Fid   -- revision.cart_global_surrogate       -> paper_final_cart_fixed/global_surrogate/
  class level, perturbation Fid -- revision.cart_global_surrogate_pert -> paper_final_cart_fixed/global_surrogate_pert/
  instance level               -- revision.containment_eval + tree_instance_eval -> containment_fix/{,tree/}
  modified CART (appendix)     -- paper_final_cart_fixed/RL_vs_CART_k1_k5.md

    python -m revision.global_surrogate_full_report [--out ../results/paper_final_cart_fixed/GLOBAL_SURROGATE_RESULTS.md]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402

RES = REPO.parent / "results"
GS = RES / "paper_final_cart_fixed" / "global_surrogate"
GSP = RES / "paper_final_cart_fixed" / "global_surrogate_pert"
CF = RES / "containment_fix"
VALTB = RES / "paper_final_valtb"
SEEDS = (42, 43, 44, 45, 46)
NAME = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot",
        "breast_cancer": "breast cancer", "uci_credit": "uci credit", "uci_adult": "uci adult"}
nm = lambda d: NAME.get(d, d)  # noqa: E731


# ---------------------------------------------------------------- helpers
def holm(ps: Sequence[float]) -> np.ndarray:
    ps = np.asarray(ps, float)
    order, n, run = np.argsort(ps), len(ps), 0.0
    adj = np.empty(n)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * ps[i]))
        adj[i] = run
    return adj


def f3(x) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def pct(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.0f}%"


def table(header: List[str], rows: List[List[str]], align: str = None) -> str:
    align = align or ("|---" + "|---:" * (len(header) - 1) + "|")
    return "\n".join(["| " + " | ".join(header) + " |", align] + ["| " + " | ".join(r) + " |" for r in rows])


def ds_mean(get: Callable[[str, int], float]) -> Dict[str, float]:
    """Per dataset: mean over seeds (NaN-aware)."""
    out = {}
    for ds in DATASETS:
        v = [get(ds, s) for s in SEEDS]
        v = [x for x in v if x is not None and np.isfinite(x)]
        out[ds] = float(np.mean(v)) if v else float("nan")
    return out


def grand(d: Dict[str, float]) -> float:
    v = [x for x in d.values() if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def contrast_rows(pairs, get_ds: Callable[[str, str], Dict[str, float]], metrics, labels) -> List[List[str]]:
    """Paired Wilcoxon over datasets for each (a, b) pair; Holm within each metric."""
    cells = {}
    for q in metrics:
        ps, stats = [], []
        for a, b in pairs:
            x = np.array([get_ds(a, q)[d] for d in DATASETS])
            y = np.array([get_ds(b, q)[d] for d in DATASETS])
            ok = np.isfinite(x) & np.isfinite(y)
            ps.append(paired_wilcoxon(list(x[ok]), list(y[ok]))["pvalue"])
            stats.append(((x[ok] - y[ok]).mean(), int((x[ok] > y[ok]).sum()), int(ok.sum())))
        cells[q] = list(zip(stats, holm(ps)))
    rows = []
    for i, (a, b) in enumerate(pairs):
        rows.append([f"{labels.get(a, a)} vs {labels.get(b, b)}"] + [
            f"{cells[q][i][0][0]:+.3f} ({cells[q][i][0][1]}/{cells[q][i][0][2]}; {cells[q][i][1]:.3f})" for q in metrics])
    return rows


# ---------------------------------------------------------------- loaders
def load_dir(d: Path) -> Dict:
    return {(ds, s): json.loads((d / f"{ds}__seed{s}.json").read_text()) for ds in DATASETS for s in SEEDS}


def rl_cell(method: str, k: int, est: str, ds: str, s: int) -> Dict:
    algo = {"rlda": "ddpg", "mada": "maddpg"}[method]
    sub = f"{est}_tc0p10/results/{algo}" if k == 1 else "k_sweep/k5"
    return json.loads((VALTB / sub / f"{ds}__{method}__seed{s}__tp0p90__tc0p10.json").read_text())


# ================================================================ sections
def sec_empirical(E: Dict) -> str:
    T = ["depth 2", "depth 3", "depth 4", "depth 5", "depth 8", "val-tuned depth", "fully grown"]
    tv = lambda t, q: ds_mean(lambda d, s: E[(d, s)]["trees"][t][q])  # noqa: E731
    rv = lambda key, q: ds_mean(lambda d, s: E[(d, s)]["rl"][key][q])  # noqa: E731
    RL = {"rlda_k1": "RLDA k=1", "mada_k1": "MADA k=1", "rlda_k5": "RLDA k=5", "mada_k5": "MADA k=5"}
    rl_comp = {}
    for key in RL:
        m, k = key.split("_k")
        rl_comp[key] = {
            "rules": ds_mean(lambda d, s: sum(len(b.get("selected_rules") or []) for b in rl_cell(m, int(k), "emp", d, s)["per_class"].values())),
            "cond": ds_mean(lambda d, s: rl_cell(m, int(k), "emp", d, s)["compactness"]["mean_active_features"]),
        }
    out = ["## 2. Class level, empirical fidelity\n",
           "Tree: one `DecisionTreeClassifier(max_depth=d, random_state=seed)` fit to f̂(D_train) in original units; every leaf is a rule for its "
           "majority class, the tree answers every D_test row (coverage 1), and Fid = Eff = agreement with f̂ on D_test. "
           "'val-tuned depth' = the shallowest depth in 1–15 or unlimited with the best D_val agreement. RL = the paper's rule sets "
           "(`paper_final_valtb`, emp τ_C = 0.10, conflicts tie-broken on D_val; k = 5 from the k-sweep). RL Fid is on the rows RL decides; "
           "Coverage = share of D_test rows decided; Eff = Fid × Coverage.\n",
           "### 2.1 Overall (means over 12 datasets of 5-seed means)\n"]
    rows = []
    for t in T:
        rows.append([f"Tree, {t}", f3(grand(tv(t, "fid"))), "1.000", f3(grand(tv(t, "fid"))),
                     f"{grand(tv(t, 'n_leaves')):.0f} / {np.median(list(tv(t, 'n_leaves').values())):.0f}",
                     f"{grand(tv(t, 'depth')):.1f}", f"{grand(tv(t, 'cond_per_leaf')):.1f}", f"{grand(tv(t, 'cond_per_row')):.1f}",
                     pct(grand(tv(t, "rows_in_weak_leaf"))), f3(grand(tv(t, "acc")))])
    for key, lab in RL.items():
        rows.append([lab, f3(grand(rv(key, "fid"))), f3(grand(rv(key, "cov"))), f3(grand(rv(key, "eff"))),
                     f"{grand(rl_comp[key]['rules']):.1f} rules", "—", f"{grand(rl_comp[key]['cond']):.1f}", "—", "—", "—"])
    out.append(table(["Method", "Fid", "Coverage", "Eff", "Leaves or rules (mean / median)", "Depth", "Conditions / rule",
                      "Conditions / test row", "Rows in a leaf with test Fid < 0.90", "Accuracy vs y"], rows))
    out.append("\nRows in a weak leaf: the share of D_test rows whose leaf agrees with f̂ on fewer than 90% of its D_test rows. "
               "For the RL rule sets the comparable figure (share of decided rows whose rule is below τ_P) is 38% RLDA and 46% MADA at k = 1 "
               "(`paper_final_cart_fixed/SURROGATE_WEAKNESSES.md`).\n")

    # same rows
    out.append("### 2.2 The same rows: tree agreement on the D_test rows each RL rule set decides, and on the rows it abstains on\n")
    rows, P = [], []
    TT = ["depth 3", "depth 5", "val-tuned depth", "fully grown"]
    for key, lab in RL.items():
        rl = rv(key, "fid")
        row = [lab, f3(grand(rl))]
        for t in TT:
            tr = tv(t, f"fid_on_{key}_decided")
            x = np.array([rl[d] for d in DATASETS]); y = np.array([tr[d] for d in DATASETS])
            p = paired_wilcoxon(list(x), list(y))["pvalue"]
            P.append(p)
            row.append(f"{y.mean():.3f} (RL higher {(x > y).sum()}/12; p={p:.3f})")
        row += [f3(grand(tv("depth 3", f"fid_on_{key}_abstained"))), f3(grand(tv("val-tuned depth", f"fid_on_{key}_abstained")))]
        rows.append(row)
    out.append(table(["RL rule set", "RL Fid", "Tree d3", "Tree d5", "Tree tuned", "Tree full", "Tree d3 on abstained rows",
                      "Tree tuned on abstained rows"], rows))
    out.append("\np: paired Wilcoxon over 12 datasets, unadjusted. The tree's agreement drops from the rows RL decides to the rows it "
               "abstains on: RL's abstention picks up the hard rows, but it is not more faithful than the tree on the rows it keeps.\n")

    # dataset-wise
    out.append("### 2.3 Dataset-wise (5-seed means)\n")
    out.append("**Fidelity** (tree: all rows; RL: decided rows) and RL coverage\n")
    rows = []
    for d in DATASETS:
        rows.append([nm(d), str(E[(d, 42)]["n_classes"])] + [f3(tv(t, "fid")[d]) for t in T] +
                    [f"{f3(rv(k, 'fid')[d])} / {f3(rv(k, 'cov')[d])}" for k in RL])
    out.append(table(["Dataset", "Classes"] + [f"Tree {t}" for t in T] + [f"{v} Fid / Cov" for v in RL.values()], rows))
    out.append("\n**Effectiveness** (tree Eff = tree Fid)\n")
    rows = [[nm(d)] + [f3(tv(t, "fid")[d]) for t in ("depth 3", "depth 5", "val-tuned depth")] + [f3(rv(k, "eff")[d]) for k in RL] for d in DATASETS]
    out.append(table(["Dataset", "Tree d3", "Tree d5", "Tree tuned"] + list(RL.values()), rows))
    out.append("\n**Tree size**: leaves, depth, conditions per leaf; val-tuned depth per seed\n")
    rows = []
    for d in DATASETS:
        depths = ",".join(str(E[(d, s)]["val_tuned_depth"]) for s in SEEDS)
        rows.append([nm(d)] + [f"{tv(t, 'n_leaves')[d]:.0f}" for t in T] + [depths] +
                    [f"{tv(t, 'cond_per_leaf')[d]:.1f}" for t in ("depth 3", "val-tuned depth", "fully grown")] +
                    [f"{rl_comp['rlda_k1']['rules'][d]:.1f} / {rl_comp['rlda_k1']['cond'][d]:.1f}"])
    out.append(table(["Dataset"] + [f"Leaves {t}" for t in T] + ["Tuned depth (s42–46)", "Cond/leaf d3", "Cond/leaf tuned",
                                                                   "Cond/leaf full", "RLDA k=1 rules / cond"], rows))
    out.append("\n**Share of D_test rows in a leaf with test Fid < 0.90**, and tree accuracy against the true label\n")
    rows = [[nm(d)] + [pct(tv(t, "rows_in_weak_leaf")[d]) for t in ("depth 3", "depth 5", "val-tuned depth", "fully grown")] +
            [f3(tv(t, "acc")[d]) for t in ("depth 3", "val-tuned depth")] for d in DATASETS]
    out.append(table(["Dataset", "Weak d3", "Weak d5", "Weak tuned", "Weak full", "Acc d3", "Acc tuned"], rows))
    out.append("\n**Tree agreement on RL's decided / abstained rows** (k = 1)\n")
    rows = [[nm(d), f3(rv("rlda_k1", "fid")[d]), f3(tv("depth 3", "fid_on_rlda_k1_decided")[d]), f3(tv("depth 3", "fid_on_rlda_k1_abstained")[d]),
             f3(rv("mada_k1", "fid")[d]), f3(tv("depth 3", "fid_on_mada_k1_decided")[d]), f3(tv("depth 3", "fid_on_mada_k1_abstained")[d])]
            for d in DATASETS]
    out.append(table(["Dataset", "RLDA Fid", "d3 on RLDA-decided", "d3 on RLDA-abstained", "MADA Fid", "d3 on MADA-decided",
                      "d3 on MADA-abstained"], rows))
    miss = {t: sorted({nm(d) for d in DATASETS for s in SEEDS
                       if E[(d, s)]["trees"][t]["classes_with_leaf"] < E[(d, s)]["n_classes"]}) for t in ("depth 2", "depth 3")}
    nmiss = {t: sum(E[(d, s)]["trees"][t]["classes_with_leaf"] < E[(d, s)]["n_classes"] for d in DATASETS for s in SEEDS) for t in miss}
    out.append(f"\nA class with no leaf at all: depth 2 in {nmiss['depth 2']} of 60 cells ({', '.join(miss['depth 2'])}); "
               f"depth 3 in {nmiss['depth 3']} ({', '.join(miss['depth 3'])}).\n")

    # class-wise
    out.append("### 2.4 Class-wise (5-seed means)\n")
    out.append("Tree precision = agreement with f̂ on the D_test rows the tree labels c (the Fid of class c's leaves); recall = share of "
               "f̂'s class-c rows the tree labels c; leaves = number of leaves labelled c. RL: class-union Fid on D_test and class-conditional "
               "coverage (share of f̂'s class-c rows inside a class-c rule), k = 1.\n")
    rows = []
    for d in DATASETS:
        for c in range(E[(d, 42)]["n_classes"]):
            def tc(t, q):
                v = [E[(d, s)]["trees"][t]["per_class"][str(c)][q] for s in SEEDS]
                v = [x for x in v if x is not None]
                return float(np.mean(v)) if v else float("nan")

            def rc(m, q):
                v = []
                for s in SEEDS:
                    u = (rl_cell(m, 1, "emp", d, s)["per_class"].get(f"class_{c}") or {}).get("union") or {}
                    if u.get(q) is not None and np.isfinite(u[q]):
                        v.append(u[q])
                return float(np.mean(v)) if v else float("nan")
            rows.append([nm(d) if c == 0 else "", str(c),
                         f"{f3(tc('depth 3', 'precision'))} / {f3(tc('depth 3', 'recall'))} ({tc('depth 3', 'n_leaves'):.1f})",
                         f"{f3(tc('val-tuned depth', 'precision'))} / {f3(tc('val-tuned depth', 'recall'))} ({tc('val-tuned depth', 'n_leaves'):.0f})",
                         f"{f3(rc('rlda', 'fidelity'))} / {f3(rc('rlda', 'coverage'))}", f"{f3(rc('mada', 'fidelity'))} / {f3(rc('mada', 'coverage'))}"])
    out.append(table(["Dataset", "Class", "Tree d3 precision / recall (leaves)", "Tree tuned precision / recall (leaves)",
                      "RLDA Fid / class cov", "MADA Fid / class cov"], rows))
    return "\n".join(out) + "\n"


def sec_pert(Pd: Dict) -> str:
    TREES = [f"{f} / {d}" for f in ("plain", "pert fit") for d in ("depth 3", "depth 5", "val-tuned depth", "fully grown")]
    RL = {"rlda_emp": "RLDA (trained on emp Fid)", "mada_emp": "MADA (trained on emp Fid)",
          "rlda_pert": "RLDA (trained on pert Fid)", "mada_pert": "MADA (trained on pert Fid)"}

    def g(c, m, q):
        return c["trees"][m][q] if m in c["trees"] else c["rl"][m].get(q, float("nan"))

    dm = lambda m, q: ds_mean(lambda d, s: g(Pd[(d, s)], m, q))  # noqa: E731
    out = ["## 3. Class level, perturbation fidelity\n",
           "Every rule (a tree leaf, or an RL selected rule) is scored with Anchors' D(z|B): draw a D_train row, replace only the coordinates "
           "that violate a predicate with a D_train value inside that predicate, and query f̂ (512 draws; the sampler reproduces the earlier "
           "Track B file within Monte Carlo noise). Target class = the rule's class. **Row-weighted** = each rule weighted by the D_test rows it "
           "covers (for a tree, the pert Fid of the leaf of each test row, averaged over rows; for RL over decided rows only). "
           "**Mean over rules** = unweighted (comparable to the paper's 0.59 / 0.61 per-rule figures). Trees: 'plain' is fit to f̂(D_train); "
           "'pert fit' to D_train plus an equal number of recombined rows (each feature drawn independently from its train marginal) labelled "
           "by f̂, which is what `run_cart` does under the perturbed estimator. RL at k = 1, τ_C = 0.10, trained on empirical (emp_tc0p10) "
           "or perturbation (pert_tc0p10) fidelity.\n",
           "### 3.1 Overall\n"]
    rows = []
    for m in TREES:
        rows.append([f"Tree, {m}", f3(grand(dm(m, "pert_fid_rows"))), f3(grand(dm(m, "pert_fid_rule_mean"))), pct(grand(dm(m, "rows_pert_ge_tau"))),
                     f3(grand(dm(m, "emp_fid"))), "1.000", f"{grand(dm(m, 'n_leaves')):.0f} / {np.median(list(dm(m, 'n_leaves').values())):.0f}",
                     f"{grand(dm(m, 'cond_per_leaf')):.1f}"])
    for m, lab in RL.items():
        rows.append([lab, f3(grand(dm(m, "pert_fid_rows"))), f3(grand(dm(m, "pert_fid_rule_mean"))), pct(grand(dm(m, "rows_pert_ge_tau"))),
                     f3(grand(dm(m, "emp_fid"))), f3(grand(dm(m, "coverage"))), f"{grand(dm(m, 'n_rules')):.1f}", f"{grand(dm(m, 'cond_per_rule')):.1f}"])
    out.append(table(["Method", "Pert Fid (row-weighted)", "Pert Fid (mean over rules)", "Rows whose rule has pert Fid ≥ 0.90",
                      "Emp Fid", "Coverage", "Rules (mean / median)", "Conditions / rule"], rows))
    out.append("\nFor reference, greedy Anchors' selected rules average 0.691 per rule under the same sampler (Track B file, 2,000 draws).\n")

    out.append("### 3.2 Paired tests (Wilcoxon over 12 datasets; Holm over the 16 contrasts per metric)\n")
    pairs = [(a, b) for a in RL for b in ("plain / depth 3", "pert fit / depth 3", "plain / val-tuned depth", "pert fit / val-tuned depth")]
    rows = contrast_rows(pairs, dm, ["pert_fid_rows", "pert_fid_rule_mean"], {**RL})
    out.append(table(["RL vs tree", "Δ row-weighted pert Fid (RL higher; p_Holm)", "Δ pert Fid, mean over rules (RL higher; p_Holm)"], rows,
                     "|---|---|---|"))
    out.append("")

    out.append("### 3.3 Dataset-wise (5-seed means)\n")
    cols = list(RL) + ["plain / depth 3", "pert fit / depth 3", "plain / val-tuned depth", "pert fit / val-tuned depth"]
    heads = ["RLDA emp", "MADA emp", "RLDA pert", "MADA pert", "Tree plain d3", "Tree pert-fit d3", "Tree plain tuned", "Tree pert-fit tuned"]
    for q, lab in (("pert_fid_rows", "Row-weighted pert Fid"), ("pert_fid_rule_mean", "Pert Fid, mean over rules"),
                   ("rows_pert_ge_tau", "Share of rows whose rule has pert Fid ≥ 0.90"), ("emp_fid", "Empirical Fid of the same rules")):
        D = {m: dm(m, q) for m in cols}
        fmt = pct if q == "rows_pert_ge_tau" else f3
        out.append(f"**{lab}**\n")
        out.append(table(["Dataset"] + heads, [[nm(d)] + [fmt(D[m][d]) for m in cols] for d in DATASETS]))
        out.append("")
    D = {m: dm(m, "n_leaves" if m in TREES else "n_rules") for m in cols}
    out.append("**Rules (RL) / leaves (tree)**\n")
    out.append(table(["Dataset"] + heads, [[nm(d)] + [f"{D[m][d]:.1f}" if m in RL else f"{D[m][d]:.0f}" for m in cols] for d in DATASETS]))
    out.append("")

    if "per_class" in Pd[(DATASETS[0], SEEDS[0])]["trees"]["plain / depth 3"]:
        out.append("### 3.4 Class-wise, row-weighted pert Fid (5-seed means; tree leaves labelled c / RL rules for c)\n")
        rows = []
        for d in DATASETS:
            classes = sorted({int(c) for s in SEEDS for m in cols for c in
                              (Pd[(d, s)]["trees"][m]["per_class"] if m in TREES else Pd[(d, s)]["rl"][m]["per_class"]).keys()})
            for c in classes:
                def pc(m, q="pert_fid_rows"):
                    v = []
                    for s in SEEDS:
                        blk = Pd[(d, s)]["trees"][m]["per_class"] if m in TREES else Pd[(d, s)]["rl"][m]["per_class"]
                        x = (blk.get(str(c)) or {}).get(q)
                        if x is not None and np.isfinite(x):
                            v.append(x)
                    return float(np.mean(v)) if v else float("nan")
                rows.append([nm(d) if c == classes[0] else "", str(c)] + [f3(pc(m)) for m in cols])
        out.append(table(["Dataset", "Class"] + heads, rows))
        out.append("\n'—' = no rule/leaf for that class in any seed.\n")
    return "\n".join(out) + "\n"


def sec_instance() -> str:
    M = ["RLDA", "MADA", "Anchors", "Tree d3", "Tree d5", "Tree tuned"]
    TK = {"Tree d3": "depth 3", "Tree d5": "depth 5", "Tree tuned": "val-tuned depth"}
    per = {d: defaultdict(list) for d in DATASETS}
    percls = defaultdict(lambda: defaultdict(list))
    stored = {d: defaultdict(list) for d in DATASETS}
    cost = defaultdict(list)
    serve = defaultdict(list)
    depths = defaultdict(list)
    mism = n_all = 0

    def add(d, m, xs, bucket):
        bucket[d][m + "|cond"].append(np.nanmean([x["cond_fid"] for x in xs]))
        bucket[d][m + "|ok"].append(np.mean([x["cond_fid"] >= 0.9 for x in xs]))
        bucket[d][m + "|cov"].append(np.mean([x["coverage"] for x in xs]))
        bucket[d][m + "|act"].append(np.mean([x["n_active"] for x in xs]))
        bucket[d][m + "|contain"].append(np.mean([x["contains_x"] for x in xs]))
        bucket[d][m + "|emp"].append(np.nanmean([x["emp_fid"] for x in xs]))
        if "tree_label_matches_y_hat" in xs[0]:
            bucket[d][m + "|agree"].append(np.mean([x["tree_label_matches_y_hat"] for x in xs]))

    for d in DATASETS:
        for s in SEEDS:
            r = json.loads((CF / f"{d}__rlda__seed{s}.json").read_text())["rows"]
            a = json.loads((CF / f"{d}__mada__seed{s}.json").read_text())["rows"]
            t = json.loads((CF / "tree" / f"{d}__seed{s}.json").read_text())
            assert [x["index"] for x in r] == [x["index"] for x in a] == [x["index"] for x in t["rows"]]
            add(d, "RLDA", [x["pi_contained"] for x in r if x.get("pi_contained")], per)
            add(d, "MADA", [x["pi_contained"] for x in a if x.get("pi_contained")], per)
            add(d, "Anchors", [x["anchors"] for x in r if x.get("anchors")], per)
            add(d, "RLDA π", [x["pi"] for x in r if x.get("pi")], stored)
            add(d, "MADA π", [x["pi"] for x in a if x.get("pi")], stored)
            for m, k in TK.items():
                add(d, m, [x[k] for x in t["rows"]], per)
            for x_r, x_a, x_t in zip(r, a, t["rows"]):
                c = x_t["y_hat"]
                for m, v in (("RLDA", x_r.get("pi_contained")), ("MADA", x_a.get("pi_contained")), ("Anchors", x_r.get("anchors")),
                             ("Tree d3", x_t["depth 3"]), ("Tree tuned", x_t["val-tuned depth"])):
                    if v:
                        percls[(d, c)][m].append(v["cond_fid"])
                percls[(d, c)]["agree"].append(x_t["depth 3"]["tree_label_matches_y_hat"])
            mism += sum(x["y_hat"] != x.get("y_hat_original_units", x["y_hat"]) for x in t["rows"])
            n_all += len(t["rows"])
            per[d]["n"].append(len(t["rows"]))
            depths[d].append(t["val_tuned_depth"])
            serve["tree"].append(t["serve_seconds_per_input"]["depth 3"])
            cost["tree_fit_queries"].append(t["queries_fit"])
            for arm, algo in (("rlda", "ddpg"), ("mada", "maddpg")):
                J = json.loads((RES / "paper_final" / "emp_tc0p10" / "results" / algo / f"{d}__{arm}__instances__seed{s}.json").read_text())
                serve[arm].append(J["pi"]["summary"]["all"]["wall_s_per_x"])
                if arm == "rlda" and J["anchors"].get("available"):
                    cost["anchors_q"].append(J["anchors"]["summary"]["queries_per_x"])
                    serve["anchors"].append(J["anchors"]["summary"]["wall_s_per_x"])
    V = {d: {k: float(np.mean(v)) for k, v in per[d].items()} for d in DATASETS}
    S = {d: {k: float(np.mean(v)) for k, v in stored[d].items()} for d in DATASETS}
    g = lambda m, q: float(np.mean([V[d].get(f"{m}|{q}", np.nan) for d in DATASETS]))  # noqa: E731

    out = ["## 4. Instance level\n",
           f"Same D_test points for every method: up to 400 per dataset and seed (all of them on the small datasets), seeds 42–46 "
           f"({n_all:,} points; 42–43 run on the Mac, 44–46 on spark). The explanation of x* is: RLDA / MADA — the box from one rollout of the "
           "policy for ŷ(x*), with the containment fix (π+, x* forced inside the box); Anchors — the classical anchor for x* (its rule parsed "
           "into a box with open faces); Tree — the leaf of the plain surrogate that contains x* (depth 3, depth 5, val-tuned). All scored with "
           "the same Anchors-style D(z|A) over D_train, 2,000 draws, **target ŷ(x*)** — if the tree's leaf is labelled with another class it "
           "explains the wrong prediction, and scores low. Coverage = share of D_test rows inside the box; conditions = active features.\n",
           "### 4.1 Overall (means over 12 datasets of 5-seed means)\n"]
    rows = []
    for m in M:
        rows.append([m, f3(g(m, "cond")), pct(g(m, "ok")), f3(g(m, "cov")), f"{g(m, 'act'):.1f}", pct(g(m, "contain")), f3(g(m, "emp")),
                     pct(g(m, "agree")) if m.startswith("Tree") else ("100% (by construction)" if m == "Anchors" else "100% (routed by ŷ)")])
    out.append(table(["Method", "Perturbation precision", "Share ≥ 0.90", "Coverage", "Conditions", "Contains x*", "Emp Fid of the box",
                      "Rule's class = ŷ(x*)"], rows))
    anc_self = []
    for d in DATASETS:
        v = []
        for s in SEEDS:
            r = json.loads((CF / f"{d}__rlda__seed{s}.json").read_text())["rows"]
            sp = [x["anchors"]["self_reported_precision"] for x in r if x.get("anchors") and x["anchors"].get("self_reported_precision") is not None]
            v.append(np.mean(sp))
        anc_self.append(np.mean(v))
    out.append(f"\nAnchors' self-reported precision (its own stopping estimate) averages {np.mean(anc_self):.3f}, against "
               f"{g('Anchors', 'cond'):.3f} measured with the shared estimator.\n")

    out.append("### 4.2 Paired tests (Wilcoxon over 12 datasets; Holm over the 8 contrasts per metric)\n")
    dm = lambda m, q: {d: V[d].get(f"{m}|{q}", np.nan) for d in DATASETS}  # noqa: E731
    pairs = [("Tree d3", "RLDA"), ("Tree d3", "MADA"), ("Tree d3", "Anchors"), ("Tree tuned", "RLDA"), ("Tree tuned", "MADA"),
             ("Tree tuned", "Anchors"), ("RLDA", "Anchors"), ("MADA", "Anchors")]
    out.append(table(["Contrast", "Pert precision Δ (first higher; p_Holm)", "Share ≥ 0.90 Δ", "Coverage Δ", "Conditions Δ"],
                     contrast_rows(pairs, dm, ["cond", "ok", "cov", "act"], {}), "|---|---|---|---|---|"))
    out.append("")

    out.append("### 4.3 Dataset-wise (5-seed means)\n")
    for q, lab, fmt in (("cond", "Perturbation precision", f3), ("ok", "Share of points with precision ≥ 0.90", pct),
                        ("cov", "Coverage (share of D_test rows in the box)", f3), ("act", "Conditions (active features)", lambda x: f"{x:.1f}"),
                        ("contain", "Contains x*", pct), ("emp", "Empirical Fid of the box on D_test", f3)):
        out.append(f"**{lab}**\n")
        rows = [[nm(d), str(int(sum(per[d]["n"])))] + [fmt(V[d].get(f"{m}|{q}", np.nan)) for m in M] for d in DATASETS]
        out.append(table(["Dataset", "Points"] + M, rows))
        out.append("")
    out.append("**Tree leaf labelled with ŷ(x*)**, and the val-tuned depth per seed\n")
    out.append(table(["Dataset", "Tree d3", "Tree d5", "Tree tuned", "Tuned depth (s42–46)"],
                     [[nm(d), pct(V[d]["Tree d3|agree"]), pct(V[d]["Tree d5|agree"]), pct(V[d]["Tree tuned|agree"]),
                       ",".join(str(x) for x in depths[d])] for d in DATASETS]))
    out.append("")

    out.append("### 4.4 Class-wise perturbation precision (points grouped by ŷ(x*), pooled over seeds)\n")
    rows = []
    for d in DATASETS:
        cls = sorted(c for (dd, c) in percls if dd == d)
        for c in cls:
            b = percls[(d, c)]
            rows.append([nm(d) if c == cls[0] else "", str(c), str(len(b["agree"]))] +
                        [f3(np.mean(b[m])) if b[m] else "—" for m in ("RLDA", "MADA", "Anchors", "Tree d3", "Tree tuned")] +
                        [pct(np.mean(b["agree"]))])
    out.append(table(["Dataset", "ŷ class", "Points", "RLDA", "MADA", "Anchors", "Tree d3", "Tree tuned", "Tree d3 leaf = ŷ"], rows))
    out.append("")

    out.append("### 4.5 The containment fix (RL, stored rollout π vs π+)\n")
    out.append("`utils.quantile_mdp.contain_point` widens the value bounds to include x* after each quantile-to-value sync "
               "(`enforce_instance_containment`, inference only, off by default). Without it the stored boxes often exclude x*, because a face "
               "at q* maps to the next class-c training value, not to x*.\n")
    gs = lambda m, q: float(np.mean([S[d][f"{m}|{q}"] for d in DATASETS]))  # noqa: E731
    rows = [[arm, pct(gs(f"{arm} π", "contain")), pct(g(arm, "contain")), f3(gs(f"{arm} π", "cond")), f3(g(arm, "cond")),
             f3(gs(f"{arm} π", "cov")), f3(g(arm, "cov"))] for arm in ("RLDA", "MADA")]
    out.append(table(["Arm", "Contains x* (π)", "Contains x* (π+)", "Pert precision (π)", "Pert precision (π+)", "Coverage (π)", "Coverage (π+)"], rows))
    out.append("\nDataset-wise containment of the stored rollout π: " + ", ".join(
        f"{nm(d)} {pct(S[d]['RLDA π|contain'])}/{pct(S[d]['MADA π|contain'])}" for d in DATASETS) + " (RLDA/MADA).\n")

    out.append("### 4.6 Cost per explained input\n")
    out.append(table(["Method", "f̂ queries per input", "Wall time per input", "One-time cost"],
                     [["Tree leaf", "1 (ŷ, for parity; the leaf itself needs none)", f"{np.mean(serve['tree']) * 1e3:.2f} ms",
                       f"fit: one f̂ query per D_train row (mean {np.mean(cost['tree_fit_queries']):,.0f}); about 0.5 s with the depth search"],
                      ["RLDA", "1 (ŷ, to route to the class policy)", f"{np.mean(serve['rlda']):.3f} s", "RL training (minutes to hours)"],
                      ["MADA", "1", f"{np.mean(serve['mada']):.3f} s", "RL training"],
                      ["Anchors", f"{np.mean(cost['anchors_q']):,.0f}", f"{np.mean(serve['anchors']):.2f} s", "none"]], "|---|---|---|---|"))
    out.append(f"\nŷ on the clipped unit inputs (used by RL and Anchors, and here) differs from ŷ on original units for {mism} of {n_all:,} points.\n")
    return "\n".join(out) + "\n"


def sec_modified_cart() -> str:
    p = RES / "paper_final_cart_fixed" / "RL_vs_CART_k1_k5.md"
    body = p.read_text().split("\n", 2)[2] if p.is_file() else "(missing)"
    body = body.replace("\nTable:", "\n**Table:**").replace("\nPaired tests", "\n**Paired tests**").replace(
        "\nDataset-wise Fidelity", "\n**Dataset-wise Fidelity")
    body = body.replace("(5-seed means)\n", "(5-seed means)**\n", 1)
    # A table must start its own block to render.
    body = "\n".join(line + "\n" if line.startswith("**") and not line.endswith("**") or line.startswith("**Dataset-wise") else line
                      for line in body.split("\n"))
    return ("## Appendix A. The modified CART (not a baseline)\n\n"
            "Kept for the record; these trees are no longer used as the baseline. 'CART (paper)' is the paper_final construction: "
            "`max_leaf_nodes = k·C` (a single split for a binary task at k = 1), leaf boxes clipped to the D_train range (on wine 31% of test rows "
            "fall outside, so coverage < 1 even for a partition), a class can end up with no rule (wyodot, every seed), then the RL selection "
            "pipeline (rank leaves on D_val, top-k per class, abstain elsewhere). The 'fixed' variants (`run_cart` default, `--cart_legacy` "
            "restores the paper tree) use a true partition with open faces and choose the tree size on D_val. All of them put a tree through "
            "RL's own selection pipeline, so they measure how much the candidate generator matters, not how RL compares with a standard "
            "surrogate.\n\n" + body + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RES / "paper_final_cart_fixed" / "GLOBAL_SURROGATE_RESULTS.md")
    args = ap.parse_args()
    E, Pd = load_dir(GS), load_dir(GSP)
    head = f"""# The unmodified CART global surrogate as the tree baseline

Generated {dt.date.today().isoformat()} by `python -m revision.global_surrogate_full_report` (branch `revision/global-surrogate-baseline`).
12 datasets × 5 seeds (42–46), f̂ = the paper's locked classifiers, RL rule sets from `paper_final_valtb` (conflicts tie-broken on D_val).
Every table is computed from the result files listed in Section 6; numbers are means over datasets of per-dataset seed means unless a
table says otherwise. Tests are paired Wilcoxon signed-rank over the 12 datasets.

## 1. Summary

**Baseline.** A single `DecisionTreeClassifier` fit to f̂(D_train), every leaf read as a rule, no leaf selection, no abstention — the
textbook global surrogate. It replaces the paper's modified CART, which put tree leaves through RL's own selection pipeline and was
handicapped (a stump for binary tasks, boxes clipped to the training range, classes without a rule); see Appendix A.

**Class level, empirical fidelity (Section 2).** A depth-3 tree (about 7 leaves of 2.5 conditions, the same rule length as RL) has
Fid 0.891 on every D_test row; RLDA has 0.903 on the 74% of rows it decides (MADA 0.890 on 80%). On RL's own decided rows the depth-3 tree
agrees with f̂ at least as often (0.921 vs 0.903, RL higher on 3/12 datasets), and from depth 5 on the tree is higher on almost every
dataset. RL's abstention does find hard rows: the depth-3 tree drops from 0.92 on the rows RL decides to 0.84 on the rows RL abstains on.
The tree's weakness is size: the validation-tuned tree has a median of 27 leaves but about 1,000 on uci adult and 5,500 on folktables, and
a depth-3 tree misses a class on wyodot and fails on housing (0.56).

**Class level, perturbation fidelity (Section 3).** Scored with Anchors' D(z|B), RL trained on empirical Fid is below a depth-3
tree on 10–11 of 12 datasets (row-weighted 0.662 / 0.671 vs 0.701; p_Holm 0.062–0.074 over 16 contrasts) and significantly below the
validation-tuned trees (p_Holm = 0.008). RLDA trained on perturbation Fid ties the depth-3 tree by row (0.710) and has the best mean over
rules of any method (0.710, above greedy Anchors' 0.691; ahead of the plain depth-3 tree on 10/12 datasets, p_Holm = 0.073, and of the plain
tuned tree, p_Holm = 0.039), at about 10⁸ training queries. Trees fit on recombined rows and tuned on D_val reach 0.809 by row.

**Instance level (Section 4).** The leaf of a depth-3 surrogate that contains x* is as precise under D(z|A) as RL's per-instance box
(0.654 vs RLDA 0.644 / MADA 0.639; p_Holm = 1.0) with fewer conditions (2.5 vs 3.6) and more coverage (0.382 vs 0.293); the tuned tree
(0.715) beats RLDA (10/12 datasets, p_Holm = 0.027). Anchors is the most precise (0.749; ahead of RLDA on 9/12 datasets, p_Holm = 0.056,
and of MADA on 10/12, p_Holm = 0.027). The tree's leaf is labelled with a class other than ŷ(x*) for 12.5% of inputs (43% on housing); RL never is, because it routes
by ŷ. Both RL and the tree explain an input with one f̂ query; Anchors needs about 15,000.

**What RL can still claim against the surrogate:** (i) per-class rules with an explicit region of abstention that picks up the hard
rows; (ii) an explanation that is always for the class f̂ actually predicted; (iii) the best per-rule robustness under perturbation when
trained on perturbation fidelity (paid in queries); (iv) a large cost advantage over Anchors. It cannot claim higher fidelity than the
surrogate at matched rule length, on either estimator or at either level, nor a cost advantage over the surrogate.

"""
    parts = [head, sec_empirical(E), sec_pert(Pd), sec_instance(), """## 5. Reading the results for the paper

- **Fidelity:** no win for RL over the plain surrogate at matched rule length, class or instance level, empirical or perturbation
  fidelity. The only lead is RLDA trained on perturbation Fid, on the unweighted mean over rules: significant after correction only
  against the plain tuned tree (Section 3.2), whose many small leaves pull its per-rule mean down.
- **Coverage:** the surrogate covers every row; RL covers 74–88% (class level) and slightly less than a depth-3 leaf per instance.
- **Size:** the surrogate becomes unreadable where it is most faithful (folktables, adult, wyodot, housing need hundreds to thousands of
  leaves); RL keeps one to three short rules per class on those datasets, with abstention.
- **Class alignment:** the surrogate's leaf explains a class other than ŷ(x*) for one input in eight; RL explanations are always for ŷ(x*).
- **Cost:** RL and the surrogate both serve an explanation with one f̂ query; the surrogate is also cheaper to build. RL's cost advantage
  is over Anchors only.

## 6. Reproduction

| Step | Command | Output |
|---|---|---|
| Class level, empirical | `python -m revision.cart_global_surrogate` | `results/paper_final_cart_fixed/global_surrogate/` |
| Class level, perturbation | `python -m revision.cart_global_surrogate_pert` | `results/paper_final_cart_fixed/global_surrogate_pert/` |
| Instance level, RL + Anchors | `python -m revision.containment_eval --seeds 42 43` (Mac); `bash revision/run_instance_spark.sh` (spark, 44–46) | `results/containment_fix/` |
| Instance level, tree | `python -m revision.tree_instance_eval` (and `--seeds 44 45 46` on spark) | `results/containment_fix/tree/` |
| Per-part summaries | `revision.global_surrogate_report`, `global_surrogate_pert_report`, `instance_surrogate_report`, `containment_report` | `SUMMARY.md` in each folder |
| This file | `python -m revision.global_surrogate_full_report` | `results/paper_final_cart_fixed/GLOBAL_SURROGATE_RESULTS.md` |

Code is on branch `revision/global-surrogate-baseline` (not merged into `main`).
""", sec_modified_cart()]
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
