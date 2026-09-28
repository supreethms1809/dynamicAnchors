"""Review report for the fixed CART baseline (revision/run_cart_fixed.sh).

Compares, per dataset (5-seed means), the paper_final CART ("legacy": k x n_classes
leaves, D_train-range boxes) with the fixed CART (true partition, tree size on
D_val, two size criteria), next to RLDA / MADA. All cells use the D_val conflict
tie-break (legacy / RL from ../results/paper_final_valtb). Writes
../results/paper_final_cart_fixed/REVIEW.md.

    python -m revision.cart_fixed_report
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402

VALTB = REPO.parent / "results" / "paper_final_valtb"
FIXED = REPO.parent / "results" / "paper_final_cart_fixed"
SEEDS = [42, 43, 44, 45, 46]
CRITERIA = ("precision_constrained", "effectiveness")
SHORT = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def cell_paths(est: str, tc: str, ds: str, seed: int):
    name = lambda m: f"{ds}__{m}__seed{seed}__tp0p90__{tc}.json"  # noqa: E731
    grid = VALTB / f"{est}_{tc}" / "results"
    out = {
        "rlda": grid / "ddpg" / name("rlda"),
        "mada": grid / "maddpg" / name("mada"),
        "cart legacy": VALTB / f"baselines_{est}" / name("cart"),
    }
    for c in CRITERIA:
        out[f"cart {c}"] = FIXED / c / f"baselines_{est}" / name("cart")
    return out


def summary(cell) -> dict:
    g = cell["global_ruleset"]
    fid = g.get("global_fidelity")
    comp = cell.get("compactness") or {}
    ex = cell.get("extra") or {}
    return {
        "fid": np.nan if fid is None else fid,
        "cov": g["coverage"],
        "eff": g.get("effectiveness") or 0.0,
        "rules": comp.get("mean_rules_per_class", np.nan) * max(1, len(cell.get("per_class") or {})),
        "feat": comp.get("mean_active_features") if comp.get("mean_active_features") is not None else np.nan,
        "L": ex.get("max_leaf_nodes", np.nan),
        "leaves": ex.get("n_leaves", np.nan),
        "n_classes_ruled": len(cell.get("per_class") or {}),
    }


def merge_path(rule: str) -> str:
    """CART display rules are raw split paths ("a > 1 and a > 3"); merge each
    feature into one interval for reading. Other rule text is returned as is."""
    import re

    parts = rule.split(" and ")
    lo, hi, order = {}, {}, []
    for part in parts:
        m = re.match(r"^(.*) (<=|>) (\S+)$", part)
        if not m:
            return rule
        name, op, v = m.group(1), m.group(2), float(m.group(3))
        if name not in order:
            order.append(name)
        if op == ">":
            lo[name] = max(lo.get(name, -np.inf), v)
        else:
            hi[name] = min(hi.get(name, np.inf), v)
    out = []
    for n in order:
        a, b = lo.get(n), hi.get(n)
        out.append(f"{a:.4g} < {n} <= {b:.4g}" if a is not None and b is not None
                   else (f"{n} > {a:.4g}" if a is not None else f"{n} <= {b:.4g}"))
    return " and ".join(out)


def fmt(v, p=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"


def table(lines, head, rows):
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    lines.append("")


def main() -> int:
    L = []
    L += ["# Fixed CART baseline: review", ""]
    L += [
        "`revision.baselines.run_cart` after the fix, re-run for the paper_final grid "
        "(`revision/run_cart_fixed.sh`, seeds 42-46, 12 datasets, k = 1, τ_P = 0.90). "
        "These results are kept apart from the paper numbers, in `../results/paper_final_cart_fixed/<criterion>/baselines_{emp,pert}/`.",
        "",
        "What changed from the paper's CART (`--cart_legacy` reproduces it, checked on iris, wine and uci_adult seed 42):",
        "",
        "1. **True partition.** Leaf boxes are the tree's own cells, with open outer faces and a strict `>` on the right child. "
        "The old boxes stopped at D_train's min/max, so held-out rows beyond that range fell in no leaf. On wine that was 31% of test rows.",
        "2. **Tree size chosen on D_val.** The old size was `max_leaf_nodes = k × n_classes`, which is a single split for a binary task at k = 1. "
        "Now L ∈ {1, 2, 5, 10, 20} × n_classes, and the choice looks only at the rule set each L would report on D_val:",
        "   - `precision_constrained`: the largest D_val coverage among sizes whose D_val Fid ≥ τ_P. If none reaches τ_P, the highest D_val Fid. "
        "This is the objective the RL arms and Anchors pursue.",
        "   - `effectiveness`: the best D_val Fid × coverage. At k = 1 this almost always keeps the smallest tree.",
        "3. Leaf ranking and selection are unchanged: per class, the top-k leaves by Wilson LCB(Fid) × (1 + class coverage) on D_val, reported on D_test.",
        "",
        "Coverage is global coverage, the share of D_test rows decided. Conditions per rule = mean active features of the selected rules. "
        "Wilcoxon tests are paired over 12 datasets (5-seed means).",
        "",
    ]

    # ---- validity -------------------------------------------------------
    n_have = collections.Counter()
    bad_partition, missing_cls = [], collections.Counter()
    for c in CRITERIA:
        for est in ("emp", "pert"):
            for tc in ("tc0p10", "tc0p20"):
                for ds in DATASETS:
                    for s in SEEDS:
                        cell = load(cell_paths(est, tc, ds, s)[f"cart {c}"])
                        if cell is None:
                            continue
                        n_have[c] += 1
                        ex = cell["extra"]
                        chosen = [x for x in ex["cart_candidates"] if x["max_leaf_nodes"] == ex["max_leaf_nodes"]][0]
                        if chosen["val_rows_in_no_leaf"] or chosen["val_rows_in_two_leaves"] or chosen["val_box_vs_apply_mismatches"]:
                            bad_partition.append(f"{c} {est} {tc} {ds} s{s}")
                        if chosen["classes_without_leaf"]:
                            missing_cls[(c, est, SHORT.get(ds, ds))] += 1
    L += ["## Validity", ""]
    L += [f"- Cells present: " + ", ".join(f"{c} {n_have[c]}/240" for c in CRITERIA) + "."]
    L += [f"- Partition check on D_val for the chosen tree (rows in no leaf, in two leaves, box vs `tree.apply` mismatches): "
          f"{'all zero' if not bad_partition else 'FAILED: ' + '; '.join(bad_partition[:10])}."]
    L += ["- Chosen trees with a class that has no leaf (that class gets no rule): "
          + ("none" if not missing_cls else ", ".join(f"{d} ({c}, {e}): {n} cells" for (c, e, d), n in sorted(missing_cls.items()))) + "."]
    L += [""]

    # ---- summaries -------------------------------------------------------
    methods = ["rlda", "mada", "cart legacy"] + [f"cart {c}" for c in CRITERIA]
    for est in ("emp", "pert"):
        for tc in ("tc0p10", "tc0p20"):
            per = {m: collections.defaultdict(list) for m in methods}
            for ds in DATASETS:
                for s in SEEDS:
                    for m, p in cell_paths(est, tc, ds, s).items():
                        cell = load(p)
                        if cell is not None:
                            per[m][ds].append(summary(cell))
            if not any(per[f"cart {c}"] for c in CRITERIA):
                continue
            L += [f"## {est}, τ_C = 0.{tc[-2:]}", ""]
            rows = []
            for m in methods:
                ds_means = {ds: {k: np.nanmean([x[k] for x in xs]) for k in xs[0]} for ds, xs in per[m].items() if xs}
                if not ds_means:
                    continue
                mean = lambda k: np.nanmean([v[k] for v in ds_means.values()])  # noqa: E731
                rows.append([m, str(len(ds_means)), fmt(mean("fid")), fmt(mean("cov")), fmt(mean("eff")),
                             fmt(mean("rules"), 1), fmt(mean("feat"), 1),
                             fmt(mean("leaves"), 1) if m.startswith("cart") else "—"])
            table(L, ["method", "datasets", "Fid", "Cov", "Eff", "rules", "conditions / rule", "tree leaves"], rows)

            tests = []
            for rl in ("rlda", "mada"):
                for m in ["cart legacy"] + [f"cart {c}" for c in CRITERIA]:
                    common = [ds for ds in DATASETS if per[rl][ds] and per[m][ds]]
                    for k in ("fid", "cov", "eff"):
                        a = [np.nanmean([x[k] for x in per[rl][ds]]) for ds in common]
                        b = [np.nanmean([x[k] for x in per[m][ds]]) for ds in common]
                        w = paired_wilcoxon(a, b)
                        tests.append([rl.upper(), m, k.capitalize(),
                                      f"{sum(x > y for x, y in zip(a, b))}/{len(common)}",
                                      f"{np.mean(np.array(a) - np.array(b)):+.3f}", f"{w['pvalue']:.4f}"])
            table(L, ["RL arm", "vs", "metric", "RL higher", "mean Δ (RL − CART)", "p"], tests)

            if est == "emp" and tc == "tc0p10":
                L += ["### Dataset-wise (emp, τ_C = 0.10, 5-seed mean): Fid / Cov", ""]
                rows = []
                for ds in DATASETS:
                    row = [SHORT.get(ds, ds)]
                    for m in methods:
                        xs = per[m][ds]
                        row.append(f"{fmt(np.nanmean([x['fid'] for x in xs]))} / {fmt(np.nanmean([x['cov'] for x in xs]))}" if xs else "—")
                    for c in CRITERIA:
                        xs = per[f"cart {c}"][ds]
                        row.append(", ".join(str(int(x["leaves"])) for x in xs) if xs else "—")
                    rows.append(row)
                table(L, ["dataset"] + methods + [f"leaves by seed ({c})" for c in CRITERIA], rows)

    # ---- Track B: Anchors-style conditional Fid -------------------------------
    tb = FIXED / "trackb_rules_emp_tc0p10.json"
    if tb.is_file():
        rows_tb = json.loads(tb.read_text())
        ms = ["rlda", "mada", "cart legacy", "cart precision_constrained", "cart effectiveness", "greedy_anchors"]
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for x in rows_tb:
            by[x["dataset"]][x["method"]].append(x)
        L += ["## Per-rule fidelity under Anchors' D(z|A) (emp, τ_C = 0.10)", "",
              "Each selected rule is re-scored with the Track B sampler (`revision.dual_estimator_rescore.Scorer`: "
              "Anchors' `sample_from_train` port over D_train, 2,000 draws). Violating coordinates are patched with "
              "train values that satisfy the rule, and the draw is labelled by f̂. Cells show the mean per-rule "
              "conditional Fid, with the share of rules ≥ τ_P = 0.90 in brackets (5 seeds).", ""]
        rows = []
        cols = {m: [] for m in ms}
        for ds in DATASETS:
            row = [SHORT.get(ds, ds)]
            for m in ms:
                xs = by[ds][m]
                v = np.nanmean([x["fid_anchors"] for x in xs]) if xs else np.nan
                sh = np.mean([x["fid_anchors"] >= 0.9 for x in xs]) if xs else np.nan
                cols[m].append(v)
                row.append(f"{fmt(v)} [{fmt(sh, 2)}]")
            rows.append(row)
        rows.append(["**mean**"] + [f"**{fmt(np.nanmean(cols[m]))}**" for m in ms])
        table(L, ["dataset"] + ms, rows)
        tests = []
        for a_, b_ in (("rlda", "cart precision_constrained"), ("mada", "cart precision_constrained"),
                       ("rlda", "cart legacy"), ("greedy_anchors", "cart precision_constrained")):
            w = paired_wilcoxon(cols[a_], cols[b_])
            tests.append([a_, b_, f"{sum(x > y for x, y in zip(cols[a_], cols[b_]))}/12",
                          f"{np.mean(np.array(cols[a_]) - np.array(cols[b_])):+.3f}", f"{w['pvalue']:.4f}"])
        table(L, ["A", "B", "A higher", "mean Δ (A − B)", "p"], tests)

    # ---- rules -----------------------------------------------------------
    L += ["## Rules, seed 42, emp τ_C = 0.10", "",
          "Each line: class, D_test Fid, D_test rows covered, rule. Legacy is the paper's CART.", ""]
    for ds in DATASETS:
        paths = cell_paths("emp", "tc0p10", ds, 42)
        L += [f"### {SHORT.get(ds, ds)}", ""]
        for m in ["cart legacy"] + [f"cart {c}" for c in CRITERIA] + ["rlda", "mada"]:
            cell = load(paths[m])
            if cell is None:
                continue
            g = cell["global_ruleset"]
            ex = cell.get("extra") or {}
            tag = (f" (max_leaf_nodes = {ex.get('max_leaf_nodes')}"
                   + (f", {ex['n_leaves']} leaves" if ex.get("n_leaves") else "") + ")") if m.startswith("cart") else ""
            L += [f"**{m}**{tag}: Fid {fmt(g.get('global_fidelity'))}, Cov {fmt(g['coverage'])}", ""]
            for key, b in sorted((cell.get("per_class") or {}).items()):
                for r in b.get("selected_rules") or []:
                    rm = r.get("report_metrics") or {}
                    rule = r.get("display_rule") or ""
                    rule = merge_path(rule) if m.startswith("cart") else rule
                    L += [f"- {key}: Fid {fmt(rm.get('fidelity'))}, n = {rm.get('n_covered')}: `{rule}`"]
            L += [""]

    (FIXED / "REVIEW.md").write_text("\n".join(L))
    print(f"wrote {FIXED / 'REVIEW.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
