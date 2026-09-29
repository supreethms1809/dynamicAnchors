"""Report for `revision.recompute_len`: Len under the new criterion, next to the old one.

    python -m revision.recompute_len --report > ../results/len_recompute/REPORT.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO.parent / "results" / "len_recompute"

from revision.paper_stats import DATASETS  # noqa: E402

# (label, group, method, path filter)
ROWS = [
    ("RLDA, per-policy + floor", "rl_perpolicy", "rlda", "emp_tc0p10"),
    ("MADA, per-policy OR + floor", "rl_perpolicy", "mada", "emp_tc0p10"),
    ("RLDA, paper cells (pooled)", "rl_paper", "rlda", "emp_tc0p10"),
    ("MADA, paper cells (pooled)", "rl_paper", "mada", "emp_tc0p10"),
    ("SP-Anchors, pool 20", "anchors_pool20", "sp_anchors", "tc0p10"),
    ("Greedy anchors, pool 20", "anchors_pool20", "greedy_anchors", "tc0p10"),
    ("SP-Anchors, pool 5", "anchors_pool5", "sp_anchors", "tc0p10"),
    ("Greedy anchors, pool 5", "anchors_pool5", "greedy_anchors", "tc0p10"),
    ("CART baseline (one leaf per class)", "cart_random", "cart", "tc0p10"),
    ("Random search", "cart_random", "random_search", "tc0p10"),
]


def load(group: str) -> List[dict]:
    p = OUT / f"{group}.jsonl"
    return [json.loads(l) for l in p.open()] if p.is_file() else []


def total_conditions(r: dict) -> int:
    return sum(x["n_new"] for c in r["per_class"] for x in c["rules"])


def by_dataset(cells: List[dict], key) -> Dict[str, float]:
    per = defaultdict(list)
    for r in cells:
        per[r["dataset"]].append(key(r))
    return {d: float(np.mean(v)) for d, v in per.items()}


def mean12(d: Dict[str, float]) -> float:
    vals = [d[x] for x in DATASETS if x in d]
    return float(np.mean(vals)) if len(vals) == len(DATASETS) else float("nan")


def seeds_of(cells) -> str:
    s = sorted({r["seed"] for r in cells})
    return f"{s[0]}–{s[-1]}" if s and s == list(range(s[0], s[-1] + 1)) and len(s) > 1 else ",".join(map(str, s))


def f2(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"


def main() -> int:
    groups = {g: load(g) for g in {r[1] for r in ROWS}}
    trees = [json.loads(l) for l in (OUT / "trees.jsonl").open()] if (OUT / "trees.jsonl").is_file() else []
    P = print
    P("# Rule length under the new criterion\n")
    P("A condition counts when it excludes at least one D_train row, decided in the space the box is "
      "scored in; the printed rule shows exactly those conditions (`utils/rule_print.py`). The old Len "
      "counted a feature whose interval was narrower than 95% of its D_train range. Len is averaged "
      "over the rules of each class, then over classes; tables average seeds within a dataset, then the "
      "12 datasets. Only Len and the printed rules change; no Fid, Cov or Eff number moves.\n")

    # --- checks
    P("## Checks\n")
    for g, cells in sorted(groups.items()):
        fails = sum(r["rebuild_failures"] for r in cells)
        mism = sum(r["old_stored_mismatch"] for r in cells)
        P(f"- `{g}`: {len(cells)} cells; boxes rebuilt to the stored D_val/D_test counts with {fails} "
          f"failures; the old count recomputed from the boxes differs from the stored one on {mism} rules.")
    if trees:
        mm = sum(t[k]["path_mismatch"] for t in trees for k in ("tree_d3", "tree_tuned"))
        lv = sum(int(t[k]["n_leaves"] != t[k]["stored_n_leaves"]) for t in trees for k in ("tree_d3", "tree_tuned"))
        P(f"- trees: {len(trees)} dataset-seed cells refit; leaves whose count differs from the number of "
          f"distinct features on their path: {mm}; trees whose leaf count differs from the stored one: {lv}.")
    P("")

    # --- headline
    P("## Len by method (emp, τ_C = 0.10, k = 1)\n")
    P("| Method | seeds | Len old (95% width) | Len new | Δ | rules whose count changed | conditions to read per dataset |")
    P("|---|---|---:|---:|---:|---:|---:|")
    ds_new: Dict[str, Dict[str, float]] = {}
    for label, g, m, filt in ROWS:
        cells = [r for r in groups[g] if r["method"] == m and filt in r["path"]]
        if not cells:
            continue
        new, old = mean12(by_dataset(cells, lambda r: r["len_new"])), mean12(by_dataset(cells, lambda r: r["len_old"]))
        tot = mean12(by_dataset(cells, total_conditions))
        rules = [x for r in cells for c in r["per_class"] for x in c["rules"]]
        ch = sum(x["n_new"] != x["n_old"] for x in rules)
        ds_new[label] = by_dataset(cells, lambda r: r["len_new"])
        P(f"| {label} | {seeds_of(cells)} | {f2(old)} | {f2(new)} | {new - old:+.2f} | "
          f"{ch}/{len(rules)} ({100 * ch / max(1, len(rules)):.0f}%) | {tot:.1f} |")
    for name, key in (("Tree, depth 3", "tree_d3"), ("Tree, tuned depth", "tree_tuned")):
        if not trees:
            continue
        per = defaultdict(list)
        per_leaf, tot = defaultdict(list), defaultdict(list)
        for t in trees:
            per[t["dataset"]].append(t[key]["len_class_mean"])
            per_leaf[t["dataset"]].append(t[key]["len_leaf_mean"])
            tot[t["dataset"]].append(t[key]["total_conditions"])
        d = {k: float(np.mean(v)) for k, v in per.items()}
        ds_new[name] = d
        leaf = mean12({k: float(np.mean(v)) for k, v in per_leaf.items()})
        P(f"| {name} | {seeds_of(trees)} | {f2(leaf)} (per leaf) | {f2(mean12(d))} | — | "
          f"— (already this criterion) | {mean12({k: float(np.mean(v)) for k, v in tot.items()}):.1f} |")
    P("\n“Len old” for the trees is the stored mean over leaves (distinct features on the path, which "
      "is this criterion); “Len new” averages over each class's leaves, then over classes, as for the "
      "rule sets. Conditions to read per dataset: all conditions of all rules (tree: all leaves).\n")

    # --- per grid for RL
    P("## RLDA and MADA, every grid (per-policy selection, floor 0.60)\n")
    P("| grid | seeds | RLDA old | RLDA new | MADA old | MADA new |")
    P("|---|---|---:|---:|---:|---:|")
    for grid in ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20"):
        vals = []
        seeds = ""
        for m in ("rlda", "mada"):
            cells = [r for r in groups["rl_perpolicy"] if r["method"] == m and f"/{grid}/" in r["path"]]
            seeds = seeds_of(cells)
            vals += [mean12(by_dataset(cells, lambda r: r["len_old"])), mean12(by_dataset(cells, lambda r: r["len_new"]))]
        P(f"| {grid} | {seeds} | " + " | ".join(f2(v) for v in vals) + " |")
    P("")

    # --- dataset-wise
    labels = [l for l in ds_new]
    P("## Len new by dataset (emp, τ_C = 0.10)\n")
    P("| dataset | " + " | ".join(labels) + " |")
    P("|---|" + "---:|" * len(labels))
    for d in DATASETS:
        P(f"| {d} | " + " | ".join(f2(ds_new[l].get(d)) for l in labels) + " |")
    P("")

    # --- class-wise for the headline rule sets
    P("## Len new by class (emp, τ_C = 0.10), seed mean\n")
    heads = ROWS[:2] + ROWS[4:6]
    P("| dataset | class | " + " | ".join(h[0] for h in heads) + " |")
    P("|---|---:|" + "---:|" * len(heads))
    table = defaultdict(lambda: defaultdict(list))
    for label, g, m, filt in heads:
        for r in groups[g]:
            if r["method"] == m and filt in r["path"]:
                for c in r["per_class"]:
                    table[(r["dataset"], c["cls"])][label].append(np.mean([x["n_new"] for x in c["rules"]]))
    for d in DATASETS:
        for cls in sorted({k[1] for k in table if k[0] == d}):
            row = table[(d, cls)]
            P(f"| {d} | {cls} | " + " | ".join(f2(np.mean(row[h[0]])) if row[h[0]] else "—" for h in heads) + " |")
    P("")

    # --- instances
    inst = [json.loads(l) for l in (OUT / "instances.jsonl").open()] if (OUT / "instances.jsonl").is_file() else []
    if inst:
        P("## Instance level: conditions per explanation\n")
        P("π: the stored rollout box; π+: with the containment fix; π∨: the OR of the policies' boxes "
          "(summed over its boxes); A: Anchors (exact bins). Mean over 12 datasets of the seed means.\n")
        P("| grid | arm | seeds | π old/new | π+ old/new | π∨ old/new | Anchors old/new |")
        P("|---|---|---|---|---|---|---|")
        for grid in ("emp", "pert"):
            for arm in ("rlda", "mada"):
                cells = [r for r in inst if r["arm"] == arm and f"_{grid}/" in r["path"]]
                if not cells:
                    continue
                parts = []
                for t in ("pi", "pi_contained", "pi_or", "anchors"):
                    o = mean12(by_dataset([c for c in cells if f"{t}|old" in c], lambda r: r[f"{t}|old"]))
                    n = mean12(by_dataset([c for c in cells if f"{t}|new" in c], lambda r: r[f"{t}|new"]))
                    parts.append(f"{f2(o)} / {f2(n)}")
                P(f"| {grid} | {arm.upper()} | {seeds_of(cells)} | " + " | ".join(parts) + " |")
        P("")

    # --- example rules
    P("## Printed rules, seed 42 (emp, τ_C = 0.10, per-policy selection with the floor)\n")
    for ds in ("iris", "wyodot"):
        for m in ("rlda", "mada"):
            cells = [r for r in groups["rl_perpolicy"]
                     if r["dataset"] == ds and r["method"] == m and r["seed"] == 42 and "/emp_tc0p10/" in r["path"]]
            for r in cells:
                P(f"**{ds}, {m.upper()}**\n")
                for c in r["per_class"]:
                    for x in c["rules"]:
                        P(f"- class {c['cls']}: `{x['printed']}` ({x['n_new']} conditions; old count {x['n_old']})")
                        P(f"  - stored string: `{x['stored']}`")
                P("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
