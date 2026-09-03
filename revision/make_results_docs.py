"""Regenerate docs/RESULTS_comparison.md and docs/RULES.md from sweep result JSON.

Multi-seed aware: every metric cell carries one value per seed plus the mean, so
a dataset's DNN and RandomForest tables can be read side by side and a seed that
has not finished shows as an em dash rather than silently shifting the mean.

    python -m revision.make_results_docs [--seeds 42 43]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
from collections import defaultdict

ROOTS = [("dnn", "runs/sweep_dnn"), ("rf", "runs/sweep_rf")]
DATASETS = ["iris", "wine", "breast_cancer", "synthetic", "housing", "uci_credit", "uci_adult"]
METHODS = ["mada", "rlda", "cart", "greedy_anchors", "sp_anchors", "random_search"]
RL_METHODS = {"mada", "rlda"}
BACKEND_LABEL = {"dnn": "DNN", "rf": "RandomForest"}

FNAME = re.compile(r"^(?P<ds>.+?)__(?P<method>[a-z_]+)__seed(?P<seed>\d+)__tp")


def load_all(seeds):
    """(backend, dataset, method, seed) -> parsed result dict."""
    out = {}
    for backend, root in ROOTS:
        for path in glob.glob(os.path.join(root, "results", "*", "*.json")):
            m = FNAME.match(os.path.basename(path))
            if not m or "__instances__" in path:
                continue
            seed = int(m["seed"])
            if seed not in seeds:
                continue
            with open(path) as fh:
                out[(backend, m["ds"], m["method"], seed)] = json.load(fh)
    return out


# ---------------------------------------------------------------- metric access

def metric(res, name):
    """Pull one scalar out of a result dict, or None when it does not apply."""
    if res is None:
        return None
    g = res.get("global_ruleset") or {}
    if name == "fid":
        return g.get("global_fidelity")
    if name == "cov":
        return g.get("coverage")
    if name == "conflict":
        return g.get("conflict_rate")
    if name == "abstain":
        return g.get("abstention_rate")
    if name == "success":
        sr = res.get("success_rate")
        return sr.get("success_rate") if isinstance(sr, dict) else None
    if name == "queries":
        return (res.get("queries") or {}).get("n_blackbox_queries")
    if name == "train_queries":
        return (res.get("queries") or {}).get("n_training_queries")
    raise KeyError(name)


def method_label(method):
    return {"mada": "MADA", "rlda": "RLDA", "cart": "CART"}.get(method, method)


def cell(values, seeds, fmt="{:.3f}", mean=True):
    """Headline cell: mean over completed seeds, with the spread when n > 1.

    Seeds that have not finished are excluded from the mean rather than counted
    as zero, and the seed count is carried alongside so a mean over two seeds is
    never mistaken for a mean over five.
    """
    have = [values[s] for s in seeds if values.get(s) is not None]
    if not have:
        return "—"
    m = sum(have) / len(have)
    if not mean or len(have) == 1:
        return fmt.format(m)
    var = sum((v - m) ** 2 for v in have) / (len(have) - 1)
    return f"{fmt.format(m)} ±{fmt.format(var ** 0.5)}"


def per_seed_cell(values, seeds, fmt="{:.3f}"):
    """Detail cell: one value per seed, in seed order, em dash where absent."""
    return " / ".join(
        fmt.format(values[s]) if values.get(s) is not None else "—" for s in seeds
    )


def n_seeds(results, seeds):
    return sum(1 for s in seeds if results.get(s) is not None)


def success_cell(results, seeds):
    parts = []
    for s in seeds:
        r = results.get(s)
        sr = (r or {}).get("success_rate")
        if not isinstance(sr, dict):
            parts.append("—")
        else:
            parts.append(f"{sr['success_rate']:.3f} ({sr['n_success']}/{sr['n_episodes']})")
    return " / ".join(parts)


# ---------------------------------------------------------------- results doc

def results_doc(data, seeds, stamp, branch):
    seed_list = " / ".join(f"seed {s}" for s in seeds)
    L = [
        "# Results: MADA vs RLDA vs Baselines",
        "",
        f"Generated {stamp}. Branch `{branch}`.",
        "",
        "Tables are grouped **by dataset**, with the DNN and RandomForest black boxes",
        "adjacent, so the effect of swapping the explained model is read top-to-bottom",
        "within one dataset.",
        "",
        "Each dataset gets a **mean ±sd** summary table per black box, followed by the",
        f"raw per-seed values in **{seed_list}** order. An em dash marks a seed that has",
        "not been run yet — it is excluded from the mean rather than counted as zero,",
        "and the `seeds` column says how many actually contributed, so a mean over two",
        "seeds is never mistaken for a mean over five.",
        "",
        "The sd is the sample standard deviation across seeds. With three seeds it is a",
        "crude spread estimate, not a confidence interval, and a difference smaller than",
        "the sd should not be reported as an effect.",
        "",
        "## What was run",
        "",
        "| | |",
        "|---|---|",
        "| Arms | MADA (MADDPG), RLDA (DDPG), each against two black boxes |",
        "| Black boxes | **DNN** (`runs/sweep_dnn/`) and **RandomForest** (`runs/sweep_rf/`) |",
        f"| Datasets | {', '.join(DATASETS)} |",
        "| Excluded | covtype, folktables (too large for the time budget) |",
        f"| Seeds | {', '.join(str(s) for s in seeds)} |",
        "| Budgets | MADA 144,000 frames/agent · RLDA 270,000 total steps |",
        "| Baselines | CART, greedy_anchors, sp_anchors, random_search |",
        "| Selection | validation split, greedy marginal-gain union (k≤5); reporting on test |",
        "| τ_P / τ_C | 0.90 / 0.10 |",
        "",
        "Both arms of a given (dataset, black box) load the **same classifier file**, so MADA",
        "and RLDA always explain an identical model.",
        "",
        "### Coverage of this sweep",
        "",
    ]
    L += coverage_matrix(data, seeds)
    L += [
        "",
        "### Metric directions",
        "",
        "| Metric | Meaning | Better |",
        "|---|---|---|",
        "| **Fid** | P(rule's class = black-box prediction \\| covered) | higher |",
        "| **Cov** | fraction of test rows some rule fires on | higher |",
        "| **Conflict** | fraction of rows covered by rules of >1 class | **lower** |",
        "| **Abstain** | fraction of rows no rule covers | **lower** |",
        "| **Success** | episodes reaching τ_P and τ_C / episodes attempted | higher |",
        "| **Extraction queries** | black-box calls to BUILD the rule set — *not* training | lower |",
        "",
        "`Cov` and `Abstain` sum to 1. Success applies only to the RL arms — baselines are",
        "not episodic and report `—`, not 0. See **Query accounting** below before",
        "quoting any cost number.",
        "",
        "---",
        "",
        "## Per-dataset results",
        "",
    ]

    for ds in DATASETS:
        L += [f"### {ds}", ""]
        for backend, _ in ROOTS:
            L += [f"**{BACKEND_LABEL[backend]} black box** — mean ±sd over completed seeds", ""]
            L += [
                "| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |",
                "|---|---|---|---|---|---|---|---|",
            ]
            rows = []
            for method in METHODS:
                res = {s: data.get((backend, ds, method, s)) for s in seeds}
                if not any(res.values()):
                    continue
                rows.append((method, res))
                vals = lambda n: {s: metric(res[s], n) for s in seeds}
                succ = cell({s: metric(res[s], "success") for s in seeds}, seeds) \
                    if method in RL_METHODS else "—"
                L.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        method_label(method),
                        n_seeds(res, seeds),
                        cell(vals("fid"), seeds),
                        cell(vals("cov"), seeds),
                        cell(vals("conflict"), seeds),
                        cell(vals("abstain"), seeds),
                        succ,
                        cell(vals("queries"), seeds, fmt="{:,.0f}"),
                    )
                )
            if not rows:
                L += ["| _not run yet_ | 0 | — | — | — | — | — | — |", ""]
                continue
            L += ["",
                  f"_per seed ({' / '.join(str(s) for s in seeds)}):_",
                  "",
                  "| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |",
                  "|---|---|---|---|---|---|---|"]
            for method, res in rows:
                vals = lambda n: {s: metric(res[s], n) for s in seeds}
                succ = per_seed_cell({s: metric(res[s], "success") for s in seeds}, seeds) \
                    if method in RL_METHODS else "—"
                L.append(
                    "| {} | {} | {} | {} | {} | {} | {} |".format(
                        method_label(method),
                        per_seed_cell(vals("fid"), seeds),
                        per_seed_cell(vals("cov"), seeds),
                        per_seed_cell(vals("conflict"), seeds),
                        per_seed_cell(vals("abstain"), seeds),
                        succ,
                        per_seed_cell(vals("queries"), seeds, fmt="{:,.0f}"),
                    )
                )
            L.append("")
        L.append("")

    L += head_to_head(data, seeds)
    L += query_accounting(data, seeds)
    return "\n".join(L) + "\n"


def coverage_matrix(data, seeds):
    """A quick map of which (backend, seed, dataset) legs actually have results."""
    L = ["| black box | seed | " + " | ".join(DATASETS) + " |",
         "|---|---|" + "---|" * len(DATASETS)]
    for backend, _ in ROOTS:
        for s in seeds:
            marks = []
            for ds in DATASETS:
                have = {m for m in METHODS if data.get((backend, ds, m, s))}
                if RL_METHODS <= have:
                    marks.append("✅")
                elif have & RL_METHODS:
                    marks.append("◐")
                else:
                    marks.append("—")
            L.append(f"| {BACKEND_LABEL[backend]} | {s} | " + " | ".join(marks) + " |")
    L += ["", "✅ both arms · ◐ one arm only · — not run"]
    return L


def head_to_head(data, seeds):
    """MADA vs RLDA per metric, counted over datasets, one block per (backend, seed)."""
    L = ["---", "", "## MADA vs RLDA head-to-head", "",
         "Counted over datasets where both arms finished. `W` = MADA better,",
         "`L` = RLDA better, `T` = tie to three decimals.", ""]
    specs = [("Fid", "fid", True), ("Cov", "cov", True),
             ("Conflict", "conflict", False), ("Abstain", "abstain", False)]

    L += ["### Pooled over all (dataset, seed) pairs", "",
          "The row to quote: one count per black box over every dataset-seed pair where",
          "both arms finished. Less seed-sensitive than the per-seed rows below.", "",
          "| black box | pairs | " + " | ".join(n for n, _, _ in specs) + " |",
          "|---|---|" + "---|" * len(specs)]
    for backend, _ in ROOTS:
        cells, npairs = [], 0
        for _, key, higher_better in specs:
            w = l = t = 0
            for s in seeds:
                for ds in DATASETS:
                    a = metric(data.get((backend, ds, "mada", s)), key)
                    b = metric(data.get((backend, ds, "rlda", s)), key)
                    if a is None or b is None:
                        continue
                    if round(a, 3) == round(b, 3):
                        t += 1
                    elif (a > b) == higher_better:
                        w += 1
                    else:
                        l += 1
            npairs = max(npairs, w + l + t)
            cells.append("—" if w + l + t == 0 else f"{w}W/{l}L" + (f"/{t}T" if t else ""))
        L.append(f"| {BACKEND_LABEL[backend]} | {npairs} | " + " | ".join(cells) + " |")

    L += ["", "### By seed", ""]
    L += ["| black box | seed | " + " | ".join(n for n, _, _ in specs) + " |",
          "|---|---|" + "---|" * len(specs)]
    for backend, _ in ROOTS:
        for s in seeds:
            cells = []
            for _, key, higher_better in specs:
                w = l = t = 0
                for ds in DATASETS:
                    a = metric(data.get((backend, ds, "mada", s)), key)
                    b = metric(data.get((backend, ds, "rlda", s)), key)
                    if a is None or b is None:
                        continue
                    if round(a, 3) == round(b, 3):
                        t += 1
                    elif (a > b) == higher_better:
                        w += 1
                    else:
                        l += 1
                cells.append("—" if w + l + t == 0 else f"{w}W/{l}L" + (f"/{t}T" if t else ""))
            L.append(f"| {BACKEND_LABEL[backend]} | {s} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def query_accounting(data, seeds):
    L = ["---", "", "## Query accounting", "",
         "`extraction queries` counts black-box calls made to **build** the rule set:",
         "rule generation plus validation-split selection. It excludes policy training,",
         "which is reported separately below, and excludes held-out test reporting,",
         "which is instrumentation rather than a cost of producing an explanation.",
         "",
         "Serving an explanation from an already-extracted box costs **0** queries in",
         "both RL arms — that is the amortisation claim the paper makes.",
         "",
         "### Training queries (RL arms only)",
         "",
         "| black box | dataset | " + " | ".join(f"MADA s{s} | RLDA s{s}" for s in seeds) + " |",
         "|---|---|" + "---|" * (2 * len(seeds))]
    for backend, _ in ROOTS:
        for ds in DATASETS:
            row = []
            for s in seeds:
                for m in ("mada", "rlda"):
                    v = metric(data.get((backend, ds, m, s)), "train_queries")
                    row.append("—" if v is None else f"{v:,}")
            if all(c == "—" for c in row):
                continue
            L.append(f"| {BACKEND_LABEL[backend]} | {ds} | " + " | ".join(row) + " |")
    L += ["",
          "> **Caveat.** MADA's training-query count is a lower bound: the torchrl",
          "> collector holds copies of the environment whose cache hits are not all",
          "> attributed back to the parent counter. RLDA's count is exact.",
          "",
          "> **Open item.** RLDA reports roughly 5x MADA's extraction queries on every",
          "> dataset, independent of class count, and `uci_adult` is far above CART for",
          "> both arms. This ratio is not yet explained and should not be quoted as a",
          "> headline cost result until it is.",
          ""]
    return L


# ------------------------------------------------------------------ rules doc

def rules_doc(data, seeds, stamp, branch):
    L = [
        "# Extracted Rules",
        "",
        f"Generated {stamp}. Branch `{branch}`.",
        "",
        "The actual rule sets behind the numbers in `RESULTS_comparison.md`.",
        "Rules are the **validation-selected union**, scored on the **test** split.",
        "`k` is how many rules the marginal-gain selector kept for that class.",
        "",
        "Seeds are listed adjacently under each class so a rule's stability across",
        "seeds can be read directly. A class that changes its selected feature between",
        "seeds is telling you something about the policy, not about the dataset.",
        "",
        "Rule strings are printed in full — never truncated — because a truncated box",
        "is a different box, and the repository has been bitten by that before.",
        "",
        "Fid/Cov below are **per rule** on the test split; the class union's own",
        "Fid/Cov are in the comparison document, and the union is not the average of",
        "its members.",
        "",
        "---",
        "",
    ]
    for ds in DATASETS:
        L += [f"## {ds}", ""]
        for backend, _ in ROOTS:
            L += [f"### {ds} — {BACKEND_LABEL[backend]} black box", ""]
            wrote_any = False
            for method in METHODS:
                res = {s: data.get((backend, ds, method, s)) for s in seeds}
                if not any(res.values()):
                    continue
                wrote_any = True
                label = "MADA" if method == "mada" else "RLDA" if method == "rlda" else method
                L += [f"**{label}**", ""]
                classes = sorted(
                    {c for r in res.values() if r for c in (r.get("per_class") or {})},
                    key=lambda c: int(c.split("_")[-1]) if c.split("_")[-1].isdigit() else 0,
                )
                for cls in classes:
                    L.append(f"- `{cls}`")
                    for s in seeds:
                        r = res.get(s)
                        pc = ((r or {}).get("per_class") or {}).get(cls)
                        if pc is None:
                            L.append(f"  - seed {s}: _not run_")
                            continue
                        rules = pc.get("selected_rules") or []
                        L.append(f"  - seed {s} (k={pc.get('k', len(rules))})")
                        if not rules:
                            L.append("    - _no rule selected_")
                        for i, rule in enumerate(rules, 1):
                            rm = rule.get("report_metrics") or {}
                            fid, cov = rm.get("fidelity"), rm.get("coverage")
                            n = rm.get("n_covered")
                            bits = []
                            if fid is not None:
                                bits.append(f"Fid {fid:.3f}")
                            if cov is not None:
                                bits.append(f"Cov {cov:.3f}")
                            if n is not None:
                                bits.append(f"n={n}")
                            tail = ("  — " + ", ".join(bits)) if bits else ""
                            L.append(f"    {i}. {rule.get('display_rule', '(no display string)')}{tail}")
                    L.append("")
                L.append("")
            if not wrote_any:
                L += ["_not run yet_", ""]
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--results-out", default="docs/RESULTS_comparison.md")
    ap.add_argument("--rules-out", default="docs/RULES.md")
    args = ap.parse_args()

    seeds = sorted(args.seeds)
    data = load_all(set(seeds))
    if not data:
        raise SystemExit("no result JSON found for the requested seeds")

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        branch = "unknown"

    with open(args.results_out, "w") as fh:
        fh.write(results_doc(data, seeds, stamp, branch))
    with open(args.rules_out, "w") as fh:
        fh.write(rules_doc(data, seeds, stamp, branch))

    legs = defaultdict(int)
    for (backend, _ds, method, seed) in data:
        legs[(backend, seed)] += 1
    print(f"wrote {args.results_out} and {args.rules_out}")
    for k in sorted(legs):
        print(f"  {k[0]} seed{k[1]}: {legs[k]} result files")


if __name__ == "__main__":
    main()
