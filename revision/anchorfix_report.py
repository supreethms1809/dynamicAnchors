#!/usr/bin/env python3
"""SP-/greedy-Anchors before and after the exact-bins fix, against the paper cells.

Old anchor-family cells and every other method come from ../results/paper_final_valtb
(the D_val tie-break re-score); new anchor-family cells from
../results/paper_final_anchorfix (`revision/rerun_anchor_family_exactbins.py`).
Only seeds present in the new cells are used, for old and new alike.

Coverage is the global rule-set coverage on D_test (share of test rows some class
union fires on) unless a table says class-wise, where it is the class union's
coverage P(x in B | f_hat(x) = c).

  python -m revision.anchorfix_report > ../results/paper_final_anchorfix/REPORT.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS, boot_ci_paired_diff, holm  # noqa: E402
from utils.metrics import paired_wilcoxon, track_a_cov_tau, track_a_eff  # noqa: E402

VALTB = REPO.parent / "results" / "paper_final_valtb"
NEW = REPO.parent / "results" / "paper_final_anchorfix"
ANCH = ("sp_anchors", "greedy_anchors")
OTHERS = ("rlda", "mada", "cart", "random_search")
NM = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}
FN = re.compile(r"^(?P<ds>.+)__(?P<m>[a-z_]+)__seed(?P<seed>\d+)__tp0p90__tc(?P<tc>0p\d+)\.json$")


def load(folders, tc):
    """(ds, method, seed) -> result json, for files at tau_C `tc` in `folders`."""
    out = {}
    for d in folders:
        for p in Path(d).glob("*.json"):
            m = FN.match(p.name)
            if m and m["tc"] == tc and "__instances__" not in p.name:
                out[(m["ds"], m["m"], int(m["seed"]))] = json.loads(p.read_text())
    return out


def glob_metrics(j):
    g = j.get("global_ruleset") or {}
    f = g.get("global_fidelity")
    return {"fid": f if f is not None and f == f else None, "cov": g.get("coverage") or 0.0,
            "eff": track_a_eff(g), "cov_tau": track_a_cov_tau(g)}


def ds_means(cells, method, seeds, key):
    """dataset -> mean over seeds of a global metric (Fid skips undefined)."""
    out = {}
    for ds in DATASETS:
        v = [glob_metrics(cells[(ds, method, s)])[key] for s in seeds if (ds, method, s) in cells]
        v = [x for x in v if x is not None]
        out[ds] = float(np.mean(v)) if v else float("nan")
    return out


def f3(x):
    return "—" if x is None or x != x else f"{x:.3f}"


def headline(seeds, methods, title):
    print(f"\n### {title}\n")
    print("Mean over the 12 datasets of each dataset's seed mean. Cov = global rule-set coverage on D_test.\n")
    print("| Method | Fid | Cov | Eff | Cov_τ |")
    print("|---|---:|---:|---:|---:|")
    for lab, (cells, m) in methods.items():
        row = [np.nanmean(list(ds_means(cells, m, seeds, k).values())) for k in ("fid", "cov", "eff", "cov_tau")]
        print(f"| {lab} | " + " | ".join(f3(x) for x in row) + " |")


def main() -> int:
    new = load([NEW / "baselines_emp"], "0p10")
    seeds = sorted({s for (_, m, s) in new if m in ANCH})
    full = [s for s in seeds if all((ds, m, s) in new for ds in DATASETS for m in ANCH)]
    old = load([VALTB / "baselines_emp", VALTB / "emp_tc0p10" / "results" / "ddpg",
                VALTB / "emp_tc0p10" / "results" / "maddpg"], "0p10")
    print("# SP-/greedy-Anchors after the exact-bins fix\n")
    print(f"Seeds with every new anchor-family cell: {full} (of {seeds} started). "
          "Old and new use the same seeds. Main grid: emp, τ_P = 0.90, τ_C = 0.10, k = 1, pool 5 per class.\n")
    seeds = full

    methods = {"RLDA": (old, "rlda"), "MADA": (old, "mada"), "CART": (old, "cart"),
               "Random search": (old, "random_search")}
    for m, lab in (("sp_anchors", "SP-Anchors"), ("greedy_anchors", "Greedy Anchors")):
        methods[f"{lab}, old"] = (old, m)
        methods[f"{lab}, exact bins, no empty"] = (new, m)
    headline(seeds, methods, "Headline, main grid")

    # --- empties and classes left without a rule
    print("\n### Empty anchors dropped and classes left without a rule\n")
    print("| dataset | empty anchors dropped (SP run, all classes, summed over seeds) | classes with no rule, SP / greedy (summed over seeds) |")
    print("|---|---:|---:|")
    for ds in DATASETS:
        e = sum(sum(new[(ds, "sp_anchors", s)]["extra"]["empty_anchors_dropped"].values()) for s in seeds)
        nr = []
        for m in ANCH:
            n = 0
            for s in seeds:
                j = new[(ds, m, s)]
                n_cls = len(j["extra"]["pool_per_class"])
                n += n_cls - sum(1 for b in j["per_class"].values() if b.get("selected_rules"))
            nr.append(n)
        print(f"| {NM.get(ds, ds)} | {e} | {nr[0]} / {nr[1]} |")

    # --- dataset-wise
    for key, lab in (("eff", "Eff"), ("fid", "Fid"), ("cov", "Cov (global, D_test)")):
        print(f"\n### Dataset-wise {lab}, seed mean\n")
        print("| dataset | RLDA | MADA | CART | SP old | SP new | Greedy old | Greedy new |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        cols = [ds_means(old, "rlda", seeds, key), ds_means(old, "mada", seeds, key),
                ds_means(old, "cart", seeds, key),
                ds_means(old, "sp_anchors", seeds, key), ds_means(new, "sp_anchors", seeds, key),
                ds_means(old, "greedy_anchors", seeds, key), ds_means(new, "greedy_anchors", seeds, key)]
        for ds in DATASETS:
            print(f"| {NM.get(ds, ds)} | " + " | ".join(f3(c[ds]) for c in cols) + " |")
        print("| **mean** | " + " | ".join(f3(np.nanmean(list(c.values()))) for c in cols) + " |")

    # --- class-wise
    print("\n### Class-wise union Fid / Cov, seed mean (Cov = P(x in union | f̂(x) = c) on D_test)\n")
    print("| dataset | class | SP old | SP new | Greedy old | Greedy new | RLDA | MADA |")
    print("|---|---:|---|---|---|---|---|---|")

    def cls_fc(cells, m, ds, c):
        f, v = [], []
        for s in seeds:
            b = (cells.get((ds, m, s)) or {}).get("per_class", {}).get(f"class_{c}")
            if not b:
                v.append(0.0)
                continue
            u = b.get("union") or {}
            if u.get("fidelity") is not None:
                f.append(u["fidelity"])
            v.append(u.get("coverage") or 0.0)
        return f"{f3(np.mean(f) if f else None)} / {f3(np.mean(v))}"

    for ds in DATASETS:
        ncls = len(new[(ds, "sp_anchors", seeds[0])]["extra"]["pool_per_class"])
        for c in range(ncls):
            print(f"| {NM.get(ds, ds)} | {c} | " + " | ".join(
                cls_fc(cells, m, ds, c) for cells, m in (
                    (old, "sp_anchors"), (new, "sp_anchors"), (old, "greedy_anchors"),
                    (new, "greedy_anchors"), (old, "rlda"), (old, "mada"))) + " |")
    print("\nA class with no rule counts as Cov 0 and has no Fid.")

    # --- Wilcoxon, the paper's primary family (9 Eff contrasts, Holm)
    print("\n### Primary family: Eff, one pair per dataset (seed mean), Wilcoxon, Holm over 9\n")
    for tag, anc in (("old", old), ("new", new)):
        pairs = [("mada", "rlda", old, old), ("rlda", "cart", old, old), ("mada", "cart", old, old),
                 ("rlda", "sp_anchors", old, anc), ("mada", "sp_anchors", old, anc),
                 ("rlda", "greedy_anchors", old, anc), ("mada", "greedy_anchors", old, anc),
                 ("rlda", "random_search", old, old), ("mada", "random_search", old, old)]
        rows = []
        for a, b, ca, cb in pairs:
            va = [ds_means(ca, a, seeds, "eff")[d] for d in DATASETS]
            vb = [ds_means(cb, b, seeds, "eff")[d] for d in DATASETS]
            w = paired_wilcoxon(va, vb)
            lo, hi = boot_ci_paired_diff(va, vb)
            rows.append((a, b, w, lo, hi, sum(x > y for x, y in zip(va, vb))))
        adj = holm([r[2].get("pvalue") for r in rows])
        print(f"\nAnchor cells: **{tag}**\n")
        print("| a vs b | a wins | mean Δ | 95% CI | p | p (Holm) |")
        print("|---|---:|---:|---|---:|---:|")
        for (a, b, w, lo, hi, n), pa in zip(rows, adj):
            print(f"| {a} vs {b} | {n}/12 | {w['mean_diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                  f"{w['pvalue']:.4f} | {pa:.4f} |")

    # --- tau_C = 0.20 grid
    new20 = load([NEW / "baselines_emp"], "0p20")
    old20 = load([VALTB / "baselines_emp", VALTB / "emp_tc0p20" / "results" / "ddpg",
                  VALTB / "emp_tc0p20" / "results" / "maddpg"], "0p20")
    s20 = [s for s in seeds if all((ds, m, s) in new20 for ds in DATASETS for m in ANCH)]
    if s20:
        m20 = {"RLDA": (old20, "rlda"), "MADA": (old20, "mada")}
        for m, lab in (("sp_anchors", "SP-Anchors"), ("greedy_anchors", "Greedy Anchors")):
            m20[f"{lab}, old"] = (old20, m)
            m20[f"{lab}, new"] = (new20, m)
        headline(s20, m20, f"Headline, emp τ_C = 0.20 grid (seeds {s20})")

    # --- k-sweep and pool-20
    def k_table(title, k_dirs_old, k_dirs_new, ks, with_rl):
        print(f"\n### {title}\n")
        print("Eff, mean over 12 datasets of seed means.\n")
        head = (["RLDA", "MADA"] if with_rl else []) + ["SP old", "SP new", "Greedy old", "Greedy new"]
        print("| k | " + " | ".join(head) + " |")
        print("|---:|" + "---:|" * len(head))
        for k in ks:
            o = load([k_dirs_old(k)], "0p10") if k_dirs_old(k) else {}
            n = load([k_dirs_new(k)], "0p10")
            vals = []
            if with_rl:
                for m in ("rlda", "mada"):
                    vals.append(np.nanmean(list(ds_means(o, m, seeds, "eff").values())) if o else None)
            for m in ANCH:
                for cells in (o, n):
                    ok = cells and all((d, m, s) in cells for d in DATASETS for s in seeds)
                    vals.append(np.nanmean(list(ds_means(cells, m, seeds, "eff").values())) if ok else None)
            print(f"| {k} | " + " | ".join(f3(v) for v in vals) + " |")

    main_old = VALTB / "emp_tc0p10" / "results"

    def ks_old(k):
        return None if k == 1 else VALTB / "k_sweep" / f"k{k}"

    def ks_new(k):
        return NEW / "baselines_emp" if k == 1 else NEW / "k_sweep" / f"k{k}"

    # k = 1 RL/old anchors come from the main grid.
    print("\n(k = 1 is the main grid; RL at k > 1 re-selects from the same RL pools.)")
    old_k1 = load([VALTB / "baselines_emp", main_old / "ddpg", main_old / "maddpg"], "0p10")
    rows_k = [1, 2, 3, 5, 10, 20]
    print("\n### Union-size k-sweep (pool max(k, 5) per class)\n")
    print("Eff, mean over 12 datasets of seed means.\n")
    print("| k | RLDA | MADA | SP old | SP new | Greedy old | Greedy new |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for k in rows_k:
        o = old_k1 if k == 1 else load([ks_old(k)], "0p10")
        n = load([ks_new(k)], "0p10")
        vals = [np.nanmean(list(ds_means(o, m, seeds, "eff").values())) for m in ("rlda", "mada")]
        for m in ANCH:
            for cells in (o, n):
                ok = all((d, m, s) in cells for d in DATASETS for s in seeds)
                vals.append(np.nanmean(list(ds_means(cells, m, seeds, "eff").values())) if ok else None)
        print(f"| {k} | " + " | ".join(f3(v) for v in vals) + " |")

    k_table("Pool of 20 anchors per class, one pool reduced at each k",
            lambda k: VALTB / "pool20" / f"k{k}" if (VALTB / "pool20" / f"k{k}").is_dir() else None,
            lambda k: NEW / "pool20" / f"k{k}", [1, 2, 3, 5, 10, 20], with_rl=False)
    print("\nOld pool-20 cells exist only at k = 1 and 5 (—: not run).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
