#!/usr/bin/env python3
"""Per-policy selection with the policy floor (0.60), against the paper cells.

Columns (seeds with every cell present, the same for all columns):
  RLDA paper       ../results/paper_final_valtb             (1 rule per class, no floor)
  RLDA floor       ../results/paper_final_perpolicy          (1-level OR: its rule if D_val Fid >= 0.60)
  MADA paper       ../results/paper_final_valtb             (all agents pooled, class top-1)
  MADA OR          ../results/paper_final_perpolicy_nofloor  (2-level OR, no floor)
  MADA OR floor    ../results/paper_final_perpolicy          (2-level OR, agents below 0.60 left out)

Global rule set on D_test:
  Fid        agreement with f_hat on the rows the rule set decides
  Cov        share of test rows it decides
  Eff        Fid x Cov
  Overlap    share of test rows two or more classes claim (settled by the D_val
             union-Fid tie-break; counted in Fid and Cov)
  Abstained  share of test rows no class claims (1 - Cov)
Class-wise Cov is the class union's P(x in union | f_hat(x) = c).
The floor sensitivity table compares no floor and 0.50 ... 0.90
(paper_final_perpolicy_floor0NN; 0.60 is paper_final_perpolicy).

  python -m revision.perpolicy_report > ../results/paper_final_perpolicy/REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon, track_a_eff  # noqa: E402

RES = REPO.parent / "results"
VALTB, PP, NOF = RES / "paper_final_valtb", RES / "paper_final_perpolicy", RES / "paper_final_perpolicy_nofloor"
FLOOR = 0.60
# floor -> (RLDA root, MADA root). Without a floor, per-policy RLDA is the paper cell
# (one policy per class), so its "none" row reads paper_final_valtb.
FLOORS = {"none": (VALTB, NOF),
          **{f"{v:.2f}": ((PP,) * 2 if v == FLOOR else (RES / f"paper_final_perpolicy_floor{round(v * 100):03d}",) * 2)
             for v in (0.50, 0.60, 0.70, 0.80, 0.90)}}
NM = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}
GRIDS = ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20")
ALG = {"rlda": "ddpg", "mada": "maddpg"}
KEYS = (("fid", "Fid"), ("cov", "Cov"), ("eff", "Eff"), ("overlap", "Overlap"), ("abstained", "Abstained"))


def fname(ds, arm, seed, tc):
    return f"{ds}__{arm}__seed{seed}__tp0p90__tc{tc}.json"


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def metrics(j):
    g = j["global_ruleset"]
    f = g.get("global_fidelity")
    cov = g.get("coverage") or 0.0
    return {"fid": f if f is not None and f == f else np.nan, "cov": cov, "eff": track_a_eff(g),
            "overlap": g.get("conflict_rate") or 0.0, "abstained": 1.0 - cov,
            "rules": float(np.mean([len(b.get("selected_rules") or []) for b in j["per_class"].values()]))
            if j["per_class"] else 0.0,
            "no_rule": (j.get("extra") or {}).get("classes_without_rule_at_floor", 0)}


def columns(sub):
    """sub: path under a results root, e.g. 'emp_tc0p10/results' or 'k_sweep/k3'."""
    def d(root, arm):
        return root / sub / ALG[arm] if sub.endswith("results") else root / sub
    return {"RLDA paper": (d(VALTB, "rlda"), "rlda"), "RLDA floor": (d(PP, "rlda"), "rlda"),
            "MADA paper": (d(VALTB, "mada"), "mada"), "MADA OR": (d(NOF, "mada"), "mada"),
            "MADA OR floor": (d(PP, "mada"), "mada")}


def complete_seeds(cols, tc):
    return [s for s in (42, 43, 44, 45, 46)
            if all((f / fname(ds, arm, s, tc)).is_file() for f, arm in cols.values() for ds in DATASETS)]


def table(cols, seeds, tc):
    out = {lab: {} for lab in cols}
    for lab, (folder, arm) in cols.items():
        for ds in DATASETS:
            ms = [metrics(load(folder / fname(ds, arm, s, tc))) for s in seeds]
            out[lab][ds] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
    return out


def mean12(t, col, key):
    return float(np.nanmean([t[col][d][key] for d in DATASETS]))


def f3(x):
    return "—" if x != x else f"{x:.3f}"


def section(grid):
    tc = grid.split("_")[1][2:]
    cols = columns(f"{grid}/results")
    seeds = complete_seeds(cols, tc)
    if not seeds:
        return
    t = table(cols, seeds, tc)
    print(f"\n## {grid}, k = 1 (seeds {', '.join(map(str, seeds))})\n")
    print("| | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for k, lab in KEYS:
        print(f"| {lab} | " + " | ".join(f3(mean12(t, c, k)) for c in cols) + " |")
    print("\nMean of 12 datasets. Rules per class: " + ", ".join(
        f"{c} {mean12(t, c, 'rules'):.2f}" for c in cols) + ". Classes left without a rule per cell: "
        + ", ".join(f"{c} {mean12(t, c, 'no_rule'):.2f}" for c in cols) + ".")
    for k, lab in KEYS:
        print(f"\nDataset-wise {lab}\n")
        print("| dataset | " + " | ".join(cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for ds in DATASETS:
            print(f"| {NM.get(ds, ds)} | " + " | ".join(f3(t[c][ds][k]) for c in cols) + " |")
    print("\nEff, paired over 12 datasets (seed means), Wilcoxon, unadjusted\n")
    for a, b in (("MADA OR floor", "RLDA floor"), ("MADA OR floor", "MADA paper"), ("MADA OR floor", "MADA OR"),
                 ("RLDA floor", "RLDA paper"), ("MADA paper", "RLDA paper")):
        x = [t[a][d]["eff"] for d in DATASETS]
        y = [t[b][d]["eff"] for d in DATASETS]
        w = paired_wilcoxon(x, y)
        print(f"- {a} vs {b}: {sum(p > q for p, q in zip(x, y))}/12 higher, "
              f"mean Δ {np.mean(np.subtract(x, y)):+.3f}, p = {w['pvalue']:.4f}")
    if grid == "emp_tc0p10":
        print("\nClass-wise union Fid / Cov (Cov = P(x in union | f̂(x) = c)), seed mean; — = no rule\n")
        print("| dataset | class | " + " | ".join(cols) + " |")
        print("|---|---:|" + "---|" * len(cols))
        for ds in DATASETS:
            ref = load(cols["RLDA paper"][0] / fname(ds, "rlda", seeds[0], tc))
            for c in range(max(int(k.split("_")[1]) for k in ref["per_class"]) + 1):
                cells = []
                for folder, arm in cols.values():
                    f, v = [], []
                    for s in seeds:
                        b = (load(folder / fname(ds, arm, s, tc))["per_class"]).get(f"class_{c}")
                        u = (b or {}).get("union") or {}
                        if u.get("fidelity") is not None:
                            f.append(u["fidelity"])
                        v.append(u.get("coverage") or 0.0)
                    cells.append(f"{f3(np.mean(f)) if f else '—'} / {f3(np.mean(v))}")
                print(f"| {NM.get(ds, ds)} | {c} | " + " | ".join(cells) + " |")


def main() -> int:
    print(f"# Per-policy selection with the policy floor ({FLOOR:.2f})\n")
    print(f"A policy's selected rule enters its class's OR only at D_val Fid ≥ {FLOOR:.2f}. MADA: each "
          "agent's best rule, OR'd (2-level OR at k > 1); an agent below the floor is left out and "
          "the others still explain the class. RLDA: its one policy's rule (1-level OR); below the "
          "floor the class has no rule.\n")
    print("Fid = agreement with f̂ on decided rows; Cov = share of test rows decided; Eff = Fid × Cov; "
          "Overlap = share of test rows two or more classes claim (settled by the D_val tie-break, "
          "counted in Fid and Cov); Abstained = share no class claims.")
    for grid in GRIDS:
        section(grid)

    print("\n## Floor sensitivity, k = 1\n")
    print("Mean of 12 datasets per grid, then the mean of the 4 grids.\n")
    avg = {}
    for grid in GRIDS:
        tc = grid.split("_")[1][2:]
        cols = {f"{arm.upper()} {fl}": (roots[i] / grid / "results" / ALG[arm], arm)
                for i, arm in enumerate(("rlda", "mada")) for fl, roots in FLOORS.items()
                if (roots[i] / grid / "results" / ALG[arm]).is_dir()}
        cols = {c: v for c, v in cols.items() if complete_seeds({c: v}, tc)}
        seeds = complete_seeds(cols, tc)
        if not seeds:
            continue
        t = table(cols, seeds, tc)
        print(f"\n{grid} (seeds {', '.join(map(str, seeds))})\n")
        print("| floor | method | Fid | Cov | Eff | Overlap | Abstained | MADA − RLDA Eff (higher/12, p) |")
        print("|---|---|---:|---:|---:|---:|---:|---|")
        for fl in FLOORS:
            r, m = f"RLDA {fl}", f"MADA {fl}"
            if r not in t or m not in t:
                continue
            x = [t[m][d]["eff"] for d in DATASETS]
            y = [t[r][d]["eff"] for d in DATASETS]
            w = paired_wilcoxon(x, y)
            for c in (r, m):
                vals = [mean12(t, c, k) for k, _ in KEYS]
                avg.setdefault(c, []).append(vals)
                diff = (f"{np.mean(np.subtract(x, y)):+.3f} ({sum(a > b for a, b in zip(x, y))}/12, "
                        f"p = {w['pvalue']:.3f})") if c == m else ""
                print(f"| {fl} | {c.split()[0]} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {diff} |")
    if avg:
        print("\nMean of the 4 grids\n")
        print("| floor | method | Fid | Cov | Eff | Overlap | Abstained |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for c, rows in avg.items():
            if len(rows) == len(GRIDS):
                print(f"| {c.split()[1]} | {c.split()[0]} | " + " | ".join(
                    f"{v:.3f}" for v in np.mean(rows, axis=0)) + " |")

    print("\n## k-sweep, emp τ_C = 0.10\n")
    print("At k, each policy keeps up to k rules (then the floor). Eff / Overlap, mean of 12.\n")
    cols1 = columns("emp_tc0p10/results")
    seeds = complete_seeds(cols1, "0p10")
    print(f"Seeds {seeds}.\n")
    print("| k | " + " | ".join(cols1) + " |")
    print("|---:|" + "---|" * len(cols1))
    for k in (1, 2, 3, 5, 10, 20):
        cols = cols1 if k == 1 else columns(f"k_sweep/k{k}")
        ok = {c: v for c, v in cols.items()
              if all((v[0] / fname(d, v[1], s, "0p10")).is_file() for d in DATASETS for s in seeds)}
        t = table(ok, seeds, "0p10")
        print(f"| {k} | " + " | ".join(
            (f"{mean12(t, c, 'eff'):.3f} / {mean12(t, c, 'overlap'):.3f}" if c in t else "—") for c in cols) + " |")

    print("\n## Ablations, emp τ_C = 0.10 (11 datasets, no wyodot): Eff, mean of 11\n")
    print("Variants with one agent per class (no_coord, no_same_class) are 1-level ORs.\n")
    print("| variant | seeds | paper (pooled) | OR floor | full MADA − variant, paper | full MADA − variant, OR floor |")
    print("|---|---|---:|---:|---:|---:|")
    ds11 = [d for d in DATASETS if d != "wyodot_kvdw_labeled"]
    for fq in sorted((PP / "ablations").glob("*/*/results/maddpg")):
        rel = fq.relative_to(PP / "ablations").parent.parent
        fp = VALTB / "ablations" / rel / "results/maddpg"
        ss = [s for s in (42, 43, 44, 45, 46) if all((fq / fname(d, "mada", s, "0p10")).is_file()
                                                     and (fp / fname(d, "mada", s, "0p10")).is_file() for d in ds11)]
        if not ss:
            continue

        def m(folder):
            return np.mean([np.mean([metrics(load(folder / fname(d, "mada", s, "0p10")))["eff"] for s in ss])
                            for d in ds11])
        vp, vq = m(fp), m(fq)
        fpm, fqm = m(VALTB / "emp_tc0p10/results/maddpg"), m(PP / "emp_tc0p10/results/maddpg")
        print(f"| {rel} | {','.join(map(str, ss))} | {vp:.3f} | {vq:.3f} | {fpm - vp:+.3f} | {fqm - vq:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
