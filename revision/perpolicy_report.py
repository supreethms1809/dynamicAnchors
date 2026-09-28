#!/usr/bin/env python3
"""Per-policy selection with the tau_P floor, against the paper cells.

Columns (seeds with every cell present, the same for all columns):
  RLDA paper          ../results/paper_final_valtb             (1 rule per class, no floor)
  RLDA floor          ../results/paper_final_perpolicy          (1-level OR: its rule if D_val Fid >= 0.90)
  MADA paper          ../results/paper_final_valtb             (all agents pooled, class top-1)
  MADA OR             ../results/paper_final_perpolicy_nofloor  (2-level OR, no floor)
  MADA OR floor       ../results/paper_final_perpolicy          (2-level OR, agents below 0.90 left out)

Global rule set on D_test: Fid on decided rows, Cov = share of D_test rows some class
union fires on, Eff = Fid x Cov (conflicts settled by the D_val union-Fid tie-break),
Conf = share of rows two or more classes claim, Eff_strict = share of rows exactly one
class claims and gets right (conflicted rows count as undecided).
Class-wise Cov is the class union's P(x in union | f_hat(x) = c).

  python -m revision.perpolicy_report > ../results/paper_final_perpolicy/REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "BenchMARL", REPO / "single_agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from revision.conflict_free_eff import strict  # noqa: E402
from revision.paper_stats import DATASETS  # noqa: E402
from utils.metrics import paired_wilcoxon, track_a_eff  # noqa: E402

RES = REPO.parent / "results"
VALTB, PP, NOF = RES / "paper_final_valtb", RES / "paper_final_perpolicy", RES / "paper_final_perpolicy_nofloor"
NM = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}
GRIDS = ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20")
ALG = {"rlda": "ddpg", "mada": "maddpg"}
_SD: dict = {}


def fname(ds, arm, seed, tc):
    return f"{ds}__{arm}__seed{seed}__tp0p90__tc{tc}.json"


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def metrics(j, with_strict):
    g = j["global_ruleset"]
    f = g.get("global_fidelity")
    out = {"eff": track_a_eff(g), "fid": f if f is not None and f == f else np.nan,
           "cov": g.get("coverage") or 0.0, "conf": g.get("conflict_rate") or 0.0,
           "rules": float(np.mean([len(b.get("selected_rules") or []) for b in j["per_class"].values()]))
           if j["per_class"] else 0.0,
           "no_rule": (j.get("extra") or {}).get("classes_without_rule_at_floor", 0)}
    if with_strict:
        out["eff_strict"] = strict(j, _SD)["eff_strict"]
    return out


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


def table(cols, seeds, tc, with_strict):
    out = {lab: {} for lab in cols}
    for lab, (folder, arm) in cols.items():
        for ds in DATASETS:
            ms = [metrics(load(folder / fname(ds, arm, s, tc)), with_strict) for s in seeds]
            out[lab][ds] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
    return out


def f3(x):
    return "—" if x != x else f"{x:.3f}"


def section(grid):
    tc = grid.split("_")[1][2:]
    cols = columns(f"{grid}/results")
    seeds = complete_seeds(cols, tc)
    if not seeds:
        return
    t = table(cols, seeds, tc, with_strict=True)
    print(f"\n## {grid}, k = 1 (seeds {', '.join(map(str, seeds))})\n")
    print("| | " + " | ".join(cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for k, lab in (("eff", "Eff"), ("fid", "Fid"), ("cov", "Cov"), ("conf", "Conf"), ("eff_strict", "Eff_strict"),
                   ("rules", "rules/class"), ("no_rule", "classes with no rule (per cell)")):
        print(f"| {lab}, mean of 12 | " + " | ".join(
            f3(np.nanmean([t[c][d][k] for d in DATASETS])) for c in cols) + " |")
    for k, lab in (("eff", "Eff"), ("eff_strict", "Eff_strict"), ("fid", "Fid"), ("cov", "Cov"), ("conf", "Conf")):
        print(f"\nDataset-wise {lab}\n")
        print("| dataset | " + " | ".join(cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for ds in DATASETS:
            print(f"| {NM.get(ds, ds)} | " + " | ".join(f3(t[c][ds][k]) for c in cols) + " |")
    print("\nPaired over 12 datasets (seed means), Wilcoxon, unadjusted\n")
    for key in ("eff", "eff_strict"):
        for a, b in (("MADA OR floor", "RLDA floor"), ("MADA OR floor", "MADA paper"), ("MADA OR floor", "MADA OR"),
                     ("RLDA floor", "RLDA paper"), ("MADA paper", "RLDA paper")):
            x = [t[a][d][key] for d in DATASETS]
            y = [t[b][d][key] for d in DATASETS]
            w = paired_wilcoxon(x, y)
            print(f"- {key}: {a} vs {b}: {sum(p > q for p, q in zip(x, y))}/12 higher, "
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
    print("# Per-policy selection with the τ_P floor\n")
    print("A policy's selected rule enters its class's OR only at D_val Fid ≥ 0.90. MADA: each "
          "agent's best rule, OR'd (2-level OR at k > 1); an agent below the floor is left out and "
          "the others still explain the class. RLDA: its one policy's rule (1-level OR); below the "
          "floor the class has no rule.")
    for grid in GRIDS:
        section(grid)

    print("\n## k-sweep, emp τ_C = 0.10\n")
    print("At k, each policy keeps up to k rules (then the floor). Eff / Eff_strict, mean of 12.\n")
    cols1 = columns("emp_tc0p10/results")
    seeds = complete_seeds(cols1, "0p10")
    print(f"Seeds {seeds}.\n")
    print("| k | " + " | ".join(cols1) + " |")
    print("|---:|" + "---|" * len(cols1))
    for k in (1, 2, 3, 5, 10, 20):
        cols = cols1 if k == 1 else columns(f"k_sweep/k{k}")
        ok = {c: v for c, v in cols.items()
              if all((v[0] / fname(d, v[1], s, "0p10")).is_file() for d in DATASETS for s in seeds)}
        t = table(ok, seeds, "0p10", with_strict=True)
        print(f"| {k} | " + " | ".join(
            (f"{np.mean([t[c][d]['eff'] for d in DATASETS]):.3f} / {np.mean([t[c][d]['eff_strict'] for d in DATASETS]):.3f}"
             if c in t else "—") for c in cols) + " |")

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
            return np.mean([np.mean([metrics(load(folder / fname(d, "mada", s, "0p10")), False)["eff"] for s in ss])
                            for d in ds11])
        vp, vq = m(fp), m(fq)
        fpm, fqm = m(VALTB / "emp_tc0p10/results/maddpg"), m(PP / "emp_tc0p10/results/maddpg")
        print(f"| {rel} | {','.join(map(str, ss))} | {vp:.3f} | {vq:.3f} | {fpm - vp:+.3f} | {fqm - vq:+.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
