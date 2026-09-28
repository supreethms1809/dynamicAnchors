#!/usr/bin/env python3
"""Eff with conflicted rows counted as undecided (a row two classes claim is a contradiction).

Eff = Fid x Cov settles a conflicted row by the D_val union-Fid tie-break and counts
it as decided. Here a row is decided only if exactly one class union fires on it:
Eff_strict = share of D_test rows with exactly one firing class that equals f_hat.
Boxes are rebuilt from the stored cells (`revision.rescore_boxes`).

  python -m revision.conflict_free_eff
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

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import cell_classifier, load_seed_data, rebuild, with_classifier  # noqa: E402
from utils.metrics import paired_wilcoxon  # noqa: E402

RES = REPO.parent / "results"
COLS = {"RLDA": ("paper_final_valtb", "ddpg", "rlda"),
        "MADA pooled": ("paper_final_valtb", "maddpg", "mada"),
        "MADA per-policy": ("paper_final_perpolicy", "maddpg", "mada")}
NM = {"folktables_income_CA_2018": "folktables", "wyodot_kvdw_labeled": "wyodot"}


def strict(cell, sd_cache):
    key = (cell["dataset"], cell["seed"], cell_classifier(cell))
    if key not in sd_cache:
        sd_cache[key] = with_classifier(load_seed_data(cell["dataset"], cell["seed"], Path(key[2])), key[2])
    sd = sd_cache[key]
    rb = rebuild(cell, sd)
    yh = sd.test.y_hat
    cls = sorted(rb.classes)
    if not cls:   # the floor left no class a rule: the rule set abstains everywhere
        return {"eff_strict": 0.0, "cov_strict": 0.0, "conf": 0.0,
                "eff": cell["global_ruleset"]["effectiveness"], "failures": 0}
    fire = np.stack([rb.classes[c].test_union for c in cls])
    n_fire = fire.sum(0)
    one = n_fire == 1
    lab = np.array(cls)[fire.argmax(0)]
    return {"eff_strict": float((one & (lab == yh)).mean()), "cov_strict": float(one.mean()),
            "conf": float((n_fire >= 2).mean()), "eff": cell["global_ruleset"]["effectiveness"],
            "failures": len(rb.failures)}


def main():
    grid, seeds = sys.argv[1] if len(sys.argv) > 1 else "emp_tc0p10", (42, 43)
    tc = grid.split("_")[1][2:]
    sd_cache, per = {}, {c: {} for c in COLS}
    for ds in DATASETS:
        for lab, (root, alg, arm) in COLS.items():
            rs = []
            for s in seeds:
                p = RES / root / grid / "results" / alg / f"{ds}__{arm}__seed{s}__tp0p90__tc{tc}.json"
                rs.append(strict(json.loads(p.read_text()), sd_cache))
            per[lab][ds] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    print(f"## {grid}, seeds {seeds}: conflicted rows counted as undecided\n")
    print("| dataset | " + " | ".join(f"{c} Eff / Eff_strict" for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for ds in DATASETS:
        print(f"| {NM.get(ds, ds)} | " + " | ".join(
            f"{per[c][ds]['eff']:.3f} / {per[c][ds]['eff_strict']:.3f}" for c in COLS) + " |")
    print("| **mean** | " + " | ".join(
        f"{np.mean([per[c][d]['eff'] for d in DATASETS]):.3f} / {np.mean([per[c][d]['eff_strict'] for d in DATASETS]):.3f}"
        for c in COLS) + " |")
    print("\nMean conflict rate: " + ", ".join(f"{c} {np.mean([per[c][d]['conf'] for d in DATASETS]):.3f}" for c in COLS))
    print("Rebuild failures: " + str(sum(per[c][d]["failures"] for c in COLS for d in DATASETS)))
    for a, b in (("MADA per-policy", "RLDA"), ("MADA per-policy", "MADA pooled"), ("MADA pooled", "RLDA")):
        x = [per[a][d]["eff_strict"] for d in DATASETS]
        y = [per[b][d]["eff_strict"] for d in DATASETS]
        w = paired_wilcoxon(x, y)
        print(f"- Eff_strict {a} vs {b}: {sum(p > q for p, q in zip(x, y))}/12 higher, "
              f"mean Δ {np.mean(np.subtract(x, y)):+.3f}, p = {w['pvalue']:.4f}")


if __name__ == "__main__":
    main()
