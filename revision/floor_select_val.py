#!/usr/bin/env python3
"""Choose the policy floor on D_val: the floor comparison with every metric on D_val.

The floor is a hyperparameter, so it must be chosen without D_test. For each floor
(none, 0.50 ... 0.90) the stored rule sets are rebuilt on the D_val rows
(`revision.rescore_boxes`, which also checks every rule's D_val and D_test row counts
against the cell) and scored as a classifier on D_val, conflicts settled by the D_val
union Fid. D_test is not read.

  Fid        agreement with f_hat on the D_val rows the rule set decides
  Cov        share of D_val rows it decides
  Eff        Fid x Cov
  Overlap    share of D_val rows two or more classes claim
  Abstained  share of D_val rows no class claims (1 - Cov)

  python -m revision.floor_select_val > ../results/paper_final_perpolicy/FLOOR_SELECT_VAL.md
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
from revision.rescore_boxes import cell_classifier, global_result, load_seed_data, rebuild, with_classifier  # noqa: E402

RES = REPO.parent / "results"
GRIDS = ("emp_tc0p10", "emp_tc0p20", "pert_tc0p10", "pert_tc0p20")
ALG = {"rlda": "ddpg", "mada": "maddpg"}
# floor -> (RLDA root, MADA root); without a floor RLDA's per-policy cell is the paper cell
FLOORS = {"none": (RES / "paper_final_valtb", RES / "paper_final_perpolicy_nofloor"),
          "0.50": (RES / "paper_final_perpolicy_floor050",) * 2,
          "0.60": (RES / "paper_final_perpolicy",) * 2,
          "0.70": (RES / "paper_final_perpolicy_floor070",) * 2,
          "0.80": (RES / "paper_final_perpolicy_floor080",) * 2,
          "0.90": (RES / "paper_final_perpolicy_floor090",) * 2}
KEYS = ("fid", "cov", "eff", "overlap", "abstained")
_SD: dict = {}


def val_metrics(cell):
    clf = cell_classifier(cell)
    key = (cell["dataset"], cell["seed"])
    if key not in _SD:
        _SD[key] = load_seed_data(cell["dataset"], cell["seed"], clf)
    sd = with_classifier(_SD[key], clf)
    rb = rebuild(cell, sd)
    if not rb.classes:   # the floor left no class a rule
        return {"fid": np.nan, "cov": 0.0, "eff": 0.0, "overlap": 0.0, "abstained": 1.0}, 0
    g = global_result(rb, sd, "val", "val").to_dict()
    f = g.get("global_fidelity")
    cov = g.get("coverage") or 0.0
    return ({"fid": f if f is not None and f == f else np.nan, "cov": cov,
             "eff": (f or 0.0) * cov if f == f else 0.0,
             "overlap": g.get("conflict_rate") or 0.0, "abstained": 1.0 - cov}, len(rb.failures))


def main() -> int:
    seeds = [int(s) for s in sys.argv[1:]] or [42, 43]
    print("# Policy floor chosen on D_val\n")
    print(f"Seeds {seeds}. Every metric on D_val (the selection split); D_test is not read. "
          "Mean over 12 datasets of the seed means.\n")
    avg: dict = {}
    fails = 0
    for grid in GRIDS:
        tc = grid.split("_")[1][2:]
        print(f"\n## {grid}\n")
        print("| floor | method | Fid | Cov | Eff | Overlap | Abstained |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for fl, roots in FLOORS.items():
            for i, arm in enumerate(("rlda", "mada")):
                folder = roots[i] / grid / "results" / ALG[arm]
                per_ds = []
                for ds in DATASETS:
                    ms = []
                    for s in seeds:
                        p = folder / f"{ds}__{arm}__seed{s}__tp0p90__tc{tc}.json"
                        if not p.is_file():
                            continue
                        m, nf = val_metrics(json.loads(p.read_text()))
                        fails += nf
                        ms.append(m)
                    if len(ms) == len(seeds):
                        per_ds.append({k: np.nanmean([m[k] for m in ms]) for k in KEYS})
                if len(per_ds) != len(DATASETS):
                    continue
                vals = [float(np.nanmean([d[k] for d in per_ds])) for k in KEYS]
                avg.setdefault((fl, arm), []).append(vals)
                print(f"| {fl} | {arm.upper()} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    print("\n## Mean of the 4 grids (D_val)\n")
    print("| floor | method | Fid | Cov | Eff | Overlap | Abstained |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (fl, arm), rows in avg.items():
        if len(rows) == len(GRIDS):
            print(f"| {fl} | {arm.upper()} | " + " | ".join(f"{v:.3f}" for v in np.mean(rows, axis=0)) + " |")
    print(f"\nRules whose rebuilt D_val/D_test row counts differ from the stored cell: {fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
