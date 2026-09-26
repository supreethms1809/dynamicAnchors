"""Choose MADA's inter-class overlap weight on D_val from the existing ablation legs.

The paper fixes the weight at 0.75; the 0.50 and 1.00 legs already exist
(ablations/overlap/w050, w100, empirical tau_C = 0.10, k = 1). This rebuilds each
leg's selected boxes on D_val and scores the rule set there (conflicts tie-broken
on D_val), so the weight is picked without touching D_test.

Criterion: the ablation section's headline metric, effectiveness = Fid x Cov
(global coverage, i.e. share of rows decided), averaged over seeds per dataset
and then over the 11 datasets the ablation table uses (wyodot excluded there;
it is reported separately). Per-dataset winners and the D_test numbers are kept
for context only.

    python -m revision.overlap_select_val
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import (  # noqa: E402
    RESULTS, cell_classifier, global_result, load_seed_data, rebuild, result_path,
    with_classifier,
)

WEIGHTS = {
    "0.50": RESULTS / "ablations" / "overlap" / "w050" / "results" / "maddpg",
    "0.75": RESULTS / "emp_tc0p10" / "results" / "maddpg",
    "1.00": RESULTS / "ablations" / "overlap" / "w100" / "results" / "maddpg",
}
OUT = RESULTS / "overlap_select_val_5seed.json"


def _scores(res) -> Dict[str, float]:
    fid = float(res.global_fidelity) if res.n_decided else float("nan")
    return {"fidelity": fid, "coverage": float(res.coverage),
            "effectiveness": 0.0 if not res.n_decided else fid * float(res.coverage)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    records: List[Dict[str, Any]] = []
    for ds in args.datasets:
        for seed in args.seeds:
            paths = {w: d / f"{ds}__mada__seed{seed}__tp0p90__tc0p10.json" for w, d in WEIGHTS.items()}
            if not any(p.is_file() for p in paths.values()):
                continue
            base = load_seed_data(ds, seed)
            for w, p in paths.items():
                if not p.is_file():
                    print(f"  missing {p.relative_to(RESULTS)}")
                    continue
                cell = json.loads(p.read_text())
                sd = with_classifier(base, cell_classifier(cell))
                rb = rebuild(cell, sd)
                records.append({
                    "dataset": ds, "seed": seed, "weight": w,
                    "path": str(p.relative_to(REPO.parent)),
                    "classifier": str(sd.classifier),
                    "rule_failures": rb.failures,
                    "val": _scores(global_result(rb, sd, "val", "val")),
                    "test": _scores(global_result(rb, sd, "test", "val")),
                })
            print(f"done {ds} {seed}", flush=True)
    OUT.write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} records to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
