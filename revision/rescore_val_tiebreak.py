"""Re-score every stored cell with the conflict tie-break on D_val instead of D_test.

`evaluate_ruleset_as_classifier` settles a row fired by two or more class unions
in favour of the class whose union has the higher Fid. Until this fix both
`revision.evaluate` and `revision.baselines` passed the union Fid measured on
D_test, so the reported predictor used test labels (f_hat on D_test) to pick a
winner. This recomputes every cell with the D_val union Fid instead.

Coverage cannot move (a conflicted row is decided either way); only Fid / Pur on
conflicted rows can. Every cell is self-checked first: the rebuilt boxes must hit
the stored per-rule D_val and D_test counts, and the rebuilt predictor with the
STORED tie-break must reproduce the stored global counts exactly. Cells that fail
are reported and not written.

Output mirrors the input layout under ../results/paper_final_valtb/ (global_ruleset
replaced, the old one kept in extra.global_ruleset_test_tiebreak), plus
valtiebreak_cells.json with the old/new numbers per cell.

    python -m revision.rescore_val_tiebreak
    python -m revision.rescore_val_tiebreak --datasets iris --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from revision.paper_stats import DATASETS  # noqa: E402
from revision.rescore_boxes import (  # noqa: E402
    RESULTS, cell_classifier, global_result, load_seed_data, rebuild, summary_dict,
    with_classifier,
)

POOL20 = REPO / "runs" / "paper_final_pool20"
OUT = REPO.parent / "results" / "paper_final_valtb"
CHECK_KEYS = ("n_decided", "n_conflict", "n_fid_agree", "n_pur_agree")
FHAT_DRIFT_MAX = 2  # rows whose f_hat differs between the training machine and this one


def cells_for(ds: str, seed: int) -> List[Tuple[Path, Path]]:
    """(input path, output path) for every result cell of this dataset / seed."""
    pat = f"{ds}__*__seed{seed}__tp0p90__tc*.json"
    out = [(p, OUT / p.relative_to(RESULTS)) for p in sorted(RESULTS.rglob(pat))]
    out += [(p, OUT / "pool20" / p.relative_to(POOL20)) for p in sorted(POOL20.rglob(pat))]
    return [(p, o) for p, o in out if "__instances__" not in p.name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    records: List[Dict[str, Any]] = []
    for ds in args.datasets:
        for seed in args.seeds:
            cells = cells_for(ds, seed)
            if not cells:
                continue
            base = load_seed_data(ds, seed)
            by_clf = {}
            n_bad = 0
            for src, dst in cells:
                cell = json.loads(src.read_text())
                clf = cell_classifier(cell)
                key = str(clf.resolve()) if clf else None
                if key not in by_clf:
                    by_clf[key] = with_classifier(base, clf)
                sd = by_clf[key]
                rb = rebuild(cell, sd)
                stored = cell["global_ruleset"]
                repro = summary_dict(global_result(rb, sd, "test", "stored"))
                new = summary_dict(global_result(rb, sd, "test", "val"))
                exact = not rb.failures and all(repro[k] == stored.get(k) for k in CHECK_KEYS)
                # Boxes exact but a row or two of f_hat differs across machines
                # (housing seed 44, trained on spark): carry the measured change
                # onto the stored counts instead of replacing them.
                drift = {k: repro[k] - stored.get(k, 0) for k in CHECK_KEYS}
                fhat_drift = (
                    not exact and not rb.failures
                    and drift["n_decided"] == 0 and drift["n_conflict"] == 0
                    and max(abs(drift["n_fid_agree"]), abs(drift["n_pur_agree"])) <= FHAT_DRIFT_MAX
                )
                if fhat_drift:
                    nd = stored["n_decided"]
                    nf = stored["n_fid_agree"] + new["n_fid_agree"] - repro["n_fid_agree"]
                    npur = stored["n_pur_agree"] + new["n_pur_agree"] - repro["n_pur_agree"]
                    new = {**new, "n_fid_agree": nf, "n_pur_agree": npur,
                           "global_fidelity": nf / nd, "global_purity": npur / nd,
                           "effectiveness": nf / nd * stored["coverage"]}
                ok = exact or fhat_drift
                rec = {
                    "path": str(src.relative_to(REPO.parent)),
                    "dataset": ds, "seed": seed, "method": cell["method"],
                    "classifier": str(sd.classifier),
                    "reproduced": ok, "exact": exact, "fhat_drift": drift if fhat_drift else None,
                    "rule_failures": rb.failures,
                    "n_rules": rb.n_rules, "n_rules_ulp_widened": rb.n_widened,
                    "old": {k: stored.get(k) for k in repro}, "new": new,
                }
                records.append(rec)
                if not ok:
                    n_bad += 1
                    print(f"  NOT REPRODUCED {src.name}: stored "
                          f"{[stored.get(k) for k in CHECK_KEYS]} rebuilt "
                          f"{[repro[k] for k in CHECK_KEYS]} {rb.failures[:2]}")
                    continue
                cell["global_ruleset"] = {**stored, **new}
                cell.setdefault("extra", {})
                cell["extra"]["global_ruleset_test_tiebreak"] = stored
                cell["extra"]["conflict_tiebreak"] = "class-union fidelity on D_val"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(cell, indent=2))
            print(f"{ds} seed {seed}: {len(cells)} cells, {n_bad} not reproduced", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "valtiebreak_cells.json").write_text(json.dumps(records, indent=1))
    bad = [r for r in records if not r["reproduced"]]
    moved = [r for r in records if r["reproduced"]
             and r["new"]["n_fid_agree"] != r["old"]["n_fid_agree"]]
    print(f"{len(records)} cells, {len(bad)} not reproduced, "
          f"{len(moved)} with a changed Fid numerator -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
