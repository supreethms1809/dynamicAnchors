#!/usr/bin/env python3
"""Index the seed-major 2×2 that lives under runs/paper_final/.

Training writes JSON in place. This collector COPIES those files into the
label folders that write_paper_final_tables.py also accepts as a fallback.

  python revision/collect_paper_results.py          # copy what exists
  python revision/collect_paper_results.py --check  # map + missing only

Live trees (do not move; launcher skip-if-exists keys off these paths):

  {emp,pert}_tc{0p10,0p20}/results/{ddpg,maddpg}/{ds}__{rlda,mada}__seedS__tp0p90__tc{tag}.json
  baselines_{emp,pert}/{ds}__{method}__seedS__tp0p90__tc{0p10,0p20}.json
  ablations/{family}/{arm}/results/{ddpg,maddpg}/...

Copies:

  results/predicted/RLDA-emp/         <- emp_tc0p10/results/ddpg
  results/predicted/MADA-emp/         <- emp_tc0p10/results/maddpg
  results/predicted/RLDA-emp-tc020/   <- emp_tc0p20/results/ddpg
  results/predicted/MADA-emp-tc020/   <- emp_tc0p20/results/maddpg
  results/predicted/RLDA-pert/        <- pert_tc0p10/results/ddpg
  results/predicted/MADA-pert/        <- pert_tc0p10/results/maddpg
  results/predicted/RLDA-pert-tc020/  <- pert_tc0p20/results/ddpg
  results/predicted/MADA-pert-tc020/  <- pert_tc0p20/results/maddpg
  results/predicted/CART-emp/         <- baselines_emp cart tc0p10
  results/predicted/CART-emp-tc020/   <- baselines_emp cart tc0p20
  (same pattern for RandS / SP-Anch / GreedyAnch and the pert CART/RandS pair)
  ablations/{family}/{arm}/            <- paper_final/ablations/... (JSON only)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "runs" / "paper_final"
SEEDS = [42, 43, 44, 45, 46]
DATASETS = [
    "iris", "synthetic", "wine", "sick", "breast_cancer", "uci_credit",
    "mammography", "housing", "heloc", "uci_adult", "folktables_income_CA_2018",
    "wyodot_kvdw_labeled",
]
ABL_DATASETS = [d for d in DATASETS if not d.startswith("wyodot")]

# (cell, algo_subdir, method, label) for the 2×2 RL arms.
RL_CELLS = [
    ("emp_tc0p10", "ddpg", "rlda", "RLDA-emp"),
    ("emp_tc0p10", "maddpg", "mada", "MADA-emp"),
    ("emp_tc0p20", "ddpg", "rlda", "RLDA-emp-tc020"),
    ("emp_tc0p20", "maddpg", "mada", "MADA-emp-tc020"),
    ("pert_tc0p10", "ddpg", "rlda", "RLDA-pert"),
    ("pert_tc0p10", "maddpg", "mada", "MADA-pert"),
    ("pert_tc0p20", "ddpg", "rlda", "RLDA-pert-tc020"),
    ("pert_tc0p20", "maddpg", "mada", "MADA-pert-tc020"),
]
# (dir, method, label_tc010, label_tc020)
BASE_CELLS = [
    ("baselines_emp", "cart", "CART-emp", "CART-emp-tc020"),
    ("baselines_emp", "random_search", "RandS-emp", "RandS-emp-tc020"),
    ("baselines_emp", "sp_anchors", "SP-Anch*", "SP-Anch-tc020"),
    ("baselines_emp", "greedy_anchors", "GreedyAnch*", "GreedyAnch-tc020"),
    ("baselines_pert", "cart", "CART-pert", "CART-pert-tc020"),
    ("baselines_pert", "random_search", "RandS-pert", "RandS-pert-tc020"),
]


def d(label: str) -> str:
    return label.replace("*", "")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def put(src: Path, dst: Path, row: dict, rows: list, changed: list) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(src)
    if not dst.exists() or sha(dst) != digest:
        if dst.exists():
            changed.append(str(dst.relative_to(DEST)))
        shutil.copy2(src, dst)
    rows.append({**row, "file": str(dst.relative_to(DEST)), "source": str(src), "sha256": digest})


def _name(ds: str, method: str, seed: int, tc: str) -> str:
    return f"{ds}__{method}__seed{seed}__tp0p90__tc{tc}.json"


def expected() -> list[tuple[Path, Path, dict]]:
    """(src, dst, meta) for every headline cell we expect after a full 5-seed run."""
    jobs = []
    for cell, algo, method, label in RL_CELLS:
        tc = "0p20" if "tc020" in label else "0p10"
        for seed in SEEDS:
            for ds in DATASETS:
                name = _name(ds, method, seed, tc)
                src = DEST / cell / "results" / algo / name
                dst = DEST / "results" / "predicted" / d(label) / name
                jobs.append((src, dst, {
                    "kind": "result", "label": label, "seed": seed,
                    "dataset": ds, "basis": "predicted",
                }))
    for folder, method, lab10, lab20 in BASE_CELLS:
        for seed in SEEDS:
            for ds in DATASETS:
                for tc, label in (("0p10", lab10), ("0p20", lab20)):
                    name = _name(ds, method, seed, tc)
                    src = DEST / folder / name
                    dst = DEST / "results" / "predicted" / d(label) / name
                    jobs.append((src, dst, {
                        "kind": "result", "label": label, "seed": seed,
                        "dataset": ds, "basis": "predicted",
                    }))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Print the map and missing files; do not copy.")
    args = ap.parse_args()

    jobs = expected()
    missing = [j for j in jobs if not j[0].is_file()]
    present = [j for j in jobs if j[0].is_file()]

    print("paper_final 2×2 collection map")
    print(f"  root={DEST}")
    print(f"  expected result JSONs: {len(jobs)}")
    print(f"  present: {len(present)}")
    print(f"  missing: {len(missing)}")
    by_label: dict[str, list] = {}
    for src, _dst, meta in missing:
        by_label.setdefault(meta["label"], []).append(f"{meta['dataset']} seed{meta['seed']}")
    for label in sorted(by_label):
        items = by_label[label]
        print(f"  missing {label}: {len(items)}")
        if args.check and len(items) <= 4:
            for it in items:
                print(f"    {it}")

    if args.check:
        return 0 if not missing else 1

    rows, changed = [], []
    for src, dst, meta in present:
        put(src, dst, meta, rows, changed)

    abl = DEST / "ablations"
    for f in sorted(abl.glob("**/results/*/*.json")) if abl.is_dir() else []:
        if "logs" in f.parts:
            continue
        rel = f.relative_to(abl)
        put(f, DEST / "ablations_indexed" / rel, {
            "kind": "ablation", "label": "/".join(rel.parts[:2]),
            "seed": "", "dataset": f.name.split("__")[0], "basis": "predicted",
        }, rows, changed)

    DEST.mkdir(parents=True, exist_ok=True)
    with open(DEST / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "label", "seed", "dataset", "basis", "file", "source", "sha256"])
        w.writeheader()
        w.writerows(rows)

    n_res = sum(1 for r in rows if r["kind"] == "result")
    print(f"{len(rows)} files copied into {DEST} ({n_res} result JSONs)")
    if changed:
        print(f"  {len(changed)} files CHANGED since last collect")
    print(f"collector ran at {datetime.now():%Y-%m-%d %H:%M}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
