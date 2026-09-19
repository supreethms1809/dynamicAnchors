#!/usr/bin/env python3
"""Copy the final-configuration artifacts the paper uses into ONE directory.

Only matched, final-config results go in (top-K-capped RLDA pools, per-class
MADA checkpoints, rule selection + reporting under the chosen coverage basis),
so numbers written into the paper can never be mixed with pre-fix trees.

  runs/paper_final/
    README.md                 what is here and how it was produced
    MANIFEST.csv              every file: label, seed, dataset, basis, source, sha256
    results/<basis>/<label>/  revision.evaluate / revision.baselines JSONs, all seeds
    anc/<basis>/<label>/      Anchors-sampler Fid rescore CSVs, one per seed
    trackb/<label>/           instance-level (Track B) JSONs from the training trees
    k_sweep/                  seed-42 union-size sweep on the matched pools
    ablations/<family>/<arm>/ multi-seed ablation JSONs (revision/run_ablation_queue.sh);
                              the reference arm is results/predicted/MADA-emp (and RLDA-emp)

Files are COPIED (not linked); rerunning refreshes changed files and reports them.

  python revision/collect_paper_results.py
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "revision"))
from recompute_seed42 import BASE_LABEL, DATASETS, ORDER, rl_sources  # noqa: E402

DEST = REPO / "runs" / "paper_final"
SEEDS = [42, 43, 44, 45, 46]
BASES = ["predicted", "true_label"]
METHOD = {"RLDA-emp": "rlda", "RLDA-pert": "rlda", "MADA-emp": "mada", "MADA-pert": "mada"}
METHOD.update({lab: m for (_, m), lab in BASE_LABEL.items()})
SUBDIR = {lab: sub for (sub, _), lab in BASE_LABEL.items()}


def d(label: str) -> str:
    """Directory name for a method label (no shell glob characters)."""
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


def main() -> int:
    raise SystemExit(
        "collect_paper_results.py still copies the C_train=y lock. That tree is now "
        "runs/archived_ctrain_y_tc0p10_20260918/. Live numbers are trained into "
        "runs/paper_final/{emp,pert}_tc{0p10,0p20}/. Do not run this collector."
    )
    rows, changed, missing = [], [], []
    for seed in SEEDS:
        rec = REPO / "runs" / f"recompute_seed{seed}"
        for basis in BASES:
            for label in ORDER:
                method = METHOD[label]
                sub = SUBDIR.get(label, label)
                for ds in DATASETS:
                    name = f"{ds}__{method}__seed{seed}__tp0p90__tc0p10.json"
                    src = rec / basis / sub / name
                    meta = {"kind": "result", "label": label, "seed": seed, "dataset": ds, "basis": basis}
                    if src.exists():
                        put(src, DEST / "results" / basis / d(label) / name, meta, rows, changed)
                    else:
                        missing.append(f"{basis} {label} seed{seed} {ds}")
                anc = rec / basis / sub / "_dual" / "rescore_rules.csv"
                if anc.exists():
                    put(anc, DEST / "anc" / basis / d(label) / f"seed{seed}_rescore_rules.csv",
                        {"kind": "anc", "label": label, "seed": seed, "dataset": "*", "basis": basis},
                        rows, changed)
        for label, (src_dir, method) in rl_sources(seed).items():
            for f in sorted(Path(src_dir).glob(f"*__{method}__instances__seed{seed}.json")):
                put(f, DEST / "trackb" / d(label) / f.name,
                    {"kind": "trackb", "label": label, "seed": seed,
                     "dataset": f.name.split("__")[0], "basis": ""}, rows, changed)
    abl = REPO / "runs" / "ablations"
    for f in sorted(abl.glob("*/*/results/*/*.json")) if abl.is_dir() else []:
        if "_archive" in f.parts[-2]:
            continue
        fam, arm = f.parts[-5], f.parts[-4]
        put(f, DEST / "ablations" / fam / arm / f.name,
            {"kind": "ablation", "label": f"{fam}/{arm}", "seed": f.name.split("__seed")[-1].split("__")[0].split(".")[0],
             "dataset": f.name.split("__")[0], "basis": "predicted"}, rows, changed)
    ks = REPO / "runs" / "k_sweep_matched_seed42"
    for f in sorted(ks.rglob("*")) if ks.is_dir() else []:
        if f.is_file() and "logs" not in f.parts:
            put(f, DEST / "k_sweep" / f.relative_to(ks),
                {"kind": "k_sweep", "label": "", "seed": 42, "dataset": "", "basis": "predicted"},
                rows, changed)

    DEST.mkdir(parents=True, exist_ok=True)
    with open(DEST / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "label", "seed", "dataset", "basis", "file", "source", "sha256"])
        w.writeheader()
        w.writerows(rows)

    n_res = {b: sum(1 for r in rows if r["kind"] == "result" and r["basis"] == b) for b in BASES}
    total = len(ORDER) * len(DATASETS) * len(SEEDS)
    (DEST / "README.md").write_text(f"""# Paper results (final configuration)

Generated by `revision/collect_paper_results.py` at {datetime.now():%Y-%m-%d %H:%M}.
Copies, not links. Every file's origin and hash is in `MANIFEST.csv`.
Do not write into this directory by hand; rerun the collector instead.

## Configuration
- RLDA: rule pools capped at top-K (5) by score after dedupe + NMS, as MADA.
- MADA: per-class checkpoint selection (fixed box-support score).
- -emp = trained / searched with empirical Fid; -pert = Anchors-style perturbed Fid.
- Selection on D_val (lcb_coverage, min_support 10, marginal-gain union, k=1),
  reporting on D_test, tau_p=0.90, tau_c=0.10.

## Metric definitions
- Fid (rule/class union) = |{{x in B : f_hat(x)=c}}| / |B|
- class Cov, basis `predicted` (headline) = P(x in B | f_hat(x)=c)
- class Cov, basis `true_label` (appendix) = P(x in B | y=c)
- global Fid/Cov/Eff: abstain if no class fires, tie-break by union Fid;
  Cov = 1 - abstention; Eff = Fid x Cov
- Anc = rule Fid under the Anchors perturbation sampler (anc/)

## Ablations (ablations/, basis predicted, 11 datasets without wyodot, seeds 42-46)
Base = headline MADA-emp config; each arm changes only:
- coord/no_same_class: agents_per_class=1 (3x frames/agent, matched agent-steps)
- coord/no_cross_class: shared_terminal_bonus, inter_class_overlap_weight, shared_reward_weight = 0
- coord/no_coord: both of the above + same_class_diversity_weight=0
- reward/no_width (gamma=0), no_drift, no_anchor_drift, no_local (all three)
- algo/sac_masac: RLDA=SAC, MADA=MASAC
- classifier/random_forest: RF black box (RLDA, MADA, 4 empirical baselines)
- overlap/w050, overlap/w100: inter_class_overlap_weight 0.5 / 1.0 (headline 0.75)
Ablation files so far: {sum(1 for r in rows if r["kind"] == "ablation")}

## Completeness
- results/predicted : {n_res['predicted']}/{total} (10 methods x 12 datasets x 5 seeds)
- results/true_label: {n_res['true_label']}/{total}
""")
    print(f"{len(rows)} files in {DEST}")
    for b in BASES:
        print(f"  results/{b}: {n_res[b]}/{total}")
    if changed:
        print(f"  {len(changed)} files CHANGED since last collect:")
        for c in changed[:20]:
            print("    ", c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
