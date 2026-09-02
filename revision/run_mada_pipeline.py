"""MADA-only pipeline (train -> infer -> evaluate). One (dataset, algo) leg per call.

Multi-agent counterpart of run_rlda_pipeline.py. No baselines (those are the
RLDA-classifier arm, already produced). Lands in the same --root tree.

  python revision/run_mada_pipeline.py --algo maddpg --datasets iris --seed 42
  python revision/run_mada_pipeline.py --algo masac  --datasets wine --seed 42

Layout under --root (default runs/rlda_ext_seed42):
  output/{ds}_mada_{algo}_seed{seed}/training/{algo}_anchor_mlp__*/
  results/{algo}/{ds}__mada__seed{seed}__tp0p90__tc0p10.json
  logs/{ds}_{algo}/{mada_train,mada_infer,eval_mada}_seed{seed}.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from print_leg import summarize  # noqa: E402
from fit_shared_classifier import fit_or_load  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# k=1: same Track A contract as the empirical-Fid RLDA sweep.
TAU_P, TAU_C, K = 0.90, 0.10, 1

# 1x base budget (matches run_paper_seed.py ma_frames).
BUDGET_MULT = 1

DATASET_CONFIGS: Dict[str, Dict[str, int]] = {
    "iris":          {"ma_frames": 360_000, "n_instances": 20, "n_classes": 3},
    "wine":          {"ma_frames": 360_000, "n_instances": 20, "n_classes": 3},
    "breast_cancer": {"ma_frames": 360_000, "n_instances": 20, "n_classes": 2},
    "synthetic":     {"ma_frames": 360_000, "n_instances": 20, "n_classes": 2},
    "uci_credit":    {"ma_frames": 720_000, "n_instances": 25, "n_classes": 2},
    "folktables_income_CA_2018": {"ma_frames": 1_080_000, "n_instances": 25, "n_classes": 2},
    "housing":       {"ma_frames": 720_000, "n_instances": 20, "n_classes": 4},
    "uci_adult":     {"ma_frames": 720_000, "n_instances": 25, "n_classes": 2},
    # C-23 additions, 2026-09-02.
    # heloc: FICO Explainable ML Challenge credit-risk set (10k x 22, balanced).
    # sick / mammography: IMBALANCED medical (6.1% and 2.3% positive), chosen to
    # stress the class-union shared reward on a minority class.
    "heloc":         {"ma_frames": 720_000, "n_instances": 25, "n_classes": 2},
    "sick":          {"ma_frames": 360_000, "n_instances": 20, "n_classes": 2},
    "mammography":   {"ma_frames": 720_000, "n_instances": 25, "n_classes": 2},
}
for _cfg in DATASET_CONFIGS.values():
    _cfg["ma_frames"] *= BUDGET_MULT

ENV = os.environ.copy()
ENV.update({
    "WANDB_MODE": "offline",
    "WANDB_SILENT": "true",
    "DISABLE_WANDB": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
})

# ---- filled from CLI ----
ALGO = "maddpg"
SUBDIR = "maddpg"
ROOT = REPO / "runs" / "rlda_ext_seed42"
OUT_DIR = ROOT / "output"
RESULTS = ROOT / "results" / ALGO
LOG_DIR = ROOT / "logs"
FORCE = False
FORCE_TRAIN = False
SKIP_TRAIN = False
CLASSIFIER_PATH = None
CLASSIFIER_DEVICE = None
MAX_N_FRAMES = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def remove_stale(path: Optional[Path], label: str) -> None:
    """Unlink an existing artifact so skip-if-exists cannot win over --force."""
    if path is None or not path.exists():
        return
    path.unlink()
    log(f"removed stale {label}: {path}")


def run(cmd, cwd=None, log_file: Optional[Path] = None) -> None:
    log("$ " + " ".join(cmd))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as fh:
            proc = subprocess.run(cmd, cwd=cwd or REPO, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=cwd or REPO, env=ENV)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def mada_out(dataset: str, seed: int) -> Path:
    return OUT_DIR / f"{dataset}_mada_{ALGO}_seed{seed}"


def ds_log(dataset: str, name: str) -> Path:
    safe = dataset.replace("/", "_")
    return LOG_DIR / f"{safe}_{ALGO}" / name


def latest_exp(output_dir: Path, must_contain: str | None = None) -> Path | None:
    training = output_dir / "training"
    if not training.is_dir():
        return None
    cands = [p for p in training.iterdir() if p.is_dir()]
    if must_contain:
        cands = [p for p in cands if must_contain in p.name]
    cands = [
        p for p in cands
        if (p / "classifier.pth").exists() or list(p.glob("*.zip")) or list(p.glob("checkpoint*"))
        or (p / "inference").is_dir()
    ]
    if not cands:
        cands = [p for p in training.iterdir() if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def mada_rules(dataset: str, seed: int) -> Path | None:
    exp = latest_exp(mada_out(dataset, seed), SUBDIR)
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules.json"
    return p if p.exists() else None


def result_path(dataset: str, method: str, seed: int) -> Path:
    def _fmt(x: float) -> str:
        return f"{x:.2f}".replace(".", "p")
    return RESULTS / f"{dataset}__{method}__seed{seed}__tp{_fmt(TAU_P)}__tc{_fmt(TAU_C)}.json"


def shared_classifier_path(dataset: str, seed: int) -> Path:
    return ROOT / "classifiers" / f"{dataset}_seed{seed}.pth"


def ensure_dataset_classifier(dataset: str, seed: int, device: str) -> Path:
    if CLASSIFIER_PATH and Path(CLASSIFIER_PATH).exists():
        return Path(CLASSIFIER_PATH)
    dest = shared_classifier_path(dataset, seed)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"{dataset}: using shared classifier {dest}")
        return dest
    clf_device = CLASSIFIER_DEVICE or device
    log(f"{dataset}: fitting ONE shared classifier -> {dest} (device={clf_device})")
    return fit_or_load(dataset, dest, seed=seed, device=clf_device)


def train_mada(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> None:
    has = latest_exp(mada_out(dataset, seed), SUBDIR)
    if has is not None and not FORCE_TRAIN:
        why = "--skip-train" if SKIP_TRAIN else "artifacts present (--force does not retrain)"
        log(f"{dataset} MADA[{ALGO}] seed {seed}: skip train ({why})")
        return
    if SKIP_TRAIN and has is None:
        raise SystemExit(f"No MADA[{ALGO}] experiment dir for {dataset} seed {seed} (--skip-train)")
    out = mada_out(dataset, seed)
    out.mkdir(parents=True, exist_ok=True)
    clf_path = ensure_dataset_classifier(dataset, seed, device)
    extra = ["--classifier_path", str(clf_path)]
    if CLASSIFIER_DEVICE:
        extra += ["--classifier_device", CLASSIFIER_DEVICE]
    log(f"{dataset} MADA[{ALGO}]: LOAD {clf_path} (no MADA-side fit)")
    run(
        [
            sys.executable, "driver.py",
            "--dataset", dataset, "--algorithm", ALGO, "--seed", str(seed),
            "--skip_eda", "--max_n_frames", str(cfg["ma_frames"]),
            "--device", device, "--output_dir", str(out) + "/",
            *extra,
        ],
        cwd=REPO / "BenchMARL",
        log_file=ds_log(dataset, f"mada_train_seed{seed}.log"),
    )


def infer_mada(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> Path:
    existing = mada_rules(dataset, seed)
    if FORCE:
        remove_stale(existing, "rules JSON")
    elif existing is not None:
        log(f"{dataset} MADA[{ALGO}] seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(mada_out(dataset, seed), SUBDIR)
    if exp is None:
        raise SystemExit(f"No MADA[{ALGO}] experiment dir for {dataset} seed {seed}")
    run(
        [
            sys.executable, "inference.py",
            "--experiment_dir", str(exp), "--dataset", dataset, "--seed", str(seed),
            "--n_instances_per_class", str(cfg["n_instances"]), "--device", device,
            "--exploration_mode", "mean",
            "--n_class_based_rollouts", "5",
        ],
        cwd=REPO / "BenchMARL",
        log_file=ds_log(dataset, f"mada_infer_seed{seed}.log"),
    )
    rules = mada_rules(dataset, seed)
    if rules is None:
        raise SystemExit(f"MADA[{ALGO}] inference produced no rules for {dataset} seed {seed}")
    return rules


def evaluate(dataset: str, method: str, rules: Path, seed: int) -> None:
    dest = result_path(dataset, method, seed)
    if FORCE:
        remove_stale(dest, "Track A JSON")
    elif dest.exists():
        log(f"{dataset} {method}[{ALGO}] seed {seed}: result JSON exists, skip evaluate")
        log("\n" + summarize(dataset, method, seed, str(rules), results_dir=str(RESULTS)))
        return
    run(
        [
            sys.executable, "-m", "revision.evaluate",
            "--rules_file", str(rules), "--dataset", dataset, "--method", method,
            "--seed", str(seed), "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
            "--k", str(K), "--out_dir", str(RESULTS),
        ],
        log_file=ds_log(dataset, f"eval_{method}_seed{seed}.log"),
    )
    log("\n" + summarize(dataset, method, seed, str(rules), results_dir=str(RESULTS)))


def instance_result_path(dataset: str, method: str, seed: int) -> Path:
    return RESULTS / f"{dataset}__{method}__instances__seed{seed}.json"


def evaluate_instances(dataset: str, method: str, rules: Path, seed: int, device: str) -> None:
    dest = instance_result_path(dataset, method, seed)
    if FORCE:
        remove_stale(dest, "Track B JSON")
    elif dest.exists():
        log(f"{dataset} {method}[{ALGO}] seed {seed}: instance JSON exists, skip Track B")
        return
    exp = latest_exp(mada_out(dataset, seed), SUBDIR)
    if exp is None:
        raise SystemExit(f"No MADA[{ALGO}] experiment dir for Track B on {dataset}")
    clf = exp / "classifier.pth"
    track_a = result_path(dataset, method, seed)
    large = dataset.startswith("folktables") or dataset in {"housing", "uci_adult"}
    max_per = 200 if large else 0
    cmd = [
        sys.executable, "-m", "revision.evaluate_instances",
        "--dataset", dataset, "--method", method, "--algo", ALGO,
        "--seed", str(seed), "--experiment_dir", str(exp),
        "--classifier_path", str(clf), "--out_dir", str(RESULTS),
        "--rules_file", str(rules), "--device", device,
        "--max_per_pred", str(max_per),
    ]
    if track_a.exists():
        cmd += ["--track_a_json", str(track_a)]
    run(cmd, log_file=ds_log(dataset, f"eval_instances_{method}_seed{seed}.log"))


def run_dataset(dataset: str, seed: int, device: str) -> None:
    cfg = dict(DATASET_CONFIGS[dataset])
    if MAX_N_FRAMES is not None:
        cfg["ma_frames"] = int(MAX_N_FRAMES)
    log(
        f"=== {dataset} seed {seed}  MADA[{ALGO}]  {cfg['ma_frames']} frames / "
        f"{cfg['n_instances']} inst/class ==="
    )
    train_mada(dataset, seed, cfg, device)
    rules = infer_mada(dataset, seed, cfg, device)
    evaluate(dataset, "mada", rules, seed)
    evaluate_instances(dataset, "mada", rules, seed, device)
    log(f"=== {dataset} seed {seed} MADA[{ALGO}] done ===")


def main() -> None:
    p = argparse.ArgumentParser(description="MADA-only pipeline (one algo)")
    p.add_argument("--algo", required=True, choices=["maddpg", "masac"])
    p.add_argument(
        "--datasets", nargs="+",
        default=["iris", "wine", "breast_cancer", "uci_credit"],
        choices=list(DATASET_CONFIGS.keys()),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--root", default=str(REPO / "runs" / "rlda_ext_seed42"))
    p.add_argument(
        "--force", action="store_true",
        help="Re-run inference, Track A, and Track B. Deletes those outputs first. "
             "Does not retrain (use --force-train).",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Never train. Fail if no experiment directory exists.",
    )
    p.add_argument(
        "--force-train", action="store_true",
        help="Retrain even if an experiment directory already exists.",
    )
    p.add_argument(
        "--classifier_path", default=None,
        help="If set, load this .pth instead of ROOT/classifiers/{ds}_seed{seed}.pth",
    )
    p.add_argument("--classifier_device", default=None)
    p.add_argument(
        "--max_n_frames", type=int, default=None,
        help="Override per-dataset ma_frames. Must be a multiple of "
             "evaluation_interval (24000) so FidCov eval fires.",
    )
    p.add_argument(
        "--k", type=int, default=None,
        help="Top-k union at evaluation (default: pipeline K).",
    )
    args = p.parse_args()

    global ALGO, SUBDIR, ROOT, OUT_DIR, RESULTS, LOG_DIR, FORCE, FORCE_TRAIN, SKIP_TRAIN
    global CLASSIFIER_PATH, CLASSIFIER_DEVICE, MAX_N_FRAMES, K
    ALGO = args.algo
    SUBDIR = ALGO
    # MUST be absolute: train_mada runs BenchMARL/driver.py with cwd=BenchMARL/,
    # so a relative --output_dir would resolve under BenchMARL/.
    ROOT = Path(args.root).resolve()
    OUT_DIR = ROOT / "output"
    RESULTS = ROOT / "results" / ALGO
    LOG_DIR = ROOT / "logs"
    FORCE = bool(args.force)
    FORCE_TRAIN = bool(args.force_train)
    SKIP_TRAIN = bool(args.skip_train)
    CLASSIFIER_PATH = args.classifier_path
    CLASSIFIER_DEVICE = args.classifier_device
    MAX_N_FRAMES = args.max_n_frames
    if args.k is not None:
        K = int(args.k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    failed: List[str] = []
    log(f"MADA-only sweep algo={ALGO} datasets={args.datasets} seed={args.seed} "
        f"device={args.device} root={ROOT} frames={MAX_N_FRAMES or 'per-dataset'} "
        f"force={FORCE} skip_train={SKIP_TRAIN} force_train={FORCE_TRAIN} k={K}")
    for dataset in args.datasets:
        try:
            run_dataset(dataset, args.seed, args.device)
        except Exception as exc:
            failed.append(dataset)
            log(f"FAILED {dataset}: {exc}")
            traceback.print_exc()
            continue
    if failed:
        log(f"Done with failures: {failed}")
        raise SystemExit(1)
    log(f"Done. MADA[{ALGO}] results in {RESULTS}")


if __name__ == "__main__":
    main()
