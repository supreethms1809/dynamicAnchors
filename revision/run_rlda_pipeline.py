"""Class-parallel RLDA-only pipeline (train -> infer -> evaluate -> baselines).

One (dataset, algo) leg per invocation. No MADA, no tables. Everything lands
under a self-contained --root so it never collides with prior sweeps.

  python revision/run_rlda_pipeline.py --algo ddpg --datasets iris  --seed 42
  python revision/run_rlda_pipeline.py --algo sac  --datasets wine  --seed 42

Layout under --root (default runs/rlda_ext_seed42):
  output/{ds}_rlda_{algo}_seed{seed}/training/{algo}_single_agent_sb3_*/
  results/{algo}/{ds}__rlda__seed{seed}__tp0p90__tc0p10.json      (+ 4 baselines)
  logs/{ds}_{algo}/{rlda_train,rlda_infer,eval_rlda,baselines}_seed{seed}.log
"""
from __future__ import annotations

import argparse
import os
import shutil
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
# k=1: best box only. k=5 OR-unions were manufacturing conflict (wine conf=1).
TAU_P, TAU_C, K = 0.90, 0.10, 1

# Extended budget: 3x the run_paper_seed.py per-class step counts.
BUDGET_MULT = 3

# base sa_timesteps == run_paper_seed.py totals; per-class = total / n_classes.
DATASET_CONFIGS: Dict[str, Dict[str, int]] = {
    "iris":          {"sa_timesteps": 90_000,  "n_instances": 20, "n_classes": 3},
    "wine":          {"sa_timesteps": 120_000, "n_instances": 20, "n_classes": 3},
    "breast_cancer": {"sa_timesteps": 90_000,  "n_instances": 20, "n_classes": 2},
    "synthetic":     {"sa_timesteps": 90_000,  "n_instances": 20, "n_classes": 2},
    "uci_credit":    {"sa_timesteps": 360_000, "n_instances": 25, "n_classes": 2},
    "folktables_income_CA_2018": {"sa_timesteps": 720_000, "n_instances": 25, "n_classes": 2},
    "housing":       {"sa_timesteps": 120_000, "n_instances": 20, "n_classes": 4},
    "covtype":       {"sa_timesteps": 315_000, "n_instances": 25, "n_classes": 7},
    "uci_adult":     {"sa_timesteps": 360_000, "n_instances": 25, "n_classes": 2},
}
for _cfg in DATASET_CONFIGS.values():
    _cfg["sa_timesteps"] *= BUDGET_MULT

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

# ---- filled in from CLI in main() ----
ALGO = "ddpg"
SUBDIR = "ddpg_single_agent"          # training-subdir filter for latest_exp
ROOT = REPO / "runs" / "rlda_ext_seed42"
OUT_DIR = ROOT / "output"
RESULTS = ROOT / "results" / ALGO
LOG_DIR = ROOT / "logs"
FORCE = False                         # re-run infer + Track A + Track B; never retrain
FORCE_TRAIN = False                   # also retrain (rare; wipes nothing, starts a new exp)
SKIP_TRAIN = False                    # never train; fail if no experiment dir
CLASSIFIER_PATH = None                # if set + exists: shards load it, skip classifier fit
CLASSIFIER_DEVICE = None              # device for the (one-time) classifier fit


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


def rlda_out(dataset: str, seed: int) -> Path:
    return OUT_DIR / f"{dataset}_rlda_{ALGO}_seed{seed}"


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
    ]
    if not cands:
        cands = [p for p in training.iterdir() if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def rlda_rules(dataset: str, seed: int) -> Path | None:
    exp = latest_exp(rlda_out(dataset, seed), SUBDIR)
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules_single_agent.json"
    return p if p.exists() else None


def result_path(dataset: str, method: str, seed: int) -> Path:
    def _fmt(x: float) -> str:
        return f"{x:.2f}".replace(".", "p")
    return RESULTS / f"{dataset}__{method}__seed{seed}__tp{_fmt(TAU_P)}__tc{_fmt(TAU_C)}.json"


def shared_classifier_path(dataset: str, seed: int) -> Path:
    return ROOT / "classifiers" / f"{dataset}_seed{seed}.pth"


def ensure_dataset_classifier(dataset: str, seed: int, device: str) -> Path:
    """One black box per (dataset, seed), shared by every class shard and algo."""
    if CLASSIFIER_PATH and Path(CLASSIFIER_PATH).exists():
        return Path(CLASSIFIER_PATH)
    dest = shared_classifier_path(dataset, seed)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"{dataset}: using shared classifier {dest}")
        return dest
    clf_device = CLASSIFIER_DEVICE or device
    log(f"{dataset}: fitting ONE shared classifier -> {dest} (device={clf_device})")
    return fit_or_load(dataset, dest, seed=seed, device=clf_device)


def train_rlda(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> None:
    has = latest_exp(rlda_out(dataset, seed), SUBDIR)
    if has is not None and not FORCE_TRAIN:
        why = "--skip-train" if SKIP_TRAIN else "artifacts present (--force does not retrain)"
        log(f"{dataset} RLDA[{ALGO}] seed {seed}: skip train ({why})")
        return
    if SKIP_TRAIN and has is None:
        raise SystemExit(f"No RLDA[{ALGO}] experiment dir for {dataset} seed {seed} (--skip-train)")
    out = rlda_out(dataset, seed)
    out.mkdir(parents=True, exist_ok=True)
    n_cls = int(cfg.get("n_classes", 1))
    per_class = cfg["sa_timesteps"] // max(n_cls, 1)
    clf_path = ensure_dataset_classifier(dataset, seed, device)
    extra = ["--skip_eda", "--classifier_path", str(clf_path)]
    log(f"{dataset} RLDA[{ALGO}]: all class shards LOAD {clf_path} (no per-shard fit)")
    if CLASSIFIER_DEVICE:
        extra += ["--classifier_device", CLASSIFIER_DEVICE]
    run(
        [
            sys.executable, str(REPO / "single_agent" / "run_parallel_classes.py"),
            "--dataset", dataset, "--algorithm", ALGO, "--seed", str(seed),
            "--n_classes", str(n_cls), "--parallel_classes", str(n_cls),
            "--total_timesteps", str(per_class), "--n_envs", "1",
            "--device", device, "--output_dir", str(out) + "/",
            "--extra_args", *extra,
        ],
        log_file=ds_log(dataset, f"rlda_train_seed{seed}.log"),
    )


def ensure_best_models(exp: Path) -> None:
    final_dir = exp / "final_model"
    if not final_dir.is_dir():
        return
    for final in sorted(final_dir.glob("class_*.zip")):
        cls = final.stem
        dest_dir = exp / "best_model" / cls
        dest = dest_dir / "best_model.zip"
        if dest.exists():
            continue
        # G-05: DO NOT copy final -> best. Inference runs with
        # prefer_model='best', so this handed it FINAL weights from a file named
        # best_model.zip, indistinguishable downstream and unrecorded in the
        # result JSON. It fired exactly when validation selection failed
        # (P-03), so the two failures chained: selection silently fails, then
        # the unselected checkpoint is silently promoted.
        raise SystemExit(
            f"{cls}: no validation-selected best_model.zip in {dest_dir}. "
            f"Refusing to promote {final.name} to 'best' -- that would report "
            f"final weights as validation-selected. Fix training (best-model "
            f"selection scored -inf for every evaluation) or rerun with "
            f"prefer_model='final' explicitly."
        )


def infer_rlda(dataset: str, seed: int, cfg: Dict[str, int]) -> Path:
    existing = rlda_rules(dataset, seed)
    if FORCE:
        remove_stale(existing, "rules JSON")
    elif existing is not None:
        log(f"{dataset} RLDA[{ALGO}] seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(rlda_out(dataset, seed), SUBDIR)
    if exp is None:
        raise SystemExit(f"No RLDA[{ALGO}] experiment dir for {dataset} seed {seed}")
    ensure_best_models(exp)
    run(
        [
            sys.executable, str(REPO / "single_agent" / "single_agent_inference.py"),
            "--experiment_dir", str(exp), "--dataset", dataset, "--seed", str(seed),
            "--n_instances_per_class", str(cfg["n_instances"]),
            "--n_rollouts_per_instance", "1",
            "--n_class_based_rollouts", "5",
        ],
        log_file=ds_log(dataset, f"rlda_infer_seed{seed}.log"),
    )
    rules = rlda_rules(dataset, seed)
    if rules is None:
        raise SystemExit(f"RLDA[{ALGO}] inference produced no rules for {dataset} seed {seed}")
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


def baselines(dataset: str, seed: int) -> None:
    exp = latest_exp(rlda_out(dataset, seed), SUBDIR)
    if exp is None:
        raise SystemExit(f"No RLDA[{ALGO}] classifier for {dataset} seed {seed}")
    clf = exp / "classifier.pth"
    if not clf.exists():
        raise SystemExit(f"Missing classifier {clf}")
    needed = ["cart", "random_search", "sp_anchors", "greedy_anchors"]
    if all(result_path(dataset, m, seed).exists() for m in needed):
        log(f"{dataset} baselines[{ALGO}] seed {seed}: all result JSONs present, skip")
        return
    run(
        [
            sys.executable, "-m", "revision.baselines",
            "--dataset", dataset, "--seed", str(seed), "--k", str(K),
            "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
            "--classifier_path", str(clf),
            "--methods", "cart", "random_search", "sp_anchors", "greedy_anchors",
            "--budget_per_class", "5", "--n_candidates", "256",
            "--out_dir", str(RESULTS),
        ],
        log_file=ds_log(dataset, f"baselines_seed{seed}.log"),
    )


def instance_result_path(dataset: str, method: str, seed: int) -> Path:
    return RESULTS / f"{dataset}__{method}__instances__seed{seed}.json"


def evaluate_instances(dataset: str, method: str, rules: Path, seed: int) -> None:
    dest = instance_result_path(dataset, method, seed)
    if FORCE:
        remove_stale(dest, "Track B JSON")
    elif dest.exists():
        log(f"{dataset} {method}[{ALGO}] seed {seed}: instance JSON exists, skip Track B")
        return
    exp = latest_exp(rlda_out(dataset, seed), SUBDIR)
    if exp is None:
        raise SystemExit(f"No RLDA[{ALGO}] experiment dir for Track B on {dataset}")
    clf = exp / "classifier.pth"
    track_a = result_path(dataset, method, seed)
    large = dataset.startswith("folktables") or dataset in {
        "covtype", "housing", "uci_adult",
    }
    max_per = 200 if large else 0
    cmd = [
        sys.executable, "-m", "revision.evaluate_instances",
        "--dataset", dataset, "--method", method, "--algo", ALGO,
        "--seed", str(seed), "--experiment_dir", str(exp),
        "--classifier_path", str(clf), "--out_dir", str(RESULTS),
        "--rules_file", str(rules),
        "--max_per_pred", str(max_per),
    ]
    if track_a.exists():
        cmd += ["--track_a_json", str(track_a)]
    run(cmd, log_file=ds_log(dataset, f"eval_instances_{method}_seed{seed}.log"))


def run_dataset(dataset: str, seed: int, device: str) -> None:
    cfg = DATASET_CONFIGS[dataset]
    n_cls = cfg["n_classes"]
    per_class = cfg["sa_timesteps"] // n_cls
    log(
        f"=== {dataset} seed {seed}  RLDA[{ALGO}]  "
        f"{per_class} steps/class x {n_cls} classes ({cfg['sa_timesteps']} total) / "
        f"{cfg['n_instances']} inst/class ==="
    )
    train_rlda(dataset, seed, cfg, device)
    rules = infer_rlda(dataset, seed, cfg)
    evaluate(dataset, "rlda", rules, seed)
    baselines(dataset, seed)
    evaluate_instances(dataset, "rlda", rules, seed)
    log(f"=== {dataset} seed {seed} RLDA[{ALGO}] done ===")


def main() -> None:
    p = argparse.ArgumentParser(description="Class-parallel RLDA-only pipeline (one algo)")
    p.add_argument("--algo", required=True, choices=["ddpg", "sac"])
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
    p.add_argument("--classifier-path", default=None,
                   help="Pre-trained classifier.pth for all shards to load (skips per-shard fit).")
    p.add_argument("--classifier-device", default=None,
                   choices=["cpu", "cuda", "mps", "auto"],
                   help="Device for the classifier fit (default: same as --device).")
    args = p.parse_args()

    global ALGO, SUBDIR, ROOT, OUT_DIR, RESULTS, LOG_DIR, FORCE, FORCE_TRAIN, SKIP_TRAIN
    global CLASSIFIER_PATH, CLASSIFIER_DEVICE
    ALGO = args.algo
    SUBDIR = f"{ALGO}_single_agent"
    ROOT = Path(args.root).resolve()
    OUT_DIR = ROOT / "output"
    RESULTS = ROOT / "results" / ALGO
    LOG_DIR = ROOT / "logs"
    FORCE = bool(args.force)
    FORCE_TRAIN = bool(args.force_train)
    SKIP_TRAIN = bool(args.skip_train)
    CLASSIFIER_PATH = args.classifier_path
    CLASSIFIER_DEVICE = args.classifier_device

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    failed: List[str] = []
    log(f"RLDA-only sweep algo={ALGO} datasets={args.datasets} seed={args.seed} "
        f"device={args.device} root={ROOT} budget_mult={BUDGET_MULT} "
        f"force={FORCE} skip_train={SKIP_TRAIN} force_train={FORCE_TRAIN}")
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
    log(f"Done. RLDA[{ALGO}] results in {RESULTS}")


if __name__ == "__main__":
    main()
