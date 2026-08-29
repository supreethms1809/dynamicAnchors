"""One-seed revision sweep on paper datasets — SAC / MASAC variant.

Sibling of run_paper_seed.py. Identical pipeline (train -> infer -> evaluate ->
baselines -> tables) but:
  * RLDA uses single-agent SAC     -> output/{ds}_rlda_sac_seed{seed}
  * MADA uses BenchMARL MASAC      -> output/{ds}_mada_masac_seed{seed}
  * eval JSONs                     -> revision/results_sac/
  * per-dataset logs               -> revision/logs/{ds}_sac/
  * tables / figures               -> paper/tables_sac/ , paper/figures_sac/

Nothing the DDPG/MADDPG sweep produced is touched. Skip-if-exists so a crash
can resume; continues to the next dataset on failure.

  python revision/run_paper_seed_sac.py --datasets wine --seed 42 --device cpu
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
from print_leg import summarize  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
# Same thresholds as run_paper_seed.py.
TAU_P, TAU_C, K = 0.90, 0.10, 5
RESULTS = REPO / "revision" / "results_sac"
LOG_DIR = REPO / "revision" / "logs"
TABLES_DIR = REPO / "paper" / "tables_sac"
FIGURES_DIR = REPO / "paper" / "figures_sac"
FORCE = False
SKIP_RLDA = set()

SA_ALGO = "sac"
MA_ALGO = "masac"
SA_SUBDIR = "sac_single_agent"   # training-subdir filter for latest_exp
MA_SUBDIR = "masac"              # BenchMARL names it masac_anchor_mlp__...

# SAC spends its budget in the high-entropy exploration phase: at the DDPG-equal
# budget, ent_coef never decayed (iris ~0.9 throughout) and best checkpoints came
# at or near the final eval (wine class_1 best = step 40k/40k). Bump the budget 3x
# to see whether the trend breaks before touching target_entropy / ent_coef.
BUDGET_MULT = 3

# sa_timesteps / ma_frames / n_instances: same table as run_paper_seed.py.
DATASET_CONFIGS: Dict[str, Dict[str, int]] = {
    "iris": {
        "sa_timesteps": 90_000,
        "ma_frames": 360_000,
        "n_instances": 20,
        "n_classes": 3,
    },
    "wine": {
        "sa_timesteps": 120_000,
        "ma_frames": 360_000,
        "n_instances": 20,
        "n_classes": 3,
    },
    "breast_cancer": {
        "sa_timesteps": 90_000,
        "ma_frames": 360_000,
        "n_instances": 20,
        "n_classes": 2,
    },
    "synthetic": {
        "sa_timesteps": 90_000,
        "ma_frames": 360_000,
        "n_instances": 20,
        "n_classes": 2,
    },
    "uci_credit": {
        "sa_timesteps": 360_000,
        "ma_frames": 720_000,
        "n_instances": 25,
        "n_classes": 2,
    },
    "folktables_income_CA_2018": {
        "sa_timesteps": 720_000,
        "ma_frames": 1_080_000,
        "n_instances": 25,
        "n_classes": 2,
    },
}

# Apply the SAC budget multiplier to the training-length knobs only.
for _cfg in DATASET_CONFIGS.values():
    _cfg["sa_timesteps"] *= BUDGET_MULT
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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return REPO / "output" / f"{dataset}_rlda_{SA_ALGO}_seed{seed}"


def mada_out(dataset: str, seed: int) -> Path:
    return REPO / "output" / f"{dataset}_mada_{MA_ALGO}_seed{seed}"


def ds_log(dataset: str, name: str) -> Path:
    safe = dataset.replace("/", "_")
    return LOG_DIR / f"{safe}_sac" / name


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
    exp = latest_exp(rlda_out(dataset, seed), SA_SUBDIR)
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules_single_agent.json"
    return p if p.exists() else None


def mada_rules(dataset: str, seed: int) -> Path | None:
    exp = latest_exp(mada_out(dataset, seed), MA_SUBDIR)
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules.json"
    return p if p.exists() else None


def result_path(dataset: str, method: str, seed: int) -> Path:
    def _fmt(x: float) -> str:
        return f"{x:.2f}".replace(".", "p")
    return RESULTS / f"{dataset}__{method}__seed{seed}__tp{_fmt(TAU_P)}__tc{_fmt(TAU_C)}.json"


def train_rlda(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> None:
    if not FORCE and latest_exp(rlda_out(dataset, seed), SA_SUBDIR) is not None:
        log(f"{dataset} RLDA seed {seed}: training artifacts present, skip train")
        return
    out = rlda_out(dataset, seed)
    out.mkdir(parents=True, exist_ok=True)
    n_cls = int(cfg.get("n_classes", 1))
    per_class = cfg["sa_timesteps"] // max(n_cls, 1)
    run(
        [
            sys.executable, str(REPO / "single_agent" / "run_parallel_classes.py"),
            "--dataset", dataset, "--algorithm", SA_ALGO, "--seed", str(seed),
            "--n_classes", str(n_cls), "--parallel_classes", str(n_cls),
            "--total_timesteps", str(per_class), "--n_envs", "1",
            "--device", device, "--output_dir", str(out) + "/",
            "--extra_args", "--skip_eda",
        ],
        log_file=ds_log(dataset, f"rlda_train_seed{seed}.log"),
    )


def train_mada(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> None:
    if not FORCE and latest_exp(mada_out(dataset, seed), MA_SUBDIR) is not None:
        log(f"{dataset} MADA seed {seed}: training artifacts present, skip train")
        return
    out = mada_out(dataset, seed)
    out.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable, "driver.py",
            "--dataset", dataset, "--algorithm", MA_ALGO, "--seed", str(seed),
            "--skip_eda", "--max_n_frames", str(cfg["ma_frames"]),
            "--device", device, "--output_dir", str(out) + "/",
        ],
        cwd=REPO / "BenchMARL",
        log_file=ds_log(dataset, f"mada_train_seed{seed}.log"),
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
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, dest)
        log(f"  {cls}: no val-selected best_model.zip; copied {final.name} -> {dest}")


def infer_rlda(dataset: str, seed: int, cfg: Dict[str, int]) -> Path:
    existing = rlda_rules(dataset, seed)
    if not FORCE and existing is not None:
        log(f"{dataset} RLDA seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(rlda_out(dataset, seed), SA_SUBDIR)
    if exp is None:
        raise SystemExit(f"No RLDA experiment dir for {dataset} seed {seed}")
    ensure_best_models(exp)
    run(
        [
            sys.executable, str(REPO / "single_agent" / "single_agent_inference.py"),
            "--experiment_dir", str(exp), "--dataset", dataset, "--seed", str(seed),
            "--n_instances_per_class", str(cfg["n_instances"]),
        ],
        log_file=ds_log(dataset, f"rlda_infer_seed{seed}.log"),
    )
    rules = rlda_rules(dataset, seed)
    if rules is None:
        raise SystemExit(f"RLDA inference produced no rules for {dataset} seed {seed}")
    return rules


def infer_mada(dataset: str, seed: int, cfg: Dict[str, int], device: str) -> Path:
    existing = mada_rules(dataset, seed)
    if not FORCE and existing is not None:
        log(f"{dataset} MADA seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(mada_out(dataset, seed), MA_SUBDIR)
    if exp is None:
        raise SystemExit(f"No MADA experiment dir for {dataset} seed {seed}")
    run(
        [
            sys.executable, "inference.py",
            "--experiment_dir", str(exp), "--dataset", dataset, "--seed", str(seed),
            "--n_instances_per_class", str(cfg["n_instances"]), "--device", device,
            "--exploration_mode", "mean",
        ],
        cwd=REPO / "BenchMARL",
        log_file=ds_log(dataset, f"mada_infer_seed{seed}.log"),
    )
    rules = mada_rules(dataset, seed)
    if rules is None:
        raise SystemExit(f"MADA inference produced no rules for {dataset} seed {seed}")
    return rules


def evaluate(dataset: str, method: str, rules: Path, seed: int) -> None:
    dest = result_path(dataset, method, seed)
    if dest.exists() and not FORCE:
        log(f"{dataset} {method} seed {seed}: result JSON exists, skip evaluate")
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
    exp = latest_exp(rlda_out(dataset, seed), SA_SUBDIR)
    if exp is None:
        raise SystemExit(f"No RLDA classifier for {dataset} seed {seed}")
    clf = exp / "classifier.pth"
    if not clf.exists():
        raise SystemExit(f"Missing classifier {clf}")
    needed = ["cart", "random_search", "sp_anchors", "greedy_anchors"]
    if all(result_path(dataset, m, seed).exists() for m in needed):
        log(f"{dataset} baselines seed {seed}: all result JSONs present, skip")
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


def tables() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, str(REPO / "paper" / "make_tables.py"),
         "--results_dir", str(RESULTS), "--out_dir", str(TABLES_DIR)],
    )
    run(
        [sys.executable, str(REPO / "paper" / "make_figures.py"),
         "--results_dir", str(RESULTS), "--out_dir", str(FIGURES_DIR)],
    )


def run_dataset(dataset: str, seed: int, device: str) -> None:
    cfg = DATASET_CONFIGS[dataset]
    log(
        f"=== {dataset} seed {seed}  [SAC/MASAC]  "
        f"RLDA {cfg['sa_timesteps']} steps / MADA {cfg['ma_frames']} frames / "
        f"{cfg['n_instances']} inst/class ==="
    )
    if dataset not in SKIP_RLDA:
        train_rlda(dataset, seed, cfg, device)
        rules = infer_rlda(dataset, seed, cfg)
        evaluate(dataset, "rlda", rules, seed)
        baselines(dataset, seed)
    else:
        log(f"{dataset} RLDA seed {seed}: skipped (--skip-rlda-datasets)")
    train_mada(dataset, seed, cfg, device)
    rules = infer_mada(dataset, seed, cfg, device)
    evaluate(dataset, "mada", rules, seed)
    tables()
    log(f"=== {dataset} seed {seed} done ===")


def main() -> None:
    p = argparse.ArgumentParser(description="Revision 1-seed SAC/MASAC sweep on paper datasets")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain/re-infer/re-eval even if artifacts exist.",
    )
    p.add_argument(
        "--skip-rlda-datasets",
        nargs="*",
        default=[],
        help="Skip RLDA train/infer/eval for these datasets.",
    )
    args = p.parse_args()
    global FORCE, SKIP_RLDA
    FORCE = bool(args.force)
    SKIP_RLDA = set(args.skip_rlda_datasets)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    log(f"Paper-dataset seed {args.seed} SAC/MASAC sweep {args.datasets} device={args.device} force={FORCE} skip_rlda={sorted(SKIP_RLDA)}")
    for dataset in args.datasets:
        try:
            run_dataset(dataset, args.seed, args.device)
        except Exception as exc:
            failed.append(dataset)
            log(f"FAILED {dataset}: {exc}")
            traceback.print_exc()
            continue
    tables()
    if failed:
        log(f"Done with failures: {failed}. Tables in {TABLES_DIR}")
        raise SystemExit(1)
    log(f"Done. Tables in {TABLES_DIR}, figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
