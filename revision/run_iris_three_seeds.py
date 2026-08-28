"""Iris 3-seed sweep: RLDA (DDPG 90k) + MADA (MADDPG 360k) + baselines + tables.

Reuses existing artifacts unless --force is set (required after env changes).
Budgets and flags match the successful 2026-08-26 Iris run.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)
TAU_P, TAU_C, K = 0.90, 0.20, 5
RLDA_STEPS = 90_000
MADA_FRAMES = 360_000
N_INSTANCES = 20
RESULTS = REPO / "revision" / "results"
LOG_DIR = REPO / "revision" / "logs"
FORCE = False

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


def run(cmd, cwd=None, log_file: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as fh:
            proc = subprocess.run(cmd, cwd=cwd or REPO, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=cwd or REPO, env=ENV)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def rlda_out(seed: int) -> Path:
    return REPO / "output" / f"iris_rlda_ddpg_seed{seed}"


def mada_out(seed: int) -> Path:
    return REPO / "output" / f"iris_mada_maddpg_seed{seed}"


def latest_exp(output_dir: Path, must_contain: str | None = None) -> Path | None:
    training = output_dir / "training"
    if not training.is_dir():
        return None
    cands = [p for p in training.iterdir() if p.is_dir()]
    if must_contain:
        cands = [p for p in cands if must_contain in p.name]
    cands = [p for p in cands if (p / "classifier.pth").exists() or list(p.glob("*.zip")) or list(p.glob("checkpoint*"))]
    if not cands:
        cands = [p for p in training.iterdir() if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def rlda_rules(seed: int) -> Path | None:
    exp = latest_exp(rlda_out(seed), "ddpg_single_agent")
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules_single_agent.json"
    return p if p.exists() else None


def mada_rules(seed: int) -> Path | None:
    exp = latest_exp(mada_out(seed), "maddpg")
    if exp is None:
        return None
    p = exp / "inference" / "extracted_rules.json"
    return p if p.exists() else None


def train_rlda(seed: int) -> None:
    if not FORCE and latest_exp(rlda_out(seed), "ddpg_single_agent") is not None:
        log(f"RLDA seed {seed}: training artifacts present, skip train")
        return
    out = rlda_out(seed)
    out.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable, str(REPO / "single_agent" / "driver.py"),
            "--dataset", "iris", "--algorithm", "ddpg", "--seed", str(seed),
            "--skip_eda", "--total_timesteps", str(RLDA_STEPS), "--device", "cpu",
            "--output_dir", str(out) + "/",
        ],
        log_file=LOG_DIR / f"rlda_train_seed{seed}.log",
    )


def train_mada(seed: int) -> None:
    if not FORCE and latest_exp(mada_out(seed), "maddpg") is not None:
        log(f"MADA seed {seed}: training artifacts present, skip train")
        return
    out = mada_out(seed)
    out.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable, "driver.py",
            "--dataset", "iris", "--algorithm", "maddpg", "--seed", str(seed),
            "--skip_eda", "--max_n_frames", str(MADA_FRAMES), "--device", "cpu",
            "--output_dir", str(out) + "/",
        ],
        cwd=REPO / "BenchMARL",
        log_file=LOG_DIR / f"mada_train_seed{seed}.log",
    )


def infer_rlda(seed: int) -> Path:
    existing = rlda_rules(seed)
    if not FORCE and existing is not None:
        log(f"RLDA seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(rlda_out(seed), "ddpg_single_agent")
    if exp is None:
        raise SystemExit(f"No RLDA experiment dir for seed {seed}")
    run(
        [
            sys.executable, str(REPO / "single_agent" / "single_agent_inference.py"),
            "--experiment_dir", str(exp), "--dataset", "iris", "--seed", str(seed),
            "--n_instances_per_class", str(N_INSTANCES),
        ],
        log_file=LOG_DIR / f"rlda_infer_seed{seed}.log",
    )
    rules = rlda_rules(seed)
    if rules is None:
        raise SystemExit(f"RLDA inference produced no rules for seed {seed}")
    return rules


def infer_mada(seed: int) -> Path:
    existing = mada_rules(seed)
    if not FORCE and existing is not None:
        log(f"MADA seed {seed}: rules exist, skip inference")
        return existing
    exp = latest_exp(mada_out(seed), "maddpg")
    if exp is None:
        raise SystemExit(f"No MADA experiment dir for seed {seed}")
    run(
        [
            sys.executable, "inference.py",
            "--experiment_dir", str(exp), "--dataset", "iris", "--seed", str(seed),
            "--n_instances_per_class", str(N_INSTANCES), "--device", "cpu",
        ],
        cwd=REPO / "BenchMARL",
        log_file=LOG_DIR / f"mada_infer_seed{seed}.log",
    )
    rules = mada_rules(seed)
    if rules is None:
        raise SystemExit(f"MADA inference produced no rules for seed {seed}")
    return rules


def result_path(method: str, seed: int) -> Path:
    def _fmt(x: float) -> str:
        return f"{x:.2f}".replace(".", "p")
    return RESULTS / f"iris__{method}__seed{seed}__tp{_fmt(TAU_P)}__tc{_fmt(TAU_C)}.json"


def evaluate(method: str, rules: Path, seed: int) -> None:
    dest = result_path(method, seed)
    if dest.exists() and method in ("rlda", "mada"):
        # Always refresh rlda/mada so success_rate / compactness stay current.
        pass
    run(
        [
            sys.executable, "-m", "revision.evaluate",
            "--rules_file", str(rules), "--dataset", "iris", "--method", method,
            "--seed", str(seed), "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
            "--k", str(K), "--out_dir", str(RESULTS),
        ],
        log_file=LOG_DIR / f"eval_{method}_seed{seed}.log",
    )


def baselines(seed: int) -> None:
    exp = latest_exp(rlda_out(seed), "ddpg_single_agent")
    if exp is None:
        raise SystemExit(f"No RLDA classifier for baselines seed {seed}")
    clf = exp / "classifier.pth"
    if not clf.exists():
        raise SystemExit(f"Missing classifier {clf}")
    needed = ["cart", "random_search", "sp_anchors", "greedy_anchors"]
    if all(result_path(m, seed).exists() for m in needed):
        log(f"Baselines seed {seed}: all result JSONs present, skip")
        return
    run(
        [
            sys.executable, "-m", "revision.baselines",
            "--dataset", "iris", "--seed", str(seed), "--k", str(K),
            "--tau_p", str(TAU_P), "--tau_c", str(TAU_C),
            "--classifier_path", str(clf),
            "--methods", "cart", "random_search", "sp_anchors", "greedy_anchors",
            "--budget_per_class", "5", "--n_candidates", "256",
            "--out_dir", str(RESULTS),
        ],
        log_file=LOG_DIR / f"baselines_seed{seed}.log",
    )


def tables() -> None:
    run([sys.executable, str(REPO / "paper" / "make_tables.py"),
         "--results_dir", str(RESULTS), "--out_dir", str(REPO / "paper" / "tables")])
    run([sys.executable, str(REPO / "paper" / "make_figures.py"),
         "--results_dir", str(RESULTS), "--out_dir", str(REPO / "paper" / "figures")])


def train_rlda_missing() -> None:
    """Train missing RLDA seeds sequentially (small; avoids CPU contention with MADA)."""
    for seed in SEEDS:
        train_rlda(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and re-infer even if artifacts already exist (required after env fixes).",
    )
    parser.add_argument(
        "--mada-only",
        action="store_true",
        help="Skip RLDA; retrain/infer/eval MADA only (use after RLDA already completed).",
    )
    args = parser.parse_args()
    global FORCE
    FORCE = bool(args.force) or bool(args.mada_only)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    log(f"Iris 3-seed sweep {list(SEEDS)} force={FORCE} mada_only={args.mada_only}")
    if not args.mada_only:
        train_rlda_missing()
        for seed in SEEDS:
            rules = infer_rlda(seed)
            evaluate("rlda", rules, seed)
            baselines(seed)
    for seed in SEEDS:
        train_mada(seed)
        rules = infer_mada(seed)
        evaluate("mada", rules, seed)
    tables()
    log("Done. Tables in paper/tables/, figures in paper/figures/")


if __name__ == "__main__":
    main()
