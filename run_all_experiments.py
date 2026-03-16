#!/usr/bin/env python3
"""
Single launcher for all dataset experiments.

Runs run_comparison_pipeline.py for every dataset with per-dataset training budgets.
Edit the DATASET_CONFIGS table below to tune training length, episode length, or
instances per class for any individual dataset.

Usage:
    python run_all_experiments.py [--device cpu|cuda|mps] [--seed 42]
                                  [--force_retrain] [--skip_training]
                                  [--datasets iris wine ...]   # subset
                                  [--algorithm maddpg]

Training budget rationale
--------------------------
| Dataset                        | SA steps  | MA frames | Rationale              |
|--------------------------------|-----------|-----------|------------------------|
| iris                           | 90K       | 360K      | 3 classes, tiny        |
| breast_cancer                  | 90K       | 360K      | 2 classes, tiny        |
| wine                           | 120K      | 360K      | 3 classes, small       |
| circles                        | 90K       | 360K      | 2 classes, tiny        |
| housing                        | 240K      | 480K      | 4 classes, medium      |
| uci_adult                      | 360K      | 720K      | 2 classes, ~32K rows   |
| uci_credit                     | 360K      | 720K      | 2 classes, ~30K rows   |
| uci_default-credit-card-clients| 360K      | 720K      | 2 classes, ~30K rows   |
| covtype                        | 1_440K    | 1_440K    | 7 classes, 580K rows   |
| folktables_income_CA_2018      | 720K      | 1_080K    | 2 classes, ~40K rows   |
"""

import subprocess
import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-dataset training configuration
# ---------------------------------------------------------------------------
# sa_timesteps  : total SB3 training steps (divided across classes by trainer)
# ma_frames     : total BenchMARL training frames (passed as --max_n_frames)
# max_cycles    : episode length in steps (same for both pipelines)
# n_instances   : anchor instances evaluated per class during inference
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "iris": {
        "sa_timesteps": 90_000,
        "ma_frames":    360_000,
        "max_cycles":   500,
        "n_instances":  20,
    },
    "breast_cancer": {
        "sa_timesteps": 90_000,
        "ma_frames":    360_000,
        "max_cycles":   500,
        "n_instances":  20,
    },
    "wine": {
        "sa_timesteps": 120_000,
        "ma_frames":    360_000,
        "max_cycles":   500,
        "n_instances":  20,
    },
    "circles": {
        "sa_timesteps": 90_000,
        "ma_frames":    360_000,
        "max_cycles":   500,
        "n_instances":  20,
    },
    "housing": {
        "sa_timesteps": 240_000,
        "ma_frames":    480_000,
        "max_cycles":   500,
        "n_instances":  25,
    },
    "uci_adult": {
        "sa_timesteps": 360_000,
        "ma_frames":    720_000,
        "max_cycles":   500,
        "n_instances":  25,
    },
    "uci_credit": {
        "sa_timesteps": 360_000,
        "ma_frames":    720_000,
        "max_cycles":   500,
        "n_instances":  25,
    },
    "uci_default-credit-card-clients": {
        "sa_timesteps": 360_000,
        "ma_frames":    720_000,
        "max_cycles":   500,
        "n_instances":  25,
    },
    "covtype": {
        "sa_timesteps": 1_440_000,
        "ma_frames":    1_440_000,
        "max_cycles":   500,
        "n_instances":  20,
    },
    "folktables_income_CA_2018": {
        "sa_timesteps": 720_000,
        "ma_frames":    1_080_000,
        "max_cycles":   500,
        "n_instances":  25,
    },
}

# Default algorithm for multi-agent (single-agent counterpart is auto-derived)
DEFAULT_ALGORITHM = "maddpg"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the full experiment suite across all datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIGS.keys()),
        choices=list(DATASET_CONFIGS.keys()),
        help="Subset of datasets to run (default: all)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=DEFAULT_ALGORITHM,
        help="Multi-agent algorithm to use (default: maddpg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use",
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if experiment already exists",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training (use existing models)",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline (static anchors) computation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    pipeline_script = project_root / "run_comparison_pipeline.py"
    if not pipeline_script.exists():
        raise FileNotFoundError(f"run_comparison_pipeline.py not found at {pipeline_script}")

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        cmd = [
            sys.executable,
            str(pipeline_script),
            "--dataset",         dataset,
            "--algorithm",       args.algorithm,
            "--seed",            str(args.seed),
            "--device",          args.device,
            "--total_timesteps", str(cfg["sa_timesteps"]),
            "--max_n_frames",    str(cfg["ma_frames"]),
            "--steps_per_episode", str(cfg["max_cycles"]),
            "--n_instances_per_class", str(cfg["n_instances"]),
        ]
        if args.force_retrain:
            cmd.append("--force_retrain")
        if args.skip_training:
            cmd.append("--skip_training")
        if args.skip_baseline:
            cmd.append("--skip_baseline")

        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}  |  SA steps: {cfg['sa_timesteps']:,}  |  "
              f"MA frames: {cfg['ma_frames']:,}  |  max_cycles: {cfg['max_cycles']}  |  "
              f"n_instances: {cfg['n_instances']}")
        print("Command:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, check=True, cwd=str(project_root))


if __name__ == "__main__":
    main()
