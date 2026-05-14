#!/usr/bin/env python3
"""
WyoDOT Complete Pipeline: Single-Agent vs Multi-Agent Comparison

Runs the full pipeline for WyoDOT datasets:
1. Training (single-agent and multi-agent)
2. Inference (single-agent and multi-agent)
3. Test extracted rules (single-agent and multi-agent)
4. Summarize and plot results for comparison

Usage:
    python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42
    python run_pipeline.py --dataset wyodot_testbed --algorithm maddpg --skip_training
"""

import os
import sys
import subprocess
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

# Add project root and BenchMARL to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BenchMARL"))
sys.path.insert(0, str(PROJECT_ROOT / "single_agent"))

# Monkey-patch TabularDatasetLoader BEFORE importing inference/test modules.
# Different scripts import via different paths:
#   - BenchMARL scripts: "from tabular_datasets import TabularDatasetLoader"
#   - single_agent scripts: "from BenchMARL.tabular_datasets import TabularDatasetLoader"
# We must patch both module entries in sys.modules so that any future import
# resolves to WyoDOTDatasetLoader regardless of the import path used.
from wyodot_dataset_loader import WyoDOTDatasetLoader
import tabular_datasets
tabular_datasets.TabularDatasetLoader = WyoDOTDatasetLoader

# Also register the patched module under "BenchMARL.tabular_datasets" in sys.modules
# so that `from BenchMARL.tabular_datasets import TabularDatasetLoader` gets the patch.
# We do this via sys.modules directly to avoid triggering BenchMARL/__init__.py
# which would import environment.py and other heavy dependencies.
sys.modules["BenchMARL.tabular_datasets"] = tabular_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Per-dataset training timesteps for single-agent
# Target ~60-90k steps per class (5 classes for both WyoDOT datasets)
DATASET_TIMESTEPS = {
    "wyodot_kvdw_labeled": 480_000,   # 39,858 rows, 5 classes → 96k/class
    "wyodot_testbed": 800_000,        # 691 rows, 5 classes → 30k/class
}
# DATASET_TIMESTEPS = {
#     "wyodot_kvdw_labeled": 30_000,   # 39,858 rows, 5 classes → 6k/class
#     "wyodot_testbed": 150_000,        # 691 rows, 5 classes → 30k/class
# }
_DEFAULT_TIMESTEPS = 320_000

RESULTS_GROUP_NAME = "wyodot_results"


def run_command(cmd: list, description: str, cwd: Optional[str] = None, capture_output: bool = False) -> Tuple[bool, Optional[str]]:
    """Run a command and return (success, output)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Running: {' '.join(cmd)}")

    if cwd is None:
        cwd = str(PROJECT_ROOT)

    try:
        process = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True
        )

        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            logger.debug(line)
            output_lines.append(line)

        return_code = process.wait()
        output = "\n".join(output_lines)

        if return_code != 0:
            logger.error(f"FAIL: {description} (return code {return_code})")
            return False, output

        logger.info(f"OK: {description}")
        return True, output
    except Exception as e:
        logger.error(f"FAIL: {description}: {e}")
        return False, None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_single_agent_training(
    dataset: str, algorithm: str, seed: int = 42, device: str = "cpu",
    output_dir: Optional[str] = None, force_retrain: bool = False,
    total_timesteps: Optional[int] = None, **kwargs
) -> Optional[str]:
    """Run single-agent training using wyodot driver."""
    if total_timesteps is None:
        total_timesteps = DATASET_TIMESTEPS.get(dataset, _DEFAULT_TIMESTEPS)

    if output_dir is None:
        base_dir = SCRIPT_DIR / "output" / f"single_agent_sb3_{dataset}_{algorithm}"
        output_dir = str(base_dir) + "/"
    elif not output_dir.endswith("/"):
        output_dir = output_dir + "/"

    # Check existing experiment
    if not force_retrain:
        training_dir = Path(output_dir.rstrip("/")) / "training"
        if training_dir.exists():
            exp_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            valid = [d for d in exp_dirs if (d / "final_model").exists() or (d / "best_model").exists()]
            if valid:
                exp = max(valid, key=lambda p: p.stat().st_mtime)
                logger.info(f"Found existing experiment: {exp} (use --force_retrain to retrain)")
                return str(exp)

    driver_script = SCRIPT_DIR / "driver_single_agent.py"
    cmd = [
        sys.executable, str(driver_script),
        "--dataset", dataset, "--algorithm", algorithm,
        "--seed", str(seed), "--device", device,
        "--total_timesteps", str(total_timesteps),
        "--output_dir", output_dir, "--skip_eda",
    ]

    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    success, _ = run_command(cmd, f"Single-Agent Training: {dataset} with {algorithm.upper()}")

    if success:
        training_dir = Path(output_dir.rstrip("/")) / "training"
        if training_dir.exists():
            exp_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            valid = [d for d in exp_dirs if (d / "final_model").exists() or (d / "best_model").exists() or (d / "classifier.pth").exists()]
            if valid:
                return str(max(valid, key=lambda p: p.stat().st_mtime))
            if exp_dirs:
                return str(max(exp_dirs, key=lambda p: p.stat().st_mtime))
        return output_dir.rstrip("/")

    return None


def run_multi_agent_training(
    dataset: str, algorithm: str, seed: int = 42, device: str = "cpu",
    output_dir: Optional[str] = None, force_retrain: bool = False,
    max_n_frames: Optional[int] = None, **kwargs
) -> Optional[str]:
    """Run multi-agent training using wyodot driver."""
    driver_script = SCRIPT_DIR / "driver_multi_agent.py"

    if output_dir is None:
        output_dir = str(SCRIPT_DIR / "output" / f"{dataset}_{algorithm}")

    # Check existing experiment
    if not force_retrain:
        output_path = Path(output_dir)
        if output_path.exists():
            # Check BenchMARL-style experiment dirs inside output
            exp_dirs = [
                d for d in output_path.rglob("individual_models")
                if d.is_dir()
            ]
            if exp_dirs:
                exp = max(exp_dirs, key=lambda p: p.stat().st_mtime).parent
                logger.info(f"Found existing experiment: {exp} (use --force_retrain to retrain)")
                return str(exp)

    cmd = [
        sys.executable, str(driver_script),
        "--dataset", dataset, "--algorithm", algorithm,
        "--seed", str(seed), "--device", device,
        "--output_dir", output_dir + "/", "--skip_eda",
    ]

    if max_n_frames is not None:
        cmd.extend(["--max_n_frames", str(max_n_frames)])

    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    success, output = run_command(
        cmd, f"Multi-Agent Training: {dataset} with {algorithm.upper()}",
        capture_output=True
    )

    if success:
        # Try to find experiment directory from output
        if output:
            for line in output.split('\n'):
                if "BenchMARL checkpoint location:" in line or "Experiment folder:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        path = parts[-1].strip()
                        if os.path.exists(path):
                            return path

        # Search in output directory
        output_path = Path(output_dir)
        if output_path.exists():
            exp_dirs = []
            for item in output_path.rglob("individual_models"):
                if item.is_dir():
                    exp_dirs.append(item.parent)
            if exp_dirs:
                return str(max(exp_dirs, key=lambda p: p.stat().st_mtime))

        # Search in BenchMARL directory (Hydra sometimes puts experiments there)
        benchmarl_dir = PROJECT_ROOT / "BenchMARL"
        if benchmarl_dir.exists():
            excluded = {'output', 'conf', 'data', 'docs', '__pycache__', '.git'}
            exp_dirs = []
            for item in benchmarl_dir.iterdir():
                if (item.is_dir() and item.name not in excluded and
                    not item.name.startswith('.') and
                    (item / "individual_models").exists() and
                    algorithm.lower() in item.name.lower()):
                    exp_dirs.append(item)
            if exp_dirs:
                return str(max(exp_dirs, key=lambda p: p.stat().st_mtime))

        logger.warning("Training completed but could not find experiment directory")
        return None

    return None


# ---------------------------------------------------------------------------
# Inference (in-process with monkey-patched TabularDatasetLoader)
# ---------------------------------------------------------------------------

def run_single_agent_inference(
    experiment_dir: str, dataset: str,
    max_features_in_rule: int = -1, steps_per_episode: int = 100,
    n_instances_per_class: int = 5, n_rollouts_per_instance: int = 5,
    device: str = "cpu", **kwargs
) -> Optional[str]:
    """Run single-agent inference in-process (uses monkey-patched loader)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Single-Agent Inference: {dataset}")
    logger.info(f"{'='*80}")

    try:
        from single_agent_inference import extract_rules_single_agent

        results = extract_rules_single_agent(
            experiment_dir=experiment_dir,
            dataset_name=dataset,
            max_features_in_rule=max_features_in_rule,
            steps_per_episode=steps_per_episode,
            n_instances_per_class=n_instances_per_class,
            n_rollouts_per_instance=n_rollouts_per_instance,
            device=device,
            eval_on_test_data=True,
            coverage_on_all_data=True,
            sample_from_full_dataset=True,
            filter_by_prediction=False,
            use_prediction_routing=kwargs.get("use_prediction_routing", True),
            use_weighted_average=False,
            filter_low_quality_rollouts=True,
            min_precision_threshold=None,
            min_coverage_threshold=0.01,
        )

        # Save results
        output_dir = os.path.join(experiment_dir, "inference")
        os.makedirs(output_dir, exist_ok=True)
        rules_file = os.path.join(output_dir, "extracted_rules_single_agent.json")

        def _convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.int_)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, dict): return {_convert(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_convert(i) for i in obj]
            return obj

        with open(rules_file, 'w') as f:
            json.dump(_convert(results), f, indent=2)

        logger.info(f"OK: Single-agent rules saved to {rules_file}")
        return rules_file

    except Exception as e:
        logger.error(f"FAIL: Single-agent inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_multi_agent_inference(
    experiment_dir: str, dataset: str,
    max_features_in_rule: int = -1, steps_per_episode: int = 100,
    n_instances_per_class: int = 5, device: str = "cpu", **kwargs
) -> Optional[str]:
    """Run multi-agent inference in-process (uses monkey-patched loader)."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Multi-Agent Inference: {dataset}")
    logger.info(f"{'='*80}")

    # inference.py loads conf/mlp.yaml via a relative path, so we must
    # chdir to BenchMARL/ while it runs.
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(PROJECT_ROOT / "BenchMARL"))

        from inference import extract_rules_from_policies

        results = extract_rules_from_policies(
            experiment_dir=experiment_dir,
            dataset_name=dataset,
            max_features_in_rule=max_features_in_rule,
            steps_per_episode=steps_per_episode,
            n_instances_per_class=n_instances_per_class,
            device=device,
            eval_on_test_data=True,
            coverage_on_all_data=True,
            filter_by_prediction=False,
        )

        # Find rules file
        inference_dir = Path(experiment_dir) / "inference"
        rules_file = inference_dir / "extracted_rules.json"
        if rules_file.exists():
            logger.info(f"OK: Multi-agent rules saved to {rules_file}")
            return str(rules_file)

        logger.warning(f"Rules file not found at {rules_file}")
        return None

    except Exception as e:
        logger.error(f"FAIL: Multi-agent inference: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        os.chdir(prev_cwd)


# ---------------------------------------------------------------------------
# Testing (in-process with monkey-patched loader)
# ---------------------------------------------------------------------------

def run_single_agent_test(rules_file: str, dataset: str, seed: int = 42, **kwargs) -> Optional[str]:
    """Run single-agent rule testing in-process. Returns path to saved test results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Single-Agent Test Rules: {dataset}")
    logger.info(f"{'='*80}")

    try:
        from test_extracted_rules_single import test_rules_from_json as test_sa

        results = test_sa(
            rules_file=rules_file,
            dataset_name=dataset,
            seed=seed,
            use_full_dataset=kwargs.get("use_full_dataset", True),
        )

        # Save test results so summarize_and_plot can use them without re-running
        def _convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.int_)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, dict): return {_convert(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_convert(i) for i in obj]
            return obj

        inference_dir = Path(rules_file).parent
        test_results_file = str(inference_dir / "test_results_single_agent.json")
        with open(test_results_file, 'w') as f:
            json.dump(_convert(results), f, indent=2)

        logger.info(f"OK: Single-agent test completed, results saved to {test_results_file}")
        return test_results_file
    except Exception as e:
        logger.error(f"FAIL: Single-agent test: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_multi_agent_test(rules_file: str, dataset: str, seed: int = 42, **kwargs) -> Optional[str]:
    """Run multi-agent rule testing in-process. Returns path to saved test results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Multi-Agent Test Rules: {dataset}")
    logger.info(f"{'='*80}")

    try:
        from test_extracted_rules import test_rules_from_json as test_ma

        results = test_ma(
            rules_file=rules_file,
            dataset_name=dataset,
            seed=seed,
            use_full_dataset=kwargs.get("use_full_dataset", True),
        )

        # Save test results so summarize_and_plot can use them without re-running
        def _convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.integer, np.int_)): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, dict): return {_convert(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)): return [_convert(i) for i in obj]
            return obj

        inference_dir = Path(rules_file).parent
        test_results_file = str(inference_dir / "test_results_multi_agent.json")
        with open(test_results_file, 'w') as f:
            json.dump(_convert(results), f, indent=2)

        logger.info(f"OK: Multi-agent test completed, results saved to {test_results_file}")
        return test_results_file
    except Exception as e:
        logger.error(f"FAIL: Multi-agent test: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Summarize & Plot (subprocess — these scripts don't use TabularDatasetLoader)
# ---------------------------------------------------------------------------

def run_summarize_and_plot(
    rules_file: str, dataset: str, output_dir: Optional[str] = None,
    test_results_file: Optional[str] = None, seed: int = 42, **kwargs
) -> bool:
    """Run summarize and plot for a rules file.

    NOTE: We never pass --run_tests to the subprocess because the summarize
    scripts import test_extracted_rules which imports TabularDatasetLoader —
    and our monkey-patch doesn't survive subprocess boundaries.  Instead, we
    run tests in-process earlier in the pipeline and pass the saved results
    via --test_results_file.
    """
    rules_path = Path(rules_file)
    is_single = "single_agent" in rules_path.name.lower()

    if is_single:
        script = PROJECT_ROOT / "single_agent" / "summarize_and_plot_rules_single.py"
        cwd = str(PROJECT_ROOT)
        script_rel = "single_agent/summarize_and_plot_rules_single.py"
    else:
        script = PROJECT_ROOT / "BenchMARL" / "summarize_and_plot_rules.py"
        cwd = str(PROJECT_ROOT / "BenchMARL")
        script_rel = "summarize_and_plot_rules.py"

    if not script.exists():
        logger.error(f"Script not found: {script}")
        return False

    cmd = [
        sys.executable, script_rel,
        "--rules_file", str(rules_path.resolve()),
        "--dataset", dataset,
        "--seed", str(seed),
    ]

    if output_dir:
        cmd.extend(["--output_dir", str(Path(output_dir).resolve())])

    # Pass pre-computed test results instead of --run_tests
    if test_results_file and Path(test_results_file).exists():
        cmd.extend(["--test_results_file", str(Path(test_results_file).resolve())])

    # Pass remaining kwargs (but filter out run_tests / use_full_dataset
    # which are only relevant when --run_tests is used)
    skip_keys = {"run_tests", "use_full_dataset"}
    for key, value in kwargs.items():
        if key in skip_keys:
            continue
        if value is not None:
            if isinstance(value, bool):
                if value: cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    label = "Single-Agent" if is_single else "Multi-Agent"
    success, _ = run_command(cmd, f"Summarize & Plot ({label}): {dataset}", cwd=cwd)
    return success


# ---------------------------------------------------------------------------
# Comparison Summary
# ---------------------------------------------------------------------------

def create_comparison_summary(
    single_agent_summary: Optional[str], multi_agent_summary: Optional[str],
    output_dir: str, dataset: str
) -> None:
    """Create comparison summary JSON combining both pipeline results."""
    logger.info(f"\n{'='*80}")
    logger.info("CREATING COMPARISON SUMMARY")
    logger.info(f"{'='*80}")

    summary = {"dataset": dataset, "single_agent": {}, "multi_agent": {}}

    for label, path in [("single_agent", single_agent_summary), ("multi_agent", multi_agent_summary)]:
        if path and Path(path).exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                summary[label] = data.get("summary", data)
                logger.info(f"  Loaded {label} summary")
            except Exception as e:
                logger.warning(f"  Could not load {label} summary: {e}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "comparison_summary.json"

    def _convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_convert(i) for i in obj]
        return obj

    with open(summary_file, 'w') as f:
        json.dump(_convert(summary), f, indent=2)

    logger.info(f"  Comparison summary saved to: {summary_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="WyoDOT Complete Pipeline: Single-Agent vs Multi-Agent Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --seed 42
  python run_pipeline.py --dataset wyodot_testbed --algorithm maddpg --skip_training
  python run_pipeline.py --dataset wyodot_kvdw_labeled --algorithm maddpg --skip_single_agent
        """
    )

    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(WyoDOTDatasetLoader.DATASETS.keys()),
                        help="WyoDOT dataset name")
    parser.add_argument("--algorithm", type=str, required=True,
                        help="Algorithm: maddpg, masac (multi-agent) or ddpg, sac (single-agent)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"], help="Device")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference")
    parser.add_argument("--skip_testing", action="store_true", help="Skip testing")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining")
    parser.add_argument("--skip_single_agent", action="store_true", help="Skip single-agent pipeline")
    parser.add_argument("--skip_multi_agent", action="store_true", help="Skip multi-agent pipeline")
    parser.add_argument("--max_features_in_rule", type=int, default=-1,
                        help="Max features in extracted rules (-1 for all)")
    parser.add_argument("--steps_per_episode", type=int, default=500,
                        help="Steps per episode for inference")
    parser.add_argument("--n_instances_per_class", type=int, default=5,
                        help="Instances per class for inference")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for comparison results")
    parser.add_argument("--single_agent_output_dir", type=str, default=None)
    parser.add_argument("--multi_agent_output_dir", type=str, default=None)
    parser.add_argument("--max_n_frames", type=int, default=None,
                        help="Override total training frames for multi-agent")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Override total timesteps for single-agent")

    args = parser.parse_args()

    # Map algorithms
    algo = args.algorithm.lower()
    if algo in ("maddpg", "masac"):
        multi_agent_algorithm = algo
        single_agent_algorithm = "ddpg" if algo == "maddpg" else "sac"
    elif algo in ("ddpg", "sac"):
        single_agent_algorithm = algo
        multi_agent_algorithm = "maddpg" if algo == "ddpg" else "masac"
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(SCRIPT_DIR / "comparison_results" / RESULTS_GROUP_NAME / f"{args.dataset}_{args.algorithm}_{dt}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Logging setup
    log_file = output_path / "pipeline_run.log"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logging.root.setLevel(logging.INFO)

    logger.info(f"\n{'='*80}")
    logger.info("WYODOT COMPLETE PIPELINE: SINGLE-AGENT vs MULTI-AGENT COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Single-Agent: {single_agent_algorithm.upper()}")
    logger.info(f"Multi-Agent:  {multi_agent_algorithm.upper()}")
    logger.info(f"Seed: {args.seed}  |  Device: {args.device}")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"Log: {log_file}")
    logger.info(f"{'='*80}\n")

    # Track paths
    sa_exp_dir = sa_rules = sa_summary = sa_test_results = None
    ma_exp_dir = ma_rules = ma_summary = ma_test_results = None

    # ---- SINGLE-AGENT PIPELINE ----
    if not args.skip_single_agent:
        logger.info(f"\n{'='*80}")
        logger.info("SINGLE-AGENT PIPELINE")
        logger.info(f"{'='*80}\n")

        # Training
        if not args.skip_training:
            sa_exp_dir = run_single_agent_training(
                dataset=args.dataset, algorithm=single_agent_algorithm,
                seed=args.seed, device=args.device,
                output_dir=args.single_agent_output_dir,
                force_retrain=args.force_retrain,
                total_timesteps=args.total_timesteps,
            )
        else:
            # Find existing
            if args.single_agent_output_dir:
                sa_exp_dir = args.single_agent_output_dir
            else:
                base = SCRIPT_DIR / "output" / f"single_agent_sb3_{args.dataset}_{single_agent_algorithm}"
                training_dir = base / "training"
                if training_dir.exists():
                    dirs = [d for d in training_dir.iterdir() if d.is_dir()]
                    if dirs:
                        sa_exp_dir = str(max(dirs, key=lambda p: p.stat().st_mtime))
                        logger.info(f"Found existing experiment: {sa_exp_dir}")

        # Inference
        if not args.skip_inference and sa_exp_dir:
            sa_rules = run_single_agent_inference(
                experiment_dir=sa_exp_dir, dataset=args.dataset,
                max_features_in_rule=args.max_features_in_rule,
                steps_per_episode=args.steps_per_episode,
                n_instances_per_class=args.n_instances_per_class,
                device=args.device,
            )
        elif args.skip_inference and sa_exp_dir:
            rf = Path(sa_exp_dir) / "inference" / "extracted_rules_single_agent.json"
            if rf.exists():
                sa_rules = str(rf)
                logger.info(f"Found existing rules: {sa_rules}")

        # Testing (in-process with monkey-patched loader)
        if not args.skip_testing and sa_rules:
            sa_test_results = run_single_agent_test(
                rules_file=sa_rules, dataset=args.dataset,
                seed=args.seed, use_full_dataset=True,
            )

        # Summarize & Plot (subprocess — pass pre-computed test results)
        if sa_rules:
            sa_out = output_path / "single_agent"
            run_summarize_and_plot(
                rules_file=sa_rules, dataset=args.dataset,
                output_dir=str(sa_out),
                test_results_file=sa_test_results, seed=args.seed,
            )
            sf = sa_out / "summary.json"
            if sf.exists():
                sa_summary = str(sf)

    # ---- MULTI-AGENT PIPELINE ----
    if not args.skip_multi_agent:
        logger.info(f"\n{'='*80}")
        logger.info("MULTI-AGENT PIPELINE")
        logger.info(f"{'='*80}\n")

        # Training
        if not args.skip_training:
            ma_exp_dir = run_multi_agent_training(
                dataset=args.dataset, algorithm=multi_agent_algorithm,
                seed=args.seed, device=args.device,
                output_dir=args.multi_agent_output_dir,
                force_retrain=args.force_retrain,
                max_n_frames=args.max_n_frames,
            )
        else:
            if args.multi_agent_output_dir:
                ma_exp_dir = args.multi_agent_output_dir
            else:
                # Search for existing experiments
                search_dirs = [
                    SCRIPT_DIR / "output" / f"{args.dataset}_{multi_agent_algorithm}",
                    PROJECT_ROOT / "BenchMARL",
                ]
                for search_dir in search_dirs:
                    if not search_dir.exists():
                        continue
                    for item in search_dir.rglob("individual_models"):
                        if item.is_dir() and multi_agent_algorithm in str(item):
                            ma_exp_dir = str(item.parent)
                            break
                    if ma_exp_dir:
                        break
                if ma_exp_dir:
                    logger.info(f"Found existing experiment: {ma_exp_dir}")
                else:
                    logger.warning("No existing multi-agent experiment found")

        # Inference
        if not args.skip_inference and ma_exp_dir:
            ma_rules = run_multi_agent_inference(
                experiment_dir=ma_exp_dir, dataset=args.dataset,
                max_features_in_rule=args.max_features_in_rule,
                steps_per_episode=args.steps_per_episode,
                n_instances_per_class=args.n_instances_per_class,
                device=args.device,
            )
        elif args.skip_inference and ma_exp_dir:
            rf = Path(ma_exp_dir) / "inference" / "extracted_rules.json"
            if rf.exists():
                ma_rules = str(rf)
                logger.info(f"Found existing rules: {ma_rules}")

        # Testing (in-process with monkey-patched loader)
        if not args.skip_testing and ma_rules:
            ma_test_results = run_multi_agent_test(
                rules_file=ma_rules, dataset=args.dataset,
                seed=args.seed, use_full_dataset=True,
            )

        # Summarize & Plot (subprocess — pass pre-computed test results)
        if ma_rules:
            ma_out = output_path / "multi_agent"
            run_summarize_and_plot(
                rules_file=ma_rules, dataset=args.dataset,
                output_dir=str(ma_out),
                test_results_file=ma_test_results, seed=args.seed,
            )
            sf = ma_out / "summary.json"
            if sf.exists():
                ma_summary = str(sf)

    # ---- COMPARISON SUMMARY ----
    if sa_summary or ma_summary:
        create_comparison_summary(
            single_agent_summary=sa_summary,
            multi_agent_summary=ma_summary,
            output_dir=str(output_path),
            dataset=args.dataset,
        )

    # ---- COMPARISON PLOTS ----
    if sa_summary or ma_summary:
        plot_script = PROJECT_ROOT / "plot_comparison.py"
        if plot_script.exists():
            cmd = [sys.executable, str(plot_script),
                   "--dataset", args.dataset, "--output_dir", str(output_path)]
            if sa_summary:
                cmd.extend(["--single_agent_summary", sa_summary])
            if ma_summary:
                cmd.extend(["--multi_agent_summary", ma_summary])
            run_command(cmd, "Generate comparison plots", cwd=str(PROJECT_ROOT))

    # ---- CLASSIFIER RULES ANALYSIS ----
    if sa_rules or ma_rules:
        analyze_script = PROJECT_ROOT / "analyze_classifier_rules.py"
        if analyze_script.exists():
            analysis_dir = output_path / "classifier_rules_analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            for label, rf in [("single_agent", sa_rules), ("multi_agent", ma_rules)]:
                if rf:
                    cmd = [sys.executable, str(analyze_script),
                           "--rules_file", rf, "--dataset", args.dataset,
                           "--output_dir", str(analysis_dir / label)]
                    run_command(cmd, f"Analyze {label} classifier rules", cwd=str(PROJECT_ROOT), capture_output=True)

    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"Log: {log_file}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
