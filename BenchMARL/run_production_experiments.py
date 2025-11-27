#!/usr/bin/env python3
"""
Production Experiment Runner

This script runs the complete pipeline (train, inference, test rules) for all
dataset and algorithm combinations in production mode.

Usage:
    python run_production_experiments.py [--datasets DATASET1 DATASET2 ...] [--algorithms ALG1 ALG2 ...] [--skip_training] [--skip_inference] [--skip_testing]

Examples:
    # Run all experiments (default)
    python run_production_experiments.py

    # Run specific datasets and algorithms
    python run_production_experiments.py --datasets breast_cancer wine --algorithms maddpg

    # Skip training (only run inference and testing)
    python run_production_experiments.py --skip_training
"""

import argparse
import subprocess
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default datasets and algorithms
DEFAULT_DATASETS = ["breast_cancer", "wine", "covtype", "housing"]
DEFAULT_ALGORITHMS = ["maddpg", "masac"]

# Base directory for BenchMARL
BENCHMARL_DIR = Path(__file__).parent.absolute()


def run_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    description: str = "Command",
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a shell command and log the output in real-time.
    
    Args:
        cmd: Command to run as a list of strings
        cwd: Working directory (default: BenchMARL directory)
        description: Description of the command for logging
        check: If True, raise exception on non-zero exit code
    
    Returns:
        CompletedProcess-like object with stdout, stderr, and returncode
    """
    if cwd is None:
        cwd = BENCHMARL_DIR
    
    logger.info(f"{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cwd}")
    logger.info(f"{'='*80}")
    
    try:
        # Use Popen to stream output in real-time
        # Combine stderr into stdout to avoid deadlocks and simplify reading
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Capture output while streaming
        output_lines = []
        
        # Stream output in real-time
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    output_lines.append(line)
                    # Log in real-time - use info level for normal output
                    logger.info(line)
        
        # Wait for process to complete
        returncode = process.wait()
        
        # Join captured output
        stdout = '\n'.join(output_lines)
        stderr = ""  # Already combined into stdout
        
        # Create a CompletedProcess-like object
        class CompletedProcess:
            def __init__(self, args, returncode, stdout, stderr):
                self.args = args
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        result = CompletedProcess(cmd, returncode, stdout, stderr)
        
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
        else:
            logger.error(f"✗ {description} failed with exit code {result.returncode}")
        
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, stdout, stderr)
        
        return result
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"✗ {description} failed with unexpected error: {e}")
        raise


def train_model(
    dataset: str,
    algorithm: str,
    seed: int = 42,
    device: str = "cuda"
) -> Optional[Path]:
    """
    Train a model using driver.py.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name (maddpg or masac)
        seed: Random seed
        device: Device to use (cuda or cpu)
    
    Returns:
        Path to experiment directory, or None if training failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING: {dataset} with {algorithm.upper()}")
    logger.info(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "driver.py",
        "--dataset", dataset,
        "--algorithm", algorithm,
        "--seed", str(seed),
        "--eval_on_test_data"  # Always use test data for evaluation
    ]
    
    try:
        result = run_command(cmd, description=f"Training {dataset} with {algorithm}", check=False)
        
        # Check if training actually succeeded
        if result.returncode != 0:
            logger.error(f"Training failed with exit code {result.returncode}")
            return None
        
        # Try to parse the checkpoint path from driver.py output
        experiment_dir = None
        
        # Look for "BenchMARL checkpoint location:" or "Experiment folder:" in output
        output_text = result.stdout + "\n" + result.stderr
        for line in output_text.split('\n'):
            if "BenchMARL checkpoint location:" in line or "Experiment folder:" in line:
                # Extract path from line
                parts = line.split(":")
                if len(parts) > 1:
                    potential_path = parts[-1].strip()
                    if os.path.exists(potential_path):
                        experiment_dir = Path(potential_path)
                        logger.info(f"✓ Found experiment directory from output: {experiment_dir}")
                        break
        
        # If not found in output, try to find it by searching common locations
        if experiment_dir is None:
            # BenchMARL typically saves experiments in various locations
            # Try common patterns:
            search_paths = [
                BENCHMARL_DIR / "output" / f"{dataset}_{algorithm}" / "training",
                BENCHMARL_DIR / "output" / f"{dataset}_{algorithm}",
                BENCHMARL_DIR / "output",
                BENCHMARL_DIR / "logs",
            ]
            
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                # Look for directories with checkpoint.pt or individual_models
                for item in search_path.rglob("*"):
                    if item.is_dir():
                        # Check if this looks like a BenchMARL experiment directory
                        has_checkpoint = (item / "checkpoint.pt").exists() or (item / "checkpoint").exists()
                        has_individual_models = (item / "individual_models").exists()
                        has_csv_logs = any(item.glob("*.csv"))
                        
                        if has_checkpoint or has_individual_models or has_csv_logs:
                            # Check if it's recent (created/modified in last hour)
                            import time
                            mtime = item.stat().st_mtime
                            if time.time() - mtime < 3600:  # Within last hour
                                experiment_dir = item
                                logger.info(f"✓ Found recent experiment directory: {experiment_dir}")
                                break
                
                if experiment_dir:
                    break
        
        if experiment_dir and experiment_dir.exists():
            return experiment_dir
        else:
            logger.warning(f"⚠ Could not automatically find experiment directory")
            logger.info("You may need to manually specify the experiment directory for inference")
            logger.info("Common locations to check:")
            for path in search_paths:
                if path.exists():
                    logger.info(f"  - {path}")
            return None
    
    except Exception as e:
        logger.error(f"✗ Training failed for {dataset} with {algorithm}: {e}")
        return None


def run_inference(
    experiment_dir: Path,
    dataset: str,
    device: str = "cpu",
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    n_instances_per_class: int = 20
) -> Optional[Path]:
    """
    Run inference to extract rules.
    
    Args:
        experiment_dir: Path to experiment directory
        dataset: Dataset name
        device: Device to use (cpu or cuda)
        max_features_in_rule: Maximum features in extracted rules
        steps_per_episode: Steps per episode
        n_instances_per_class: Instances per class for evaluation
    
    Returns:
        Path to extracted_rules.json file, or None if inference failed
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"INFERENCE: {dataset} from {experiment_dir}")
    logger.info(f"{'='*80}")
    
    # Inference uses test data by default (eval_on_test_data is the default)
    cmd = [
        sys.executable,
        "inference.py",
        "--experiment_dir", str(experiment_dir),
        "--dataset", dataset,
        "--max_features_in_rule", str(max_features_in_rule),
        "--steps_per_episode", str(steps_per_episode),
        "--n_instances_per_class", str(n_instances_per_class)
    ]
    
    try:
        result = run_command(cmd, description=f"Inference for {dataset}")
        
        # Find the extracted_rules.json file
        inference_dir = experiment_dir / "inference"
        rules_file = inference_dir / "extracted_rules.json"
        
        if rules_file.exists():
            logger.info(f"✓ Found extracted rules: {rules_file}")
            return rules_file
        else:
            logger.warning(f"⚠ Extracted rules file not found at {rules_file}")
            return None
    
    except Exception as e:
        logger.error(f"✗ Inference failed for {dataset}: {e}")
        return None


def test_extracted_rules(
    rules_file: Path,
    dataset: str,
    seed: int = 42,
    output_file: Optional[Path] = None
) -> bool:
    """
    Test extracted rules using test_extracted_rules.py.
    
    Args:
        rules_file: Path to extracted_rules.json
        dataset: Dataset name
        seed: Random seed
        output_file: Optional path to save test results
    
    Returns:
        True if testing succeeded, False otherwise
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING RULES: {dataset} from {rules_file}")
    logger.info(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "test_extracted_rules.py",
        "--rules_file", str(rules_file),
        "--dataset", dataset,
        "--seed", str(seed)
    ]
    
    if output_file:
        cmd.extend(["--output", str(output_file)])
    
    try:
        result = run_command(cmd, description=f"Testing rules for {dataset}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Rule testing failed for {dataset}: {e}")
        return False


def run_experiment(
    dataset: str,
    algorithm: str,
    seed: int = 42,
    device: str = "cuda",
    skip_training: bool = False,
    skip_inference: bool = False,
    skip_testing: bool = False,
    experiment_dir: Optional[Path] = None,
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    n_instances_per_class: int = 20
) -> Dict:
    """
    Run complete experiment pipeline for a dataset-algorithm combination.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name
        seed: Random seed
        device: Device for training (inference uses cpu by default)
        skip_training: Skip training step
        skip_inference: Skip inference step
        skip_testing: Skip testing step
        experiment_dir: Pre-existing experiment directory (if skipping training)
        max_features_in_rule: Maximum features in extracted rules
        steps_per_episode: Steps per episode for inference
        n_instances_per_class: Instances per class for inference
    
    Returns:
        Dictionary with experiment results
    """
    results = {
        "dataset": dataset,
        "algorithm": algorithm,
        "seed": seed,
        "device": device,
        "status": "pending",
        "experiment_dir": None,
        "rules_file": None,
        "test_results_file": None,
        "errors": []
    }
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# EXPERIMENT: {dataset} with {algorithm.upper()}")
    logger.info(f"{'#'*80}")
    
    # Step 1: Training
    if not skip_training:
        try:
            exp_dir = train_model(dataset, algorithm, seed, device)
            if exp_dir:
                results["experiment_dir"] = str(exp_dir)
                experiment_dir = exp_dir
            else:
                results["status"] = "failed"
                results["errors"].append("Training completed but experiment directory not found")
                return results
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Training failed: {str(e)}")
            logger.error(f"Experiment failed at training stage: {e}")
            return results
    else:
        if experiment_dir is None:
            results["status"] = "failed"
            results["errors"].append("skip_training=True but no experiment_dir provided")
            return results
        results["experiment_dir"] = str(experiment_dir)
        logger.info(f"Skipping training, using existing experiment directory: {experiment_dir}")
    
    # Step 2: Inference
    if not skip_inference:
        try:
            rules_file = run_inference(
                experiment_dir,
                dataset,
                device="cpu",  # Inference typically uses CPU
                max_features_in_rule=max_features_in_rule,
                steps_per_episode=steps_per_episode,
                n_instances_per_class=n_instances_per_class
            )
            if rules_file:
                results["rules_file"] = str(rules_file)
            else:
                results["status"] = "failed"
                results["errors"].append("Inference completed but rules file not found")
                return results
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Inference failed: {str(e)}")
            logger.error(f"Experiment failed at inference stage: {e}")
            return results
    else:
        # Try to find rules file automatically
        inference_dir = experiment_dir / "inference"
        rules_file = inference_dir / "extracted_rules.json"
        if rules_file.exists():
            results["rules_file"] = str(rules_file)
            logger.info(f"Skipping inference, using existing rules file: {rules_file}")
        else:
            results["status"] = "failed"
            results["errors"].append("skip_inference=True but no rules file found")
            return results
    
    # Step 3: Test rules
    if not skip_testing:
        try:
            test_output_file = experiment_dir / "inference" / "test_results.json"
            success = test_extracted_rules(
                Path(results["rules_file"]),
                dataset,
                seed,
                test_output_file
            )
            if success:
                results["test_results_file"] = str(test_output_file)
                results["status"] = "completed"
            else:
                results["status"] = "failed"
                results["errors"].append("Rule testing failed")
        except Exception as e:
            results["status"] = "failed"
            results["errors"].append(f"Rule testing failed: {str(e)}")
            logger.error(f"Experiment failed at testing stage: {e}")
            return results
    else:
        logger.info("Skipping rule testing")
        results["status"] = "completed" if results["rules_file"] else "partial"
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"# EXPERIMENT COMPLETE: {dataset} with {algorithm.upper()} - Status: {results['status']}")
    logger.info(f"{'#'*80}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run production experiments for all dataset-algorithm combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_production_experiments.py

  # Run specific datasets and algorithms
  python run_production_experiments.py --datasets breast_cancer wine --algorithms maddpg

  # Skip training (use existing models)
  python run_production_experiments.py --skip_training

  # Skip inference (use existing rules)
  python run_production_experiments.py --skip_inference

  # Only test rules (skip training and inference)
  python run_production_experiments.py --skip_training --skip_inference
        """
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=["breast_cancer", "wine", "covtype", "housing", "iris", "synthetic", "moons", "circles"],
        help=f"Datasets to run (default: {DEFAULT_DATASETS})"
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=DEFAULT_ALGORITHMS,
        choices=["maddpg", "masac"],
        help=f"Algorithms to run (default: {DEFAULT_ALGORITHMS})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for training (default: cuda)"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training step (use existing models)"
    )
    
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference step (use existing rules)"
    )
    
    parser.add_argument(
        "--skip_testing",
        action="store_true",
        help="Skip rule testing step"
    )
    
    parser.add_argument(
        "--max_features_in_rule",
        type=int,
        default=5,
        help="Maximum features in extracted rules (default: 5)"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=100,
        help="Steps per episode for inference (default: 100)"
    )
    
    parser.add_argument(
        "--n_instances_per_class",
        type=int,
        default=20,
        help="Instances per class for inference (default: 20)"
    )
    
    parser.add_argument(
        "--output_summary",
        type=str,
        default=None,
        help="Path to save experiment summary JSON (default: production_experiments_summary.json)"
    )
    
    args = parser.parse_args()
    
    # Change to BenchMARL directory
    os.chdir(BENCHMARL_DIR)
    
    logger.info(f"\n{'='*80}")
    logger.info("PRODUCTION EXPERIMENT RUNNER")
    logger.info(f"{'='*80}")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Algorithms: {args.algorithms}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Skip training: {args.skip_training}")
    logger.info(f"Skip inference: {args.skip_inference}")
    logger.info(f"Skip testing: {args.skip_testing}")
    logger.info(f"{'='*80}\n")
    
    # Run all experiments
    all_results = []
    total_experiments = len(args.datasets) * len(args.algorithms)
    completed = 0
    failed = 0
    
    for dataset in args.datasets:
        for algorithm in args.algorithms:
            try:
                result = run_experiment(
                    dataset=dataset,
                    algorithm=algorithm,
                    seed=args.seed,
                    device=args.device,
                    skip_training=args.skip_training,
                    skip_inference=args.skip_inference,
                    skip_testing=args.skip_testing,
                    max_features_in_rule=args.max_features_in_rule,
                    steps_per_episode=args.steps_per_episode,
                    n_instances_per_class=args.n_instances_per_class
                )
                all_results.append(result)
                
                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"✗ Unexpected error in experiment {dataset}/{algorithm}: {e}")
                all_results.append({
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "status": "failed",
                    "errors": [f"Unexpected error: {str(e)}"]
                })
                failed += 1
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed,
        "results": all_results
    }
    
    output_file = args.output_summary or (BENCHMARL_DIR / "production_experiments_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Summary saved to: {output_file}")
    logger.info(f"{'='*80}\n")
    
    # Print detailed results
    logger.info("\nDetailed Results:")
    for result in all_results:
        status_icon = "✓" if result["status"] == "completed" else "✗"
        logger.info(f"{status_icon} {result['dataset']}/{result['algorithm']}: {result['status']}")
        if result.get("errors"):
            for error in result["errors"]:
                logger.info(f"    Error: {error}")
    
    # Exit with non-zero code if any experiments failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

