#!/usr/bin/env python3
"""
Complete Pipeline Script: Single-Agent vs Multi-Agent Comparison

This script runs the complete pipeline for both single-agent and multi-agent approaches:
1. Training (single-agent and multi-agent)
2. Inference (single-agent and multi-agent)
3. Test extracted rules (single-agent and multi-agent)
4. Summarize and plot results for comparison

Usage:
    python run_comparison_pipeline.py --dataset <dataset_name> --algorithm <algorithm_name> [options]

Example:
    python run_comparison_pipeline.py --dataset wine --algorithm maddpg --seed 42
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the project root directory (where this script is located)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR


def run_command(cmd: list, description: str, cwd: Optional[str] = None, capture_output: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Run a command and return True if successful, False otherwise.
    
    Args:
        cmd: Command to run as a list
        description: Description of what the command does
        cwd: Working directory for the command (defaults to project root)
        capture_output: If True, capture stdout/stderr and return it (while still showing in real-time)
    
    Returns:
        Tuple of (success: bool, output: Optional[str])
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    # Default to project root if cwd not specified
    if cwd is None:
        cwd = str(PROJECT_ROOT)
    
    try:
        if capture_output:
            # Use Popen to stream output in real-time while also capturing it
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            output_lines = []
            # Read and print output line by line in real-time
            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)  # Print immediately
                output_lines.append(line)
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                output = "\n".join(output_lines)
                logger.error(f"✗ {description} failed with return code {return_code}")
                logger.error(f"Output: {output}")
                return False, output
            
            output = "\n".join(output_lines)
            logger.info(f"✓ {description} completed successfully")
            return True, output
        else:
            # Show output in real-time without capturing
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            logger.info(f"✓ {description} completed successfully")
            return True, None
    except subprocess.CalledProcessError as e:
        if capture_output:
            output = (e.stdout or "") + "\n" + (e.stderr or "")
            logger.error(f"✗ {description} failed with return code {e.returncode}")
            logger.error(f"Output: {output}")
            return False, output
        else:
            logger.error(f"✗ {description} failed with return code {e.returncode}")
            return False, None
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {e}")
        return False, None


def check_existing_experiment(
    output_dir: Path,
    experiment_type: str = "single_agent",
    dataset: Optional[str] = None,
    algorithm: Optional[str] = None
) -> Optional[Path]:
    """
    Check if a complete experiment directory already exists.
    
    Args:
        output_dir: Output directory to search in
        experiment_type: "single_agent" or "multi_agent"
        dataset: Dataset name (optional, for filtering)
        algorithm: Algorithm name (optional, for filtering)
    
    Returns:
        Path to existing experiment directory if found, None otherwise
    """
    if experiment_type == "single_agent":
        # Single-agent: check in training/ subdirectory
        training_dir = output_dir / "training"
        if training_dir.exists():
            experiment_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                # Filter by dataset/algorithm if provided
                if dataset and algorithm:
                    # Check if directory name contains dataset and algorithm
                    filtered = [
                        d for d in experiment_dirs
                        if dataset in d.name.lower() and algorithm in d.name.lower()
                    ]
                    if filtered:
                        experiment_dirs = filtered
                
                # Check if most recent has models
                most_recent = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
                # Check for model files (final_model or best_model directories)
                if (most_recent / "final_model").exists() or (most_recent / "best_model").exists():
                    return most_recent
    else:
        # Multi-agent: check in BenchMARL directory for experiment folders
        benchmarl_dir = PROJECT_ROOT / "BenchMARL"
        excluded_dirs = {'output', 'conf', 'docs', '__pycache__', '.git'}
        
        experiment_dirs = []
        for item in benchmarl_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                item.name not in excluded_dirs):
                # Check if it has individual_models (complete training)
                if (item / "individual_models").exists():
                    # Filter by algorithm if provided (experiment names contain algorithm)
                    if algorithm and algorithm.lower() not in item.name.lower():
                        continue
                    experiment_dirs.append(item)
        
        if experiment_dirs:
            # Return most recent complete experiment
            return max(experiment_dirs, key=lambda p: p.stat().st_mtime)
    
    return None


def run_single_agent_training(
    dataset: str,
    algorithm: str,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    force_retrain: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Run single-agent training.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name (ddpg or sac)
        seed: Random seed
        device: Device to use
        output_dir: Output directory (optional)
        force_retrain: If True, retrain even if experiment exists
        **kwargs: Additional arguments to pass to driver
    
    Returns:
        Path to experiment directory if successful, None otherwise
    """
    if output_dir is None:
        # The driver appends "training/" to the output_dir, so we need to ensure it has a trailing slash
        # This creates: single_agent_sb3_{dataset}_{algorithm}/training/ instead of concatenating
        base_output_dir = PROJECT_ROOT / "single_agent" / "output" / f"single_agent_sb3_{dataset}_{algorithm}"
        # Ensure trailing slash so driver creates training/ subdirectory
        output_dir = str(base_output_dir) + "/"
    else:
        # Resolve relative paths relative to project root
        output_dir = str(Path(output_dir).resolve())
        # Ensure trailing slash so driver creates training/ subdirectory
        if not output_dir.endswith("/"):
            output_dir = output_dir + "/"
    
    # Check if experiment already exists
    if not force_retrain:
        existing_exp = check_existing_experiment(Path(output_dir), "single_agent", dataset, algorithm)
        if existing_exp:
            logger.info(f"✓ Found existing complete experiment directory: {existing_exp}")
            logger.info(f"  Skipping training. Use --force_retrain to retrain anyway.")
            return str(existing_exp)
    
    # Resolve paths relative to project root
    driver_script = PROJECT_ROOT / "single_agent" / "driver.py"
    if not driver_script.exists():
        logger.error(f"✗ Single-agent driver script not found: {driver_script}")
        return None
    
    cmd = [
        sys.executable,
        str(driver_script),
        "--dataset", dataset,
        "--algorithm", algorithm,
        "--seed", str(seed),
        "--device", device,
        "--output_dir", output_dir,
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    success, _ = run_command(cmd, f"Single-Agent Training: {dataset} with {algorithm.upper()}")
    
    if success:
        # Find the experiment directory (checkpoint path)
        # The checkpoint can be in different locations:
        # 1. output_dir/training/<experiment_name>/ (standard structure with trailing slash)
        # 2. output_dir/<experiment_name>/ (direct structure)
        
        # Remove trailing slash if present for Path operations
        output_dir_clean = output_dir.rstrip("/")
        output_path = Path(output_dir_clean)
        
        # First, try the standard structure: output_dir/training/<experiment_name>/
        training_dir = output_path / "training"
        if training_dir.exists():
            experiment_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                # Filter for experiment directories (should contain final_model or best_model)
                valid_dirs = [
                    d for d in experiment_dirs
                    if (d / "final_model").exists() or (d / "best_model").exists() or (d / "classifier.pth").exists()
                ]
                if valid_dirs:
                    experiment_dir = max(valid_dirs, key=lambda p: p.stat().st_mtime)
                    logger.info(f"✓ Found experiment directory: {experiment_dir}")
                    return str(experiment_dir)
                # If no valid dirs, use most recent anyway
                experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Found experiment directory: {experiment_dir}")
                return str(experiment_dir)
        
        # Second, try direct structure: output_dir/<experiment_name>/
        # Look for directories that look like experiment directories
        if output_path.exists():
            experiment_dirs = [
                d for d in output_path.iterdir()
                if d.is_dir() and not d.name.startswith('.') and
                ((d / "final_model").exists() or (d / "best_model").exists() or (d / "classifier.pth").exists() or
                 d.name.startswith(f"{algorithm}_single_agent_sb3_"))
            ]
            if experiment_dirs:
                experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Found experiment directory: {experiment_dir}")
                return str(experiment_dir)
        
        # Fallback: return training directory if it exists, otherwise output_dir
        if training_dir.exists():
            logger.warning(f"⚠ Could not find experiment directory, using training directory: {training_dir}")
            return str(training_dir)
        else:
            logger.warning(f"⚠ Could not find experiment directory, using output directory: {output_path}")
            return str(output_path)
    
    return None


def run_single_agent_inference(
    experiment_dir: str,
    dataset: str,
    max_features_in_rule: int = -1,
    steps_per_episode: int = 500,
    n_instances_per_class: int = 20,
    device: str = "cpu",
    **kwargs
) -> Optional[str]:
    """
    Run single-agent inference.
    
    Args:
        experiment_dir: Path to experiment directory
        dataset: Dataset name
        max_features_in_rule: Maximum features in rules
        steps_per_episode: Steps per episode
        n_instances_per_class: Instances per class
        device: Device to use
        **kwargs: Additional arguments
    
    Returns:
        Path to extracted_rules_single_agent.json if successful, None otherwise
    """
    # Resolve paths relative to project root
    inference_script = PROJECT_ROOT / "single_agent" / "single_agent_inference.py"
    if not inference_script.exists():
        logger.error(f"✗ Single-agent inference script not found: {inference_script}")
        return None
    
    cmd = [
        sys.executable,
        str(inference_script),
        "--experiment_dir", experiment_dir,
        "--dataset", dataset,
        "--max_features_in_rule", str(max_features_in_rule),
        "--steps_per_episode", str(steps_per_episode),
        "--n_instances_per_class", str(n_instances_per_class),
        "--device", device,
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    success, _ = run_command(cmd, f"Single-Agent Inference: {dataset}")
    
    if success:
        # Find the extracted rules file
        inference_dir = Path(experiment_dir) / "inference"
        rules_file = inference_dir / "extracted_rules_single_agent.json"
        
        if rules_file.exists():
            logger.info(f"✓ Found extracted rules: {rules_file}")
            return str(rules_file)
        else:
            logger.warning(f"⚠ Extracted rules file not found at {rules_file}")
            return None
    
    return None


def run_single_agent_test(
    rules_file: str,
    dataset: str,
    seed: int = 42,
    **kwargs
) -> bool:
    """
    Run single-agent test extracted rules.
    
    Args:
        rules_file: Path to extracted_rules_single_agent.json
        dataset: Dataset name
        seed: Random seed
        **kwargs: Additional arguments
    
    Returns:
        True if successful, False otherwise
    """
    # Resolve paths relative to project root
    test_script = PROJECT_ROOT / "single_agent" / "test_extracted_rules_single.py"
    if not test_script.exists():
        logger.error(f"✗ Single-agent test script not found: {test_script}")
        return False
    
    cmd = [
        sys.executable,
        str(test_script),
        "--rules_file", rules_file,
        "--dataset", dataset,
        "--seed", str(seed),
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    success, _ = run_command(cmd, f"Single-Agent Test Rules: {dataset}")
    return success


def run_multi_agent_training(
    dataset: str,
    algorithm: str,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    force_retrain: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Run multi-agent training.
    
    Args:
        dataset: Dataset name
        algorithm: Algorithm name (maddpg or masac)
        seed: Random seed
        device: Device to use
        output_dir: Output directory (optional)
        force_retrain: If True, retrain even if experiment exists
        **kwargs: Additional arguments
    
    Returns:
        Path to experiment directory if successful, None otherwise
    """
    # Resolve paths relative to project root
    driver_script = PROJECT_ROOT / "BenchMARL" / "driver.py"
    if not driver_script.exists():
        logger.error(f"✗ Multi-agent driver script not found: {driver_script}")
        return None
    
    # Check if experiment already exists (in BenchMARL directory)
    if not force_retrain:
        benchmarl_dir = PROJECT_ROOT / "BenchMARL"
        existing_exp = check_existing_experiment(benchmarl_dir, "multi_agent", dataset, algorithm)
        if existing_exp:
            logger.info(f"✓ Found existing complete experiment directory: {existing_exp}")
            logger.info(f"  Skipping training. Use --force_retrain to retrain anyway.")
            return str(existing_exp)
    
    if output_dir is None:
        # Default output_dir relative to BenchMARL directory
        output_dir = f"output/{dataset}_{algorithm}/"
    else:
        # If absolute path, make it relative to BenchMARL or keep as absolute
        output_path = Path(output_dir)
        if output_path.is_absolute():
            # Check if it's within BenchMARL directory
            benchmarl_dir = PROJECT_ROOT / "BenchMARL"
            try:
                output_dir = str(output_path.relative_to(benchmarl_dir))
            except ValueError:
                # Not within BenchMARL, use absolute path
                output_dir = str(output_path.resolve())
        else:
            # Already relative, use as-is (will be relative to BenchMARL when we run from there)
            pass
    
    cmd = [
        sys.executable,
        "driver.py",  # Use relative path since we'll run from BenchMARL directory
        "--dataset", dataset,
        "--algorithm", algorithm,
        "--seed", str(seed)
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Run from BenchMARL directory so relative config paths work
    benchmarl_dir = str(PROJECT_ROOT / "BenchMARL")
    success, output = run_command(cmd, f"Multi-Agent Training: {dataset} with {algorithm.upper()}", cwd=benchmarl_dir, capture_output=True)
    
    # Try to extract experiment directory from output
    experiment_dir_from_output = None
    if output:
        for line in output.split('\n'):
            if "BenchMARL checkpoint location:" in line or "Experiment folder:" in line or "BenchMARL experiment folder:" in line:
                # Extract path from line
                parts = line.split(":")
                if len(parts) > 1:
                    potential_path = parts[-1].strip()
                    if os.path.exists(potential_path):
                        experiment_dir_from_output = Path(potential_path)
                        logger.info(f"✓ Found experiment directory from training output: {experiment_dir_from_output}")
                        break
    
    if success:
        # First, use the experiment directory from training output if we found it
        if experiment_dir_from_output and experiment_dir_from_output.exists():
            logger.info(f"✓ Using experiment directory from training output: {experiment_dir_from_output}")
            return str(experiment_dir_from_output)
        
        # Fallback: Find the experiment directory (checkpoint path)
        # BenchMARL creates experiment directories directly in the BenchMARL directory
        # with names like: maddpg_anchor_mlp__e30d851d_25_11_29-19_41_35
        benchmarl_dir = PROJECT_ROOT / "BenchMARL"
        
        # Look for experiment directories directly in BenchMARL directory
        # These are created by BenchMARL and contain checkpoints/individual_models
        # BenchMARL creates folders like: maddpg_anchor_mlp__e30d851d_25_11_29-19_41_35
        experiment_dirs = []
        excluded_dirs = {'output', 'conf', 'docs', '__pycache__', '.git'}
        
        for item in benchmarl_dir.iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                item.name not in excluded_dirs):
                # Check if it looks like an experiment directory (contains checkpoints or individual_models)
                if (item / "checkpoints").exists() or (item / "individual_models").exists():
                    experiment_dirs.append(item)
        
        if experiment_dirs:
            # Sort by modification time, get most recent (should be the one we just created)
            experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
            logger.info(f"✓ Found experiment directory (most recent): {experiment_dir}")
            return str(experiment_dir)
        
        # Fallback: Check in output directory structure
        if Path(output_dir).is_absolute():
            output_path = Path(output_dir)
        else:
            output_path = benchmarl_dir / output_dir
        
        training_dir = output_path / "training"
        if training_dir.exists():
            experiment_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Found experiment directory in training folder: {experiment_dir}")
                return str(experiment_dir)
        
        # Also check if experiment directory is directly in output_dir
        experiment_dirs = [d for d in output_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if experiment_dirs:
            for exp_dir in experiment_dirs:
                if (exp_dir / "checkpoints").exists() or (exp_dir / "individual_models").exists():
                    logger.info(f"✓ Found experiment directory: {exp_dir}")
                    return str(exp_dir)
        
        logger.warning(f"⚠ Could not find experiment directory, using output directory: {output_path}")
        return str(output_path)
    
    return None


def run_multi_agent_inference(
    experiment_dir: str,
    dataset: str,
    max_features_in_rule: int = -1,
    steps_per_episode: int = 500,
    n_instances_per_class: int = 20,
    device: str = "cpu",
    **kwargs
) -> Optional[str]:
    """
    Run multi-agent inference.
    
    Args:
        experiment_dir: Path to experiment directory
        dataset: Dataset name
        max_features_in_rule: Maximum features in rules
        steps_per_episode: Steps per episode
        n_instances_per_class: Instances per class
        device: Device to use
        **kwargs: Additional arguments
    
    Returns:
        Path to extracted_rules.json if successful, None otherwise
    """
    # Resolve paths relative to project root
    inference_script = PROJECT_ROOT / "BenchMARL" / "inference.py"
    if not inference_script.exists():
        logger.error(f"✗ Multi-agent inference script not found: {inference_script}")
        return None
    
    # Make experiment_dir absolute or relative to BenchMARL directory
    experiment_dir_abs = str(Path(experiment_dir).resolve())
    
    cmd = [
        sys.executable,
        "inference.py",  # Use relative path since we'll run from BenchMARL directory
        "--experiment_dir", experiment_dir_abs,  # Use absolute path for experiment_dir
        "--dataset", dataset,
        "--max_features_in_rule", str(max_features_in_rule),
        "--steps_per_episode", str(steps_per_episode),
        "--n_instances_per_class", str(n_instances_per_class),
        "--device", device,
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Run from BenchMARL directory so relative config paths work
    benchmarl_dir = str(PROJECT_ROOT / "BenchMARL")
    success, _ = run_command(cmd, f"Multi-Agent Inference: {dataset}", cwd=benchmarl_dir)
    
    if success:
        # Find the extracted rules file
        inference_dir = Path(experiment_dir) / "inference"
        rules_file = inference_dir / "extracted_rules.json"
        
        if rules_file.exists():
            logger.info(f"✓ Found extracted rules: {rules_file}")
            return str(rules_file)
        else:
            logger.warning(f"⚠ Extracted rules file not found at {rules_file}")
            return None
    
    return None


def run_multi_agent_test(
    rules_file: str,
    dataset: str,
    seed: int = 42,
    **kwargs
) -> bool:
    """
    Run multi-agent test extracted rules.
    
    Args:
        rules_file: Path to extracted_rules.json
        dataset: Dataset name
        seed: Random seed
        **kwargs: Additional arguments
    
    Returns:
        True if successful, False otherwise
    """
    # Resolve paths relative to project root
    test_script = PROJECT_ROOT / "BenchMARL" / "test_extracted_rules.py"
    if not test_script.exists():
        logger.error(f"✗ Multi-agent test script not found: {test_script}")
        return False
    
    # Make rules_file absolute
    rules_file_abs = str(Path(rules_file).resolve())
    
    cmd = [
        sys.executable,
        "test_extracted_rules.py",  # Use relative path since we'll run from BenchMARL directory
        "--rules_file", rules_file_abs,  # Use absolute path for rules_file
        "--dataset", dataset,
        "--seed", str(seed),
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Run from BenchMARL directory so relative config paths work
    benchmarl_dir = str(PROJECT_ROOT / "BenchMARL")
    success, _ = run_command(cmd, f"Multi-Agent Test Rules: {dataset}", cwd=benchmarl_dir)
    return success


def run_baseline_establishment(
    dataset: str,
    seed: int = 42,
    n_instances_per_class: int = 20,
    methods: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    force_rerun: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Run baseline explainability methods.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        n_instances_per_class: Number of instances per class to explain
        methods: List of methods to run (None = all methods)
        output_dir: Output directory for results
        force_rerun: If True, rerun even if results exist
        **kwargs: Additional arguments
    
    Returns:
        Path to baseline_results JSON file if successful, None otherwise
    """
    # Resolve paths relative to project root
    baseline_script = PROJECT_ROOT / "baseline" / "establish_baseline.py"
    if not baseline_script.exists():
        logger.error(f"✗ Baseline script not found: {baseline_script}")
        return None
    
    # Default output directory
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output" / f"{dataset}_baseline")
    else:
        output_dir = str(Path(output_dir).resolve())
    
    # Check if baseline results already exist
    if not force_rerun:
        output_path = Path(output_dir)
        if output_path.exists():
            # Look for baseline_results JSON files
            baseline_files = list(output_path.glob("baseline_results_*.json"))
            if baseline_files:
                # Use most recent
                latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Found existing baseline results: {latest_file}")
                logger.info(f"  Skipping baseline establishment. Use --force_rerun to rerun anyway.")
                return str(latest_file)
    
    # Default methods if not specified
    if methods is None:
        methods = ["static_anchors"]  # Focus on static anchors for comparison
    
    cmd = [
        sys.executable,
        "-m", "baseline.establish_baseline",
        "--dataset", dataset,
        "--seed", str(seed),
        "--n_instances_per_class", str(n_instances_per_class),
        "--output_dir", output_dir,
        "--methods"
    ] + methods
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    success, _ = run_command(cmd, f"Baseline Establishment: {dataset}")
    
    if success:
        # Find the baseline results file
        output_path = Path(output_dir)
        baseline_files = list(output_path.glob("baseline_results_*.json"))
        if baseline_files:
            latest_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"✓ Found baseline results: {latest_file}")
            return str(latest_file)
        else:
            logger.warning(f"⚠ Baseline results file not found in {output_dir}")
            return None
    
    return None


def run_baseline_analysis(
    baseline_results_file: str,
    output_dir: Optional[str] = None
) -> bool:
    """
    Run baseline analysis and generate plots.
    
    Args:
        baseline_results_file: Path to baseline_results JSON file
        output_dir: Output directory for analysis plots (defaults to same as results file)
    
    Returns:
        True if successful, False otherwise
    """
    # Resolve paths relative to project root
    analysis_script = PROJECT_ROOT / "baseline" / "analyze_baseline.py"
    if not analysis_script.exists():
        logger.error(f"✗ Baseline analysis script not found: {analysis_script}")
        return False
    
    # Default output directory is same as results file directory
    if output_dir is None:
        output_dir = str(Path(baseline_results_file).parent)
    else:
        output_dir = str(Path(output_dir).resolve())
    
    cmd = [
        sys.executable,
        "-m", "baseline.analyze_baseline",
        baseline_results_file
    ]
    
    success, _ = run_command(cmd, f"Baseline Analysis: {Path(baseline_results_file).name}")
    return success


def run_summarize_and_plot(
    rules_file: str,
    dataset: str,
    output_dir: Optional[str] = None,
    run_tests: bool = False,
    seed: int = 42,
    **kwargs
) -> bool:
    """
    Run summarize and plot for a rules file.
    
    Args:
        rules_file: Path to extracted_rules.json or extracted_rules_single_agent.json
        dataset: Dataset name
        output_dir: Output directory for plots
        run_tests: Whether to run tests before plotting
        seed: Random seed
        **kwargs: Additional arguments
    
    Returns:
        True if successful, False otherwise
    """
    # Resolve paths relative to project root
    summarize_script = PROJECT_ROOT / "BenchMARL" / "summarize_and_plot_rules.py"
    if not summarize_script.exists():
        logger.error(f"✗ Summarize script not found: {summarize_script}")
        return False
    
    # Make paths absolute
    rules_file_abs = str(Path(rules_file).resolve())
    output_dir_abs = str(Path(output_dir).resolve()) if output_dir else None
    
    cmd = [
        sys.executable,
        "summarize_and_plot_rules.py",  # Use relative path since we'll run from BenchMARL directory
        "--rules_file", rules_file_abs,  # Use absolute path for rules_file
        "--dataset", dataset,
        "--seed", str(seed),
    ]
    
    if output_dir_abs:
        cmd.extend(["--output_dir", output_dir_abs])
    
    if run_tests:
        cmd.append("--run_tests")
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    # Run from BenchMARL directory so relative config paths work
    benchmarl_dir = str(PROJECT_ROOT / "BenchMARL")
    success, _ = run_command(cmd, f"Summarize and Plot: {dataset}", cwd=benchmarl_dir)
    return success


def create_comparison_summary(
    single_agent_summary_file: Optional[str],
    multi_agent_summary_file: Optional[str],
    output_dir: str,
    dataset: str
) -> None:
    """
    Create a comparison summary between single-agent and multi-agent results.
    
    Args:
        single_agent_summary_file: Path to single-agent summary.json
        multi_agent_summary_file: Path to multi-agent summary.json
        output_dir: Output directory for comparison
        dataset: Dataset name
    """
    logger.info(f"\n{'='*80}")
    logger.info("CREATING COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    
    comparison = {
        "dataset": dataset,
        "single_agent": {},
        "multi_agent": {},
        "comparison": {}
    }
    
    # Load single-agent summary if available
    if single_agent_summary_file and Path(single_agent_summary_file).exists():
        try:
            with open(single_agent_summary_file, 'r') as f:
                sa_data = json.load(f)
            comparison["single_agent"] = sa_data.get("summary", {})
            logger.info("✓ Loaded single-agent summary")
        except Exception as e:
            logger.warning(f"⚠ Could not load single-agent summary: {e}")
    
    # Load multi-agent summary if available
    if multi_agent_summary_file and Path(multi_agent_summary_file).exists():
        try:
            with open(multi_agent_summary_file, 'r') as f:
                ma_data = json.load(f)
            comparison["multi_agent"] = ma_data.get("summary", {})
            logger.info("✓ Loaded multi-agent summary")
        except Exception as e:
            logger.warning(f"⚠ Could not load multi-agent summary: {e}")
    
    # Create comparison metrics
    sa_stats = comparison["single_agent"].get("overall_stats", {})
    ma_stats = comparison["multi_agent"].get("overall_stats", {})
    
    # Handle both formats: single-agent uses "mean_precision", multi-agent uses "mean_instance_precision"
    if sa_stats and ma_stats:
        # Instance-level metrics
        sa_instance_precision = sa_stats.get("mean_instance_precision") or sa_stats.get("mean_precision", 0.0)
        sa_instance_coverage = sa_stats.get("mean_instance_coverage") or sa_stats.get("mean_coverage", 0.0)
        ma_instance_precision = ma_stats.get("mean_instance_precision", 0.0)
        ma_instance_coverage = ma_stats.get("mean_instance_coverage", 0.0)
        
        # Class-level metrics
        sa_class_precision = sa_stats.get("mean_class_precision", 0.0)
        sa_class_coverage = sa_stats.get("mean_class_coverage", 0.0)
        ma_class_precision = ma_stats.get("mean_class_precision", 0.0)
        ma_class_coverage = ma_stats.get("mean_class_coverage", 0.0)
        
        comparison["comparison"] = {
            # Instance-level comparison
            "instance_precision_diff": sa_instance_precision - ma_instance_precision,
            "instance_coverage_diff": sa_instance_coverage - ma_instance_coverage,
            # Class-level comparison
            "class_precision_diff": sa_class_precision - ma_class_precision,
            "class_coverage_diff": sa_class_coverage - ma_class_coverage,
            "rules_diff": sa_stats.get("total_unique_rules", 0) - ma_stats.get("total_unique_rules", 0),
            # Legacy fields for backward compatibility
            "precision_diff": sa_instance_precision - ma_instance_precision,
            "coverage_diff": sa_instance_coverage - ma_instance_coverage,
        }
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "comparison_summary.json"
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"✓ Comparison summary saved to: {comparison_file}")
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    if sa_stats:
        sa_instance_precision = sa_stats.get("mean_instance_precision") or sa_stats.get("mean_precision", 0.0)
        sa_instance_coverage = sa_stats.get("mean_instance_coverage") or sa_stats.get("mean_coverage", 0.0)
        sa_class_precision = sa_stats.get("mean_class_precision", 0.0)
        sa_class_coverage = sa_stats.get("mean_class_coverage", 0.0)
        logger.info(f"Single-Agent:")
        logger.info(f"  Instance-Level - Precision: {sa_instance_precision:.4f}, Coverage: {sa_instance_coverage:.4f}")
        logger.info(f"  Class-Level - Precision: {sa_class_precision:.4f}, Coverage: {sa_class_coverage:.4f}")
        logger.info(f"  Total Unique Rules: {sa_stats.get('total_unique_rules', 0)}")
    if ma_stats:
        ma_instance_precision = ma_stats.get("mean_instance_precision", 0.0)
        ma_instance_coverage = ma_stats.get("mean_instance_coverage", 0.0)
        ma_class_precision = ma_stats.get("mean_class_precision", 0.0)
        ma_class_coverage = ma_stats.get("mean_class_coverage", 0.0)
        logger.info(f"Multi-Agent:")
        logger.info(f"  Instance-Level - Precision: {ma_instance_precision:.4f}, Coverage: {ma_instance_coverage:.4f}")
        logger.info(f"  Class-Level - Precision: {ma_class_precision:.4f}, Coverage: {ma_class_coverage:.4f}")
        logger.info(f"  Total Unique Rules: {ma_stats.get('total_unique_rules', 0)}")
    if comparison["comparison"]:
        logger.info(f"Differences (Single - Multi):")
        logger.info(f"  Instance-Level - Precision: {comparison['comparison']['instance_precision_diff']:.4f}, Coverage: {comparison['comparison']['instance_coverage_diff']:.4f}")
        logger.info(f"  Class-Level - Precision: {comparison['comparison']['class_precision_diff']:.4f}, Coverage: {comparison['comparison']['class_coverage_diff']:.4f}")
        logger.info(f"  Rules: {comparison['comparison']['rules_diff']}")
    logger.info(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete pipeline: Single-Agent vs Multi-Agent Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline for wine dataset with MADDPG
  python run_comparison_pipeline.py --dataset wine --algorithm maddpg --seed 42
  
  # Skip training (use existing models)
  python run_comparison_pipeline.py --dataset wine --algorithm maddpg --skip_training
  
  # Skip inference (use existing rules)
  python run_comparison_pipeline.py --dataset wine --algorithm maddpg --skip_inference
  
  # Skip testing (use existing test results)
  python run_comparison_pipeline.py --dataset wine --algorithm maddpg --skip_testing
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing", "uci_adult", "uci_credit"],
        help="Dataset name"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Algorithm name. For single-agent: ddpg, sac. For multi-agent: maddpg, masac"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training (use existing models)"
    )
    
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference (use existing rules)"
    )
    
    parser.add_argument(
        "--skip_testing",
        action="store_true",
        help="Skip testing (use existing test results)"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if experiment directory already exists"
    )
    
    parser.add_argument(
        "--skip_single_agent",
        action="store_true",
        help="Skip single-agent pipeline (only run multi-agent)"
    )
    
    parser.add_argument(
        "--skip_multi_agent",
        action="store_true",
        help="Skip multi-agent pipeline (only run single-agent)"
    )
    
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip baseline establishment (use existing baseline results)"
    )
    
    parser.add_argument(
        "--force_rerun_baseline",
        action="store_true",
        help="Force rerun baseline even if results exist"
    )
    
    parser.add_argument(
        "--baseline_methods",
        type=str,
        nargs="+",
        default=None,
        choices=["lime", "static_anchors", "shap", "feature_importance"],
        help="Baseline methods to run (default: static_anchors only)"
    )
    
    parser.add_argument(
        "--max_features_in_rule",
        type=int,
        default=-1,
        help="Maximum features in extracted rules (use -1 for all features)"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=500,
        help="Steps per episode"
    )
    
    parser.add_argument(
        "--n_instances_per_class",
        type=int,
        default=20,
        help="Number of instances per class"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for comparison results (default: ./comparison_results/{dataset}_{algorithm}/)"
    )
    
    parser.add_argument(
        "--single_agent_output_dir",
        type=str,
        default=None,
        help="Output directory for single-agent results"
    )
    
    parser.add_argument(
        "--multi_agent_output_dir",
        type=str,
        default=None,
        help="Output directory for multi-agent results"
    )
    
    args = parser.parse_args()
    
    # Determine single-agent algorithm from multi-agent algorithm
    if args.algorithm.lower() == "maddpg":
        single_agent_algorithm = "ddpg"
    elif args.algorithm.lower() == "masac":
        single_agent_algorithm = "sac"
    else:
        # Assume algorithm is for single-agent, need to determine multi-agent equivalent
        if args.algorithm.lower() == "ddpg":
            single_agent_algorithm = "ddpg"
            multi_agent_algorithm = "maddpg"
        elif args.algorithm.lower() == "sac":
            single_agent_algorithm = "sac"
            multi_agent_algorithm = "masac"
        else:
            logger.error(f"Unknown algorithm: {args.algorithm}")
            logger.error("For single-agent, use: ddpg, sac")
            logger.error("For multi-agent, use: maddpg, masac")
            sys.exit(1)
    
    # If algorithm is multi-agent, use it directly
    if args.algorithm.lower() in ["maddpg", "masac"]:
        multi_agent_algorithm = args.algorithm.lower()
    else:
        # Already determined above
        pass
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE PIPELINE: SINGLE-AGENT vs MULTI-AGENT COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Single-Agent Algorithm: {single_agent_algorithm.upper()}")
    logger.info(f"Multi-Agent Algorithm: {multi_agent_algorithm.upper()}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"{'='*80}\n")
    
    # Set up output directories
    if args.output_dir is None:
        # Add datetime stamp to prevent overwriting previous results
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "comparison_results" / f"{args.dataset}_{args.algorithm}_{datetime_str}")
    else:
        # Resolve relative paths relative to project root
        args.output_dir = str(Path(args.output_dir).resolve())
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track paths for comparison
    single_agent_experiment_dir = None
    single_agent_rules_file = None
    single_agent_summary_file = None
    
    multi_agent_experiment_dir = None
    multi_agent_rules_file = None
    multi_agent_summary_file = None
    
    baseline_results_file = None
    
    # Run baseline pipeline first (it's independent and typically faster)
    if not args.skip_baseline:
        logger.info(f"\n{'='*80}")
        logger.info("BASELINE PIPELINE")
        logger.info(f"{'='*80}\n")
        
        baseline_results_file = run_baseline_establishment(
            dataset=args.dataset,
            seed=args.seed,
            n_instances_per_class=args.n_instances_per_class,
            methods=args.baseline_methods,
            output_dir=str(output_path / "baseline"),
            force_rerun=args.force_rerun_baseline
        )
        
        # Run baseline analysis to generate plots
        if baseline_results_file:
            run_baseline_analysis(
                baseline_results_file=baseline_results_file,
                output_dir=str(output_path / "baseline")
            )
    else:
        # Try to find existing baseline results
        baseline_dir = output_path / "baseline"
        if baseline_dir.exists():
            baseline_files = list(baseline_dir.glob("baseline_results_*.json"))
            if baseline_files:
                baseline_results_file = str(max(baseline_files, key=lambda p: p.stat().st_mtime))
                logger.info(f"Found existing baseline results: {baseline_results_file}")
    
    # Run single-agent pipeline
    if not args.skip_single_agent:
        logger.info(f"\n{'='*80}")
        logger.info("SINGLE-AGENT PIPELINE")
        logger.info(f"{'='*80}\n")
        
        # Training
        if not args.skip_training:
            single_agent_experiment_dir = run_single_agent_training(
                dataset=args.dataset,
                algorithm=single_agent_algorithm,
                seed=args.seed,
                device=args.device,
                output_dir=args.single_agent_output_dir,
                force_retrain=args.force_retrain
            )
        else:
            # Try to find existing experiment directory
            if args.single_agent_output_dir:
                single_agent_experiment_dir = args.single_agent_output_dir
            else:
                # The driver creates output_dir/training/ structure
                base_dir = PROJECT_ROOT / "single_agent" / "output" / f"single_agent_sb3_{args.dataset}_{single_agent_algorithm}"
                # Check the standard structure: base_dir/training/
                training_dir = base_dir / "training"
                if training_dir.exists():
                    experiment_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
                    if experiment_dirs:
                        single_agent_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                        logger.info(f"Found existing experiment directory: {single_agent_experiment_dir}")
                # Also check if directory with "training" concatenated exists (old structure)
                training_dir_old = Path(str(base_dir) + "training")
                if training_dir_old.exists() and not single_agent_experiment_dir:
                    experiment_dirs = [d for d in training_dir_old.iterdir() if d.is_dir()]
                    if experiment_dirs:
                        single_agent_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                        logger.info(f"Found existing experiment directory (old structure): {single_agent_experiment_dir}")
        
        # Inference
        if not args.skip_inference and single_agent_experiment_dir:
            single_agent_rules_file = run_single_agent_inference(
                experiment_dir=single_agent_experiment_dir,
                dataset=args.dataset,
                max_features_in_rule=args.max_features_in_rule,
                steps_per_episode=args.steps_per_episode,
                n_instances_per_class=args.n_instances_per_class,
                device=args.device
            )
        else:
            # Try to find existing rules file
            if single_agent_experiment_dir:
                rules_file = Path(single_agent_experiment_dir) / "inference" / "extracted_rules_single_agent.json"
                if rules_file.exists():
                    single_agent_rules_file = str(rules_file)
                    logger.info(f"Found existing rules file: {single_agent_rules_file}")
        
        # Testing
        if not args.skip_testing and single_agent_rules_file:
            run_single_agent_test(
                rules_file=single_agent_rules_file,
                dataset=args.dataset,
                seed=args.seed
            )
        
        # Summarize and plot
        if single_agent_rules_file:
            sa_output_dir = output_path / "single_agent"
            run_summarize_and_plot(
                rules_file=single_agent_rules_file,
                dataset=args.dataset,
                output_dir=str(sa_output_dir),
                run_tests=False,  # Tests already run
                seed=args.seed
            )
            # Find summary file
            summary_file = sa_output_dir / "summary.json"
            if summary_file.exists():
                single_agent_summary_file = str(summary_file)
    
    # Run multi-agent pipeline
    if not args.skip_multi_agent:
        logger.info(f"\n{'='*80}")
        logger.info("MULTI-AGENT PIPELINE")
        logger.info(f"{'='*80}\n")
        
        # Training
        if not args.skip_training:
            multi_agent_experiment_dir = run_multi_agent_training(
                dataset=args.dataset,
                algorithm=multi_agent_algorithm,
                seed=args.seed,
                output_dir=args.multi_agent_output_dir,
                force_retrain=args.force_retrain
            )
        else:
            # Try to find existing experiment directory
            if args.multi_agent_output_dir:
                multi_agent_experiment_dir = args.multi_agent_output_dir
            else:
                # Check BenchMARL output directory
                default_dir = PROJECT_ROOT / "BenchMARL" / "output" / f"{args.dataset}_{multi_agent_algorithm}" / "training"
                if default_dir.exists():
                    experiment_dirs = [d for d in default_dir.iterdir() if d.is_dir()]
                    if experiment_dirs:
                        multi_agent_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                        logger.info(f"Found existing experiment directory: {multi_agent_experiment_dir}")
                else:
                    # Also check if experiment directory is directly in output (BenchMARL structure)
                    output_dir = PROJECT_ROOT / "BenchMARL" / "output" / f"{args.dataset}_{multi_agent_algorithm}"
                    if output_dir.exists():
                        experiment_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        for exp_dir in experiment_dirs:
                            if (exp_dir / "checkpoints").exists() or (exp_dir / "individual_models").exists():
                                multi_agent_experiment_dir = str(exp_dir)
                                logger.info(f"Found existing experiment directory: {multi_agent_experiment_dir}")
                                break
        
        # Inference
        if not args.skip_inference and multi_agent_experiment_dir:
            multi_agent_rules_file = run_multi_agent_inference(
                experiment_dir=multi_agent_experiment_dir,
                dataset=args.dataset,
                max_features_in_rule=args.max_features_in_rule,
                steps_per_episode=args.steps_per_episode,
                n_instances_per_class=args.n_instances_per_class,
                device=args.device
            )
        else:
            # Try to find existing rules file
            if multi_agent_experiment_dir:
                rules_file = Path(multi_agent_experiment_dir) / "inference" / "extracted_rules.json"
                if rules_file.exists():
                    multi_agent_rules_file = str(rules_file)
                    logger.info(f"Found existing rules file: {multi_agent_rules_file}")
        
        # Testing
        if not args.skip_testing and multi_agent_rules_file:
            run_multi_agent_test(
                rules_file=multi_agent_rules_file,
                dataset=args.dataset,
                seed=args.seed
            )
        
        # Summarize and plot
        if multi_agent_rules_file:
            ma_output_dir = output_path / "multi_agent"
            run_summarize_and_plot(
                rules_file=multi_agent_rules_file,
                dataset=args.dataset,
                output_dir=str(ma_output_dir),
                run_tests=False,  # Tests already run
                seed=args.seed
            )
            # Find summary file
            summary_file = ma_output_dir / "summary.json"
            if summary_file.exists():
                multi_agent_summary_file = str(summary_file)
    
    # Create comparison summary
    if single_agent_summary_file or multi_agent_summary_file:
        create_comparison_summary(
            single_agent_summary_file=single_agent_summary_file,
            multi_agent_summary_file=multi_agent_summary_file,
            output_dir=str(output_path),
            dataset=args.dataset
        )
    
    # Generate comparison plots if summaries are available
    if single_agent_summary_file or multi_agent_summary_file:
        logger.info(f"\n{'='*80}")
        logger.info("GENERATING COMPARISON PLOTS")
        logger.info(f"{'='*80}")
        
        plot_comparison_script = PROJECT_ROOT / "plot_comparison.py"
        if plot_comparison_script.exists():
            cmd = [
                sys.executable,
                str(plot_comparison_script),
                "--dataset", args.dataset,
                "--output_dir", str(output_path)
            ]
            
            if single_agent_summary_file:
                cmd.extend(["--single_agent_summary", single_agent_summary_file])
            if multi_agent_summary_file:
                cmd.extend(["--multi_agent_summary", multi_agent_summary_file])
            if baseline_results_file:
                cmd.extend(["--baseline_results", baseline_results_file])
            
            success, output = run_command(
                cmd,
                description="Generate comparison plots",
                cwd=str(PROJECT_ROOT),
                capture_output=False
            )
            if success:
                logger.info("✓ Comparison plots generated successfully")
            else:
                logger.warning(f"⚠ Failed to generate comparison plots: {output}")
        else:
            logger.warning(f"⚠ plot_comparison.py not found at {plot_comparison_script}")
    
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

