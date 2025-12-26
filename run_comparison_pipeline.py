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
import numpy as np


def load_nashconv_metrics(experiment_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load NashConv metrics from training_history.json and evaluation_history.json.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with training and evaluation NashConv metrics
    """
    nashconv_data = {
        "training": None,
        "evaluation": None,
        "available": False
    }
    
    if not experiment_dir:
        return nashconv_data
    
    experiment_path = Path(experiment_dir)
    
    # Load training history
    training_history_path = experiment_path / "training_history.json"
    if training_history_path.exists():
        try:
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
            
            # Extract NashConv metrics from training history
            training_nashconv = []
            for entry in training_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("training/nashconv") or key.startswith("training/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    training_nashconv.append(nashconv_entry)
            
            if training_nashconv:
                nashconv_data["training"] = training_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load training NashConv metrics: {e}")
    
    # Load evaluation history
    evaluation_history_path = experiment_path / "evaluation_history.json"
    if evaluation_history_path.exists():
        try:
            with open(evaluation_history_path, 'r') as f:
                evaluation_history = json.load(f)
            
            # Extract NashConv metrics from evaluation history
            evaluation_nashconv = []
            for entry in evaluation_history:
                nashconv_entry = {}
                for key, value in entry.items():
                    if key.startswith("evaluation/nashconv") or key.startswith("evaluation/exploitability"):
                        nashconv_entry[key] = value
                if nashconv_entry:
                    nashconv_entry["step"] = entry.get("step")
                    nashconv_entry["total_frames"] = entry.get("total_frames")
                    evaluation_nashconv.append(nashconv_entry)
            
            if evaluation_nashconv:
                nashconv_data["evaluation"] = evaluation_nashconv
                nashconv_data["available"] = True
        except Exception as e:
            logger.debug(f"Could not load evaluation NashConv metrics: {e}")
    
    return nashconv_data

# Set up basic logging (will be reconfigured in main() with file handler)
# Don't use basicConfig here to avoid duplicate handlers - we'll configure in main()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get the project root directory (where this script is located)
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR


def build_dataset_choices() -> list:
    """
    Build dataset choices dynamically, including UCIML and Folktables datasets if available.
    
    Returns:
        List of available dataset names
    """
    dataset_choices = ["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"]
    
    # Add UCIML datasets if available
    try:
        from ucimlrepo import fetch_ucirepo
        dataset_choices.extend([
            "uci_adult", "uci_car", "uci_credit", "uci_nursery", 
            "uci_mushroom", "uci_tic-tac-toe", "uci_vote", "uci_zoo"
        ])
    except ImportError:
        pass
    
    # Add Folktables datasets if available
    try:
        from folktables import ACSDataSource
        # Add common Folktables combinations
        states = ["CA", "NY", "TX", "FL", "IL"]
        years = ["2018", "2019", "2020"]
        tasks = ["income", "coverage", "mobility", "employment", "travel"]
        for task in tasks:
            for state in states[:2]:  # Limit to first 2 states to avoid too many choices
                for year in years[:1]:  # Limit to first year
                    dataset_choices.append(f"folktables_{task}_{state}_{year}")
    except ImportError:
        pass
    
    return dataset_choices


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
                print(line, flush=True)  # Print immediately to console (subprocess output)
                logger.debug(line)  # Log to file only (don't duplicate to console)
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
            # Show output in real-time while also logging to file
            # Use Popen to capture output for logging while still showing in real-time
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time and log to file
            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)  # Print immediately to console (subprocess output)
                logger.debug(line)  # Log to file only (don't duplicate to console)
            
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"✗ {description} failed with return code {return_code}")
                return False, None
            
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
    total_timesteps: int = 240_000,
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
        total_timesteps: Total timesteps for training
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
        "--total_timesteps", str(total_timesteps),
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
    steps_per_episode: int = 100,
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
        
        # Fallback: Find the experiment directory by searching BenchMARL/ root
        # BenchMARL/Hydra creates experiment directories directly in the BenchMARL directory
        # with names like: maddpg_anchor_mlp__{hash}_{date}-{time}
        benchmarl_dir = PROJECT_ROOT / "BenchMARL"
        if benchmarl_dir.exists():
            # Find most recent experiment directory that matches algorithm and has checkpoints/individual_models
            experiment_dirs = []
            excluded_dirs = {'output', 'conf', 'data', 'docs', '__pycache__', 'old_results', '.git'}
            
            for item in benchmarl_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name not in excluded_dirs:
                    # Check if this looks like an experiment directory
                    if (item / "checkpoints").exists() or (item / "individual_models").exists() or (item / "config.pkl").exists():
                        # Check if it matches the algorithm pattern (e.g., "maddpg" in name)
                        if algorithm.lower() in item.name.lower():
                            # Check if it was created recently (within last hour for newly trained models)
                            import time
                            mtime = item.stat().st_mtime
                            if time.time() - mtime < 3600:  # Within last hour
                                experiment_dirs.append(item)
            
            if experiment_dirs:
                # Get most recent
                experiment_dir = max(experiment_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"✓ Found experiment directory (fallback search, recent): {experiment_dir}")
                return str(experiment_dir)
        
        logger.warning("⚠ Training completed but could not find experiment directory")
        return None
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
    steps_per_episode: int = 100,
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
    # Detect if this is a single-agent or multi-agent rules file
    rules_file_path = Path(rules_file)
    is_single_agent = "single_agent" in rules_file_path.name.lower()
    
    # Choose the correct summarize script
    if is_single_agent:
        summarize_script = PROJECT_ROOT / "single_agent" / "summarize_and_plot_rules_single.py"
        working_dir = str(PROJECT_ROOT)  # Run from project root for single-agent
        script_name = "single_agent/summarize_and_plot_rules_single.py"
    else:
        summarize_script = PROJECT_ROOT / "BenchMARL" / "summarize_and_plot_rules.py"
        working_dir = str(PROJECT_ROOT / "BenchMARL")  # Run from BenchMARL directory
        script_name = "summarize_and_plot_rules.py"
    
    if not summarize_script.exists():
        logger.error(f"✗ Summarize script not found: {summarize_script}")
        return False
    
    # Make paths absolute
    rules_file_abs = str(rules_file_path.resolve())
    output_dir_abs = str(Path(output_dir).resolve()) if output_dir else None
    
    cmd = [
        sys.executable,
        script_name,  # Use relative or absolute path depending on working directory
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
    
    # Run from appropriate directory
    success, _ = run_command(cmd, f"Summarize and Plot ({'Single-Agent' if is_single_agent else 'Multi-Agent'}): {dataset}", cwd=working_dir)
    return success


def create_consolidated_metrics_json(
    baseline_results_file: Optional[str],
    single_agent_summary_file: Optional[str],
    multi_agent_summary_file: Optional[str],
    output_dir: str,
    dataset: str
) -> None:
    """
    Create a consolidated metrics JSON file with all three methods (baseline, single-agent, multi-agent)
    including their wandb URLs for easy access and comparison.
    
    Args:
        baseline_results_file: Path to baseline results JSON file
        single_agent_summary_file: Path to single-agent summary.json
        multi_agent_summary_file: Path to multi-agent summary.json
        output_dir: Output directory for consolidated metrics JSON
        dataset: Dataset name
    """
    logger.info(f"\n{'='*80}")
    logger.info("CREATING CONSOLIDATED METRICS JSON")
    logger.info(f"{'='*80}")
    
    output_path = Path(output_dir)
    consolidated = {
        "dataset": dataset,
        "baseline": {},
        "single_agent": {},
        "multi_agent": {}
    }
    
    # Load baseline metrics
    if baseline_results_file and Path(baseline_results_file).exists():
        try:
            with open(baseline_results_file, 'r') as f:
                baseline_data = json.load(f)
            
            # Extract static_anchors metrics (baseline doesn't use wandb, so no URL)
            static_anchors = baseline_data.get("methods", {}).get("static_anchors", {}) or baseline_data.get("static_anchors", {})
            if static_anchors:
                baseline_stats = static_anchors.get("overall_stats", {})
                baseline_per_class = static_anchors.get("per_class_results", {})
                baseline_metadata = static_anchors.get("metadata", {})
                
                consolidated["baseline"] = {
                    "wandb_run_url": None,  # Baseline doesn't use wandb
                    "metrics": {
                        "instance_level": {
                            "mean_precision": baseline_stats.get("mean_precision", 0.0),
                            "mean_coverage": baseline_stats.get("mean_coverage", 0.0),
                            "std_precision": baseline_stats.get("std_precision", 0.0),
                            "std_coverage": baseline_stats.get("std_coverage", 0.0),
                        }
                    },
                    "timing": {
                        "total_inference_time_seconds": baseline_metadata.get("total_inference_time_seconds", 0.0),
                        "total_rollout_time_seconds": baseline_metadata.get("total_rollout_time_seconds", 0.0),
                    },
                    "per_class": {}
                }
                
                # Add per-class baseline metrics (instance-level only)
                for class_key, class_data in baseline_per_class.items():
                    cls = class_data.get("class", -1) if isinstance(class_data, dict) else int(class_key.split("_")[-1]) if "_" in class_key else int(class_key)
                    if isinstance(class_data, dict):
                        consolidated["baseline"]["per_class"][f"class_{cls}"] = {
                            "instance_precision": class_data.get("instance_precision", class_data.get("mean_precision", 0.0)),
                            "instance_coverage": class_data.get("instance_coverage", class_data.get("mean_coverage", 0.0)),
                            "timing": {
                                "avg_rollout_time_seconds": class_data.get("avg_rollout_time_seconds", 0.0),
                                "total_rollout_time_seconds": class_data.get("total_rollout_time_seconds", 0.0),
                                "class_total_time_seconds": class_data.get("class_total_time_seconds", 0.0),
                            }
                        }
                
                logger.info("✓ Loaded baseline metrics")
        except Exception as e:
            logger.warning(f"⚠ Could not load baseline metrics: {e}")
    
    # Load single-agent consolidated metrics (preferred) or summary
    if single_agent_summary_file:
        sa_summary_path = Path(single_agent_summary_file)
        sa_consolidated_file = sa_summary_path.parent / "consolidated_metrics.json"
        
        if sa_consolidated_file.exists():
            try:
                with open(sa_consolidated_file, 'r') as f:
                    sa_consolidated = json.load(f)
                consolidated["single_agent"] = {
                    "wandb_run_url": sa_consolidated.get("wandb_run_url"),
                    "metrics": sa_consolidated.get("metrics", {}),
                    "per_class": sa_consolidated.get("per_class", {}),
                    "timing": sa_consolidated.get("timing", {}),
                    "algorithm": sa_consolidated.get("algorithm"),
                    "model_type": sa_consolidated.get("model_type")
                }
                # If timing not in consolidated or empty, try to extract from inference file or aggregate from per_class
                timing_dict = consolidated["single_agent"]["timing"]
                if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                    # First try inference file
                    sa_inference_file = sa_summary_path.parent / "extracted_rules_single_agent.json"
                    if sa_inference_file.exists():
                        try:
                            with open(sa_inference_file, 'r') as f:
                                sa_inference_data = json.load(f)
                            sa_metadata = sa_inference_data.get("metadata", {})
                            if sa_metadata:
                                consolidated["single_agent"]["timing"] = {
                                    "total_inference_time_seconds": sa_metadata.get("total_inference_time_seconds", 0.0),
                                    "total_rollout_time_seconds": sa_metadata.get("total_rollout_time_seconds", 0.0),
                                }
                        except Exception:
                            pass
                    
                    # If still no timing, aggregate from per_class
                    timing_dict = consolidated["single_agent"]["timing"]
                    if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                        total_rollout = sum(
                            pc.get("timing", {}).get("total_rollout_time_seconds", 0.0)
                            for pc in consolidated["single_agent"]["per_class"].values()
                        )
                        if total_rollout > 0:
                            consolidated["single_agent"]["timing"] = {
                                "total_rollout_time_seconds": total_rollout,
                                "total_inference_time_seconds": 0.0,  # Not available without inference file
                            }
                logger.info("✓ Loaded single-agent consolidated metrics")
            except Exception as e:
                logger.warning(f"⚠ Could not load single-agent consolidated metrics: {e}")
        else:
            # Fallback to summary.json
            try:
                with open(single_agent_summary_file, 'r') as f:
                    sa_data = json.load(f)
                sa_summary = sa_data.get("summary", {})
                sa_stats = sa_summary.get("overall_stats", {})
                
                consolidated["single_agent"] = {
                    "wandb_run_url": None,  # Try to get from wandb_run_url.txt if available
                    "algorithm": sa_summary.get("algorithm"),
                    "model_type": sa_summary.get("model_type"),
                    "metrics": {
                        "instance_level": {
                            "mean_precision": sa_stats.get("mean_precision", sa_stats.get("mean_instance_precision", 0.0)),
                            "mean_coverage": sa_stats.get("mean_coverage", sa_stats.get("mean_instance_coverage", 0.0)),
                            "std_precision": sa_stats.get("std_precision", sa_stats.get("std_instance_precision", 0.0)),
                            "std_coverage": sa_stats.get("std_coverage", sa_stats.get("std_instance_coverage", 0.0)),
                        },
                        "class_union": {  # Class union metrics (union of class-based anchors only)
                            "mean_precision": sa_stats.get("mean_class_precision", 0.0),
                            "mean_coverage": sa_stats.get("mean_class_coverage", 0.0),
                            "std_precision": sa_stats.get("std_class_precision", 0.0),
                            "std_coverage": sa_stats.get("std_class_coverage", 0.0),
                        },
                        "class_based": {  # Class-based metrics (centroid-based rollouts)
                            "mean_precision": sa_stats.get("mean_class_based_precision", sa_stats.get("mean_class_level_precision", 0.0)),
                            "mean_coverage": sa_stats.get("mean_class_based_coverage", sa_stats.get("mean_class_level_coverage", 0.0)),
                            "std_precision": sa_stats.get("std_class_based_precision", 0.0),
                            "std_coverage": sa_stats.get("std_class_based_coverage", 0.0),
                        },
                        "global": {
                            "total_unique_rules": sa_stats.get("total_unique_rules", 0),
                            "mean_unique_rules_per_class": sa_stats.get("mean_unique_rules_per_class", 0.0),
                        }
                    },
                    "per_class": {},
                    "timing": {}
                }
                
                # Add per-class metrics
                per_class_summary = sa_summary.get("per_class_summary", {})
                # Filter out _class_based entries - only process main class entries
                per_class_filtered = {
                    k: v for k, v in per_class_summary.items() 
                    if not k.endswith('_class_based') and v.get('rollout_type') != 'class_based'
                }
                for class_key, class_data in per_class_filtered.items():
                    cls = class_data.get("class", -1)
                    # Extract all three metric types
                    instance_prec = class_data.get("instance_precision", 0.0)
                    instance_cov = class_data.get("instance_coverage", 0.0)
                    class_union_prec = class_data.get("class_union_precision", class_data.get("class_precision", 0.0))
                    class_union_cov = class_data.get("class_union_coverage", class_data.get("class_coverage", 0.0))
                    class_based_prec = class_data.get("class_level_precision", class_data.get("class_based_precision", 0.0))
                    class_based_cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", 0.0))
                    
                    consolidated["single_agent"]["per_class"][f"class_{cls}"] = {
                        "instance_precision": instance_prec,
                        "instance_coverage": instance_cov,
                        "class_union_precision": class_union_prec,
                        "class_union_coverage": class_union_cov,
                        # Legacy fields for backward compatibility
                        "class_precision": class_union_prec,
                        "class_coverage": class_union_cov,
                        # Class-based metrics (centroid-based rollouts)
                        "class_based_precision": class_based_prec,
                        "class_based_coverage": class_based_cov,
                        "n_unique_rules": class_data.get("n_unique_rules", 0),
                        "timing": {
                            "avg_rollout_time_seconds": class_data.get("avg_rollout_time_seconds", 0.0),
                            "total_rollout_time_seconds": class_data.get("total_rollout_time_seconds", 0.0),
                            "class_total_time_seconds": class_data.get("class_total_time_seconds", 0.0),
                        }
                    }
                
                # Try to extract timing from inference results file
                sa_inference_file = sa_summary_path.parent / "extracted_rules_single_agent.json"
                if sa_inference_file.exists():
                    try:
                        with open(sa_inference_file, 'r') as f:
                            sa_inference_data = json.load(f)
                        sa_metadata = sa_inference_data.get("metadata", {})
                        if sa_metadata:
                            consolidated["single_agent"]["timing"] = {
                                "total_inference_time_seconds": sa_metadata.get("total_inference_time_seconds", 0.0),
                                "total_rollout_time_seconds": sa_metadata.get("total_rollout_time_seconds", 0.0),
                            }
                    except Exception as e:
                        logger.debug(f"Could not extract timing from single-agent inference file: {e}")
                
                # If still no timing, aggregate from per_class timing data
                timing_dict = consolidated["single_agent"]["timing"]
                if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                    total_rollout = sum(
                        pc.get("timing", {}).get("total_rollout_time_seconds", 0.0)
                        for pc in consolidated["single_agent"]["per_class"].values()
                    )
                    if total_rollout > 0:
                        consolidated["single_agent"]["timing"] = {
                            "total_rollout_time_seconds": total_rollout,
                            "total_inference_time_seconds": 0.0,  # Not available without inference file
                        }
                
                # Try to get wandb URL from file
                wandb_url_file = sa_summary_path.parent.parent / "wandb_run_url.txt"
                if not wandb_url_file.exists():
                    # Also check in training directory
                    wandb_url_file = sa_summary_path.parent.parent.parent / "wandb_run_url.txt"
                if wandb_url_file.exists():
                    try:
                        with open(wandb_url_file, 'r') as f:
                            consolidated["single_agent"]["wandb_run_url"] = f.read().strip()
                    except Exception:
                        pass
                
                logger.info("✓ Loaded single-agent metrics from summary")
            except Exception as e:
                logger.warning(f"⚠ Could not load single-agent summary: {e}")
    
    # Load multi-agent consolidated metrics (preferred) or summary
    if multi_agent_summary_file:
        ma_summary_path = Path(multi_agent_summary_file)
        ma_consolidated_file = ma_summary_path.parent / "consolidated_metrics.json"
        
        if ma_consolidated_file.exists():
            try:
                with open(ma_consolidated_file, 'r') as f:
                    ma_consolidated = json.load(f)
                consolidated["multi_agent"] = {
                    "wandb_run_url": ma_consolidated.get("wandb_run_url"),
                    "metrics": ma_consolidated.get("metrics", {}),
                    "per_class": ma_consolidated.get("per_class", {}),
                    "timing": ma_consolidated.get("timing", {}),
                    "algorithm": ma_consolidated.get("algorithm"),
                    "model_type": ma_consolidated.get("model_type")
                }
                # If timing not in consolidated or empty, try to extract from inference file or aggregate from per_class
                timing_dict = consolidated["multi_agent"]["timing"]
                if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                    # First try inference file
                    ma_inference_file = ma_summary_path.parent / "extracted_rules.json"
                    if ma_inference_file.exists():
                        try:
                            with open(ma_inference_file, 'r') as f:
                                ma_inference_data = json.load(f)
                            ma_metadata = ma_inference_data.get("metadata", {})
                            if ma_metadata:
                                consolidated["multi_agent"]["timing"] = {
                                    "total_inference_time_seconds": ma_metadata.get("total_inference_time_seconds", 0.0),
                                    "total_rollout_time_seconds": ma_metadata.get("total_rollout_time_seconds", 0.0),
                                }
                        except Exception:
                            pass
                    
                    # If still no timing, aggregate from per_class
                    timing_dict = consolidated["multi_agent"]["timing"]
                    if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                        total_rollout = sum(
                            pc.get("timing", {}).get("total_rollout_time_seconds", 0.0)
                            for pc in consolidated["multi_agent"]["per_class"].values()
                        )
                        if total_rollout > 0:
                            consolidated["multi_agent"]["timing"] = {
                                "total_rollout_time_seconds": total_rollout,
                                "total_inference_time_seconds": 0.0,  # Not available without inference file
                            }
                logger.info("✓ Loaded multi-agent consolidated metrics")
            except Exception as e:
                logger.warning(f"⚠ Could not load multi-agent consolidated metrics: {e}")
        else:
            # Fallback to summary.json
            try:
                with open(multi_agent_summary_file, 'r') as f:
                    ma_data = json.load(f)
                ma_summary = ma_data.get("summary", {})
                ma_stats = ma_summary.get("overall_stats", {})
                
                consolidated["multi_agent"] = {
                    "wandb_run_url": None,  # Try to get from wandb_run_url.txt if available
                    "algorithm": ma_summary.get("algorithm"),
                    "model_type": ma_summary.get("model_type"),
                    "metrics": {
                        "instance_level": {
                            "mean_precision": ma_stats.get("mean_instance_precision", 0.0),
                            "mean_coverage": ma_stats.get("mean_instance_coverage", 0.0),
                            "std_precision": ma_stats.get("std_instance_precision", 0.0),
                            "std_coverage": ma_stats.get("std_instance_coverage", 0.0),
                        },
                        "class_union": {  # Class union metrics (union of class-based anchors only)
                            "mean_precision": ma_stats.get("mean_class_precision", 0.0),
                            "mean_coverage": ma_stats.get("mean_class_coverage", 0.0),
                            "std_precision": ma_stats.get("std_class_precision", 0.0),
                            "std_coverage": ma_stats.get("std_class_coverage", 0.0),
                        },
                        "class_based": {  # Class-based metrics (centroid-based rollouts)
                            "mean_precision": ma_stats.get("mean_class_based_precision", ma_stats.get("mean_class_level_precision", 0.0)),
                            "mean_coverage": ma_stats.get("mean_class_based_coverage", ma_stats.get("mean_class_level_coverage", 0.0)),
                            "std_precision": ma_stats.get("std_class_based_precision", 0.0),
                            "std_coverage": ma_stats.get("std_class_based_coverage", 0.0),
                        },
                        "global": {
                            "total_unique_rules": ma_stats.get("total_unique_rules", 0),
                            "mean_unique_rules_per_class": ma_stats.get("mean_unique_rules_per_class", 0.0),
                        }
                    },
                    "per_class": {},
                    "timing": {}
                }
                
                # Add per-class metrics
                per_class_summary = ma_summary.get("per_class_summary", {})
                # Filter out _class_based entries - only process main class entries
                per_class_filtered = {
                    k: v for k, v in per_class_summary.items() 
                    if not k.endswith('_class_based') and v.get('rollout_type') != 'class_based'
                }
                for class_key, class_data in per_class_filtered.items():
                    cls = class_data.get("class", -1)
                    # Extract all three metric types
                    instance_prec = class_data.get("instance_precision", 0.0)
                    instance_cov = class_data.get("instance_coverage", 0.0)
                    class_union_prec = class_data.get("class_union_precision", class_data.get("class_precision", 0.0))
                    class_union_cov = class_data.get("class_union_coverage", class_data.get("class_coverage", 0.0))
                    class_based_prec = class_data.get("class_level_precision", class_data.get("class_based_precision", 0.0))
                    class_based_cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", 0.0))
                    
                    consolidated["multi_agent"]["per_class"][f"class_{cls}"] = {
                        "instance_precision": instance_prec,
                        "instance_coverage": instance_cov,
                        "class_union_precision": class_union_prec,
                        "class_union_coverage": class_union_cov,
                        # Legacy fields for backward compatibility
                        "class_precision": class_union_prec,
                        "class_coverage": class_union_cov,
                        # Class-based metrics (centroid-based rollouts)
                        "class_based_precision": class_based_prec,
                        "class_based_coverage": class_based_cov,
                        "n_unique_rules": class_data.get("n_unique_rules", 0),
                        "timing": {
                            "avg_rollout_time_seconds": class_data.get("avg_rollout_time_seconds", 0.0),
                            "total_rollout_time_seconds": class_data.get("total_rollout_time_seconds", 0.0),
                            "class_total_time_seconds": class_data.get("class_total_time_seconds", 0.0),
                        }
                    }
                
                # Try to extract timing from inference results file
                ma_inference_file = ma_summary_path.parent / "extracted_rules.json"
                if ma_inference_file.exists():
                    try:
                        with open(ma_inference_file, 'r') as f:
                            ma_inference_data = json.load(f)
                        ma_metadata = ma_inference_data.get("metadata", {})
                        if ma_metadata:
                            consolidated["multi_agent"]["timing"] = {
                                "total_inference_time_seconds": ma_metadata.get("total_inference_time_seconds", 0.0),
                                "total_rollout_time_seconds": ma_metadata.get("total_rollout_time_seconds", 0.0),
                            }
                    except Exception as e:
                        logger.debug(f"Could not extract timing from multi-agent inference file: {e}")
                
                # If still no timing, aggregate from per_class timing data
                timing_dict = consolidated["multi_agent"]["timing"]
                if not timing_dict or (isinstance(timing_dict, dict) and not any(v for v in timing_dict.values() if v)):
                    total_rollout = sum(
                        pc.get("timing", {}).get("total_rollout_time_seconds", 0.0)
                        for pc in consolidated["multi_agent"]["per_class"].values()
                    )
                    if total_rollout > 0:
                        consolidated["multi_agent"]["timing"] = {
                            "total_rollout_time_seconds": total_rollout,
                            "total_inference_time_seconds": 0.0,  # Not available without inference file
                        }
                
                # Try to get wandb URL from file
                wandb_url_file = ma_summary_path.parent / "wandb_run_url.txt"
                if not wandb_url_file.exists():
                    # Check parent directories (BenchMARL experiment structure)
                    for parent in [ma_summary_path.parent.parent, ma_summary_path.parent.parent.parent]:
                        potential_file = parent / "wandb_run_url.txt"
                        if potential_file.exists():
                            wandb_url_file = potential_file
                            break
                if wandb_url_file.exists():
                    try:
                        with open(wandb_url_file, 'r') as f:
                            consolidated["multi_agent"]["wandb_run_url"] = f.read().strip()
                    except Exception:
                        pass
                
                logger.info("✓ Loaded multi-agent metrics from summary")
            except Exception as e:
                logger.warning(f"⚠ Could not load multi-agent summary: {e}")
        
        # Load NashConv metrics for multi-agent (only multi-agent has NashConv)
        if multi_agent_summary_file:
            ma_summary_path = Path(multi_agent_summary_file)
            # Try to infer experiment directory from summary path
            # Summary is typically at: experiment_dir/inference/summary.json or experiment_dir/summary.json
            ma_experiment_dir = None
            if "inference" in str(ma_summary_path):
                ma_experiment_dir = str(ma_summary_path.parent.parent)  # Go up from inference/
            else:
                ma_experiment_dir = str(ma_summary_path.parent)  # Same directory
            
            ma_nashconv = load_nashconv_metrics(ma_experiment_dir)
            if ma_nashconv.get("available", False):
                consolidated["multi_agent"]["nashconv"] = ma_nashconv
                logger.info("✓ Loaded multi-agent NashConv metrics")
    
    # Save consolidated metrics JSON
    consolidated_file = output_path / "consolidated_metrics_all_methods.json"
    with open(consolidated_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    logger.info(f"✓ Consolidated metrics JSON saved to: {consolidated_file}")
    logger.info(f"  This file contains all metrics, timing information, and wandb URLs for easy copying and comparison")


def create_comparison_summary(
    single_agent_summary_file: Optional[str],
    multi_agent_summary_file: Optional[str],
    baseline_results_file: Optional[str],
    output_dir: str,
    dataset: str
) -> None:
    """
    Create a comparison summary between baseline, single-agent, and multi-agent results.
    Also creates a consolidated metrics JSON with all metrics and wandb URLs.
    
    Args:
        single_agent_summary_file: Path to single-agent summary.json
        multi_agent_summary_file: Path to multi-agent summary.json
        baseline_results_file: Path to baseline results JSON file
        output_dir: Output directory for comparison
        dataset: Dataset name
    """
    logger.info(f"\n{'='*80}")
    logger.info("CREATING COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    
    comparison = {
        "dataset": dataset,
        "baseline": {},
        "single_agent": {},
        "multi_agent": {},
        "comparison": {}
    }
    
    # Load baseline results if available
    if baseline_results_file and Path(baseline_results_file).exists():
        try:
            with open(baseline_results_file, 'r') as f:
                baseline_data = json.load(f)
            comparison["baseline"] = baseline_data
            logger.info("✓ Loaded baseline results")
        except Exception as e:
            logger.warning(f"⚠ Could not load baseline results: {e}")
    
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
    baseline_data_full = comparison.get("baseline", {})
    baseline_static_anchors = baseline_data_full.get("methods", {}).get("static_anchors", {}) or baseline_data_full.get("static_anchors", {})
    baseline_stats = baseline_static_anchors.get("overall_stats", {}) if baseline_static_anchors else {}
    baseline_metadata = baseline_static_anchors.get("metadata", {}) if baseline_static_anchors else {}
    
    sa_stats = comparison["single_agent"].get("overall_stats", {})
    ma_stats = comparison["multi_agent"].get("overall_stats", {})
    
    # Load NashConv metrics for multi-agent
    ma_nashconv = None
    if multi_agent_summary_file:
        ma_summary_path = Path(multi_agent_summary_file)
        # Try to infer experiment directory
        if "inference" in str(ma_summary_path):
            ma_experiment_dir = str(ma_summary_path.parent.parent)
        else:
            ma_experiment_dir = str(ma_summary_path.parent)
        ma_nashconv = load_nashconv_metrics(ma_experiment_dir)
        if ma_nashconv.get("available", False):
            comparison["multi_agent"]["nashconv_metrics"] = ma_nashconv
    
    # Extract timing from inference files if not in summary
    sa_timing = {}
    ma_timing = {}
    baseline_timing = baseline_metadata or {}
    
    if single_agent_summary_file:
        sa_inference_file = Path(single_agent_summary_file).parent / "extracted_rules_single_agent.json"
        if sa_inference_file.exists():
            try:
                with open(sa_inference_file, 'r') as f:
                    sa_inference_data = json.load(f)
                sa_timing = sa_inference_data.get("metadata", {}).get("timing", {}) or sa_inference_data.get("metadata", {})
            except Exception:
                pass
    
    if multi_agent_summary_file:
        ma_inference_file = Path(multi_agent_summary_file).parent / "extracted_rules.json"
        if ma_inference_file.exists():
            try:
                with open(ma_inference_file, 'r') as f:
                    ma_inference_data = json.load(f)
                ma_timing = ma_inference_data.get("metadata", {}).get("timing", {}) or ma_inference_data.get("metadata", {})
            except Exception:
                pass
    
    # Handle both formats: single-agent uses "mean_precision", multi-agent uses "mean_instance_precision"
    # Calculate means from per_class_summary if not in overall_stats
    sa_per_class = comparison.get("single_agent", {}).get("per_class_summary", {})
    ma_per_class = comparison.get("multi_agent", {}).get("per_class_summary", {})
    
    # Compute single-agent metrics (available for logging even if multi-agent is missing)
    sa_instance_precision = 0.0
    sa_instance_coverage = 0.0
    sa_class_precision = 0.0
    sa_class_coverage = 0.0
    sa_class_based_precision = 0.0
    sa_class_based_coverage = 0.0
    
    if sa_stats:
        sa_instance_precisions = []
        sa_instance_coverages = []
        sa_class_union_precisions = []
        sa_class_union_coverages = []
        
        # Extract instance-level and class-union metrics from per_class_summary (skip _class_based entries)
        # Track seen classes to avoid duplicates
        seen_classes = set()
        for class_key, class_data in sa_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            class_val = class_data.get("class")
            # Only process each class once (in case there are duplicate entries)
            if class_val is not None and class_val not in seen_classes:
                seen_classes.add(class_val)
                inst_prec = class_data.get("instance_precision", 0.0)
                inst_cov = class_data.get("instance_coverage", 0.0)
                union_prec = class_data.get("class_union_precision", class_data.get("class_precision", 0.0))
                union_cov = class_data.get("class_union_coverage", class_data.get("class_coverage", 0.0))
                # Add all values (including 0.0) to get correct averages
                sa_instance_precisions.append(inst_prec)
                sa_instance_coverages.append(inst_cov)
                sa_class_union_precisions.append(union_prec)
                sa_class_union_coverages.append(union_cov)
        
        sa_instance_precision = sa_stats.get("mean_instance_precision") or sa_stats.get("mean_precision") or (
            float(np.mean(sa_instance_precisions)) if sa_instance_precisions else 0.0
        )
        sa_instance_coverage = sa_stats.get("mean_instance_coverage") or sa_stats.get("mean_coverage") or (
            float(np.mean(sa_instance_coverages)) if sa_instance_coverages else 0.0
        )
        sa_class_precision = sa_stats.get("mean_class_precision") or (
            float(np.mean(sa_class_union_precisions)) if sa_class_union_precisions else 0.0
        )
        sa_class_coverage = sa_stats.get("mean_class_coverage") or (
            float(np.mean(sa_class_union_coverages)) if sa_class_union_coverages else 0.0
        )
        
        # Class-based metrics (centroid-based rollouts)
        sa_class_based_precisions = []
        sa_class_based_coverages = []
        # Track seen classes to avoid duplicates (skip _class_based entries)
        seen_classes_cb = set()
        for class_key, class_data in sa_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            class_val = class_data.get("class")
            # Only process each class once
            if class_val is not None and class_val not in seen_classes_cb:
                seen_classes_cb.add(class_val)
                prec = class_data.get("class_level_precision", class_data.get("class_based_precision", None))
                cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", None))
                if prec is not None:
                    sa_class_based_precisions.append(prec)
                if cov is not None:
                    sa_class_based_coverages.append(cov)
        
        sa_class_based_precision = sa_stats.get("mean_class_based_precision", sa_stats.get("mean_class_level_precision", 
            float(np.mean(sa_class_based_precisions)) if sa_class_based_precisions else 0.0))
        sa_class_based_coverage = sa_stats.get("mean_class_based_coverage", sa_stats.get("mean_class_level_coverage",
            float(np.mean(sa_class_based_coverages)) if sa_class_based_coverages else 0.0))
    
    # Compute multi-agent metrics
    ma_instance_precision = 0.0
    ma_instance_coverage = 0.0
    ma_class_precision = 0.0
    ma_class_coverage = 0.0
    ma_class_based_precision = 0.0
    ma_class_based_coverage = 0.0
    
    if ma_stats:
        ma_instance_precisions = []
        ma_instance_coverages = []
        ma_class_union_precisions = []
        ma_class_union_coverages = []
        
        # Track seen classes to avoid duplicates
        seen_classes = set()
        for class_key, class_data in ma_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            class_val = class_data.get("class")
            # Only process each class once (in case there are duplicate entries)
            if class_val is not None and class_val not in seen_classes:
                seen_classes.add(class_val)
                inst_prec = class_data.get("instance_precision", 0.0)
                inst_cov = class_data.get("instance_coverage", 0.0)
                union_prec = class_data.get("class_union_precision", class_data.get("class_precision", 0.0))
                union_cov = class_data.get("class_union_coverage", class_data.get("class_coverage", 0.0))
                # Add all values (including 0.0) to get correct averages
                ma_instance_precisions.append(inst_prec)
                ma_instance_coverages.append(inst_cov)
                ma_class_union_precisions.append(union_prec)
                ma_class_union_coverages.append(union_cov)
        
        ma_instance_precision = ma_stats.get("mean_instance_precision") or (
            float(np.mean(ma_instance_precisions)) if ma_instance_precisions else 0.0
        )
        ma_instance_coverage = ma_stats.get("mean_instance_coverage") or (
            float(np.mean(ma_instance_coverages)) if ma_instance_coverages else 0.0
        )
        ma_class_precision = ma_stats.get("mean_class_precision") or (
            float(np.mean(ma_class_union_precisions)) if ma_class_union_precisions else 0.0
        )
        ma_class_coverage = ma_stats.get("mean_class_coverage") or (
            float(np.mean(ma_class_union_coverages)) if ma_class_union_coverages else 0.0
        )
        
        ma_class_based_precisions = []
        ma_class_based_coverages = []
        # Track seen classes to avoid duplicates (skip _class_based entries)
        seen_classes_cb = set()
        for class_key, class_data in ma_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            class_val = class_data.get("class")
            # Only process each class once
            if class_val is not None and class_val not in seen_classes_cb:
                seen_classes_cb.add(class_val)
                prec = class_data.get("class_level_precision", class_data.get("class_based_precision", None))
                cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", None))
                if prec is not None:
                    ma_class_based_precisions.append(prec)
                if cov is not None:
                    ma_class_based_coverages.append(cov)
        
        ma_class_based_precision = ma_stats.get("mean_class_based_precision", ma_stats.get("mean_class_level_precision",
            float(np.mean(ma_class_based_precisions)) if ma_class_based_precisions else 0.0))
        ma_class_based_coverage = ma_stats.get("mean_class_based_coverage", ma_stats.get("mean_class_level_coverage",
            float(np.mean(ma_class_based_coverages)) if ma_class_based_coverages else 0.0))
    
    if sa_stats and ma_stats:
        comparison["comparison"] = {
            # Instance-based comparison
            "instance_precision_diff": sa_instance_precision - ma_instance_precision,
            "instance_coverage_diff": sa_instance_coverage - ma_instance_coverage,
            # Class union comparison (Union of Class-Based Anchors Only)
            "class_precision_diff": sa_class_precision - ma_class_precision,
            "class_coverage_diff": sa_class_coverage - ma_class_coverage,
            # Class-based comparison (centroid-based rollouts)
            "class_based_precision_diff": sa_class_based_precision - ma_class_based_precision,
            "class_based_coverage_diff": sa_class_based_coverage - ma_class_based_coverage,
            "rules_diff": sa_stats.get("total_unique_rules", 0) - ma_stats.get("total_unique_rules", 0),
            "rules_class_based_diff": sa_stats.get("total_unique_rules_class_based", 0) - ma_stats.get("total_unique_rules_class_based", 0),
            # NashConv metrics (multi-agent only)
            "nashconv": {},
            # Timing comparison
            "timing": {
                "baseline": {
                    "total_inference_time_seconds": baseline_timing.get("total_inference_time_seconds", 0.0),
                    "total_rollout_time_seconds": baseline_timing.get("total_rollout_time_seconds", 0.0),
                },
                "single_agent": {
                    "total_inference_time_seconds": sa_timing.get("total_inference_time_seconds", 0.0),
                    "total_rollout_time_seconds": sa_timing.get("total_rollout_time_seconds", 0.0),
                },
                "multi_agent": {
                    "total_inference_time_seconds": ma_timing.get("total_inference_time_seconds", 0.0),
                    "total_rollout_time_seconds": ma_timing.get("total_rollout_time_seconds", 0.0),
                },
                "single_vs_multi_time_diff_seconds": (sa_timing.get("total_rollout_time_seconds", 0.0) - 
                                                      ma_timing.get("total_rollout_time_seconds", 0.0)),
                "baseline_vs_single_time_diff_seconds": (baseline_timing.get("total_rollout_time_seconds", 0.0) - 
                                                         sa_timing.get("total_rollout_time_seconds", 0.0)),
                "baseline_vs_multi_time_diff_seconds": (baseline_timing.get("total_rollout_time_seconds", 0.0) - 
                                                        ma_timing.get("total_rollout_time_seconds", 0.0)),
            },
            # Legacy fields for backward compatibility
            "precision_diff": sa_instance_precision - ma_instance_precision,
            "coverage_diff": sa_instance_coverage - ma_instance_coverage,
        }
        
        # Add NashConv metrics to comparison if available
        if ma_nashconv and ma_nashconv.get("available", False):
            nashconv_comparison = {}
            
            # Training metrics
            if ma_nashconv.get("training"):
                training_data = ma_nashconv["training"]
                if training_data:
                    final_training = training_data[-1]
                    nashconv_comparison["training"] = {
                        "final_nashconv_sum": final_training.get("training/nashconv_sum", 0.0),
                        "final_exploitability_max": final_training.get("training/exploitability_max", 0.0),
                        "final_class_nashconv_sum": final_training.get("training/class_nashconv_sum", None),
                        "final_step": final_training.get("step"),
                        "total_data_points": len(training_data)
                    }
                    
                    # Calculate convergence trend (first vs last)
                    if len(training_data) > 1:
                        first_training = training_data[0]
                        nashconv_comparison["training"]["convergence"] = {
                            "initial_nashconv_sum": first_training.get("training/nashconv_sum", 0.0),
                            "final_nashconv_sum": final_training.get("training/nashconv_sum", 0.0),
                            "improvement": first_training.get("training/nashconv_sum", 0.0) - final_training.get("training/nashconv_sum", 0.0),
                            "improvement_pct": ((first_training.get("training/nashconv_sum", 0.0) - final_training.get("training/nashconv_sum", 0.0)) / 
                                              max(first_training.get("training/nashconv_sum", 1e-6), 1e-6) * 100) if first_training.get("training/nashconv_sum", 0.0) > 0 else 0.0
                        }
            
            # Evaluation metrics
            if ma_nashconv.get("evaluation"):
                eval_data = ma_nashconv["evaluation"]
                if eval_data:
                    final_eval = eval_data[-1]
                    nashconv_comparison["evaluation"] = {
                        "final_nashconv_sum": final_eval.get("evaluation/nashconv_sum", 0.0),
                        "final_exploitability_max": final_eval.get("evaluation/exploitability_max", 0.0),
                        "final_class_nashconv_sum": final_eval.get("evaluation/class_nashconv_sum", None),
                        "final_step": final_eval.get("step"),
                        "total_data_points": len(eval_data)
                    }
                    
                    # Calculate convergence trend (first vs last)
                    if len(eval_data) > 1:
                        first_eval = eval_data[0]
                        nashconv_comparison["evaluation"]["convergence"] = {
                            "initial_nashconv_sum": first_eval.get("evaluation/nashconv_sum", 0.0),
                            "final_nashconv_sum": final_eval.get("evaluation/nashconv_sum", 0.0),
                            "improvement": first_eval.get("evaluation/nashconv_sum", 0.0) - final_eval.get("evaluation/nashconv_sum", 0.0),
                            "improvement_pct": ((first_eval.get("evaluation/nashconv_sum", 0.0) - final_eval.get("evaluation/nashconv_sum", 0.0)) / 
                                              max(first_eval.get("evaluation/nashconv_sum", 1e-6), 1e-6) * 100) if first_eval.get("evaluation/nashconv_sum", 0.0) > 0 else 0.0
                        }
            
            if nashconv_comparison:
                comparison["comparison"]["nashconv"] = nashconv_comparison
    
    # Save comparison
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_file = output_path / "comparison_summary.json"
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"✓ Comparison summary saved to: {comparison_file}")
    
    # Create consolidated metrics JSON with all methods and wandb URLs
    create_consolidated_metrics_json(
        baseline_results_file=baseline_results_file,
        single_agent_summary_file=single_agent_summary_file,
        multi_agent_summary_file=multi_agent_summary_file,
        output_dir=str(output_path),
        dataset=dataset
    )
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}")
    if baseline_stats:
        baseline_instance_precision = baseline_stats.get("mean_precision", 0.0)
        baseline_instance_coverage = baseline_stats.get("mean_coverage", 0.0)
        baseline_total_time = baseline_timing.get("total_rollout_time_seconds", 0.0)
        logger.info(f"Baseline (Static Anchors):")
        logger.info(f"  Instance-Based - Precision: {baseline_instance_precision:.4f}, Coverage: {baseline_instance_coverage:.4f}")
        if baseline_total_time > 0:
            logger.info(f"  Timing - Total Rollout Time: {baseline_total_time:.4f}s")
    if sa_stats:
        sa_total_time = sa_timing.get("total_rollout_time_seconds", 0.0)
        
        # Check if class-based data exists by looking at per_class_summary
        sa_per_class = comparison.get("single_agent", {}).get("per_class_summary", {})
        has_class_based_data = False
        if sa_per_class:
            # Check if any class has class-based metrics (filter out _class_based entries)
            for class_key, class_data in sa_per_class.items():
                if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                    continue
                if "class_level_precision" in class_data or "class_based_precision" in class_data:
                    has_class_based_data = True
                    break
            # Also check if class-based rules exist
            if not has_class_based_data and sa_stats.get("total_unique_rules_class_based", 0) > 0:
                has_class_based_data = True
        
        logger.info(f"Single-Agent:")
        logger.info(f"  Instance-Based - Precision: {sa_instance_precision:.4f}, Coverage: {sa_instance_coverage:.4f}")
        logger.info(f"  Class Union (Union of Class-Based Anchors Only) - Precision: {sa_class_precision:.4f}, Coverage: {sa_class_coverage:.4f}")
        if has_class_based_data or sa_stats.get("total_unique_rules_class_based", 0) > 0:
            logger.info(f"  Class-Based - Precision: {sa_class_based_precision:.4f}, Coverage: {sa_class_based_coverage:.4f}")
        else:
            logger.info(f"  Class-Based - Precision: N/A (no class-based data), Coverage: N/A (no class-based data)")
        logger.info(f"  Total Unique Rules (Instance-Based): {sa_stats.get('total_unique_rules_instance_based', sa_stats.get('total_unique_rules', 0))}")
        logger.info(f"  Total Unique Rules (Class-Based): {sa_stats.get('total_unique_rules_class_based', 0)}")
        if sa_total_time > 0:
            logger.info(f"  Timing - Total Rollout Time: {sa_total_time:.4f}s")
    if ma_stats:
        ma_total_time = ma_timing.get("total_rollout_time_seconds", 0.0)
        
        # Check if class-based data exists by looking at per_class_summary
        ma_per_class = comparison.get("multi_agent", {}).get("per_class_summary", {})
        has_class_based_data = False
        if ma_per_class:
            # Check if any class has class-based metrics (filter out _class_based entries)
            for class_key, class_data in ma_per_class.items():
                if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                    continue
                if "class_level_precision" in class_data or "class_based_precision" in class_data:
                    prec = class_data.get("class_level_precision", class_data.get("class_based_precision", 0.0))
                    cov = class_data.get("class_level_coverage", class_data.get("class_based_coverage", 0.0))
                    if prec > 0 or cov > 0:
                        has_class_based_data = True
                        break
            # Also check if class-based rules exist
            if not has_class_based_data and ma_stats.get("total_unique_rules_class_based", 0) > 0:
                has_class_based_data = True
        
        logger.info(f"Multi-Agent:")
        logger.info(f"  Instance-Based - Precision: {ma_instance_precision:.4f}, Coverage: {ma_instance_coverage:.4f}")
        logger.info(f"  Class Union (Union of Class-Based Anchors Only) - Precision: {ma_class_precision:.4f}, Coverage: {ma_class_coverage:.4f}")
        if has_class_based_data or ma_stats.get("total_unique_rules_class_based", 0) > 0:
            logger.info(f"  Class-Based - Precision: {ma_class_based_precision:.4f}, Coverage: {ma_class_based_coverage:.4f}")
        else:
            logger.info(f"  Class-Based - Precision: N/A (no class-based data), Coverage: N/A (no class-based data)")
        logger.info(f"  Total Unique Rules (Instance-Based): {ma_stats.get('total_unique_rules', 0)}")
        logger.info(f"  Total Unique Rules (Class-Based): {ma_stats.get('total_unique_rules_class_based', 0)}")
        if ma_total_time > 0:
            logger.info(f"  Timing - Total Rollout Time: {ma_total_time:.4f}s")
        
        # Log NashConv metrics if available
        if ma_nashconv and ma_nashconv.get("available", False):
            logger.info(f"\nMulti-Agent Nash Equilibrium Convergence:")
            if ma_nashconv.get("training"):
                training_data = ma_nashconv["training"]
                if training_data:
                    final_training = training_data[-1]
                    logger.info(f"  Training NashConv (final):")
                    logger.info(f"    NashConv sum: {final_training.get('training/nashconv_sum', 'N/A'):.6f}")
                    logger.info(f"    Max exploitability: {final_training.get('training/exploitability_max', 'N/A'):.6f}")
            if ma_nashconv.get("evaluation"):
                eval_data = ma_nashconv["evaluation"]
                if eval_data:
                    final_eval = eval_data[-1]
                    logger.info(f"  Evaluation NashConv (final):")
                    logger.info(f"    NashConv sum: {final_eval.get('evaluation/nashconv_sum', 'N/A'):.6f}")
                    logger.info(f"    Max exploitability: {final_eval.get('evaluation/exploitability_max', 'N/A'):.6f}")
    
    if comparison["comparison"]:
        logger.info(f"Differences (Single - Multi):")
        logger.info(f"  Instance-Based - Precision: {comparison['comparison']['instance_precision_diff']:.4f}, Coverage: {comparison['comparison']['instance_coverage_diff']:.4f}")
        logger.info(f"  Class Union (Union of Class-Based Anchors Only) - Precision: {comparison['comparison']['class_precision_diff']:.4f}, Coverage: {comparison['comparison']['class_coverage_diff']:.4f}")
        if "class_based_precision_diff" in comparison["comparison"]:
            logger.info(f"  Class-Based - Precision: {comparison['comparison']['class_based_precision_diff']:.4f}, Coverage: {comparison['comparison']['class_based_coverage_diff']:.4f}")
        logger.info(f"  Rules (Instance-Based): {comparison['comparison']['rules_diff']}")
        if "rules_class_based_diff" in comparison["comparison"]:
            logger.info(f"  Rules (Class-Based): {comparison['comparison']['rules_class_based_diff']}")
        if "timing" in comparison["comparison"]:
            timing = comparison["comparison"]["timing"]
            logger.info(f"Timing Comparison:")
            if timing.get("single_vs_multi_time_diff_seconds", 0) != 0:
                logger.info(f"  Single vs Multi Time Difference: {timing['single_vs_multi_time_diff_seconds']:.4f}s")
            if timing.get("baseline_vs_single_time_diff_seconds", 0) != 0:
                logger.info(f"  Baseline vs Single Time Difference: {timing['baseline_vs_single_time_diff_seconds']:.4f}s")
            if timing.get("baseline_vs_multi_time_diff_seconds", 0) != 0:
                logger.info(f"  Baseline vs Multi Time Difference: {timing['baseline_vs_multi_time_diff_seconds']:.4f}s")
    
    # Log NashConv comparison if available
    if comparison.get("comparison", {}).get("nashconv"):
        nashconv_comp = comparison["comparison"]["nashconv"]
        logger.info(f"\nNash Equilibrium Convergence (Multi-Agent):")
        if nashconv_comp.get("training"):
            train_nc = nashconv_comp["training"]
            logger.info(f"  Training (final):")
            logger.info(f"    NashConv sum: {train_nc.get('final_nashconv_sum', 'N/A'):.6f}")
            logger.info(f"    Max exploitability: {train_nc.get('final_exploitability_max', 'N/A'):.6f}")
            if train_nc.get("convergence"):
                conv = train_nc["convergence"]
                logger.info(f"    Convergence: {conv.get('initial_nashconv_sum', 0.0):.6f} → {conv.get('final_nashconv_sum', 0.0):.6f} "
                          f"(improvement: {conv.get('improvement', 0.0):.6f}, {conv.get('improvement_pct', 0.0):.1f}%)")
        if nashconv_comp.get("evaluation"):
            eval_nc = nashconv_comp["evaluation"]
            logger.info(f"  Evaluation (final):")
            logger.info(f"    NashConv sum: {eval_nc.get('final_nashconv_sum', 'N/A'):.6f}")
            logger.info(f"    Max exploitability: {eval_nc.get('final_exploitability_max', 'N/A'):.6f}")
            if eval_nc.get("convergence"):
                conv = eval_nc["convergence"]
                logger.info(f"    Convergence: {conv.get('initial_nashconv_sum', 0.0):.6f} → {conv.get('final_nashconv_sum', 0.0):.6f} "
                          f"(improvement: {conv.get('improvement', 0.0):.6f}, {conv.get('improvement_pct', 0.0):.1f}%)")
    
    # Log class union rules (class-based union rules)
    logger.info(f"\n{'='*80}")
    logger.info("CLASS UNION RULES (Class-Based Union - Smallest Set of General Rules)")
    logger.info(f"{'='*80}")
    
    # Single-agent class union rules
    sa_per_class = comparison.get("single_agent", {}).get("per_class_summary", {})
    if sa_per_class:
        logger.info(f"\nSingle-Agent Class Union Rules:")
        seen_classes_sa = set()
        for class_key, class_data in sa_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            target_class = class_data.get("class")
            if target_class is not None and target_class not in seen_classes_sa:
                seen_classes_sa.add(target_class)
                # Get class union rules (class-based unique rules)
                union_rules = class_data.get("class_level_unique_rules", [])
                # Fallback to separate class_based entry
                if not union_rules:
                    class_based_key = f"class_{target_class}_class_based"
                    if class_based_key in sa_per_class:
                        union_rules = sa_per_class[class_based_key].get("unique_rules", [])
                
                logger.info(f"\n  Class {target_class} - Class Union Rules ({len(union_rules)} rules):")
                logger.info(f"    Precision: {class_data.get('class_precision', 0.0):.4f}, Coverage: {class_data.get('class_coverage', 0.0):.4f}")
                if union_rules:
                    for i, rule in enumerate(union_rules[:5], 1):  # Show first 5 rules
                        rule_display = rule[:120] + "..." if len(rule) > 120 else rule
                        logger.info(f"    Rule {i}: {rule_display}")
                    if len(union_rules) > 5:
                        logger.info(f"    ... and {len(union_rules) - 5} more rules")
                else:
                    logger.info(f"    No class union rules found")
    
    # Multi-agent class union rules
    ma_per_class = comparison.get("multi_agent", {}).get("per_class_summary", {})
    if ma_per_class:
        logger.info(f"\nMulti-Agent Class Union Rules:")
        seen_classes_ma = set()
        for class_key, class_data in ma_per_class.items():
            # Skip class-based result keys (they're stored separately, not as actual classes)
            if class_key.endswith("_class_based") or class_data.get("rollout_type") == "class_based":
                continue
            target_class = class_data.get("class")
            if target_class is not None and target_class not in seen_classes_ma:
                seen_classes_ma.add(target_class)
                # Get class union rules (class-based unique rules)
                union_rules = class_data.get("class_level_unique_rules", [])
                # Fallback: check if class_based_results has rules
                if not union_rules and "class_based_results" in class_data:
                    # Collect rules from all agents' class-based results
                    union_rules = []
                    for agent_result in class_data["class_based_results"].values():
                        agent_rules = agent_result.get("unique_rules", [])
                        union_rules.extend(agent_rules)
                    # Deduplicate
                    union_rules = list(set(union_rules))
                
                logger.info(f"\n  Class {target_class} - Class Union Rules ({len(union_rules)} rules):")
                logger.info(f"    Precision: {class_data.get('class_precision', 0.0):.4f}, Coverage: {class_data.get('class_coverage', 0.0):.4f}")
                if union_rules:
                    for i, rule in enumerate(union_rules[:5], 1):  # Show first 5 rules
                        rule_display = rule[:120] + "..." if len(rule) > 120 else rule
                        logger.info(f"    Rule {i}: {rule_display}")
                    if len(union_rules) > 5:
                        logger.info(f"    ... and {len(union_rules) - 5} more rules")
                else:
                    logger.info(f"    No class union rules found")
    
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
    
    # Build dataset choices dynamically
    dataset_choices = build_dataset_choices()
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=dataset_choices,
        help="Dataset name. For UCIML: uci_<name_or_id>. For Folktables: folktables_<task>_<state>_<year>"
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
        default=100,
        help="Steps per episode (default: 100)"
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
    
    # Set up output directories first (needed for log file location)
    if args.output_dir is None:
        # Add datetime stamp to prevent overwriting previous results
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "comparison_results" / f"{args.dataset}_{args.algorithm}_{datetime_str}")
    else:
        # Resolve relative paths relative to project root
        args.output_dir = str(Path(args.output_dir).resolve())
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging to save all output to a log file
    log_file = output_path / "pipeline_run.log"
    
    # Remove ALL existing handlers from root logger and our logger to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logger.handlers.clear()
    
    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler (stderr) - for logger messages only
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (write to log file)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture more detail in file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set root logger level
    logging.root.setLevel(logging.INFO)
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPLETE PIPELINE: SINGLE-AGENT vs MULTI-AGENT COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Single-Agent Algorithm: {single_agent_algorithm.upper()}")
    logger.info(f"Multi-Agent Algorithm: {multi_agent_algorithm.upper()}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Logging all output to: {log_file}")
    logger.info(f"Results will be saved to: {args.output_dir}")
    logger.info(f"{'='*80}\n")
    
    # Track paths for comparison
    single_agent_experiment_dir = None
    single_agent_rules_file = None
    single_agent_summary_file = None
    
    multi_agent_experiment_dir = None
    multi_agent_rules_file = None
    multi_agent_summary_file = None
    
    baseline_results_file = None
    
    # Run baseline pipeline first (it's independent and typically faster)
    # Skip if either --skip_baseline or --skip_training is set (baseline includes training)
    if not args.skip_baseline and not args.skip_training:
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
        if args.skip_training:
            logger.info(f"\n{'='*80}")
            logger.info("BASELINE PIPELINE (SKIPPED - --skip_training enabled)")
            logger.info(f"{'='*80}\n")
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
            logger.info("Training skipped (--skip_training). Looking for existing experiment directory...")
            # Try to find existing experiment directory
            if args.multi_agent_output_dir:
                multi_agent_experiment_dir = args.multi_agent_output_dir
                logger.info(f"Using specified multi-agent output directory: {multi_agent_experiment_dir}")
            else:
                # BenchMARL/Hydra stores experiments directly in the BenchMARL root directory
                # with names like: maddpg_anchor_mlp__{hash}_{date}-{time}
                # The experiment directory name is printed in training output: "BenchMARL checkpoint location: {path}"
                benchmarl_dir = PROJECT_ROOT / "BenchMARL"
                
                if benchmarl_dir.exists():
                    experiment_dirs = []
                    for item in benchmarl_dir.iterdir():
                        if item.is_dir() and not item.name.startswith('.') and item.name not in ['output', 'conf', 'data', 'docs', '__pycache__', 'old_results']:
                            # Check if this looks like an experiment directory
                            if (item / "checkpoints").exists() or (item / "individual_models").exists() or (item / "config.pkl").exists():
                                # Check if it matches the algorithm pattern (e.g., "maddpg" in name)
                                if multi_agent_algorithm.lower() in item.name.lower():
                                    experiment_dirs.append(item)
                    
                    if experiment_dirs:
                        # Sort by modification time (most recent first)
                        multi_agent_experiment_dir = str(max(experiment_dirs, key=lambda p: p.stat().st_mtime))
                        logger.info(f"✓ Found existing experiment directory in BenchMARL/: {multi_agent_experiment_dir}")
                    else:
                        logger.warning(f"⚠ No experiment directories found in {benchmarl_dir} matching algorithm '{multi_agent_algorithm}'")
                else:
                    logger.warning(f"⚠ BenchMARL directory not found at {benchmarl_dir}")
                
                if not multi_agent_experiment_dir:
                    logger.warning("⚠ No existing multi-agent experiment directory found.")
                    logger.warning("  Multi-agent inference, testing, and summarization will be skipped.")
                    logger.warning("  To run multi-agent pipeline, either:")
                    logger.warning("    1. Run without --skip_training to train a new model")
                    logger.warning("    2. Specify --multi_agent_output_dir to point to an existing experiment")
        
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
        elif args.skip_inference:
            logger.info("Inference skipped (--skip_inference). Looking for existing rules file...")
            # Try to find existing rules file
            if multi_agent_experiment_dir:
                rules_file = Path(multi_agent_experiment_dir) / "inference" / "extracted_rules.json"
                if rules_file.exists():
                    multi_agent_rules_file = str(rules_file)
                    logger.info(f"✓ Found existing rules file: {multi_agent_rules_file}")
                else:
                    logger.warning(f"⚠ Rules file not found at {rules_file}")
            else:
                logger.warning("⚠ Cannot search for rules file: no experiment directory found")
        elif not multi_agent_experiment_dir:
            logger.warning("⚠ Inference skipped: no experiment directory found")
        
        # Testing
        if not args.skip_testing and multi_agent_rules_file:
            run_multi_agent_test(
                rules_file=multi_agent_rules_file,
                dataset=args.dataset,
                seed=args.seed
            )
        elif args.skip_testing:
            logger.info("Testing skipped (--skip_testing)")
        elif not multi_agent_rules_file:
            logger.warning("⚠ Testing skipped: no rules file found")
        
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
                logger.info(f"✓ Multi-agent summary saved to: {multi_agent_summary_file}")
            else:
                logger.warning(f"⚠ Multi-agent summary file not found at {summary_file}")
        else:
            logger.warning("⚠ Summarization skipped: no rules file available")
    
    # Create comparison summary
    if single_agent_summary_file or multi_agent_summary_file:
        create_comparison_summary(
            single_agent_summary_file=single_agent_summary_file,
            multi_agent_summary_file=multi_agent_summary_file,
            baseline_results_file=baseline_results_file,
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
    
    # Run classifier rules analysis if rules files are available
    if single_agent_rules_file or multi_agent_rules_file:
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING CLASSIFIER RULES ANALYSIS")
        logger.info(f"{'='*80}")
        
        analyze_rules_script = PROJECT_ROOT / "analyze_classifier_rules.py"
        if analyze_rules_script.exists():
            # Create output directory for rules analysis
            rules_analysis_dir = output_path / "classifier_rules_analysis"
            rules_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Analyze single-agent rules if available
            if single_agent_rules_file:
                logger.info(f"Analyzing single-agent classifier rules...")
                cmd = [
                    sys.executable,
                    str(analyze_rules_script),
                    "--rules_file", single_agent_rules_file,
                    "--dataset", args.dataset,
                    "--output_dir", str(rules_analysis_dir / "single_agent")
                ]
                
                success, output = run_command(
                    cmd,
                    description="Analyze single-agent classifier rules",
                    cwd=str(PROJECT_ROOT),
                    capture_output=True
                )
                
                if success:
                    logger.info("✓ Single-agent classifier rules analysis completed")
                    # Save output to log file
                    log_file = rules_analysis_dir / "single_agent_analysis.log"
                    with open(log_file, 'w') as f:
                        f.write(output)
                    logger.info(f"  Analysis log saved to: {log_file}")
                else:
                    logger.warning(f"⚠ Failed to analyze single-agent classifier rules: {output}")
            
            # Analyze multi-agent rules if available
            if multi_agent_rules_file:
                logger.info(f"Analyzing multi-agent classifier rules...")
                cmd = [
                    sys.executable,
                    str(analyze_rules_script),
                    "--rules_file", multi_agent_rules_file,
                    "--dataset", args.dataset,
                    "--output_dir", str(rules_analysis_dir / "multi_agent")
                ]
                
                success, output = run_command(
                    cmd,
                    description="Analyze multi-agent classifier rules",
                    cwd=str(PROJECT_ROOT),
                    capture_output=True
                )
                
                if success:
                    logger.info("✓ Multi-agent classifier rules analysis completed")
                    # Save output to log file
                    log_file = rules_analysis_dir / "multi_agent_analysis.log"
                    with open(log_file, 'w') as f:
                        f.write(output)
                    logger.info(f"  Analysis log saved to: {log_file}")
                else:
                    logger.warning(f"⚠ Failed to analyze multi-agent classifier rules: {output}")
        else:
            logger.warning(f"⚠ analyze_classifier_rules.py not found at {analyze_rules_script}")
    
    # Run EDA-informed result analysis if results directory exists
    if output_path.exists():
        logger.info(f"\n{'='*80}")
        logger.info("RUNNING EDA-INFORMED RESULT ANALYSIS")
        logger.info(f"{'='*80}")
        
        analyze_eda_script = PROJECT_ROOT / "analyze_results_with_eda.py"
        if analyze_eda_script.exists():
            # Create output directory for EDA analysis
            eda_analysis_dir = output_path / "eda_analysis"
            eda_analysis_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Running EDA-informed result analysis...")
            cmd = [
                sys.executable,
                str(analyze_eda_script),
                "--dataset", args.dataset,
                "--results_dir", str(output_path),
                "--output_dir", str(eda_analysis_dir)
            ]
            
            success, output = run_command(
                cmd,
                description="Analyze results with EDA context",
                cwd=str(PROJECT_ROOT),
                capture_output=True
            )
            
            if success:
                logger.info("✓ EDA-informed result analysis completed")
                # Save output to log file
                log_file = eda_analysis_dir / "eda_analysis.log"
                with open(log_file, 'w') as f:
                    f.write(output)
                logger.info(f"  Analysis log saved to: {log_file}")
                logger.info(f"  Analysis results saved to: {eda_analysis_dir}")
            else:
                logger.warning(f"⚠ Failed to run EDA-informed result analysis: {output}")
                # Save error output to log file anyway
                log_file = eda_analysis_dir / "eda_analysis_error.log"
                with open(log_file, 'w') as f:
                    f.write(output or "No output captured")
                logger.info(f"  Error log saved to: {log_file}")
        else:
            logger.warning(f"⚠ analyze_results_with_eda.py not found at {analyze_eda_script}")
    
    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Complete log saved to: {log_file}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()

