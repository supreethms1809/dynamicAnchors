"""
Automated script to run sklearn_datasets_anchors.py with all datasets
using TD3 (continuous actions) in both joint and non-joint modes.

This script will:
1. Iterate through all available datasets
2. Run each dataset in both joint and non-joint training modes
3. Use TD3 algorithm for continuous actions
4. Handle appropriate sample sizes for large datasets
5. Log all runs with timestamps
6. Generate rules with ALL features (max_features_in_rule=-1 for comprehensive analysis)

Usage:
    python -m trainers.auto_run_td3_experiments
    python -m trainers.auto_run_td3_experiments --datasets breast_cancer wine
    python -m trainers.auto_run_td3_experiments --skip-joint  # Skip joint training
    python -m trainers.auto_run_td3_experiments --skip-non-joint  # Skip non-joint training
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
from typing import List, Optional, Tuple


# Production-optimized dataset configurations
# These configurations are optimized for:
# - Better performance (increased episodes, steps_per_episode)
# - Higher computational utilization (larger batch sizes, more parallel envs)
# - Better precision estimates (more perturbation samples)
# Conservative estimates for reliable production runs
def get_available_gpus() -> List[int]:
    """
    Detect available GPUs on the system.
    
    Returns:
        List of available GPU IDs (e.g., [0, 1, 2, 3]).
        Returns empty list if no GPUs found or if CUDA is not available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return []
        
        return list(range(n_gpus))
    except ImportError:
        # PyTorch not installed, try using nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output like "GPU 0: NVIDIA GeForce RTX 3090"
                gpu_ids = []
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('GPU'):
                        parts = line.split(':')
                        if len(parts) > 0:
                            gpu_id_str = parts[0].replace('GPU', '').strip()
                            try:
                                gpu_ids.append(int(gpu_id_str))
                            except ValueError:
                                pass
                return sorted(gpu_ids)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    return []


DATASET_CONFIGS = {
    "breast_cancer": {
        "sample_size": None,  # Small dataset, use all
        "description": "Binary classification (2 classes, 30 features, 569 samples)",
        # Production-optimized training parameters
        "episodes": 200,  # Increased from default 150 for better convergence
        "steps_per_episode": 2000,  # Increased from default 1500 (conservative but better performance)
        "batch_size": 512,  # Increased from 256 for better GPU utilization
        "n_perturb": 4096,  # Increased from 2048 for better precision estimates
        "n_envs": 2,  # For PPO (TD3/DDPG uses 1 automatically)
    },
    "wine": {
        "sample_size": None,  # Small dataset, use all
        "description": "Multi-class classification (3 classes, 13 features, 178 samples)",
        # Production-optimized training parameters
        "episodes": 200,
        "steps_per_episode": 2000,
        "batch_size": 512,
        "n_perturb": 4096,
        "n_envs": 2,
    },
    "covtype": {
        "sample_size": 20000,  # Large dataset, sample for reasonable runtime (original: 581k samples)
        "description": "Multi-class classification (7 classes, 54 features, 581k samples)",
        # Production-optimized training parameters for complex dataset
        "episodes": 250,  # More episodes for complex dataset
        "steps_per_episode": 2500,  # More steps for complex dataset
        "batch_size": 1024,  # Larger batch for large dataset and better GPU utilization
        "n_perturb": 8192,  # More samples for complex dataset precision estimates
        "n_envs": 4,  # More parallel envs for large dataset (PPO only)
    },
    "housing": {
        "sample_size": 10000,  # Large dataset, sample for reasonable runtime (original: 20k samples)
        "description": "Multi-class classification (4 classes, 8 features, 20640 samples)",
        # Production-optimized training parameters
        "episodes": 200,
        "steps_per_episode": 2500,  # More steps for better convergence
        "batch_size": 1024,  # Larger batch for better GPU utilization
        "n_perturb": 8192,  # More samples for better precision estimates
        "n_envs": 4,  # More parallel envs for large dataset (PPO only)
    }
}


def run_experiment(
    dataset: str,
    joint: bool,
    sample_size: Optional[int] = None,
    classifier_type: str = "dnn",
    device: str = "auto",
    gpu_id: Optional[int] = None,
    episodes: Optional[int] = None,
    steps_per_episode: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_perturb: Optional[int] = None,
    n_envs: Optional[int] = None,
) -> bool:
    """
    Run a single experiment with specified parameters.
    
    Args:
        dataset: Dataset name
        joint: Whether to use joint training
        sample_size: Optional sample size for large datasets
        classifier_type: Classifier type ("dnn", "random_forest", or "gradient_boosting")
        device: Device to use ("auto", "cpu", "cuda", "mps")
        gpu_id: Optional GPU ID to use (sets CUDA_VISIBLE_DEVICES). If None, uses all available GPUs.
        episodes: Optional number of training episodes (uses dataset config if None)
        steps_per_episode: Optional steps per episode (uses dataset config if None)
        batch_size: Optional batch size (uses dataset config if None)
        n_perturb: Optional number of perturbation samples (uses dataset config if None)
        n_envs: Optional number of parallel envs (uses dataset config if None, ignored for TD3)
    
    Returns:
        True if successful, False otherwise
    """
    # Set up environment for GPU selection
    env = os.environ.copy()
    if gpu_id is not None:
        # Set CUDA_VISIBLE_DEVICES to only show the specified GPU
        # This makes the specified GPU appear as GPU 0 to the subprocess
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
    elif device == "cuda":
        # If cuda is specified but no gpu_id, use all available GPUs
        # Don't set CUDA_VISIBLE_DEVICES, let PyTorch use all GPUs
        pass
    
    # Build command
    cmd = [
        sys.executable, "-m", "trainers.sklearn_datasets_anchors",
        "--dataset", dataset,
        "--continuous-actions",
        "--continuous-algorithm", "td3",
        "--classifier-type", classifier_type,
        "--device", device
    ]
    
    # Add sample size if specified
    if sample_size is not None:
        cmd.extend(["--sample_size", str(sample_size)])
    
    # Add --no-joint flag if not using joint training
    if not joint:
        cmd.append("--no-joint")
    
    # Add production-optimized training parameters if provided
    if episodes is not None:
        cmd.extend(["--episodes", str(episodes)])
    if steps_per_episode is not None:
        cmd.extend(["--steps-per-episode", str(steps_per_episode)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if n_perturb is not None:
        cmd.extend(["--n-perturb", str(n_perturb)])
    if n_envs is not None:
        cmd.extend(["--n-envs", str(n_envs)])
    
    # Print header
    mode_str = "JOINT" if joint else "NON-JOINT"
    gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print("\n" + "="*80)
    print(f"Running: {dataset.upper()} - {mode_str} - TD3{gpu_info}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"CUDA_VISIBLE_DEVICES: {gpu_id}")
    print("="*80 + "\n")
    
    # Run the experiment
    metrics_path = None
    captured_output = []
    
    try:
        # Run with real-time output streaming and capture for parsing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env=env,  # Pass environment with CUDA_VISIBLE_DEVICES set
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Stream output in real-time and capture it
        for line in process.stdout:
            print(line, end='')  # Print in real-time (line already includes newline)
            captured_output.append(line)
            # Check for metrics file path while streaming
            if 'Saved metrics and rules to:' in line:
                parts = line.split('Saved metrics and rules to:')
                if len(parts) > 1:
                    metrics_path = parts[1].strip()
            elif 'Output Directory:' in line and not metrics_path:
                # Fallback: construct from output directory
                parts = line.split('Output Directory:')
                if len(parts) > 1:
                    output_dir = parts[1].strip()
                    metrics_path = os.path.join(output_dir, 'metrics_and_rules.json')
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        
        # If we still don't have metrics_path, search the captured output
        if not metrics_path:
            full_output = ''.join(captured_output)
            for line in full_output.split('\n'):
                if 'Saved metrics and rules to:' in line:
                    parts = line.split('Saved metrics and rules to:')
                    if len(parts) > 1:
                        metrics_path = parts[1].strip()
                        break
                elif 'Output Directory:' in line:
                    parts = line.split('Output Directory:')
                    if len(parts) > 1:
                        output_dir = parts[1].strip()
                        metrics_path = os.path.join(output_dir, 'metrics_and_rules.json')
                        break
        
        print(f"\n✓ Successfully completed: {dataset} - {mode_str}")
        
        # Run analyze_metrics.py on the metrics file
        if metrics_path:
            # Convert to absolute path if relative
            if not os.path.isabs(metrics_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                metrics_path = os.path.join(base_dir, metrics_path)
            
            if os.path.exists(metrics_path):
                print(f"\n{'='*80}")
                print(f"Running metrics analysis on: {metrics_path}")
                print(f"{'='*80}")
                try:
                    analyze_cmd = [
                        sys.executable, "-m", "trainers.analyze_metrics",
                        metrics_path
                    ]
                    analyze_result = subprocess.run(
                        analyze_cmd,
                        check=True,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    print(f"\n✓ Metrics analysis completed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"\n⚠ Warning: Metrics analysis failed (error code: {e.returncode})")
                    print(f"   Metrics file is still available at: {metrics_path}")
                except Exception as e:
                    print(f"\n⚠ Warning: Error running metrics analysis: {str(e)}")
                    print(f"   Metrics file is still available at: {metrics_path}")
            else:
                print(f"\n⚠ Warning: Metrics file not found at: {metrics_path}")
        else:
            print(f"\n⚠ Warning: Could not find metrics file path in output")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset} - {mode_str}")
        print(f"Error code: {e.returncode}")
        # Output was already printed in real-time
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted: {dataset} - {mode_str}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {dataset} - {mode_str}")
        print(f"Error: {str(e)}")
        return False


def main(
    datasets: Optional[List[str]] = None,
    skip_joint: bool = False,
    skip_non_joint: bool = False,
    classifier_type: str = "dnn",
    device: str = "auto",
    gpu_ids: Optional[List[int]] = None,
    parallel: bool = False
):
    """
    Main function to run all experiments.
    
    Args:
        datasets: List of datasets to run (None = all datasets)
        skip_joint: Skip joint training experiments
        skip_non_joint: Skip non-joint training experiments
        classifier_type: Classifier type to use
        device: Device to use ("auto", "cpu", "cuda", "mps")
        gpu_ids: List of GPU IDs to use for experiments (None = use all available or auto-detect)
        parallel: If True, run experiments in parallel on different GPUs (requires gpu_ids)
    """
    # Determine which datasets to run
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    else:
        # Validate datasets
        invalid = [d for d in datasets if d not in DATASET_CONFIGS]
        if invalid:
            print(f"Error: Invalid datasets: {invalid}")
            print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
            return
        datasets = list(datasets)
    
    # Print summary
    print("\n" + "="*80)
    print("AUTOMATED TD3 EXPERIMENTS - ALL DATASETS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Training modes:")
    if not skip_joint:
        print(f"  ✓ Joint training")
    if not skip_non_joint:
        print(f"  ✓ Non-joint training")
    if skip_joint and skip_non_joint:
        print("  ⚠ No training modes selected!")
        return
    print(f"Algorithm: TD3 (continuous actions)")
    print(f"Classifier type: {classifier_type}")
    print(f"Device: {device} ({'auto-detect' if device == 'auto' else device})")
    print("="*80)
    
    # Show dataset configurations
    print("\nDataset Configurations (Production-Optimized):")
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        sample_str = f"sample_size={config['sample_size']}" if config['sample_size'] else "all samples"
        print(f"  {dataset:15s}: {sample_str:20s} - {config['description']}")
        # Show production-optimized training parameters
        if 'episodes' in config:
            print(f"                 Training: {config['episodes']} episodes × {config['steps_per_episode']} steps/episode")
            print(f"                          batch_size={config['batch_size']}, n_perturb={config['n_perturb']:,}")
    
    # Calculate total number of experiments
    n_modes = (0 if skip_joint else 1) + (0 if skip_non_joint else 1)
    total_experiments = len(datasets) * n_modes
    print(f"\nTotal experiments to run: {total_experiments}")
    
    # Auto-detect GPUs if not specified and device supports CUDA
    if gpu_ids is None and device in ["cuda", "auto"]:
        detected_gpus = get_available_gpus()
        if detected_gpus:
            gpu_ids = detected_gpus
            print(f"\nAuto-detected {len(gpu_ids)} GPU(s): {gpu_ids}")
        else:
            if device == "cuda":
                print("Warning: CUDA requested but no GPUs detected. Falling back to CPU.")
                device = "cpu"
            else:
                print("No GPUs detected. Will use CPU if available.")
    
    # Validate GPU IDs if provided
    if gpu_ids is not None:
        if device != "cuda" and device != "auto":
            print(f"Warning: GPU IDs specified but device is '{device}'. GPU IDs will be ignored.")
            gpu_ids = None
        else:
            print(f"GPU assignment: {len(gpu_ids)} GPU(s) available: {gpu_ids}")
            # Automatically enable parallel execution if GPUs are available
            if not parallel and len(gpu_ids) > 0:
                parallel = True
                print(f"Auto-enabled parallel execution with {len(gpu_ids)} GPU(s)")
    
    if parallel and (gpu_ids is None or len(gpu_ids) == 0):
        print("Warning: --parallel specified but no GPUs available. Running sequentially.")
        parallel = False
    
    if parallel:
        print(f"Parallel execution: Enabled (max {len(gpu_ids)} concurrent experiments)")
    else:
        print(f"Parallel execution: Disabled (sequential execution)")
    
    # Ask for confirmation
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    results = []
    experiment_num = 0
    
    # Prepare separate lists for joint and non-joint experiments
    joint_experiments = []
    non_joint_experiments = []
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        sample_size = config['sample_size']
        
        # Extract production-optimized training parameters from config
        training_params = {
            'episodes': config.get('episodes'),
            'steps_per_episode': config.get('steps_per_episode'),
            'batch_size': config.get('batch_size'),
            'n_perturb': config.get('n_perturb'),
            'n_envs': config.get('n_envs'),  # Note: ignored for TD3/DDPG, but kept for documentation
        }
        
        # Add joint training experiment
        if not skip_joint:
            joint_experiments.append({
                'dataset': dataset,
                'joint': True,
                'sample_size': sample_size,
                'training_params': training_params
            })
        
        # Add non-joint training experiment
        if not skip_non_joint:
            non_joint_experiments.append({
                'dataset': dataset,
                'joint': False,
                'sample_size': sample_size,
                'training_params': training_params
            })
    
    # Assign GPU IDs to experiments
    def assign_gpu_ids(experiments_list, gpu_list):
        """Assign GPU IDs to experiments, cycling through available GPUs."""
        if gpu_list is not None and len(gpu_list) > 0 and device in ["cuda", "auto"]:
            for i, exp in enumerate(experiments_list):
                exp['gpu_id'] = gpu_list[i % len(gpu_list)]
        else:
            for exp in experiments_list:
                exp['gpu_id'] = None
    
    assign_gpu_ids(joint_experiments, gpu_ids)
    assign_gpu_ids(non_joint_experiments, gpu_ids)
    
    # Run experiments in two phases: joint first, then non-joint
    # Each phase runs in parallel across available GPUs
    
    def run_experiment_batch(experiments_list, phase_name, phase_num, total_phases, start_num):
        """Run a batch of experiments (joint or non-joint) in parallel."""
        if not experiments_list:
            return []
        
        batch_results = []
        current_exp_num = start_num
        
        if parallel and gpu_ids and len(gpu_ids) > 0:
            # Parallel execution using multiprocessing
            import multiprocessing
            from functools import partial
            
            def run_experiment_wrapper(exp_dict, classifier_type, device):
                """Wrapper function for parallel execution."""
                try:
                    return run_experiment(
                        dataset=exp_dict['dataset'],
                        joint=exp_dict['joint'],
                        sample_size=exp_dict['sample_size'],
                        classifier_type=classifier_type,
                        device=device,
                        gpu_id=exp_dict['gpu_id'],
                        **exp_dict['training_params']
                    )
                except Exception as e:
                    print(f"Error in parallel experiment {exp_dict['dataset']} ({'joint' if exp_dict['joint'] else 'non-joint'}): {e}")
                    return False
            
            # Create pool of workers (limited by number of GPUs or experiments)
            n_workers = min(len(gpu_ids), len(experiments_list))
            print(f"\n{'='*80}")
            print(f"Phase {phase_num}/{total_phases}: Running {len(experiments_list)} {phase_name} experiments in parallel on {n_workers} GPU(s)...")
            print(f"{'='*80}")
            print("Note: Output from parallel experiments may be interleaved.\n")
            
            with multiprocessing.Pool(processes=n_workers) as pool:
                # Create partial function with fixed arguments
                run_func = partial(run_experiment_wrapper, classifier_type=classifier_type, device=device)
                # Run experiments in parallel
                success_list = pool.map(run_func, experiments_list)
            
            # Collect results
            for i, exp in enumerate(experiments_list):
                mode_str = "joint" if exp['joint'] else "non-joint"
                batch_results.append({
                    'dataset': exp['dataset'],
                    'mode': mode_str,
                    'gpu_id': exp['gpu_id'],
                    'success': success_list[i]
                })
        else:
            # Sequential execution
            for exp in experiments_list:
                current_exp_num += 1
                mode_str = "JOINT" if exp['joint'] else "NON-JOINT"
                gpu_info = f" (GPU {exp['gpu_id']})" if exp['gpu_id'] is not None else ""
                print(f"\n[{current_exp_num}/{total_experiments}] ", end="")
                success = run_experiment(
                    dataset=exp['dataset'],
                    joint=exp['joint'],
                    sample_size=exp['sample_size'],
                    classifier_type=classifier_type,
                    device=device,
                    gpu_id=exp['gpu_id'],
                    **exp['training_params']
                )
                batch_results.append({
                    'dataset': exp['dataset'],
                    'mode': mode_str.lower(),
                    'gpu_id': exp['gpu_id'],
                    'success': success
                })
        
        return batch_results
    
    # Phase 1: Run joint experiments first (in parallel across GPUs)
    if joint_experiments:
        phase_results = run_experiment_batch(
            joint_experiments, 
            "JOINT", 
            phase_num=1, 
            total_phases=(2 if non_joint_experiments else 1),
            start_num=experiment_num
        )
        results.extend(phase_results)
        experiment_num += len(phase_results)
        print(f"\n✓ Completed {len(phase_results)} joint experiment(s)")
    
    # Phase 2: Run non-joint experiments after joint completes (in parallel across GPUs)
    if non_joint_experiments:
        phase_results = run_experiment_batch(
            non_joint_experiments,
            "NON-JOINT",
            phase_num=2 if joint_experiments else 1,
            total_phases=(2 if joint_experiments else 1),
            start_num=experiment_num
        )
        results.extend(phase_results)
        experiment_num += len(phase_results)
        print(f"\n✓ Completed {len(phase_results)} non-joint experiment(s)")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(results)}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successful experiments:")
        for r in successful:
            gpu_info = f" [GPU {r.get('gpu_id')}]" if r.get('gpu_id') is not None else ""
            print(f"  - {r['dataset']} ({r['mode']}){gpu_info}")
    
    if failed:
        print("\n✗ Failed experiments:")
        for r in failed:
            gpu_info = f" [GPU {r.get('gpu_id')}]" if r.get('gpu_id') is not None else ""
            print(f"  - {r['dataset']} ({r['mode']}){gpu_info}")
    
    print("="*80)
    
    # Return exit code based on results
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated script to run sklearn_datasets_anchors.py with all datasets using TD3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all datasets in both modes
  python -m trainers.auto_run_td3_experiments
  
  # Run specific datasets
  python -m trainers.auto_run_td3_experiments --datasets breast_cancer wine
  
  # Skip joint training (only run non-joint)
  python -m trainers.auto_run_td3_experiments --skip-joint
  
  # Skip non-joint training (only run joint)
  python -m trainers.auto_run_td3_experiments --skip-non-joint
  
  # Use different classifier
  python -m trainers.auto_run_td3_experiments --classifier-type random_forest
  
  # Force CPU usage
  python -m trainers.auto_run_td3_experiments --device cpu
  
  # Force CUDA (if available)
  python -m trainers.auto_run_td3_experiments --device cuda
  
  # Run each experiment on a different GPU (sequential)
  python -m trainers.auto_run_td3_experiments --gpu-ids 0 1 2 3
  
  # Run experiments in parallel on different GPUs
  python -m trainers.auto_run_td3_experiments --gpu-ids 0 1 2 3 --parallel
  
  # Use specific GPUs for specific experiments (will cycle if more experiments than GPUs)
  python -m trainers.auto_run_td3_experiments --datasets breast_cancer wine --gpu-ids 0 1
        """
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to run (default: all datasets)"
    )
    parser.add_argument(
        "--skip-joint",
        action="store_true",
        help="Skip joint training experiments"
    )
    parser.add_argument(
        "--skip-non-joint",
        action="store_true",
        help="Skip non-joint training experiments"
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="dnn",
        choices=["dnn", "random_forest", "gradient_boosting"],
        help="Classifier type to use (default: dnn)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use: 'auto' (default, auto-detect), 'cpu', 'cuda', or 'mps'"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use for experiments (e.g., --gpu-ids 0 1 2 3). "
             "Each experiment will be assigned a GPU in round-robin fashion. "
             "Sets CUDA_VISIBLE_DEVICES for each subprocess."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel on different GPUs (requires --gpu-ids). "
             "If not specified, experiments run sequentially."
    )
    
    args = parser.parse_args()
    main(
        datasets=args.datasets,
        skip_joint=args.skip_joint,
        skip_non_joint=args.skip_non_joint,
        classifier_type=args.classifier_type,
        device=args.device,
        gpu_ids=args.gpu_ids,
        parallel=args.parallel
    )

