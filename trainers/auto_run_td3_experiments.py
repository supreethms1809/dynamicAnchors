"""
Automated script to run sklearn_datasets_anchors.py with all datasets
using TD3 (continuous actions) in both joint and non-joint modes.

This script will:
1. Iterate through all available datasets
2. Run each dataset in both joint and non-joint training modes
3. Use TD3 algorithm for continuous actions
4. Handle appropriate sample sizes for large datasets
5. Log all runs with timestamps

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
from typing import List, Optional


# Dataset configurations with recommended sample sizes
DATASET_CONFIGS = {
    "breast_cancer": {
        "sample_size": None,  # Small dataset, use all
        "description": "Binary classification (2 classes, 30 features, 569 samples)"
    },
    "covtype": {
        "sample_size": 20000,  # Large dataset, sample for reasonable runtime (original: 581k samples)
        "description": "Multi-class classification (7 classes, 54 features, 581k samples)"
    },
    "wine": {
        "sample_size": None,  # Small dataset, use all
        "description": "Multi-class classification (3 classes, 13 features, 178 samples)"
    },
    "housing": {
        "sample_size": 10000,  # Large dataset, sample for reasonable runtime (original: 20k samples)
        "description": "Multi-class classification (4 classes, 8 features, 20640 samples)"
    }
}


def run_experiment(
    dataset: str,
    joint: bool,
    sample_size: Optional[int] = None,
    classifier_type: str = "dnn",
    device: str = "auto"
) -> bool:
    """
    Run a single experiment with specified parameters.
    
    Args:
        dataset: Dataset name
        joint: Whether to use joint training
        sample_size: Optional sample size for large datasets
        classifier_type: Classifier type ("dnn", "random_forest", or "gradient_boosting")
    
    Returns:
        True if successful, False otherwise
    """
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
    
    # Print header
    mode_str = "JOINT" if joint else "NON-JOINT"
    print("\n" + "="*80)
    print(f"Running: {dataset.upper()} - {mode_str} - TD3")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        print(f"\n✓ Successfully completed: {dataset} - {mode_str}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset} - {mode_str}")
        print(f"Error code: {e.returncode}")
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
    device: str = "auto"
):
    """
    Main function to run all experiments.
    
    Args:
        datasets: List of datasets to run (None = all datasets)
        skip_joint: Skip joint training experiments
        skip_non_joint: Skip non-joint training experiments
        classifier_type: Classifier type to use
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
    print("\nDataset Configurations:")
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        sample_str = f"sample_size={config['sample_size']}" if config['sample_size'] else "all samples"
        print(f"  {dataset:15s}: {sample_str:20s} - {config['description']}")
    
    # Calculate total number of experiments
    n_modes = (0 if skip_joint else 1) + (0 if skip_non_joint else 1)
    total_experiments = len(datasets) * n_modes
    print(f"\nTotal experiments to run: {total_experiments}")
    
    # Ask for confirmation
    response = input("\nProceed? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    results = []
    experiment_num = 0
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        sample_size = config['sample_size']
        
        # Run joint training
        if not skip_joint:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] ", end="")
            success = run_experiment(
                dataset=dataset,
                joint=True,
                sample_size=sample_size,
                classifier_type=classifier_type,
                device=device
            )
            results.append({
                'dataset': dataset,
                'mode': 'joint',
                'success': success
            })
        
        # Run non-joint training
        if not skip_non_joint:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] ", end="")
            success = run_experiment(
                dataset=dataset,
                joint=False,
                sample_size=sample_size,
                classifier_type=classifier_type,
                device=device
            )
            results.append({
                'dataset': dataset,
                'mode': 'non-joint',
                'success': success
            })
    
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
            print(f"  - {r['dataset']} ({r['mode']})")
    
    if failed:
        print("\n✗ Failed experiments:")
        for r in failed:
            print(f"  - {r['dataset']} ({r['mode']})")
    
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
    
    args = parser.parse_args()
    main(
        datasets=args.datasets,
        skip_joint=args.skip_joint,
        skip_non_joint=args.skip_non_joint,
        classifier_type=args.classifier_type,
        device=args.device
    )

