"""
Inference script for extracting anchor rules from trained BenchMARL models.

This script:
1. Loads a trained BenchMARL checkpoint
2. Runs rollouts on instances to extract anchors
3. Extracts rules from the anchors
4. Saves evaluation data and rules
"""

from tabular_datasets import TabularDatasetLoader
from anchor_trainer import AnchorTrainer
from environment import AnchorEnv
import argparse
import os
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import json


def extract_rules_from_checkpoint(
    checkpoint_path: str,
    dataset_name: str,
    algorithm: str = "maddpg",
    algorithm_config_path: Optional[str] = None,
    experiment_config_path: str = "conf/base_experiment.yaml",
    mlp_config_path: str = "conf/mlp.yaml",
    max_features_in_rule: int = 5,
    steps_per_episode: int = 100,
    n_instances_per_class: int = 20,
    eval_on_test_data: bool = False,
    output_dir: Optional[str] = None,
    seed: int = 42,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extract anchor rules from a trained BenchMARL checkpoint.
    
    Args:
        checkpoint_path: Path to BenchMARL experiment folder or checkpoint file
        dataset_name: Name of the dataset
        algorithm: Algorithm name (must match training)
        algorithm_config_path: Path to algorithm config (default: conf/{algorithm}.yaml)
        experiment_config_path: Path to experiment config
        mlp_config_path: Path to MLP config
        max_features_in_rule: Maximum features to include in rules
        steps_per_episode: Maximum steps per rollout
        n_instances_per_class: Number of instances to evaluate per class
        eval_on_test_data: Whether to evaluate on test data
        output_dir: Output directory for results (default: checkpoint_path/inference/)
        seed: Random seed
        device: Device to use
    
    Returns:
        Dictionary containing extracted rules and evaluation data
    """
    print("="*80)
    print("ANCHOR RULE EXTRACTION FROM TRAINED MODEL")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Algorithm: {algorithm}")
    print("="*80)
    
    # Load dataset
    dataset_loader = TabularDatasetLoader(
        dataset_name=dataset_name,
        test_size=0.2,
        random_state=seed
    )
    
    dataset_loader.load_dataset()
    dataset_loader.preprocess_data()
    
    # Try to load classifier from checkpoint directory
    classifier_path = os.path.join(checkpoint_path, "classifier.pth")
    classifier_loaded = False
    
    if os.path.exists(classifier_path):
        print(f"\nLoading classifier from: {classifier_path}")
        try:
            classifier = dataset_loader.load_classifier(
                filepath=classifier_path,
                classifier_type="dnn",  # Default to DNN, can be made configurable
                device=device
            )
            dataset_loader.classifier = classifier
            classifier_loaded = True
            print(f"Classifier loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load classifier from {classifier_path}: {e}")
            print("  Will train a new classifier...")
    
    # If classifier not loaded, train a new one
    if not classifier_loaded:
        print(f"\nTraining new classifier (not found in checkpoint)...")
        classifier = dataset_loader.create_classifier(
            classifier_type="dnn",
            dropout_rate=0.3,
            use_batch_norm=True,
            device=device
        )
        
        trained_classifier, test_acc, history = dataset_loader.train_classifier(
            classifier,
            epochs=100,
            batch_size=256,
            lr=1e-3,
            patience=10,
            device=device
        )
        
        print(f"Classifier trained with test accuracy: {test_acc:.4f}")
        
        # Save it for future use
        os.makedirs(checkpoint_path, exist_ok=True)
        dataset_loader.save_classifier(trained_classifier, classifier_path)
        print(f"Classifier saved to: {classifier_path}")
    
    if dataset_loader.classifier is None:
        raise ValueError("Failed to load or train classifier")
    
    print(f"\nClassifier ready (test accuracy: {getattr(dataset_loader, 'classifier_test_accuracy', 'N/A')})")
    
    # Create trainer (just for setup, we'll load checkpoint)
    trainer = AnchorTrainer(
        dataset_loader=dataset_loader,
        algorithm=algorithm,
        algorithm_config_path=algorithm_config_path,
        experiment_config_path=experiment_config_path,
        mlp_config_path=mlp_config_path,
        output_dir=output_dir or os.path.join(checkpoint_path, "inference"),
        seed=seed
    )
    
    # Find BenchMARL checkpoint file in the experiment folder
    # Reference: https://benchmarl.readthedocs.io/en/latest/concepts/features.html#reloading
    checkpoint_file = None
    if os.path.isdir(checkpoint_path):
        # Check for checkpoints in the checkpoints subdirectory (BenchMARL standard location)
        checkpoints_dir = os.path.join(checkpoint_path, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoint_files = [
                f for f in os.listdir(checkpoints_dir)
                if (f.endswith('.pt') or f.endswith('.pth'))
            ]
            if checkpoint_files:
                # Use the most recent checkpoint
                checkpoint_file = os.path.join(
                    checkpoints_dir,
                    max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
                )
        
        # If not found in checkpoints subdirectory, check root directory
        if not checkpoint_file:
            all_files = os.listdir(checkpoint_path)
            checkpoint_files = [
                f for f in all_files 
                if (f.endswith('.pt') or f.endswith('.pth')) 
                and f != 'classifier.pth'
                and not f.startswith('classifier')
            ]
            
            # Also check for BenchMARL's standard checkpoint naming
            possible_names = ['checkpoint.pt', 'checkpoint.pth', 'model.pt', 'model.pth']
            for name in possible_names:
                if os.path.exists(os.path.join(checkpoint_path, name)):
                    checkpoint_file = os.path.join(checkpoint_path, name)
                    break
            
            if not checkpoint_file and checkpoint_files:
                # Use most recent checkpoint file
                checkpoint_file = os.path.join(
                    checkpoint_path, 
                    max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f)))
                )
    elif os.path.isfile(checkpoint_path):
        checkpoint_file = checkpoint_path
    
    # Method 1: Try using BenchMARL's official reload_experiment_from_file() (recommended)
    # Reference: https://benchmarl.readthedocs.io/en/latest/concepts/features.html#reloading
    experiment_loaded = False
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            from benchmarl.hydra_config import reload_experiment_from_file
            
            print(f"\n{'='*80}")
            print("METHOD 1: Using BenchMARL's reload_experiment_from_file()")
            print(f"{'='*80}")
            print(f"Checkpoint: {checkpoint_file}")
            print("  Reference: https://benchmarl.readthedocs.io/en/latest/concepts/features.html#reloading")
            
            # Reload experiment from checkpoint
            # This automatically restores the experiment state
            experiment = reload_experiment_from_file(checkpoint_file)
            
            # Assign the reloaded experiment to the trainer
            trainer.experiment = experiment
            
            # Find the callback if it exists in the experiment
            if hasattr(experiment, 'callbacks') and experiment.callbacks:
                for cb in experiment.callbacks:
                    if hasattr(cb, 'get_evaluation_anchor_data'):
                        trainer.callback = cb
                        break
            
            print("  ✓ Experiment reloaded successfully using BenchMARL's official method")
            experiment_loaded = True
            
        except ImportError:
            print(f"\n{'='*80}")
            print("METHOD 1: BenchMARL's reload_experiment_from_file() not available")
            print(f"{'='*80}")
            print("  Falling back to restore_file mechanism...")
        except Exception as e:
            print(f"\n{'='*80}")
            print("METHOD 1: Failed to use reload_experiment_from_file()")
            print(f"{'='*80}")
            print(f"  Error: {e}")
            print("  Falling back to restore_file mechanism...")
    
    # Method 2: Fallback to restore_file mechanism (if Method 1 failed)
    if not experiment_loaded:
        restore_file = checkpoint_file if checkpoint_file and os.path.exists(checkpoint_file) else None
        
        if restore_file:
            print(f"\n{'='*80}")
            print("METHOD 2: Using BenchMARL restore_file mechanism")
            print(f"{'='*80}")
            print(f"Checkpoint: {restore_file}")
            
            # Update experiment config to use restore_file before setup
            # We need to modify the config file before setup_experiment loads it
            import yaml
            import tempfile
            
            # Load the original config
            original_config_path = experiment_config_path
            with open(original_config_path, 'r') as f:
                exp_config_dict = yaml.safe_load(f)
            
            # Set restore_file
            exp_config_dict['restore_file'] = restore_file
            
            # Create a temporary config file with restore_file set
            temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(exp_config_dict, temp_config_path)
            temp_config_path.close()
            
            # Update trainer's config path to use the temporary config
            trainer.experiment_config_path = temp_config_path.name
            
            # Setup experiment (will restore from checkpoint via restore_file)
            trainer.setup_experiment(
                env_config=None,
                target_classes=None,  # Will use all classes from dataset
                max_cycles=steps_per_episode,
                device=device,
                eval_on_test_data=eval_on_test_data
            )
            print("  ✓ Experiment restored using restore_file mechanism")
            experiment_loaded = True
        else:
            # Method 3: Setup normally and try manual loading
            print(f"\n{'='*80}")
            print("METHOD 3: Manual checkpoint loading")
            print(f"{'='*80}")
            
            # Setup experiment normally
            trainer.setup_experiment(
                env_config=None,
                target_classes=None,  # Will use all classes from dataset
                max_cycles=steps_per_episode,
                device=device,
                eval_on_test_data=eval_on_test_data
            )
            
            # Try manual loading
            print(f"Attempting to load checkpoint manually from: {checkpoint_path}")
            try:
                trainer.load_checkpoint(checkpoint_path)
                print("  ✓ Checkpoint loaded manually")
                experiment_loaded = True
            except Exception as e:
                print(f"  ✗ Could not load checkpoint: {e}")
                print("  The model may not have been checkpointed.")
                print("  For future runs, set checkpoint_at_end: True in base_experiment.yaml")
                raise ValueError(
                    "No checkpoint found. Please retrain with checkpoint_at_end: True enabled, "
                    "or provide a valid checkpoint file path."
                )
    
    if not experiment_loaded:
        raise RuntimeError("Failed to load experiment from checkpoint using all available methods")
    
    print(f"{'='*80}\n")
    
    # Get environment data
    env_data = dataset_loader.get_anchor_env_data()
    target_classes = list(np.unique(dataset_loader.y_train))
    
    print(f"\nExtracting rules for classes: {target_classes}")
    print(f"  Instances per class: {n_instances_per_class}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Max features in rule: {max_features_in_rule}")
    
    # Extract rules using the trainer's method
    # This will use the loaded model to run rollouts
    results = trainer.extract_rules(
        max_features_in_rule=max_features_in_rule,
        steps_per_episode=steps_per_episode,
        n_instances_per_class=n_instances_per_class,
        eval_on_test_data=eval_on_test_data
    )
    
    # Add metadata
    results["metadata"]["checkpoint_path"] = checkpoint_path
    results["metadata"]["inference_timestamp"] = str(np.datetime64('now'))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract anchor rules from trained BenchMARL model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to BenchMARL experiment folder or checkpoint file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "wine", "iris", "synthetic", "moons", "circles", "covtype", "housing"],
        help="Dataset name (must match training)"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        default="maddpg",
        help="Algorithm name (must match training)"
    )
    
    parser.add_argument(
        "--algorithm_config",
        type=str,
        default=None,
        help="Path to algorithm config YAML (default: conf/{algorithm}.yaml)"
    )
    
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="conf/base_experiment.yaml",
        help="Path to experiment config YAML"
    )
    
    parser.add_argument(
        "--mlp_config",
        type=str,
        default="conf/mlp.yaml",
        help="Path to MLP model config YAML"
    )
    
    parser.add_argument(
        "--max_features_in_rule",
        type=int,
        default=5,
        help="Maximum number of features to include in extracted rules"
    )
    
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=100,
        help="Maximum steps per rollout episode"
    )
    
    parser.add_argument(
        "--n_instances_per_class",
        type=int,
        default=20,
        help="Number of instances to evaluate per class"
    )
    
    parser.add_argument(
        "--eval_on_test_data",
        action="store_true",
        help="Evaluate on test data instead of training data"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint_path/inference/)"
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
        choices=["cpu", "cuda"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Extract rules
    results = extract_rules_from_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        algorithm=args.algorithm,
        algorithm_config_path=args.algorithm_config,
        experiment_config_path=args.experiment_config,
        mlp_config_path=args.mlp_config,
        max_features_in_rule=args.max_features_in_rule,
        steps_per_episode=args.steps_per_episode,
        n_instances_per_class=args.n_instances_per_class,
        eval_on_test_data=args.eval_on_test_data,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device
    )
    
    # Save results
    output_dir = args.output_dir or os.path.join(args.checkpoint, "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a minimal trainer just for saving (doesn't need full setup)
    class MinimalTrainer:
        def __init__(self, output_dir):
            self.output_dir = output_dir
        
        def _convert_to_serializable(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (float, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: self._convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        def save_rules(self, results: Dict[str, Any], filepath: Optional[str] = None):
            if filepath is None:
                filepath = os.path.join(self.output_dir, "extracted_rules.json")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            serializable_results = self._convert_to_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            n_anchors_total = sum(
                len(class_data.get("anchors", []))
                for class_data in serializable_results.get("per_class_results", {}).values()
            )
            n_rules_total = sum(
                len(class_data.get("rules", []))
                for class_data in serializable_results.get("per_class_results", {}).values()
            )
            
            print(f"Rules and anchors saved to: {filepath}")
            print(f"  Total anchors saved: {n_anchors_total}")
            print(f"  Total rules saved: {n_rules_total}")
            return filepath
    
    minimal_trainer = MinimalTrainer(output_dir)
    rules_filepath = minimal_trainer.save_rules(results)
    print(f"\n{'='*80}")
    print(f"Rule extraction complete!")
    print(f"Results saved to: {rules_filepath}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

