"""
Complete example: sklearn Datasets with Dynamic Anchors using Ray RLlib (SAC)

This script demonstrates the full pipeline:
1. Load dataset (Breast Cancer, Covtype, Wine, or Housing)
2. Train a classifier
3. Train RL policy (SAC for continuous actions) to find anchors using Ray RLlib
4. Evaluate anchors on test instances

Post-hoc training only: Train classifier first, then train RL policy.

Usage:
    python -m ray.sklearn_datasets_anchors_ray --dataset breast_cancer
    python -m ray.sklearn_datasets_anchors_ray --dataset covtype --sample_size 10000
    python -m ray.sklearn_datasets_anchors_ray --dataset wine
    python -m ray.sklearn_datasets_anchors_ray --dataset housing --sample_size 10000
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
from datetime import datetime
from sklearn.datasets import load_breast_cancer, fetch_covtype, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Handle imports when running as script vs module
# Add parent directory to path to handle both direct execution and module import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from standalone module (no dependency on trainers directory)
# Use relative import when running as script, absolute when running as module
try:
    # Try absolute import (when running as module: python -m ray.sklearn_datasets_anchors_ray)
    from ray.ray_modules_standalone import SimpleClassifier, get_device_pair
    from ray.tabular_dynAnchors_ray import train_and_evaluate_dynamic_anchors_ray
except ImportError:
    # Fallback to relative import (when running directly: python ray/sklearn_datasets_anchors_ray.py)
    from ray_modules_standalone import SimpleClassifier, get_device_pair
    from tabular_dynAnchors_ray import train_and_evaluate_dynamic_anchors_ray


class TeeLogger:
    """
    Logger that writes to both console and file simultaneously.
    Captures all print statements and saves them to a log file.
    Properly forwards attributes to the original stdout for compatibility.
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        """Write to both console and file."""
        self.original_stdout.write(text)
        # Check if file is still open before writing (handles atexit callbacks)
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except (ValueError, OSError):
                # File is closed or error occurred, ignore silently
                pass
        
    def flush(self):
        """Flush both streams."""
        self.original_stdout.flush()
        # Check if file is still open before flushing
        if self.log_file and not self.log_file.closed:
            try:
                self.log_file.flush()
            except (ValueError, OSError):
                # File is closed or error occurred, ignore silently
                pass
        
    def close(self):
        """Close the log file and restore original stdout."""
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def __getattr__(self, name):
        """Forward attribute access to original stdout for compatibility."""
        # Forward attributes like 'encoding', 'mode', 'name', etc. to original stdout
        return getattr(self.original_stdout, name)
        
    def __enter__(self):
        """Context manager entry."""
        sys.stdout = self
        sys.stderr = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_dataset(dataset_name: str, sample_size: int = None, seed: int = 42):
    """
    Load a sklearn dataset.
    
    Args:
        dataset_name: Name of dataset ("breast_cancer", "covtype", "wine", or "housing")
        sample_size: Optional size to sample (for large datasets like covtype or housing)
        seed: Random seed for sampling
        
    Returns:
        Tuple of (X, y, feature_names, class_names)
    """
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        feature_names = list(data.feature_names)
        class_names = list(data.target_names)
    elif dataset_name == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = y.astype(int) - 1  # Convert from 1-7 to 0-6
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"covertype_{i+1}" for i in range(7)]
    elif dataset_name == "wine":
        data = load_wine()
        X = data.data.astype(np.float32)
        y = data.target.astype(int)
        feature_names = list(data.feature_names)
        class_names = list(data.target_names)
    elif dataset_name == "housing":
        # California Housing dataset - convert regression to classification by binning prices
        data = fetch_california_housing()
        X = data.data.astype(np.float32)
        prices = data.target.astype(np.float32)
        
        # Convert regression target to classification by binning prices into quartiles
        quartiles = np.percentile(prices, [25, 50, 75])
        y = np.digitize(prices, quartiles).astype(int)  # Creates 0, 1, 2, 3 (4 classes)
        
        feature_names = list(data.feature_names)
        class_names = ["very_low_price", "low_price", "medium_price", "high_price"]
        
        print(f"\nConverted housing prices to 4 classes:")
        print(f"  Class 0 (very_low): < ${quartiles[0]*100:.0f}K (25th percentile)")
        print(f"  Class 1 (low): ${quartiles[0]*100:.0f}K - ${quartiles[1]*100:.0f}K (25th-50th percentile)")
        print(f"  Class 2 (medium): ${quartiles[1]*100:.0f}K - ${quartiles[2]*100:.0f}K (50th-75th percentile)")
        print(f"  Class 3 (high): >= ${quartiles[2]*100:.0f}K (75th percentile+)")
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: 'breast_cancer', 'covtype', 'wine', or 'housing'."
        )
    
    # Sample subset if requested
    if sample_size is not None and len(X) > sample_size:
        np.random.seed(seed)
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Sampling {sample_size} instances for faster execution")
    
    return X, y, feature_names, class_names


def train_classifier(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    n_features, 
    n_classes, 
    device, 
    epochs=100, 
    batch_size=256, 
    lr=1e-3, 
    patience=100,
    weight_decay=1e-4,
    use_lr_scheduler=True
):
    """
    Train a SimpleClassifier on tabular data.
    
    Returns:
        Trained classifier and test accuracy
    """
    print("\n" + "="*80)
    print("Training Classifier")
    print("="*80)
    
    classifier = SimpleClassifier(n_features, n_classes, dropout_rate=0.3, use_batch_norm=True).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//3, 
            min_lr=1e-6
        )
    
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_test_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate on test set
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(torch.from_numpy(X_test).float().to(device))
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, test_preds)
        
        if scheduler:
            scheduler.step(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = classifier.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            lr_str = f", LR: {current_lr:.2e}" if scheduler else ""
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Test Acc: {test_acc:.3f}{lr_str}")
        
        if patience_counter >= patience and epoch >= 50:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    print(f"\nClassifier training complete. Best test accuracy: {best_test_acc:.3f}")
    print("="*80)
    
    return classifier, best_test_acc


def get_output_directory(
    dataset_name: str,
    timestamp: str = None
) -> str:
    """Generate consistent output directory name with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = f"./output/{dataset_name}_sac_ray_{timestamp}/"
    return output_dir


def main(
    dataset_name: str = "breast_cancer", 
    sample_size: int = None, 
    classifier_type: str = "dnn"
):
    """
    Main function: Complete pipeline for sklearn datasets using Ray RLlib.
    
    Args:
        dataset_name: Dataset to use ("breast_cancer", "covtype", "wine", or "housing")
        sample_size: Optional size to sample (for large datasets)
        classifier_type: "dnn", "random_forest", or "gradient_boosting"
    """
    # Generate timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_directory(dataset_name=dataset_name, timestamp=timestamp)
    
    # Setup logging
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.log")
    
    print(f"\n{'='*80}")
    print(f"Output Directory: {output_dir}")
    print(f"Logging enabled: Output will be saved to {log_file_path}")
    print(f"{'='*80}\n")
    
    with TeeLogger(log_file_path) as logger:
        print("\n" + "="*80)
        print(f"Dataset: {dataset_name.upper().replace('_', ' ')} - Dynamic Anchors Pipeline (Ray RLlib SAC)")
        print("="*80)
        print(f"Log file: {log_file_path}")
        print("="*80)
        
        # Set random seeds
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # STEP 1: Load Dataset
        print("\n" + "="*80)
        print("STEP 1: Loading Dataset")
        print("="*80)
        
        X, y, feature_names, class_names = load_dataset(dataset_name, sample_size=sample_size, seed=seed)
        
        print(f"Dataset: {dataset_name.upper().replace('_', ' ')}")
        print(f"Shape: {X.shape}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Class names: {class_names}")
        print(f"Number of features: {len(feature_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # STEP 2: Set Device
        device, device_str = get_device_pair("auto")
        print(f"\nUsing device: {device} ({device_str})")
        
        n_features = X_train.shape[1]
        unique_classes_train = np.unique(y_train)
        n_classes_train = len(unique_classes_train)
        target_classes = tuple(unique_classes_train)
        
        # STEP 3: Train Classifier
        print("\n" + "="*80)
        print("STEP 2: Training Classifier")
        print("="*80)
        
        classifier, test_acc = train_classifier(
            X_train, y_train, X_test, y_test,
            n_features=n_features,
            n_classes=n_classes_train,
            device=device,
            epochs=2000,
            batch_size=256,
            lr=1e-3,
            patience=100,
            weight_decay=1e-4,
            use_lr_scheduler=True
        )
        
        print(f"\nClassifier trained successfully!")
        print(f"Test accuracy: {test_acc:.3f}")
        
        # STEP 4: Train Dynamic Anchors Policy with Ray RLlib SAC
        print("\n" + "="*80)
        print("STEP 3: Training Dynamic Anchors with Ray RLlib SAC")
        print("="*80)
        
        episodes = 150
        steps_per_episode = 1500
        total_timesteps = episodes * steps_per_episode
        
        print(f"\nTraining Configuration:")
        print(f"  Episodes: {episodes}")
        print(f"  Steps per episode: {steps_per_episode}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Algorithm: SAC (Ray RLlib)")
        
        results = train_and_evaluate_dynamic_anchors_ray(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            classifier=classifier,
            target_classes=target_classes,
            device=device_str,
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            use_perturbation=True,
            perturbation_mode="adaptive",
            n_perturb=2048,
            step_fracs=(0.005, 0.01, 0.02),
            min_width=0.05,
            precision_target=0.98,
            coverage_target=0.5,
            n_eval_instances_per_class=20,
            max_features_in_rule=-1,
            steps_per_episode=steps_per_episode,
            use_random_sampling=True,
            output_dir=output_dir,
            save_checkpoints=True,
            checkpoint_freq=2000,
            verbose=1,
            num_workers=0,  # Single process for now
            num_gpus=1 if device_str == "cuda" else 0,  # Use GPU if CUDA is available
            num_envs_per_env_runner=4 if device_str == "cuda" else 1,  # More parallel envs for GPU
            tau=0.005,
            target_network_update_freq=1,
            buffer_size=1000000,
            learning_starts=1000,
            train_batch_size=512 if device_str == "cuda" else 256,  # Larger batches for GPU
        )
        
        # STEP 5: Display Results
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nTraining completed successfully!")
        print(f"Models saved to: {output_dir}/models/")
        print(f"Checkpoints saved to: {output_dir}/checkpoints/")
        
        if "training_history" in results and len(results["training_history"]) > 0:
            print(f"\nTraining History:")
            print(f"  Total episodes: {len(results['training_history'])}")
            if len(results["training_history"]) > 0:
                last_episode = results["training_history"][-1]
                print(f"  Last episode reward: {last_episode.get('reward', 0.0):.3f}")
                print(f"  Last episode precision: {last_episode.get('precision', 0.0):.3f}")
                print(f"  Last episode coverage: {last_episode.get('coverage', 0.0):.3f}")
        
        print("\n" + "="*80)
        print("Pipeline Complete!")
        print("="*80)
        print(f"\n{'='*80}")
        print(f"Log file saved to: {log_file_path}")
        print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dynamic anchors on sklearn datasets using Ray RLlib SAC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ray.sklearn_datasets_anchors_ray --dataset breast_cancer
  python -m ray.sklearn_datasets_anchors_ray --dataset covtype --sample_size 10000
  python -m ray.sklearn_datasets_anchors_ray --dataset wine
  python -m ray.sklearn_datasets_anchors_ray --dataset housing --sample_size 10000
  
Datasets:
  breast_cancer  - Binary classification (2 classes, 30 features, 569 samples)
  covtype        - Multi-class classification (7 classes, 54 features, 581k samples)
  wine           - Multi-class classification (3 classes, 13 features, 178 samples)
  housing        - Multi-class classification (4 classes, 8 features, 20640 samples)
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "covtype", "wine", "housing"],
        help="Dataset to use (default: breast_cancer)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for large datasets (recommended: 10000-50000 for covtype)"
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default="dnn",
        choices=["dnn", "random_forest", "gradient_boosting"],
        help="Classifier type: 'dnn' (default), 'random_forest', or 'gradient_boosting'"
    )
    
    args = parser.parse_args()
    main(
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        classifier_type=args.classifier_type
    )

