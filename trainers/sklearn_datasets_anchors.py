"""
Complete example: sklearn Datasets with Dynamic Anchors

This script demonstrates the full pipeline:
1. Load dataset (Breast Cancer, Covtype, Wine, or Housing)
2. Train a classifier
3. Train RL policy (PPO for discrete actions, DDPG/TD3 for continuous actions) to find anchors
4. Evaluate anchors on test instances

Perturbation Modes:
- "bootstrap": Resample empirical points with replacement (requires points in box)
- "uniform": Generate uniform samples within box bounds (works even with 0 points)
- "adaptive": Use bootstrap when plenty of points, uniform when sparse (recommended)

Usage:
    python -m trainers.sklearn_datasets_anchors --dataset breast_cancer
    python -m trainers.sklearn_datasets_anchors --dataset covtype --sample_size 10000
    python -m trainers.sklearn_datasets_anchors --dataset breast_cancer --continuous-actions
    python -m trainers.sklearn_datasets_anchors --dataset wine
    python -m trainers.sklearn_datasets_anchors --dataset housing --sample_size 10000
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
try:
    from trainers.networks import SimpleClassifier
    from trainers.tabular_dynAnchors import train_and_evaluate_dynamic_anchors
except ImportError:
    # Add parent directory to path if running directly from trainers/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trainers.networks import SimpleClassifier
    from trainers.tabular_dynAnchors import train_and_evaluate_dynamic_anchors


class TeeLogger:
    """
    Logger that writes to both console and file simultaneously.
    Captures all print statements and saves them to a log file.
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def write(self, text):
        """Write to both console and file."""
        self.original_stdout.write(text)
        self.log_file.write(text)
        self.log_file.flush()  # Ensure immediate write
        
    def flush(self):
        """Flush both streams."""
        self.original_stdout.flush()
        self.log_file.flush()
        
    def close(self):
        """Close the log file and restore original stdout."""
        if self.log_file:
            self.log_file.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
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
        # This creates 4 classes: very_low, low, medium, high
        # Note: Prices are in hundreds of thousands of dollars (e.g., 1.5 = $150,000)
        quartiles = np.percentile(prices, [25, 50, 75])
        # np.digitize returns: 0 for <q1, 1 for q1-<q2, 2 for q2-<q3, 3 for >=q3
        # This gives us 4 bins (0,1,2,3) which is what we want!
        y = np.digitize(prices, quartiles).astype(int)  # Creates 0, 1, 2, 3 (4 classes)
        # No need to subtract 1 - np.digitize already gives us 0-3
        
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
    
    # Sample subset if requested (for faster execution on large datasets)
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
    Train a SimpleClassifier on tabular data with optimized training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_features: Number of input features
        n_classes: Number of classes
        device: Torch device
        epochs: Maximum epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        weight_decay: L2 regularization weight decay (default: 1e-4)
        use_lr_scheduler: Whether to use learning rate scheduling (default: True)
        
    Returns:
        Trained classifier and test accuracy
    """
    print("\n" + "="*80)
    print("Training Classifier (Optimized)")
    print("="*80)
    
    # Create optimized classifier with dropout and batch norm
    classifier = SimpleClassifier(n_features, n_classes, dropout_rate=0.3, use_batch_norm=True).to(device)
    
    # Use Adam optimizer with weight decay for L2 regularization
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler: reduce LR on plateau
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
    
    print(f"Training configuration:")
    print(f"  Architecture: Input({n_features}) -> 256 -> 256 -> 128 -> Output({n_classes})")
    print(f"  Batch normalization: Enabled")
    print(f"  Dropout: 0.3 (hidden), 0.15 (final)")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    if scheduler:
        print(f"  LR scheduler: ReduceLROnPlateau (factor=0.5, patience={patience//3})")
    print()
    
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
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate on test set
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(torch.from_numpy(X_test).float().to(device))
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, test_preds)
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(test_acc)
            current_lr = optimizer.param_groups[0]['lr']  # Update after scheduler step
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = classifier.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            lr_str = f", LR: {current_lr:.2e}" if scheduler else ""
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Test Acc: {test_acc:.3f}{lr_str}")
        
        if patience_counter >= patience and epoch >= 50:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        classifier.load_state_dict(best_model_state)
    
    print(f"\nClassifier training complete. Best test accuracy: {best_test_acc:.3f}")
    print("="*80)
    
    return classifier, best_test_acc


def get_output_directory(
    dataset_name: str,
    continuous_algorithm: str,
    joint: bool,
    timestamp: str = None
) -> str:
    """
    Generate consistent output directory name with timestamp.
    
    Args:
        dataset_name: Name of the dataset
        continuous_algorithm: "ddpg" or "td3" (or "ppo" for discrete)
        joint: Whether using joint training
        timestamp: Optional timestamp string (if None, generates one)
    
    Returns:
        Output directory path with timestamp
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine algorithm name
    if continuous_algorithm.lower() in ["ddpg", "td3"]:
        algo_name = continuous_algorithm.lower()
    else:
        algo_name = "ppo"
    
    # Consistent naming: {dataset}_{algorithm}_{training_mode}_{timestamp}
    training_mode = "joint" if joint else "post_hoc"
    output_dir = f"./output/{dataset_name}_{algo_name}_{training_mode}_{timestamp}/"
    
    return output_dir


def main(dataset_name: str = "breast_cancer", sample_size: int = None, joint: bool = True, use_continuous_actions: bool = False, continuous_algorithm: str = "ddpg", classifier_type: str = "dnn"):
    """
    Main function: Complete pipeline for sklearn datasets.
    
    Args:
        dataset_name: Dataset to use ("breast_cancer", "covtype", "wine", or "housing")
        sample_size: Optional size to sample (for large datasets like covtype or housing)
        joint: If True, use joint training (alternating classifier and RL)
        use_continuous_actions: If True, use continuous actions (DDPG/TD3) instead of discrete (PPO)
        continuous_algorithm: "ddpg" or "td3" (only used if use_continuous_actions=True)
        classifier_type: "dnn", "random_forest", or "gradient_boosting"
    
    Note:
        Perturbation modes (set in training functions):
        - "bootstrap": Resample empirical points with replacement (requires points in box)
        - "uniform": Generate uniform samples within box bounds (works even with 0 points)
        - "adaptive": Use bootstrap when plenty of points, uniform when sparse (recommended)
    """
    # Generate timestamp and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_output_directory(
        dataset_name=dataset_name,
        continuous_algorithm=continuous_algorithm if use_continuous_actions else "ppo",
        joint=joint,
        timestamp=timestamp
    )
    
    # Setup logging to file
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.log")
    
    print(f"\n{'='*80}")
    print(f"Output Directory: {output_dir}")
    print(f"Logging enabled: Output will be saved to {log_file_path}")
    print(f"{'='*80}\n")
    
    # Use TeeLogger context manager to capture all output
    with TeeLogger(log_file_path) as logger:
        print("\n" + "="*80)
        print(f"Dataset: {dataset_name.upper().replace('_', ' ')} - Dynamic Anchors Pipeline")
        print("="*80)
        print(f"Log file: {log_file_path}")
        print("="*80)
        
        # Set random seeds for reproducibility
        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # ==========================================================================
        # STEP 1: Load Dataset
        # ==========================================================================
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
        
        # ==========================================================================
        # STEP 2: Set Device (Standardized - Set Once at Beginning)
        # ==========================================================================
        from trainers.device_utils import get_device_pair
        device, device_str = get_device_pair("auto")  # Can be changed to "cuda" or "mps" or "cpu"
        print(f"\nUsing device: {device} ({device_str})")
        
        n_features = X_train.shape[1]
        
        # Get actual unique classes from training data (after split, some classes might be missing)
        unique_classes_train = np.unique(y_train)
        unique_classes_test = np.unique(y_test)
        n_classes_train = len(unique_classes_train)
        n_classes_total = len(class_names)
        
        # Use classes that are actually present in training data
        # This ensures the classifier has the right number of output classes
        target_classes = tuple(unique_classes_train)
        
        if joint:
            # ======================================================================
            # JOINT TRAINING MODE: Alternate classifier and RL training
            # ======================================================================
            print("\n" + "="*80)
            if use_continuous_actions:
                print("MODE: JOINT TRAINING (Classifier + RL) - CONTINUOUS ACTIONS")
            else:
                print("MODE: JOINT TRAINING (Classifier + RL)")
            print("="*80)
            
            # Always use the stable-baselines3 trainer from trainers.tabular_dynAnchors_joint
            # It supports both discrete and continuous actions via use_continuous_actions parameter
            from trainers.tabular_dynAnchors_joint import train_and_evaluate_joint
            
            # Warn if some classes are missing from training data
            if len(unique_classes_train) < n_classes_total:
                missing_classes = set(range(n_classes_total)) - set(unique_classes_train)
                print(f"\n⚠ WARNING: Some classes are missing from training data after split:")
                print(f"  Expected classes: {list(range(n_classes_total))}")
                print(f"  Classes in training: {list(unique_classes_train)}")
                print(f"  Missing classes: {list(missing_classes)}")
                print(f"  Will only train on classes present in training data: {target_classes}")
            
            # Dataset-specific presets (adaptive based on dataset complexity)
            dataset_presets = {
                "breast_cancer": {
                    "episodes": 20,
                    "steps_per_episode": 1000,
                    "classifier_epochs_per_round": 1,  # Very simple classifier, 1 epoch is enough
                    "classifier_update_every": 2,  # Update every 2 episodes (more RL between updates)
                    "n_envs": 4,
                },
                "covtype": {
                    "episodes": 20,
                    "steps_per_episode": 1000,
                    "classifier_epochs_per_round": 1,  # Larger dataset, 2 epochs per update
                    "classifier_update_every": 2,  # Update every 5 episodes (many RL episodes between updates)
                    "n_envs": 4,
                },
                "wine": {
                    "episodes": 20,
                    "steps_per_episode": 1000,
                    "classifier_epochs_per_round": 1,  # Small dataset, 1 epoch is enough
                    "classifier_update_every": 2,  # Update every 8 episodes
                    "n_envs": 3,
                },
                "housing": {
                    "episodes": 20,
                    "steps_per_episode": 1000,
                    "classifier_epochs_per_round": 1,  # Larger dataset, 2 epochs per update
                    "classifier_update_every": 2,  # Update every 5 episodes
                    "n_envs": 4,
                },
                "default": {
                    "episodes": 20,
                    "steps_per_episode": 1000,
                    "classifier_epochs_per_round": 1,  # Default: 2 epochs per update
                    "classifier_update_every": 2,  # Default: update every 3 episodes
                    "n_envs": 2,
                }
            }
            
            # Select preset based on dataset
            preset = dataset_presets.get(dataset_name, dataset_presets["default"])
            
            episodes = preset["episodes"]
            steps_per_episode = preset["steps_per_episode"]
            classifier_epochs_per_round = preset["classifier_epochs_per_round"]
            classifier_update_every = preset["classifier_update_every"]
            n_envs = preset["n_envs"]
            
            print(f"\nJoint Training Configuration (dataset-specific presets for {dataset_name}):")
            print(f"  Episodes: {episodes}")
            print(f"  Steps per episode: {steps_per_episode}")
            print(f"  Classifier epochs per update: {classifier_epochs_per_round} (optimized for {dataset_name})")
            print(f"  Classifier update every: {classifier_update_every} episode(s) (allows {classifier_update_every-1} RL episodes between updates)")
            if not use_continuous_actions:
                print(f"  Parallel envs: {n_envs}")
                print(f"  Total RL timesteps: {episodes * steps_per_episode * n_envs}")
            print(f"  Classifier updates: {episodes // classifier_update_every} times × {classifier_epochs_per_round} epochs = {episodes // classifier_update_every * classifier_epochs_per_round} total epochs")
            
            # Use stable-baselines3 trainer (supports both discrete and continuous actions)
            if use_continuous_actions:
                algorithm_name = continuous_algorithm.upper() if continuous_algorithm.lower() == "td3" else "DDPG"
                print(f"\n[Continuous Actions] Using {algorithm_name} (Stable Baselines 3)")
            else:
                print(f"\n[Discrete Actions] Using PPO (Stable Baselines 3)")
            
            results = train_and_evaluate_joint(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                target_classes=target_classes,
                device=device,
                episodes=episodes,
                steps_per_episode=steps_per_episode,
                classifier_type=classifier_type,  # "dnn" or "random_forest"
                classifier_epochs_per_round=classifier_epochs_per_round,
                classifier_update_every=classifier_update_every,
                classifier_lr=1e-3,
                classifier_batch_size=256,
                classifier_patience=5,
                use_continuous_actions=use_continuous_actions,  # Enable continuous actions if requested
                continuous_algorithm=continuous_algorithm,  # "ddpg" or "td3"
                n_envs=n_envs if not use_continuous_actions else 1,  # DDPG doesn't use vectorized envs the same way
                learning_rate=3e-4,  # For PPO (discrete actions)
                continuous_learning_rate=5e-5,  # Lower LR for TD3/DDPG to reduce reward variance
                n_steps=steps_per_episode,
                batch_size=48,
                n_epochs=10,
                use_perturbation=True,
                perturbation_mode="adaptive",  # "bootstrap", "uniform", or "adaptive" (recommended: "adaptive")
                n_perturb=2048,
                step_fracs=(0.005, 0.01, 0.02),
                min_width=0.05,
                precision_target=0.95,  # High precision target
                coverage_target=0.01,  # Realistic target (1% coverage) - can be increased as agent learns
                n_eval_instances_per_class=50,
                max_features_in_rule=-1,  # Use -1 or None to include all tightened features (for feature importance)
                use_random_sampling=True,  # Enable random sampling to reduce variance (avoids deterministic cycling patterns)
                output_dir=output_dir,
                save_checkpoints=True,
                checkpoint_freq=2000,
                verbose=1,
            )
            
            classifier = results['classifier']
        
        else:
            # ======================================================================
            # STANDARD TRAINING MODE: Train classifier first, then RL
            # ======================================================================
            print("\n" + "="*80)
            print("MODE: STANDARD TRAINING (Classifier → RL)")
            print("="*80)
            
            # ======================================================================
            # STEP 2a: Train Classifier
            # ======================================================================
            classifier, test_acc = train_classifier(
                X_train, y_train, X_test, y_test,
                n_features=n_features,
                n_classes=n_classes_train,  # Use actual number of classes in training data
                device=device,
                epochs=2000,  # Sufficient epochs with better training
                batch_size=256,
                lr=1e-3,  # Fixed: was 1e-5 which is too low!
                patience=100,  # Reasonable patience with LR scheduling
                weight_decay=1e-4,  # L2 regularization
                use_lr_scheduler=True  # Enable learning rate scheduling
            )
            
            print(f"\nClassifier trained successfully!")
            print(f"Test accuracy: {test_acc:.3f}")
            
            # ======================================================================
            # STEP 3: Train Dynamic Anchors Policy (PPO for discrete, TD3/DDPG for continuous)
            # ======================================================================
            print("\n" + "="*80)
            if use_continuous_actions:
                algorithm_name = continuous_algorithm.upper() if continuous_algorithm.lower() == "td3" else "DDPG"
                print(f"STEP 3: Training Dynamic Anchors with {algorithm_name}")
            else:
                print("STEP 3: Training Dynamic Anchors with PPO")
            print("="*80)
        
            # Note: We pass raw unscaled data - tabular_dynAnchors will standardize
            # Use classes that are actually present in training data (already set above)
            # target_classes is already set from unique_classes_train

            # Training configuration using episodes and steps_per_episode convention
            # From POC: Breast Cancer uses 25 episodes × 40 steps
            # We'll use similar but with more steps per episode for better learning
            episodes = 150
            steps_per_episode = 1500
            
            # Calculate PPO parameters from episodes convention
            # total_timesteps = episodes × steps_per_episode × n_envs
            # n_steps should be approximately steps_per_episode or slightly less
            n_envs = 2
            total_timesteps = episodes * steps_per_episode * n_envs  # 25 × 40 × 2 = 2000
            n_steps = steps_per_episode  # Use same as steps_per_episode for alignment
            
            # Evaluation uses same steps_per_episode
            eval_steps_per_episode = steps_per_episode
            
            print(f"\nTraining Configuration:")
            print(f"  Episodes: {episodes}")
            print(f"  Steps per episode: {steps_per_episode}")
            print(f"  Parallel envs: {n_envs}")
            print(f"  Total timesteps: {total_timesteps}")
            print(f"  N steps per rollout: {n_steps}")
            
            results = train_and_evaluate_dynamic_anchors(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                classifier=classifier,
                target_classes=target_classes,
                device=device,
                n_envs=n_envs if not use_continuous_actions else 1,  # DDPG/TD3 don't use vectorized envs the same way
                total_timesteps=total_timesteps,
                learning_rate=3e-4,
                n_steps=n_steps,
                batch_size=256,
                n_epochs=10,
                use_continuous_actions=use_continuous_actions,  # Enable continuous actions if requested
                continuous_algorithm=continuous_algorithm,  # "ddpg" or "td3"
                continuous_learning_rate=5e-5,  # Lower LR for TD3/DDPG to reduce reward variance
                use_perturbation=True,
                perturbation_mode="adaptive",  # "bootstrap", "uniform", or "adaptive" (recommended: "adaptive")
                n_perturb=2048,
                step_fracs=(0.005, 0.01, 0.02),
                min_width=0.05,
                precision_target=0.98,
                coverage_target=0.5,  # 50% coverage - very high threshold, requires large boxes
                n_eval_instances_per_class=20,
                max_features_in_rule=-1,  # Use -1 or None to include all tightened features (for feature importance)
                steps_per_episode=eval_steps_per_episode,
                use_random_sampling=True,  # Enable random sampling to reduce variance (avoids deterministic cycling patterns)
                output_dir=output_dir,
                save_checkpoints=True,
                checkpoint_freq=2000,
                eval_freq=0,
                verbose=1,
            )
        
        # ==========================================================================
        # STEP 4: Display Results (same for both modes)
        # ==========================================================================
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nOverall Metrics:")
        # Handle both new structure (instance_level/class_level) and old structure (backward compatibility)
        overall_stats = results.get('overall_stats', {})
        if 'instance_level' in overall_stats:
            # New structure: separate instance-level and class-level
            instance_stats = overall_stats['instance_level']
            class_stats = overall_stats.get('class_level', {})
            
            print(f"\n[Instance-Level Evaluation] (One anchor per test instance, like static anchors):")
            print(f"  Average Precision: {instance_stats.get('avg_precision', 0.0):.3f}")
            print(f"  Average Coverage:  {instance_stats.get('avg_coverage', 0.0):.3f}")
            
            if class_stats:
                print(f"\n[Class-Level Evaluation] (One anchor per class, dynamic anchors advantage):")
                print(f"  Average Precision: {class_stats.get('avg_precision', 0.0):.3f}")
                print(f"  Average Coverage:  {class_stats.get('avg_coverage', 0.0):.3f}")
                
                # Show comparison
                instance_cov = instance_stats.get('avg_coverage', 0.0)
                class_cov = class_stats.get('avg_coverage', 0.0)
                if instance_cov > 0:
                    improvement = class_cov - instance_cov
                    improvement_pct = (improvement / instance_cov) * 100
                    print(f"\n[Coverage Comparison]:")
                    print(f"  Coverage Improvement: {improvement:+.3f} ({improvement_pct:+.1f}% increase)")
                    if improvement > 0:
                        print(f"    → Dynamic anchors (class-level) achieve higher coverage!")
        else:
            # Old structure: backward compatibility
            print(f"  Average Precision: {overall_stats.get('avg_precision', 0.0):.3f}")
            print(f"  Average Coverage:  {overall_stats.get('avg_coverage', 0.0):.3f}")
        
        # Show joint training history if available
        if 'joint_training_history' in results:
            print(f"\nJoint Training History:")
            for episode_info in results['joint_training_history']:
                episode_num = episode_info.get('episode', 'N/A')
                classifier_acc = episode_info.get('classifier_test_acc', 0.0)
                rl_timesteps = episode_info.get('rl_timesteps', 0)
                rl_precision = episode_info.get('rl_avg_precision')
                rl_coverage = episode_info.get('rl_avg_coverage')
                
                print(f"  Episode {episode_num}: "
                      f"Classifier Acc: {classifier_acc:.3f}, "
                      f"RL Timesteps: {rl_timesteps}", end="")
                if rl_precision is not None and rl_coverage is not None:
                    print(f", RL Precision: {rl_precision:.3f}, RL Coverage: {rl_coverage:.3f}")
                else:
                    print()
        
        print(f"\nPer-Class Results:")
        eval_results = results.get('eval_results', {})
        
        # Handle both new structure (instance_level/class_level) and old structure
        if 'instance_level' in eval_results:
            # New structure: separate instance-level and class-level
            instance_results = eval_results.get('instance_level', {})
            class_results = eval_results.get('class_level', {})
            
            # Show instance-level results
            if 'per_class_results' in instance_results:
                print(f"\n[Instance-Level Results] (One anchor per test instance):")
                for cls, class_data in instance_results['per_class_results'].items():
                    cls_int = int(cls.replace('class_', '')) if isinstance(cls, str) and cls.startswith('class_') else int(cls) if isinstance(cls, (int, str)) else cls
                    cls_name = class_names[cls_int] if cls_int < len(class_names) else f"Class {cls_int}"
                    print(f"\n  Class {cls_int} ({cls_name}):")
                    # Try multiple possible keys for precision and coverage
                    # Debug: print available keys if precision/coverage are 0
                    precision = class_data.get('hard_precision', class_data.get('precision', class_data.get('avg_hard_precision', class_data.get('avg_precision', 0.0))))
                    coverage = class_data.get('coverage', class_data.get('avg_coverage', class_data.get('avg_local_coverage', class_data.get('avg_global_coverage', 0.0))))
                    
                    # If precision/coverage are 0 but best_rule exists, try to get from individual_results
                    if (precision == 0.0 or coverage == 0.0) and 'best_rule' in class_data and class_data.get('best_rule'):
                        # Try to get from individual_results if available
                        if 'individual_results' in class_data and len(class_data['individual_results']) > 0:
                            # Get average from individual results
                            individual_precisions = [r.get('hard_precision', r.get('precision', 0.0)) for r in class_data['individual_results']]
                            individual_coverages = [r.get('coverage', r.get('local_coverage', r.get('global_coverage', 0.0))) for r in class_data['individual_results']]
                            if individual_precisions:
                                precision = np.mean(individual_precisions)
                            if individual_coverages:
                                coverage = np.mean(individual_coverages)
                        # Also try best_precision if available
                        if precision == 0.0 and 'best_precision' in class_data:
                            precision = class_data['best_precision']
                    
                    print(f"    Precision: {precision:.3f}")
                    print(f"    Coverage:  {coverage:.3f}")
                    if 'best_rule' in class_data:
                        print(f"    Best Rule: {class_data['best_rule'][:100]}...")
            
            # Show class-level results
            if 'per_class_results' in class_results:
                print(f"\n[Class-Level Results] (One anchor per class, dynamic anchors advantage):")
                for cls, class_data in class_results['per_class_results'].items():
                    cls_int = int(cls.replace('class_', '')) if isinstance(cls, str) and cls.startswith('class_') else int(cls) if isinstance(cls, (int, str)) else cls
                    cls_name = class_names[cls_int] if cls_int < len(class_names) else f"Class {cls_int}"
                    print(f"\n  Class {cls_int} ({cls_name}):")
                    
                    # Debug: Check what keys are available
                    available_keys = list(class_data.keys())
                    precision = class_data.get('hard_precision', class_data.get('precision', 0.0))
                    coverage = class_data.get('coverage', class_data.get('global_coverage', 0.0))
                    
                    # If precision/coverage are 0, check if values exist under different keys
                    if precision == 0.0 and coverage == 0.0:
                        # Debug output to help diagnose
                        print(f"    [DEBUG] Available keys: {available_keys}")
                        print(f"    [DEBUG] hard_precision: {class_data.get('hard_precision', 'NOT FOUND')}")
                        print(f"    [DEBUG] precision: {class_data.get('precision', 'NOT FOUND')}")
                        print(f"    [DEBUG] coverage: {class_data.get('coverage', 'NOT FOUND')}")
                        print(f"    [DEBUG] global_coverage: {class_data.get('global_coverage', 'NOT FOUND')}")
                    
                    print(f"    Precision: {precision:.3f}")
                    print(f"    Coverage:  {coverage:.3f}")
                    if 'rule' in class_data:
                        print(f"    Rule: {class_data['rule'][:100]}...")
        elif 'per_class_results' in eval_results:
            for cls, class_results in eval_results['per_class_results'].items():
                cls_int = int(cls) if isinstance(cls, (int, str)) else cls
                cls_name = class_names[cls_int] if cls_int < len(class_names) else f"Class {cls_int}"
                print(f"\n  Class {cls_int} ({cls_name}):")
                if 'avg_precision' in class_results:
                    print(f"    Avg Precision: {class_results['avg_precision']:.3f}")
                    print(f"    Avg Coverage:  {class_results['avg_coverage']:.3f}")
                elif 'precision' in class_results:
                    print(f"    Precision: {class_results['precision']:.3f}")
                    print(f"    Coverage:  {class_results['coverage']:.3f}")
                if 'best_rule' in class_results:
                    print(f"    Best Rule: {class_results['best_rule']}")
                
                # Try to create 2D visualization if not already created
                if 'anchors' in class_results and len(class_results.get('anchors', [])) > 0:
                    try:
                        from trainers.dynAnchors_inference import plot_rules_2d
                        # Only create if it doesn't exist
                        plot_path = f"{results.get('metadata', {}).get('output_dir', './output/')}/plots/rules_2d_visualization.png"
                        if not os.path.exists(plot_path):
                            print(f"\n  Creating 2D visualization...")
                            # Need X_test_scaled, X_min, X_range from results
                            # These might not be available, so skip if missing
                            if 'X_test_scaled' in results and 'X_min' in results and 'X_range' in results:
                                plot_path = plot_rules_2d(
                                    eval_results=eval_results,
                                    X_test=results['X_test_scaled'],
                                    y_test=y_test,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    output_path=plot_path,
                                    X_min=results['X_min'],
                                    X_range=results['X_range'],
                                )
                                print(f"    Saved: {plot_path}")
                    except Exception as e:
                        pass  # Silently skip if visualization fails
                
                # Show all rules with their individual coverage
                if 'individual_results' in class_results and len(class_results['individual_results']) > 0:
                    individual_results = class_results['individual_results']
                    print(f"\n    All Rules ({len(individual_results)} total):")
                    print(f"    {'Rule':<60} {'Precision':<12} {'Coverage':<12}")
                    print(f"    {'-'*60} {'-'*12} {'-'*12}")
                    
                    # Sort by coverage (descending) for better readability
                    sorted_results = sorted(individual_results, key=lambda x: x.get('coverage', 0.0), reverse=True)
                    
                    for i, result in enumerate(sorted_results, 1):
                        rule = result.get('rule', 'no rule')
                        precision = result.get('hard_precision', result.get('precision', 0.0))
                        coverage = result.get('coverage', 0.0)
                        # Truncate long rules
                        rule_display = rule[:57] + "..." if len(rule) > 60 else rule
                        print(f"    {i:2d}. {rule_display:<60} {precision:>10.3f}  {coverage:>10.3f}")
                    
                    # Calculate total unique coverage (union of all anchors)
                    # Note: Coverage is per-anchor (fraction of test data in that box)
                    # These coverages are NOT additive - anchors may overlap
                    # Total coverage would be 100% only if anchors collectively cover all test instances
                    total_coverage_sum = sum(r.get('coverage', 0.0) for r in individual_results)
                    avg_coverage = np.mean([r.get('coverage', 0.0) for r in individual_results])
                    print(f"\n    Coverage Statistics:")
                    print(f"      - Average per-anchor coverage: {avg_coverage:.3f}")
                    print(f"      - Sum of all coverages: {total_coverage_sum:.3f} (NOT additive - anchors may overlap)")
                    print(f"      - Note: Total unique coverage (union) would require checking which")
                    print(f"        test instances are covered by at least one anchor.")
                
                # Show unique rules if available (legacy code)
                if 'individual_results' in class_results and len(class_results['individual_results']) > 0:
                    from collections import Counter
                    all_rules = [r['rule'] for r in class_results['individual_results']]
                    rule_counts = Counter(all_rules)
                    n_unique = len(rule_counts)
                    
                    if n_unique > 1:
                        print(f"\n    Unique Rules: {n_unique} different rules found")
                        print(f"    Top Rules:")
                        for rule, count in rule_counts.most_common(3):
                            percentage = (count / len(all_rules)) * 100
                            print(f"      [{count}/{len(all_rules)} ({percentage:.1f}%)] {rule}")
    
        print("\n" + "="*80)
        print("Pipeline Complete!")
        print("="*80)
        
        # Add note about coverage interpretation
        if 'overall_stats' in results and 'all_individual_results' in results.get('overall_stats', {}):
            all_results = results['overall_stats'].get('all_individual_results', [])
            if all_results:
                print(f"\nNote on Coverage:")
                data_source = results.get('eval_results', {}).get('data_source', 'training')
                print(f"  - Coverage is per-anchor (fraction of {data_source} data in that anchor box)")
                print(f"  - Each anchor is evaluated independently on different test instances")
                print(f"  - By default, metrics are computed on TRAINING data (explains classifier behavior)")
                print(f"  - Set eval_on_test_data=True to compute metrics on test data")
                print(f"  - Total coverage is NOT necessarily 100% because:")
                print(f"    1. Anchors may overlap (same instance covered by multiple anchors)")
                print(f"    2. Some instances may not be covered by any anchor")
                print(f"    3. Coverage is calculated per-anchor, not as a union")
                if data_source == "test" and 'overall_union_coverage' in results.get('eval_results', {}):
                    union_cov = results['eval_results'].get('overall_union_coverage')
                    if union_cov is not None:
                        print(f"  - Union coverage (unique instances covered by at least one anchor): {union_cov:.3f}")
                else:
                    print(f"  - To get union coverage, use eval_on_test_data=True")
        
        print(f"\n{'='*80}")
        print(f"Log file saved to: {log_file_path}")
        print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dynamic anchors on sklearn datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trainers.sklearn_datasets_anchors --dataset breast_cancer
  python -m trainers.sklearn_datasets_anchors --dataset covtype --sample_size 10000
  python -m trainers.sklearn_datasets_anchors --dataset wine
  python -m trainers.sklearn_datasets_anchors --dataset housing --sample_size 10000
  
Datasets:
  breast_cancer  - Binary classification (2 classes, 30 features, 569 samples)
  covtype        - Multi-class classification (7 classes, 54 features, 581k samples)
                   Use --sample_size to limit samples for faster execution
  wine           - Multi-class classification (3 classes, 13 features, 178 samples)
  housing        - Multi-class classification (4 classes, 8 features, 20640 samples)
                   Converted from regression by binning prices into quartiles
                   Use --sample_size to limit samples for faster execution
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
        "--no-joint",
        dest="joint",
        action="store_false",
        default=True,
        help="Disable joint training mode (default: joint training enabled, use this flag to disable)"
    )
    parser.add_argument(
        "--continuous-actions",
        dest="use_continuous_actions",
        action="store_true",
        default=False,
        help="Use continuous actions version (DDPG or TD3) instead of discrete actions (PPO)"
    )
    parser.add_argument(
        "--continuous-algorithm",
        type=str,
        default="ddpg",
        choices=["ddpg", "td3"],
        help="Continuous action algorithm: 'ddpg' (default) or 'td3' (only used with --continuous-actions)"
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
        joint=args.joint,
        use_continuous_actions=args.use_continuous_actions,
        continuous_algorithm=args.continuous_algorithm,
        classifier_type=args.classifier_type
    )

