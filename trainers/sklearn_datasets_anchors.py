"""
Complete example: sklearn Datasets with Dynamic Anchors

This script demonstrates the full pipeline:
1. Load dataset (Breast Cancer or Covtype)
2. Train a classifier
3. Train PPO policy to find anchors
4. Evaluate anchors on test instances

Usage:
    python -m trainers.sklearn_datasets_anchors --dataset breast_cancer
    python -m trainers.sklearn_datasets_anchors --dataset covtype
    python -m trainers.sklearn_datasets_anchors --dataset breast_cancer --continuous-actions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
from sklearn.datasets import load_breast_cancer, fetch_covtype
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


def load_dataset(dataset_name: str, sample_size: int = None, seed: int = 42):
    """
    Load a sklearn dataset.
    
    Args:
        dataset_name: Name of dataset ("breast_cancer" or "covtype")
        sample_size: Optional size to sample (for large datasets like covtype)
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
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'breast_cancer' or 'covtype'.")
    
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
    patience=10
):
    """
    Train a SimpleClassifier on tabular data.
    
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
        
    Returns:
        Trained classifier and test accuracy
    """
    print("\n" + "="*80)
    print("Training Classifier")
    print("="*80)
    
    classifier = SimpleClassifier(n_features, n_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_test_acc = 0.0
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
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluate on test set
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(torch.from_numpy(X_test).float().to(device))
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, test_preds)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | Test Acc: {test_acc:.3f}")
        
        if patience_counter >= patience and epoch >= 50:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Classifier training complete. Best test accuracy: {best_test_acc:.3f}")
    print("="*80)
    
    return classifier, best_test_acc


def main(dataset_name: str = "breast_cancer", sample_size: int = None, joint: bool = True, use_continuous_actions: bool = False):
    """
    Main function: Complete pipeline for sklearn datasets.
    
    Args:
        dataset_name: Dataset to use ("breast_cancer" or "covtype")
        sample_size: Optional size to sample (for covtype, use 10000-50000)
        joint: If True, use joint training (alternating classifier and RL)
        use_continuous_actions: If True, use continuous actions version (from POC)
    """
    print("\n" + "="*80)
    print(f"Dataset: {dataset_name.upper().replace('_', ' ')} - Dynamic Anchors Pipeline")
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
    n_classes = len(class_names)
    
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
        
        # Note: Classifier is initialized inside joint training
        # For multi-class datasets, use all classes; for binary, use both
        if n_classes == 2:
            target_classes = (0, 1)  # Both classes
        else:
            target_classes = tuple(range(n_classes))  # All classes
        
        # Dataset-specific presets (adaptive based on dataset complexity)
        dataset_presets = {
            "breast_cancer": {
                "episodes": 100,
                "steps_per_episode": 500,
                "classifier_epochs_per_round": 1,  # Very simple classifier, 1 epoch is enough
                "classifier_update_every": 10,  # Update every 2 episodes (more RL between updates)
                "n_envs": 4,
            },
            "covtype": {
                "episodes": 100,
                "steps_per_episode": 500,
                "classifier_epochs_per_round": 2,  # Larger dataset, 2 epochs per update
                "classifier_update_every": 5,  # Update every 5 episodes (many RL episodes between updates)
                "n_envs": 4,
            },
            "default": {
                "episodes": 40,
                "steps_per_episode": 60,
                "classifier_epochs_per_round": 2,  # Default: 2 epochs per update
                "classifier_update_every": 3,  # Default: update every 3 episodes
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
            print(f"\n[Continuous Actions] Using DDPG (Stable Baselines 3)")
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
            classifier_epochs_per_round=classifier_epochs_per_round,
            classifier_update_every=classifier_update_every,
            classifier_lr=1e-3,
            classifier_batch_size=256,
            classifier_patience=5,
            use_continuous_actions=use_continuous_actions,  # Enable continuous actions if requested
            n_envs=n_envs if not use_continuous_actions else 1,  # DDPG doesn't use vectorized envs the same way
            learning_rate=3e-4,
            n_steps=steps_per_episode,
            batch_size=48,
            n_epochs=10,
            use_perturbation=True,
            perturbation_mode="bootstrap",
            n_perturb=2048,
            step_fracs=(0.005, 0.01, 0.02),
            min_width=0.05,
            precision_target=0.95,  # Match POC breast_cancer preset (easier than 0.98)
            coverage_target=0.5,  # 50% coverage - very high threshold, requires large boxes
            n_eval_instances_per_class=20,
            max_features_in_rule=5,
            output_dir=f"./{dataset_name}_joint_output/",
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
            n_classes=n_classes,
            device=device,
            epochs=100,
            batch_size=256,
            lr=1e-3,
            patience=10
        )
        
        print(f"\nClassifier trained successfully!")
        print(f"Test accuracy: {test_acc:.3f}")
        
        # ======================================================================
        # STEP 3: Train Dynamic Anchors PPO Policy
        # ======================================================================
        print("\n" + "="*80)
        print("STEP 3: Training Dynamic Anchors with PPO")
        print("="*80)
    
        # Note: We pass raw unscaled data - tabular_dynAnchors will standardize
        # For multi-class datasets, use all classes; for binary, use both
        if n_classes == 2:
            target_classes = (0, 1)  # Both classes
        else:
            target_classes = tuple(range(n_classes))  # All classes

        # Training configuration using episodes and steps_per_episode convention
        # From POC: Breast Cancer uses 25 episodes × 40 steps
        # We'll use similar but with more steps per episode for better learning
        episodes = 600
        steps_per_episode = 90
        
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
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            use_perturbation=True,
            perturbation_mode="bootstrap",
            n_perturb=2048,
            step_fracs=(0.005, 0.01, 0.02),
            min_width=0.05,
            precision_target=0.98,
            coverage_target=0.5,  # 50% coverage - very high threshold, requires large boxes
            n_eval_instances_per_class=20,
            max_features_in_rule=5,
            steps_per_episode=eval_steps_per_episode,
            output_dir=f"./{dataset_name}_anchors_output/",
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
    print(f"  Average Precision: {results['overall_stats']['avg_precision']:.3f}")
    print(f"  Average Coverage:  {results['overall_stats']['avg_coverage']:.3f}")
    
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
    eval_results = results['eval_results']
    if 'per_class_results' in eval_results:
        for cls, class_results in eval_results['per_class_results'].items():
            cls_name = class_names[int(cls)]
            print(f"\n  Class {cls} ({cls_name}):")
            if 'avg_precision' in class_results:
                print(f"    Avg Precision: {class_results['avg_precision']:.3f}")
                print(f"    Avg Coverage:  {class_results['avg_coverage']:.3f}")
            if 'best_rule' in class_results:
                print(f"    Best Rule: {class_results['best_rule']}")
            
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
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dynamic anchors on sklearn datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m trainers.sklearn_datasets_anchors --dataset breast_cancer
  python -m trainers.sklearn_datasets_anchors --dataset covtype --sample_size 10000
  
Datasets:
  breast_cancer  - Binary classification (2 classes, 30 features, 569 samples)
  covtype        - Multi-class classification (7 classes, 54 features, 581k samples)
                   Use --sample_size to limit samples for faster execution
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "covtype"],
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
        help="Use continuous actions version (from POC) with GAE (default: False, uses discrete actions)"
    )
    
    args = parser.parse_args()
    main(dataset_name=args.dataset, sample_size=args.sample_size, joint=args.joint, use_continuous_actions=args.use_continuous_actions)

