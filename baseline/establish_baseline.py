"""
Baseline Explainability Methods for Comparison

This script implements baseline explainability methods:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. Static Anchors (using anchor-exp library)
3. SHAP (SHapley Additive exPlanations)
4. Feature Importance (Permutation Importance and Tree-based)

These baselines can be compared against dynamic anchors for the same datasets.

Usage:
    python -m baseline.establish_baseline --dataset breast_cancer
    python -m baseline.establish_baseline --dataset covtype --sample_size 10000
    python -m baseline.establish_baseline --dataset wine
    python -m baseline.establish_baseline --dataset housing --sample_size 10000
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import (
    load_breast_cancer, fetch_covtype, load_wine, load_iris, 
    fetch_california_housing, make_classification, make_moons, make_circles
)
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Handle optional imports for UCI and Folktables
try:
    from ucimlrepo import fetch_ucirepo
    UCIML_AVAILABLE = True
except ImportError:
    UCIML_AVAILABLE = False

try:
    from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage, ACSMobility, ACSEmployment, ACSTravelTime
    FOLKTABLES_AVAILABLE = True
except ImportError:
    FOLKTABLES_AVAILABLE = False

# Handle imports when running as script vs module
try:
    from utils.networks import SimpleClassifier
    from utils.device_utils import get_device_pair
except ImportError:
    # Add parent directory to path if running directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.networks import SimpleClassifier
    from utils.device_utils import get_device_pair


def load_dataset(dataset_name: str, sample_size: int = None, seed: int = 42):
    """
    Load a sklearn dataset (same function as in sklearn_datasets_anchors.py).
    
    Args:
        dataset_name: Name of dataset ("breast_cancer", "covtype", "wine", "iris", "housing", 
                      "uci_<id_or_name>" (e.g., "uci_adult", "uci_credit"), or 
                      "folktables_<task>_<state>_<year>" (e.g., "folktables_income_CA_2018"))
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
    elif dataset_name == "iris":
        data = load_iris()
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
    elif dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=seed
        )
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
    elif dataset_name == "moons":
        X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
    elif dataset_name.startswith("uci_"):
        # UCIML Repository dataset
        if not UCIML_AVAILABLE:
            raise ImportError(
                "ucimlrepo package is required for UCIML datasets. "
                "Install with: pip install ucimlrepo"
            )
        
        # Parse dataset identifier (can be ID or name)
        dataset_id_str = dataset_name.replace("uci_", "")
        
        # Common UCIML dataset IDs
        uci_dataset_map = {
            "adult": 2,
            "car": 19,
            "credit": 27,
            "nursery": 76,
            "mushroom": 73,
            "tic-tac-toe": 101,
            "vote": 56,
            "zoo": 111,
        }
        
        # Try to get ID from map or parse as integer
        try:
            if dataset_id_str in uci_dataset_map:
                dataset_id = uci_dataset_map[dataset_id_str]
            else:
                dataset_id = int(dataset_id_str)
        except ValueError:
            raise ValueError(
                f"Invalid UCIML dataset identifier: {dataset_id_str}. "
                f"Use format 'uci_<id>' or 'uci_<name>'. "
                f"Supported names: {list(uci_dataset_map.keys())}"
            )
        
        print(f"Fetching UCIML dataset (ID: {dataset_id})...")
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Extract features and targets
        X_df = dataset.data.features
        y_df = dataset.data.targets
        
        # Get feature names before processing
        if hasattr(X_df, 'columns'):
            feature_names = list(X_df.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_df.shape[1])]
        
        # Convert to DataFrame if not already
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=feature_names)
        
        # Handle missing values
        if X_df.isnull().any().any():
            missing_count = X_df.isnull().sum().sum()
            print(f"  Found {missing_count} missing values, filling with median/mode...")
            # Fill numeric columns with median, categorical with mode
            for col in X_df.columns:
                if X_df[col].dtype in ['int64', 'float64']:
                    X_df[col].fillna(X_df[col].median(), inplace=True)
                else:
                    mode_val = X_df[col].mode()
                    X_df[col].fillna(mode_val[0] if len(mode_val) > 0 else 0, inplace=True)
        
        # Encode categorical features
        label_encoders = {}
        for col in X_df.columns:
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                label_encoders[col] = le
        
        # Convert to numpy array
        X = X_df.values.astype(np.float32)
        
        # Handle target
        y = y_df.values if hasattr(y_df, 'values') else y_df
        
        # Handle target shape (may be 1D or 2D)
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.flatten()
            else:
                # Multi-label case - use first column
                print(f"  Warning: Multi-column target detected, using first column")
                y = y[:, 0]
        
        # Convert target to integer labels if needed
        if isinstance(y, pd.Series):
            y = y.values
        
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.integer):
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str)).astype(int)
            class_names = le.classes_.tolist()
        else:
            y = y.astype(int)
            unique_classes = np.unique(y)
            class_names = [f"class_{i}" for i in unique_classes]
        
        print(f"  Loaded UCIML dataset: {dataset.metadata.name if hasattr(dataset, 'metadata') else 'Unknown'}")
        print(f"  Features: {len(feature_names)}, Classes: {len(class_names)}")
        if label_encoders:
            print(f"  Encoded {len(label_encoders)} categorical features")
    
    elif dataset_name.startswith("folktables_"):
        # Folktables dataset
        if not FOLKTABLES_AVAILABLE:
            raise ImportError(
                "folktables package is required for Folktables datasets. "
                "Install with: pip install folktables"
            )
        
        # Parse dataset specification: folktables_<task>_<state>_<year>
        # Example: folktables_income_CA_2018
        parts = dataset_name.replace("folktables_", "").split("_")
        
        if len(parts) < 3:
            raise ValueError(
                f"Invalid Folktables dataset format: {dataset_name}. "
                f"Use format: folktables_<task>_<state>_<year>\n"
                f"Example: folktables_income_CA_2018\n"
                f"Available tasks: income, coverage, mobility, employment, travel"
            )
        
        task_name = parts[0].lower()
        state = parts[1].upper()
        year = parts[2]
        
        # Map task names to Folktables tasks
        task_map = {
            "income": ACSIncome,
            "coverage": ACSPublicCoverage,
            "mobility": ACSMobility,
            "employment": ACSEmployment,
            "travel": ACSTravelTime,
        }
        
        if task_name not in task_map:
            raise ValueError(
                f"Unknown Folktables task: {task_name}. "
                f"Available tasks: {list(task_map.keys())}"
            )
        
        task_class = task_map[task_name]
        
        print(f"Loading Folktables dataset: {task_name} for {state} ({year})...")
        
        # Create data source
        data_source = ACSDataSource(
            survey_year=year,
            horizon='1-Year',
            survey='person'
        )
        
        # Download and extract data
        acs_data = data_source.get_data(states=[state], download=True)
        
        # Extract features and labels using the task
        task = task_class()
        X, y = task.df_to_numpy(acs_data)
        
        # Convert to float32
        X = X.astype(np.float32)
        y = y.astype(int)
        
        # Get feature names from task
        feature_names = task.features
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Get class names
        unique_classes = np.unique(y)
        if task_name == "income":
            class_names = ["income_<=50K", "income_>50K"]
        elif task_name == "coverage":
            class_names = ["no_coverage", "has_coverage"]
        elif task_name == "mobility":
            class_names = ["not_moved", "moved"]
        elif task_name == "employment":
            class_names = ["not_employed", "employed"]
        elif task_name == "travel":
            class_names = ["travel_<=30min", "travel_>30min"]
        else:
            class_names = [f"class_{i}" for i in unique_classes]
        
        print(f"  Loaded Folktables dataset: {task_name} ({state}, {year})")
        print(f"  Features: {len(feature_names)}, Classes: {len(class_names)}")
    
    else:
        supported = ['breast_cancer', 'covtype', 'wine', 'iris', 'housing', 'synthetic', 'moons', 'circles']
        if UCIML_AVAILABLE:
            supported.append('uci_<id_or_name> (e.g., uci_adult, uci_2)')
        if FOLKTABLES_AVAILABLE:
            supported.append('folktables_<task>_<state>_<year> (e.g., folktables_income_CA_2018)')
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported datasets: {', '.join(supported)}"
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
    patience=10
):
    """
    Train a SimpleClassifier on tabular data (same as in sklearn_datasets_anchors.py).
    
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


def predict_proba_wrapper(classifier, device):
    """Wrapper for classifier prediction function."""
    def predict_fn(X):
        classifier.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
            logits = classifier(X_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    return predict_fn


def parse_anchor_rule(anchor_rule: Any, feature_names: List[str]) -> List[Tuple[str, float, float]]:
    """
    Parse an anchor rule from anchor-exp library into feature ranges.
    
    Args:
        anchor_rule: Anchor rule from anchor-exp (can be string, list, or other format)
        feature_names: List of feature names
    
    Returns:
        List of (feature_name, lower_bound, upper_bound) tuples
    """
    conditions = []
    
    # Handle different formats from anchor-exp
    if isinstance(anchor_rule, str):
        rule_str = anchor_rule
    elif isinstance(anchor_rule, list):
        # anchor-exp sometimes returns list of strings or tuples
        if len(anchor_rule) == 0:
            return []
        # Check if it's a list of strings
        if isinstance(anchor_rule[0], str):
            rule_str = " and ".join(str(item) for item in anchor_rule)
        else:
            # Might be list of tuples or other format - convert to string
            rule_str = " and ".join(str(item) for item in anchor_rule)
    else:
        # Try to convert to string
        rule_str = str(anchor_rule)
    
    if not rule_str or rule_str.strip() == "":
        return []
    
    # Pattern to match: "feature_name ∈ [lower, upper]" or "feature_name <= value" or "feature_name >= value"
    # Also handle "feature_name > value" and "feature_name < value"
    # Also handle formats like "feature_name <= value <= feature_name" (range)
    pattern_range = r'(.+?)\s*∈\s*\[([-\d.]+),\s*([-\d.]+)\]'
    pattern_range_alt = r'(.+?)\s*<=\s*([-\d.]+)\s*<=\s*(.+)'  # "feature <= val <= feature" (same feature)
    pattern_le = r'(.+?)\s*<=\s*([-\d.]+)'
    pattern_ge = r'(.+?)\s*>=\s*([-\d.]+)'
    pattern_lt = r'(.+?)\s*<\s*([-\d.]+)'
    pattern_gt = r'(.+?)\s*>\s*([-\d.]+)'
    
    # Split by " and " to get individual conditions
    condition_strings = rule_str.split(" and ")
    
    for condition_str in condition_strings:
        condition_str = condition_str.strip()
        if not condition_str:
            continue
        
        # Try range pattern first (∈ [lower, upper])
        match = re.search(pattern_range, condition_str)
        if match:
            feature_name = match.group(1).strip()
            lower = float(match.group(2))
            upper = float(match.group(3))
            conditions.append((feature_name, lower, upper))
            continue
        
        # Try alternative range pattern (feature <= val <= feature)
        match = re.search(pattern_range_alt, condition_str)
        if match:
            feature_name = match.group(1).strip()
            lower_str = match.group(2).strip()
            upper_str = match.group(3).strip()
            # Check if upper_str is a number or same feature name
            try:
                lower = float(lower_str)
                upper = float(upper_str)
                conditions.append((feature_name, lower, upper))
                continue
            except ValueError:
                pass
        
        # Try <= pattern
        match = re.search(pattern_le, condition_str)
        if match:
            feature_name = match.group(1).strip()
            upper = float(match.group(2))
            # For <=, we use -inf as lower bound (will be replaced with feature min later)
            conditions.append((feature_name, float('-inf'), upper))
            continue
        
        # Try >= pattern
        match = re.search(pattern_ge, condition_str)
        if match:
            feature_name = match.group(1).strip()
            lower = float(match.group(2))
            # For >=, we use +inf as upper bound (will be replaced with feature max later)
            conditions.append((feature_name, lower, float('inf')))
            continue
        
        # Try < pattern
        match = re.search(pattern_lt, condition_str)
        if match:
            feature_name = match.group(1).strip()
            upper = float(match.group(2))
            conditions.append((feature_name, float('-inf'), upper))
            continue
        
        # Try > pattern
        match = re.search(pattern_gt, condition_str)
        if match:
            feature_name = match.group(1).strip()
            lower = float(match.group(2))
            conditions.append((feature_name, lower, float('inf')))
            continue
    
    return conditions


def compute_class_union_metrics(
    anchor_rules: List[Any],
    X_data: np.ndarray,
    y_data: np.ndarray,
    target_class: int,
    feature_names: List[str],
    X_train: np.ndarray
) -> Tuple[float, float]:
    """
    Compute class-level union metrics from a list of anchor rules.
    
    For each feature, takes the union of all ranges across all anchors:
    - Lower bound: minimum of all lower bounds (or -inf if any anchor has no lower bound)
    - Upper bound: maximum of all upper bounds (or +inf if any anchor has no upper bound)
    
    Then computes precision and coverage on the union.
    
    Args:
        anchor_rules: List of anchor rules (from anchor-exp)
        X_data: Data matrix (n_samples, n_features) in original feature space
        y_data: Labels (n_samples,)
        target_class: Target class to compute metrics for
        feature_names: List of feature names
        X_train: Training data to get feature ranges for handling -inf/+inf
    
    Returns:
        Tuple of (class_precision, class_coverage)
    """
    if len(anchor_rules) == 0:
        return 0.0, 0.0
    
    # Parse all anchor rules
    all_conditions = []
    for anchor_rule in anchor_rules:
        conditions = parse_anchor_rule(anchor_rule, feature_names)
        if conditions:
            all_conditions.append(conditions)
    
    if len(all_conditions) == 0:
        # No valid conditions found - union covers everything
        return 0.0, 0.0
    
    # Compute feature ranges from training data
    feature_mins = X_train.min(axis=0)
    feature_maxs = X_train.max(axis=0)
    
    # Build union: for each feature, take min of all lower bounds and max of all upper bounds
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    union_lower = np.full(len(feature_names), float('-inf'))
    union_upper = np.full(len(feature_names), float('inf'))
    
    for conditions in all_conditions:
        for feature_name, lower, upper in conditions:
            if feature_name in feature_to_idx:
                feat_idx = feature_to_idx[feature_name]
                # Update union bounds
                if lower != float('-inf'):
                    if union_lower[feat_idx] == float('-inf'):
                        union_lower[feat_idx] = lower
                    else:
                        union_lower[feat_idx] = min(union_lower[feat_idx], lower)
                if upper != float('inf'):
                    if union_upper[feat_idx] == float('inf'):
                        union_upper[feat_idx] = upper
                    else:
                        union_upper[feat_idx] = max(union_upper[feat_idx], upper)
    
    # Replace -inf with feature min and +inf with feature max
    for feat_idx in range(len(feature_names)):
        if union_lower[feat_idx] == float('-inf'):
            union_lower[feat_idx] = feature_mins[feat_idx]
        if union_upper[feat_idx] == float('inf'):
            union_upper[feat_idx] = feature_maxs[feat_idx]
    
    # Check which samples satisfy the union
    in_union = np.all((X_data >= union_lower) & (X_data <= union_upper), axis=1)
    is_class = (y_data == target_class)
    
    # Coverage: fraction of class samples in union
    n_class_samples = is_class.sum()
    if n_class_samples == 0:
        coverage = 0.0
    else:
        coverage = float((in_union & is_class).sum() / n_class_samples)
    
    # Precision: fraction of union samples that belong to class
    n_union_samples = in_union.sum()
    if n_union_samples == 0:
        precision = 0.0
    else:
        precision = float((in_union & is_class).sum() / n_union_samples)
    
    return precision, coverage


def run_lime(
    classifier: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    device: torch.device,
    n_instances_per_class: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run LIME explanations on test instances.
    
    Returns:
        Dictionary with LIME results including feature importance per instance
    """
    print("\n" + "="*80)
    print("Running LIME Explanations")
    print("="*80)
    print("NOTE: LIME is an INSTANCE-LEVEL method.")
    print("  - Each instance gets its own explanation")
    print("  - Results are aggregated (averaged) by class")
    print("  - Unlike Dynamic Anchors, which produce ONE unified explanation per class")
    print("="*80)
    
    try:
        from lime import lime_tabular
    except ImportError:
        raise ImportError(
            "LIME is required. Install with: pip install lime"
        )
    
    # Validate inputs
    if X_train.shape[1] != len(feature_names):
        raise ValueError(f"X_train has {X_train.shape[1]} features but {len(feature_names)} feature names")
    if X_test.shape[1] != len(feature_names):
        raise ValueError(f"X_test has {X_test.shape[1]} features but {len(feature_names)} feature names")
    
    # Create LIME explainer
    try:
        explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            discretize_continuous=True,
            random_state=seed
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create LIME explainer: {e}") from e
    
    # Wrapper for prediction function
    predict_fn = predict_proba_wrapper(classifier, device)
    
    # Test prediction function
    try:
        test_pred = predict_fn(X_test[:1])
        if test_pred.shape[1] != len(class_names):
            raise ValueError(
                f"Prediction function returns {test_pred.shape[1]} classes "
                f"but expected {len(class_names)} classes"
            )
        print(f"Prediction function test passed: shape={test_pred.shape}, classes={len(class_names)}")
    except Exception as e:
        raise RuntimeError(f"Prediction function test failed: {e}") from e
    
    # Wrap predict_fn to catch any issues and provide better error messages
    def safe_predict_fn(X):
        """Wrapper that catches errors and provides diagnostics."""
        try:
            return predict_fn(X)
        except Exception as e:
            print(f"  Error in prediction function: {e}")
            raise
    
    results = {}
    unique_classes = np.unique(y_test)
    
    for cls in unique_classes:
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        
        # Sample instances
        np.random.seed(seed)
        sel = np.random.choice(idx_cls, size=min(n_instances_per_class, idx_cls.size), replace=False)
        
        class_results = []
        feature_importance_sum = np.zeros(len(feature_names))
        
        for i, instance_idx in enumerate(sel):
            instance = X_test[instance_idx]
            true_label = y_test[instance_idx]
            
            try:
                # Get explanation with multiple fallback strategies
                # Strategy 1: Try with predicted class label (more stable)
                num_features_to_use = min(len(feature_names), 10)  # Limit to top 10 features
                
                # First, get the predicted class
                pred_probs = predict_fn(instance.reshape(1, -1))
                predicted_class = int(np.argmax(pred_probs[0]))
                
                # Try explaining for predicted class first (more stable)
                explanation = None
                exp_list = None
                
                # Strategy 1: Try explaining for predicted class
                try:
                    explanation = explainer.explain_instance(
                        instance,
                        predict_fn,
                        num_features=num_features_to_use,
                        top_labels=1,
                        labels=(predicted_class,),  # Explain for predicted class
                        num_samples=5000,  # More samples for better stability
                    )
                    # When labels is specified, as_list() works without label parameter
                    exp_list = explanation.as_list()
                except (SystemExit, Exception) as e1:
                    # Strategy 2: Try without specifying label
                    try:
                        explanation = explainer.explain_instance(
                            instance,
                            predict_fn,
                            num_features=num_features_to_use,
                            top_labels=1,
                            num_samples=5000,
                        )
                        exp_list = explanation.as_list()
                    except (SystemExit, Exception) as e2:
                        # Strategy 3: Try with fewer features
                        try:
                            num_features_small = min(5, len(feature_names))
                            explanation = explainer.explain_instance(
                                instance,
                                predict_fn,
                                num_features=num_features_small,
                                top_labels=1,
                                num_samples=10000,  # Even more samples
                            )
                            exp_list = explanation.as_list()
                        except (SystemExit, Exception) as e3:
                            # Strategy 4: Try using as_map() instead of as_list()
                            try:
                                if explanation is None:
                                    explanation = explainer.explain_instance(
                                        instance,
                                        predict_fn,
                                        num_features=num_features_to_use,
                                        top_labels=1,
                                        num_samples=5000,
                                    )
                                # Try to extract from as_map()
                                exp_map = explanation.as_map()
                                if exp_map:
                                    # as_map() returns dict: {label: [(feature_idx, importance), ...]}
                                    # Convert to as_list() format: [(feature_name, importance), ...]
                                    label = list(exp_map.keys())[0]
                                    feature_indices_and_importance = exp_map[label]
                                    exp_list = []
                                    for feat_idx, importance in feature_indices_and_importance:
                                        if feat_idx < len(feature_names):
                                            exp_list.append((feature_names[feat_idx], importance))
                                    if len(exp_list) == 0:
                                        raise ValueError("Empty explanation after conversion")
                                else:
                                    raise ValueError("Empty explanation map")
                            except (SystemExit, Exception) as e4:
                                # All strategies failed
                                error_codes = []
                                for e in [e1, e2, e3, e4]:
                                    if isinstance(e, SystemExit):
                                        code = e.code if hasattr(e, 'code') else 1
                                        error_codes.append(f"sys.exit({code})")
                                    elif e is not None:
                                        error_codes.append(f"{type(e).__name__}")
                                
                                if i == 0:  # Only print detailed error for first instance
                                    print(f"  Warning: All LIME strategies failed for instance {instance_idx} (class {cls})")
                                    print(f"    Errors: {', '.join(error_codes)}")
                                    print(f"    Predicted class: {predicted_class}, True class: {true_label}")
                                    print(f"    Prediction probs: {pred_probs[0]}")
                                continue
                
                # If we still don't have exp_list, skip this instance
                if exp_list is None or len(exp_list) == 0:
                    if i == 0:
                        print(f"  Warning: No explanation extracted for instance {instance_idx} (class {cls})")
                    continue
                
                feature_importance = np.zeros(len(feature_names))
                for feature_name, importance in exp_list:
                    # LIME returns feature names that might be formatted differently
                    # Try exact match first, then try to find partial matches
                    if feature_name in feature_names:
                        idx = feature_names.index(feature_name)
                        feature_importance[idx] = importance
                    else:
                        # Try to find by partial match (LIME might add prefixes/suffixes)
                        for idx, fn in enumerate(feature_names):
                            if fn in feature_name or feature_name in fn:
                                feature_importance[idx] = importance
                                break
                
                # Check if we got any non-zero importance
                if np.sum(np.abs(feature_importance)) == 0 and len(exp_list) > 0:
                    print(f"  Warning: Instance {instance_idx} explanation extracted but all importances are zero")
                    print(f"    Explanation list length: {len(exp_list)}")
                    if len(exp_list) > 0:
                        print(f"    First few explanation items: {exp_list[:3]}")
                
                feature_importance_sum += np.abs(feature_importance)
                
                class_results.append({
                    "instance_idx": int(instance_idx),
                    "true_label": int(true_label),
                    "feature_importance": feature_importance.tolist(),
                    "explanation": exp_list,
                })
            except SystemExit as e:
                # LIME sometimes calls sys.exit() - catch and provide better error
                error_code = e.code if hasattr(e, 'code') else 1
                print(f"  Warning: LIME called sys.exit({error_code}) for instance {instance_idx} (class {cls})")
                print(f"    This often happens when LIME cannot find a valid explanation.")
                print(f"    Try reducing num_features or adjusting LIME parameters.")
                continue
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"  Warning: Failed to explain instance {instance_idx} (class {cls}): {error_type}: {error_msg}")
                # Only print traceback for first error to avoid spam
                if i == 0:
                    print("  First error traceback:")
                    traceback.print_exc()
                continue
        
        if len(class_results) == 0:
            print(f"\nClass {cls}: No successful explanations (all failed)")
            continue
        
        avg_feature_importance = feature_importance_sum / len(class_results)
        
        results[int(cls)] = {
            "n_instances": len(class_results),
            "avg_feature_importance": avg_feature_importance.tolist(),
            "individual_results": class_results,
        }
        
        # Print top features for this class
        top_indices = np.argsort(np.abs(avg_feature_importance))[-5:][::-1]
        print(f"\nClass {cls} ({class_names[cls] if cls < len(class_names) else cls}):")
        print(f"  Top 5 features (avg importance):")
        for idx in top_indices:
            print(f"    {feature_names[idx]}: {avg_feature_importance[idx]:.4f}")
    
    if len(results) == 0:
        print("\n" + "="*80)
        print("WARNING: No successful LIME explanations for any class")
        print("="*80)
        print("This can happen when:")
        print("  - LIME's sampling algorithm fails to find valid explanations")
        print("  - Prediction probabilities are too extreme (close to 0 or 1)")
        print("  - Data distribution issues prevent LIME from generating explanations")
        print("\nConsider:")
        print("  - Using other methods (SHAP, Static Anchors, Feature Importance)")
        print("  - Adjusting LIME parameters (num_features, sampling parameters)")
        print("  - Checking if the classifier predictions are reasonable")
        print("="*80)
        raise RuntimeError("No successful LIME explanations for any class")
    
    # Print summary
    successful_classes = list(results.keys())
    print(f"\nLIME Summary: Successfully explained {len(successful_classes)} out of {len(unique_classes)} classes")
    if len(successful_classes) < len(unique_classes):
        failed_classes = set(unique_classes) - set(successful_classes)
        print(f"  Failed classes: {sorted(failed_classes)}")
        print(f"  Successful classes: {sorted(successful_classes)}")
    
    return {
        "method": "LIME",
        "per_class_results": results,
        "successful_classes": sorted(successful_classes),
        "failed_classes": sorted(set(unique_classes) - set(successful_classes)),
    }


def run_static_anchors(
    classifier: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    device: torch.device,
    anchor_threshold: float = 0.95,
    n_instances_per_class: int = 20,
    disc_perc: List[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run static anchors using anchor-exp library.
    
    Returns:
        Dictionary with static anchor results including precision and coverage
    """
    print("\n" + "="*80)
    print("Running Static Anchors")
    print("="*80)
    print("NOTE: Static Anchors is an INSTANCE-LEVEL method.")
    print("  - Each instance gets its own anchor explanation")
    print("  - Results (precision/coverage) are aggregated (averaged) by class")
    print("  - Unlike Dynamic Anchors, which produce ONE unified anchor per class")
    print("="*80)
    
    try:
        from anchor import anchor_tabular
    except ImportError:
        raise ImportError(
            "anchor-exp is required for static anchors. Install with: pip install anchor-exp"
        )
    
    # Default discretization percentiles
    if disc_perc is None:
        disc_perc = [25, 50, 75]
    
    def predict_labels(x: np.ndarray) -> np.ndarray:
        classifier.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(device)
            preds = classifier(t).argmax(dim=1).cpu().numpy()
        return preds
    
    categorical_names = {}
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names,
        feature_names,
        X_train,
        categorical_names,
    )
    # Note: anchor-exp's AnchorTabularExplainer doesn't have a fit() method
    # The data is passed directly to the constructor
    
    results = {}
    unique_classes = np.unique(y_test)
    
    for cls in unique_classes:
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        
        # Sample instances
        np.random.seed(seed)
        sel = np.random.choice(idx_cls, size=min(n_instances_per_class, idx_cls.size), replace=False)
        
        class_results = []
        
        for i in sel:
            exp = explainer.explain_instance(
                X_test[i], 
                predict_labels, 
                threshold=anchor_threshold,
            )
            
            def _metric(val):
                try:
                    return float(val() if callable(val) else val)
                except Exception:
                    return 0.0
            
            # Extract precision/coverage
            prec_val = _metric(getattr(exp, 'precision', 0.0))
            cov_val = _metric(getattr(exp, 'coverage', 0.0))
            
            # Extract anchor rule
            anchor_names = []
            if hasattr(exp, 'names'):
                names_attr = getattr(exp, 'names')
                try:
                    anchor_names = list(names_attr() if callable(names_attr) else names_attr)
                except Exception:
                    anchor_names = []
            elif hasattr(exp, 'as_list'):
                try:
                    anchor_names = list(exp.as_list())
                except Exception:
                    anchor_names = []
            
            class_results.append({
                "instance_idx": int(i),
                "precision": prec_val,
                "coverage": cov_val,
                "anchor": anchor_names,
            })
        
        if len(class_results) > 0:
            # Instance-level metrics (average across all instances)
            avg_prec = float(np.mean([r["precision"] for r in class_results]))
            avg_cov = float(np.mean([r["coverage"] for r in class_results]))
            
            # Class-level metrics (union of all anchors for this class)
            anchor_rules = [r["anchor"] for r in class_results if r.get("anchor")]
            class_prec, class_cov = compute_class_union_metrics(
                anchor_rules=anchor_rules,
                X_data=X_test,
                y_data=y_test,
                target_class=cls,
                feature_names=feature_names,
                X_train=X_train
            )
            
            results[int(cls)] = {
                # Instance-level metrics (averaged across instances)
                "instance_precision": avg_prec,
                "instance_coverage": avg_cov,
                # Class-level metrics (union of all anchors)
                "class_precision": class_prec,
                "class_coverage": class_cov,
                # Legacy fields for backward compatibility
                "avg_precision": avg_prec,
                "avg_coverage": avg_cov,
                "n_instances": len(class_results),
                "individual_results": class_results,
            }
            
            print(f"\nClass {cls} ({class_names[cls] if cls < len(class_names) else cls}):")
            print(f"  Instance-level (avg across {len(class_results)} instances):")
            print(f"    Avg Precision: {avg_prec:.3f}")
            print(f"    Avg Coverage:  {avg_cov:.3f}")
            print(f"  Class-level (union of all {len(anchor_rules)} anchors):")
            print(f"    Union Precision: {class_prec:.3f}")
            print(f"    Union Coverage:  {class_cov:.3f}")
    
    return {
        "method": "Static Anchors",
        "per_class_results": results,
    }


def run_shap(
    classifier: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    device: torch.device,
    n_instances_per_class: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run SHAP explanations on test instances.
    
    Returns:
        Dictionary with SHAP results including feature importance per instance
    """
    print("\n" + "="*80)
    print("Running SHAP Explanations")
    print("="*80)
    print("NOTE: SHAP is an INSTANCE-LEVEL method.")
    print("  - Each instance gets its own SHAP values")
    print("  - Results are aggregated (averaged) by class")
    print("  - Unlike Dynamic Anchors, which produce ONE unified explanation per class")
    print("="*80)
    
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP is required. Install with: pip install shap"
        )
    
    # Wrapper for prediction function
    predict_fn = predict_proba_wrapper(classifier, device)
    
    # Use KernelExplainer for model-agnostic SHAP values
    # For faster computation, we'll use a background dataset
    background_size = min(100, len(X_train))
    np.random.seed(seed)
    background_indices = np.random.choice(len(X_train), size=background_size, replace=False)
    X_background = X_train[background_indices]
    
    explainer = shap.KernelExplainer(predict_fn, X_background)
    
    results = {}
    unique_classes = np.unique(y_test)
    
    for cls in unique_classes:
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        
        # Sample instances
        np.random.seed(seed)
        sel = np.random.choice(idx_cls, size=min(n_instances_per_class, idx_cls.size), replace=False)
        
        class_results = []
        feature_importance_sum = np.zeros(len(feature_names))
        
        for i in sel:
            instance = X_test[i:i+1]  # Keep 2D shape
            true_label = int(y_test[i])
            
            try:
                # Get SHAP values
                shap_values = explainer.shap_values(instance, nsamples=100)
                
                # Debug: Print shape information for first instance
                if i == sel[0] and cls == unique_classes[0]:
                    print(f"  Debug: SHAP values type: {type(shap_values)}")
                    if isinstance(shap_values, list):
                        print(f"  Debug: SHAP is list with {len(shap_values)} elements")
                        for idx, sv in enumerate(shap_values):
                            print(f"    Class {idx}: shape={np.asarray(sv).shape}")
                    else:
                        print(f"  Debug: SHAP is array with shape={np.asarray(shap_values).shape}")
                
                # Handle multi-class: shap_values is a list for each class
                if isinstance(shap_values, list):
                    # shap_values is a list of arrays, one per class
                    # Each array has shape (n_instances, n_features) = (1, n_features) for single instance
                    if true_label < len(shap_values):
                        # Get SHAP values for the true class
                        class_shap = shap_values[true_label]
                        # Handle shape: could be (1, n_features) or (n_features,)
                        class_shap_arr = np.asarray(class_shap)
                        if class_shap_arr.ndim == 2:
                            instance_shap = class_shap_arr[0]  # Shape: (n_features,)
                        elif class_shap_arr.ndim == 1:
                            instance_shap = class_shap_arr  # Already 1D
                        else:
                            # Flatten and take first n_features
                            instance_shap = class_shap_arr.flatten()[:len(feature_names)]
                    else:
                        # Fallback to first class
                        class_shap = shap_values[0]
                        class_shap_arr = np.asarray(class_shap)
                        if class_shap_arr.ndim == 2:
                            instance_shap = class_shap_arr[0]
                        elif class_shap_arr.ndim == 1:
                            instance_shap = class_shap_arr
                        else:
                            instance_shap = class_shap_arr.flatten()[:len(feature_names)]
                else:
                    # Single array case - could be (n_features,) or (n_features, n_classes) or (1, n_features, n_classes) or flattened (n_features * n_classes,)
                    shap_array = np.asarray(shap_values)
                    
                    # Handle different possible shapes
                    if shap_array.ndim == 1:
                        # Could be (n_features,) or (n_features * n_classes,) flattened
                        if len(shap_array) == len(feature_names):
                            # Shape: (n_features,) - single class
                            instance_shap = shap_array
                        elif len(shap_array) == len(feature_names) * len(class_names):
                            # Flattened array with all classes - extract for true class
                            n_features = len(feature_names)
                            start_idx = true_label * n_features
                            end_idx = start_idx + n_features
                            instance_shap = shap_array[start_idx:end_idx]
                        else:
                            # Take first n_features
                            instance_shap = shap_array[:len(feature_names)]
                    elif shap_array.ndim == 2:
                        # Could be (1, n_features) or (n_features, n_classes) or (1, n_features * n_classes)
                        if shap_array.shape[0] == 1:
                            # Shape: (1, n_features) or (1, n_features * n_classes)
                            flat_val = shap_array[0]
                            if len(flat_val) == len(feature_names):
                                instance_shap = flat_val
                            elif len(flat_val) == len(feature_names) * len(class_names):
                                n_features = len(feature_names)
                                start_idx = true_label * n_features
                                end_idx = start_idx + n_features
                                instance_shap = flat_val[start_idx:end_idx]
                            else:
                                instance_shap = flat_val[:len(feature_names)]
                        elif shap_array.shape[1] == len(feature_names):
                            # Shape: (n_features, n_classes) - need to select class
                            instance_shap = shap_array[:, true_label] if true_label < shap_array.shape[1] else shap_array[:, 0]
                        else:
                            # Try flattening and extracting
                            flat_val = shap_array.flatten()
                            if len(flat_val) == len(feature_names) * len(class_names):
                                n_features = len(feature_names)
                                start_idx = true_label * n_features
                                end_idx = start_idx + n_features
                                instance_shap = flat_val[start_idx:end_idx]
                            else:
                                instance_shap = flat_val[:len(feature_names)]
                    elif shap_array.ndim == 3:
                        # Shape: (1, n_features, n_classes) - single instance, multiple classes
                        instance_shap = shap_array[0, :, true_label] if true_label < shap_array.shape[2] else shap_array[0, :, 0]
                    else:
                        # Fallback: flatten and extract
                        flat_val = shap_array.flatten()
                        if len(flat_val) == len(feature_names) * len(class_names):
                            n_features = len(feature_names)
                            start_idx = true_label * n_features
                            end_idx = start_idx + n_features
                            instance_shap = flat_val[start_idx:end_idx]
                        else:
                            instance_shap = flat_val[:len(feature_names)]
                
                # Ensure instance_shap is 1D and has correct length
                instance_shap = np.asarray(instance_shap).flatten()
                
                # Final check: if still wrong length, try to fix it
                if len(instance_shap) != len(feature_names):
                    # Check if it's a multiple (e.g., 60 = 30 * 2 for binary classification)
                    if len(instance_shap) == len(feature_names) * len(class_names):
                        # Likely concatenated values for all classes - extract for true class
                        n_features = len(feature_names)
                        start_idx = true_label * n_features
                        end_idx = start_idx + n_features
                        instance_shap = instance_shap[start_idx:end_idx]
                    elif len(instance_shap) > len(feature_names):
                        # Take first n_features
                        instance_shap = instance_shap[:len(feature_names)]
                    else:
                        # Pad with zeros if too short (shouldn't happen, but handle gracefully)
                        padded = np.zeros(len(feature_names))
                        padded[:len(instance_shap)] = instance_shap
                        instance_shap = padded
                
                # Final validation
                if len(instance_shap) != len(feature_names):
                    # Last resort: try to reshape
                    original_shape = np.asarray(shap_values).shape if not isinstance(shap_values, list) else [arr.shape for arr in shap_values]
                    raise ValueError(
                        f"SHAP values length {len(instance_shap)} doesn't match "
                        f"feature count {len(feature_names)} after processing. "
                        f"Original shape: {original_shape}, "
                        f"n_classes: {len(class_names)}, "
                        f"true_label: {true_label}"
                    )
                
                feature_importance = np.abs(instance_shap)
                feature_importance_sum += feature_importance
                
                class_results.append({
                    "instance_idx": int(i),
                    "true_label": true_label,
                    "feature_importance": instance_shap.tolist(),
                    "feature_importance_abs": feature_importance.tolist(),
                })
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"  Warning: Failed to compute SHAP for instance {i} (class {cls}): {error_type}: {error_msg}")
                if len(class_results) == 0:  # Only print traceback for first error
                    print("  First error traceback:")
                    traceback.print_exc()
                continue
        
        if len(class_results) == 0:
            print(f"\nClass {cls}: No successful SHAP explanations (all failed)")
            continue
        
        avg_feature_importance = feature_importance_sum / len(class_results)
        
        results[int(cls)] = {
            "n_instances": len(class_results),
            "avg_feature_importance": avg_feature_importance.tolist(),
            "individual_results": class_results,
        }
        
        # Print top features for this class
        top_indices = np.argsort(avg_feature_importance)[-5:][::-1]
        print(f"\nClass {cls} ({class_names[cls] if cls < len(class_names) else cls}):")
        print(f"  Top 5 features (avg |SHAP|):")
        for idx in top_indices:
            print(f"    {feature_names[idx]}: {avg_feature_importance[idx]:.4f}")
    
    return {
        "method": "SHAP",
        "per_class_results": results,
    }


def run_feature_importance(
    classifier: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    device: torch.device,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute feature importance using permutation importance.
    
    Returns:
        Dictionary with feature importance results
    """
    print("\n" + "="*80)
    print("Computing Feature Importance (Permutation Importance)")
    print("="*80)
    print("NOTE: Feature Importance is a GLOBAL method.")
    print("  - Computes one importance score per feature across all data")
    print("  - Not instance-level or class-level - measures overall feature impact")
    print("="*80)
    
    # Get baseline accuracy
    classifier.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        logits = classifier(X_test_tensor)
        baseline_preds = logits.argmax(dim=1).cpu().numpy()
    baseline_acc = accuracy_score(y_test, baseline_preds)
    
    # Manual permutation importance (since sklearn's permutation_importance expects sklearn estimator)
    n_features = X_test.shape[1]
    importances = np.zeros(n_features)
    importances_std = np.zeros(n_features)
    
    np.random.seed(random_state)
    
    print(f"Computing permutation importance (baseline accuracy: {baseline_acc:.4f})...")
    print("This may take a while...")
    
    for feature_idx in range(n_features):
        feature_scores = []
        
        for repeat in range(n_repeats):
            # Permute feature
            X_test_permuted = X_test.copy()
            perm_indices = np.random.permutation(len(X_test_permuted))
            X_test_permuted[:, feature_idx] = X_test_permuted[perm_indices, feature_idx]
            
            # Evaluate with permuted feature
            classifier.eval()
            with torch.no_grad():
                X_test_tensor = torch.from_numpy(X_test_permuted.astype(np.float32)).to(device)
                logits = classifier(X_test_tensor)
                preds = logits.argmax(dim=1).cpu().numpy()
            
            perm_acc = accuracy_score(y_test, preds)
            # Importance is the decrease in accuracy
            feature_scores.append(baseline_acc - perm_acc)
        
        importances[feature_idx] = np.mean(feature_scores)
        importances_std[feature_idx] = np.std(feature_scores)
        
        if (feature_idx + 1) % max(1, n_features // 10) == 0:
            print(f"  Progress: {feature_idx + 1}/{n_features} features")
    
    # Sort features by importance
    sorted_indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(sorted_indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f} (+/- {importances_std[idx]:.4f})")
    
    return {
        "method": "Permutation Importance",
        "baseline_accuracy": float(baseline_acc),
        "feature_importance": importances.tolist(),
        "feature_importance_std": importances_std.tolist(),
        "feature_names": feature_names,
        "sorted_features": [feature_names[i] for i in sorted_indices],
        "sorted_importance": importances[sorted_indices].tolist(),
    }


def main(
    dataset_name: str = "breast_cancer",
    sample_size: int = None,
    n_instances_per_class: int = 20,
    methods: List[str] = None,
    seed: int = 42,
    output_dir: str = None,
):
    """
    Main function: Run all baseline explainability methods.
    
    Args:
        dataset_name: Dataset to use
        sample_size: Optional size to sample (for large datasets)
        n_instances_per_class: Number of instances per class to explain
        methods: List of methods to run (None = all methods)
        seed: Random seed
        output_dir: Output directory for results
    """
    if methods is None:
        methods = ["lime", "static_anchors", "shap", "feature_importance"]
    
    print("\n" + "="*80)
    print(f"Baseline Explainability Methods - {dataset_name.upper().replace('_', ' ')}")
    print("="*80)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
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
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Set device
    device, device_str = get_device_pair("auto")
    print(f"\nUsing device: {device} ({device_str})")
    
    n_features = X_train.shape[1]
    unique_classes_train = np.unique(y_train)
    n_classes = len(unique_classes_train)
    
    # Train classifier
    print("\n" + "="*80)
    print("STEP 2: Training Classifier")
    print("="*80)
    
    classifier, test_acc = train_classifier(
        X_train_scaled, y_train, X_test_scaled, y_test,
        n_features=n_features,
        n_classes=n_classes,
        device=device,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        patience=10
    )
    
    print(f"\nClassifier test accuracy: {test_acc:.3f}")
    
    # Run baseline methods
    all_results = {
        "dataset": dataset_name,
        "test_accuracy": float(test_acc),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "class_names": class_names,
        "feature_names": feature_names,
        "n_instances_per_class": n_instances_per_class,
        "methods": {},
    }
    
    # LIME
    if "lime" in methods:
        try:
            lime_results = run_lime(
                classifier, X_train_scaled, X_test_scaled, y_test,
                feature_names, class_names, device,
                n_instances_per_class=n_instances_per_class,
                seed=seed,
            )
            all_results["methods"]["lime"] = lime_results
        except (Exception, SystemExit, KeyboardInterrupt) as e:
            import traceback
            if isinstance(e, SystemExit):
                error_msg = f"SystemExit: LIME library called sys.exit({e.code})"
            else:
                error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nError running LIME: {error_msg}")
            print("Full traceback:")
            traceback.print_exc()
            all_results["methods"]["lime"] = {
                "error": error_msg,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    # Static Anchors
    if "static_anchors" in methods:
        try:
            # Dataset-specific presets
            presets = {
                "breast_cancer": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "covtype": {"anchor_threshold": 0.95, "disc_perc": [10, 25, 50, 75, 90]},
                "wine": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "iris": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "housing": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "synthetic": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "moons": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
                "circles": {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]},
            }
            # For UCI and Folktables datasets, use default presets
            # Check if dataset_name starts with uci_ or folktables_
            if dataset_name.startswith("uci_") or dataset_name.startswith("folktables_"):
                preset = {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]}
            else:
                preset = presets.get(dataset_name, {"anchor_threshold": 0.95, "disc_perc": [25, 50, 75]})
            
            anchor_results = run_static_anchors(
                classifier, X_train_scaled, X_test_scaled, y_train, y_test,
                feature_names, class_names, device,
                anchor_threshold=preset["anchor_threshold"],
                n_instances_per_class=n_instances_per_class,
                disc_perc=preset["disc_perc"],
                seed=seed,
            )
            all_results["methods"]["static_anchors"] = anchor_results
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nError running Static Anchors: {error_msg}")
            print("Full traceback:")
            traceback.print_exc()
            all_results["methods"]["static_anchors"] = {
                "error": error_msg,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    # SHAP
    if "shap" in methods:
        try:
            shap_results = run_shap(
                classifier, X_train_scaled, X_test_scaled, y_test,
                feature_names, class_names, device,
                n_instances_per_class=n_instances_per_class,
                seed=seed,
            )
            all_results["methods"]["shap"] = shap_results
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nError running SHAP: {error_msg}")
            print("Full traceback:")
            traceback.print_exc()
            all_results["methods"]["shap"] = {
                "error": error_msg,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    # Feature Importance
    if "feature_importance" in methods:
        try:
            fi_results = run_feature_importance(
                classifier, X_train_scaled, X_test_scaled, y_train, y_test,
                feature_names, device,
                n_repeats=10,
                random_state=seed,
            )
            all_results["methods"]["feature_importance"] = fi_results
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"\nError running Feature Importance: {error_msg}")
            print("Full traceback:")
            traceback.print_exc()
            all_results["methods"]["feature_importance"] = {
                "error": error_msg,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
    
    # Save results
    if output_dir is None:
        output_dir = f"./output/{dataset_name}_baseline/"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"baseline_results_{timestamp}.json")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("\nIMPORTANT NOTE:")
    print("  - LIME, Static Anchors, and SHAP are INSTANCE-LEVEL methods")
    print("    (each instance gets its own explanation, then aggregated by class)")
    print("  - Static Anchors now also computes CLASS-LEVEL union metrics")
    print("    (union of all anchors for each class, similar to Dynamic Anchors)")
    print("  - Feature Importance is a GLOBAL method (one importance score per feature)")
    print("  - Dynamic Anchors produce CLASS-LEVEL explanations")
    print("    (ONE unified explanation per class that applies to all instances)")
    print("="*80)
    
    # Print summary for each method
    for method_name, method_results in all_results["methods"].items():
        if "error" in method_results:
            print(f"\n{method_name.upper()}: Error - {method_results['error']}")
            continue
        
        print(f"\n{method_name.upper()}:")
        
        if method_name == "static_anchors":
            # Print precision/coverage summary
            per_class = method_results.get("per_class_results", {})
            if per_class:
                # Instance-level metrics
                instance_precisions = [r.get("instance_precision", r.get("avg_precision", 0.0)) for r in per_class.values()]
                instance_coverages = [r.get("instance_coverage", r.get("avg_coverage", 0.0)) for r in per_class.values()]
                print(f"  Instance-level (avg across instances):")
                print(f"    Overall Avg Precision: {np.mean(instance_precisions):.3f}")
                print(f"    Overall Avg Coverage:  {np.mean(instance_coverages):.3f}")
                # Class-level metrics
                class_precisions = [r.get("class_precision", 0.0) for r in per_class.values()]
                class_coverages = [r.get("class_coverage", 0.0) for r in per_class.values()]
                if any(cp > 0 or cc > 0 for cp, cc in zip(class_precisions, class_coverages)):
                    print(f"  Class-level (union of all anchors):")
                    print(f"    Overall Union Precision: {np.mean(class_precisions):.3f}")
                    print(f"    Overall Union Coverage:  {np.mean(class_coverages):.3f}")
        
        elif method_name in ["lime", "shap"]:
            # Print top features summary
            per_class = method_results.get("per_class_results", {})
            if per_class:
                # Aggregate feature importance across classes
                all_importance = []
                for cls_result in per_class.values():
                    if "avg_feature_importance" in cls_result:
                        all_importance.append(cls_result["avg_feature_importance"])
                if all_importance:
                    avg_importance = np.mean(all_importance, axis=0)
                    top_indices = np.argsort(avg_importance)[-5:][::-1]
                    print(f"  Top 5 Features (avg across classes):")
                    for idx in top_indices:
                        print(f"    {feature_names[idx]}: {avg_importance[idx]:.4f}")
        
        elif method_name == "feature_importance":
            sorted_features = method_results.get("sorted_features", [])
            sorted_importance = method_results.get("sorted_importance", [])
            if sorted_features:
                print(f"  Top 5 Features:")
                for i, (feat, imp) in enumerate(zip(sorted_features[:5], sorted_importance[:5])):
                    print(f"    {i+1}. {feat}: {imp:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)
    
    return all_results


def main_cli():
    """Command-line interface entry point for establish_baseline."""
    parser = argparse.ArgumentParser(
        description="Run baseline explainability methods for comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m baseline.establish_baseline --dataset breast_cancer
  python -m baseline.establish_baseline --dataset covtype --sample_size 10000
  python -m baseline.establish_baseline --dataset wine
  python -m baseline.establish_baseline --dataset housing --sample_size 10000
  python -m baseline.establish_baseline --dataset circles
  python -m baseline.establish_baseline --dataset moons
  python -m baseline.establish_baseline --dataset synthetic
  python -m baseline.establish_baseline --dataset breast_cancer --methods lime shap
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        # Note: choices list doesn't include uci_* and folktables_* patterns
        # as they have dynamic formats, but they are supported
        choices=["breast_cancer", "covtype", "wine", "iris", "housing", "synthetic", "moons", "circles"],
        help="Dataset to use (default: breast_cancer)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for large datasets (recommended: 10000-50000 for covtype/housing)"
    )
    parser.add_argument(
        "--n_instances_per_class",
        type=int,
        default=20,
        help="Number of instances per class to explain (default: 20)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        choices=["lime", "static_anchors", "shap", "feature_importance"],
        help="Methods to run (default: all methods)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./output/{dataset}_baseline/)"
    )
    
    args = parser.parse_args()
    
    try:
        results = main(
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            n_instances_per_class=args.n_instances_per_class,
            methods=args.methods,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        return results
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_cli()

