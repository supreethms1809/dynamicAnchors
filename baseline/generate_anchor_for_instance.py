#!/usr/bin/env python3
"""
Generate anchor explanation for a specific instance ID using static anchors.

This script loads the dataset, trains a classifier (or loads an existing one),
and generates an anchor explanation for a given instance_idx using the anchor-exp library.

Usage:
    python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89
    python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89 --json
    python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89 --anchor_threshold 0.95
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import functions from establish_baseline
try:
    from baseline.establish_baseline import (
        load_dataset,
        train_classifier,
        compute_anchor_metrics_on_full_dataset,
        get_device_pair,
    )
except ImportError:
    # Add parent directory to path if running directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from baseline.establish_baseline import (
        load_dataset,
        train_classifier,
        compute_anchor_metrics_on_full_dataset,
        get_device_pair,
    )


def generate_anchor_for_instance(
    dataset_name: str,
    instance_idx: int,
    anchor_threshold: float = 0.95,
    seed: int = 42,
    sample_size: Optional[int] = None,
    classifier_model_path: Optional[str] = None,
    output_json: bool = False,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate anchor explanation for a specific instance.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'iris', 'breast_cancer')
        instance_idx: Instance index in the full dataset (train+test combined)
        anchor_threshold: Threshold for anchor explanation (default: 0.95)
        seed: Random seed for reproducibility (default: 42)
        sample_size: Optional sample size for large datasets
        classifier_model_path: Optional path to saved classifier model
        output_json: If True, output as JSON (doesn't affect return value)
        device: Device to use ('cpu', 'cuda', or 'auto')
    
    Returns:
        Dictionary with anchor explanation and metrics
    """
    try:
        from anchor import anchor_tabular
    except ImportError:
        raise ImportError(
            "anchor-exp is required. Install with: pip install anchor-exp"
        )
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    if not output_json:
        print("\n" + "="*80)
        print(f"Generating Anchor Explanation for Instance {instance_idx}")
        print("="*80)
        print(f"Dataset: {dataset_name.upper().replace('_', ' ')}")
        print("="*80)
    
    X, y, feature_names, class_names = load_dataset(dataset_name, sample_size=sample_size, seed=seed)
    
    # Split data (same split as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Combine train and test to get full dataset (same as baseline)
    X_full = np.vstack([X_train, X_test])
    y_full = np.hstack([y_train, y_test])
    X_full_scaled = np.vstack([X_train_scaled, X_test_scaled])
    
    # Validate instance_idx
    if instance_idx < 0 or instance_idx >= len(X_full):
        raise ValueError(
            f"Instance index {instance_idx} is out of range. "
            f"Dataset has {len(X_full)} instances (indices 0-{len(X_full)-1})"
        )
    
    # Get device
    if device is None:
        device, device_str = get_device_pair("auto")
    else:
        device = torch.device(device)
        device_str = str(device)
    
    if not output_json:
        print(f"\nUsing device: {device} ({device_str})")
        print(f"Full dataset size: {len(X_full)}")
        print(f"Instance index: {instance_idx}")
        print(f"Instance class: {class_names[y_full[instance_idx]]} (class {y_full[instance_idx]})")
    
    # Load or train classifier
    n_features = X_train.shape[1]
    n_classes = len(class_names)
    
    if classifier_model_path and os.path.exists(classifier_model_path):
        if not output_json:
            print(f"\nLoading classifier from: {classifier_model_path}")
        classifier = torch.load(classifier_model_path, map_location=device)
        classifier.eval()
        # Test accuracy
        classifier.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test_scaled).to(device)
            test_logits = classifier(X_test_tensor)
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            test_acc = float((test_preds == y_test).mean())
        if not output_json:
            print(f"Classifier test accuracy: {test_acc:.4f}")
    else:
        if not output_json:
            print("\nTraining classifier...")
        
        # Determine training parameters based on dataset size
        n_train_samples = len(X_train)
        is_large_dataset = (
            dataset_name.startswith("folktables_") or 
            dataset_name.startswith("uci_") or 
            dataset_name in ["housing", "covtype"] or
            n_train_samples > 10000
        )
        
        if is_large_dataset:
            classifier_epochs = 500
            classifier_patience = 100
        else:
            classifier_epochs = 100
            classifier_patience = 10
        
        classifier, test_acc = train_classifier(
            X_train_scaled, y_train, X_test_scaled, y_test,
            n_features=n_features,
            n_classes=n_classes,
            device=device,
            epochs=classifier_epochs,
            batch_size=256,
            lr=1e-3,
            patience=classifier_patience,
        )
        
        if not output_json:
            print(f"Classifier trained. Test accuracy: {test_acc:.4f}")
    
    # Initialize anchor explainer
    if not output_json:
        print("\nInitializing anchor explainer...")
    
    categorical_names = {}
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names,
        feature_names,
        X_train,  # Use unscaled X_train for anchor explainer
        categorical_names,
    )
    
    # Get predictions for full dataset
    classifier.eval()
    with torch.no_grad():
        X_full_tensor = torch.from_numpy(X_full_scaled.astype(np.float32)).to(device)
        full_logits = classifier(X_full_tensor)
        full_predictions = full_logits.argmax(dim=1).cpu().numpy()
    
    # Get instance from full dataset
    instance = X_full[instance_idx]  # Unscaled instance
    instance_scaled = X_full_scaled[instance_idx]  # Scaled instance
    instance_label = y_full[instance_idx]
    original_prediction = full_predictions[instance_idx]
    
    if not output_json:
        print(f"\nGenerating anchor explanation...")
        print(f"  Instance label: {class_names[instance_label]} (class {instance_label})")
        print(f"  Instance prediction: {class_names[original_prediction]} (class {original_prediction})")
        print(f"  Anchor threshold: {anchor_threshold}")
    
    # Define prediction function
    def predict_labels(x: np.ndarray) -> np.ndarray:
        classifier.eval()
        # x is in original (unscaled) space, need to scale it
        x_scaled = scaler.transform(x.astype(np.float32)).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x_scaled).to(device)
            preds = classifier(t).argmax(dim=1).cpu().numpy()
        return preds
    
    # Generate anchor explanation
    start_time = time.perf_counter()
    exp = explainer.explain_instance(
        instance, 
        predict_labels, 
        threshold=anchor_threshold,
    )
    end_time = time.perf_counter()
    explanation_time = end_time - start_time
    
    # Extract anchor rule
    def _metric(val):
        try:
            return float(val() if callable(val) else val)
        except Exception:
            return 0.0
    
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
    
    # Compute precision and coverage on full dataset
    prec_full, cov_full = compute_anchor_metrics_on_full_dataset(
        anchor_rule=anchor_names,
        X_full=X_full,
        y_full=y_full,
        original_instance=instance,
        original_prediction=original_prediction,
        feature_names=feature_names,
        X_train=X_train,
        classifier=classifier,
        device=device,
        explainer=explainer
    )
    
    # Get original precision/coverage from anchor-exp
    prec_original = _metric(getattr(exp, 'precision', 0.0))
    cov_original = _metric(getattr(exp, 'coverage', 0.0))
    
    # Build result
    result = {
        "instance_idx": int(instance_idx),
        "instance_label": int(instance_label),
        "instance_label_name": class_names[instance_label],
        "original_prediction": int(original_prediction),
        "original_prediction_name": class_names[original_prediction],
        "anchor": anchor_names,
        "precision": float(prec_full),
        "coverage": float(cov_full),
        "precision_original": float(prec_original),
        "coverage_original": float(cov_original),
        "rollout_time_seconds": float(explanation_time),
        "anchor_threshold": float(anchor_threshold),
        "dataset": dataset_name,
    }
    
    return result


def print_anchor_result(result: Dict[str, Any], output_json: bool = False):
    """Print anchor result in a formatted way."""
    if output_json:
        print(json.dumps(result, indent=2))
        return
    
    print("\n" + "="*80)
    print("ANCHOR EXPLANATION")
    print("="*80)
    print(f"Dataset: {result['dataset'].upper().replace('_', ' ')}")
    print(f"Instance Index: {result['instance_idx']}")
    print(f"Instance Label: {result['instance_label_name']} (class {result['instance_label']})")
    print(f"Original Prediction: {result['original_prediction_name']} (class {result['original_prediction']})")
    print(f"\nAnchor Rules ({len(result['anchor'])} features):")
    print("-" * 80)
    
    for i, rule in enumerate(result['anchor'], 1):
        print(f"  {i}. {rule}")
    
    print("\nMetrics:")
    print("-" * 80)
    print(f"  Precision (recomputed):  {result['precision']:.4f}")
    print(f"  Coverage (recomputed):   {result['coverage']:.4f}")
    if result.get('precision_original') is not None:
        print(f"  Precision (original):    {result['precision_original']:.4f}")
    if result.get('coverage_original') is not None:
        print(f"  Coverage (original):     {result['coverage_original']:.4f}")
    print(f"  Explanation Time:        {result['rollout_time_seconds']:.4f} seconds")
    print(f"  Anchor Threshold:        {result['anchor_threshold']:.4f}")
    print("="*80 + "\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate anchor explanation for a specific instance ID using static anchors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate anchor for instance 89 in iris dataset
  python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89
  
  # Output as JSON
  python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89 --json
  
  # Use custom anchor threshold
  python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89 --anchor_threshold 0.90
  
  # Use saved classifier model
  python -m baseline.generate_anchor_for_instance --dataset iris --instance_idx 89 --classifier_model_path model.pth
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., iris, breast_cancer, wine)'
    )
    
    parser.add_argument(
        '--instance_idx',
        type=int,
        required=True,
        help='Instance index in the full dataset (train+test combined)'
    )
    
    parser.add_argument(
        '--anchor_threshold',
        type=float,
        default=0.95,
        help='Threshold for anchor explanation (default: 0.95)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Optional sample size for large datasets'
    )
    
    parser.add_argument(
        '--classifier_model_path',
        type=str,
        default=None,
        help='Optional path to saved classifier model (if not provided, will train a new one)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON instead of formatted text'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    try:
        result = generate_anchor_for_instance(
            dataset_name=args.dataset,
            instance_idx=args.instance_idx,
            anchor_threshold=args.anchor_threshold,
            seed=args.seed,
            sample_size=args.sample_size,
            classifier_model_path=args.classifier_model_path,
            output_json=args.json,
            device=args.device,
        )
        
        print_anchor_result(result, output_json=args.json)
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
