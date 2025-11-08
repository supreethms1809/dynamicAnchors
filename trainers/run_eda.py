#!/usr/bin/env python3
"""
Quick EDA script for sklearn datasets.
Usage: python run_eda.py --dataset housing
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_breast_cancer, fetch_covtype, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name, sample_size=None, seed=42):
    """Load sklearn dataset."""
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        class_names = list(data.target_names)
    elif dataset_name == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        y = y - 1  # Convert from 1-7 to 0-6
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        class_names = [f"covertype_{i+1}" for i in range(7)]
    elif dataset_name == "wine":
        data = load_wine()
        X = data.data
        y = data.target
        feature_names = list(data.feature_names)
        class_names = list(data.target_names)
    elif dataset_name == "housing":
        data = fetch_california_housing()
        X = data.data
        prices = data.target
        # Convert to classification
        quartiles = np.percentile(prices, [25, 50, 75])
        y = np.digitize(prices, quartiles).astype(int)
        y = np.clip(y - 1, 0, 3)
        feature_names = list(data.feature_names)
        class_names = ["very_low_price", "low_price", "medium_price", "high_price"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if sample_size is not None and len(X) > sample_size:
        np.random.seed(seed)
        indices = np.random.choice(len(X), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X, y, feature_names, class_names

def quick_eda(X_train, y_train, X_test, y_test, feature_names, class_names=None, output_dir='./output/eda_output/', dataset_name='dataset'):
    """Perform quick EDA."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataframes
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train
    
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test
    
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"Training samples: {len(df_train):,}")
    print(f"Test samples: {len(df_test):,}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {len(np.unique(y_train))}")
    
    if class_names:
        print("\nClass names:", class_names)
    
    print("\nClass distribution (train):")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for cls, count in train_dist.items():
        cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        print(f"  {cls_name}: {count:,} ({count/len(y_train)*100:.1f}%)")
    
    print("\nClass distribution (test):")
    test_dist = pd.Series(y_test).value_counts().sort_index()
    for cls, count in test_dist.items():
        cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        print(f"  {cls_name}: {count:,} ({count/len(y_test)*100:.1f}%)")
    
    # Feature statistics
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    print(df_train[feature_names].describe())
    
    # Missing values
    print("\n" + "="*80)
    print("MISSING VALUES")
    print("="*80)
    missing_train = df_train[feature_names].isnull().sum()
    missing_test = df_test[feature_names].isnull().sum()
    if missing_train.sum() == 0 and missing_test.sum() == 0:
        print("✓ No missing values!")
    else:
        print("Missing values (train):")
        print(missing_train[missing_train > 0])
        print("\nMissing values (test):")
        print(missing_test[missing_test > 0])
    
    # Correlation heatmap
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("1. Correlation heatmap...")
    corr = df_train[feature_names].corr()
    plt.figure(figsize=(max(12, len(feature_names)*0.8), max(10, len(feature_names)*0.8)))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, 
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}correlation_heatmap.png")
    
    # Distribution plots
    print("2. Feature distributions by class...")
    n_features = len(feature_names)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, col in enumerate(feature_names):
        ax = axes[i]
        for cls in np.unique(y_train):
            mask = df_train['target'] == cls
            cls_name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
            ax.hist(df_train.loc[mask, col], alpha=0.6, label=cls_name, bins=30, density=True)
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}feature_distributions.png")
    
    # Box plots
    print("3. Box plots by class...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, col in enumerate(feature_names):
        ax = axes[i]
        sns.boxplot(data=df_train, x='target', y=col, ax=ax)
        ax.set_title(f'{col}', fontsize=10)
        ax.set_xlabel('Class')
        ax.set_ylabel('Value')
    
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions (Box Plots) by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}boxplots_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved to {output_dir}boxplots_by_class.png")
    
    # Pair plot for first few features
    if len(feature_names) >= 2:
        print("4. Pair plot (first 5 features)...")
        n_pair_features = min(5, len(feature_names))
        pair_features = feature_names[:n_pair_features]
        
        # Sample if too many points
        sample_size = min(1000, len(df_train))
        df_sample = df_train.sample(n=sample_size, random_state=42)
        
        g = sns.pairplot(df_sample[pair_features + ['target']], 
                        hue='target', 
                        diag_kind='kde',
                        plot_kws={'alpha': 0.6, 's': 20})
        g.fig.suptitle('Pair Plot (First 5 Features)', y=1.02)
        plt.savefig(f'{output_dir}pairplot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved to {output_dir}pairplot.png")
    
    # Try to generate comprehensive report with ydata-profiling if available
    try:
        from ydata_profiling import ProfileReport
        print("\n5. Generating comprehensive automated report (ydata-profiling)...")
        print("   This may take a few minutes...")
        profile = ProfileReport(
            df_train,
            title=f"{dataset_name.replace('_', ' ').title()} Dataset - Training Data EDA",
            explorative=True,
            minimal=False
        )
        report_path = f'{output_dir}comprehensive_report.html'
        profile.to_file(report_path)
        print(f"   ✓ Saved comprehensive report to {report_path}")
    except ImportError:
        pass  # ydata-profiling not installed, skip
    
    print("\n" + "="*80)
    print("EDA COMPLETE!")
    print("="*80)
    print(f"All visualizations saved to: {output_dir}")
    
    # Only show install message if ydata-profiling is not available
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("\n💡 Tip: Install ydata-profiling for comprehensive automated reports:")
        print("  pip install ydata-profiling")

def main():
    parser = argparse.ArgumentParser(description='Run EDA on sklearn datasets')
    parser.add_argument('--dataset', type=str, default='housing',
                       choices=['breast_cancer', 'covtype', 'wine', 'housing'],
                       help='Dataset to analyze')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for large datasets')
    parser.add_argument('--output_dir', type=str, default='./output/eda_output/',
                       help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"EDA FOR {args.dataset.upper().replace('_', ' ')} DATASET")
    print("="*80)
    
    # Load dataset
    X, y, feature_names, class_names = load_dataset(
        args.dataset, 
        sample_size=args.sample_size,
        seed=args.seed
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y if len(np.unique(y)) < 20 else None
    )
    
    # Run EDA
    quick_eda(X_train, y_train, X_test, y_test, feature_names, class_names, args.output_dir, args.dataset)

if __name__ == '__main__':
    main()

