import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris, 
    make_classification, make_moons, make_circles,
    fetch_california_housing,
    fetch_covtype
)
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Optional, List
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainers.networks import SimpleClassifier, UnifiedClassifier


class TabularDatasetLoader:
    
    def __init__(
        self,
        dataset_name: str = "breast_cancer",
        test_size: float = 0.2,
        random_state: int = 42,
        sample_size: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_unit = None
        self.X_test_unit = None
        self.X_min = None
        self.X_range = None
        self.feature_names = None
        self.class_names = None
        self.n_features = None
        self.n_classes = None
        self.classifier = None
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        print(f"\nLoading dataset: {self.dataset_name}")
        print("="*80)
        
        if self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            class_names = list(data.target_names)
        elif self.dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        elif self.dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            class_names = list(data.target_names)
        elif self.dataset_name == "synthetic":
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=5,
                n_redundant=2,
                n_classes=2,
                random_state=self.random_state
            )
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        elif self.dataset_name == "moons":
            X, y = make_moons(n_samples=1000, noise=0.1, random_state=self.random_state)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        elif self.dataset_name == "circles":
            X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=self.random_state)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        elif self.dataset_name == "covtype":
            X, y = fetch_covtype(return_X_y=True, as_frame=False)
            X = X.astype(np.float32)
            y = (y - 1).astype(int)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = [f"covertype_{i+1}" for i in range(7)]
        elif self.dataset_name == "housing":
            data = fetch_california_housing()
            X = data.data.astype(np.float32)
            prices = data.target.astype(np.float32)
            
            quartiles = np.percentile(prices, [25, 50, 75])
            y = np.digitize(prices, quartiles).astype(int)
            
            feature_names = list(data.feature_names)
            class_names = ["very_low_price", "low_price", "medium_price", "high_price"]
            
            print(f"\nConverted housing prices to 4 classes:")
            print(f"  Class 0 (very_low): < ${quartiles[0]*100:.0f}K (25th percentile)")
            print(f"  Class 1 (low): ${quartiles[0]*100:.0f}K - ${quartiles[1]*100:.0f}K (25th-50th percentile)")
            print(f"  Class 2 (medium): ${quartiles[1]*100:.0f}K - ${quartiles[2]*100:.0f}K (50th-75th percentile)")
            print(f"  Class 3 (high): >= ${quartiles[2]*100:.0f}K (75th percentile+)")
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Supported: 'breast_cancer', 'wine', 'iris', 'synthetic', 'moons', 'circles', 'covtype', 'housing'"
            )
        
        if self.sample_size is not None and self.sample_size < len(X):
            indices = np.random.RandomState(self.random_state).choice(
                len(X), size=self.sample_size, replace=False
            )
            X = X[indices]
            y = y[indices]
            print(f"Sampled {self.sample_size} instances from dataset")
        
        stratify = y if len(np.unique(y)) < 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.astype(int)
        self.y_test = y_test.astype(int)
        self.feature_names = feature_names
        self.class_names = class_names
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y))
        
        print(f"Dataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {self.n_features}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Class distribution (train): {np.bincount(y_train)}")
        print(f"  Class distribution (test): {np.bincount(y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test, feature_names, class_names
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("\nPreprocessing data...")
        print("="*80)
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train).astype(np.float32)
        self.X_test_scaled = self.scaler.transform(self.X_test).astype(np.float32)
        
        self.X_min = self.X_train_scaled.min(axis=0)
        self.X_max = self.X_train_scaled.max(axis=0)
        self.X_range = np.where((self.X_max - self.X_min) == 0, 1.0, (self.X_max - self.X_min))
        
        self.X_train_unit = (self.X_train_scaled - self.X_min) / self.X_range
        self.X_train_unit = np.clip(self.X_train_unit, 0.0, 1.0).astype(np.float32)
        
        self.X_test_unit = (self.X_test_scaled - self.X_min) / self.X_range
        self.X_test_unit = np.clip(self.X_test_unit, 0.0, 1.0).astype(np.float32)
        
        print("Data preprocessing complete:")
        print(f"  Scaled train shape: {self.X_train_scaled.shape}")
        print(f"  Scaled test shape: {self.X_test_scaled.shape}")
        print(f"  Unit train range: [{self.X_train_unit.min():.3f}, {self.X_train_unit.max():.3f}]")
        print(f"  Unit test range: [{self.X_test_unit.min():.3f}, {self.X_test_unit.max():.3f}]")
        
        return (
            self.X_train_scaled, self.X_test_scaled,
            self.X_train_unit, self.X_test_unit,
            self.X_min, self.X_range
        )
    
    def create_classifier(
        self,
        classifier_type: str = "dnn",
        hidden_size: int = 256,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        device: str = "cpu"
    ) -> torch.nn.Module:
        print(f"\nCreating classifier: {classifier_type}")
        print("="*80)
        
        if classifier_type.lower() == "dnn":
            classifier = SimpleClassifier(
                input_dim=self.n_features,
                num_classes=self.n_classes,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            ).to(device)
            print(f"  Architecture: Input({self.n_features}) -> 256 -> 256 -> 128 -> Output({self.n_classes})")
            print(f"  Dropout: {dropout_rate}, BatchNorm: {use_batch_norm}")
        elif classifier_type.lower() == "random_forest":
            classifier = UnifiedClassifier(
                classifier_type="random_forest",
                input_dim=self.n_features,
                num_classes=self.n_classes,
                device=device
            )
            print(f"  Type: Random Forest")
        elif classifier_type.lower() == "gradient_boosting":
            classifier = UnifiedClassifier(
                classifier_type="gradient_boosting",
                input_dim=self.n_features,
                num_classes=self.n_classes,
                device=device
            )
            print(f"  Type: Gradient Boosting")
        else:
            raise ValueError(
                f"Unknown classifier type: {classifier_type}. "
                f"Supported: 'dnn', 'random_forest', 'gradient_boosting'"
            )
        
        self.classifier = classifier
        return classifier
    
    def train_classifier(
        self,
        classifier: torch.nn.Module,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 1e-4,
        use_lr_scheduler: bool = True,
        device: str = "cpu",
        verbose: bool = True
    ) -> Tuple[torch.nn.Module, float, Dict[str, List[float]]]:
        print(f"\nTraining classifier")
        print("="*80)
        
        classifier_type = "dnn"
        if isinstance(classifier, UnifiedClassifier):
            classifier_type = classifier.classifier_type
        
        if classifier_type == "dnn":
            return self._train_dnn_classifier(
                classifier, epochs, batch_size, lr, patience,
                weight_decay, use_lr_scheduler, device, verbose
            )
        elif classifier_type == "random_forest":
            return self._train_sklearn_classifier(
                classifier, "random_forest", verbose
            )
        elif classifier_type == "gradient_boosting":
            return self._train_sklearn_classifier(
                classifier, "gradient_boosting", verbose
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _train_dnn_classifier(
        self,
        classifier: torch.nn.Module,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int,
        weight_decay: float,
        use_lr_scheduler: bool,
        device: str,
        verbose: bool
    ) -> Tuple[torch.nn.Module, float, Dict[str, List[float]]]:
        optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        scheduler = None
        if use_lr_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=patience//3, min_lr=1e-6
            )
        
        dataset = TensorDataset(
            torch.from_numpy(self.X_train_scaled).float(),
            torch.from_numpy(self.y_train).long()
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        best_test_acc = 0.0
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "test_acc": []}
        
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
            
            classifier.eval()
            with torch.no_grad():
                test_logits = classifier(torch.from_numpy(self.X_test_scaled).float().to(device))
                test_preds = test_logits.argmax(dim=1).cpu().numpy()
                test_acc = accuracy_score(self.y_test, test_preds)
            
            history["train_loss"].append(epoch_loss / len(loader))
            history["test_acc"].append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = classifier.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if scheduler is not None:
                scheduler.step(test_acc)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | "
                      f"Test Acc: {test_acc:.4f} | LR: {current_lr:.2e} | "
                      f"Best: {best_test_acc:.4f}")
            
            if patience_counter >= patience and epoch >= 50:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if best_model_state is not None:
            classifier.load_state_dict(best_model_state)
        
        classifier.eval()
        print(f"\nTraining complete. Best test accuracy: {best_test_acc:.4f}")
        print("="*80)
        
        return classifier, best_test_acc, history
    
    def _train_sklearn_classifier(
        self,
        classifier: UnifiedClassifier,
        classifier_type: str,
        verbose: bool
    ) -> Tuple[torch.nn.Module, float, Dict[str, List[float]]]:
        if verbose:
            print(f"Training {classifier_type} classifier...")
        
        classifier.fit(self.X_train_scaled, self.y_train)
        
        train_preds = classifier.predict(self.X_train_scaled)
        test_preds = classifier.predict(self.X_test_scaled)
        
        train_acc = accuracy_score(self.y_train, train_preds)
        test_acc = accuracy_score(self.y_test, test_preds)
        
        history = {
            "train_acc": [train_acc],
            "test_acc": [test_acc]
        }
        
        if verbose:
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
            print("="*80)
        
        return classifier, test_acc, history
    
    def get_anchor_env_data(self) -> Dict[str, np.ndarray]:
        return {
            "X_unit": self.X_train_unit,
            "X_std": self.X_train_scaled,
            "y": self.y_train,
            "X_test_unit": self.X_test_unit,
            "X_test_std": self.X_test_scaled,
            "y_test": self.y_test,
            "X_min": self.X_min,
            "X_range": self.X_range,
            "feature_names": self.feature_names
        }
    
    def save_classifier(self, classifier: torch.nn.Module, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if isinstance(classifier, UnifiedClassifier) and classifier.classifier_type != "dnn":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(classifier, f)
        else:
            torch.save(classifier.state_dict(), filepath)
        
        print(f"Classifier saved to {filepath}")
    
    def load_classifier(
        self,
        filepath: str,
        classifier_type: str = "dnn",
        device: str = "cpu"
    ) -> torch.nn.Module:
        classifier = self.create_classifier(classifier_type=classifier_type, device=device)
        
        if isinstance(classifier, UnifiedClassifier) and classifier.classifier_type != "dnn":
            import pickle
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
        else:
            classifier.load_state_dict(torch.load(filepath, map_location=device))
        
        classifier.eval()
        print(f"Classifier loaded from {filepath}")
        return classifier

    def get_classifier(self) -> torch.nn.Module:
        if self.classifier is None:
            raise ValueError("Classifier not created yet. Call create_classifier() first.")
        return self.classifier
    
    def perform_eda_analysis(
        self,
        output_dir: str = "./output/eda_output/",
        verbose: bool = True,
        use_ydata_profiling: bool = True,
        minimal: bool = False
    ) -> Dict[str, any]:
        if self.X_train is None or self.X_test is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose:
            print(f"\nPerforming EDA analysis for dataset: {self.dataset_name}")
            print("="*80)
        
        eda_results = {}
        
        if use_ydata_profiling:
            try:
                from ydata_profiling import ProfileReport
                eda_results["ydata_profiling"] = self._perform_ydata_profiling(
                    output_dir, verbose, minimal
                )
            except ImportError:
                if verbose:
                    print("\n⚠ ydata-profiling not installed. Falling back to custom EDA.")
                    print("  Install with: pip install ydata-profiling")
                use_ydata_profiling = False
        
        if not use_ydata_profiling:
            eda_results["dataset_overview"] = self._analyze_dataset_overview(verbose)
            eda_results["feature_statistics"] = self._analyze_feature_statistics(verbose)
            eda_results["class_distribution"] = self._analyze_class_distribution(verbose)
            eda_results["feature_correlations"] = self._analyze_feature_correlations(output_dir, verbose)
            eda_results["class_separability"] = self._analyze_class_separability(verbose)
            eda_results["data_quality"] = self._analyze_data_quality(verbose)
        
        if verbose:
            print("\n" + "="*80)
            print("EDA COMPLETE!")
            print("="*80)
            print(f"Results saved to: {output_dir}")
        
        return eda_results
    
    def _perform_ydata_profiling(
        self,
        output_dir: str,
        verbose: bool,
        minimal: bool
    ) -> Dict[str, any]:
        try:
            import pandas as pd
            from ydata_profiling import ProfileReport
        except ImportError:
            raise ImportError(
                "ydata-profiling is required. Install with: pip install ydata-profiling"
            )
        
        if verbose:
            print("\n" + "="*80)
            print("YDATA PROFILING ANALYSIS")
            print("="*80)
            print("Generating comprehensive EDA report...")
            print("  This may take a few minutes for large datasets...")
        
        df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        df_train['target'] = self.y_train
        
        df_test = pd.DataFrame(self.X_test, columns=self.feature_names)
        df_test['target'] = self.y_test
        
        results = {}
        
        if verbose:
            print("\n1. Generating training data profile...")
        
        profile_train = ProfileReport(
            df_train,
            title=f"{self.dataset_name.replace('_', ' ').title()} Dataset - Training Data EDA",
            explorative=True,
            minimal=minimal,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True}
            },
            interactions={"continuous": True},
            missing_diagrams={
                "bar": True,
                "matrix": True,
                "heatmap": True,
                "dendrogram": True
            },
            duplicates={"head": 10}
        )
        
        train_report_path = os.path.join(output_dir, "training_data_profile.html")
        profile_train.to_file(train_report_path)
        results["training_report_path"] = train_report_path
        
        if verbose:
            print(f"   ✓ Saved training data profile to {train_report_path}")
        
        if verbose:
            print("\n2. Generating test data profile...")
        
        profile_test = ProfileReport(
            df_test,
            title=f"{self.dataset_name.replace('_', ' ').title()} Dataset - Test Data EDA",
            explorative=True,
            minimal=minimal,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True},
                "kendall": {"calculate": True},
                "phi_k": {"calculate": True},
                "cramers": {"calculate": True}
            },
            interactions={"continuous": True},
            missing_diagrams={
                "bar": True,
                "matrix": True,
                "heatmap": True,
                "dendrogram": True
            },
            duplicates={"head": 10}
        )
        
        test_report_path = os.path.join(output_dir, "test_data_profile.html")
        profile_test.to_file(test_report_path)
        results["test_report_path"] = test_report_path
        
        if verbose:
            print(f"   ✓ Saved test data profile to {test_report_path}")
        
        if verbose:
            print("\n3. Extracting key metrics for XAI...")
        
        train_description = profile_train.get_description()
        test_description = profile_test.get_description()
        
        results["training_summary"] = {
            "n_variables": train_description.get("table", {}).get("n_variables", 0),
            "n_observations": train_description.get("table", {}).get("n_observations", 0),
            "n_cells": train_description.get("table", {}).get("n_cells", 0),
            "n_duplicates": train_description.get("table", {}).get("n_duplicates", 0),
            "p_duplicates": train_description.get("table", {}).get("p_duplicates", 0),
            "n_missing": train_description.get("table", {}).get("n_missing", 0),
            "p_missing": train_description.get("table", {}).get("p_missing", 0),
            "memory_size": train_description.get("table", {}).get("memory_size", 0)
        }
        
        results["test_summary"] = {
            "n_variables": test_description.get("table", {}).get("n_variables", 0),
            "n_observations": test_description.get("table", {}).get("n_observations", 0),
            "n_cells": test_description.get("table", {}).get("n_cells", 0),
            "n_duplicates": test_description.get("table", {}).get("n_duplicates", 0),
            "p_duplicates": test_description.get("table", {}).get("p_duplicates", 0),
            "n_missing": test_description.get("table", {}).get("n_missing", 0),
            "p_missing": test_description.get("table", {}).get("p_missing", 0),
            "memory_size": test_description.get("table", {}).get("memory_size", 0)
        }
        
        results["correlations"] = {}
        if "correlations" in train_description:
            results["correlations"]["training"] = train_description["correlations"]
        if "correlations" in test_description:
            results["correlations"]["test"] = test_description["correlations"]
        
        results["alerts"] = {}
        if "alerts" in train_description:
            results["alerts"]["training"] = train_description["alerts"]
        if "alerts" in test_description:
            results["alerts"]["test"] = test_description["alerts"]
        
        if verbose:
            print("\n4. Key Statistics:")
            print(f"   Training samples: {results['training_summary']['n_observations']:,}")
            print(f"   Test samples: {results['test_summary']['n_observations']:,}")
            print(f"   Features: {results['training_summary']['n_variables'] - 1}")
            print(f"   Missing values (train): {results['training_summary']['n_missing']} "
                  f"({results['training_summary']['p_missing']:.2f}%)")
            print(f"   Missing values (test): {results['test_summary']['n_missing']} "
                  f"({results['test_summary']['p_missing']:.2f}%)")
            print(f"   Duplicates (train): {results['training_summary']['n_duplicates']} "
                  f"({results['training_summary']['p_duplicates']:.2f}%)")
            print(f"   Duplicates (test): {results['test_summary']['n_duplicates']} "
                  f"({results['test_summary']['p_duplicates']:.2f}%)")
        
        if verbose:
            print("\n5. Saving JSON summary...")
        
        import json
        summary_path = os.path.join(output_dir, "eda_summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "dataset_name": self.dataset_name,
                "training_summary": results["training_summary"],
                "test_summary": results["test_summary"],
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "feature_names": self.feature_names,
                "class_names": self.class_names
            }, f, indent=2)
        
        results["summary_json_path"] = summary_path
        
        if verbose:
            print(f"   ✓ Saved summary to {summary_path}")
        
        return results
    
    def _analyze_dataset_overview(self, verbose: bool) -> Dict[str, any]:
        overview = {
            "dataset_name": self.dataset_name,
            "n_train_samples": len(self.X_train),
            "n_test_samples": len(self.X_test),
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "class_names": self.class_names
        }
        
        if verbose:
            print("\n" + "="*80)
            print("DATASET OVERVIEW")
            print("="*80)
            print(f"Dataset: {overview['dataset_name']}")
            print(f"Training samples: {overview['n_train_samples']:,}")
            print(f"Test samples: {overview['n_test_samples']:,}")
            print(f"Features: {overview['n_features']}")
            print(f"Classes: {overview['n_classes']}")
            if self.class_names:
                print(f"Class names: {self.class_names}")
        
        return overview
    
    def _analyze_feature_statistics(self, verbose: bool) -> Dict[str, any]:
        stats = {}
        
        for i, feat_name in enumerate(self.feature_names):
            feat_train = self.X_train[:, i]
            feat_test = self.X_test[:, i]
            
            stats[feat_name] = {
                "train_mean": float(np.mean(feat_train)),
                "train_std": float(np.std(feat_train)),
                "train_min": float(np.min(feat_train)),
                "train_max": float(np.max(feat_train)),
                "train_median": float(np.median(feat_train)),
                "test_mean": float(np.mean(feat_test)),
                "test_std": float(np.std(feat_test)),
                "test_min": float(np.min(feat_test)),
                "test_max": float(np.max(feat_test)),
                "test_median": float(np.median(feat_test))
            }
        
        if verbose:
            print("\n" + "="*80)
            print("FEATURE STATISTICS")
            print("="*80)
            print(f"{'Feature':<20} {'Train Mean':<12} {'Train Std':<12} {'Test Mean':<12} {'Test Std':<12}")
            print("-" * 80)
            for feat_name, feat_stats in list(stats.items())[:10]:
                print(f"{feat_name:<20} {feat_stats['train_mean']:>11.4f} "
                      f"{feat_stats['train_std']:>11.4f} {feat_stats['test_mean']:>11.4f} "
                      f"{feat_stats['test_std']:>11.4f}")
            if len(stats) > 10:
                print(f"... and {len(stats) - 10} more features")
        
        return stats
    
    def _analyze_class_distribution(self, verbose: bool) -> Dict[str, any]:
        train_dist = {}
        test_dist = {}
        
        unique_classes = np.unique(self.y_train)
        
        for cls in unique_classes:
            train_count = np.sum(self.y_train == cls)
            test_count = np.sum(self.y_test == cls)
            train_pct = (train_count / len(self.y_train)) * 100
            test_pct = (test_count / len(self.y_test)) * 100
            
            cls_name = self.class_names[cls] if self.class_names and cls < len(self.class_names) else f"Class {cls}"
            
            train_dist[cls] = {
                "name": cls_name,
                "count": int(train_count),
                "percentage": float(train_pct)
            }
            test_dist[cls] = {
                "name": cls_name,
                "count": int(test_count),
                "percentage": float(test_pct)
            }
        
        distribution = {
            "train": train_dist,
            "test": test_dist,
            "is_balanced": self._check_class_balance()
        }
        
        if verbose:
            print("\n" + "="*80)
            print("CLASS DISTRIBUTION")
            print("="*80)
            print("Training set:")
            for cls, info in train_dist.items():
                print(f"  {info['name']}: {info['count']:,} ({info['percentage']:.1f}%)")
            print("\nTest set:")
            for cls, info in test_dist.items():
                print(f"  {info['name']}: {info['count']:,} ({info['percentage']:.1f}%)")
            print(f"\nClass balance: {'Balanced' if distribution['is_balanced'] else 'Imbalanced'}")
        
        return distribution
    
    def _check_class_balance(self) -> bool:
        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            return True
        
        class_counts = [np.sum(self.y_train == cls) for cls in unique_classes]
        max_count = max(class_counts)
        min_count = min(class_counts)
        
        balance_ratio = min_count / max_count
        return balance_ratio > 0.7
    
    def _analyze_feature_correlations(self, output_dir: str, verbose: bool) -> Dict[str, any]:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            if verbose:
                print("\n" + "="*80)
                print("FEATURE CORRELATIONS")
                print("="*80)
                print("Skipping correlation analysis (pandas/matplotlib/seaborn not available)")
            return {}
        
        df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        corr_matrix = df_train.corr()
        
        correlations = {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": []
        }
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    correlations["high_correlations"].append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })
        
        if verbose:
            print("\n" + "="*80)
            print("FEATURE CORRELATIONS")
            print("="*80)
            print(f"High correlations (|r| > 0.7): {len(correlations['high_correlations'])}")
            if correlations["high_correlations"]:
                print("\nTop high correlations:")
                for corr_info in sorted(correlations["high_correlations"], 
                                       key=lambda x: abs(x["correlation"]), 
                                       reverse=True)[:10]:
                    print(f"  {corr_info['feature1']} <-> {corr_info['feature2']}: "
                          f"{corr_info['correlation']:.3f}")
        
        try:
            plt.figure(figsize=(max(12, len(self.feature_names)*0.5), 
                               max(10, len(self.feature_names)*0.5)))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, xticklabels=True, yticklabels=True,
                       cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            plt.savefig(f'{output_dir}correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            if verbose:
                print(f"  ✓ Saved correlation heatmap to {output_dir}correlation_heatmap.png")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not save correlation heatmap: {e}")
        
        return correlations
    
    def _analyze_class_separability(self, verbose: bool) -> Dict[str, any]:
        separability = {
            "feature_importance_scores": {},
            "class_overlap_metrics": {}
        }
        
        unique_classes = np.unique(self.y_train)
        
        for i, feat_name in enumerate(self.feature_names):
            feat_values = self.X_train[:, i]
            
            class_means = {}
            class_stds = {}
            
            for cls in unique_classes:
                mask = self.y_train == cls
                class_values = feat_values[mask]
                class_means[cls] = float(np.mean(class_values))
                class_stds[cls] = float(np.std(class_values))
            
            mean_diff = max(class_means.values()) - min(class_means.values())
            mean_std = np.mean(list(class_stds.values()))
            
            separability_score = mean_diff / (mean_std + 1e-8)
            separability["feature_importance_scores"][feat_name] = float(separability_score)
        
        sorted_features = sorted(separability["feature_importance_scores"].items(), 
                                key=lambda x: x[1], reverse=True)
        
        separability["top_features"] = [feat for feat, score in sorted_features[:10]]
        
        for cls1 in unique_classes:
            for cls2 in unique_classes:
                if cls1 >= cls2:
                    continue
                
                cls1_mask = self.y_train == cls1
                cls2_mask = self.y_train == cls2
                
                cls1_data = self.X_train[cls1_mask]
                cls2_data = self.X_train[cls2_mask]
                
                mean1 = np.mean(cls1_data, axis=0)
                mean2 = np.mean(cls2_data, axis=0)
                
                distance = np.linalg.norm(mean1 - mean2)
                
                cls1_name = self.class_names[cls1] if self.class_names and cls1 < len(self.class_names) else f"Class {cls1}"
                cls2_name = self.class_names[cls2] if self.class_names and cls2 < len(self.class_names) else f"Class {cls2}"
                
                separability["class_overlap_metrics"][f"{cls1_name}_vs_{cls2_name}"] = {
                    "mean_distance": float(distance),
                    "class1": cls1_name,
                    "class2": cls2_name
                }
        
        if verbose:
            print("\n" + "="*80)
            print("CLASS SEPARABILITY")
            print("="*80)
            print("Top 10 features by separability score:")
            for feat, score in sorted_features[:10]:
                print(f"  {feat}: {score:.4f}")
        
        return separability
    
    def _analyze_data_quality(self, verbose: bool) -> Dict[str, any]:
        quality = {
            "missing_values_train": int(np.isnan(self.X_train).sum()),
            "missing_values_test": int(np.isnan(self.X_test).sum()),
            "infinite_values_train": int(np.isinf(self.X_train).sum()),
            "infinite_values_test": int(np.isinf(self.X_test).sum()),
            "outliers": {}
        }
        
        for i, feat_name in enumerate(self.feature_names):
            feat_values = self.X_train[:, i]
            Q1 = np.percentile(feat_values, 25)
            Q3 = np.percentile(feat_values, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((feat_values < lower_bound) | (feat_values > upper_bound))
            outlier_pct = (outliers / len(feat_values)) * 100
            
            quality["outliers"][feat_name] = {
                "count": int(outliers),
                "percentage": float(outlier_pct),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        
        quality["has_missing"] = quality["missing_values_train"] > 0 or quality["missing_values_test"] > 0
        quality["has_infinite"] = quality["infinite_values_train"] > 0 or quality["infinite_values_test"] > 0
        
        if verbose:
            print("\n" + "="*80)
            print("DATA QUALITY")
            print("="*80)
            print(f"Missing values (train): {quality['missing_values_train']}")
            print(f"Missing values (test): {quality['missing_values_test']}")
            print(f"Infinite values (train): {quality['infinite_values_train']}")
            print(f"Infinite values (test): {quality['infinite_values_test']}")
            
            high_outlier_features = [
                (feat, info["percentage"]) 
                for feat, info in quality["outliers"].items() 
                if info["percentage"] > 5.0
            ]
            
            if high_outlier_features:
                print(f"\nFeatures with >5% outliers: {len(high_outlier_features)}")
                for feat, pct in sorted(high_outlier_features, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  {feat}: {pct:.1f}%")
            else:
                print("\n✓ No significant outliers detected (>5%)")
        
        return quality