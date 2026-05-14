"""
WyoDOT Dataset Loader for Dynamic Anchors.

Subclasses TabularDatasetLoader to handle WyoDOT road surface condition datasets
with a Random Forest classifier using paper-specified parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# Add project root and BenchMARL directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "BenchMARL"))

from tabular_datasets import TabularDatasetLoader
from utils.networks import UnifiedClassifier

import logging
logger = logging.getLogger(__name__)

# Directory containing the CSV files (same directory as this script)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class WyoDOTDatasetLoader(TabularDatasetLoader):
    """Dataset loader for WyoDOT road surface condition datasets."""

    DATASETS = {
        "wyodot_kvdw_labeled": {
            "file": "KVDW_labeled.csv",
            "label_col": "label",
            "drop_cols": [],
            "label_map": {},
        },
        "wyodot_testbed": {
            "file": "merged_dataset_new.csv",
            "label_col": "label",
            "drop_cols": ["timestamp"],
            # Per RF_Anchors.pdf: Snow/Frost → Snow, Moist → Wet, Error → drop
            "label_map": {"Snow/Frost": "Snow", "Moist": "Wet"},
            "drop_labels": ["Error"],
        },
    }

    # Random Forest parameters from RF_Anchors.pdf
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        print(f"\nLoading WyoDOT dataset: {self.dataset_name}")
        print("=" * 80)

        if self.dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown WyoDOT dataset: {self.dataset_name}. "
                f"Supported: {list(self.DATASETS.keys())}"
            )

        config = self.DATASETS[self.dataset_name]
        csv_path = os.path.join(DATA_DIR, config["file"])
        df = pd.read_csv(csv_path)
        logger.info(f"  Loaded {len(df)} rows from {config['file']}")

        # Drop non-feature columns
        for col in config["drop_cols"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        label_col = config["label_col"]

        # Remap labels (e.g., Snow/Frost → Snow, Moist → Wet)
        label_map = config.get("label_map", {})
        if label_map:
            df[label_col] = df[label_col].replace(label_map)
            logger.info(f"  Label remapping applied: {label_map}")

        # Drop rows with specific labels (e.g., Error)
        drop_labels = config.get("drop_labels", [])
        if drop_labels:
            before = len(df)
            df = df[~df[label_col].isin(drop_labels)]
            logger.info(f"  Dropped {before - len(df)} rows with labels: {drop_labels}")

        # Drop rows with NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            before = len(df)
            df = df.dropna()
            logger.info(f"  Dropped {before - len(df)} rows with NaN values ({nan_count} total NaNs)")

        # Separate features and labels
        y_raw = df[label_col].values
        X = df.drop(columns=[label_col]).values.astype(np.float32)
        feature_names = [c for c in df.columns if c != label_col]

        # Encode string labels to integers
        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(int)
        class_names = list(le.classes_)

        logger.info(f"  Features: {feature_names}")
        logger.info(f"  Classes: {class_names}")

        # Sample if requested
        if self.sample_size is not None and self.sample_size < len(X):
            indices = np.random.RandomState(self.random_state).choice(
                len(X), size=self.sample_size, replace=False
            )
            X = X[indices]
            y = y[indices]
            logger.info(f"  Sampled {self.sample_size} instances")

        # Train/test split (stratified)
        stratify = y if len(np.unique(y)) < 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        # Set instance attributes (matches parent contract exactly)
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.astype(int)
        self.y_test = y_test.astype(int)
        self.feature_names = feature_names
        self.class_names = class_names
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y))

        logger.info(f"Dataset loaded:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Classes: {self.n_classes}")
        logger.info(f"  Class distribution (train): {np.bincount(y_train)}")
        logger.info(f"  Class distribution (test): {np.bincount(y_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test, feature_names, class_names

    def create_classifier(
        self,
        classifier_type: str = "random_forest",
        hidden_size: int = 256,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        device: str = "cpu",
        hidden_sizes: Optional[List[int]] = None
    ):
        if classifier_type.lower() == "random_forest":
            rf_params = {**self.RF_PARAMS, "random_state": self.random_state}
            logger.info(f"\nCreating Random Forest classifier with WyoDOT paper parameters")
            logger.info("=" * 80)
            logger.info(f"  Parameters: {rf_params}")

            classifier = UnifiedClassifier(
                classifier_type="random_forest",
                input_dim=self.n_features,
                num_classes=self.n_classes,
                device=device,
                **rf_params,
            )

            self.classifier = classifier
            return classifier
        else:
            # Delegate to parent for DNN / gradient_boosting
            return super().create_classifier(
                classifier_type=classifier_type,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                device=device,
                hidden_sizes=hidden_sizes,
            )

    # load_classifier is handled by the parent — it auto-detects pickle vs. torch.save
    # by reading the file's magic bytes, so no override is needed.
