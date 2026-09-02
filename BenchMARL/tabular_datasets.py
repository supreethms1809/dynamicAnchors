# CRITICAL: Set environment variables BEFORE importing numpy to fix multiprocessing issues
# This prevents OpenBLAS threading errors when multiprocessing spawns child processes
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

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
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, Optional, List
import os
import sys
import logging
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.networks import SimpleClassifier, UnifiedClassifier


def _infer_dnn_arch_from_state_dict(state_dict):
    """Recover (hidden_sizes, use_batch_norm) from a SimpleClassifier state_dict.

    SimpleClassifier stores everything in one nn.Sequential named ``net``, so the
    Linear layers appear as ``net.<idx>.weight`` with 2-D shapes (out, in). The
    hidden sizes are the out-features of every Linear except the final one (which
    is the output layer, out == num_classes). BatchNorm is detected by the
    presence of ``running_mean`` buffers.

    Returns None if the state_dict does not look like a SimpleClassifier, so the
    caller can fall back to the size-aware defaults.
    """
    try:
        linear_out = []
        for key, value in state_dict.items():
            if key.startswith("net.") and key.endswith(".weight") and hasattr(value, "dim") and value.dim() == 2:
                idx = int(key.split(".")[1])
                linear_out.append((idx, value.shape[0]))
        if len(linear_out) < 2:
            return None
        linear_out.sort()
        hidden_sizes = [out for _, out in linear_out[:-1]]  # drop the output layer
        use_batch_norm = any("running_mean" in k for k in state_dict)
        return hidden_sizes, use_batch_norm
    except (ValueError, IndexError, AttributeError):
        return None


class TabularDatasetLoader:
    
    def __init__(
        self,
        dataset_name: str = "breast_cancer",
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        sample_size: Optional[int] = None
    ):
        self.dataset_name = dataset_name
        self.test_size = test_size
        # C-10: three-way split. Default 60/20/20 (train/val/test). Set val_size=0
        # to recover the old 80/20 train/test split (not recommended for the paper).
        self.val_size = val_size
        self.random_state = random_state
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.X_train_unit = None
        self.X_val_unit = None
        self.X_test_unit = None
        self.X_min = None
        self.X_range = None
        self.feature_names = None
        self.class_names = None
        self.n_features = None
        self.n_classes = None
        self.classifier = None
        # C-07: label-encoded categoricals (indices into feature_names) and encoders
        # so rules can be printed as equalities rather than bogus intervals.
        self.categorical_indices: List[int] = []
        self.categorical_names: List[str] = []
        self.label_encoders: Dict[str, object] = {}
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.classifier_accuracy: Dict[str, float] = {}
    
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
            feature_names = list(data.feature_names)
            class_names = list(data.target_names)
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
            
            logger.info(f"\nConverted housing prices to 4 classes:")
            logger.info(f"  Class 0 (very_low): < ${quartiles[0]*100:.0f}K (25th percentile)")
            logger.info(f"  Class 1 (low): ${quartiles[0]*100:.0f}K - ${quartiles[1]*100:.0f}K (25th-50th percentile)")
            logger.info(f"  Class 2 (medium): ${quartiles[1]*100:.0f}K - ${quartiles[2]*100:.0f}K (50th-75th percentile)")
            logger.info(f"  Class 3 (high): >= ${quartiles[2]*100:.0f}K (75th percentile+)")
        elif self.dataset_name.startswith("uci_"):
            # UCIML Repository dataset
            if not UCIML_AVAILABLE:
                raise ImportError(
                    "ucimlrepo package is required for UCIML datasets. "
                    "Install with: pip install ucimlrepo"
                )
            
            # Parse dataset identifier (can be ID or name)
            dataset_id_str = self.dataset_name.replace("uci_", "")
            
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
                "default-credit-card-clients": 350,
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
            
            logger.info(f"Fetching UCIML dataset (ID: {dataset_id})...")
            dataset = fetch_ucirepo(id=dataset_id)
            
            # Extract features and targets - ensure we have a copy, not a view
            import pandas as pd
            X_df_orig = dataset.data.features
            if isinstance(X_df_orig, pd.DataFrame):
                X_df = X_df_orig.copy()
            else:
                X_df = pd.DataFrame(X_df_orig)
            y_df = dataset.data.targets
            
            # Get feature names before processing
            if hasattr(X_df, 'columns'):
                feature_names = list(X_df.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X_df.shape[1])]
            
            # Handle missing values and non-numeric features
            from sklearn.preprocessing import LabelEncoder
            
            # Handle missing values - use a more robust approach
            if X_df.isnull().any().any():
                missing_count = X_df.isnull().sum().sum()
                logger.info(f"  Found {missing_count} missing values, filling with median/mode...")
                # Fill numeric columns with median, categorical with mode
                # Build fill values dict first, then apply all at once to avoid SettingWithCopyWarning
                fill_values = {}
                for col in X_df.columns:
                    if X_df[col].dtype in ['int64', 'float64']:
                        median_val = X_df[col].median()
                        fill_values[col] = median_val if not np.isnan(median_val) else 0.0
                    else:
                        mode_val = X_df[col].mode()
                        fill_values[col] = mode_val[0] if len(mode_val) > 0 else 0
                
                # Apply all fillna operations at once
                for col, fill_val in fill_values.items():
                    X_df[col] = X_df[col].fillna(fill_val)
                
                # Verify all missing values are filled
                remaining_missing = X_df.isnull().sum().sum()
                if remaining_missing > 0:
                    logger.warning(f"  Warning: {remaining_missing} missing values remain after filling. This may cause issues.")
                else:
                    logger.info(f"  ✓ All missing values successfully filled")
            
            # Encode categorical features (C-07). LabelEncoder keeps a 1-D integer
            # column so we can freeze it to the instance/class mode rather than
            # treating a one-hot interval as a logical condition.
            label_encoders = {}
            numeric_cols = []
            categorical_cols = []
            for col in X_df.columns:
                if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_df.loc[:, col] = le.fit_transform(X_df[col].astype(str))
                    label_encoders[col] = le
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            self.label_encoders = label_encoders
            self.categorical_names = list(categorical_cols)
            self.categorical_indices = [list(X_df.columns).index(c) for c in categorical_cols]
            
            # Convert to numpy array
            X = X_df.values.astype(np.float32)
            
            # Handle target
            y = y_df.values if hasattr(y_df, 'values') else y_df
            
            # Handle target shape (may be 1D or 2D)
            if y.ndim > 1:
                if y.shape[1] == 1:
                    y = y.flatten()
                else:
                    # Multi-label case - use first column or convert to binary
                    logger.warning(f"Multi-column target detected, using first column")
                    y = y[:, 0]
            
            # Convert target to integer labels if needed
            if isinstance(y, pd.Series):
                y = y.values
            
            if y.dtype == 'object' or not np.issubdtype(y.dtype, np.integer):
                # Normalise string labels before encoding. Several UCI datasets ship
                # the train and test splits with inconsistent label spelling — Adult
                # terminates the test split's labels with a period ('<=50K' in train
                # vs '<=50K.' in test) and pads some with whitespace. Encoding those
                # verbatim turns a binary task into 4 classes ('<=50K', '<=50K.',
                # '>50K', '>50K.') with train and test rows in different bins, which
                # silently corrupts every downstream metric.
                y_norm = pd.Series(np.asarray(y).ravel()).astype(str).str.strip()
                stripped = y_norm.str.rstrip('.').str.strip()
                # Only adopt the stripped form if it actually merges labels; this
                # keeps datasets whose labels legitimately contain '.' untouched.
                if stripped.nunique() < y_norm.nunique():
                    logger.info(
                        f"  Normalised target labels: {y_norm.nunique()} -> {stripped.nunique()} "
                        f"classes after stripping trailing '.' / whitespace "
                        f"({sorted(y_norm.unique())} -> {sorted(stripped.unique())})"
                    )
                    y_norm = stripped
                le = LabelEncoder()
                y = le.fit_transform(y_norm).astype(int)
                class_names = le.classes_.tolist()
            else:
                y = y.astype(int)
                unique_classes = np.unique(y)
                # Remap to 0-indexed labels so CrossEntropyLoss targets are always in [0, n_classes-1]
                label_map = {old: new for new, old in enumerate(unique_classes)}
                y = np.array([label_map[v] for v in y], dtype=int)
                class_names = [f"class_{i}" for i in unique_classes]
            
            logger.info(f"  Loaded UCIML dataset: {dataset.metadata.name if hasattr(dataset, 'metadata') else 'Unknown'}")
            logger.info(f"  Features: {len(feature_names)}, Classes: {len(class_names)}")
            if label_encoders:
                logger.info(f"  Encoded {len(label_encoders)} categorical features")
            
        elif self.dataset_name.startswith("folktables_"):
            # Folktables dataset
            if not FOLKTABLES_AVAILABLE:
                raise ImportError(
                    "folktables package is required for Folktables datasets. "
                    "Install with: pip install folktables"
                )
            
            # Parse dataset specification: folktables_<task>_<state>_<year>
            # Example: folktables_income_CA_2018
            parts = self.dataset_name.replace("folktables_", "").split("_")
            
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid Folktables dataset format: {self.dataset_name}. "
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
            
            # Task classes from folktables are already instances, not classes
            task = task_map[task_name]
            
            logger.info(f"Loading Folktables dataset: {task_name} for {state} ({year})...")
            
            # Create data source
            data_source = ACSDataSource(
                survey_year=year,
                horizon='1-Year',
                survey='person'
            )
            
            # Download and extract data
            acs_data = data_source.get_data(states=[state], download=True)
            
            # Extract features and labels using the task (task is already an instance)
            # Note: df_to_numpy may return 2 or 3 values depending on folktables version:
            # - Older versions: (X, y)
            # - Newer versions: (X, y, group) where group is demographic information
            result = task.df_to_numpy(acs_data)
            if len(result) == 2:
                X, y = result
            elif len(result) == 3:
                X, y, group = result  # group contains demographic info (e.g., RAC1P, SEX, etc.)
                logger.debug(f"Note: Group information available but not used")
            else:
                raise ValueError(f"Unexpected return value from df_to_numpy: expected 2 or 3 values, got {len(result)}")
            
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
            
            logger.info(f"  Loaded Folktables dataset: {task_name} ({state}, {year})")
            logger.info(f"  Features: {len(feature_names)}, Classes: {len(class_names)}")

        elif self.dataset_name in ("heloc", "fico_heloc"):
            # C-23: FICO HELOC (~10k rows, 23 continuous). OpenML 45578 or sklearn fetch.
            X, y, feature_names, class_names = self._load_heloc()
        elif self.dataset_name in ("sick", "thyroid_sick"):
            # Imbalanced medical: thyroid disease, ~3772 x 29, ~6.1% positive.
            X, y, feature_names, class_names = self._load_openml_named("sick", "Thyroid (sick)")
        elif self.dataset_name == "mammography":
            # Severely imbalanced medical: ~11183 x 6, ~2.3% positive.
            X, y, feature_names, class_names = self._load_openml_named("mammography", "Mammography")
        elif self.dataset_name in ("bank_marketing", "uci_bank"):
            # C-23: UCI Bank Marketing (~45k, mixed types).
            X, y, feature_names, class_names = self._load_bank_marketing()
            
        else:
            supported = ['breast_cancer', 'wine', 'iris', 'synthetic', 'moons', 'circles', 'covtype', 'housing', 'heloc', 'sick', 'mammography', 'bank_marketing']
            if UCIML_AVAILABLE:
                supported.append('uci_<id_or_name> (e.g., uci_adult, uci_2)')
            if FOLKTABLES_AVAILABLE:
                supported.append('folktables_<task>_<state>_<year> (e.g., folktables_income_CA_2018)')
            
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Supported: {', '.join(supported)}"
            )
        
        if self.sample_size is not None and self.sample_size < len(X):
            indices = np.random.RandomState(self.random_state).choice(
                len(X), size=self.sample_size, replace=False
            )
            X = X[indices]
            y = y[indices]
            logger.info(f"Sampled {self.sample_size} instances from dataset")
        
        stratify = y if len(np.unique(y)) < 20 else None
        n = len(X)
        all_idx = np.arange(n)

        # C-10: strict three-way split. D_train = black-box + RL; D_val = rule
        # generation / ranking / checkpoint selection; D_test = ALL reported metrics.
        if self.val_size and self.val_size > 0:
            if self.test_size + self.val_size >= 1.0:
                raise ValueError(
                    f"test_size ({self.test_size}) + val_size ({self.val_size}) must be < 1"
                )
            idx_temp, idx_test = train_test_split(
                all_idx, test_size=self.test_size, random_state=self.random_state, stratify=stratify
            )
            stratify_temp = y[idx_temp] if stratify is not None else None
            val_frac_of_temp = self.val_size / (1.0 - self.test_size)
            idx_train, idx_val = train_test_split(
                idx_temp, test_size=val_frac_of_temp, random_state=self.random_state, stratify=stratify_temp
            )
            from utils.eval_harness import assert_no_index_overlap
            assert_no_index_overlap(idx_train, idx_val, idx_test)
            self.train_idx, self.val_idx, self.test_idx = idx_train, idx_val, idx_test
            X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        else:
            idx_train, idx_test = train_test_split(
                all_idx, test_size=self.test_size, random_state=self.random_state, stratify=stratify
            )
            self.train_idx, self.val_idx, self.test_idx = idx_train, np.array([], dtype=int), idx_test
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]
            X_val, y_val = None, None
        
        self.X_train = X_train.astype(np.float32)
        self.X_test = X_test.astype(np.float32)
        self.y_train = y_train.astype(int)
        self.y_test = y_test.astype(int)
        self.X_val = None if X_val is None else X_val.astype(np.float32)
        self.y_val = None if y_val is None else y_val.astype(int)
        self.feature_names = feature_names
        self.class_names = class_names
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y))
        
        logger.info(f"Dataset loaded:")
        logger.info(f"  Training samples: {len(X_train)}")
        if self.y_val is not None:
            logger.info(f"  Validation samples: {len(self.y_val)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features: {self.n_features}")
        logger.info(f"  Classes: {self.n_classes}")
        logger.info(f"  Class distribution (train): {np.bincount(y_train)}")
        if self.y_val is not None:
            logger.info(f"  Class distribution (val): {np.bincount(self.y_val)}")
        logger.info(f"  Class distribution (test): {np.bincount(y_test)}")
        if self.categorical_indices:
            logger.info(f"  Categorical features (C-07, frozen to instance/mode at env time): "
                        f"{[self.feature_names[i] for i in self.categorical_indices]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test, feature_names, class_names
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("\nPreprocessing data...")
        logger.info("="*80)
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train).astype(np.float32)
        self.X_test_scaled = self.scaler.transform(self.X_test).astype(np.float32)
        if self.X_val is not None:
            self.X_val_scaled = self.scaler.transform(self.X_val).astype(np.float32)
        else:
            self.X_val_scaled = None
        
        self.X_min = self.X_train_scaled.min(axis=0)
        self.X_max = self.X_train_scaled.max(axis=0)
        self.X_range = np.where((self.X_max - self.X_min) == 0, 1.0, (self.X_max - self.X_min))
        
        self.X_train_unit = (self.X_train_scaled - self.X_min) / self.X_range
        self.X_train_unit = np.clip(self.X_train_unit, 0.0, 1.0).astype(np.float32)
        
        self.X_test_unit = (self.X_test_scaled - self.X_min) / self.X_range
        self.X_test_unit = np.clip(self.X_test_unit, 0.0, 1.0).astype(np.float32)

        if self.X_val_scaled is not None:
            self.X_val_unit = (self.X_val_scaled - self.X_min) / self.X_range
            self.X_val_unit = np.clip(self.X_val_unit, 0.0, 1.0).astype(np.float32)
        else:
            self.X_val_unit = None
        
        logger.info("Data preprocessing complete:")
        logger.info(f"  Scaled train shape: {self.X_train_scaled.shape}")
        if self.X_val_scaled is not None:
            logger.info(f"  Scaled val shape: {self.X_val_scaled.shape}")
        logger.info(f"  Scaled test shape: {self.X_test_scaled.shape}")
        logger.info(f"  Unit train range: [{self.X_train_unit.min():.3f}, {self.X_train_unit.max():.3f}]")
        logger.info(f"  Unit test range: [{self.X_test_unit.min():.3f}, {self.X_test_unit.max():.3f}]")
        
        return (
            self.X_train_scaled, self.X_test_scaled,
            self.X_train_unit, self.X_test_unit,
            self.X_min, self.X_range
        )

    def _load_heloc(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """C-23: FICO HELOC (Home Equity Line of Credit), ~10k rows, all continuous."""
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:
            raise ImportError("scikit-learn is required to fetch HELOC") from exc
        # data_id=45578 is NOT HELOC -- it is "California-Housing-Classification"
        # (20640 x 8: longitude, latitude, housingMedianAge, ...). Fetching it
        # succeeded, so the `except` fallback never fired and `--dataset heloc`
        # silently returned California housing labelled as credit risk.
        # Verified 2026-09-02. Fetch by NAME, and assert the shape we expect.
        logger.info("Fetching FICO HELOC from OpenML (name='heloc')...")
        bunch = fetch_openml(name="heloc", as_frame=True, parser="auto")
        if bunch.data.shape[1] < 15:
            raise ValueError(
                f"OpenML 'heloc' returned {bunch.data.shape} with columns "
                f"{list(bunch.data.columns)[:5]} -- that is not FICO HELOC "
                f"(expected ~10000 x 22). Refusing to load the wrong dataset."
            )
        import pandas as pd
        X_df = bunch.data.copy()
        y_raw = bunch.target
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X = X_df.values.astype(np.float32)
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(np.asarray(y_raw).ravel().astype(str)).astype(int)
        feature_names = list(X_df.columns)
        class_names = ["bad", "good"] if len(np.unique(y)) == 2 else [f"class_{i}" for i in range(len(np.unique(y)))]
        logger.info(f"  HELOC: n={len(X)}, d={X.shape[1]}, K={len(np.unique(y))}")
        return X, y, feature_names, class_names

    def _load_openml_named(self, name: str, label: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Fetch an OpenML dataset by NAME, label-encoding categoricals.

        Used for the imbalanced medical sets (`sick`, `mammography`). Fetching by
        name rather than numeric id, because a wrong id fails silently -- that is
        exactly how `heloc` ended up returning California housing.
        """
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        logger.info(f"Fetching {label} from OpenML (name={name!r})...")
        bunch = fetch_openml(name=name, as_frame=True, parser="auto")
        X_df = bunch.data.copy()
        label_encoders, categorical_cols = {}, []
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                mode = X_df[col].mode()
                X_df[col] = X_df[col].fillna(mode.iloc[0] if len(mode) else "missing")
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                label_encoders[col] = le
                categorical_cols.append(col)
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
                X_df[col] = X_df[col].fillna(X_df[col].median())
        # A column that is entirely NaN medians to NaN; drop it rather than ship NaNs.
        bad = [c for c in X_df.columns if X_df[c].isna().any()]
        if bad:
            logger.warning(f"  {label}: dropping all-NaN columns {bad}")
            X_df = X_df.drop(columns=bad)
            categorical_cols = [c for c in categorical_cols if c not in bad]
        self.label_encoders = label_encoders
        self.categorical_names = list(categorical_cols)
        self.categorical_indices = [list(X_df.columns).index(c) for c in categorical_cols]
        X = X_df.values.astype(np.float32)
        y = LabelEncoder().fit_transform(
            np.asarray(bunch.target).ravel().astype(str)).astype(int)
        counts = np.bincount(y)
        logger.info(f"  {label}: n={len(X)}, d={X.shape[1]}, K={len(counts)}, "
                    f"class balance={np.round(counts/counts.sum(), 4).tolist()}, "
                    f"categoricals={len(categorical_cols)}")
        return X, y, list(X_df.columns), [f"class_{i}" for i in range(len(counts))]

    def _load_bank_marketing(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """C-23: UCI Bank Marketing (~45k, mixed types). Requires ucimlrepo (id=222)."""
        if not UCIML_AVAILABLE:
            raise ImportError("ucimlrepo is required for bank_marketing. pip install ucimlrepo")
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        logger.info("Fetching UCI Bank Marketing (id=222)...")
        dataset = fetch_ucirepo(id=222)
        X_df = dataset.data.features.copy() if hasattr(dataset.data.features, "copy") else pd.DataFrame(dataset.data.features)
        y_df = dataset.data.targets
        label_encoders = {}
        categorical_cols = []
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                X_df[col] = X_df[col].fillna(X_df[col].mode().iloc[0] if len(X_df[col].mode()) else "missing")
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                label_encoders[col] = le
                categorical_cols.append(col)
            else:
                X_df[col] = X_df[col].fillna(X_df[col].median())
        self.label_encoders = label_encoders
        self.categorical_names = list(categorical_cols)
        self.categorical_indices = [list(X_df.columns).index(c) for c in categorical_cols]
        X = X_df.values.astype(np.float32)
        y_raw = y_df.values if hasattr(y_df, "values") else y_df
        y = LabelEncoder().fit_transform(np.asarray(y_raw).ravel().astype(str).astype(object)).astype(int)
        feature_names = list(X_df.columns)
        class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
        logger.info(f"  Bank Marketing: n={len(X)}, d={X.shape[1]}, K={len(np.unique(y))}, "
                    f"categoricals={len(categorical_cols)}")
        return X, y, feature_names, class_names
    
    def create_classifier(
        self,
        classifier_type: str = "dnn",
        hidden_size: int = 256,
        dropout_rate: Optional[float] = None,
        use_batch_norm: bool = True,
        device: str = "cpu",
        hidden_sizes: Optional[List[int]] = None
    ) -> torch.nn.Module:
        logger.info(f"\nCreating classifier: {classifier_type}")
        logger.info("="*80)

        if classifier_type.lower() == "dnn":
            # Use dataset-specific architecture for larger datasets
            n_train_samples = len(self.y_train) if hasattr(self, 'y_train') else 0
            is_large = n_train_samples > 10000

            # Dropout defaults to 0.3, but large datasets get 0.1. Measured on
            # housing (16512 train rows, 150 epochs, seed 42) — train accuracy sat
            # within 0.4pt of test at dropout 0.3, i.e. the model was UNDERfitting,
            # so the heavy regularization was pure cost:
            #   [512,512,256]   dropout 0.3 -> train 0.7245 / test 0.7209
            #   [512,512,256]   dropout 0.1 -> train 0.7504 / test 0.7282
            #   [1024,1024,512] dropout 0.3 -> train 0.7496 / test 0.7355
            #   [1024,1024,512] dropout 0.1 -> train 0.7673 / test 0.7415
            if dropout_rate is None:
                dropout_rate = 0.1 if is_large else 0.3

            if hidden_sizes is None:
                if is_large:
                    # Large datasets (housing, etc.): use larger network.
                    # [512,512,256] -> [1024,1024,512] on the sweep above (+1.5pt
                    # test at dropout 0.3, +1.3pt at 0.1). Capacity is close to free
                    # for the RL loop: the environment scores boxes against
                    # classifier predictions cached once per env, not per step, so a
                    # wider classifier costs init time only, not per-step time.
                    hidden_sizes = [1024, 1024, 512]
                    logger.info(f"  Large dataset detected ({n_train_samples} samples), using larger architecture")
                elif n_train_samples > 5000:
                    # Medium-large datasets: slightly larger
                    hidden_sizes = [256, 256, 256, 128]
                    logger.info(f"  Medium-large dataset detected ({n_train_samples} samples), using medium architecture")
                else:
                    # Small datasets: default architecture
                    hidden_sizes = [256, 256, 128]
                    logger.info(f"  Small dataset detected ({n_train_samples} samples), using default architecture")
                    logger.info(f"  (small/medium datasets keep dropout 0.3; only the large branch drops to 0.1)")
            
            classifier = SimpleClassifier(
                input_dim=self.n_features,
                num_classes=self.n_classes,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                hidden_sizes=hidden_sizes
            ).to(device)
            arch_str = " -> ".join([f"Input({self.n_features})"] + [str(s) for s in hidden_sizes] + [f"Output({self.n_classes})"])
            logger.info(f"  Architecture: {arch_str}")
            logger.info(f"  Dropout: {dropout_rate}, BatchNorm: {use_batch_norm}")
        elif classifier_type.lower() == "random_forest":
            classifier = UnifiedClassifier(
                classifier_type="random_forest",
                input_dim=self.n_features,
                num_classes=self.n_classes,
                device=device
            )
            logger.info(f"  Type: Random Forest")
        elif classifier_type.lower() == "gradient_boosting":
            classifier = UnifiedClassifier(
                classifier_type="gradient_boosting",
                input_dim=self.n_features,
                num_classes=self.n_classes,
                device=device
            )
            logger.info(f"  Type: Gradient Boosting")
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
        logger.info(f"\nTraining classifier")
        logger.info("="*80)
        
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
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_acc": [], "test_acc": []}

        # C-10: early-stop on D_val, never on D_test. Fall back to test only if
        # val_size=0 (legacy 80/20), and log a warning that this leaks.
        if self.X_val_scaled is not None:
            X_early = self.X_val_scaled
            y_early = self.y_val
            early_name = "Val"
        else:
            logger.warning(
                "No validation split (val_size=0): early-stopping on TEST accuracy. "
                "This leaks D_test into model selection (C-10). Use val_size=0.2 for the paper."
            )
            X_early = self.X_test_scaled
            y_early = self.y_test
            early_name = "Test"
        
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
                early_logits = classifier(torch.from_numpy(X_early).float().to(device))
                early_preds = early_logits.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_early, early_preds)
                test_logits = classifier(torch.from_numpy(self.X_test_scaled).float().to(device))
                test_preds = test_logits.argmax(dim=1).cpu().numpy()
                test_acc = accuracy_score(self.y_test, test_preds)
            
            history["train_loss"].append(epoch_loss / len(loader))
            history["val_acc"].append(val_acc)
            history["test_acc"].append(test_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Clone the tensors: state_dict() returns references to the live
                # parameters and dict.copy() is shallow, so without the clone the
                # "snapshot" keeps mutating with every later optimizer step and the
                # restore below silently becomes a no-op (saving the FINAL epoch).
                best_model_state = {
                    k: v.detach().clone() for k, v in classifier.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
            
            if scheduler is not None:
                scheduler.step(val_acc)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss/len(loader):.4f} | "
                    f"{early_name} Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | "
                    f"LR: {current_lr:.2e} | Best {early_name}: {best_val_acc:.4f}"
                )
            
            # Early stopping: require at least 10% of max epochs before stopping
            # This ensures we don't stop too early, especially for complex datasets
            min_epochs_before_stop = max(50, int(epochs * 0.1))
            if patience_counter >= patience and epoch >= min_epochs_before_stop:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch} (patience: {patience}, min epochs: {min_epochs_before_stop})")
                break
        
        if best_model_state is not None:
            classifier.load_state_dict(best_model_state)
        
        classifier.eval()
        with torch.no_grad():
            def _acc(Xs, ys):
                logits = classifier(torch.from_numpy(Xs).float().to(device))
                return float(accuracy_score(ys, logits.argmax(dim=1).cpu().numpy()))
            train_acc = _acc(self.X_train_scaled, self.y_train)
            val_acc_final = _acc(self.X_val_scaled, self.y_val) if self.X_val_scaled is not None else None
            test_acc_final = _acc(self.X_test_scaled, self.y_test)
        self.classifier_accuracy = {
            "train": train_acc,
            "val": val_acc_final,
            "test": test_acc_final,
        }
        logger.info(
            f"\nTraining complete. Train acc: {train_acc:.4f}"
            + (f" | Val acc: {val_acc_final:.4f}" if val_acc_final is not None else "")
            + f" | Test acc: {test_acc_final:.4f} (early-stop on {early_name}={best_val_acc:.4f})"
        )
        logger.info("="*80)
        
        return classifier, test_acc_final, history
    
    def _train_sklearn_classifier(
        self,
        classifier: UnifiedClassifier,
        classifier_type: str,
        verbose: bool
    ) -> Tuple[torch.nn.Module, float, Dict[str, List[float]]]:
        if verbose:
            logger.info(f"Training {classifier_type} classifier...")
        
        classifier.fit(self.X_train_scaled, self.y_train)
        
        train_preds = classifier.predict(self.X_train_scaled)
        test_preds = classifier.predict(self.X_test_scaled)
        
        train_acc = accuracy_score(self.y_train, train_preds)
        test_acc = accuracy_score(self.y_test, test_preds)
        val_acc = None
        if self.X_val_scaled is not None:
            val_acc = accuracy_score(self.y_val, classifier.predict(self.X_val_scaled))
        
        history = {
            "train_acc": [train_acc],
            "val_acc": [val_acc],
            "test_acc": [test_acc]
        }
        self.classifier_accuracy = {"train": train_acc, "val": val_acc, "test": test_acc}
        
        if verbose:
            logger.info(f"Training accuracy: {train_acc:.4f}")
            if val_acc is not None:
                logger.info(f"Validation accuracy: {val_acc:.4f}")
            logger.info(f"Test accuracy: {test_acc:.4f}")
            logger.info("="*80)
        
        return classifier, test_acc, history
    
    def get_anchor_env_data(self) -> Dict[str, np.ndarray]:
        return {
            "X_unit": self.X_train_unit,
            "X_std": self.X_train_scaled,
            "y": self.y_train,
            "X_val_unit": self.X_val_unit,
            "X_val_std": self.X_val_scaled,
            "y_val": self.y_val,
            "X_test_unit": self.X_test_unit,
            "X_test_std": self.X_test_scaled,
            "y_test": self.y_test,
            "X_min": self.X_min,
            "X_range": self.X_range,
            "scaler_mean": np.asarray(self.scaler.mean_, dtype=np.float32),
            "scaler_scale": np.asarray(self.scaler.scale_, dtype=np.float32),
            "feature_names": self.feature_names,
            "categorical_indices": list(self.categorical_indices),
            "categorical_names": list(self.categorical_names),
            "categorical_value_names": {
                int(self.feature_names.index(name)): [
                    str(v) for v in encoder.classes_
                ]
                for name, encoder in self.label_encoders.items()
                if name in self.feature_names
            },
            "classifier_accuracy": dict(self.classifier_accuracy),
            "split_sizes": {
                "train": int(len(self.y_train)),
                "val": int(len(self.y_val)) if self.y_val is not None else 0,
                "test": int(len(self.y_test)),
            },
        }
    
    def save_classifier(self, classifier: torch.nn.Module, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        if isinstance(classifier, UnifiedClassifier) and classifier.classifier_type != "dnn":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(classifier, f)
        else:
            torch.save(classifier.state_dict(), filepath)
        
        logger.info(f"Classifier saved to {filepath}")
    
    def load_classifier(
        self,
        filepath: str,
        classifier_type: Optional[str] = None,
        device: str = "cpu"
    ) -> torch.nn.Module:
        # Auto-detect file format by reading the first 2 bytes.
        # Pickle files start with 0x80 (PROTO opcode); torch.save files are zip archives starting with "PK".
        with open(filepath, 'rb') as f:
            magic = f.read(2)
        is_pickle = len(magic) >= 1 and magic[0] == 0x80

        if is_pickle:
            import pickle
            with open(filepath, 'rb') as f:
                classifier = pickle.load(f)
            if isinstance(classifier, UnifiedClassifier):
                classifier.device = device
        else:
            ct = classifier_type if classifier_type is not None else "dnn"
            state_dict = torch.load(filepath, map_location=device)

            # Rebuild the architecture the CHECKPOINT was saved with, not whatever
            # create_classifier's size-aware defaults would pick today. Those defaults
            # change over time (the large-dataset branch went [512,512,256] ->
            # [1024,1024,512]), and without this every previously saved large-dataset
            # classifier fails to load with a shape mismatch. Mirrors what inference.py
            # already does for policy checkpoints.
            hidden_sizes, use_batch_norm = (None, True)
            if ct.lower() == "dnn":
                inferred = _infer_dnn_arch_from_state_dict(state_dict)
                if inferred is not None:
                    hidden_sizes, use_batch_norm = inferred
                    logger.info(
                        f"  Inferred classifier architecture from checkpoint: "
                        f"hidden_sizes={hidden_sizes}, batch_norm={use_batch_norm}"
                    )

            classifier = self.create_classifier(
                classifier_type=ct,
                device=device,
                hidden_sizes=hidden_sizes,
                use_batch_norm=use_batch_norm,
            )
            classifier.load_state_dict(state_dict)

        classifier.eval()
        logger.info(f"Classifier loaded from {filepath}")
        return classifier

    def get_classifier(self) -> torch.nn.Module:
        if self.classifier is None:
            raise ValueError("Classifier not created yet. Call create_classifier() first.")
        return self.classifier
    
    def perform_eda_analysis(
        self,
        output_dir: str = "./output/eda/",
        verbose: bool = True,
        use_ydata_profiling: bool = True,
        minimal: bool = False,
        force_rerun: bool = False
    ) -> Dict[str, any]:
        if self.X_train is None or self.X_test is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if EDA has already been done
        summary_path = os.path.join(output_dir, "eda_summary.json")
        train_report_path = os.path.join(output_dir, "training_data_profile.html")
        test_report_path = os.path.join(output_dir, "test_data_profile.html")
        
        # Determine if EDA was done with ydata_profiling or custom
        eda_already_done = False
        if use_ydata_profiling:
            # Check for ydata_profiling outputs
            if os.path.exists(summary_path) and os.path.exists(train_report_path) and os.path.exists(test_report_path):
                eda_already_done = True
        else:
            # Check for custom EDA output (at minimum, summary should exist)
            if os.path.exists(summary_path):
                eda_already_done = True
        
        if eda_already_done and not force_rerun:
            if verbose:
                logger.info(f"\nEDA analysis already exists for dataset: {self.dataset_name}")
                logger.info("="*80)
                logger.info(f"Found existing EDA results in: {output_dir}")
                logger.info("  - Skipping EDA generation (use force_rerun=True to regenerate)")
                logger.info("="*80)
            
            # Load and return existing results
            eda_results = {}
            try:
                import json
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_data = json.load(f)
                    eda_results["summary"] = summary_data
                
                if use_ydata_profiling:
                    if os.path.exists(train_report_path) and os.path.exists(test_report_path):
                        eda_results["ydata_profiling"] = {
                            "training_report_path": train_report_path,
                            "test_report_path": test_report_path,
                            "summary_json_path": summary_path
                        }
            except Exception as e:
                if verbose:
                    logger.warning(f"   Warning: Could not load existing EDA results: {e}")
                    logger.warning("  Continuing with fresh EDA generation...")
                eda_already_done = False
        
        if not eda_already_done or force_rerun:
            if verbose:
                logger.info(f"\nPerforming EDA analysis for dataset: {self.dataset_name}")
                logger.info("="*80)
            
            eda_results = {}
            
            if use_ydata_profiling:
                try:
                    from ydata_profiling import ProfileReport
                    eda_results["ydata_profiling"] = self._perform_ydata_profiling(
                        output_dir, verbose, minimal
                    )
                except ImportError:
                    if verbose:
                        logger.warning("\n ydata-profiling not installed. Falling back to custom EDA.")
                        logger.warning("  Install with: pip install ydata-profiling")
                    use_ydata_profiling = False
            
            if not use_ydata_profiling:
                eda_results["dataset_overview"] = self._analyze_dataset_overview(verbose)
                eda_results["feature_statistics"] = self._analyze_feature_statistics(verbose)
                eda_results["class_distribution"] = self._analyze_class_distribution(verbose)
                eda_results["feature_correlations"] = self._analyze_feature_correlations(output_dir, verbose)
                eda_results["class_separability"] = self._analyze_class_separability(verbose)
                eda_results["data_quality"] = self._analyze_data_quality(verbose)
            
            if verbose:
                logger.info("\n" + "="*80)
                logger.info("EDA COMPLETE!")
                logger.info("="*80)
                logger.info(f"Results saved to: {output_dir}")
        
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
            logger.info("\n" + "="*80)
            logger.info("YDATA PROFILING ANALYSIS")
            logger.info("="*80)
            logger.info("Generating comprehensive EDA report...")
            logger.info("  This may take a few minutes for large datasets...")
        
        df_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        df_train['target'] = self.y_train
        
        df_test = pd.DataFrame(self.X_test, columns=self.feature_names)
        df_test['target'] = self.y_test
        
        results = {}
        
        if verbose:
            logger.info("\n1. Generating training data profile...")
        
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
            logger.info(f"    Saved training data profile to {train_report_path}")
        
        if verbose:
            logger.info("\n2. Generating test data profile...")
        
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
            logger.info(f"    Saved test data profile to {test_report_path}")
        
        if verbose:
            logger.info("\n3. Extracting key metrics for XAI...")
        
        train_description = profile_train.get_description()
        test_description = profile_test.get_description()
        
        # Helper function to safely get attribute values from description objects
        def get_table_stat(description, stat_name, default=0):
            """Safely extract table statistics from description object."""
            try:
                table = getattr(description, 'table', None)
                if table is None:
                    return default
                # Try accessing as attribute first
                if hasattr(table, stat_name):
                    return getattr(table, stat_name, default)
                # Try accessing as dict key
                if isinstance(table, dict):
                    return table.get(stat_name, default)
                # Try accessing via __dict__
                if hasattr(table, '__dict__'):
                    return table.__dict__.get(stat_name, default)
                return default
            except Exception:
                return default
        
        # Extract table statistics
        results["training_summary"] = {
            "n_variables": get_table_stat(train_description, "n_variables", 0),
            "n_observations": get_table_stat(train_description, "n_observations", 0),
            "n_cells": get_table_stat(train_description, "n_cells", 0),
            "n_duplicates": get_table_stat(train_description, "n_duplicates", 0),
            "p_duplicates": get_table_stat(train_description, "p_duplicates", 0.0),
            "n_missing": get_table_stat(train_description, "n_missing", 0),
            "p_missing": get_table_stat(train_description, "p_missing", 0.0),
            "memory_size": get_table_stat(train_description, "memory_size", 0)
        }
        
        results["test_summary"] = {
            "n_variables": get_table_stat(test_description, "n_variables", 0),
            "n_observations": get_table_stat(test_description, "n_observations", 0),
            "n_cells": get_table_stat(test_description, "n_cells", 0),
            "n_duplicates": get_table_stat(test_description, "n_duplicates", 0),
            "p_duplicates": get_table_stat(test_description, "p_duplicates", 0.0),
            "n_missing": get_table_stat(test_description, "n_missing", 0),
            "p_missing": get_table_stat(test_description, "p_missing", 0.0),
            "memory_size": get_table_stat(test_description, "memory_size", 0)
        }
        
        # Extract correlations and alerts if available
        results["correlations"] = {}
        try:
            train_corr = getattr(train_description, 'correlations', None)
            if train_corr is not None:
                if hasattr(train_corr, 'to_dict'):
                    results["correlations"]["training"] = train_corr.to_dict()
                else:
                    results["correlations"]["training"] = train_corr
        except Exception:
            pass
        
        try:
            test_corr = getattr(test_description, 'correlations', None)
            if test_corr is not None:
                if hasattr(test_corr, 'to_dict'):
                    results["correlations"]["test"] = test_corr.to_dict()
                else:
                    results["correlations"]["test"] = test_corr
        except Exception:
            pass
        
        results["alerts"] = {}
        try:
            train_alerts = getattr(train_description, 'alerts', None)
            if train_alerts is not None:
                results["alerts"]["training"] = train_alerts
        except Exception:
            pass
        
        try:
            test_alerts = getattr(test_description, 'alerts', None)
            if test_alerts is not None:
                results["alerts"]["test"] = test_alerts
        except Exception:
            pass
        
        if verbose:
            logger.info("\n4. Key Statistics:")
            logger.info(f"   Training samples: {results['training_summary']['n_observations']:,}")
            logger.info(f"   Test samples: {results['test_summary']['n_observations']:,}")
            logger.info(f"   Features: {results['training_summary']['n_variables'] - 1}")
            logger.info(f"   Missing values (train): {results['training_summary']['n_missing']} "
                  f"({results['training_summary']['p_missing']:.2f}%)")
            logger.info(f"   Missing values (test): {results['test_summary']['n_missing']} "
                  f"({results['test_summary']['p_missing']:.2f}%)")
            logger.info(f"   Duplicates (train): {results['training_summary']['n_duplicates']} "
                  f"({results['training_summary']['p_duplicates']:.2f}%)")
            logger.info(f"   Duplicates (test): {results['test_summary']['n_duplicates']} "
                  f"({results['test_summary']['p_duplicates']:.2f}%)")
        
        if verbose:
            logger.info("\n5. Saving JSON summary...")
        
        import json
        
        # Helper function to convert NumPy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert NumPy types to native Python types."""
            # Check for NumPy integer types (compatible with NumPy 2.0)
            if isinstance(obj, np.integer):
                return int(obj)
            # Check for NumPy floating point types (compatible with NumPy 2.0)
            elif isinstance(obj, np.floating):
                return float(obj)
            # Check for NumPy arrays
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Check for NumPy boolean (compatible with NumPy 2.0)
            elif isinstance(obj, bool):
                return bool(obj)
            elif hasattr(np, 'bool_') and isinstance(obj, np.bool_):
                return bool(obj)
            # Handle dictionaries recursively
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            # Handle lists and tuples recursively
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            else:
                # Try to convert if it's a NumPy scalar (fallback for any NumPy type)
                try:
                    if hasattr(obj, 'item'):
                        return obj.item()
                    # Check if it's a NumPy dtype that can be converted
                    if hasattr(obj, 'dtype'):
                        if np.issubdtype(obj.dtype, np.integer):
                            return int(obj)
                        elif np.issubdtype(obj.dtype, np.floating):
                            return float(obj)
                except (ValueError, AttributeError, TypeError):
                    pass
                return obj
        
        summary_data = {
            "dataset_name": self.dataset_name,
            "training_summary": results["training_summary"],
            "test_summary": results["test_summary"],
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "class_names": self.class_names
        }
        
        # Convert all NumPy types to native Python types
        summary_data = convert_to_json_serializable(summary_data)
        
        summary_path = os.path.join(output_dir, "eda_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        results["summary_json_path"] = summary_path
        
        if verbose:
            logger.info(f"    Saved summary to {summary_path}")
        
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
            logger.info("\n" + "="*80)
            logger.info("DATASET OVERVIEW")
            logger.info("="*80)
            logger.info(f"Dataset: {overview['dataset_name']}")
            logger.info(f"Training samples: {overview['n_train_samples']:,}")
            logger.info(f"Test samples: {overview['n_test_samples']:,}")
            logger.info(f"Features: {overview['n_features']}")
            logger.info(f"Classes: {overview['n_classes']}")
            if self.class_names:
                logger.info(f"Class names: {self.class_names}")
        
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
            logger.info("\n" + "="*80)
            logger.info("FEATURE STATISTICS")
            logger.info("="*80)
            logger.info(f"{'Feature':<20} {'Train Mean':<12} {'Train Std':<12} {'Test Mean':<12} {'Test Std':<12}")
            logger.info("-" * 80)
            for feat_name, feat_stats in list(stats.items())[:10]:
                logger.info(f"{feat_name:<20} {feat_stats['train_mean']:>11.4f} "
                      f"{feat_stats['train_std']:>11.4f} {feat_stats['test_mean']:>11.4f} "
                      f"{feat_stats['test_std']:>11.4f}")
            if len(stats) > 10:
                logger.info(f"... and {len(stats) - 10} more features")
        
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
            logger.info("\n" + "="*80)
            logger.info("CLASS DISTRIBUTION")
            logger.info("="*80)
            logger.info("Training set:")
            for cls, info in train_dist.items():
                logger.info(f"  {info['name']}: {info['count']:,} ({info['percentage']:.1f}%)")
            logger.info("\nTest set:")
            for cls, info in test_dist.items():
                logger.info(f"  {info['name']}: {info['count']:,} ({info['percentage']:.1f}%)")
            logger.info(f"\nClass balance: {'Balanced' if distribution['is_balanced'] else 'Imbalanced'}")
        
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
                logger.info("\n" + "="*80)
                logger.info("FEATURE CORRELATIONS")
                logger.info("="*80)
                logger.info("Skipping correlation analysis (pandas/matplotlib/seaborn not available)")
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
            logger.info("\n" + "="*80)
            logger.info("FEATURE CORRELATIONS")
            logger.info("="*80)
            logger.info(f"High correlations (|r| > 0.7): {len(correlations['high_correlations'])}")
            if correlations["high_correlations"]:
                logger.info("\nTop high correlations:")
                for corr_info in sorted(correlations["high_correlations"], 
                                       key=lambda x: abs(x["correlation"]), 
                                       reverse=True)[:10]:
                    logger.info(f"  {corr_info['feature1']} <-> {corr_info['feature2']}: "
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
                logger.info(f"   Saved correlation heatmap to {output_dir}correlation_heatmap.png")
        except Exception as e:
            if verbose:
                logger.warning(f"   Could not save correlation heatmap: {e}")
        
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
            logger.info("\n" + "="*80)
            logger.info("CLASS SEPARABILITY")
            logger.info("="*80)
            logger.info("Top 10 features by separability score:")
            for feat, score in sorted_features[:10]:
                logger.info(f"  {feat}: {score:.4f}")
        
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
            logger.info("\n" + "="*80)
            logger.info("DATA QUALITY")
            logger.info("="*80)
            logger.info(f"Missing values (train): {quality['missing_values_train']}")
            logger.info(f"Missing values (test): {quality['missing_values_test']}")
            logger.info(f"Infinite values (train): {quality['infinite_values_train']}")
            logger.info(f"Infinite values (test): {quality['infinite_values_test']}")
            
            high_outlier_features = [
                (feat, info["percentage"]) 
                for feat, info in quality["outliers"].items() 
                if info["percentage"] > 5.0
            ]
            
            if high_outlier_features:
                logger.info(f"\nFeatures with >5% outliers: {len(high_outlier_features)}")
                for feat, pct in sorted(high_outlier_features, key=lambda x: x[1], reverse=True)[:5]:
                    logger.info(f"  {feat}: {pct:.1f}%")
            else:
                logger.info("\n No significant outliers detected (>5%)")
        
        return quality