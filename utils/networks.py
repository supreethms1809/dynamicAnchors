"""
Neural network architectures for Dynamic Anchor RL.

This module provides all neural network definitions used in the project.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class bigClassifier(nn.Module):
    """Big neural network classifier for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        layers = []
        layers.append(nn.Linear(input_dim, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(256, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(256, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.5))

        layers.append(nn.Linear(128, 64))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.5))
        
        layers.append(nn.Linear(64, num_classes))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)

class SimpleClassifier(nn.Module):
    """Simple neural network classifier for tabular data."""
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        dropout_rate: float = 0.3, 
        use_batch_norm: bool = True,
        hidden_sizes: Optional[List[int]] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization (default: 0.3)
            use_batch_norm: Whether to use batch normalization (default: True)
            hidden_sizes: List of hidden layer sizes (default: [256, 256, 128] for small datasets)
                         Use [512, 512, 256] or [256, 256, 256, 128] for larger datasets
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Default architecture for small datasets
        if hidden_sizes is None:
            hidden_sizes = [256, 256, 128]
        
        layers = []
        prev_size = input_dim
        
        # Build hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            # Use full dropout for first layers, reduced for later layers
            dropout = dropout_rate if i < len(hidden_sizes) - 1 else dropout_rate * 0.5
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)


def predict_proba_torch(classifier, x: torch.Tensor) -> torch.Tensor:
    """Return class probabilities as a torch.Tensor on the same device as ``x``.

    Works with both raw nn.Module DNN classifiers (e.g. SimpleClassifier) and
    UnifiedClassifier wrappers around tree-based models. For tree models this
    skips the log/clip/softmax round-trip done by UnifiedClassifier.forward.
    """
    if isinstance(classifier, UnifiedClassifier):
        return classifier.predict_proba_torch(x)
    # Raw nn.Module — assume DNN producing logits.
    logits = classifier(x)
    return torch.softmax(logits, dim=-1)


class UnifiedClassifier(nn.Module):
    """
    Unified classifier wrapper that supports PyTorch DNN, sklearn Random Forest, and Gradient Boosting.
    
    This wrapper allows the same interface for all types of classifiers:
    - PyTorch models: Used directly
    - Random Forest: Wrapped to provide PyTorch-compatible interface
    - Gradient Boosting: Wrapped to provide PyTorch-compatible interface
    """
    
    def __init__(
        self,
        classifier_type: str = "dnn",
        input_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        dnn_model: Optional[nn.Module] = None,
        rf_model: Optional[RandomForestClassifier] = None,
        gb_model: Optional[GradientBoostingClassifier] = None,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize unified classifier.
        
        Args:
            classifier_type: "dnn", "random_forest", or "gradient_boosting"
            input_dim: Number of input features (required for DNN)
            num_classes: Number of output classes (required for DNN)
            dnn_model: Pre-initialized PyTorch model (optional)
            rf_model: Pre-trained Random Forest model (optional)
            gb_model: Pre-trained Gradient Boosting model (optional)
            device: Device to use for DNN models
            **kwargs: Additional arguments for tree-based models (n_estimators, max_depth, etc.)
        """
        super().__init__()
        
        self.classifier_type = classifier_type.lower()
        self.device = device
        
        if self.classifier_type == "dnn":
            if dnn_model is not None:
                self.model = dnn_model
            else:
                if input_dim is None or num_classes is None:
                    raise ValueError("input_dim and num_classes required for DNN classifier")
                self.model = SimpleClassifier(input_dim, num_classes)
            self.model.to(device)
            self.rf_model = None
            self.gb_model = None
        elif self.classifier_type == "random_forest":
            if rf_model is not None:
                self.rf_model = rf_model
            else:
                # Default Random Forest parameters
                default_rf_kwargs = {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                default_rf_kwargs.update(kwargs)
                self.rf_model = RandomForestClassifier(**default_rf_kwargs)
            self.model = None  # No PyTorch model for RF
            self.gb_model = None
        elif self.classifier_type == "gradient_boosting":
            if gb_model is not None:
                self.gb_model = gb_model
            else:
                # Default Gradient Boosting parameters
                default_gb_kwargs = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42,
                }
                default_gb_kwargs.update(kwargs)
                self.gb_model = GradientBoostingClassifier(**default_gb_kwargs)
            self.model = None  # No PyTorch model for GB
            self.rf_model = None
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}. Use 'dnn', 'random_forest', or 'gradient_boosting'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that works for DNN, Random Forest, and Gradient Boosting.

        For DNN this returns true logits. For tree-based models there are no
        logits — we return log(probs) so that ``softmax(forward(x)) ≈ probs``,
        which keeps existing call sites that wrap this in ``torch.softmax``
        approximately correct. Prefer ``predict_proba_torch`` when you only
        need probabilities — it skips the redundant log/softmax round-trip
        and avoids the small precision loss from clipping.

        Returns a tensor on the same device as ``x``.
        """
        if self.classifier_type == "dnn":
            return self.model(x)

        out_device = x.device if isinstance(x, torch.Tensor) else torch.device(self.device)
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        if self.classifier_type == "random_forest":
            probs = self.rf_model.predict_proba(x_np)
        elif self.classifier_type == "gradient_boosting":
            probs = self.gb_model.predict_proba(x_np)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        eps = 1e-8
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs)
        return torch.from_numpy(logits).float().to(out_device)

    def predict_proba_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities as a torch.Tensor on the same device as ``x``.

        Use this instead of ``torch.softmax(classifier(x), dim=-1)`` — it gives
        exact probabilities for tree-based models (no log/clip/softmax round-trip)
        and matches the DNN behavior.
        """
        if self.classifier_type == "dnn":
            logits = self.model(x)
            return torch.softmax(logits, dim=-1)

        out_device = x.device if isinstance(x, torch.Tensor) else torch.device(self.device)
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

        if self.classifier_type == "random_forest":
            probs = self.rf_model.predict_proba(x_np)
        elif self.classifier_type == "gradient_boosting":
            probs = self.gb_model.predict_proba(x_np)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")

        return torch.from_numpy(probs).float().to(out_device)
    
    def train(self, mode: bool = True):
        """Set training/eval mode (only relevant for DNN)."""
        if self.classifier_type == "dnn":
            self.model.train(mode)
        return self
    
    def eval(self):
        """Set eval mode (only relevant for DNN)."""
        if self.classifier_type == "dnn":
            self.model.eval()
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        
        For DNN: This is a no-op (use optimizer and backward pass)
        For Random Forest: Trains the Random Forest model
        For Gradient Boosting: Trains the Gradient Boosting model
        """
        if self.classifier_type == "random_forest":
            self.rf_model.fit(X, y)
        elif self.classifier_type == "gradient_boosting":
            self.gb_model.fit(X, y)
        # For DNN, training is done via optimizer, so this is a no-op
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.classifier_type == "dnn":
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float().to(self.device)
                logits = self.model(X_tensor)
                preds = logits.argmax(dim=1).cpu().numpy()
            return preds
        elif self.classifier_type == "random_forest":
            return self.rf_model.predict(X)
        elif self.classifier_type == "gradient_boosting":
            return self.gb_model.predict(X)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.classifier_type == "dnn":
            with torch.no_grad():
                X_tensor = torch.from_numpy(X).float().to(self.device)
                logits = self.model(X_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        elif self.classifier_type == "random_forest":
            return self.rf_model.predict_proba(X)
        elif self.classifier_type == "gradient_boosting":
            return self.gb_model.predict_proba(X)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
    
    def state_dict(self):
        """Get state dict (only for DNN)."""
        if self.classifier_type == "dnn":
            return self.model.state_dict()
        else:
            return {}  # Tree-based models don't have state_dict
    
    def load_state_dict(self, state_dict):
        """Load state dict (only for DNN)."""
        if self.classifier_type == "dnn":
            self.model.load_state_dict(state_dict)
    
    def parameters(self):
        """Get parameters (only for DNN)."""
        if self.classifier_type == "dnn":
            return self.model.parameters()
        else:
            return []  # Tree-based models don't have trainable parameters


class PolicyNet(nn.Module):
    """Policy network for dynamic anchor generation using PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_dim)
        
        Returns:
            Output action logits of shape (batch_size, action_dim)
        """
        return self.net(x)


class ValueNet(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, state_dim)
        
        Returns:
            Output state values of shape (batch_size,)
        """
        return self.net(x).view(-1)
