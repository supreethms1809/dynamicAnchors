"""
Neural network architectures for Dynamic Anchor RL.

This module provides all neural network definitions used in the project.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3, use_batch_norm: bool = True):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization (default: 0.3)
            use_batch_norm: Whether to use batch normalization (default: True)
        """
        super().__init__()
        self.num_classes = num_classes
        
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Second layer
        layers.append(nn.Linear(256, 256))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Third layer (optional, for deeper network)
        layers.append(nn.Linear(256, 128))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate * 0.5))  # Less dropout in later layers
        
        # Output layer
        layers.append(nn.Linear(128, num_classes))
        
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
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        if self.classifier_type == "dnn":
            return self.model(x)
        else:  # Tree-based models (Random Forest or Gradient Boosting)
            # Convert torch tensor to numpy
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x
            
            # Get probabilities from tree-based model
            if self.classifier_type == "random_forest":
                probs = self.rf_model.predict_proba(x_np)
            elif self.classifier_type == "gradient_boosting":
                probs = self.gb_model.predict_proba(x_np)
            else:
                raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
            
            # Convert probabilities to logits
            # For multi-class: logits = log(probs) - log(sum(probs))
            # Since probs are already normalized (sum to 1), we can use log(probs)
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            probs = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs)
            
            # Convert back to torch tensor
            return torch.from_numpy(logits).float().to(self.device)
    
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
