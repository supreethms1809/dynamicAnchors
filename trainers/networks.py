"""
Neural network architectures for Dynamic Anchor RL.

This module provides all neural network definitions used in the project.
"""

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Simple neural network classifier for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.net(x)


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

