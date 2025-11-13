"""
BenchMARL - Multi-Agent Reinforcement Learning for Dynamic Anchor Discovery

This package provides tools for training multi-agent RL models to discover
dynamic anchors in tabular classification datasets.

Main Components:
- TabularDatasetLoader: Load and preprocess tabular datasets
- AnchorEnv: Multi-agent environment for anchor discovery
- AnchorTrainer: Training pipeline for multi-agent RL models
- AnchorTask, AnchorTaskClass: BenchMARL task wrappers
- AnchorMetricsCallback: Metrics collection callback
"""

from .tabular_datasets import TabularDatasetLoader
from .environment import AnchorEnv
from .anchor_trainer import AnchorTrainer
from .benchmarl_wrappers import (
    AnchorTask,
    AnchorTaskClass,
    AnchorMetricsCallback
)

__all__ = [
    "TabularDatasetLoader",
    "AnchorEnv",
    "AnchorTrainer",
    "AnchorTask",
    "AnchorTaskClass",
    "AnchorMetricsCallback",
]

__version__ = "0.1.0"

