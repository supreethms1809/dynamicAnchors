"""
Multi-Agent Global Interpretation with Dynamic Anchors

This script loads datasets and sets the number of agents based on the number of classes.
Each agent represents one class and learns to find anchors for its target class.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from trainers.multiagent_utils import SimpleClassifier, train_classifier, \
                                    globalBuffer, AnchorEnv, \
                                    MultiAgentEnvironment, CentralizedCritic, \
                                    MultiAgentPolicyNet, CentralizedTrainer, \
                                    DecentralizedExecutor
from trainers.sklearn_datasets_anchors import load_dataset
import gymnasium as gym

def get_number_of_classes(dataset_name: str, sample_size: int = None, seed: int = 42):
    X, y, feature_names, class_names = load_dataset(dataset_name, sample_size=sample_size, seed=seed)
    n_classes = len(class_names)
    unique_classes = np.unique(y)
    n_classes_from_y = len(unique_classes)
    if n_classes != n_classes_from_y:
        print(f"Warning: Number of classes from class_names ({n_classes}) "
              f"does not match unique values in y ({n_classes_from_y}). Using {n_classes_from_y}.")
        n_classes = n_classes_from_y
    
    return n_classes


def main(dataset_name: str = "breast_cancer", sample_size: int = None, seed: int = 42):
    print("\n" + "="*80)
    print(f"Multi-Agent Global Interpretation - Dataset: {dataset_name.upper().replace('_', ' ')}")
    print("="*80)
    
    # Load dataset
    X, y, feature_names, class_names = load_dataset(dataset_name, sample_size=sample_size, seed=seed)
    n_classes = get_number_of_classes(dataset_name, sample_size=sample_size, seed=seed)
    print(f"Number of classes: {n_classes}")
    
    # Set number of agents
    n_agents = n_classes
    print(f"Number of agents: {n_agents}")

    # Check if classifier model exists and load it, otherwise train a new one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 512
    
    # Create model path based on dataset parameters
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    sample_size_str = f"_{sample_size}" if sample_size is not None else ""
    model_path = os.path.join(model_dir, f"classifier_{dataset_name}{sample_size_str}_seed{seed}.pth")
    
    # Initialize classifier
    classifier = SimpleClassifier(X.shape[1], hidden_size, n_classes)
    
    # Check if model exists and try to load it
    model_loaded = False
    if os.path.exists(model_path):
        print(f"Found existing classifier at {model_path}")
        print("Attempting to load classifier...")
        classifier.to(device)
        model_loaded = classifier.load(model_path, device=device, strict=True)
        
        if model_loaded:
            classifier.eval()
            print(f"Classifier loaded successfully from {model_path}")
        else:
            print(f"Failed to load classifier (architecture mismatch or corrupted file)")
            print("Will train a new classifier...")
    
    # Train classifier if not loaded
    if not model_loaded:
        if not os.path.exists(model_path):
            print(f"No existing classifier found at {model_path}")
        print("Training new classifier...")
        classifier.to(device)
        classifier, best_test_acc = train_classifier(classifier, X, y, device, epochs=50, batch_size=256, lr=1e-4)
        print(f"Best test accuracy: {best_test_acc:.3f}")
        classifier.save(model_path)
        print(f"Classifier saved to {model_path}")
    print("="*80)
    
    # Multi-Agent Global Interpretation
    # Observation space: [lower_bounds, upper_bounds, precision, coverage]
    obs_dim = X.shape[1] + 2  # Features + precision + coverage
    action_dim = n_classes  # Action dimension
    
    # Initialize centralized critic
    # Critic input: concatenated observations from all agents
    centralized_critic = CentralizedCritic(obs_dim * n_agents, hidden_size, n_classes * n_agents, device)
    # Initialize policy nets - each agent's policy takes its own observation
    policy_nets = [MultiAgentPolicyNet(n_agents, obs_dim, hidden_size, action_dim, device) for agent_id in range(n_agents)]

    # Initialize observation and action spaces
    run_config = {
        "precision_target": 0.95,
        "coverage_target": 0.05,
        "buffer_size": 10000,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "num_episodes": 100,
        "num_steps_per_episode": 100,
        "learning_starts": 1000,  # Warm-up: collect this many samples before training starts
        "train_frequency": 1,  # Train every N steps after learning_starts
        "device": device,
        "num_agents": n_agents,
    }
    obs_space = {agent_id: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)) for agent_id in range(n_agents)}
    # Action space: [lower_deltas, upper_deltas] each in [-1, 1]
    action_space = {agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,)) for agent_id in range(n_agents)}
    
    # Initialize multiagent environment
    multiagent_env = MultiAgentEnvironment(classifier, centralized_critic, policy_nets, n_agents, obs_space, action_space, run_config)
    for agent_id in range(n_agents):
        print(f"Observation space agent {agent_id}: {multiagent_env.get_observation_space(agent_id)}")
        print(f"Action space agent {agent_id}: {multiagent_env.get_action_space(agent_id)}")
    print("="*80)

    # Initialize centralized trainer and train
    centralized_trainer = CentralizedTrainer(multiagent_env, centralized_critic, policy_nets, n_agents, obs_space, action_space, run_config)
    training_results = centralized_trainer.train()
    print(f"Centralized trainer trained successfully with results: {training_results}")
    print("="*80)

    # Initialize decentralized executor and execute
    decentralized_executor = DecentralizedExecutor(multiagent_env, policy_nets, n_agents, obs_space, action_space, run_config)
    execution_results = decentralized_executor.execute()
    print(f"Decentralized executor executed successfully with results: {execution_results}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load datasets and set number of agents based on number of classes",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["breast_cancer", "covtype", "wine", "housing"],
        help="Dataset to use (default: breast_cancer)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for large datasets (recommended: 10000-50000 for covtype)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    results = main(
        dataset_name=args.dataset,
        sample_size=args.sample_size,
        seed=args.seed
    )

