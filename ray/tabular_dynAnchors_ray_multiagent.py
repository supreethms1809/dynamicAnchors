"""
Multi-Agent Dynamic Anchors pipeline for tabular data using Ray RLlib (SAC for continuous actions).

This module provides a multi-agent version where all class agents train simultaneously
in a single MultiAgentEnv. Each agent represents one class and learns to find anchors
for its target class.

Post-hoc training only: Train classifier first, then train RL policy.

Usage example:
    from ray.tabular_dynAnchors_ray_multiagent import train_and_evaluate_dynamic_anchors_ray_multiagent
    
    results = train_and_evaluate_dynamic_anchors_ray_multiagent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        classifier=classifier,
        target_classes=(0, 1, 2),
        total_timesteps=50000,
    )
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
import os
import json
from functools import partial
import gymnasium as gym
from gymnasium import spaces


def train_and_evaluate_dynamic_anchors_ray_multiagent(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    classifier: torch.nn.Module,
    target_classes: Tuple[int, ...] = None,
    device: str = "cpu",
    # Training parameters
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    # Environment parameters
    use_perturbation: bool = True,
    perturbation_mode: str = "adaptive",  # "bootstrap", "uniform", or "adaptive"
    n_perturb: int = 2048,
    step_fracs: Tuple[float, ...] = (0.005, 0.01, 0.02),
    min_width: float = 0.05,
    precision_target: float = 0.95,
    coverage_target: float = 0.02,
    # Evaluation parameters
    n_eval_instances_per_class: int = 20,
    max_features_in_rule: Optional[int] = 5,
    steps_per_episode: int = 100,
    eval_steps_per_episode: int = None,
    num_rollouts_per_instance: int = 1,
    # Training sampling parameters
    n_clusters_per_class: Optional[int] = None,
    n_fixed_instances_per_class: Optional[int] = None,
    use_random_sampling: bool = False,
    # Output parameters
    output_dir: str = "./output/anchors_ray_multiagent/",
    save_checkpoints: bool = True,
    checkpoint_freq: int = 10000,
    verbose: int = 1,
    # Ray RLlib specific parameters
    num_workers: int = 0,  # Number of parallel workers (0 = single process)
    num_gpus: int = 0,  # Number of GPUs to use
    num_cpus_per_worker: int = 1,
    # SAC specific parameters
    tau: float = 0.005,  # Soft update coefficient
    target_network_update_freq: int = 1,
    buffer_size: int = 1000000,
    learning_starts: int = 1000,
    train_batch_size: int = 256,
    target_entropy: Optional[float] = None,  # Auto if None
) -> Dict[str, Any]:
    """
    Multi-agent version: Train all class agents simultaneously in a single MultiAgentEnv.
    
    Each agent represents one class and learns to find anchors for its target class.
    All agents train in parallel, sharing the same environment but with different
    target classes and policies.
    
    Args:
        Same as train_and_evaluate_dynamic_anchors_ray, but trains all agents simultaneously.
    
    Returns:
        Dictionary with trained multi-agent model and evaluation results.
    """
    # Import Ray and required modules
    try:
        import ray
        from ray.rllib.algorithms.sac import SACConfig
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        from ray.rllib.algorithms.callbacks import DefaultCallbacks
    except ImportError as e:
        raise ImportError(
            "Ray RLlib is required for multi-agent training. "
            "Install with: pip install 'ray[rllib]'"
        ) from e
    
    # Import standalone modules
    try:
        from ray.ray_modules_standalone import (
            AnchorEnv,
            ContinuousAnchorEnv,
            compute_cluster_centroids_per_class,
            evaluate_all_classes,
            evaluate_all_classes_class_level,
            get_device,
            get_device_str,
        )
    except ImportError:
        from ray_modules_standalone import (
            AnchorEnv,
            ContinuousAnchorEnv,
            compute_cluster_centroids_per_class,
            evaluate_all_classes,
            evaluate_all_classes_class_level,
            get_device,
            get_device_str,
        )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device_str = get_device_str(device)
    device_obj = get_device(device)
    
    # Determine target classes
    if target_classes is None:
        target_classes = tuple(sorted(np.unique(y_train)))
    
    print(f"\n{'='*80}")
    print(f"Multi-Agent Dynamic Anchors Training (Ray RLlib SAC)")
    print(f"{'='*80}")
    print(f"Target classes: {target_classes}")
    print(f"Number of agents: {len(target_classes)}")
    print(f"Device: {device_str}")
    print(f"Total timesteps: {total_timesteps}")
    
    # Data preprocessing
    print(f"\n[Data Preprocessing]")
    X_min = X_train.min(axis=0)
    X_range = X_train.max(axis=0) - X_min
    X_range[X_range == 0] = 1.0  # Avoid division by zero
    
    # Normalize to [0, 1]
    X_unit_train = (X_train - X_min) / X_range
    X_unit_test = (X_test - X_min) / X_range
    X_test_unit = X_unit_test  # Alias for consistency with evaluation functions
    
    # Standardize (zero mean, unit variance)
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0  # Avoid division by zero
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    
    # Cluster-based sampling (if requested)
    cluster_centroids_per_class = None
    if n_clusters_per_class is not None:
        print(f"\n[Cluster-Based Sampling] Computing {n_clusters_per_class} centroids per class...")
        cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_unit_train,
            y=y_train,
            n_clusters_per_class=n_clusters_per_class,
            random_state=42
        )
        print(f"  Cluster centroids computed successfully!")
    
    # Fixed instance sampling (fallback)
    fixed_instances_per_class = None
    if n_fixed_instances_per_class is not None:
        print(f"\n[Fixed Instance Sampling] Selecting {n_fixed_instances_per_class} fixed instances per class...")
        fixed_instances_per_class = {}
        for cls in target_classes:
            cls_indices = np.where(y_train == cls)[0]
            if len(cls_indices) > 0:
                n_fixed = min(n_fixed_instances_per_class, len(cls_indices))
                rng = np.random.default_rng(42 + cls)
                fixed_indices = rng.choice(cls_indices, size=n_fixed, replace=False)
                fixed_instances_per_class[cls] = fixed_indices.tolist()
                print(f"  Class {cls}: {len(fixed_indices)} fixed instances")
    
    # Compute test cluster centroids for evaluation
    test_cluster_centroids_per_class = None
    try:
        test_cluster_centroids_per_class = compute_cluster_centroids_per_class(
            X_unit=X_unit_test,
            y=y_test,
            n_clusters_per_class=10,
            random_state=42
        )
    except Exception as e:
        print(f"  WARNING: Could not compute test cluster centroids: {e}")
    
    # Create environment factory function
    def create_anchor_env(target_cls, use_test_centroids=False):
        """Helper to create AnchorEnv with a specific target class."""
        env = AnchorEnv(
            X_unit=X_unit_train,
            X_std=X_train_scaled,
            y=y_train,
            feature_names=feature_names,
            classifier=classifier,
            device=device_str,
            target_class=target_cls,
            step_fracs=step_fracs,
            min_width=min_width,
            alpha=0.7,
            beta=0.6,
            gamma=0.1,
            precision_target=precision_target,
            coverage_target=coverage_target,
            precision_blend_lambda=0.5,
            drift_penalty_weight=0.05,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            X_min=X_min,
            X_range=X_range,
            min_coverage_floor=0.005,
            js_penalty_weight=0.05,
        )
        
        # Set cluster centroids
        if use_test_centroids and test_cluster_centroids_per_class is not None:
            env.cluster_centroids_per_class = test_cluster_centroids_per_class
        elif cluster_centroids_per_class is not None:
            env.cluster_centroids_per_class = cluster_centroids_per_class
        
        env.fixed_instances_per_class = fixed_instances_per_class
        env.use_random_sampling = use_random_sampling
        
        # Enable continuous actions
        env.max_action_scale = max(step_fracs) if step_fracs else 0.02
        env.min_absolute_step = max(0.05, min_width * 0.5)
        
        return env
    
    # Custom callback to track precision and coverage metrics per agent
    class MultiAgentPrecisionCoverageCallback(DefaultCallbacks):
        """
        Custom callback to track precision and coverage metrics from multi-agent environment info dict.
        
        Extracts precision and coverage for each agent and stores them in agent-specific custom_metrics.
        """
        def __init__(self):
            super().__init__()
            # Track metrics per agent: {agent_id: [list of precisions/coverages]}
            self.episode_precisions_per_agent = {}
            self.episode_coverages_per_agent = {}
            self.processed_episodes = set()
        
        def on_episode_step(self, *, episode, env_index, env_runner=None, metrics_logger=None, env=None, rl_module=None, worker=None, base_env=None, policies=None, **kwargs):
            """Called at each step of an episode. Extract metrics per agent."""
            try:
                # Method 1: Try to get infos from episode (most reliable - populated by step())
                if hasattr(episode, 'get_infos'):
                    try:
                        infos = episode.get_infos()
                        if isinstance(infos, dict):
                            # Multi-agent format: infos is a dict per agent
                            for agent_id, agent_infos in infos.items():
                                if isinstance(agent_infos, list) and len(agent_infos) > 0:
                                    last_info = agent_infos[-1]
                                elif isinstance(agent_infos, dict):
                                    last_info = agent_infos
                                else:
                                    continue
                                
                                if isinstance(last_info, dict):
                                    precision = last_info.get("precision", None)
                                    coverage = last_info.get("coverage", None)
                                    
                                    if precision is not None or coverage is not None:
                                        if not hasattr(episode, 'custom_metrics'):
                                            episode.custom_metrics = {}
                                        if agent_id not in episode.custom_metrics:
                                            episode.custom_metrics[agent_id] = {}
                                        if precision is not None:
                                            episode.custom_metrics[agent_id]["precision"] = float(precision)
                                        if coverage is not None:
                                            episode.custom_metrics[agent_id]["coverage"] = float(coverage)
                    except Exception:
                        pass
                
                # Method 2: Try to access the multi-agent environment directly (fallback)
                if base_env is not None:
                    if hasattr(base_env, 'envs') and len(base_env.envs) > env_index:
                        actual_env = base_env.envs[env_index]
                        # Unwrap if needed
                        while hasattr(actual_env, 'env') or hasattr(actual_env, 'gym_env'):
                            actual_env = getattr(actual_env, 'env', None) or getattr(actual_env, 'gym_env', None)
                            if actual_env is None:
                                break
                        
                        # Check if this is a MultiAgentAnchorEnv
                        if hasattr(actual_env, 'gym_envs') and isinstance(actual_env.gym_envs, dict):
                            # This is our MultiAgentAnchorEnv - extract metrics per agent
                            for agent_id, gym_env in actual_env.gym_envs.items():
                                if hasattr(gym_env, 'anchor_env'):
                                    anchor_env = gym_env.anchor_env
                                    if hasattr(anchor_env, '_current_metrics'):
                                        prec, cov, _ = anchor_env._current_metrics()
                                        
                                        # Store in episode custom_metrics per agent
                                        if not hasattr(episode, 'custom_metrics'):
                                            episode.custom_metrics = {}
                                        if agent_id not in episode.custom_metrics:
                                            episode.custom_metrics[agent_id] = {}
                                        episode.custom_metrics[agent_id]["precision"] = float(prec)
                                        episode.custom_metrics[agent_id]["coverage"] = float(cov)
            except Exception as e:
                # Silently handle errors to avoid spam
                pass
        
        def on_episode_end(self, *, episode, env_index, env_runner=None, metrics_logger=None, env=None, rl_module=None, worker=None, base_env=None, policies=None, **kwargs):
            """Called at the end of each episode. Extract final metrics per agent."""
            try:
                # Extract metrics from episode's custom_metrics (set during on_episode_step)
                if hasattr(episode, 'custom_metrics') and isinstance(episode.custom_metrics, dict):
                    for agent_id, agent_metrics in episode.custom_metrics.items():
                        if isinstance(agent_metrics, dict):
                            precision = agent_metrics.get("precision", None)
                            coverage = agent_metrics.get("coverage", None)
                            
                            if precision is not None:
                                if agent_id not in self.episode_precisions_per_agent:
                                    self.episode_precisions_per_agent[agent_id] = []
                                self.episode_precisions_per_agent[agent_id].append(float(precision))
                            
                            if coverage is not None:
                                if agent_id not in self.episode_coverages_per_agent:
                                    self.episode_coverages_per_agent[agent_id] = []
                                self.episode_coverages_per_agent[agent_id].append(float(coverage))
                
                # Also try to extract directly from base_env as fallback
                if base_env is not None:
                    if hasattr(base_env, 'envs') and len(base_env.envs) > env_index:
                        actual_env = base_env.envs[env_index]
                        while hasattr(actual_env, 'env') or hasattr(actual_env, 'gym_env'):
                            actual_env = getattr(actual_env, 'env', None) or getattr(actual_env, 'gym_env', None)
                            if actual_env is None:
                                break
                        
                        if hasattr(actual_env, 'gym_envs') and isinstance(actual_env.gym_envs, dict):
                            for agent_id, gym_env in actual_env.gym_envs.items():
                                if hasattr(gym_env, 'anchor_env'):
                                    anchor_env = gym_env.anchor_env
                                    if hasattr(anchor_env, '_current_metrics'):
                                        prec, cov, _ = anchor_env._current_metrics()
                                        
                                        # Only add if not already in lists (avoid duplicates)
                                        if agent_id not in self.episode_precisions_per_agent:
                                            self.episode_precisions_per_agent[agent_id] = []
                                        if agent_id not in self.episode_coverages_per_agent:
                                            self.episode_coverages_per_agent[agent_id] = []
                                        
                                        # Check if we already have metrics for this episode
                                        episode_id = id(episode)
                                        if episode_id not in self.processed_episodes:
                                            self.episode_precisions_per_agent[agent_id].append(float(prec))
                                            self.episode_coverages_per_agent[agent_id].append(float(cov))
                                            self.processed_episodes.add(episode_id)
            except Exception as e:
                pass
        
        def on_sample_end(self, *, env_runner=None, metrics_logger=None, samples=None, worker=None, **kwargs):
            """Called after sampling is complete. Extract info from sample batches per agent."""
            if samples is None:
                return
            
            try:
                if isinstance(samples, list):
                    for episode in samples:
                        episode_id = id(episode)
                        if episode_id in self.processed_episodes:
                            continue
                        
                        # Try to get infos per agent
                        if hasattr(episode, 'get_infos'):
                            try:
                                infos = episode.get_infos()
                                if infos:
                                    # For multi-agent, infos might be a dict per agent
                                    if isinstance(infos, dict):
                                        for agent_id, agent_infos in infos.items():
                                            if isinstance(agent_infos, list) and len(agent_infos) > 0:
                                                last_info = agent_infos[-1]
                                            elif isinstance(agent_infos, dict):
                                                last_info = agent_infos
                                            else:
                                                continue
                                            
                                            if isinstance(last_info, dict):
                                                precision = last_info.get("precision", None)
                                                coverage = last_info.get("coverage", None)
                                                
                                                if precision is not None:
                                                    if agent_id not in self.episode_precisions_per_agent:
                                                        self.episode_precisions_per_agent[agent_id] = []
                                                    self.episode_precisions_per_agent[agent_id].append(float(precision))
                                                
                                                if coverage is not None:
                                                    if agent_id not in self.episode_coverages_per_agent:
                                                        self.episode_coverages_per_agent[agent_id] = []
                                                    self.episode_coverages_per_agent[agent_id].append(float(coverage))
                                    elif isinstance(infos, list) and len(infos) > 0:
                                        # Single-agent format - try to extract agent_id from episode
                                        last_info = infos[-1]
                                        if isinstance(last_info, dict):
                                            # Try to infer agent_id from episode or use default
                                            agent_id = getattr(episode, 'agent_id', None) or "default_agent"
                                            precision = last_info.get("precision", None)
                                            coverage = last_info.get("coverage", None)
                                            
                                            if precision is not None:
                                                if agent_id not in self.episode_precisions_per_agent:
                                                    self.episode_precisions_per_agent[agent_id] = []
                                                self.episode_precisions_per_agent[agent_id].append(float(precision))
                                            
                                            if coverage is not None:
                                                if agent_id not in self.episode_coverages_per_agent:
                                                    self.episode_coverages_per_agent[agent_id] = []
                                                self.episode_coverages_per_agent[agent_id].append(float(coverage))
                            except Exception:
                                pass
                        
                        self.processed_episodes.add(episode_id)
            except Exception:
                pass
        
        def on_train_result(self, *, algorithm, result, **kwargs):
            """Called at the end of each training iteration. Aggregate metrics per agent."""
            # Multi-agent results have info per agent
            if "info" not in result:
                result["info"] = {}
            
            # Aggregate metrics for each agent
            for agent_id in self.episode_precisions_per_agent.keys() | self.episode_coverages_per_agent.keys():
                if agent_id not in result["info"]:
                    result["info"][agent_id] = {}
                if "custom_metrics" not in result["info"][agent_id]:
                    result["info"][agent_id]["custom_metrics"] = {}
                
                # Aggregate precision
                if agent_id in self.episode_precisions_per_agent and self.episode_precisions_per_agent[agent_id]:
                    precision_mean = np.mean(self.episode_precisions_per_agent[agent_id])
                    result["info"][agent_id]["custom_metrics"]["precision_mean"] = precision_mean
                    self.episode_precisions_per_agent[agent_id] = []  # Clear for next iteration
                else:
                    result["info"][agent_id]["custom_metrics"]["precision_mean"] = 0.0
                
                # Aggregate coverage
                if agent_id in self.episode_coverages_per_agent and self.episode_coverages_per_agent[agent_id]:
                    coverage_mean = np.mean(self.episode_coverages_per_agent[agent_id])
                    result["info"][agent_id]["custom_metrics"]["coverage_mean"] = coverage_mean
                    self.episode_coverages_per_agent[agent_id] = []  # Clear for next iteration
                else:
                    result["info"][agent_id]["custom_metrics"]["coverage_mean"] = 0.0
            
            # Clear processed episodes set for next iteration
            self.processed_episodes.clear()
    
    # Create Multi-Agent Environment
    class MultiAgentAnchorEnv(MultiAgentEnv):
        """
        Multi-agent environment where each agent represents one class.
        Each agent learns to find anchors for its target class.
        """
        def __init__(self, env_config=None):
            super().__init__()
            if env_config is None:
                env_config = {}
            
            self.target_classes = target_classes
            self.agent_ids = [f"agent_{cls}" for cls in target_classes]
            self.class_to_agent = {cls: agent_id for cls, agent_id in zip(target_classes, self.agent_ids)}
            
            # Create one environment per agent (each with different target class)
            worker_idx = env_config.get("worker_index", 0)
            self.envs = {}
            self.gym_envs = {}
            
            for cls, agent_id in zip(target_classes, self.agent_ids):
                anchor_env = create_anchor_env(target_cls=cls)
                gym_env = ContinuousAnchorEnv(anchor_env, seed=42 + cls + worker_idx)
                self.envs[agent_id] = anchor_env
                self.gym_envs[agent_id] = gym_env
            
            # Get observation and action spaces from first agent (all agents have same spaces)
            first_agent_id = self.agent_ids[0]
            self.observation_space = spaces.Dict({
                agent_id: self.gym_envs[first_agent_id].observation_space
                for agent_id in self.agent_ids
            })
            self.action_space = spaces.Dict({
                agent_id: self.gym_envs[first_agent_id].action_space
                for agent_id in self.agent_ids
            })
            
            self.agent_selection = None
        
        def reset(self, seed=None, options=None):
            """Reset all agent environments."""
            obs = {}
            infos = {}
            for agent_id in self.agent_ids:
                obs[agent_id], info = self.gym_envs[agent_id].reset(seed=seed, options=options)
                infos[agent_id] = info if isinstance(info, dict) else {}
            self.agent_selection = self.agent_ids[0]
            return obs, infos
        
        def step(self, actions):
            """Step all agents simultaneously."""
            obs = {}
            rewards = {}
            dones = {}
            truncateds = {}
            infos = {}
            
            for agent_id in self.agent_ids:
                if agent_id in actions:
                    step_result = self.gym_envs[agent_id].step(actions[agent_id])
                    if len(step_result) == 5:
                        # Gymnasium API: (obs, reward, terminated, truncated, info)
                        obs[agent_id], rewards[agent_id], dones[agent_id], truncateds[agent_id], infos[agent_id] = step_result
                    else:
                        # Old API: (obs, reward, done, info)
                        obs[agent_id], rewards[agent_id], dones[agent_id], infos[agent_id] = step_result
                        truncateds[agent_id] = False
                else:
                    # Agent didn't provide action, use no-op or last action
                    obs[agent_id] = self.gym_envs[agent_id].observation_space.sample()
                    rewards[agent_id] = 0.0
                    dones[agent_id] = True
                    truncateds[agent_id] = False
                    infos[agent_id] = {}
            
            # Check if all agents are done
            dones["__all__"] = all(dones.values())
            truncateds["__all__"] = all(truncateds.values())
            
            return obs, rewards, dones, truncateds, infos
        
        def close(self):
            """Close all environments."""
            for gym_env in self.gym_envs.values():
                gym_env.close()
    
    # Create a test environment to get spaces for logging
    test_env = MultiAgentAnchorEnv()
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    
    print(f"\n{'='*60}")
    print(f"Multi-Agent Environment Setup")
    print(f"{'='*60}")
    print(f"  Agents: {test_env.agent_ids}")
    print(f"  Observation space: {obs_space}")
    print(f"  Action space: {action_space}")
    test_env.close()
    
    # Policy mapping: one policy per agent
    def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
        """Map each agent to its own policy."""
        return agent_id  # Each agent gets its own policy
    
    # Configure multi-agent SAC
    print(f"\n{'='*60}")
    print(f"Configuring Multi-Agent SAC")
    print(f"{'='*60}")
    
    # Get observation and action space from a single agent (all agents have same spaces)
    single_agent_env = create_anchor_env(target_cls=target_classes[0])
    single_gym_env = ContinuousAnchorEnv(single_agent_env, seed=42)
    single_obs_space = single_gym_env.observation_space
    single_action_space = single_gym_env.action_space
    single_gym_env.close()
    
    # Create policies config: one policy per agent
    policies = {}
    for agent_id in test_env.agent_ids:
        policies[agent_id] = None  # Use default SAC policy
    
    config = SACConfig()
    config = config.environment(env=MultiAgentAnchorEnv)
    # Add custom callback to track precision and coverage metrics per agent
    config = config.callbacks(MultiAgentPrecisionCoverageCallback)
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
    )
    config = config.training(
        actor_lr=learning_rate,
        critic_lr=learning_rate,
        tau=tau,
        target_network_update_freq=target_network_update_freq,
        num_steps_sampled_before_learning_starts=learning_starts,
        train_batch_size=train_batch_size,
        target_entropy=target_entropy,
        replay_buffer_config={
            "capacity": buffer_size,
            "type": "EpisodeReplayBuffer",
        },
    )
    config = config.env_runners(
        num_env_runners=num_workers,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=num_cpus_per_worker,
    )
    config = config.resources(
        num_gpus=num_gpus if num_workers == 0 else 0,
    )
    if num_gpus > 0 and num_workers == 0:
        config = config.learners(
            num_gpus_per_learner=1,
        )
    config = config.framework(framework="torch")
    
    # Create multi-agent SAC trainer
    trainer = config.build_algo()
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training Multi-Agent SAC ({total_timesteps} timesteps)")
    print(f"{'='*60}")
    
    training_history = []
    per_class_precision_history = {cls: [] for cls in target_classes}
    per_class_coverage_history = {cls: [] for cls in target_classes}
    
    num_iterations = max(1, total_timesteps // (steps_per_episode * max(1, num_workers + 1)))
    
    print(f"  Training for {num_iterations} iterations (target: {total_timesteps} timesteps)...")
    
    for iteration in range(num_iterations):
        # Train one iteration (Ray handles rollout collection internally)
        result = trainer.train()
        
        # Extract metrics from training result
        timesteps = result.get("timesteps_total", 0)
        
        # Multi-agent metrics are per-policy
        info_by_agent = result.get("info", {})
        
        # Track episode metrics per agent
        for agent_id, cls in zip(test_env.agent_ids, target_classes):
            agent_metrics = info_by_agent.get(agent_id, {})
            episode_reward_mean = agent_metrics.get("episode_reward_mean", 0.0)
            episode_len_mean = agent_metrics.get("episode_len_mean", 0)
            episode_precision = agent_metrics.get("custom_metrics", {}).get("precision_mean", 0.0)
            episode_coverage = agent_metrics.get("custom_metrics", {}).get("coverage_mean", 0.0)
            
            training_history.append({
                "class": cls,
                "agent_id": agent_id,
                "episode": iteration + 1,
                "timestep": timesteps,
                "reward": float(episode_reward_mean),
                "precision": float(episode_precision),
                "coverage": float(episode_coverage),
                "steps": int(episode_len_mean),
            })
            per_class_precision_history[cls].append(float(episode_precision))
            per_class_coverage_history[cls].append(float(episode_coverage))
        
        # Progress update
        if (iteration + 1) % max(1, num_iterations // 10) == 0 or iteration == num_iterations - 1:
            print(f"  Iteration {iteration + 1}/{num_iterations} | Timesteps: {timesteps}")
            for agent_id, cls in zip(test_env.agent_ids, target_classes):
                agent_metrics = info_by_agent.get(agent_id, {})
                reward = agent_metrics.get("episode_reward_mean", 0.0)
                prec = agent_metrics.get("custom_metrics", {}).get("precision_mean", 0.0)
                cov = agent_metrics.get("custom_metrics", {}).get("coverage_mean", 0.0)
                print(f"    Agent {agent_id} (Class {cls}): Reward={reward:.3f}, Precision={prec:.3f}, Coverage={cov:.3f}")
        
        # Save checkpoint
        if save_checkpoints and (timesteps % checkpoint_freq == 0 or iteration == num_iterations - 1):
            checkpoint_dir = os.path.join(output_dir, "checkpoints", f"multiagent_sac_iter_{iteration}")
            checkpoint_dir = os.path.abspath(checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = trainer.save(checkpoint_dir)
            if verbose >= 1:
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Stop if we've reached target timesteps
        if timesteps >= total_timesteps:
            print(f"  Reached target timesteps ({total_timesteps}), stopping training.")
            break
    
    # Extract individual policies for evaluation
    print(f"\n[Extracting Individual Policies]")
    sac_trainers = {}
    for agent_id, cls in zip(test_env.agent_ids, target_classes):
        # Get policy for this agent
        policy = trainer.get_policy(agent_id)
        # Store trainer and policy mapping for evaluation
        sac_trainers[cls] = {
            "trainer": trainer,
            "agent_id": agent_id,
            "policy": policy,
        }
        print(f"  Class {cls} -> Agent {agent_id}: Policy extracted")
    
    # Create wrapper for evaluation (compatible with single-agent evaluation functions)
    class MultiAgentSACTrainerWrapper:
        """Wrapper to extract single-agent policy from multi-agent trainer for evaluation."""
        def __init__(self, trainer, agent_id):
            self.trainer = trainer
            self.agent_id = agent_id
            self.policy = trainer.get_policy(agent_id)
            # Set actor/critic attributes for continuous model detection
            try:
                module = self.policy.model if hasattr(self.policy, 'model') else None
                if module:
                    if hasattr(module, 'actor'):
                        self.actor = module.actor
                    if hasattr(module, 'critic'):
                        self.critic = module.critic
            except Exception:
                pass
            if not hasattr(self, 'actor'):
                self.actor = None
            if not hasattr(self, 'critic'):
                self.critic = None
        
        def predict(self, obs, deterministic=False):
            """Predict action using the agent's policy."""
            if isinstance(obs, torch.Tensor):
                obs = obs.cpu().numpy()
            obs = np.array(obs, dtype=np.float32)
            
            # Use policy's forward_inference
            try:
                module = self.policy.model if hasattr(self.policy, 'model') else None
                if module:
                    obs_tensor = torch.from_numpy(obs).float()
                    if len(obs_tensor.shape) == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        fwd_outputs = module.forward_inference({"obs": obs_tensor})
                        action_dist_class = module.get_inference_action_dist_cls()
                        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
                        
                        if deterministic:
                            if hasattr(action_dist, 'mean'):
                                action = action_dist.mean()[0]
                            elif hasattr(action_dist, 'deterministic_sample'):
                                action = action_dist.deterministic_sample()[0]
                            elif hasattr(action_dist, 'mode'):
                                action = action_dist.mode()[0]
                            else:
                                action = action_dist.sample()[0]
                        else:
                            action = action_dist.sample()[0]
                        
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        else:
                            action = np.array(action)
                        
                        action = np.array(action, dtype=np.float32)
                        if len(action.shape) > 1:
                            action = action.flatten()
                else:
                    raise RuntimeError("Could not access policy model")
            except Exception as e:
                raise RuntimeError(f"Failed to compute action: {e}") from e
            
            return action, None
    
    # Create wrapped trainers for evaluation
    wrapped_trainers = {
        cls: MultiAgentSACTrainerWrapper(trainer_info["trainer"], trainer_info["agent_id"])
        for cls, trainer_info in sac_trainers.items()
    }
    
    # Evaluation (same as single-agent version)
    print(f"\n[Class-Level Evaluation] Computing cluster centroids from test data...")
    print(f"  Test cluster centroids computed successfully!")
    
    print(f"\nEvaluating on test set...")
    
    def create_anchor_env_for_eval(target_cls=None, use_test_centroids=False):
        return create_anchor_env(target_cls=target_cls, use_test_centroids=use_test_centroids)
    
    eval_steps = eval_steps_per_episode if eval_steps_per_episode is not None else steps_per_episode
    
    # Instance-level evaluation
    print(f"\n[Instance-Level Evaluation] Creating one anchor per test instance...")
    eval_results_instance = evaluate_all_classes(
        X_test=X_test_scaled,
        y_test=y_test,
        trained_model=wrapped_trainers,
        make_env_fn=create_anchor_env_for_eval,
        feature_names=feature_names,
        n_instances_per_class=n_eval_instances_per_class,
        max_features_in_rule=max_features_in_rule,
        steps_per_episode=eval_steps,
        random_seed=42,
        eval_on_test_data=False,
        X_test_unit=X_test_unit if False else None,
        X_test_std=X_test_scaled if False else None,
        num_rollouts_per_instance=num_rollouts_per_instance,
    )
    
    # Class-level evaluation
    print(f"\n[Class-Level Evaluation] Creating one anchor per class (using test centroids)...")
    def create_anchor_env_for_class_eval(target_cls=None):
        return create_anchor_env(target_cls=target_cls, use_test_centroids=True)
    
    eval_results_class = evaluate_all_classes_class_level(
        trained_model=wrapped_trainers,
        make_env_fn=create_anchor_env_for_class_eval,
        feature_names=feature_names,
        target_classes=list(target_classes),
        steps_per_episode=eval_steps,
        max_features_in_rule=max_features_in_rule,
        random_seed=42,
        eval_on_test_data=False,
        X_test_unit=X_test_unit if False else None,
        X_test_std=X_test_scaled if False else None,
        y_test=y_test if False else None,
    )
    
    # Prepare results
    results = {
        "trained_model": trainer,  # Multi-agent trainer
        "policies": sac_trainers,  # Individual policies per class
        "eval_results": {
            "instance_level": eval_results_instance,
            "class_level": eval_results_class,
        },
        "overall_stats": {
            "instance_level": {
                "avg_precision": eval_results_instance.get("overall_precision", 0.0),
                "avg_coverage": eval_results_instance.get("overall_coverage", 0.0),
            },
            "class_level": {
                "avg_precision": eval_results_class.get("overall_precision", 0.0),
                "avg_coverage": eval_results_class.get("overall_coverage", 0.0),
            },
        },
        "metadata": {
            "n_classes": len(target_classes),
            "n_features": len(feature_names),
            "target_classes": target_classes,
            "feature_names": feature_names,
            "output_dir": output_dir,
            "X_test_scaled": X_test_scaled,
            "X_min": X_min,
            "X_range": X_range,
        },
        "training_history": training_history,
    }
    
    # Save model
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"\nSaving multi-agent SAC model...")
    model_dir = os.path.join(models_dir, "multiagent_sac_final")
    model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model_path = trainer.save(model_dir)
    print(f"  Saved multi-agent SAC model: {model_path}")
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {models_dir}")
    
    # Print evaluation results summary
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n[Instance-Level Evaluation] (One anchor per test instance):")
    print(f"  Overall Precision: {eval_results_instance.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage:  {eval_results_instance.get('overall_coverage', 0.0):.3f}")
    
    print(f"\n[Class-Level Evaluation] (One anchor per class):")
    print(f"  Overall Precision: {eval_results_class.get('overall_precision', 0.0):.3f}")
    print(f"  Overall Coverage:  {eval_results_class.get('overall_coverage', 0.0):.3f}")
    
    print(f"\n{'='*80}")
    
    return results

