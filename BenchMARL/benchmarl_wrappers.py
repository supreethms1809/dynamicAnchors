"""
This code was created by following the tutorial in the BenchMARL documentation.
"""
import copy
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import torch

from torchrl.data import Composite
from torchrl.envs import EnvBase, PettingZooEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from tensordict import TensorDictBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.experiment.callback import Callback
from benchmarl.utils import DEVICE_TYPING

import sys
import os
import logging
logger = logging.getLogger(__name__)

benchmarl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmarl_dir)
from environment import AnchorEnv

class AnchorTaskClass(TaskClass):
    
    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        
        # SS: Track where this max_cycles is set. The reading from yaml is not getting passed to the environment.
        # Below code needs fixing.
        max_cycles = config.get("max_cycles", 500)

        env_config = {k: v for k, v in config.items() if k != "max_cycles"}
        
        if "env_config" not in env_config:
            env_config["env_config"] = {}
        if not isinstance(env_config["env_config"], dict):
            env_config["env_config"] = {}
        env_config["env_config"]["max_cycles"] = max_cycles
        
        # Verify agents_per_class is in env_config (for debugging)
        agents_per_class = env_config.get("env_config", {}).get("agents_per_class", None)
        if agents_per_class is not None:
            logger.info(f"  agents_per_class={agents_per_class} found in env_config")
        else:
            # Using default value of 1 is expected and fine, so use info level instead of warning
            logger.debug(f"  agents_per_class not found in env_config, using default (1)")
        
        def _make_env():
            anchor_env = AnchorEnv(**env_config)
            
            return PettingZooWrapper(
                env=anchor_env,
                return_state=False,
                categorical_actions=False,
                seed=seed,
                device=device,
                use_mask=True,
            )
        
        return _make_env
    
    def supports_continuous_actions(self) -> bool:
        return True
    
    def supports_discrete_actions(self) -> bool:
        return False
    
    def has_state(self) -> bool:
        return False
    
    def has_render(self, env: EnvBase) -> bool:
        return False
    
    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_cycles", 500)
    
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return env.group_map
    
    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None
    
    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None
    
    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "observation":
                    del group_obs_spec[key]
        return observation_spec
    
    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        observation_spec = env.observation_spec.clone()
        for group in self.group_map(env):
            group_obs_spec = observation_spec[group]
            for key in list(group_obs_spec.keys()):
                if key != "info":
                    del group_obs_spec[key]
        if observation_spec.is_empty():
            return None
        return observation_spec
    
    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec
    
    @staticmethod
    def env_name() -> str:
        return "anchor"


class AnchorTask(Task):
    
    ANCHOR = None
    
    @staticmethod
    def associated_class():
        return AnchorTaskClass

# SS: This is the callback that is used to collect the metrics and the anchor data.
# Bug: This is not working right now.
class AnchorMetricsCallback(Callback):
    
    def __init__(self, log_training_metrics: bool = True, log_evaluation_metrics: bool = True, save_to_file: bool = True, collect_anchor_data: bool = False, save_frequency: int = 10, save_during_training: bool = True, save_best_model: bool = True):
        super().__init__()
        self.log_training_metrics = log_training_metrics
        self.log_evaluation_metrics = log_evaluation_metrics
        self.save_to_file = save_to_file
        self.collect_anchor_data = collect_anchor_data
        self.save_frequency = save_frequency  # Save every N aggregated metrics
        self.save_during_training = save_during_training  # Whether to save periodically during training
        self.save_best_model = save_best_model  # Whether to save best model based on evaluation
        self.training_metrics = []
        self.training_history = []
        self.evaluation_history = []
        # Store anchor data collected during evaluation
        self.evaluation_anchor_data = []  # List of episodes, each episode contains agent data
        # Store training episode details (bounds, precision, coverage per episode)
        self.training_episode_details = []  # List of episode details from training batches
        self.experiment_folder = None  # Will be set when experiment starts
        # Track best model
        self.best_eval_score = -float('inf')  # Track best evaluation score (precision + coverage)
        self.best_model_path = None  # Path to best model checkpoint
        
        # Track best models per class (for multi-agent equilibrium evaluation)
        self.best_eval_score_per_class = {}  # class -> best score
        self.best_model_path_per_class = {}  # class -> best model path
        
        # Equilibrium evaluation: track if all classes meet targets
        self.equilibrium_eval_mode = True  # If True, save best model when all classes meet targets
        self.best_equilibrium_score = -float('inf')  # Track best equilibrium score (min class score when all meet targets)
    
    def _extract_class_from_group_name(self, group_name: str) -> Optional[int]:
        """
        Extract class ID from agent/group name.
        Handles both formats:
        - agent_0, agent_1 (agents_per_class == 1)
        - agent_0_0, agent_0_1, agent_1_0 (agents_per_class > 1)
        Returns None if format is unrecognized.
        """
        try:
            # Remove 'agent_' prefix
            if group_name.startswith("agent_"):
                parts = group_name.replace("agent_", "").split("_")
                # First part is always the class ID
                if parts:
                    return int(parts[0])
        except (ValueError, AttributeError):
            pass
        return None
    
    def _extract_metrics_from_info(self, info: TensorDictBase, prefix: str = "") -> Dict[str, float]:
        metrics = {}
        
        if info is None:
            return metrics
        
        metric_keys = [
            "anchor_precision", "anchor_coverage",
            "class_union_precision", "class_union_coverage",  # Union metrics for multi-agent per class
            "drift", "anchor_drift", "js_penalty",
            "precision_gain", "coverage_gain", "coverage_bonus", "target_class_bonus",
            "overlap_penalty", "drift_penalty", "anchor_drift_penalty", 
            "inter_class_overlap_penalty", "shared_reward", "total_reward",
            "coverage_floor_hits", "coverage_clipped"
        ]
        
        for key in metric_keys:
            if key in info.keys():
                value = info[key]
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        metrics[f"{prefix}{key}"] = value.item()
                    elif value.numel() > 0:
                        metrics[f"{prefix}{key}_mean"] = value.float().mean().item()
                        metrics[f"{prefix}{key}_std"] = value.float().std().item()
                elif isinstance(value, (int, float, np.number)):
                    metrics[f"{prefix}{key}"] = float(value)
        
        return metrics
    
    def on_batch_collected(self, batch: TensorDictBase):
        if not self.log_training_metrics:
            return
        
        all_metrics = {}
        episode_details = {}
        
        for group in self.experiment.group_map.keys():
            group_key = ("next", group, "info")
            obs_key = ("next", group, "observation")
            
            # Extract info metrics
            if group_key in batch.keys(include_nested=True):
                info = batch[group_key]
                
                if info.shape[0] > 0:
                    final_info = info[-1]
                    group_metrics = self._extract_metrics_from_info(
                        final_info, 
                        prefix=f"training/{group}/"
                    )
                    all_metrics.update(group_metrics)
                    
                    # Extract final state information (bounds, precision, coverage)
                    episode_detail = {}
                    
                    # Get precision and coverage from info
                    def safe_get(key, default=0.0):
                        try:
                            if hasattr(final_info, 'get'):
                                val = final_info.get(key, default)
                            elif hasattr(final_info, 'keys') and key in final_info.keys():
                                val = final_info[key]
                            else:
                                val = getattr(final_info, key, default)
                            
                            if isinstance(val, torch.Tensor):
                                return float(val.item() if val.numel() == 1 else val[-1].item())
                            return float(val)
                        except:
                            return default
                    
                    episode_detail["anchor_precision"] = safe_get("anchor_precision", 0.0)
                    episode_detail["anchor_coverage"] = safe_get("anchor_coverage", 0.0)
                    episode_detail["total_reward"] = safe_get("total_reward", 0.0)
                    
                    # Extract final observation (contains bounds)
                    # SS: This is very messy. We need to fix this. (its messy but it works - change later)
                    obs = None
                    obs_keys_to_try = [
                        obs_key,  # ("next", group, "observation")
                        (group, "observation"),  # (group, "observation")
                        ("next", group, "observation"),  # Explicit nested
                    ]
                    
                    for key in obs_keys_to_try:
                        try:
                            if key in batch.keys(include_nested=True):
                                obs = batch[key]
                                break
                        except (KeyError, TypeError):
                            # Try accessing directly if nested key check fails
                            try:
                                if isinstance(key, tuple) and len(key) == 3:
                                    # Try accessing as batch["next"][group]["observation"]
                                    if "next" in batch.keys() and group in batch["next"].keys():
                                        if "observation" in batch["next"][group].keys():
                                            obs = batch["next"][group]["observation"]
                                            break
                                elif isinstance(key, tuple) and len(key) == 2:
                                    # Try accessing as batch[group]["observation"]
                                    if group in batch.keys() and "observation" in batch[group].keys():
                                        obs = batch[group]["observation"]
                                        break
                            except (KeyError, TypeError, AttributeError):
                                continue
                    
                    if obs is not None and hasattr(obs, 'shape') and obs.shape[0] > 0:
                        final_obs = obs[-1]
                        
                        # Convert to numpy if tensor
                        if isinstance(final_obs, torch.Tensor):
                            final_obs = final_obs.cpu().numpy()
                        elif not isinstance(final_obs, np.ndarray):
                            final_obs = np.array(final_obs)
                        
                        # Observation structure: [lower_bounds (n_features), upper_bounds (n_features), precision, coverage]
                        # We need to infer n_features from the observation length
                        obs_len = len(final_obs) if hasattr(final_obs, '__len__') else final_obs.shape[0] if hasattr(final_obs, 'shape') else 0
                        if obs_len >= 4:  # At least 2 features + precision + coverage
                            # n_features = (obs_len - 2) / 2
                            n_features = (obs_len - 2) // 2
                            
                            if n_features > 0:
                                lower_bounds = final_obs[:n_features]
                                upper_bounds = final_obs[n_features:2*n_features]
                                
                                episode_detail["lower_bounds"] = lower_bounds.tolist() if hasattr(lower_bounds, 'tolist') else list(lower_bounds)
                                episode_detail["upper_bounds"] = upper_bounds.tolist() if hasattr(upper_bounds, 'tolist') else list(upper_bounds)
                                
                                # Calculate box metrics
                                box_widths = upper_bounds - lower_bounds
                                episode_detail["box_widths"] = box_widths.tolist() if hasattr(box_widths, 'tolist') else list(box_widths)
                                episode_detail["box_volume"] = float(np.prod(np.maximum(box_widths, 1e-9)))
                                
                                # Add bounds summary metrics to aggregated metrics
                                all_metrics[f"training/{group}/box_volume"] = episode_detail["box_volume"]
                                all_metrics[f"training/{group}/mean_box_width"] = float(np.mean(box_widths))
                                all_metrics[f"training/{group}/min_box_width"] = float(np.min(box_widths))
                                all_metrics[f"training/{group}/max_box_width"] = float(np.max(box_widths))
                    
                    episode_detail["group"] = group
                    episode_detail["step"] = self.experiment.n_iters_performed if hasattr(self.experiment, 'n_iters_performed') else None
                    episode_detail["total_frames"] = self.experiment.total_frames if hasattr(self.experiment, 'total_frames') else None
                    
                    episode_details[group] = episode_detail
        
        if all_metrics:
            self.training_metrics.append(all_metrics)
            
            # Store episode details if available
            if episode_details:
                self.training_episode_details.append(episode_details.copy())
            
            # Aggregate and save when we have enough batches, or save remaining data at end
            if len(self.training_metrics) >= self.save_frequency:
                aggregated = {}
                for key in self.training_metrics[0].keys():
                    values = [m[key] for m in self.training_metrics if key in m]
                    if values:
                        aggregated[key] = sum(values) / len(values)
                
                # Try to log to wandb, but handle case where run is finished
                # SS: This is not working right now.
                try:
                    self.experiment.logger.log(
                        aggregated,
                        step=self.experiment.n_iters_performed
                    )
                except Exception as e:
                    # Handle wandb run finished error gracefully
                    if "wandb" in str(type(e)).lower() or "finished" in str(e).lower():
                        logger.warning(f"Warning: Could not log to wandb (run may be finished): {e}")
                        # Still save to file even if wandb logging fails
                    else:
                        # Re-raise if it's a different error
                        raise
                
                if self.save_to_file:
                    aggregated["step"] = self.experiment.n_iters_performed
                    aggregated["total_frames"] = self.experiment.total_frames
                    self.training_history.append(aggregated.copy())
                    
                    # Save periodically during training if enabled
                    if self.save_during_training and self.experiment_folder:
                        try:
                            self._save_data_periodically()
                        except Exception as e:
                            logger.warning(f"Warning: Could not save metrics during training: {e}")
                
                self.training_metrics = []
    
    # SS: This function is way too messy. We need to fix this. Its not working right now.
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not self.log_evaluation_metrics:
            return
        
        all_metrics = {}
        n_episodes = 0
        
        # Collect anchor data if requested
        if self.collect_anchor_data:
            # Initialize if not exists, otherwise keep existing data (append mode)
            if not hasattr(self, 'evaluation_anchor_data'):
                self.evaluation_anchor_data = []

        # SS: REMOVE THIS DEBUG LATER
        # # Debug: check if rollouts is empty
        # if not rollouts:
        #     logger.warning("Warning: on_evaluation_end received empty rollouts list")
        #     logger.warning(f"  Callback collect_anchor_data={self.collect_anchor_data}")
        #     logger.warning(f"  Experiment group_map: {getattr(self.experiment, 'group_map', 'N/A')}")
        #     return
        
        # logger.info(f"Debug: on_evaluation_end received {len(rollouts)} rollouts")
        # if rollouts:
        #     logger.info(f"  First rollout type: {type(rollouts[0])}")
        #     if hasattr(rollouts[0], 'keys'):
        #         logger.info(f"  First rollout keys: {list(rollouts[0].keys())}")
        
        for rollout in rollouts:
            n_episodes += 1
            episode_data = {}
            
            # Rollout structure: ['agent', 'done', 'terminated', 'truncated', 'next']
            # Data is nested: rollout['next']['agent'] or rollout['agent'] for current state
            # Info is in: rollout['next']['agent']['info'] or rollout['next'][group]['info']
            
            rollout_keys = list(rollout.keys()) if hasattr(rollout, 'keys') else []
            
            # Check if we have 'next' key (contains next state info)
            next_data = None
            if "next" in rollout_keys:
                next_data = rollout["next"]
            elif hasattr(rollout, 'get'):
                next_data = rollout.get("next", None)
            
            # Also check for direct 'agent' key
            agent_data = None
            if "agent" in rollout_keys:
                agent_data = rollout["agent"]
            elif hasattr(rollout, 'get'):
                agent_data = rollout.get("agent", None)
            
            # Try to get groups from experiment, or use 'agent' as default
            groups_to_check = list(self.experiment.group_map.keys()) if hasattr(self.experiment, 'group_map') else []
            if not groups_to_check and agent_data is not None:
                # If no groups, try 'agent' directly
                groups_to_check = ["agent"]
            
            # If we have next_data, check for nested groups
            if next_data is not None:
                if hasattr(next_data, 'keys'):
                    next_keys = list(next_data.keys())
                    # Check if groups are nested in 'next'
                    for group in groups_to_check:
                        if group in next_keys:
                            groups_to_check = [group]
                            break
                    # If no groups found, try 'agent' in next
                    if "agent" in next_keys and not any(g in next_keys for g in groups_to_check):
                        groups_to_check = ["agent"]
            
            # Extract data for each group
            for group in groups_to_check:
                group_data = None
                group_info = None
                group_obs = None
                
                # Try to get group data from 'next'
                if next_data is not None:
                    if hasattr(next_data, 'keys') and group in next_data.keys():
                        group_data = next_data[group]
                    elif hasattr(next_data, 'get'):
                        group_data = next_data.get(group, None)
                
                # Fallback to direct access
                if group_data is None:
                    if group in rollout_keys:
                        group_data = rollout[group]
                    elif agent_data is not None and group == "agent":
                        group_data = agent_data
                
                if group_data is None:
                    continue
                
                # Extract info from group_data
                if hasattr(group_data, 'keys') and "info" in group_data.keys():
                    group_info = group_data["info"]
                elif hasattr(group_data, 'get'):
                    group_info = group_data.get("info", None)
                
                # Extract observation
                if hasattr(group_data, 'keys') and "observation" in group_data.keys():
                    group_obs = group_data["observation"]
                elif hasattr(group_data, 'get'):
                    group_obs = group_data.get("observation", None)
                
                # Process info if available
                if group_info is not None:
                    # Handle TensorDict or tensor
                    if hasattr(group_info, 'shape') and group_info.shape[0] > 0:
                        # Get final info (last step of episode)
                        final_info = group_info[-1] if len(group_info.shape) > 0 else group_info
                        
                        # Extract metrics
                        group_metrics = self._extract_metrics_from_info(
                            final_info,
                            prefix=f"evaluation/{group}/"
                        )
                        
                        for key, value in group_metrics.items():
                            if key not in all_metrics:
                                all_metrics[key] = []
                            all_metrics[key].append(value)
                        
                        # Extract bounds and box metrics from observation
                        if group_obs is not None:
                            try:
                                # Get final observation (last step of episode)
                                if hasattr(group_obs, 'shape') and group_obs.shape[0] > 0:
                                    final_obs = group_obs[-1] if len(group_obs.shape) > 1 else group_obs
                                else:
                                    final_obs = group_obs
                                
                                # Convert to numpy if tensor
                                if isinstance(final_obs, torch.Tensor):
                                    final_obs = final_obs.cpu().numpy()
                                elif not isinstance(final_obs, np.ndarray):
                                    final_obs = np.array(final_obs)
                                
                                # Observation structure: [lower_bounds (n_features), upper_bounds (n_features), precision, coverage]
                                obs_len = len(final_obs) if hasattr(final_obs, '__len__') else final_obs.shape[0] if hasattr(final_obs, 'shape') else 0
                                if obs_len >= 4:  # At least 2 features + precision + coverage
                                    n_features = (obs_len - 2) // 2
                                    
                                    if n_features > 0:
                                        lower_bounds = final_obs[:n_features]
                                        upper_bounds = final_obs[n_features:2*n_features]
                                        
                                        # Calculate box metrics
                                        box_widths = upper_bounds - lower_bounds
                                        box_volume = float(np.prod(np.maximum(box_widths, 1e-9)))
                                        
                                        # Add box metrics to aggregated metrics for wandb logging
                                        box_metrics = {
                                            f"evaluation/{group}/box_volume": box_volume,
                                            f"evaluation/{group}/mean_box_width": float(np.mean(box_widths)),
                                            f"evaluation/{group}/min_box_width": float(np.min(box_widths)),
                                            f"evaluation/{group}/max_box_width": float(np.max(box_widths)),
                                        }
                                        
                                        for key, value in box_metrics.items():
                                            if key not in all_metrics:
                                                all_metrics[key] = []
                                            all_metrics[key].append(value)
                            except Exception as e:
                                # If bounds extraction fails, continue without box metrics
                                pass
                        
                        # Collect anchor data for rule extraction
                        if self.collect_anchor_data:
                            # Extract metrics from final info
                            def safe_get(key, default=0.0):
                                try:
                                    if hasattr(final_info, 'get'):
                                        val = final_info.get(key, default)
                                    elif hasattr(final_info, 'keys') and key in final_info.keys():
                                        val = final_info[key]
                                    else:
                                        val = getattr(final_info, key, default)
                                    
                                    if isinstance(val, torch.Tensor):
                                        if val.numel() == 1:
                                            return float(val.item())
                                        else:
                                            return float(val[-1].item()) if len(val.shape) > 0 else float(val.item())
                                    return float(val)
                                except:
                                    return default
                            
                            precision_val = safe_get("anchor_precision", 0.0)
                            coverage_val = safe_get("anchor_coverage", 0.0)
                            episode_data[group] = {
                                "anchor_precision": precision_val,
                                "anchor_coverage": coverage_val,
                                "total_reward": safe_get("total_reward", 0.0),
                            }
                            
                            # Get anchor bounds from observation
                            if group_obs is not None:
                                if hasattr(group_obs, 'shape') and group_obs.shape[0] > 0:
                                    # Last observation contains final anchor state
                                    final_obs = group_obs[-1] if len(group_obs.shape) > 1 else group_obs
                                    if isinstance(final_obs, torch.Tensor):
                                        episode_data[group]["final_observation"] = final_obs.cpu().numpy().tolist()
                                    else:
                                        episode_data[group]["final_observation"] = np.array(final_obs).tolist()
            
            if self.collect_anchor_data and episode_data:
                self.evaluation_anchor_data.append(episode_data)
                logger.info(f"  Collected anchor data for episode {n_episodes}: {list(episode_data.keys())}")
        
        if all_metrics:
            aggregated = {}
            for key, values in all_metrics.items():
                if values:
                    aggregated[f"{key}_mean"] = sum(values) / len(values)
                    aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
                    aggregated[f"{key}_min"] = min(values)
                    aggregated[f"{key}_max"] = max(values)
            
            aggregated["evaluation/n_episodes"] = n_episodes
            
            # Try to log to wandb, but handle case where run is finished
            try:
                self.experiment.logger.log(
                    aggregated,
                    step=self.experiment.n_iters_performed
                )
            except Exception as e:
                # Handle wandb run finished error gracefully
                if "wandb" in str(type(e)).lower() or "finished" in str(e).lower():
                    logger.warning(f"Warning: Could not log to wandb (run may be finished): {e}")
                    # Still save to file even if wandb logging fails
                else:
                    # Re-raise if it's a different error
                    raise
            
            if self.save_to_file:
                aggregated["step"] = self.experiment.n_iters_performed
                aggregated["total_frames"] = self.experiment.total_frames
                self.evaluation_history.append(aggregated.copy())
            
            precision_key = [k for k in aggregated.keys() if "anchor_precision" in k and "mean" in k and "evaluation" in k]
            coverage_key = [k for k in aggregated.keys() if "anchor_coverage" in k and "mean" in k and "evaluation" in k]
            if precision_key and coverage_key:
                precision = aggregated[precision_key[0]]
                coverage = aggregated[coverage_key[0]]
                logger.info(
                    f"Evaluation - Precision: {precision:.4f}, "
                    f"Coverage: {coverage:.4f} "
                    f"(n={n_episodes})"
                )
                
                # Multi-agent best model selection: track per-class metrics and evaluate equilibrium
                if self.save_best_model and self.experiment is not None:
                    # Extract per-class metrics for equilibrium evaluation
                    class_metrics = {}  # class -> list of (precision, coverage, score) from agents
                    class_union_metrics = {}  # class -> (union_precision, union_coverage, score)
                    groups_to_check = list(self.experiment.group_map.keys()) if hasattr(self.experiment, 'group_map') else []
                    
                    # Determine if we have multiple agents per class by checking group names
                    # If we see agent_0_0, agent_0_1, etc., we have multiple agents per class
                    agents_per_class = 1
                    for group in groups_to_check:
                        if group.startswith("agent_"):
                            parts = group.replace("agent_", "").split("_")
                            if len(parts) >= 2:  # Has format agent_class_idx (e.g., agent_0_0, agent_0_1)
                                class_prefix = parts[0]
                                # Count how many groups start with agent_{class_prefix}_
                                count = len([g for g in groups_to_check if g.startswith(f"agent_{class_prefix}_")])
                                agents_per_class = max(agents_per_class, count)
                    
                    for group in groups_to_check:
                        class_id = self._extract_class_from_group_name(group)
                        if class_id is not None:
                            # When agents_per_class > 1: use union metrics (what class achieves when all agents work together)
                            # When agents_per_class == 1: use individual agent metrics
                            if agents_per_class > 1:
                                # Try to get union metrics (represent class performance when all agents work together)
                                # Union metrics are the same for all agents in a class, so we only need to store once
                                union_precision_key = f"evaluation/{group}/class_union_precision_mean"
                                union_coverage_key = f"evaluation/{group}/class_union_coverage_mean"
                                
                                union_precision = aggregated.get(union_precision_key, 0.0)
                                union_coverage = aggregated.get(union_coverage_key, 0.0)
                                
                                # Store union metrics once per class (they're the same for all agents in the class)
                                if class_id not in class_union_metrics:
                                    if union_precision > 0 or union_coverage > 0:
                                        # Union metrics available - use them
                                        class_union_metrics[class_id] = (union_precision, union_coverage, 
                                                                         union_precision + union_coverage)
                                    else:
                                        # Union metrics not available - fallback to individual agent metrics
                                        # This shouldn't happen if environment is configured correctly, but handle gracefully
                                        group_precision_key = f"evaluation/{group}/anchor_precision_mean"
                                        group_coverage_key = f"evaluation/{group}/anchor_coverage_mean"
                                        group_precision = aggregated.get(group_precision_key, 0.0)
                                        group_coverage = aggregated.get(group_coverage_key, 0.0)
                                        if group_precision > 0 or group_coverage > 0:
                                            class_union_metrics[class_id] = (group_precision, group_coverage,
                                                                             group_precision + group_coverage)
                            else:
                                # Single agent per class: use individual agent metrics
                                group_precision_key = f"evaluation/{group}/anchor_precision_mean"
                                group_coverage_key = f"evaluation/{group}/anchor_coverage_mean"
                                
                                group_precision = aggregated.get(group_precision_key, 0.0)
                                group_coverage = aggregated.get(group_coverage_key, 0.0)
                                group_score = group_precision + group_coverage
                                
                                if class_id not in class_metrics:
                                    class_metrics[class_id] = []
                                class_metrics[class_id].append((group_precision, group_coverage, group_score))
                    
                    # Build class_scores: use union metrics if available (multi-agent per class), otherwise individual metrics
                    class_scores = {}
                    
                    # First, use union metrics for classes with multiple agents (agents_per_class > 1)
                    for class_id, (p, c, s) in class_union_metrics.items():
                        class_scores[class_id] = (p, c, s)
                    
                    # Then, add individual metrics for classes with single agent (agents_per_class == 1)
                    for class_id, metrics_list in class_metrics.items():
                        if class_id not in class_scores and metrics_list:
                            # For single agent per class, there should be only one metric
                            if len(metrics_list) == 1:
                                p, c, s = metrics_list[0]
                                class_scores[class_id] = (p, c, s)
                            else:
                                # Fallback: use mean if somehow multiple metrics exist (shouldn't happen)
                                avg_precision = sum(m[0] for m in metrics_list) / len(metrics_list)
                                avg_coverage = sum(m[1] for m in metrics_list) / len(metrics_list)
                                avg_score = sum(m[2] for m in metrics_list) / len(metrics_list)
                                class_scores[class_id] = (avg_precision, avg_coverage, avg_score)
                    
                    # Evaluate equilibrium: check if all classes meet targets
                    # Get targets from environment if available (may be wrapped)
                    precision_target = 0.95  # Default
                    coverage_target = 0.5    # Default
                    env = None
                    if hasattr(self.experiment, 'env'):
                        env = self.experiment.env
                        # Try to unwrap if it's a wrapped environment
                        while hasattr(env, 'env') or hasattr(env, '_env'):
                            env = getattr(env, 'env', None) or getattr(env, '_env', None)
                            if env is None:
                                break
                    if env is not None:
                        if hasattr(env, 'precision_target'):
                            precision_target = env.precision_target
                        if hasattr(env, 'coverage_target'):
                            coverage_target = env.coverage_target
                    
                    # Evaluate equilibrium: check if all classes meet targets
                    all_classes_meet_targets = True
                    classes_meeting_targets = []
                    classes_not_meeting_targets = []
                    equilibrium_details = {}
                    n_classes_total = 0
                    n_classes_meeting = 0
                    equilibrium_fraction = 0.0
                    
                    if class_scores:
                        n_classes_total = len(class_scores)
                        for class_id, (p, c, s) in class_scores.items():
                            meets_precision = p >= precision_target
                            meets_coverage = c >= coverage_target
                            meets_both = meets_precision and meets_coverage
                            
                            equilibrium_details[class_id] = {
                                "precision": p,
                                "coverage": c,
                                "score": s,
                                "meets_precision": meets_precision,
                                "meets_coverage": meets_coverage,
                                "meets_both": meets_both,
                                "precision_gap": max(0, precision_target - p),
                                "coverage_gap": max(0, coverage_target - c)
                            }
                            
                            if meets_both:
                                classes_meeting_targets.append(class_id)
                                n_classes_meeting += 1
                            else:
                                all_classes_meet_targets = False
                                classes_not_meeting_targets.append(class_id)
                        
                        equilibrium_fraction = n_classes_meeting / n_classes_total if n_classes_total > 0 else 0.0
                    
                    # Log equilibrium status (always, not just when reached)
                    if class_scores:
                        if all_classes_meet_targets:
                            logger.info(f"  ✓ EQUILIBRIUM: All {n_classes_total} classes meet targets!")
                        else:
                            logger.info(
                                f"  Equilibrium status: {n_classes_meeting}/{n_classes_total} classes meet targets "
                                f"({equilibrium_fraction:.1%})"
                            )
                            if classes_not_meeting_targets:
                                for class_id in classes_not_meeting_targets:
                                    details = equilibrium_details[class_id]
                                    gaps = []
                                    if not details["meets_precision"]:
                                        gaps.append(f"P gap: {details['precision_gap']:.4f}")
                                    if not details["meets_coverage"]:
                                        gaps.append(f"C gap: {details['coverage_gap']:.4f}")
                                    logger.info(
                                        f"    Class {class_id}: P={details['precision']:.4f} (target: {precision_target:.2f}), "
                                        f"C={details['coverage']:.4f} (target: {coverage_target:.4f}) - {', '.join(gaps)}"
                                    )
                    
                    # Add equilibrium metrics to aggregated for wandb/file logging
                    if class_scores:
                        aggregated["evaluation/equilibrium_fraction"] = equilibrium_fraction
                        aggregated["evaluation/equilibrium_reached"] = 1.0 if all_classes_meet_targets else 0.0
                        aggregated["evaluation/classes_meeting_targets"] = float(n_classes_meeting)
                        aggregated["evaluation/total_classes"] = float(n_classes_total)
                        
                        # Add per-class equilibrium status
                        for class_id, details in equilibrium_details.items():
                            aggregated[f"evaluation/class_{class_id}/meets_targets"] = 1.0 if details["meets_both"] else 0.0
                            aggregated[f"evaluation/class_{class_id}/precision_gap"] = details["precision_gap"]
                            aggregated[f"evaluation/class_{class_id}/coverage_gap"] = details["coverage_gap"]
                    
                    # Save best model based on strategy
                    save_model = False
                    save_reason = ""
                    
                    # Strategy 1: Equilibrium-based (all classes meet targets)
                    if self.equilibrium_eval_mode and all_classes_meet_targets:
                        # Check if this is better than previous equilibrium
                        if class_scores:
                            min_class_score = min(s for _, _, s in class_scores.values())
                            if not hasattr(self, 'best_equilibrium_score') or min_class_score > self.best_equilibrium_score:
                                save_model = True
                                save_reason = "equilibrium"
                                self.best_equilibrium_score = min_class_score
                                logger.info(f"  ✓ New equilibrium checkpoint! Min class score: {min_class_score:.4f}")
                    
                    # Strategy 2: Global aggregate (fallback or if equilibrium not reached)
                    eval_score = precision + coverage  # Combined score
                    if not save_model and eval_score > self.best_eval_score:
                        save_model = True
                        save_reason = "aggregate"
                        self.best_eval_score = eval_score
                    
                    # Strategy 3: Per-class best (optional - save if any class improves)
                    if class_scores:
                        for class_id, (p, c, s) in class_scores.items():
                            if class_id not in self.best_eval_score_per_class or s > self.best_eval_score_per_class[class_id]:
                                self.best_eval_score_per_class[class_id] = s
                    
                    if save_model:
                        try:
                            # Save best model checkpoint
                            best_model_dir = os.path.join(str(self.experiment.folder_name), "best_model")
                            os.makedirs(best_model_dir, exist_ok=True)
                            best_model_path = os.path.join(best_model_dir, "best_checkpoint.pt")
                            
                            # Save experiment state
                            self.experiment.save(best_model_path)
                            self.best_model_path = best_model_path
                            
                            if save_reason == "equilibrium":
                                logger.info(
                                    f"  ✓ New best model saved (EQUILIBRIUM)! "
                                    f"All classes meet targets. "
                                    f"Global: P={precision:.4f}, C={coverage:.4f}"
                                )
                                if class_scores:
                                    for class_id, (p, c, s) in sorted(class_scores.items()):
                                        logger.info(f"    Class {class_id}: P={p:.4f}, C={c:.4f}, Score={s:.4f}")
                            else:
                                logger.info(
                                    f"  ✓ New best model saved! (Score: {eval_score:.4f}, "
                                    f"Precision: {precision:.4f}, Coverage: {coverage:.4f})"
                                )
                            logger.info(f"    Best model path: {best_model_path}")
                        except Exception as e:
                            logger.warning(f"  ⚠ Could not save best model: {e}")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        return self.training_history.copy()
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        return self.evaluation_history.copy()
    
    def get_training_episode_details(self) -> List[Dict[str, Any]]:
        """Get episode details collected during training (bounds, precision, coverage, etc.)."""
        return self.training_episode_details.copy()
    
    def get_evaluation_anchor_data(self) -> List[Dict[str, Any]]:
        """Get anchor data collected during evaluation for rule extraction."""
        return self.evaluation_anchor_data.copy()
    
    def set_experiment_folder(self, folder_path: str):
        """Set the experiment folder path for periodic saving during training."""
        self.experiment_folder = folder_path
    
    def _flush_remaining_metrics(self):
        """
        Aggregate and save any remaining training metrics that haven't been saved yet.
        Called at the end of training to ensure all data is saved.
        """
        if not self.training_metrics:
            return
        
        # Aggregate remaining metrics
        aggregated = {}
        for key in self.training_metrics[0].keys():
            values = [m[key] for m in self.training_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        if aggregated:
            aggregated["step"] = self.experiment.n_iters_performed if hasattr(self, 'experiment') and hasattr(self.experiment, 'n_iters_performed') else None
            aggregated["total_frames"] = self.experiment.total_frames if hasattr(self, 'experiment') and hasattr(self.experiment, 'total_frames') else None
            self.training_history.append(aggregated.copy())
            self.training_metrics = []
    
    def _save_data_periodically(self):
        """
        Save current training metrics and episode details to files during training.
        This is called periodically when metrics are aggregated.
        """
        if not self.experiment_folder:
            return
        
        import json
        from pathlib import Path
        
        experiment_path = Path(self.experiment_folder)
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        def convert_to_serializable(obj: Any) -> Any:
            """Convert numpy arrays and other non-serializable types to JSON-compatible formats."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (float, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist() if obj.numel() > 0 else []
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        # Save training history (append mode - incremental saves)
        if self.training_history:
            training_history_path = experiment_path / "training_history.json"
            serializable_history = convert_to_serializable(self.training_history)
            with open(training_history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        # Save training episode details (append mode - incremental saves)
        if self.training_episode_details:
            training_episodes_path = experiment_path / "training_episode_details.json"
            serializable_episodes = convert_to_serializable(self.training_episode_details)
            with open(training_episodes_path, 'w') as f:
                json.dump(serializable_episodes, f, indent=2, ensure_ascii=False)
    
    def save_data_to_files(self, experiment_folder: str) -> Dict[str, str]:
        """
        Save all collected callback data to JSON files in the experiment folder.
        
        Args:
            experiment_folder: Path to the experiment folder where data should be saved
            
        Returns:
            Dictionary mapping data type to file path where it was saved
        """
        import json
        import os
        from pathlib import Path
        
        saved_files = {}
        experiment_path = Path(experiment_folder)
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        def convert_to_serializable(obj: Any) -> Any:
            """Convert numpy arrays and other non-serializable types to JSON-compatible formats."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (float, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist() if obj.numel() > 0 else []
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        # Save training history (aggregated metrics)
        if self.training_history:
            training_history_path = experiment_path / "training_history.json"
            serializable_history = convert_to_serializable(self.training_history)
            with open(training_history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            saved_files["training_history"] = str(training_history_path)
            logger.info(f"  ✓ Saved training history ({len(self.training_history)} entries) to: {training_history_path}")
        
        # Save evaluation history (aggregated metrics)
        if self.evaluation_history:
            evaluation_history_path = experiment_path / "evaluation_history.json"
            serializable_history = convert_to_serializable(self.evaluation_history)
            with open(evaluation_history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            saved_files["evaluation_history"] = str(evaluation_history_path)
            logger.info(f"  ✓ Saved evaluation history ({len(self.evaluation_history)} entries) to: {evaluation_history_path}")
        
        # Save training episode details (detailed per-episode data with bounds)
        if self.training_episode_details:
            training_episodes_path = experiment_path / "training_episode_details.json"
            serializable_episodes = convert_to_serializable(self.training_episode_details)
            with open(training_episodes_path, 'w') as f:
                json.dump(serializable_episodes, f, indent=2, ensure_ascii=False)
            saved_files["training_episode_details"] = str(training_episodes_path)
            logger.info(f"  ✓ Saved training episode details ({len(self.training_episode_details)} episodes) to: {training_episodes_path}")
        
        # Save evaluation anchor data (for rule extraction)
        if self.evaluation_anchor_data:
            evaluation_anchor_path = experiment_path / "evaluation_anchor_data.json"
            serializable_anchor_data = convert_to_serializable(self.evaluation_anchor_data)
            with open(evaluation_anchor_path, 'w') as f:
                json.dump(serializable_anchor_data, f, indent=2, ensure_ascii=False)
            saved_files["evaluation_anchor_data"] = str(evaluation_anchor_path)
            logger.info(f"  ✓ Saved evaluation anchor data ({len(self.evaluation_anchor_data)} episodes) to: {evaluation_anchor_path}")
        
        # Log diagnostic information if no data was saved
        if not saved_files:
            logger.info("  No callback data to save. Diagnostic info:")
            logger.info(f"    - training_history: {len(self.training_history)} entries")
            logger.info(f"    - evaluation_history: {len(self.evaluation_history)} entries")
            logger.info(f"    - training_episode_details: {len(self.training_episode_details)} episodes")
            logger.info(f"    - evaluation_anchor_data: {len(self.evaluation_anchor_data)} episodes")
            logger.info(f"    - log_training_metrics: {self.log_training_metrics}")
            logger.info(f"    - log_evaluation_metrics: {self.log_evaluation_metrics}")
            logger.info(f"    - collect_anchor_data: {self.collect_anchor_data}")
        
        return saved_files

