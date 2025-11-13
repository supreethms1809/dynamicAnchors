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
        max_cycles = config.get("max_cycles", 100)

        env_config = {k: v for k, v in config.items() if k != "max_cycles"}
        
        if "env_config" not in env_config:
            env_config["env_config"] = {}
        if not isinstance(env_config["env_config"], dict):
            env_config["env_config"] = {}
        env_config["env_config"]["max_cycles"] = max_cycles
        
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
        return self.config.get("max_cycles", 100)
    
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
    
    def __init__(self, log_training_metrics: bool = True, log_evaluation_metrics: bool = True, save_to_file: bool = True, collect_anchor_data: bool = False):
        super().__init__()
        self.log_training_metrics = log_training_metrics
        self.log_evaluation_metrics = log_evaluation_metrics
        self.save_to_file = save_to_file
        self.collect_anchor_data = collect_anchor_data
        self.training_metrics = []
        self.training_history = []
        self.evaluation_history = []
        # Store anchor data collected during evaluation
        self.evaluation_anchor_data = []  # List of episodes, each episode contains agent data
        # Store training episode details (bounds, precision, coverage per episode)
        self.training_episode_details = []  # List of episode details from training batches
    
    def _extract_metrics_from_info(self, info: TensorDictBase, prefix: str = "") -> Dict[str, float]:
        metrics = {}
        
        if info is None:
            return metrics
        
        metric_keys = [
            "precision", "coverage", "drift", "anchor_drift", "js_penalty",
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
                    
                    episode_detail["precision"] = safe_get("precision", 0.0)
                    episode_detail["coverage"] = safe_get("coverage", 0.0)
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
            
            if len(self.training_metrics) >= 10:
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
                            
                            episode_data[group] = {
                                "precision": safe_get("precision", 0.0),
                                "coverage": safe_get("coverage", 0.0),
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
            
            precision_key = [k for k in aggregated.keys() if "precision" in k and "mean" in k and "evaluation" in k]
            coverage_key = [k for k in aggregated.keys() if "coverage" in k and "mean" in k and "evaluation" in k]
            if precision_key and coverage_key:
                logger.info(
                    f"Evaluation - Precision: {aggregated[precision_key[0]]:.4f}, "
                    f"Coverage: {aggregated[coverage_key[0]]:.4f} "
                    f"(n={n_episodes})"
                )
    
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

