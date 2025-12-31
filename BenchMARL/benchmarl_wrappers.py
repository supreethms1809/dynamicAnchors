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
from tensordict import TensorDict, TensorDictBase

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
        
        # Read max_cycles from config - no hardcoded default to ensure YAML settings are respected
        max_cycles = config.get("max_cycles")
        if max_cycles is None:
            raise ValueError("max_cycles must be specified in config. Check your YAML config file.")
        max_cycles = int(max_cycles)

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
    
    def __init__(self, log_training_metrics: bool = True, log_evaluation_metrics: bool = True, save_to_file: bool = True, collect_anchor_data: bool = False, save_frequency: int = 10, save_during_training: bool = True, save_best_model: bool = True, compute_nashconv: bool = True, nashconv_batch_size: int = 32, nashconv_lr: float = 0.01, nashconv_steps: int = 10, nashconv_compute_frequency: int = 10, nashconv_threshold: float = 0.01):
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
        # Track training batches for NashConv computation (compute periodically, not every batch)
        self.training_batches_for_nashconv = []  # Store recent batches for NashConv computation
        # NashConv computation frequency recommendations:
        # - Short training (< 20 iterations): 2-3 batches (more frequent for visibility)
        # - Medium training (20-100 iterations): 5-10 batches (balanced)
        # - Long training (> 100 iterations): 10-20 batches (less frequent, still sufficient data points)
        # Note: NashConv computation is expensive, so higher frequency = more compute time
        self.nashconv_compute_frequency = nashconv_compute_frequency  # Compute NashConv every N batches (configurable via __init__)
        self.training_batch_count = 0  # Counter for tracking batches
        # Track best model
        self.best_eval_score = -float('inf')  # Track best evaluation score (precision + coverage)
        self.best_model_path = None  # Path to best model checkpoint
        
        # Track best models per class (for multi-agent equilibrium evaluation)
        self.best_eval_score_per_class = {}  # class -> best score
        self.best_model_path_per_class = {}  # class -> best model path
        
        # Equilibrium evaluation: track if all classes meet targets
        self.equilibrium_eval_mode = True  # If True, save best model when all classes meet targets
        self.best_equilibrium_score = -float('inf')  # Track best equilibrium score (min class score when all meet targets)
        self.best_equilibrium_nashconv = float('inf')  # Track best NashConv for equilibrium models (lower is better)
        
        # NashConv/exploitability computation settings
        self.compute_nashconv = compute_nashconv  # Whether to compute NashConv during evaluation
        self.nashconv_batch_size = nashconv_batch_size  # Batch size for best response computation
        self.nashconv_lr = nashconv_lr  # Learning rate for gradient ascent
        self.nashconv_steps = nashconv_steps  # Number of gradient ascent steps
        self.nashconv_threshold = nashconv_threshold  # ε-Nash threshold for model selection (default: 0.01)
        self.best_eval_nashconv = float('inf')  # Track best NashConv for aggregate models (lower is better)
        
        # Experiment reference (will be set by BenchMARL)
        self.experiment = None
    
    def on_experiment_start(self, experiment):
        """Called when experiment starts. Sets the experiment reference."""
        self.experiment = experiment
        print(f"\n{'='*80}")
        print(f"CALLBACK: on_experiment_start called!")
        print(f"  Experiment: {experiment}")
        print(f"  Experiment folder: {getattr(experiment, 'folder_name', 'N/A')}")
        print(f"{'='*80}\n")
        logger.info("Callback: Experiment started, experiment reference set")
    
    def _extract_class_from_group_name(self, group_name: str) -> Optional[int]:
        """Extract class id from a GROUP name.

        Supported group naming conventions:
          - class_0, class_1, ...           (recommended: GROUP == class/player)
          - agent_0, agent_1, ...           (legacy: GROUP == class/player)
          - agent_0_0, agent_0_1, ...       (legacy: per-agent groups; first index is class)

        Returns None if format is unrecognized.
        """
        try:
            if not isinstance(group_name, str):
                return None

            # Preferred: group == class/player
            if group_name.startswith("class_"):
                parts = group_name.replace("class_", "").split("_")
                return int(parts[0]) if parts and parts[0] != "" else None

            # Backward compatible: group starts with agent_
            if group_name.startswith("agent_"):
                parts = group_name.replace("agent_", "").split("_")
                return int(parts[0]) if parts and parts[0] != "" else None

            return None
        except (ValueError, TypeError):
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
    
    def _compute_nashconv_metrics(self, rollouts: List[TensorDictBase]) -> Dict[str, float]:
        """Compute a critic-based *one-step deviation* exploitability proxy at the CLASS level.

        IMPORTANT:
        - This treats each CLASS as a player, not individual agents.
        - For classes with multiple agents, we compute joint best response for all agents in that class.
        - This is commonly called "NashConv" in codebases, but it is *not* a full best-response-over-policies
          computation. It measures whether a CLASS can improve its Q-value at sampled states by deviating
          the joint action of all agents in that class while holding other classes fixed.

        Proxy definition (per-class):
          Δ_c(s) = max_{a_c} Q_c(s, a_c, a_{-c}) - Q_c(s, a^π)
          where a_c is the joint action of all agents in class c, and a_{-c} are actions of all other classes.

        Returns:
          - evaluation/nashconv_sum: sum_c E[Δ_c] (sum over classes)
          - evaluation/exploitability_max: max_c E[Δ_c]
          - evaluation/exploitability_class_{c}: per-class E[Δ_c]
          - evaluation/class_nashconv_sum: same as nashconv_sum (for backward compatibility)
          - evaluation/class_exploitability_max: same as exploitability_max (for backward compatibility)
        """
        if not self.compute_nashconv or self.experiment is None:
            return {}

        algorithm = getattr(self.experiment, "algorithm", None)
        if algorithm is None:
            return {}

        group_map = getattr(self.experiment, "group_map", None)
        groups = list(group_map.keys()) if isinstance(group_map, dict) else []
        if not groups:
            return {}
        group_order = list(groups)  # consistent ordering everywhere

        from tensordict import TensorDict

        def _to_tensor(x: Any) -> Optional[torch.Tensor]:
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.float()
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            try:
                return torch.as_tensor(x).float()
            except Exception:
                return None

        # Collect per-group (obs, action) pairs from rollouts
        obs_buf: Dict[str, List[torch.Tensor]] = {g: [] for g in group_order}
        act_buf: Dict[str, List[torch.Tensor]] = {g: [] for g in group_order}

        for rollout in rollouts or []:
            if rollout is None:
                continue

            # next TD
            try:
                next_td = rollout.get("next", None) if hasattr(rollout, "get") else (rollout["next"] if "next" in rollout.keys() else None)
            except Exception:
                next_td = None

            for g in group_order:
                try:
                    # observation from next state
                    gd_next = None
                    if next_td is not None:
                        gd_next = next_td.get(g, None) if hasattr(next_td, "get") else (next_td[g] if hasattr(next_td, "keys") and g in next_td.keys() else None)
                    obs = None
                    if gd_next is not None:
                        obs = gd_next.get("observation", None) if hasattr(gd_next, "get") else (gd_next["observation"] if hasattr(gd_next, "keys") and "observation" in gd_next.keys() else None)

                    # action from current state
                    gd_cur = rollout.get(g, None) if hasattr(rollout, "get") else (rollout[g] if hasattr(rollout, "keys") and g in rollout.keys() else None)
                    act = None
                    if gd_cur is not None:
                        act = gd_cur.get("action", None) if hasattr(gd_cur, "get") else (gd_cur["action"] if hasattr(gd_cur, "keys") and "action" in gd_cur.keys() else None)

                    # Fallback: some rollouts store all agents under a shared key (often "agent")
                    # In that case, we read from that container and slice by the agent index.
                    if (obs is None or act is None) and next_td is not None:
                        try:
                            shared_key = None
                            if hasattr(next_td, "keys") and "agent" in next_td.keys():
                                shared_key = "agent"
                            elif hasattr(next_td, "keys") and "agents" in next_td.keys():
                                shared_key = "agents"

                            if shared_key is not None:
                                shared_next = next_td.get(shared_key, None) if hasattr(next_td, "get") else next_td[shared_key]
                                if obs is None and shared_next is not None:
                                    obs = shared_next.get("observation", None) if hasattr(shared_next, "get") else (shared_next["observation"] if hasattr(shared_next, "keys") and "observation" in shared_next.keys() else None)

                                shared_cur = None
                                if hasattr(rollout, "keys") and shared_key in rollout.keys():
                                    shared_cur = rollout.get(shared_key, None) if hasattr(rollout, "get") else rollout[shared_key]
                                if act is None and shared_cur is not None:
                                    act = shared_cur.get("action", None) if hasattr(shared_cur, "get") else (shared_cur["action"] if hasattr(shared_cur, "keys") and "action" in shared_cur.keys() else None)
                        except Exception:
                            pass

                    obs_t = _to_tensor(obs)
                    act_t = _to_tensor(act)
                    if obs_t is None or act_t is None:
                        continue

                    # NOTE: Do NOT slice away the agent dimension here.
                    # For grouped policies (e.g., `agent_0`, `agent_1`), observations/actions are often
                    # shaped as [T, agents_per_class, dim]. The critic expects the per-group agent dimension
                    # to be preserved (e.g., shape[-2] == agents_per_class).

                    if obs_t.dim() == 1:
                        obs_t = obs_t.unsqueeze(0)
                    if act_t.dim() == 1:
                        act_t = act_t.unsqueeze(0)

                    T = min(obs_t.shape[0], act_t.shape[0])
                    if T <= 0:
                        continue

                    for t in range(T):
                        obs_buf[g].append(obs_t[t])
                        act_buf[g].append(act_t[t])

                except Exception:
                    continue

        min_len = min((len(obs_buf[g]) for g in group_order), default=0)
        if min_len <= 0:
            logger.warning(
                f"NashConv: no (obs, action) pairs collected. groups={group_order}. "
                f"counts={{g: (len(obs_buf[g]), len(act_buf[g])) for g in group_order}}"
            )
            return {}

        batch_size = min(self.nashconv_batch_size, min_len)
        indices = np.random.choice(min_len, size=batch_size, replace=False)

        # Choose device (prefer algorithm params; fall back to cpu)
        try:
            device = next(algorithm.parameters()).device  # type: ignore[attr-defined]
        except Exception:
            device = torch.device("cpu")

        # Infer agents_per_class for reshaping flattened tensors (when each group == class/player)
        def _infer_agents_per_class() -> int:
            try:
                cfg = getattr(getattr(self.experiment, "task", None), "config", None)
                if isinstance(cfg, dict):
                    env_cfg = cfg.get("env_config", None)
                    if isinstance(env_cfg, dict) and env_cfg.get("agents_per_class", None) is not None:
                        return int(env_cfg["agents_per_class"])
            except Exception:
                pass
            try:
                env = getattr(self.experiment, "env", None)
                while env is not None and (hasattr(env, "env") or hasattr(env, "_env")):
                    env = getattr(env, "env", None) or getattr(env, "_env", None)
                apc = getattr(env, "agents_per_class", None)
                if apc is not None:
                    return int(apc)
            except Exception:
                pass
            return 1

        agents_per_class = _infer_agents_per_class()

        obs_batch: Dict[str, torch.Tensor] = {}
        act_batch: Dict[str, torch.Tensor] = {}
        for g in group_order:
            obs_batch[g] = torch.stack([obs_buf[g][i] for i in indices]).to(device)
            act_batch[g] = torch.stack([act_buf[g][i] for i in indices]).to(device)

        # If wrappers flattened internal agent dimension, reshape back to [B, A, D]
        if agents_per_class and agents_per_class > 1:
            for g in group_order:
                ob = obs_batch[g]
                ac = act_batch[g]
                if ob.dim() == 2 and ob.shape[1] % agents_per_class == 0:
                    obs_batch[g] = ob.view(batch_size, agents_per_class, ob.shape[1] // agents_per_class)
                if ac.dim() == 2 and ac.shape[1] % agents_per_class == 0:
                    act_batch[g] = ac.view(batch_size, agents_per_class, ac.shape[1] // agents_per_class)

        def _list_value_module_keys() -> List[str]:
            """Best-effort introspection of available value/critic module keys on the algorithm."""
            candidates = []
            for attr in ("value_modules", "value_module", "critics", "critic_modules", "_value_modules", "_critics"):
                if hasattr(algorithm, attr):
                    obj = getattr(algorithm, attr)
                    if isinstance(obj, dict):
                        candidates.extend([str(k) for k in obj.keys()])
            # Deduplicate while preserving order
            seen = set()
            out = []
            for k in candidates:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

        def _get_value_module_for(group_name: str):
            """Try multiple naming conventions to find the critic/value module."""
            tried = []

            def _try(name: str):
                if not name:
                    return None
                tried.append(name)
                if hasattr(algorithm, "get_value_module"):
                    try:
                        return algorithm.get_value_module(name)
                    except Exception:
                        return None
                return None

            # 1) direct
            mod = _try(group_name)
            if mod is not None:
                return mod, tried

            # 2) common shared group keys
            for alt in ("agent", "agents", "group", "default"):
                mod = _try(alt)
                if mod is not None:
                    return mod, tried

            # 3) normalize group names like agent_0 -> agent
            if group_name.startswith("agent_"):
                mod = _try("agent")
                if mod is not None:
                    return mod, tried

            # 4) try class-only prefix (agent_0_1 -> agent_0)
            if group_name.startswith("agent_") and "_" in group_name.replace("agent_", ""):
                parts = group_name.split("_")
                if len(parts) >= 2:
                    base = "_".join(parts[:2])  # agent_0
                    mod = _try(base)
                    if mod is not None:
                        return mod, tried

            # 5) dict-based fallbacks if present
            for attr in ("value_modules", "critics", "critic_modules", "_value_modules", "_critics"):
                if hasattr(algorithm, attr):
                    obj = getattr(algorithm, attr)
                    if isinstance(obj, dict):
                        # direct lookup
                        if group_name in obj:
                            return obj[group_name], tried
                        # shared key lookups
                        for alt in ("agent", "agents", "default"):
                            if alt in obj:
                                return obj[alt], tried

            return None, tried

        def _critic_q(critic_module, agent_group: str, act_by_group: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Return Q-values for a given *target group* while providing the critic the full joint (s, a).

            Why: Most BenchMARL critics are centralized (CTDE) and expect observations/actions for *all* groups
            to be present in the input TensorDict. Passing only the current group's (obs, action) often yields
            missing outputs (None), which then caused the old flatten(None) crash.
            """
            # Ensure the input TensorDict is built on the SAME device as the critic.
            # This avoids cpu<->mps (or cpu<->cuda) mismatches during evaluation.
            try:
                td_device = next(critic_module.parameters()).device
            except Exception:
                td_device = obs_batch[agent_group].device
            td = TensorDict({}, batch_size=[batch_size], device=td_device)

            # Always provide the full joint observation/action keyed by group.
            for g in group_order:
                if g not in obs_batch:
                    raise KeyError(f"NashConv: obs_batch missing group '{g}'. Available={list(obs_batch.keys())}")
                if g not in act_by_group:
                    raise KeyError(f"NashConv: act_by_group missing group '{g}'. Available={list(act_by_group.keys())}")

                obs_g = obs_batch[g].to(td_device)
                act_g = act_by_group[g].to(td_device)

                # TorchRL-style tuple keys
                td[(g, "observation")] = obs_g
                td[(g, "action")] = act_g

                # Nested group TensorDict
                td[g] = TensorDict({"observation": obs_g, "action": act_g}, batch_size=[batch_size], device=td_device)

            # Also provide flat keys for compatibility (set to the *target* group)
            td["observation"] = obs_batch[agent_group].to(td_device)
            td["action"] = act_by_group[agent_group].to(td_device)

            out = critic_module(td)

            # -----------------------------
            # Robust Q extraction
            # -----------------------------
            q = None

            # Candidates we will try in priority order
            candidate_keys = [
                (agent_group, "state_action_value"),
                (agent_group, "chosen_action_value"),
                (agent_group, "action_value"),
                (agent_group, "q"),
                (agent_group, "Q"),
                "state_action_value",
                "chosen_action_value",
                "action_value",
                "q",
                "Q",
            ]

            # Try direct extraction from dict or TensorDict
            try:
                if isinstance(out, TensorDictBase):
                    out_keys = list(out.keys(True))
                    for k in candidate_keys:
                        if k in out.keys(True):
                            q = out.get(k)
                            if q is not None:
                                break

                    # Fallback: if critic returns per-group nested TDs
                    if q is None and agent_group in out.keys(True):
                        out_g = out.get(agent_group)
                        if isinstance(out_g, TensorDictBase):
                            for k2 in ("state_action_value", "chosen_action_value", "action_value", "q", "Q"):
                                if k2 in out_g.keys(True):
                                    q = out_g.get(k2)
                                    if q is not None:
                                        break
                elif isinstance(out, dict):
                    out_keys = list(out.keys())
                    for k in candidate_keys:
                        if k in out:
                            q = out[k]
                            if q is not None:
                                break
                else:
                    out_keys = [type(out).__name__]
                    q = out
            except Exception:
                out_keys = ["<unavailable>"]
                q = None

            if q is None:
                raise TypeError(
                    f"NashConv: critic output did not contain an action-dependent Q tensor for group='{agent_group}'. "
                    f"Tried keys={candidate_keys}. Available out_keys={out_keys}"
                )

            if not torch.is_tensor(q):
                raise TypeError(f"NashConv: extracted Q is not a Tensor for group='{agent_group}'. type={type(q)}")

            # Normalize Q to shape [B] by averaging over any remaining dims.
            # Common outputs: [B], [B,1], [B,agents_per_class], [B,agents_per_class,1], etc.
            if q.dim() == 2 and q.shape[-1] == 1:
                q = q.squeeze(-1)
            if q.dim() >= 2:
                q = q.reshape(q.shape[0], -1).mean(dim=1)

            return q

        # -----------------------------
        # CLASS-AS-PLAYER MODEL
        # -----------------------------
        # We want each PLAYER to be one CLASS. In BenchMARL terms, this means:
        #   - `group_map` should expose ONE group per class/player (recommended name: "class_{c}")
        #   - each group tensor can still carry an *internal* agent dimension, e.g. [B, agents_per_class, dim]
        #
        # For backward compatibility, if the user still exposes multiple groups per class (e.g. agent_0_0, agent_0_1),
        # we will aggregate them under the same class id.

        class_to_groups: Dict[int, List[str]] = {}
        for g in group_order:
            cls = self._extract_class_from_group_name(g)
            if cls is None:
                continue
            class_to_groups.setdefault(cls, []).append(g)

        if not class_to_groups:
            logger.warning(
                f"NashConv: Could not extract class ids from group names. groups={group_order}"
            )
            return {}

        classes = sorted(class_to_groups.keys())
        logger.debug(f"NashConv: Computing CLASS-level exploitability for classes: {classes}")
        logger.debug(f"NashConv: Class-to-groups mapping: {class_to_groups}")

        def _class_q_value(class_id: int, act_by_group: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
            groups_in_class = class_to_groups.get(class_id, [])
            if not groups_in_class:
                logger.debug(f"NashConv: Class {class_id} has no groups")
                return None

            q_values = []
            failed = []

            for group_name in groups_in_class:
                critic_module, tried_names = _get_value_module_for(group_name)
                if critic_module is None:
                    failed.append((group_name, "no_critic"))
                    continue

                try:
                    q = _critic_q(critic_module, group_name, act_by_group)
                    if q is not None:
                        q_values.append(q)
                    else:
                        failed.append((group_name, "q_none"))
                except Exception as e:
                    failed.append((group_name, f"{type(e).__name__}: {e}"))

            if not q_values:
                logger.warning(
                    f"NashConv: Class {class_id} Q-value computation failed for all groups. "
                    f"Groups: {groups_in_class}, Failed: {failed}"
                )
                return None

            # If there is exactly one group per class (recommended), this is just that group's Q.
            # If there are multiple groups (legacy), we average them.
            q_stack = torch.stack(q_values, dim=0)  # [n_groups, B]
            return q_stack.mean(dim=0)

        def _approx_class_br_random_search(class_id: int, act_by_group: Dict[str, torch.Tensor]) -> float:
            """Approximate class-level best-response via black-box random search over *joint actions* of the class.

            Here a "class" corresponds to the group(s) in `class_to_groups[class_id]`. In the recommended setup,
            there is exactly one group per class (e.g., 'class_0' or 'agent_0'), and its action tensor can still
            contain an internal agent dimension: [B, agents_per_class, act_dim].
            """
            groups_in_class = class_to_groups.get(class_id, [])
            if not groups_in_class:
                return 0.0

            # Hyperparams
            num_samples = int(getattr(self, "nashconv_num_samples", 256))
            noise_std = float(getattr(self, "nashconv_noise_std", 0.25))
            eval_chunk = int(getattr(self, "nashconv_eval_chunk", 64))

            # Base actions for this class (by group)
            base_acts = {g: act_by_group[g] for g in groups_in_class if g in act_by_group}
            if not base_acts:
                return 0.0

            B = next(iter(base_acts.values())).shape[0]
            best_q = None
            candidates_total = num_samples + 1  # include the original action as candidate 0

            for start in range(0, candidates_total, eval_chunk):
                end = min(candidates_total, start + eval_chunk)
                k = end - start

                # Sample candidates for each group in the class
                cand_by_group = {}
                for g, base_act in base_acts.items():
                    if base_act.dim() == 3:
                        # [B, A, act_dim]
                        n_agents = base_act.shape[1]
                        act_dim = base_act.shape[2]
                        noise = torch.randn((k, B, n_agents, act_dim), device=base_act.device, dtype=base_act.dtype) * noise_std
                        cand = base_act.unsqueeze(0) + noise  # [k, B, A, act_dim]
                        if start == 0:
                            cand[0] = base_act
                        cand_by_group[g] = cand.clamp(-1.0, 1.0)
                    elif base_act.dim() == 2:
                        # [B, act_dim]
                        act_dim = base_act.shape[1]
                        noise = torch.randn((k, B, act_dim), device=base_act.device, dtype=base_act.dtype) * noise_std
                        cand = base_act.unsqueeze(0) + noise  # [k, B, act_dim]
                        if start == 0:
                            cand[0] = base_act
                        cand_by_group[g] = cand.clamp(-1.0, 1.0)
                    else:
                        # Unexpected shape
                        continue

                if not cand_by_group:
                    continue

                # Evaluate candidates
                q_vals = []
                for j in range(k):
                    act_tmp = dict(act_by_group)
                    for g in groups_in_class:
                        if g in cand_by_group:
                            act_tmp[g] = cand_by_group[g][j]
                    q_class = _class_q_value(class_id, act_tmp)
                    if q_class is not None:
                        q_vals.append(q_class)

                if q_vals:
                    q_stack = torch.stack(q_vals, dim=0)  # [k, B]
                    q_chunk_best = q_stack.max(dim=0).values  # [B]
                    best_q = q_chunk_best if best_q is None else torch.maximum(best_q, q_chunk_best)

            return best_q.mean().item() if best_q is not None else 0.0

        # -----------------------------
        # Compute class-level exploitabilities
        # -----------------------------
        class_exploitabilities: Dict[int, float] = {}
        available_value_keys = _list_value_module_keys()
        
        # Get critic modules for all groups (need at least one per class)
        critic_modules_by_group: Dict[str, Any] = {}
        for g in group_order:
            critic_module, tried_names = _get_value_module_for(g)
            if critic_module is not None:
                critic_modules_by_group[g] = critic_module

        if not critic_modules_by_group:
            logger.warning(
                f"NashConv: No critic modules found. Available_keys={available_value_keys}"
            )
            return {}

        # Freeze all critic parameters
        saved_rg_dicts: Dict[str, List[bool]] = {}
        for g, critic_module in critic_modules_by_group.items():
            saved_rg = []
            try:
                for p in critic_module.parameters():
                    saved_rg.append(p.requires_grad)
                    p.requires_grad_(False)
            except Exception:
                saved_rg = []
            saved_rg_dicts[g] = saved_rg

        try:
            for critic_module in critic_modules_by_group.values():
                critic_module.eval()

            for class_id in classes:
                groups_in_class = class_to_groups[class_id]

                # Check if we have at least one critic for this class
                has_critic = any(g in critic_modules_by_group for g in groups_in_class)
                if not has_critic:
                    logger.warning(
                        f"NashConv: No critic module found for class {class_id} (groups: {groups_in_class}). "
                        f"Available critics: {list(critic_modules_by_group.keys())}"
                    )
                    continue

                # Check if we have actions for all groups in this class
                missing_actions = [g for g in groups_in_class if g not in act_batch]
                if missing_actions:
                    logger.warning(
                        f"NashConv: Class {class_id} missing actions for groups: {missing_actions}. "
                        f"Available actions: {list(act_batch.keys())}"
                    )
                    continue

                logger.debug(f"NashConv: Computing exploitability for class {class_id} with groups: {groups_in_class}")

                try:
                    # Current class Q-value (aggregate over all groups in class)
                    with torch.no_grad():
                        q_cur = _class_q_value(class_id, act_batch)
                        if q_cur is None:
                            logger.warning(
                                f"NashConv: Class {class_id} current Q-value is None, skipping exploitability computation. "
                                f"Groups in class: {groups_in_class}, Actions available: {list(act_batch.keys())}"
                            )
                            continue
                        q_cur_mean = q_cur.mean().item()
                        logger.debug(f"NashConv: Class {class_id} current Q-value mean: {q_cur_mean:.6f}")

                    # Probe if Q is differentiable w.r.t. actions of groups in this class
                    q_has_action_grad = False
                    with torch.inference_mode(False):
                        act_probe = dict(act_batch)
                        probes = {}
                        for g in groups_in_class:
                            if g in act_batch:
                                a_probe = act_batch[g].clone().detach().requires_grad_(True)
                                act_probe[g] = a_probe
                                probes[g] = a_probe

                        if probes:
                            with torch.enable_grad():
                                q_probe = _class_q_value(class_id, act_probe)
                                if q_probe is not None:
                                    q_has_action_grad = bool(getattr(q_probe, "requires_grad", False))

                    # Optimize joint actions of all groups in class
                    if q_has_action_grad and probes:
                        with torch.inference_mode(False):
                            # Create optimizable actions for all groups in class
                            optimizable_acts = {}
                            opt_params = []
                            for g in groups_in_class:
                                if g in act_batch:
                                    a_br = act_batch[g].clone().detach().requires_grad_(True)
                                    optimizable_acts[g] = a_br
                                    opt_params.append(a_br)

                            if not opt_params:
                                continue

                            opt = torch.optim.Adam(opt_params, lr=self.nashconv_lr)
                            act_tmp = dict(act_batch)

                            for _ in range(self.nashconv_steps):
                                opt.zero_grad()
                                for g, a_br in optimizable_acts.items():
                                    act_tmp[g] = a_br

                                with torch.enable_grad():
                                    q_br = _class_q_value(class_id, act_tmp)
                                    if q_br is None:
                                        break
                                    loss = -q_br.mean()
                                    loss.backward()

                                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                                opt.step()
                                with torch.no_grad():
                                    for a_br in optimizable_acts.values():
                                        a_br.clamp_(-1.0, 1.0)

                            with torch.no_grad():
                                for g, a_br in optimizable_acts.items():
                                    act_tmp[g] = a_br
                                q_br_final = _class_q_value(class_id, act_tmp)
                                q_br_mean = q_br_final.mean().item() if q_br_final is not None else q_cur_mean
                    else:
                        # Fall back to random search for class-level best response
                        logger.debug(
                            f"NashConv: Class {class_id} Q not differentiable w.r.t. actions. "
                            f"Using random-search BR (num_samples={int(getattr(self, 'nashconv_num_samples', 256))})."
                        )
                        q_br_mean = _approx_class_br_random_search(class_id, act_batch)

                    delta = max(0.0, q_br_mean - q_cur_mean)
                    class_exploitabilities[class_id] = float(delta)

                except Exception as e:
                    logger.warning(
                        f"NashConv: Class {class_id} exploitability computation failed. "
                        f"Error={type(e).__name__}: {e}"
                    )
                    continue

        finally:
            # Restore critic requires_grad flags
            for g, saved_rg in saved_rg_dicts.items():
                if g in critic_modules_by_group:
                    try:
                        critic_module = critic_modules_by_group[g]
                        for p, rg in zip(critic_module.parameters(), saved_rg):
                            p.requires_grad_(rg)
                    except Exception:
                        pass
        
        if not class_exploitabilities:
            logger.warning(
                f"NashConv: No class exploitabilities computed. Classes tried: {classes}"
            )
            return {}
        
        # Compute metrics
        nashconv_sum = float(sum(class_exploitabilities.values()))
        exploitability_max = float(max(class_exploitabilities.values()))
        
        metrics: Dict[str, float] = {
            "evaluation/nashconv_sum": nashconv_sum,
            "evaluation/exploitability_max": exploitability_max,
            "evaluation/class_nashconv_sum": nashconv_sum,  # Alias for clarity
            "evaluation/class_exploitability_max": exploitability_max,  # Alias for clarity
        }
        
        # Per-class exploitability
        for class_id, v in class_exploitabilities.items():
            metrics[f"evaluation/exploitability_class_{class_id}"] = float(v)
            metrics[f"evaluation/class_exploitability_class_{class_id}"] = float(v)  # Alias for backward compatibility
        
        return metrics
    
    def on_batch_collected(self, batch: TensorDictBase):
        if not self.log_training_metrics:
            return
        
        # Debug: Log that on_batch_collected is being called (use print for visibility)
        self.training_batch_count = getattr(self, 'training_batch_count', 0) + 1
        # Log every batch for debugging (can be reduced later)
        print(f"[TRAINING] on_batch_collected: batch_count={self.training_batch_count}, compute_nashconv={self.compute_nashconv}, frequency={self.nashconv_compute_frequency}")
        
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
        
        # Compute NashConv periodically during training (epsilon-Nash equilibrium convergence)
        # Do this BEFORE checking all_metrics so we can compute even if other metrics are empty
        # Note: training_batch_count was already incremented above
        training_nashconv_metrics = {}
        
        if self.compute_nashconv and self.training_batch_count % self.nashconv_compute_frequency == 0:
            print(f"[TRAINING NASHCONV] ⚡ Computing at batch {self.training_batch_count} (iter {self.experiment.n_iters_performed if hasattr(self, 'experiment') and hasattr(self.experiment, 'n_iters_performed') else 'N/A'})")
            try:
                # Store batch for NashConv computation
                # Training batches have structure: {group: {...}, "next": {group: {...}}}
                # This is compatible with _compute_nashconv_metrics which expects rollouts
                from tensordict import TensorDict
                
                # Clone batch to avoid modifying the original
                rollout_like = batch.clone()
                self.training_batches_for_nashconv.append(rollout_like)
                
                # Keep only recent batches (last 2-3) to avoid memory issues
                if len(self.training_batches_for_nashconv) > 3:
                    self.training_batches_for_nashconv.pop(0)
                
                # Compute NashConv from recent training batches
                if len(self.training_batches_for_nashconv) >= 1:
                    logger.debug(f"Computing NashConv from {len(self.training_batches_for_nashconv)} training batch(es)")
                    nashconv_metrics = self._compute_nashconv_metrics(self.training_batches_for_nashconv)
                    if nashconv_metrics:
                        logger.debug(f"NashConv computation succeeded, got {len(nashconv_metrics)} metrics")
                        # Convert evaluation/ prefix to training/ prefix for training metrics
                        for key, value in nashconv_metrics.items():
                            if key.startswith("evaluation/"):
                                training_key = key.replace("evaluation/", "training/")
                                training_nashconv_metrics[training_key] = value
                            else:
                                training_nashconv_metrics[f"training/{key}"] = value
                        
                        # Log NashConv metrics for monitoring epsilon-Nash equilibrium convergence
                        nashconv_sum = training_nashconv_metrics.get('training/nashconv_sum', 0.0)
                        exploitability_max = training_nashconv_metrics.get('training/exploitability_max', 0.0)
                        class_nashconv_sum = training_nashconv_metrics.get('training/class_nashconv_sum', None)
                        
                        # Print to console for visibility
                        print(f"[TRAINING NASHCONV] ✓ Success! batch={self.training_batch_count}, iter={self.experiment.n_iters_performed if hasattr(self, 'experiment') and hasattr(self.experiment, 'n_iters_performed') else 'N/A'}")
                        print(f"  NashConv sum: {nashconv_sum:.6f}, max exploitability: {exploitability_max:.6f}" + (f", class_sum: {class_nashconv_sum:.6f}" if class_nashconv_sum is not None else ""))
                        
                        logger.info(
                            f"Training NashConv (iter {self.experiment.n_iters_performed}, batch {self.training_batch_count}): "
                            f"sum={nashconv_sum:.6f}, max={exploitability_max:.6f}"
                            + (f", class_sum={class_nashconv_sum:.6f}" if class_nashconv_sum is not None else "")
                        )
                        
                        # Also log per-agent exploitability if available
                        for key, value in training_nashconv_metrics.items():
                            if key.startswith("training/exploitability_") and not key.endswith("_max") and not key.startswith("training/class_"):
                                logger.debug(f"  {key}: {value:.6f}")
                        
                        # Log class-level exploitability if available
                        for key, value in training_nashconv_metrics.items():
                            if key.startswith("training/class_exploitability_class_"):
                                class_id = key.replace("training/class_exploitability_class_", "")
                                logger.debug(f"  Class {class_id} exploitability: {value:.6f}")
                    else:
                        print(f"[TRAINING NASHCONV] ⚠ No metrics returned (insufficient data or computation failed)")
                        logger.warning(f"NashConv computation returned empty metrics (this may indicate insufficient data or computation failure)")
            except Exception as e:
                # NashConv computation failure during training is non-fatal
                print(f"[TRAINING NASHCONV] ✗ Error: {type(e).__name__}: {e}")
                logger.warning(f"Could not compute NashConv during training (non-fatal): {type(e).__name__}: {e}")
                import traceback
                logger.debug(f"NashConv training computation traceback: {traceback.format_exc()}")
        
        # Add NashConv metrics to all_metrics if computed
        if training_nashconv_metrics:
            all_metrics.update(training_nashconv_metrics)
        
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
            logger.debug("Evaluation metrics logging disabled, skipping on_evaluation_end")
            return
        
        # Check if experiment is available
        if not hasattr(self, 'experiment') or self.experiment is None:
            logger.warning("on_evaluation_end called but experiment is not set. Callback may not be properly initialized.")
            return
        
        print(f"\n{'='*80}")
        print(f"CALLBACK: on_evaluation_end called!")
        print(f"  Number of rollouts: {len(rollouts) if rollouts else 0}")
        print(f"  Experiment: {self.experiment}")
        print(f"  Compute NashConv: {self.compute_nashconv}")
        print(f"{'='*80}\n")
        logger.info(f"on_evaluation_end called with {len(rollouts) if rollouts else 0} rollouts")
        
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
        
        # Always create aggregated dict, even if all_metrics is empty (for NashConv)
        aggregated = {}
        if all_metrics:
            for key, values in all_metrics.items():
                if values:
                    aggregated[f"{key}_mean"] = sum(values) / len(values)
                    aggregated[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
                    aggregated[f"{key}_min"] = min(values)
                    aggregated[f"{key}_max"] = max(values)
        
        aggregated["evaluation/n_episodes"] = n_episodes
        
        # Compute NashConv/exploitability metrics (always compute, even if other metrics are empty)
        if self.compute_nashconv:
            try:
                print(f"  Computing NashConv metrics from {len(rollouts)} rollouts...")
                nashconv_metrics = self._compute_nashconv_metrics(rollouts)
                if nashconv_metrics:
                    aggregated.update(nashconv_metrics)
                    nashconv_sum = nashconv_metrics.get('evaluation/nashconv_sum', 0.0)
                    exploitability_max = nashconv_metrics.get('evaluation/exploitability_max', 0.0)
                    print(f"  ✓ NashConv computed successfully!")
                    print(f"    NashConv sum: {nashconv_sum:.6f}")
                    print(f"    Exploitability max: {exploitability_max:.6f}")
                    logger.info(f"NashConv metrics: sum={nashconv_sum:.6f}, max={exploitability_max:.6f}")
                else:
                    print(f"  ⚠ NashConv computation returned empty metrics (insufficient data or computation failed)")
                    logger.debug("NashConv computation returned empty metrics (this is normal if computation fails or data is insufficient)")
            except Exception as e:
                print(f"  ✗ Error during NashConv computation: {e}")
                logger.warning(f"Error during NashConv computation (non-fatal): {e}")
                import traceback
                logger.debug(f"NashConv error traceback: {traceback.format_exc()}")
        
        if aggregated:
            
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
                    coverage_target = 0.1    # Default (updated to match config files)
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
                    
                    # Get NashConv from aggregated metrics (if available)
                    nashconv_sum = aggregated.get("evaluation/nashconv_sum", float('inf'))
                    nashconv_available = nashconv_sum != float('inf')
                    
                    # Strategy 1: Equilibrium-based (all classes meet targets) with NashConv check
                    if self.equilibrium_eval_mode and all_classes_meet_targets:
                        # Check if this is better than previous equilibrium
                        if class_scores:
                            min_class_score = min(s for _, _, s in class_scores.values())
                            
                            # Check NashConv if available
                            if nashconv_available:
                                # Prefer models with NashConv <= threshold (ε-Nash equilibrium)
                                if nashconv_sum <= self.nashconv_threshold:
                                    # NashConv is acceptable - save if score improved or NashConv improved
                                    score_improved = min_class_score > self.best_equilibrium_score
                                    nashconv_improved = nashconv_sum < self.best_equilibrium_nashconv
                                    
                                    if score_improved or (not hasattr(self, 'best_equilibrium_score') or 
                                                          (abs(min_class_score - self.best_equilibrium_score) < 0.01 and nashconv_improved)):
                                        save_model = True
                                        save_reason = "equilibrium"
                                        self.best_equilibrium_score = min_class_score
                                        self.best_equilibrium_nashconv = nashconv_sum
                                        logger.info(
                                            f"  ✓ New equilibrium checkpoint! "
                                            f"Min class score: {min_class_score:.4f}, "
                                            f"NashConv: {nashconv_sum:.6f} (≤ {self.nashconv_threshold:.3f})"
                                        )
                                else:
                                    # NashConv above threshold - only save if NashConv improved significantly
                                    # and we don't have a good equilibrium model yet
                                    if (not hasattr(self, 'best_equilibrium_nashconv') or 
                                        self.best_equilibrium_nashconv > self.nashconv_threshold):
                                        # No good equilibrium model yet - save if NashConv improved
                                        if nashconv_sum < self.best_equilibrium_nashconv:
                                            save_model = True
                                            save_reason = "equilibrium_improving"
                                            self.best_equilibrium_score = min_class_score
                                            self.best_equilibrium_nashconv = nashconv_sum
                                            logger.info(
                                                f"  ✓ New equilibrium checkpoint (improving NashConv)! "
                                                f"Min class score: {min_class_score:.4f}, "
                                                f"NashConv: {nashconv_sum:.6f} (target: ≤ {self.nashconv_threshold:.3f})"
                                            )
                            else:
                                # NashConv not available - fallback to score-based selection
                                if not hasattr(self, 'best_equilibrium_score') or min_class_score > self.best_equilibrium_score:
                                    save_model = True
                                    save_reason = "equilibrium"
                                    self.best_equilibrium_score = min_class_score
                                    logger.info(
                                        f"  ✓ New equilibrium checkpoint! "
                                        f"Min class score: {min_class_score:.4f} "
                                        f"(NashConv not available)"
                                    )
                    
                    # Strategy 2: Global aggregate (fallback or if equilibrium not reached) with NashConv tiebreaker
                    eval_score = precision + coverage  # Combined score
                    if not save_model:
                        score_improved = eval_score > self.best_eval_score
                        nashconv_improved = (nashconv_available and 
                                            hasattr(self, 'best_eval_nashconv') and 
                                            nashconv_sum < self.best_eval_nashconv)
                        
                        if score_improved:
                            save_model = True
                            save_reason = "aggregate"
                            self.best_eval_score = eval_score
                            if nashconv_available:
                                self.best_eval_nashconv = nashconv_sum
                        elif (nashconv_available and 
                              hasattr(self, 'best_eval_score') and 
                              abs(eval_score - self.best_eval_score) < 0.01 and  # Score within 0.01
                              nashconv_improved):
                            # Tiebreaker: prefer lower NashConv when scores are similar
                            save_model = True
                            save_reason = "aggregate_nashconv"
                            self.best_eval_nashconv = nashconv_sum
                            logger.info(
                                f"  ✓ New best model (NashConv tiebreaker)! "
                                f"Score: {eval_score:.4f} (similar to {self.best_eval_score:.4f}), "
                                f"NashConv: {nashconv_sum:.6f} (improved from {self.best_eval_nashconv:.6f})"
                            )
                    
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
                                nashconv_info = ""
                                if nashconv_available:
                                    nashconv_info = f", NashConv: {nashconv_sum:.6f}"
                                logger.info(
                                    f"  ✓ New best model saved (EQUILIBRIUM)! "
                                    f"All classes meet targets. "
                                    f"Global: P={precision:.4f}, C={coverage:.4f}{nashconv_info}"
                                )
                                if class_scores:
                                    for class_id, (p, c, s) in sorted(class_scores.items()):
                                        logger.info(f"    Class {class_id}: P={p:.4f}, C={c:.4f}, Score={s:.4f}")
                            elif save_reason == "equilibrium_improving":
                                logger.info(
                                    f"  ✓ New best model saved (EQUILIBRIUM - improving NashConv)! "
                                    f"All classes meet targets. "
                                    f"Global: P={precision:.4f}, C={coverage:.4f}, "
                                    f"NashConv: {nashconv_sum:.6f} (target: ≤ {self.nashconv_threshold:.3f})"
                                )
                                if class_scores:
                                    for class_id, (p, c, s) in sorted(class_scores.items()):
                                        logger.info(f"    Class {class_id}: P={p:.4f}, C={c:.4f}, Score={s:.4f}")
                            elif save_reason == "aggregate_nashconv":
                                logger.info(
                                    f"  ✓ New best model saved (NashConv tiebreaker)! "
                                    f"Score: {eval_score:.4f}, "
                                    f"Precision: {precision:.4f}, Coverage: {coverage:.4f}, "
                                    f"NashConv: {nashconv_sum:.6f}"
                                )
                            else:
                                nashconv_info = ""
                                if nashconv_available:
                                    nashconv_info = f", NashConv: {nashconv_sum:.6f}"
                                logger.info(
                                    f"  ✓ New best model saved! (Score: {eval_score:.4f}, "
                                    f"Precision: {precision:.4f}, Coverage: {coverage:.4f}{nashconv_info})"
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

