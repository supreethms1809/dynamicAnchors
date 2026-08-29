"""
The outline of the file was created by following the Custom env tutorial in the PettingZoo documentation.
"""
import functools
from copy import copy
import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any, List
from collections import defaultdict
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.device_utils import get_device
from utils.networks import predict_proba_torch
from utils.metrics import active_feature_mask, ranking_score as _ranking_score
from utils import quantile_mdp as qmdp
import logging
logger = logging.getLogger(__name__)


class AnchorEnv(ParallelEnv):
    metadata = {
        "name": "AnchorEnv",
        "description": "AnchorEnv is a multi-agent environment for finding anchors",
        "keywords": ["multi-agent", "anchor", "environment"],
        "render_modes": None,
    }

    def __init__(
        self,
        X_unit: Optional[np.ndarray] = None,
        X_std: Optional[np.ndarray] = None,
        y: np.ndarray = None,
        feature_names: list = None,
        classifier = None,
        device: str = "cpu",
        target_class: Optional[int] = None,
        target_classes: Optional[List[int]] = None,
        env_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        if env_config is None:
            env_config = {}
        
        normalize_data = env_config.get("normalize_data", False)
        
        if normalize_data:
            if X_std is None:
                raise ValueError("X_std must be provided when normalize_data=True")
            X_unit_normalized, X_min, X_range = self._normalize_data(X_std, env_config)
            X_unit = X_unit_normalized
            if env_config.get("X_min") is None:
                env_config["X_min"] = X_min
            if env_config.get("X_range") is None:
                env_config["X_range"] = X_range
        else:
            if X_unit is None or X_std is None:
                raise ValueError("Both X_unit and X_std must be provided when normalize_data=False")
        
        self.X_unit = X_unit
        self.X_std = X_std
        self.y = y.astype(int)
        self.feature_names = feature_names
        self.n_features = X_unit.shape[1]
        self.classifier = classifier
        self.device = get_device(device)
        
        if target_classes is None:
            if target_class is not None:
                target_classes = [target_class]
            else:
                target_classes = sorted(np.unique(y).tolist())
        
        self.target_classes = target_classes

        self.agents_per_class = env_config.get("agents_per_class", 1)

        # SS - Reference needed for policy extraction and inference
        # For agents_per_class == 1:
        #   class c -> "agent_c"
        # For agents_per_class > 1:
        #   class c -> ["agent_c_0", "agent_c_1", ..., "agent_c_{K-1}"]
        self.possible_agents = []
        self.agent_to_class = {}
        for cls in target_classes:
            if self.agents_per_class == 1:
                name = f"agent_{cls}"
                self.possible_agents.append(name)
                self.agent_to_class[name] = cls
            else:
                for k in range(self.agents_per_class):
                    name = f"agent_{cls}_{k}"
                    self.possible_agents.append(name)
                    self.agent_to_class[name] = cls
        
        # SS - Reference needed for class-level metrics and within-class diversity penalties
        # This is used for class-level metrics (union coverage/precision) and
        # for within-class diversity penalties when multiple agents represent
        # the same class.
        self.class_to_agents = defaultdict(list)
        for agent, cls in self.agent_to_class.items():
            self.class_to_agents[cls].append(agent)
        
        # Group mapping for BenchMARL: maps group names to lists of agent names.
        # By default, each agent is its own group.
        self._group_map = self._compute_group_map()
        
        step_fracs = env_config.get("step_fracs", (0.005, 0.01, 0.02))
        if step_fracs is None or len(step_fracs) == 0:
            raise ValueError("step_fracs cannot be empty. Provide at least one step fraction value.")
        self.step_fracs = step_fracs
        self.min_width = env_config.get("min_width", 0.05)
        self.alpha = env_config.get("alpha", 0.7)
        self.beta = env_config.get("beta", 0.6)
        self.gamma = env_config.get("gamma", 0.1)
        # Discount used by the Ng et al. (1999) shaping term. Deliberately NOT
        # self.gamma, which is the narrow-width overlap penalty weight. Kept in sync
        # with single_agentENV so RLDA and MADA share one shaping definition.
        self.discount = float(env_config.get("discount", 0.99))
        self.coverage_target = env_config.get("coverage_target", 0.2)
        self.precision_target = env_config.get("precision_target", 0.9)
        self.precision_blend_lambda = env_config.get("precision_blend_lambda", 1.0)
        self.drift_penalty_weight = env_config.get("drift_penalty_weight", 0.05)
        
        # SS - Termination reason counters (diagnostics only). The old max-count
        # mechanism that permanently disabled overused reasons mid-training was
        # removed: it made the MDP non-stationary under the replay buffer. The
        # terminal bonus now provides the incentive it was approximating.
        self._reset_termination_counters()

        # Quantile-position MDP (Anchors procedure order), mirroring
        # single_agentENV. OPT-IN for MADA: the default stays "neighbor_hull" so
        # in-flight runs and the WyoDOT/CDEA pipeline keep their current behaviour
        # and their existing 2d+3 checkpoints. Flip conf/anchor.yaml to
        # init_mode: full_space for the ported MADA runs.
        self.init_mode = str(env_config.get("init_mode", "neighbor_hull")).lower()
        self.precision_estimator = str(
            env_config.get(
                "precision_estimator",
                "empirical" if self.init_mode == "neighbor_hull" else "conditional",
            )
        ).lower()
        self.quantile_eps = float(env_config.get("quantile_eps", 1e-3))
        self.max_quantile_step = float(env_config.get("max_quantile_step", 0.10))
        self.min_quantile_width = float(env_config.get("min_quantile_width", 0.02))
        self.leave_threshold = float(env_config.get("leave_threshold", 0.85))
        self.max_new_constraints_per_step = int(
            env_config.get("max_new_constraints_per_step", 1)
        )
        self.activation_quantile = float(env_config.get("activation_quantile", 0.05))
        self.reset_diversity_frac = float(env_config.get("reset_diversity_frac", 0.3))
        self.n_reset_landings = int(env_config.get("n_reset_landings", 2))
        self.sparsity_terminal_weight = float(
            env_config.get("sparsity_terminal_weight", 1.0)
        )
        self.require_precision_gain_to_terminate = bool(
            env_config.get(
                "require_precision_gain_to_terminate", self.init_mode != "neighbor_hull"
            )
        )
        # Per-agent quantile state. Unit bounds stay the source of truth for every
        # downstream consumer; (a, b) is the action/observation space.
        self.a: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        self.q_star: Dict[str, np.ndarray] = {}
        self._crn_idx: Dict[str, np.ndarray] = {}
        self._crn_U: Dict[str, np.ndarray] = {}
        self._precision_at_reset: Dict[str, float] = {}
        # Best feasible box seen this episode, per agent. With the floor enforced
        # only at the end the trajectory is free to collapse, so the episode needs a
        # memory of the best anchor it actually found. Measured: under a random
        # policy breast_cancer (d=30) ends collapsed on 100% of agent-episodes
        # (k=27/30, coverage 0.000) -- without this the episode returns nothing
        # usable, while RLDA retreats and does not.
        self._best_box: Dict[str, tuple] = {}
        self._best_box_score: Dict[str, float] = {}
        self._class_centroid_unit: Dict[int, np.ndarray] = {}
        # CDFs are class-conditional, so cache per target class rather than per agent
        # (agents of the same class share them).
        self._quantile_cdfs_by_class: Dict[int, Dict[str, np.ndarray]] = {}

        self.use_perturbation = env_config.get("use_perturbation", False)
        self.perturbation_mode = env_config.get("perturbation_mode", "bootstrap")
        self.n_perturb = int(env_config.get("n_perturb_train", env_config.get("n_perturb", 2048)))
        self.n_perturb_eval = int(env_config.get("n_perturb_eval", self.n_perturb))
        
        # # Log perturbation settings during initialization
        # logger.info(f"AnchorEnv initialized with perturbation settings: "
        #            f"use_perturbation={self.use_perturbation}, "
        #            f"perturbation_mode={self.perturbation_mode}, "
        #            f"n_perturb={self.n_perturb}")
        if self.use_perturbation and self.perturbation_mode == "adaptive":
            min_points_threshold = max(1, int(0.1 * self.n_perturb))
            # logger.info(f"  Adaptive mode threshold: will use uniform sampling if covered points < {min_points_threshold}")
        self.X_min = env_config.get("X_min", None)
        self.X_range = env_config.get("X_range", None)
        self.scaler_mean = env_config.get("scaler_mean", None)
        self.scaler_scale = env_config.get("scaler_scale", None)
        if self.scaler_mean is not None:
            self.scaler_mean = np.asarray(self.scaler_mean, dtype=np.float32)
        if self.scaler_scale is not None:
            self.scaler_scale = np.asarray(self.scaler_scale, dtype=np.float32)
        self.rng = env_config.get("rng", None)
        if self.rng is None:
            self.rng = np.random.default_rng(42)
        self.min_coverage_floor = env_config.get("min_coverage_floor", 0.005)
        # The floor is only an in-episode "do not collapse to nothing" guard. It must
        # never exceed coverage_target, or a box that legitimately meets the target is
        # judged infeasible and retreated away -- and since the retreat pool only
        # holds boxes already above the floor, that trades precision for coverage.
        _ct = float(env_config.get("coverage_target", 0.2))
        # Relative floor: "at least min_support class rows in the box", matching
        # single_agentENV. n_class varies per class, so use the SMALLEST class --
        # the floor is a shared scalar here and the smallest class gives the
        # least-binding value, which is the safe direction for a collapse guard.
        self.coverage_floor_relative = bool(env_config.get("coverage_floor_relative", False))
        self.coverage_floor_min_support = int(env_config.get("min_support", 10))
        if self.coverage_floor_relative:
            try:
                _counts = [int((np.asarray(y) == c).sum()) for c in np.unique(np.asarray(y))]
                _n_min = max(1, min([c for c in _counts if c > 0], default=1))
                self.min_coverage_floor = float(self.coverage_floor_min_support) / float(_n_min)
            except Exception:
                pass
        if _ct > 0.0:
            self.min_coverage_floor = max(min(float(self.min_coverage_floor), _ct), 1e-6)
        # "step" preserves CDEA/WyoDOT revert-on-hit. "terminal" matches RLDA.
        self.coverage_floor_mode = str(env_config.get("coverage_floor_mode", "step")).lower()
        self.partial_terminal_credit = float(env_config.get("partial_terminal_credit", 0.0))
        self.js_penalty_weight = env_config.get("js_penalty_weight", 0.05)
        self.initial_window = env_config.get("initial_window", 0.1)
        # Neighborhood-hull init: cover k nearest class points around the start,
        # not a ±initial_window cube (high-d cubes are singletons and do not transfer).
        self.init_min_neighbors = int(env_config.get("init_min_neighbors", 5))
        self.init_neighbor_frac = float(env_config.get("init_neighbor_frac", 0.1))
        self.init_max_neighbors = int(env_config.get("init_max_neighbors", 20))
        # Cap start hull so class-cond C is at most this fraction of τ_C.
        # Wins over init_min_neighbors (wine n≈35, τ_C=0.2 → k=3, not 5).
        self.init_coverage_frac_of_target = float(
            env_config.get("init_coverage_frac_of_target", 0.5)
        )
        # Only terminate on both_targets_met (disable excellent/high_prec/close).
        self.strict_target_termination = bool(
            env_config.get("strict_target_termination", True)
        )
        # Terminal bonus / target termination require C > C_init + 1/n_class.
        self.require_coverage_gain_to_terminate = bool(
            env_config.get("require_coverage_gain_to_terminate", True)
        )
        self._coverage_at_reset: Dict[str, float] = {}
        self._coverage_gain_eps: Dict[str, float] = {}
        self._union_coverage_at_reset: Dict[int, float] = {}
        # 0.0 = always snap a mean/cluster centroid to the nearest class row.
        # A 0.5 threshold left means sitting in empty space at distance 0.3.
        self.centroid_snap_threshold = float(env_config.get("centroid_snap_threshold", 0.0))
        self.fixed_instances_per_class = env_config.get("fixed_instances_per_class", None)
        self.cluster_centroids_per_class = env_config.get("cluster_centroids_per_class", None)
        self.training_instances_per_class = env_config.get("training_instances_per_class", None)
        self.training_instance_ratio = env_config.get("training_instance_ratio", 0.5)  # Base ratio (fallback)
        self.training_instance_ratios_per_class = env_config.get("training_instance_ratios_per_class", None)  # Class-specific ratios
        self.use_random_sampling = env_config.get("use_random_sampling", False)
        self.use_class_centroids = env_config.get("use_class_centroids", True)
        # B2 rebalance: precision ramp width for the coverage gate (mirrors SA env).
        self.gate_margin = float(env_config.get("gate_margin", 0.10))  # Default: use centroids for initialization
        # Explicit class-based start point (unit space). Set per rollout by inference
        # to cycle through diversified starts (k-means centroids + random class
        # samples) instead of reusing the same agent-indexed centroid every rollout.
        self.class_init_point = env_config.get("class_init_point", None)

        self.eval_on_test_data = env_config.get("eval_on_test_data", False)
        # C-10: eval_split in {train, val, test}. eval_on_test_data=True maps to
        # "test" for backward compatibility. Training must stay on train.
        self.eval_split = env_config.get("eval_split", "test" if self.eval_on_test_data else "train")
        self.X_val_unit = env_config.get("X_val_unit", None)
        self.X_val_std = env_config.get("X_val_std", None)
        self.y_val = None if env_config.get("y_val") is None else np.asarray(env_config["y_val"]).astype(int)

        # --- Potential-based reward shaping state (mirrors single_agentENV.py) ---
        # Terminal bonus paid when an agent's episode terminates with targets met.
        self.terminal_bonus = env_config.get("terminal_bonus", 5.0)
        # Cooperative analogue of terminal_bonus: paid once per episode per class
        # the first step that class's UNION of boxes meets both targets
        # (union_precision >= effective target AND union_coverage >= coverage_target).
        # The local terminal_bonus is unreachable for minority classes — a
        # high-precision box on a 2%-prior class covers far fewer than
        # coverage_target samples — so those agents only ever see the smooth
        # shaping signal (~0). The class union IS reachable (multiple boxes cover
        # more than one), and the union is exactly what inference extracts, so a
        # union-target bonus gives the failing classes an attainable discrete goal.
        # Latched per episode (see _class_union_bonus_paid) so it cannot be farmed.
        # Default 0.0 = opt-in: existing runs are unchanged unless the config sets it.
        self.shared_terminal_bonus = env_config.get("shared_terminal_bonus", 0.0)
        # Cached classifier probabilities over the train/val/test sets (lazy, once).
        self._cached_probs = {"train": None, "val": None, "test": None}
        # C-12: black-box query counter (cache fills count as queries; lookups do not).
        self.n_blackbox_queries = 0
        # C-07: freeze these feature indices to the instance/class-mode value.
        self.categorical_indices = list(env_config.get("categorical_indices") or [])
        self.categorical_value_names = {
            int(k): list(v)
            for k, v in (env_config.get("categorical_value_names") or {}).items()
        }
        self.categorical_freeze = env_config.get("categorical_freeze", "instance")  # instance | class_mode | none
        # C-08: drop a printed feature when its width >= this fraction of the full range.
        self.sparsity_width_ratio = float(env_config.get("sparsity_width_ratio", 0.95))
        # C-52: ranking score formula used at inference (documented hyperparameter).
        self.ranking_score_formula = env_config.get("ranking_score_formula", "precision_coverage")
        self.top_k_rules_by_score = env_config.get("top_k_rules_by_score", 5)
        self.min_support = int(env_config.get("min_support", 10))
        # Per-agent metrics from the last reset()/step(), reused as prev metrics by
        # the next step() (halves classifier calls; common-random-number gains).
        self._last_step_metrics: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
        # Class-aware effective precision targets (lazy per class).
        self.use_class_aware_targets = env_config.get("use_class_aware_targets", True)
        self._effective_precision_targets: Dict[int, float] = {}
        # Per-class union potential from the previous step, for the same-class
        # shared reward (class-union potential gain).
        self._prev_class_phi: Dict[int, float] = {}
        # Latch: has the one-time shared_terminal_bonus already been paid for this
        # class this episode? Reset in reset() alongside _prev_class_phi.
        self._class_union_bonus_paid: Dict[int, bool] = {}
        # Per-agent inter-class overlap level from the previous step (the overlap
        # penalty charges the change in this level; seeded in reset())
        self._prev_inter_overlap: Dict[str, float] = {}
        
        # Track warnings per agent per episode to reduce log spam
        # Only warn once per episode about adaptive mode switching and precision-coverage mismatch
        self._adaptive_uniform_warned_this_episode = defaultdict(bool)  # agent -> bool
        self._precision_coverage_mismatch_warned_this_episode = defaultdict(bool)  # agent -> bool
        self._rt_shaping = defaultdict(float)
        self._rt_overlap = defaultdict(float)
        self._rt_drift = defaultdict(float)
        self._rt_anchor_drift = defaultdict(float)
        self._rt_cov_floor = defaultdict(float)
        self._rt_terminal = defaultdict(float)
        self._rt_total = defaultdict(float)
        
        if self.eval_on_test_data:
            X_test_unit = env_config.get("X_test_unit", None)
            X_test_std = env_config.get("X_test_std", None)
            y_test = env_config.get("y_test", None)
            if X_test_unit is None or X_test_std is None or y_test is None:
                raise ValueError("eval_on_test_data=True requires X_test_unit, X_test_std, and y_test")
            self.X_test_unit = X_test_unit
            self.X_test_std = X_test_std
            self.y_test = y_test.astype(int)
        else:
            self.X_test_unit = None
            self.X_test_std = None
            self.y_test = None

        self.max_action_scale = env_config.get("max_action_scale", 0.1)
        self.min_absolute_step = env_config.get("min_absolute_step", 0.001)
        
        # Environment mode: "training", "evaluation", or "inference"
        self.mode = env_config.get("mode", "training")
        
        self.inter_class_overlap_weight = env_config.get("inter_class_overlap_weight", 0.1)
        # Shared reward weight for cooperative behavior (applied to all agents)
        self.shared_reward_weight = env_config.get("shared_reward_weight", 0.5)
        
        # Weights for class-level rewards and within-class diversity.
        # Defaults are 0.0 so the existing reward structure remains unchanged
        # unless explicitly enabled via env_config.
        self.class_union_cov_weight = env_config.get("class_union_cov_weight", 0.0)
        self.class_union_prec_weight = env_config.get("class_union_prec_weight", 0.0)
        self.same_class_diversity_weight = env_config.get("same_class_diversity_weight", 0.0)
        
        # Global coverage reward: weight and threshold for rewarding when all agents together cover the dataset
        self.global_coverage_weight = env_config.get("global_coverage_weight", 0.0)
        self.global_coverage_threshold = env_config.get("global_coverage_threshold", 0.0)
        
        # Coverage bonus weights (read from config, defaults match reduced values for fair comparison with single-agent)
        self.coverage_bonus_weight_met = env_config.get("coverage_bonus_weight_met", 0.01)
        self.coverage_bonus_weight_high_prec = env_config.get("coverage_bonus_weight_high_prec", 0.03)
        self.coverage_bonus_weight_high_prec_progress = env_config.get("coverage_bonus_weight_high_prec_progress", 0.07)
        self.coverage_bonus_weight_high_prec_distance = env_config.get("coverage_bonus_weight_high_prec_distance", 0.02)
        self.coverage_bonus_weight_reasonable_prec = env_config.get("coverage_bonus_weight_reasonable_prec", 0.01)
        self.coverage_bonus_weight_reasonable_prec_progress = env_config.get("coverage_bonus_weight_reasonable_prec_progress", 0.02)
        
        # Target class bonus weight (read from config, default matches reduced value for fair comparison)
        self.target_class_bonus_weight = env_config.get("target_class_bonus_weight", 0.02)
        
        x_star_unit_config = env_config.get("x_star_unit", None)
        if isinstance(x_star_unit_config, dict):
            self.x_star_unit = x_star_unit_config
        else:
            self.x_star_unit = {}
        
        # Store original predictions for instance-based anchors (matches original Anchor paper)
        # Key: agent name, Value: original prediction (int) for that instance
        self.original_predictions = {}

        self.lower = {}
        self.upper = {}
        self.prev_lower = {}
        self.prev_upper = {}
        self.box_history = {}
        self.coverage_floor_hits = {}
        self.timestep = None
        # Read max_cycles from config - no hardcoded default to ensure YAML settings are respected
        self.max_cycles = env_config.get("max_cycles")
        if self.max_cycles is None:
            raise ValueError("max_cycles must be specified in env_config. Check your YAML config file.")
        self.max_cycles = int(self.max_cycles)

        # -----------------------------
        # Stabilization-based early termination (per-agent)
        # -----------------------------
        # If an agent's box and metrics stop changing for a window of steps, terminate early.
        # Minimum steps before target-based termination is allowed. Without this,
        # an agent can terminate on step 1 from the initial box (the degenerate
        # 1-step termination), claiming the terminal bonus without ever shaping
        # the anchor. Mirrors single_agentENV.min_steps_before_termination (=2)
        # so MA and SA training dynamics match.
        self.min_steps_before_termination = int(env_config.get("min_steps_before_termination", 2))
        self.enable_stability_termination = env_config.get("enable_stability_termination", True)
        self.stability_window = int(env_config.get("stability_window", 10))
        self.stability_min_steps = int(env_config.get("stability_min_steps", 20))
        self.stability_precision_tol = float(env_config.get("stability_precision_tol", 1e-3))
        self.stability_coverage_tol = float(env_config.get("stability_coverage_tol", 1e-3))
        self.stability_drift_tol = float(env_config.get("stability_drift_tol", 1e-3))

        # Counter of consecutive "stable" steps per agent
        self._stable_counts: Dict[str, int] = {}
        self._log_effective_config()

    def _log_effective_config(self) -> None:
        """Print the knobs this env actually uses (YAML/CLI after any trainer merge)."""
        bits = [
            f"mode={self.mode}",
            f"precision_target={self.precision_target}",
            f"coverage_target={self.coverage_target}",
            f"max_cycles={self.max_cycles}",
            f"use_perturbation={self.use_perturbation}",
            f"precision_blend_lambda={self.precision_blend_lambda}",
            f"training_instance_ratio={self.training_instance_ratio}",
            f"eval_split={self.eval_split}",
            f"eval_on_test_data={self.eval_on_test_data}",
            f"strict_target_termination={self.strict_target_termination}",
            f"require_coverage_gain_to_terminate={self.require_coverage_gain_to_terminate}",
            f"init_coverage_frac_of_target={self.init_coverage_frac_of_target}",
            f"init_min_neighbors={self.init_min_neighbors}",
            f"centroid_snap_threshold={self.centroid_snap_threshold}",
            f"min_steps_before_termination={self.min_steps_before_termination}",
            f"alpha={self.alpha} beta={self.beta} gamma={self.gamma}",
            f"terminal_bonus={self.terminal_bonus}",
            f"shared_terminal_bonus={self.shared_terminal_bonus}",
            f"shared_reward_weight={self.shared_reward_weight}",
            f"agents_per_class={self.agents_per_class}",
            f"min_coverage_floor={self.min_coverage_floor}",
            f"use_class_aware_targets={self.use_class_aware_targets}",
            f"categorical_freeze={self.categorical_freeze}",
        ]
        logger.info("EFFECTIVE ENV CONFIG (honored YAML/CLI): " + " ".join(bits))

    # SS: This is a helper method to normalize the data. It is used to normalize the data for the perturbation sampling.
    @staticmethod
    def _normalize_data(X_std: np.ndarray, env_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_min = env_config.get("X_min", None)
        X_range = env_config.get("X_range", None)
        
        if X_min is None or X_range is None:
            X_min = X_std.min(axis=0)
            X_max = X_std.max(axis=0)
            X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
        
        X_unit = (X_std - X_min) / X_range
        X_unit = np.clip(X_unit, 0.0, 1.0).astype(np.float32)
        
        return X_unit, X_min, X_range

    # SS: It is used to mask the data in the box for the perturbation sampling.
    def _mask_in_box(self, agent: str) -> np.ndarray:
        X_eval_unit, _, _, _ = self._active_data()
        
        # FIX: Ensure we have valid data and box bounds
        if X_eval_unit is None or X_eval_unit.shape[0] == 0:
            logger.warning(f"Agent {agent}: X_eval_unit is None or empty in _mask_in_box")
            return np.zeros(0, dtype=bool)
        
        if agent not in self.lower or agent not in self.upper:
            logger.warning(f"Agent {agent}: Box bounds not found in _mask_in_box")
            return np.zeros(X_eval_unit.shape[0], dtype=bool)
        
        # FIX: Verify that box bounds are in normalized space [0, 1]
        # This ensures the mask computation uses the same normalized space as the data
        lower = self.lower[agent]
        upper = self.upper[agent]
        
        # Debug: Log if box bounds are outside [0, 1] range (indicates normalization issue)
        if np.any(lower < 0) or np.any(lower > 1) or np.any(upper < 0) or np.any(upper > 1):
            logger.warning(
                f"Agent {agent}: Box bounds outside [0, 1] range in _mask_in_box. "
                f"lower range: [{lower.min():.3f}, {lower.max():.3f}], "
                f"upper range: [{upper.min():.3f}, {upper.max():.3f}]. "
                f"This may indicate a normalization mismatch."
            )
        
        conds = []
        for j in range(self.n_features):
            conds.append((X_eval_unit[:, j] >= lower[j]) & (X_eval_unit[:, j] <= upper[j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(X_eval_unit.shape[0], dtype=bool)
        return mask

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min

    def _std_to_orig(self, X_std_samples: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_scale is None:
            return X_std_samples
        return X_std_samples * self.scaler_scale + self.scaler_mean

    def _unit_to_orig(self, X_unit_samples: np.ndarray) -> np.ndarray:
        return self._std_to_orig(self._unit_to_std(X_unit_samples))

    def _active_split(self) -> str:
        """Which split the live metrics (reward / termination) are computed on.

        Training: always 'train'. Inference/eval: env_config['eval_split']
        (val for rule generation, test only in the post-hoc evaluator).
        C-06: this choice is an env constant, not a joint-action-dependent ξ_t.
        """
        if getattr(self, "mode", "training") == "training":
            return "train"
        split = getattr(self, "eval_split", None)
        if split in ("train", "val", "test"):
            return split
        return "test" if self.eval_on_test_data else "train"

    def _active_data(self):
        split = self._active_split()
        if split == "test":
            if self.X_test_unit is None:
                raise ValueError("eval_split='test' but X_test_unit is not set")
            return self.X_test_unit, self.X_test_std, self.y_test, "test"
        if split == "val":
            if self.X_val_unit is None:
                logger.warning("eval_split='val' but no val data; falling back to test")
                return self.X_test_unit, self.X_test_std, self.y_test, "test"
            return self.X_val_unit, self.X_val_std, self.y_val, "val"
        return self.X_unit, self.X_std, self.y, "training"

    def _init_n_neighbors(self, n_class: int) -> int:
        """Size the start hull so class-cond C starts below τ_C.

        k is min(frac·n_class, init_max_neighbors, n_class, cap_tau), then
        floored at init_min_neighbors unless that would exceed cap_tau.
        cap_tau = max(1, int(init_coverage_frac_of_target · τ_C · n_class)),
        so wine n≈35, τ_C=0.2, frac=0.5 → k=3 (C≈0.086), not min_neighbors=5.
        """
        n_class = int(max(1, n_class))
        cap_tau = max(
            1,
            int(self.init_coverage_frac_of_target * float(self.coverage_target) * n_class),
        )
        n = max(1, int(self.init_neighbor_frac * n_class))
        n = max(n, int(self.init_min_neighbors))
        n = min(n, int(self.init_max_neighbors), n_class, cap_tau)
        return int(max(1, n))

    @staticmethod
    def _class_conditional_coverage(
        mask: np.ndarray, y_data: np.ndarray, target_class: int
    ) -> Tuple[float, float, int, int]:
        """Return (class_cond_C, marginal_C, n_class_in_box, n_class)."""
        coverage_marginal = float(mask.mean()) if len(mask) else 0.0
        class_mask = y_data == target_class
        n_class = int(class_mask.sum())
        if n_class == 0 or len(mask) != len(class_mask):
            return 0.0, coverage_marginal, 0, n_class
        n_in = int((mask & class_mask).sum())
        return float(n_in / n_class), coverage_marginal, n_in, n_class

    def _coverage_improved(self, agent: str, coverage: float) -> bool:
        if not self.require_coverage_gain_to_terminate:
            return True
        c_reset = float(self._coverage_at_reset.get(agent, 0.0))
        # Already at τ_C at reset: extra gain is not required to terminate.
        if c_reset >= float(self.coverage_target):
            return True
        return float(coverage) > (
            c_reset + float(self._coverage_gain_eps.get(agent, 0.0))
        )

    def _snap_to_nearest_class_point(self, point: np.ndarray, class_data: np.ndarray) -> np.ndarray:
        """Snap a mean/cluster centroid onto the nearest real class row.

        Threshold 0.0 means any positive distance snaps. A 0.5 cutoff left
        high-d means sitting in empty space (distance ~0.3) unsnapped.
        """
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if class_data is None or len(class_data) == 0:
            return point
        distances = np.linalg.norm(class_data - point, axis=1)
        nearest = class_data[int(np.argmin(distances))]
        if float(distances.min()) > self.centroid_snap_threshold:
            return np.asarray(nearest, dtype=np.float32)
        return point.astype(np.float32)

    def _include_point_in_box(self, agent: str, point: np.ndarray) -> None:
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        self.lower[agent] = np.minimum(self.lower[agent], point).astype(np.float32)
        self.upper[agent] = np.maximum(self.upper[agent], point).astype(np.float32)
        self.lower[agent] = np.clip(self.lower[agent], 0.0, 1.0).astype(np.float32)
        self.upper[agent] = np.clip(self.upper[agent], 0.0, 1.0).astype(np.float32)

    def _window_box_around(self, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = max(self.initial_window, self.min_width)
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        lower = np.clip(point - w, 0.0, 1.0).astype(np.float32)
        upper = np.clip(point + w, 0.0, 1.0).astype(np.float32)
        return lower, upper

    def _freeze_categorical_bounds(self, agent: str) -> None:
        """C-07 option (a): do not refine categorical dimensions; pin to instance/mode."""
        if not self.categorical_indices or self.categorical_freeze == "none":
            return
        x_star = self.x_star_unit.get(agent)
        cls = self._get_class_for_agent(agent)
        X_data, _, y_data, _ = self._active_data()
        for j in self.categorical_indices:
            width = float(self.upper[agent][j] - self.lower[agent][j])
            if width >= float(getattr(self, "sparsity_width_ratio", 0.95)):
                continue
            if x_star is not None and self.categorical_freeze == "instance":
                v = float(np.asarray(x_star).reshape(-1)[j])
            elif cls is not None:
                class_rows = X_data[y_data == cls]
                if class_rows.shape[0] == 0:
                    continue
                # mode in unit space (label-encoded -> scaled -> unit, still unimodal)
                vals, counts = np.unique(np.round(class_rows[:, j], 6), return_counts=True)
                v = float(vals[int(np.argmax(counts))])
            else:
                continue
            # Tight interval around the frozen value so the mask is an equality.
            self.lower[agent][j] = np.clip(v - 1e-4, 0.0, 1.0)
            self.upper[agent][j] = np.clip(v + 1e-4, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Quantile-position MDP helpers (mirror single_agentENV; per-agent state)
    # ------------------------------------------------------------------
    def _uses_quantile_mdp(self) -> bool:
        return str(getattr(self, "init_mode", "neighbor_hull")).lower() != "neighbor_hull"

    def _cdfs_for_class(self, cls: Optional[int]) -> Dict[str, np.ndarray]:
        """Per-class empirical CDFs, fitted once on D_train and shared by that
        class's agents (the quantile space is class-conditional)."""
        key = int(cls) if cls is not None else -1
        if key not in self._quantile_cdfs_by_class:
            self._quantile_cdfs_by_class[key] = qmdp.fit_train_cdfs(
                self.X_unit, self.y, key if key >= 0 else int(np.unique(self.y)[0])
            )
        return self._quantile_cdfs_by_class[key]

    def _cdfs(self, agent: str) -> Dict[str, np.ndarray]:
        return self._cdfs_for_class(self._get_class_for_agent(agent))

    def _constrained_mask(self, agent: str) -> np.ndarray:
        q_active = None
        if self._uses_quantile_mdp() and self.a.get(agent) is not None:
            q_active = qmdp.constrained_mask(self.a[agent], self.b[agent], self.quantile_eps)
        lo, up = self.lower.get(agent), self.upper.get(agent)
        if lo is None or up is None:
            if q_active is not None:
                return np.asarray(q_active, dtype=bool)
            return np.zeros(self.n_features, dtype=bool)
        return active_feature_mask(
            lo, up,
            sparsity_width_ratio=getattr(self, "sparsity_width_ratio", 0.95),
            quantile_active=q_active,
        )

    def n_predicates(self, agent: str) -> int:
        return int(self._constrained_mask(agent).sum())

    def _sync_unit_bounds_from_quantiles(self, agent: str) -> None:
        lo, up = qmdp.quantile_to_unit_bounds(
            self.a[agent], self.b[agent], self._cdfs(agent)["v_class"], self.quantile_eps
        )
        self.lower[agent], self.upper[agent] = lo, up

    def _values_to_q(self, agent: str, x_unit: np.ndarray) -> np.ndarray:
        v = self._cdfs(agent)["v_class"]
        x_unit = np.asarray(x_unit, dtype=np.float64).reshape(-1)
        return np.array(
            [qmdp.value_to_quantile(v[j], float(x_unit[j])) for j in range(self.n_features)],
            dtype=np.float64,
        )

    def _class_centroid_quantiles(self, agent: str) -> np.ndarray:
        cls = self._get_class_for_agent(agent)
        mask = self.y == cls if cls is not None else np.ones(len(self.y), dtype=bool)
        if not mask.any():
            return np.full(self.n_features, 0.5, dtype=np.float64)
        centroid = np.median(self.X_unit[mask], axis=0)
        if cls is not None:
            self._class_centroid_unit[int(cls)] = centroid.astype(np.float32)
        return self._values_to_q(agent, centroid)

    def _draw_crn(self, agent: str) -> None:
        n = max(1, int(self.n_perturb if self.mode == "training" else self.n_perturb_eval))
        self._crn_idx[agent] = self.rng.integers(0, int(self.X_unit.shape[0]), size=n)
        self._crn_U[agent] = self.rng.random((n, self.n_features))

    def _empty_rule_eligible(self, agent: str, precision: float, k: Optional[int] = None) -> bool:
        if k is None:
            k = self.n_predicates(agent)
        p_reset = float(self._precision_at_reset.get(agent, 0.0))
        return int(k) >= 1 and float(precision) > p_reset + 1e-12

    def _precision_improved(self, agent: str, precision: float) -> bool:
        if not self.require_precision_gain_to_terminate:
            return True
        return float(precision) > float(self._precision_at_reset.get(agent, 0.0)) + 1e-12

    def _pin_constrained_categoricals(self, agent: str) -> None:
        """Categoricals are an equality predicate only while the dim is ACTIVE.

        The hull path froze every categorical for the whole episode, so the policy
        never controlled them and every rule carried a forced equality on each --
        which alone can pin class-conditional coverage near zero on the categorical
        datasets (uci_credit, uci_adult).
        """
        if not self.categorical_indices or self.categorical_freeze == "none":
            return
        if not self._uses_quantile_mdp():
            return
        X_data, _, y_data, _ = self._active_data()
        cls = self._get_class_for_agent(agent)
        active = self._constrained_mask(agent)
        v_class = self._cdfs(agent)["v_class"]
        x_star = self.x_star_unit.get(agent)
        for j in self.categorical_indices:
            if not active[j]:
                self.a[agent][j], self.b[agent][j] = 0.0, 1.0
                continue
            if x_star is not None and self.categorical_freeze == "instance":
                v = float(np.asarray(x_star).reshape(-1)[j])
            else:
                rows = X_data[y_data == cls] if cls is not None else X_data
                if rows.shape[0] == 0:
                    continue
                vals, counts = np.unique(np.round(rows[:, j], 6), return_counts=True)
                v = float(vals[int(np.argmax(counts))])
            self.a[agent][j], self.b[agent][j] = qmdp.categorical_atom_quantiles(v_class[j], v)

    def _maybe_reset_diversity_landing(self, agent: str) -> None:
        """Randomised initial landing so deterministic rollouts are not identical.

        Without it every reset yields the same all-corner observation and a
        deterministic policy produces ONE rule per class, which makes the class
        union / NMS / set-cover story vacuous.
        """
        if not self._uses_quantile_mdp():
            return
        if self.reset_diversity_frac <= 0.0 or self.rng.random() >= self.reset_diversity_frac:
            return
        k_land = max(0, int(self.n_reset_landings))
        if k_land <= 0:
            return
        dims = self.rng.choice(self.n_features, size=min(k_land, self.n_features), replace=False)
        q = float(np.clip(self.activation_quantile, 0.0, 0.49))
        x_star = self.x_star_unit.get(agent)
        for j in dims:
            if x_star is not None:
                qj = float(self.q_star[agent][j])
                self.a[agent][j] = max(0.0, qj - q)
                self.b[agent][j] = min(1.0, qj + q)
            else:
                self.a[agent][j], self.b[agent][j] = q, 1.0 - q
        if x_star is not None:
            self.a[agent], self.b[agent] = qmdp.clip_quantiles_around_qstar(
                self.a[agent], self.b[agent], self.q_star[agent],
                self._constrained_mask(agent), self.quantile_eps,
            )

    def _get_observation(self, agent: str, precision: float, coverage: float,
                         episode_phase: float = 0.0) -> np.ndarray:
        if not self._uses_quantile_mdp():
            return np.concatenate([
                self.lower[agent], self.upper[agent],
                np.array([precision, coverage, episode_phase], dtype=np.float32),
            ]).astype(np.float32)
        mode_bit = 1.0 if self.x_star_unit.get(agent) is not None else 0.0
        return np.concatenate([
            np.asarray(self.a[agent], dtype=np.float32),
            np.asarray(self.b[agent], dtype=np.float32),
            np.asarray(self.q_star[agent], dtype=np.float32),
            np.array([precision, coverage, mode_bit], dtype=np.float32),
        ]).astype(np.float32)

    def _apply_quantile_action(self, agent: str, action: np.ndarray) -> None:
        self.a[agent], self.b[agent] = qmdp.apply_leave_corner_action(
            self.a[agent], self.b[agent], action,
            eta=self.max_quantile_step,
            leave_threshold=self.leave_threshold,
            max_new_constraints=self.max_new_constraints_per_step,
            min_quantile_width=self.min_quantile_width,
            eps=self.quantile_eps,
        )
        if self.x_star_unit.get(agent) is not None:
            self.a[agent], self.b[agent] = qmdp.clip_quantiles_around_qstar(
                self.a[agent], self.b[agent], self.q_star[agent],
                self._constrained_mask(agent), self.quantile_eps,
            )
        self._pin_constrained_categoricals(agent)
        self._sync_unit_bounds_from_quantiles(agent)

    def _conditional_precision_metrics(self, agent: str, coverage: float,
                                       coverage_marginal: float, n_class_in_box: int,
                                       n_class_samples: int, data_source: str,
                                       covered: np.ndarray) -> tuple:
        """Anchors' D(z|A) with frozen (idx, U) so Phi(s')-Phi(s) is CRN-coupled."""
        if self._crn_idx.get(agent) is None or self._crn_U.get(agent) is None:
            self._draw_crn(agent)
        cls = self._get_class_for_agent(agent)
        if cls is not None and int(cls) not in self._class_centroid_unit:
            self._class_centroid_quantiles(agent)
        z_unit = qmdp.crn_perturb(
            self.X_unit, self._crn_idx[agent], self._crn_U[agent],
            self.lower[agent], self.upper[agent], self._constrained_mask(agent),
            self._cdfs(agent)["v_all"],
            x_star_unit=self.x_star_unit.get(agent),
            class_mode_values=self._class_centroid_unit.get(int(cls)) if cls is not None else None,
        )
        z_std = self._unit_to_std(z_unit)
        if hasattr(self.classifier, "eval"):
            self.classifier.eval()
        with torch.no_grad():
            probs = predict_proba_torch(
                self.classifier, torch.from_numpy(z_std.astype(np.float32)).to(self.device)
            ).cpu().numpy()
        self.n_blackbox_queries += int(z_std.shape[0])
        preds = probs.argmax(axis=1)
        orig = self.original_predictions.get(agent)
        if self.x_star_unit.get(agent) is not None and orig is not None:
            hard_precision = float((preds == int(orig)).mean())
            avg_prob = float(probs[:, int(orig)].mean())
        else:
            tgt = int(cls) if cls is not None else 0
            hard_precision = float((preds == tgt).mean())
            avg_prob = float(probs[:, tgt].mean())
        return hard_precision, coverage, {
            "hard_precision": hard_precision,
            "precision_proxy": self.precision_blend_lambda * hard_precision
                               + (1.0 - self.precision_blend_lambda) * avg_prob,
            "purity": float("nan"),
            "avg_prob": avg_prob,
            "n_points": int(z_std.shape[0]),
            "n_covered": int(covered.size),
            "sampler": "conditional_crn",
            "target_class_fraction": hard_precision,
            "data_source": data_source,
            "cov_real": float(coverage),
            "coverage_marginal": float(coverage_marginal),
            "n_class_in_box": int(n_class_in_box),
            "n_class_samples": int(n_class_samples),
        }

    def _get_class_for_agent(self, agent: str) -> Optional[int]:
        if agent in self.agent_to_class:
            return self.agent_to_class[agent]
        
        try:
            parts = agent.split("_")
            for p in parts[1:]:
                if p.isdigit():
                    cls = int(p)
                    self.agent_to_class[agent] = cls
                    self.class_to_agents[cls].append(agent)
                    return cls
        except Exception:
            pass
        
        return None

    def _get_class_centroid(self, agent: str) -> Optional[np.ndarray]:
        """
        Get centroid for an agent, ensuring different agents per class get different centroids.
        
        When agents_per_class > 1, each agent should get a different centroid/instance to ensure
        policy diversity. This is achieved by:
        1. Using agent index to deterministically assign centroids/instances
        2. When centroids/instances are available, cycling through them based on agent index
        3. When falling back to class data, sampling different instances for each agent
        """
        target_class = self._get_class_for_agent(agent)
        if target_class is None:
            return None

        # Priority 0: Explicit start point set by inference for diversified
        # class-based rollouts. Inference snaps these to class data when needed,
        # so use the point as-is.
        if self.class_init_point is not None:
            return np.array(self.class_init_point, dtype=np.float32)

        # Extract agent index from agent name (e.g., "agent_0_1" -> 1, "agent_0" -> 0)
        agent_idx = 0
        if self.agents_per_class > 1 and "_" in agent:
            parts = agent.split("_")
            if len(parts) >= 3 and parts[2].isdigit():
                agent_idx = int(parts[2])
        
        # Use precomputed cluster centroids if its class level
        if self.cluster_centroids_per_class is not None:
            if target_class in self.cluster_centroids_per_class:
                centroids = self.cluster_centroids_per_class[target_class]
                if len(centroids) > 0:
                    # Assign centroids deterministically based on agent index to ensure diversity
                    # Cycle through available centroids if we have more agents than centroids
                    centroid_idx = agent_idx % len(centroids)
                    centroid = np.array(centroids[centroid_idx], dtype=np.float32)
                    
                    # CRITICAL FIX: For scattered data, centroids might be mean centroids (not actual data points)
                    # Check if centroid is close to any actual data point. If not, use the nearest data point instead.
                    X_data, _, y_data, _ = self._active_data()
                    class_mask = (y_data == target_class)
                    
                    if class_mask.sum() > 0:
                        return self._snap_to_nearest_class_point(centroid, X_data[class_mask])
                    return centroid
        
        # Use fixed instances if its instance level
        if self.fixed_instances_per_class is not None:
            if target_class in self.fixed_instances_per_class:
                instances = self.fixed_instances_per_class[target_class]
                if len(instances) > 0:
                    # Assign instances deterministically based on agent index to ensure diversity
                    # Cycle through available instances if we have more agents than instances
                    instance_idx = agent_idx % len(instances)
                    return np.array(instances[instance_idx], dtype=np.float32)
        
        # Fallback: Sample different instances from class data for each agent
        # This ensures different agents get different starting points even without precomputed centroids
        X_data, _, y_data, _ = self._active_data()
        
        class_mask = (y_data == target_class)
        if class_mask.sum() == 0:
            logger.warning(f"No instances found for class {target_class} to compute centroid")
            return None
        
        class_data = X_data[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # If we have enough instances, assign different instances to different agents
        # Otherwise, fall back to mean centroid (but this should be rare)
        if len(class_data) >= self.agents_per_class:
            # Assign instances to agents deterministically to ensure diversity
            # Each agent gets a different subset of instances
            instances_per_agent = len(class_data) // self.agents_per_class
            start_idx = agent_idx * instances_per_agent
            end_idx = start_idx + instances_per_agent if agent_idx < self.agents_per_class - 1 else len(class_data)
            agent_instances = class_data[start_idx:end_idx]
            
            # Randomly sample from agent's assigned subset for exploration across episodes
            # This ensures each agent always gets a different instance from other agents
            # while still having randomness for exploration
            if len(agent_instances) > 0:
                instance_idx = self.rng.integers(0, len(agent_instances))
                centroid = agent_instances[instance_idx].astype(np.float32)
            else:
                # Fallback (shouldn't happen)
                centroid = class_data[agent_idx % len(class_data)].astype(np.float32)
        else:
            # Not enough instances: use mean centroid (fallback)
            # Log warning if this happens with multiple agents per class
            if self.agents_per_class > 1:
                logger.warning(
                    f"Class {target_class} has only {len(class_data)} instances but {self.agents_per_class} agents. "
                    f"All agents will use the same mean centroid. Consider using cluster_centroids_per_class or "
                    f"fixed_instances_per_class to ensure diversity."
                )
            mean_centroid = np.mean(class_data, axis=0).astype(np.float32)
            centroid = self._snap_to_nearest_class_point(mean_centroid, class_data)
        
        return centroid
    
    # Initialize the box from the centroid
    def _compute_box_from_centroid(self, agent: str, centroid: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        target_class = self._get_class_for_agent(agent)
        if target_class is None:
            return None
        
        # Get class data
        X_data, _, y_data, _ = self._active_data()
        
        class_mask = (y_data == target_class)
        if class_mask.sum() == 0:
            return None
        
        class_data = X_data[class_mask]
        
        n_neighbors = self._init_n_neighbors(len(class_data))
        
        # Compute distances to centroid
        distances = np.linalg.norm(class_data - centroid, axis=1)
        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_points = class_data[nearest_indices]
        
        # Compute box bounds from nearest points
        padding = 0.05
        lower = np.maximum(0.0, nearest_points.min(axis=0) - padding)
        upper = np.minimum(1.0, nearest_points.max(axis=0) + padding)
        
        # Ensure minimum width
        widths = upper - lower
        min_width_mask = widths < self.min_width
        if min_width_mask.any():
            # For features with width < min_width, center the box on the centroid
            for f in np.where(min_width_mask)[0]:
                half_width = self.min_width / 2.0
                lower[f] = np.clip(centroid[f] - half_width, 0.0, 1.0 - self.min_width)
                upper[f] = np.clip(lower[f] + self.min_width, lower[f] + self.min_width, 1.0)
                # If upper was clipped to 1.0, adjust lower to maintain minimum width
                if upper[f] - lower[f] < self.min_width:
                    lower[f] = np.clip(1.0 - self.min_width, 0.0, 1.0 - self.min_width)
                    upper[f] = 1.0
        
        return lower.astype(np.float32), upper.astype(np.float32)

    # Reward calculation: Compute the current metrics for the agent
    def _current_metrics(self, agent: str) -> tuple:
        target_class = self._get_class_for_agent(agent)
        if target_class is None:
            raise KeyError(f"Could not determine class for agent '{agent}'")
        
        # C-10: metrics on the active split (train during training; val at
        # inference/selection; never D_test inside the env — D_test is for
        # revision.evaluate only).
        X_data_unit, X_data_std, y_data, data_source = self._active_data()
        
        mask = self._mask_in_box(agent)
        covered = np.where(mask)[0]
        
        # Ensure mask and y_data have the same length (safety check)
        if len(mask) != len(y_data):
            logger.error(
                f"Agent {agent}: Mask length ({len(mask)}) != y_data length ({len(y_data)}). "
                f"This indicates a data mismatch. eval_on_test_data={self.eval_on_test_data}"
            )
            # Try to fix: use the data source that matches the mask
            expected_len = (
                len(self.y_test) if self.eval_on_test_data and self.y_test is not None
                else len(self.y)
            )
            if len(mask) == expected_len:
                # Mask seems to be from test, but we're using train y_data (or vice versa)
                y_data = self.y_test if self.eval_on_test_data else self.y
                logger.warning(f"  Corrected y_data to match mask length")
        
        # Coverage is always class-conditional P(x in box | y = target_class),
        # including instance-based episodes. Marginal P(x in box) is logged only.
        # Precision stays instance prediction-matching when x_star is set.
        is_instance_based = self.x_star_unit.get(agent) is not None
        
        if self.mode == "inference" and not is_instance_based:
            logger.debug(
                f"Agent {agent}: Instance-based mode not detected during inference (x_star_unit is None). "
                f"This may indicate x_star_unit was not set before reset() or was cleared during reset()."
            )
        
        if is_instance_based and agent not in self.original_predictions:
            x_star = self.x_star_unit[agent]
            x_star_std = self._unit_to_std(x_star.reshape(1, -1))[0]
            
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()
            
            with torch.no_grad():
                instance_tensor = torch.from_numpy(x_star_std.astype(np.float32)).unsqueeze(0).to(self.device)
                probs = predict_proba_torch(self.classifier, instance_tensor).cpu().numpy()[0]
                self.original_predictions[agent] = int(np.argmax(probs))
                logger.debug(f"Agent {agent}: Computed original prediction {self.original_predictions[agent]} for instance-based anchor")
        
        coverage, coverage_marginal, n_class_in_box, n_class_samples = (
            self._class_conditional_coverage(mask, y_data, target_class)
        )
        if n_class_samples == 0:
            logger.warning(
                f"Agent {agent}: No samples found for target class {target_class} in {data_source} data. "
                f"Total samples: {len(y_data)}, Classes present: {sorted(np.unique(y_data).tolist())}"
            )
        elif coverage == 0.0 and covered.size > 0:
            logger.debug(
                f"Agent {agent}: Box covers {int(mask.sum())} total samples but 0 target class samples. "
                f"Target class: {target_class}, Class samples in dataset: {n_class_samples}"
            )
        
        if covered.size == 0 and self.precision_estimator != "conditional" and not (
            self.use_perturbation and self.perturbation_mode in ["uniform", "adaptive"]
        ):
            logger.debug(f"Agent {agent}: No covered points (coverage={coverage:.4f}) and perturbation not enabled for uniform/adaptive modes - returning precision=0")
            return 0.0, coverage, {
                "hard_precision": 0.0, 
                "avg_prob": 0.0, 
                "n_points": 0, 
                "sampler": "none",
                "data_source": data_source,
                "coverage_marginal": float(coverage_marginal),
                "n_class_in_box": int(n_class_in_box),
                "n_class_samples": int(n_class_samples),
            }

        # Row indices into the active dataset when evaluation uses real rows
        # (empirical/bootstrap paths). Lets us look up cached classifier
        # probabilities instead of re-running the classifier every step.
        if self.precision_estimator == "conditional":
            return self._conditional_precision_metrics(
                agent, coverage, coverage_marginal, n_class_in_box, n_class_samples,
                data_source, covered,
            )

        eval_row_idx = None

        if not self.use_perturbation:
            X_eval = X_data_std[covered]
            y_eval = y_data[covered]
            eval_row_idx = covered
            n_points = int(X_eval.shape[0])
            sampler_note = f"empirical_{data_source}"
            logger.debug(f"Agent {agent}: Perturbation disabled - using {n_points} empirical points from {data_source} data")
        else:
            if self.perturbation_mode == "bootstrap":
                if covered.size == 0:
                    data_source = "test" if self.eval_on_test_data else "training"
                    logger.debug(f"Agent {agent}: Bootstrap mode - no covered points, returning precision=0, coverage={coverage:.4f}")
                    return 0.0, coverage, {
                        "hard_precision": 0.0, 
                        "avg_prob": 0.0, 
                        "n_points": 0, 
                        "sampler": "none",
                        "data_source": data_source,
                        "coverage_marginal": float(coverage_marginal),
                        "n_class_in_box": int(n_class_in_box),
                        "n_class_samples": int(n_class_samples),
                    }
                n_samp = min(self.n_perturb, max(1, covered.size))
                idx = self.rng.choice(covered, size=n_samp, replace=True)
                X_eval = X_data_std[idx]
                y_eval = y_data[idx]
                eval_row_idx = idx
                n_points = int(n_samp)
                sampler_note = f"bootstrap_{data_source}"
                logger.debug(f"Agent {agent}: Bootstrap mode - sampled {n_points} points from {covered.size} covered points (n_perturb={self.n_perturb})")
            elif self.perturbation_mode == "uniform":
                n_samp = self.n_perturb
                U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                for j in range(self.n_features):
                    low, up = float(self.lower[agent][j]), float(self.upper[agent][j])
                    width = max(up - low, self.min_width)
                    mid = 0.5 * (low + up)
                    low = max(0.0, mid - width / 2.0)
                    up = min(1.0, mid + width / 2.0)
                    U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                
                # CRITICAL FIX: For instance-based anchors, always include the original instance
                # This ensures precision calculation includes at least one point (the instance itself)
                # that matches the original prediction, preventing precision from being incorrectly 0.0
                if is_instance_based and self.x_star_unit.get(agent) is not None:
                    x_star_unit = np.array(self.x_star_unit[agent], dtype=np.float32).reshape(1, -1)
                    # Prepend the original instance to the uniform samples
                    U = np.vstack([x_star_unit, U])
                    n_samp = n_samp + 1
                
                X_eval = self._unit_to_std(U)
                y_eval = None
                n_points = int(n_samp)
                sampler_note = f"uniform_{data_source}"
                logger.debug(f"Agent {agent}: Uniform mode - generating {n_points} synthetic samples (coverage={coverage:.4f} from {covered.size} real points)")
            elif self.perturbation_mode == "adaptive":
                # Adaptive mode: use bootstrap when enough covered points, otherwise use uniform
                min_points_for_bootstrap = max(1, int(0.1 * self.n_perturb))
                
                if covered.size >= min_points_for_bootstrap:
                    # Use bootstrap sampling from real data points
                    n_samp = min(self.n_perturb, covered.size)
                    idx = self.rng.choice(covered, size=n_samp, replace=True)
                    X_eval = X_data_std[idx]
                    y_eval = y_data[idx]
                    eval_row_idx = idx
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_bootstrap_{data_source}"
                    logger.debug(f"Agent {agent}: Adaptive mode - using bootstrap sampling: {n_points} points from {covered.size} covered points "
                               f"(threshold={min_points_for_bootstrap}, coverage={coverage:.4f})")
                else:
                    # Fall back to uniform sampling when not enough covered points
                    n_samp = self.n_perturb
                    U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                    for j in range(self.n_features):
                        low, up = float(self.lower[agent][j]), float(self.upper[agent][j])
                        width = max(up - low, self.min_width)
                        mid = 0.5 * (low + up)
                        low = max(0.0, mid - width / 2.0)
                        up = min(1.0, mid + width / 2.0)
                        U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                    
                    # CRITICAL FIX: For instance-based anchors, always include the original instance
                    # This ensures precision calculation includes at least one point (the instance itself)
                    # that matches the original prediction, preventing precision from being incorrectly 0.0
                    if is_instance_based and self.x_star_unit.get(agent) is not None:
                        x_star_unit = np.array(self.x_star_unit[agent], dtype=np.float32).reshape(1, -1)
                        # Prepend the original instance to the uniform samples
                        U = np.vstack([x_star_unit, U])
                        n_samp = n_samp + 1
                    
                    X_eval = self._unit_to_std(U)
                    y_eval = None
                    n_points = int(n_samp)
                    sampler_note = f"adaptive_uniform_{data_source}"
                    # Only warn once per episode to reduce log spam
                    # if not self._adaptive_uniform_warned_this_episode[agent]:
                    #     logger.warning(f"Agent {agent}: Adaptive mode - switching to uniform sampling! "
                    #                  f"covered.size={covered.size} < threshold={min_points_for_bootstrap}, "
                    #                  f"coverage={coverage:.4f}. Precision will be calculated from synthetic samples, "
                    #                  f"but coverage is from real data points. This may cause precision-coverage mismatch. "
                    #                  f"(This warning will appear once per episode)")
                    #     self._adaptive_uniform_warned_this_episode[agent] = True
                    # else:
                    #     logger.debug(f"Agent {agent}: Adaptive mode using uniform sampling (covered.size={covered.size} < threshold={min_points_for_bootstrap}, coverage={coverage:.4f})")
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap', 'uniform', or 'adaptive'.")

        if eval_row_idx is not None:
            # Dataset rows: look up cached probabilities (computed once per env)
            probs = self._get_cached_probs(self._active_split())[eval_row_idx]
        else:
            # Fresh uniform samples: must run the classifier
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()

            with torch.no_grad():
                inputs = torch.from_numpy(X_eval).float().to(self.device)
                probs = predict_proba_torch(self.classifier, inputs).cpu().numpy()
            self.n_blackbox_queries += int(X_eval.shape[0])

        preds = probs.argmax(axis=1)
        positive_idx = (preds == target_class)
        
        # Precision calculation depends on mode:
        # - Instance-based: P(prediction matches original instance | anchor conditions hold) - matches original Anchor paper
        # - Class-based: P(y = target_class | x in box) - measures class correctness
        if is_instance_based:
            # Instance-based mode: Match original Anchor paper definition
            # Precision = fraction of samples where prediction matches original instance's prediction
            if agent in self.original_predictions:
                original_pred = self.original_predictions[agent]
                matches_original = (preds == original_pred)
                hard_precision = float(matches_original.mean())
                logger.debug(f"Agent {agent}: Instance-based precision (prediction matching): {hard_precision:.4f} "
                           f"(original prediction: {original_pred}, target class: {target_class})")
            else:
                # Fallback: if original prediction not stored, use class-based precision
                logger.warning(f"Agent {agent}: Original prediction not found for instance-based anchor, "
                             f"falling back to class-based precision calculation")
                if y_eval is None:
                    hard_precision = float(positive_idx.mean())
                else:
                    hard_precision = float((preds == target_class).mean())
            purity = float((y_eval == (self.original_predictions.get(agent, target_class))).mean()) if y_eval is not None else float("nan")
        else:
            # C-09: class-based PRIMARY is model fidelity P(f_hat(x) = c | x in B).
            # Label purity is logged as a secondary diagnostic and is NOT the
            # optimization target. Training previously used y here, so class-mode
            # policies must be retrained after this change.
            hard_precision = float(positive_idx.mean())
            using_synthetic_samples = y_eval is None
            purity = float((y_eval == target_class).mean()) if y_eval is not None else float("nan")

        # CRITICAL FIX: For instance-based mode, use original prediction probability instead of target_class
        # This ensures reward signal matches single-agent behavior and original Anchor paper
        if is_instance_based and agent in self.original_predictions:
            original_pred = self.original_predictions[agent]
            # Use probability of original prediction class, not target_class
            avg_prob = float(probs[:, original_pred].mean())
            logger.debug(f"Agent {agent}: Instance-based precision proxy using original prediction class {original_pred} probability")
        else:
            # Class-based mode: use target_class probability
            avg_prob = float(probs[:, target_class].mean())
        
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        target_class_fraction = hard_precision  # Same as hard_precision when y_eval is available
        
        # Control signal is hard Fid. Softmax blend made τ_P unreachable on
        # low-confidence but accurate DNNs (wine p̂_max≈0.37 → proxy 0.68).
        return hard_precision, coverage, {
            "hard_precision": hard_precision,
            "precision_proxy": precision_proxy,
            "purity": purity,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "n_covered": int(covered.size),
            "sampler": sampler_note,
            "target_class_fraction": target_class_fraction,
            "data_source": data_source,
            "cov_real": float(coverage),
            "coverage_marginal": float(coverage_marginal),
            "n_class_in_box": int(n_class_in_box),
            "n_class_samples": int(n_class_samples),
        }

    # SS: Added as part of the termination reason counters
    def _reset_termination_counters(self):
        agents_list = getattr(self, 'agents', None) or getattr(self, 'possible_agents', [])
        if not agents_list:
            self.termination_reason_counts = {}
            self.termination_reason_enabled = {}
            return
        
        self.termination_reason_counts = {agent: {
            "both_targets_met": 0,
            "excellent_precision": 0,
            "high_precision_reasonable_coverage": 0,
            "both_reasonably_close": 0
        } for agent in agents_list}
        self.termination_reason_enabled = {agent: {
            "both_targets_met": True,
            "excellent_precision": True,
            "high_precision_reasonable_coverage": True,
            "both_reasonably_close": True
        } for agent in agents_list}
        if getattr(self, "strict_target_termination", True) or getattr(self, "mode", "training") == "inference":
            for agent in agents_list:
                self.termination_reason_enabled[agent]["excellent_precision"] = False
                self.termination_reason_enabled[agent]["high_precision_reasonable_coverage"] = False
        if getattr(self, "strict_target_termination", True):
            for agent in agents_list:
                self.termination_reason_enabled[agent]["both_reasonably_close"] = False
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        if hasattr(self.classifier, 'eval'):
            self.classifier.eval()
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Rebuild termination-enabled flags against the CURRENT mode. __init__ runs
        # this once with mode="training"; inference sets env.mode="inference" only
        # afterwards, so without this call the inference-mode disabling of lenient
        # conditions (excellent_precision / high_precision_reasonable_coverage)
        # would never take effect. Re-running it here honors the mode set before
        # the rollout, matching the "reset in reset()" intent in inference.py.
        self._reset_termination_counters()
        self._coverage_at_reset = {}
        self._coverage_gain_eps = {}
        self._union_coverage_at_reset = {}
        self._rt_shaping = defaultdict(float)
        self._rt_overlap = defaultdict(float)
        self._rt_drift = defaultdict(float)
        self._rt_anchor_drift = defaultdict(float)
        self._rt_cov_floor = defaultdict(float)
        self._rt_terminal = defaultdict(float)
        self._rt_total = defaultdict(float)

        observations = {}
        infos = {}
        
        # CRITICAL: Preserve x_star_unit during inference/evaluation mode
        # During inference, x_star_unit is set externally (e.g., by inference code) to indicate instance-based mode
        # We must preserve it so precision stays instance prediction-matching.
        x_star_unit_preserved = {}
        if self.mode in ["inference", "evaluation"]:
            for agent in self.possible_agents:
                if agent in self.x_star_unit and self.x_star_unit[agent] is not None:
                    x_star_unit_preserved[agent] = self.x_star_unit[agent].copy() if isinstance(self.x_star_unit[agent], np.ndarray) else self.x_star_unit[agent]
        
        # Mixed initialization during training: randomly choose between instance-based and centroid-based
        # For each agent, decide whether to use instance-based or centroid-based
        for agent in self.agents:
            use_instance_based = False
            if self.mode == "training" and self.training_instances_per_class is not None:
                # Get class-specific ratio if available, otherwise use base ratio
                target_class = self._get_class_for_agent(agent)
                if (self.training_instance_ratios_per_class is not None and 
                    target_class is not None and 
                    target_class in self.training_instance_ratios_per_class):
                    class_ratio = self.training_instance_ratios_per_class[target_class]
                else:
                    class_ratio = self.training_instance_ratio
                
                # Randomly decide whether to use instance-based or centroid-based for this agent
                use_instance_based = class_ratio > 0.0 and self.rng.random() < class_ratio
                
                if use_instance_based:
                    target_class = self._get_class_for_agent(agent)
                    if target_class is not None and target_class in self.training_instances_per_class:
                        # Instance-based: randomly select an instance from training instances
                        instances = self.training_instances_per_class[target_class]
                        if len(instances) > 0:
                            # Extract agent index for diversity (different agents get different instances)
                            agent_idx = 0
                            if self.agents_per_class > 1 and "_" in agent:
                                parts = agent.split("_")
                                if len(parts) >= 3 and parts[2].isdigit():
                                    agent_idx = int(parts[2])
                            
                            # Cycle through instances based on agent index, then add randomness
                            instance_idx = (agent_idx + self.rng.integers(0, len(instances))) % len(instances)
                            instance = np.array(instances[instance_idx], dtype=np.float32)
                            # CRITICAL: Reset original_prediction when selecting a new instance
                            # This ensures we recompute it for the new instance, not reuse from previous episode
                            if agent in self.original_predictions:
                                del self.original_predictions[agent]
                            self.x_star_unit[agent] = instance
                            ratio_info = f" (ratio: {class_ratio:.1%})" if target_class is not None else ""
                            logger.debug(f"Training: Agent {agent} using instance-based initialization (instance {instance_idx}/{len(instances)}{ratio_info})")
                        else:
                            use_instance_based = False  # Fall back to centroid-based if no instances
                    else:
                        use_instance_based = False  # Fall back to centroid-based if class not found
                else:
                    # Centroid-based: clear x_star_unit for this agent
                    if agent in self.x_star_unit:
                        del self.x_star_unit[agent]
                    # Also clear original prediction when switching to class-based
                    if agent in self.original_predictions:
                        del self.original_predictions[agent]
            
            # CRITICAL: Restore x_star_unit during inference/evaluation mode if it was set externally
            # This ensures instance-based mode is preserved and correct coverage (overall) is used for termination
            if self.mode in ["inference", "evaluation"] and agent in x_star_unit_preserved:
                self.x_star_unit[agent] = x_star_unit_preserved[agent].copy() if isinstance(x_star_unit_preserved[agent], np.ndarray) else x_star_unit_preserved[agent]
                logger.debug(f"Agent {agent}: Preserved x_star_unit for instance-based mode during {self.mode}")
            
            if self._uses_quantile_mdp():
                # Anchors step 1: the EMPTY rule. All dims at the corner (a=0, b=1),
                # so the box is the full space and k = 0. Predicates are then added
                # by leaving the corner, and coverage falls from 1.0 -- the reverse
                # of the hull path, which had to expand a singleton and got a flat
                # Phi for ~24 of 30 dims.
                self.a[agent] = np.zeros(self.n_features, dtype=np.float64)
                self.b[agent] = np.ones(self.n_features, dtype=np.float64)
                x_star = self.x_star_unit.get(agent)
                if x_star is not None:
                    x_star = np.asarray(x_star, dtype=np.float32).reshape(-1)
                    self.q_star[agent] = self._values_to_q(agent, x_star)
                    if agent not in self.original_predictions:
                        x_star_std = self._unit_to_std(x_star.reshape(1, -1))[0]
                        if hasattr(self.classifier, "eval"):
                            self.classifier.eval()
                        with torch.no_grad():
                            probs = predict_proba_torch(
                                self.classifier,
                                torch.from_numpy(x_star_std.astype(np.float32)).unsqueeze(0).to(self.device),
                            ).cpu().numpy()[0]
                            self.original_predictions[agent] = int(np.argmax(probs))
                else:
                    if self.class_init_point is not None:
                        pt = np.asarray(self.class_init_point, dtype=np.float32).reshape(-1)
                        cls = self._get_class_for_agent(agent)
                        if cls is not None:
                            self._class_centroid_unit[int(cls)] = pt
                        self.q_star[agent] = self._values_to_q(agent, pt)
                    else:
                        self.q_star[agent] = self._class_centroid_quantiles(agent)
                self._maybe_reset_diversity_landing(agent)
                self._pin_constrained_categoricals(agent)
                self._sync_unit_bounds_from_quantiles(agent)
                self._draw_crn(agent)
            elif self.x_star_unit.get(agent) is not None:
                # Get target_class for this agent (needed for validation and logging)
                target_class = self._get_class_for_agent(agent)
                
                centroid = self.x_star_unit[agent]
                box_bounds = self._compute_box_from_centroid(agent, centroid)
                if box_bounds is not None:
                    self.lower[agent], self.upper[agent] = box_bounds
                    self._include_point_in_box(agent, centroid)
                else:
                    self.lower[agent], self.upper[agent] = self._window_box_around(centroid)
                
                # Compute and store original prediction for instance-based anchors (matches original Anchor paper)
                # This is used for precision calculation: P(prediction matches original | anchor conditions hold)
                if agent not in self.original_predictions:
                    # Convert instance from unit space to standardized space for prediction
                    x_star_std = self._unit_to_std(centroid.reshape(1, -1))[0]
                    
                    # Get prediction for original instance
                    if hasattr(self.classifier, 'eval'):
                        self.classifier.eval()
                    if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                        self.classifier.model.eval()
                    
                    with torch.no_grad():
                        instance_tensor = torch.from_numpy(x_star_std.astype(np.float32)).unsqueeze(0).to(self.device)
                        probs = predict_proba_torch(self.classifier, instance_tensor).cpu().numpy()[0]
                        original_pred = int(np.argmax(probs))
                        self.original_predictions[agent] = original_pred
                        # CRITICAL VALIDATION: Verify original_prediction matches target_class (if target_class is available)
                        # This should be true after filtering instances by prediction, but check as safety
                        if target_class is not None and original_pred != target_class:
                            logger.warning(
                                f"Agent {agent}: original_prediction ({original_pred}) != target_class ({target_class})! "
                                f"This may cause precision calculation issues. "
                                f"Instance should have been filtered during training instance selection."
                            )
                        logger.debug(f"Agent {agent}: Stored original prediction {original_pred} for instance-based anchor (target_class={target_class})")
            # Use class centroid if enabled
            elif self.use_class_centroids:
                centroid = self._get_class_centroid(agent)
                if centroid is not None:
                    box_bounds = self._compute_box_from_centroid(agent, centroid)
                    if box_bounds is not None:
                        self.lower[agent], self.upper[agent] = box_bounds
                        self._include_point_in_box(agent, centroid)
                    else:
                        self.lower[agent], self.upper[agent] = self._window_box_around(centroid)
                else:
                    self.lower[agent] = np.zeros(self.n_features, dtype=np.float32)
                    self.upper[agent] = np.ones(self.n_features, dtype=np.float32)
            # Fallback: Full space initialization
            else:
                self.lower[agent] = np.zeros(self.n_features, dtype=np.float32)
                self.upper[agent] = np.ones(self.n_features, dtype=np.float32)

            # Pin categoricals before the first metric computation (not after step 1).
            self._freeze_categorical_bounds(agent)
            
            self.prev_lower[agent] = self.lower[agent].copy()
            self.prev_upper[agent] = self.upper[agent].copy()
            self.box_history[agent] = [(self.lower[agent].copy(), self.upper[agent].copy())]
            self.coverage_floor_hits[agent] = 0
            # Reset stabilization counter
            self._stable_counts[agent] = 0
            # Reset warning flags for this episode
            self._adaptive_uniform_warned_this_episode[agent] = False
            self._precision_coverage_mismatch_warned_this_episode[agent] = False
            
            precision, coverage, initial_details = self._current_metrics(agent)
            # Seed the prev-metrics cache for the first step of the episode
            self._last_step_metrics[agent] = (precision, coverage, initial_details)
            n_class = int(initial_details.get("n_class_samples") or 0)
            if n_class <= 0:
                cls = self._get_class_for_agent(agent)
                if cls is not None:
                    n_class = int((self._active_data()[2] == cls).sum())
            self._coverage_at_reset[agent] = float(coverage)
            self._coverage_gain_eps[agent] = 1.0 / max(n_class, 1)
            # episode_phase is 0.0 at reset (self.timestep was just set to 0)
            self._precision_at_reset[agent] = float(precision)
            self._best_box[agent] = None
            self._best_box_score[agent] = float("-inf")
            state = self._get_observation(agent, precision, coverage, 0.0)

            observations[agent] = np.array(state, dtype=np.float32)
            # Keep reset infos empty. TorchRL infers the info spec from reset;
            # a short dict here drops step keys (coverage_at_reset is in step info).
            infos[agent] = {}

        # Reset the per-class union potential used by the same-class shared reward
        # so the first step of an episode measures gain from the initial boxes.
        self._prev_class_phi = {}
        # Clear the shared-terminal-bonus latch so each episode can pay it once.
        self._class_union_bonus_paid = {}
        self._union_coverage_at_reset = {}
        if (
            len(self.agents) > 1
            and self.shared_terminal_bonus != 0.0
        ):
            for cls, m in self._compute_class_union_metrics().items():
                self._union_coverage_at_reset[cls] = float(m.get("union_coverage", 0.0))
        # Seed inter-class overlap levels from the initial boxes so the first
        # step charges only the overlap CHANGE the first actions cause.
        self._prev_inter_overlap = {
            agent: self._compute_inter_class_overlap_penalty(agent) for agent in self.agents
        }

        return observations, infos

    # Discrete action: Older version (not used anymore)
    def _apply_action(self, agent: str, action: int):
        f = action // (len(self.directions) * len(self.step_fracs))
        rem = action % (len(self.directions) * len(self.step_fracs))
        d = rem // len(self.step_fracs)
        m = rem % len(self.step_fracs)

        direction = self.directions[d]
        step = float(self.step_fracs[m])
        cur_width = max(1e-6, self.upper[agent][f] - self.lower[agent][f])
        rel_step = step * cur_width

        if direction == "shrink_lower":
            self.lower[agent][f] = min(self.lower[agent][f] + rel_step, self.upper[agent][f] - self.min_width)
        elif direction == "expand_lower":
            self.lower[agent][f] = max(self.lower[agent][f] - rel_step, 0.0)
        elif direction == "shrink_upper":
            self.upper[agent][f] = max(self.upper[agent][f] - rel_step, self.lower[agent][f] + self.min_width)
        elif direction == "expand_upper":
            self.upper[agent][f] = min(self.upper[agent][f] + rel_step, 1.0)

        if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
            mid = 0.5 * (self.upper[agent][f] + self.lower[agent][f])
            self.lower[agent][f] = max(0.0, mid - self.min_width / 2.0)
            self.upper[agent][f] = min(1.0, mid + self.min_width / 2.0)

    # SS: Currently used 
    def _apply_continuous_action(self, agent: str, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        
        lower_deltas = action[:self.n_features]
        upper_deltas = action[self.n_features:]
        
        widths = np.maximum(self.upper[agent] - self.lower[agent], 1e-6)
        max_delta_proportional = self.max_action_scale * widths
        max_delta = np.maximum(max_delta_proportional, self.min_absolute_step)
        
        # Store before state for debugging
        lower_before = self.lower[agent].copy()
        upper_before = self.upper[agent].copy()
        
        lower_changes = lower_deltas * max_delta
        self.lower[agent] = np.clip(self.lower[agent] + lower_changes, 0.0, self.upper[agent] - self.min_width)
        
        upper_changes = upper_deltas * max_delta
        self.upper[agent] = np.clip(self.upper[agent] + upper_changes, self.lower[agent] + self.min_width, 1.0)
        
        # CRITICAL: For instance-based anchors, ensure box always covers x_star_unit FIRST
        # This must happen BEFORE min_width adjustment to prevent the box from moving away
        if agent in self.x_star_unit and self.x_star_unit[agent] is not None:
            x_star = self.x_star_unit[agent]
            if isinstance(x_star, np.ndarray):
                x_star_arr = x_star
            else:
                x_star_arr = np.array(x_star, dtype=np.float32)
            
            # Ensure box covers x_star_unit in all dimensions
            for f in range(self.n_features):
                if x_star_arr[f] < self.lower[agent][f]:
                    # Anchor is below lower bound: expand lower to include it
                    self.lower[agent][f] = max(0.0, x_star_arr[f] - self.min_width / 2.0)
                    # Ensure upper is still above lower + min_width
                    if self.upper[agent][f] < self.lower[agent][f] + self.min_width:
                        self.upper[agent][f] = min(1.0, self.lower[agent][f] + self.min_width)
                elif x_star_arr[f] > self.upper[agent][f]:
                    # Anchor is above upper bound: expand upper to include it
                    self.upper[agent][f] = min(1.0, x_star_arr[f] + self.min_width / 2.0)
                    # Ensure lower is still below upper - min_width
                    if self.lower[agent][f] > self.upper[agent][f] - self.min_width:
                        self.lower[agent][f] = max(0.0, self.upper[agent][f] - self.min_width)
                # Ensure x_star_unit is within bounds (safety check)
                if x_star_arr[f] < self.lower[agent][f] or x_star_arr[f] > self.upper[agent][f]:
                    # If still outside, center box around x_star_unit
                    self.lower[agent][f] = max(0.0, x_star_arr[f] - self.min_width / 2.0)
                    self.upper[agent][f] = min(1.0, self.lower[agent][f] + self.min_width)
                    # If upper was clipped, adjust lower
                    if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
                        self.upper[agent][f] = min(1.0, x_star_arr[f] + self.min_width / 2.0)
                        self.lower[agent][f] = max(0.0, self.upper[agent][f] - self.min_width)
        
        # Now ensure min_width constraint is satisfied (AFTER ensuring x_star_unit is covered)
        for f in range(self.n_features):
            if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
                # If x_star_unit exists, try to center around it while maintaining min_width
                if agent in self.x_star_unit and self.x_star_unit[agent] is not None:
                    x_star = self.x_star_unit[agent]
                    if isinstance(x_star, np.ndarray):
                        x_star_arr = x_star
                    else:
                        x_star_arr = np.array(x_star, dtype=np.float32)
                    # Center around x_star_unit
                    mid = x_star_arr[f]
                else:
                    # No x_star_unit, use box center
                    mid = 0.5 * (self.upper[agent][f] + self.lower[agent][f])
                
                self.lower[agent][f] = max(0.0, mid - self.min_width / 2.0)
                self.upper[agent][f] = min(1.0, mid + self.min_width / 2.0)
                # If upper was clipped, adjust lower
                if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
                    if self.upper[agent][f] >= 1.0:
                        self.lower[agent][f] = max(0.0, 1.0 - self.min_width)
                        self.upper[agent][f] = 1.0
                    elif self.lower[agent][f] <= 0.0:
                        self.lower[agent][f] = 0.0
                        self.upper[agent][f] = min(1.0, self.min_width)
                
                # Final safety check: ensure x_star_unit is still in box
                if agent in self.x_star_unit and self.x_star_unit[agent] is not None:
                    x_star = self.x_star_unit[agent]
                    if isinstance(x_star, np.ndarray):
                        x_star_arr = x_star
                    else:
                        x_star_arr = np.array(x_star, dtype=np.float32)
                    if x_star_arr[f] < self.lower[agent][f]:
                        self.lower[agent][f] = max(0.0, x_star_arr[f] - self.min_width / 2.0)
                        self.upper[agent][f] = min(1.0, self.lower[agent][f] + self.min_width)
                    elif x_star_arr[f] > self.upper[agent][f]:
                        self.upper[agent][f] = min(1.0, x_star_arr[f] + self.min_width / 2.0)
                        self.lower[agent][f] = max(0.0, self.upper[agent][f] - self.min_width)
        
        # SS: Debugging (disable later)
        # Debug: Log if action was applied (only for first call per agent)
        if not hasattr(self, '_action_debug_logged'):
            self._action_debug_logged = set()
        
        if agent not in self._action_debug_logged:
            lower_diff = np.abs(self.lower[agent] - lower_before).max()
            upper_diff = np.abs(self.upper[agent] - upper_before).max()
            logger.debug(f"  _apply_continuous_action for {agent}: lower_diff={lower_diff:.6f}, upper_diff={upper_diff:.6f}, max_delta={max_delta.max():.6f}, action_mean={action.mean():.4f}")
            if lower_diff < 1e-6 and upper_diff < 1e-6:
                logger.warning(f"  ⚠ Action did not change box for {agent}! lower_deltas mean={lower_deltas.mean():.4f}, upper_deltas mean={upper_deltas.mean():.4f}, max_delta={max_delta.max():.6f}")
            self._action_debug_logged.add(agent)
        self._freeze_categorical_bounds(agent)
    
    def step(
        self, 
        actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}
        
        metrics_cache: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
        
        # SS: R_local: We first compute all local rewards (without the shared component),
        # and only after all agents have moved do we compute the R_shared
        # and add it to each agent's reward.
        reward_without_shared: Dict[str, float] = {}
        # Phi(s') per agent, for the absorbing-terminal shaping correction applied
        # after ALL termination paths have been resolved (see below).
        phi_curr_by_agent: Dict[str, float] = {}

        for agent in self.agents:
            if agent not in actions:
                # Agent did not act in this step; keep its box unchanged
                precision, coverage, details = self._current_metrics(agent)
                # Skip action application for agents without actions
                continue
            else:
                # Agent has an action; read it and apply it
                # CRITICAL FIX: Do not compute metrics here - observation will be created with post-action metrics
                # This avoids wasted compute and prevents stochastic variance from duplicate calls
                # No local reward contribution; shared reward will be added later
                reward_without_shared[agent] = 0.0
                terminations[agent] = False
                truncations[agent] = False
                
                # SS: This is part of the metrics callback of BenchMARL. (currently not working BUG: Need to fix this)
                # Initialize infos with placeholder values - will be updated after action with actual metrics
                infos[agent] = {
                    "anchor_precision": 0.0,
                    "anchor_coverage": 0.0,
                    "drift": 0.0,
                    "anchor_drift": 0.0,
                    "js_penalty": 0.0,
                    "coverage_clipped": 0.0,
                    "termination_reason": 0.0,
                    "coverage_floor_hits": float(self.coverage_floor_hits.get(agent, 0)),
                    "coverage_before_revert": 0.0,
                    "coverage_after_revert": 0.0,
                    "precision_gain": 0.0,
                    "coverage_gain": 0.0,
                    "coverage_gain_scaled": 0.0,
                    "precision_gain_component": 0.0,
                    "coverage_gain_component": 0.0,
                    "coverage_bonus": 0.0,
                    "target_class_bonus": 0.0,
                    "overlap_penalty": 0.0,
                    "drift_penalty": 0.0,
                    "anchor_drift_penalty": 0.0,
                    "inter_class_overlap_penalty": 0.0,
                    "same_class_overlap_penalty": 0.0,
                    "coverage_floor_penalty": 0.0,
                    # Shared reward and global coverage are added after all agents have been processed
                    "shared_reward": 0.0,
                    "shared_terminal_bonus": 0.0,
                    "global_coverage": 0.0,
                    "total_reward": 0.0,
                }
            
            # Read and apply action (only reached for agents with actions in the else branch)
            action = actions[agent]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            action = np.array(action, dtype=np.float32)
            
            # Prev metrics: reuse the metrics computed at the end of the previous
            # step()/reset() for this agent. This halves classifier work and makes
            # prev and current come from the same sample draw, so the shaping gain
            # below measures the action's effect rather than resampling noise.
            if agent in self._last_step_metrics:
                prev_precision, prev_coverage, _ = self._last_step_metrics[agent]
            else:
                prev_precision, prev_coverage, _ = self._current_metrics(agent)
            prev_lower = self.lower[agent].copy()
            prev_upper = self.upper[agent].copy()
            if self._uses_quantile_mdp():
                _prev_a = self.a[agent].copy()
                _prev_b = self.b[agent].copy()
                _prev_active = self._constrained_mask(agent)
                _k_prev = int(_prev_active.sum())
            else:
                _prev_a = _prev_b = None
                _prev_active = None
                _k_prev = None
            
            #SS: Apply either continuous or discrete action (currently only continuous is used)
            if isinstance(action, np.ndarray) and action.shape[0] == 2 * self.n_features:
                if self._uses_quantile_mdp():
                    self._apply_quantile_action(agent, np.clip(action, -1.0, 1.0))
                else:
                    self._apply_continuous_action(agent, action)
            else:
                self._apply_action(agent, int(action))
            
            # CRITICAL VALIDATION: Ensure box is valid (lower <= upper in all dimensions)
            # This prevents coverage from being 0.0 due to invalid boxes
            for f in range(self.n_features):
                if self.lower[agent][f] > self.upper[agent][f]:
                    logger.warning(
                        f"  ⚠ Invalid box detected for {agent}: lower[{f}]={self.lower[agent][f]:.6f} > upper[{f}]={self.upper[agent][f]:.6f}. "
                        f"Fixing by centering around x_star_unit if available."
                    )
                    # Fix invalid box
                    if agent in self.x_star_unit and self.x_star_unit[agent] is not None:
                        x_star = self.x_star_unit[agent]
                        if isinstance(x_star, np.ndarray):
                            x_star_arr = x_star
                        else:
                            x_star_arr = np.array(x_star, dtype=np.float32)
                        # Center around x_star_unit
                        mid = x_star_arr[f]
                    else:
                        # No x_star_unit, use midpoint
                        mid = 0.5 * (self.lower[agent][f] + self.upper[agent][f])
                    
                    self.lower[agent][f] = max(0.0, mid - self.min_width / 2.0)
                    self.upper[agent][f] = min(1.0, mid + self.min_width / 2.0)
                    if self.upper[agent][f] - self.lower[agent][f] < self.min_width:
                        if self.upper[agent][f] >= 1.0:
                            self.lower[agent][f] = max(0.0, 1.0 - self.min_width)
                            self.upper[agent][f] = 1.0
                        else:
                            self.lower[agent][f] = 0.0
                            self.upper[agent][f] = min(1.0, self.min_width)
            
            precision, coverage, details = self._current_metrics(agent)
            
            if not np.isfinite(precision):
                precision = 0.0
            if not np.isfinite(coverage):
                coverage = 0.0
            if not np.isfinite(prev_precision):
                prev_precision = 0.0
            if not np.isfinite(prev_coverage):
                prev_coverage = 0.0
            
            coverage_clipped = False
            coverage_before_revert = None
            coverage_after_revert = None
            if coverage < self.min_coverage_floor and self.coverage_floor_mode == "step":
                coverage_before_revert = float(coverage)
                logger.debug(
                    f"  Coverage floor hit for {agent}: coverage={coverage:.6f} < "
                    f"min_coverage_floor={self.min_coverage_floor:.6f}, reverting box bounds"
                )
                self.lower[agent] = prev_lower
                self.upper[agent] = prev_upper
                precision, coverage, details = self._current_metrics(agent)
                if not np.isfinite(precision):
                    precision = 0.0
                if not np.isfinite(coverage):
                    coverage = 0.0
                coverage_after_revert = float(coverage)
                self.coverage_floor_hits[agent] = self.coverage_floor_hits.get(agent, 0) + 1
                coverage_clipped = True
                logger.debug(
                    f"  Box reverted for {agent}: coverage after revert={coverage_after_revert:.6f}"
                )
                # Box revert indicates instability - reset stability counter
                if self.enable_stability_termination:
                    self._stable_counts[agent] = 0

            # -----------------------------
            # Stabilization tracking (for early termination)
            # -----------------------------
            # Only track stability if box was not reverted (coverage_clipped == False)
            if not coverage_clipped:
                try:
                    anchor_drift = float(
                        max(
                            np.abs(self.lower[agent] - prev_lower).max(),
                            np.abs(self.upper[agent] - prev_upper).max(),
                        )
                    )
                except Exception:
                    anchor_drift = 0.0

                dp = float(abs(precision - prev_precision))
                dc = float(abs(coverage - prev_coverage))

                if self.enable_stability_termination:
                    if (
                        dp <= self.stability_precision_tol
                        and dc <= self.stability_coverage_tol
                        and anchor_drift <= self.stability_drift_tol
                    ):
                        self._stable_counts[agent] = int(self._stable_counts.get(agent, 0)) + 1
                    else:
                        self._stable_counts[agent] = 0
            
            # Cache post-action metrics (after possible revert)
            metrics_cache[agent] = (precision, coverage, details)
            
            precision_gain = precision - prev_precision
            coverage_gain = coverage - prev_coverage
            
            if not np.isfinite(precision_gain):
                precision_gain = 0.0
            if not np.isfinite(coverage_gain):
                coverage_gain = 0.0
            
            # Potential-based shaping: reward = Phi(s') - Phi(s) (see _potential).
            # Telescoping makes any A->B->A oscillation net exactly zero. The previous
            # relative-gain normalization (gain / prev value) paid more on the way up
            # than it charged on the way back down, so wiggling a bound forever was the
            # return-maximizing policy. The clips, phase weights, JS proxy, and
            # coverage/target-class bonuses existed to manage that scheme and are
            # retired with it. Mirrors single_agentENV.py for fair comparison.
            agent_target_class = self._get_class_for_agent(agent)
            _k_now = self.n_predicates(agent) if self._uses_quantile_mdp() else None
            _p_reset = self._precision_at_reset.get(agent) if self._uses_quantile_mdp() else None
            phi_prev = self._potential(prev_precision, prev_coverage, agent_target_class,
                                       k=_k_prev, p_reset=_p_reset)
            phi_curr = self._potential(precision, coverage, agent_target_class,
                                       k=_k_now, p_reset=_p_reset)
            # F = gamma*Phi(s') - Phi(s). The undiscounted form is only invariance-
            # preserving at gamma = 1, and MADDPG/MASAC do discount.
            shaping_gain = self.discount * phi_curr - phi_prev
            phi_curr_by_agent[agent] = float(phi_curr)
            coverage_gain_for_reward = coverage_gain  # raw gain, kept for logging

            _curr_active = self._constrained_mask(agent) if self._uses_quantile_mdp() else None
            if self._uses_quantile_mdp() and np.isfinite(coverage) and coverage >= self.min_coverage_floor:
                _k_b = int(_curr_active.sum())
                if self._empty_rule_eligible(agent, precision, _k_b):
                    _sc = _ranking_score(
                        precision, coverage,
                        getattr(self, "ranking_score_formula", "lcb_coverage"),
                        n_covered=int(details.get("n_covered", 0) or 0),
                    )
                    if np.isfinite(_sc) and _sc > self._best_box_score.get(agent, float("-inf")):
                        self._best_box_score[agent] = float(_sc)
                        self._best_box[agent] = (
                            self.lower[agent].copy(), self.upper[agent].copy(),
                            self.a[agent].copy(), self.b[agent].copy(),
                        )
            if self._uses_quantile_mdp():
                _both = _prev_active & _curr_active
                _wq = self.b[agent] - self.a[agent]
                overlap_penalty = self.gamma * float(
                    ((_wq < (2 * self.min_quantile_width)) & _both).mean()
                ) if _both.any() else 0.0
                drift = float(
                    np.abs(self.a[agent] - _prev_a)[_both].sum()
                    + np.abs(self.b[agent] - _prev_b)[_both].sum()
                ) if _both.any() else 0.0
            else:
                widths = self.upper[agent] - self.lower[agent]
                overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())
                drift = float(
                    np.linalg.norm(self.upper[agent] - prev_upper)
                    + np.linalg.norm(self.lower[agent] - prev_lower)
                )
            drift_penalty = self.drift_penalty_weight * drift

            anchor_drift_penalty = self._compute_anchor_drift_penalty(agent, prev_lower, prev_upper)
            # Inter-class overlap is charged AFTER the agent loop as a potential-style
            # delta (change in overlap level), not a per-step level: the level form
            # accumulated 0.15-0.3/step (~30-60 per episode) against a positive
            # signal bounded by ~7, so penalty avoidance dominated the learned policy.
            same_class_overlap_penalty = self._compute_same_class_overlap_penalty(agent)

            # Retired terms (keys kept for logging compatibility): the JS proxy was a
            # third movement penalty; the bonuses belonged to the relative-gain scheme.
            js_penalty = 0.0
            coverage_bonus = 0.0
            target_class_bonus = 0.0

            # When action is reverted (coverage_clipped), reduce penalties significantly
            coverage_floor_penalty = 0.0
            if coverage_clipped:
                # Reduce all penalties since action didn't actually take effect
                # (the inter-class overlap delta needs no reduction: a reverted box
                # leaves the overlap level unchanged, so its delta is already ~0)
                penalty_reduction_factor = 0.1
                overlap_penalty *= penalty_reduction_factor
                anchor_drift_penalty *= penalty_reduction_factor
                same_class_overlap_penalty *= penalty_reduction_factor
                # Give a small negative reward for attempting invalid action
                coverage_floor_penalty = -0.05

            # SS: R_local: Local reward component (inter-class overlap delta added post-loop)
            reward_local = (
                shaping_gain
                - overlap_penalty
                - drift_penalty
                - anchor_drift_penalty
                - same_class_overlap_penalty
                + coverage_floor_penalty )

            if not np.isfinite(reward_local):
                reward_local = 0.0

            reward_without_shared[agent] = float(reward_local)
            self._rt_shaping[agent] += float(shaping_gain)
            self._rt_overlap[agent] += float(overlap_penalty)
            self._rt_drift[agent] += float(drift_penalty)
            self._rt_anchor_drift[agent] += float(anchor_drift_penalty)
            self._rt_cov_floor[agent] += float(coverage_floor_penalty)
            self._rt_total[agent] += float(reward_local)
            
            self.box_history[agent].append((self.lower[agent].copy(), self.upper[agent].copy()))
            self.prev_lower[agent] = prev_lower
            self.prev_upper[agent] = prev_upper
            # Cache final metrics for the next step's prev (common random numbers)
            self._last_step_metrics[agent] = (precision, coverage, details)
            
            # self.timestep is incremented after this agent loop, so the state being
            # observed here belongs to time (timestep + 1).
            episode_phase = min(1.0, float(self.timestep + 1) / float(self.max_cycles))
            state = self._get_observation(agent, precision, coverage, episode_phase)
            
            ## SS: Target change here:
            # Termination uses the class-aware effective target so minority/overlapping
            # classes have reachable conditions (see _get_effective_precision_target).
            agent_precision_target = self._get_effective_precision_target(agent_target_class) \
                if agent_target_class is not None else self.precision_target
            both_targets_met = (
                precision >= agent_precision_target
                and coverage >= self.coverage_target
                and (
                    (self._uses_quantile_mdp()
                     and self.n_predicates(agent) >= 1
                     and self._precision_improved(agent, precision))
                    or ((not self._uses_quantile_mdp())
                        and self._coverage_improved(agent, coverage))
                )
            )
            high_precision_with_reasonable_coverage = (
                precision >= 0.95 * agent_precision_target
                and coverage >= 0.7 * self.coverage_target
            )
            both_reasonably_close = (
                precision >= 0.90 * agent_precision_target
                and coverage >= 0.90 * self.coverage_target
            )
            excellent_precision = (
                precision >= agent_precision_target
                and coverage >= 0.5 * self.coverage_target
            )
            
            # Check if termination reasons are enabled (not overused) for this agent
            both_targets_met_enabled = self.termination_reason_enabled[agent]["both_targets_met"]
            excellent_precision_enabled = self.termination_reason_enabled[agent]["excellent_precision"]
            high_precision_enabled = self.termination_reason_enabled[agent]["high_precision_reasonable_coverage"]
            both_close_enabled = self.termination_reason_enabled[agent]["both_reasonably_close"]
            
            # Only consider conditions that are enabled
            both_targets_met = both_targets_met and both_targets_met_enabled
            excellent_precision = excellent_precision and excellent_precision_enabled
            high_precision_with_reasonable_coverage = high_precision_with_reasonable_coverage and high_precision_enabled
            both_reasonably_close = both_reasonably_close and both_close_enabled
            
            # Validate rule validity before allowing termination:
            # bounds ordered, inside [0, 1], and finite.
            #
            # Whether a ZERO-WIDTH dim is legal depends on the representation:
            #
            #   neighbor_hull -- the box is grown from a point, so lower == upper
            #       means the box collapsed back onto that point and the rule is
            #       degenerate. Reject it.
            #
            #   quantile MDP -- bounds are Q_j(a_j), Q_j(b_j) through the class
            #       empirical CDF. Whenever the band [a_j, b_j) falls between two
            #       tied values, Q_j(a_j) == Q_j(b_j) and the dim becomes an
            #       EQUALITY predicate "f_j = v" -- exactly Anchors' categorical
            #       predicate, and box_mask() is inclusive (utils/metrics.py), so
            #       it carries the real support of every row tied at v. Rejecting
            #       it forces `done = False` forever on any dataset with ties:
            #       uci_credit collapses 5/15 dims at a [0.30, 0.70] band before
            #       the policy narrows anything, iris 4/4 at width 0.02. That
            #       killed `both_targets_met` and all partial terminal credit for
            #       the whole run. Only a genuinely INVERTED bound is invalid here.
            bounds_valid = True
            agent_lower = self.lower[agent]
            agent_upper = self.upper[agent]
            _degenerate = (
                agent_lower > agent_upper
                if self._uses_quantile_mdp()
                else agent_lower >= agent_upper
            )
            if np.any(_degenerate):
                bounds_valid = False
                invalid_features = np.where(_degenerate)[0]
                logger.warning(
                    f"Agent {agent}: Invalid bounds detected: lower > upper for features {invalid_features[:5]}. "
                    f"Preventing termination until bounds are fixed."
                )
            if np.any(agent_lower < 0) or np.any(agent_upper > 1):
                bounds_valid = False
                logger.warning(
                    f"Agent {agent}: Invalid bounds detected: bounds outside [0, 1] range. "
                    f"Preventing termination until bounds are fixed."
                )
            if not np.all(np.isfinite(agent_lower)) or not np.all(np.isfinite(agent_upper)):
                bounds_valid = False
                logger.warning(
                    f"Agent {agent}: Invalid bounds detected: NaN or Inf values in bounds. "
                    f"Preventing termination until bounds are fixed."
                )
            
            # Require a minimum number of steps before termination is allowed, so
            # an agent cannot terminate straight from the initial box. self.timestep
            # is 0 on the first step (incremented post-loop), so +1 = steps taken.
            # Matches SA's `can_terminate = step_count >= min_steps_before_termination`.
            can_terminate = (self.timestep + 1) >= self.min_steps_before_termination

            # Only allow termination if bounds are valid AND minimum steps taken AND targets are met
            done = bool(
                bounds_valid and can_terminate and (
                    both_targets_met
                    or high_precision_with_reasonable_coverage
                    or both_reasonably_close
                    or excellent_precision
                )
            )
            
            # SS: Need some kind of priority order for the termination reasons. (currently not implemented)
            termination_reason = None
            if done:
                # Determine which condition was met (check in priority order)
                if both_targets_met and both_targets_met_enabled:
                    termination_reason = "both_targets_met"
                elif excellent_precision and excellent_precision_enabled:
                    termination_reason = "excellent_precision"
                elif high_precision_with_reasonable_coverage and high_precision_enabled:
                    termination_reason = "high_precision_reasonable_coverage"
                elif both_reasonably_close and both_close_enabled:
                    termination_reason = "both_reasonably_close"
                
                # Track counts for diagnostics only. The old mechanism that permanently
                # DISABLED a termination reason after N uses changed the MDP mid-training
                # while stale transitions sat in the replay buffer — removed. The
                # terminal bonus below makes terminating at targets strictly better
                # than hovering below them.
                if termination_reason:
                    self.termination_reason_counts[agent][termination_reason] += 1

                    # Terminal bonus: pays once, dwarfs anything farmable from
                    # per-step terms in the remaining steps of the episode.
                    reward_without_shared[agent] = float(reward_without_shared.get(agent, 0.0) + self.terminal_bonus)
                    self._rt_terminal[agent] += float(self.terminal_bonus)
                    self._rt_total[agent] += float(self.terminal_bonus)

            # Decompose the shaping gain for logging: precision part is exact, the
            # remainder is the gated-coverage part of the potential.
            precision_gain_component = self.alpha * precision_gain
            coverage_gain_component = shaping_gain - precision_gain_component
            
            termination_reason_code = 0.0
            if termination_reason == "both_targets_met":
                termination_reason_code = 1.0
            elif termination_reason == "excellent_precision":
                termination_reason_code = 2.0
            elif termination_reason == "high_precision_reasonable_coverage":
                termination_reason_code = 3.0
            elif termination_reason == "both_reasonably_close":
                termination_reason_code = 4.0
            
            info: Dict[str, Any] = {
                "anchor_precision": float(precision),
                "anchor_coverage": float(coverage),
                # Layout stamp so consumers never have to infer the observation
                # layout from its length (2n+3 and 3m+3 collide).
                "obs_layout_quantile": float(1.0 if self._uses_quantile_mdp() else 0.0),
                "n_features": float(self.n_features),
                "n_predicates": float(self.n_predicates(agent)),
                "drift": float(drift),
                "anchor_drift": float(anchor_drift_penalty),
                "js_penalty": float(js_penalty),
                "coverage_clipped": float(1.0 if coverage_clipped else 0.0),
                "termination_reason": float(termination_reason_code),
                "coverage_floor_hits": float(self.coverage_floor_hits.get(agent, 0)),
                "coverage_before_revert": float(coverage_before_revert) if coverage_before_revert is not None else 0.0,
                "coverage_after_revert": float(coverage_after_revert) if coverage_after_revert is not None else 0.0,
                "precision_gain": float(precision_gain),
                "coverage_gain": float(coverage_gain),
                "coverage_gain_scaled": float(coverage_gain_for_reward),
                "precision_gain_component": float(precision_gain_component),
                "coverage_gain_component": float(coverage_gain_component),
                "coverage_bonus": float(coverage_bonus),
                "target_class_bonus": float(target_class_bonus),
                "overlap_penalty": float(overlap_penalty),
                "drift_penalty": float(drift_penalty),
                "anchor_drift_penalty": float(anchor_drift_penalty),
                # Placeholder: replaced with the post-loop overlap delta below
                "inter_class_overlap_penalty": 0.0,
                "same_class_overlap_penalty": float(same_class_overlap_penalty),
                "coverage_floor_penalty": float(coverage_floor_penalty),
                "coverage_at_reset": float(self._coverage_at_reset.get(agent, 0.0)),
                "coverage_improved": float(1.0 if self._coverage_improved(agent, coverage) else 0.0),
                # Shared reward will be added later once we have processed all agents
                "shared_reward": 0.0,
                "shared_terminal_bonus": 0.0,
                "total_reward": float(reward_local),
            }
            
            # Attach classifier-level details if available
            for key, value in details.items():
                if value is not None:
                    if isinstance(value, (int, float, np.number)):
                        info[key] = float(value)
                    elif isinstance(value, bool):
                        info[key] = float(1.0 if value else 0.0)
                    elif isinstance(value, np.ndarray):
                        info[key] = float(value.item()) if value.size == 1 else value.tolist()
                    elif isinstance(value, str):
                        # Skip string entries; they are not easily logged numerically
                        continue
            # Provide a clearer alias for target_class_fraction, if available
            if "target_class_fraction" in details:
                try:
                    info["anchor_class_purity"] = float(details["target_class_fraction"])
                except Exception:
                    pass
            
            observations[agent] = np.array(state, dtype=np.float32)
            terminations[agent] = bool(done)
            truncations[agent] = False
            infos[agent] = info
        
        # SS: R_shared: same-class cooperative reward as a class-union potential gain.
        # Each class c has Phi_c = alpha * union_precision(c) + beta * sqrt(union_coverage(c)),
        # and every agent of class c receives w_shared * (Phi_c(s') - Phi_c(s)).
        # This replaces the old global shared reward, which (a) was identical for ALL
        # agents regardless of class — an Ice agent was rewarded when Snow improved —
        # (b) was positive-only and clipped to [0, 1.25] per step, a farmable income
        # stream that dwarfed the local signal, and (c) had no credit assignment.
        # As a potential gain it is oscillation-proof for the same reason the local
        # term is, and the cooperative task it rewards (grow YOUR class's union) is
        # exactly what inference extracts.
        shared_by_class: Dict[int, float] = {}
        class_union_metrics: Dict[int, Dict[str, float]] = {}
        union_terminal_bonus_by_class: Dict[int, float] = {}
        # Compute union metrics when EITHER the smooth shared term or the one-time
        # union-target bonus is active — the bonus is an independent lever and must
        # work even if shared_reward_weight is 0.
        if len(self.agents) > 1 and (
            self.shared_reward_weight != 0.0 or self.shared_terminal_bonus != 0.0
        ):
            class_union_metrics = self._compute_class_union_metrics()
            for cls, m in class_union_metrics.items():
                union_prec = max(0.0, float(m.get("union_precision", 0.0)))
                union_cov = max(0.0, float(m.get("union_coverage", 0.0)))
                # Gate the coverage term by union precision, mirroring _potential:
                # ungated, the shared term paid for raw union-coverage expansion
                # regardless of precision, rewarding box inflation.
                cls_target = self._get_effective_precision_target(cls)
                gate = min(1.0, union_prec / max(cls_target * 0.8, 1e-6))
                if self._uses_quantile_mdp():
                    # Same two corrections as the local potential. Without them the
                    # SHARED term reproduces the degenerate optimum on its own: at
                    # reset every agent is at the corner, so union coverage is 1.0
                    # and Phi_c is near maximal -- the class is paid for doing
                    # nothing, and any predicate a member adds lowers it.
                    _k_cls = sum(self.n_predicates(a) for a in self.class_to_agents.get(cls, []))
                    _cov_ok = 1.0 if _k_cls >= 1 else 0.0
                    phi_class = (self.alpha * min(union_prec, cls_target)
                                 + self.beta * np.sqrt(union_cov) * gate * _cov_ok)
                else:
                    phi_class = self.alpha * union_prec + self.beta * np.sqrt(union_cov) * gate
                prev_phi = self._prev_class_phi.get(cls)
                shared_by_class[cls] = (
                    self.shared_reward_weight * (phi_class - prev_phi) if prev_phi is not None else 0.0
                ) if self.shared_reward_weight != 0.0 else 0.0
                self._prev_class_phi[cls] = phi_class

                # One-time cooperative terminal bonus: the first step this class's
                # union of boxes clears BOTH targets, pay every agent of the class
                # (added per-agent below) and latch so it can't be re-collected this
                # episode. Uses the same effective per-class precision target and the
                # same coverage_target as local termination, so "union hits target"
                # means the same thing for the cooperative objective as for the local.
                if (
                    self.shared_terminal_bonus != 0.0
                    and not self._class_union_bonus_paid.get(cls, False)
                    and union_prec >= cls_target
                    and union_cov >= self.coverage_target
                    and (
                        not self.require_coverage_gain_to_terminate
                        or union_cov
                        > self._union_coverage_at_reset.get(cls, 0.0)
                        + (1.0 / max(int((self._active_data()[2] == cls).sum()), 1))
                    )
                ):
                    union_terminal_bonus_by_class[cls] = float(self.shared_terminal_bonus)
                    self._class_union_bonus_paid[cls] = True

        # Inter-class overlap as a potential-style delta: charge each agent the
        # CHANGE in its (Jaccard-normalized, weighted) overlap level, computed here
        # from everyone's post-action boxes so all agents see a consistent state.
        # Telescoping bounds the episode total by level_final - level_init (|.| <= 1)
        # instead of growing linearly with episode length.
        inter_overlap_delta: Dict[str, float] = {}
        for agent in self.agents:
            level = self._compute_inter_class_overlap_penalty(agent)
            prev_level = self._prev_inter_overlap.get(agent, level)
            inter_overlap_delta[agent] = level - prev_level
            self._prev_inter_overlap[agent] = level

        # Global coverage kept for logging only (weight-gated; no longer a reward stream)
        global_coverage = self._compute_global_coverage() if self.global_coverage_weight > 0 else 0.0

        # Add the shared reward to each agent's local reward and update infos
        for agent in self.agents:
            local_r = reward_without_shared.get(agent, 0.0)

            cls = self._get_class_for_agent(agent)
            shared_reward = float(shared_by_class.get(cls, 0.0)) if cls is not None else 0.0
            union_cov = 0.0
            union_prec = 0.0
            union_purity = 0.0
            if class_union_metrics and cls is not None and cls in class_union_metrics:
                m = class_union_metrics[cls]
                union_cov = float(m.get("union_coverage", 0.0))
                union_prec = float(m.get("union_precision", 0.0))
                union_purity = float(m.get("union_purity", 0.0))

            overlap_delta = float(inter_overlap_delta.get(agent, 0.0))
            # One-time union-target bonus for this class (latched), paid to every
            # agent of the class on the step the union first clears both targets.
            union_bonus = float(union_terminal_bonus_by_class.get(cls, 0.0)) if cls is not None else 0.0
            final_reward = float(local_r + shared_reward + union_bonus - overlap_delta)
            rewards[agent] = final_reward
            self._rt_total[agent] = float(self._rt_total.get(agent, 0.0) + shared_reward + union_bonus - overlap_delta)

            # Update info to reflect the final reward decomposition
            if agent in infos:
                infos[agent]["inter_class_overlap_penalty"] = overlap_delta
                infos[agent]["shared_reward"] = float(shared_reward)
                infos[agent]["shared_terminal_bonus"] = float(union_bonus)
                infos[agent]["class_union_coverage"] = float(union_cov)
                infos[agent]["class_union_precision"] = float(union_prec)
                infos[agent]["class_union_fidelity"] = float(union_prec)
                infos[agent]["class_union_purity"] = float(union_purity)
                infos[agent]["class_union_bonus"] = 0.0  # retired level bonus (was w*cov + w*prec per step)
                infos[agent]["global_coverage"] = float(global_coverage)
                infos[agent]["total_reward"] = float(final_reward)
            else:
                # In principle this should not happen, but guard just in case
                infos[agent] = {
                    "anchor_precision": 0.0,
                    "anchor_coverage": 0.0,
                    "drift": 0.0,
                    "anchor_drift": 0.0,
                    "js_penalty": 0.0,
                    "coverage_clipped": 0.0,
                    "termination_reason": 0.0,
                    "coverage_floor_hits": float(self.coverage_floor_hits.get(agent, 0)),
                    "coverage_before_revert": 0.0,
                    "coverage_after_revert": 0.0,
                    "precision_gain": 0.0,
                    "coverage_gain": 0.0,
                    "coverage_gain_scaled": 0.0,
                    "precision_gain_component": 0.0,
                    "coverage_gain_component": 0.0,
                    "coverage_bonus": 0.0,
                    "target_class_bonus": 0.0,
                    "overlap_penalty": 0.0,
                    "drift_penalty": 0.0,
                    "anchor_drift_penalty": 0.0,
                    "inter_class_overlap_penalty": overlap_delta,
                    "same_class_overlap_penalty": 0.0,
                    "coverage_floor_penalty": 0.0,
                    "class_union_coverage": float(union_cov),
                    "class_union_precision": float(union_prec),
                    "class_union_fidelity": float(union_prec),
                    "class_union_purity": float(union_purity),
                    "class_union_bonus": 0.0,  # retired level bonus (see above)
                    "shared_reward": float(shared_reward),
                    "shared_terminal_bonus": float(union_bonus),
                    "global_coverage": float(global_coverage),
                    "total_reward": float(final_reward),
                }
        
        self.timestep += 1
        
        max_steps_reached = self.timestep >= self.max_cycles

        # -----------------------------
        # Post-process: stabilization-based early termination
        # -----------------------------
        if self.enable_stability_termination:
            if int(self.timestep) >= int(self.stability_min_steps):
                for agent in list(self.agents):
                    if not terminations.get(agent, False) and not truncations.get(agent, False):
                        if int(self._stable_counts.get(agent, 0)) >= int(self.stability_window):
                            terminations[agent] = True
                            if agent in infos and isinstance(infos[agent], dict):
                                infos[agent]["stabilized"] = 1.0
                                infos[agent]["termination_reason_str"] = "stabilized"
        
        # If we hit the maximum number of steps, truncate any agents that are not yet terminated
        if max_steps_reached:
            for agent in self.agents:
                if not terminations.get(agent, False):
                    truncations[agent] = True
                    if self.partial_terminal_credit > 0.0:
                        precision = float(infos.get(agent, {}).get("anchor_precision", 0.0) or 0.0)
                        coverage = float(infos.get(agent, {}).get("anchor_coverage", 0.0) or 0.0)
                        cls = self._get_class_for_agent(agent)
                        _t = max(float(self._get_effective_precision_target(cls) if cls is not None else self.precision_target), 1e-6)
                        _m = max(float(getattr(self, "gate_margin", 0.10)), 1e-6)
                        _gate = float(min(1.0, max(0.0, (precision - (_t - _m)) / _m)))
                        _frac = float(max(0.0, min(1.0, coverage)))
                        _partial = float(self.terminal_bonus) * self.partial_terminal_credit * _frac * _gate
                        if _partial > 0.0:
                            rewards[agent] = float(rewards.get(agent, 0.0) + _partial)
                            self._rt_terminal[agent] += _partial
                            self._rt_total[agent] += _partial
                            if agent in infos:
                                infos[agent]["partial_terminal_credit"] = float(_partial)
        
        # Terminal coverage-floor enforcement (quantile mode). Runs after every
        # termination path is resolved, on the box the episode will actually keep.
        # Mirrors single_agentENV.step: the floor is enforced ONCE at the end rather
        # than per step, so the trajectory can explore -- which means a collapsed
        # final box must be able to retreat to the best feasible one it found.
        if self._uses_quantile_mdp() and self.coverage_floor_mode == "terminal":
            for agent in list(self.agents):
                if not (terminations.get(agent, False) or truncations.get(agent, False)):
                    continue
                try:
                    _p_end, _c_end, _d_end = self._current_metrics(agent)
                except Exception:
                    continue
                if not (np.isfinite(_c_end) and _c_end < self.min_coverage_floor):
                    continue
                _ref = self._best_box.get(agent)
                if _ref is None:
                    continue  # nothing feasible was ever found; keep the final box
                self.lower[agent], self.upper[agent] = _ref[0].copy(), _ref[1].copy()
                self.a[agent], self.b[agent] = _ref[2].copy(), _ref[3].copy()
                self.coverage_floor_hits[agent] = self.coverage_floor_hits.get(agent, 0) + 1
                _p2, _c2, _d2 = self._current_metrics(agent)
                if agent in infos and isinstance(infos[agent], dict):
                    infos[agent]["anchor_precision"] = float(_p2)
                    infos[agent]["anchor_coverage"] = float(_c2)
                    infos[agent]["precision"] = float(_p2)
                    infos[agent]["coverage"] = float(_c2)
                    infos[agent]["n_predicates"] = float(self.n_predicates(agent))
                    infos[agent]["terminal_floor_retreat"] = 1.0
                observations[agent] = self._get_observation(
                    agent, _p2, _c2, min(1.0, float(self.timestep) / float(self.max_cycles))
                )

        # Absorbing-terminal shaping correction: Phi of the terminal state is 0, so
        # the final transition's shaping is gamma*0 - Phi(s) and we subtract the
        # gamma*Phi(s') already added above. Applied here, after BOTH termination
        # paths (target-based in the agent loop, stability-based just above) have
        # been resolved.
        #
        # TERMINATIONS ONLY, never truncations. BenchMARL/TorchRL bootstraps a
        # time-limit truncation, so s' is not absorbing there; under shaping the
        # bootstrapped value already carries -Phi(s'), and applying this correction
        # too would double-count it and penalise exactly the episodes that end in a
        # good box. Same rule as single_agentENV.step.
        for agent in list(self.agents):
            if not terminations.get(agent, False):
                continue
            if truncations.get(agent, False):
                # Belt and braces: a truncated agent is bootstrapped, never absorbing.
                continue
            _phi_end = float(phi_curr_by_agent.get(agent, 0.0))
            if _phi_end == 0.0:
                continue
            _corr = -self.discount * _phi_end
            rewards[agent] = float(rewards.get(agent, 0.0) + _corr)
            self._rt_shaping[agent] += _corr
            self._rt_total[agent] += _corr
            if agent in infos and isinstance(infos[agent], dict):
                infos[agent]["shaping_terminal_correction"] = _corr
                infos[agent]["total_reward"] = float(rewards[agent])

        # Remove agents that are done (terminated or truncated) from the active list.
        # The episode for the environment ends once all agents are finished, but
        self.agents = [
            agent for agent in self.agents
            if not (terminations.get(agent, False) or truncations.get(agent, False))
        ]
        
        return observations, rewards, terminations, truncations, infos

    @staticmethod
    def _active_dims(lower, upper, ratio: float = 0.95) -> np.ndarray:
        return active_feature_mask(lower, upper, sparsity_width_ratio=ratio)

    def _volume_active(self, lower, upper) -> float:
        active = self._active_dims(lower, upper, getattr(self, "sparsity_width_ratio", 0.95))
        if not active.any():
            return 0.0
        w = np.maximum(np.asarray(upper) - np.asarray(lower), 1e-9)
        return float(np.prod(w[active]))
    
    # SS: Competative part of the game
    def _compute_inter_class_overlap_penalty(self, agent: str) -> float:
        """
        Weighted inter-class overlap LEVEL for this agent's box (clipped to [0, 1]).
        Jaccard-normalized (inter / union): the old inter / own_volume form punished
        being contained rather than overlapping, so inflating to a near-full-space
        box drove the penalty to ~0 — the degenerate minority-class solution.
        Iterates over all agents with boxes (not just alive ones) so another agent
        terminating does not discontinuously change this level mid-episode; step()
        charges the CHANGE in this level, not the level itself.
        """
        cls_agent = self._get_class_for_agent(agent)
        if cls_agent is None:
            return 0.0

        agent_lower = self.lower[agent]
        agent_upper = self.upper[agent]
        agent_vol = self._volume_active(agent_lower, agent_upper)

        if agent_vol <= 1e-12:
            return 0.0

        total_overlap_vol = 0.0

        agents_with_boxes = set(self.lower.keys()) & set(self.upper.keys())
        for other_agent in agents_with_boxes:
            if other_agent == agent:
                continue

            cls_other = self._get_class_for_agent(other_agent)
            # Only penalize overlap with agents belonging to different classes
            if cls_other is None or cls_other == cls_agent:
                continue

            other_lower = self.lower[other_agent]
            other_upper = self.upper[other_agent]
            other_vol = self._volume_active(other_lower, other_upper)
            if other_vol <= 1e-12:
                continue

            inter_lower = np.maximum(agent_lower, other_lower)
            inter_upper = np.minimum(agent_upper, other_upper)
            union_active = self._active_dims(agent_lower, agent_upper) | self._active_dims(other_lower, other_upper)
            if not union_active.any():
                continue
            inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
            inter_vol = float(np.prod(np.maximum(inter_widths[union_active], 1e-12)))

            if inter_vol > 1e-12:
                union_vol = agent_vol + other_vol - inter_vol
                if union_vol > 1e-12:
                    total_overlap_vol += inter_vol / union_vol

        inter_class_overlap_weight = getattr(self, 'inter_class_overlap_weight', 0.1)
        penalty = inter_class_overlap_weight * total_overlap_vol

        return float(np.clip(penalty, 0.0, 1.0))

    def _compute_same_class_overlap_penalty(self, agent: str) -> float:
        if self.same_class_diversity_weight <= 0.0:
            return 0.0
        
        cls = self._get_class_for_agent(agent)
        if cls is None:
            return 0.0
        
        # Same-class agents currently active
        same_class_agents = [
            a for a in self.agents
            if a != agent and self._get_class_for_agent(a) == cls
        ]
        if not same_class_agents:
            return 0.0
        
        lower_i = self.lower[agent]
        upper_i = self.upper[agent]
        vol_i = self._volume_active(lower_i, upper_i)
        if vol_i <= 1e-12:
            return 0.0
        
        jacc_sum = 0.0
        count = 0
        
        for other in same_class_agents:
            lower_j = self.lower[other]
            upper_j = self.upper[other]
            vol_j = self._volume_active(lower_j, upper_j)
            if vol_j <= 1e-12:
                continue
            
            inter_lower = np.maximum(lower_i, lower_j)
            inter_upper = np.minimum(upper_i, upper_j)
            union_active = self._active_dims(lower_i, upper_i) | self._active_dims(lower_j, upper_j)
            if not union_active.any():
                continue
            inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
            inter_vol = float(np.prod(np.maximum(inter_widths[union_active], 1e-12)))
            if inter_vol <= 1e-12:
                continue
            
            union_vol = vol_i + vol_j - inter_vol
            if union_vol <= 1e-12:
                continue
            
            jacc = inter_vol / union_vol
            jacc_sum += jacc
            count += 1
        
        if count == 0:
            return 0.0
        
        avg_jacc = jacc_sum / count
        penalty = self.same_class_diversity_weight * avg_jacc
        return float(np.clip(penalty, 0.0, 1.0))

    def _get_cached_probs(self, split: str) -> np.ndarray:
        """
        Classifier probabilities for every row of the given split, computed once
        and cached. C-12: filling the cache counts as black-box queries.
        """
        if isinstance(split, bool):
            split = "test" if split else "train"
        if split == "training":
            split = "train"
        key = split
        if self._cached_probs.get(key) is None:
            if key == "test":
                X = self.X_test_std
            elif key == "val":
                X = self.X_val_std
            else:
                X = self.X_std
            if X is None:
                raise ValueError(f"No {key} data available for cached predictions")
            if hasattr(self.classifier, 'eval'):
                self.classifier.eval()
            if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'eval'):
                self.classifier.model.eval()
            with torch.no_grad():
                inputs = torch.from_numpy(X.astype(np.float32)).to(self.device)
                self._cached_probs[key] = predict_proba_torch(self.classifier, inputs).cpu().numpy()
            self.n_blackbox_queries += int(X.shape[0])
        return self._cached_probs[key]

    def _get_effective_precision_target(self, target_class: int) -> float:
        """
        Class-aware precision target: min(precision_target, max(prior, 0.9 x the
        classifier's own precision for this class on training data)). The classifier's
        per-class precision is an upper bound on what any anchor box can achieve, so
        an absolute target above it makes the precision gate and the termination
        conditions unreachable for minority/overlapping classes.
        """
        if not self.use_class_aware_targets:
            return float(self.precision_target)
        if target_class in self._effective_precision_targets:
            return self._effective_precision_targets[target_class]
        try:
            probs = self._get_cached_probs("train")
        except Exception as e:
            logger.warning(f"Could not compute class-aware precision target ({e}); "
                           f"falling back to absolute target {self.precision_target}")
            self._effective_precision_targets[target_class] = float(self.precision_target)
            return self._effective_precision_targets[target_class]
        preds = probs.argmax(axis=1)
        prior = float((self.y == target_class).mean())
        pred_mask = (preds == target_class)
        if pred_mask.any():
            clf_class_precision = float((self.y[pred_mask] == target_class).mean())
        else:
            clf_class_precision = prior
        ceiling = max(prior, 0.9 * clf_class_precision)
        effective = float(np.clip(min(self.precision_target, ceiling), 0.05, self.precision_target))
        if effective < self.precision_target:
            logger.info(f"Class {target_class}: effective precision target {effective:.4f} "
                        f"(absolute={self.precision_target}, prior={prior:.4f}, "
                        f"classifier class precision={clf_class_precision:.4f})")
        self._effective_precision_targets[target_class] = effective
        return effective

    def _potential(self, precision: float, coverage: float, target_class: Optional[int],
                   k: Optional[int] = None, p_reset: Optional[float] = None) -> float:
        """
        State potential for reward shaping: Phi(s) = alpha * precision +
        beta * sqrt(coverage) * gate(precision). sqrt amplifies the early-coverage
        signal; the gate scales coverage credit by precision quality relative to the
        class-aware threshold, preserving the precision-first curriculum without any
        non-potential bonus terms. Phi depends only on state, so shaping with
        Phi(s') - Phi(s) leaves the optimal policy unchanged (Ng et al., 1999).
        Mirrors single_agentENV.py for fair comparison.
        """
        precision = max(0.0, float(precision))
        coverage = max(0.0, float(coverage))
        target = self._get_effective_precision_target(target_class) if target_class is not None else self.precision_target
        # B2 rebalance (mirrors single_agentENV._potential): gate coverage credit on
        # precision relative to the target. The old gate saturated at 0.8*target,
        # granting full coverage credit below target; with beta raised so coverage
        # drives the policy, that becomes a licence to trade fidelity for coverage.
        target = max(float(target), 1e-6)
        margin = max(float(getattr(self, "gate_margin", 0.10)), 1e-6)
        gate = (precision - (target - margin)) / margin
        gate = float(min(1.0, max(0.0, gate)))
        if not self._uses_quantile_mdp():
            return float(self.alpha * precision + self.beta * np.sqrt(coverage) * gate)
        # Quantile MDP, mirroring single_agentENV._potential:
        #  * p_tilde = min(P, target): precision above target is worth nothing, so
        #    once the constraint binds the only remaining gradient is coverage.
        #    Without this the gate slope (alpha + beta*sqrt(C)/margin ~ 6x) makes
        #    relaxing any dim strictly costly and the policy shrinks to k = d.
        #  * cov_ok: an empty rule (k = 0) covers everything at the class base rate.
        #    On an imbalanced majority class that already clears the gate, so Phi at
        #    reset would be near maximal and DOING NOTHING would be optimal.
        p_tilde = min(precision, target)
        cov_ok = 1.0 if (k is not None and int(k) >= 1
                         and (p_reset is None or precision > float(p_reset) + 1e-12)) else 0.0
        return float(self.alpha * p_tilde + self.beta * np.sqrt(coverage) * gate * cov_ok)

    def _compute_class_union_metrics(self) -> Dict[int, Dict[str, float]]:
        X_data, _, y_data, _ = self._active_data()
        
        if X_data is None or y_data is None or X_data.shape[0] == 0:
            return {}
        
        n_samples = X_data.shape[0]
        class_ids = set()

        # Only agents currently in the episode. Idle init boxes of other
        # agents (single-agent inference rollouts) must not enter the union.
        agents_with_boxes = [a for a in self.agents if a in self.lower and a in self.upper]
        for agent in agents_with_boxes:
            cls = self._get_class_for_agent(agent)
            if cls is not None:
                class_ids.add(cls)
        
        metrics: Dict[int, Dict[str, float]] = {}
        
        for cls in sorted(class_ids):
            agents_c = [a for a in agents_with_boxes if self._get_class_for_agent(a) == cls]
            if not agents_c:
                continue
            
            union_mask = np.zeros(n_samples, dtype=bool)
            # FIX: Only include agents that have valid boxes (both lower and upper bounds exist)
            # and have been updated (not just initialized). During inference with single-agent rollouts,
            # other agents' boxes might be initialized but never updated, causing incorrect union metrics.
            valid_agents = []
            for agent in agents_c:
                if agent in self.lower and agent in self.upper:
                    # Check if box is valid (not all zeros or all ones, which might indicate uninitialized state)
                    lower = self.lower[agent]
                    upper = self.upper[agent]
                    # Valid box should have upper > lower for at least some features
                    if np.any(upper > lower):
                        valid_agents.append(agent)
            
            # If no valid agents, skip this class
            if not valid_agents:
                continue
            
            for agent in valid_agents:
                mask_agent = self._mask_in_box(agent)
                if mask_agent.shape[0] == union_mask.shape[0]:
                    union_mask |= mask_agent
            
            # Class-conditional coverage: P(x in union | y = cls)
            mask_cls = (y_data == cls)
            if mask_cls.sum() > 0:
                cov_union = float(union_mask[mask_cls].mean())
            else:
                cov_union = 0.0
            
            # C-09: union PRIMARY is model fidelity, matching local rewards and
            # the revision evaluator. Ground-truth purity remains diagnostic.
            if union_mask.any():
                preds = self._get_cached_probs(self._active_split()).argmax(axis=1)
                fidelity_union = float((preds[union_mask] == cls).mean())
                purity_union = float((y_data[union_mask] == cls).mean())
            else:
                fidelity_union = 0.0
                purity_union = 0.0
            
            metrics[cls] = {
                "union_coverage": cov_union,
                "union_precision": fidelity_union,
                "union_fidelity": fidelity_union,
                "union_purity": purity_union,
            }
        
        return metrics
    
    # Added as part of debugging multiagent after presentation
    def _compute_global_coverage(self) -> float:
        if len(self.agents) == 0:
            return 0.0
        
        X_data, _, y_data, _ = self._active_data()
        
        if X_data is None or y_data is None or X_data.shape[0] == 0:
            return 0.0
        
        n_samples = X_data.shape[0]
        
        # Build union mask
        union_mask = np.zeros(n_samples, dtype=bool)
        
        for agent in self.agents:
            mask_agent = self._mask_in_box(agent)
            if mask_agent.shape[0] == union_mask.shape[0]:
                union_mask |= mask_agent
        
        # Global coverage
        global_coverage = float(union_mask.mean()) if n_samples > 0 else 0.0
        
        return global_coverage
    
    # SS: Drift penalty
    def _compute_anchor_drift_penalty(self, agent: str, prev_lower: np.ndarray, prev_upper: np.ndarray) -> float:
        anchor_drift_penalty = 0.0
        if self.x_star_unit.get(agent) is not None:
            box_center = 0.5 * (self.lower[agent] + self.upper[agent])
            anchor_distance = float(np.linalg.norm(box_center - self.x_star_unit[agent]))
            max_allowed_distance = self.initial_window * 2.0
            if anchor_distance > max_allowed_distance:
                excess = anchor_distance - max_allowed_distance
                anchor_drift_penalty = self.drift_penalty_weight * excess * 0.5
        return anchor_drift_penalty
    
    # SS: Reward weights and penalties
    # NOTE: _compute_reward_weights_and_penalties, _compute_coverage_bonus,
    # _compute_target_class_bonus and _compute_shared_reward were removed with the
    # move to potential-based reward shaping (see _potential and the same-class
    # shared reward in step()).

    def _compute_group_map(self) -> Dict[str, List[str]]:
        group_map = {}
        
        for agent in self.possible_agents:
            group_map[agent] = [agent]
        
        return group_map
    
    @property
    def group_map(self) -> Dict[str, List[str]]:
        return self._group_map.copy()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            # [lower(n), upper(n), precision, coverage, episode_phase]
            # C-06: episode_phase = t / max_cycles is the quantity the paper called
            # ξ_t. It is a shared clock (same t for every agent) but it does NOT
            # depend on joint actions — it just increments. That is agent-wise
            # decoupled (Condition B for a Markov potential game). The old shared
            # termination-reason counters that disabled conditions mid-training
            # WERE joint-history-dependent; they are deprecated (config comment)
            # and no longer couple the agents.
            # episode_phase also makes the observation time-aware so the critic
            # has a consistent value function under per-step costs.
            #
            # Quantile MDP: [a(n), b(n), q*(n), precision, coverage, mode_bit].
            # Shape is CONDITIONAL so hull-mode runs (WyoDOT/CDEA, in-flight jobs)
            # keep 2n+3 and their existing checkpoints stay loadable.
            shape=((3 if self._uses_quantile_mdp() else 2) * self.n_features + 3,),
            dtype=np.float32
        )
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.n_features,),
            dtype=np.float32
        )

    def export_rule_state(self, agent: str) -> Dict[str, Any]:
        """Authoritative rule state for inference.

        In quantile mode the observation carries (a, b, q*) -- QUANTILE positions --
        so unit bounds cannot be recovered from the observation alone without the
        class CDF knots. Anything that decodes obs[:n] as `lower` is silently wrong
        there. Inference must read this instead.
        """
        active = self._constrained_mask(agent)
        out = {
            "active_features": active.astype(int).tolist(),
            "n_predicates": int(active.sum()),
            "lower_bounds_normalized": np.asarray(self.lower[agent], dtype=float).tolist(),
            "upper_bounds_normalized": np.asarray(self.upper[agent], dtype=float).tolist(),
        }
        if self._uses_quantile_mdp():
            out.update({
                "a": np.asarray(self.a[agent], dtype=float).tolist(),
                "b": np.asarray(self.b[agent], dtype=float).tolist(),
                "q_star": np.asarray(self.q_star[agent], dtype=float).tolist(),
                "quantile_knots": qmdp.export_knots(self._cdfs(agent)),
            })
        return out

    def get_anchor_bounds(self, agent: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.lower[agent].copy(), self.upper[agent].copy()
    
    def extract_rule(
        self, 
        agent: str, 
        max_features_in_rule: Optional[int] = 5,
        initial_lower: Optional[np.ndarray] = None,
        initial_upper: Optional[np.ndarray] = None,
        denormalize: bool = False
    ) -> str:
        from utils.metrics import sparsify_box

        if self._uses_quantile_mdp() and self.n_predicates(agent) < 1:
            return "any values (no tightened features)"
        lower, upper, active = sparsify_box(
            self.lower[agent],
            self.upper[agent],
            sparsity_width_ratio=self.sparsity_width_ratio,
            max_features=max_features_in_rule,
            active_mask=self._constrained_mask(agent) if self._uses_quantile_mdp() else None,
        )

        # Denormalize bounds if requested: unit [0,1] -> standardized -> original raw units
        if denormalize:
            if self.X_min is None or self.X_range is None:
                logger.warning("Cannot denormalize: X_min or X_range not available. Using normalized bounds.")
                denormalize = False
            else:
                lower = self._unit_to_orig(lower)
                upper = self._unit_to_orig(upper)

        to_show_idx = np.flatnonzero(active)
        if to_show_idx.size == 0:
            return "any values (no tightened features)"
        
        cond_parts = []
        for i in to_show_idx:
            if i in self.categorical_indices and denormalize:
                code = int(round(float((lower[i] + upper[i]) / 2.0)))
                labels = self.categorical_value_names.get(int(i), [])
                value = labels[code] if 0 <= code < len(labels) else str(code)
                cond_parts.append(f"{self.feature_names[i]} = {value!r}")
            else:
                cond_parts.append(f"{self.feature_names[i]} ∈ [{lower[i]:.6f}, {upper[i]:.6f}]")
        
        return " and ".join(cond_parts)
    
    def render(self):
        raise NotImplementedError("Render not implemented for AnchorEnv")
    
    def close(self):
        pass

# main function to test the environment compatibility with PettingZoo. 
# The AnchorEnv is inherited from the ParallelEnv class in PettingZoo.
# This is needed for the environment to be compatible with BenchMARL.
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 5
    n_classes = 2
    
    X_raw = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw).astype(np.float32)
    
    X_min = X_std.min(axis=0)
    X_max = X_std.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit = (X_std - X_min) / X_range
    X_unit = np.clip(X_unit, 0.0, 1.0).astype(np.float32)
    
    try:
        from trainers.networks import SimpleClassifier
        classifier = SimpleClassifier(input_dim=n_features, num_classes=n_classes, dropout_rate=0.3, use_batch_norm=True)
    except (ImportError, TypeError):
        try:
            from trainers.multiagent_networks import SimpleClassifier
            classifier = SimpleClassifier(input_size=n_features, hidden_size=128, output_size=n_classes)
        except ImportError:
            class TestClassifier(torch.nn.Module):
                def __init__(self, input_dim, num_classes):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_dim, 64)
                    self.fc2 = torch.nn.Linear(64, 64)
                    self.fc3 = torch.nn.Linear(64, num_classes)
                    self.relu = torch.nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x
            
            classifier = TestClassifier(input_dim=n_features, num_classes=n_classes)
    
    classifier.eval()
    
    env_config = {
        "precision_target": 0.8,
        "coverage_target": 0.1,
        "use_perturbation": False,
        "X_min": X_min,
        "X_range": X_range,
    }
    
    test_env = AnchorEnv(
        X_unit=X_unit,
        X_std=X_std,
        y=y,
        feature_names=feature_names,
        classifier=classifier,
        device="cpu",
        target_classes=[0, 1],
        env_config=env_config
    )
    
    from pettingzoo.test import parallel_api_test
    parallel_api_test(test_env, num_cycles=10)


if __name__ == "__main__":
    main()
