import numpy as np
from typing import Dict, List, Tuple, Optional

from gymnasium import spaces
from pettingzoo.utils import ParallelEnv


class MultiBoxAnchorEnv(ParallelEnv):
    """
    Multi-box per class environment for dynamic anchors.

    - One agent per BOX
    - Each agent belongs to a class: agent_id -> class_id
    - Each class has boxes_per_class boxes
    - State per agent: [lower (d), upper (d), precision, coverage]
    - Continuous action: [-1, 1]^(2d) -> delta for lower & upper bounds
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        target_classes: Optional[List[int]] = None,
        boxes_per_class: int = 3,
        max_cycles: int = 50,
        precision_target: float = 0.9,
        coverage_target: float = 0.5,
        max_action_scale: float = 0.2,
        min_absolute_step: float = 0.02,
        min_width: float = 0.02,
        initial_window: float = 0.3,
        inter_class_overlap_weight: float = 1.0,
        intra_class_overlap_weight: float = 0.1,
        class_union_weight: float = 0.5,
        shared_reward_weight: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            X: (N, d) features, assumed already normalized to [0, 1].
            y: (N,) labels.
            classifier: callable f(X_batch) -> predicted class labels (N,)
            target_classes: list of classes to model. If None, use np.unique(y).
            boxes_per_class: number of boxes per class.
        """
        assert X.ndim == 2
        assert len(X) == len(y)

        self.X_unit = X.astype(np.float32)
        self.y = y
        self.classifier = classifier

        if target_classes is None:
            self.target_classes = sorted(np.unique(y).tolist())
        else:
            self.target_classes = sorted(target_classes)

        self.boxes_per_class = boxes_per_class
        self.n_samples, self.n_features = self.X_unit.shape

        self.max_cycles = max_cycles
        self.precision_target = precision_target
        self.coverage_target = coverage_target

        self.max_action_scale = max_action_scale
        self.min_absolute_step = min_absolute_step
        self.min_width = min_width
        self.initial_window = initial_window

        self.inter_class_overlap_weight = inter_class_overlap_weight
        self.intra_class_overlap_weight = intra_class_overlap_weight
        self.class_union_weight = class_union_weight
        self.shared_reward_weight = shared_reward_weight

        # Agent sets
        self.possible_agents: List[str] = []
        self.agent_to_class: Dict[str, int] = {}
        self.agent_to_box: Dict[str, int] = {}

        for cls in self.target_classes:
            for b in range(self.boxes_per_class):
                agent_id = f"agent_{cls}_box{b}"
                self.possible_agents.append(agent_id)
                self.agent_to_class[agent_id] = cls
                self.agent_to_box[agent_id] = b

        # PettingZoo required attribute
        self.agents: List[str] = []

        # Box parameters
        self.lower: Dict[str, np.ndarray] = {}
        self.upper: Dict[str, np.ndarray] = {}
        self.prev_lower: Dict[str, np.ndarray] = {}
        self.prev_upper: Dict[str, np.ndarray] = {}

        # History for delta-based rewards
        self.prev_precision: Dict[str, float] = {}
        self.prev_coverage: Dict[str, float] = {}

        # Anchor points (per box); you can extend this later
        self.x_star_unit: Dict[str, Optional[np.ndarray]] = {}

        # Step counter
        self.timestep: int = 0

        # RNG
        self._rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------------
    # PettingZoo API
    # ---------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Box:
        # [lower (d), upper (d), precision, coverage]
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.n_features + 2,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> spaces.Box:
        # continuous deltas for lower and upper
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * self.n_features,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents.copy()
        self.timestep = 0

        self.lower.clear()
        self.upper.clear()
        self.prev_lower.clear()
        self.prev_upper.clear()
        self.prev_precision.clear()
        self.prev_coverage.clear()
        self.x_star_unit.clear()

        # Initialize anchors and boxes
        for cls in self.target_classes:
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) == 0:
                continue

            chosen_indices = self._rng.choice(
                class_indices,
                size=self.boxes_per_class,
                replace=(len(class_indices) < self.boxes_per_class),
            )

            for b, idx in enumerate(chosen_indices):
                agent = f"agent_{cls}_box{b}"
                x_anchor = self.X_unit[idx]
                self.x_star_unit[agent] = x_anchor

                lower = np.clip(
                    x_anchor - self.initial_window / 2.0, 0.0, 1.0
                )
                upper = np.clip(
                    x_anchor + self.initial_window / 2.0, 0.0, 1.0
                )

                # Ensure min_width
                widths = np.maximum(upper - lower, self.min_width)
                center = 0.5 * (lower + upper)
                lower = np.clip(center - widths / 2.0, 0.0, 1.0)
                upper = np.clip(center + widths / 2.0, 0.0, 1.0)

                self.lower[agent] = lower
                self.upper[agent] = upper
                self.prev_lower[agent] = lower.copy()
                self.prev_upper[agent] = upper.copy()

                # initial metrics
                p, c = self._compute_box_metrics(agent)
                self.prev_precision[agent] = p
                self.prev_coverage[agent] = c

        observations = {
            agent: self._build_observation(agent) for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        assert set(actions.keys()) == set(self.agents)

        self.timestep += 1

        # Apply actions: update boxes
        for agent, action in actions.items():
            self._apply_continuous_action(agent, action)

        # Compute union metrics once per class
        class_union_metrics = {
            cls: self._compute_class_union_metrics(cls)
            for cls in self.target_classes
        }

        # Compute shared reward once
        shared_reward = self._compute_shared_reward(class_union_metrics)

        # Rewards, terminations, truncations, infos
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}

        # Check global success condition: all classes meet targets
        all_classes_ok = all(
            (prec >= self.precision_target and cov >= self.coverage_target)
            for (prec, cov) in class_union_metrics.values()
        )
        global_done = all_classes_ok or (self.timestep >= self.max_cycles)

        # Per-agent reward
        for agent in self.agents:
            box_precision, box_coverage = self._compute_box_metrics(agent)
            prev_p = self.prev_precision[agent]
            prev_c = self.prev_coverage[agent]

            precision_gain = box_precision - prev_p
            coverage_gain = box_coverage - prev_c

            # Simple clipping to avoid huge spikes
            precision_gain_clipped = np.clip(precision_gain, -0.5, 0.5)
            coverage_gain_clipped = np.clip(coverage_gain, -0.5, 0.5)

            # Box-level reward
            box_reward = precision_gain_clipped + 0.5 * coverage_gain_clipped

            # Overlap penalties (intra-/inter-class)
            inter_pen, intra_pen = self._compute_overlap_penalties(agent)
            box_reward -= inter_pen
            box_reward -= intra_pen

            # Class-union bonus
            cls = self.agent_to_class[agent]
            union_prec, union_cov = class_union_metrics[cls]
            class_union_bonus = self._compute_class_union_bonus(
                union_prec, union_cov
            )

            # Final reward: box + class-union + shared
            reward = box_reward + class_union_bonus + shared_reward
            rewards[agent] = float(reward)

            # Update histories
            self.prev_precision[agent] = box_precision
            self.prev_coverage[agent] = box_coverage
            self.prev_lower[agent] = self.lower[agent].copy()
            self.prev_upper[agent] = self.upper[agent].copy()

            terminations[agent] = global_done
            truncations[agent] = False
            infos[agent] = {
                "box_precision": box_precision,
                "box_coverage": box_coverage,
                "class_union_precision": union_prec,
                "class_union_coverage": union_cov,
            }

        observations = {
            agent: self._build_observation(agent) for agent in self.agents
        }

        if global_done:
            # PettingZoo convention: empty agents list after done
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # No-op for now; you can add visualization later
        pass

    def close(self):
        pass

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _build_observation(self, agent: str) -> np.ndarray:
        lower = self.lower[agent]
        upper = self.upper[agent]
        p = self.prev_precision[agent]
        c = self.prev_coverage[agent]
        return np.concatenate(
            [lower, upper, np.array([p, c], dtype=np.float32)], axis=0
        ).astype(np.float32)

    def _apply_continuous_action(self, agent: str, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        lower_deltas = action[: self.n_features]
        upper_deltas = action[self.n_features :]

        lower = self.lower[agent].copy()
        upper = self.upper[agent].copy()

        widths = np.maximum(upper - lower, 1e-6)
        max_delta_proportional = self.max_action_scale * widths
        max_delta = np.maximum(max_delta_proportional, self.min_absolute_step)

        lower_changes = lower_deltas * max_delta
        upper_changes = upper_deltas * max_delta

        # Update lower
        new_lower = lower + lower_changes
        # Ensure new_lower <= upper - min_width
        new_lower = np.minimum(new_lower, upper - self.min_width)
        new_lower = np.clip(new_lower, 0.0, 1.0)

        # Update upper
        new_upper = upper + upper_changes
        # Ensure new_upper >= new_lower + min_width
        new_upper = np.maximum(new_upper, new_lower + self.min_width)
        new_upper = np.clip(new_upper, 0.0, 1.0)

        self.lower[agent] = new_lower
        self.upper[agent] = new_upper

    def _in_box(
        self, X: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        return np.all((X >= lower) & (X <= upper), axis=1)

    # ------------------- METRICS -------------------

    def _compute_box_metrics(self, agent: str) -> Tuple[float, float]:
        """
        Precision and coverage for this single box (agent) only.
        """
        cls = self.agent_to_class[agent]
        lower = self.lower[agent]
        upper = self.upper[agent]

        in_box = self._in_box(self.X_unit, lower, upper)
        is_cls = (self.y == cls)

        # coverage: fraction of class samples inside this box
        if is_cls.sum() == 0:
            coverage = 0.0
        else:
            coverage = float((in_box & is_cls).sum() / is_cls.sum())

        # precision: among in-box samples, fraction belonging to this class
        if in_box.sum() == 0:
            precision = 0.0
        else:
            precision = float((in_box & is_cls).sum() / in_box.sum())

        return precision, coverage

    def _compute_class_union_metrics(self, cls: int) -> Tuple[float, float]:
        """
        Precision and coverage for the union of all boxes belonging
        to class 'cls'.
        """
        class_agents = [
            a for a in self.possible_agents if self.agent_to_class[a] == cls
        ]
        if not class_agents:
            return 0.0, 0.0

        in_union = np.zeros(self.n_samples, dtype=bool)
        for agent in class_agents:
            lower = self.lower[agent]
            upper = self.upper[agent]
            in_box = self._in_box(self.X_unit, lower, upper)
            in_union |= in_box

        is_cls = (self.y == cls)

        if is_cls.sum() == 0:
            coverage = 0.0
        else:
            coverage = float((in_union & is_cls).sum() / is_cls.sum())

        if in_union.sum() == 0:
            precision = 0.0
        else:
            precision = float((in_union & is_cls).sum() / in_union.sum())

        return precision, coverage

    def _compute_overlap_penalties(self, agent: str) -> Tuple[float, float]:
        """
        Computes inter-class and intra-class overlap penalties for the given agent.
        Overlap is measured as intersection_volume / own_volume.
        """
        own_cls = self.agent_to_class[agent]
        lower_i = self.lower[agent]
        upper_i = self.upper[agent]
        width_i = np.maximum(upper_i - lower_i, 1e-6)
        vol_i = float(np.prod(width_i))

        inter_class_overlap = 0.0
        intra_class_overlap = 0.0

        for other in self.possible_agents:
            if other == agent:
                continue

            lower_j = self.lower[other]
            upper_j = self.upper[other]

            inter_lower = np.maximum(lower_i, lower_j)
            inter_upper = np.minimum(upper_i, upper_j)
            inter_width = np.maximum(inter_upper - inter_lower, 0.0)
            inter_vol = float(np.prod(inter_width))

            if vol_i <= 0.0:
                continue

            overlap_ratio = inter_vol / vol_i

            if self.agent_to_class[other] == own_cls:
                intra_class_overlap += overlap_ratio
            else:
                inter_class_overlap += overlap_ratio

        inter_pen = self.inter_class_overlap_weight * inter_class_overlap
        intra_pen = self.intra_class_overlap_weight * intra_class_overlap

        return inter_pen, intra_pen

    def _compute_class_union_bonus(
        self, union_precision: float, union_coverage: float
    ) -> float:
        """
        Simple shaping bonus for how good the union of boxes is for this class.
        """
        if self.class_union_weight == 0.0:
            return 0.0

        precision_target = self.precision_target
        coverage_target = self.coverage_target

        precision_progress = union_precision / max(precision_target, 1e-6)
        coverage_progress = union_coverage / max(coverage_target, 1e-6)

        bonus = 0.0
        if union_precision >= 0.8 * precision_target:
            bonus += 0.5 * precision_progress
        if union_coverage >= 0.5 * coverage_target:
            bonus += 0.5 * coverage_progress

        return self.class_union_weight * bonus

    def _compute_shared_reward(
        self, class_union_metrics: Dict[int, Tuple[float, float]]
    ) -> float:
        """
        Shared reward across all agents based on class-union metrics.
        """
        if self.shared_reward_weight == 0.0:
            return 0.0

        precisions = [p for (p, _) in class_union_metrics.values()]
        coverages = [c for (_, c) in class_union_metrics.values()]

        avg_precision = float(np.mean(precisions)) if precisions else 0.0
        avg_coverage = float(np.mean(coverages)) if coverages else 0.0

        precision_target = self.precision_target
        coverage_target = self.coverage_target

        precision_progress = avg_precision / max(precision_target, 1e-6)
        coverage_progress = avg_coverage / max(coverage_target, 1e-6)

        shared = 0.0

        # reward getting close to targets on average
        if avg_precision >= 0.8 * precision_target:
            shared += 0.5 * precision_progress
        if avg_coverage >= 0.5 * coverage_target:
            shared += 0.5 * coverage_progress

        return self.shared_reward_weight * shared