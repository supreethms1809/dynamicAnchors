"""
Revision evaluation harness (C-05, C-10, C-12, C-13, C-14, C-51).

Built once, used by every method and baseline so tables are generated from the
same schema with no hand transcription.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.metrics import (
    MIN_SUPPORT_DEFAULT,
    RANKING_SCORE_PRECISION_COVERAGE,
    BoxMetrics,
    RankedRule,
    UnionResult,
    assert_print_matches_box,
    box_mask,
    compactness_of_ruleset,
    evaluate_mask,
    ranking_score,
    select_topk_union,
    sparsify_box,
)

LEGACY_RULE_TESTER_WARNING = (
    "This script unions printed unique_rules strings and is NOT the paper "
    "reporting path (C-01 / C-10). Use `python -m revision.evaluate` for "
    "held-out Fid/Pur, n_covered, and top-k unions from stored boxes."
)


# ---------------------------------------------------------------------------
# C-10 — three-way split helpers
# ---------------------------------------------------------------------------

def assert_no_index_overlap(train_idx, val_idx, test_idx) -> None:
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)
    if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & set(test_idx)) or len(set(val_idx) & set(test_idx)):
        raise AssertionError(
            "Train/val/test index overlap detected. Class-level metrics would leak."
        )


# ---------------------------------------------------------------------------
# C-14 — global rule-set-as-classifier
# ---------------------------------------------------------------------------

@dataclass
class GlobalRuleSetResult:
    n_eval: int
    n_abstain: int
    n_conflict: int
    n_decided: int
    n_fid_agree: int
    n_pur_agree: int
    global_fidelity: float
    global_purity: float
    abstention_rate: float
    conflict_rate: float
    coverage: float
    per_class_fired: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        def _f(x):
            return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
        return {
            "n_eval": int(self.n_eval),
            "n_abstain": int(self.n_abstain),
            "n_conflict": int(self.n_conflict),
            "n_decided": int(self.n_decided),
            "n_fid_agree": int(self.n_fid_agree),
            "n_pur_agree": int(self.n_pur_agree),
            "global_fidelity": _f(self.global_fidelity),
            "global_purity": _f(self.global_purity),
            "abstention_rate": _f(self.abstention_rate),
            "conflict_rate": _f(self.conflict_rate),
            "coverage": _f(self.coverage),
            "per_class_fired": {str(k): int(v) for k, v in self.per_class_fired.items()},
        }


def evaluate_ruleset_as_classifier(
    class_union_masks: Dict[int, np.ndarray],
    class_union_fid: Dict[int, float],
    y: np.ndarray,
    y_hat: np.ndarray,
) -> GlobalRuleSetResult:
    """predict(x): fire class unions; empty -> ABSTAIN; 2+ classes -> CONFLICT
    (tie-break by union fidelity). Coverage = 1 - abstention.
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    n = y.shape[0]
    classes = sorted(class_union_masks.keys())
    if not classes:
        return GlobalRuleSetResult(
            n_eval=n, n_abstain=n, n_conflict=0, n_decided=0,
            n_fid_agree=0, n_pur_agree=0,
            global_fidelity=float("nan"), global_purity=float("nan"),
            abstention_rate=1.0, conflict_rate=0.0, coverage=0.0,
        )

    stacked = np.stack([np.asarray(class_union_masks[c], dtype=bool) for c in classes], axis=1)
    n_fired = stacked.sum(axis=1)
    abstain = n_fired == 0
    conflict = n_fired >= 2
    decided_single = n_fired == 1

    pred = np.full(n, -1, dtype=int)
    if decided_single.any():
        pred[decided_single] = np.array(classes)[stacked[decided_single].argmax(axis=1)]

    if conflict.any():
        fid_vec = np.array([class_union_fid.get(c, float("nan")) for c in classes], dtype=np.float64)
        fid_vec = np.where(np.isfinite(fid_vec), fid_vec, -np.inf)
        # among fired classes, pick the one with highest union fidelity
        fired_fid = np.where(stacked[conflict], fid_vec[None, :], -np.inf)
        pred[conflict] = np.array(classes)[fired_fid.argmax(axis=1)]

    decided = ~abstain
    n_decided = int(decided.sum())
    if n_decided == 0:
        n_fid = n_pur = 0
        gfid = gpur = float("nan")
    else:
        n_fid = int((pred[decided] == y_hat[decided]).sum())
        n_pur = int((pred[decided] == y[decided]).sum())
        gfid = n_fid / n_decided
        gpur = n_pur / n_decided

    per_class_fired = {int(c): int(class_union_masks[c].sum()) for c in classes}
    return GlobalRuleSetResult(
        n_eval=n,
        n_abstain=int(abstain.sum()),
        n_conflict=int(conflict.sum()),
        n_decided=n_decided,
        n_fid_agree=n_fid,
        n_pur_agree=n_pur,
        global_fidelity=gfid,
        global_purity=gpur,
        abstention_rate=float(abstain.mean()),
        conflict_rate=float(conflict.mean()),
        coverage=float(1.0 - abstain.mean()),
        per_class_fired=per_class_fired,
    )


# ---------------------------------------------------------------------------
# C-13 — original-unit bounds inside observed range
# ---------------------------------------------------------------------------

def assert_bounds_in_observed_range(
    lower: np.ndarray,
    upper: np.ndarray,
    feature_min: np.ndarray,
    feature_max: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    atol: float = 1e-6,
) -> None:
    """Every printed continuous bound must lie inside the observed feature range."""
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    feature_min = np.asarray(feature_min, dtype=np.float64).reshape(-1)
    feature_max = np.asarray(feature_max, dtype=np.float64).reshape(-1)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(len(lower))]
    for i, name in enumerate(names):
        if lower[i] < feature_min[i] - atol:
            raise AssertionError(
                f"{name}: lower bound {lower[i]} is below observed min {feature_min[i]} "
                f"(negative petal-length style bug; C-13)"
            )
        if upper[i] > feature_max[i] + atol:
            raise AssertionError(
                f"{name}: upper bound {upper[i]} is above observed max {feature_max[i]} (C-13)"
            )


def unit_to_original(
    bounds_unit: np.ndarray,
    x_min_std: np.ndarray,
    x_range_std: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    """unit [0,1] -> StandardScaler space -> original feature units."""
    std = np.asarray(bounds_unit, dtype=np.float64) * np.asarray(x_range_std) + np.asarray(x_min_std)
    return std * np.asarray(scaler_scale) + np.asarray(scaler_mean)


# ---------------------------------------------------------------------------
# C-12 — query / wall-clock counters (method-agnostic)
# ---------------------------------------------------------------------------

class QueryCounter:
    """Hardware-independent black-box query budget."""

    def __init__(self):
        self.n_queries = 0
        self.n_reporting_queries = 0
        self.n_cache_hits = 0
        self.wall_train_s = 0.0
        self.wall_infer_s = 0.0

    def add_queries(self, n: int, *, reporting: bool = False) -> None:
        if reporting:
            self.n_reporting_queries += int(n)
        else:
            self.n_queries += int(n)

    def add_cache_hits(self, n: int) -> None:
        self.n_cache_hits += int(n)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_blackbox_queries": int(self.n_queries),
            "n_reporting_queries": int(self.n_reporting_queries),
            "query_policy": (
                "n_blackbox_queries counts generation/selection calls; "
                "held-out test reporting calls are separate"
            ),
            "n_cache_hits": int(self.n_cache_hits),
            "wall_train_seconds": float(self.wall_train_s),
            "wall_infer_seconds": float(self.wall_infer_s),
        }


# ---------------------------------------------------------------------------
# C-05 / C-51 — result schema + config dump
# ---------------------------------------------------------------------------

def git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
    except Exception:
        return None


def config_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def config_snapshot(obj: Any) -> Any:
    """Convert config objects into stable, JSON-friendly values."""
    if is_dataclass(obj):
        return config_snapshot(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): config_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [config_snapshot(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return {
            str(k): config_snapshot(v)
            for k, v in vars(obj).items()
            if not str(k).startswith("_")
        }
    return str(obj)


def library_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for pkg in ("torch", "sklearn", "gymnasium", "stable_baselines3", "pettingzoo"):
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not-imported"
    return versions


def dump_reproducibility_artifact(
    path: str,
    *,
    dataset: str,
    method: str,
    seed: int,
    tau_p: float,
    tau_c: float,
    split_sizes: Dict[str, int],
    classifier_info: Dict[str, Any],
    rl_info: Dict[str, Any],
    reward_info: Dict[str, Any],
    env_info: Dict[str, Any],
    rule_reporting: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """C-51: one JSON the appendix / repo can point at."""
    artifact = {
        "schema": "rlda_mada_rebuttal_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "tau_p": float(tau_p),
        "tau_c": float(tau_c),
        "split_sizes": split_sizes,
        "classifier": classifier_info,
        "rl": rl_info,
        "reward": reward_info,
        "environment": env_info,
        "rule_reporting": rule_reporting,
        "libraries": library_versions(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "extra": extra or {},
    }
    artifact["config_hash"] = config_hash(artifact)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    return artifact


def result_filename(dataset: str, method: str, seed: int, tau_p: float, tau_c: float) -> str:
    """One result file per (dataset, method, seed, tau_P, tau_C) — C-05."""
    def _fmt(x: float) -> str:
        return f"{x:.2f}".replace(".", "p")
    return f"{dataset}__{method}__seed{seed}__tp{_fmt(tau_p)}__tc{_fmt(tau_c)}.json"


def write_result_artifact(
    out_dir: str,
    *,
    dataset: str,
    method: str,
    seed: int,
    tau_p: float,
    tau_c: float,
    per_class: Dict[str, Any],
    global_ruleset: Optional[Dict[str, Any]],
    classifier_accuracy: Dict[str, Any],
    queries: Dict[str, Any],
    n_covered_note: str,
    git: Optional[str] = None,
    config_h: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    success_rate: Optional[Dict[str, Any]] = None,
    compactness: Optional[Dict[str, Any]] = None,
    ranking_formula: Optional[str] = None,
    min_support: Optional[int] = None,
) -> str:
    extra = dict(extra or {})
    if ranking_formula is not None:
        extra.setdefault("ranking_formula", ranking_formula)
    if min_support is not None:
        extra.setdefault("min_support", int(min_support))
    payload = {
        "schema": "rlda_mada_result_v1",
        "dataset": dataset,
        "method": method,
        "seed": int(seed),
        "tau_p": float(tau_p),
        "tau_c": float(tau_c),
        "git_commit": git or git_commit_hash(),
        "config_hash": config_h or config_hash({
            "dataset": dataset, "method": method, "seed": seed,
            "tau_p": tau_p, "tau_c": tau_c, "extra": extra,
        }),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_accuracy": classifier_accuracy,
        "queries": queries,
        "per_class": per_class,
        "global_ruleset": global_ruleset,
        "success_rate": success_rate,
        "compactness": compactness,
        "ranking_formula": ranking_formula or extra.get("ranking_formula"),
        "min_support": min_support if min_support is not None else extra.get("min_support"),
        "notes": n_covered_note,
        "extra": extra,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, result_filename(dataset, method, seed, tau_p, tau_c))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Build RankedRule list from stored anchors (box bounds, not printed strings)
# ---------------------------------------------------------------------------

def rules_from_anchors(
    anchors: Sequence[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    target_class: int,
    *,
    class_conditional: bool,
    min_support: int,
    ranking_formula: str,
    bounds_key_lower: Optional[str] = None,
    bounds_key_upper: Optional[str] = None,
    space: str = "unit",
) -> List[RankedRule]:
    """Evaluate stored boxes in an explicitly declared coordinate space.

    Inference artifacts store original-unit bounds under ``lower_bounds`` and
    unit-space bounds under ``lower_bounds_normalized``.  Choosing the keys from
    ``space`` here prevents silently comparing original-unit bounds with X_unit.
    """
    if space not in {"unit", "original"}:
        raise ValueError(f"space must be 'unit' or 'original', got {space!r}")
    if bounds_key_lower is None:
        bounds_key_lower = "lower_bounds_normalized" if space == "unit" else "lower_bounds"
    if bounds_key_upper is None:
        bounds_key_upper = "upper_bounds_normalized" if space == "unit" else "upper_bounds"

    out: List[RankedRule] = []
    for i, anc in enumerate(anchors):
        lo = anc.get(bounds_key_lower)
        up = anc.get(bounds_key_upper)
        if lo is None or up is None:
            continue
        lo = np.asarray(lo, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)
        if lo.shape[0] != X.shape[1]:
            continue
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(up)) or np.any(lo > up):
            raise ValueError(f"Invalid bounds for anchor {i}: lower/upper are non-finite or inverted")
        if space == "unit" and (np.any(lo < -1e-6) or np.any(up > 1.0 + 1e-6)):
            raise ValueError(
                f"Anchor {i} declares unit-space bounds outside [0,1]. "
                "Use lower_bounds_normalized/upper_bounds_normalized with X_unit."
            )
        mask = box_mask(X, lo, up)
        metrics = evaluate_mask(
            y=y, y_hat=y_hat, mask=mask, target_class=target_class,
            class_conditional=class_conditional, min_support=min_support,
        )
        display = str(anc.get("rule") or anc.get("display_rule") or "")
        rid = anc.get("rule_id") or f"{target_class}:{i}:{display[:80]}"
        out.append(RankedRule(
            rule_id=str(rid),
            lower=lo,
            upper=up,
            mask=mask,
            metrics=metrics,
            score=ranking_score(metrics.fidelity, metrics.coverage, ranking_formula,
                                n_covered=metrics.n_covered),
            display_rule=display,
            extra={"source_anchor_index": i, "space": space},
        ))
    return out


def reevaluate_ranked_rules(
    selected_rules: Sequence[RankedRule],
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    target_class: int,
    *,
    class_conditional: bool,
    min_support: int,
) -> List[RankedRule]:
    """Evaluate a validation-selected rule list on a reporting split.

    Ranking scores are deliberately preserved from validation.  Test labels and
    predictions therefore cannot alter which rules are selected or which rule is
    called ``best``.
    """
    out: List[RankedRule] = []
    for rule in selected_rules:
        if rule.lower is None or rule.upper is None:
            raise ValueError(f"Selected rule {rule.rule_id!r} has no box bounds")
        mask = box_mask(X, rule.lower, rule.upper)
        metrics = evaluate_mask(
            y=y,
            y_hat=y_hat,
            mask=mask,
            target_class=target_class,
            class_conditional=class_conditional,
            min_support=min_support,
        )
        out.append(RankedRule(
            rule_id=rule.rule_id,
            lower=np.asarray(rule.lower, dtype=np.float32),
            upper=np.asarray(rule.upper, dtype=np.float32),
            mask=mask,
            metrics=metrics,
            score=rule.score,
            display_rule=rule.display_rule,
            extra={
                **rule.extra,
                "selection_metrics": rule.metrics.to_dict(),
                "selection_score": None if not np.isfinite(rule.score) else float(rule.score),
            },
        ))
    return out


def per_class_block(
    union: UnionResult,
    instance_metrics: Optional[BoxMetrics] = None,
) -> Dict[str, Any]:
    """One class row for the result JSON / tables.

    `best` is rank-1 of the same top-k used for the union (C-01).
    Instance metrics (overall coverage) are optional and separate.
    """
    block = {
        "k": union.k,
        "n_selected": union.n_selected,
        "best": {
            "rule_id": union.best.rule_id,
            "display_rule": union.best.display_rule,
            "fidelity": union.best.metrics.to_dict(),
            "score": None if not np.isfinite(union.best.score) else float(union.best.score),
            "lower_bounds": (
                None if union.best.lower is None
                else np.asarray(union.best.lower, dtype=float).tolist()
            ),
            "upper_bounds": (
                None if union.best.upper is None
                else np.asarray(union.best.upper, dtype=float).tolist()
            ),
        },
        "union": union.union_metrics.to_dict(),
        "selected_ids": union.selected_ids,
        "selected_rules": [
            {
                "rule_id": rule.rule_id,
                "display_rule": rule.display_rule,
                "selection_score": (
                    None if not np.isfinite(rule.score) else float(rule.score)
                ),
                "lower_bounds": (
                    None if rule.lower is None
                    else np.asarray(rule.lower, dtype=float).tolist()
                ),
                "upper_bounds": (
                    None if rule.upper is None
                    else np.asarray(rule.upper, dtype=float).tolist()
                ),
                "report_metrics": rule.metrics.to_dict(),
                "selection_metrics": rule.extra.get("selection_metrics"),
            }
            for rule in union.individual
        ],
    }
    if instance_metrics is not None:
        block["instance"] = instance_metrics.to_dict()
    if union.individual:
        sparsity = float((union.best.extra or {}).get("sparsity_width_ratio") or 0.95)
        block["compactness"] = compactness_of_ruleset(
            union.individual, sparsity_width_ratio=sparsity
        )
    return block


def audit_selected_unit_rules(
    rules: Sequence[RankedRule],
    X_unit: np.ndarray,
    *,
    sparsity_width_ratio: float,
    x_min_std: np.ndarray,
    x_range_std: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_min_orig: np.ndarray,
    feature_max_orig: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """C-08 / C-13: sparsified printed box vs evaluated coverage; original bounds in range."""
    problems: List[str] = []
    for rule in rules:
        if rule.lower is None or rule.upper is None:
            continue
        sparse_lo, sparse_up, _ = sparsify_box(
            rule.lower, rule.upper, sparsity_width_ratio=sparsity_width_ratio
        )
        printed_mask = box_mask(X_unit, sparse_lo, sparse_up)
        try:
            assert_print_matches_box(rule.mask, printed_mask, rule.rule_id)
        except AssertionError as exc:
            problems.append(str(exc))
        orig_lo = unit_to_original(rule.lower, x_min_std, x_range_std, scaler_mean, scaler_scale)
        orig_up = unit_to_original(rule.upper, x_min_std, x_range_std, scaler_mean, scaler_scale)
        try:
            assert_bounds_in_observed_range(
                orig_lo, orig_up, feature_min_orig, feature_max_orig, feature_names,
            )
        except AssertionError as exc:
            problems.append(f"{rule.rule_id}: {exc}")
    return problems


def resolve_extracted_models_dir(experiment_dir: str, prefer_model: str = "best") -> str:
    """Return individual_models_best or individual_models. Fail hard if best is requested but missing."""
    pm = str(prefer_model).lower()
    if pm not in ("best", "final"):
        pm = "best"
    final_dir = os.path.join(experiment_dir, "individual_models")
    best_dir = os.path.join(experiment_dir, "individual_models_best")
    if pm == "best":
        if not os.path.isdir(best_dir):
            raise FileNotFoundError(
                f"prefer_model='best' but {best_dir} is missing. "
                "Re-run training so individual_models_best is extracted at save-best "
                "time, or pass prefer_model='final'."
            )
        return best_dir
    if not os.path.isdir(final_dir):
        raise FileNotFoundError(
            f"prefer_model='final' but {final_dir} is missing."
        )
    return final_dir


def apply_train_val_slots(env_data: Dict[str, Any], env_config: Dict[str, Any], metric_split: str = "val") -> Dict[str, Any]:
    """Keep constructor X_unit/X_std/y as train. Attach val/test for live metrics.

    min_coverage_floor is 1/n of the *metric* split (val during paper generation).
    """
    cfg = dict(env_config)
    cfg["eval_on_test_data"] = bool(metric_split == "test")
    if metric_split == "val":
        if env_data.get("X_val_unit") is None:
            raise ValueError("Validation data is required for rule generation")
        cfg["eval_split"] = "val"
        cfg["X_val_unit"] = env_data["X_val_unit"]
        cfg["X_val_std"] = env_data["X_val_std"]
        cfg["y_val"] = env_data["y_val"]
        n = int(len(env_data["y_val"]))
    elif metric_split == "test":
        if env_data.get("X_test_unit") is None:
            raise ValueError("Test data is required for eval_split='test'")
        cfg["eval_split"] = "test"
        cfg["X_test_unit"] = env_data["X_test_unit"]
        cfg["X_test_std"] = env_data["X_test_std"]
        cfg["y_test"] = env_data["y_test"]
        n = int(len(env_data["y_test"]))
    else:
        cfg["eval_split"] = "train"
        n = int(len(env_data["y"])) if env_data.get("y") is not None else 0
    if n > 0:
        cfg["min_coverage_floor"] = max(1.0 / float(n), 1e-6)
    return cfg
