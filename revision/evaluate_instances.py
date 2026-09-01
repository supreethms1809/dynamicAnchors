"""Held-out instance evaluation (Track B). Does not replace revision.evaluate.

For each test x*:
  ŷ = f(x*)  →  one greedy rollout of π_ŷ  →  perturb Fid + empirical test Fid/Pur
then the same x* list vs classical Anchors, plus lookup vs local vs ŷ.

Usage:
    python -m revision.evaluate_instances \\
        --dataset iris --method rlda --algo ddpg --seed 42 \\
        --experiment_dir <exp> --classifier_path <clf.pth> \\
        --track_a_json <class-level result JSON> \\
        --rules_file <extracted_rules.json> \\
        --out_dir runs/rlda_ext_seed42/results/ddpg
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))
sys.path.insert(0, str(REPO / "single_agent"))

from utils.eval_harness import git_commit_hash  # noqa: E402
from utils.inference_extract import persist_box_from_episode  # noqa: E402
from utils.metrics import (  # noqa: E402
    MIN_SUPPORT_DEFAULT,
    active_feature_mask,
    box_mask,
    evaluate_mask,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("revision.evaluate_instances")

N_PERTURB_DEFAULT = 2048
MAX_PER_PRED_DEFAULT = 200


def _json_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _maybe_call(x: Any) -> Any:
    """anchor-exp stores precision/coverage/names as methods, not fields."""
    return x() if callable(x) else x


def _mean(xs: Sequence[Any]) -> Optional[float]:
    arr = [float(v) for v in xs if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else None


def sample_test_indices(
    y_hat: np.ndarray,
    *,
    max_per_pred: Optional[int],
    seed: int,
) -> np.ndarray:
    """Stratify by predicted class. max_per_pred=None keeps every test row."""
    y_hat = np.asarray(y_hat)
    rng = np.random.default_rng(int(seed))
    picked: List[int] = []
    for cls in sorted(np.unique(y_hat).tolist()):
        idx = np.flatnonzero(y_hat == cls)
        if max_per_pred is None or int(max_per_pred) <= 0 or idx.size <= int(max_per_pred):
            picked.extend(idx.tolist())
            continue
        chosen = rng.choice(idx, size=int(max_per_pred), replace=False)
        picked.extend(chosen.tolist())
    return np.asarray(sorted(picked), dtype=int)


def unit_to_std(X_unit: np.ndarray, x_min: np.ndarray, x_range: np.ndarray) -> np.ndarray:
    return np.asarray(X_unit, dtype=np.float32) * np.asarray(x_range, dtype=np.float32) + np.asarray(
        x_min, dtype=np.float32
    )


def box_contains(x_unit: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    x_unit = np.asarray(x_unit, dtype=np.float32).reshape(-1)
    lower = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(upper, dtype=np.float32).reshape(-1)
    return bool(np.all((x_unit >= lower) & (x_unit <= upper)))


def perturb_precision(
    *,
    X_train_unit: np.ndarray,
    x_star_unit: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray,
    predict_std: Callable[[np.ndarray], np.ndarray],
    x_min: np.ndarray,
    x_range: np.ndarray,
    y_hat_star: int,
    n_perturb: int,
    rng: np.random.Generator,
) -> Tuple[float, int]:
    """Anchors-style D_{x*}(z|A): pin active dims to x*, resample the rest from train.

    Returns (precision, n_queries).
    """
    X_train_unit = np.asarray(X_train_unit, dtype=np.float32)
    x_star_unit = np.asarray(x_star_unit, dtype=np.float32).reshape(-1)
    active = np.asarray(active, dtype=bool).reshape(-1)
    n = min(int(n_perturb), int(X_train_unit.shape[0]))
    if n <= 0:
        return float("nan"), 0
    idx = rng.integers(0, X_train_unit.shape[0], size=n)
    z = X_train_unit[idx].copy()
    if np.any(active):
        z[:, active] = x_star_unit[active]
    z_std = unit_to_std(z, x_min, x_range)
    pred = np.asarray(predict_std(z_std)).reshape(-1)
    return float((pred == int(y_hat_star)).mean()), int(n)


def break_even_n(cost_train: float, cost_pi: float, cost_anchors: float) -> Optional[float]:
    """N such that train + N * cost_pi = N * cost_anchors. None if π is not cheaper."""
    gap = float(cost_anchors) - float(cost_pi)
    if gap <= 0 or not np.isfinite(cost_train):
        return None
    return float(cost_train) / gap


def lookup_fired_classes(
    x_unit: np.ndarray,
    class_boxes: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
) -> List[int]:
    """Which class unions (OR of selected boxes) contain x."""
    fired = []
    for cls, boxes in class_boxes.items():
        if any(box_contains(x_unit, lo, up) for lo, up in boxes):
            fired.append(int(cls))
    return sorted(fired)


def class_boxes_from_track_a(track_a: Dict[str, Any]) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    out: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    per_class = track_a.get("per_class") or {}
    for key, block in per_class.items():
        if not isinstance(block, dict):
            continue
        cls = block.get("best", {}).get("fidelity", {}).get("target_class")
        if cls is None:
            try:
                cls = int(str(key).split("_")[-1])
            except ValueError:
                continue
        boxes = []
        for rule in block.get("selected_rules") or []:
            lo, up = rule.get("lower_bounds"), rule.get("upper_bounds")
            if lo is None or up is None:
                continue
            boxes.append((np.asarray(lo, dtype=np.float32), np.asarray(up, dtype=np.float32)))
        out[int(cls)] = boxes
    return out


def summarize_instance_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _col(name: str, pred: Optional[Callable] = None) -> List[Any]:
        vals = []
        for r in rows:
            if pred is not None and not pred(r):
                continue
            vals.append(r.get(name))
        return vals

    def _block(pred=None) -> Dict[str, Any]:
        subset = [r for r in rows if pred is None or pred(r)]
        return {
            "n": len(subset),
            "perturb_fid": _mean(_col("perturb_fid", pred)),
            "emp_fid": _mean(_col("emp_fid", pred)),
            "emp_pur": _mean(_col("emp_pur", pred)),
            "emp_cov_marginal": _mean(_col("emp_cov_marginal", pred)),
            "n_covered_mean": _mean(_col("n_covered", pred)),
            "frac_min_support": _mean(
                [1.0 if (r.get("n_covered") or 0) >= MIN_SUPPORT_DEFAULT else 0.0 for r in subset]
            ),
            "containment_rate": _mean(_col("contains_x", pred)),
            "queries_per_x": _mean(_col("queries", pred)),
            "perturb_queries_per_x": _mean(_col("perturb_queries", pred)),
            "wall_s_per_x": _mean(_col("wall_s", pred)),
            "n_steps_mean": _mean(_col("n_steps", pred)),
            "empty_rule_rate": _mean(_col("empty_rule", pred)),
        }

    return {
        "all": _block(),
        "f_correct": _block(lambda r: bool(r.get("f_correct"))),
        "f_wrong": _block(lambda r: r.get("f_correct") is False),
    }


def _predict_std_fn(loader, device: str):
    import torch
    from utils.networks import predict_proba_torch

    clf = loader.classifier

    def _predict(X_std: np.ndarray) -> np.ndarray:
        if hasattr(clf, "eval"):
            clf.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(X_std, dtype=np.float32))
            if str(device) != "cpu":
                t = t.to(device)
            return predict_proba_torch(clf, t).cpu().numpy().argmax(axis=1)

    return _predict


# ---------------------------------------------------------------------------
# Policy loading / one rollout
# ---------------------------------------------------------------------------

def _load_rlda_models(experiment_dir: str, loader, env_config: Dict[str, Any], algo: str, device: str):
    from stable_baselines3 import DDPG, SAC
    from single_agentENV import SingleAgentAnchorEnv

    env_data = loader.get_anchor_env_data()
    model_cls = DDPG if str(algo).lower() == "ddpg" else SAC
    models = {}
    n_classes = int(loader.n_classes)
    for cls in range(n_classes):
        path = os.path.join(experiment_dir, "best_model", f"class_{cls}", "best_model.zip")
        if not os.path.exists(path):
            alt = os.path.join(experiment_dir, "final_model", f"class_{cls}.zip")
            path = alt if os.path.exists(alt) else path
        if not os.path.exists(path):
            logger.warning("No RLDA model for class %s under %s", cls, experiment_dir)
            continue
        env = SingleAgentAnchorEnv(
            X_unit=env_data["X_unit"],
            X_std=env_data["X_std"],
            y=env_data["y"],
            feature_names=list(loader.feature_names),
            classifier=loader.classifier,
            device=device,
            target_class=cls,
            env_config=dict(env_config),
        )
        models[cls] = model_cls.load(path, env=env, device=device)
        logger.info("  RLDA class %s: %s", cls, path)
    if not models:
        raise FileNotFoundError(f"No RLDA policies in {experiment_dir}")
    return models, env_data


def _rollout_rlda(
    *,
    models,
    loader,
    env_config: Dict[str, Any],
    env_data: Dict[str, Any],
    x_star_unit: np.ndarray,
    y_hat: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    from single_agentENV import SingleAgentAnchorEnv
    from single_agent_inference import run_single_agent_rollout

    model = models.get(int(y_hat))
    if model is None:
        return {"error": f"no policy for predicted class {y_hat}"}
    cfg = dict(env_config)
    cfg["mode"] = "inference"
    env = SingleAgentAnchorEnv(
        X_unit=env_data["X_unit"],
        X_std=env_data["X_std"],
        y=env_data["y"],
        feature_names=list(loader.feature_names),
        classifier=loader.classifier,
        device=device,
        target_class=int(y_hat),
        env_config=cfg,
    )
    env.x_star_unit = np.asarray(x_star_unit, dtype=np.float32).copy()
    env.n_blackbox_queries = 0
    return run_single_agent_rollout(env, model, seed=seed)


def _resolve_mada_models_dir(experiment_dir: str) -> str:
    from BenchMARL.inference import resolve_extracted_models_dir

    return resolve_extracted_models_dir(experiment_dir, "best")


def _load_mada_policies(experiment_dir: str, device: str) -> Tuple[Dict[int, Any], Dict[int, str], Dict[str, Any], str]:
    from BenchMARL.inference import load_policy_model

    models_dir = _resolve_mada_models_dir(experiment_dir)
    index_path = os.path.join(models_dir, "policies_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing {index_path}")
    with open(index_path) as f:
        index = json.load(f)
    mlp_config = os.path.join(str(REPO / "BenchMARL"), "conf", "mlp.yaml")
    policies: Dict[int, Any] = {}
    agents: Dict[int, str] = {}
    for key, slot in (index.get("policies_by_class") or {}).items():
        cls = int(slot.get("class", str(key).split("_")[-1]))
        plist = slot.get("policies") or []
        if not plist:
            continue
        info = plist[0]
        agent = info.get("agent") or info.get("group")
        pfile = os.path.join(models_dir, info["policy_file"])
        meta = info.get("metadata_file") or ""
        meta_path = os.path.join(models_dir, meta) if meta else ""
        policies[cls] = load_policy_model(pfile, meta_path, mlp_config, device=device)
        agents[cls] = str(agent)
        logger.info("  MADA class %s: %s (%s)", cls, agent, pfile)
    if not policies:
        raise FileNotFoundError(f"No MADA policies in {models_dir}")
    return policies, agents, index, models_dir


def _rollout_mada(
    *,
    policies,
    agents: Dict[int, str],
    index: Dict[str, Any],
    loader,
    env_config: Dict[str, Any],
    env_data: Dict[str, Any],
    x_star_unit: np.ndarray,
    y_hat: int,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    from BenchMARL.environment import AnchorEnv
    from BenchMARL.inference import run_rollout_with_policy

    policy = policies.get(int(y_hat))
    agent_id = agents.get(int(y_hat))
    if policy is None or agent_id is None:
        return {"error": f"no policy for predicted class {y_hat}"}
    apc = int(index.get("agents_per_class") or 1)
    cfg = dict(env_config)
    cfg.update({
        "mode": "inference",
        "normalize_data": False,
        "agents_per_class": apc,
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
    })
    env = AnchorEnv(
        X_unit=env_data["X_unit"],
        X_std=env_data["X_std"],
        y=env_data["y"],
        feature_names=list(loader.feature_names),
        classifier=loader.classifier,
        device=device,
        target_classes=[int(y_hat)],
        env_config=cfg,
    )
    env.x_star_unit[agent_id] = np.asarray(x_star_unit, dtype=np.float32).copy()
    env.n_blackbox_queries = 0
    env.agents = [agent_id]
    return run_rollout_with_policy(
        env=env,
        policy=policy,
        agent_id=agent_id,
        device=device,
        seed=seed,
        exploration_mode="mean",
        action_noise_scale=0.0,
        verbose_logging=False,
    )


def _env_yaml(kind: str, loader, experiment_dir: str, seed: int) -> Dict[str, Any]:
    if kind == "rlda":
        from anchor_trainer_sb3 import AnchorTrainerSB3
        trainer = AnchorTrainerSB3(
            dataset_loader=loader, algorithm="ddpg",
            output_dir=os.path.join(experiment_dir, "inference"), seed=seed,
        )
        cfg = trainer._get_default_env_config()
    else:
        from BenchMARL.anchor_trainer import AnchorTrainer
        trainer = AnchorTrainer(
            dataset_loader=loader, algorithm="maddpg",
            output_dir=os.path.join(experiment_dir, "inference"), seed=seed,
        )
        try:
            cfg = trainer._load_env_config_from_yaml()
        except Exception:
            cfg = trainer._get_default_env_config()
    if isinstance(cfg, dict) and isinstance(cfg.get("env_config"), dict):
        nested, top = cfg["env_config"], {k: v for k, v in cfg.items() if k != "env_config"}
        cfg = {**nested, **top}
    env_data = loader.get_anchor_env_data()
    cfg.update({
        "X_min": env_data["X_min"],
        "X_range": env_data["X_range"],
        "scaler_mean": env_data.get("scaler_mean"),
        "scaler_scale": env_data.get("scaler_scale"),
        "normalize_data": False,
        "mode": "inference",
        "eval_split": "train",
        "X_val_unit": env_data.get("X_val_unit"),
        "X_val_std": env_data.get("X_val_std"),
        "y_val": env_data.get("y_val"),
        "categorical_indices": env_data.get("categorical_indices") or [],
        "categorical_value_names": env_data.get("categorical_value_names") or {},
    })
    return cfg


def _score_episode(
    episode: Dict[str, Any],
    *,
    env_config: Dict[str, Any],
    n_features: int,
    x_star_unit: np.ndarray,
    y: int,
    y_hat: int,
    X_test_unit: np.ndarray,
    y_test: np.ndarray,
    y_hat_test: np.ndarray,
    X_train_unit: np.ndarray,
    x_min: np.ndarray,
    x_range: np.ndarray,
    predict_std,
    n_perturb: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "y": int(y),
        "y_hat": int(y_hat),
        "f_correct": bool(int(y) == int(y_hat)),
        "empty_rule": 1.0,
        "contains_x": 0.0,
        "rollout_fid": _json_float(
            episode.get("precision", episode.get("anchor_precision", episode.get("instance_precision")))
        ),
        "n_steps": int(episode.get("n_steps") or 0),
        "queries": int(episode.get("n_blackbox_queries") or 0),
        "wall_s": _json_float(episode.get("rollout_time_seconds")),
    }
    if episode.get("error"):
        row["error"] = episode["error"]
        return row
    box = persist_box_from_episode(episode, env_config, n_features)
    if box is None:
        return row
    lower = np.asarray(box["lower_normalized"], dtype=np.float32)
    upper = np.asarray(box["upper_normalized"], dtype=np.float32)
    active = np.asarray(box["active_features"], dtype=bool)
    if active.size != n_features:
        active = active_feature_mask(lower, upper)
    row["empty_rule"] = 0.0
    row["contains_x"] = 1.0 if box_contains(x_star_unit, lower, upper) else 0.0
    row["n_active"] = int(active.sum())
    pert, nq = perturb_precision(
        X_train_unit=X_train_unit, x_star_unit=x_star_unit,
        lower=lower, upper=upper, active=active,
        predict_std=predict_std, x_min=x_min, x_range=x_range,
        y_hat_star=int(y_hat), n_perturb=n_perturb, rng=rng,
    )
    row["perturb_fid"] = _json_float(pert)
    # Measurement-only: do not fold into serving `queries` (rollout CRN).
    row["perturb_queries"] = int(nq)
    mask = box_mask(X_test_unit, lower, upper)
    emp = evaluate_mask(
        y=y_test, y_hat=y_hat_test, mask=mask, target_class=int(y_hat),
        class_conditional=False, min_support=MIN_SUPPORT_DEFAULT,
    )
    row["emp_fid"] = _json_float(emp.fidelity)
    row["emp_pur"] = _json_float(emp.purity)
    row["emp_cov_marginal"] = _json_float(emp.coverage_marginal)
    row["n_covered"] = int(emp.n_covered)
    row["fid_ci_low"], row["fid_ci_high"] = (
        _json_float(emp.fid_ci[0]), _json_float(emp.fid_ci[1])
    )
    return row


# ---------------------------------------------------------------------------
# Classical Anchors on the same x* list
# ---------------------------------------------------------------------------

def run_anchors_on_indices(
    loader,
    indices: np.ndarray,
    y_hat_test: np.ndarray,
    tau_p: float,
    seed: int,
) -> List[Dict[str, Any]]:
    from revision.baselines import _try_import_anchor

    anchor_tabular = _try_import_anchor()
    if anchor_tabular is None:
        logger.warning("anchor-exp not installed; skipping B2 classical Anchors")
        return []
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=[str(c) for c in range(loader.n_classes)],
        feature_names=list(loader.feature_names),
        train_data=loader.X_train,
    )
    X_test = loader.X_test
    rows = []
    predict_count = {"n": 0}

    def predict_fn(X_raw):
        X_raw = np.asarray(X_raw, dtype=np.float32)
        predict_count["n"] += int(X_raw.shape[0])
        Xs = loader.scaler.transform(X_raw)
        import torch
        from utils.networks import predict_proba_torch
        clf = loader.classifier
        if hasattr(clf, "eval"):
            clf.eval()
        with torch.no_grad():
            probs = predict_proba_torch(
                clf, torch.from_numpy(np.asarray(Xs, dtype=np.float32))
            ).cpu().numpy()
        return probs.argmax(axis=1)

    for i in indices:
        yhat = int(y_hat_test[int(i)])
        instance = X_test[int(i)]
        predict_count["n"] = 0
        t0 = time.perf_counter()
        try:
            exp = explainer.explain_instance(instance, predict_fn, threshold=float(tau_p))
            ok = True
            err = None
        except Exception as exc:
            ok = False
            exp = None
            err = str(exc)
        wall = time.perf_counter() - t0
        pred_names = []
        precision = coverage = None
        if ok and exp is not None:
            raw = _maybe_call(getattr(exp, "names", []))
            pred_names = list(raw or [])
            precision = _json_float(_maybe_call(getattr(exp, "precision", None)))
            coverage = _json_float(_maybe_call(getattr(exp, "coverage", None)))
        rows.append({
            "index": int(i),
            "y_hat": yhat,
            "ok": bool(ok),
            "error": err,
            "perturb_fid": precision,
            "coverage": coverage,
            "n_predicates": len(pred_names),
            "rule": " and ".join(str(p) for p in pred_names) if pred_names else None,
            "queries": int(predict_count["n"]),
            "wall_s": float(wall),
        })
    return rows


def _b5_train_vs_test(track_a: Dict[str, Any], rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pair Track A test Fid of selected boxes with rollout precision from extracted_rules."""
    pairs = []
    if not rules:
        return {"n": 0, "pairs": []}
    per_class_rules = rules.get("per_class_results") or {}
    anchors_by_cls: Dict[int, List[Dict[str, Any]]] = {}
    for key, cd in per_class_rules.items():
        if not isinstance(cd, dict):
            continue
        cls = cd.get("class")
        if cls is None:
            try:
                cls = int(str(key).replace("_class_based", "").split("_")[-1])
            except ValueError:
                continue
        bag = list(cd.get("all_anchors") or cd.get("anchors") or [])
        nested = cd.get("class_based_results") or {}
        if isinstance(nested, dict):
            for v in nested.values():
                if isinstance(v, dict):
                    bag.extend(v.get("anchors") or v.get("all_anchors") or [])
        anchors_by_cls.setdefault(int(cls), []).extend(bag)

    def _key(lo, up):
        return (
            tuple(np.round(np.asarray(lo, dtype=float), 5)),
            tuple(np.round(np.asarray(up, dtype=float), 5)),
        )

    for key, block in (track_a.get("per_class") or {}).items():
        if not isinstance(block, dict):
            continue
        cls = None
        try:
            cls = int(str(key).split("_")[-1])
        except ValueError:
            continue
        for rule in block.get("selected_rules") or []:
            lo, up = rule.get("lower_bounds"), rule.get("upper_bounds")
            if lo is None or up is None:
                continue
            test_fid = (rule.get("report_metrics") or {}).get("fidelity")
            rollout_p = None
            want = _key(lo, up)
            for anc in anchors_by_cls.get(cls, []):
                alo = anc.get("lower_bounds_normalized") or anc.get("lower_normalized")
                aup = anc.get("upper_bounds_normalized") or anc.get("upper_normalized")
                if alo is None or aup is None:
                    continue
                if _key(alo, aup) == want:
                    for pk in (
                        "precision_rollout_estimated", "precision",
                        "anchor_precision", "instance_precision",
                    ):
                        if anc.get(pk) is not None:
                            rollout_p = float(anc[pk])
                            break
                    break
            pairs.append({
                "class": cls,
                "rule_id": rule.get("rule_id"),
                "test_fid": _json_float(test_fid),
                "rollout_p": _json_float(rollout_p),
                "delta": (
                    None if test_fid is None or rollout_p is None
                    else float(rollout_p) - float(test_fid)
                ),
            })
    deltas = [p["delta"] for p in pairs if p.get("delta") is not None]
    return {
        "n": len(pairs),
        "mean_delta_rollout_minus_test": _mean(deltas),
        "pairs": pairs,
    }


def evaluate_instances(
    *,
    dataset: str,
    method: str,
    algo: str,
    seed: int,
    experiment_dir: str,
    classifier_path: str,
    out_dir: str,
    track_a_json: Optional[str] = None,
    rules_file: Optional[str] = None,
    tau_p: float = 0.90,
    max_per_pred: Optional[int] = MAX_PER_PRED_DEFAULT,
    n_perturb: int = N_PERTURB_DEFAULT,
    skip_anchors: bool = False,
    device: str = "cpu",
) -> str:
    from utils.dataset_factory import make_tabular_loader

    loader = make_tabular_loader(dataset, random_state=seed)
    loader.load_dataset()
    loader.preprocess_data()
    if not classifier_path or not os.path.exists(classifier_path):
        raise FileNotFoundError(f"classifier not found: {classifier_path}")
    loader.classifier = loader.load_classifier(filepath=classifier_path, device=device)

    env_data = loader.get_anchor_env_data()
    kind = "mada" if str(method).lower().startswith("mada") else "rlda"
    env_config = _env_yaml(kind, loader, experiment_dir, seed)
    predict_std = _predict_std_fn(loader, device)
    y_hat_test = predict_std(loader.X_test_scaled)
    y_test = np.asarray(loader.y_test)
    X_test_unit = np.asarray(loader.X_test_unit, dtype=np.float32)
    X_train_unit = np.asarray(loader.X_train_unit, dtype=np.float32)
    x_min = np.asarray(env_data["X_min"], dtype=np.float32)
    x_range = np.asarray(env_data["X_range"], dtype=np.float32)
    n_features = int(X_test_unit.shape[1])

    indices = sample_test_indices(y_hat_test, max_per_pred=max_per_pred, seed=seed)
    logger.info(
        "Scoring %s/%s test rows (max_per_pred=%s)  method=%s algo=%s",
        len(indices), len(y_test), max_per_pred, method, algo,
    )

    if kind == "rlda":
        models, env_data = _load_rlda_models(experiment_dir, loader, env_config, algo, device)
        mada_pack = None
    else:
        policies, agents, index, _ = _load_mada_policies(experiment_dir, device)
        mada_pack = (policies, agents, index)
        models = None

    rng = np.random.default_rng(int(seed) + 17)
    pi_rows: List[Dict[str, Any]] = []
    t_all = time.perf_counter()
    for k, idx in enumerate(indices):
        x_star = X_test_unit[int(idx)]
        yhat = int(y_hat_test[int(idx)])
        ytrue = int(y_test[int(idx)])
        if kind == "rlda":
            ep = _rollout_rlda(
                models=models, loader=loader, env_config=env_config, env_data=env_data,
                x_star_unit=x_star, y_hat=yhat, device=device,
                seed=int(seed) + int(idx),
            )
        else:
            policies, agents, index = mada_pack
            ep = _rollout_mada(
                policies=policies, agents=agents, index=index, loader=loader,
                env_config=env_config, env_data=env_data,
                x_star_unit=x_star, y_hat=yhat, device=device,
                seed=int(seed) + int(idx),
            )
        row = _score_episode(
            ep, env_config=env_config, n_features=n_features,
            x_star_unit=x_star, y=ytrue, y_hat=yhat,
            X_test_unit=X_test_unit, y_test=y_test, y_hat_test=y_hat_test,
            X_train_unit=X_train_unit, x_min=x_min, x_range=x_range,
            predict_std=predict_std, n_perturb=n_perturb, rng=rng,
        )
        row["index"] = int(idx)
        pi_rows.append(row)
        if (k + 1) % 25 == 0 or k == 0:
            logger.info("  π  %s/%s  ŷ=%s empty=%s emp_fid=%s",
                        k + 1, len(indices), yhat, row.get("empty_rule"), row.get("emp_fid"))
    wall_pi = time.perf_counter() - t_all

    anchors_rows: List[Dict[str, Any]] = []
    if not skip_anchors:
        logger.info("Classical Anchors on the same %s indices", len(indices))
        anchors_rows = run_anchors_on_indices(loader, indices, y_hat_test, tau_p, seed)

    track_a = None
    if track_a_json and os.path.exists(track_a_json):
        with open(track_a_json) as f:
            track_a = json.load(f)
    class_boxes = class_boxes_from_track_a(track_a) if track_a else {}
    lookup_rows = []
    for row in pi_rows:
        idx = int(row["index"])
        fired = lookup_fired_classes(X_test_unit[idx], class_boxes) if class_boxes else []
        yhat = int(row["y_hat"])
        lookup_rows.append({
            "index": idx,
            "y_hat": yhat,
            "n_fired": len(fired),
            "fired": fired,
            "abstain": len(fired) == 0,
            "conflict": len(fired) >= 2,
            "lookup_agrees_yhat": (len(fired) == 1 and fired[0] == yhat),
            "local_empty": bool(row.get("empty_rule")),
            "local_contains": bool(row.get("contains_x")),
        })

    def _rate(key, rows=lookup_rows):
        if not rows:
            return None
        return float(np.mean([1.0 if r.get(key) else 0.0 for r in rows]))

    queries_train = None
    if track_a:
        queries_train = (track_a.get("queries") or {}).get("n_blackbox_queries")
    mean_q_pi = _mean([r.get("queries") for r in pi_rows])
    mean_q_pi_perturb = _mean([r.get("perturb_queries") for r in pi_rows])
    mean_q_anc = _mean([r.get("queries") for r in anchors_rows]) if anchors_rows else None
    mean_t_pi = _mean([r.get("wall_s") for r in pi_rows])
    mean_t_anc = _mean([r.get("wall_s") for r in anchors_rows]) if anchors_rows else None

    rules_obj = None
    if rules_file and os.path.exists(rules_file):
        with open(rules_file) as f:
            rules_obj = json.load(f)
    b5 = _b5_train_vs_test(track_a, rules_obj) if track_a else {"n": 0, "pairs": []}

    payload = {
        "dataset": dataset,
        "method": method,
        "algo": algo,
        "seed": int(seed),
        "tau_p": float(tau_p),
        "git_commit": git_commit_hash(),
        "n_test": int(len(y_test)),
        "n_scored": int(len(indices)),
        "max_per_pred": max_per_pred,
        "n_perturb": int(n_perturb),
        "routing": "y_hat = f(x*)",
        "pi": {
            "summary": summarize_instance_rows(pi_rows),
            "wall_s_total": float(wall_pi),
            "rows": pi_rows,
        },
        "anchors": {
            "available": bool(anchors_rows),
            "summary": {
                "n": len(anchors_rows),
                "perturb_fid": _mean([r.get("perturb_fid") for r in anchors_rows]),
                "coverage": _mean([r.get("coverage") for r in anchors_rows]),
                "queries_per_x": _mean([r.get("queries") for r in anchors_rows]),
                "wall_s_per_x": _mean([r.get("wall_s") for r in anchors_rows]),
                "n_predicates": _mean([r.get("n_predicates") for r in anchors_rows]),
            } if anchors_rows else None,
            "rows": anchors_rows,
        },
        "cost": {
            "queries_train": queries_train,
            "queries_per_x_pi": mean_q_pi,
            "queries_per_x_pi_perturb": mean_q_pi_perturb,
            "queries_per_x_anchors": mean_q_anc,
            "note": (
                "queries_per_x_pi is serving cost (rollout CRN). "
                "queries_per_x_pi_perturb is Track B measurement only and is not in break-even."
            ),
            "wall_s_per_x_pi": mean_t_pi,
            "wall_s_per_x_anchors": mean_t_anc,
            "break_even_n_queries": (
                None if queries_train is None or mean_q_pi is None or mean_q_anc is None
                else break_even_n(float(queries_train), float(mean_q_pi), float(mean_q_anc))
            ),
            "break_even_n_seconds": (
                None if mean_t_pi is None or mean_t_anc is None
                else break_even_n(
                    float((track_a or {}).get("queries", {}).get("wall_train_seconds") or 0.0),
                    float(mean_t_pi), float(mean_t_anc),
                )
            ),
        },
        "lookup": {
            "n": len(lookup_rows),
            "abstain_rate": _rate("abstain"),
            "conflict_rate": _rate("conflict"),
            "lookup_yhat_agree": _rate("lookup_agrees_yhat"),
            "local_containment": _rate("local_contains"),
            "local_empty_rate": _rate("local_empty"),
            "rows": lookup_rows,
        },
        "train_vs_test_fid": b5,
        "experiment_dir": os.path.abspath(experiment_dir),
        "classifier_path": os.path.abspath(classifier_path),
        "track_a_json": os.path.abspath(track_a_json) if track_a_json else None,
        "rules_file": os.path.abspath(rules_file) if rules_file else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}__{method}__instances__seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote %s", path)
    s = payload["pi"]["summary"]["all"]
    logger.info(
        "π  perturb_fid=%s emp_fid=%s contain=%s empty=%s  q/x=%s  vs Anchors q/x=%s  N*=%s",
        s.get("perturb_fid"), s.get("emp_fid"), s.get("containment_rate"),
        s.get("empty_rule_rate"), mean_q_pi, mean_q_anc,
        payload["cost"]["break_even_n_queries"],
    )
    return path


def main():
    p = argparse.ArgumentParser(description="Track B: held-out instance eval (ŷ-routed, 1 rollout)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True, help="rlda / mada")
    p.add_argument("--algo", required=True, help="ddpg / sac / maddpg / masac")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_dir", required=True)
    p.add_argument("--classifier_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--track_a_json", default=None)
    p.add_argument("--rules_file", default=None)
    p.add_argument("--tau_p", type=float, default=0.90)
    p.add_argument("--max_per_pred", type=int, default=MAX_PER_PRED_DEFAULT)
    p.add_argument("--n_perturb", type=int, default=N_PERTURB_DEFAULT)
    p.add_argument("--skip_anchors", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate_instances(
        dataset=args.dataset, method=args.method, algo=args.algo, seed=args.seed,
        experiment_dir=args.experiment_dir, classifier_path=args.classifier_path,
        out_dir=args.out_dir, track_a_json=args.track_a_json, rules_file=args.rules_file,
        tau_p=args.tau_p, max_per_pred=args.max_per_pred, n_perturb=args.n_perturb,
        skip_anchors=args.skip_anchors, device=args.device,
    )


if __name__ == "__main__":
    main()
