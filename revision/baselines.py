"""
Class-level baselines for the revision (C-19, C-20, C-21, C-22).

All baselines emit the same result schema as revision.evaluate so they drop
straight into paper/make_tables.py and the C-14 global evaluator.

Usage:
    python -m revision.baselines --dataset iris --seed 42 --k 5 --out_dir revision/results
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))

from utils.eval_harness import (  # noqa: E402
    QueryCounter,
    evaluate_ruleset_as_classifier,
    per_class_block,
    reevaluate_ranked_rules,
    write_result_artifact,
)
from utils.metrics import (  # noqa: E402
    MIN_SUPPORT_DEFAULT,
    RANKING_SCORE_LCB_COVERAGE,
    RankedRule,
    box_mask,
    evaluate_mask,
    ranking_score,
    select_topk_union,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("revision.baselines")


def _load(dataset: str, seed: int):
    from BenchMARL.tabular_datasets import TabularDatasetLoader
    loader = TabularDatasetLoader(dataset_name=dataset, random_state=seed)
    loader.load_dataset()
    loader.preprocess_data()
    return loader


def _ensure_classifier(loader, classifier_path: str, device: str = "cpu"):
    """Load the exact black box used by RLDA/MADA; never retrain a baseline copy."""
    if not classifier_path or not os.path.exists(classifier_path):
        raise FileNotFoundError(
            "A trained RLDA/MADA classifier checkpoint is required for comparable "
            f"fidelity; not found: {classifier_path!r}"
        )
    loader.load_classifier(filepath=classifier_path, device=device)
    return loader.classifier


def _select_on_val_report_on_test(
    per_class_val: Dict[int, List[RankedRule]],
    *,
    X_test: np.ndarray,
    y_val: np.ndarray,
    y_hat_val: np.ndarray,
    y_test: np.ndarray,
    y_hat_test: np.ndarray,
    k: int,
    min_support: int,
) -> Dict[int, List[RankedRule]]:
    reported: Dict[int, List[RankedRule]] = {}
    for cls, rules in per_class_val.items():
        selected = select_topk_union(
            rules, y_val, y_hat_val, cls, k=k,
            class_conditional=True, min_support=min_support,
        )
        if selected is None:
            reported[cls] = []
            continue
        reported[cls] = reevaluate_ranked_rules(
            selected.individual,
            X_test,
            y_test,
            y_hat_test,
            cls,
            class_conditional=True,
            min_support=min_support,
        )
    return reported


def _predict(loader, X_std: np.ndarray) -> np.ndarray:
    import torch
    from utils.networks import predict_proba_torch
    clf = loader.classifier
    if hasattr(clf, "eval"):
        clf.eval()
    with torch.no_grad():
        probs = predict_proba_torch(clf, torch.from_numpy(np.asarray(X_std, dtype=np.float32))).cpu().numpy()
    return probs.argmax(axis=1)


def _split_arrays(loader, split: str):
    if split == "test":
        return loader.X_test, loader.X_test_scaled, loader.X_test_unit, loader.y_test
    if split == "val":
        if getattr(loader, "X_val", None) is None:
            raise ValueError("No val split")
        return loader.X_val, loader.X_val_scaled, loader.X_val_unit, loader.y_val
    return loader.X_train, loader.X_train_scaled, loader.X_train_unit, loader.y_train


def _emit(
    *,
    dataset, method, seed, tau_p, tau_c, out_dir, k, min_support,
    loader, y_eval, y_hat_eval, per_class_ranked: Dict[int, List[RankedRule]],
    queries: QueryCounter, extra: Dict[str, Any],
) -> str:
    # Artifact accuracy reporting performs one classifier pass per split.
    queries.add_queries(len(loader.X_train), reporting=True)
    if getattr(loader, "X_val_scaled", None) is not None:
        queries.add_queries(len(loader.X_val_scaled), reporting=True)
    queries.add_queries(len(loader.X_test), reporting=True)
    per_class_out = {}
    class_union_masks = {}
    class_union_fid = {}
    for cls, ranked in per_class_ranked.items():
        union = select_topk_union(
            ranked, y_eval, y_hat_eval, cls, k=len(ranked),
            class_conditional=True, min_support=min_support,
            enforce_min_support=False,
        )
        if union is None:
            continue
        inst = evaluate_mask(
            y=y_eval, y_hat=y_hat_eval, mask=union.best.mask, target_class=cls,
            class_conditional=False, min_support=min_support,
        )
        per_class_out[f"class_{cls}"] = per_class_block(union, instance_metrics=inst)
        umask = np.zeros(len(y_eval), dtype=bool)
        for r in union.individual:
            umask |= r.mask
        class_union_masks[cls] = umask
        class_union_fid[cls] = (
            union.union_metrics.fidelity if np.isfinite(union.union_metrics.fidelity) else -np.inf
        )
    global_res = evaluate_ruleset_as_classifier(class_union_masks, class_union_fid, y_eval, y_hat_eval)
    return write_result_artifact(
        out_dir,
        dataset=dataset,
        method=method,
        seed=seed,
        tau_p=tau_p,
        tau_c=tau_c,
        per_class=per_class_out,
        global_ruleset=global_res.to_dict(),
        classifier_accuracy={
            "train_accuracy": float((_predict(loader, loader.X_train_scaled) == loader.y_train).mean()),
            "val_accuracy": (
                float((_predict(loader, loader.X_val_scaled) == loader.y_val).mean())
                if getattr(loader, "X_val_scaled", None) is not None else None
            ),
            "test_accuracy": float((_predict(loader, loader.X_test_scaled) == loader.y_test).mean()),
            "n": int(len(y_eval)),
        },
        queries=queries.to_dict(),
        n_covered_note=(
            f"Baseline {method}; selection on D_val and reporting on D_test; "
            f"union over top-k={k} ranked by {RANKING_SCORE_LCB_COVERAGE}; "
            f"min_support={min_support}."
        ),
        extra=extra,
        compactness=_compactness_summary(per_class_out),
        ranking_formula=RANKING_SCORE_LCB_COVERAGE,
        min_support=min_support,
    )


def _compactness_summary(per_class_out: Dict[str, Any]) -> Dict[str, Any]:
    acts = [
        (b.get("compactness") or {}).get("mean_active_features")
        for b in per_class_out.values()
    ]
    acts = [a for a in acts if a is not None and np.isfinite(a)]
    return {
        "mean_rules_per_class": (
            float(np.mean([b.get("n_selected", b.get("k", 0)) for b in per_class_out.values()]))
            if per_class_out else 0.0
        ),
        "mean_active_features": float(np.mean(acts)) if acts else None,
        "sparsity_width_ratio": 0.95,
    }


# ---------------------------------------------------------------------------
# C-21 — depth-limited CART surrogate on model predictions
# ---------------------------------------------------------------------------

def run_cart(
    dataset: str, seed: int, k: int, tau_p: float, tau_c: float, out_dir: str,
    classifier_path: str,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> str:
    from sklearn.tree import DecisionTreeClassifier, _tree

    loader = _load(dataset, seed)
    _ensure_classifier(loader, classifier_path)
    # Train on D_train predictions, rank leaves on D_val, report on D_test.
    y_hat_train = _predict(loader, loader.X_train_scaled)
    # Leaf count matched to k * n_classes (one path-rule per class, k of them)
    n_classes = int(loader.n_classes)
    max_leaf = max(n_classes, k * n_classes)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf, random_state=seed)
    t0 = time.time()
    tree.fit(loader.X_train, y_hat_train)  # original units, like printed rules
    queries = QueryCounter()
    queries.wall_train_s = time.time() - t0
    queries.add_queries(len(loader.X_train))  # one f_hat call per train row

    X_val, X_val_std, _, y_val = _split_arrays(loader, "val")
    X_test, X_test_std, _, y_test = _split_arrays(loader, "test")
    y_hat_val = _predict(loader, X_val_std)
    y_hat_test = _predict(loader, X_test_std)
    queries.add_queries(len(X_val))
    queries.add_queries(len(X_test), reporting=True)

    feature_names = list(loader.feature_names)
    t = tree.tree_
    per_class_ranked: Dict[int, List[RankedRule]] = {c: [] for c in range(n_classes)}

    def recurse(node, lower, upper, path):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            counts = t.value[node][0]
            pred_cls = int(np.argmax(counts))
            lo = np.array(lower, dtype=np.float32)
            up = np.array(upper, dtype=np.float32)
            mask = box_mask(X_val, lo, up)
            metrics = evaluate_mask(
                y=y_val, y_hat=y_hat_val, mask=mask, target_class=pred_cls,
                class_conditional=True, min_support=min_support,
            )
            display = " and ".join(path) if path else "any values"
            per_class_ranked[pred_cls].append(RankedRule(
                rule_id=f"cart:{pred_cls}:{node}",
                lower=lo, upper=up, mask=mask, metrics=metrics,
                score=ranking_score(
                    metrics.fidelity, metrics.coverage,
                    formula=RANKING_SCORE_LCB_COVERAGE,
                    n_covered=metrics.n_covered,
                ),
                display_rule=display,
            ))
            return
        feat = t.feature[node]
        thr = t.threshold[node]
        name = feature_names[feat]
        left_up = list(upper); left_up[feat] = min(left_up[feat], thr)
        right_lo = list(lower); right_lo[feat] = max(right_lo[feat], thr)
        recurse(t.children_left[node], lower, left_up, path + [f"{name} <= {thr:.6f}"])
        recurse(t.children_right[node], right_lo, upper, path + [f"{name} > {thr:.6f}"])

    lo0 = np.min(loader.X_train, axis=0).tolist()
    up0 = np.max(loader.X_train, axis=0).tolist()
    recurse(0, lo0, up0, [])

    per_class_reported = _select_on_val_report_on_test(
        per_class_ranked,
        X_test=X_test,
        y_val=y_val,
        y_hat_val=y_hat_val,
        y_test=y_test,
        y_hat_test=y_hat_test,
        k=k,
        min_support=min_support,
    )

    return _emit(
        dataset=dataset, method="cart", seed=seed, tau_p=tau_p, tau_c=tau_c,
        out_dir=out_dir, k=k, min_support=min_support,
        loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
        per_class_ranked=per_class_reported, queries=queries,
        extra={
            "max_leaf_nodes": max_leaf,
            "k": k,
            "classifier_path": os.path.abspath(classifier_path),
            "selection_split": "val",
            "report_split": "test",
        },
    )


# ---------------------------------------------------------------------------
# C-20 — greedy set-cover union of instance boxes (Anchors or any box list)
# ---------------------------------------------------------------------------

def greedy_set_cover(
    ranked: List[RankedRule],
    y: np.ndarray,
    target_class: int,
    k: int,
    tau_p: float,
) -> List[RankedRule]:
    """Greedily add rules that maximize marginal class coverage subject to Fid >= tau_P."""
    class_idx = set(np.where(y == target_class)[0].tolist())
    eligible = [r for r in ranked if np.isfinite(r.metrics.fidelity) and r.metrics.fidelity + 1e-12 >= tau_p]
    if not eligible:
        eligible = sorted(ranked, key=lambda r: r.metrics.fidelity if np.isfinite(r.metrics.fidelity) else -1, reverse=True)
    selected = []
    covered = set()
    remaining = list(eligible)
    while remaining and len(selected) < k:
        best = None
        best_gain = -1
        for r in remaining:
            idxs = set(np.where(r.mask)[0].tolist()) & class_idx
            gain = len(idxs - covered)
            if gain > best_gain:
                best_gain = gain
                best = r
        if best is None or best_gain <= 0:
            break
        selected.append(best)
        covered |= set(np.where(best.mask)[0].tolist()) & class_idx
        remaining = [r for r in remaining if r.rule_id != best.rule_id]
    return selected


# ---------------------------------------------------------------------------
# C-19 — SP-Anchors (submodular pick over per-instance Anchors)
# ---------------------------------------------------------------------------

def _try_import_anchor():
    try:
        from anchor import anchor_tabular
        return anchor_tabular
    except ImportError:
        return None


def run_anchors_family(
    dataset: str, seed: int, k: int, tau_p: float, tau_c: float, out_dir: str,
    classifier_path: str,
    budget_per_class: int = 20,
    query_budget: Optional[int] = None,
    min_support: int = MIN_SUPPORT_DEFAULT,
    methods: Sequence[str] = ("sp_anchors", "greedy_anchors"),
) -> List[str]:
    """Generate per-instance Anchors on D_val, pick a subset, evaluate on D_test.

    SP-Anchors (C-19): submodular pick for coverage diversity.
    Greedy set-cover (C-20): marginal class coverage with Fid >= tau_P.
    """
    anchor_tabular = _try_import_anchor()
    if anchor_tabular is None:
        logger.error("anchor-exp is not installed; skipping SP-Anchors / greedy Anchors. pip install anchor-exp")
        return []

    loader = _load(dataset, seed)
    _ensure_classifier(loader, classifier_path)
    if getattr(loader, "X_val", None) is None:
        raise ValueError("SP-Anchors requires a val split (set val_size > 0)")

    X_val, X_val_std, _, y_val = _split_arrays(loader, "val")
    X_test, X_test_std, _, y_test = _split_arrays(loader, "test")
    y_hat_val = _predict(loader, X_val_std)
    y_hat_test = _predict(loader, X_test_std)

    feature_names = list(loader.feature_names)
    class_names = [str(c) for c in range(loader.n_classes)]

    queries = QueryCounter()

    class QueryBudgetExceeded(RuntimeError):
        pass

    def predict_fn(X_raw):
        # X_raw is original units; scale then classify
        X_raw = np.asarray(X_raw, dtype=np.float32)
        n = int(X_raw.shape[0])
        if query_budget is not None and queries.n_queries + n > query_budget:
            raise QueryBudgetExceeded(
                f"Anchor query budget exhausted ({queries.n_queries}/{query_budget})"
            )
        queries.add_queries(n)
        Xs = loader.scaler.transform(X_raw)
        return _predict(loader, Xs)

    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_names,
        train_data=loader.X_train,
    )

    queries.add_queries(len(X_val))
    queries.add_queries(len(X_test), reporting=True)
    per_class_boxes: Dict[int, List[RankedRule]] = {c: [] for c in range(loader.n_classes)}

    rng = np.random.default_rng(seed)
    t0 = time.time()
    budget_exhausted = False
    for cls in range(loader.n_classes):
        idx = np.where(y_hat_val == cls)[0]
        if idx.size == 0:
            continue
        take = min(int(budget_per_class), int(idx.size))
        chosen = rng.choice(idx, size=take, replace=False)
        for j, row_i in enumerate(chosen):
            if query_budget is not None and queries.n_queries >= query_budget:
                budget_exhausted = True
                break
            instance = X_val[row_i]
            try:
                kwargs = {"threshold": float(tau_p)}
                if query_budget is not None:
                    kwargs["max_samples"] = max(
                        1, query_budget - queries.n_queries
                    )
                exp = explainer.explain_instance(instance, predict_fn, **kwargs)
            except QueryBudgetExceeded:
                budget_exhausted = True
                break
            except Exception as e:
                logger.warning("Anchors failed on class %s instance %s: %s", cls, row_i, e)
                continue
            # Convert predicate list into a box in original units
            lo = np.min(loader.X_train, axis=0).astype(np.float32).copy()
            up = np.max(loader.X_train, axis=0).astype(np.float32).copy()
            names = list(getattr(exp, "names", lambda: [])() if callable(getattr(exp, "names", None)) else getattr(exp, "names", []))
            # Best-effort parse of "feature > v" / "feature <= v" / "feature = v"
            for pred in names:
                _apply_anchor_predicate(pred, feature_names, lo, up)
            mask_val = box_mask(X_val, lo, up)
            metrics = evaluate_mask(
                y=y_val, y_hat=y_hat_val, mask=mask_val, target_class=cls,
                class_conditional=True, min_support=min_support,
            )
            display = " and ".join(str(p) for p in names) if names else "any values"
            per_class_boxes[cls].append(RankedRule(
                rule_id=f"anchor:{cls}:{int(row_i)}",
                lower=lo, upper=up, mask=mask_val, metrics=metrics,
                score=ranking_score(
                    metrics.fidelity, metrics.coverage,
                    formula=RANKING_SCORE_LCB_COVERAGE,
                    n_covered=metrics.n_covered,
                ),
                display_rule=display,
            ))
        if budget_exhausted:
            logger.info("Anchor query budget exhausted after class %s", cls)
            break
    queries.wall_infer_s = time.time() - t0

    written = []
    if "sp_anchors" in methods:
        picked_val = {
            cls: _submodular_pick(rules, y_val, cls, k)
            for cls, rules in per_class_boxes.items()
        }
        picked = _select_on_val_report_on_test(
            picked_val, X_test=X_test, y_val=y_val, y_hat_val=y_hat_val,
            y_test=y_test, y_hat_test=y_hat_test, k=k,
            min_support=min_support,
        )
        written.append(_emit(
            dataset=dataset, method="sp_anchors", seed=seed, tau_p=tau_p, tau_c=tau_c,
            out_dir=out_dir, k=k, min_support=min_support,
            loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
            per_class_ranked=picked, queries=queries,
            extra={
                "budget_per_class": budget_per_class,
                "query_budget": query_budget,
                "budget_exhausted": budget_exhausted,
                "k": k,
                "picker": "submodular",
                "classifier_path": os.path.abspath(classifier_path),
                "selection_split": "val",
                "report_split": "test",
            },
        ))
    if "greedy_anchors" in methods:
        picked_val = {
            cls: greedy_set_cover(rules, y_val, cls, k, tau_p)
            for cls, rules in per_class_boxes.items()
        }
        picked = _select_on_val_report_on_test(
            picked_val, X_test=X_test, y_val=y_val, y_hat_val=y_hat_val,
            y_test=y_test, y_hat_test=y_hat_test, k=k,
            min_support=min_support,
        )
        # greedy_set_cover returns a subset; wrap as ranked list (already scored)
        written.append(_emit(
            dataset=dataset, method="greedy_anchors", seed=seed, tau_p=tau_p, tau_c=tau_c,
            out_dir=out_dir, k=k, min_support=min_support,
            loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
            per_class_ranked=picked, queries=queries,
            extra={
                "budget_per_class": budget_per_class,
                "query_budget": query_budget,
                "budget_exhausted": budget_exhausted,
                "k": k,
                "picker": "greedy_set_cover",
                "classifier_path": os.path.abspath(classifier_path),
                "selection_split": "val",
                "report_split": "test",
            },
        ))
    return written


def _apply_anchor_predicate(pred: str, feature_names: List[str], lo: np.ndarray, up: np.ndarray) -> None:
    """Best-effort conversion of an Anchors predicate into interval bounds."""
    import re
    s = str(pred)
    for i, name in enumerate(feature_names):
        if name not in s:
            continue
        m = re.search(rf"{re.escape(name)}\s*<=\s*([-+eE0-9.]+)", s)
        if m:
            up[i] = min(up[i], float(m.group(1)))
        m = re.search(rf"{re.escape(name)}\s*>=\s*([-+eE0-9.]+)", s)
        if m:
            lo[i] = max(lo[i], float(m.group(1)))
        m = re.search(rf"{re.escape(name)}\s*<\s*([-+eE0-9.]+)", s)
        if m:
            up[i] = min(up[i], float(m.group(1)))
        m = re.search(rf"{re.escape(name)}\s*>\s*([-+eE0-9.]+)", s)
        if m:
            lo[i] = max(lo[i], np.nextafter(float(m.group(1)), np.inf))
        m = re.search(rf"{re.escape(name)}\s*=\s*([-+eE0-9.]+)", s)
        if m and "<" not in s and ">" not in s:
            v = float(m.group(1))
            lo[i] = v
            up[i] = v


def _submodular_pick(rules: List[RankedRule], y: np.ndarray, cls: int, k: int) -> List[RankedRule]:
    """SP-LIME style: iteratively pick the rule covering the most still-uncovered class samples,
    weighted by fidelity (Ribeiro et al. submodular pick)."""
    if not rules:
        return []
    class_idx = np.where(y == cls)[0]
    n = len(y)
    covered = np.zeros(n, dtype=bool)
    selected = []
    remaining = list(rules)
    while remaining and len(selected) < k:
        best = None
        best_gain = -1.0
        for r in remaining:
            new = r.mask & ~covered
            new_class = int((new[class_idx]).sum()) if class_idx.size else int(new.sum())
            fid = r.metrics.fidelity if np.isfinite(r.metrics.fidelity) else 0.0
            gain = fid * new_class
            if gain > best_gain:
                best_gain = gain
                best = r
        if best is None or best_gain <= 0:
            break
        selected.append(best)
        covered |= best.mask
        remaining = [r for r in remaining if r.rule_id != best.rule_id]
    return selected


# ---------------------------------------------------------------------------
# C-22 — non-RL box optimizer (random search over (ℓ, u), matched query budget)
# ---------------------------------------------------------------------------

def run_random_search(
    dataset: str, seed: int, k: int, tau_p: float, tau_c: float, out_dir: str,
    classifier_path: str,
    n_candidates: int = 512,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> str:
    """Random axis-aligned boxes, scored on D_val, reported on D_test.

    Query budget is n_candidates * |D_val| classifier lookups — but we use the
    cached y_hat_val, so the black-box cost is one pass over val (+ train for the
    classifier). This is the honest non-amortized optimizer baseline.
    """
    loader = _load(dataset, seed)
    _ensure_classifier(loader, classifier_path)
    if getattr(loader, "X_val_unit", None) is None:
        raise ValueError("random_search requires a val split")

    _, X_val_std, X_val_unit, y_val = _split_arrays(loader, "val")
    _, X_test_std, X_test_unit, y_test = _split_arrays(loader, "test")
    y_hat_val = _predict(loader, X_val_std)
    y_hat_test = _predict(loader, X_test_std)
    queries = QueryCounter()
    queries.add_queries(len(X_val_unit))
    queries.add_queries(len(X_test_unit), reporting=True)

    rng = np.random.default_rng(seed)
    d = X_val_unit.shape[1]
    per_class_ranked: Dict[int, List[RankedRule]] = {c: [] for c in range(loader.n_classes)}
    t0 = time.time()
    for cls in range(loader.n_classes):
        cls_pts = X_val_unit[y_hat_val == cls]
        if cls_pts.shape[0] == 0:
            continue
        for i in range(n_candidates):
            # Sample a seed point of this class and a random width in [min_w, 1]
            seed_pt = cls_pts[rng.integers(0, len(cls_pts))]
            width = rng.uniform(0.05, 0.6, size=d).astype(np.float32)
            lo = np.clip(seed_pt - width / 2.0, 0.0, 1.0)
            up = np.clip(seed_pt + width / 2.0, 0.0, 1.0)
            up = np.maximum(up, lo + 0.05)
            up = np.clip(up, 0.0, 1.0)
            mask_val = box_mask(X_val_unit, lo, up)
            m_val = evaluate_mask(
                y=y_val, y_hat=y_hat_val, mask=mask_val, target_class=cls,
                class_conditional=True, min_support=min_support,
            )
            # Candidate metrics and scores are validation-only.
            per_class_ranked[cls].append(RankedRule(
                rule_id=f"rs:{cls}:{i}",
                lower=lo, upper=up, mask=mask_val, metrics=m_val,
                score=ranking_score(
                    m_val.fidelity, m_val.coverage,
                    formula=RANKING_SCORE_LCB_COVERAGE,
                    n_covered=m_val.n_covered,
                ),
                display_rule=f"random box {i}",
            ))
    queries.wall_infer_s = time.time() - t0
    per_class_reported = _select_on_val_report_on_test(
        per_class_ranked,
        X_test=X_test_unit,
        y_val=y_val,
        y_hat_val=y_hat_val,
        y_test=y_test,
        y_hat_test=y_hat_test,
        k=k,
        min_support=min_support,
    )
    return _emit(
        dataset=dataset, method="random_search", seed=seed, tau_p=tau_p, tau_c=tau_c,
        out_dir=out_dir, k=k, min_support=min_support,
        loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
        per_class_ranked=per_class_reported, queries=queries,
        extra={
            "n_candidates": n_candidates,
            "k": k,
            "classifier_path": os.path.abspath(classifier_path),
            "selection_split": "val",
            "report_split": "test",
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--tau_p", type=float, default=0.90)
    p.add_argument("--tau_c", type=float, default=0.20)
    p.add_argument("--out_dir", default="revision/results")
    p.add_argument(
        "--classifier_path",
        required=True,
        help="Exact classifier.pth checkpoint used by RLDA/MADA.",
    )
    p.add_argument(
        "--methods", nargs="+",
        default=["cart", "random_search"],
        choices=["cart", "random_search", "sp_anchors", "greedy_anchors"],
    )
    p.add_argument("--budget_per_class", type=int, default=20)
    p.add_argument(
        "--query_budget",
        type=int,
        default=None,
        help="Maximum generation/selection black-box calls for Anchor baselines.",
    )
    p.add_argument("--n_candidates", type=int, default=512)
    args = p.parse_args()

    written = []
    if "cart" in args.methods:
        written.append(run_cart(
            args.dataset, args.seed, args.k, args.tau_p, args.tau_c,
            args.out_dir, args.classifier_path,
        ))
    if "random_search" in args.methods:
        written.append(run_random_search(
            args.dataset, args.seed, args.k, args.tau_p, args.tau_c, args.out_dir,
            args.classifier_path, n_candidates=args.n_candidates,
        ))
    anchor_methods = [m for m in args.methods if m in ("sp_anchors", "greedy_anchors")]
    if anchor_methods:
        written.extend(run_anchors_family(
            args.dataset, args.seed, args.k, args.tau_p, args.tau_c, args.out_dir,
            args.classifier_path,
            budget_per_class=args.budget_per_class,
            query_budget=args.query_budget,
            methods=anchor_methods,
        ))
    for w in written:
        logger.info("Wrote %s", w)


if __name__ == "__main__":
    main()
