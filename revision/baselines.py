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
import dataclasses
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
    active_feature_mask,
    get_coverage_basis,
    RANKING_SCORE_LCB_COVERAGE,
    RANKING_SCORE_LCB_GATED,
    RankedRule,
    box_mask,
    evaluate_mask,
    ranking_score,
    select_topk_union,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("revision.baselines")


def _load(dataset: str, seed: int):
    from utils.dataset_factory import make_tabular_loader
    loader = make_tabular_loader(dataset, random_state=seed)
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
    # load_classifier returns the model but does not assign loader.classifier.
    loader.classifier = loader.load_classifier(filepath=classifier_path, device=device)
    return loader.classifier


FID_EMPIRICAL, FID_PERTURBED = "empirical", "perturbed"


def _perturbed_fid_scorer(loader, space: str, n_samples: int, sparsity_width_ratio: float = 0.95):
    """Score a candidate box by Fid under Anchors' D(z|B) instead of real rows.

    CART and random search rank candidates by agreement with f_hat on the real
    D_val rows inside the box; the Anchors family already searches on perturbed
    precision (`explain_instance(threshold=tau_P)`). Without this, comparing all
    of them on perturbed Fid measures two different objectives. Coverage stays on
    real rows either way -- perturbed samples have no coverage semantics.

    Returns None when the estimator is empirical, else fn(lower, upper, cls, rng).
    """
    from revision.dual_estimator_rescore import sample_anchors_conditional

    pool = np.asarray(
        loader.X_train_unit if space == "unit" else loader.X_train, dtype=np.float32
    )
    if space == "unit":
        X_min = np.asarray(loader.X_min, dtype=np.float64)
        X_range = np.asarray(loader.X_range, dtype=np.float64)
        to_std = lambda Z: np.asarray(Z, dtype=np.float64) * X_range + X_min  # noqa: E731
        f_min = f_max = None
    else:
        mean = np.asarray(loader.scaler.mean_, dtype=np.float64)
        scale = np.asarray(loader.scaler.scale_, dtype=np.float64)
        to_std = lambda Z: (np.asarray(Z, dtype=np.float64) - mean) / scale  # noqa: E731
        f_min = np.min(pool, axis=0).astype(np.float64)
        f_max = np.max(pool, axis=0).astype(np.float64)

    def score(lower, upper, cls: int, rng) -> float:
        lower = np.asarray(lower, dtype=np.float32).reshape(-1)
        upper = np.asarray(upper, dtype=np.float32).reshape(-1)
        active = active_feature_mask(
            lower, upper, sparsity_width_ratio=sparsity_width_ratio,
            feature_min=f_min, feature_max=f_max,
        )
        z, _ = sample_anchors_conditional(pool, lower, upper, active, n_samples, rng)
        preds = _predict(loader, to_std(z).astype(np.float32))
        return float((preds == int(cls)).mean())

    return score


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
    queries: QueryCounter, extra: Dict[str, Any], box_space: str = "unit",
    ranking_formula: str = RANKING_SCORE_LCB_COVERAGE,
) -> str:
    # Compactness compares a box side against the feature's full range. The RL
    # arms build boxes in unit space, where that range is [0, 1] and the default
    # holds. CART and the Anchors family build boxes in ORIGINAL feature units
    # and seed the root at X_train's min/max, so their range is X_train's own
    # span -- without it, an unconstrained iris dimension (width 3.6 cm) is read
    # as constrained and a real one-sided split reported 0 active features,
    # while breast_cancer's small-magnitude features reported all 16 as active.
    if box_space == "original":
        feature_min = np.min(loader.X_train, axis=0)
        feature_max = np.max(loader.X_train, axis=0)
    elif box_space == "unit":
        feature_min = feature_max = None
    else:
        raise ValueError(f"box_space must be 'original' or 'unit', got {box_space!r}")
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
        per_class_out[f"class_{cls}"] = per_class_block(
            union, instance_metrics=inst,
            feature_min=feature_min, feature_max=feature_max,
        )
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
            f"union over top-k={k} ranked by {ranking_formula}; "
            f"min_support={min_support}."
        ),
        extra={**extra, "coverage_basis": get_coverage_basis()},
        compactness=_compactness_summary(per_class_out),
        ranking_formula=ranking_formula,
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


def _meter_for(loader) -> "QueryMeter":
    """Meter every baseline on the marginal-call unit, alongside the legacy counts.

    The legacy `n_blackbox_queries` bills cached table re-reads and is not
    comparable across methods; `marginal_cost.marginal_rows` is.
    """
    from utils.query_meter import QueryMeter

    refs = [
        getattr(loader, n, None)
        for n in ("X_train_scaled", "X_val_scaled", "X_test_scaled")
    ]
    return QueryMeter([r for r in refs if r is not None])


def run_cart(
    dataset: str, seed: int, k: int, tau_p: float, tau_c: float, out_dir: str,
    classifier_path: str,
    min_support: int = MIN_SUPPORT_DEFAULT,
    ranking_formula: str = RANKING_SCORE_LCB_COVERAGE,
    fid_estimator: str = FID_EMPIRICAL,
    perturb_samples: int = 512,
) -> str:
    from sklearn.tree import DecisionTreeClassifier, _tree

    loader = _load(dataset, seed)
    _ensure_classifier(loader, classifier_path)
    meter = _meter_for(loader)
    meter.__enter__()
    _t_construct = time.time()
    # Train on D_train predictions, rank leaves on D_val, report on D_test.
    y_hat_train = _predict(loader, loader.X_train_scaled)
    # Leaf count matched to k * n_classes (one path-rule per class, k of them)
    n_classes = int(loader.n_classes)
    max_leaf = max(n_classes, k * n_classes)
    tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf, random_state=seed)
    t0 = time.time()
    queries = QueryCounter()
    X_fit, y_fit = loader.X_train, y_hat_train
    queries.add_queries(len(loader.X_train))  # one f_hat call per train row
    if fid_estimator == FID_PERTURBED:
        # At k=1 the tree has one leaf per class, so ranking leaves on perturbed
        # Fid changes nothing -- there is nothing to choose between. The fit is
        # what has to move: augment D_train with coordinate-independent draws
        # from the per-feature train marginals, labelled by f_hat. That is the
        # same recombination D(z|B) samples from, so the surrogate is fit to
        # agree off the data manifold as well as on it.
        arng = np.random.default_rng(seed)
        Xtr = np.asarray(loader.X_train, dtype=np.float64)
        Z = np.column_stack([
            Xtr[arng.integers(0, Xtr.shape[0], size=Xtr.shape[0]), j]
            for j in range(Xtr.shape[1])
        ])
        y_hat_aug = _predict(loader, loader.scaler.transform(Z))
        queries.add_queries(len(Z))
        X_fit = np.vstack([Xtr, Z])
        y_fit = np.concatenate([y_hat_train, y_hat_aug])
        logger.info("CART perturbed fit: %s real + %s recombined rows", len(Xtr), len(Z))
    tree.fit(X_fit, y_fit)  # original units, like printed rules
    queries.wall_train_s = time.time() - t0

    X_val, X_val_std, _, y_val = _split_arrays(loader, "val")
    X_test, X_test_std, _, y_test = _split_arrays(loader, "test")
    y_hat_val = _predict(loader, X_val_std)
    y_hat_test = _predict(loader, X_test_std)
    queries.add_queries(len(X_val))
    queries.add_queries(len(X_test), reporting=True)

    feature_names = list(loader.feature_names)
    t = tree.tree_
    per_class_ranked: Dict[int, List[RankedRule]] = {c: [] for c in range(n_classes)}
    pfid = (_perturbed_fid_scorer(loader, "original", perturb_samples)
            if fid_estimator == FID_PERTURBED else None)
    prng = np.random.default_rng(seed)

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
            sel_fid = metrics.fidelity if pfid is None else pfid(lo, up, pred_cls, prng)
            per_class_ranked[pred_cls].append(RankedRule(
                rule_id=f"cart:{pred_cls}:{node}",
                lower=lo, upper=up, mask=mask, metrics=metrics,
                score=ranking_score(
                    sel_fid, metrics.coverage,
                    formula=ranking_formula,
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

    queries.wall_construct_s = time.time() - _t_construct
    queries.attach_meter(meter); meter.__exit__(None, None, None)
    return _emit(
        dataset=dataset, method="cart", seed=seed, tau_p=tau_p, tau_c=tau_c, ranking_formula=ranking_formula, box_space="original",
        out_dir=out_dir, k=k, min_support=min_support,
        loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
        per_class_ranked=per_class_reported, queries=queries,
        extra={
            "max_leaf_nodes": max_leaf,
            "k": k,
            "fid_estimator": fid_estimator,
            "perturb_samples": int(perturb_samples) if fid_estimator == FID_PERTURBED else 0,
            "cart_fit_rows": int(len(X_fit)),
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
    k_values: Optional[Sequence[int]] = None,
    ranking_formulas: Sequence[str] = (RANKING_SCORE_LCB_COVERAGE,),
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
    meter = _meter_for(loader)
    meter.__enter__()
    _t_construct = time.time()
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
                    formula=ranking_formulas[0],
                    n_covered=metrics.n_covered,
                ),
                display_rule=display,
            ))
        if budget_exhausted:
            logger.info("Anchor query budget exhausted after class %s", cls)
            break
    queries.wall_infer_s = time.time() - t0

    queries.wall_construct_s = time.time() - _t_construct
    queries.attach_meter(meter)
    meter.__exit__(None, None, None)
    written = []
    base_boxes = per_class_boxes
    for formula in ranking_formulas:
        # Same pool for every formula: only the candidate score changes.
        per_class_boxes = {
            cls: [dataclasses.replace(r, score=ranking_score(
                r.metrics.fidelity, r.metrics.coverage, formula=formula,
                n_covered=r.metrics.n_covered, tau_p=tau_p,
            )) for r in rules]
            for cls, rules in base_boxes.items()
        }
        f_out = out_dir if len(ranking_formulas) == 1 else os.path.join(out_dir, formula)
        # One pool, many k. `anchor-exp`'s internal sampling is not seeded by our
        # seed, so regenerating the pool per k would confound the union-size sweep
        # with explainer noise (measured at +-0.014 Eff run-to-run). Reduce the same
        # `per_class_boxes` at every k instead; the k subdirectory is the only thing
        # that changes downstream.
        for k_i in (k_values if k_values else [k]):
            k_out = f_out if len(k_values or [k]) == 1 else os.path.join(f_out, f"k{k_i}")
            common_extra = {
                "budget_per_class": budget_per_class,
                "query_budget": query_budget,
                "budget_exhausted": budget_exhausted,
                "k": k_i,
                "pool_per_class": {
                    str(c): len(v) for c, v in per_class_boxes.items()
                },
                "pool_shared_across_k": bool(k_values and len(k_values) > 1),
                "classifier_path": os.path.abspath(classifier_path),
                "selection_split": "val",
                "report_split": "test",
            }
            if "sp_anchors" in methods:
                picked_val = {
                    cls: _submodular_pick(rules, y_val, cls, k_i)
                    for cls, rules in per_class_boxes.items()
                }
                picked = _select_on_val_report_on_test(
                    picked_val, X_test=X_test, y_val=y_val, y_hat_val=y_hat_val,
                    y_test=y_test, y_hat_test=y_hat_test, k=k_i,
                    min_support=min_support,
                )
                written.append(_emit(
                    dataset=dataset, method="sp_anchors", seed=seed, tau_p=tau_p,
                    tau_c=tau_c, box_space="original",
                    out_dir=k_out, k=k_i, min_support=min_support,
                    loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
                    per_class_ranked=picked, queries=queries, ranking_formula=formula,
                    extra={**common_extra, "picker": "submodular"},
                ))
            if "greedy_anchors" in methods:
                picked_val = {
                    cls: greedy_set_cover(rules, y_val, cls, k_i, tau_p)
                    for cls, rules in per_class_boxes.items()
                }
                picked = _select_on_val_report_on_test(
                    picked_val, X_test=X_test, y_val=y_val, y_hat_val=y_hat_val,
                    y_test=y_test, y_hat_test=y_hat_test, k=k_i,
                    min_support=min_support,
                )
                written.append(_emit(
                    dataset=dataset, method="greedy_anchors", seed=seed, tau_p=tau_p,
                    tau_c=tau_c, box_space="original",
                    out_dir=k_out, k=k_i, min_support=min_support,
                    loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
                    per_class_ranked=picked, queries=queries, ranking_formula=formula,
                    extra={**common_extra, "picker": "greedy_set_cover"},
                ))
    return written


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _apply_anchor_predicate(pred: str, feature_names: List[str], lo: np.ndarray, up: np.ndarray) -> None:
    """Convert one Anchors predicate into interval bounds on (lo, up).

    `anchor-exp` discretizes continuous features, so a predicate is one of

        feat <= v        feat > v         feat >= v        feat = v
        a < feat <= b    a <= feat <= b                    (two-sided bin)

    The two-sided form is the one that matters: it is ~13% of the predicates the
    explainer emits on these datasets, and the value sits on the *left* of the
    feature name. Matching only `feat <op> value` silently dropped every such
    lower bound and left the box at the training minimum on that feature, so the
    baseline's boxes were strictly wider than the anchors they encode.

    Feature names are matched longest-first and must not be part of a longer
    identifier, so `road_temp_set_1 <= 5` cannot also bind a feature named `temp`.
    """
    import re

    s = str(pred)
    order = sorted(range(len(feature_names)), key=lambda i: -len(str(feature_names[i])))
    bound_this_pred = False
    for i in order:
        name = str(feature_names[i])
        if name not in s or bound_this_pred:
            continue
        nm = rf"(?<![\w.]){re.escape(name)}(?![\w.])"

        def _lower(v: float, strict: bool) -> None:
            lo[i] = max(lo[i], np.nextafter(v, np.inf) if strict else v)

        def _upper(v: float, strict: bool) -> None:
            # Boxes are closed, so a strict `<` is represented by the value
            # itself; shrinking below it would drop rows the anchor accepts.
            up[i] = min(up[i], v)

        matched = False
        # value-on-the-left forms first: "a < feat <= b", "a <= feat < b"
        m = re.search(rf"({_NUM})\s*(<=?)\s*{nm}", s)
        if m:
            _lower(float(m.group(1)), strict=(m.group(2) == "<"))
            matched = True
        m = re.search(rf"({_NUM})\s*(>=?)\s*{nm}", s)
        if m:
            _upper(float(m.group(1)), strict=(m.group(2) == ">"))
            matched = True
        # name-on-the-left forms
        m = re.search(rf"{nm}\s*<=\s*({_NUM})", s)
        if m:
            _upper(float(m.group(1)), strict=False)
            matched = True
        elif (m := re.search(rf"{nm}\s*<\s*({_NUM})", s)):
            _upper(float(m.group(1)), strict=True)
            matched = True
        m = re.search(rf"{nm}\s*>=\s*({_NUM})", s)
        if m:
            _lower(float(m.group(1)), strict=False)
            matched = True
        elif (m := re.search(rf"{nm}\s*>\s*({_NUM})", s)):
            _lower(float(m.group(1)), strict=True)
            matched = True
        if not matched:
            m = re.search(rf"{nm}\s*=\s*({_NUM})", s)
            if m and "<" not in s and ">" not in s:
                v = float(m.group(1))
                lo[i] = v
                up[i] = v
                matched = True
        bound_this_pred = matched


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
    ranking_formula: str = RANKING_SCORE_LCB_COVERAGE,
    fid_estimator: str = FID_EMPIRICAL,
    perturb_samples: int = 512,
) -> str:
    """Random axis-aligned boxes, scored on D_val, reported on D_test.

    Query budget is n_candidates * |D_val| classifier lookups — but we use the
    cached y_hat_val, so the black-box cost is one pass over val (+ train for the
    classifier). This is the honest non-amortized optimizer baseline.
    """
    loader = _load(dataset, seed)
    _ensure_classifier(loader, classifier_path)
    meter = _meter_for(loader)
    meter.__enter__()
    _t_construct = time.time()
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
    pfid = (_perturbed_fid_scorer(loader, "unit", perturb_samples)
            if fid_estimator == FID_PERTURBED else None)
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
            sel_fid = m_val.fidelity if pfid is None else pfid(lo, up, cls, rng)
            per_class_ranked[cls].append(RankedRule(
                rule_id=f"rs:{cls}:{i}",
                lower=lo, upper=up, mask=mask_val, metrics=m_val,
                score=ranking_score(
                    sel_fid, m_val.coverage,
                    formula=ranking_formula,
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
    queries.wall_construct_s = time.time() - _t_construct
    queries.attach_meter(meter); meter.__exit__(None, None, None)
    return _emit(
        dataset=dataset, method="random_search", seed=seed, tau_p=tau_p, tau_c=tau_c, ranking_formula=ranking_formula,
        out_dir=out_dir, k=k, min_support=min_support,
        loader=loader, y_eval=y_test, y_hat_eval=y_hat_test,
        per_class_ranked=per_class_reported, queries=queries,
        extra={
            "n_candidates": n_candidates,
            "k": k,
            "fid_estimator": fid_estimator,
            "perturb_samples": int(perturb_samples) if fid_estimator == FID_PERTURBED else 0,
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
    p.add_argument(
        "--fid_estimator", default=FID_EMPIRICAL, choices=[FID_EMPIRICAL, FID_PERTURBED],
        help="Candidate ranking Fid for CART / random_search: real D_val rows in the box "
             "(empirical) or Anchors' D(z|B) (perturbed). The Anchors family always "
             "searches on perturbed precision; coverage is always on real rows.",
    )
    p.add_argument("--perturb_samples", type=int, default=512)
    p.add_argument(
        "--coverage_basis", default=os.environ.get("DYNANC_COVERAGE_BASIS", "true_label"),
        choices=["true_label", "predicted"],
        help="Class-conditional coverage denominator: P(x in B | y=c) or P(x in B | f_hat=c).",
    )
    p.add_argument(
        "--ranking_formulas", nargs="+",
        default=[RANKING_SCORE_LCB_COVERAGE],
        choices=[RANKING_SCORE_LCB_COVERAGE, RANKING_SCORE_LCB_GATED],
        help="Candidate ranking on D_val. Several => one <formula>/ subdir each; the "
             "Anchors pool is generated once and re-scored per formula.",
    )
    p.add_argument(
        "--k_values", type=int, nargs="+", default=None,
        help="Reduce ONE anchor pool at several union sizes; writes k<K>/ "
             "subdirectories under --out_dir. Keeps the explainer's unseeded "
             "sampling constant across k.",
    )
    args = p.parse_args()
    from utils.metrics import set_coverage_basis
    set_coverage_basis(args.coverage_basis)

    written = []
    multi = len(args.ranking_formulas) > 1
    for formula in args.ranking_formulas:
        f_out = os.path.join(args.out_dir, formula) if multi else args.out_dir
        if "cart" in args.methods:
            written.append(run_cart(
                args.dataset, args.seed, args.k, args.tau_p, args.tau_c,
                f_out, args.classifier_path, ranking_formula=formula,
                fid_estimator=args.fid_estimator, perturb_samples=args.perturb_samples,
            ))
        if "random_search" in args.methods:
            written.append(run_random_search(
                args.dataset, args.seed, args.k, args.tau_p, args.tau_c, f_out,
                args.classifier_path, n_candidates=args.n_candidates,
                ranking_formula=formula,
                fid_estimator=args.fid_estimator, perturb_samples=args.perturb_samples,
            ))
    anchor_methods = [m for m in args.methods if m in ("sp_anchors", "greedy_anchors")]
    if anchor_methods:
        written.extend(run_anchors_family(
            args.dataset, args.seed, args.k, args.tau_p, args.tau_c, args.out_dir,
            args.classifier_path,
            budget_per_class=args.budget_per_class,
            query_budget=args.query_budget,
            methods=anchor_methods,
            k_values=args.k_values,
            ranking_formulas=args.ranking_formulas,
        ))
    for w in written:
        logger.info("Wrote %s", w)


if __name__ == "__main__":
    main()
