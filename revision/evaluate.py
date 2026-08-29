"""
Rebuttal evaluation entry point.

Reads an extracted_rules JSON (RLDA or MADA), scores every stored BOX on the
held-out test split, reports Fid/Pur + n_covered + Wilson CIs, builds the top-k
union from the same pool as `best`, and writes a result artifact that
`paper/make_tables.py` can turn into LaTeX.

Usage:
    python -m revision.evaluate \\
        --rules_file <extracted_rules.json> \\
        --dataset iris --method maddpg --seed 42 \\
        --tau_p 0.90 --tau_c 0.20 \\
        --out_dir revision/results
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "BenchMARL"))

from BenchMARL.tabular_datasets import TabularDatasetLoader  # noqa: E402
from utils.eval_harness import (  # noqa: E402
    audit_selected_unit_rules,
    evaluate_ruleset_as_classifier,
    per_class_block,
    reevaluate_ranked_rules,
    rules_from_anchors,
    write_result_artifact,
)
from utils.metrics import (
    RANKING_SCORE_LCB_COVERAGE,  # noqa: E402
    MIN_SUPPORT_DEFAULT,
    active_feature_mask,
    collect_success_rate,
    compactness_of_ruleset,
    evaluate_mask,
    select_topk_union,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("revision.evaluate")


def _classifier_predictions(loader, X_std: np.ndarray) -> np.ndarray:
    import torch
    from utils.networks import predict_proba_torch

    clf = loader.classifier
    if clf is None:
        raise ValueError("Classifier not loaded")
    if hasattr(clf, "eval"):
        clf.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(np.asarray(X_std, dtype=np.float32))
        probs = predict_proba_torch(clf, inputs).cpu().numpy()
    return probs.argmax(axis=1)


def _dedupe_anchor_boxes(anchors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop identical unit boxes so instance + class-based slots don't double-count."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for a in anchors:
        lo = a.get("lower_bounds_normalized") or a.get("lower_bounds")
        up = a.get("upper_bounds_normalized") or a.get("upper_bounds")
        if lo is None or up is None:
            out.append(a)
            continue
        key = (tuple(np.round(np.asarray(lo, dtype=float), 6)),
               tuple(np.round(np.asarray(up, dtype=float), 6)))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def _pool_class_anchors(per_class_results: Dict[str, Any], cls: int) -> List[Dict[str, Any]]:
    """Rank instance-based and class-based boxes together (do not starve the hull)."""
    class_data = per_class_results.get(f"class_{cls}") or {}
    anchors = list(_collect_anchors(class_data))
    cb_key = f"class_{cls}_class_based"
    if cb_key in per_class_results:
        anchors.extend(_collect_anchors(per_class_results[cb_key]))
    return _dedupe_anchor_boxes(anchors)


def _has_scorable_box(anchor: Dict[str, Any]) -> bool:
    """Admit unit-space or original-space bounds (evaluate scores unit keys).

    D-01: reject the EMPTY rule. A box with no tightened dimension covers every row,
    so it enters top-k on coverage alone and drags the union's fidelity down to the
    class prior. This guard lives here as well as in the producers because every
    method (RLDA, MADA, and the baselines) is scored through this function.
    """
    unit = (
        anchor.get("lower_bounds_normalized") is not None
        and anchor.get("upper_bounds_normalized") is not None
    )
    orig = anchor.get("lower_bounds") is not None and anchor.get("upper_bounds") is not None
    if not (unit or orig):
        return False
    if unit:
        active = active_feature_mask(
            np.asarray(anchor["lower_bounds_normalized"], dtype=float),
            np.asarray(anchor["upper_bounds_normalized"], dtype=float),
        )
        return bool(np.any(active))
    return True


def _collect_anchors(class_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prefer stored box bounds over printed unique_rules strings (C-01 / C-08)."""
    anchors = []
    for key in ("all_anchors", "anchors"):
        for a in class_data.get(key) or []:
            if _has_scorable_box(a):
                anchors.append(a)
    # Class-based nested slot
    cb = class_data.get("class_based_results") or {}
    if isinstance(cb, dict):
        for agent_cb in cb.values() if cb and isinstance(next(iter(cb.values()), None), dict) else [cb]:
            if not isinstance(agent_cb, dict):
                continue
            for a in agent_cb.get("anchors") or agent_cb.get("all_anchors") or []:
                if _has_scorable_box(a):
                    anchors.append(a)
    return anchors


def _load_classifier(loader, rules_file: str, seed: int, dataset: str):
    try:
        return loader.get_classifier()
    except ValueError:
        pass
    rules_parent = os.path.dirname(os.path.abspath(rules_file))
    exp_dir = os.path.dirname(rules_parent)
    candidates = [
        os.path.join(exp_dir, "training", "classifier.pth"),
        os.path.join(exp_dir, "classifier.pth"),
        os.path.join(rules_parent, "classifier.pth"),
        os.path.join("models", f"classifier_{dataset}_seed{seed}.pth"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"No classifier.pth found in {candidates}")
    return loader.load_classifier(filepath=path, device="cpu")


def evaluate_rules_file(
    rules_file: str,
    dataset: str,
    method: str,
    seed: int,
    tau_p: float,
    tau_c: float,
    out_dir: str,
    k: int = 5,
    min_support: int = MIN_SUPPORT_DEFAULT,
    ranking_formula: str = RANKING_SCORE_LCB_COVERAGE,
    split: str = "test",
) -> str:
    if split != "test":
        raise ValueError(
            "Reported revision metrics are test-only. Rule ranking and top-k "
            "selection always use validation; pass split='test'."
        )
    with open(rules_file) as f:
        rules = json.load(f)

    loader = TabularDatasetLoader(dataset_name=dataset, random_state=seed)
    loader.load_dataset()
    loader.preprocess_data()
    loader.classifier = _load_classifier(loader, rules_file, seed, dataset)

    if loader.X_val_unit is None:
        raise ValueError("A validation split is required for rule selection")
    X_val_unit = loader.X_val_unit
    y_val = loader.y_val
    y_hat_val = _classifier_predictions(loader, loader.X_val_scaled)
    X_test_unit = loader.X_test_unit
    y_test = loader.y_test
    y_hat_test = _classifier_predictions(loader, loader.X_test_scaled)

    # Classifier accuracy on this split (C-09 action 4)
    acc = {
        "selection_split": "val",
        "report_split": "test",
        "n": int(len(y_test)),
        "test_accuracy": float((y_hat_test == y_test).mean()),
        "split_accuracy": float((y_hat_test == y_test).mean()),
    }
    # Also report train/val/test if available
    def _acc(Xs, ys):
        if Xs is None or ys is None:
            return None
        pred = _classifier_predictions(loader, Xs)
        return float((pred == ys).mean())
    acc["train_accuracy"] = _acc(loader.X_train_scaled, loader.y_train)
    acc["val_accuracy"] = float((y_hat_val == y_val).mean())
    queries = dict(rules.get("queries") or rules.get("query_accounting") or {})
    queries.setdefault("n_blackbox_queries", 0)
    queries["n_reporting_queries"] = int(
        queries.get("n_reporting_queries", 0)
        + len(loader.y_train) + len(y_val) + len(y_test)
    )
    queries.setdefault(
        "query_policy",
        "Generation/selection calls exclude held-out reporting calls",
    )

    per_class_results = rules.get("per_class_results", {})
    classes = sorted({
        int(cd.get("class"))
        for cd in per_class_results.values()
        if cd.get("class") is not None
    })

    per_class_out: Dict[str, Any] = {}
    class_union_masks: Dict[int, np.ndarray] = {}
    class_union_fid: Dict[int, float] = {}
    compactness_rows: List[Dict[str, Any]] = []
    audit_problems: List[str] = []
    sparsity = float(rules.get("metadata", {}).get("sparsity_width_ratio", 0.95))
    X_all_orig = np.vstack(
        [x for x in (loader.X_train, loader.X_val, loader.X_test) if x is not None]
    )
    feat_min_orig = X_all_orig.min(axis=0)
    feat_max_orig = X_all_orig.max(axis=0)

    for cls in classes:
        anchors = _pool_class_anchors(per_class_results, int(cls))
        class_data = per_class_results.get(f"class_{cls}") or per_class_results.get(
            f"class_{cls}_class_based"
        ) or {}

        ranked_val = rules_from_anchors(
            anchors, X_val_unit, y_val, y_hat_val, cls,
            class_conditional=True,
            min_support=min_support,
            ranking_formula=ranking_formula,
            space="unit",
        )
        if not ranked_val:
            logger.warning("Class %s: no anchors with box bounds; skipping", cls)
            continue

        selected_val = select_topk_union(
            ranked_val, y_val, y_hat_val, cls, k=k,
            class_conditional=True, min_support=min_support,
        )
        if selected_val is None:
            continue

        ranked_test = reevaluate_ranked_rules(
            selected_val.individual,
            X_test_unit,
            y_test,
            y_hat_test,
            cls,
            class_conditional=True,
            min_support=min_support,
        )
        # Reporting-side union: the rule set is already fixed by validation
        # selection above, so support must NOT be re-filtered here — doing so
        # would make the published rule set depend on test data.
        union = select_topk_union(
            ranked_test, y_test, y_hat_test, cls, k=len(ranked_test),
            class_conditional=True, min_support=min_support,
            enforce_min_support=False,
        )
        if union is None:
            continue

        # Instance-level: overall (marginal) coverage of the best box
        inst = evaluate_mask(
            y=y_test, y_hat=y_hat_test, mask=union.best.mask, target_class=cls,
            class_conditional=False, min_support=min_support,
        )
        per_class_out[f"class_{cls}"] = per_class_block(union, instance_metrics=inst)
        compactness_rows.append(
            compactness_of_ruleset(union.individual, sparsity_width_ratio=sparsity)
        )
        audit_problems.extend(
            audit_selected_unit_rules(
                union.individual,
                X_test_unit,
                sparsity_width_ratio=sparsity,
                x_min_std=loader.X_min,
                x_range_std=loader.X_range,
                scaler_mean=np.asarray(loader.scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(loader.scaler.scale_, dtype=np.float64),
                feature_min_orig=feat_min_orig,
                feature_max_orig=feat_max_orig,
                feature_names=list(loader.feature_names),
            )
        )
        umask = np.zeros(len(y_test), dtype=bool)
        for r in union.individual:
            umask |= r.mask
        class_union_masks[cls] = umask
        class_union_fid[cls] = union.union_metrics.fidelity if np.isfinite(union.union_metrics.fidelity) else -np.inf

        logger.info(
            "Class %s: k=%s best Fid=%.3f Pur=%.3f cov=%.3f n=%s | "
            "union Fid=%.3f Pur=%.3f cov=%.3f n=%s",
            cls, union.n_selected,
            union.best.metrics.fidelity, union.best.metrics.purity,
            union.best.metrics.coverage, union.best.metrics.n_covered,
            union.union_metrics.fidelity, union.union_metrics.purity,
            union.union_metrics.coverage, union.union_metrics.n_covered,
        )

    if not per_class_out:
        raise RuntimeError(
            f"No scorable boxes in {rules_file} after scoring "
            f"({len(classes)} classes in the rules file). Inference likely failed "
            "to persist unit bounds; refusing to write an empty result."
        )

    global_res = evaluate_ruleset_as_classifier(
        class_union_masks, class_union_fid, y_test, y_hat_test
    )
    success = collect_success_rate(rules, tau_p, tau_c)
    compactness = {
        "sparsity_width_ratio": sparsity,
        "n_classes_with_rules": len(compactness_rows),
        "mean_active_features": (
            float(np.nanmean([r["mean_active_features"] for r in compactness_rows
                              if r.get("mean_active_features") is not None]))
            if compactness_rows else None
        ),
        "mean_rules_per_class": (
            float(np.mean([r["n_rules"] for r in compactness_rows]))
            if compactness_rows else 0.0
        ),
        "per_class": compactness_rows,
    }
    if audit_problems:
        for msg in audit_problems:
            logger.error("Rule audit: %s", msg)
        logger.error(
            "%s selected-rule audit problem(s) (C-08/C-13). Metrics stay on the "
            "evaluated box; printed-rule mismatches are stored in extra.",
            len(audit_problems),
        )

    path = write_result_artifact(
        out_dir,
        dataset=dataset,
        method=method,
        seed=seed,
        tau_p=tau_p,
        tau_c=tau_c,
        per_class=per_class_out,
        global_ruleset=global_res.to_dict(),
        classifier_accuracy=acc,
        queries=queries,
        n_covered_note=(
            f"Rules selected/ranked on D_val; all reported metrics on "
            f"D_test (n={len(y_test)}). Union over top-k={k} ranked by "
            f"{ranking_formula} (score = Wilson LCB(fid) * (1 + cov) when formula is "
            f"lcb_coverage); "
            f"best = rank-1 of that same set. Empty boxes -> NaN precision. "
            f"min_support={min_support}. "
            f"Cells with n_covered < min_support are statistically uninformative."
        ),
        extra={
            "rules_file": os.path.abspath(rules_file),
            "selection_split": "val",
            "report_split": "test",
            "bounds_space": "unit",
            "k": k,
            "ranking_formula": ranking_formula,
            "sparsity_width_ratio": sparsity,
            "print_box_mismatches": audit_problems,
            "split_sizes": {
                "train": int(len(loader.y_train)),
                "val": int(len(loader.y_val)) if getattr(loader, "y_val", None) is not None else 0,
                "test": int(len(loader.y_test)),
            },
        },
        success_rate=success,
        compactness=compactness,
        ranking_formula=ranking_formula,
        min_support=min_support,
    )
    logger.info("Wrote %s", path)
    logger.info(
        "Global test: fidelity=%s abstention=%.3f conflict=%.3f coverage=%.3f",
        global_res.global_fidelity, global_res.abstention_rate,
        global_res.conflict_rate, global_res.coverage,
    )
    return path


def main():
    p = argparse.ArgumentParser(description="Rebuttal evaluation (box-based Fid/Pur on D_test)")
    p.add_argument("--rules_file", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--method", required=True, help="rlda / mada / maddpg / sac / ...")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tau_p", type=float, default=None)
    p.add_argument("--tau_c", type=float, default=None)
    p.add_argument("--k", type=int, default=None, help="override config top-k")
    p.add_argument("--min_support", type=int, default=None)
    p.add_argument("--ranking_formula", default=None)
    p.add_argument(
        "--split", default="test", choices=["test"],
        help="Reported metrics are always test-only; selection always uses validation.",
    )
    p.add_argument("--out_dir", default="revision/results")
    args = p.parse_args()
    with open(args.rules_file) as f:
        stored = json.load(f)
    metadata = stored.get("metadata", {})
    dataset = args.dataset or metadata.get("dataset")
    if not dataset:
        p.error("--dataset is required when rules metadata has no dataset")
    seed = args.seed if args.seed is not None else int(metadata.get("seed", 42))
    tau_p = (
        args.tau_p if args.tau_p is not None
        else float(metadata.get("precision_target", 0.90))
    )
    tau_c = (
        args.tau_c if args.tau_c is not None
        else float(metadata.get("coverage_target", 0.20))
    )
    k = (
        args.k if args.k is not None
        else int(metadata.get("top_k_rules_by_score", 5))
    )
    min_support = (
        args.min_support if args.min_support is not None
        else int(metadata.get("min_support", MIN_SUPPORT_DEFAULT))
    )
    ranking_formula = (
        args.ranking_formula
        or metadata.get("ranking_score_formula")
        # A1: default to the support-aware score. Rules files written before the
        # fix carry no formula key and must not silently fall back to the
        # point-estimate ranking that selected single-sample rules.
        or RANKING_SCORE_LCB_COVERAGE
    )
    evaluate_rules_file(
        rules_file=args.rules_file,
        dataset=dataset,
        method=args.method,
        seed=seed,
        tau_p=tau_p,
        tau_c=tau_c,
        out_dir=args.out_dir,
        k=k,
        min_support=min_support,
        ranking_formula=ranking_formula,
        split=args.split,
    )


if __name__ == "__main__":
    main()
