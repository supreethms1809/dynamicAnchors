#!/usr/bin/env python3
"""
EBM reference for the California Housing feature-influence comparison.

Produces the EBM side of "dynamic anchors vs EBM" on the SAME dataset and the
SAME 4-class quartile target the anchors pipeline uses.

Two fits:

1. VERIFICATION (regressor). Reproduces notebooks/explainable_boosting_machines_tutorial.ipynb
   exactly: continuous MedHouseVal, unstratified 80/20 split, random_state=42.
   Expected test R2 = 0.8332 (the value saved in the notebook). This only exists to
   prove the data/hyperparameters match the notebook before we trust the classifier fit.

2. COMPARISON (classifier). Same EBM hyperparameters, but fit on the quartile-binned
   4-class target and the STRATIFIED split produced by BenchMARL/tabular_datasets.py,
   which is what the anchor agents and the MLP classifier see. The notebook's split is
   unstratified, so it is NOT the same partition -- this fit is the one that is
   comparable to the anchors.

Feature influence exported per class as mean |term contribution| to that class's logit,
which is the EBM analogue of "how much does this feature drive class c".
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

# EBM hyperparameters, verbatim from the notebook
EBM_KWARGS = dict(interactions=10, max_bins=256, learning_rate=0.01, random_state=42, n_jobs=-1)

OUT_DIR = Path(__file__).parent / "ebm_housing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["very_low_price", "low_price", "medium_price", "high_price"]


def load_housing():
    """Raw features, continuous target, and the quartile 4-class target.

    Class binning mirrors BenchMARL/tabular_datasets.py:118-124 exactly.
    """
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    prices = data.target.astype(np.float32)
    quartiles = np.percentile(prices, [25, 50, 75])
    y_cls = np.digitize(prices, quartiles).astype(int)
    feature_names = list(data.feature_names)
    # EBM takes term names from DataFrame columns; a bare ndarray yields feature_0000...
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, prices, y_cls, feature_names, quartiles


def aggregate_to_features(term_names, term_scores, feature_names):
    """Collapse 1D + 2D term scores onto individual features.

    Interaction terms are split evenly between their two features -- the usual
    convention when attributing a pairwise term to its members.
    """
    main = {f: 0.0 for f in feature_names}
    total = {f: 0.0 for f in feature_names}
    for name, score in zip(term_names, term_scores):
        if " & " in name:
            for part in name.split(" & "):
                total[part] += 0.5 * score
        else:
            main[name] += score
            total[name] += score
    return main, total


def verify_against_notebook(X, prices, feature_names):
    """Fit 1: reproduce the notebook regressor and check R2 == 0.8332."""
    # Notebook split: train_test_split(X, y, test_size=0.20, random_state=42), NO stratify
    Xtr, Xte, ytr, yte = train_test_split(X, prices, test_size=0.20, random_state=42)

    ebm = ExplainableBoostingRegressor(**EBM_KWARGS)
    ebm.fit(Xtr, ytr)

    r2 = r2_score(yte, ebm.predict(Xte))
    notebook_r2 = 0.8332
    r2_matches = abs(r2 - notebook_r2) < 5e-4

    # The exact 18 terms printed by notebook cell 11
    notebook_terms = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
        "Latitude", "Longitude",
        "MedInc & HouseAge", "MedInc & Population", "MedInc & AveOccup",
        "MedInc & Longitude", "HouseAge & AveOccup", "AveBedrms & Longitude",
        "Population & AveOccup", "AveOccup & Latitude", "AveOccup & Longitude",
        "Latitude & Longitude",
    ]
    terms_match = list(ebm.term_names_) == notebook_terms
    matches = r2_matches and terms_match

    print(f"[verify] regressor test R2 = {r2:.4f}  (notebook: {notebook_r2})  match={r2_matches}")
    print(f"[verify] terms found = {len(ebm.term_names_)}, identical to notebook = {terms_match}")

    importances = list(ebm.term_importances())
    main, total = aggregate_to_features(ebm.term_names_, importances, feature_names)

    return {
        "test_r2": float(r2),
        "notebook_test_r2": notebook_r2,
        "reproduces_notebook": bool(matches),
        "r2_matches": bool(r2_matches),
        "term_list_matches_notebook": bool(terms_match),
        "term_names": list(ebm.term_names_),
        "term_importances": [float(v) for v in importances],
        "feature_importance_main_effect_only": {k: float(v) for k, v in main.items()},
        "feature_importance_with_interactions": {k: float(v) for k, v in total.items()},
        "intercept": float(np.ravel(ebm.intercept_)[0]),
    }, matches


def fit_classifier(X, y_cls, feature_names):
    """Fit 2: the comparison model -- 4-class quartile target, anchors' stratified split."""
    # Anchors split: tabular_datasets.py:398-400, stratified on the class label
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    ebm = ExplainableBoostingClassifier(**EBM_KWARGS)
    ebm.fit(Xtr, ytr)

    acc = accuracy_score(yte, ebm.predict(Xte))
    print(f"[classifier] 4-class test accuracy = {acc:.4f}")
    print(f"[classifier] terms found = {len(ebm.term_names_)}")

    term_names = list(ebm.term_names_)

    # Global (class-averaged) importances as interpret reports them
    global_imp = list(ebm.term_importances())
    g_main, g_total = aggregate_to_features(term_names, global_imp, feature_names)

    # Per-class influence: mean |contribution| of each term to class c's logit,
    # averaged over TEST rows whose true label is c.
    contribs = ebm.eval_terms(Xte)
    contribs = np.asarray(contribs)
    print(f"[classifier] eval_terms shape = {contribs.shape} "
          f"(expect n_samples={len(Xte)}, n_terms={len(term_names)}, n_classes=4)")

    per_class = {}
    for c in range(4):
        mask = yte == c
        if contribs.ndim == 3:
            # (n_samples, n_terms, n_classes) -> contribution to class c's own logit
            cls_scores = np.abs(contribs[mask, :, c]).mean(axis=0)
        else:
            # binary/regression fallback: (n_samples, n_terms)
            cls_scores = np.abs(contribs[mask, :]).mean(axis=0)
        m, t = aggregate_to_features(term_names, list(cls_scores), feature_names)
        per_class[f"class_{c}"] = {
            "class": c,
            "class_name": CLASS_NAMES[c],
            "n_test_samples": int(mask.sum()),
            "term_influence": {n: float(s) for n, s in zip(term_names, cls_scores)},
            "feature_influence_main_effect_only": {k: float(v) for k, v in m.items()},
            "feature_influence_with_interactions": {k: float(v) for k, v in t.items()},
        }

    return {
        "test_accuracy": float(acc),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "term_names": term_names,
        "global_term_importances": [float(v) for v in global_imp],
        "global_feature_importance_main_effect_only": {k: float(v) for k, v in g_main.items()},
        "global_feature_importance_with_interactions": {k: float(v) for k, v in g_total.items()},
        "per_class": per_class,
    }


def main():
    X, prices, y_cls, feature_names, quartiles = load_housing()
    print(f"Housing: X={X.shape}, classes={np.bincount(y_cls)}, quartiles={quartiles}")

    verify, matched = verify_against_notebook(X, prices, feature_names)
    clf = fit_classifier(X, y_cls, feature_names)

    out = {
        "dataset": "housing (California Housing)",
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "quartile_thresholds": [float(q) for q in quartiles],
        "ebm_hyperparameters": {k: v for k, v in EBM_KWARGS.items()},
        "verification_regressor": verify,
        "comparison_classifier": clf,
        "notes": {
            "split_difference": (
                "The notebook uses an UNSTRATIFIED 80/20 split; the anchors pipeline "
                "(tabular_datasets.py:398) stratifies on the 4-class label. Same seed, "
                "different partition. The classifier fit here uses the anchors' split."
            ),
            "per_class_influence": (
                "mean |contribution of term to class c's logit| over test rows with true label c"
            ),
            "interaction_attribution": "2D terms split 50/50 between their two features",
        },
    }

    path = OUT_DIR / "ebm_influence.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")

    # Console summary
    print("\n=== EBM global feature importance (4-class classifier, with interactions) ===")
    ranked = sorted(clf["global_feature_importance_with_interactions"].items(),
                    key=lambda kv: -kv[1])
    for i, (f, v) in enumerate(ranked, 1):
        print(f"  {i}. {f:<12} {v:.4f}")

    if not matched:
        print("\nWARNING: regressor did not reproduce the notebook R2. "
              "Check interpret version / data before trusting the comparison.", file=sys.stderr)


if __name__ == "__main__":
    main()
