#!/usr/bin/env python3
"""
Feature influence from dynamic-anchor rules, and comparison against the EBM reference.

EBM gives feature influence directly (mean |contribution| of a term). Anchors give
boxes, not contributions, so influence has to be *derived*. Two derivations here:

1. ABLATION INFLUENCE (headline). For anchor box A of class c, drop the constraint on
   feature j (relax it to the full data range) and re-measure the box's precision:

       influence_j(A) = precision(A) - precision(A without the condition on j)

   where precision = P(MLP predicts class c | x inside box). If removing feature j's
   condition lets in a flood of other-class points, that feature was doing real work.
   If precision is unchanged, the condition was decorative. This is the anchor analogue
   of EBM's "how much does this term move the class-c logit", and it explains the SAME
   object EBM explains: the trained classifier's decision function, not the labels.

2. TIGHTNESS. Purely geometric: 1 - (box width on j / data range on j). Cheap, and it
   is what a reader eyeballs off a printed rule, but it does NOT know whether a tight
   bound matters -- a feature can be pinned tight and be irrelevant. Reported as a
   cross-check on (1), not as the headline.

Both are aggregated per class as an UNWEIGHTED mean over the class's anchors -- each
rule is one explanation and counts once -- then normalized to a share so they can sit
next to EBM's logit-unit numbers. An earlier version weighted by box coverage; that was
wrong in practice (see the note in anchor_influence), because coverage is skewed enough
that a few huge boxes decided the result, and huge boxes are the ones where ablating a
single feature changes almost nothing.

CAVEAT on single-feature ablation: it under-reports features that are redundant with
others. Latitude and Longitude jointly define a neighbourhood, so relaxing one while the
other still binds may barely move precision even though the pair matters a great deal.
This is a structural limit of one-at-a-time ablation, not of the anchors, and it is the
most likely reason anchors under-weight Latitude relative to EBM's Latitude & Longitude
interaction term. `binding_fraction` is reported so a feature that IS constrained but
precision-neutral can be told apart from one that is simply unconstrained.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "BenchMARL"))

from BenchMARL.tabular_datasets import TabularDatasetLoader  # noqa: E402
from utils.networks import predict_proba_torch  # noqa: E402


def load_data_and_classifier(dataset, experiment_dir, device="cpu"):
    """Full (train+test) data in both spaces, plus the MLP's predictions.

    Box membership is tested in the space the rule bounds live in; the classifier is
    fed standardized inputs, since that is what it was trained on.
    """
    loader = TabularDatasetLoader(dataset_name=dataset, test_size=0.2, random_state=42)
    loader.load_dataset()
    loader.preprocess_data()

    X_raw = np.vstack([loader.X_train, loader.X_test]).astype(np.float32)
    X_scaled = np.vstack([loader.X_train_scaled, loader.X_test_scaled]).astype(np.float32)
    y = np.concatenate([loader.y_train, loader.y_test])

    clf_path = None
    for cand in Path(experiment_dir).rglob("classifier.pth"):
        clf_path = cand
        break
    if clf_path is None:
        raise FileNotFoundError(f"No classifier.pth found under {experiment_dir}")

    classifier = loader.load_classifier(filepath=str(clf_path), device=device)
    classifier.eval()
    with torch.no_grad():
        probs = predict_proba_torch(classifier, torch.tensor(X_scaled, device=device))
        preds = torch.argmax(probs, dim=1).cpu().numpy()

    print(f"  classifier: {clf_path}")
    print(f"  data: {X_raw.shape[0]} rows, {X_raw.shape[1]} features")
    print(f"  MLP accuracy on full data: {(preds == y).mean():.4f}")
    return loader, X_raw, X_scaled, y, preds


def detect_bound_space(anchors, X_raw, X_scaled):
    """Rule bounds are written in raw feature units, but verify rather than assume.

    Picks whichever space actually puts data inside the boxes.
    """
    def containment(X):
        hits = 0
        for a in anchors[:50]:
            lb = np.asarray(a["lower_bounds"], dtype=np.float64)
            ub = np.asarray(a["upper_bounds"], dtype=np.float64)
            if lb.shape[0] != X.shape[1]:
                return -1.0
            hits += int(np.all((X >= lb) & (X <= ub), axis=1).sum())
        return hits / max(len(anchors[:50]), 1)

    raw_hits, scaled_hits = containment(X_raw), containment(X_scaled)
    space = "raw" if raw_hits >= scaled_hits else "standardized"
    print(f"  bound space: {space} (mean rows/box raw={raw_hits:.1f}, scaled={scaled_hits:.1f})")
    return space, raw_hits, scaled_hits


def collect_anchors(rules_json):
    """Flatten per-class -> per-agent -> anchors, keeping only boxes with real bounds."""
    per_class = {}
    for class_key, cdata in rules_json["per_class_results"].items():
        cls = int(cdata["class"])
        anchors = []
        for agent, adata in cdata.get("per_agent_results", {}).items():
            for a in adata.get("anchors", []):
                if "lower_bounds" in a and "upper_bounds" in a:
                    anchors.append({**a, "agent": agent})
        per_class[cls] = anchors
    return per_class


def _rule_feature_set(rule_str):
    """Features named in a rule string, e.g. "MedInc in [..] and Latitude in [..]"."""
    out = set()
    for cond in (rule_str or "").split(" and "):
        if "\u2208" in cond:
            out.add(cond.split("\u2208")[0].strip())
    return frozenset(out)


def anchor_influence(anchors, X, y, preds, target_class, feature_names):
    """Per-feature ablation influence + tightness for one class."""
    n_features = X.shape[1]
    data_lo, data_hi = X.min(axis=0), X.max(axis=0)
    data_range = np.where((data_hi - data_lo) == 0, 1.0, data_hi - data_lo)

    per_anchor = []
    for a in anchors:
        lb = np.asarray(a["lower_bounds"], dtype=np.float64)
        ub = np.asarray(a["upper_bounds"], dtype=np.float64)

        # (n_samples, n_features) boolean: does each row satisfy each single condition?
        inside = (X >= lb) & (X <= ub)
        mask = np.all(inside, axis=1)
        n_in = int(mask.sum())
        if n_in == 0:
            continue

        prec_pred = float((preds[mask] == target_class).mean())
        prec_label = float((y[mask] == target_class).mean())
        cov = n_in / X.shape[0]

        infl, tight, relaxed = np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)
        binding = np.zeros(n_features)
        for j in range(n_features):
            # Relax feature j only: satisfy all OTHER conditions
            others = np.delete(inside, j, axis=1)
            mask_j = np.all(others, axis=1)
            n_j = int(mask_j.sum())
            prec_j = float((preds[mask_j] == target_class).mean()) if n_j > 0 else 0.0
            relaxed[j] = prec_j
            infl[j] = prec_pred - prec_j
            # Binding = the bound on j actually excludes rows. Distinguishes "this
            # feature is unconstrained" from "constrained but precision-neutral".
            binding[j] = 1.0 if n_j > n_in else 0.0
            width = (ub[j] - lb[j]) / data_range[j]
            tight[j] = float(np.clip(1.0 - width, 0.0, 1.0))

        per_anchor.append({
            "agent": a.get("agent"),
            "n_samples_in_box": n_in,
            "coverage": cov,
            "precision_prediction_match": prec_pred,
            "precision_class_label": prec_label,
            "influence": infl,
            "tightness": tight,
            "relaxed_precision": relaxed,
            "binding": binding,
            "feature_set": _rule_feature_set(a.get("rule", "")),
            "mask": mask,
        })

    if not per_anchor:
        return None

    # Aggregation: the UNWEIGHTED mean over anchors is the headline. Each anchor is
    # one explanation and counts once.
    #
    # The coverage-weighted mean it replaced was actively misleading. Box coverage is
    # severely skewed -- in v5 class_0 the single largest box carried 14.3% of the
    # weight and the top 5 of 60 carried 50.4%, against a median box weight of 0.39%
    # -- so a handful of boxes decided the whole vector. Those boxes are exactly
    # where single-feature ablation is least informative: a box already holding a
    # quarter of the data barely changes when one bound is relaxed, and can even gain
    # precision, yielding a NEGATIVE influence. The result contradicted the rules
    # themselves: Latitude is binding in 82% of v5 boxes (mean precision drop +0.0324
    # when relaxed) and appears in 32.5% of v5 rule strings, yet the weighted mean
    # reported -0.0020, i.e. a 0.0% share. It also made the EBM rank correlation swing
    # wildly between runs (+0.810 to +0.000) when the unweighted values were stable
    # (+0.643 vs +0.643) -- the swings were an artifact of which giant box dominated.
    #
    # coverage_weighted is still reported, clearly labelled, for reference.
    w = np.array([p["coverage"] for p in per_anchor], dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(per_anchor)) / len(per_anchor)
    I = np.vstack([p["influence"] for p in per_anchor])
    T = np.vstack([p["tightness"] for p in per_anchor])

    infl_w = (I * w[:, None]).sum(axis=0)
    infl_u = I.mean(axis=0)
    tight_w = (T * w[:, None]).sum(axis=0)
    tight_u = T.mean(axis=0)

    # Diagnostic: how concentrated is the coverage weighting, and is each feature's
    # constraint actually binding? "binding" = relaxing it admits at least one extra
    # row. A feature can be binding and still precision-neutral (redundant with the
    # other bounds), which single-feature ablation cannot separate -- see the
    # interaction caveat in the module docstring.
    w_sorted = np.sort(w)[::-1]
    binding_frac = (np.vstack([p["binding"] for p in per_anchor])).mean(axis=0)

    # Sensitivity check: does the headline depend on how OFTEN a feature is used?
    # The plain mean is an expected influence over sampled instances (EBM's per-class
    # number is likewise a mean over rows), so a feature used by more instances
    # legitimately weighs more. But that conflates frequency-of-use with
    # magnitude-when-used. The dedup variant averages within each distinct rule
    # feature-set first, so {MedInc} counts once no matter how many anchors use it.
    # On housing the two agree closely (rho moves <= 0.1), which is what licenses
    # reading the plain mean as a statement about magnitude and not just count.
    groups = {}
    for pa in per_anchor:
        groups.setdefault(pa["feature_set"], []).append(pa["influence"])
    infl_dedup = np.mean([np.mean(np.vstack(v), axis=0) for v in groups.values()], axis=0)

    # Region overlap: if boxes covered the same rows, features would be double
    # counted. Reported so that assumption is checkable rather than assumed.
    M = np.vstack([pa["mask"] for pa in per_anchor])
    sizes = M.sum(axis=1).astype(np.float64)
    inter = M.astype(np.float64) @ M.T.astype(np.float64)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    iu = np.triu_indices(len(per_anchor), 1)
    pair_j = jac[iu] if len(per_anchor) > 1 else np.array([0.0])

    def share(v):
        pos = np.clip(v, 0.0, None)
        return pos / pos.sum() if pos.sum() > 0 else np.zeros_like(pos)

    return {
        "n_anchors_used": len(per_anchor),
        "n_anchors_total": len(anchors),
        "mean_precision_prediction_match": float(np.mean([p["precision_prediction_match"] for p in per_anchor])),
        "mean_precision_class_label": float(np.mean([p["precision_class_label"] for p in per_anchor])),
        "mean_coverage": float(np.mean([p["coverage"] for p in per_anchor])),
        # HEADLINE (unweighted mean over anchors)
        "influence": dict(zip(feature_names, infl_u.tolist())),
        "influence_share": dict(zip(feature_names, share(infl_u).tolist())),
        "tightness": dict(zip(feature_names, tight_u.tolist())),
        "tightness_share": dict(zip(feature_names, share(tight_u).tolist())),
        # Secondary / diagnostic
        "influence_coverage_weighted": dict(zip(feature_names, infl_w.tolist())),
        "tightness_coverage_weighted": dict(zip(feature_names, tight_w.tolist())),
        "binding_fraction": dict(zip(feature_names, binding_frac.tolist())),
        "influence_featureset_dedup": dict(zip(feature_names, infl_dedup.tolist())),
        "influence_featureset_dedup_share": dict(zip(feature_names, share(infl_dedup).tolist())),
        "n_distinct_feature_sets": len(groups),
        "rule_overlap": {
            "median_pairwise_jaccard": float(np.median(pair_j)),
            "max_pairwise_jaccard": float(pair_j.max()),
            "frac_pairs_jaccard_gt_0.5": float((pair_j > 0.5).mean()),
        },
        "coverage_weight_concentration": {
            "top1": float(w_sorted[0]),
            "top5": float(w_sorted[:5].sum()),
            "median": float(np.median(w)),
        },
        "_vectors": {"influence": infl_u, "tightness": tight_u, "influence_dedup": infl_dedup},
    }


def compare_with_ebm(anchor_res, ebm_json, feature_names):
    """Rank + share comparison, per class and globally."""
    ebm_clf = ebm_json["comparison_classifier"]
    out = {"per_class": {}, "global": {}}

    def norm(d):
        v = np.array([d[f] for f in feature_names], dtype=np.float64)
        pos = np.clip(v, 0.0, None)
        return pos / pos.sum() if pos.sum() > 0 else np.zeros_like(pos)

    for cls, res in anchor_res.items():
        if res is None:
            continue
        ebm_c = ebm_clf["per_class"].get(f"class_{cls}")
        if ebm_c is None:
            continue
        a_vec = res["_vectors"]["influence"]
        e_vec = np.array([ebm_c["feature_influence_with_interactions"][f] for f in feature_names])

        rho, p = spearmanr(a_vec, e_vec)
        out["per_class"][f"class_{cls}"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "anchor_share": dict(zip(feature_names, norm(res["influence"]).tolist())),
            "ebm_share": dict(zip(feature_names, (e_vec / e_vec.sum()).tolist())),
            "anchor_top3": [f for f, _ in sorted(
                res["influence"].items(), key=lambda kv: -kv[1])[:3]],
            "ebm_top3": [f for f, _ in sorted(
                ebm_c["feature_influence_with_interactions"].items(), key=lambda kv: -kv[1])[:3]],
        }

    # Global: average anchor influence over classes vs EBM's global importances
    vecs = [r["_vectors"]["influence"] for r in anchor_res.values() if r is not None]
    if vecs:
        a_glob = np.mean(np.vstack(vecs), axis=0)
        e_glob = np.array([ebm_clf["global_feature_importance_with_interactions"][f]
                           for f in feature_names])
        rho, p = spearmanr(a_glob, e_glob)
        d_vecs = [r["_vectors"]["influence_dedup"] for r in anchor_res.values() if r is not None]
        d_glob = np.mean(np.vstack(d_vecs), axis=0)
        rho_d, p_d = spearmanr(d_glob, e_glob)
        out["global"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            # Same correlation after collapsing repeated rule feature-sets to one vote
            "spearman_rho_featureset_dedup": float(rho_d),
            "spearman_p_featureset_dedup": float(p_d),
            "anchor_influence": dict(zip(feature_names, a_glob.tolist())),
            "anchor_share": dict(zip(feature_names,
                                    (np.clip(a_glob, 0, None) / max(np.clip(a_glob, 0, None).sum(), 1e-12)).tolist())),
            "ebm_importance": dict(zip(feature_names, e_glob.tolist())),
            "ebm_share": dict(zip(feature_names, (e_glob / e_glob.sum()).tolist())),
            "anchor_ranking": [f for f, _ in sorted(zip(feature_names, a_glob), key=lambda kv: -kv[1])],
            "ebm_ranking": [f for f, _ in sorted(zip(feature_names, e_glob), key=lambda kv: -kv[1])],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules_file", required=True)
    ap.add_argument("--dataset", default="housing")
    ap.add_argument("--ebm_json", default=str(PROJECT_ROOT / "paper/ebm_housing/ebm_influence.json"))
    ap.add_argument("--output_dir", default=str(PROJECT_ROOT / "paper/ebm_housing"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rules_json = json.load(open(args.rules_file))
    experiment_dir = rules_json.get("metadata", {}).get("experiment_dir") or \
        str(Path(args.rules_file).parent.parent)

    print("Loading data + classifier...")
    loader, X_raw, X_scaled, y, preds = load_data_and_classifier(
        args.dataset, experiment_dir, args.device)
    feature_names = list(loader.feature_names)

    per_class_anchors = collect_anchors(rules_json)
    all_anchors = [a for v in per_class_anchors.values() for a in v]
    print(f"  anchors: {len(all_anchors)} across {len(per_class_anchors)} classes")
    if not all_anchors:
        print("No anchors with bounds found in rules file.", file=sys.stderr)
        sys.exit(1)

    space, raw_hits, scaled_hits = detect_bound_space(all_anchors, X_raw, X_scaled)
    X = X_raw if space == "raw" else X_scaled

    print("\nComputing ablation influence per class...")
    anchor_res = {}
    for cls, anchors in sorted(per_class_anchors.items()):
        res = anchor_influence(anchors, X, y, preds, cls, feature_names)
        anchor_res[cls] = res
        if res is None:
            print(f"  class {cls}: no non-empty boxes")
            continue
        top = sorted(res["influence"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  class {cls}: {res['n_anchors_used']}/{res['n_anchors_total']} boxes, "
              f"prec={res['mean_precision_prediction_match']:.3f}, "
              f"cov={res['mean_coverage']:.4f}, top={[f'{a}:{b:+.3f}' for a, b in top]}")

    ebm_json = json.load(open(args.ebm_json))
    comparison = compare_with_ebm(anchor_res, ebm_json, feature_names)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for cls, res in anchor_res.items():
        if res is None:
            serializable[f"class_{cls}"] = None
            continue
        serializable[f"class_{cls}"] = {k: v for k, v in res.items() if k != "_vectors"}

    payload = {
        "dataset": args.dataset,
        "rules_file": str(args.rules_file),
        "experiment_dir": experiment_dir,
        "feature_names": feature_names,
        "bound_space": space,
        "mlp_accuracy_full_data": float((preds == y).mean()),
        "method": {
            "influence": "precision(box) - precision(box with feature j relaxed), "
                         "precision = P(MLP predicts class c | x in box)",
            "aggregation": "unweighted mean over the class's anchor boxes (each rule counts once)",
            "tightness": "1 - (box width on j / data range on j)",
        },
        "anchor_influence": serializable,
        "comparison_vs_ebm": comparison,
    }
    path = out_dir / "anchor_influence.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {path}")

    if comparison.get("global"):
        g = comparison["global"]
        print("\n=== GLOBAL: anchor influence vs EBM importance ===")
        print(f"  Spearman rho = {g['spearman_rho']:+.3f} (p={g['spearman_p']:.4f})")
        print(f"  {'feature':<12} {'anchor share':>13} {'EBM share':>11}")
        for f in sorted(feature_names, key=lambda f: -g["anchor_share"][f]):
            print(f"  {f:<12} {g['anchor_share'][f]:>12.1%} {g['ebm_share'][f]:>11.1%}")


if __name__ == "__main__":
    main()
