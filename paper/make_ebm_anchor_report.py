#!/usr/bin/env python3
"""Build the EBM-vs-dynamic-anchors comparison figure and markdown report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

# Output dir may be overridden (e.g. paper/ebm_housing_v2 for the retuned run).
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "ebm_housing"

EBM_COLOR = "#4C78A8"
ANCHOR_COLOR = "#F58518"


def main():
    ebm = json.load(open(OUT / "ebm_influence.json"))
    anc = json.load(open(OUT / "anchor_influence.json"))

    features = ebm["feature_names"]
    class_names = ebm["class_names"]
    cmp_ = anc["comparison_vs_ebm"]
    g = cmp_["global"]

    # ---------- Figure ----------
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(4.2 * (n_classes + 1), 5.0), sharey=True)

    order = sorted(features, key=lambda f: -g["ebm_share"][f])
    ypos = np.arange(len(order))
    h = 0.38

    ax = axes[0]
    ax.barh(ypos + h / 2, [g["ebm_share"][f] for f in order], height=h,
            color=EBM_COLOR, label="EBM")
    ax.barh(ypos - h / 2, [g["anchor_share"][f] for f in order], height=h,
            color=ANCHOR_COLOR, label="Dynamic anchors")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_title(f"Global\n" + r"Spearman $\rho$ = " + f"{g['spearman_rho']:+.2f}", fontweight="bold")
    ax.set_xlabel("share of total influence")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)

    for i, cname in enumerate(class_names):
        key = f"class_{i}"
        ax = axes[i + 1]
        if key not in cmp_["per_class"]:
            ax.set_title(f"{cname}\n(no anchors)")
            ax.axis("off")
            continue
        c = cmp_["per_class"][key]
        ax.barh(ypos + h / 2, [c["ebm_share"][f] for f in order], height=h, color=EBM_COLOR)
        ax.barh(ypos - h / 2, [c["anchor_share"][f] for f in order], height=h, color=ANCHOR_COLOR)
        ax.set_yticks(ypos)
        ax.invert_yaxis()
        ax.set_title(f"{cname}\n" + r"$\rho$ = " + f"{c['spearman_rho']:+.2f}")
        ax.set_xlabel("share of total influence")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Feature influence on the same California Housing decision problem:\n"
        "EBM (glass-box, exact additive terms) vs dynamic anchors (rules over an MLP)",
        fontweight="bold", y=1.02)
    fig.tight_layout()
    fig_path = OUT / "ebm_vs_anchors_influence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {fig_path}")

    # ---------- Markdown ----------
    clf = ebm["comparison_classifier"]
    ver = ebm["verification_regressor"]
    lines = []
    A = lines.append
    A("# EBM vs Dynamic Anchors — California Housing\n")
    A("Both methods explain the **same 4-class quartile decision problem** on the same "
      "80/20 stratified split (seed 42).\n")

    A("## Models\n")
    A("| Model | Role | Test performance |")
    A("|---|---|---|")
    A(f"| EBM regressor | notebook reproduction check | R² = {ver['test_r2']:.4f} "
      f"(notebook: {ver['notebook_test_r2']}, terms identical: {ver['term_list_matches_notebook']}) |")
    A(f"| EBM classifier | glass-box reference | accuracy = {clf['test_accuracy']:.4f} |")
    acc = anc["mlp_accuracy_full_data"]
    A(f"| MLP (dnn) | the model anchors explain | accuracy = {acc:.4f} (full data) |")
    A("")
    if acc < 0.70:
        A("> **Caveat on the MLP.** This run predates the best-model checkpoint fix, so the saved "
          "classifier is the FINAL epoch rather than the best one — `tabular_datasets.py` snapshotted "
          "the best epoch with `state_dict().copy()`, a shallow dict copy that aliases the live tensors, "
          "making the restore a no-op. The log's \"best test accuracy\" therefore overstates the model "
          "actually explained here (0.7418 reported vs 0.6672 real on the v1 run). Later runs use the "
          "fixed path and a larger [1024,1024,512] network.\n")
    A("> **Caveat on all reported accuracies.** Model selection maximises accuracy on X_test across "
      "epochs and the LR scheduler also steps on test accuracy, so these figures are optimistic. "
      "EBM's number is an honest held-out score; the MLP's is not. A validation split carved from "
      "training data would fix this and has not been implemented.\n")

    A("## How feature influence is defined\n")
    A("| | EBM | Dynamic anchors |")
    A("|---|---|---|")
    A("| Quantity | mean \\|contribution of term to class-c logit\\| | precision(box) − precision(box with feature *j* relaxed) |")
    A("| Explains | its own additive structure | the trained MLP's decision function |")
    A("| Aggregation | mean over test rows of class c | unweighted mean over the class's anchor boxes (each rule counts once) |")
    A("| Interactions | explicit 2D terms, split 50/50 onto their features | implicit — a box is a conjunction, so influence is already joint |")
    A("")

    A("## Global agreement\n")
    A(f"Spearman ρ = **{g['spearman_rho']:+.3f}** (p = {g['spearman_p']:.4f})\n")
    if "spearman_rho_featureset_dedup" in g:
        A(f"Sensitivity — collapsing repeated rule feature-sets to one vote each "
          f"(so `{{MedInc}}` counts once however many anchors use it): "
          f"ρ = **{g['spearman_rho_featureset_dedup']:+.3f}** "
          f"(p = {g['spearman_p_featureset_dedup']:.4f}). A large gap between the two "
          f"would mean the ranking is driven by how OFTEN a feature is used rather than "
          f"how much it matters when used.\n")
    ov = []
    for k, v in sorted((anc.get("anchor_influence") or {}).items()):
        if v and "rule_overlap" in v:
            ov.append((k, v["n_distinct_feature_sets"], v["rule_overlap"]))
    if ov:
        A("| Class | Distinct feature-sets | Median pairwise Jaccard | Max Jaccard | Pairs J>0.5 |")
        A("|---|---:|---:|---:|---:|")
        for k, nfs, o in ov:
            A(f"| {k} | {nfs} | {o['median_pairwise_jaccard']:.3f} | "
              f"{o['max_pairwise_jaccard']:.3f} | {o['frac_pairs_jaccard_gt_0.5']:.1%} |")
        A("")
        A("Near-zero Jaccard means the boxes cover essentially disjoint rows, so no region "
          "contributes its features twice. Feature-sets do repeat across those disjoint "
          "regions, which is what the dedup sensitivity above tests.\n")
    A("| Feature | Anchor influence share | EBM importance share | Anchor rank | EBM rank |")
    A("|---|---:|---:|---:|---:|")
    ar, er = g["anchor_ranking"], g["ebm_ranking"]
    for f in order:
        A(f"| {f} | {g['anchor_share'][f]:.1%} | {g['ebm_share'][f]:.1%} | "
          f"{ar.index(f)+1} | {er.index(f)+1} |")
    A("")

    A("## Per-class agreement\n")
    A("| Class | ρ | Anchor top-3 | EBM top-3 | Anchor boxes | Mean rule precision | Mean coverage |")
    A("|---|---:|---|---|---:|---:|---:|")
    for i, cname in enumerate(class_names):
        key = f"class_{i}"
        if key not in cmp_["per_class"]:
            A(f"| {cname} | — | no anchors | — | 0 | — | — |")
            continue
        c = cmp_["per_class"][key]
        a = anc["anchor_influence"].get(key) or {}
        A(f"| {cname} | {c['spearman_rho']:+.2f} | {', '.join(c['anchor_top3'])} | "
          f"{', '.join(c['ebm_top3'])} | {a.get('n_anchors_used','—')} | "
          f"{a.get('mean_precision_prediction_match', float('nan')):.3f} | "
          f"{a.get('mean_coverage', float('nan')):.4f} |")
    A("")

    A("![influence comparison](ebm_vs_anchors_influence.png)\n")

    A("## Notes\n")
    A(f"- Rule bounds were read in **{anc['bound_space']}** feature space.")
    A("- Anchor precision is prediction-match, P(MLP predicts class c | x in box) — the "
      "same object EBM explains, so the two are measuring influence on a model, not on labels.")
    A("- EBM interaction terms are attributed 50/50 to their two features; the "
      "main-effect-only breakdown is in `ebm_influence.json` if you prefer it.")
    A("- Training was truncated to 144k frames (30 iterations) for every run in this series; "
      "none is converged. Rules come from the FINAL extracted models (inference defaults to "
      "`prefer_model=\"final\"`), not from best_model.")
    A("- Geometric tightness (in `anchor_influence.json`) can rank quite differently from "
      "ablation influence: the features the boxes pin most tightly are not always the ones "
      "carrying precision. Tight != important, which is why ablation is the headline metric.")
    A("- Influence is an UNWEIGHTED mean over anchors. An earlier coverage-weighted version was "
      "dropped: coverage is skewed enough (top 5 of 60 boxes carrying ~40-50% of the weight) that "
      "a few huge boxes decided each result, and huge boxes barely react to relaxing one bound. "
      "That version reported Latitude at a 0.0% share for runs whose rules constrain Latitude in "
      "a third of cases, and made the EBM rank correlation swing between +0.81 and 0.00 across "
      "runs whose unweighted values were stable.")
    A("- `binding_fraction` in the JSON separates 'this feature is unconstrained' from "
      "'constrained but precision-neutral'. Single-feature ablation cannot see redundancy: "
      "Latitude and Longitude jointly define a neighbourhood, so relaxing one while the other "
      "still binds understates the pair. That is the most likely reason anchors under-weight "
      "Latitude against EBM's Latitude & Longitude interaction term.")

    md_path = OUT / "COMPARISON.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
