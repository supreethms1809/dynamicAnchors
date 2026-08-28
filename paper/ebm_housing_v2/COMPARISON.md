# EBM vs Dynamic Anchors — California Housing

Both methods explain the **same 4-class quartile decision problem** on the same 80/20 stratified split (seed 42).

## Models

| Model | Role | Test performance |
|---|---|---|
| EBM regressor | notebook reproduction check | R² = 0.8333 (notebook: 0.8332, terms identical: True) |
| EBM classifier | glass-box reference | accuracy = 0.7643 |
| MLP (dnn) | the model anchors explain | accuracy = 0.7643 (full data) |

> **Caveat on all reported accuracies.** Model selection maximises accuracy on X_test across epochs and the LR scheduler also steps on test accuracy, so these figures are optimistic. EBM's number is an honest held-out score; the MLP's is not. A validation split carved from training data would fix this and has not been implemented.

## How feature influence is defined

| | EBM | Dynamic anchors |
|---|---|---|
| Quantity | mean \|contribution of term to class-c logit\| | precision(box) − precision(box with feature *j* relaxed) |
| Explains | its own additive structure | the trained MLP's decision function |
| Aggregation | mean over test rows of class c | unweighted mean over the class's anchor boxes (each rule counts once) |
| Interactions | explicit 2D terms, split 50/50 onto their features | implicit — a box is a conjunction, so influence is already joint |

## Global agreement

Spearman ρ = **+0.643** (p = 0.0856)

Sensitivity — collapsing repeated rule feature-sets to one vote each (so `{MedInc}` counts once however many anchors use it): ρ = **+0.667** (p = 0.0710). A large gap between the two would mean the ranking is driven by how OFTEN a feature is used rather than how much it matters when used.

| Class | Distinct feature-sets | Median pairwise Jaccard | Max Jaccard | Pairs J>0.5 |
|---|---:|---:|---:|---:|
| class_0 | 25 | 0.000 | 0.667 | 0.3% |
| class_1 | 16 | 0.000 | 0.873 | 4.6% |
| class_2 | 10 | 0.062 | 0.737 | 1.4% |
| class_3 | 24 | 0.000 | 0.755 | 0.3% |

Near-zero Jaccard means the boxes cover essentially disjoint rows, so no region contributes its features twice. Feature-sets do repeat across those disjoint regions, which is what the dedup sensitivity above tests.

| Feature | Anchor influence share | EBM importance share | Anchor rank | EBM rank |
|---|---:|---:|---:|---:|
| Latitude | 24.4% | 35.7% | 3 | 1 |
| Longitude | 24.9% | 32.7% | 2 | 2 |
| MedInc | 40.6% | 13.3% | 1 | 3 |
| AveOccup | 0.0% | 7.9% | 8 | 4 |
| AveRooms | 2.4% | 5.4% | 5 | 5 |
| HouseAge | 7.0% | 2.9% | 4 | 6 |
| AveBedrms | 0.6% | 1.1% | 6 | 7 |
| Population | 0.2% | 0.9% | 7 | 8 |

## Per-class agreement

| Class | ρ | Anchor top-3 | EBM top-3 | Anchor boxes | Mean rule precision | Mean coverage |
|---|---:|---|---|---:|---:|---:|
| very_low_price | +0.83 | Latitude, Longitude, HouseAge | Latitude, Longitude, MedInc | 60 | 0.694 | 0.0190 |
| low_price | +0.69 | Longitude, Latitude, AveRooms | Longitude, Latitude, MedInc | 60 | 0.346 | 0.1645 |
| medium_price | +0.45 | MedInc, Longitude, Population | Latitude, Longitude, MedInc | 60 | 0.301 | 0.1548 |
| high_price | +0.10 | MedInc, HouseAge, Longitude | Latitude, Longitude, MedInc | 60 | 0.679 | 0.0352 |

![influence comparison](ebm_vs_anchors_influence.png)

## Notes

- Rule bounds were read in **raw** feature space.
- Anchor precision is prediction-match, P(MLP predicts class c | x in box) — the same object EBM explains, so the two are measuring influence on a model, not on labels.
- EBM interaction terms are attributed 50/50 to their two features; the main-effect-only breakdown is in `ebm_influence.json` if you prefer it.
- Training was truncated to 144k frames (30 iterations) for every run in this series; none is converged. Rules come from the FINAL extracted models (inference defaults to `prefer_model="final"`), not from best_model.
- Geometric tightness (in `anchor_influence.json`) can rank quite differently from ablation influence: the features the boxes pin most tightly are not always the ones carrying precision. Tight != important, which is why ablation is the headline metric.
- Influence is an UNWEIGHTED mean over anchors. An earlier coverage-weighted version was dropped: coverage is skewed enough (top 5 of 60 boxes carrying ~40-50% of the weight) that a few huge boxes decided each result, and huge boxes barely react to relaxing one bound. That version reported Latitude at a 0.0% share for runs whose rules constrain Latitude in a third of cases, and made the EBM rank correlation swing between +0.81 and 0.00 across runs whose unweighted values were stable.
- `binding_fraction` in the JSON separates 'this feature is unconstrained' from 'constrained but precision-neutral'. Single-feature ablation cannot see redundancy: Latitude and Longitude jointly define a neighbourhood, so relaxing one while the other still binds understates the pair. That is the most likely reason anchors under-weight Latitude against EBM's Latitude & Longitude interaction term.