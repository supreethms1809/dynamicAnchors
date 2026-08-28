---
type: Note
---
# Reviewer's Comments Dynamic Anchors

Reviewer 1

The idea proposed by the authors is promising: amortizing anchor discovery with continuous-control RL, and using multi-agent coordination to reduce redundant or conflicting rule regions, is a plausible research direction. However, the manuscript currently has several serious issues that affect correctness, coherence, and evidential support.

The most important problems are:

Lack of strong references. The related work claim that neural decision trees' hardening typically degrades precision and performance may be true in many settings, but it needs evidence or a citation. Similarly, the claim that continuous-control rule discovery is the key novelty needs positioning against existing continuous threshold optimization methods.

The manuscript would benefit from a substantially stronger and more systematic positioning within the XAI literature. At present, the related-work discussion is relatively narrow: it moves mainly from classical rule learners such as CN2/RIPPER to Anchors, neural decision trees, and a few RL/tree-distillation approaches, before motivating the proposed continuous-control formulation. This does not sufficiently situate the work within the broader XAI taxonomy, nor does it adequately connect the paper’s terminology (local vs. global explanations, model-agnostic post-hoc explanations, fidelity, coverage, explanation targets, and evaluation of explanations) to established conceptual frameworks in the field.

I recommend that the authors revise the Introduction, Related Work, and Evaluation Metrics sections to include and engage with stronger foundational and recent XAI references, not merely as additional citations but as part of the conceptual framing of the contribution. In particular, the paper should discuss (briefly and supported by references) how RLDA and MADA fit within standard XAI taxonomies: whether they are local, global, or semi-global; model-agnostic or model-specific; instance-level or class-level; post-hoc or intrinsic; and whether their objective is model fidelity, data/label purity, human interpretability, or decision-region summarization. A few relevant references came to my mind to be considered include:

- **Barredo Arrieta et al. (2020), Information Fusion, DOI: 10.1016/j.inffus.2019.12.012. This survey provides an important conceptual and responsible-AI framing of XAI, including transparency, interpretability, comprehensibility, regulatory motivations, and ethical considerations.**
- **Ali et al. (2023), Information Fusion, DOI: 10.1016/j.inffus.2023.101805. This more recent survey offers a comprehensive organization of XAI methods along data, model, post-hoc explainability, and explanation-assessment axes, and would help connect the manuscript to the broader trustworthy-AI discussion.**
- **Weber et al. (2023), Information Fusion, DOI: 10.1016/j.inffus.2022.11.013. This work is relevant because it studies how explanations can be used not only for post-hoc interpretation but also for model improvement, data curation, feature engineering, and architecture refinement. It may help the authors clarify whether their method is only explanatory or could also support model debugging and refinement.**
- **Ortigossa et al. (2024), IEEE Access, DOI: 10.1109/ACCESS.2024.3409843. This broad theory-to-practice overview covers many of the same methodological tools invoked in the manuscript, including SHAP, LIME, Anchors, and other post-hoc methods. It would help the authors avoid presenting a partial view of the XAI toolbox.**
- **Nauta et al. (2023), ACM Computing Surveys, DOI: 10.1145/3583558. This systematic review of XAI evaluation is highly relevant. It proposes conceptual properties for evaluating explanations and shows that many XAI papers rely too heavily on anecdotal examples. Given that the present manuscript uses selected rules, plots, precision/coverage tables, and overlap matrices, it should explicitly relate its evaluation protocol to established XAI evaluation properties such as fidelity, stability, compactness, robustness, completeness, and human interpretability.**

Furthermore, the paper confuses model fidelity with ground-truth label purity. For XAI, anchors should explain the black-box model’s behavior, not the ground-truth labels. The original Anchors paper frames anchors as high-precision, model-agnostic rules explaining the behavior of a complex model and representing local sufficient conditions for predictions. The manuscript's evaluation section instead defines final hard precision using y = c on the dataset labels. That is not a fidelity metric unless the black-box classifier is perfect.

Several mathematical definitions are wrong or underspecified. Examples include the action dimension in RLDA, the sign of the MADA width constraint, undefined or ambiguous reward terms, unclear transition/state definitions, and an equilibrium analysis that does not establish actual Nash or e-Nash convergence.

The empirical claims are not supported by the tables. The paper repeatedly claims that RLDA improves coverage, but Table 1 shows RLDA and MADA have much lower average instance coverage than classical Anchors on almost every dataset and class. Many reported instance coverages are approximately one sample, making high precision almost trivial.

The experimental design is too weak for a top-tier venue. There are no confidence intervals, seeds, runtime comparisons, ablation on reward terms, exact hyperparameters, model accuracies per dataset, success rates, or fair class-level Anchors baseline. The datasets are also small, despite scalability claims.

There are visible inconsistencies in preprocessing and rule reporting. The paper says data are normalized to [0,1]^d and later says Iris rule bounds are denormalized to original units, but the reported Iris thresholds include negative petal/sepal lengths such as petal length <= -1.23, which are impossible in centimeters and inconsistent with [0,1] normalization.=

Several claims are too strong:

- The abstract claim that RLDA provides more precise rules and comparable performance to Anchors is only partially supported. In Table 1, RLDA sometimes improves precision, especially on UCI Credit, but this comes with a dramatic coverage collapse. For Wine, RLDA instance coverage is around 0.006--0.008, while baseline Anchors coverage is around 0.117--0.165. For Breast Cancer, RLDA instance coverage is 0.002, roughly one test sample depending on split size. High precision at near-zero coverage is not strong evidence of useful explanations. The introduction says RLDA achieves precision comparable to Anchors while significantly improving coverage. That statement is also contradicted by Table 1 for instance-level coverage.
- The claim that MADA provides a more global and structured explanation is plausible, but not demonstrated rigorously. The paper reports class-union metrics and overlap matrices, but it does not provide a fair global baseline, statistical testing, or a complete measure of cross-class ambiguity.
- The phrase operating under defined equilibrium conditions is misleading. The manuscript defines an approximate Nash-style metric, but it does not prove convergence, and the empirical NashConv calculation appears to be only an approximate critic-based diagnostic.

In Section 3, we have:

- Equation 2 uses D\_x(z | A) and f̂, but these are not clearly defined. Equation 3 writes E\_D(x)[A(z)], mixing x and z. These are notation problems, but they matter because the later method changes the meaning of precision and coverage.
- The problematic sentence is that algorithms often utilize fictitious play or a variant in cooperative MARL. That is too broad and not well supported. Many MARL algorithms do not use fictitious play, and MADDPG is not naturally described as a fictitious-play variant.

In Section 4, we have:

- several reward components depend on variables not included in the state: the previous box for trajectory drift, the initialization centroid for anchor drift, progress schedules, termination counters, and possibly perturbation/sampling configuration. This can be made Markov by including these variables or defining the reward carefully over transitions, but the manuscript does not do so.
- the action dimension is wrong. The paper says the action is (\\Delta l\_t, \\Delta u\_t) ∈ [-1,1]^d, but this is two d-dimensional vectors, so the action should be in [-1,1]^{2d}. The MADA section later uses 2d, so RLDA is internally inconsistent.
- the update equation is incomplete. The text says actions are scaled by box width (Equation 6). If the action is scaled by current width, the equation should explicitly include that scaling, for example with w\_t = u\_t - l\_t and a step-size parameter.
- the reward function is not reproducible. Terms such as R\_cov, R\_purity, D\_t, A\_t, G\_t, and M\_t are described verbally but not mathematically defined. The phrase proxy for Jensen-Shannon volume stability is especially concerning: Jensen-Shannon divergence is a distributional divergence, not a generic volume stability measure. If the authors use a volume-ratio or geometric penalty, it should be defined as such.

In Section 5, the multi-agent motivation is fine. However, the formalization is not coherent. The paper says there are one or more agents per class, then defines J = {1,...,K} as the agent/class index. But if there are M anchors per class, the number of agents is K \* M, not K. Later, the paper says each class player is a coalition of agents. This could be a valid formulation, but the notation never cleanly distinguishes: class index, anchor index, individual agent, class coalition/player, joint policy over agents in a class. The reward equation is also ambiguous. It is unclear whether w\_shared multiplies only the global-success term or the whole shared-reward bracket.

Section 6 is the most serious conceptual failure. The paper says training uses a soft/probability-weighted proxy precision based on model prediction stability, but final evaluation uses hard precision computed from actual ground-truth labels. This is not strict fidelity. In post-hoc XAI, fidelity should be with respect to the black-box model's predictions, not the ground truth. A rule can be perfectly faithful to a wrong classifier prediction, and a ground-truth-pure rule can fail to describe the classifier.

Section 7. The dataset choices are too small to support scalability claims. Iris, Wine, Breast Cancer, and UCI Credit are useful smoke tests, but they are not sufficient for a top-tier empirical claim about scalable XAI. The paper calls the range small to large, but the largest reported dataset has only 600 samples. This is not large.

The Iris examples reveal a major preprocessing/reporting inconsistency. The paper states that learning occurs in normalized space and rules are reported in original units after denormalization. But the reported Iris rules contain negative values. These are not original Iris centimeter units.

The authors acknowledge important limitations: data efficiency, imbalance, computational cost, and rule overload. That is positive. But the limitations section omits several deeper limitations, such as: fidelity is evaluated against labels rather than model predictions; no strong baseline for class-level rule sets; no reproducibility details; no statistical testing; unstable reward design; no categorical-feature treatment; no guarantee that learned boxes satisfy anchor precision constraints; no discussion of out-of-distribution perturbations; no human interpretability study; no runtime comparison despite claiming amortized inference

Therefore, I do not recommend acceptance in its current form. The idea could become a strong paper after substantial revision, but the current version has enough conceptual, mathematical, and empirical problems that the main claims are not yet reliable

Reviewer 2

Summary:

The manuscript proposes RLDA and MADA, two extensions of classical Anchors for explaining tabular classifiers. RLDA formulates anchor discovery as a reinforcement learning problem, while MADA extends this to a cooperative multi-agent setting intended to produce more coordinated class-level rules.

Strengths:

I found the idea interesting and well motivated. Classical Anchors are local and computed independently for each instance, so learning reusable policies and coordinating rules across classes is a promising direction. The paper also includes useful analyses, including class-union metrics, overlap matrices, threshold studies, and NashConv diagnostics.

Weaknesses:

That said, I think the manuscript needs major clarification before the results can be interpreted reliably.

My main concern is the evaluation target. Anchors are usually evaluated by fidelity to the black-box model, meaning whether the model keeps making the same prediction inside the anchor region. In the manuscript, final hard precision appears to be computed using ground-truth labels. This measures class purity or label agreement, not necessarily explanation fidelity. Both quantities can be useful, but they should be clearly separated. Since this is an XAI paper, the authors should explicitly report fidelity-to-model precision.

The evaluation protocol for class-level and class-union metrics also needs to be clearer. The paper states that instance-level metrics are evaluated on a held-out test set, while class-level metrics are computed on the full dataset. Because many of the main claims about MADA rely on these class-level results, the authors should evaluate them on held-out data as well, or clearly explain the data split used for training, anchor selection, and evaluation.

Some conclusions should also be softened. The results are promising, but the tables show a mixed picture. In particular, RLDA and MADA often have much lower average instance coverage than classical Anchors. This limitation is mentioned in the discussion, but it should be reflected more clearly in the abstract, results, and conclusion.

The comparison with classical Anchors should be strengthened. Since MADA is mainly proposed as a class-level/global explanation method, it would be helpful to compare it with a simple baseline that aggregates classical Anchors across multiple instances. Without this, it is difficult to know whether MADA improves over a simpler rule-selection or rule-merging approach.

The NashConv analysis is interesting, but should be presented more cautiously. More details are needed on how best responses and exploitability are estimated. Otherwise, I would describe NashConv mainly as a training diagnostic, rather than strong evidence of convergence to a meaningful equilibrium.

Finally, the paper would benefit from clearer reproducibility and robustness details: train/test splits, random seeds, classifier architectures and accuracies, RL/MARL hyperparameters, reward weights, initialization settings, checkpoint selection, and the number of rules used in the union metrics. Since RL and MARL methods can be sensitive to initialization and seeds, reporting variability across runs would also strengthen the empirical claims. The authors should also clarify preprocessing and rule units, since the method is described in normalized ([0,1]) space but some reported rule bounds are negative.

Recommendation:

Overall, I think this is a promising contribution, but the current version does not yet support all of its claims clearly enough. I recommend major revision, focused on clarifying the evaluation, adding fidelity-to-model metrics, evaluating class-level results on held-out data, improving the baseline comparison, and aligning the conclusions more closely with the reported evidence.


## Summary

Both reviews land on the same root cause, so this is more tractable than it looks — but it is a rebuild of Sections 4–7, not a patch. Let me separate what they said, what they *missed* (there's a serious one), and what you actually have to run.

---

## 1. The verdict, read honestly

R1 is a reject-in-current-form. R2 is a major revision. They converge on three things:

1. **You measure the wrong thing** (ground-truth purity, not model fidelity).
2. **Your headline claims are contradicted by your own Table 1** (coverage).
3. **The formalism has enough errors that the method isn't reproducible from the paper.**

Everything else — references, dataset size, seeds — is secondary and mechanical. If you fix only the references and add citations, you will be rejected again.

Realistic scope: 3–5 months of work. Most of it is re-running evaluation and adding baselines, not new method development. The method itself is not what they're attacking.

---

## 2. The three decisive issues

### A. Fidelity vs. label purity

This is the one that would sink the paper on its own, and both reviewers independently flagged it.

Your Eq. 2 (background) correctly states anchor precision as $\\mathbb{E}[\\mathbb{1}\_{\\hat f(x)=\\hat f(z)}]$ — agreement with the *model*. Your Eq. 19 (evaluation) then defines $\\text{Precision}(B,c) = P(y=c \\mid x\\in B)$ — agreement with the *labels*. Those are different quantities and the paper contradicts itself between Section 3 and Section 6.

Why it matters technically: your MLPs are 80–90+% accurate. A rule can be 100% faithful to a wrong prediction region and score badly on label purity, and vice versa. On UCI Credit at ~80% accuracy, up to a fifth of your precision signal is measuring classifier error, not explanation quality.

**Fix:** make fidelity primary. Define
@@TOLARIA\_MATH\_BLOCK:5c-74-65-78-74-7b-46-69-64-7d-28-42-2c-63-29-3d-50-5c-62-69-67-28-5c-68-61-74-20-66-28-78-29-3d-63-20-5c-6d-69-64-20-78-5c-69-6e-20-42-5c-62-69-67-29-2c-5c-71-71-75-61-64-20-5c-74-65-78-74-7b-50-75-72-7d-28-42-2c-63-29-3d-50-5c-62-69-67-28-79-3d-63-20-5c-6d-69-64-20-78-5c-69-6e-20-42-5c-62-69-67-29@@
Report both in every table, with fidelity as the headline. Add a sentence stating that the gap between them is bounded by classifier error on the region, and report per-dataset classifier accuracy so the reader can calibrate.

Good news: your *training* reward already uses model probabilities ("probability-weighted proxy precision based on model prediction stability"). So the policies are already optimizing roughly the right thing — **you likely do not need to retrain, only re-evaluate.** Verify this in the code before you promise it in the response letter. Also: define that proxy mathematically. Right now it's a sentence, and R1 explicitly called out the training/eval precision mismatch as underspecified.

Also fix Eq. 19's typo: "$I$ is a trained model" — $\\mathbb{I}$ is the indicator function.

### B. The coverage claim is false as written

Your intro says RLDA "achieves precision comparable to that of classical anchors while significantly improving coverage." Table 1 says:

| Dataset | Anchors instC | RLDA instC |
| --- | --- | --- |
| Wine c0 | 0.165 | 0.008 |
| Wine c1 | 0.117 | 0.006 |
| Breast cancer c0 | 0.168 | 0.002 |
| Breast cancer c1 | 0.158 | 0.002 |

That's a 20–80× *degradation*. R1 is right and you cannot argue with it. Worse: at instC = 0.002 on a ~170-sample test set, the average anchor covers **less than one sample**. Precision = 1.000 on zero-or-one samples is not a measurement.

Two separate problems here, and you must handle both:

**B1 — the writing.** Retract the instance-coverage claim entirely. The honest claim is: *RLDA/MADA trade instance coverage for a reusable policy and for class-level union coverage.* See §6 below for draft replacement sentences.

**B2 — the diagnosis.** Why is coverage collapsing when your termination conditions require $C \\ge 0.5\\tau\_C$ (i.e. ≥ 0.1 at $\\tau\_C=0.2$), yet you report 0.002? My hypothesis — and I want to be clear this is a hypothesis, I can't see your code:

> Training coverage is estimated on the *locally perturbed* sample set around the instance; evaluation coverage is computed on the *real* test set. The perturbed set is concentrated near the box, so training coverage is inflated by orders of magnitude, the termination check passes, and the box that satisfies it covers almost nothing real.

If that's what's happening, it's a train/eval mismatch and it's fixable: compute the coverage used for termination on real data (or a real-data holdout mini-batch), not on perturbations. Check this first — it may substantially improve your numbers and change what you can claim. Also check what your code does when $|B \\cap D\_{\\text{test}}| = 0$; if it returns precision 1.0 instead of NaN, several table entries are meaningless and you must exclude or flag them.

### C. Formalism errors

All confirmed, all cheap to fix:

- **Action dimension.** §4.2 says $(\\Delta\\ell\_t,\\Delta u\_t)\\in[-1,1]^d$; two $d$-vectors is $[-1,1]^{2d}$. MADA already uses $2d$. Fix RLDA.
- **Width constraint sign.** §5.2 writes $\\ell\_{k,j,t+1}-u\_{k,j,t+1}\\ge w\_{\\min}$. Should be $u-\\ell \\ge w\_{\\min}$.
- **Eq. 6 is incomplete.** You say actions are scaled by box width but the equation doesn't show it. Write it properly: $w\_t = u\_t-\\ell\_t$, $\\ell\_{t+1}=\\text{clip}(\\ell\_t+\\kappa\\, w\_t \\odot a^{(\\ell)}\_t)$, and state $\\kappa$.
- **Eq. 2/3 notation.** $\\mathcal{D}\_x(z\\mid A)$ and $\\hat f$ undefined; Eq. 3 mixes $x$ and $z$. Write $\\text{cov}(A)=\\mathbb{E}\_{z\\sim\\mathcal{D}}[\\mathbb{1}\\{z\\in A\\}]$.
- **Reward terms are verbal only.** $R\_{\\text{cov}}, R\_{\\text{purity}}, D\_t, A\_t, G\_t, M\_t, B\_{\\text{global cov}}, P\_{\\text{inter}}, P\_{\\text{same}}$ all need equations. Put them in a table with symbol, formula, default weight.
- **"Proxy for Jensen–Shannon volume stability"** (the $G\_t$ term) — delete this phrase. JS divergence is a divergence between distributions; it is not a volume-stability measure. If you're computing a volume ratio or a geometric feasibility penalty, say exactly that. R1 singled this out and it reads as a red flag about technical care.
- **Markov violation.** $D\_t$ depends on $B\_{t-1}$, $A\_t$ on $x^\*$, $\\eta$ on a progress schedule, termination on counters. Either augment the state $s\_t \\supseteq (B\_{t-1}, x^\*, \\eta\_t, \\text{counters})$ or define reward over transitions $r(s\_t,a\_t,s\_{t+1})$ and put the rest in state. One paragraph.
- **MADA agent indexing is broken.** You set $J=\\{1,\\dots,K\\}$ as the agent index, then say $M$ anchors per class (so $K\\cdot M$ agents), then say players are class coalitions. Introduce three distinct symbols: agent $(k,i)$, class $k$, coalition player $c$. Then restate Eqs. 13–18 with them.
- **Eq. 13 ambiguity.** State explicitly whether $w\_{\\text{shared}}$ multiplies only the bracket or the whole expression including $B\_{\\text{global cov}}$.
- **Equilibrium overclaim.** "Operating under defined equilibrium conditions" in the abstract must go. You define NE/ε-NE and then never establish either. Reframe NashConv as a *training-stability diagnostic* (R2's exact suggestion), and add the estimation details: how many gradient-ascent steps for the best response, what step size, and the caveat that $Q\_c$ is a learned approximation so $\\Delta\_c$ is a biased lower bound on true exploitability.

One structural note on the game theory, since it's your research area: your reward is *mixed*, not identical-interest — agents have local terms plus a shared term. You should say which, because in a common-payoff game NE is a weak solution concept (every local optimum is one) and reporting NashConv would prove little. The mixed structure is what makes it non-trivial. Make that argument explicitly; it's a strength you're currently leaving on the table.

---

## 3. What the reviewers missed — and someone will catch next round

These are mine, not theirs. The first one is serious.

### 3.1 Several union-coverage entries are mathematically impossible

Union coverage over a rule set must be **≥** the coverage of any single rule in that set. Your tables violate this repeatedly:

| Table | Dataset / method / class | best clsC | unionC |
| --- | --- | --- | --- |
| 1 | Iris RLDA c1 | 0.740 | **0.440** |
| 1 | Wine RLDA c2 | 0.521 | **0.396** |
| 1 | Breast cancer RLDA c0 | 0.321 | **0.090** |
| 1 | Breast cancer RLDA c1 | 0.678 | **0.081** |
| 1 | Breast cancer MADA c1 | 0.877 | **0.487** |
| 1 | UCI credit MADA c1 | 0.966 | **0.266** |
| 2 | Wine RLDA c2 | 0.708 | **0.396** |

Both Eq. 21 and Eq. 23 are class-conditional, so the denominators match — this isn't a marginal-vs-conditional artifact.

Most likely explanation: the union is taken over *unique* rules only (consistent with your UCI Credit c0 = 0.000 case, "RLDA fails to produce any unique rules"), and the best rule is being filtered out before the union. If so, you are reporting a "best rule" that is not in the rule set you evaluate — which is misleading and must be stated. Otherwise it's a bug. **Check this before anything else.** If a reviewer finds a mathematical impossibility in your results table on round two, you're done.

Related: reporting unionP = 0.000 for an empty rule set is wrong. Precision of an empty set is undefined — report N/A.

### 3.2 Eq. 11 is a no-op

$w\_P=\\max(2.0,\\ 1+\\frac{P\_t-\\tau\_P}{1-\\tau\_P})$ when $P\_t\\ge\\tau\_P$. For $P\_t\\in[\\tau\_P,1]$ the second argument lies in $[1,2]$, so the max is always 2.0 — identical to the else-branch. Your "dynamic precision weight" is the constant 2. Either the formula is a typo or the mechanism doesn't exist. And §5.2.1 says the local reward is "the same as Eq. 7," which uses fixed $\\alpha=0.7,\\beta=0.6$ — so it's unclear whether $w\_P,w\_C$ are used at all. Reconcile.

### 3.3 Duplicated / inconsistent rows across the three ablation tables

- Anchors baseline for Breast Cancer: (0.990, 0.168), (0.990, 0.160), (0.996, 0.168) across Tables 1/2/3. The Anchors baseline does not depend on $\\tau\_C$. Either you resampled instances (→ you need seeds and CIs, which is R1's point anyway) or these are transcription errors.
- UCI Credit RLDA clsP/clsC = 0.459/0.606 and 0.762/0.775 are **byte-identical in all three tables**. So are several MADA instC = 0.002 entries. This looks like copied rows and a reviewer will read it as fabrication risk. Regenerate all tables programmatically from result files.

### 3.4 The amortization claim is never tested

Your central selling point — a *reusable* policy that amortizes cost — has no supporting experiment. No inference time, no query count, no explicit held-out-instance generalization test. R1 noticed the missing runtime; nobody noticed that this is the paper's main claim. Fix with P0 experiment #4 below.

### 3.5 Categorical features

UCI Credit (crx) is 6 continuous + 9 categorical attributes. Your method constructs axis-aligned boxes in continuous space. How did you handle the 9 categoricals? One-hot + continuous interval bounds gives rules like "$0.3 \\le \\text{is\_category\_A} \\le 0.8$," which is not human-readable and not a valid logical condition. The paper never mentions this. Classical Anchors handles categoricals natively, so this is also a fairness issue in the baseline comparison. Address explicitly, even if the answer is "we restrict boxes to continuous features and fix categoricals to the instance value."

### 3.6 How does a rule become sparse?

Your reported rules use 1–2 features out of 4/13/30. There is no sparsity term anywhere in Eq. 7. Presumably a dimension is dropped from the printed rule when its interval spans nearly the full range — but that mechanism is never described, and the threshold for "nearly" is a free parameter that materially changes the reported rule. Document it. Better: add an explicit complexity penalty (number of active dimensions), which also gives you a principled answer to your own "rule overload" limitation and to Nauta's *compactness* property.

### 3.7 MADA observation space vs. the overlap penalty

Agent $k$ observes only $(\\ell\_k, u\_k, P\_k, C\_k)$ — its own box. But $P\_{\\text{inter}}$ and $P\_{\\text{same}}$ depend entirely on *other* agents' boxes. From the agent's local view, the overlap penalty is an unexplained non-stationary reward shift. CTDE's centralized critic sees the joint state, so it's learnable in principle, but you should either (a) justify why the local observation suffices, or (b) run an ablation adding other agents' box summary statistics to $o\_{k,t}$. My guess is (b) helps a lot and it's a cheap win.

---

## 4. New results you must collect

Prioritized. P0 = the paper fails without it.

### P0

**1. Fidelity re-evaluation.** All datasets × classes × methods, reporting Fid and Pur side by side, plus per-dataset classifier train/test accuracy. Re-evaluation only, no retraining (verify first).

**2. Held-out class-level evaluation.** R2 asked directly. Three-way split: train the black box and the policies on train; select/rank rules on validation; report all class-level and class-union metrics on test. Currently class-level metrics use the full dataset, so rule selection and evaluation share data — that's leakage and it inflates every MADA claim.

**3. Seeds and confidence intervals.** ≥5 seeds (10 preferred). Every number becomes mean ± 95% CI. Add paired Wilcoxon signed-rank tests vs. Anchors over instances. Also report **Wilson intervals on precision** — this is what makes "precision 1.000 on one sample" visibly meaningless instead of impressive, and it's better to surface that yourself than to have a reviewer do it.

**4. Runtime and query budget.** One-time training cost (wall-clock, hardware) vs. per-explanation inference cost, for RLDA/MADA vs. Anchors. Also count black-box queries per explanation — that's the cleaner, hardware-independent number. Report break-even: after how many explained instances does amortization pay off? This single plot defends your core claim.

**5. A fair class-level Anchors baseline.** R2 is right that without this, MADA's global claim is untestable. Build at least:

- **SP-Anchors:** submodular pick over per-instance anchors (this is Ribeiro et al.'s own local→global recipe — its absence is the most glaring baseline gap), then union the selected rules per class.
- **Greedy set-cover union** of instance anchors under a precision constraint, matched to MADA's rule count.
- A **depth-limited CART surrogate** trained on *model predictions*, complexity-matched.

**6. Global rule-set fidelity.** This is the experiment that would actually earn "more global and structured explanation," and neither reviewer asked for it. Turn the class-union anchors into a classifier: predict $k$ if $x\\in\\bigcup\\mathcal{A}\_k$, tie-break by rule precision, abstain if no rule fires. Report:

- **Global fidelity:** agreement with the black box over $\\mathcal{D}\_{\\text{test}}$
- **Abstention rate:** fraction covered by no rule
- **Conflict rate:** fraction covered by ≥2 *different-class* rule sets

Compare RLDA / MADA / SP-Anchors / CART. Conflict rate is exactly R2's "complete measure of cross-class ambiguity," and it converts your overlap heatmaps from anecdote into a number.

**7. Fix the normalization/units reporting.** Both reviewers caught the negative Iris "centimeters." Your Fig. 9 axes literally say "z-score," so the pipeline is StandardScaler, not min-max. Either change all the math (`0 ≤ ℓ < u ≤ 1`) to standardized space with bounds in $\\mathbb{R}^d$, or actually switch to min-max. Then **report every rule in original units** — a rule that says "petal length ≤ −1.23" fails the paper's own interpretability premise.

### P1

**8. Precision–coverage frontier curves.** Since raw coverage differs by 50×, single-point comparison is meaningless. Sweep $\\tau\_P \\times \\tau\_C$ and plot the frontier per dataset for all methods. Then compare **precision at matched coverage**. This is the fair comparison and it may well show RLDA/MADA are competitive where the current tables show a rout.

**9. Reward ablation.** R1 asked. Drop one at a time: $P\_{\\text{inter}}$, $P\_{\\text{same}}$, $R\_{\\text{shared}}$, $R\_{\\text{purity}}$, $R\_{\\text{class}}$, static vs. dynamic $\\eta$. Report Δ on fidelity, union coverage, conflict rate.

**10. Agents-per-class ablation.** $M\\in\\{1,2,3,5\\}$. You currently assert $M{=}3$ with no justification, and "increasing the number of agents per class... can lead to more non-overlapping rules" is stated without evidence.

**11. Two larger datasets.** 600 samples is not "large" and R1 says so. Adult (~48k, 14 feat), Bank Marketing (~45k), HELOC/FICO (~10k, 23 feat), Covertype subsample. You need ≥2 with >10k rows. **If you can't run these, delete every scalability claim from the paper** — that's a legitimate and much cheaper option, and I'd rather you scope honestly than overclaim on Iris.

**12. Non-RL optimization baseline.** Random search and/or CMA-ES over box bounds, matched query budget, per instance. If simple black-box optimization matches RLDA per-instance, then RL's value is *only* amortization — which is fine, but you need to know it and say it. A reviewer will ask "why RL?" and right now you have no answer.

**13. Stability.** Re-explain the same instance across seeds and perturbation draws; report Jaccard/IoU of resulting boxes. Nauta's *stability* property, and RL is exactly where a reviewer expects instability.

### P2

**14. Compactness metrics:** active features per rule, rules per class, total description length.
**15. Perturbation-distribution sensitivity:** your bootstrap-vs-uniform choice changes the meaning of precision and makes it non-comparable to Anchors' $\\mathcal{D}\_x(\\cdot|A)$. Show results under both.
**16. NashConv validation:** on one small case, compare your critic-gradient best response against a brute-force/fine-grid best response to bound the approximation error.
**17. Categorical-aware variant** for crx.

---

## 5. Positioning and references

R1 handed you five citations and a structural request. Do exactly what's asked — add a **taxonomy paragraph** in the Introduction placing RLDA/MADA on each axis:

> post-hoc; model-agnostic (given query access to $f$); *semi-global* — instance-level rules from a policy trained across instances, aggregated to class-level decision-region summaries; objective is **model fidelity**, not label purity; explanation target is the decision region, not feature attribution.

That last clause is the sentence that answers R1's Section 6 complaint at the framing level rather than just the metric level.

Add the five reviewer-suggested references (Barredo Arrieta 2020, Ali 2023, Weber 2023, Ortigossa 2024, Nauta 2023) and map your evaluation onto Nauta's properties explicitly — fidelity → #1/#6, compactness → #14, stability → #13, completeness → union coverage + abstention rate.

Beyond those, the gap I'd close is **local→global rule aggregation**, which is your actual contribution and is currently uncited:

- **GLocalX** (Setzu et al., *Artificial Intelligence*, 2021) — merges local rule explanations into a global one. This is the closest prior work to your class-union idea and its absence is a real hole.
- **SP-LIME / SP-Anchors** (Ribeiro et al.) — the original authors' own local→global recipe; also your baseline.
- **Interpretable Decision Sets** (Lakkaraju et al., KDD 2016) and **MUSE / "Faithful and Customizable Explanations of Black Box Models"** (Lakkaraju et al., AIES 2019) — global rule-set approximation with explicit overlap/coverage objectives. Your $P\_{\\text{inter}}$ and $P\_{\\text{same}}$ terms are close cousins of their objectives and you should say so.
- For "continuous threshold optimization," which R1 explicitly asked you to position against: **DR-Net** (Qiao et al., AAAI 2021) and **RRL** (Wang et al., NeurIPS 2021).

On R1's request for a citation that hardening soft rules degrades performance: **I don't know a single canonical citation for that claim.** I'd either soften it to a hypothesis, or — better — run a 20-line experiment: train a soft neural decision tree on one of your datasets, harden the thresholds, report the accuracy drop. That converts a contested assertion into your own evidence and costs a day.

Please verify all venues/DOIs yourself before submitting; I'm working from memory on the bibliographic details.

---

## 6. Draft replacement claims

Current abstract: *"RLDA provides more precise rules and the performance is comparable to classical anchors while producing reusable policies."*

Replace with something like:

> RLDA attains fidelity comparable to classical Anchors while amortizing explanation cost: a single trained policy produces anchors for unseen instances in $X$ ms and $Y$ black-box queries, versus $Z$ ms and $W$ queries for per-instance search. This comes at the cost of substantially lower per-instance coverage, which we quantify and analyze.

Current intro: *"...precision comparable to classical anchors while significantly improving coverage."*

> ...precision comparable to classical Anchors at the instance level, with markedly lower instance coverage; the coverage benefit of our approach appears at the class level, where MADA's coordinated rule sets achieve higher class-union coverage and lower cross-class conflict rate than unioned per-instance Anchors.

Current abstract: *"...operating under defined equilibrium conditions."* → delete. Replace with: *"...with NashConv reported as a training-stability diagnostic."*

Current: *"MADA yields... a more global and structured explanation of the classifier."* → keep only if experiment #6 supports it, and state it as: *"...achieving global rule-set fidelity of X% at a conflict rate of Y%, versus X'/Y' for unioned per-instance Anchors."*

---

## 7. Sequencing

**Weeks 1–2 — audit before you write anything.
**Check the union-coverage impossibilities (§3.1). Check the empty-box precision handling. Check whether the termination coverage is computed on perturbed or real data (§B2). Regenerate all tables programmatically. You may find the results are better than reported, or that some are wrong — either way you need to know before you commit to claims in a response letter.

**Weeks 3–5 — evaluation harness rebuild.
**Fidelity + purity, three-way splits, seeds, Wilson/bootstrap CIs, runtime and query counters, global rule-set metrics. Write this once, cleanly; everything downstream depends on it.

**Weeks 6–9 — baselines and reruns.
**SP-Anchors, greedy union, CART surrogate, CMA-ES/random-search. Rerun everything under the new harness. Larger datasets if feasible.

**Weeks 10–12 — ablations.
**Reward terms, $M$, observation space, P–C frontiers, stability.

**Weeks 13–16 — rewrite.
**Sections 3–6 formalism (that's mostly a week of careful math), taxonomy/related work, results with the honest framing, expanded limitations, response letter.

---

## 8. Where to concede and where to push back

**Concede fully:** fidelity metric, coverage claims, all math errors, equilibrium overclaim, normalization/units, missing seeds/CIs/hyperparameters, missing class-level baseline, held-out class-level evaluation. Don't argue any of these.

**Push back respectfully on one thing:** R1's "too weak for a top-tier venue" on dataset size. The legitimate response is scope reduction, not capitulation — either add two large datasets *or* remove every scalability claim and reposition as a method paper on small-to-medium tabular data. Don't do the thing where you keep the claim and add a hedge; reviewers read that as evasion.

**Expand the limitations section** with everything R1 listed that you omitted: fidelity-vs-labels (now fixed, but note the residual gap), no human study, no OOD perturbation analysis, no categorical treatment, no formal guarantee that learned boxes satisfy the precision constraint (your termination is a heuristic, not a certificate — say so explicitly).

---

One meta-point, since you asked me to be a critical mentor. The reviewers were tough but they were fair, and neither of them attacked the idea. R1 opens by calling it "promising"; R2 calls it "interesting and well motivated." The core contribution — amortized anchor discovery plus coordination-as-a-game — survived. What failed is measurement discipline and mathematical care. That's the more fixable failure mode, but it's also the one that will recur unless you rebuild the evaluation harness properly rather than patching numbers.
