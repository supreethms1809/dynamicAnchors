# RLDA / MADA — Code & Experiment Plan for Major Revision

**Scope:** everything that requires running, computing, or verifying something.
Writing-only items live in `WRITING_CHANGES.md`.

**ID convention:** `C-##` = code/experiment item. Reviewer tags: `R1-#`, `R2-#`.
`SELF-#` = issue neither reviewer raised but that a second-round reviewer will
likely catch.

**Priority:** `P0` = paper fails without it. `P1` = a reviewer explicitly asked, or
a claim is unsupported without it. `P2` = strengthens, defers if time-bound.

---

## PART 0 — AUDIT AND BUG FIXES (do this before anything else)

Nothing downstream is trustworthy until these are resolved. Several may change
your numbers *in your favour*, so resolving them first prevents you from
committing to claims in the response letter that you'd then have to walk back.

---

### C-01 — Union coverage is smaller than best single-rule coverage `P0` `SELF-1`

**Problem.** Union coverage over a rule set must be **≥** the coverage of any
single rule in that set. Your tables violate this. Both Eq. 21 and Eq. 23 are
class-conditional, so this is not a marginal-vs-conditional artifact.

| Table | Dataset / method / class | best `clsC` | reported `unionC` |
|---|---|---|---|
| 1 | Iris RLDA c1 | 0.740 | **0.440** |
| 1 | Wine RLDA c2 | 0.521 | **0.396** |
| 1 | Breast cancer RLDA c0 | 0.321 | **0.090** |
| 1 | Breast cancer RLDA c1 | 0.678 | **0.081** |
| 1 | Breast cancer MADA c1 | 0.877 | **0.487** |
| 1 | UCI credit MADA c1 | 0.966 | **0.266** |
| 2 | Wine RLDA c2 | 0.708 | **0.396** |

**Most likely cause.** The union is formed over *unique* rules only (consistent
with the UCI-credit c0 case where "RLDA fails to produce any unique rules" →
0.000), and the `best` rule is filtered out of the set before the union is taken.

**Actions**
1. Locate the union construction. Log, for one failing case, the exact rule set
   used for `unionC` and the exact rule used for `clsC`. Confirm whether they are
   drawn from the same pool.
2. Add a hard assertion in the metrics module:
   `assert union_cov >= max(individual_cov) - 1e-9` — fail loudly, do not warn.
3. Add an assertion that the union and best-rule denominators are identical
   (same class-conditional test set).
4. Decide the reporting convention and apply it consistently:
   - **Recommended:** union over the top-`k` ranked rules, with `best` = the
     highest-scoring rule *within that same top-`k` set*. Report `k` in the caption.
   - If uniqueness filtering is retained, the paper must state that `best` may be
     excluded from the union set, and you should report both filtered and
     unfiltered union metrics.
5. Regenerate all three tables.

**Deliverable:** a short note stating whether this was a bug or a reporting
convention, for the response letter (see `WRITING_CHANGES.md` W-31).

---

### C-02 — Empty-box precision handling `P0` `SELF-9`

**Problem.** Breast-cancer instance coverage of 0.002 on a ~170-sample test split
is **fewer than one sample**. Precision of 1.000 on an empty or singleton
intersection is not a measurement. Several reported 1.000 values are likely
computed on `|B ∩ D_test| ∈ {0, 1}`.

**Actions**
1. Inspect the precision function's behaviour when `|B ∩ D| == 0`. If it returns
   1.0 (or 0.0) instead of `NaN`, that is a bug.
2. Return `NaN` for empty intersections. Propagate `NaN` through aggregation
   (`np.nanmean`) rather than silently coercing.
3. Log `n_covered` (raw sample count) alongside every precision value, and
   **report it in the tables**.
4. Add a `min_support` threshold (suggest `n ≥ 10`); rules below it are reported
   but flagged as statistically uninformative.
5. Compute a **Wilson score interval** on every precision estimate. This is what
   makes "precision 1.000 on one sample" visibly meaningless — better that you
   surface it than a reviewer.

---

### C-03 — Diagnose the instance-coverage collapse `P0` `R1-5` `R2-3` `SELF-10`

**Problem.** Your termination conditions (Sec. 5.2.2) require `C ≥ 0.5·τ_C`,
i.e. ≥ 0.10 at `τ_C = 0.20`. Yet you report instance coverage of 0.002–0.008.
Those two facts are inconsistent. Something is measuring coverage differently at
train time and eval time.

**Working hypothesis (verify, do not assume).** Training coverage is estimated on
the *locally perturbed* sample set around the instance; evaluation coverage is
computed on the *real* test set. The perturbed set is concentrated near the box,
so training coverage is inflated by orders of magnitude, the termination check
passes, and the accepted box covers almost nothing real.

**Actions**
1. Instrument the environment: log, per step, `cov_perturbed` and `cov_real` side
   by side for the same box. Plot their ratio over an episode.
2. If the hypothesis holds, change the **termination and reward coverage** to be
   computed on real data (a real-data mini-batch is fine; it need not be the full
   set every step). Keep perturbation sampling for the *precision/fidelity* proxy
   if you want, but say so.
3. Re-run one dataset and check whether instance coverage recovers.
4. Record the outcome either way — if the hypothesis is wrong, you still need an
   explanation for the reviewer.

**Note:** this is the single change most likely to materially improve your
headline numbers. Do it before you finalize any claims.

---

### C-04 — Eq. 11 dynamic precision weight is a no-op `P0` `SELF-2`

**Problem.** `w_P = max(2.0, 1 + (P_t − τ_P)/(1 − τ_P))` for `P_t ≥ τ_P`.
For `P_t ∈ [τ_P, 1]` the second argument lies in `[1, 2]`, so the `max` is always
`2.0` — identical to the else-branch. The "dynamic" weight is the constant 2.

**Compounding problem.** Sec. 5.2.1 states the MADA local reward is "the same as
Eq. 7", which uses fixed `α = 0.7, β = 0.6`. So it is unclear whether `w_P`/`w_C`
are used at all.

**Actions**
1. Read the code and determine which of `{α, β}` and `{w_P, w_C}` is actually
   applied, in RLDA and in MADA separately.
2. If `w_P` was intended to *increase* past `τ_P`, the formula is probably meant
   to be `min(2.0, ...)` or `max(1.0, ...)`. Fix and state the intent.
3. If it was never active, either delete it from the paper or activate it and
   re-run.
4. Add a unit test asserting `w_P` is non-constant over `P_t ∈ [0, 1]`.

---

### C-05 — Regenerate all tables programmatically `P0` `SELF-3`

**Problem.** Cross-table inconsistencies indicate hand-transcription:

- Breast-cancer **Anchors** baseline appears as `(0.990, 0.168)`, `(0.990, 0.160)`,
  `(0.996, 0.168)` in Tables 1/2/3. The Anchors baseline does **not** depend on
  `τ_C`, so these should be identical (or you resampled instances, which then
  requires seeds and CIs).
- UCI-credit RLDA `clsP/clsC = 0.459/0.606` and `0.762/0.775` are **byte-identical
  across all three tables**. Several MADA `instC = 0.002` entries likewise.

A reviewer reading this on round two will read it as fabrication risk, whatever
the true cause.

**Actions**
1. Write a `make_tables.py` that reads result JSON/CSV artifacts and emits LaTeX
   directly. No manual editing of tables, ever again.
2. Store one result file per `(dataset, method, seed, τ_P, τ_C)` with a schema
   including: metric values, `n_covered`, CI bounds, rule bounds, wall-clock,
   query counts, git commit hash, config hash.
3. Add a check that the Anchors baseline row is byte-identical across ablation
   tables (it must be, by construction), or that a differing seed is documented.

---

### C-06 — Undefined `ξ_t` in the MADA transition `P0` `SELF-13`

**Problem.** Sec. 5.2 "Model transition" says the next state is obtained "by
replacing `{B_k,t}` with `{B_k,t+1}` and updating `ξ_t`". `ξ_t` is never defined
anywhere in the paper.

**Why it matters beyond notation.** If `ξ_t` is a *shared* variable updated from
joint actions, it couples the agents through the dynamics — which breaks the
agent-wise decoupling needed for the Markov-potential-game argument (C-16).

**Actions**
1. Identify what `ξ_t` is in the code (step counter? termination counters?
   coverage-floor violation tally?).
2. Classify it: per-agent, or shared-and-joint-action-dependent.
3. If shared and joint-action-dependent, either move it into the state and accept
   coupling, or restructure it as a per-agent quantity.
4. Same analysis for the **shared termination counters** (Sec. 5.2.2), which are
   global and history-dependent and *do* couple agents.

---

### C-07 — Categorical features in UCI Credit `P0` `SELF-5`

**Problem.** UCI Credit Approval (crx) has 6 continuous + 9 categorical
attributes. Your method builds axis-aligned boxes in continuous space. With
one-hot encoding, an interval bound produces rules like
`0.3 ≤ is_category_A ≤ 0.8`, which is neither human-readable nor a valid logical
condition. The paper never mentions this. Classical Anchors handles categoricals
natively, so this is *also* a baseline-fairness problem.

**Actions**
1. Determine what the code currently does (one-hot? ordinal encode? drop?).
2. Implement and document one defensible option:
   - **(a)** Restrict box refinement to continuous dimensions; fix categoricals to
     the target instance's value (instance mode) or the class mode (class mode).
   - **(b)** Treat one-hot dimensions as a group with a binary
     include/exclude action, emitting `feature ∈ {A, C}` style conditions.
   - **(c)** Threshold one-hot interval bounds at 0.5 post-hoc and emit equality
     conditions; must be validated to not change the covered set.
3. Report the chosen handling for both your method and the Anchors baseline.

---

### C-08 — Document the rule sparsity mechanism `P1` `SELF-6`

**Problem.** Reported rules use 1–2 features out of 4/13/30. There is no sparsity
term in Eq. 7. Presumably a dimension is dropped from the *printed* rule when its
interval spans nearly the full range — but the mechanism and its threshold are
undocumented, and the threshold materially changes the reported rule.

**Actions**
1. Locate the rule-printing code; extract the exact drop criterion and threshold.
2. Verify the printed rule and the evaluated box define the **same covered set**;
   if dropping a dimension changes coverage, the printed rule is not the rule you
   measured. Add an assertion.
3. Report the threshold as a hyperparameter.
4. **Recommended addition:** an explicit complexity penalty in `Φ` on the number
   of active dimensions. Gives a principled answer to your own "rule overload"
   limitation and to Nauta et al.'s *compactness* property.

---

## PART 1 — EVALUATION HARNESS REBUILD

Write this **once, cleanly**. Every experiment below depends on it.

---

### C-09 — Separate fidelity from label purity `P0` `R1-3` `R1-17` `R2-1`

The single most important change in the revision. Both reviewers raised it
independently.

**Problem.** Eq. 2 (background) correctly defines anchor precision as agreement
with the **model**. Eq. 19 (evaluation) defines it as `P(y = c | x ∈ B)` —
agreement with **labels**. The paper contradicts itself between Sec. 3 and Sec. 6.
With MLPs at 80–90% accuracy, up to a fifth of your precision signal on UCI Credit
is measuring classifier error, not explanation quality.

**Actions**
1. Implement both metrics explicitly:
   ```
   Fid(B, c) = P( f_hat(x) = c | x ∈ B )     # model fidelity — PRIMARY
   Pur(B, c) = P( y        = c | x ∈ B )     # label purity   — SECONDARY
   ```
2. Same split for the union versions: `Fid_∪(c)`, `Pur_∪(c)`.
3. Make **fidelity** the optimization target, the termination criterion, and the
   headline reported metric everywhere. Purity is reported as a secondary column.
4. Log per-dataset classifier train/test accuracy so readers can calibrate the
   gap between the two.
5. **Verify before promising:** the paper says training already uses a
   "probability-weighted proxy precision based on model prediction stability".
   If true, the policies already optimize approximately the right thing and you
   need **re-evaluation only, not retraining**. Confirm in code. If the training
   proxy actually uses `y`, you must retrain.
6. Define the training proxy mathematically (currently one English sentence) so
   the train/eval relationship is explicit — R1 flagged this gap directly.

---

### C-10 — Three-way split; class-level metrics on held-out data `P0` `R2-2`

**Problem.** Instance metrics use a test set; class-level metrics use the **full
dataset**. Since rule selection/ranking also uses that data, selection and
evaluation share samples. That is leakage, and it inflates every MADA class-level
claim — which is where MADA's whole case rests.

**Actions**
1. Implement a strict three-way split:

   | Split | Used for |
   |---|---|
   | `D_train` | black-box training; RL/MARL policy training |
   | `D_val` | rule generation, ranking, top-`k` selection, checkpoint selection |
   | `D_test` | **all reported metrics**, instance-level and class-level |

2. Suggested ratio 60/20/20, stratified. For Iris/Wine (150/178 samples) this
   leaves ~30 test samples — report the raw counts and use nested CV
   (5×5) instead if the test split is too small to support any claim.
3. Assert no index overlap between splits at runtime.
4. Recompute all class-level and class-union metrics on `D_test`.
5. Record and report `k` (number of rules in each union) — R2 asked for this
   explicitly.

---

### C-11 — Seeds, confidence intervals, statistical testing `P0` `R1-6` `R2-6`

**Actions**
1. Run **≥ 5 seeds** (10 preferred) for every configuration. Seed the black-box
   init, the data split, the RL policy, the replay buffer, and the perturbation
   sampler — separately and reproducibly.
2. Report every number as **mean ± 95% CI** (bootstrap over seeds).
3. **Paired Wilcoxon signed-rank test** vs. the Anchors baseline over matched
   instances, per dataset and metric. Report the p-value and effect size.
4. **Wilson intervals** on all precision/fidelity values (see C-02).
5. Report **success rate** — fraction of episodes reaching `τ_P` *and* `τ_C`
   before `T_max`. R1 asked for this and it is currently absent. It is also the
   honest way to describe the "not every episode reaches the threshold" sentence
   in Sec. 7.1.
6. Report **variability across runs** for the same instance (see C-24).

---

### C-12 — Runtime and query-budget instrumentation `P0` `R1-6` `SELF-4`

**Problem.** Amortization is the paper's central selling point and it has **no
supporting experiment**. R1 noticed the missing runtime; nobody noticed the
missing amortization test.

**Actions**
1. Instrument a counter for **black-box queries per explanation** — this is the
   hardware-independent number and it is the fairer comparison.
2. Instrument wall-clock for: one-time policy training; per-instance inference
   (RLDA/MADA); per-instance search (Anchors).
3. Report hardware, and whether training used the vectorized envs mentioned in
   Sec. 4.4.
4. Produce a **break-even plot**: cumulative cost vs. number of explained
   instances, for RLDA/MADA (flat training cost + cheap per-instance) against
   Anchors (zero training + expensive per-instance). Mark the crossover point.
   This one figure defends the core claim.

---

### C-13 — Fix normalization and report rules in original units `P0` `R1-7` `R1-19` `R2-7`

**Problem.** The paper says data are normalized to `[0,1]^d` (Eqs. 4, 8) and that
Iris rules are "denormalized to original units", yet reports
`petal length ≤ −1.23`. Fig. 9 axes literally read "z-score", so the pipeline is
`StandardScaler`, not min-max. A rule stating a negative petal length in
centimetres fails the paper's own interpretability premise.

**Actions**
1. Decide and implement **one** pipeline:
   - **(a)** Switch to genuine min-max scaling, keeping all `[0,1]` math intact.
     Note that min-max is sensitive to outliers on Breast Cancer / UCI Credit.
   - **(b)** Keep `StandardScaler` and change the math in Secs. 4–5 to
     standardized space (bounds in `ℝ^d`, box constraints stated accordingly).
     **Recommended** — fewer code changes, and z-scoring is the better choice for
     these datasets.
2. Implement an inverse-transform in the rule printer so **every reported rule is
   in original units** (cm, chemical units, currency).
3. Add an assertion that all printed continuous bounds are inside the observed
   feature range.
4. Regenerate every rule box in Figs. 2–8 and Fig. 9 axis labels.

---

### C-14 — Global rule-set fidelity evaluator `P0` `SELF` (extends `R2-4`)

Neither reviewer asked for this, but it is the experiment that would actually
*earn* the claim "a more global and structured explanation", and it directly
answers R2's request for "a complete measure of cross-class ambiguity".

**Actions**

Build a rule-set-as-classifier evaluator:

```
predict(x):
    fired = { k : x ∈ ∪ A_k }
    if fired is empty          -> ABSTAIN
    if |fired| == 1            -> that class
    else                       -> tie-break by rule fidelity; record CONFLICT
```

Report on `D_test`:

| Metric | Definition |
|---|---|
| **Global fidelity** | agreement with black box on non-abstained samples |
| **Abstention rate** | fraction of `D_test` covered by no rule |
| **Conflict rate** | fraction covered by ≥ 2 *different-class* rule sets |
| **Coverage** | 1 − abstention rate |

Compare RLDA / MADA / SP-Anchors / CART surrogate. Conflict rate converts your
overlap heatmaps (Figs. 11, 12) from anecdote into a number.

---

## PART 2 — METHOD CHANGES (recommended, not required)

These are the potential-based reformulations discussed separately. They are
optional in the sense that the paper can be repaired without them — but they
close R1-4, R1-11, R1-14 and the equilibrium overclaim at the *design* level
rather than patching each symptom.

**Cost:** ~2 weeks of derivation + a full rerun. Since C-09 already forces a
rerun, the marginal cost is mostly the derivation.

**Sequencing rule:** verify C-15's symmetry condition and C-16's decoupling
condition **in code first**. If they fail, you have a potential-game-*flavoured*
heuristic, not a potential game, and you must say that instead. Finding out after
drafting a proposition is expensive.

---

### C-15 — Potential-based reward shaping for RLDA `P1`

**Observation.** Your reward is already nearly potential-based. The gain terms
`α ΔP_t + β ΔC_t` are exactly `γΦ(s') − Φ(s)` at `γ = 1` with
`Φ = αP + βC`. You found PBRS empirically and then bolted seven heuristic terms
onto it.

**Target form** (Ng, Harada & Russell 1999):

```
R_true(s_T) = 1[ Fid(B_T) ≥ τ_P ] · Cov(B_T)

Φ(s) = w_P·P(s) + w_C·C(s) + w_π·R_purity(s)
       − w_a·||center(B) − x*||²  − w_g·G(s)  − w_m·M(s)

r_t = R_true + γ·Φ(s_{t+1}) − Φ(s_t)
```

**Implementation actions**
1. Refactor the reward into exactly two functions: `terminal_reward(s_T)` and
   `potential(s)`. Delete the per-term ad-hoc assembly.
2. **Delete the `η` progress schedule.** Its purpose was to stop shaping terms
   from dominating the true objective; policy invariance makes that impossible,
   so the mechanism is unnecessary. This also removes a history-dependent term
   from the reward (closes part of R1-11).
3. `x*` is constant within an episode → include it in the state, making every
   `Φ` argument a state function by construction.
4. **Caveat to handle:** trajectory drift `D_t = ||B_t − B_{t−1}||²` **cannot** be
   folded into `Φ(s)` without augmenting the state with `B_{t−1}`. Either augment,
   or drop it (note that `γΦ(s') − Φ(s)` already penalizes large potential swings
   implicitly). Test both.
5. If you want the schedule back, use **dynamic PBRS** (Devlin & Kudenko, AAMAS
   2012), which permits `Φ(s, t)` while preserving invariance.
6. Keep the original reward as an ablation arm: `RLDA-v1 (heuristic, as reviewed)`
   vs `RLDA-v2 (potential-based)`. Free ablation answering R1's "unstable reward
   design", and it preserves continuity for the reviewer.

---

### C-16 — Markov potential game structure for MADA `P1`

**Verify these two conditions in code before claiming anything.**

**Condition A — symmetry of the coupling term.** A game with utilities
`u_i = V_i(a_i) + Σ_{j≠i} w_ij·g(a_i,a_j)` where `g` is symmetric and
`w_ij = w_ji` is an *exact potential game* with
`Φ = Σ_i V_i + Σ_{i<j} w_ij·g(a_i,a_j)`.

Your overlap measure (Eq. 24, min-IoU across dimensions) **is** symmetric. So the
whole thing reduces to one concrete code requirement:

> `P_inter` and `P_same` must be charged **symmetrically, with equal weight, to
> both agents** in each overlapping pair.

- **Check:** does the code penalize only agent `i` for overlapping with `j`? Does
  it use asymmetric weights? Does it penalize only the "loser" of a comparison?
  Any of these breaks the potential structure.
- **Fix:** enforce `w_ij == w_ji` and charge both agents. One-line condition,
  unlocks a theorem.

**Condition B — agent-wise decoupled dynamics.** Stage-game potential structure
does **not** automatically lift to a Markov potential game; transitions can couple
agents. In your environment `B_k` evolves under `a_k` alone, and `P_k, C_k` depend
on `B_k` and the data alone — so coupling is reward-only, which is the sufficient
condition. **But** verify:
- `ξ_t` (see C-06) — if shared and joint-action-dependent, decoupling breaks.
- Shared termination counters (Sec. 5.2.2) — these are global and
  history-dependent and **do** couple agents. Either move into state, or argue
  they are training curriculum rather than part of the game.

**Actions**
1. Audit and enforce Condition A. Add a unit test on symmetry of the penalty
   matrix.
2. Audit Condition B; document the resolution for `ξ_t` and the counters.
3. Implement `Φ` explicitly as a logged diagnostic and verify empirically that
   `Δu_c` matches `ΔΦ` under unilateral deviations (numerical potential check on
   a small instance). This is a strong, cheap sanity test of the whole claim.
4. **Algorithm caveat:** MPG convergence results (Leonardos et al., ICLR 2022;
   Ding et al., ICML 2022) assume specific parameterizations and exact/unbiased
   gradients. They **do not cover MADDPG** (deterministic policies, replay buffer,
   learned centralized critic). Either state the gap, or add a **MAPPO /
   independent policy gradient** arm where the theory bites. Recommended: run
   both and report — "does the theoretically-covered algorithm behave better?" is
   itself a result.

---

### C-17 — NashConv estimation details and validation `P1` `R1-4` `R2-5`

**Problem.** R2 asked for details on how best responses and exploitability are
estimated. Currently `Δ_c(s) = max_{a_c} Q_c(...) − Q_c(...)` is approximated "via
gradient ascent" with no specification.

**Actions**
1. Log and report: number of gradient-ascent steps, step size, initialization,
   whether ascent is projected onto the valid action box, number of states
   averaged over, and which critic checkpoint is used.
2. **Validation experiment** (`P2`): on one small case (Iris, `M = 1`), compare
   the critic-gradient best response against a **brute-force fine-grid** best
   response. Report the approximation gap. This bounds the bias.
3. Note in code and paper that `Q_c` is a *learned* approximation, so `Δ_c` is a
   biased estimate and generally a **lower bound** on true exploitability.
4. Reframe output as a training-stability diagnostic (writing item W-24).

---

### C-18 — MADA observation space ablation `P1` `SELF-7`

**Problem.** Agent `k` observes only `(ℓ_k, u_k, P_k, C_k)` — its own box. But
`P_inter` and `P_same` depend entirely on *other* agents' boxes. From the local
view, the overlap penalty is an unexplained non-stationary reward shift. CTDE's
centralized critic sees the joint state so it is learnable in principle, but this
is a plausible cause of the higher residual exploitability on Iris/Wine.

**Action.** Ablate: augment `o_k,t` with summary statistics of other agents'
boxes (e.g. per-dimension mean/min/max of other boxes, or the current pairwise
overlap vector). Compare convergence, conflict rate, union coverage. Cheap; my
expectation is that it helps noticeably.

---

## PART 3 — BASELINES TO IMPLEMENT

R2's central methodological request: MADA is proposed as a class-level/global
method but is only compared against an *instance-level* baseline. Without a
class-level baseline the global claim is untestable.

---

### C-19 — SP-Anchors (submodular pick) `P0` `R2-4`

The most glaring gap: this is Ribeiro et al.'s **own** local→global recipe, from
the same line of work, and it is absent.

**Actions**
1. Generate per-instance Anchors for a budget of `B` instances per class.
2. Apply submodular pick (as in SP-LIME) to select a diverse, high-coverage
   subset of size `k`.
3. Union the selected rules per class.
4. Evaluate with the **identical** harness (C-14): union fidelity, union coverage,
   overlap, conflict rate, abstention.
5. **Match `k`** to MADA's rule count so the comparison is fair.

---

### C-20 — Greedy set-cover union of instance Anchors `P1` `R2-4`

Simpler and stronger-than-it-sounds baseline: greedily add per-instance anchors
that maximize marginal class coverage subject to a fidelity floor `≥ τ_P`. Stop
at `k` rules. Same evaluation harness, same `k`.

---

### C-21 — Depth-limited CART surrogate on model predictions `P1`

Train a decision tree on `(x, f_hat(x))` — **not** on `y` — with depth limited so
the leaf count matches your rule count. Extract per-class leaf-path rules; union
per class. This is the classic global-surrogate baseline and reviewers expect it.

---

### C-22 — Non-RL box optimization baseline `P1` `SELF-12`

**Why this matters:** a reviewer will ask "why RL?" and right now the paper has no
answer. If simple black-box optimization matches RLDA per-instance, RL's value is
*only* amortization — which is a perfectly fine claim, but you need to know it and
say it.

**Actions**
1. Implement random search and/or **CMA-ES** directly over `(ℓ, u)` per instance,
   maximizing the same objective, under a **matched query budget** to RLDA.
2. Report fidelity, coverage, and query count.
3. If it matches RLDA per-instance, reposition the RLDA contribution around
   amortization (C-12) rather than solution quality. That is a defensible and
   honest framing.

---

## PART 4 — DATASETS

### C-23 — Add larger datasets, or delete the scalability claim `P1` `R1-6` `R1-18`

**Problem.** The paper describes 150–600 samples as spanning "small to large
sample regimes" and claims scalability. 600 samples is not large; R1 says so
plainly.

**Current suite (keep):**

| Dataset | n | d | K | Role |
|---|---|---|---|---|
| Iris | 150 | 4 | 3 | visualizable, low-dim |
| Wine | 178 | 13 | 3 | multi-class, mid-dim |
| Breast Cancer | 569 | 30 | 2 | high-dim, correlated |
| UCI Credit (crx) | ~690 | 15 | 2 | mixed types, imbalanced |

**Add ≥ 2 with > 10k rows:**

| Candidate | n | d | Notes |
|---|---|---|---|
| **Adult / Census Income** | ~48k | 14 | mixed categorical; XAI standard |
| **Bank Marketing** | ~45k | 16 | imbalanced; mixed types |
| **HELOC / FICO** | ~10k | 23 | all continuous; credit domain, fits your narrative |
| **Covertype** (subsample) | 50k+ | 54 | high-dim, 7 classes; stress-tests `K` |

Recommended minimum: **HELOC** (all-continuous, so no categorical confound, and
domain-relevant) + **Adult** (categorical stress test, forces C-07 to be solved
properly).

**If you cannot run these:** delete every scalability claim from the paper. This
is a legitimate and much cheaper option and I'd rather you scope honestly than
overclaim on Iris. Do **not** keep the claim and add a hedge — reviewers read
that as evasion.

---

## PART 5 — EXPERIMENTS

### P0 — paper fails without these

| ID | Experiment | Closes |
|---|---|---|
| **C-24** | **Fidelity re-evaluation.** All datasets × classes × methods, reporting `Fid` and `Pur` side by side + per-dataset classifier train/test accuracy + `n_covered`. Re-evaluation only if C-09 step 5 confirms the training proxy is model-based. | R1-3, R1-17, R2-1 |
| **C-25** | **Held-out class-level evaluation.** All class-level and class-union metrics recomputed on `D_test` under the three-way split. | R2-2 |
| **C-26** | **Seeds + CIs + significance.** ≥5 seeds; mean ± 95% CI on every cell; paired Wilcoxon vs. Anchors; Wilson intervals on precision; success rates. | R1-6, R2-6 |
| **C-27** | **Runtime & query budget + break-even plot.** | R1-6, SELF-4 |
| **C-28** | **Class-level Anchors baselines.** SP-Anchors (C-19), greedy set-cover (C-20), CART surrogate (C-21), all `k`-matched. | R2-4 |
| **C-29** | **Global rule-set fidelity.** Global fidelity / abstention / conflict rate across all methods (C-14). | R2-4, R1-8 |
| **C-30** | **Units/normalization regeneration.** All rules and figures reissued in original units. | R1-7, R1-19, R2-7 |

### P1 — explicitly requested, or a claim is unsupported without it

| ID | Experiment | Closes |
|---|---|---|
| **C-31** | **Precision–coverage frontier curves.** Sweep `τ_P × τ_C`; plot the frontier per dataset for all methods; compare **fidelity at matched coverage**. Since raw coverage differs by ~50×, single-point comparison is meaningless — this is the fair comparison, and it may well show RLDA/MADA competitive where the current tables show a rout. | R1-5, R2-3 |
| **C-32** | **Stability under re-explanation.** Re-explain the same instances across seeds and perturbation draws; report Jaccard/IoU of resulting boxes and variance of reported bounds. RL is exactly where reviewers expect instability. Maps to Nauta et al.'s *stability*. | R2-6 |
| **C-33** | **Larger-dataset runs** (C-23), or scope reduction. | R1-6, R1-18 |
| **C-34** | **Non-RL optimizer comparison** (C-22). | SELF-12 |
| **C-35** | **Soft→hard decision-tree hardening experiment.** R1 asked for a citation that hardening soft neural decision trees degrades performance. I do not know a canonical citation for this. Cheaper and stronger: train a soft NDT on one of your datasets, harden the thresholds, report the accuracy drop. Converts a contested assertion into your own evidence; ~1 day. | R1-1 |

### P2 — strengthens; defer if time-bound

| ID | Experiment | Closes |
|---|---|---|
| **C-36** | **Compactness metrics:** active features per rule, rules per class, total description length. | R1-2 (Nauta) |
| **C-37** | **Perturbation-distribution sensitivity.** Your bootstrap-vs-uniform choice changes the meaning of precision and makes it non-comparable to Anchors' `D_x(·\|A)`. Report results under both. | R1-9 |
| **C-38** | **NashConv validation** vs. brute-force best response (C-17 step 2). | R2-5 |
| **C-39** | **Categorical-aware MADA variant** for crx/Adult. | SELF-5 |
| **C-40** | **OOD perturbation analysis.** Do perturbed samples used for precision estimation lie on-manifold? Report the fraction outside the data hull. | R1-20 |

---

## PART 6 — ABLATIONS

R1 explicitly asked for reward-term ablation. Report Δ on **fidelity**, **union
coverage**, **conflict rate**, and **NashConv** for each.

| ID | Ablation | Arms |
|---|---|---|
| **C-41** | **Reward terms** (one-at-a-time removal) | `− P_inter`, `− P_same`, `− R_shared`, `− R_purity`, `− R_class`, `− R_cov`, `− A_t` (anchor drift), `− D_t` (trajectory drift) |
| **C-42** | **`η` schedule** | static `η = 1.0` / static `0.5` / dynamic (current) / removed under PBRS (C-15) |
| **C-43** | **Reward formulation** | `RLDA-v1` heuristic (as reviewed) vs `RLDA-v2` potential-based |
| **C-44** | **Agents per class `M`** | `M ∈ {1, 2, 3, 5}` — you assert `M = 3` with no justification, and the claim that more agents yields more non-overlapping rules is currently unsupported |
| **C-45** | **Initialization strategy** | instance-wise / centroid / hybrid at ratios `{0.25, 0.5, 0.75}` — the paper describes a hybrid but never reports the ratio used |
| **C-46** | **Observation space** | own-box only vs. augmented with other-agent summaries (C-18) |
| **C-47** | **MARL algorithm** | MADDPG vs MAPPO vs independent PG (ties to C-16 step 4) |
| **C-48** | **Threshold grid** | `τ_P ∈ {0.90, 0.95, 0.99} × τ_C ∈ {0.1, 0.2, 0.3}` — extends your existing 3-table study into a full grid feeding C-31 |
| **C-49** | **Perturbation strategy** | bootstrap vs uniform vs none (real-data-only) — also settles C-03 |
| **C-50** | **Termination criteria** | full hierarchy vs. ideal-only vs. no counter-disabling |

---

## PART 7 — REPRODUCIBILITY ARTIFACTS

R2 asked for these by name; R1 listed the absence under "experimental design too
weak".

### C-51 — Config and hyperparameter dump `P0` `R2-6`

Emit and include as an appendix / repo file:

- Train/val/test split sizes, stratification, seeds
- Classifier: architecture (layer widths, activations), optimizer, LR, epochs,
  batch size, **train and test accuracy per dataset**
- RL: algorithm, actor/critic architectures, LR, `γ`, `τ`, buffer size, batch
  size, exploration noise, `T_max`, episodes, total frames, vectorized env count
- MARL: same + `M`, CTDE critic input spec, BenchMARL config
- Reward: **every** weight (`α, β, w_P, w_C, w_shared, w_union_cov,
  w_union_prec, λ_d, λ_a, λ_g, λ_min, η` schedule)
- Environment: `ε_min`, `w_min`, coverage floor, perturbation strategy and
  sample counts, termination counter limits
- Rule reporting: sparsity drop threshold (C-08), top-`k` for unions, ranking
  score definition
- Hardware, library versions, git commit hash

### C-52 — Rule-ranking score must be defined `P0`

Sec. 7.2 says rules are scored on "precision and coverage balance" and ranked.
The formula is never given, yet it determines every `best` and every union set.
Extract it, define it, report it, and make it a configurable hyperparameter.

### C-53 — Release artifacts `P1`

Repo should contain: exact configs per table, result JSONs, `make_tables.py`,
`make_figures.py`, seed lists, and a one-command reproduction script. The paper's
data-availability statement should point at a **versioned tag/DOI**, not just
"Dynamic Anchors" on GitHub.

---

## TRACEABILITY — reviewer point → code item

| Reviewer point | Code items |
|---|---|
| R1-1 references / hardening claim | C-35 |
| R1-3, R1-17, R2-1 fidelity vs purity | C-09, C-24 |
| R1-4 math errors / equilibrium | C-16, C-17 (rest in `WRITING_CHANGES.md`) |
| R1-5 coverage claim unsupported | C-03, C-31 |
| R1-6 weak experimental design | C-11, C-12, C-23, C-26, C-27, C-28, C-41–C-50, C-51 |
| R1-7, R1-19, R2-7 normalization/units | C-13, C-30 |
| R1-8 overclaims | C-14, C-29, C-31 |
| R1-11 Markov violation | C-15 (+ writing W-14) |
| R1-14 reward not reproducible | C-15, C-51 |
| R1-18 dataset size | C-23, C-33 |
| R1-20 missing limitations | C-32, C-37, C-40 |
| R2-2 held-out class-level eval | C-10, C-25 |
| R2-3 soften conclusions | C-03, C-31 (evidence for revised claims) |
| R2-4 class-level baseline | C-19, C-20, C-21, C-28, C-29 |
| R2-5 NashConv caution | C-17, C-38 |
| R2-6 reproducibility | C-11, C-26, C-32, C-51, C-52, C-53 |
| SELF-1 union coverage impossible | C-01 |
| SELF-2 Eq. 11 no-op | C-04 |
| SELF-3 duplicated rows | C-05 |
| SELF-4 amortization untested | C-12, C-27 |
| SELF-5 categoricals | C-07, C-39 |
| SELF-6 sparsity mechanism | C-08 |
| SELF-7 MADA observations | C-18, C-46 |
| SELF-9 empty-box precision | C-02 |
| SELF-10 coverage collapse | C-03, C-49 |
| SELF-12 why RL | C-22, C-34 |
| SELF-13 undefined `ξ_t` | C-06 |

---

## SUGGESTED SEQUENCING

| Weeks | Work |
|---|---|
| **1–2** | Part 0 audit: C-01 – C-08. Do not write anything yet. Resolve the union-coverage impossibility and the coverage-collapse diagnosis first — they may change what you can claim. |
| **3–5** | Part 1 harness: C-09 – C-14, C-51, C-52. Build once, cleanly. |
| **6–7** | Part 2 method changes if adopting: C-15 – C-18. Verify conditions in code **before** drafting theory. |
| **8–10** | Part 3 baselines + Part 4 datasets: C-19 – C-23. Rerun everything under the new harness. |
| **11–13** | Part 5 P0/P1 experiments: C-24 – C-35. |
| **14–15** | Part 6 ablations: C-41 – C-50. |
| **16** | Artifacts, table/figure regeneration, C-53. |

Request a deadline extension now. What R1 asked for does not fit in 60 days.
