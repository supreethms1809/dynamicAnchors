# External Review (GPT 5.6) — Verification Verdicts

**What this is.** Every finding from the external RLDA/MADA audit, checked against
the working tree at `ma-training-config-bump` on 2026-09-01. Verdicts are based on
reading the cited code and, where possible, measuring the artifacts. Nothing here
was taken on the reviewer's word.

**Headline: all 13 findings are real.** None was a false positive. Three were
already in our own registers; ten are new to us. One (G-01) is more severe than we
had it, because it lands on the actual paper-source tree rather than the probe runs
we had checked.

Cross-references: `docs/BUGS_mada_multiagent_design.md` (M-##),
`docs/BUGS_train_inference_mismatch.md` (B-##), `docs/BUGS_pipeline_review.md` (P-##).

---

## Verdict table

| ID | Finding | Verdict | Ours? |
|---|---|---|---|
| G-01 | Published MADA artifacts predate the grouping fix | **CONFIRMED — worse than stated** | extends M-01 |
| G-02 | Inter-class reward is outside every critic's input | **CONFIRMED** | = M-04 |
| G-03 | Success-rate denominator drops failed episodes | **CONFIRMED** | new |
| G-04 | Break-even accounting is not serving cost | **CONFIRMED** | new |
| G-05 | Paper runner can use stale artifacts; final relabelled as best | **CONFIRMED** | new, compounds P-03 |
| G-06 | RLDA quality filter uses marginal, not class-conditional, coverage | **CONFIRMED** | new |
| G-07 | Class-mode train/inference initialisation differs | **CONFIRMED** | = B-04/B-05/R-5 |
| G-08 | Optional test-label leakage via `eval_on_test_data` | **CONFIRMED — dormant** | new |
| G-09 | ParallelEnv missing-action path violates the contract | **CONFIRMED — dormant** | new |
| G-10 | MADA metadata fallback crashes | **CONFIRMED** | new |
| G-11 | Quantile box fallback parses the wrong state | **CONFIRMED** | new |
| G-12 | Finite-horizon state is not Markov | **CONFIRMED** | new |
| G-13 | Per-class compactness uses a hardcoded 0.95 | **CONFIRMED** | new |

---

## G-01 — Published MADA artifacts predate the grouping fix `S0`

**Confirmed, and broader than the reviewer stated.** We had verified the collapse
only on the `mada_coord_*` probe runs. It holds on **every dataset in the actual
paper-source tree**, `runs/rlda_tuned_seed42/`:

```
iris           groups=['loss_agent_0','loss_agent_1','loss_agent_2']  9 policies  max same-class |Δw| = 0.0
breast_cancer  groups=['loss_agent_0','loss_agent_1']                 6 policies  max same-class |Δw| = 0.0
wine           groups=['loss_agent_0','loss_agent_1','loss_agent_2']  9 policies  max same-class |Δw| = 0.0
housing        groups=[... 4 groups ...]                             12 policies  max same-class |Δw| = 0.0
uci_credit     groups=['loss_agent_0','loss_agent_1']                 6 policies  max same-class |Δw| = 0.0
uci_adult      groups=['loss_agent_0','loss_agent_1']                 6 policies  max same-class |Δw| = 0.0
```

Not "close" — bit-identical, `max |Δw| = 0.0` exactly. And the result JSONs that
`paper/make_tables.py` reads (`revision/results/*mada*.json`, its default
`--results_dir`) are dated 2026-08-28, i.e. generated from those checkpoints.

**Action: regenerate every MADA table from post-fix checkpoints.** Six datasets ×
MADDPG, plus MASAC if it appears in the paper — MASAC carried the same
`share_policy_params` collapse.

---

## G-02 — Inter-class reward outside every critic's input `S1`

**Confirmed; this is our M-04.** `has_state()` returns `False` and `state_spec()`
returns `None` (`benchmarl_wrappers.py:94,111`), so MADDPG takes the
no-global-state branch and the critic input is `observation_spec[group] +
action_spec[group]`. One group is one class, so no other class's box is visible to
any critic, and no agent's observation carries one either.

**Important:** today's `share_param_critic: False` change does **not** fix this. It
gives per-agent Q heads *within* a class; the cross-class term is still
unattributable. G-02 needs a real `state_spec`, or the term must leave the reward
and become a rule-selection filter.

---

## G-03 — Success-rate denominator drops failed episodes `S1`

**Confirmed.** `collect_success_rate` (`utils/metrics.py:644`) sums
`stats["n_episodes"]` from `episode_success_rate(_collect_episode_anchors(cd), …)`,
and `_collect_episode_anchors` (line 561) walks `all_anchors` / `anchors` — the
**persisted** anchors. An episode that ended empty, errored, or was dropped by the
quality filter never appears, so the denominator is "episodes that produced a
box", not "episodes attempted".

Measured on the published iris MADA result:

```
revision/results/iris__mada__seed42__tp0p90__tc0p10.json
  success: n_episodes=45, n_success=34, success_rate=0.756
           (15 per class = 5 anchors x 3 agents)
```

Inference ran 20 instances/class × 3 agents × 3 classes = 180 instance episodes
plus class-based rollouts. The reported denominator is 45. The published 0.756 is
therefore a success rate **conditional on having produced a rule at all** — not the
quantity the paper's τ_P/τ_C success claim describes.

For scale: the fresh post-fix runs report `success=0.167` and `0.233` on the same
denominator convention. Fixing G-03 moves the published number down further.

**Action:** persist one record per attempted episode, flagged, and count all of
them. This also fixes the same denominator in `episode_success_rate`.

---

## G-04 — Break-even accounting is not serving cost `S1`

**Confirmed.** `_fixed_and_marginal` (`paper/make_figures.py:44`) for the RL arms:

```python
per = (n_rep / n_test) if n_test > 0 else 0.0
return n_bb, per          # fixed = extraction queries, marginal = reporting/test
```

Two errors of kind, not degree:

1. **Training queries are absent from the fixed cost.** `n_blackbox_queries` comes
   from the inference rules file. The classifier calls spent *training* the policy
   — `n_perturb_train = 512` per env step, over 144k–360k frames, times the number
   of agents — are the dominant term and are counted nowhere. A break-even curve
   exists precisely to amortise an upfront cost; omitting the largest component of
   that cost is not a calibration issue.
2. **The marginal term is the wrong quantity.** `n_reporting_queries` is set in
   `revision/evaluate.py:212` as `len(y_train) + len(y_val) + len(y_test)` — the
   cost of *measuring* the method. Dividing it by `n_test` and calling it
   per-explanation cost conflates evaluation instrumentation with serving. At
   serving time a learned box costs **zero** black-box queries: the box is fixed.

So the true shape for the RL arms is a very large fixed cost and a ~zero marginal,
while the figure plots a small fixed cost and a non-zero marginal. That inverts the
break-even argument rather than shifting it.

**Action:** three separate counters — training, extraction/construction, and
per-explanation serving — and rebuild `break_even.json` / `.png` from them.

---

## G-05 — Stale artifacts, and `final` relabelled as `best` `S1`

**Confirmed, and it compounds our P-03.** `ensure_best_models`
(`revision/run_rlda_pipeline.py:190`, duplicated at `run_paper_seed_sac.py:217`):

```python
for final in sorted(final_dir.glob("class_*.zip")):
    dest = exp / "best_model" / cls / "best_model.zip"
    if dest.exists(): continue
    shutil.copy2(final, dest)
    log(f"  {cls}: no val-selected best_model.zip; copied {final.name} -> {dest}")
```

Inference then runs with `prefer_model='best'` and loads **final** weights from a
file named `best_model.zip`. It is logged, but the artifact is indistinguishable
downstream and nothing in the result JSON records the substitution.

This is not a rare path. Our **P-03** shows RLDA best-model selection returns
`-inf` and writes nothing whenever eval episodes collapse — which is exactly when
`ensure_best_models` fires. The two failures chain: selection silently fails, then
the unselected checkpoint is silently promoted.

Confirming the provenance gap: `revision/results/*.json` carry `git_commit` and
`config_hash` but **no `rules_file` or `experiment_dir`** — we could not determine
which checkpoint produced a published result from the result file alone.

**Action:** never write `best_model.zip` from `final_model/`; let inference fail
loudly with `prefer_model='best'`. Record the source experiment dir and a
completion manifest in every result JSON.

---

## G-06 — RLDA quality filter uses marginal coverage `S1`

**Confirmed, and the code contradicts itself three lines apart.**
`single_agent/single_agent_inference.py`:

```python
1082  rollout_data["coverage_recomputed"] = float(cov_full)                       # MARGINAL
1083  rollout_data["coverage_class_conditional_recomputed"] = float(cov_class_conditional_full)
...
1087  # Primary coverage stays class-conditional (same as training tau_C).
1089  rollout_data["coverage"] = float(cov_class_conditional_full)
...
1109  anchor_coverage = rollout_data["coverage_recomputed"]                        # MARGINAL again
1111  if anchor_precision >= min_precision_threshold and anchor_coverage >= min_coverage_threshold:
```

The comment at 1087 states the intended convention; the filter at 1109 uses the
other quantity, against a threshold calibrated for the class-conditional one. On
an imbalanced dataset marginal ≪ class-conditional, so good rules are discarded.

Partly masked by the fallback at 1119 (`if len(kept_rollouts) == 0: kept_rollouts =
valid_rollouts`), which turns the filter into a no-op exactly when it is most
wrong — making the behaviour data-dependent rather than simply broken.

**Action:** filter on `coverage_class_conditional_recomputed`. Then re-check
whether the all-filtered fallback is still wanted.

---

## G-07 — Class-mode train/inference initialisation differs `S2`

**Confirmed; already ours as B-04 / B-05 / R-5.** Training class episodes start at
`_class_centroid_quantiles` (the unsnapped per-feature class median); inference
passes diversified k-means centroids and class samples via `class_init_point`. The
policy meets `(mode=class, q*)` pairs at inference it never saw in training.

Note the current state is the *reverse* of the original bug (R-5): inference is now
the diversified side. Still a train/test start-distribution mismatch.

---

## G-08 — Optional test-label leakage `S2` (dormant)

**Confirmed as a code path; not active in the shipped configuration.** Both
`conf/anchor.yaml:229` and `conf/anchor_single.yaml:156` set
`eval_on_test_data: false`, and the pipelines do not override it. The reviewer's
"optional" framing is accurate — this is a latent hazard, not a live one.

**Action:** guard rather than fix — make candidate generation refuse test arrays
outright, so the flag cannot leak selection data if someone flips it.

---

## G-09 — ParallelEnv missing-action path `S2` (dormant)

**Confirmed.** `BenchMARL/environment.py:1857`:

```python
for agent in self.agents:
    if agent not in actions:
        precision, coverage, details = self._current_metrics(agent)   # result discarded
        continue
```

The agent gets no entry in `observations`, `rewards`, `terminations` or
`truncations`, violating the PettingZoo ParallelEnv contract. Dormant because
BenchMARL always supplies every agent's action. The discarded `_current_metrics`
call is also pure waste — it is the expensive call in the loop.

**Action:** raise on an incomplete action dict; delete the dead metrics call.

---

## G-10 — MADA metadata fallback crashes `S2`

**Confirmed exactly as described.** `BenchMARL/inference.py:1510`:

```python
metadata_path = metadata_files.get(group)
if metadata_path is None:
    logger.warning(f"  Warning: No metadata found for {group}, using defaults")
...
    metadata_path=metadata_path or "",
```

`load_policy_model` then does `with open(metadata_path)` (line 574) on `""`. The
warning promises defaults that do not exist; the call raises instead.

---

## G-11 — Quantile box fallback parses the wrong state `S2`

**Confirmed, and the function's own docstring names the hazard.**
`utils/inference_extract.py:persist_box_from_episode`:

> "Quantile obs[:n] is `a`, not `lower`."

Then the fallback:

```python
if obs.shape[0] < 2 * n_features:
    return None
lo, up = obs[:n_features], obs[n_features:2 * n_features]
```

The comment distinguishes the layouts ("Hull-era synthetic final_obs is 2n+2 unit
bounds; quantile policy obs is 3n+3") but the guard does not branch on them: 3n+3
passes `>= 2n`, so a quantile observation is sliced as if it were unit bounds and
**quantile positions a, b are persisted as a box**. Silently wrong rather than
absent. Rare since the B-02 fix populates the preferred keys.

**Action:** return `None` when the observation is the 3n+3 quantile layout.

---

## G-12 — Finite-horizon state is not Markov `S2`

**Confirmed, and the codebase argues our reviewer's case for us.** The hull
observation space carries `episode_phase = t / max_cycles`, and the comment at
`environment.py:2971` explains why:

> "C-06: episode_phase = t / max_cycles is the quantity the paper called ξ_t …
> episode_phase also makes the observation time-aware so the critic has a
> consistent value function under per-step costs."

The quantile layout is `[a, b, q*, precision, coverage, **mode_bit**]` — the clock
channel is replaced by the instance/class mode flag. With `max_cycles: 50`
truncation and per-step costs, the value function is genuinely non-stationary in
`t`, so the critic is fitting a target that depends on an unobserved variable. The
paper's own ξ_t is absent from the representation the paper actually runs.

**Action:** append normalised `t / max_cycles` to the quantile observation (3n+4)
and retrain. Note this invalidates existing quantile checkpoints — batch it with
the G-01 regeneration.

---

## G-13 — Per-class compactness uses a hardcoded 0.95 `S3`

**Confirmed.** `utils/eval_harness.py:576`:

```python
sparsity = float((union.best.extra or {}).get("sparsity_width_ratio") or 0.95)
```

`sparsity_width_ratio` is never *written* into `extra` anywhere (grep: only reads),
so this always falls back to 0.95, while `revision/evaluate.py:234` reads the real
value from artifact metadata and passes it to a second `compactness_of_ruleset`
call. One JSON can therefore contain two compactness numbers computed at different
thresholds whenever metadata is not 0.95.

---

## Consolidated fix order (both reviews)

**Blockers for any published MADA number**
1. **G-01** — regenerate all MADA artifacts post-fix. Everything else is measured
   on top of this.
2. **P-02** — MADA passes no `n_covered`, so `ranking_score_formula: lcb_coverage`
   is inert and checkpoint selection is support-blind. Fix before regenerating, or
   the regenerated run selects the same collapsed-box checkpoints.
3. **G-12** — if the clock is going back into the observation, it must happen
   before the regeneration run, not after.

**Correctness of reported numbers**
4. **G-03** (success denominator), **G-06** (marginal vs class-conditional filter),
   **P-01** (class-union gate), **P-03** (collapse invisible to selection),
   **G-05** (final relabelled as best).

**Honesty of the comparison**
5. **G-04** (query accounting), **P-04** (4x per-policy budget gap at defaults --
   corrected from an earlier 12x; undocumented in the paper), **G-02** (drop the cross-class term or give critics a state).

**Hygiene before submission**
6. **G-07**, **G-08**, **G-09**, **G-10**, **G-11**, **G-13**, **P-05**, **P-06**.

The in-flight `mada_critic_fix_seed42` run addresses none of these; it isolates
`share_param_critic`. Treat it as a diagnostic, not as a paper artifact.
