# RLDA + MADA — Cross-Arm Pipeline Review

**Scope.** Training, inference and evaluation on BOTH arms, read for bugs and
logical issues, with emphasis on anything that makes the RLDA-vs-MADA comparison
not like-for-like. Complements `docs/BUGS_train_inference_mismatch.md` (MADA
train/inference/reporting) and `docs/BUGS_mada_multiagent_design.md` (MADA
multi-agent layer). **ID convention:** `P-##`.

**Status.** Audit only — no code changed by this pass.

**Verified against:** working tree at `ma-training-config-bump`, 2026-09-01,
after the grouping / `share_policy_params` / `share_param_critic` fixes.

---

## What is already right

Worth stating, because these are the places this class of bug usually lives and
they are clean here:

- **Reward parity is enforced by a test, not by hand.**
  `test_shipped_configs_agree_between_rlda_and_mada` derives the shared knob list
  by regexing every `env_config.get("…")` out of both env sources and diffing the
  two YAMLs. A hand-written predecessor passed while `beta` was 5.0 in MADA and
  0.6 in RLDA. Everything excluded is named in `PLUMBING` / `MULTI_AGENT_ONLY` /
  `DEAD`, so adding a divergence is a deliberate act.
- **The knobs that test excludes still agree.** Checked by hand: `n_perturb`
  (2048), `n_perturb_train` (512), `max_cycles` (50), `eval_split` (val),
  `reset_diversity_frac` (0.3), `n_reset_landings` (2), `min_support` (10),
  `coverage_floor_mode` (terminal), `precision_estimator` (empirical),
  `init_mode` (full_space) — identical on both sides.
- **`_potential` matches between the arms**, including the B2 gate ramp,
  `p_tilde = min(P, target)` and the `cov_ok` empty-rule guard.
- **Split hygiene in `revision/evaluate.py` is correct.** Candidates are ranked
  and top-k selected on **val**; `reevaluate_ranked_rules` explicitly carries the
  validation `score` forward (`utils/eval_harness.py:513`) so test labels cannot
  re-rank; the reporting union passes `enforce_min_support=False` so test support
  cannot re-filter the published rule set. This is the part reviewers attack
  first and it holds.
- **The reset-landing mode gate (MADA B-03) has no RLDA counterpart bug** —
  `single_agentENV._maybe_reset_diversity_landing` (line 544) is ungated in both
  modes, matching the fixed MADA version.

---

## P-01 — The class-union potential uses the RETIRED gate `S1`

`_class_union_potential` (`BenchMARL/environment.py:780`):

```python
gate = min(1.0, union_prec / max(cls_target * 0.8, 1e-6))
```

`_potential`, the local shaping term, uses the B2 ramp instead
(`environment.py:2842`, mirrored in `single_agentENV.py:706`):

```python
gate = (precision - (target - margin)) / margin   # margin = gate_margin = 0.10
```

So with `tau_P = 0.9` the two reward streams disagree about when coverage is
worth paying for:

| union / local fidelity | local gate | class-union gate |
|---|---|---|
| 0.50 | 0.00 | 0.69 |
| 0.72 | 0.00 | **1.00** |
| 0.80 | 0.00 | 1.00 |
| 0.85 | 0.50 | 1.00 |
| 0.90 | 1.00 | 1.00 |

The local term's own comment calls the 0.8·target form "a licence to trade
fidelity for coverage" and says it was replaced for exactly that reason. The
replacement never reached the shared term — which is the stream carrying the
multi-agent cooperation signal. A class whose union sits at fidelity 0.72 is paid
full coverage credit for inflating boxes, while each agent's local term pays it
nothing until 0.80.

**Fix.** Factor the gate into one helper and call it from both, or inline the
ramp in `_class_union_potential`. Then re-check `shared_reward_weight`: it was
tuned against the lenient gate.

---

## P-02 — The two arms select `best_model` with different scores `S1`

Same function, different arguments.

**RLDA** (`single_agent/anchor_trainer_sb3.py:297`):
```python
score = ranking_score(mean_p, mean_c, n_covered=mean_n)
```
`formula` defaults to `RANKING_SCORE_LCB_COVERAGE` and `n_covered` **is** given,
so this is the real thing: `wilson_lower_bound(fid, n) * (1 + cov)`.

**MADA** (`BenchMARL/benchmarl_wrappers.py:1327` and `:1335`):
```python
ranking_score(prec, cov, self.ranking_score_formula)     # no n_covered
```
`n_covered` is never passed, so `ranking_score` takes its documented fallback
(`utils/metrics.py:262`) and returns the **point estimate** `fid * (1 + cov)`.

Two consequences:

1. **`ranking_score_formula: lcb_coverage` in `conf/anchor.yaml:239` is inert.**
   The config names the support-aware formula; the code silently delivers
   `precision_coverage`. A config key that reads as active and is not.
2. **MADA checkpoint selection is support-blind while RLDA's is not.** The
   docstring is explicit about what that costs: "fid=1.0 on n=1 scores ~0.21,
   while fid=0.92 on n=25 scores ~0.75". Under the point estimate, fid=1.0 on
   n=1 scores 1.0·(1+cov) and wins outright.

This is not hypothetical. The `mada_coord_probe` iris run selected a checkpoint
reporting `Fid=1.000, success=0.848` whose class-1 union covered n=5 and class-2
union n=2, with `abst=0.433` — a perfect-fidelity, no-support checkpoint, which is
precisely what the LCB exists to reject and what `anchor-box-collapse` recorded
independently.

**Fix.** Thread `n_covered` through the MADA eval aggregation (the box-support
count is already available per rollout in `evaluation_anchor_data.json`) and pass
`self.ranking_score_formula` on the RLDA side so both arms read their config.

---

## P-03 — RLDA best-model selection ignores how often the policy collapses `S1`

`FidCovEvalCallback._evaluate` (`anchor_trainer_sb3.py:263`):

```python
if k < 1:
    continue          # episode dropped entirely
```

Episodes that end with the empty rule never reach `precs` / `covs` / `ns`. The
score is therefore the mean over **surviving** episodes only, with no term for
what fraction survived. A policy that collapses on 9 of 10 evaluation episodes and
produces one excellent box outscores a policy that reliably produces good boxes.

The comment 20 lines below claims the opposite:

> "keeping the zeros in it correctly penalises a checkpoint that often collapses"

That is true only of the *other* zero — a `k >= 1` box with `n_covered = 0`, which
does survive the filter. Collapse to `k = 0` is invisible to selection.

This interacts with D-03 in the other register (the deterministic `-inf` on the
smoke test): both are symptoms of empty-rule episodes being handled as "no data"
rather than "a bad outcome".

**Fix.** Score collapsed episodes as 0 rather than dropping them, or multiply the
score by the non-collapse rate. Either makes `k = 0` cost something.

---

## P-04 — The training budgets differ, and are undocumented `S2`

`base_experiment.yaml:66` states the semantics unambiguously:

> "A frame is one shared-env transition; every agent in the group acts on each
> frame, so max_n_frames is the per-agent timestep budget."

**Correction (2026-09-01).** An earlier version of this entry put the RLDA
budget at 30 000 timesteps/class and the gap at 12x per policy. That was wrong:
`revision/run_rlda_pipeline.py:36` sets `BUDGET_MULT = 3` and applies it at line
51 (`_cfg["sa_timesteps"] *= BUDGET_MULT`); the `90_000` literal in
`DATASET_CONFIGS` is the pre-multiplier value. MADA's `BUDGET_MULT` is 1. The
real figures, at pipeline defaults:

| iris (3 classes, agents_per_class 3) | per-policy timesteps | total env interaction |
|---|---|---|
| RLDA (`90_000 x 3 // 3`) | 90 000 | 270 000 |
| MADA (`ma_frames: 360_000`) | 360 000 | 3 240 000 (x9 agents) |

| breast_cancer (2 classes) | per-policy | total |
|---|---|---|
| RLDA (`90_000 x 3 // 2`) | 135 000 | 270 000 |
| MADA (`ma_frames: 360_000`) | 360 000 | 2 160 000 (x6 agents) |

So the gap is **4x per policy** on iris (2.7x on breast_cancer) and **12x / 8x in
total environment interaction** -- not 12x per policy. Smaller than first
reported, and the direction of the concern is unchanged but its weight is lower.

In the runs launched on 2026-09-01 (`--max_n_frames 144000`) the per-policy
budgets are 144 000 (MADA) vs 90 000 (RLDA) on iris: **1.6x**, close to parity.

`paper/02_training.md` still states no budget for either arm. That is the part
worth fixing regardless of the ratio: a reviewer cannot check a comparison whose
compute budgets are unstated.

## P-05 — Test coverage breaks ties in val-based ranking `S2`

`select_topk_union` (`utils/metrics.py:383`):

```python
scored.sort(key=lambda r: (r.score, r.metrics.coverage ...), reverse=True)
```

On the reporting call, `r.score` is correctly the preserved validation score, but
`r.metrics` is the **test** metrics. So when two rules tie on validation score —
common with near-duplicate boxes, and `reevaluate_ranked_rules` preserves ties
exactly — the rule called `best` is chosen by test coverage.

Small, and it only moves which of two val-equivalent rules is displayed. But it is
a test-data-dependent choice inside a function whose docstring promises the
opposite, so it is cheap to close and awkward to explain if a reviewer finds it.

**Fix.** Break ties on the stored `selection_metrics` coverage (already carried in
`extra`), or on `rule_id` for full determinism.

---

## P-06 — `ranking_score_formula` is read by neither arm's trainer as configured `S2`

Collecting the config-vs-code drift in one place:

- `conf/anchor.yaml:239` → `ranking_score_formula: lcb_coverage`, passed by MADA
  but neutralised by the missing `n_covered` (P-02).
- `conf/anchor_single.yaml:164` → `ranking_score_formula: lcb_coverage`, **never
  read** by `anchor_trainer_sb3.py`; RLDA gets the LCB only because it happens to
  be the function default.
- `revision/evaluate.py:171` defaults to `RANKING_SCORE_LCB_COVERAGE` independently.

Three sites, one intended value, no single source of truth. Today they agree by
coincidence on the RLDA/evaluator side and disagree in effect on the MADA side.
Changing the YAML would silently change one arm and not the other.

---

## Suggested order

1. **P-02** — one line on each side; it changes which checkpoints the paper
   reports and directly explains the collapsed-box selections already on record.
2. **P-01** — one shared gate helper; then re-tune `shared_reward_weight`, which
   was fitted against the lenient gate.
3. **P-03** — makes RLDA checkpoint selection honest about collapse.
4. **P-04** — a paper change, not a code change: state both budgets in
   `02_training.md` (4x per policy on iris at defaults, 1.6x in the reruns),
   and add a matched-budget row if one can be afforded.
5. **P-05**, **P-06** — cheap hygiene; close before submission so neither becomes
   a reviewer question.

**None of P-01…P-06 is invalidated by the grouping fixes**; P-01 and P-02 became
*more* consequential once MADA's agents stopped being identical copies, because
both terms now act on genuinely different per-agent outcomes.
