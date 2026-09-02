# MADA — Multi-Agent Design Defect Register

**Scope.** The *multi-agent* layer of MADA only: agent construction → torchrl
grouping → BenchMARL actor/critic wiring → the coordination reward terms added in
the working tree (`inter_class_overlap_weight`, `same_class_diversity_weight`,
shared class-union Φ). Companion to `docs/BUGS_train_inference_mismatch.md`,
which covers the train/inference and reporting path. **ID convention:** `M-##`.

**Status.** Audit only — no code changed.

**Verified against:** working tree at `ma-training-config-bump`
(`BenchMARL/conf/anchor.yaml` with `agents_per_class: 3`,
`inter_class_overlap_weight: 0.25`, `same_class_diversity_weight: 0.25`,
`init_mode: full_space` → quantile MDP), BenchMARL at
`/Users/ssuresh/BenchMARL`, and the three MADDPG seed-42 runs of 2026-09-01:

| run | `agents_per_class` | frames | datasets |
|---|---|---|---|
| `runs/mada_coord_probe_seed42` | 3 | 96 000 | iris, breast_cancer |
| `runs/mada_coord_retune_m1_w025_seed42` | 1 | 144 000 | iris, breast_cancer |
| `runs/mada_coord_retune_m3_w025_seed42` | 3 | 144 000 | iris, breast_cancer |

---

## Summary

`agents_per_class` does not create multiple agents. It creates **one policy per
class, replicated K times**. Every coordination term added in this working tree —
same-class diversity, same-class union Φ, inter-class overlap — is optimising over
a degree of freedom the learner does not have. Tuning
`inter_class_overlap_weight` (1.0 → 0.25) and `same_class_diversity_weight`
(0 → 0.25) cannot fix this, which is why the retune moved the numbers around
without a trend.

---

## M-01 — The three "agents" of a class are byte-identical networks `S0`

**Cause — two independent settings compose badly.**

1. `BenchMARL/benchmarl_wrappers.py:77` constructs `PettingZooWrapper(...)`
   **without a `group_map` argument**. torchrl then applies its default parallel
   grouping: *agents named `str_int` are grouped by `str`*
   (`torchrl/envs/libs/pettingzoo.py::_get_default_group_map`). With
   `agents_per_class: 3` the agent names are `agent_{c}_{k}`, so the groups are
   `agent_0 = [agent_0_0, agent_0_1, agent_0_2]`, etc. — **one group per class**.
2. `BenchMARL/conf/base_experiment.yaml:17` sets `share_policy_params: True`,
   which BenchMARL passes as `share_params` to the group's `MultiAgentMLP`
   (`benchmarl/algorithms/maddpg.py:100-110`). All agents **in a group share one
   parameter set**.

Composed: the K agents of a class are one network.

**`AnchorEnv._compute_group_map` (`environment.py:2951`), which assigns one group
per agent, is dead code on the training path.** `benchmarl_wrappers.group_map(env)`
(line 108) returns `env.group_map` where `env` is the *torchrl* wrapper, not
`AnchorEnv`, so torchrl's name-derived map wins. Inference bookkeeping still reads
`AnchorEnv.group_map`, so the two disagree — see M-05.

**Measured.** MD5 of the extracted per-agent actor weights,
`runs/mada_coord_retune_m3_w025_seed42/.../iris/individual_models`:

```
class_0/policy_agent_0_0.pth  0da0e3896ef28b60
class_0/policy_agent_0_1.pth  0da0e3896ef28b60   <- identical
class_0/policy_agent_0_2.pth  0da0e3896ef28b60   <- identical
class_1/policy_agent_1_*.pth  66d12dad3cad96a5   (all three)
class_2/policy_agent_2_*.pth  1c4abb42e3bfb1a1   (all three)
```

The training checkpoint agrees: `checkpoint_144000.pt` contains exactly three loss
modules — `loss_agent_0`, `loss_agent_1`, `loss_agent_2` — for what the config
calls nine agents. `individual_models/policies_index.json` records
`"agent": "agent_0_0", "group": "agent_0"`.

**Consequence.** `agents_per_class: 3` costs 3× the environment interaction and
buys one policy per class. "Multiple agents per class compete/cooperate to find
diverse, non-overlapping rules" (CLAUDE.md) does not describe what runs.

---

## M-02 — Same-class diversity is unlearnable by construction `S0`

Given M-01, the three same-class agents share θ. Under the quantile MDP they also
share their reset state: `a = 0`, `b = 1`, `q* = _class_centroid_quantiles(agent)`
(`environment.py:1579`) is a *per-class* quantity, and precision/coverage at reset
are class-level scalars. So all three agents observe the **same** vector and, being
the same deterministic network, emit the same action.

The only symmetry breaker is `_maybe_reset_diversity_landing`
(`environment.py:863`), which fires on `reset_diversity_frac` of episodes and
constrains `n_reset_landings` random dims. On the other ~70% of episodes the three
agents run identical trajectories, and `same_class_diversity_weight` charges each
of them a Simpson overlap of 1.0 against teammates they are exact copies of.

Worse, the gradient cannot resolve it: the shared-parameter actor update sums
`∇_θ Q(o_i, π_θ(o_i))` over i with identical `o_i`, so the diversity term pushes
one parameter set in three mutually cancelling directions.

**Measured** (m3 run, class-based rollouts, distinct rule strings per agent vs.
the class union):

| dataset / class | unique per agent | union over 3 agents | produced by **all three** |
|---|---|---|---|
| iris class_0 | 2, 2, 2 | 4 | 1 (`petal width ∈ [0.10, 0.40]`) |
| iris class_1 | 5, 5, 5 | 9 | 2 |
| iris class_2 | 5, 5, 5 | **6** | 3 (all `sepal width ∈ [2.0, ·]` variants) |
| breast class_0 | 5, 5, 5 | 13 | 1 |

iris class_2: 15 rollouts, 6 distinct rules, half of them emitted identically by
all three agents. That is the copy the diversity term was added to prevent.

---

## M-03 — `agents_per_class: 1` shares one policy across *all classes* `S0`

With `agents_per_class: 1` the names are `agent_0, agent_1, agent_2` — one
underscore — so torchrl's rule groups **every class into a single group `"agent"`**
(the checkpoint confirms: one loss module, `loss_agent`). With
`share_policy_params: True`, one network serves all classes.

**The observation carries no class identity.** `_get_observation`
(`environment.py:893`) returns `[a, b, q*, precision, coverage, mode_bit]`. In
class-based mode `q*` is the class median mapped through *that class's own*
conditional CDF, so it collapses to ≈0.5 for every class:

```
iris          class 0: q* mean 0.635   class 1: 0.530   class 2: 0.565
breast_cancer class 0: q* mean 0.501   class 1: 0.501
```

On breast_cancer the reset observation is identical across classes to three
decimals in every one of 90 box dims; the only class-dependent input is the reset
precision scalar (the class base rate). A single deterministic policy is asked to
emit a different anchor from the same input.

**Measured.** m1 run actor hashes: `policy_agent_0/1/2.pth` all
`155ece428bf7d698` (iris), both classes `e403be170d3f0644` (breast_cancer). The
iris m1 evaluation reports `class_0: best Fid=0.000 Pur=0.000 clsC=0.00` — the
shared policy never produced a class-0 rule.

**`agents_per_class: 1` must not be used as a MADA baseline**, and the m1 row of
the retune sweep is not interpretable.

---

## M-04 — The inter-class overlap term is outside every critic's input `S1`

`has_state()` returns `False` and `state_spec()` returns `None`
(`benchmarl_wrappers.py:94, 111`). BenchMARL's MADDPG therefore takes the
no-global-state branch (`maddpg.py:220-228`):

```python
critic_input_spec = Composite({group: observation_spec[group].update(action_spec[group])})
```

The critic sees **only its own group's** observations and actions. With one group
per class (M-01), the inter-class Simpson delta enters the reward but no
other-class box enters any critic input, and no agent's own observation carries
one either. It is unattributable noise on the TD target.

`centralised=True` in `get_value_module` is real but vacuous here: with
`share_param_critic: True` it centralises over the K identical same-class copies.
Same-class union Φ and same-class diversity *are* inside the class critic's input;
inter-class overlap is not, in any configuration except M=1 (where the single
group holds everything — but M=1 is broken for the separate reason in M-03).

Net: the reward has three coordination streams; **at most one is both observable
to a critic and actionable by an actor, and it is the one the shared parameters
cannot act on (M-02).**

---

## M-05 — Training and inference disagree about grouping `S2`

`extracted_rules.json` records `"agent": "agent_0_0", "group": "agent_0_0"` (from
`AnchorEnv._group_map`), while `policies_index.json` and the checkpoint record
`"group": "agent_0"` (torchrl). Harmless today because the extraction step writes
one file per agent name, but it means any code branching on `group` gets a
different answer on the two sides, and it is why M-01 is easy to miss when reading
the inference outputs.

---

## M-06 — The retune sweep has no signal to read `S1`

`evaluation/episode_reward_mean` across training (mean over eval episodes):

```
iris  m3_w025:   9.48  7.71  1.54  0.38  6.45  5.84  1.77
iris  m1_w025:   6.37  5.78  8.70 -1.39 -0.68 -0.12  1.09
iris  probe:     7.78  7.28  6.45  1.83 -2.22
breast m3_w025: 19.41 -8.23  4.30  8.34 -1.66 -7.45 12.15
breast m1_w025:  8.40 17.29  9.78  2.04  7.39  8.81  8.29
breast probe:    6.87 -9.07 -7.96 -9.68 -9.75
```

Five of six curves peak at the **first** evaluation (4 800 frames, i.e. before
meaningful training) and none is monotone. `box_precision_mean` behaves the same
way. This is the expected signature of a return dominated by terms the learner
cannot attribute (M-02, M-04), not of a hyperparameter that needs more tuning.

The test-set consequence, comparing M=1 → M=3 at fixed weights (0.25/0.25):

| | iris Fid | iris success | breast Fid | breast success |
|---|---|---|---|---|
| M=1 | 0.850 | 0.333 | 0.922 | 0.500 |
| M=3 | 0.733 | 0.167 | 0.884 | 0.233 |

Adding agents makes every headline number worse on both datasets — consistent with
M-01/M-02 (identical copies add coupling penalty and interaction cost, no
capacity) and inconsistent with any "more agents explore more" reading.

The `probe` run's iris result (`Fid=1.000, success=0.848`) is not a
counter-example: `abst=0.433` with class-1 union `n=5` and class-2 union `n=2`.
Those are the 2-row corner boxes already recorded in memory as the anchor-box
collapse — perfect fidelity on almost no support.

---

## Fix order

1. **M-01 / M-03 first — nothing below is measurable until agents are distinct.**
   Two independent changes, both needed:
   - Pass an explicit `group_map` to `PettingZooWrapper`
     (`benchmarl_wrappers.py:77`), e.g. `group_map=anchor_env.group_map`, so the
     intended grouping is the one that runs rather than a name-parsing accident.
   - Decide the grouping deliberately and set `share_policy_params` to match:
     - *one group per class, `share_policy_params: False`* → K distinct actors per
       class with a critic centralised over the class. This is the configuration
       in which same-class diversity and same-class union Φ are actually
       learnable, and is the smallest change that makes MADA a multi-agent method.
     - *one group per agent* → fully independent DDPG; state the paper's method as
       IDDPG, and drop or re-derive the coordination claims.
2. **M-03 also needs a class identity in the observation** (one-hot target class,
   or per-agent index) if any parameter sharing spans heterogeneous agents. Without
   it a shared actor is class-blind on breast_cancer by measurement, not by theory.
3. **M-04** — for the inter-class term to be learnable, either return a real
   `state_spec` (concatenated per-agent boxes) from the task so MADDPG's critic
   becomes centralised in the textbook sense, or accept that the term is a
   post-hoc filter and remove it from the reward, applying non-overlap at rule
   selection instead.
4. **M-02** — once actors are distinct, re-check whether the reset landing is still
   needed as a symmetry breaker, or whether per-agent `q*` (B-04/B-05 in the other
   register) is the principled one.
5. **M-05** — make the two group maps one source of truth.
6. **M-06** — re-run the weight sweep only after 1–3. The current sweep measured
   noise; `inter_class_overlap_weight: 0.25` and `same_class_diversity_weight: 0.25`
   in `conf/anchor.yaml` are not supported by it either way.

**Reporting.** No MADA number from the 2026-09-01 runs should enter the paper as a
multi-agent result. They are single-policy-per-class results (M=3) or
single-policy-overall results (M=1) with an unattributable coordination penalty
added to the return.
