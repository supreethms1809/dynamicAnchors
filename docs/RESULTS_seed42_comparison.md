# Seed-42 Results: MADA vs RLDA vs Baselines

Generated 2026-09-02 11:48. Branch `ma-training-config-bump`.

Tables are grouped **by dataset**, with the DNN and RandomForest black boxes
adjacent, so the effect of swapping the explained model is read top-to-bottom
within one dataset.

## What was run

| | |
|---|---|
| Arms | MADA (MADDPG), RLDA (DDPG), each against two black boxes |
| Black boxes | **DNN** (`runs/sweep_dnn/`) and **RandomForest** (`runs/sweep_rf/`) |
| Datasets | iris, wine, breast_cancer, synthetic, housing, uci_credit, uci_adult |
| Excluded | covtype, folktables (too large for the time budget) |
| Seeds | **42 only** — seeds 43–46 not yet run |
| Budgets | MADA 144,000 frames/agent · RLDA 270,000 total steps |
| Baselines | CART, greedy_anchors, sp_anchors, random_search |
| Selection | validation split, greedy marginal-gain union (k≤5); reporting on test |
| τ_P / τ_C | 0.90 / 0.10 |

Both arms of a given (dataset, black box) load the **same classifier file**, so MADA
and RLDA always explain an identical model.

### Metric directions

| Metric | Meaning | Better |
|---|---|---|
| **Fid** | P(rule's class = black-box prediction \| covered) | higher |
| **Cov** | fraction of test rows some rule fires on | higher |
| **Conflict** | fraction of rows covered by rules of >1 class | **lower** |
| **Abstain** | fraction of rows no rule covers | **lower** |
| **Success** | episodes reaching τ_P and τ_C / episodes attempted | higher |
| **Extraction queries** | black-box calls to BUILD the rule set — *not* training | lower |

`Cov` and `Abstain` sum to 1. Success applies only to the RL arms — baselines are
not episodic and report `null`, not 0. See **Query accounting** below before
quoting any cost number.

---

## Per-dataset results

### iris

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.958 | 0.800 | 0.000 | 0.200 | 0.122 (22/180) | 1,170 |
| RLDA | 0.967 | 1.000 | 0.567 | 0.000 | 0.400 (24/60) | 5,730 |
| CART | 0.964 | 0.933 | 0.000 | 0.067 | — | 120 |
| greedy_anchors | 1.000 | 0.567 | 0.000 | 0.433 | — | 65,614 |
| sp_anchors | 1.000 | 0.567 | 0.033 | 0.433 | — | 65,614 |
| random_search | 0.864 | 0.733 | 0.133 | 0.267 | — | 30 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.964 | 0.933 | 0.300 | 0.067 | 0.178 (32/180) | 1,170 |
| RLDA | 0.960 | 0.833 | 0.100 | 0.167 | 0.350 (21/60) | 5,730 |
| CART | 1.000 | 0.933 | 0.000 | 0.067 | — | 120 |
| greedy_anchors | 0.933 | 0.500 | 0.000 | 0.500 | — | 100,349 |
| sp_anchors | 0.933 | 0.500 | 0.033 | 0.500 | — | 100,349 |
| random_search | 0.864 | 0.733 | 0.133 | 0.267 | — | 30 |

*DNN → RF:* MADA Fid 0.958→0.964, Cov 0.800→0.933 · RLDA Fid 0.967→0.960, Cov 1.000→0.833

---

### wine

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.962 | 0.722 | 0.306 | 0.278 | 0.150 (27/180) | 1,384 |
| RLDA | 0.923 | 0.722 | 0.139 | 0.278 | 0.250 (15/60) | 6,738 |
| CART | 0.920 | 0.694 | 0.000 | 0.306 | — | 142 |
| greedy_anchors | 1.000 | 0.389 | 0.000 | 0.611 | — | 336,401 |
| sp_anchors | 1.000 | 0.389 | 0.000 | 0.611 | — | 336,401 |
| random_search | 1.000 | 0.028 | 0.000 | 0.972 | — | 36 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.657 | 0.972 | 0.333 | 0.028 | 0.128 (23/180) | 1,384 |
| RLDA | 0.968 | 0.861 | 0.111 | 0.139 | 0.367 (22/60) | 6,738 |
| CART | 0.920 | 0.694 | 0.000 | 0.306 | — | 142 |
| greedy_anchors | 1.000 | 0.556 | 0.000 | 0.444 | — | 481,231 |
| sp_anchors | 1.000 | 0.556 | 0.000 | 0.444 | — | 481,231 |
| random_search | — | 0.000 | 0.000 | 1.000 | — | 36 |

*DNN → RF:* MADA Fid 0.962→0.657, Cov 0.722→0.972 · RLDA Fid 0.923→0.968, Cov 0.722→0.861

---

### breast_cancer

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 | 0.877 | 0.053 | 0.123 | 0.133 (16/120) | 3,071 |
| RLDA | 0.918 | 0.746 | 0.000 | 0.254 | 0.250 (10/40) | 14,362 |
| CART | 0.908 | 0.956 | 0.000 | 0.044 | — | 455 |
| greedy_anchors | 1.000 | 0.526 | 0.000 | 0.474 | — | 794,676 |
| sp_anchors | 1.000 | 0.526 | 0.000 | 0.474 | — | 794,676 |
| random_search | 1.000 | 0.035 | 0.000 | 0.965 | — | 114 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.920 | 0.877 | 0.035 | 0.123 | 0.158 (19/120) | 3,071 |
| RLDA | 0.952 | 0.912 | 0.000 | 0.088 | 0.100 (4/40) | 14,362 |
| CART | 0.908 | 0.956 | 0.000 | 0.044 | — | 455 |
| greedy_anchors | 1.000 | 0.711 | 0.000 | 0.289 | — | 908,911 |
| sp_anchors | 1.000 | 0.711 | 0.000 | 0.289 | — | 908,911 |
| random_search | 1.000 | 0.026 | 0.000 | 0.974 | — | 114 |

*DNN → RF:* MADA Fid 0.930→0.920, Cov 0.877→0.877 · RLDA Fid 0.918→0.952, Cov 0.746→0.912

---

### synthetic

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.895 | 0.905 | 0.415 | 0.095 | 0.075 (9/120) | 5,400 |
| RLDA | 0.907 | 0.645 | 0.050 | 0.355 | 0.250 (10/40) | 25,240 |
| CART | 0.705 | 0.950 | 0.000 | 0.050 | — | 800 |
| greedy_anchors | 0.914 | 0.465 | 0.000 | 0.535 | — | 95,493 |
| sp_anchors | 0.914 | 0.465 | 0.000 | 0.535 | — | 95,493 |
| random_search | 1.000 | 0.100 | 0.000 | 0.900 | — | 200 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.890 | 0.865 | 0.305 | 0.135 | 0.150 (18/120) | 5,400 |
| RLDA | 1.000 | 0.475 | 0.000 | 0.525 | 0.525 (21/40) | 25,240 |
| CART | 0.716 | 0.950 | 0.000 | 0.050 | — | 800 |
| greedy_anchors | 0.955 | 0.440 | 0.000 | 0.560 | — | 49,689 |
| sp_anchors | 0.955 | 0.440 | 0.000 | 0.560 | — | 49,689 |
| random_search | 0.810 | 0.105 | 0.015 | 0.895 | — | 200 |

*DNN → RF:* MADA Fid 0.895→0.890, Cov 0.905→0.865 · RLDA Fid 0.907→1.000, Cov 0.645→0.475

---

### housing

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.569 | 0.740 | 0.080 | 0.260 | 0.004 (1/240) | 210,528 |
| RLDA | 0.515 | 0.646 | 0.100 | 0.354 | 0.037 (3/80) | 1,040,336 |
| CART | 0.563 | 0.998 | 0.000 | 0.002 | — | 16,512 |
| greedy_anchors | 0.681 | 0.238 | 0.000 | 0.762 | — | 482,158 |
| sp_anchors | 0.681 | 0.238 | 0.000 | 0.762 | — | 482,158 |
| random_search | 0.681 | 0.169 | 0.000 | 0.831 | — | 4,128 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.902 | 0.246 | 0.001 | 0.754 | 0.008 (2/240) | 210,528 |
| RLDA | 0.912 | 0.193 | 0.001 | 0.807 | 0.100 (8/80) | 1,040,336 |
| CART | 0.578 | 0.888 | 0.000 | 0.112 | — | 16,512 |
| greedy_anchors | 0.685 | 0.225 | 0.004 | 0.775 | — | 370,950 |
| sp_anchors | 0.692 | 0.271 | 0.000 | 0.729 | — | 370,950 |
| random_search | 0.825 | 0.101 | 0.000 | 0.899 | — | 4,128 |

*DNN → RF:* MADA Fid 0.569→0.902, Cov 0.740→0.246 · RLDA Fid 0.515→0.912, Cov 0.646→0.193

---

### uci_credit

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 | 0.935 | 0.043 | 0.065 | 0.120 (18/150) | 3,726 |
| RLDA | 0.926 | 0.884 | 0.000 | 0.116 | 0.060 (3/50) | 21,578 |
| CART | 0.905 | 0.993 | 0.000 | 0.007 | — | 552 |
| greedy_anchors | 0.980 | 0.710 | 0.007 | 0.290 | — | 36,117 |
| sp_anchors | 0.891 | 0.993 | 0.442 | 0.007 | — | 36,117 |
| random_search | — | 0.000 | 0.000 | 1.000 | — | 138 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.935 | 1.000 | 0.000 | 0.000 | 0.113 (17/150) | 3,726 |
| RLDA | 0.989 | 0.630 | 0.000 | 0.370 | 0.100 (5/50) | 21,578 |
| CART | 0.934 | 0.993 | 0.000 | 0.007 | — | 552 |
| greedy_anchors | 0.934 | 0.993 | 0.457 | 0.007 | — | 63,141 |
| sp_anchors | 0.934 | 0.993 | 0.457 | 0.007 | — | 63,141 |
| random_search | — | 0.000 | 0.000 | 1.000 | — | 138 |

*DNN → RF:* MADA Fid 0.930→0.935, Cov 0.935→1.000 · RLDA Fid 0.926→0.989, Cov 0.884→0.630

---

### uci_adult

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.877 | 0.894 | 0.080 | 0.106 | 0.047 (7/150) | 263,742 |
| RLDA | 0.871 | 0.746 | 0.054 | 0.254 | 0.020 (1/50) | 1,523,858 |
| CART | 0.800 | 1.000 | 0.000 | 0.000 | — | 39,073 |
| greedy_anchors | 0.936 | 0.573 | 0.000 | 0.427 | — | 66,155 |
| sp_anchors | 0.936 | 0.573 | 0.000 | 0.427 | — | 66,155 |
| random_search | 0.996 | 0.029 | 0.000 | 0.971 | — | 9,769 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.896 | 1.000 | 0.079 | 0.000 | 0.040 (6/150) | 263,742 |
| RLDA | 0.917 | 0.743 | 0.054 | 0.257 | 0.020 (1/50) | 1,523,858 |
| CART | 0.844 | 1.000 | 0.000 | 0.000 | — | 39,073 |
| greedy_anchors | 0.955 | 0.791 | 0.000 | 0.209 | — | 102,564 |
| sp_anchors | 0.932 | 1.000 | 0.117 | 0.000 | — | 102,564 |
| random_search | 0.973 | 0.049 | 0.000 | 0.951 | — | 9,769 |

*DNN → RF:* MADA Fid 0.877→0.896, Cov 0.894→1.000 · RLDA Fid 0.871→0.917, Cov 0.746→0.743

---

## Head-to-head: MADA vs RLDA

Wins / losses / ties across the 7 datasets.

| Metric | DNN | RF |
|---|---|---|
| Fidelity | **5**W / 2L / 0T | **1**W / 6L / 0T |
| Coverage | **5**W / 1L / 1T | **6**W / 1L / 0T |
| Conflict (lower better) | **2**W / 5L / 0T | **0**W / 6L / 1T |
| Abstention (lower better) | **5**W / 1L / 1T | **6**W / 1L / 0T |

## Head-to-head: RL arms vs CART

| Comparison | Fidelity | Coverage | Conflict | Abstention |
|---|---|---|---|---|
| MADA vs CART (DNN) | 6W/1L/0T | 1W/6L/0T | 0W/6L/1T | 1W/6L/0T |
| MADA vs CART (RF) | 5W/2L/0T | 3W/3L/1T | 0W/6L/1T | 3W/3L/1T |
| RLDA vs CART (DNN) | 6W/1L/0T | 2W/5L/0T | 0W/5L/2T | 2W/5L/0T |
| RLDA vs CART (RF) | 6W/1L/0T | 1W/6L/0T | 0W/4L/3T | 1W/6L/0T |

---

## Reading of the results

### 1. MADA beats RLDA on the semi-global objective (DNN)

Fidelity, coverage and abstention all favour MADA on **5 of 7** datasets, and the
three move together — the wins are not spread across different datasets. The
exceptions are `iris` (RLDA reaches full coverage with zero abstention) and
`synthetic` (RLDA slightly higher fidelity, but at 0.645 coverage vs 0.905).

This is the multi-agent claim doing what it is supposed to: more of each class
covered, without giving up fidelity.

**But this reverses under RandomForest.** Against RF, MADA still wins coverage
(6/7) and abstention (6/7) — more decisively than under the DNN — yet **loses
fidelity 1W/6L**. So the multi-agent advantage is not uniform: it buys coverage
under both black boxes, but only preserves fidelity under the DNN. Under RF the
single agent produces more faithful rules over a smaller region.

Any claim of the form "MADA dominates RLDA" is therefore not supported. The
supportable claim is narrower: MADA reliably converts abstention into coverage,
and whether that costs fidelity depends on the model being explained.

### 2. Inter-class coordination is still NOT working

(Counts below are from the generated tables above; where this prose and those
tables disagree, the tables are authoritative — they are computed from the JSONs.)

MADA loses on conflict 5/7. Adding rules adds overlap, and the
`inter_class_overlap_weight` term is not preventing it. `iris` is the one clear
counter-example (0.000 vs RLDA's 0.567).

This matters because reduced overlap is an explicit design goal, and it is the
one axis where a decision tree is structurally unbeatable — CART partitions the
space, so its conflict is 0.000 everywhere by construction.

### 3. Against CART: a trade-off, not a loss

MADA wins **fidelity** on 6/7 (DNN); CART wins **coverage** on 6/7. The largest
fidelity gaps are `synthetic` (0.895 vs 0.705) and `uci_adult` (0.877 vs 0.800).
CART's coverage advantage is largest on `housing` (0.998 vs 0.740).

The honest framing for the paper is that the RL arms produce *more faithful*
rules and the tree covers *more of the space* — not that the RL arms dominate.
CART also remains far cheaper in black-box queries.

### 4. Tree-based black boxes are easier to explain with boxes

RLDA fidelity rises going DNN → RF on 6 of 7 datasets (`wine` 0.923→0.968,
`synthetic` 0.907→1.000, `housing` 0.515→0.912, `uci_credit` 0.926→0.989).

Axis-aligned boxes approximate axis-aligned tree ensembles better than a smooth
DNN boundary — a hypothesis-class alignment effect. Coverage often pays for it
(`housing` RLDA drops to 0.193). Worth stating explicitly: it also demonstrates
the model-agnostic claim, which a DNN-only evaluation does not.

---

## Anomalies to resolve before publishing

### A. `wine` MADA/RF — Fid 0.657, far below every neighbour

RLDA gets 0.968 and CART 0.920 on the **identical** classifier; every other
MADA/RF fidelity is 0.89–0.96.

Checked: the divergence guard did **not** fire, so it is not a stopped run. The
per-class breakdown shows the cause is class_0:

    class_0: best Fid=0.303  clsC=0.83  n=33   <- covers 33 rows at 30% fidelity
    class_1: best Fid=1.000  clsC=0.64  n=9
    class_2: best Fid=0.429  clsC=0.30  n=7

A wide, low-fidelity class_0 box drags the global average down while pushing
coverage to 0.972. The marginal-gain selector kept it because it improves the
union's ranking score (coverage gain outweighs the fidelity loss in
`LCB(fid)*(1+cov)`), which is the selector behaving as designed but producing an
undesirable rule set. Compare `wine` MADA/DNN, which has the opposite shape:
Fid 0.962 at coverage 0.722.

Not a crash — a genuine failure of the fidelity/coverage trade-off on this cell.

### B. `housing` under RF — both arms collapse on coverage

MADA 0.246 and RLDA 0.193 coverage (abstention 0.75–0.81), while fidelity jumps
to ~0.90. Both arms find small, high-fidelity boxes and leave most of the space
uncovered. CART gets 0.888 coverage on the same model. `housing` is also the
weakest dataset under the DNN (Fid 0.515–0.569), so it is a consistently hard
case rather than a one-off.

### C. `random_search` produces degenerate rule sets

Coverage 0.026–0.100 on most datasets, and `null` fidelity on `uci_credit` and
`wine` (RF) where it covers nothing at all. It is not a meaningful competitor at
these settings; report it as such or drop it.

---

## Caveats

1. **Single seed.** Every count above is n=1. Yesterday's measurements showed
   large run-to-run variance on identical configs (RLDA breast_cancer success
   0.400 vs 0.200 across two runs). The 5/7 and 6/7 counts are provisional until
   seeds 43–46 land. The Wilcoxon table in `paper/tables/` still reports
   "too few seeds (need >= 2)" on every row.

2. **Reduced budgets.** MADA 144k frames/agent and RLDA 270k steps, against
   `DATASET_CONFIGS` defaults of 360k–720k frames and up to 1.08M steps. Results
   are internally consistent but not comparable to runs at default budgets.

3. **Residual critic divergence.** `gamma`/`discount` 0.99 → 0.95 cut peak
   critic loss by ~9 orders of magnitude (5.9e11 → ~950) and removed the drift
   that made Q estimates take the wrong sign, but the divergence guard still
   fires on some shards. Training is stabilised, not fixed.

4. **Baselines re-scored under the current evaluator.** The G-03 success
   denominator, G-13 compactness threshold and P-05 tie-break all changed
   yesterday; these baseline numbers were produced by the pipeline after those
   changes, so they are comparable to the RL rows here — but NOT to any numbers
   in `paper/tables/`, which predate them.

## Provenance

    runs/sweep_dnn/results/{ddpg,maddpg}/*__seed42__tp0p90__tc0p10.json
    runs/sweep_rf/results/{ddpg,maddpg}/*__seed42__tp0p90__tc0p10.json
    runs/sweep_logs/seed42_<arm>_<blackbox>_<dataset>.log
    runs/sweep_logs/sweep_progress.log

Generated by `revision/run_overnight_sweep.py` (commit 67c9e9c).

---

## Query accounting — READ THIS BEFORE QUOTING ANY COST NUMBER

The `extraction queries` column above is **`n_blackbox_queries`, which is
inference-time only**: the calls made to generate candidate boxes and select
among them. It does **not** include policy training. It was labelled
"black-box queries" in the first version of this document, which was misleading.

The three costs are separate quantities and amortise differently:

| Cost | Field | Paid | RL arms | CART / Anchors baselines |
|---|---|---|---|---|
| Training | `n_training_queries` | once, before any explanation exists | yes | n/a (no policy) |
| Extraction | `n_blackbox_queries` | once, to turn the policy into rules | yes | this IS their construction cost |
| Serving | `n_serving_queries_per_explanation` | per explanation | **0** (box is fixed) | per-instance methods pay here |
| Reporting | `n_reporting_queries` | measuring the method | excluded from cost | excluded from cost |

**Construction cost = training + extraction.** For the baselines, construction is
`n_blackbox_queries` alone, since they have no training phase. So the columns are
NOT directly comparable across arms as printed — an RL row's extraction figure
understates its true construction cost, while a CART row's is complete.

### The training figures in these runs are unreliable — do not publish them

Two independent defects, both found 2026-09-02 while preparing this document:

1. **RLDA undercounts by roughly a factor of n_classes.** RLDA trains one
   process per class (`run_parallel_classes.py`), and every shard wrote the same
   `training_queries.json` in a shared experiment folder — last writer wins.
   `wine` (3 classes) recorded `per_class={'1': 248}`: one shard's count
   presented as the run's total.

2. **MADA counts only reachable env instances.** The collector holds its own env
   copies in worker processes that the trainer cannot read, so the figure is a
   lower bound. iris MADA reports 120 training queries, which is not credible for
   144,000 frames.

The `complete: true` flag in `training_queries.json` means only "total > 0" and
should not be trusted as a completeness guarantee.

**Fixed in code** (commit following this document): each shard now writes
`training_queries_class{N}.json` and both inference paths sum across shards. The
fix does not retroactively repair these runs — **the training-query numbers for
seed 42 are wrong and must be regenerated** before any cost claim, break-even
figure, or query table is published.

### What this does not affect

Fid, Cov, Conflict, Abstain and Success are unaffected — they are computed from
the rule sets on the test split and have nothing to do with query accounting.
Only the cost columns and anything derived from them (`table_queries.tex`,
`break_even.json`) are implicated.

---

## Update 2026-09-02: extraction queries corrected, but still not trustworthy

Inference was re-run across all 28 cells (`--force --skip-train`, no retraining,
0 failures) after fixing a prediction-cache defect. `Fid/Cov/Conflict/Abstain`
are unchanged; only the cost column moved.

**Fixed.** `inference.py` builds a fresh `AnchorEnv` per episode, and each one's
prediction cache started cold, so the same split was re-classified once per
episode. iris MADA: 225 envs x 90 train rows = 20,250 of a reported 21,330.
`_get_cached_probs` now shares a process-wide cache keyed on
`(id(classifier), split, X.shape, sha1(X))`, and counts only real misses.
iris MADA fell **21,330 -> 1,170 (18x)**.

**Still unexplained — do not publish these as costs.** Two things do not add up:

1. **RLDA reports ~5x MADA on every dataset** (4.67–5.79), independent of class
   count. MADA runs *more* episodes (9 agents vs 3), so the ratio should favour
   MADA if it tracked work done. RLDA runs one process per class, so each shard
   has its own cache — but that predicts a factor of `n_classes` (2–4), not a
   flat ~5.

2. **Large datasets remain implausibly high**: `uci_adult` RLDA 1,523,858 and
   MADA 263,742, against CART's 39,073. For an amortised method whose serving
   cost is zero, extraction should be a small multiple of the split size, not
   ~50x it. The per-episode cache should be hitting; evidently it is not on
   these cells.

So the cache fix removed one over-count but did not make the column correct. The
cost columns, `table_queries.tex` and `break_even.json` all still need work
before publication. Fidelity/coverage/conflict/abstention conclusions in this
document do not depend on any of it.
