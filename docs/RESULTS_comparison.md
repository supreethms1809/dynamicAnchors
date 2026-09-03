# Results: MADA vs RLDA vs Baselines

Generated 2026-09-03 16:38. Branch `ma-training-config-bump`.

Tables are grouped **by dataset**, with the DNN and RandomForest black boxes
adjacent, so the effect of swapping the explained model is read top-to-bottom
within one dataset.

Each dataset gets a **mean ±sd** summary table per black box, followed by the
raw per-seed values in **seed 42 / seed 43 / seed 44** order. An em dash marks a seed that has
not been run yet — it is excluded from the mean rather than counted as zero,
and the `seeds` column says how many actually contributed, so a mean over two
seeds is never mistaken for a mean over five.

The sd is the sample standard deviation across seeds. With three seeds it is a
crude spread estimate, not a confidence interval, and a difference smaller than
the sd should not be reported as an effect.

## What was run

| | |
|---|---|
| Arms | MADA (MADDPG), RLDA (DDPG), each against two black boxes |
| Black boxes | **DNN** (`runs/sweep_dnn/`) and **RandomForest** (`runs/sweep_rf/`) |
| Datasets | iris, wine, breast_cancer, synthetic, housing, uci_credit, uci_adult |
| Excluded | covtype, folktables (too large for the time budget) |
| Seeds | 42, 43, 44 |
| Budgets | MADA 144,000 frames/agent · RLDA 270,000 total steps |
| Baselines | CART, greedy_anchors, sp_anchors, random_search |
| Selection | validation split, greedy marginal-gain union (k≤5); reporting on test |
| τ_P / τ_C | 0.90 / 0.10 |

Both arms of a given (dataset, black box) load the **same classifier file**, so MADA
and RLDA always explain an identical model.

### Coverage of this sweep

| black box | seed | iris | wine | breast_cancer | synthetic | housing | uci_credit | uci_adult |
|---|---|---|---|---|---|---|---|---|
| DNN | 42 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DNN | 43 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DNN | 44 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RandomForest | 42 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RandomForest | 43 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RandomForest | 44 | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |

✅ both arms · ◐ one arm only · — not run

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
not episodic and report `—`, not 0. See **Query accounting** below before
quoting any cost number.

---

## Per-dataset results

### iris

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.903 ±0.050 | 0.889 ±0.084 | 0.044 ±0.077 | 0.111 ±0.084 | 0.150 ±0.043 | 1,170 ±0 |
| RLDA | 3 | 0.895 ±0.062 | 0.967 ±0.033 | 0.367 ±0.219 | 0.033 ±0.033 | 0.400 ±0.067 | 5,730 ±0 |
| CART | 3 | 0.942 ±0.072 | 0.944 ±0.019 | 0.000 ±0.000 | 0.056 ±0.019 | — | 120 ±0 |
| greedy_anchors | 3 | 1.000 ±0.000 | 0.578 ±0.019 | 0.000 ±0.000 | 0.422 ±0.019 | — | 77,037 ±9,917 |
| sp_anchors | 3 | 0.944 ±0.096 | 0.644 ±0.135 | 0.022 ±0.019 | 0.356 ±0.135 | — | 77,037 ±9,917 |
| random_search | 3 | 0.929 ±0.068 | 0.733 ±0.133 | 0.078 ±0.051 | 0.267 ±0.133 | — | 30 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.958 / 0.862 / 0.889 | 0.800 / 0.967 / 0.900 | 0.000 / 0.000 / 0.133 | 0.200 / 0.033 / 0.100 | 0.122 / 0.200 / 0.128 | 1,170 / 1,170 / 1,170 |
| RLDA | 0.967 / 0.862 / 0.857 | 1.000 / 0.967 / 0.933 | 0.567 / 0.133 / 0.400 | 0.000 / 0.033 / 0.067 | 0.400 / 0.333 / 0.467 | 5,730 / 5,730 / 5,730 |
| CART | 0.964 / 0.862 / 1.000 | 0.933 / 0.967 / 0.933 | 0.000 / 0.000 / 0.000 | 0.067 / 0.033 / 0.067 | — | 120 / 120 / 120 |
| greedy_anchors | 1.000 / 1.000 / 1.000 | 0.567 / 0.600 / 0.567 | 0.000 / 0.000 / 0.000 | 0.433 / 0.400 / 0.433 | — | 65,614 / 83,451 / 82,045 |
| sp_anchors | 1.000 / 0.833 / 1.000 | 0.567 / 0.800 / 0.567 | 0.033 / 0.033 / 0.000 | 0.433 / 0.200 / 0.433 | — | 65,614 / 83,451 / 82,045 |
| random_search | 0.864 / 0.923 / 1.000 | 0.733 / 0.867 / 0.600 | 0.133 / 0.067 / 0.033 | 0.267 / 0.133 / 0.400 | — | 30 / 30 / 30 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.946 ±0.025 | 0.933 ±0.000 | 0.150 ±0.212 | 0.067 ±0.000 | 0.189 ±0.016 | 1,170 ±0 |
| RLDA | 3 | 0.947 ±0.061 | 0.800 ±0.058 | 0.067 ±0.033 | 0.200 ±0.058 | 0.478 ±0.155 | 5,730 ±0 |
| CART | 3 | 0.954 ±0.080 | 0.944 ±0.019 | 0.000 ±0.000 | 0.056 ±0.019 | — | 120 ±0 |
| greedy_anchors | 3 | 0.959 ±0.036 | 0.511 ±0.084 | 0.000 ±0.000 | 0.489 ±0.084 | — | 76,511 ±20,778 |
| sp_anchors | 3 | 0.960 ±0.035 | 0.622 ±0.117 | 0.033 ±0.033 | 0.378 ±0.117 | — | 76,511 ±20,778 |
| random_search | 3 | 0.893 ±0.096 | 0.756 ±0.135 | 0.044 ±0.077 | 0.244 ±0.135 | — | 30 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.964 / 0.929 / — | 0.933 / 0.933 / — | 0.300 / 0.000 / — | 0.067 / 0.067 / — | 0.178 / 0.200 / — | 1,170 / 1,170 / — |
| RLDA | 0.960 / 0.880 / 1.000 | 0.833 / 0.833 / 0.733 | 0.100 / 0.067 / 0.033 | 0.167 / 0.167 / 0.267 | 0.350 / 0.433 / 0.650 | 5,730 / 5,730 / 5,730 |
| CART | 1.000 / 0.862 / 1.000 | 0.933 / 0.967 / 0.933 | 0.000 / 0.000 / 0.000 | 0.067 / 0.033 / 0.067 | — | 120 / 120 / 120 |
| greedy_anchors | 0.933 / 1.000 / 0.944 | 0.500 / 0.433 / 0.600 | 0.000 / 0.000 / 0.000 | 0.500 / 0.567 / 0.400 | — | 100,349 / 62,238 / 66,945 |
| sp_anchors | 0.933 / 1.000 / 0.947 | 0.500 / 0.733 / 0.633 | 0.033 / 0.067 / 0.000 | 0.500 / 0.267 / 0.367 | — | 100,349 / 62,238 / 66,945 |
| random_search | 0.864 / 0.815 / 1.000 | 0.733 / 0.900 / 0.633 | 0.133 / 0.000 / 0.000 | 0.267 / 0.100 / 0.367 | — | 30 / 30 / 30 |


### wine

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.894 ±0.084 | 0.704 ±0.016 | 0.157 ±0.131 | 0.296 ±0.016 | 0.165 ±0.017 | 1,384 ±0 |
| RLDA | 3 | 0.785 ±0.129 | 0.796 ±0.064 | 0.250 ±0.127 | 0.204 ±0.064 | 0.267 ±0.060 | 6,738 ±0 |
| CART | 3 | 0.840 ±0.070 | 0.787 ±0.085 | 0.000 ±0.000 | 0.213 ±0.085 | — | 142 ±0 |
| greedy_anchors | 3 | 0.985 ±0.026 | 0.519 ±0.116 | 0.000 ±0.000 | 0.481 ±0.116 | — | 338,979 ±24,313 |
| sp_anchors | 3 | 0.985 ±0.026 | 0.519 ±0.116 | 0.000 ±0.000 | 0.481 ±0.116 | — | 338,979 ±24,313 |
| random_search | 3 | 1.000 ±0.000 | 0.019 ±0.016 | 0.000 ±0.000 | 0.981 ±0.016 | — | 36 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.962 / 0.800 / 0.920 | 0.722 / 0.694 / 0.694 | 0.306 / 0.056 / 0.111 | 0.278 / 0.306 / 0.306 | 0.150 / 0.161 / 0.183 | 1,384 / 1,384 / 1,384 |
| RLDA | 0.923 / 0.667 / 0.767 | 0.722 / 0.833 / 0.833 | 0.139 / 0.389 / 0.222 | 0.278 / 0.167 / 0.167 | 0.250 / 0.217 / 0.333 | 6,738 / 6,738 / 6,738 |
| CART | 0.920 / 0.806 / 0.793 | 0.694 / 0.861 / 0.806 | 0.000 / 0.000 / 0.000 | 0.306 / 0.139 / 0.194 | — | 142 / 142 / 142 |
| greedy_anchors | 1.000 / 0.955 / 1.000 | 0.389 / 0.611 / 0.556 | 0.000 / 0.000 / 0.000 | 0.611 / 0.389 / 0.444 | — | 336,401 / 316,057 / 364,478 |
| sp_anchors | 1.000 / 0.955 / 1.000 | 0.389 / 0.611 / 0.556 | 0.000 / 0.000 / 0.000 | 0.611 / 0.389 / 0.444 | — | 336,401 / 316,057 / 364,478 |
| random_search | 1.000 / — / 1.000 | 0.028 / 0.000 / 0.028 | 0.000 / 0.000 / 0.000 | 0.972 / 1.000 / 0.972 | — | 36 / 36 / 36 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.704 ±0.066 | 0.875 ±0.137 | 0.194 ±0.196 | 0.125 ±0.137 | 0.158 ±0.043 | 1,384 ±0 |
| RLDA | 3 | 0.866 ±0.091 | 0.796 ±0.112 | 0.194 ±0.100 | 0.204 ±0.112 | 0.217 ±0.132 | 6,738 ±0 |
| CART | 3 | 0.861 ±0.064 | 0.787 ±0.085 | 0.000 ±0.000 | 0.213 ±0.085 | — | 142 ±0 |
| greedy_anchors | 3 | 0.979 ±0.036 | 0.556 ±0.111 | 0.009 ±0.016 | 0.444 ±0.111 | — | 471,417 ±26,420 |
| sp_anchors | 3 | 0.979 ±0.036 | 0.556 ±0.111 | 0.009 ±0.016 | 0.444 ±0.111 | — | 471,417 ±26,420 |
| random_search | 3 | 1.000 | 0.009 ±0.016 | 0.000 ±0.000 | 0.991 ±0.016 | — | 36 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.657 / 0.750 / — | 0.972 / 0.778 / — | 0.333 / 0.056 / — | 0.028 / 0.222 / — | 0.128 / 0.189 / — | 1,384 / 1,384 / — |
| RLDA | 0.968 / 0.792 / 0.839 | 0.861 / 0.667 / 0.861 | 0.111 / 0.167 / 0.306 | 0.139 / 0.333 / 0.139 | 0.367 / 0.167 / 0.117 | 6,738 / 6,738 / 6,738 |
| CART | 0.920 / 0.871 / 0.793 | 0.694 / 0.861 / 0.806 | 0.000 / 0.000 / 0.000 | 0.306 / 0.139 / 0.194 | — | 142 / 142 / 142 |
| greedy_anchors | 1.000 / 0.938 / 1.000 | 0.556 / 0.444 / 0.667 | 0.000 / 0.028 / 0.000 | 0.444 / 0.556 / 0.333 | — | 481,231 / 491,526 / 441,495 |
| sp_anchors | 1.000 / 0.938 / 1.000 | 0.556 / 0.444 / 0.667 | 0.000 / 0.028 / 0.000 | 0.444 / 0.556 / 0.333 | — | 481,231 / 491,526 / 441,495 |
| random_search | — / — / 1.000 | 0.000 / 0.000 / 0.028 | 0.000 / 0.000 / 0.000 | 1.000 / 1.000 / 0.972 | — | 36 / 36 / 36 |


### breast_cancer

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.921 ±0.032 | 0.959 ±0.071 | 0.073 ±0.051 | 0.041 ±0.071 | 0.144 ±0.035 | 3,071 ±0 |
| RLDA | 3 | 0.943 ±0.049 | 0.743 ±0.048 | 0.006 ±0.010 | 0.257 ±0.048 | 0.308 ±0.101 | 14,362 ±0 |
| CART | 3 | 0.923 ±0.017 | 0.912 ±0.038 | 0.000 ±0.000 | 0.088 ±0.038 | — | 455 ±0 |
| greedy_anchors | 3 | 1.000 ±0.000 | 0.421 ±0.098 | 0.000 ±0.000 | 0.579 ±0.098 | — | 745,680 ±57,033 |
| sp_anchors | 3 | 0.993 ±0.012 | 0.427 ±0.097 | 0.000 ±0.000 | 0.573 ±0.097 | — | 745,680 ±57,033 |
| random_search | 3 | 0.939 ±0.105 | 0.056 ±0.035 | 0.006 ±0.010 | 0.944 ±0.035 | — | 114 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 / 0.886 / 0.947 | 0.877 / 1.000 / 1.000 | 0.053 / 0.035 / 0.132 | 0.123 / 0.000 / 0.000 | 0.133 / 0.117 / 0.183 | 3,071 / 3,071 / 3,071 |
| RLDA | 0.918 / 0.911 / 1.000 | 0.746 / 0.693 / 0.789 | 0.000 / 0.000 / 0.018 | 0.254 / 0.307 / 0.211 | 0.250 / 0.250 / 0.425 | 14,362 / 14,362 / 14,362 |
| CART | 0.908 / 0.921 / 0.941 | 0.956 / 0.886 / 0.895 | 0.000 / 0.000 / 0.000 | 0.044 / 0.114 / 0.105 | — | 455 / 455 / 455 |
| greedy_anchors | 1.000 / 1.000 / 1.000 | 0.526 / 0.333 / 0.404 | 0.000 / 0.000 / 0.000 | 0.474 / 0.667 / 0.596 | — | 794,676 / 683,073 / 759,292 |
| sp_anchors | 1.000 / 1.000 / 0.979 | 0.526 / 0.333 / 0.421 | 0.000 / 0.000 / 0.000 | 0.474 / 0.667 / 0.579 | — | 794,676 / 683,073 / 759,292 |
| random_search | 1.000 / 1.000 / 0.818 | 0.035 / 0.035 / 0.096 | 0.000 / 0.000 / 0.018 | 0.965 / 0.965 / 0.904 | — | 114 / 114 / 114 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.887 ±0.047 | 0.917 ±0.056 | 0.070 ±0.050 | 0.083 ±0.056 | 0.179 ±0.029 | 3,071 ±0 |
| RLDA | 3 | 0.956 ±0.043 | 0.798 ±0.129 | 0.000 ±0.000 | 0.202 ±0.129 | 0.167 ±0.058 | 14,362 ±0 |
| CART | 3 | 0.943 ±0.032 | 0.912 ±0.038 | 0.000 ±0.000 | 0.088 ±0.038 | — | 455 ±0 |
| greedy_anchors | 3 | 0.980 ±0.034 | 0.544 ±0.145 | 0.000 ±0.000 | 0.456 ±0.145 | — | 885,418 ±36,706 |
| sp_anchors | 3 | 0.980 ±0.034 | 0.544 ±0.145 | 0.000 ±0.000 | 0.456 ±0.145 | — | 885,418 ±36,706 |
| random_search | 3 | 0.667 ±0.577 | 0.035 ±0.032 | 0.000 ±0.000 | 0.965 ±0.032 | — | 114 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.920 / 0.853 / — | 0.877 / 0.956 / — | 0.035 / 0.105 / — | 0.123 / 0.044 / — | 0.158 / 0.200 / — | 3,071 / 3,071 / — |
| RLDA | 0.952 / 0.915 / 1.000 | 0.912 / 0.825 / 0.658 | 0.000 / 0.000 / 0.000 | 0.088 / 0.175 / 0.342 | 0.100 / 0.200 / 0.200 | 14,362 / 14,362 / 14,362 |
| CART | 0.908 / 0.970 / 0.951 | 0.956 / 0.886 / 0.895 | 0.000 / 0.000 / 0.000 | 0.044 / 0.114 / 0.105 | — | 455 / 455 / 455 |
| greedy_anchors | 1.000 / 1.000 / 0.941 | 0.711 / 0.474 / 0.447 | 0.000 / 0.000 / 0.000 | 0.289 / 0.526 / 0.553 | — | 908,911 / 843,120 / 904,222 |
| sp_anchors | 1.000 / 1.000 / 0.941 | 0.711 / 0.474 / 0.447 | 0.000 / 0.000 / 0.000 | 0.289 / 0.526 / 0.553 | — | 908,911 / 843,120 / 904,222 |
| random_search | 1.000 / 1.000 / 0.000 | 0.026 / 0.070 / 0.009 | 0.000 / 0.000 / 0.000 | 0.974 / 0.930 / 0.991 | — | 114 / 114 / 114 |


### synthetic

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.909 ±0.017 | 0.767 ±0.140 | 0.200 ±0.186 | 0.233 ±0.140 | 0.083 ±0.022 | 5,400 ±0 |
| RLDA | 3 | 0.915 ±0.041 | 0.598 ±0.209 | 0.017 ±0.029 | 0.402 ±0.209 | 0.333 ±0.088 | 25,240 ±0 |
| CART | 3 | 0.702 ±0.047 | 0.960 ±0.017 | 0.000 ±0.000 | 0.040 ±0.017 | — | 800 ±0 |
| greedy_anchors | 3 | 0.948 ±0.030 | 0.350 ±0.103 | 0.000 ±0.000 | 0.650 ±0.103 | — | 82,369 ±21,138 |
| sp_anchors | 3 | 0.948 ±0.030 | 0.350 ±0.103 | 0.000 ±0.000 | 0.650 ±0.103 | — | 82,369 ±21,138 |
| random_search | 3 | 0.865 ±0.176 | 0.102 ±0.033 | 0.003 ±0.006 | 0.898 ±0.033 | — | 200 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.895 / 0.903 / 0.928 | 0.905 / 0.770 / 0.625 | 0.415 / 0.100 / 0.085 | 0.095 / 0.230 / 0.375 | 0.075 / 0.108 / 0.067 | 5,400 / 5,400 / 5,400 |
| RLDA | 0.907 / 0.959 / 0.878 | 0.645 / 0.370 / 0.780 | 0.050 / 0.000 / 0.000 | 0.355 / 0.630 / 0.220 | 0.250 / 0.425 / 0.325 | 25,240 / 25,240 / 25,240 |
| CART | 0.705 / 0.653 / 0.747 | 0.950 / 0.980 / 0.950 | 0.000 / 0.000 / 0.000 | 0.050 / 0.020 / 0.050 | — | 800 / 800 / 800 |
| greedy_anchors | 0.914 / 0.969 / 0.962 | 0.465 / 0.320 / 0.265 | 0.000 / 0.000 / 0.000 | 0.535 / 0.680 / 0.735 | — | 95,493 / 93,630 / 57,985 |
| sp_anchors | 0.914 / 0.969 / 0.962 | 0.465 / 0.320 / 0.265 | 0.000 / 0.000 / 0.000 | 0.535 / 0.680 / 0.735 | — | 95,493 / 93,630 / 57,985 |
| random_search | 1.000 / 0.929 / 0.667 | 0.100 / 0.070 / 0.135 | 0.000 / 0.000 / 0.010 | 0.900 / 0.930 / 0.865 | — | 200 / 200 / 200 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.924 ±0.048 | 0.735 ±0.184 | 0.180 ±0.177 | 0.265 ±0.184 | 0.129 ±0.029 | 5,400 ±0 |
| RLDA | 3 | 0.978 ±0.020 | 0.460 ±0.059 | 0.000 ±0.000 | 0.540 ±0.059 | 0.425 ±0.195 | 25,240 ±0 |
| CART | 3 | 0.730 ±0.059 | 0.960 ±0.017 | 0.000 ±0.000 | 0.040 ±0.017 | — | 800 ±0 |
| greedy_anchors | 3 | 0.970 ±0.015 | 0.368 ±0.063 | 0.000 ±0.000 | 0.632 ±0.063 | — | 65,787 ±20,140 |
| sp_anchors | 3 | 0.970 ±0.015 | 0.368 ±0.063 | 0.000 ±0.000 | 0.632 ±0.063 | — | 65,787 ±20,140 |
| random_search | 3 | 0.796 ±0.086 | 0.107 ±0.028 | 0.008 ±0.008 | 0.893 ±0.028 | — | 200 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.890 / 0.959 / — | 0.865 / 0.605 / — | 0.305 / 0.055 / — | 0.135 / 0.395 / — | 0.150 / 0.108 / — | 5,400 / 5,400 / — |
| RLDA | 1.000 / 0.975 / 0.961 | 0.475 / 0.395 / 0.510 | 0.000 / 0.000 / 0.000 | 0.525 / 0.605 / 0.490 | 0.525 / 0.200 / 0.550 | 25,240 / 25,240 / 25,240 |
| CART | 0.716 / 0.679 / 0.795 | 0.950 / 0.980 / 0.950 | 0.000 / 0.000 / 0.000 | 0.050 / 0.020 / 0.050 | — | 800 / 800 / 800 |
| greedy_anchors | 0.955 / 0.971 / 0.984 | 0.440 / 0.345 / 0.320 | 0.000 / 0.000 / 0.000 | 0.560 / 0.655 / 0.680 | — | 49,689 / 88,371 / 59,302 |
| sp_anchors | 0.955 / 0.971 / 0.984 | 0.440 / 0.345 / 0.320 | 0.000 / 0.000 / 0.000 | 0.560 / 0.655 / 0.680 | — | 49,689 / 88,371 / 59,302 |
| random_search | 0.810 / 0.875 / 0.704 | 0.105 / 0.080 / 0.135 | 0.015 / 0.000 / 0.010 | 0.895 / 0.920 / 0.865 | — | 200 / 200 / 200 |


### housing

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.568 ±0.035 | 0.725 ±0.014 | 0.038 ±0.037 | 0.275 ±0.014 | 0.011 ±0.006 | 210,528 ±0 |
| RLDA | 3 | 0.521 ±0.050 | 0.756 ±0.152 | 0.175 ±0.197 | 0.244 ±0.152 | 0.033 ±0.019 | 1,040,336 ±0 |
| CART | 3 | 0.532 ±0.028 | 0.999 ±0.001 | 0.000 ±0.000 | 0.001 ±0.001 | — | 16,512 ±0 |
| greedy_anchors | 3 | 0.606 ±0.155 | 0.334 ±0.158 | 0.028 ±0.046 | 0.666 ±0.158 | — | 498,579 ±23,622 |
| sp_anchors | 3 | 0.609 ±0.149 | 0.340 ±0.169 | 0.034 ±0.057 | 0.660 ±0.169 | — | 498,579 ±23,622 |
| random_search | 3 | 0.758 ±0.072 | 0.115 ±0.049 | 0.000 ±0.000 | 0.885 ±0.049 | — | 4,128 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.569 / 0.532 / 0.602 | 0.740 / 0.713 / 0.721 | 0.080 / 0.013 / 0.021 | 0.260 / 0.287 / 0.279 | 0.004 / 0.017 / 0.013 | 210,528 / 210,528 / 210,528 |
| RLDA | 0.515 / 0.474 / 0.574 | 0.646 / 0.930 / 0.692 | 0.100 / 0.399 / 0.026 | 0.354 / 0.070 / 0.308 | 0.037 / 0.013 / 0.050 | 1,040,336 / 1,040,336 / 1,040,336 |
| CART | 0.563 / 0.512 / 0.520 | 0.998 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.002 / 0.000 / 0.000 | — | 16,512 / 16,512 / 16,512 |
| greedy_anchors | 0.681 / 0.427 / 0.709 | 0.238 / 0.516 / 0.247 | 0.000 / 0.080 / 0.003 | 0.762 / 0.484 / 0.753 | — | 482,158 / 525,651 / 487,929 |
| sp_anchors | 0.681 / 0.438 / 0.709 | 0.238 / 0.535 / 0.247 | 0.000 / 0.100 / 0.003 | 0.762 / 0.465 / 0.753 | — | 482,158 / 525,651 / 487,929 |
| random_search | 0.681 / 0.768 / 0.824 | 0.169 / 0.074 / 0.102 | 0.000 / 0.000 / 0.000 | 0.831 / 0.926 / 0.898 | — | 4,128 / 4,128 / 4,128 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.735 ±0.235 | 0.565 ±0.451 | 0.155 ±0.217 | 0.435 ±0.451 | 0.008 ±0.000 | 210,528 ±0 |
| RLDA | 3 | 0.876 ±0.033 | 0.270 ±0.078 | 0.005 ±0.006 | 0.730 ±0.078 | 0.067 ±0.029 | 1,040,336 ±0 |
| CART | 3 | 0.575 ±0.013 | 0.921 ±0.068 | 0.000 ±0.000 | 0.079 ±0.068 | — | 16,512 ±0 |
| greedy_anchors | 3 | 0.717 ±0.071 | 0.224 ±0.098 | 0.008 ±0.011 | 0.776 ±0.098 | — | 367,352 ±27,564 |
| sp_anchors | 3 | 0.707 ±0.037 | 0.269 ±0.005 | 0.000 ±0.000 | 0.731 ±0.005 | — | 367,352 ±27,564 |
| random_search | 3 | 0.812 ±0.019 | 0.102 ±0.001 | 0.000 ±0.000 | 0.898 ±0.001 | — | 4,128 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.902 / 0.569 / — | 0.246 / 0.884 / — | 0.001 / 0.308 / — | 0.754 / 0.116 / — | 0.008 / 0.008 / — | 210,528 / 210,528 / — |
| RLDA | 0.912 / 0.850 / 0.864 | 0.193 / 0.270 / 0.348 | 0.001 / 0.003 / 0.012 | 0.807 / 0.730 / 0.652 | 0.100 / 0.050 / 0.050 | 1,040,336 / 1,040,336 / 1,040,336 |
| CART | 0.578 / 0.587 / 0.562 | 0.888 / 0.876 / 1.000 | 0.000 / 0.000 / 0.000 | 0.112 / 0.124 / 0.000 | — | 16,512 / 16,512 / 16,512 |
| greedy_anchors | 0.685 / 0.666 / 0.798 | 0.225 / 0.322 / 0.125 | 0.004 / 0.021 / 0.000 | 0.775 / 0.678 / 0.875 | — | 370,950 / 392,940 / 338,166 |
| sp_anchors | 0.692 / 0.750 / 0.680 | 0.271 / 0.273 / 0.263 | 0.000 / 0.000 / 0.000 | 0.729 / 0.727 / 0.737 | — | 370,950 / 392,940 / 338,166 |
| random_search | 0.825 / 0.822 / 0.790 | 0.101 / 0.104 / 0.102 | 0.000 / 0.000 / 0.000 | 0.899 / 0.896 / 0.898 | — | 4,128 / 4,128 / 4,128 |


### uci_credit

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.909 ±0.018 | 0.978 ±0.038 | 0.179 ±0.273 | 0.022 ±0.038 | 0.078 ±0.057 | 3,726 ±0 |
| RLDA | 3 | 0.916 ±0.026 | 0.870 ±0.032 | 0.000 ±0.000 | 0.130 ±0.032 | 0.080 ±0.053 | 21,578 ±0 |
| CART | 3 | 0.907 ±0.011 | 0.986 ±0.007 | 0.000 ±0.000 | 0.014 ±0.007 | — | 552 ±0 |
| greedy_anchors | 3 | 0.966 ±0.042 | 0.778 ±0.177 | 0.162 ±0.268 | 0.222 ±0.177 | — | 29,226 ±6,279 |
| sp_anchors | 3 | 0.936 ±0.057 | 0.872 ±0.197 | 0.307 ±0.260 | 0.128 ±0.197 | — | 29,226 ±6,279 |
| random_search | 3 | 1.000 ±0.000 | 0.022 ±0.032 | 0.000 ±0.000 | 0.978 ±0.032 | — | 138 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 / 0.899 / 0.899 | 0.935 / 1.000 / 1.000 | 0.043 / 0.493 / 0.000 | 0.065 / 0.000 / 0.000 | 0.120 / 0.013 / 0.100 | 3,726 / 3,726 / 3,726 |
| RLDA | 0.926 / 0.935 / 0.887 | 0.884 / 0.891 / 0.833 | 0.000 / 0.000 / 0.000 | 0.116 / 0.109 / 0.167 | 0.060 / 0.040 / 0.140 | 21,578 / 21,578 / 21,578 |
| CART | 0.905 / 0.919 / 0.897 | 0.993 / 0.978 / 0.986 | 0.000 / 0.000 / 0.000 | 0.007 / 0.022 / 0.014 | — | 552 / 552 / 552 |
| greedy_anchors | 0.980 / 0.919 / 1.000 | 0.710 / 0.978 / 0.645 | 0.007 / 0.471 / 0.007 | 0.290 / 0.022 / 0.355 | — | 36,117 / 27,732 / 23,828 |
| sp_anchors | 0.891 / 0.919 / 1.000 | 0.993 / 0.978 / 0.645 | 0.442 / 0.471 / 0.007 | 0.007 / 0.022 / 0.355 | — | 36,117 / 27,732 / 23,828 |
| random_search | — / 1.000 / 1.000 | 0.000 / 0.058 / 0.007 | 0.000 / 0.000 / 0.000 | 1.000 / 0.942 / 0.993 | — | 138 / 138 / 138 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.952 ±0.024 | 0.967 ±0.046 | 0.000 ±0.000 | 0.033 ±0.046 | 0.090 ±0.033 | 3,726 ±0 |
| RLDA | 3 | 0.897 ±0.169 | 0.703 ±0.171 | 0.010 ±0.017 | 0.297 ±0.171 | 0.120 ±0.020 | 21,578 ±0 |
| CART | 3 | 0.912 ±0.051 | 0.986 ±0.007 | 0.000 ±0.000 | 0.014 ±0.007 | — | 552 ±0 |
| greedy_anchors | 3 | 0.978 ±0.038 | 0.797 ±0.178 | 0.162 ±0.255 | 0.203 ±0.178 | — | 83,055 ±19,382 |
| sp_anchors | 3 | 0.978 ±0.038 | 0.797 ±0.178 | 0.162 ±0.255 | 0.203 ±0.178 | — | 83,055 ±19,382 |
| random_search | 3 | 1.000 ±0.000 | 0.022 ±0.026 | 0.000 ±0.000 | 0.978 ±0.026 | — | 138 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.935 / 0.969 / — | 1.000 / 0.935 / — | 0.000 / 0.000 / — | 0.000 / 0.065 / — | 0.113 / 0.067 / — | 3,726 / 3,726 / — |
| RLDA | 0.989 / 0.702 / 1.000 | 0.630 / 0.899 / 0.580 | 0.000 / 0.029 / 0.000 | 0.370 / 0.101 / 0.420 | 0.100 / 0.120 / 0.140 | 21,578 / 21,578 / 21,578 |
| CART | 0.934 / 0.948 / 0.853 | 0.993 / 0.978 / 0.986 | 0.000 / 0.000 / 0.000 | 0.007 / 0.022 / 0.014 | — | 552 / 552 / 552 |
| greedy_anchors | 0.934 / 1.000 / 1.000 | 0.993 / 0.754 / 0.645 | 0.457 / 0.022 / 0.007 | 0.007 / 0.246 / 0.355 | — | 63,141 / 101,858 / 84,166 |
| sp_anchors | 0.934 / 1.000 / 1.000 | 0.993 / 0.754 / 0.645 | 0.457 / 0.022 / 0.007 | 0.007 / 0.246 / 0.355 | — | 63,141 / 101,858 / 84,166 |
| random_search | — / 1.000 / 1.000 | 0.000 / 0.051 / 0.014 | 0.000 / 0.000 / 0.000 | 1.000 / 0.949 / 0.986 | — | 138 / 138 / 138 |


### uci_adult

**DNN black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 3 | 0.873 ±0.005 | 0.924 ±0.058 | 0.069 ±0.040 | 0.076 ±0.058 | 0.049 ±0.010 | 263,742 ±0 |
| RLDA | 3 | 0.872 ±0.029 | 0.712 ±0.036 | 0.024 ±0.026 | 0.288 ±0.036 | 0.067 ±0.064 | 1,523,858 ±0 |
| CART | 3 | 0.881 ±0.071 | 0.833 ±0.145 | 0.000 ±0.000 | 0.167 ±0.145 | — | 39,073 ±0 |
| greedy_anchors | 3 | 0.946 ±0.011 | 0.525 ±0.045 | 0.000 ±0.000 | 0.475 ±0.045 | — | 150,193 ±72,828 |
| sp_anchors | 3 | 0.905 ±0.048 | 0.700 ±0.205 | 0.037 ±0.065 | 0.300 ±0.205 | — | 150,193 ±72,828 |
| random_search | 3 | 0.976 ±0.022 | 0.034 ±0.019 | 0.000 ±0.000 | 0.966 ±0.019 | — | 9,769 ±0 |

_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.877 / 0.874 / 0.867 | 0.894 / 0.991 / 0.887 | 0.080 / 0.103 / 0.025 | 0.106 / 0.009 / 0.113 | 0.047 / 0.040 / 0.060 | 263,742 / 263,742 / 263,742 |
| RLDA | 0.871 / 0.843 / 0.901 | 0.746 / 0.717 / 0.674 | 0.054 / 0.004 / 0.014 | 0.254 / 0.283 / 0.326 | 0.020 / 0.040 / 0.140 | 1,523,858 / 1,523,858 / 1,523,858 |
| CART | 0.800 / 0.910 / 0.932 | 1.000 / 0.747 / 0.751 | 0.000 / 0.000 / 0.000 | 0.000 / 0.253 / 0.249 | — | 39,073 / 39,073 / 39,073 |
| greedy_anchors | 0.936 / 0.943 / 0.958 | 0.573 / 0.483 / 0.520 | 0.000 / 0.000 / 0.000 | 0.427 / 0.517 / 0.480 | — | 66,155 / 189,547 / 194,877 |
| sp_anchors | 0.936 / 0.850 / 0.930 | 0.573 / 0.937 / 0.591 | 0.000 / 0.112 / 0.000 | 0.427 / 0.063 / 0.409 | — | 66,155 / 189,547 / 194,877 |
| random_search | 0.996 / 0.979 / 0.952 | 0.029 / 0.019 / 0.055 | 0.000 / 0.000 / 0.000 | 0.971 / 0.981 / 0.945 | — | 9,769 / 9,769 / 9,769 |

**RandomForest black box** — mean ±sd over completed seeds

| method | seeds | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|---|
| MADA | 2 | 0.919 ±0.032 | 0.936 ±0.091 | 0.081 ±0.003 | 0.064 ±0.091 | 0.057 ±0.024 | 263,742 ±0 |
| RLDA | 3 | 0.909 ±0.020 | 0.794 ±0.102 | 0.058 ±0.005 | 0.206 ±0.102 | 0.093 ±0.070 | 1,523,858 ±0 |
| CART | 3 | 0.845 ±0.002 | 1.000 ±0.000 | 0.000 ±0.000 | 0.000 ±0.000 | — | 39,073 ±0 |
| greedy_anchors | 3 | 0.953 ±0.004 | 0.820 ±0.026 | 0.002 ±0.004 | 0.180 ±0.026 | — | 114,989 ±34,860 |
| sp_anchors | 3 | 0.923 ±0.013 | 1.000 ±0.000 | 0.110 ±0.018 | 0.000 ±0.000 | — | 114,989 ±34,860 |
| random_search | 3 | 0.982 ±0.013 | 0.037 ±0.011 | 0.000 ±0.000 | 0.963 ±0.011 | — | 9,769 ±0 |

> ⚠️ **Unequal seeds in this table** (MADA n=2, RLDA n=3). The arms are averaged
> over different runs, so the MADA-vs-RLDA means here are not
> like-for-like. This pair is excluded from the pooled head-to-head.


_per seed (42 / 43 / 44):_

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.896 / 0.941 / — | 1.000 / 0.872 / — | 0.079 / 0.083 / — | 0.000 / 0.128 / — | 0.040 / 0.073 / — | 263,742 / 263,742 / — |
| RLDA | 0.917 / 0.886 / 0.923 | 0.743 / 0.912 / 0.728 | 0.054 / 0.063 / 0.058 | 0.257 / 0.088 / 0.272 | 0.020 / 0.100 / 0.160 | 1,523,858 / 1,523,858 / 1,523,858 |
| CART | 0.844 / 0.847 / 0.845 | 1.000 / 1.000 / 1.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | — | 39,073 / 39,073 / 39,073 |
| greedy_anchors | 0.955 / 0.956 / 0.948 | 0.791 / 0.829 / 0.840 | 0.000 / 0.006 / 0.000 | 0.209 / 0.171 / 0.160 | — | 102,564 / 88,044 / 154,360 |
| sp_anchors | 0.932 / 0.930 / 0.908 | 1.000 / 1.000 / 1.000 | 0.117 / 0.123 / 0.089 | 0.000 / 0.000 / 0.000 | — | 102,564 / 88,044 / 154,360 |
| random_search | 0.973 / 0.997 / 0.975 | 0.049 / 0.033 / 0.029 | 0.000 / 0.000 / 0.000 | 0.951 / 0.967 / 0.971 | — | 9,769 / 9,769 / 9,769 |


---

## MADA vs RLDA head-to-head

Counted over datasets where both arms finished. `W` = MADA better,
`L` = RLDA better, `T` = tie to three decimals.

### Pooled over all (dataset, seed) pairs

The row to quote: one count per black box over every dataset-seed pair where
both arms finished. Less seed-sensitive than the per-seed rows below.

| black box | pairs | Fid | Cov | Conflict | Abstain |
|---|---|---|---|---|---|
| DNN | 21 | 13W/7L/1T | 13W/6L/2T | 8W/12L/1T | 13W/6L/2T |
| RandomForest | 14 | 4W/10L | 12W/2L | 3W/9L/2T | 12W/2L |

### By seed

| black box | seed | Fid | Cov | Conflict | Abstain |
|---|---|---|---|---|---|
| DNN | 42 | 5W/2L | 5W/1L/1T | 2W/5L | 5W/1L/1T |
| DNN | 43 | 3W/3L/1T | 4W/2L/1T | 3W/4L | 4W/2L/1T |
| DNN | 44 | 5W/2L | 4W/3L | 3W/3L/1T | 4W/3L |
| RandomForest | 42 | 1W/6L | 6W/1L | 0W/5L/2T | 6W/1L |
| RandomForest | 43 | 3W/4L | 6W/1L | 3W/4L | 6W/1L |
| RandomForest | 44 | — | — | — | — |

---

## Query accounting

`extraction queries` counts black-box calls made to **build** the rule set:
rule generation plus validation-split selection. It excludes policy training,
which is reported separately below, and excludes held-out test reporting,
which is instrumentation rather than a cost of producing an explanation.

Serving an explanation from an already-extracted box costs **0** queries in
both RL arms — that is the amortisation claim the paper makes.

### Training queries (RL arms only)

| black box | dataset | MADA s42 | RLDA s42 | MADA s43 | RLDA s43 | MADA s44 | RLDA s44 |
|---|---|---|---|---|---|---|---|
| DNN | iris | 120 | 210 | 120 | 360 | 120 | 360 |
| DNN | wine | 142 | 248 | 142 | 426 | 142 | 426 |
| DNN | breast_cancer | 455 | 796 | 455 | 910 | 455 | 910 |
| DNN | synthetic | 800 | 1,400 | 800 | 1,600 | 800 | 1,600 |
| DNN | housing | 16,512 | 28,896 | 16,512 | 66,048 | 16,512 | 66,048 |
| DNN | uci_credit | 552 | 966 | 552 | 1,104 | 552 | 1,104 |
| DNN | uci_adult | 39,073 | 68,377 | 39,073 | 78,146 | 39,073 | 78,146 |
| RandomForest | iris | 120 | 210 | 120 | 360 | — | 360 |
| RandomForest | wine | 142 | 248 | 142 | 426 | — | 426 |
| RandomForest | breast_cancer | 455 | 796 | 455 | 910 | — | 910 |
| RandomForest | synthetic | 800 | 1,400 | 800 | 1,600 | — | 1,600 |
| RandomForest | housing | 16,512 | 28,896 | 16,512 | 66,048 | — | 66,048 |
| RandomForest | uci_credit | 552 | 966 | 552 | 1,104 | — | 1,104 |
| RandomForest | uci_adult | 39,073 | 68,377 | 39,073 | 78,146 | — | 78,146 |

> **Caveat.** MADA's training-query count is a lower bound: the torchrl
> collector holds copies of the environment whose cache hits are not all
> attributed back to the parent counter. RLDA's count is exact.

> **Open item.** RLDA reports roughly 5x MADA's extraction queries on every
> dataset, independent of class count, and `uci_adult` is far above CART for
> both arms. This ratio is not yet explained and should not be quoted as a
> headline cost result until it is.

