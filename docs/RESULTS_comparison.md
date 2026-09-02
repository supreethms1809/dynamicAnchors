# Results: MADA vs RLDA vs Baselines

Generated 2026-09-02 17:12. Branch `ma-training-config-bump`.

Tables are grouped **by dataset**, with the DNN and RandomForest black boxes
adjacent, so the effect of swapping the explained model is read top-to-bottom
within one dataset.

Every metric cell reads **seed 42 / seed 43**, followed by the mean in parentheses
once more than one seed is present. An em dash marks a seed that has not been
run yet — it is excluded from the mean rather than counted as zero.

## What was run

| | |
|---|---|
| Arms | MADA (MADDPG), RLDA (DDPG), each against two black boxes |
| Black boxes | **DNN** (`runs/sweep_dnn/`) and **RandomForest** (`runs/sweep_rf/`) |
| Datasets | iris, wine, breast_cancer, synthetic, housing, uci_credit, uci_adult |
| Excluded | covtype, folktables (too large for the time budget) |
| Seeds | 42, 43 |
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
| RandomForest | 42 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RandomForest | 43 | — | ◐ | ◐ | — | — | — | — |

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

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.958 / 0.862 (0.910) | 0.800 / 0.967 (0.883) | 0.000 / 0.000 (0.000) | 0.200 / 0.033 (0.117) | 0.122 (22/180) / 0.200 (36/180) | 1,170 / 1,170 |
| RLDA | 0.967 / 0.862 (0.914) | 1.000 / 0.967 (0.983) | 0.567 / 0.133 (0.350) | 0.000 / 0.033 (0.017) | 0.400 (24/60) / 0.333 (20/60) | 5,730 / 5,730 |
| CART | 0.964 / 0.862 (0.913) | 0.933 / 0.967 (0.950) | 0.000 / 0.000 (0.000) | 0.067 / 0.033 (0.050) | — | 120 / 120 |
| greedy_anchors | 1.000 / 1.000 (1.000) | 0.567 / 0.600 (0.583) | 0.000 / 0.000 (0.000) | 0.433 / 0.400 (0.417) | — | 65,614 / 83,451 |
| sp_anchors | 1.000 / 0.833 (0.917) | 0.567 / 0.800 (0.683) | 0.033 / 0.033 (0.033) | 0.433 / 0.200 (0.317) | — | 65,614 / 83,451 |
| random_search | 0.864 / 0.923 (0.893) | 0.733 / 0.867 (0.800) | 0.133 / 0.067 (0.100) | 0.267 / 0.133 (0.200) | — | 30 / 30 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.964 / — | 0.933 / — | 0.300 / — | 0.067 / — | 0.178 (32/180) / — | 1,170 / — |
| RLDA | 0.960 / — | 0.833 / — | 0.100 / — | 0.167 / — | 0.350 (21/60) / — | 5,730 / — |
| CART | 1.000 / — | 0.933 / — | 0.000 / — | 0.067 / — | — | 120 / — |
| greedy_anchors | 0.933 / — | 0.500 / — | 0.000 / — | 0.500 / — | — | 100,349 / — |
| sp_anchors | 0.933 / — | 0.500 / — | 0.033 / — | 0.500 / — | — | 100,349 / — |
| random_search | 0.864 / — | 0.733 / — | 0.133 / — | 0.267 / — | — | 30 / — |


### wine

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.962 / 0.800 (0.881) | 0.722 / 0.694 (0.708) | 0.306 / 0.056 (0.181) | 0.278 / 0.306 (0.292) | 0.150 (27/180) / 0.161 (29/180) | 1,384 / 1,384 |
| RLDA | 0.923 / 0.667 (0.795) | 0.722 / 0.833 (0.778) | 0.139 / 0.389 (0.264) | 0.278 / 0.167 (0.222) | 0.250 (15/60) / 0.217 (13/60) | 6,738 / 6,738 |
| CART | 0.920 / 0.806 (0.863) | 0.694 / 0.861 (0.778) | 0.000 / 0.000 (0.000) | 0.306 / 0.139 (0.222) | — | 142 / 142 |
| greedy_anchors | 1.000 / 0.955 (0.977) | 0.389 / 0.611 (0.500) | 0.000 / 0.000 (0.000) | 0.611 / 0.389 (0.500) | — | 336,401 / 316,057 |
| sp_anchors | 1.000 / 0.955 (0.977) | 0.389 / 0.611 (0.500) | 0.000 / 0.000 (0.000) | 0.611 / 0.389 (0.500) | — | 336,401 / 316,057 |
| random_search | 1.000 / — | 0.028 / 0.000 (0.014) | 0.000 / 0.000 (0.000) | 0.972 / 1.000 (0.986) | — | 36 / 36 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.657 / — | 0.972 / — | 0.333 / — | 0.028 / — | 0.128 (23/180) / — | 1,384 / — |
| RLDA | 0.968 / 0.792 (0.880) | 0.861 / 0.667 (0.764) | 0.111 / 0.167 (0.139) | 0.139 / 0.333 (0.236) | 0.367 (22/60) / 0.167 (10/60) | 6,738 / 6,738 |
| CART | 0.920 / 0.871 (0.895) | 0.694 / 0.861 (0.778) | 0.000 / 0.000 (0.000) | 0.306 / 0.139 (0.222) | — | 142 / 142 |
| greedy_anchors | 1.000 / 0.938 (0.969) | 0.556 / 0.444 (0.500) | 0.000 / 0.028 (0.014) | 0.444 / 0.556 (0.500) | — | 481,231 / 491,526 |
| sp_anchors | 1.000 / 0.938 (0.969) | 0.556 / 0.444 (0.500) | 0.000 / 0.028 (0.014) | 0.444 / 0.556 (0.500) | — | 481,231 / 491,526 |
| random_search | — / — | 0.000 / 0.000 (0.000) | 0.000 / 0.000 (0.000) | 1.000 / 1.000 (1.000) | — | 36 / 36 |


### breast_cancer

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 / 0.886 (0.908) | 0.877 / 1.000 (0.939) | 0.053 / 0.035 (0.044) | 0.123 / 0.000 (0.061) | 0.133 (16/120) / 0.117 (14/120) | 3,071 / 3,071 |
| RLDA | 0.918 / 0.911 (0.915) | 0.746 / 0.693 (0.719) | 0.000 / 0.000 (0.000) | 0.254 / 0.307 (0.281) | 0.250 (10/40) / 0.250 (10/40) | 14,362 / 14,362 |
| CART | 0.908 / 0.921 (0.915) | 0.956 / 0.886 (0.921) | 0.000 / 0.000 (0.000) | 0.044 / 0.114 (0.079) | — | 455 / 455 |
| greedy_anchors | 1.000 / 1.000 (1.000) | 0.526 / 0.333 (0.430) | 0.000 / 0.000 (0.000) | 0.474 / 0.667 (0.570) | — | 794,676 / 683,073 |
| sp_anchors | 1.000 / 1.000 (1.000) | 0.526 / 0.333 (0.430) | 0.000 / 0.000 (0.000) | 0.474 / 0.667 (0.570) | — | 794,676 / 683,073 |
| random_search | 1.000 / 1.000 (1.000) | 0.035 / 0.035 (0.035) | 0.000 / 0.000 (0.000) | 0.965 / 0.965 (0.965) | — | 114 / 114 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.920 / — | 0.877 / — | 0.035 / — | 0.123 / — | 0.158 (19/120) / — | 3,071 / — |
| RLDA | 0.952 / 0.915 (0.933) | 0.912 / 0.825 (0.868) | 0.000 / 0.000 (0.000) | 0.088 / 0.175 (0.132) | 0.100 (4/40) / 0.200 (8/40) | 14,362 / 14,362 |
| CART | 0.908 / 0.970 (0.939) | 0.956 / 0.886 (0.921) | 0.000 / 0.000 (0.000) | 0.044 / 0.114 (0.079) | — | 455 / 455 |
| greedy_anchors | 1.000 / — | 0.711 / — | 0.000 / — | 0.289 / — | — | 908,911 / — |
| sp_anchors | 1.000 / — | 0.711 / — | 0.000 / — | 0.289 / — | — | 908,911 / — |
| random_search | 1.000 / 1.000 (1.000) | 0.026 / 0.070 (0.048) | 0.000 / 0.000 (0.000) | 0.974 / 0.930 (0.952) | — | 114 / 114 |


### synthetic

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.895 / 0.903 (0.899) | 0.905 / 0.770 (0.838) | 0.415 / 0.100 (0.258) | 0.095 / 0.230 (0.163) | 0.075 (9/120) / 0.108 (13/120) | 5,400 / 5,400 |
| RLDA | 0.907 / 0.959 (0.933) | 0.645 / 0.370 (0.508) | 0.050 / 0.000 (0.025) | 0.355 / 0.630 (0.492) | 0.250 (10/40) / 0.425 (17/40) | 25,240 / 25,240 |
| CART | 0.705 / 0.653 (0.679) | 0.950 / 0.980 (0.965) | 0.000 / 0.000 (0.000) | 0.050 / 0.020 (0.035) | — | 800 / 800 |
| greedy_anchors | 0.914 / 0.969 (0.941) | 0.465 / 0.320 (0.392) | 0.000 / 0.000 (0.000) | 0.535 / 0.680 (0.608) | — | 95,493 / 93,630 |
| sp_anchors | 0.914 / 0.969 (0.941) | 0.465 / 0.320 (0.392) | 0.000 / 0.000 (0.000) | 0.535 / 0.680 (0.608) | — | 95,493 / 93,630 |
| random_search | 1.000 / 0.929 (0.964) | 0.100 / 0.070 (0.085) | 0.000 / 0.000 (0.000) | 0.900 / 0.930 (0.915) | — | 200 / 200 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.890 / — | 0.865 / — | 0.305 / — | 0.135 / — | 0.150 (18/120) / — | 5,400 / — |
| RLDA | 1.000 / — | 0.475 / — | 0.000 / — | 0.525 / — | 0.525 (21/40) / — | 25,240 / — |
| CART | 0.716 / — | 0.950 / — | 0.000 / — | 0.050 / — | — | 800 / — |
| greedy_anchors | 0.955 / — | 0.440 / — | 0.000 / — | 0.560 / — | — | 49,689 / — |
| sp_anchors | 0.955 / — | 0.440 / — | 0.000 / — | 0.560 / — | — | 49,689 / — |
| random_search | 0.810 / — | 0.105 / — | 0.015 / — | 0.895 / — | — | 200 / — |


### housing

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.569 / 0.532 (0.551) | 0.740 / 0.713 (0.727) | 0.080 / 0.013 (0.047) | 0.260 / 0.287 (0.273) | 0.004 (1/240) / 0.017 (4/240) | 210,528 / 210,528 |
| RLDA | 0.515 / 0.474 (0.494) | 0.646 / 0.930 (0.788) | 0.100 / 0.399 (0.250) | 0.354 / 0.070 (0.212) | 0.037 (3/80) / 0.013 (1/80) | 1,040,336 / 1,040,336 |
| CART | 0.563 / 0.512 (0.537) | 0.998 / 1.000 (0.999) | 0.000 / 0.000 (0.000) | 0.002 / 0.000 (0.001) | — | 16,512 / 16,512 |
| greedy_anchors | 0.681 / 0.427 (0.554) | 0.238 / 0.516 (0.377) | 0.000 / 0.080 (0.040) | 0.762 / 0.484 (0.623) | — | 482,158 / 525,651 |
| sp_anchors | 0.681 / 0.438 (0.559) | 0.238 / 0.535 (0.386) | 0.000 / 0.100 (0.050) | 0.762 / 0.465 (0.614) | — | 482,158 / 525,651 |
| random_search | 0.681 / 0.768 (0.724) | 0.169 / 0.074 (0.122) | 0.000 / 0.000 (0.000) | 0.831 / 0.926 (0.878) | — | 4,128 / 4,128 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.902 / — | 0.246 / — | 0.001 / — | 0.754 / — | 0.008 (2/240) / — | 210,528 / — |
| RLDA | 0.912 / — | 0.193 / — | 0.001 / — | 0.807 / — | 0.100 (8/80) / — | 1,040,336 / — |
| CART | 0.578 / — | 0.888 / — | 0.000 / — | 0.112 / — | — | 16,512 / — |
| greedy_anchors | 0.685 / — | 0.225 / — | 0.004 / — | 0.775 / — | — | 370,950 / — |
| sp_anchors | 0.692 / — | 0.271 / — | 0.000 / — | 0.729 / — | — | 370,950 / — |
| random_search | 0.825 / — | 0.101 / — | 0.000 / — | 0.899 / — | — | 4,128 / — |


### uci_credit

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.930 / 0.899 (0.914) | 0.935 / 1.000 (0.967) | 0.043 / 0.493 (0.268) | 0.065 / 0.000 (0.033) | 0.120 (18/150) / 0.013 (2/150) | 3,726 / 3,726 |
| RLDA | 0.926 / 0.935 (0.931) | 0.884 / 0.891 (0.888) | 0.000 / 0.000 (0.000) | 0.116 / 0.109 (0.112) | 0.060 (3/50) / 0.040 (2/50) | 21,578 / 21,578 |
| CART | 0.905 / 0.919 (0.912) | 0.993 / 0.978 (0.986) | 0.000 / 0.000 (0.000) | 0.007 / 0.022 (0.014) | — | 552 / 552 |
| greedy_anchors | 0.980 / 0.919 (0.949) | 0.710 / 0.978 (0.844) | 0.007 / 0.471 (0.239) | 0.290 / 0.022 (0.156) | — | 36,117 / 27,732 |
| sp_anchors | 0.891 / 0.919 (0.905) | 0.993 / 0.978 (0.986) | 0.442 / 0.471 (0.457) | 0.007 / 0.022 (0.014) | — | 36,117 / 27,732 |
| random_search | — / 1.000 | 0.000 / 0.058 (0.029) | 0.000 / 0.000 (0.000) | 1.000 / 0.942 (0.971) | — | 138 / 138 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.935 / — | 1.000 / — | 0.000 / — | 0.000 / — | 0.113 (17/150) / — | 3,726 / — |
| RLDA | 0.989 / — | 0.630 / — | 0.000 / — | 0.370 / — | 0.100 (5/50) / — | 21,578 / — |
| CART | 0.934 / — | 0.993 / — | 0.000 / — | 0.007 / — | — | 552 / — |
| greedy_anchors | 0.934 / — | 0.993 / — | 0.457 / — | 0.007 / — | — | 63,141 / — |
| sp_anchors | 0.934 / — | 0.993 / — | 0.457 / — | 0.007 / — | — | 63,141 / — |
| random_search | — / — | 0.000 / — | 0.000 / — | 1.000 / — | — | 138 / — |


### uci_adult

**DNN black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.877 / 0.874 (0.876) | 0.894 / 0.991 (0.942) | 0.080 / 0.103 (0.091) | 0.106 / 0.009 (0.058) | 0.047 (7/150) / 0.040 (6/150) | 263,742 / 263,742 |
| RLDA | 0.871 / 0.843 (0.857) | 0.746 / 0.717 (0.732) | 0.054 / 0.004 (0.029) | 0.254 / 0.283 (0.268) | 0.020 (1/50) / 0.040 (2/50) | 1,523,858 / 1,523,858 |
| CART | 0.800 / 0.910 (0.855) | 1.000 / 0.747 (0.873) | 0.000 / 0.000 (0.000) | 0.000 / 0.253 (0.127) | — | 39,073 / 39,073 |
| greedy_anchors | 0.936 / 0.943 (0.939) | 0.573 / 0.483 (0.528) | 0.000 / 0.000 (0.000) | 0.427 / 0.517 (0.472) | — | 66,155 / 189,547 |
| sp_anchors | 0.936 / 0.850 (0.893) | 0.573 / 0.937 (0.755) | 0.000 / 0.112 (0.056) | 0.427 / 0.063 (0.245) | — | 66,155 / 189,547 |
| random_search | 0.996 / 0.979 (0.988) | 0.029 / 0.019 (0.024) | 0.000 / 0.000 (0.000) | 0.971 / 0.981 (0.976) | — | 9,769 / 9,769 |

**RandomForest black box**

| method | Fid | Cov | Conflict | Abstain | Success | extraction queries |
|---|---|---|---|---|---|---|
| MADA | 0.896 / — | 1.000 / — | 0.079 / — | 0.000 / — | 0.040 (6/150) / — | 263,742 / — |
| RLDA | 0.917 / — | 0.743 / — | 0.054 / — | 0.257 / — | 0.020 (1/50) / — | 1,523,858 / — |
| CART | 0.844 / — | 1.000 / — | 0.000 / — | 0.000 / — | — | 39,073 / — |
| greedy_anchors | 0.955 / — | 0.791 / — | 0.000 / — | 0.209 / — | — | 102,564 / — |
| sp_anchors | 0.932 / — | 1.000 / — | 0.117 / — | 0.000 / — | — | 102,564 / — |
| random_search | 0.973 / — | 0.049 / — | 0.000 / — | 0.951 / — | — | 9,769 / — |


---

## MADA vs RLDA head-to-head

Counted over datasets where both arms finished. `W` = MADA better,
`L` = RLDA better, `T` = tie to three decimals.

| black box | seed | Fid | Cov | Conflict | Abstain |
|---|---|---|---|---|---|
| DNN | 42 | 5W/2L | 5W/1L/1T | 2W/5L | 5W/1L/1T |
| DNN | 43 | 3W/3L/1T | 4W/2L/1T | 3W/4L | 4W/2L/1T |
| RandomForest | 42 | 1W/6L | 6W/1L | 0W/5L/2T | 6W/1L |
| RandomForest | 43 | — | — | — | — |

---

## Query accounting

`extraction queries` counts black-box calls made to **build** the rule set:
rule generation plus validation-split selection. It excludes policy training,
which is reported separately below, and excludes held-out test reporting,
which is instrumentation rather than a cost of producing an explanation.

Serving an explanation from an already-extracted box costs **0** queries in
both RL arms — that is the amortisation claim the paper makes.

### Training queries (RL arms only)

| black box | dataset | MADA s42 | RLDA s42 | MADA s43 | RLDA s43 |
|---|---|---|---|---|---|
| DNN | iris | 120 | 210 | 120 | 360 |
| DNN | wine | 142 | 248 | 142 | 426 |
| DNN | breast_cancer | 455 | 796 | 455 | 910 |
| DNN | synthetic | 800 | 1,400 | 800 | 1,600 |
| DNN | housing | 16,512 | 28,896 | 16,512 | 66,048 |
| DNN | uci_credit | 552 | 966 | 552 | 1,104 |
| DNN | uci_adult | 39,073 | 68,377 | 39,073 | 78,146 |
| RandomForest | iris | 120 | 210 | — | — |
| RandomForest | wine | 142 | 248 | — | 426 |
| RandomForest | breast_cancer | 455 | 796 | — | 910 |
| RandomForest | synthetic | 800 | 1,400 | — | — |
| RandomForest | housing | 16,512 | 28,896 | — | — |
| RandomForest | uci_credit | 552 | 966 | — | — |
| RandomForest | uci_adult | 39,073 | 68,377 | — | — |

> **Caveat.** MADA's training-query count is a lower bound: the torchrl
> collector holds copies of the environment whose cache hits are not all
> attributed back to the parent counter. RLDA's count is exact.

> **Open item.** RLDA reports roughly 5x MADA's extraction queries on every
> dataset, independent of class count, and `uci_adult` is far above CART for
> both arms. This ratio is not yet explained and should not be quoted as a
> headline cost result until it is.

