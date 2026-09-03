# Extracted Rules

Generated 2026-09-03 17:18. Branch `ma-training-config-bump`.

The actual rule sets behind the numbers in `RESULTS_comparison.md`.
Rules are the **validation-selected union**, scored on the **test** split.
`k` is how many rules the marginal-gain selector kept for that class.

Seeds are listed adjacently under each class so a rule's stability across
seeds can be read directly. A class that changes its selected feature between
seeds is telling you something about the policy, not about the dataset.

Rule strings are printed in full — never truncated — because a truncated box
is a different box, and the repository has been bitten by that before.

Fid/Cov below are **per rule** on the test split; the class union's own
Fid/Cov are in the comparison document, and the union is not the average of
its members.

---

## iris

### iris — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) ∈ [1.300000, 6.900000] and petal width (cm) ∈ [0.100000, 0.406667]  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [4.300000, 5.520007] and petal length (cm) ∈ [1.100000, 1.706667]  — Fid 1.000, Cov 1.000, n=10
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [4.490000, 7.900000] and petal width (cm) ∈ [0.100000, 0.406667]  — Fid 1.000, Cov 0.800, n=8

- `class_1`
  - seed 42 (k=1)
    1. sepal length (cm) ∈ [5.096666, 7.900000] and petal length (cm) ∈ [3.590000, 6.900000] and petal width (cm) ∈ [0.100000, 1.500000]  — Fid 0.857, Cov 0.600, n=7
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [4.300000, 6.710988] and petal length (cm) ∈ [3.290000, 4.720357]  — Fid 0.600, Cov 1.000, n=10
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [4.300000, 6.420022] and sepal width (cm) ∈ [2.400000, 3.203333] and petal width (cm) ∈ [1.000000, 2.500000]  — Fid 0.800, Cov 0.800, n=10

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) ∈ [4.790000, 6.900000] and petal width (cm) ∈ [1.793333, 2.500000]  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [5.576667, 7.900000] and petal width (cm) ∈ [1.789987, 2.500000]  — Fid 1.000, Cov 0.900, n=9
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [5.989960, 7.900000] and petal width (cm) ∈ [0.100000, 2.406667]  — Fid 0.692, Cov 0.900, n=13


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.488068]  — Fid 1.000, Cov 0.900, n=9
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [4.686482, 7.900000] and sepal width (cm) ∈ [3.137640, 4.400000] and petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 0.500, n=5

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.000000, 1.500000]  — Fid 0.889, Cov 0.800, n=9
  - seed 43 (k=1)
    1. sepal width (cm) ∈ [2.200000, 3.110024] and petal length (cm) ∈ [3.479993, 4.720000]  — Fid 0.750, Cov 0.800, n=8
  - seed 44 (k=1)
    1. petal length (cm) ∈ [1.100000, 4.913185] and petal width (cm) ∈ [1.000000, 2.500000]  — Fid 1.000, Cov 1.000, n=10

- `class_2`
  - seed 42 (k=1)
    1. sepal width (cm) ∈ [2.500000, 3.900000]  — Fid 0.429, Cov 1.000, n=28
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [5.800000, 7.900000]  — Fid 0.812, Cov 1.000, n=16
  - seed 44 (k=1)
    1. sepal width (cm) ∈ [2.000000, 3.344453] and petal width (cm) ∈ [0.100000, 2.313334]  — Fid 0.360, Cov 0.900, n=25


**cart**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 1.000, n=10
  - seed 44 (k=1)
    1. petal width (cm) <= 0.800000  — Fid 1.000, Cov 0.900, n=9

- `class_1`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) <= 1.550000  — Fid 0.889, Cov 0.800, n=9
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal length (cm) <= 4.650000  — Fid 0.625, Cov 0.800, n=8
  - seed 44 (k=1)
    1. petal width (cm) > 0.800000 and petal length (cm) <= 4.750000  — Fid 1.000, Cov 1.000, n=10

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) > 1.550000  — Fid 1.000, Cov 0.900, n=11
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal length (cm) > 4.650000  — Fid 0.909, Cov 0.900, n=11
  - seed 44 (k=1)
    1. petal width (cm) > 0.800000 and petal length (cm) > 4.750000  — Fid 1.000, Cov 0.900, n=9


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 1.60 and petal width (cm) <= 0.30  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.75 and sepal width (cm) > 2.80  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. sepal width (cm) > 3.38 and petal length (cm) <= 4.45 and sepal length (cm) <= 5.85  — Fid 1.000, Cov 0.300, n=3

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.30 and petal length (cm) > 1.60 and sepal width (cm) <= 2.73  — Fid 1.000, Cov 0.300, n=3
  - seed 43 (k=1)
    1. petal width (cm) <= 1.30 and petal length (cm) > 1.60 and sepal width (cm) <= 2.80  — Fid 1.000, Cov 0.400, n=4
  - seed 44 (k=1)
    1. 0.30 < petal width (cm) <= 1.35 and sepal length (cm) > 5.10 and sepal width (cm) <= 3.00  — Fid 1.000, Cov 0.600, n=6

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) > 5.10 and petal width (cm) > 1.30  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. petal width (cm) > 1.35 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 1.60 and petal width (cm) <= 0.30  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.75 and sepal width (cm) > 2.80  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. sepal width (cm) > 3.38 and petal length (cm) <= 4.45 and sepal length (cm) <= 5.85  — Fid 1.000, Cov 0.300, n=3

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.80 and 1.60 < petal length (cm) <= 4.25 and sepal width (cm) <= 3.00  — Fid 0.750, Cov 0.300, n=4
  - seed 43 (k=1)
    1. petal width (cm) <= 1.87 and sepal width (cm) <= 3.00 and 1.60 < petal length (cm) <= 5.10 and 5.10 < sepal length (cm) <= 6.48  — Fid 0.545, Cov 0.800, n=11
  - seed 44 (k=1)
    1. 0.30 < petal width (cm) <= 1.35 and sepal length (cm) > 5.10 and sepal width (cm) <= 3.00  — Fid 1.000, Cov 0.600, n=6

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) > 5.10 and petal width (cm) > 1.30  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. petal width (cm) > 1.35 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 111  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. random box 100  — Fid 1.000, Cov 0.900, n=9
  - seed 44 (k=1)
    1. random box 10  — Fid 1.000, Cov 0.600, n=6

- `class_1`
  - seed 42 (k=1)
    1. random box 57  — Fid 0.571, Cov 0.400, n=7
  - seed 43 (k=1)
    1. random box 164  — Fid 0.556, Cov 0.800, n=9
  - seed 44 (k=1)
    1. random box 116  — Fid 1.000, Cov 0.900, n=9

- `class_2`
  - seed 42 (k=1)
    1. random box 51  — Fid 0.778, Cov 0.500, n=9
  - seed 43 (k=1)
    1. random box 235  — Fid 1.000, Cov 0.800, n=10
  - seed 44 (k=1)
    1. random box 201  — Fid 0.750, Cov 0.300, n=4


### iris — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) ∈ [1.293334, 6.900000] and petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) ∈ [1.100000, 1.706667] and petal width (cm) ∈ [0.100000, 0.406667]  — Fid 1.000, Cov 0.900, n=9
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.000000, 2.500000]  — Fid 0.500, Cov 1.000, n=20
  - seed 43 (k=1)
    1. petal length (cm) ∈ [3.479986, 4.813334] and petal width (cm) ∈ [1.000000, 2.500000]  — Fid 1.000, Cov 1.000, n=10
  - seed 44: _not run_

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.800000, 2.500000]  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. petal length (cm) ∈ [4.886666, 6.700000] and petal width (cm) ∈ [1.790000, 2.500000]  — Fid 0.778, Cov 0.900, n=9
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.488068]  — Fid 1.000, Cov 0.900, n=9
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [4.686482, 7.900000] and sepal width (cm) ∈ [3.137640, 4.400000] and petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 0.500, n=5

- `class_1`
  - seed 42 (k=1)
    1. sepal length (cm) ∈ [5.464249, 7.900000] and petal width (cm) ∈ [0.100000, 1.500000]  — Fid 0.750, Cov 0.500, n=8
  - seed 43 (k=1)
    1. petal length (cm) ∈ [3.147385, 4.800000]  — Fid 1.000, Cov 1.000, n=10
  - seed 44 (k=1)
    1. petal length (cm) ∈ [3.530310, 6.700000] and petal width (cm) ∈ [0.100000, 1.606667]  — Fid 1.000, Cov 0.900, n=10

- `class_2`
  - seed 42 (k=1)
    1. sepal width (cm) ∈ [2.500000, 3.613333] and petal length (cm) ∈ [4.894028, 6.900000]  — Fid 0.900, Cov 0.900, n=10
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [5.486940, 7.900000] and petal length (cm) ∈ [4.546051, 6.100000] and petal width (cm) ∈ [0.100000, 2.310000]  — Fid 0.375, Cov 0.600, n=8
  - seed 44 (k=1)
    1. sepal length (cm) ∈ [6.117279, 7.900000] and sepal width (cm) ∈ [2.700000, 4.400000]  — Fid 0.875, Cov 0.700, n=8


**cart**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 1.000, n=10
  - seed 44 (k=1)
    1. petal width (cm) <= 0.800000  — Fid 1.000, Cov 0.900, n=9

- `class_1`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) <= 1.650000  — Fid 1.000, Cov 0.900, n=10
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) <= 1.550000  — Fid 1.000, Cov 0.800, n=9
  - seed 44 (k=1)
    1. petal width (cm) > 0.800000 and petal width (cm) <= 1.750000  — Fid 1.000, Cov 1.000, n=11

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) > 1.650000  — Fid 1.000, Cov 0.900, n=10
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) > 1.550000  — Fid 0.600, Cov 0.800, n=10
  - seed 44 (k=1)
    1. petal width (cm) > 0.800000 and petal width (cm) > 1.750000  — Fid 1.000, Cov 0.800, n=8


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.10  — Fid 1.000, Cov 0.400, n=4
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and petal length (cm) <= 1.60  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. petal width (cm) <= 1.35 and sepal width (cm) > 3.38  — Fid 1.000, Cov 0.300, n=3

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.30 and sepal width (cm) <= 2.73 and petal length (cm) <= 5.10  — Fid 1.000, Cov 0.300, n=3
  - seed 43 (k=1)
    1. 1.60 < petal length (cm) <= 4.40 and petal width (cm) <= 1.87 and sepal length (cm) > 5.75  — Fid 1.000, Cov 0.200, n=2
  - seed 44 (k=1)
    1. 0.30 < petal width (cm) <= 1.35 and sepal width (cm) <= 3.00  — Fid 0.875, Cov 0.700, n=8

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 0.875, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal width (cm) > 1.87 and petal length (cm) > 4.40  — Fid 1.000, Cov 0.400, n=4
  - seed 44 (k=1)
    1. petal width (cm) > 1.80 and petal length (cm) > 4.45 and sepal width (cm) <= 3.00  — Fid 1.000, Cov 0.700, n=7


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.10  — Fid 1.000, Cov 0.400, n=4
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and petal length (cm) <= 1.60  — Fid 1.000, Cov 0.700, n=7
  - seed 44 (k=1)
    1. petal width (cm) <= 1.35 and sepal width (cm) > 3.38  — Fid 1.000, Cov 0.300, n=3

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.80 and sepal width (cm) <= 3.00 and 1.60 < petal length (cm) <= 4.25  — Fid 0.750, Cov 0.300, n=4
  - seed 43 (k=1)
    1. 0.30 < petal width (cm) <= 1.87 and 1.60 < petal length (cm) <= 5.10 and sepal width (cm) <= 3.27  — Fid 0.846, Cov 0.800, n=13
  - seed 44 (k=1)
    1. 0.30 < petal width (cm) <= 1.35 and sepal width (cm) <= 3.00  — Fid 0.875, Cov 0.700, n=8

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 0.875, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal width (cm) > 1.87 and petal length (cm) > 4.40  — Fid 1.000, Cov 0.400, n=4
  - seed 44 (k=1)
    1. petal length (cm) > 5.10 and petal width (cm) > 1.35  — Fid 1.000, Cov 0.800, n=8


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 111  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. random box 100  — Fid 1.000, Cov 0.900, n=9
  - seed 44 (k=1)
    1. random box 10  — Fid 1.000, Cov 0.600, n=6

- `class_1`
  - seed 42 (k=1)
    1. random box 57  — Fid 0.714, Cov 0.400, n=7
  - seed 43 (k=1)
    1. random box 133  — Fid 1.000, Cov 0.800, n=8
  - seed 44 (k=1)
    1. random box 116  — Fid 1.000, Cov 0.900, n=9

- `class_2`
  - seed 42 (k=1)
    1. random box 51  — Fid 0.667, Cov 0.500, n=9
  - seed 43 (k=1)
    1. random box 235  — Fid 0.500, Cov 0.800, n=10
  - seed 44 (k=1)
    1. random box 188  — Fid 1.000, Cov 0.400, n=4



## wine

### wine — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. nonflavanoid_phenols ∈ [0.220000, 0.630000] and proline ∈ [845.000000, 1515.000000]  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. ash ∈ [1.700000, 2.801143] and total_phenols ∈ [1.100000, 3.170000] and flavanoids ∈ [2.524000, 3.930000]  — Fid 0.778, Cov 0.583, n=9
  - seed 44 (k=1)
    1. ash ∈ [2.127996, 3.220000] and color_intensity ∈ [4.475971, 6.931163]  — Fid 0.545, Cov 0.500, n=11

- `class_1`
  - seed 42 (k=1)
    1. flavanoids ∈ [1.380000, 3.130474] and nonflavanoid_phenols ∈ [0.210000, 0.630000]  — Fid 0.600, Cov 0.643, n=15
  - seed 43 (k=1)
    1. color_intensity ∈ [1.280000, 3.400000]  — Fid 0.800, Cov 0.714, n=10
  - seed 44 (k=1)
    1. alcohol ∈ [11.030000, 12.712003] and flavanoids ∈ [1.419993, 3.750000] and proanthocyanins ∈ [0.418324, 3.286977]  — Fid 1.000, Cov 0.714, n=10

- `class_2`
  - seed 42 (k=1)
    1. nonflavanoid_phenols ∈ [0.130000, 0.530000] and proanthocyanins ∈ [0.420000, 1.560000]  — Fid 0.467, Cov 0.700, n=15
  - seed 43 (k=1)
    1. alcohol ∈ [12.532412, 13.615375] and color_intensity ∈ [4.275000, 13.000000]  — Fid 0.625, Cov 0.400, n=8
  - seed 44 (k=1)
    1. ash ∈ [2.196996, 3.220000] and flavanoids ∈ [0.479643, 1.268104]  — Fid 0.875, Cov 0.700, n=8


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. alcalinity_of_ash ∈ [10.600001, 20.000000] and nonflavanoid_phenols ∈ [0.220000, 0.340000]  — Fid 0.643, Cov 0.750, n=14
  - seed 43 (k=1)
    1. total_phenols ∈ [2.586000, 3.880000] and flavanoids ∈ [0.470000, 3.745429]  — Fid 0.800, Cov 0.667, n=10
  - seed 44 (k=1)
    1. alcohol ∈ [13.067714, 14.830000]  — Fid 0.688, Cov 0.917, n=16

- `class_1`
  - seed 42 (k=1)
    1. alcohol ∈ [11.030000, 12.963396] and od280/od315_of_diluted_wines ∈ [2.271620, 3.481030]  — Fid 1.000, Cov 0.643, n=9
  - seed 43 (k=1)
    1. od280/od315_of_diluted_wines ∈ [2.076000, 4.000000]  — Fid 0.440, Cov 0.929, n=25
  - seed 44 (k=1)
    1. ash ∈ [1.360000, 2.583559] and nonflavanoid_phenols ∈ [0.242459, 0.558984]  — Fid 0.438, Cov 0.500, n=16

- `class_2`
  - seed 42 (k=1)
    1. alcalinity_of_ash ∈ [18.928572, 30.000002] and hue ∈ [0.540000, 0.835001]  — Fid 0.750, Cov 0.600, n=8
  - seed 43 (k=1)
    1. magnesium ∈ [90.399208, 151.000000] and proanthocyanins ∈ [0.410000, 1.657692]  — Fid 0.556, Cov 0.500, n=9
  - seed 44 (k=1)
    1. ash ∈ [2.196889, 3.220000] and proanthocyanins ∈ [0.410000, 1.259010] and color_intensity ∈ [1.280000, 10.081224]  — Fid 0.750, Cov 0.600, n=8


**cart**

- `class_0`
  - seed 42 (k=1)
    1. color_intensity > 3.945000 and flavanoids > 1.795000  — Fid 0.889, Cov 0.667, n=9
  - seed 43 (k=1)
    1. proline > 875.000000  — Fid 0.909, Cov 0.833, n=11
  - seed 44 (k=1)
    1. proline > 787.500000  — Fid 0.833, Cov 0.833, n=12

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 3.945000  — Fid 0.857, Cov 0.429, n=7
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity <= 3.970000  — Fid 0.750, Cov 0.571, n=8
  - seed 44 (k=1)
    1. proline <= 787.500000 and flavanoids > 1.385000  — Fid 0.875, Cov 0.500, n=8

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 3.945000 and flavanoids <= 1.795000  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity > 3.970000  — Fid 0.750, Cov 0.800, n=12
  - seed 44 (k=1)
    1. proline <= 787.500000 and flavanoids <= 1.385000  — Fid 0.667, Cov 0.600, n=9


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. proline > 985.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.583, n=7
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84  — Fid 1.000, Cov 0.667, n=8
  - seed 44 (k=1)
    1. proline > 645.00 and flavanoids > 2.23 and alcohol > 13.05  — Fid 1.000, Cov 0.833, n=10

- `class_1`
  - seed 42 (k=1)
    1. proline <= 500.00 and color_intensity <= 3.18  — Fid 1.000, Cov 0.143, n=2
  - seed 43 (k=1)
    1. proline <= 677.50 and color_intensity <= 3.19  — Fid 0.833, Cov 0.429, n=6
  - seed 44 (k=1)
    1. od280/od315_of_diluted_wines > 1.89 and color_intensity <= 3.25  — Fid 1.000, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 6.29 and od280/od315_of_diluted_wines <= 1.95  — Fid 1.000, Cov 0.500, n=5
  - seed 43 (k=1)
    1. color_intensity > 4.55 and flavanoids <= 1.21  — Fid 1.000, Cov 0.800, n=8
  - seed 44 (k=1)
    1. color_intensity > 6.10 and flavanoids <= 1.21  — Fid 1.000, Cov 0.400, n=4


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. proline > 985.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.583, n=7
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84  — Fid 1.000, Cov 0.667, n=8
  - seed 44 (k=1)
    1. proline > 645.00 and flavanoids > 2.23 and alcohol > 13.05  — Fid 1.000, Cov 0.833, n=10

- `class_1`
  - seed 42 (k=1)
    1. proline <= 500.00 and color_intensity <= 3.18  — Fid 1.000, Cov 0.143, n=2
  - seed 43 (k=1)
    1. proline <= 677.50 and color_intensity <= 3.19  — Fid 0.833, Cov 0.429, n=6
  - seed 44 (k=1)
    1. od280/od315_of_diluted_wines > 1.89 and color_intensity <= 3.25  — Fid 1.000, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 6.29 and od280/od315_of_diluted_wines <= 1.95  — Fid 1.000, Cov 0.500, n=5
  - seed 43 (k=1)
    1. color_intensity > 4.55 and flavanoids <= 1.21  — Fid 1.000, Cov 0.800, n=8
  - seed 44 (k=1)
    1. color_intensity > 6.10 and flavanoids <= 1.21  — Fid 1.000, Cov 0.400, n=4


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 18  — Fid 1.000, Cov 0.083, n=1
  - seed 43 (k=1)
    1. random box 111  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 142  — Fid 1.000, Cov 0.083, n=1

- `class_1`
  - seed 42 (k=1)
    1. random box 134  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 130  — Cov 0.000, n=0

- `class_2`
  - seed 42 (k=1)
    1. random box 181  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 125  — Cov 0.000, n=0


### wine — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. magnesium ∈ [78.000000, 120.228577]  — Fid 0.303, Cov 0.833, n=33
    2. alcalinity_of_ash ∈ [15.900001, 30.000002] and proanthocyanins ∈ [1.460000, 3.280000]  — Fid 0.533, Cov 0.667, n=15
  - seed 43 (k=1)
    1. alcalinity_of_ash ∈ [12.000000, 17.920000] and flavanoids ∈ [2.648000, 3.930000] and color_intensity ∈ [3.984000, 13.000000]  — Fid 1.000, Cov 0.333, n=4
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=1)
    1. total_phenols ∈ [1.703721, 3.880000] and flavanoids ∈ [1.380000, 2.915189] and proline ∈ [341.999969, 750.000000]  — Fid 1.000, Cov 0.643, n=9
  - seed 43 (k=2)
    1. malic_acid ∈ [0.740000, 3.072000] and alcalinity_of_ash ∈ [16.859970, 30.000000] and proanthocyanins ∈ [1.184813, 2.960000]  — Fid 0.438, Cov 0.500, n=16
    2. malic_acid ∈ [0.740000, 3.072000] and total_phenols ∈ [1.100000, 2.740000] and od280/od315_of_diluted_wines ∈ [2.239000, 4.000000]  — Fid 0.727, Cov 0.571, n=11
  - seed 44: _not run_

- `class_2`
  - seed 42 (k=1)
    1. malic_acid ∈ [0.740000, 4.036001] and ash ∈ [2.302000, 2.840000] and proline ∈ [341.999969, 759.000061]  — Fid 0.429, Cov 0.300, n=7
  - seed 43 (k=1)
    1. malic_acid ∈ [0.740000, 5.021999] and proanthocyanins ∈ [0.752449, 1.545500] and color_intensity ∈ [4.408572, 10.185068]  — Fid 1.000, Cov 0.800, n=8
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. ash ∈ [2.127401, 2.840000] and magnesium ∈ [93.318634, 115.000000] and flavanoids ∈ [2.406571, 3.740000]  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. alcalinity_of_ash ∈ [15.160000, 19.542856] and proanthocyanins ∈ [1.359143, 2.960000]  — Fid 0.583, Cov 0.583, n=12
  - seed 44 (k=1)
    1. alcohol ∈ [13.067714, 14.830000]  — Fid 0.688, Cov 0.917, n=16

- `class_1`
  - seed 42 (k=1)
    1. alcohol ∈ [11.030000, 12.946000]  — Fid 0.800, Cov 0.857, n=15
  - seed 43 (k=1)
    1. ash ∈ [1.707252, 2.580461] and alcalinity_of_ash ∈ [18.500000, 30.000000] and flavanoids ∈ [1.277151, 3.930000]  — Fid 0.875, Cov 0.571, n=8
  - seed 44 (k=1)
    1. ash ∈ [1.360000, 2.583559] and nonflavanoid_phenols ∈ [0.242459, 0.558984]  — Fid 0.438, Cov 0.500, n=16

- `class_2`
  - seed 42 (k=1)
    1. od280/od315_of_diluted_wines ∈ [1.290000, 2.075003]  — Fid 0.909, Cov 1.000, n=11
  - seed 43 (k=1)
    1. proanthocyanins ∈ [0.698541, 2.960000] and hue ∈ [0.573932, 0.894984]  — Fid 0.545, Cov 0.500, n=11
  - seed 44 (k=1)
    1. malic_acid ∈ [1.635714, 5.800000] and magnesium ∈ [70.000000, 122.035713] and total_phenols ∈ [1.150000, 2.125000]  — Fid 0.750, Cov 0.900, n=12


**cart**

- `class_0`
  - seed 42 (k=1)
    1. color_intensity > 3.820000 and flavanoids > 1.580000  — Fid 0.889, Cov 0.667, n=9
  - seed 43 (k=1)
    1. proline > 875.000000  — Fid 0.909, Cov 0.833, n=11
  - seed 44 (k=1)
    1. proline > 787.500000  — Fid 0.833, Cov 0.833, n=12

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 3.820000  — Fid 0.857, Cov 0.429, n=7
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity <= 3.970000  — Fid 1.000, Cov 0.571, n=8
  - seed 44 (k=1)
    1. proline <= 787.500000 and flavanoids > 1.385000  — Fid 0.875, Cov 0.500, n=8

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 3.820000 and flavanoids <= 1.580000  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity > 3.970000  — Fid 0.750, Cov 0.800, n=12
  - seed 44 (k=1)
    1. proline <= 787.500000 and flavanoids <= 1.385000  — Fid 0.667, Cov 0.600, n=9


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. total_phenols > 1.65 and proline > 675.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84 and alcalinity_of_ash <= 17.00  — Fid 1.000, Cov 0.167, n=2
  - seed 44 (k=1)
    1. alcohol > 12.37 and proline > 645.00 and flavanoids > 2.23  — Fid 1.000, Cov 0.917, n=11

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 4.80 and alcohol <= 12.37  — Fid 1.000, Cov 0.357, n=5
  - seed 43 (k=1)
    1. color_intensity <= 4.55 and proline <= 498.75  — Fid 1.000, Cov 0.357, n=5
  - seed 44 (k=1)
    1. proline <= 981.25 and color_intensity <= 3.25  — Fid 1.000, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. hue <= 0.96 and flavanoids <= 1.21 and nonflavanoid_phenols > 0.43  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. flavanoids <= 2.11 and hue <= 0.80 and od280/od315_of_diluted_wines <= 2.82  — Fid 0.800, Cov 0.800, n=10
  - seed 44 (k=1)
    1. total_phenols <= 2.38 and od280/od315_of_diluted_wines <= 1.89  — Fid 1.000, Cov 0.700, n=7


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. total_phenols > 1.65 and proline > 675.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84 and alcalinity_of_ash <= 17.00  — Fid 1.000, Cov 0.167, n=2
  - seed 44 (k=1)
    1. alcohol > 12.37 and proline > 645.00 and flavanoids > 2.23  — Fid 1.000, Cov 0.917, n=11

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 4.80 and alcohol <= 12.37  — Fid 1.000, Cov 0.357, n=5
  - seed 43 (k=1)
    1. color_intensity <= 4.55 and proline <= 498.75  — Fid 1.000, Cov 0.357, n=5
  - seed 44 (k=1)
    1. proline <= 981.25 and color_intensity <= 3.25  — Fid 1.000, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. hue <= 0.96 and flavanoids <= 1.21 and nonflavanoid_phenols > 0.43  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. flavanoids <= 2.11 and hue <= 0.80 and od280/od315_of_diluted_wines <= 2.82  — Fid 0.800, Cov 0.800, n=10
  - seed 44 (k=1)
    1. total_phenols <= 2.38 and od280/od315_of_diluted_wines <= 1.89  — Fid 1.000, Cov 0.700, n=7


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 125  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 111  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 142  — Fid 1.000, Cov 0.083, n=1

- `class_1`
  - seed 42 (k=1)
    1. random box 134  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 128  — Cov 0.000, n=0

- `class_2`
  - seed 42 (k=1)
    1. random box 181  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 125  — Cov 0.000, n=0



## breast_cancer

### breast_cancer — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=3)
    1. mean radius ∈ [14.254001, 28.110001] and mean texture ∈ [17.068001, 39.279999]  — Fid 0.925, Cov 0.810, n=40
    2. mean perimeter ∈ [100.215797, 128.030701] and mean symmetry ∈ [0.144769, 0.214480] and mean fractal dimension ∈ [0.053873, 0.069224] and fractal dimension error ∈ [0.000895, 0.006160]  — Fid 0.882, Cov 0.333, n=17
    3. mean perimeter ∈ [86.754547, 133.779999] and mean symmetry ∈ [0.160460, 0.304000] and mean fractal dimension ∈ [0.057457, 0.067795] and worst compactness ∈ [0.274700, 0.470600]  — Fid 0.714, Cov 0.190, n=14
  - seed 43 (k=3)
    1. mean radius ∈ [15.052000, 27.219999] and mean concavity ∈ [0.082230, 0.426800]  — Fid 0.962, Cov 0.595, n=26
    2. mean area ∈ [674.135498, 1365.889648] and radius error ∈ [0.334100, 1.509000] and fractal dimension error ∈ [0.000895, 0.006286]  — Fid 0.950, Cov 0.452, n=20
    3. radius error ∈ [0.284031, 0.835620] and concavity error ∈ [0.000000, 0.064282] and worst area ∈ [987.049988, 1739.500000]  — Fid 1.000, Cov 0.333, n=14
  - seed 44 (k=1)
    1. mean concavity ∈ [0.073542, 0.410800]  — Fid 0.800, Cov 1.000, n=55

- `class_1`
  - seed 42 (k=3)
    1. mean texture ∈ [13.312542, 20.743551] and mean perimeter ∈ [43.790001, 94.270996]  — Fid 0.977, Cov 0.583, n=43
    2. mean texture ∈ [15.385001, 39.279999] and mean perimeter ∈ [43.790001, 87.867996] and mean symmetry ∈ [0.116700, 0.208226]  — Fid 0.927, Cov 0.542, n=41
    3. mean texture ∈ [10.889020, 17.190001] and mean concave points ∈ [0.019768, 0.191300] and mean symmetry ∈ [0.116700, 0.184550] and texture error ∈ [0.360200, 1.596600] and area error ∈ [6.802006, 31.115002]  — Fid 1.000, Cov 0.097, n=9
  - seed 43 (k=2)
    1. mean radius ∈ [6.980999, 14.970747]  — Fid 0.821, Cov 0.958, n=84
    2. mean smoothness ∈ [0.068830, 0.100626] and perimeter error ∈ [1.170300, 2.577400] and area error ∈ [6.801998, 31.215996]  — Fid 0.933, Cov 0.569, n=45
  - seed 44 (k=3)
    1. mean smoothness ∈ [0.052630, 0.107413] and mean concave points ∈ [0.000000, 0.047422]  — Fid 0.984, Cov 0.847, n=61
    2. mean area ∈ [143.500061, 684.623413] and mean compactness ∈ [0.052326, 0.129540] and mean concavity ∈ [0.008109, 0.410800]  — Fid 0.925, Cov 0.694, n=53
    3. mean radius ∈ [6.981001, 14.530001] and mean texture ∈ [14.727999, 33.810001] and mean symmetry ∈ [0.160100, 0.194340]  — Fid 0.909, Cov 0.444, n=33


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. mean radius ∈ [14.762913, 28.110001] and mean perimeter ∈ [94.392006, 188.500000] and mean smoothness ∈ [0.090566, 0.142500]  — Fid 0.867, Cov 0.595, n=30
  - seed 43 (k=1)
    1. mean radius ∈ [15.052000, 27.219999] and mean perimeter ∈ [89.575996, 182.100006]  — Fid 0.929, Cov 0.619, n=28
  - seed 44 (k=1)
    1. mean radius ∈ [14.796000, 28.110001] and mean texture ∈ [17.091999, 33.810001]  — Fid 1.000, Cov 0.833, n=35

- `class_1`
  - seed 42 (k=1)
    1. mean radius ∈ [6.981000, 13.602000] and mean texture ∈ [10.939300, 39.279999]  — Fid 0.945, Cov 0.736, n=55
  - seed 43 (k=1)
    1. mean radius ∈ [6.980999, 13.498000] and mean texture ∈ [10.380001, 23.188000]  — Fid 0.902, Cov 0.625, n=51
  - seed 44 (k=1)
    1. mean smoothness ∈ [0.075121, 0.142500] and mean concavity ∈ [0.000000, 0.091914]  — Fid 0.965, Cov 0.764, n=57


**cart**

- `class_0`
  - seed 42 (k=1)
    1. worst concave points > 0.145450  — Fid 0.971, Cov 0.762, n=34
  - seed 43 (k=1)
    1. worst concave points > 0.135950  — Fid 0.892, Cov 0.762, n=37
  - seed 44 (k=1)
    1. worst perimeter > 116.049999  — Fid 1.000, Cov 0.714, n=30

- `class_1`
  - seed 42 (k=1)
    1. worst concave points <= 0.145450  — Fid 0.880, Cov 0.931, n=75
  - seed 43 (k=1)
    1. worst concave points <= 0.135950  — Fid 0.938, Cov 0.819, n=64
  - seed 44 (k=1)
    1. worst perimeter <= 116.049999  — Fid 0.917, Cov 0.944, n=72


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst texture > 25.09 and worst area > 985.50  — Fid 1.000, Cov 0.643, n=28
  - seed 43 (k=1)
    1. area error > 45.42 and worst perimeter > 126.90  — Fid 1.000, Cov 0.357, n=15
  - seed 44 (k=1)
    1. radius error > 0.47 and worst perimeter > 124.90  — Fid 1.000, Cov 0.500, n=21

- `class_1`
  - seed 42 (k=1)
    1. worst texture <= 25.09 and worst concave points <= 0.10  — Fid 1.000, Cov 0.444, n=32
  - seed 43 (k=1)
    1. mean compactness <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.306, n=23
  - seed 44 (k=1)
    1. mean compactness <= 0.13 and mean concave points <= 0.02  — Fid 1.000, Cov 0.347, n=25


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst texture > 25.09 and worst area > 985.50  — Fid 1.000, Cov 0.643, n=28
  - seed 43 (k=1)
    1. area error > 45.42 and worst perimeter > 126.90  — Fid 1.000, Cov 0.357, n=15
  - seed 44 (k=1)
    1. radius error > 0.47 and area error > 45.38  — Fid 0.957, Cov 0.524, n=23

- `class_1`
  - seed 42 (k=1)
    1. worst texture <= 25.09 and worst concave points <= 0.10  — Fid 1.000, Cov 0.444, n=32
  - seed 43 (k=1)
    1. mean compactness <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.306, n=23
  - seed 44 (k=1)
    1. mean compactness <= 0.13 and mean concave points <= 0.02  — Fid 1.000, Cov 0.347, n=25


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 131  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 121  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 201  — Fid 0.000, Cov 0.000, n=4

- `class_1`
  - seed 42 (k=1)
    1. random box 207  — Fid 1.000, Cov 0.056, n=4
  - seed 43 (k=1)
    1. random box 45  — Fid 1.000, Cov 0.056, n=4
  - seed 44 (k=1)
    1. random box 107  — Fid 1.000, Cov 0.125, n=9


### breast_cancer — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=3)
    1. mean radius ∈ [15.268000, 28.110001] and mean perimeter ∈ [43.790001, 133.779999] and mean smoothness ∈ [0.062510, 0.118340]  — Fid 0.962, Cov 0.548, n=26
    2. mean smoothness ∈ [0.090566, 0.114100] and mean symmetry ∈ [0.116700, 0.230460] and worst radius ∈ [17.384983, 23.760021]  — Fid 1.000, Cov 0.310, n=15
    3. mean concave points ∈ [-0.000000, 0.093986] and mean fractal dimension ∈ [0.054622, 0.097440] and worst texture ∈ [26.315998, 49.540001] and worst area ∈ [766.293335, 1839.000000] and worst smoothness ∈ [0.109392, 0.218400]  — Fid 0.867, Cov 0.310, n=15
  - seed 43 (k=3)
    1. mean radius ∈ [15.051996, 21.009918] and perimeter error ∈ [2.052192, 11.070000]  — Fid 0.957, Cov 0.524, n=23
    2. worst perimeter ∈ [99.313354, 176.606293] and worst area ∈ [185.200012, 2094.291504] and worst concavity ∈ [0.261619, 1.105000]  — Fid 0.833, Cov 0.738, n=36
    3. mean area ∈ [702.639404, 1357.714111] and mean fractal dimension ∈ [0.053491, 0.073578] and area error ∈ [24.559948, 233.000000]  — Fid 1.000, Cov 0.429, n=18
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=1)
    1. mean area ∈ [143.500031, 654.570007]  — Fid 0.929, Cov 0.875, n=70
  - seed 43 (k=2)
    1. mean perimeter ∈ [43.790001, 94.713501]  — Fid 0.883, Cov 0.917, n=77
    2. mean area ∈ [386.129639, 690.314514] and worst smoothness ∈ [0.104755, 0.151980] and worst compactness ∈ [0.027290, 0.301980]  — Fid 0.930, Cov 0.528, n=43
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. mean radius ∈ [15.262047, 28.110001] and mean texture ∈ [9.710000, 30.635748] and mean area ∈ [540.960022, 2499.000000]  — Fid 0.971, Cov 0.738, n=34
  - seed 43 (k=1)
    1. mean radius ∈ [15.052000, 27.219999] and mean texture ∈ [16.384251, 39.279999]  — Fid 0.962, Cov 0.595, n=26
  - seed 44 (k=1)
    1. mean radius ∈ [14.796000, 28.110001] and mean texture ∈ [17.091999, 33.810001]  — Fid 1.000, Cov 0.833, n=35

- `class_1`
  - seed 42 (k=1)
    1. mean radius ∈ [6.981000, 14.643740] and mean perimeter ∈ [43.790001, 94.270996]  — Fid 0.943, Cov 0.875, n=70
  - seed 43 (k=1)
    1. mean texture ∈ [12.954393, 39.279999] and mean perimeter ∈ [43.790001, 92.480003]  — Fid 0.897, Cov 0.819, n=68
  - seed 44 (k=1)
    1. mean radius ∈ [6.981001, 14.530001] and mean concave points ∈ [0.006222, 0.184500] and worst smoothness ∈ [0.085289, 0.140540]  — Fid 1.000, Cov 0.542, n=40


**cart**

- `class_0`
  - seed 42 (k=1)
    1. worst concave points > 0.145450  — Fid 0.941, Cov 0.762, n=34
  - seed 43 (k=1)
    1. worst concave points > 0.135950  — Fid 0.946, Cov 0.762, n=37
  - seed 44 (k=1)
    1. worst perimeter > 114.399998  — Fid 0.968, Cov 0.714, n=31

- `class_1`
  - seed 42 (k=1)
    1. worst concave points <= 0.145450  — Fid 0.893, Cov 0.931, n=75
  - seed 43 (k=1)
    1. worst concave points <= 0.135950  — Fid 0.984, Cov 0.819, n=64
  - seed 44 (k=1)
    1. worst perimeter <= 114.399998  — Fid 0.944, Cov 0.931, n=71


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst area > 985.50 and worst radius > 17.98  — Fid 1.000, Cov 0.738, n=32
  - seed 43 (k=1)
    1. worst area > 489.50 and worst concave points > 0.16  — Fid 1.000, Cov 0.595, n=26
  - seed 44 (k=1)
    1. mean concavity > 0.03 and worst perimeter > 97.59 and perimeter error > 2.29 and worst texture > 25.48  — Fid 0.885, Cov 0.571, n=26

- `class_1`
  - seed 42 (k=1)
    1. worst area <= 680.60 and worst radius <= 14.85  — Fid 1.000, Cov 0.667, n=49
  - seed 43 (k=1)
    1. worst concave points <= 0.10 and mean concavity <= 0.03  — Fid 1.000, Cov 0.375, n=28
  - seed 44 (k=1)
    1. worst concave points <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.347, n=25


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst area > 985.50 and worst radius > 17.98  — Fid 1.000, Cov 0.738, n=32
  - seed 43 (k=1)
    1. worst area > 489.50 and worst concave points > 0.16  — Fid 1.000, Cov 0.595, n=26
  - seed 44 (k=1)
    1. mean concavity > 0.03 and worst perimeter > 97.59 and perimeter error > 2.29 and worst texture > 25.48  — Fid 0.885, Cov 0.571, n=26

- `class_1`
  - seed 42 (k=1)
    1. worst area <= 680.60 and worst radius <= 14.85  — Fid 1.000, Cov 0.667, n=49
  - seed 43 (k=1)
    1. worst concave points <= 0.10 and mean concavity <= 0.03  — Fid 1.000, Cov 0.375, n=28
  - seed 44 (k=1)
    1. worst concave points <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.347, n=25


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 187  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 228  — Cov 0.000, n=0
  - seed 44 (k=1)
    1. random box 146  — Fid 0.000, Cov 0.000, n=1

- `class_1`
  - seed 42 (k=1)
    1. random box 102  — Fid 1.000, Cov 0.042, n=3
  - seed 43 (k=1)
    1. random box 166  — Fid 1.000, Cov 0.111, n=8
  - seed 44 (k=1)
    1. random box 186  — Cov 0.000, n=0



## synthetic

### synthetic — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 ∈ [-0.318884, 6.036793] and feature_5 ∈ [-0.199768, 5.608411]  — Fid 0.919, Cov 0.758, n=86
  - seed 43 (k=1)
    1. feature_0 ∈ [-4.417074, 1.684969] and feature_2 ∈ [-4.120684, 3.430854] and feature_3 ∈ [-3.325108, 5.864470]  — Fid 0.762, Cov 0.760, n=101
  - seed 44 (k=3)
    1. feature_3 ∈ [-2.480249, 1.185962] and feature_7 ∈ [-4.907111, 1.090601] and feature_8 ∈ [-1.279566, 0.769683]  — Fid 0.736, Cov 0.390, n=53
    2. feature_0 ∈ [-3.467809, 1.155576] and feature_3 ∈ [-4.063193, -0.235157] and feature_7 ∈ [-4.907111, -0.648375] and feature_8 ∈ [-4.323597, -0.656904]  — Fid 0.882, Cov 0.150, n=17
    3. feature_0 ∈ [-5.354650, 2.350787] and feature_5 ∈ [-1.319647, 4.624685] and feature_6 ∈ [-5.952446, 0.545578] and feature_8 ∈ [-1.279635, 0.318314]  — Fid 0.818, Cov 0.250, n=33

- `class_1`
  - seed 42 (k=1)
    1. feature_6 ∈ [-2.030793, 7.456970]  — Fid 0.506, Cov 0.921, n=178
  - seed 43 (k=5)
    1. feature_0 ∈ [1.383111, 5.282501] and feature_1 ∈ [-4.578733, 2.826565] and feature_2 ∈ [-3.033984, 0.397191] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.944, Cov 0.510, n=54
    2. feature_0 ∈ [0.460024, 2.997764] and feature_2 ∈ [-2.115639, -0.032339] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.971, Cov 0.330, n=34
    3. feature_0 ∈ [1.383111, 5.282501] and feature_1 ∈ [-4.578733, 1.366603] and feature_2 ∈ [-1.657719, 0.397191] and feature_3 ∈ [-9.130013, 0.318074]  — Fid 0.952, Cov 0.200, n=21
    4. feature_0 ∈ [0.214604, 5.282501] and feature_1 ∈ [-4.578733, 0.988761] and feature_2 ∈ [-1.657719, 0.397191] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.909, Cov 0.300, n=33
    5. feature_2 ∈ [-1.687017, -0.380612] and feature_3 ∈ [-4.579780, -0.603626] and feature_4 ∈ [-1.285808, 4.651225]  — Fid 0.963, Cov 0.250, n=27
  - seed 44 (k=3)
    1. feature_0 ∈ [-5.354650, 0.995057] and feature_4 ∈ [-2.512787, 2.956028] and feature_6 ∈ [-0.139703, 5.589865] and feature_7 ∈ [0.053238, 5.722162]  — Fid 0.966, Cov 0.560, n=59
    2. feature_0 ∈ [-5.354650, 0.159569] and feature_4 ∈ [-0.935434, 2.956028] and feature_5 ∈ [-4.771687, 1.965541] and feature_6 ∈ [0.372845, 5.589865] and feature_7 ∈ [-0.946757, 5.722162]  — Fid 0.970, Cov 0.320, n=33
    3. feature_0 ∈ [-2.629460, -0.364153] and feature_3 ∈ [-7.780115, 3.185455] and feature_4 ∈ [-3.494597, 1.174024] and feature_6 ∈ [-1.045622, 2.160554] and feature_7 ∈ [0.858891, 3.329683] and feature_8 ∈ [-1.195984, 5.380715]  — Fid 0.923, Cov 0.110, n=13


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 ∈ [0.281729, 6.036793] and feature_5 ∈ [-0.928563, 5.608411]  — Fid 0.867, Cov 0.747, n=90
  - seed 43 (k=1)
    1. feature_1 ∈ [-1.563525, 3.900472] and feature_6 ∈ [-3.060544, 0.578759] and feature_7 ∈ [-0.154771, 3.962004] and feature_8 ∈ [-3.468153, 0.271785] and feature_9 ∈ [-1.297152, 3.438132]  — Fid 1.000, Cov 0.280, n=29
  - seed 44 (k=1)
    1. feature_0 ∈ [-2.148176, 4.254667] and feature_7 ∈ [-4.907111, 0.046012]  — Fid 0.825, Cov 0.790, n=97

- `class_1`
  - seed 42 (k=1)
    1. feature_1 ∈ [-2.232415, 4.585218] and feature_3 ∈ [-0.513922, 4.843362] and feature_4 ∈ [-3.716660, 1.196453]  — Fid 0.796, Cov 0.406, n=49
  - seed 43 (k=1)
    1. feature_0 ∈ [0.399875, 3.672356] and feature_2 ∈ [-1.408125, 3.430854] and feature_8 ∈ [-0.546124, 5.184010]  — Fid 0.933, Cov 0.410, n=45
  - seed 44 (k=1)
    1. feature_0 ∈ [-5.354650, 0.995069] and feature_6 ∈ [-0.139591, 5.589865] and feature_7 ∈ [0.053238, 5.722162]  — Fid 0.966, Cov 0.560, n=59


**cart**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > -0.070581  — Fid 0.659, Cov 0.869, n=135
  - seed 43 (k=1)
    1. feature_2 > 0.610751  — Fid 0.804, Cov 0.380, n=46
  - seed 44 (k=1)
    1. feature_7 <= -0.204200  — Fid 0.745, Cov 0.730, n=102

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.070581  — Fid 0.818, Cov 0.475, n=55
  - seed 43 (k=1)
    1. feature_2 <= 0.610751  — Fid 0.607, Cov 0.900, n=150
  - seed 44 (k=1)
    1. feature_7 > -0.204200  — Fid 0.750, Cov 0.660, n=88


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.918, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_8 <= -0.56 and feature_3 > 0.38 and feature_4 > -0.48  — Fid 0.935, Cov 0.290, n=31
  - seed 44 (k=1)
    1. feature_0 > 0.73 and feature_7 <= 1.18 and feature_9 > -0.09  — Fid 0.950, Cov 0.180, n=20

- `class_1`
  - seed 42 (k=1)
    1. feature_5 <= -0.69 and feature_9 > 2.36  — Fid 0.906, Cov 0.307, n=32
  - seed 43 (k=1)
    1. feature_0 > 2.24 and feature_8 > -0.56 and feature_2 <= 0.52  — Fid 1.000, Cov 0.330, n=33
  - seed 44 (k=1)
    1. feature_6 > 0.59 and feature_0 <= -0.51 and feature_5 <= 1.00  — Fid 0.970, Cov 0.320, n=33


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.918, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_8 <= -0.56 and feature_3 > 0.38 and feature_4 > -0.48  — Fid 0.935, Cov 0.290, n=31
  - seed 44 (k=1)
    1. feature_0 > 0.73 and feature_7 <= 1.18 and feature_9 > -0.09  — Fid 0.950, Cov 0.180, n=20

- `class_1`
  - seed 42 (k=1)
    1. feature_5 <= -0.69 and feature_9 > 2.36  — Fid 0.906, Cov 0.307, n=32
  - seed 43 (k=1)
    1. feature_0 > 2.24 and feature_8 > -0.56 and feature_2 <= 0.52  — Fid 1.000, Cov 0.330, n=33
  - seed 44 (k=1)
    1. feature_6 > 0.59 and feature_0 <= -0.51 and feature_5 <= 1.00  — Fid 0.970, Cov 0.320, n=33


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 148  — Fid 1.000, Cov 0.152, n=16
  - seed 43 (k=1)
    1. random box 11  — Fid 1.000, Cov 0.040, n=4
  - seed 44 (k=1)
    1. random box 119  — Fid 0.636, Cov 0.090, n=11

- `class_1`
  - seed 42 (k=1)
    1. random box 135  — Fid 1.000, Cov 0.040, n=4
  - seed 43 (k=1)
    1. random box 235  — Fid 0.900, Cov 0.090, n=10
  - seed 44 (k=1)
    1. random box 7  — Fid 0.667, Cov 0.100, n=18


### synthetic — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. feature_2 ∈ [-2.604953, 3.276399] and feature_4 ∈ [0.543823, 2.671431] and feature_5 ∈ [-0.199769, 5.608411] and feature_9 ∈ [-7.018609, 2.676795]  — Fid 1.000, Cov 0.545, n=56
    2. feature_0 ∈ [-4.661168, 1.743295] and feature_2 ∈ [-1.930220, 0.849564] and feature_4 ∈ [0.173260, 1.658094] and feature_5 ∈ [-0.928314, 5.608411]  — Fid 0.902, Cov 0.465, n=51
  - seed 43 (k=3)
    1. feature_2 ∈ [-1.378004, 3.430854] and feature_4 ∈ [-1.886569, 1.410273] and feature_5 ∈ [-3.644516, 1.315525] and feature_7 ∈ [-0.554244, 3.962004] and feature_9 ∈ [-3.441975, 1.198165]  — Fid 0.867, Cov 0.260, n=30
    2. feature_0 ∈ [-4.417074, 1.195978] and feature_4 ∈ [-4.987191, 1.709342] and feature_5 ∈ [-3.644516, 0.586923] and feature_9 ∈ [-3.441975, 0.811636]  — Fid 0.822, Cov 0.380, n=45
    3. feature_1 ∈ [-2.209138, 0.409050] and feature_3 ∈ [-2.787386, 5.864470] and feature_5 ∈ [-0.557768, 2.870024] and feature_6 ∈ [-3.060544, 0.310358] and feature_7 ∈ [-1.467845, 1.824637]  — Fid 0.667, Cov 0.080, n=12
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=3)
    1. feature_0 ∈ [-0.944462, 4.724046]  — Fid 0.554, Cov 0.891, n=157
    2. feature_2 ∈ [-0.988432, 2.235508] and feature_3 ∈ [-1.985323, 4.843362] and feature_4 ∈ [-3.716660, -0.444956] and feature_5 ∈ [-0.887323, 5.608411] and feature_9 ∈ [-4.863406, 7.933945]  — Fid 0.931, Cov 0.277, n=29
    3. feature_3 ∈ [-0.876189, 0.541337] and feature_4 ∈ [-3.716660, 0.476419] and feature_5 ∈ [-1.721353, 5.608411] and feature_9 ∈ [-2.559016, 7.933945]  — Fid 0.909, Cov 0.099, n=11
  - seed 43 (k=3)
    1. feature_0 ∈ [-0.343942, 5.282501] and feature_2 ∈ [-2.116620, 0.312801] and feature_3 ∈ [-7.876883, -1.026891]  — Fid 1.000, Cov 0.440, n=46
    2. feature_0 ∈ [-0.343928, 5.282501] and feature_2 ∈ [-2.462268, -0.032334] and feature_3 ∈ [-3.648659, -1.026899]  — Fid 1.000, Cov 0.390, n=41
    3. feature_2 ∈ [-2.115676, -0.032333] and feature_5 ∈ [-1.569860, 0.499156] and feature_6 ∈ [-1.228518, 2.859548] and feature_7 ∈ [-2.227543, 0.562137] and feature_8 ∈ [-0.261707, 5.184010]  — Fid 0.969, Cov 0.310, n=32
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. feature_0 ∈ [-4.661168, 1.503932] and feature_4 ∈ [-0.318891, 6.036793] and feature_5 ∈ [0.199058, 5.608411]  — Fid 1.000, Cov 0.596, n=60
  - seed 43 (k=1)
    1. feature_0 ∈ [-4.417074, 1.563181] and feature_2 ∈ [-2.248454, 3.430854] and feature_3 ∈ [-1.474722, 5.864470] and feature_6 ∈ [-3.060544, 0.157763] and feature_7 ∈ [-0.555029, 3.962004] and feature_8 ∈ [-3.468153, -0.050120] and feature_9 ∈ [-0.636340, 3.438132]  — Fid 1.000, Cov 0.150, n=15
  - seed 44 (k=1)
    1. feature_0 ∈ [-1.024776, 4.254667] and feature_7 ∈ [-3.526631, -0.318895] and feature_9 ∈ [-0.252362, 5.524426]  — Fid 0.972, Cov 0.320, n=36

- `class_1`
  - seed 42 (k=1)
    1. feature_0 ∈ [-4.661168, 2.774007] and feature_3 ∈ [-1.186223, 4.843362] and feature_4 ∈ [-3.716660, -0.019826] and feature_6 ∈ [-0.992333, 7.456970]  — Fid 1.000, Cov 0.347, n=35
  - seed 43 (k=1)
    1. feature_0 ∈ [0.894977, 5.282501] and feature_3 ∈ [-9.130013, 0.266369] and feature_8 ∈ [-0.089434, 5.184010]  — Fid 0.969, Cov 0.630, n=64
  - seed 44 (k=1)
    1. feature_0 ∈ [-5.354650, 0.995045] and feature_6 ∈ [-0.486886, 5.589865] and feature_7 ∈ [-0.234976, 5.722162]  — Fid 0.955, Cov 0.610, n=66


**cart**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.031331  — Fid 0.667, Cov 0.859, n=132
  - seed 43 (k=1)
    1. feature_2 > 0.610751  — Fid 0.870, Cov 0.380, n=46
  - seed 44 (k=1)
    1. feature_7 <= -0.241016  — Fid 0.822, Cov 0.720, n=101

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= 0.031331  — Fid 0.828, Cov 0.495, n=58
  - seed 43 (k=1)
    1. feature_2 <= 0.610751  — Fid 0.620, Cov 0.900, n=150
  - seed 44 (k=1)
    1. feature_7 > -0.241016  — Fid 0.764, Cov 0.660, n=89


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.934, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_0 <= -0.46 and feature_2 > -0.52 and feature_4 <= 1.40  — Fid 0.968, Cov 0.310, n=31
  - seed 44 (k=1)
    1. feature_7 <= -0.10 and feature_0 > 0.73  — Fid 0.966, Cov 0.250, n=29

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.61 and feature_3 > 0.29  — Fid 1.000, Cov 0.267, n=27
  - seed 43 (k=1)
    1. feature_8 > -0.56 and feature_2 <= -0.52 and feature_7 <= -0.63  — Fid 0.974, Cov 0.350, n=38
  - seed 44 (k=1)
    1. feature_7 > -0.10 and feature_6 > 0.59 and feature_0 <= -0.51  — Fid 1.000, Cov 0.350, n=35


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.934, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_0 <= -0.46 and feature_2 > -0.52 and feature_4 <= 1.40  — Fid 0.968, Cov 0.310, n=31
  - seed 44 (k=1)
    1. feature_7 <= -0.10 and feature_0 > 0.73  — Fid 0.966, Cov 0.250, n=29

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.61 and feature_3 > 0.29  — Fid 1.000, Cov 0.267, n=27
  - seed 43 (k=1)
    1. feature_8 > -0.56 and feature_2 <= -0.52 and feature_7 <= -0.63  — Fid 0.974, Cov 0.350, n=38
  - seed 44 (k=1)
    1. feature_7 > -0.10 and feature_6 > 0.59 and feature_0 <= -0.51  — Fid 1.000, Cov 0.350, n=35


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 203  — Fid 0.750, Cov 0.081, n=12
  - seed 43 (k=1)
    1. random box 167  — Fid 0.833, Cov 0.050, n=6
  - seed 44 (k=1)
    1. random box 119  — Fid 0.818, Cov 0.090, n=11

- `class_1`
  - seed 42 (k=1)
    1. random box 97  — Fid 0.750, Cov 0.099, n=12
  - seed 43 (k=1)
    1. random box 235  — Fid 0.900, Cov 0.090, n=10
  - seed 44 (k=1)
    1. random box 7  — Fid 0.556, Cov 0.100, n=18



## housing

### housing — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=5)
    1. MedInc ∈ [0.499900, 2.823835] and AveRooms ∈ [4.898873, 62.422218] and AveBedrms ∈ [1.001818, 14.111111] and AveOccup ∈ [0.692308, 3.786198] and Latitude ∈ [34.157082, 41.880001]  — Fid 0.837, Cov 0.187, n=258
    2. MedInc ∈ [0.499900, 2.276656] and AveRooms ∈ [4.022934, 62.422218] and AveBedrms ∈ [1.028851, 14.111111] and AveOccup ∈ [2.623818, 3.819907] and Latitude ∈ [34.077488, 41.880001]  — Fid 0.908, Cov 0.102, n=119
    3. MedInc ∈ [1.446400, 3.139460] and AveRooms ∈ [0.846154, 5.981257] and AveBedrms ∈ [0.444444, 1.270933] and AveOccup ∈ [2.848440, 502.461578] and Longitude ∈ [-124.349991, -119.099991]  — Fid 0.797, Cov 0.178, n=251
    4. MedInc ∈ [1.264652, 2.323702] and HouseAge ∈ [10.999998, 39.000000] and AveRooms ∈ [0.846154, 7.424706] and AveBedrms ∈ [0.444444, 1.728300] and Population ∈ [81.983276, 1427.000000] and AveOccup ∈ [2.455009, 502.461578] and Longitude ∈ [-124.349991, -115.999710]  — Fid 0.765, Cov 0.135, n=187
    5. MedInc ∈ [1.124935, 3.139460] and AveRooms ∈ [0.846154, 5.491123] and AveBedrms ∈ [0.444444, 1.270933] and AveOccup ∈ [2.921917, 502.461578] and Longitude ∈ [-124.349991, -119.099991]  — Fid 0.769, Cov 0.151, n=216
  - seed 43 (k=5)
    1. MedInc ∈ [0.499900, 2.979764] and AveRooms ∈ [4.399619, 132.533340] and AveBedrms ∈ [0.333333, 1.155628]  — Fid 0.707, Cov 0.348, n=559
    2. MedInc ∈ [1.261479, 2.814030] and AveBedrms ∈ [1.004359, 1.255662] and AveOccup ∈ [0.692308, 4.109674] and Latitude ∈ [34.150002, 41.950001]  — Fid 0.696, Cov 0.278, n=471
    3. MedInc ∈ [0.499900, 2.336201] and AveRooms ∈ [4.343571, 5.688532] and AveBedrms ∈ [0.960144, 1.097496]  — Fid 0.870, Cov 0.133, n=169
    4. MedInc ∈ [0.499900, 2.567241] and AveRooms ∈ [5.243216, 132.533340] and AveBedrms ∈ [0.333333, 1.284522]  — Fid 0.828, Cov 0.087, n=128
    5. MedInc ∈ [1.769288, 2.336200] and AveRooms ∈ [3.854238, 132.533340] and AveBedrms ∈ [1.048989, 1.117666] and Population ∈ [793.000000, 2951.588135] and AveOccup ∈ [0.692308, 4.270623] and Latitude ∈ [33.809013, 41.950001]  — Fid 0.841, Cov 0.047, n=69
  - seed 44 (k=5)
    1. MedInc ∈ [0.499900, 2.546029] and HouseAge ∈ [11.000000, 39.000000] and AveRooms ∈ [3.844063, 141.909088] and Latitude ∈ [34.240002, 41.950005]  — Fid 0.872, Cov 0.259, n=336
    2. MedInc ∈ [0.499900, 2.792084] and AveRooms ∈ [4.237589, 6.212649] and AveOccup ∈ [2.232184, 1243.333374]  — Fid 0.797, Cov 0.426, n=602
    3. MedInc ∈ [0.499900, 2.975000] and HouseAge ∈ [0.999998, 45.000000] and AveRooms ∈ [4.381483, 141.909088] and Latitude ∈ [33.990002, 41.950005]  — Fid 0.797, Cov 0.385, n=557
    4. MedInc ∈ [0.499900, 2.313617] and Population ∈ [2.999878, 2511.898926] and AveOccup ∈ [2.607345, 1243.333374] and Latitude ∈ [34.240002, 38.621994] and Longitude ∈ [-121.580002, -117.110001]  — Fid 0.935, Cov 0.123, n=139
    5. MedInc ∈ [0.499900, 3.994378] and Population ∈ [2.999878, 1040.530029] and AveOccup ∈ [2.762905, 1243.333374] and Latitude ∈ [35.750473, 36.834488] and Longitude ∈ [-121.300003, -117.532974]  — Fid 1.000, Cov 0.036, n=39

- `class_1`
  - seed 42 (k=4)
    1. MedInc ∈ [3.295002, 4.067696] and AveOccup ∈ [4.302743, 502.461578] and Latitude ∈ [32.924000, 41.880001] and Longitude ∈ [-124.349991, -117.550003]  — Fid 0.767, Cov 0.019, n=30
    2. MedInc ∈ [2.068848, 15.000100] and HouseAge ∈ [14.999999, 52.000000] and AveRooms ∈ [0.846154, 5.611066] and AveOccup ∈ [3.017442, 502.461578] and Longitude ∈ [-119.074013, -114.489990]  — Fid 0.643, Cov 0.252, n=476
    3. MedInc ∈ [2.051857, 2.875000] and HouseAge ∈ [10.000000, 52.000000] and AveRooms ∈ [0.846154, 4.407454] and AveBedrms ∈ [0.444444, 1.112226] and Population ∈ [1504.000000, 16304.999023] and AveOccup ∈ [2.662883, 4.417292] and Latitude ∈ [32.549999, 34.660000] and Longitude ∈ [-118.284897, -114.489990]  — Fid 0.812, Cov 0.030, n=48
    4. MedInc ∈ [2.160012, 3.158869] and HouseAge ∈ [14.999999, 52.000000] and AveRooms ∈ [0.846154, 3.628435] and AveBedrms ∈ [0.444444, 1.087446] and Population ∈ [1758.000122, 16304.999023] and AveOccup ∈ [2.838722, 4.636699] and Latitude ∈ [32.549999, 34.099998] and Longitude ∈ [-118.334534, -114.489990]  — Fid 1.000, Cov 0.008, n=11
  - seed 43 (k=2)
    1. MedInc ∈ [2.037470, 2.682440] and HouseAge ∈ [35.000000, 52.000000] and AveRooms ∈ [3.446317, 4.667093] and Population ∈ [1297.000000, 35682.003906] and AveOccup ∈ [2.855951, 502.461548] and Latitude ∈ [33.939999, 41.950001] and Longitude ∈ [-121.440002, -117.120003]  — Fid 0.897, Cov 0.016, n=29
    2. MedInc ∈ [3.237500, 4.200259] and HouseAge ∈ [24.000000, 52.000000] and AveRooms ∈ [4.004460, 5.564185] and Population ∈ [789.787354, 1976.487427] and AveOccup ∈ [4.088190, 502.461548] and Latitude ∈ [33.830002, 41.950001] and Longitude ∈ [-122.136002, -114.309998]  — Fid 0.944, Cov 0.016, n=18
  - seed 44 (k=4)
    1. MedInc ∈ [2.523466, 3.236916] and AveRooms ∈ [0.846154, 4.373985] and AveBedrms ∈ [0.333333, 1.114590] and Population ∈ [930.000000, 28566.000000] and AveOccup ∈ [2.835620, 1243.333374] and Latitude ∈ [32.540001, 38.540001] and Longitude ∈ [-118.230011, -117.877563]  — Fid 0.809, Cov 0.034, n=47
    2. MedInc ∈ [2.062500, 4.173901] and AveRooms ∈ [0.846154, 5.583738] and AveBedrms ∈ [0.333333, 1.166243] and Population ∈ [1857.266968, 28566.000000] and AveOccup ∈ [4.379761, 1243.333374] and Longitude ∈ [-120.939308, -118.160004]  — Fid 0.833, Cov 0.015, n=18
    3. MedInc ∈ [2.062500, 4.173901] and AveRooms ∈ [0.846154, 3.993431] and Population ∈ [548.000000, 28566.000000] and AveOccup ∈ [4.552453, 1243.333374] and Longitude ∈ [-121.449997, -117.970001]  — Fid 0.875, Cov 0.014, n=24
    4. MedInc ∈ [0.499900, 3.000000] and AveRooms ∈ [2.636078, 4.373990] and Population ∈ [1096.999878, 1758.232910] and AveOccup ∈ [2.634380, 1243.333374] and Longitude ∈ [-118.221497, -118.160004]  — Fid 1.000, Cov 0.016, n=20

- `class_2`
  - seed 42 (k=5)
    1. MedInc ∈ [3.036780, 5.638802] and Population ∈ [564.800049, 16304.999023]  — Fid 0.440, Cov 0.611, n=1777
    2. MedInc ∈ [4.356608, 15.000100] and AveRooms ∈ [0.846154, 5.986365] and AveBedrms ∈ [0.908449, 1.181023] and Population ∈ [1348.888184, 16304.999023] and AveOccup ∈ [2.628149, 502.461578] and Latitude ∈ [32.769859, 33.840000] and Longitude ∈ [-124.349991, -116.969894]  — Fid 0.611, Cov 0.011, n=18
    3. MedInc ∈ [3.036780, 5.638807] and Latitude ∈ [33.680000, 41.880001]  — Fid 0.437, Cov 0.594, n=1766
    4. MedInc ∈ [4.341707, 6.039940] and Population ∈ [669.854126, 16304.999023] and AveOccup ∈ [3.111642, 502.461578] and Latitude ∈ [33.910000, 37.680000] and Longitude ∈ [-124.349991, -117.660004]  — Fid 0.812, Cov 0.065, n=112
    5. MedInc ∈ [2.810060, 5.332797] and Population ∈ [3.000000, 2787.000488] and Latitude ∈ [33.779999, 37.730000]  — Fid 0.472, Cov 0.424, n=1188
  - seed 43 (k=4)
    1. MedInc ∈ [3.008880, 5.631200] and HouseAge ∈ [1.000000, 45.000000]  — Fid 0.398, Cov 0.631, n=1847
    2. MedInc ∈ [2.805551, 4.352780] and HouseAge ∈ [1.000000, 33.000000] and AveRooms ∈ [0.846154, 5.806704] and Population ∈ [943.000000, 2457.256836] and AveOccup ∈ [0.692308, 3.280338] and Latitude ∈ [33.810001, 37.959999] and Longitude ∈ [-122.260002, -114.309998]  — Fid 0.543, Cov 0.085, n=210
    3. MedInc ∈ [3.322452, 4.180600] and HouseAge ∈ [16.999998, 52.000000] and AveBedrms ∈ [0.333333, 1.138781] and AveOccup ∈ [2.532245, 2.573802]  — Fid 0.471, Cov 0.005, n=17
    4. MedInc ∈ [3.330480, 15.000101] and HouseAge ∈ [14.875121, 52.000000] and AveRooms ∈ [3.788420, 132.533340] and Latitude ∈ [34.419998, 35.133526] and Longitude ∈ [-120.600029, -118.067032]  — Fid 0.208, Cov 0.007, n=48
  - seed 44 (k=5)
    1. MedInc ∈ [2.986699, 5.845356] and Population ∈ [247.978638, 3028.934082] and Longitude ∈ [-124.300003, -117.639999]  — Fid 0.454, Cov 0.582, n=1694
    2. MedInc ∈ [3.301492, 15.000101] and AveRooms ∈ [0.846154, 6.489146] and AveBedrms ∈ [0.948442, 1.055898] and AveOccup ∈ [2.946866, 3.094756] and Latitude ∈ [33.119411, 34.270000] and Longitude ∈ [-118.230011, -117.049759]  — Fid 0.743, Cov 0.027, n=35
    3. MedInc ∈ [2.986700, 5.587268] and Population ∈ [913.399597, 4095.101074] and Longitude ∈ [-124.300003, -117.639999]  — Fid 0.477, Cov 0.418, n=1157
    4. MedInc ∈ [3.969356, 6.008106] and HouseAge ∈ [21.702806, 40.000000] and AveBedrms ∈ [1.068490, 25.636364] and Population ∈ [2.999878, 2459.123047] and AveOccup ∈ [2.533110, 3.508057] and Longitude ∈ [-122.430000, -114.470001]  — Fid 0.676, Cov 0.035, n=71
    5. MedInc ∈ [2.986694, 15.000101] and AveRooms ∈ [0.846154, 6.606678] and AveBedrms ∈ [1.018442, 1.055898] and AveOccup ∈ [2.817885, 3.662817] and Latitude ∈ [33.820000, 37.259998] and Longitude ∈ [-118.919998, -118.089996]  — Fid 0.659, Cov 0.023, n=41

- `class_3`
  - seed 42 (k=5)
    1. MedInc ∈ [5.833889, 15.000100] and AveRooms ∈ [0.846154, 7.460154] and AveBedrms ∈ [0.444444, 1.042160] and Population ∈ [414.927673, 815.000000] and Longitude ∈ [-124.349991, -118.309998]  — Fid 0.792, Cov 0.035, n=48
    2. MedInc ∈ [5.924500, 8.295311] and HouseAge ∈ [11.926678, 52.000000] and AveRooms ∈ [6.738949, 62.422218] and Population ∈ [815.000000, 16304.999023] and AveOccup ∈ [2.580318, 3.568241] and Latitude ∈ [32.549999, 36.810017]  — Fid 0.926, Cov 0.047, n=54
    3. MedInc ∈ [5.367700, 8.045593] and HouseAge ∈ [35.355297, 52.000000] and AveRooms ∈ [6.343899, 62.422218] and Population ∈ [671.000000, 16304.999023] and AveOccup ∈ [2.446483, 2.850975] and Latitude ∈ [32.549999, 37.880692]  — Fid 0.944, Cov 0.016, n=18
    4. MedInc ∈ [7.642435, 15.000100] and HouseAge ∈ [1.000000, 23.678293] and AveBedrms ∈ [0.444444, 1.149152] and Population ∈ [815.000000, 3221.816406] and AveOccup ∈ [1.880553, 3.088417]  — Fid 0.923, Cov 0.023, n=26
    5. MedInc ∈ [6.486750, 15.000100] and AveRooms ∈ [0.846154, 8.130054] and AveBedrms ∈ [0.444444, 1.043256] and Population ∈ [815.000000, 1251.000000] and Longitude ∈ [-124.349991, -118.207428]  — Fid 0.857, Cov 0.035, n=42
  - seed 43 (k=5)
    1. MedInc ∈ [5.368300, 15.000101] and HouseAge ∈ [16.837208, 52.000000] and AveBedrms ∈ [0.994438, 34.066666] and AveOccup ∈ [0.692308, 2.843699] and Latitude ∈ [32.549995, 37.799995]  — Fid 0.922, Cov 0.138, n=153
    2. MedInc ∈ [6.993691, 15.000101] and HouseAge ∈ [13.902454, 52.000000] and AveBedrms ∈ [1.000765, 1.057012] and AveOccup ∈ [0.692308, 3.002717] and Latitude ∈ [32.549995, 37.799995]  — Fid 0.978, Cov 0.044, n=46
    3. MedInc ∈ [6.421195, 13.402983] and AveRooms ∈ [4.133965, 132.533340] and AveOccup ∈ [2.219352, 2.651572]  — Fid 0.956, Cov 0.086, n=91
    4. MedInc ∈ [5.903900, 15.000101] and HouseAge ∈ [19.767765, 52.000000] and AveBedrms ∈ [1.025806, 34.066666] and AveOccup ∈ [0.692308, 3.965324] and Latitude ∈ [32.549995, 37.659996]  — Fid 0.917, Cov 0.088, n=96
    5. MedInc ∈ [5.903900, 15.000101] and AveRooms ∈ [4.445275, 6.298551] and Population ∈ [50.993164, 1478.500244] and AveOccup ∈ [1.852852, 2.843699] and Latitude ∈ [33.830002, 41.950001] and Longitude ∈ [-124.349998, -117.769997]  — Fid 0.896, Cov 0.037, n=48
  - seed 44 (k=4)
    1. MedInc ∈ [6.847146, 15.000101] and AveRooms ∈ [6.736480, 141.909088] and Latitude ∈ [37.500000, 41.950005]  — Fid 1.000, Cov 0.038, n=41
    2. MedInc ∈ [6.120412, 7.827309] and AveRooms ∈ [5.155161, 141.909088] and AveBedrms ∈ [0.333333, 1.058315] and Latitude ∈ [33.708694, 41.950005]  — Fid 0.818, Cov 0.138, n=170
    3. MedInc ∈ [5.389354, 15.000101] and AveRooms ∈ [6.345844, 141.909088] and AveOccup ∈ [0.692308, 3.021777] and Latitude ∈ [34.090000, 41.950005]  — Fid 0.838, Cov 0.169, n=204
    4. MedInc ∈ [5.432141, 6.826800] and AveBedrms ∈ [0.963529, 25.636364] and AveOccup ∈ [1.568169, 2.050102]  — Fid 1.000, Cov 0.013, n=15


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 1.910540] and HouseAge ∈ [12.999999, 39.000000] and AveRooms ∈ [0.846154, 4.784486] and Population ∈ [1072.500000, 16304.999023] and AveOccup ∈ [2.758413, 502.461578] and Latitude ∈ [34.249001, 38.619999]  — Fid 0.925, Cov 0.034, n=40
  - seed 43 (k=1)
    1. MedInc ∈ [0.499900, 2.336201] and AveRooms ∈ [3.362297, 132.533340] and Population ∈ [325.932129, 35682.003906] and Latitude ∈ [35.619999, 41.950001] and Longitude ∈ [-124.349998, -117.019310]  — Fid 0.890, Cov 0.266, n=336
  - seed 44 (k=1)
    1. MedInc ∈ [0.499900, 2.792080] and AveRooms ∈ [4.237593, 141.909088] and Latitude ∈ [33.770000, 41.950005]  — Fid 0.786, Cov 0.475, n=693

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 4.704296] and AveOccup ∈ [2.459047, 502.461578]  — Fid 0.374, Cov 0.716, n=2146
  - seed 43 (k=1)
    1. MedInc ∈ [0.499900, 4.656301]  — Fid 0.289, Cov 0.893, n=3042
  - seed 44 (k=1)
    1. MedInc ∈ [0.499900, 5.490491] and HouseAge ∈ [0.999998, 48.000000] and AveRooms ∈ [0.846154, 6.014778] and AveBedrms ∈ [0.960762, 1.042052] and AveOccup ∈ [3.250759, 1243.333374] and Latitude ∈ [32.540001, 37.509998] and Longitude ∈ [-118.651985, -114.470001]  — Fid 0.692, Cov 0.106, n=208

- `class_2`
  - seed 42 (k=1)
    1. MedInc ∈ [3.579812, 5.640269] and AveOccup ∈ [2.407553, 3.344623] and Latitude ∈ [32.759998, 37.740002]  — Fid 0.599, Cov 0.288, n=628
  - seed 43 (k=1)
    1. MedInc ∈ [3.008880, 5.671132] and Latitude ∈ [32.549995, 37.959999]  — Fid 0.425, Cov 0.640, n=1791
  - seed 44 (k=1)
    1. MedInc ∈ [2.986700, 5.590778] and Latitude ∈ [33.669998, 41.950005]  — Fid 0.425, Cov 0.611, n=1855

- `class_3`
  - seed 42 (k=1)
    1. MedInc ∈ [5.924500, 15.000100] and AveRooms ∈ [4.549819, 62.422218] and AveBedrms ∈ [0.953696, 14.111111] and Population ∈ [3.000000, 1469.000000] and Latitude ∈ [32.549999, 37.940437]  — Fid 0.921, Cov 0.229, n=267
  - seed 43 (k=1)
    1. MedInc ∈ [5.887044, 15.000101] and AveRooms ∈ [5.240104, 132.533340] and Latitude ∈ [33.599998, 37.580002] and Longitude ∈ [-122.470001, -114.309998]  — Fid 0.860, Cov 0.270, n=315
  - seed 44 (k=1)
    1. MedInc ∈ [5.955995, 15.000101] and HouseAge ∈ [26.000000, 51.999996]  — Fid 0.885, Cov 0.183, n=208


**cart**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 3.198700 and Latitude > 34.445000  — Fid 0.694, Cov 0.530, n=863
  - seed 43 (k=1)
    1. MedInc <= 3.187950 and Latitude > 34.455000  — Fid 0.690, Cov 0.536, n=899
  - seed 44 (k=1)
    1. MedInc <= 5.590200 and MedInc <= 2.787150 and Latitude > 34.465000  — Fid 0.772, Cov 0.460, n=676

- `class_1`
  - seed 42 (k=1)
    1. MedInc <= 3.198700 and Latitude <= 34.445000  — Fid 0.476, Cov 0.330, n=908
  - seed 43 (k=1)
    1. MedInc <= 3.187950 and Latitude <= 34.455000  — Fid 0.375, Cov 0.303, n=839
  - seed 44 (k=1)
    1. MedInc <= 5.590200 and MedInc <= 2.787150 and Latitude <= 34.465000  — Fid 0.463, Cov 0.224, n=635

- `class_2`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc <= 5.764050  — Fid 0.455, Cov 0.636, n=1815
  - seed 43 (k=1)
    1. MedInc > 3.187950 and MedInc <= 5.776200  — Fid 0.405, Cov 0.664, n=1888
  - seed 44 (k=1)
    1. MedInc <= 5.590200 and MedInc > 2.787150  — Fid 0.397, Cov 0.726, n=2239

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc > 5.764050  — Fid 0.867, Cov 0.430, n=535
  - seed 43 (k=1)
    1. MedInc > 3.187950 and MedInc > 5.776200  — Fid 0.826, Cov 0.412, n=500
  - seed 44 (k=1)
    1. MedInc > 5.590200  — Fid 0.763, Cov 0.443, n=577


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. Latitude > 34.27 and Longitude > -121.80 and MedInc <= 2.57  — Fid 0.909, Cov 0.277, n=352
  - seed 43 (k=1)
    1. Latitude > 34.24 and MedInc <= 3.54 and Longitude > -121.81 and AveOccup > 2.43 and AveRooms <= 6.04 and 1.01 < AveBedrms <= 1.10  — Fid 0.782, Cov 0.244, n=340
  - seed 44 (k=1)
    1. Latitude > 34.25 and MedInc <= 2.56 and Longitude > -121.79  — Fid 0.880, Cov 0.287, n=368

- `class_1`
  - seed 42 (k=1)
    1. Latitude <= 34.27 and 2.57 < MedInc <= 3.55 and AveRooms <= 5.25 and AveOccup > 3.28 and Population > 1167.00 and AveBedrms <= 1.10 and HouseAge > 29.00  — Fid 0.710, Cov 0.095, n=162
  - seed 43 (k=1)
    1. Longitude <= -118.48 and 2.57 < MedInc <= 3.54  — Fid 0.209, Cov 0.250, n=1074
  - seed 44 (k=1)
    1. Latitude <= 33.93 and 2.56 < MedInc <= 4.74 and AveOccup > 2.82 and AveRooms <= 6.05 and AveBedrms <= 1.10  — Fid 0.525, Cov 0.107, n=257

- `class_2`
  - seed 42 (k=1)
    1. AveOccup > 2.44 and MedInc > 3.55 and Longitude <= -121.80  — Fid 0.430, Cov 0.125, n=405
  - seed 43 (k=1)
    1. AveOccup <= 3.28 and Longitude > -118.48 and 2.57 < MedInc <= 4.74 and Latitude <= 37.72  — Fid 0.428, Cov 0.339, n=1008
  - seed 44 (k=1)
    1. 2.43 < AveOccup <= 3.29 and MedInc > 3.53 and 18.00 < HouseAge <= 37.00 and Latitude <= 37.72 and AveRooms <= 6.05 and Population > 782.75 and Longitude <= -118.01  — Fid 0.609, Cov 0.116, n=322

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and AveOccup <= 2.44 and Longitude <= -121.80 and HouseAge > 18.00  — Fid 0.952, Cov 0.056, n=62
  - seed 43 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.43 and Latitude <= 34.24 and HouseAge > 29.00 and -121.81 < Longitude <= -118.00  — Fid 0.929, Cov 0.040, n=42
  - seed 44 (k=1)
    1. MedInc > 4.74 and AveRooms > 6.05 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -118.01 and 782.75 < Population <= 1706.25 and HouseAge > 18.00 and 1.01 < AveBedrms <= 1.10  — Fid 0.824, Cov 0.069, n=85


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. Latitude > 34.27 and Longitude > -121.80 and MedInc <= 2.57  — Fid 0.909, Cov 0.277, n=352
  - seed 43 (k=1)
    1. Latitude > 34.24 and MedInc <= 3.54 and Longitude > -121.81 and AveOccup > 2.43 and AveRooms <= 6.04 and 1.01 < AveBedrms <= 1.10  — Fid 0.782, Cov 0.244, n=340
  - seed 44 (k=1)
    1. Latitude > 34.25 and MedInc <= 2.56 and Longitude > -121.79  — Fid 0.880, Cov 0.287, n=368

- `class_1`
  - seed 42 (k=1)
    1. Latitude <= 34.27 and 2.57 < MedInc <= 3.55 and AveRooms <= 5.25 and AveOccup > 3.28 and Population > 1167.00 and AveBedrms <= 1.10 and HouseAge > 29.00  — Fid 0.710, Cov 0.095, n=162
  - seed 43 (k=1)
    1. Longitude <= -118.48 and 2.57 < MedInc <= 3.54  — Fid 0.209, Cov 0.250, n=1074
  - seed 44 (k=1)
    1. Latitude <= 33.93 and 2.56 < MedInc <= 4.74 and AveOccup > 2.82 and AveRooms <= 6.05 and AveBedrms <= 1.10  — Fid 0.525, Cov 0.107, n=257

- `class_2`
  - seed 42 (k=1)
    1. AveOccup > 2.44 and MedInc > 3.55 and Longitude <= -121.80  — Fid 0.430, Cov 0.125, n=405
  - seed 43 (k=1)
    1. AveOccup <= 3.28 and Longitude > -118.48 and 2.57 < MedInc <= 4.74 and Latitude <= 37.72  — Fid 0.428, Cov 0.339, n=1008
  - seed 44 (k=1)
    1. 2.43 < AveOccup <= 3.29 and MedInc > 3.53 and 18.00 < HouseAge <= 37.00 and Latitude <= 37.72 and AveRooms <= 6.05 and Population > 782.75 and Longitude <= -118.01  — Fid 0.609, Cov 0.116, n=322

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and AveOccup <= 2.44 and Longitude <= -121.80 and HouseAge > 18.00  — Fid 0.952, Cov 0.056, n=62
  - seed 43 (k=1)
    1. AveOccup <= 2.43 and MedInc > 2.57 and HouseAge > 18.00 and Latitude <= 37.72 and Longitude <= -118.00 and AveRooms > 4.45  — Fid 0.625, Cov 0.137, n=200
  - seed 44 (k=1)
    1. MedInc > 4.74 and AveRooms > 6.05 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -118.01 and 782.75 < Population <= 1706.25 and HouseAge > 18.00 and 1.01 < AveBedrms <= 1.10  — Fid 0.824, Cov 0.069, n=85


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 160  — Fid 0.985, Cov 0.062, n=67
  - seed 43 (k=1)
    1. random box 249  — Fid 0.987, Cov 0.072, n=75
  - seed 44 (k=1)
    1. random box 37  — Fid 1.000, Cov 0.088, n=94

- `class_1`
  - seed 42 (k=1)
    1. random box 7  — Fid 0.897, Cov 0.023, n=39
  - seed 43 (k=1)
    1. random box 198  — Fid 0.598, Cov 0.058, n=127
  - seed 44 (k=1)
    1. random box 187  — Fid 0.923, Cov 0.007, n=13

- `class_2`
  - seed 42 (k=1)
    1. random box 239  — Fid 0.577, Cov 0.218, n=482
  - seed 43 (k=1)
    1. random box 212  — Fid 0.704, Cov 0.021, n=54
  - seed 44 (k=1)
    1. random box 150  — Fid 0.715, Cov 0.117, n=253

- `class_3`
  - seed 42 (k=1)
    1. random box 221  — Fid 0.874, Cov 0.096, n=111
  - seed 43 (k=1)
    1. random box 121  — Fid 0.940, Cov 0.047, n=50
  - seed 44 (k=1)
    1. random box 188  — Fid 0.983, Cov 0.057, n=60


### housing — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=5)
    1. MedInc ∈ [0.499900, 3.139491] and HouseAge ∈ [14.874919, 35.000000] and AveOccup ∈ [2.232897, 502.461578] and Latitude ∈ [34.249001, 40.760330]  — Fid 0.883, Cov 0.253, n=367
    2. MedInc ∈ [0.499900, 2.827969] and AveRooms ∈ [3.971788, 62.422218] and AveBedrms ∈ [0.935340, 1.092194] and Population ∈ [630.341125, 16304.999023] and AveOccup ∈ [2.224380, 502.461578] and Longitude ∈ [-124.349991, -119.096008]  — Fid 0.922, Cov 0.159, n=206
    3. MedInc ∈ [0.499900, 3.139503] and HouseAge ∈ [24.000000, 35.000000] and AveOccup ∈ [2.232894, 502.461578] and Latitude ∈ [34.249001, 41.030151]  — Fid 0.905, Cov 0.151, n=210
    4. MedInc ∈ [0.499900, 2.832603] and AveRooms ∈ [4.721093, 62.422218] and AveBedrms ∈ [1.002617, 1.092001] and Population ∈ [396.000000, 16304.999023] and AveOccup ∈ [2.108800, 4.292845] and Longitude ∈ [-124.349991, -120.480003]  — Fid 0.956, Cov 0.066, n=91
    5. MedInc ∈ [0.499900, 2.826523] and HouseAge ∈ [1.000000, 46.000000] and AveRooms ∈ [4.760034, 62.422218] and AveBedrms ∈ [1.053951, 1.092015] and Population ∈ [634.000000, 16304.999023] and Longitude ∈ [-124.349991, -118.946472]  — Fid 1.000, Cov 0.044, n=50
  - seed 43 (k=5)
    1. MedInc ∈ [0.499900, 3.125000] and HouseAge ∈ [12.999999, 52.000000]  — Fid 0.576, Cov 0.721, n=1539
    2. AveRooms ∈ [4.253340, 6.199045] and AveBedrms ∈ [1.019608, 1.091060] and Population ∈ [793.000000, 1739.102905] and AveOccup ∈ [0.692308, 4.109745] and Latitude ∈ [34.913776, 36.821499]  — Fid 0.872, Cov 0.039, n=47
    3. MedInc ∈ [0.499900, 2.567331] and HouseAge ∈ [1.000000, 48.057205] and AveRooms ∈ [3.865956, 5.485728] and AveBedrms ∈ [1.117661, 1.363717] and AveOccup ∈ [0.692308, 4.128926] and Longitude ∈ [-117.680000, -117.138863]  — Fid 0.941, Cov 0.013, n=17
    4. MedInc ∈ [0.499900, 3.568838] and HouseAge ∈ [1.000000, 35.000000] and AveRooms ∈ [3.835562, 16.886786] and AveBedrms ∈ [1.091060, 3.692989] and AveOccup ∈ [0.692308, 3.098574] and Longitude ∈ [-120.124359, -119.050003]  — Fid 0.860, Cov 0.026, n=43
    5. HouseAge ∈ [27.999998, 29.000000] and Latitude ∈ [32.549995, 39.729996] and Longitude ∈ [-121.652870, -119.790001]  — Fid 0.583, Cov 0.012, n=24
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [2.424211, 4.704300] and HouseAge ∈ [30.999998, 52.000000] and AveRooms ∈ [0.846154, 4.056044] and AveBedrms ∈ [1.014826, 14.111111] and Population ∈ [949.690186, 16304.999023] and AveOccup ∈ [3.206178, 502.461578] and Latitude ∈ [33.937145, 34.660000] and Longitude ∈ [-124.349991, -118.160004]  — Fid 1.000, Cov 0.015, n=19
  - seed 43 (k=2)
    1. MedInc ∈ [2.225480, 4.375000] and AveBedrms ∈ [0.962676, 34.066666] and AveOccup ∈ [2.311464, 502.461548]  — Fid 0.351, Cov 0.560, n=1546
    2. MedInc ∈ [2.389097, 15.000101] and HouseAge ∈ [29.000000, 44.090786] and AveRooms ∈ [0.846154, 4.524498] and AveBedrms ∈ [0.333333, 1.030509] and Population ∈ [459.787659, 2140.000000] and Latitude ∈ [33.939999, 34.290001]  — Fid 0.487, Cov 0.019, n=39
  - seed 44: _not run_

- `class_2`
  - seed 42 (k=2)
    1. MedInc ∈ [4.512644, 6.854567] and HouseAge ∈ [21.000000, 52.000000] and AveRooms ∈ [4.285494, 6.612618] and AveBedrms ∈ [0.966175, 14.111111] and AveOccup ∈ [3.192369, 502.461578]  — Fid 0.855, Cov 0.037, n=69
    2. MedInc ∈ [0.499900, 5.131864] and HouseAge ∈ [17.778315, 52.000000] and AveBedrms ∈ [0.444444, 1.014778] and AveOccup ∈ [2.254367, 502.461578] and Latitude ∈ [32.549999, 33.897167] and Longitude ∈ [-118.110001, -117.967789]  — Fid 0.812, Cov 0.013, n=16
  - seed 43 (k=3)
    1. MedInc ∈ [3.008879, 5.934507] and HouseAge ∈ [16.999998, 45.000000] and AveRooms ∈ [0.846154, 7.287680]  — Fid 0.456, Cov 0.507, n=1433
    2. MedInc ∈ [3.665660, 15.000101] and HouseAge ∈ [21.000000, 38.000000] and AveRooms ∈ [0.846154, 6.153610] and Population ∈ [3.000000, 690.834778] and AveOccup ∈ [2.769115, 3.612663] and Latitude ∈ [32.549995, 37.959999]  — Fid 0.815, Cov 0.012, n=27
    3. MedInc ∈ [3.826011, 5.594357] and HouseAge ∈ [25.000000, 52.000000] and AveRooms ∈ [4.273380, 132.533340] and AveBedrms ∈ [1.010607, 1.112964] and Population ∈ [544.999939, 35682.003906] and AveOccup ∈ [3.173123, 502.461548]  — Fid 0.729, Cov 0.032, n=59
  - seed 44: _not run_

- `class_3`
  - seed 42 (k=5)
    1. MedInc ∈ [5.924500, 10.229071] and HouseAge ∈ [14.887274, 52.000000] and AveBedrms ∈ [0.953049, 14.111111] and Population ∈ [3.000000, 2711.179688] and AveOccup ∈ [2.182507, 3.839206] and Latitude ∈ [32.549999, 37.380001]  — Fid 0.944, Cov 0.171, n=198
    2. MedInc ∈ [5.367700, 9.220963] and AveRooms ∈ [5.561594, 8.657566] and AveBedrms ∈ [0.444444, 1.099567] and Population ∈ [495.000000, 16304.999023] and AveOccup ∈ [0.692308, 2.924684]  — Fid 0.927, Cov 0.189, n=232
    3. MedInc ∈ [5.367700, 8.617228] and AveRooms ∈ [5.134445, 8.213058] and AveBedrms ∈ [0.444444, 1.099567] and Population ∈ [495.000000, 16304.999023] and AveOccup ∈ [0.692308, 2.898188]  — Fid 0.930, Cov 0.193, n=242
    4. MedInc ∈ [5.367700, 12.543919] and HouseAge ∈ [10.935076, 52.000000] and AveBedrms ∈ [1.010427, 14.111111] and AveOccup ∈ [2.074328, 3.178025] and Latitude ∈ [32.549999, 37.709999]  — Fid 0.954, Cov 0.170, n=196
    5. MedInc ∈ [6.486750, 15.000100] and HouseAge ∈ [13.904070, 52.000000] and AveBedrms ∈ [0.988432, 14.111111] and Population ∈ [3.000000, 2405.566895] and AveOccup ∈ [2.421327, 3.322512] and Latitude ∈ [32.549999, 37.040630]  — Fid 1.000, Cov 0.086, n=90
  - seed 43 (k=5)
    1. MedInc ∈ [5.903900, 15.000101] and AveRooms ∈ [5.933887, 132.533340] and AveOccup ∈ [2.213753, 3.737775]  — Fid 0.931, Cov 0.338, n=394
    2. MedInc ∈ [5.903900, 15.000101] and AveRooms ∈ [5.539680, 9.867174] and AveOccup ∈ [2.444720, 3.221675]  — Fid 0.948, Cov 0.234, n=270
    3. MedInc ∈ [5.903900, 15.000101] and HouseAge ∈ [1.000000, 37.295864] and AveRooms ∈ [5.933883, 132.533340] and AveBedrms ∈ [0.959359, 34.066666] and Population ∈ [657.999939, 35682.003906] and AveOccup ∈ [2.414443, 502.461548]  — Fid 0.900, Cov 0.207, n=251
    4. MedInc ∈ [5.903900, 15.000101] and HouseAge ∈ [14.886950, 52.000000] and AveRooms ∈ [4.133904, 132.533340] and AveBedrms ∈ [1.006938, 34.066666] and Population ∈ [3.000000, 2025.311401] and Longitude ∈ [-121.589996, -114.309998]  — Fid 0.923, Cov 0.111, n=130
    5. MedInc ∈ [5.903900, 15.000101] and HouseAge ∈ [14.000000, 52.000000] and AveRooms ∈ [4.655259, 132.533340] and AveBedrms ∈ [1.026954, 34.066666] and Population ∈ [3.000000, 1478.500244] and Longitude ∈ [-118.514999, -114.309998]  — Fid 0.955, Cov 0.062, n=67
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 2.564511] and AveBedrms ∈ [1.068926, 14.111111] and Latitude ∈ [34.240002, 41.880001] and Longitude ∈ [-121.619995, -114.489990]  — Fid 0.960, Cov 0.140, n=174
  - seed 43 (k=1)
    1. MedInc ∈ [0.499900, 2.814031] and AveRooms ∈ [4.253335, 132.533340] and Latitude ∈ [33.812988, 41.950001]  — Fid 0.858, Cov 0.456, n=696
  - seed 44 (k=1)
    1. MedInc ∈ [0.499900, 2.792080] and AveRooms ∈ [4.237594, 141.909088] and Latitude ∈ [33.770000, 41.950005]  — Fid 0.861, Cov 0.475, n=693

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [3.012578, 15.000100] and AveBedrms ∈ [0.444444, 1.038734] and AveOccup ∈ [4.307551, 502.461578] and Longitude ∈ [-120.469994, -114.489990]  — Fid 0.889, Cov 0.024, n=36
  - seed 43 (k=1)
    1. MedInc ∈ [2.013499, 15.000101] and HouseAge ∈ [16.999998, 52.000000] and AveRooms ∈ [3.294988, 5.484581] and AveOccup ∈ [4.948204, 502.461548] and Longitude ∈ [-121.811623, -114.309998]  — Fid 0.839, Cov 0.017, n=31
  - seed 44 (k=1)
    1. HouseAge ∈ [23.000000, 51.999996] and AveRooms ∈ [0.846154, 6.121200] and AveBedrms ∈ [1.008751, 25.636364] and AveOccup ∈ [3.524548, 1243.333374] and Latitude ∈ [32.540001, 34.435696] and Longitude ∈ [-122.120003, -114.470001]  — Fid 0.654, Cov 0.130, n=263

- `class_2`
  - seed 42 (k=1)
    1. MedInc ∈ [3.035674, 15.000100] and AveRooms ∈ [0.846154, 5.274409] and AveOccup ∈ [2.560740, 3.393601] and Latitude ∈ [32.549999, 37.958115] and Longitude ∈ [-119.106606, -118.089989]  — Fid 0.854, Cov 0.054, n=103
  - seed 43 (k=1)
    1. MedInc ∈ [4.347200, 6.176807] and HouseAge ∈ [1.000000, 39.000000] and AveRooms ∈ [4.950607, 6.022175] and AveBedrms ∈ [0.908449, 1.180356] and AveOccup ∈ [2.532744, 502.461548]  — Fid 0.683, Cov 0.118, n=221
  - seed 44 (k=1)
    1. MedInc ∈ [0.499900, 5.462598] and HouseAge ∈ [10.000000, 51.999996] and AveRooms ∈ [0.846154, 5.452035] and AveOccup ∈ [2.018638, 3.125831] and Latitude ∈ [34.145229, 34.380001] and Longitude ∈ [-121.680481, -117.860001]  — Fid 0.838, Cov 0.061, n=99

- `class_3`
  - seed 42 (k=1)
    1. MedInc ∈ [5.367700, 15.000100] and HouseAge ∈ [1.000000, 52.000000] and AveRooms ∈ [4.021782, 62.422218] and Population ∈ [3.000000, 3550.383789] and AveOccup ∈ [0.692308, 3.116878]  — Fid 0.903, Cov 0.384, n=487
  - seed 43 (k=1)
    1. MedInc ∈ [5.367352, 15.000101] and HouseAge ∈ [23.000000, 52.000000] and AveBedrms ∈ [0.333333, 1.165973] and AveOccup ∈ [0.692308, 3.002718] and Longitude ∈ [-124.349998, -118.400002]  — Fid 0.977, Cov 0.155, n=177
  - seed 44 (k=1)
    1. MedInc ∈ [5.956371, 15.000101] and AveOccup ∈ [2.136319, 1243.333374] and Latitude ∈ [32.540001, 38.590084]  — Fid 0.949, Cov 0.358, n=434


**cart**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 3.198700  — Fid 0.557, Cov 0.821, n=1771
  - seed 43 (k=1)
    1. MedInc <= 3.133950  — Fid 0.583, Cov 0.782, n=1662
  - seed 44 (k=1)
    1. MedInc <= 3.176250 and Latitude > 34.455000  — Fid 0.787, Cov 0.544, n=892

- `class_1`
  - seed 42: _not run_
  - seed 43: _not run_
  - seed 44 (k=1)
    1. MedInc <= 3.176250 and Latitude <= 34.455000  — Fid 0.403, Cov 0.299, n=836

- `class_2`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc <= 5.776350 and AveOccup > 2.395127  — Fid 0.477, Cov 0.515, n=1364
  - seed 43 (k=1)
    1. MedInc > 3.133950 and MedInc <= 5.951500 and AveOccup > 2.418130  — Fid 0.485, Cov 0.553, n=1500
  - seed 44 (k=1)
    1. MedInc > 3.176250 and MedInc <= 5.617250  — Fid 0.423, Cov 0.630, n=1833

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc > 5.776350  — Fid 0.906, Cov 0.427, n=531
  - seed 43 (k=1)
    1. MedInc > 3.133950 and MedInc > 5.951500  — Fid 0.936, Cov 0.380, n=456
  - seed 44 (k=1)
    1. MedInc > 3.176250 and MedInc > 5.617250  — Fid 0.888, Cov 0.435, n=565


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 2.57 and Latitude > 37.72  — Fid 0.897, Cov 0.210, n=301
  - seed 43 (k=1)
    1. MedInc <= 2.57 and Latitude > 34.24 and Longitude > -121.81  — Fid 0.962, Cov 0.303, n=394
  - seed 44 (k=1)
    1. MedInc <= 2.56 and Latitude > 37.72 and Longitude > -121.79  — Fid 0.978, Cov 0.111, n=139

- `class_1`
  - seed 42 (k=1)
    1. 3.55 < MedInc <= 4.78 and AveOccup > 2.82 and Longitude > -118.02 and 33.93 < Latitude <= 34.27 and HouseAge <= 18.00 and AveRooms <= 6.09 and Population > 1729.25 and AveBedrms <= 1.10  — Fid 0.759, Cov 0.016, n=29
  - seed 43 (k=1)
    1. AveOccup > 3.28 and Longitude > -118.48 and MedInc <= 4.74 and AveBedrms <= 1.10 and 33.93 < Latitude <= 34.24 and AveRooms <= 4.45 and Population > 1166.00  — Fid 0.698, Cov 0.091, n=159
  - seed 44 (k=1)
    1. Latitude > 37.72 and HouseAge <= 18.00 and 2.56 < MedInc <= 4.74 and AveRooms > 5.22 and AveOccup > 2.82 and AveBedrms <= 1.10 and -121.79 < Longitude <= -118.01 and Population <= 782.75  — Fid 0.600, Cov 0.003, n=5

- `class_2`
  - seed 42 (k=1)
    1. 33.93 < Latitude <= 34.27 and 2.57 < MedInc <= 4.78 and AveRooms <= 6.09 and 2.44 < AveOccup <= 3.28 and 18.00 < HouseAge <= 37.00 and AveBedrms <= 1.10  — Fid 0.494, Cov 0.208, n=526
  - seed 43 (k=1)
    1. -121.81 < Longitude <= -118.00 and 3.54 < MedInc <= 4.74 and 2.82 < AveOccup <= 3.28 and HouseAge > 18.00 and AveBedrms <= 1.10 and Population > 788.00  — Fid 0.409, Cov 0.235, n=767
  - seed 44 (k=1)
    1. Latitude <= 34.25 and 2.56 < MedInc <= 3.53 and Longitude <= -118.01 and 2.43 < AveOccup <= 3.29 and AveRooms <= 4.44  — Fid 0.600, Cov 0.077, n=200

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and Longitude <= -121.80 and AveOccup <= 3.28 and AveRooms > 6.09 and Latitude <= 37.72  — Fid 0.977, Cov 0.080, n=88
  - seed 43 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -121.81  — Fid 0.903, Cov 0.078, n=93
  - seed 44 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -118.49  — Fid 0.889, Cov 0.128, n=171


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 2.57 and Latitude > 37.72  — Fid 0.897, Cov 0.210, n=301
  - seed 43 (k=1)
    1. MedInc <= 2.57 and Latitude > 34.24 and Longitude > -121.81  — Fid 0.962, Cov 0.303, n=394
  - seed 44 (k=1)
    1. Latitude > 34.25 and MedInc <= 4.74 and -121.79 < Longitude <= -118.49 and AveRooms <= 6.05 and AveOccup > 2.43 and Population <= 1706.25  — Fid 0.669, Cov 0.346, n=643

- `class_1`
  - seed 42 (k=1)
    1. AveOccup > 3.28 and -118.52 < Longitude <= -118.02 and AveRooms <= 5.25 and 2.57 < MedInc <= 3.55 and Latitude <= 34.27 and HouseAge > 29.00 and Population > 786.00  — Fid 0.777, Cov 0.123, n=202
  - seed 43 (k=1)
    1. AveOccup > 3.28 and Longitude > -118.48 and MedInc <= 4.74 and AveBedrms <= 1.10 and 33.93 < Latitude <= 34.24 and AveRooms <= 4.45 and Population > 1166.00  — Fid 0.698, Cov 0.091, n=159
  - seed 44 (k=1)
    1. AveOccup > 3.29 and Longitude > -118.01 and 3.53 < MedInc <= 4.74 and 1161.00 < Population <= 1706.25 and 33.93 < Latitude <= 34.25 and AveRooms <= 6.05 and 1.01 < AveBedrms <= 1.10 and HouseAge <= 37.00  — Fid 0.507, Cov 0.024, n=71

- `class_2`
  - seed 42 (k=1)
    1. 33.93 < Latitude <= 34.27 and 2.57 < MedInc <= 4.78 and AveRooms <= 6.09 and 2.44 < AveOccup <= 3.28 and 18.00 < HouseAge <= 37.00 and AveBedrms <= 1.10  — Fid 0.494, Cov 0.208, n=526
  - seed 43 (k=1)
    1. Latitude <= 34.24 and MedInc > 2.57 and AveRooms <= 6.04 and 2.43 < AveOccup <= 3.28 and HouseAge <= 37.00 and AveBedrms <= 1.10  — Fid 0.564, Cov 0.211, n=479
  - seed 44 (k=1)
    1. Latitude <= 34.25 and 2.56 < MedInc <= 3.53 and Longitude <= -118.01 and 2.43 < AveOccup <= 3.29 and AveRooms <= 4.44  — Fid 0.600, Cov 0.077, n=200

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and Longitude <= -121.80 and AveOccup <= 3.28 and AveRooms > 6.09 and Latitude <= 37.72  — Fid 0.977, Cov 0.080, n=88
  - seed 43 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -121.81  — Fid 0.903, Cov 0.078, n=93
  - seed 44 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.82 and Latitude <= 37.72 and Longitude <= -118.49  — Fid 0.889, Cov 0.128, n=171


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 239  — Fid 0.985, Cov 0.113, n=132
  - seed 43 (k=1)
    1. random box 87  — Fid 0.967, Cov 0.101, n=121
  - seed 44 (k=1)
    1. random box 71  — Fid 1.000, Cov 0.065, n=72

- `class_1`
  - seed 42 (k=1)
    1. random box 130  — Fid 0.667, Cov 0.004, n=18
  - seed 43 (k=1)
    1. random box 10  — Fid 0.870, Cov 0.014, n=23
  - seed 44 (k=1)
    1. random box 138  — Fid 0.889, Cov 0.036, n=63

- `class_2`
  - seed 42 (k=1)
    1. random box 0  — Fid 0.678, Cov 0.113, n=202
  - seed 43 (k=1)
    1. random box 235  — Fid 0.678, Cov 0.118, n=211
  - seed 44 (k=1)
    1. random box 78  — Fid 0.638, Cov 0.099, n=224

- `class_3`
  - seed 42 (k=1)
    1. random box 41  — Fid 1.000, Cov 0.058, n=64
  - seed 43 (k=1)
    1. random box 178  — Fid 0.986, Cov 0.063, n=73
  - seed 44 (k=1)
    1. random box 255  — Fid 1.000, Cov 0.054, n=61



## uci_credit

### uci_credit — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=3)
    1. A10 = 't' and A5 = 'g'  — Fid 0.870, Cov 0.607, n=46
    2. A9 = 't' and A6 = 'q' and A5 = 'g'  — Fid 1.000, Cov 0.197, n=15
    3. A10 = 'f' and A9 = 't' and A5 = 'g'  — Fid 0.640, Cov 0.213, n=25
  - seed 43 (k=1)
    1. A13 = 'g'  — Fid 0.472, Cov 0.934, n=127
  - seed 44 (k=1)
    1. A9 = 't'  — Fid 0.816, Cov 0.934, n=76

- `class_1`
  - seed 42 (k=1)
    1. A9 = 'f'  — Fid 1.000, Cov 0.727, n=64
  - seed 43 (k=2)
    1. A10 = 'f'  — Fid 0.892, Cov 0.753, n=74
    2. A12 = 'f' and A11 ∈ [0.000000, 67.000000] and A10 = 't' and A9 = 'f'  — Fid 1.000, Cov 0.065, n=5
  - seed 44 (k=1)
    1. A9 = 'f'  — Fid 1.000, Cov 0.753, n=62


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. A13 = 'g' and A9 = 't'  — Fid 0.864, Cov 0.803, n=66
  - seed 43 (k=1)
    1. A13 = 'g' and A9 = 't'  — Fid 0.882, Cov 0.902, n=68
  - seed 44 (k=1)
    1. A13 = 'g' and A9 = 't'  — Fid 0.817, Cov 0.902, n=71

- `class_1`
  - seed 42 (k=1)
    1. A15 ∈ [0.000000, 100000.007812] and A11 ∈ [0.000000, 1.000000] and A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.636, n=56
  - seed 43 (k=1)
    1. A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.662, n=55
  - seed 44 (k=1)
    1. A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.532, n=44


**cart**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.500000  — Fid 0.824, Cov 0.869, n=74
  - seed 43 (k=1)
    1. A9 > 0.500000  — Fid 0.843, Cov 0.918, n=70
  - seed 44 (k=1)
    1. A9 > 0.500000  — Fid 0.811, Cov 0.918, n=74

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.714, n=63
  - seed 43 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.753, n=62


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.921, Cov 0.557, n=38
  - seed 43 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.437, Cov 0.984, n=135
  - seed 44 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.964, Cov 0.393, n=28

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00 and A11 <= 3.00  — Fid 1.000, Cov 0.688, n=61
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.753, n=62


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.445, Cov 1.000, n=137
  - seed 43 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.437, Cov 0.984, n=135
  - seed 44 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.964, Cov 0.393, n=28

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00 and A11 <= 3.00  — Fid 1.000, Cov 0.688, n=61
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.753, n=62


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 105  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 197  — Fid 1.000, Cov 0.033, n=3
  - seed 44 (k=1)
    1. random box 117  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 42  — Fid 1.000, Cov 0.052, n=5
  - seed 44 (k=1)
    1. random box 108  — Fid 1.000, Cov 0.013, n=1


### uci_credit — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. A9 = 't'  — Fid 0.919, Cov 0.869, n=74
  - seed 43 (k=4)
    1. A10 = 't' and A9 = 't' and A4 = 'u' and A1 = 'b'  — Fid 1.000, Cov 0.475, n=32
    2. A9 = 't' and A7 = 'o'  — Fid 0.932, Cov 0.607, n=44
    3. A11 ∈ [5.000000, 10.141304] and A10 = 't' and A9 = 't' and A8 ∈ [0.375000, 28.500000]  — Fid 1.000, Cov 0.180, n=11
    4. A9 = 't' and A7 = 'h'  — Fid 0.929, Cov 0.180, n=14
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A9 = 'f'  — Fid 0.953, Cov 0.727, n=64
  - seed 43 (k=3)
    1. A9 = 'f' and A4 = 'u'  — Fid 1.000, Cov 0.558, n=47
    2. A12 = 't' and A11 ∈ [0.000000, 2.000000] and A9 = 'f'  — Fid 1.000, Cov 0.338, n=28
    3. A9 = 'f' and A4 = 'y'  — Fid 1.000, Cov 0.247, n=19
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. A10 = 't' and A9 = 't'  — Fid 1.000, Cov 0.639, n=44
  - seed 43 (k=1)
    1. A15 ∈ [252.499939, 50000.000000] and A13 = 'g' and A9 = 't' and A8 ∈ [0.000000, 5.000000] and A7 = 'o'  — Fid 1.000, Cov 0.279, n=18
  - seed 44 (k=1)
    1. A10 = 't' and A9 = 't'  — Fid 1.000, Cov 0.607, n=41

- `class_1`
  - seed 42 (k=1)
    1. A13 = 'g' and A10 = 'f' and A9 = 'f'  — Fid 0.977, Cov 0.519, n=43
  - seed 43 (k=1)
    1. A15 ∈ [-0.000061, 552.000000]  — Fid 0.627, Cov 0.948, n=110
  - seed 44 (k=1)
    1. A13 = 'g' and A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.481, n=39


**cart**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.500000  — Fid 0.919, Cov 0.869, n=74
  - seed 43 (k=1)
    1. A9 > 0.500000  — Fid 0.900, Cov 0.918, n=70
  - seed 44 (k=1)
    1. A9 > 0.500000  — Fid 0.730, Cov 0.918, n=74

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.500000  — Fid 0.952, Cov 0.714, n=63
  - seed 43 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.753, n=62


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.518, Cov 1.000, n=137
  - seed 43 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.929, Cov 0.574, n=42
  - seed 44 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.964, Cov 0.393, n=28

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00  — Fid 0.952, Cov 0.714, n=63
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.753, n=62


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.518, Cov 1.000, n=137
  - seed 43 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.929, Cov 0.574, n=42
  - seed 44 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.964, Cov 0.393, n=28

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00  — Fid 0.952, Cov 0.714, n=63
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65
  - seed 44 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.753, n=62


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 147  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 247  — Fid 1.000, Cov 0.033, n=3
  - seed 44 (k=1)
    1. random box 117  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 154  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 140  — Fid 1.000, Cov 0.039, n=4
  - seed 44 (k=1)
    1. random box 159  — Fid 1.000, Cov 0.026, n=2



## uci_adult

### uci_adult — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=5)
    1. age ∈ [16.999998, 49.000000] and fnlwgt ∈ [13769.015625, 420088.937500]  — Fid 0.823, Cov 0.776, n=7340
    2. age ∈ [16.999998, 49.000000] and education-num ∈ [6.000000, 16.000000]  — Fid 0.817, Cov 0.767, n=7308
    3. age ∈ [16.999998, 54.131168] and fnlwgt ∈ [13769.015625, 327389.625000]  — Fid 0.805, Cov 0.778, n=7511
    4. age ∈ [16.999998, 57.999996] and education = 'HS-grad' and education-num ∈ [8.000000, 13.000000]  — Fid 0.930, Cov 0.327, n=2886
    5. age ∈ [19.933697, 90.000000] and fnlwgt ∈ [130021.000000, 302963.812500] and education-num ∈ [9.000001, 13.000000] and marital-status = 'Never-married'  — Fid 0.982, Cov 0.177, n=1369
  - seed 43 (k=5)
    1. education-num ∈ [6.897139, 13.000001] and native-country = 'United-States'  — Fid 0.838, Cov 0.789, n=7531
    2. hours-per-week ∈ [20.889063, 99.000000]  — Fid 0.793, Cov 0.888, n=8893
    3. education-num ∈ [1.000000, 13.000001] and relationship = 'Not-in-family'  — Fid 0.970, Cov 0.278, n=2251
    4. education-num ∈ [1.000000, 13.000001] and relationship = 'Own-child'  — Fid 0.995, Cov 0.195, n=1468
    5. fnlwgt ∈ [12285.000000, 323985.250000] and education-num ∈ [9.000000, 14.010363] and sex = 'Female' and native-country = 'United-States'  — Fid 0.921, Cov 0.286, n=2370
  - seed 44 (k=5)
    1. age ∈ [16.999998, 49.000000] and education-num ∈ [6.000000, 16.000000]  — Fid 0.835, Cov 0.765, n=7306
    2. age ∈ [23.999998, 90.000000] and education = 'HS-grad'  — Fid 0.944, Cov 0.306, n=2792
    3. age ∈ [23.999998, 68.026024] and education-num ∈ [4.954468, 16.000000] and relationship = 'Not-in-family'  — Fid 0.951, Cov 0.242, n=2026
    4. age ∈ [18.960342, 57.999996] and education-num ∈ [6.899023, 16.000000] and relationship = 'Own-child'  — Fid 0.995, Cov 0.162, n=1224
    5. age ∈ [19.931768, 90.000000] and education = 'Some-college'  — Fid 0.906, Cov 0.211, n=1950

- `class_1`
  - seed 42 (k=5)
    1. age ∈ [35.000000, 90.000000] and education = 'Masters' and relationship = 'Husband'  — Fid 0.947, Cov 0.076, n=227
    2. age ∈ [35.000000, 90.000000] and education = 'Bachelors' and relationship = 'Husband'  — Fid 0.891, Cov 0.177, n=568
    3. workclass = 'Private' and fnlwgt ∈ [108140.007812, 1490400.000000] and education = 'Bachelors' and education-num ∈ [10.000001, 13.000000] and marital-status = 'Married-civ-spouse'  — Fid 0.850, Cov 0.124, n=420
    4. fnlwgt ∈ [13769.015625, 194025.218750] and education = 'Preschool' and relationship = 'Husband' and capital-gain ∈ [-0.000122, 667.330811] and capital-loss ∈ [0.000000, 62.021965] and native-country = 'United-States'  — Fid 0.868, Cov 0.012, n=38
    5. age ∈ [48.320023, 90.000000] and education = 'Doctorate' and marital-status = 'Married-civ-spouse' and relationship = 'Husband' and race = 'White' and sex = 'Male' and hours-per-week ∈ [45.384628, 99.000000] and native-country = 'United-States'  — Fid 1.000, Cov 0.006, n=13
  - seed 43 (k=5)
    1. relationship = 'Husband' and capital-gain ∈ [0.000000, 650.514893] and capital-loss ∈ [1887.000000, 1980.511841]  — Fid 0.926, Cov 0.056, n=135
    2. education = 'Bachelors' and relationship = 'Husband' and capital-loss ∈ [0.000000, 143.798462] and hours-per-week ∈ [40.000000, 99.000000]  — Fid 0.790, Cov 0.184, n=632
    3. workclass = 'Private' and education = 'Bachelors' and occupation = 'Sales' and relationship = 'Husband' and sex = 'Male' and capital-loss ∈ [0.000000, 4356.000000] and hours-per-week ∈ [50.000000, 99.000000] and native-country = 'United-States'  — Fid 0.902, Cov 0.013, n=41
    4. age ∈ [16.999998, 45.000000] and education = 'Bachelors' and relationship = 'Husband' and capital-loss ∈ [143.798462, 4356.000000] and native-country = 'United-States'  — Fid 0.929, Cov 0.017, n=42
    5. workclass = 'Private' and education-num ∈ [13.000001, 16.000000] and relationship = 'Husband' and capital-loss ∈ [0.000000, 1484.999878]  — Fid 0.814, Cov 0.195, n=634
  - seed 44 (k=4)
    1. age ∈ [38.000000, 53.000000] and workclass = 'Private' and education = 'Bachelors' and relationship = 'Husband'  — Fid 0.966, Cov 0.064, n=203
    2. age ∈ [40.999996, 49.000000] and workclass = 'Never-worked' and fnlwgt ∈ [12285.000000, 256393.187500] and education-num ∈ [13.000000, 16.000000] and marital-status = 'Married-civ-spouse' and hours-per-week ∈ [35.000000, 65.000000]  — Fid 1.000, Cov 0.017, n=47
    3. age ∈ [38.000000, 54.166569] and workclass = 'Self-emp-inc' and marital-status = 'Married-civ-spouse' and occupation = 'Prof-specialty' and race = 'White' and sex = 'Male' and capital-loss ∈ [0.000000, 4356.000000]  — Fid 0.957, Cov 0.009, n=23
    4. age ∈ [40.999996, 52.211208] and workclass = 'Self-emp-inc' and fnlwgt ∈ [12285.000000, 256393.187500] and education = 'Prof-school'  — Fid 0.938, Cov 0.006, n=16


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. workclass = 'Private'  — Fid 0.820, Cov 0.734, n=6959
  - seed 43 (k=1)
    1. workclass = 'Private'  — Fid 0.836, Cov 0.742, n=6986
  - seed 44 (k=1)
    1. workclass = 'Private' and capital-gain ∈ [-0.000122, -0.000122]  — Fid 0.882, Cov 0.708, n=6419

- `class_1`
  - seed 42 (k=1)
    1. age ∈ [16.999998, 58.098831] and education-num ∈ [12.000000, 14.088848] and marital-status = 'Married-civ-spouse' and capital-gain ∈ [-0.000122, 667.330811] and native-country = 'United-States'  — Fid 0.831, Cov 0.243, n=857
  - seed 43 (k=1)
    1. age ∈ [36.000000, 65.030380] and education = 'Masters' and education-num ∈ [1.000000, 14.088706] and occupation = 'Exec-managerial' and relationship = 'Husband' and race = 'White' and sex = 'Male' and capital-loss ∈ [0.000000, 597.680176] and native-country = 'United-States'  — Fid 1.000, Cov 0.025, n=63
  - seed 44 (k=1)
    1. education = 'Masters' and marital-status = 'Married-civ-spouse'  — Fid 0.923, Cov 0.100, n=300


**cart**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 12.500000  — Fid 0.898, Cov 0.824, n=7289
  - seed 43 (k=1)
    1. education-num <= 12.500000  — Fid 0.910, Cov 0.828, n=7298
  - seed 44 (k=1)
    1. education-num <= 12.500000  — Fid 0.932, Cov 0.826, n=7337

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.500000  — Fid 0.509, Cov 0.500, n=2478
  - seed 43: _not run_
  - seed 44: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 9.00  — Fid 0.947, Cov 0.519, n=4449
  - seed 43 (k=1)
    1. education-num <= 9.00  — Fid 0.944, Cov 0.517, n=4408
  - seed 44 (k=1)
    1. education-num <= 9.00  — Fid 0.962, Cov 0.520, n=4474

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.893, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and age > 37.00 and education > 11.00  — Fid 0.920, Cov 0.109, n=314
  - seed 44 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and age > 37.00 and marital-status <= 2.00 and sex > 0.00 and occupation > 3.00 and workclass <= 4.00  — Fid 0.927, Cov 0.197, n=602


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 9.00  — Fid 0.947, Cov 0.519, n=4449
  - seed 43 (k=1)
    1. capital-gain <= 0.00 and relationship > 0.00  — Fid 0.848, Cov 0.955, n=8920
  - seed 44 (k=1)
    1. education-num <= 9.00  — Fid 0.962, Cov 0.520, n=4474

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.893, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. capital-loss > 0.00 and relationship <= 0.00 and education-num > 12.00  — Fid 0.792, Cov 0.402, n=1330
  - seed 44 (k=1)
    1. capital-gain > 0.00 and education-num > 12.00 and relationship <= 0.00  — Fid 0.816, Cov 0.389, n=1302


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 131  — Fid 1.000, Cov 0.009, n=85
  - seed 43 (k=1)
    1. random box 177  — Fid 0.987, Cov 0.010, n=77
  - seed 44 (k=1)
    1. random box 108  — Fid 0.943, Cov 0.043, n=458

- `class_1`
  - seed 42 (k=1)
    1. random box 208  — Fid 0.995, Cov 0.065, n=196
  - seed 43 (k=1)
    1. random box 33  — Fid 0.973, Cov 0.040, n=110
  - seed 44 (k=1)
    1. random box 126  — Fid 1.000, Cov 0.026, n=84


### uci_adult — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. age ∈ [16.999998, 57.999996]  — Fid 0.838, Cov 0.911, n=8871
    2. age ∈ [23.807999, 90.000000]  — Fid 0.808, Cov 0.804, n=8295
  - seed 43 (k=5)
    1. age ∈ [20.901308, 90.000000] and fnlwgt ∈ [12285.000000, 329288.312500] and race = 'White' and capital-gain ∈ [0.000000, 4.796875]  — Fid 0.846, Cov 0.655, n=6382
    2. age ∈ [23.999998, 90.000000] and workclass = 'Private' and fnlwgt ∈ [63910.789062, 1490400.000000] and education-num ∈ [1.000000, 10.000000]  — Fid 0.961, Cov 0.404, n=3575
    3. age ∈ [26.999998, 90.000000] and fnlwgt ∈ [12285.000000, 389254.718750] and education = 'HS-grad'  — Fid 0.954, Cov 0.266, n=2433
    4. age ∈ [18.999996, 28.000000] and fnlwgt ∈ [63910.789062, 1490400.000000] and education-num ∈ [9.000000, 10.000000]  — Fid 0.993, Cov 0.180, n=1385
    5. age ∈ [24.773773, 90.000000] and fnlwgt ∈ [105685.992188, 442275.906250] and education = 'HS-grad'  — Fid 0.954, Cov 0.234, n=2117
  - seed 44: _not run_

- `class_1`
  - seed 42 (k=5)
    1. age ∈ [35.000000, 90.000000] and workclass = 'Private' and education = 'Bachelors' and relationship = 'Husband' and race = 'White'  — Fid 0.955, Cov 0.101, n=309
    2. fnlwgt ∈ [132235.500000, 1490400.000000] and education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.874, Cov 0.164, n=565
    3. workclass = 'Private' and education = 'Bachelors' and education-num ∈ [10.000001, 13.000000] and marital-status = 'Married-civ-spouse' and capital-loss ∈ [0.000000, 0.000000]  — Fid 0.885, Cov 0.138, n=477
    4. workclass = 'Private' and fnlwgt ∈ [49108.750000, 1490400.000000] and education = 'Masters' and marital-status = 'Married-civ-spouse' and occupation = 'Exec-managerial' and relationship = 'Husband' and native-country = 'United-States'  — Fid 0.982, Cov 0.021, n=56
    5. workclass = 'Private' and fnlwgt ∈ [108225.531250, 1490400.000000] and education = 'Masters' and marital-status = 'Married-civ-spouse' and occupation = 'Exec-managerial'  — Fid 0.983, Cov 0.022, n=60
  - seed 43 (k=4)
    1. fnlwgt ∈ [12285.000000, 510083.125000] and education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.892, Cov 0.243, n=843
    2. age ∈ [16.999998, 73.006699] and fnlwgt ∈ [12285.000000, 402363.125000] and education = 'Masters' and marital-status = 'Married-civ-spouse'  — Fid 0.942, Cov 0.093, n=293
    3. age ∈ [49.000000, 90.000000] and workclass = 'Private' and fnlwgt ∈ [108103.179688, 1490400.000000] and education = 'Bachelors' and marital-status = 'Married-civ-spouse' and relationship = 'Husband' and hours-per-week ∈ [40.000000, 40.000000] and native-country = 'United-States'  — Fid 0.976, Cov 0.012, n=41
    4. workclass = 'Private' and marital-status = 'Married-civ-spouse' and occupation = 'Exec-managerial' and relationship = 'Husband' and sex = 'Male' and capital-gain ∈ [0.000000, 15049.435547] and capital-loss ∈ [143.798462, 4356.000000] and native-country = 'United-States'  — Fid 0.821, Cov 0.014, n=39
  - seed 44: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. workclass = 'Private'  — Fid 0.861, Cov 0.734, n=6959
  - seed 43 (k=1)
    1. age ∈ [16.999998, 58.000000]  — Fid 0.829, Cov 0.905, n=8835
  - seed 44 (k=1)
    1. fnlwgt ∈ [64686.898438, 1226583.000000] and race = 'White' and capital-gain ∈ [-0.000122, 594.000000]  — Fid 0.860, Cov 0.718, n=6849

- `class_1`
  - seed 42 (k=1)
    1. education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.870, Cov 0.241, n=829
  - seed 43 (k=1)
    1. education = 'Bachelors' and education-num ∈ [1.000000, 13.213348] and marital-status = 'Married-civ-spouse' and relationship = 'Husband' and race = 'White' and hours-per-week ∈ [1.000000, 75.017540]  — Fid 0.901, Cov 0.203, n=686
  - seed 44 (k=1)
    1. education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.885, Cov 0.231, n=832


**cart**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 12.500000  — Fid 0.955, Cov 0.824, n=7289
  - seed 43 (k=1)
    1. education-num <= 12.500000  — Fid 0.953, Cov 0.828, n=7298
  - seed 44 (k=1)
    1. education-num <= 12.500000  — Fid 0.950, Cov 0.826, n=7337

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.500000  — Fid 0.516, Cov 0.500, n=2478
  - seed 43 (k=1)
    1. education-num > 12.500000  — Fid 0.537, Cov 0.512, n=2471
  - seed 44 (k=1)
    1. education-num > 12.500000  — Fid 0.530, Cov 0.487, n=2429


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 10.00  — Fid 0.963, Cov 0.754, n=6587
  - seed 43 (k=1)
    1. education-num <= 12.00  — Fid 0.953, Cov 0.828, n=7298
  - seed 44 (k=1)
    1. education-num <= 12.00  — Fid 0.950, Cov 0.826, n=7337

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.913, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00 and age > 37.00  — Fid 0.952, Cov 0.287, n=866
  - seed 44 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and education > 9.00 and age > 37.00  — Fid 0.936, Cov 0.277, n=870


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. relationship > 0.00  — Fid 0.836, Cov 1.000, n=9767
  - seed 43 (k=1)
    1. relationship > 0.00  — Fid 0.829, Cov 1.000, n=9769
  - seed 44 (k=1)
    1. relationship > 0.00  — Fid 0.830, Cov 1.000, n=9766

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.913, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.912, Cov 0.371, n=1198
  - seed 44 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and education > 9.00 and age > 37.00  — Fid 0.936, Cov 0.277, n=870


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 73  — Fid 0.963, Cov 0.032, n=320
  - seed 43 (k=1)
    1. random box 57  — Fid 1.000, Cov 0.022, n=215
  - seed 44 (k=1)
    1. random box 59  — Fid 1.000, Cov 0.013, n=98

- `class_1`
  - seed 42 (k=1)
    1. random box 223  — Fid 0.994, Cov 0.057, n=156
  - seed 43 (k=1)
    1. random box 74  — Fid 0.991, Cov 0.030, n=107
  - seed 44 (k=1)
    1. random box 40  — Fid 0.962, Cov 0.058, n=182



