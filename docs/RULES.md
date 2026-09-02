# Extracted Rules

Generated 2026-09-02 17:12. Branch `ma-training-config-bump`.

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

- `class_1`
  - seed 42 (k=1)
    1. sepal length (cm) ∈ [5.096666, 7.900000] and petal length (cm) ∈ [3.590000, 6.900000] and petal width (cm) ∈ [0.100000, 1.500000]  — Fid 0.857, Cov 0.600, n=7
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [4.300000, 6.710988] and petal length (cm) ∈ [3.290000, 4.720357]  — Fid 0.600, Cov 1.000, n=10

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) ∈ [4.790000, 6.900000] and petal width (cm) ∈ [1.793333, 2.500000]  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [5.576667, 7.900000] and petal width (cm) ∈ [1.789987, 2.500000]  — Fid 1.000, Cov 0.900, n=9


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.488068]  — Fid 1.000, Cov 0.900, n=9

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.000000, 1.500000]  — Fid 0.889, Cov 0.800, n=9
  - seed 43 (k=1)
    1. sepal width (cm) ∈ [2.200000, 3.110024] and petal length (cm) ∈ [3.479993, 4.720000]  — Fid 0.750, Cov 0.800, n=8

- `class_2`
  - seed 42 (k=1)
    1. sepal width (cm) ∈ [2.500000, 3.900000]  — Fid 0.429, Cov 1.000, n=28
  - seed 43 (k=1)
    1. sepal length (cm) ∈ [5.800000, 7.900000]  — Fid 0.812, Cov 1.000, n=16


**cart**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 1.000, n=10

- `class_1`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) <= 1.550000  — Fid 0.889, Cov 0.800, n=9
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal length (cm) <= 4.650000  — Fid 0.625, Cov 0.800, n=8

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) > 1.550000  — Fid 1.000, Cov 0.900, n=11
  - seed 43 (k=1)
    1. petal length (cm) > 2.450000 and petal length (cm) > 4.650000  — Fid 0.909, Cov 0.900, n=11


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 1.60 and petal width (cm) <= 0.30  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.75 and sepal width (cm) > 2.80  — Fid 1.000, Cov 0.700, n=7

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.30 and petal length (cm) > 1.60 and sepal width (cm) <= 2.73  — Fid 1.000, Cov 0.300, n=3
  - seed 43 (k=1)
    1. petal width (cm) <= 1.30 and petal length (cm) > 1.60 and sepal width (cm) <= 2.80  — Fid 1.000, Cov 0.400, n=4

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) > 5.10 and petal width (cm) > 1.30  — Fid 1.000, Cov 0.700, n=7


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 1.60 and petal width (cm) <= 0.30  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.75 and sepal width (cm) > 2.80  — Fid 1.000, Cov 0.700, n=7

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.80 and 1.60 < petal length (cm) <= 4.25 and sepal width (cm) <= 3.00  — Fid 0.750, Cov 0.300, n=4
  - seed 43 (k=1)
    1. petal width (cm) <= 1.87 and sepal width (cm) <= 3.00 and 1.60 < petal length (cm) <= 5.10 and 5.10 < sepal length (cm) <= 6.48  — Fid 0.545, Cov 0.800, n=11

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 1.000, Cov 0.800, n=8
  - seed 43 (k=1)
    1. petal length (cm) > 5.10 and petal width (cm) > 1.30  — Fid 1.000, Cov 0.700, n=7


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 111  — Fid 1.000, Cov 1.000, n=10
  - seed 43 (k=1)
    1. random box 100  — Fid 1.000, Cov 0.900, n=9

- `class_1`
  - seed 42 (k=1)
    1. random box 57  — Fid 0.571, Cov 0.400, n=7
  - seed 43 (k=1)
    1. random box 164  — Fid 0.556, Cov 0.800, n=9

- `class_2`
  - seed 42 (k=1)
    1. random box 51  — Fid 0.778, Cov 0.500, n=9
  - seed 43 (k=1)
    1. random box 235  — Fid 1.000, Cov 0.800, n=10


### iris — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) ∈ [1.293334, 6.900000] and petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 0.800, n=8
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.000000, 2.500000]  — Fid 0.500, Cov 1.000, n=20
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [1.800000, 2.500000]  — Fid 1.000, Cov 0.900, n=9
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) ∈ [0.100000, 0.400000]  — Fid 1.000, Cov 1.000, n=10
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. sepal length (cm) ∈ [5.464249, 7.900000] and petal width (cm) ∈ [0.100000, 1.500000]  — Fid 0.750, Cov 0.500, n=8
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. sepal width (cm) ∈ [2.500000, 3.613333] and petal length (cm) ∈ [4.894028, 6.900000]  — Fid 0.900, Cov 0.900, n=10
  - seed 43: _not run_


**cart**

- `class_0`
  - seed 42 (k=1)
    1. petal length (cm) <= 2.450000  — Fid 1.000, Cov 0.800, n=8
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) <= 1.650000  — Fid 1.000, Cov 0.900, n=10
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. petal length (cm) > 2.450000 and petal width (cm) > 1.650000  — Fid 1.000, Cov 0.900, n=10
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.10  — Fid 1.000, Cov 0.400, n=4
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.30 and sepal width (cm) <= 2.73 and petal length (cm) <= 5.10  — Fid 1.000, Cov 0.300, n=3
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 0.875, Cov 0.800, n=8
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. petal width (cm) <= 0.30 and sepal length (cm) <= 5.10  — Fid 1.000, Cov 0.400, n=4
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. petal width (cm) <= 1.80 and sepal width (cm) <= 3.00 and 1.60 < petal length (cm) <= 4.25  — Fid 0.750, Cov 0.300, n=4
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. petal width (cm) > 1.30 and petal length (cm) > 5.10  — Fid 0.875, Cov 0.800, n=8
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 111  — Fid 1.000, Cov 1.000, n=10
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. random box 57  — Fid 0.714, Cov 0.400, n=7
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. random box 51  — Fid 0.667, Cov 0.500, n=9
  - seed 43: _not run_



## wine

### wine — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. nonflavanoid_phenols ∈ [0.220000, 0.630000] and proline ∈ [845.000000, 1515.000000]  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. ash ∈ [1.700000, 2.801143] and total_phenols ∈ [1.100000, 3.170000] and flavanoids ∈ [2.524000, 3.930000]  — Fid 0.778, Cov 0.583, n=9

- `class_1`
  - seed 42 (k=1)
    1. flavanoids ∈ [1.380000, 3.130474] and nonflavanoid_phenols ∈ [0.210000, 0.630000]  — Fid 0.600, Cov 0.643, n=15
  - seed 43 (k=1)
    1. color_intensity ∈ [1.280000, 3.400000]  — Fid 0.800, Cov 0.714, n=10

- `class_2`
  - seed 42 (k=1)
    1. nonflavanoid_phenols ∈ [0.130000, 0.530000] and proanthocyanins ∈ [0.420000, 1.560000]  — Fid 0.467, Cov 0.700, n=15
  - seed 43 (k=1)
    1. alcohol ∈ [12.532412, 13.615375] and color_intensity ∈ [4.275000, 13.000000]  — Fid 0.625, Cov 0.400, n=8


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. alcalinity_of_ash ∈ [10.600001, 20.000000] and nonflavanoid_phenols ∈ [0.220000, 0.340000]  — Fid 0.643, Cov 0.750, n=14
  - seed 43 (k=1)
    1. total_phenols ∈ [2.586000, 3.880000] and flavanoids ∈ [0.470000, 3.745429]  — Fid 0.800, Cov 0.667, n=10

- `class_1`
  - seed 42 (k=1)
    1. alcohol ∈ [11.030000, 12.963396] and od280/od315_of_diluted_wines ∈ [2.271620, 3.481030]  — Fid 1.000, Cov 0.643, n=9
  - seed 43 (k=1)
    1. od280/od315_of_diluted_wines ∈ [2.076000, 4.000000]  — Fid 0.440, Cov 0.929, n=25

- `class_2`
  - seed 42 (k=1)
    1. alcalinity_of_ash ∈ [18.928572, 30.000002] and hue ∈ [0.540000, 0.835001]  — Fid 0.750, Cov 0.600, n=8
  - seed 43 (k=1)
    1. magnesium ∈ [90.399208, 151.000000] and proanthocyanins ∈ [0.410000, 1.657692]  — Fid 0.556, Cov 0.500, n=9


**cart**

- `class_0`
  - seed 42 (k=1)
    1. color_intensity > 3.945000 and flavanoids > 1.795000  — Fid 0.889, Cov 0.667, n=9
  - seed 43 (k=1)
    1. proline > 875.000000  — Fid 0.909, Cov 0.833, n=11

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 3.945000  — Fid 0.857, Cov 0.429, n=7
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity <= 3.970000  — Fid 0.750, Cov 0.571, n=8

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 3.945000 and flavanoids <= 1.795000  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity > 3.970000  — Fid 0.750, Cov 0.800, n=12


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. proline > 985.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.583, n=7
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84  — Fid 1.000, Cov 0.667, n=8

- `class_1`
  - seed 42 (k=1)
    1. proline <= 500.00 and color_intensity <= 3.18  — Fid 1.000, Cov 0.143, n=2
  - seed 43 (k=1)
    1. proline <= 677.50 and color_intensity <= 3.19  — Fid 0.833, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 6.29 and od280/od315_of_diluted_wines <= 1.95  — Fid 1.000, Cov 0.500, n=5
  - seed 43 (k=1)
    1. color_intensity > 4.55 and flavanoids <= 1.21  — Fid 1.000, Cov 0.800, n=8


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. proline > 985.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.583, n=7
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84  — Fid 1.000, Cov 0.667, n=8

- `class_1`
  - seed 42 (k=1)
    1. proline <= 500.00 and color_intensity <= 3.18  — Fid 1.000, Cov 0.143, n=2
  - seed 43 (k=1)
    1. proline <= 677.50 and color_intensity <= 3.19  — Fid 0.833, Cov 0.429, n=6

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 6.29 and od280/od315_of_diluted_wines <= 1.95  — Fid 1.000, Cov 0.500, n=5
  - seed 43 (k=1)
    1. color_intensity > 4.55 and flavanoids <= 1.21  — Fid 1.000, Cov 0.800, n=8


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 18  — Fid 1.000, Cov 0.083, n=1
  - seed 43 (k=1)
    1. random box 111  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 134  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0

- `class_2`
  - seed 42 (k=1)
    1. random box 181  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0


### wine — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. magnesium ∈ [78.000000, 120.228577]  — Fid 0.303, Cov 0.833, n=33
    2. alcalinity_of_ash ∈ [15.900001, 30.000002] and proanthocyanins ∈ [1.460000, 3.280000]  — Fid 0.533, Cov 0.667, n=15
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. total_phenols ∈ [1.703721, 3.880000] and flavanoids ∈ [1.380000, 2.915189] and proline ∈ [341.999969, 750.000000]  — Fid 1.000, Cov 0.643, n=9
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. malic_acid ∈ [0.740000, 4.036001] and ash ∈ [2.302000, 2.840000] and proline ∈ [341.999969, 759.000061]  — Fid 0.429, Cov 0.300, n=7
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. ash ∈ [2.127401, 2.840000] and magnesium ∈ [93.318634, 115.000000] and flavanoids ∈ [2.406571, 3.740000]  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. alcalinity_of_ash ∈ [15.160000, 19.542856] and proanthocyanins ∈ [1.359143, 2.960000]  — Fid 0.583, Cov 0.583, n=12

- `class_1`
  - seed 42 (k=1)
    1. alcohol ∈ [11.030000, 12.946000]  — Fid 0.800, Cov 0.857, n=15
  - seed 43 (k=1)
    1. ash ∈ [1.707252, 2.580461] and alcalinity_of_ash ∈ [18.500000, 30.000000] and flavanoids ∈ [1.277151, 3.930000]  — Fid 0.875, Cov 0.571, n=8

- `class_2`
  - seed 42 (k=1)
    1. od280/od315_of_diluted_wines ∈ [1.290000, 2.075003]  — Fid 0.909, Cov 1.000, n=11
  - seed 43 (k=1)
    1. proanthocyanins ∈ [0.698541, 2.960000] and hue ∈ [0.573932, 0.894984]  — Fid 0.545, Cov 0.500, n=11


**cart**

- `class_0`
  - seed 42 (k=1)
    1. color_intensity > 3.820000 and flavanoids > 1.580000  — Fid 0.889, Cov 0.667, n=9
  - seed 43 (k=1)
    1. proline > 875.000000  — Fid 0.909, Cov 0.833, n=11

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 3.820000  — Fid 0.857, Cov 0.429, n=7
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity <= 3.970000  — Fid 1.000, Cov 0.571, n=8

- `class_2`
  - seed 42 (k=1)
    1. color_intensity > 3.820000 and flavanoids <= 1.580000  — Fid 1.000, Cov 0.900, n=9
  - seed 43 (k=1)
    1. proline <= 875.000000 and color_intensity > 3.970000  — Fid 0.750, Cov 0.800, n=12


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. total_phenols > 1.65 and proline > 675.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84 and alcalinity_of_ash <= 17.00  — Fid 1.000, Cov 0.167, n=2

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 4.80 and alcohol <= 12.37  — Fid 1.000, Cov 0.357, n=5
  - seed 43 (k=1)
    1. color_intensity <= 4.55 and proline <= 498.75  — Fid 1.000, Cov 0.357, n=5

- `class_2`
  - seed 42 (k=1)
    1. hue <= 0.96 and flavanoids <= 1.21 and nonflavanoid_phenols > 0.43  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. flavanoids <= 2.11 and hue <= 0.80 and od280/od315_of_diluted_wines <= 2.82  — Fid 0.800, Cov 0.800, n=10


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. total_phenols > 1.65 and proline > 675.00 and flavanoids > 2.03 and alcohol > 13.01  — Fid 1.000, Cov 0.750, n=9
  - seed 43 (k=1)
    1. proline > 677.50 and flavanoids > 2.84 and alcalinity_of_ash <= 17.00  — Fid 1.000, Cov 0.167, n=2

- `class_1`
  - seed 42 (k=1)
    1. color_intensity <= 4.80 and alcohol <= 12.37  — Fid 1.000, Cov 0.357, n=5
  - seed 43 (k=1)
    1. color_intensity <= 4.55 and proline <= 498.75  — Fid 1.000, Cov 0.357, n=5

- `class_2`
  - seed 42 (k=1)
    1. hue <= 0.96 and flavanoids <= 1.21 and nonflavanoid_phenols > 0.43  — Fid 1.000, Cov 0.600, n=6
  - seed 43 (k=1)
    1. flavanoids <= 2.11 and hue <= 0.80 and od280/od315_of_diluted_wines <= 2.82  — Fid 0.800, Cov 0.800, n=10


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 125  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 111  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 134  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0

- `class_2`
  - seed 42 (k=1)
    1. random box 181  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 0  — Cov 0.000, n=0



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

- `class_1`
  - seed 42 (k=3)
    1. mean texture ∈ [13.312542, 20.743551] and mean perimeter ∈ [43.790001, 94.270996]  — Fid 0.977, Cov 0.583, n=43
    2. mean texture ∈ [15.385001, 39.279999] and mean perimeter ∈ [43.790001, 87.867996] and mean symmetry ∈ [0.116700, 0.208226]  — Fid 0.927, Cov 0.542, n=41
    3. mean texture ∈ [10.889020, 17.190001] and mean concave points ∈ [0.019768, 0.191300] and mean symmetry ∈ [0.116700, 0.184550] and texture error ∈ [0.360200, 1.596600] and area error ∈ [6.802006, 31.115002]  — Fid 1.000, Cov 0.097, n=9
  - seed 43 (k=2)
    1. mean radius ∈ [6.980999, 14.970747]  — Fid 0.821, Cov 0.958, n=84
    2. mean smoothness ∈ [0.068830, 0.100626] and perimeter error ∈ [1.170300, 2.577400] and area error ∈ [6.801998, 31.215996]  — Fid 0.933, Cov 0.569, n=45


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. mean radius ∈ [14.762913, 28.110001] and mean perimeter ∈ [94.392006, 188.500000] and mean smoothness ∈ [0.090566, 0.142500]  — Fid 0.867, Cov 0.595, n=30
  - seed 43 (k=1)
    1. mean radius ∈ [15.052000, 27.219999] and mean perimeter ∈ [89.575996, 182.100006]  — Fid 0.929, Cov 0.619, n=28

- `class_1`
  - seed 42 (k=1)
    1. mean radius ∈ [6.981000, 13.602000] and mean texture ∈ [10.939300, 39.279999]  — Fid 0.945, Cov 0.736, n=55
  - seed 43 (k=1)
    1. mean radius ∈ [6.980999, 13.498000] and mean texture ∈ [10.380001, 23.188000]  — Fid 0.902, Cov 0.625, n=51


**cart**

- `class_0`
  - seed 42 (k=1)
    1. worst concave points > 0.145450  — Fid 0.971, Cov 0.762, n=34
  - seed 43 (k=1)
    1. worst concave points > 0.135950  — Fid 0.892, Cov 0.762, n=37

- `class_1`
  - seed 42 (k=1)
    1. worst concave points <= 0.145450  — Fid 0.880, Cov 0.931, n=75
  - seed 43 (k=1)
    1. worst concave points <= 0.135950  — Fid 0.938, Cov 0.819, n=64


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst texture > 25.09 and worst area > 985.50  — Fid 1.000, Cov 0.643, n=28
  - seed 43 (k=1)
    1. area error > 45.42 and worst perimeter > 126.90  — Fid 1.000, Cov 0.357, n=15

- `class_1`
  - seed 42 (k=1)
    1. worst texture <= 25.09 and worst concave points <= 0.10  — Fid 1.000, Cov 0.444, n=32
  - seed 43 (k=1)
    1. mean compactness <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.306, n=23


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst texture > 25.09 and worst area > 985.50  — Fid 1.000, Cov 0.643, n=28
  - seed 43 (k=1)
    1. area error > 45.42 and worst perimeter > 126.90  — Fid 1.000, Cov 0.357, n=15

- `class_1`
  - seed 42 (k=1)
    1. worst texture <= 25.09 and worst concave points <= 0.10  — Fid 1.000, Cov 0.444, n=32
  - seed 43 (k=1)
    1. mean compactness <= 0.10 and worst concavity <= 0.12  — Fid 1.000, Cov 0.306, n=23


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 131  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 121  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 207  — Fid 1.000, Cov 0.056, n=4
  - seed 43 (k=1)
    1. random box 45  — Fid 1.000, Cov 0.056, n=4


### breast_cancer — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=3)
    1. mean radius ∈ [15.268000, 28.110001] and mean perimeter ∈ [43.790001, 133.779999] and mean smoothness ∈ [0.062510, 0.118340]  — Fid 0.962, Cov 0.548, n=26
    2. mean smoothness ∈ [0.090566, 0.114100] and mean symmetry ∈ [0.116700, 0.230460] and worst radius ∈ [17.384983, 23.760021]  — Fid 1.000, Cov 0.310, n=15
    3. mean concave points ∈ [-0.000000, 0.093986] and mean fractal dimension ∈ [0.054622, 0.097440] and worst texture ∈ [26.315998, 49.540001] and worst area ∈ [766.293335, 1839.000000] and worst smoothness ∈ [0.109392, 0.218400]  — Fid 0.867, Cov 0.310, n=15
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. mean area ∈ [143.500031, 654.570007]  — Fid 0.929, Cov 0.875, n=70
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. mean radius ∈ [15.262047, 28.110001] and mean texture ∈ [9.710000, 30.635748] and mean area ∈ [540.960022, 2499.000000]  — Fid 0.971, Cov 0.738, n=34
  - seed 43 (k=1)
    1. mean radius ∈ [15.052000, 27.219999] and mean texture ∈ [16.384251, 39.279999]  — Fid 0.962, Cov 0.595, n=26

- `class_1`
  - seed 42 (k=1)
    1. mean radius ∈ [6.981000, 14.643740] and mean perimeter ∈ [43.790001, 94.270996]  — Fid 0.943, Cov 0.875, n=70
  - seed 43 (k=1)
    1. mean texture ∈ [12.954393, 39.279999] and mean perimeter ∈ [43.790001, 92.480003]  — Fid 0.897, Cov 0.819, n=68


**cart**

- `class_0`
  - seed 42 (k=1)
    1. worst concave points > 0.145450  — Fid 0.941, Cov 0.762, n=34
  - seed 43 (k=1)
    1. worst concave points > 0.135950  — Fid 0.946, Cov 0.762, n=37

- `class_1`
  - seed 42 (k=1)
    1. worst concave points <= 0.145450  — Fid 0.893, Cov 0.931, n=75
  - seed 43 (k=1)
    1. worst concave points <= 0.135950  — Fid 0.984, Cov 0.819, n=64


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst area > 985.50 and worst radius > 17.98  — Fid 1.000, Cov 0.738, n=32
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. worst area <= 680.60 and worst radius <= 14.85  — Fid 1.000, Cov 0.667, n=49
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. worst area > 985.50 and worst radius > 17.98  — Fid 1.000, Cov 0.738, n=32
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. worst area <= 680.60 and worst radius <= 14.85  — Fid 1.000, Cov 0.667, n=49
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 187  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 228  — Cov 0.000, n=0

- `class_1`
  - seed 42 (k=1)
    1. random box 102  — Fid 1.000, Cov 0.042, n=3
  - seed 43 (k=1)
    1. random box 166  — Fid 1.000, Cov 0.111, n=8



## synthetic

### synthetic — DNN black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 ∈ [-0.318884, 6.036793] and feature_5 ∈ [-0.199768, 5.608411]  — Fid 0.919, Cov 0.758, n=86
  - seed 43 (k=1)
    1. feature_0 ∈ [-4.417074, 1.684969] and feature_2 ∈ [-4.120684, 3.430854] and feature_3 ∈ [-3.325108, 5.864470]  — Fid 0.762, Cov 0.760, n=101

- `class_1`
  - seed 42 (k=1)
    1. feature_6 ∈ [-2.030793, 7.456970]  — Fid 0.506, Cov 0.921, n=178
  - seed 43 (k=5)
    1. feature_0 ∈ [1.383111, 5.282501] and feature_1 ∈ [-4.578733, 2.826565] and feature_2 ∈ [-3.033984, 0.397191] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.944, Cov 0.510, n=54
    2. feature_0 ∈ [0.460024, 2.997764] and feature_2 ∈ [-2.115639, -0.032339] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.971, Cov 0.330, n=34
    3. feature_0 ∈ [1.383111, 5.282501] and feature_1 ∈ [-4.578733, 1.366603] and feature_2 ∈ [-1.657719, 0.397191] and feature_3 ∈ [-9.130013, 0.318074]  — Fid 0.952, Cov 0.200, n=21
    4. feature_0 ∈ [0.214604, 5.282501] and feature_1 ∈ [-4.578733, 0.988761] and feature_2 ∈ [-1.657719, 0.397191] and feature_3 ∈ [-9.130013, 0.265433]  — Fid 0.909, Cov 0.300, n=33
    5. feature_2 ∈ [-1.687017, -0.380612] and feature_3 ∈ [-4.579780, -0.603626] and feature_4 ∈ [-1.285808, 4.651225]  — Fid 0.963, Cov 0.250, n=27


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 ∈ [0.281729, 6.036793] and feature_5 ∈ [-0.928563, 5.608411]  — Fid 0.867, Cov 0.747, n=90
  - seed 43 (k=1)
    1. feature_1 ∈ [-1.563525, 3.900472] and feature_6 ∈ [-3.060544, 0.578759] and feature_7 ∈ [-0.154771, 3.962004] and feature_8 ∈ [-3.468153, 0.271785] and feature_9 ∈ [-1.297152, 3.438132]  — Fid 1.000, Cov 0.280, n=29

- `class_1`
  - seed 42 (k=1)
    1. feature_1 ∈ [-2.232415, 4.585218] and feature_3 ∈ [-0.513922, 4.843362] and feature_4 ∈ [-3.716660, 1.196453]  — Fid 0.796, Cov 0.406, n=49
  - seed 43 (k=1)
    1. feature_0 ∈ [0.399875, 3.672356] and feature_2 ∈ [-1.408125, 3.430854] and feature_8 ∈ [-0.546124, 5.184010]  — Fid 0.933, Cov 0.410, n=45


**cart**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > -0.070581  — Fid 0.659, Cov 0.869, n=135
  - seed 43 (k=1)
    1. feature_2 > 0.610751  — Fid 0.804, Cov 0.380, n=46

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.070581  — Fid 0.818, Cov 0.475, n=55
  - seed 43 (k=1)
    1. feature_2 <= 0.610751  — Fid 0.607, Cov 0.900, n=150


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.918, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_8 <= -0.56 and feature_3 > 0.38 and feature_4 > -0.48  — Fid 0.935, Cov 0.290, n=31

- `class_1`
  - seed 42 (k=1)
    1. feature_5 <= -0.69 and feature_9 > 2.36  — Fid 0.906, Cov 0.307, n=32
  - seed 43 (k=1)
    1. feature_0 > 2.24 and feature_8 > -0.56 and feature_2 <= 0.52  — Fid 1.000, Cov 0.330, n=33


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.918, Cov 0.566, n=61
  - seed 43 (k=1)
    1. feature_8 <= -0.56 and feature_3 > 0.38 and feature_4 > -0.48  — Fid 0.935, Cov 0.290, n=31

- `class_1`
  - seed 42 (k=1)
    1. feature_5 <= -0.69 and feature_9 > 2.36  — Fid 0.906, Cov 0.307, n=32
  - seed 43 (k=1)
    1. feature_0 > 2.24 and feature_8 > -0.56 and feature_2 <= 0.52  — Fid 1.000, Cov 0.330, n=33


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 148  — Fid 1.000, Cov 0.152, n=16
  - seed 43 (k=1)
    1. random box 11  — Fid 1.000, Cov 0.040, n=4

- `class_1`
  - seed 42 (k=1)
    1. random box 135  — Fid 1.000, Cov 0.040, n=4
  - seed 43 (k=1)
    1. random box 235  — Fid 0.900, Cov 0.090, n=10


### synthetic — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. feature_2 ∈ [-2.604953, 3.276399] and feature_4 ∈ [0.543823, 2.671431] and feature_5 ∈ [-0.199769, 5.608411] and feature_9 ∈ [-7.018609, 2.676795]  — Fid 1.000, Cov 0.545, n=56
    2. feature_0 ∈ [-4.661168, 1.743295] and feature_2 ∈ [-1.930220, 0.849564] and feature_4 ∈ [0.173260, 1.658094] and feature_5 ∈ [-0.928314, 5.608411]  — Fid 0.902, Cov 0.465, n=51
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=3)
    1. feature_0 ∈ [-0.944462, 4.724046]  — Fid 0.554, Cov 0.891, n=157
    2. feature_2 ∈ [-0.988432, 2.235508] and feature_3 ∈ [-1.985323, 4.843362] and feature_4 ∈ [-3.716660, -0.444956] and feature_5 ∈ [-0.887323, 5.608411] and feature_9 ∈ [-4.863406, 7.933945]  — Fid 0.931, Cov 0.277, n=29
    3. feature_3 ∈ [-0.876189, 0.541337] and feature_4 ∈ [-3.716660, 0.476419] and feature_5 ∈ [-1.721353, 5.608411] and feature_9 ∈ [-2.559016, 7.933945]  — Fid 0.909, Cov 0.099, n=11
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. feature_0 ∈ [-4.661168, 1.503932] and feature_4 ∈ [-0.318891, 6.036793] and feature_5 ∈ [0.199058, 5.608411]  — Fid 1.000, Cov 0.596, n=60
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. feature_0 ∈ [-4.661168, 2.774007] and feature_3 ∈ [-1.186223, 4.843362] and feature_4 ∈ [-3.716660, -0.019826] and feature_6 ∈ [-0.992333, 7.456970]  — Fid 1.000, Cov 0.347, n=35
  - seed 43: _not run_


**cart**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.031331  — Fid 0.667, Cov 0.859, n=132
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= 0.031331  — Fid 0.828, Cov 0.495, n=58
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.934, Cov 0.566, n=61
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.61 and feature_3 > 0.29  — Fid 1.000, Cov 0.267, n=27
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. feature_4 > 0.53 and feature_5 > -0.69 and feature_0 <= 1.65  — Fid 0.934, Cov 0.566, n=61
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. feature_4 <= -0.61 and feature_3 > 0.29  — Fid 1.000, Cov 0.267, n=27
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 203  — Fid 0.750, Cov 0.081, n=12
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. random box 97  — Fid 0.750, Cov 0.099, n=12
  - seed 43: _not run_



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

- `class_1`
  - seed 42 (k=4)
    1. MedInc ∈ [3.295002, 4.067696] and AveOccup ∈ [4.302743, 502.461578] and Latitude ∈ [32.924000, 41.880001] and Longitude ∈ [-124.349991, -117.550003]  — Fid 0.767, Cov 0.019, n=30
    2. MedInc ∈ [2.068848, 15.000100] and HouseAge ∈ [14.999999, 52.000000] and AveRooms ∈ [0.846154, 5.611066] and AveOccup ∈ [3.017442, 502.461578] and Longitude ∈ [-119.074013, -114.489990]  — Fid 0.643, Cov 0.252, n=476
    3. MedInc ∈ [2.051857, 2.875000] and HouseAge ∈ [10.000000, 52.000000] and AveRooms ∈ [0.846154, 4.407454] and AveBedrms ∈ [0.444444, 1.112226] and Population ∈ [1504.000000, 16304.999023] and AveOccup ∈ [2.662883, 4.417292] and Latitude ∈ [32.549999, 34.660000] and Longitude ∈ [-118.284897, -114.489990]  — Fid 0.812, Cov 0.030, n=48
    4. MedInc ∈ [2.160012, 3.158869] and HouseAge ∈ [14.999999, 52.000000] and AveRooms ∈ [0.846154, 3.628435] and AveBedrms ∈ [0.444444, 1.087446] and Population ∈ [1758.000122, 16304.999023] and AveOccup ∈ [2.838722, 4.636699] and Latitude ∈ [32.549999, 34.099998] and Longitude ∈ [-118.334534, -114.489990]  — Fid 1.000, Cov 0.008, n=11
  - seed 43 (k=2)
    1. MedInc ∈ [2.037470, 2.682440] and HouseAge ∈ [35.000000, 52.000000] and AveRooms ∈ [3.446317, 4.667093] and Population ∈ [1297.000000, 35682.003906] and AveOccup ∈ [2.855951, 502.461548] and Latitude ∈ [33.939999, 41.950001] and Longitude ∈ [-121.440002, -117.120003]  — Fid 0.897, Cov 0.016, n=29
    2. MedInc ∈ [3.237500, 4.200259] and HouseAge ∈ [24.000000, 52.000000] and AveRooms ∈ [4.004460, 5.564185] and Population ∈ [789.787354, 1976.487427] and AveOccup ∈ [4.088190, 502.461548] and Latitude ∈ [33.830002, 41.950001] and Longitude ∈ [-122.136002, -114.309998]  — Fid 0.944, Cov 0.016, n=18

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


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 1.910540] and HouseAge ∈ [12.999999, 39.000000] and AveRooms ∈ [0.846154, 4.784486] and Population ∈ [1072.500000, 16304.999023] and AveOccup ∈ [2.758413, 502.461578] and Latitude ∈ [34.249001, 38.619999]  — Fid 0.925, Cov 0.034, n=40
  - seed 43 (k=1)
    1. MedInc ∈ [0.499900, 2.336201] and AveRooms ∈ [3.362297, 132.533340] and Population ∈ [325.932129, 35682.003906] and Latitude ∈ [35.619999, 41.950001] and Longitude ∈ [-124.349998, -117.019310]  — Fid 0.890, Cov 0.266, n=336

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 4.704296] and AveOccup ∈ [2.459047, 502.461578]  — Fid 0.374, Cov 0.716, n=2146
  - seed 43 (k=1)
    1. MedInc ∈ [0.499900, 4.656301]  — Fid 0.289, Cov 0.893, n=3042

- `class_2`
  - seed 42 (k=1)
    1. MedInc ∈ [3.579812, 5.640269] and AveOccup ∈ [2.407553, 3.344623] and Latitude ∈ [32.759998, 37.740002]  — Fid 0.599, Cov 0.288, n=628
  - seed 43 (k=1)
    1. MedInc ∈ [3.008880, 5.671132] and Latitude ∈ [32.549995, 37.959999]  — Fid 0.425, Cov 0.640, n=1791

- `class_3`
  - seed 42 (k=1)
    1. MedInc ∈ [5.924500, 15.000100] and AveRooms ∈ [4.549819, 62.422218] and AveBedrms ∈ [0.953696, 14.111111] and Population ∈ [3.000000, 1469.000000] and Latitude ∈ [32.549999, 37.940437]  — Fid 0.921, Cov 0.229, n=267
  - seed 43 (k=1)
    1. MedInc ∈ [5.887044, 15.000101] and AveRooms ∈ [5.240104, 132.533340] and Latitude ∈ [33.599998, 37.580002] and Longitude ∈ [-122.470001, -114.309998]  — Fid 0.860, Cov 0.270, n=315


**cart**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 3.198700 and Latitude > 34.445000  — Fid 0.694, Cov 0.530, n=863
  - seed 43 (k=1)
    1. MedInc <= 3.187950 and Latitude > 34.455000  — Fid 0.690, Cov 0.536, n=899

- `class_1`
  - seed 42 (k=1)
    1. MedInc <= 3.198700 and Latitude <= 34.445000  — Fid 0.476, Cov 0.330, n=908
  - seed 43 (k=1)
    1. MedInc <= 3.187950 and Latitude <= 34.455000  — Fid 0.375, Cov 0.303, n=839

- `class_2`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc <= 5.764050  — Fid 0.455, Cov 0.636, n=1815
  - seed 43 (k=1)
    1. MedInc > 3.187950 and MedInc <= 5.776200  — Fid 0.405, Cov 0.664, n=1888

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc > 5.764050  — Fid 0.867, Cov 0.430, n=535
  - seed 43 (k=1)
    1. MedInc > 3.187950 and MedInc > 5.776200  — Fid 0.826, Cov 0.412, n=500


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. Latitude > 34.27 and Longitude > -121.80 and MedInc <= 2.57  — Fid 0.909, Cov 0.277, n=352
  - seed 43 (k=1)
    1. Latitude > 34.24 and MedInc <= 3.54 and Longitude > -121.81 and AveOccup > 2.43 and AveRooms <= 6.04 and 1.01 < AveBedrms <= 1.10  — Fid 0.782, Cov 0.244, n=340

- `class_1`
  - seed 42 (k=1)
    1. Latitude <= 34.27 and 2.57 < MedInc <= 3.55 and AveRooms <= 5.25 and AveOccup > 3.28 and Population > 1167.00 and AveBedrms <= 1.10 and HouseAge > 29.00  — Fid 0.710, Cov 0.095, n=162
  - seed 43 (k=1)
    1. Longitude <= -118.48 and 2.57 < MedInc <= 3.54  — Fid 0.209, Cov 0.250, n=1074

- `class_2`
  - seed 42 (k=1)
    1. AveOccup > 2.44 and MedInc > 3.55 and Longitude <= -121.80  — Fid 0.430, Cov 0.125, n=405
  - seed 43 (k=1)
    1. AveOccup <= 3.28 and Longitude > -118.48 and 2.57 < MedInc <= 4.74 and Latitude <= 37.72  — Fid 0.428, Cov 0.339, n=1008

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and AveOccup <= 2.44 and Longitude <= -121.80 and HouseAge > 18.00  — Fid 0.952, Cov 0.056, n=62
  - seed 43 (k=1)
    1. MedInc > 4.74 and AveOccup <= 2.43 and Latitude <= 34.24 and HouseAge > 29.00 and -121.81 < Longitude <= -118.00  — Fid 0.929, Cov 0.040, n=42


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. Latitude > 34.27 and Longitude > -121.80 and MedInc <= 2.57  — Fid 0.909, Cov 0.277, n=352
  - seed 43 (k=1)
    1. Latitude > 34.24 and MedInc <= 3.54 and Longitude > -121.81 and AveOccup > 2.43 and AveRooms <= 6.04 and 1.01 < AveBedrms <= 1.10  — Fid 0.782, Cov 0.244, n=340

- `class_1`
  - seed 42 (k=1)
    1. Latitude <= 34.27 and 2.57 < MedInc <= 3.55 and AveRooms <= 5.25 and AveOccup > 3.28 and Population > 1167.00 and AveBedrms <= 1.10 and HouseAge > 29.00  — Fid 0.710, Cov 0.095, n=162
  - seed 43 (k=1)
    1. Longitude <= -118.48 and 2.57 < MedInc <= 3.54  — Fid 0.209, Cov 0.250, n=1074

- `class_2`
  - seed 42 (k=1)
    1. AveOccup > 2.44 and MedInc > 3.55 and Longitude <= -121.80  — Fid 0.430, Cov 0.125, n=405
  - seed 43 (k=1)
    1. AveOccup <= 3.28 and Longitude > -118.48 and 2.57 < MedInc <= 4.74 and Latitude <= 37.72  — Fid 0.428, Cov 0.339, n=1008

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and AveOccup <= 2.44 and Longitude <= -121.80 and HouseAge > 18.00  — Fid 0.952, Cov 0.056, n=62
  - seed 43 (k=1)
    1. AveOccup <= 2.43 and MedInc > 2.57 and HouseAge > 18.00 and Latitude <= 37.72 and Longitude <= -118.00 and AveRooms > 4.45  — Fid 0.625, Cov 0.137, n=200


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 160  — Fid 0.985, Cov 0.062, n=67
  - seed 43 (k=1)
    1. random box 249  — Fid 0.987, Cov 0.072, n=75

- `class_1`
  - seed 42 (k=1)
    1. random box 7  — Fid 0.897, Cov 0.023, n=39
  - seed 43 (k=1)
    1. random box 198  — Fid 0.598, Cov 0.058, n=127

- `class_2`
  - seed 42 (k=1)
    1. random box 239  — Fid 0.577, Cov 0.218, n=482
  - seed 43 (k=1)
    1. random box 212  — Fid 0.704, Cov 0.021, n=54

- `class_3`
  - seed 42 (k=1)
    1. random box 221  — Fid 0.874, Cov 0.096, n=111
  - seed 43 (k=1)
    1. random box 121  — Fid 0.940, Cov 0.047, n=50


### housing — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=5)
    1. MedInc ∈ [0.499900, 3.139491] and HouseAge ∈ [14.874919, 35.000000] and AveOccup ∈ [2.232897, 502.461578] and Latitude ∈ [34.249001, 40.760330]  — Fid 0.883, Cov 0.253, n=367
    2. MedInc ∈ [0.499900, 2.827969] and AveRooms ∈ [3.971788, 62.422218] and AveBedrms ∈ [0.935340, 1.092194] and Population ∈ [630.341125, 16304.999023] and AveOccup ∈ [2.224380, 502.461578] and Longitude ∈ [-124.349991, -119.096008]  — Fid 0.922, Cov 0.159, n=206
    3. MedInc ∈ [0.499900, 3.139503] and HouseAge ∈ [24.000000, 35.000000] and AveOccup ∈ [2.232894, 502.461578] and Latitude ∈ [34.249001, 41.030151]  — Fid 0.905, Cov 0.151, n=210
    4. MedInc ∈ [0.499900, 2.832603] and AveRooms ∈ [4.721093, 62.422218] and AveBedrms ∈ [1.002617, 1.092001] and Population ∈ [396.000000, 16304.999023] and AveOccup ∈ [2.108800, 4.292845] and Longitude ∈ [-124.349991, -120.480003]  — Fid 0.956, Cov 0.066, n=91
    5. MedInc ∈ [0.499900, 2.826523] and HouseAge ∈ [1.000000, 46.000000] and AveRooms ∈ [4.760034, 62.422218] and AveBedrms ∈ [1.053951, 1.092015] and Population ∈ [634.000000, 16304.999023] and Longitude ∈ [-124.349991, -118.946472]  — Fid 1.000, Cov 0.044, n=50
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [2.424211, 4.704300] and HouseAge ∈ [30.999998, 52.000000] and AveRooms ∈ [0.846154, 4.056044] and AveBedrms ∈ [1.014826, 14.111111] and Population ∈ [949.690186, 16304.999023] and AveOccup ∈ [3.206178, 502.461578] and Latitude ∈ [33.937145, 34.660000] and Longitude ∈ [-124.349991, -118.160004]  — Fid 1.000, Cov 0.015, n=19
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=2)
    1. MedInc ∈ [4.512644, 6.854567] and HouseAge ∈ [21.000000, 52.000000] and AveRooms ∈ [4.285494, 6.612618] and AveBedrms ∈ [0.966175, 14.111111] and AveOccup ∈ [3.192369, 502.461578]  — Fid 0.855, Cov 0.037, n=69
    2. MedInc ∈ [0.499900, 5.131864] and HouseAge ∈ [17.778315, 52.000000] and AveBedrms ∈ [0.444444, 1.014778] and AveOccup ∈ [2.254367, 502.461578] and Latitude ∈ [32.549999, 33.897167] and Longitude ∈ [-118.110001, -117.967789]  — Fid 0.812, Cov 0.013, n=16
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=5)
    1. MedInc ∈ [5.924500, 10.229071] and HouseAge ∈ [14.887274, 52.000000] and AveBedrms ∈ [0.953049, 14.111111] and Population ∈ [3.000000, 2711.179688] and AveOccup ∈ [2.182507, 3.839206] and Latitude ∈ [32.549999, 37.380001]  — Fid 0.944, Cov 0.171, n=198
    2. MedInc ∈ [5.367700, 9.220963] and AveRooms ∈ [5.561594, 8.657566] and AveBedrms ∈ [0.444444, 1.099567] and Population ∈ [495.000000, 16304.999023] and AveOccup ∈ [0.692308, 2.924684]  — Fid 0.927, Cov 0.189, n=232
    3. MedInc ∈ [5.367700, 8.617228] and AveRooms ∈ [5.134445, 8.213058] and AveBedrms ∈ [0.444444, 1.099567] and Population ∈ [495.000000, 16304.999023] and AveOccup ∈ [0.692308, 2.898188]  — Fid 0.930, Cov 0.193, n=242
    4. MedInc ∈ [5.367700, 12.543919] and HouseAge ∈ [10.935076, 52.000000] and AveBedrms ∈ [1.010427, 14.111111] and AveOccup ∈ [2.074328, 3.178025] and Latitude ∈ [32.549999, 37.709999]  — Fid 0.954, Cov 0.170, n=196
    5. MedInc ∈ [6.486750, 15.000100] and HouseAge ∈ [13.904070, 52.000000] and AveBedrms ∈ [0.988432, 14.111111] and Population ∈ [3.000000, 2405.566895] and AveOccup ∈ [2.421327, 3.322512] and Latitude ∈ [32.549999, 37.040630]  — Fid 1.000, Cov 0.086, n=90
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. MedInc ∈ [0.499900, 2.564511] and AveBedrms ∈ [1.068926, 14.111111] and Latitude ∈ [34.240002, 41.880001] and Longitude ∈ [-121.619995, -114.489990]  — Fid 0.960, Cov 0.140, n=174
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. MedInc ∈ [3.012578, 15.000100] and AveBedrms ∈ [0.444444, 1.038734] and AveOccup ∈ [4.307551, 502.461578] and Longitude ∈ [-120.469994, -114.489990]  — Fid 0.889, Cov 0.024, n=36
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. MedInc ∈ [3.035674, 15.000100] and AveRooms ∈ [0.846154, 5.274409] and AveOccup ∈ [2.560740, 3.393601] and Latitude ∈ [32.549999, 37.958115] and Longitude ∈ [-119.106606, -118.089989]  — Fid 0.854, Cov 0.054, n=103
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=1)
    1. MedInc ∈ [5.367700, 15.000100] and HouseAge ∈ [1.000000, 52.000000] and AveRooms ∈ [4.021782, 62.422218] and Population ∈ [3.000000, 3550.383789] and AveOccup ∈ [0.692308, 3.116878]  — Fid 0.903, Cov 0.384, n=487
  - seed 43: _not run_


**cart**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 3.198700  — Fid 0.557, Cov 0.821, n=1771
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc <= 5.776350 and AveOccup > 2.395127  — Fid 0.477, Cov 0.515, n=1364
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 3.198700 and MedInc > 5.776350  — Fid 0.906, Cov 0.427, n=531
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 2.57 and Latitude > 37.72  — Fid 0.897, Cov 0.210, n=301
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. 3.55 < MedInc <= 4.78 and AveOccup > 2.82 and Longitude > -118.02 and 33.93 < Latitude <= 34.27 and HouseAge <= 18.00 and AveRooms <= 6.09 and Population > 1729.25 and AveBedrms <= 1.10  — Fid 0.759, Cov 0.016, n=29
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. 33.93 < Latitude <= 34.27 and 2.57 < MedInc <= 4.78 and AveRooms <= 6.09 and 2.44 < AveOccup <= 3.28 and 18.00 < HouseAge <= 37.00 and AveBedrms <= 1.10  — Fid 0.494, Cov 0.208, n=526
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and Longitude <= -121.80 and AveOccup <= 3.28 and AveRooms > 6.09 and Latitude <= 37.72  — Fid 0.977, Cov 0.080, n=88
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. MedInc <= 2.57 and Latitude > 37.72  — Fid 0.897, Cov 0.210, n=301
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. AveOccup > 3.28 and -118.52 < Longitude <= -118.02 and AveRooms <= 5.25 and 2.57 < MedInc <= 3.55 and Latitude <= 34.27 and HouseAge > 29.00 and Population > 786.00  — Fid 0.777, Cov 0.123, n=202
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. 33.93 < Latitude <= 34.27 and 2.57 < MedInc <= 4.78 and AveRooms <= 6.09 and 2.44 < AveOccup <= 3.28 and 18.00 < HouseAge <= 37.00 and AveBedrms <= 1.10  — Fid 0.494, Cov 0.208, n=526
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=1)
    1. MedInc > 4.78 and Longitude <= -121.80 and AveOccup <= 3.28 and AveRooms > 6.09 and Latitude <= 37.72  — Fid 0.977, Cov 0.080, n=88
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 239  — Fid 0.985, Cov 0.113, n=132
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. random box 130  — Fid 0.667, Cov 0.004, n=18
  - seed 43: _not run_

- `class_2`
  - seed 42 (k=1)
    1. random box 0  — Fid 0.678, Cov 0.113, n=202
  - seed 43: _not run_

- `class_3`
  - seed 42 (k=1)
    1. random box 41  — Fid 1.000, Cov 0.058, n=64
  - seed 43: _not run_



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

- `class_1`
  - seed 42 (k=1)
    1. A9 = 'f'  — Fid 1.000, Cov 0.727, n=64
  - seed 43 (k=2)
    1. A10 = 'f'  — Fid 0.892, Cov 0.753, n=74
    2. A12 = 'f' and A11 ∈ [0.000000, 67.000000] and A10 = 't' and A9 = 'f'  — Fid 1.000, Cov 0.065, n=5


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. A13 = 'g' and A9 = 't'  — Fid 0.864, Cov 0.803, n=66
  - seed 43 (k=1)
    1. A13 = 'g' and A9 = 't'  — Fid 0.882, Cov 0.902, n=68

- `class_1`
  - seed 42 (k=1)
    1. A15 ∈ [0.000000, 100000.007812] and A11 ∈ [0.000000, 1.000000] and A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.636, n=56
  - seed 43 (k=1)
    1. A10 = 'f' and A9 = 'f'  — Fid 1.000, Cov 0.662, n=55


**cart**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.500000  — Fid 0.824, Cov 0.869, n=74
  - seed 43 (k=1)
    1. A9 > 0.500000  — Fid 0.843, Cov 0.918, n=70

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.714, n=63
  - seed 43 (k=1)
    1. A9 <= 0.500000  — Fid 1.000, Cov 0.792, n=65


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A11 > 3.00  — Fid 0.921, Cov 0.557, n=38
  - seed 43 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.437, Cov 0.984, n=135

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00 and A11 <= 3.00  — Fid 1.000, Cov 0.688, n=61
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.445, Cov 1.000, n=137
  - seed 43 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.437, Cov 0.984, n=135

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00 and A11 <= 3.00  — Fid 1.000, Cov 0.688, n=61
  - seed 43 (k=1)
    1. A9 <= 0.00  — Fid 1.000, Cov 0.792, n=65


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 105  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 197  — Fid 1.000, Cov 0.033, n=3

- `class_1`
  - seed 42 (k=1)
    1. random box 0  — Cov 0.000, n=0
  - seed 43 (k=1)
    1. random box 42  — Fid 1.000, Cov 0.052, n=5


### uci_credit — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=1)
    1. A9 = 't'  — Fid 0.919, Cov 0.869, n=74
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A9 = 'f'  — Fid 0.953, Cov 0.727, n=64
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. A10 = 't' and A9 = 't'  — Fid 1.000, Cov 0.639, n=44
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A13 = 'g' and A10 = 'f' and A9 = 'f'  — Fid 0.977, Cov 0.519, n=43
  - seed 43: _not run_


**cart**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.500000  — Fid 0.919, Cov 0.869, n=74
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.500000  — Fid 0.952, Cov 0.714, n=63
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.518, Cov 1.000, n=137
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00  — Fid 0.952, Cov 0.714, n=63
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. A9 > 0.00 and A10 > 0.00  — Fid 0.518, Cov 1.000, n=137
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. A9 <= 0.00  — Fid 0.952, Cov 0.714, n=63
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 147  — Cov 0.000, n=0
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. random box 154  — Cov 0.000, n=0
  - seed 43: _not run_



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


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. workclass = 'Private'  — Fid 0.820, Cov 0.734, n=6959
  - seed 43 (k=1)
    1. workclass = 'Private'  — Fid 0.836, Cov 0.742, n=6986

- `class_1`
  - seed 42 (k=1)
    1. age ∈ [16.999998, 58.098831] and education-num ∈ [12.000000, 14.088848] and marital-status = 'Married-civ-spouse' and capital-gain ∈ [-0.000122, 667.330811] and native-country = 'United-States'  — Fid 0.831, Cov 0.243, n=857
  - seed 43 (k=1)
    1. age ∈ [36.000000, 65.030380] and education = 'Masters' and education-num ∈ [1.000000, 14.088706] and occupation = 'Exec-managerial' and relationship = 'Husband' and race = 'White' and sex = 'Male' and capital-loss ∈ [0.000000, 597.680176] and native-country = 'United-States'  — Fid 1.000, Cov 0.025, n=63


**cart**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 12.500000  — Fid 0.898, Cov 0.824, n=7289
  - seed 43 (k=1)
    1. education-num <= 12.500000  — Fid 0.910, Cov 0.828, n=7298

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.500000  — Fid 0.509, Cov 0.500, n=2478
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 9.00  — Fid 0.947, Cov 0.519, n=4449
  - seed 43 (k=1)
    1. education-num <= 9.00  — Fid 0.944, Cov 0.517, n=4408

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.893, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and age > 37.00 and education > 11.00  — Fid 0.920, Cov 0.109, n=314


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 9.00  — Fid 0.947, Cov 0.519, n=4449
  - seed 43 (k=1)
    1. capital-gain <= 0.00 and relationship > 0.00  — Fid 0.848, Cov 0.955, n=8920

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.893, Cov 0.361, n=1145
  - seed 43 (k=1)
    1. capital-loss > 0.00 and relationship <= 0.00 and education-num > 12.00  — Fid 0.792, Cov 0.402, n=1330


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 131  — Fid 1.000, Cov 0.009, n=85
  - seed 43 (k=1)
    1. random box 177  — Fid 0.987, Cov 0.010, n=77

- `class_1`
  - seed 42 (k=1)
    1. random box 208  — Fid 0.995, Cov 0.065, n=196
  - seed 43 (k=1)
    1. random box 33  — Fid 0.973, Cov 0.040, n=110


### uci_adult — RandomForest black box

**MADA**

- `class_0`
  - seed 42 (k=2)
    1. age ∈ [16.999998, 57.999996]  — Fid 0.838, Cov 0.911, n=8871
    2. age ∈ [23.807999, 90.000000]  — Fid 0.808, Cov 0.804, n=8295
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=5)
    1. age ∈ [35.000000, 90.000000] and workclass = 'Private' and education = 'Bachelors' and relationship = 'Husband' and race = 'White'  — Fid 0.955, Cov 0.101, n=309
    2. fnlwgt ∈ [132235.500000, 1490400.000000] and education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.874, Cov 0.164, n=565
    3. workclass = 'Private' and education = 'Bachelors' and education-num ∈ [10.000001, 13.000000] and marital-status = 'Married-civ-spouse' and capital-loss ∈ [0.000000, 0.000000]  — Fid 0.885, Cov 0.138, n=477
    4. workclass = 'Private' and fnlwgt ∈ [49108.750000, 1490400.000000] and education = 'Masters' and marital-status = 'Married-civ-spouse' and occupation = 'Exec-managerial' and relationship = 'Husband' and native-country = 'United-States'  — Fid 0.982, Cov 0.021, n=56
    5. workclass = 'Private' and fnlwgt ∈ [108225.531250, 1490400.000000] and education = 'Masters' and marital-status = 'Married-civ-spouse' and occupation = 'Exec-managerial'  — Fid 0.983, Cov 0.022, n=60
  - seed 43: _not run_


**RLDA**

- `class_0`
  - seed 42 (k=1)
    1. workclass = 'Private'  — Fid 0.861, Cov 0.734, n=6959
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. education = 'Bachelors' and marital-status = 'Married-civ-spouse'  — Fid 0.870, Cov 0.241, n=829
  - seed 43: _not run_


**cart**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 12.500000  — Fid 0.955, Cov 0.824, n=7289
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.500000  — Fid 0.516, Cov 0.500, n=2478
  - seed 43: _not run_


**greedy_anchors**

- `class_0`
  - seed 42 (k=1)
    1. education-num <= 10.00  — Fid 0.963, Cov 0.754, n=6587
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.913, Cov 0.361, n=1145
  - seed 43: _not run_


**sp_anchors**

- `class_0`
  - seed 42 (k=1)
    1. relationship > 0.00  — Fid 0.836, Cov 1.000, n=9767
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. education-num > 12.00 and relationship <= 0.00 and hours-per-week > 40.00  — Fid 0.913, Cov 0.361, n=1145
  - seed 43: _not run_


**random_search**

- `class_0`
  - seed 42 (k=1)
    1. random box 73  — Fid 0.963, Cov 0.032, n=320
  - seed 43: _not run_

- `class_1`
  - seed 42 (k=1)
    1. random box 223  — Fid 0.994, Cov 0.057, n=156
  - seed 43: _not run_



