# Paired statistical analysis

## Pre-specified primary comparison

The primary logistic policy obtains 77.6% versus 78.8% for fixed N=16, a paired
difference of -1.2 percentage points. The 10,000-resample paired bootstrap 95%
interval is [-3.0, 0.6] points. Its correctness contingency is 380 both correct,
8 adaptive-only correct, 14 N=16-only correct, and 98 both wrong. Exact
McNemar p=0.286.

This is neither evidence of equivalence nor a confirmatory non-inferiority
result: no equivalence margin was pre-specified, and a confidence interval
including zero does not establish equivalence.

## Other central comparisons

| Policy | Accuracy difference vs N=16 | Paired bootstrap 95% CI | Adaptive only / N=16 only | Exact McNemar p |
|---|---:|---:|---:|---:|
| Primary logistic | -1.2 pp | [-3.0, 0.6] | 8 / 14 | 0.286 |
| Primary boosted tree | -2.0 pp | [-4.0, -0.2] | 7 / 17 | 0.064 |
| Nested consensus | -1.2 pp | [-3.2, 0.6] | 9 / 15 | 0.307 |
| Nested margin | -0.4 pp | [-2.2, 1.4] | 9 / 11 | 0.824 |
| Fixed N=8 | -0.4 pp | [-2.4, 1.6] | 11 / 13 | 0.839 |

The tree bootstrap interval and exact McNemar test do not give identical
thresholded conclusions, which can occur with discrete, asymmetric paired
outcomes and different inferential constructions. Given multiple exploratory
comparisons, neither is used to claim significance.

## Fold stability

Primary logistic fold accuracies are 77%, 85%, 74%, 76%, and 76%; mean sample
counts range from 3.14 to 6.29. The corresponding per-fold accuracy differences
from each fold's N=16 result are 0, -1, -3, -1, and -1 points. Primary tree
accuracies are 77%, 86%, 73%, 73%, and 75%, with 2.97 to 6.76 samples and
differences 0, 0, -4, -4, and -2 points. The controller is therefore not stable
enough across folds for a strong deployment claim.

Bootstrap results, deterministic policy-specific seeds, and every contingency
are in `results/paired_comparisons.csv`. The base bootstrap seed is 20260915;
policy-specific seeds are stable SHA-256-derived offsets. All resampling units
are questions, never prefix rows.
