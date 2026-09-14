# Feature, calibration, and robustness results

## Feature ablation

These are nested outer-fold results for a fixed balanced L2 logistic model with
sigmoid calibration. Model discrimination values are outer-fold macro means;
policy results concatenate the 500 disjoint outer-test questions.

| Feature group | ROC-AUC | AP | Policy accuracy | Mean samples | Token reduction | Missed recoveries |
|---|---:|---:|---:|---:|---:|---:|
| A: consensus | 0.808 | 0.162 | 76.8% | 3.42 | 75.8% | 18 |
| B: + entropy | 0.825 | 0.176 | 77.2% | 4.00 | 71.9% | 15 |
| C: + vote margin | 0.827 | 0.188 | 78.4% | 3.96 | 71.9% | 11 |
| D: all disagreement | 0.819 | 0.185 | 77.4% | 4.02 | 71.9% | 14 |
| E: + stability | 0.878 | 0.158 | 76.6% | 3.66 | 73.4% | 19 |
| F: + prefix/token cost | 0.888 | 0.204 | 77.6% | 4.32 | 69.5% | 15 |

The full feature set has the best AP and ROC-AUC but not the best stopping
accuracy. Consensus+entropy+margin is the best observed learned ablation,
matching the 78.4% simple margin baseline with similar compute. This mismatch
between ranking metrics and sequential utility is a central negative result.

## Calibration

| Model / calibration | ROC-AUC | AP | Brier | 10-bin ECE | Policy accuracy | Mean samples |
|---|---:|---:|---:|---:|---:|---:|
| Logistic uncalibrated | 0.802 | 0.119 | 0.0772 | 0.0934 | 77.2% | 4.03 |
| Logistic sigmoid | 0.893 | 0.198 | 0.0306 | 0.0097 | 77.2% | 3.12 |
| Logistic isotonic | 0.877 | 0.166 | 0.0317 | 0.0114 | 77.2% | 3.97 |
| Tree uncalibrated | 0.842 | 0.149 | 0.0905 | 0.1100 | 77.0% | 3.75 |
| Tree sigmoid | 0.867 | 0.176 | 0.0309 | 0.0069 | 78.2% | 3.85 |
| Tree isotonic | 0.846 | 0.159 | 0.0321 | 0.0086 | 77.0% | 4.17 |

Calibration substantially improves Brier score and ECE. Sigmoid calibration
also produces the strongest tree robustness policy (78.2%, 2,442 tokens per
question), but the inner selection does not choose it consistently across all
folds. At the ordinary 0.5 classification cutoff, the fold-selected primary
logistic confusion matrix is TN=6,947, FP=289, FN=223, TP=41 (recovery
recall 0.155, precision 0.124); the tree matrix is TN=6,791, FP=445, FN=189,
TP=75 (recall 0.284, precision 0.144). Pure sigmoid variants predict almost no
positives at 0.5 because recovery prevalence is only 3.52%. The sequential
policy correctly uses much lower, inner-selected thresholds. Full calibration-
specific confusion matrices and reliability bins are in
`results/calibration_results.csv`.

## Interpretability

Mean standardized logistic coefficients are most negative for prefix (-1.266),
vote margin (-0.867), stability (-0.570), and consensus (-0.263), while entropy
is positive (+0.317). The tree's largest held-out permutation AP decrease is
vote margin (0.099), followed by stability (0.030) and prefix (0.026). Token
cost and prefix are correlated, and fold coefficient standard deviations are
large; these values support complementarity, not causal importance.

## Robustness conclusion

Starting at t=4 generally spends more compute without consistently restoring
accuracy. Checkpoint decisions are not consistently better than every-sample
decisions. Larger error penalties trade compute for fewer missed recoveries,
but no setting makes the primary learned models dominate nested margin
stopping. Nonlinearity is not reliably beneficial, while calibration is
important for probability quality and sometimes policy quality. Detailed cost,
start, and schedule rows are in `results/adaptive_policy_results.csv`.
