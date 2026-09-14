# Phase 2B feature ablations

Each ablation uses balanced L2 logistic regression (C=1), sigmoid calibration,
the locked fold-specific margin rule, lambda=10, every-sample decisions from
t=2, and inner out-of-fold tau selection. This fixes the estimator so feature
sets, rather than hyperparameter-search outcomes, drive the comparison.

| Feature group | Mean ROC-AUC | Mean PR-AUC | Accuracy | Mean samples | Token reduction | Missed recoveries |
|---|---:|---:|---:|---:|---:|---:|
| Consensus only | 0.830 | 0.353 | 78.4% | 4.316 | 69.146% | 11 |
| Vote margin only | 0.842 | 0.359 | 77.8% | 4.522 | 67.887% | 11 |
| Entropy only | 0.714 | 0.174 | 78.0% | 4.516 | 68.008% | 11 |
| Consensus + entropy + margin | 0.847 | 0.399 | 77.8% | 4.356 | 69.106% | 11 |
| Disagreement + stability | 0.919 | 0.429 | 77.8% | 5.010 | 63.564% | 10 |
| All original features | 0.926 | 0.501 | 78.4% | 4.694 | 65.506% | 10 |
| All original + trajectory deltas | 0.931 | 0.508 | 78.2% | 4.324 | 68.336% | 11 |

Trajectory features add a small, consistent classification improvement:
ROC-AUC +0.0048, PR-AUC +0.0068, Brier 0.0503 to 0.0491, and ECE 0.0234 to
0.0230. In policy simulation they reduce mean samples by 0.370 and improve
token reduction by 2.830 points, but accuracy falls by 0.2 points and one more
recovery is missed. The evidence supports incremental predictive information,
not an improved high-stakes stopping policy.

Across selected primary logistic folds, the largest absolute standardized
coefficients are top-two count gap (-2.153), prefix (-1.021), answer stability
(-0.696), vote margin (-0.670), and entropy (+0.574). Signs are conditional on
correlated features and should not be read causally. The aggregate permutation
and coefficient diagnostics are saved in `results/feature_importance.csv`.

