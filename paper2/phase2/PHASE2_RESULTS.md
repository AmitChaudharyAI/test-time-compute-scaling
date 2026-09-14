# Phase 2 learned-controller results

## Executive result

The primary nested logistic controller does not improve on nested simple
threshold stopping. It reaches 77.6% accuracy with 4.12 samples and 2,653
output tokens per question, reducing tokens by 70.1% relative to N=16 but
losing 1.2 accuracy points. Nested consensus obtains the same accuracy with
fewer samples, and nested vote-margin stopping reaches 78.4% with 4.04 samples.

The learned models retain scientific value: recovery prediction is feasible,
and a pre-specified logistic ablation using consensus, entropy, and margin also
reaches 78.4% with 3.96 samples. But the main nested model/calibration/threshold
selection is unstable, so the evidence does not justify new GPU inference yet.

## Main accuracy-compute results

All adaptive rows concatenate five disjoint outer-test folds. Parameters for
each question were selected without that question or any of its prefixes.

| Policy | Accuracy | Mean / median samples | Mean tokens | Token reduction vs N=16 | Accuracy difference | Missed recoveries |
|---|---:|---:|---:|---:|---:|---:|
| Fixed N=16 | 78.8% | 16.00 / 16 | 8,879 | 0.0% | 0.0 pp | 0 |
| Fixed N=8 | 78.4% | 8.00 / 8 | 4,426 | 50.1% | -0.4 pp | 13 |
| Nested margin | 78.4% | 4.04 / 2 | 2,561 | 71.2% | -0.4 pp | 11 |
| Learned ablation: consensus+entropy+margin | 78.4% | 3.96 / 2 | 2,497 | 71.9% | -0.4 pp | 11 |
| Primary logistic | 77.6% | 4.12 / 2 | 2,653 | 70.1% | -1.2 pp | 14 |
| Nested consensus | 77.6% | 3.54 / 2 | 2,252 | 74.6% | -1.2 pp | 15 |
| Primary boosted tree | 76.8% | 3.79 / 2 | 2,412 | 72.8% | -2.0 pp | 17 |

The primary logistic policy uses 2,060 samples and 1,326,554 output tokens in
total, versus 8,000 samples and 4,439,465 tokens for N=16. It makes 94 early
wrong stops, including 14 recoveries that N=16 would have obtained, while
avoiding eight N=16 regressions.

The sample-based Pareto frontier among all aggregate policy rows includes low
cost learned cost-ratio variants, nested consensus, nested margin, and N=16.
The primary logistic and tree policies are Pareto-dominated. Near the
high-accuracy end, nested margin is the best observed accuracy-compute trade-off
among the main policies; this is descriptive, not a universal optimum.

## Controller prediction and calibration

Across outer folds, the selected logistic models have mean ROC-AUC 0.890 and
mean AP 0.218; selected trees have mean ROC-AUC 0.871 and AP 0.200. These remain
well above the rare recovery prevalence, but ranking performance alone does not
ensure good sequential stopping.

Sigmoid calibration reduces aggregate logistic Brier score from 0.0772 to
0.0306 and 10-bin ECE from 0.0934 to 0.0097; the analogous tree values improve
from 0.0905/0.1100 to 0.0309/0.0069. Calibration is therefore important for
probability quality. Its policy benefit is less stable: calibrated tree
robustness reaches 78.2%, while fold-wise inner selection often chooses other
methods.

## Robustness and interpretation

Starting at t=4 and using checkpoint-only decisions do not consistently improve
the accuracy-compute result. Increasing the recovery-error penalty generally
spends more compute and misses fewer recoveries, as intended, but no cost ratio
makes the primary learned controller dominate nested margin. Outer-fold
logistic accuracy ranges from 74% to 85% and tree accuracy from 73% to 86%, so
the result is not stable across folds.

Vote margin is the most important common signal: its mean standardized logistic
coefficient is -0.867 and its held-out tree permutation AP decrease is 0.099.
Lower stability, lower consensus, higher entropy, and earlier prefixes also
indicate future recovery. Adding prefix/token information improves AP, while
the compact consensus+entropy+margin group gives better policy accuracy. These
are predictive associations, not causal effects.

## Answers to the Phase 2 decision questions

1. **Does the learned controller outperform simple threshold stopping?** No.
   Primary logistic ties nested consensus on accuracy at higher cost and is
   below nested margin. The best logistic ablation only matches margin.
2. **How close is it to N=16?** Primary logistic is 1.2 points lower; the best
   learned ablation is 0.4 points lower.
3. **How much compute does it save?** Primary logistic saves 74.3% of samples
   and 70.1% of output tokens. The best learned ablation saves 75.2% of samples
   and 71.9% of tokens.
4. **How many N=16 recoveries are missed?** Fourteen by the primary logistic
   policy; eleven by the best learned ablation.
5. **Is logistic regression sufficient?** It is sufficient to show predictive
   signal, but the current full-feature selection and stopping objective are not
   sufficient to beat a margin threshold.
6. **Does nonlinearity add meaningful benefit?** Not reliably. The primary tree
   is worse; a sigmoid-calibrated robustness variant is competitive but is not
   selected consistently.
7. **Which signals contribute most?** Vote margin is strongest, with consensus,
   entropy, stability, and prefix contributing complementary information.
8. **Is calibration important?** Yes for Brier score and reliability; its
   stopping-policy advantage is configuration-dependent.
9. **Is the controller stable across outer folds?** No. Accuracy and selected
   calibration/thresholds vary materially across folds.
10. **Does this justify prospective GPU experiments?** No. The offline method
    should first be revised around sequential policy selection, compact feature
    sets, and calibration stability, then re-evaluated under the same frozen
    outer protocol or a separately locked holdout.

A.
The offline learned controller is insufficient; revise the method before
new LLM inference.
