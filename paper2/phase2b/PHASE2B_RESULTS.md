# Phase 2B results

## Executive result

The pre-specified hybrid does not improve the accuracy-compute trade-off over
nested margin. It obtains **78.2% accuracy with 4.678 mean samples and 3,054
output tokens per question**, saving 65.604% of N=16 output tokens. Nested
margin obtains 78.4% with 4.038 samples and saves 71.155% of tokens.

The risk check prevents two of margin's 11 missed recoveries, but loses three
questions where margin's early answer was correct and later computation
regressed. The net paired accuracy change is therefore -0.2 points. This is a
useful failure mode of a controller trained on future answer risk: continuing
can both recover and regress.

## Main controller comparison

All adaptive results concatenate five untouched outer folds. Simple rules and
the Phase 2 recovery controller are their original nested results.

| Controller | Accuracy | Mean / median samples | Mean tokens | Token reduction | Missed recoveries |
|---|---:|---:|---:|---:|---:|
| Fixed N=1 | 72.0% | 1.000 / 1 | 557 | 93.732% | 42 |
| Fixed N=2 | 73.0% | 2.000 / 2 | 1,122 | 87.365% | 39 |
| Fixed N=4 | 74.8% | 4.000 / 4 | 2,226 | 74.926% | 30 |
| Fixed N=8 | 78.4% | 8.000 / 8 | 4,426 | 50.148% | 13 |
| Fixed N=16 | 78.8% | 16.000 / 16 | 8,879 | 0% | 0 |
| Nested consensus | 77.6% | 3.538 / 2 | 2,252 | 74.640% | 15 |
| Nested entropy | 78.0% | 5.006 / 2 | 3,157 | 64.444% | 9 |
| Nested margin | 78.4% | 4.038 / 2 | 2,561 | 71.155% | 11 |
| Nested stability | 74.2% | 2.974 / 2 | 1,786 | 79.890% | 33 |
| Phase 2 recovery logistic | 77.6% | 4.120 / 2 | 2,653 | 70.119% | 14 |
| Safe-stop logistic, primary settings | 77.0% | 4.952 / 3 | 3,272 | 63.143% | 11 |
| Safe-stop HistGB, primary settings | 77.2% | 5.030 / 3 | 3,344 | 62.339% | 12 |
| **Hybrid margin + logistic risk (primary)** | **78.2%** | **4.678 / 2** | **3,054** | **65.604%** | **9** |
| Hybrid margin + HistGB risk, primary settings | 78.0% | 5.154 / 2 | 3,431 | 61.354% | 10 |

The primary hybrid uses 2,339 samples and 1,527,017 output tokens, versus 8,000
samples and 4,439,465 tokens at N=16. Sample reduction is 70.763%. It makes 62
wrong early stops, avoids six N=16 regressions, and reaches N=16 on 71
questions; 68 of those had a final plurality that stabilized before N=16.

## Risk prediction

Across outer folds, selected logistic models average ROC-AUC 0.927, PR-AUC
0.508, Brier 0.0711, and 10-bin ECE 0.0711. At probability 0.5 their mean
unsafe recall is 0.583 and precision is 0.497. Selected HistGB models average
ROC-AUC 0.921, PR-AUC 0.489, Brier 0.0918, and ECE 0.1058. These numbers are
substantially higher than Phase 2 recovery prediction's mean ROC-AUC 0.890 and
AP 0.218, but the targets have different prevalence and meaning; this is not a
controlled claim that one prediction problem is intrinsically easier.

Inner selection chose uncalibrated logistic probabilities in four folds and
sigmoid in one. HistGB was uncalibrated in all five. Calibration/model choice
therefore remains unstable even though the ranking signal is strong.

## Pre-specified loss and sensitivity analyses

| Hybrid logistic, t=2 every sample | Accuracy | Mean samples | Token reduction | Missed recoveries |
|---|---:|---:|---:|---:|
| 2:1 | 78.4% | 4.076 | 70.797% | 11 |
| 5:1 | 78.2% | 4.144 | 70.082% | 11 |
| **10:1 (primary)** | **78.2%** | **4.678** | **65.604%** | **9** |
| 20:1 | 78.8% | 6.794 | 50.547% | 1 |

The conservative 20:1 analysis matches N=16 accuracy while saving about half
the tokens, but it spends 2.756 more samples than margin and is a sensitivity
setting, not a post-hoc primary result. Starting decisions at t=4 also spends
substantially more compute: the 10:1 every-sample variant reaches 78.8% with
6.218 samples and 56.666% token reduction. Checkpoint-only results show no
consistent advantage. These results motivate robustness work, not a claim of a
universally optimal policy.

## Pareto and stability

The primary hybrid is Pareto-dominated in both samples and tokens by nested
margin. Some conservative Phase 2B settings extend the high-accuracy part of
the descriptive frontier, trading materially more compute for fewer missed
recoveries. The plots identify efficient points only within the evaluated
grid.

Primary-hybrid outer-fold accuracy is 78%, 85%, 77%, 75%, and 76%; mean samples
are 4.29, 4.49, 6.30, 4.19, and 4.12. Missed recoveries concentrate in the last
two folds. This variability is not stable enough for a strong prospective
claim.

## Answers to the Phase 2B questions

1. **Does safe-stop prediction work better than future-recovery prediction?**
   It has much stronger discrimination (mean PR-AUC 0.508 vs 0.218), but the
   labels differ and the standalone safe-stop policy is worse on the primary
   accuracy-compute comparison. Thus prediction improves, policy utility does
   not clearly improve.
2. **Does the hybrid policy improve on the margin-only baseline?** No. It
   prevents two missed recoveries but loses 0.2 accuracy points and uses 0.640
   more samples per question.
3. **How much accuracy does the hybrid policy achieve?** 78.2% (391/500).
4. **How many mean samples/question does it use?** 4.678.
5. **How much token compute does it save relative to N=16?** 65.604%, or
   2,912,448 output tokens over 500 questions.
6. **How many missed recoveries are prevented?** Two relative to margin (9 vs
   11).
7. **Which features are most useful for detecting risky early stops?** The
   current top-two vote-count gap is strongest, followed by prefix, answer
   stability, vote margin, and entropy in absolute standardized coefficients.
8. **Do temporal trajectory features help?** Slightly for classification:
   mean PR-AUC rises from 0.501 to 0.508 and Brier falls from 0.0503 to 0.0491.
   They do not improve hybrid accuracy in the fixed ablation; they reduce cost
   while losing 0.2 points and one additional recovery.
9. **Is performance stable across outer folds?** No. Accuracy ranges from 75%
   to 85%, mean samples from 4.12 to 6.30, and calibration selection varies.
10. **Does the result justify prospective GPU validation?** No. The primary
    method does not beat margin and the favorable conservative analyses require
    additional robustness testing on frozen data first.

A.
The hybrid/safe-stop method does not improve the trade-off enough.
Revise the method before new LLM inference.

