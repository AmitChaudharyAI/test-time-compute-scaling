# Preregistered prospective statistical analysis

All 500 questions and all methods use the same completed generation panel. No exclusions or outcome-based termination occurred. The sole primary comparison is adaptive minus fixed N=16, with two-sided alpha=0.05.

## Exact paired primary test

Both correct: 383; adaptive correct / N16 wrong: 4; adaptive wrong / N16 correct: 10; both wrong: 103.

The preregistered exact two-sided binomial McNemar test conditions on 14 discordant questions. Its binomial statistic is the adaptive-only count, k=4 of n=14, under null p=0.5. Exact p=0.1795654297. No asymptotic chi-square statistic or continuity correction is used. If there were zero discordant pairs, p would be 1.

Adaptive minus N16 accuracy: **-1.20 percentage points**, paired percentile 95% bootstrap interval **[-2.60, 0.20] pp**.

## Frozen bootstrap and secondary tests

NumPy default_rng(40260915), 10,000 resamples of 500 question indices with replacement. Identical indices are reused across every method; confidence bounds are the 2.5th and 97.5th percentiles. Individual method accuracy intervals are in prospective_summary.csv. Paired difference intervals and the prespecified secondary N=8 and N=4 exact tests are below. Secondary p-values are exploratory and unadjusted; no familywise confirmatory claim is made.

| Comparison | Scope | Difference (pp) | 95% CI low (pp) | 95% CI high (pp) | Adaptive only | Fixed only | Exact p-value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_minus_fixed_n16 | primary confirmatory | -1.20 | -2.60 | 0.20 | 4 | 10 | 0.179565 |
| adaptive_minus_fixed_n8 | secondary exploratory unadjusted | 1.00 | -0.60 | 2.60 | 11 | 6 | 0.332306 |
| adaptive_minus_fixed_n4 | secondary exploratory unadjusted | 1.20 | 0.00 | 2.60 | 9 | 3 | 0.145996 |

There is no formal equivalence or non-inferiority test. Non-significance and overlapping intervals cannot demonstrate equivalence. The preregistered 0.5, 1.0 and 2.0 pp observed-shortfall bands are descriptive only and are not acceptance thresholds or policy-selection rules. Conditional stopping accuracy, error trajectories and historical-versus-fresh comparisons are secondary descriptive analyses. No new statistical tests were introduced.
