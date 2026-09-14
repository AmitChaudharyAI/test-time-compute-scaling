# Phase 2B paired statistical analysis

The pre-specified hybrid was compared question by question with nested margin,
fixed N=8, and fixed N=16. Accuracy differences are hybrid minus comparator.
Each percentile interval uses 10,000 paired resamples of the 500 question
pairs. Seeds are fixed and derived deterministically from base seed 20260923.
McNemar p-values are exact two-sided binomial tests of discordant pairs.

| Comparator | Hybrid − comparator | Paired 95% bootstrap CI | Both correct | Hybrid only | Comparator only | Both wrong | Exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nested margin | -0.2 pp | [-1.0, 0.6] | 389 | 2 | 3 | 106 | 1.000 |
| Fixed N=8 | -0.2 pp | [-2.4, 2.0] | 377 | 14 | 15 | 94 | 1.000 |
| Fixed N=16 | -0.6 pp | [-2.2, 0.8] | 385 | 6 | 9 | 100 | 0.607 |

The nested-margin comparison has only five discordant questions. Two are
successful risk overrides; three are margin-correct early stops for which the
hybrid continued into an N=16 regression. The intervals are compatible with
small positive or negative differences. Non-significance is not evidence of
equivalence or non-inferiority; no equivalence margin was pre-specified.

