# Paper 2: offline adaptive-compute results

## Bottom line

There is substantial predictive signal in early disagreement, but the simple
threshold policies selected on development data did not preserve N=16 accuracy
on the 100-question held-out split. **Recommendation: B. A learned lightweight
controller should be developed next.** It should be evaluated with a fresh
question-level split (or nested cross-validation) and a pre-specified
compute/accuracy objective; no controller has been trained in this phase.

## Data and reproducibility

The audit passed: 500 questions x 16 ordered samples = 8,000 unique valid JSON
records, with no missing, duplicated, malformed, out-of-range, or
within-question-inconsistent records. See `REPOSITORY_AUDIT.md` and
`audit_summary.json`. The analysis recreated the frozen fixed accuracies:
N=1 72.0%, N=2 73.0%, N=4 74.8%, N=8 78.4%, and N=16 78.8%.

All reported adaptive costs use actual cumulative output tokens, not an assumed
constant cost per sample. The online signals never use the reference answer;
the frozen per-generation evaluator label is used only for retrospective labels
and evaluation.

## Which signals predict benefit from more computation?

At prefix 4, the 30 questions whose current answer was wrong but N=16 was
correct had substantially less agreement than the 460 questions with no future
correctness change: mean consensus 0.439 vs 0.881, entropy 0.839 vs 0.196,
margin 0.164 vs 0.811, and stability 2.80 vs 3.75 prefixes. The 10 regression
questions were also more ambiguous than the no-change group (consensus 0.650,
entropy 0.676, margin 0.383, stability 3.30). Thus low consensus / margin and
high entropy are the clearest associations; stability adds a smaller but
consistent signal.

As a diagnostic only, a logistic model trained on 400 questions and evaluated
on 100 held-out questions obtained ROC-AUC 0.888 and average precision 0.172
for future correctness improvement, against a 0.040 held-out prevalence. Its
Brier score was 0.141. This establishes discriminative information, not a
calibrated stopping policy or statistical significance. Full details are in
`predictability_analysis.csv` and `figures/signal_vs_future_utility.png`.

## Simple stopping policies

The best observed *exploratory* trade-off in the predefined grid was
consensus >=0.70 after at least four samples: 79.0% accuracy, 6.22 mean samples,
3,830 mean output tokens, and 56.9% token reduction versus full-panel N=16.
It is exploratory because its parameters and result use the same 500 questions;
the 0.2-point improvement over N=16 must not be interpreted as an unbiased gain.

The development-selected policies did not transfer at the same quality level:

| Held-out policy (100 questions) | Accuracy | Difference vs held-out N=16 | Mean samples | Token reduction | Early-stop errors | Missed recoveries |
|---|---:|---:|---:|---:|---:|---:|
| Consensus >=0.60, min 2 | 72.0% | -3.0 pp | 3.73 | 73.7% | 21 | 5 |
| Entropy <=0.50, min 2 | 71.0% | -4.0 pp | 4.99 | 64.5% | 17 | 4 |
| Margin >=0.50, min 2 | 72.0% | -3.0 pp | 4.34 | 68.8% | 20 | 5 |
| Stability >=5 prefixes | 71.0% | -4.0 pp | 5.65 | 62.5% | 28 | 6 |

Held-out fixed N=16 accuracy is 75.0% (75/100); its total output is 876,042
tokens. For example, held-out consensus stopping consumes 230,769 tokens
(2,308/question), saving 645,273 tokens, but misses five questions that N=16
would recover. These are paired comparisons: the consensus policy has 70 both
correct, 2 policy-only correct, 5 N=16-only correct, and 23 both wrong.

Consequently, no simple policy in this limited, development-selected family can
yet be said to approach N=16 accuracy while substantially reducing compute on
unseen questions. The full grid and its Pareto labels are descriptive only;
see `policy_results.csv` and `figures/accuracy_vs_mean_samples.png`.

## Premature stopping errors and next experiment

The available files do not include MATH domain/difficulty labels, so it would
not be supportable to label a mathematical subdomain as the cause of premature
stops. Behaviourally, the missed recoveries are questions that look settled to
a single threshold rule early yet later change to a correct N=16 plurality;
the broader recoverable group is characterized by low consensus/margin and
high entropy. The consensus hold-out policy also reaches N=16 for ten questions,
so it is not blindly stopping every trajectory.

The next experiment should keep generation frozen and compare a lightweight,
probability-calibrated controller against these baselines with nested
question-level cross-validation or a locked development/test protocol. The
controller should use the recorded online signals plus prefix/token cost, use a
pre-specified asymmetric penalty for missed recoveries, and report paired
accuracy/cost trade-offs. Before subject-level claims, join a versioned MATH-500
metadata source or pre-register a reproducible taxonomy.
